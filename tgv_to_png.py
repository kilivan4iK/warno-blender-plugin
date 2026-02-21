import argparse
import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
import zstandard as zstd

try:
    import numpy as np
except ImportError:  # optional, used only for BC5 normal-Z reconstruction
    np = None


TABLE_CANDIDATES = (0x30, 0x34, 0x38, 0x3C)


@dataclass(frozen=True)
class TGVInfo:
    path: Path
    version: int
    unk: int
    width: int
    height: int
    mip_count: int
    fmt: str
    data: bytes
    table_start: int
    offsets: list[int]
    sizes: list[int]


@dataclass(frozen=True)
class LayoutInfo:
    size: tuple[int, int]
    main_box: tuple[int, int, int, int] | None
    aux_boxes: list[tuple[int, int, int, int]]


def normalize_format(raw_fmt: bytes) -> str:
    text = raw_fmt.decode("ascii", errors="ignore").upper()
    patterns = (
        r"BC[1-7](?:_[A-Z0-9]+)?",
        r"A8B8G8R8(?:_[A-Z0-9]+)?",
        r"L16(?:_[A-Z0-9]+)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return text.strip() or "UNKNOWN"


def try_table(data: bytes, table_start: int, mip_count: int) -> tuple[int, list[int] | None, list[int] | None]:
    try:
        offsets = list(struct.unpack_from(f"<{mip_count}I", data, table_start))
        sizes = list(struct.unpack_from(f"<{mip_count}I", data, table_start + 4 * mip_count))
    except struct.error:
        return 0, None, None

    valid = 0
    for offset, size in zip(offsets, sizes):
        if (
            0 <= offset <= len(data) - 12
            and 12 <= size <= len(data)
            and offset + size <= len(data)
            and data[offset : offset + 4] == b"ZSTD"
        ):
            valid += 1
    return valid, offsets, sizes


def parse_tgv(path: Path) -> TGVInfo:
    data = path.read_bytes()
    if len(data) < 0x2C:
        raise RuntimeError(f"{path.name}: file too small to be valid TGV")

    version, unk, width, height = struct.unpack_from("<4I", data, 0)
    mip_count = struct.unpack_from("<H", data, 0x18)[0]
    fmt = normalize_format(data[0x1C : 0x1C + 16])

    best = (0, None, None, None)
    for table_start in TABLE_CANDIDATES:
        valid, offsets, sizes = try_table(data, table_start, mip_count)
        if valid > best[0]:
            best = (valid, table_start, offsets, sizes)

    valid, table_start, offsets, sizes = best
    if valid == 0 or table_start is None or offsets is None or sizes is None:
        raise RuntimeError(f"{path.name}: could not find valid mip offset/size table")

    return TGVInfo(
        path=path,
        version=version,
        unk=unk,
        width=width,
        height=height,
        mip_count=mip_count,
        fmt=fmt,
        data=data,
        table_start=table_start,
        offsets=offsets,
        sizes=sizes,
    )


def iter_valid_mips(info: TGVInfo) -> Iterable[tuple[int, int, int, int]]:
    for idx, (offset, size) in enumerate(zip(info.offsets, info.sizes)):
        if (
            0 <= offset <= len(info.data) - 12
            and 12 <= size <= len(info.data)
            and offset + size <= len(info.data)
            and info.data[offset : offset + 4] == b"ZSTD"
        ):
            raw_size = struct.unpack_from("<I", info.data, offset + 4)[0]
            yield idx, offset, size, raw_size


def decompress_mip(info: TGVInfo, offset: int, size: int, raw_size: int) -> bytes:
    comp = info.data[offset + 8 : offset + size]
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(comp))
    try:
        raw = reader.read(raw_size)
    finally:
        reader.close()

    if len(raw) != raw_size:
        raise RuntimeError(
            f"{info.path.name}: decompression size mismatch at 0x{offset:X}: "
            f"got {len(raw)}, expected {raw_size}"
        )
    return raw


def expected_fullres_size(width: int, height: int, fmt: str) -> int | None:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4

    if "BC1" in fmt:
        return blocks_x * blocks_y * 8
    if "BC3" in fmt or "BC5" in fmt or "BC7" in fmt:
        return blocks_x * blocks_y * 16
    if "A8B8G8R8" in fmt:
        return width * height * 4
    if "L16" in fmt:
        return width * height * 2
    return None


def pick_fullres_mip(info: TGVInfo) -> tuple[int, int, int, int]:
    mips = list(iter_valid_mips(info))
    if not mips:
        raise RuntimeError(f"{info.path.name}: no valid ZSTD mip entries found")

    expected = expected_fullres_size(info.width, info.height, info.fmt)
    if expected is not None:
        for mip in reversed(mips):
            if mip[3] == expected:
                return mip

    return max(mips, key=lambda x: x[3])


def build_dds_header_compressed(
    width: int,
    height: int,
    top_linear_size: int,
    fourcc: bytes,
    dxgi_format: int | None = None,
) -> bytes:
    dds_magic = b"DDS "
    dds_header_size = 124
    dds_pf_size = 32

    ddsd_caps = 0x1
    ddsd_height = 0x2
    ddsd_width = 0x4
    ddsd_pixel_format = 0x1000
    ddsd_linear_size = 0x80000

    ddpf_fourcc = 0x4
    ddscaps_texture = 0x1000

    flags = ddsd_caps | ddsd_height | ddsd_width | ddsd_pixel_format | ddsd_linear_size

    header = struct.pack("<I", dds_header_size)
    header += struct.pack("<I", flags)
    header += struct.pack("<I", height)
    header += struct.pack("<I", width)
    header += struct.pack("<I", top_linear_size)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 1)
    header += struct.pack("<11I", *([0] * 11))

    pf = struct.pack("<I", dds_pf_size)
    pf += struct.pack("<I", ddpf_fourcc)
    pf += fourcc
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    header += pf

    header += struct.pack("<I", ddscaps_texture)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 0)

    out = dds_magic + header
    if dxgi_format is not None:
        out += struct.pack("<5I", dxgi_format, 3, 0, 1, 0)
    return out


def decode_block_compressed(raw: bytes, width: int, height: int, fmt: str) -> Image.Image:
    fmt_up = fmt.upper()

    if "BC1" in fmt_up:
        fourcc = b"DXT1"
        dxgi = None
    elif "BC3" in fmt_up:
        fourcc = b"DXT5"
        dxgi = None
    elif "BC5" in fmt_up:
        fourcc = b"ATI2"
        dxgi = None
    elif "BC7" in fmt_up:
        fourcc = b"DX10"
        dxgi = 99 if "SRGB" in fmt_up else 98  # BC7_UNORM_SRGB / BC7_UNORM
    else:
        raise RuntimeError(f"Unsupported compressed format: {fmt}")

    dds_blob = build_dds_header_compressed(width, height, len(raw), fourcc, dxgi) + raw
    image = Image.open(io.BytesIO(dds_blob))
    image.load()

    if "BC5" in fmt_up:
        return image.convert("RGB")
    return image.convert("RGBA")


def decode_uncompressed(raw: bytes, width: int, height: int, fmt: str) -> Image.Image:
    fmt_up = fmt.upper()

    if "A8B8G8R8" in fmt_up:
        expected = width * height * 4
        if len(raw) != expected:
            raise RuntimeError(f"A8B8G8R8 size mismatch: got {len(raw)}, expected {expected}")
        return Image.frombytes("RGBA", (width, height), raw)

    if "L16" in fmt_up:
        expected = width * height * 2
        if len(raw) != expected:
            raise RuntimeError(f"L16 size mismatch: got {len(raw)}, expected {expected}")
        return Image.frombytes("I;16", (width, height), raw)

    raise RuntimeError(f"Unsupported uncompressed format: {fmt}")


def decode_tgv_image(info: TGVInfo, raw: bytes) -> Image.Image:
    fmt_up = info.fmt.upper()
    if "BC" in fmt_up:
        return decode_block_compressed(raw, info.width, info.height, info.fmt)
    return decode_uncompressed(raw, info.width, info.height, info.fmt)


def detect_texture_role(path: Path, fmt: str) -> str:
    name = path.stem.lower()
    fmt_up = fmt.upper()

    if "combinedda" in name or "coloralpha" in name:
        return "combined_da"
    if "normal" in name or "tscnm" in name or "BC5" in fmt_up:
        return "normal"
    if "combinedorm" in name or "_orm" in name or "ormtexture" in name:
        return "orm"
    if "splat" in name:
        return "splat"
    if "height" in name or "L16" in fmt_up:
        return "height"
    return "generic"


def extract_unit_name_from_atlas(atlas_path: Path) -> str | None:
    try:
        text = atlas_path.read_bytes().decode("latin1", errors="ignore")
    except OSError:
        return None

    match = re.search(r"/([A-Za-z0-9_]+)/TSC", text)
    if match:
        return match.group(1)
    return None


def atlas_category_from_text(text: str) -> str | None:
    up = text.upper()
    if "/UNITS/" in up:
        return "unit"
    if "/DECORS/" in up:
        return "decor"
    return None


def detect_atlas_category_in_folder(folder: Path) -> str | None:
    for atlas_path in sorted(folder.glob("*.atlas")):
        try:
            text = atlas_path.read_bytes().decode("latin1", errors="ignore")
        except OSError:
            continue
        category = atlas_category_from_text(text)
        if category:
            return category
    return None


def find_unit_name_in_folder(folder: Path) -> str | None:
    for atlas_path in sorted(folder.glob("*.atlas")):
        unit_name = extract_unit_name_from_atlas(atlas_path)
        if unit_name:
            return unit_name
    return None


def canonical_stem_for_file(in_file: Path, unit_name: str | None) -> str:
    if not unit_name:
        return in_file.stem

    stem_low = in_file.stem.lower()
    if "diffusetexturenoalpha" in stem_low:
        return f"{unit_name}_D"
    if "combinedormtexture" in stem_low:
        return f"{unit_name}_ORM"
    if "normaltexture" in stem_low or "tscnm" in stem_low:
        return f"{unit_name}_NM"
    if "combineddatexture" in stem_low or "coloralpha" in stem_low:
        return f"{unit_name}_DA"
    return in_file.stem


def split_base_and_tag(stem: str) -> tuple[str, str | None]:
    stem_up = stem.upper()
    for tag in ("_NM", "_ORM", "_DA", "_D", "_A", "_AO", "_R", "_M"):
        if stem_up.endswith(tag):
            return stem[: -len(tag)], tag[1:]
    return stem, None


def is_track_like_source_name(path: Path) -> bool:
    low = path.stem.lower()
    return "_trk" in low or "track" in low or "chenille" in low


def true_ranges(mask_1d: "np.ndarray") -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask_1d.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx))
            start = None
    if start is not None:
        ranges.append((start, len(mask_1d)))
    return ranges


def smooth_1d(values: "np.ndarray", window: int) -> "np.ndarray":
    if window <= 1:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def find_main_cut_row(row_counts: "np.ndarray", width: int) -> int | None:
    height = len(row_counts)
    if height < 32:
        return None

    smooth = smooth_1d(row_counts, max(5, height // 300))
    ref_window_end = max(12, height // 3)
    ref = float(np.percentile(smooth[:ref_window_end], 90))
    if ref <= 0:
        return None

    # 1) Prefer a strong one-step drop (common for atlas packing main -> lower parts).
    start = max(8, height // 6)
    end = max(start + 2, height - 8)
    deltas = smooth[1:] - smooth[:-1]
    if end > start:
        for idx in range(start, end):
            drop = float(-deltas[idx])
            if drop < max(8.0, width * 0.01):
                continue

            before = float(np.mean(smooth[max(0, idx - 32) : idx + 1]))
            after = float(np.mean(smooth[idx + 1 : min(height, idx + 33)]))
            if before <= 0:
                continue
            if after <= before * 0.93:
                return idx + 1

    # 2) Fallback: persistent drop band.
    drop_threshold = ref * 0.80
    min_run = max(8, height // 256)
    min_aux_signal = max(4, width // 40)
    search_start = max(16, height // 10)

    for y in range(search_start, height - min_run):
        if np.all(smooth[y : y + min_run] <= drop_threshold):
            if float(np.max(smooth[y + min_run :])) >= min_aux_signal:
                return y
    return None


def split_range_by_valley(col_counts: "np.ndarray", x0: int, x1: int) -> list[tuple[int, int]]:
    width = x1 - x0
    if width < 128:
        return [(x0, x1)]

    segment = col_counts[x0:x1].astype(np.float32)
    if segment.size < 8:
        return [(x0, x1)]

    smooth = smooth_1d(segment, max(9, width // 40))
    peak_floor = float(smooth.max()) * 0.35
    if peak_floor <= 0:
        return [(x0, x1)]

    margin = max(12, width // 18)
    best_valley = None
    best_depth = 0.0
    for v in range(margin, len(smooth) - margin):
        if not (smooth[v] <= smooth[v - 1] and smooth[v] <= smooth[v + 1]):
            continue

        left_peak = float(np.max(smooth[:v]))
        right_peak = float(np.max(smooth[v + 1 :]))
        peak_min = min(left_peak, right_peak)
        if peak_min < peak_floor:
            continue

        depth = peak_min - float(smooth[v])
        if depth <= peak_min * 0.14:
            continue

        if depth > best_depth:
            best_depth = depth
            best_valley = (v, peak_min, float(smooth[v]))

    if best_valley is None:
        return [(x0, x1)]

    valley_rel, peak_min, valley_value = best_valley
    if valley_value > peak_min * 0.86:
        return [(x0, x1)]

    cut = x0 + valley_rel
    if cut - x0 < 16 or x1 - cut < 16:
        return [(x0, x1)]
    return [(x0, cut), (cut, x1)]


def split_range_by_color_jump(
    band_rgb: "np.ndarray",
    band_mask: "np.ndarray",
    x0: int,
    x1: int,
) -> list[tuple[int, int]]:
    width = x1 - x0
    if width < 96:
        return [(x0, x1)]

    region_rgb = band_rgb[:, x0:x1, :].astype(np.float32)
    region_mask = band_mask[:, x0:x1]
    if not region_mask.any():
        return [(x0, x1)]

    counts = region_mask.sum(axis=0).astype(np.float32)
    sums = (region_rgb * region_mask[:, :, None].astype(np.float32)).sum(axis=0)
    means = np.zeros_like(sums, dtype=np.float32)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid, None]

    # Forward-fill empty columns to avoid artificial jumps.
    for i in range(1, width):
        if not valid[i]:
            means[i] = means[i - 1]
    for i in range(width - 2, -1, -1):
        if not valid[i]:
            means[i] = means[i + 1]

    diffs = np.linalg.norm(means[1:] - means[:-1], axis=1)
    if diffs.size == 0:
        return [(x0, x1)]

    diffs = smooth_1d(diffs, max(5, width // 200))
    lo = max(8, width // 8)
    hi = max(lo + 1, width - lo)
    search = diffs[lo:hi]
    if search.size == 0:
        return [(x0, x1)]

    cut_rel = int(np.argmax(search)) + lo + 1
    jump = float(diffs[cut_rel - 1])
    median_jump = float(np.median(diffs))
    if jump < max(2.5, median_jump * 2.2):
        return [(x0, x1)]

    left_mass = float(counts[:cut_rel].sum())
    right_mass = float(counts[cut_rel:].sum())
    total_mass = left_mass + right_mass
    if total_mass <= 0:
        return [(x0, x1)]
    if min(left_mass, right_mass) < total_mass * 0.10:
        return [(x0, x1)]

    cut = x0 + cut_rel
    if cut - x0 < 16 or x1 - cut < 16:
        return [(x0, x1)]
    return [(x0, cut), (cut, x1)]


def align_bbox_to_grid(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    grid: int = 2,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    g = max(1, int(grid))
    x0 = max(0, (x0 // g) * g)
    y0 = max(0, (y0 // g) * g)
    x1 = min(width, ((x1 + g - 1) // g) * g)
    y1 = min(height, ((y1 + g - 1) // g) * g)
    return x0, y0, x1, y1


def snap_value_to_anchors(value: int, anchors: list[int], tol: int) -> int:
    if not anchors:
        return value
    nearest = min(anchors, key=lambda a: abs(a - value))
    if abs(nearest - value) <= tol:
        return nearest
    return value


def snap_box_to_major_grid(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    tol_x: int | None = None,
    tol_y: int | None = None,
    grid: int = 2,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = box

    if tol_x is None:
        tol_x = max(4, width // 512)
    if tol_y is None:
        tol_y = max(4, height // 512)

    anchors_x = sorted(set([0, width // 4, width // 2, (width * 3) // 4, width]))
    anchors_y = sorted(set([0, height // 4, height // 2, (height * 3) // 4, height]))

    x0 = snap_value_to_anchors(x0, anchors_x, tol_x)
    x1 = snap_value_to_anchors(x1, anchors_x, tol_x)
    y0 = snap_value_to_anchors(y0, anchors_y, tol_y)
    y1 = snap_value_to_anchors(y1, anchors_y, tol_y)

    return align_bbox_to_grid((x0, y0, x1, y1), width, height, grid=grid)


def bbox_from_row_range(mask: "np.ndarray", y0: int, y1: int) -> tuple[int, int, int, int] | None:
    if y1 <= y0:
        return None

    sub = mask[y0:y1]
    if not sub.any():
        return None

    col_counts = sub.sum(axis=0)
    max_col = int(col_counts.max())
    if max_col <= 0:
        return None

    col_threshold = max(4, int(max_col * 0.40))
    col_ranges = true_ranges(col_counts >= col_threshold)
    if not col_ranges:
        return None

    x0, x1 = max(col_ranges, key=lambda r: r[1] - r[0])
    return int(x0), int(y0), int(x1), int(y1)


def detect_main_and_aux_bboxes(
    image: Image.Image,
    allow_subsplit: bool = True,
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    if np is None:
        return None, []

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        return None, []

    height, width, _ = arr.shape

    # In WARNO atlas-like textures, background padding is usually at corners.
    corner_sample = np.array(
        [
            arr[0, 0],
            arr[0, width - 1],
            arr[height - 1, 0],
            arr[height - 1, width - 1],
            arr[max(0, height - 2), max(0, width - 2)],
        ],
        dtype=np.uint8,
    )
    colors, counts = np.unique(corner_sample, axis=0, return_counts=True)
    bg = colors[counts.argmax()]

    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    fg = diff > 6

    row_counts = fg.sum(axis=1)
    max_row = int(row_counts.max())
    min_row_pixels = max(8, width // 10)
    if max_row < min_row_pixels:
        return None, []

    main_cut = find_main_cut_row(row_counts, width)
    if main_cut is not None and main_cut > 0:
        row_active = np.where(row_counts > max(1, width // 200))[0]
        if row_active.size == 0:
            return None, []
        main_y0 = int(row_active[0])
        main_y1 = int(main_cut)
    else:
        high_threshold = max(min_row_pixels, int(max_row * 0.75))
        main_ranges = true_ranges(row_counts >= high_threshold)
        if not main_ranges:
            return None, []
        main_y0, main_y1 = max(main_ranges, key=lambda r: r[1] - r[0])

    main_box = bbox_from_row_range(fg, main_y0, main_y1)
    if main_box is None:
        return None, []
    main_box = align_bbox_to_grid(main_box, width, height, grid=2)

    aux_boxes: list[tuple[int, int, int, int]] = []
    if main_y1 < height:
        lower = fg[main_y1:]
        lower_row_counts = lower.sum(axis=1)
        min_aux_row_pixels = max(4, width // 40)
        y_ranges = true_ranges(lower_row_counts >= min_aux_row_pixels)

        for y0_rel, y1_rel in y_ranges:
            band = lower[y0_rel:y1_rel]
            if not band.any():
                continue

            band_rgb = arr[main_y1 + y0_rel : main_y1 + y1_rel]
            col_counts = band.sum(axis=0)
            peak = int(col_counts.max())
            if peak <= 0:
                continue

            col_threshold = max(2, int(peak * 0.18))
            x_ranges = true_ranges(col_counts >= col_threshold)

            if len(x_ranges) <= 1:
                strong_threshold = max(2, int(peak * 0.40))
                strong_ranges = true_ranges(col_counts >= strong_threshold)
                if len(strong_ranges) >= 2:
                    x_ranges = strong_ranges

            if allow_subsplit and len(x_ranges) == 1:
                x_ranges = split_range_by_valley(col_counts, x_ranges[0][0], x_ranges[0][1])
            if allow_subsplit and len(x_ranges) == 1:
                x_ranges = split_range_by_color_jump(band_rgb, band, x_ranges[0][0], x_ranges[0][1])

            for x0, x1 in x_ranges:
                region = band[:, x0:x1]
                if not region.any():
                    continue

                rows_any = np.where(region.any(axis=1))[0]
                cols_any = np.where(region.any(axis=0))[0]
                if rows_any.size == 0 or cols_any.size == 0:
                    continue

                bx0 = int(x0 + cols_any[0])
                bx1 = int(x0 + cols_any[-1] + 1)
                by0 = int(main_y1 + y0_rel + rows_any[0])
                by1 = int(main_y1 + y0_rel + rows_any[-1] + 1)

                box = align_bbox_to_grid((bx0, by0, bx1, by1), width, height, grid=2)
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area >= (width * height) // 500:
                    aux_boxes.append(box)

    # Keep unique boxes and sort by area (largest first).
    unique_boxes = list(dict.fromkeys(aux_boxes))
    unique_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return main_box, unique_boxes


def part_output_path(out_main: Path, role: str, part_kind: str) -> Path:
    base, tag = split_base_and_tag(out_main.stem)
    if tag:
        return out_main.with_name(f"{base}_{part_kind}_{tag}.png")
    return out_main.with_name(f"{out_main.stem}_{part_kind.lower()}.png")


def assign_part_kinds(
    aux_boxes: list[tuple[int, int, int, int]],
    atlas_category: str | None,
) -> list[str]:
    if not aux_boxes:
        return []

    if atlas_category != "unit":
        kinds = ["" for _ in aux_boxes]
        order = sorted(
            range(len(aux_boxes)),
            key=lambda i: (
                (aux_boxes[i][0] + aux_boxes[i][2]) * 0.5,
                (aux_boxes[i][1] + aux_boxes[i][3]) * 0.5,
            ),
        )
        for part_no, idx in enumerate(order, start=1):
            kinds[idx] = f"PART{part_no}"
        return kinds

    centers = [((b[0] + b[2]) * 0.5, i) for i, b in enumerate(aux_boxes)]
    track_idx = max(centers, key=lambda t: t[0])[1]  # right-most block is usually tracks

    kinds = ["" for _ in aux_boxes]
    kinds[track_idx] = "TRK"

    remaining = [i for i in range(len(aux_boxes)) if i != track_idx]
    remaining.sort(key=lambda i: (aux_boxes[i][0] + aux_boxes[i][2]) * 0.5)
    for pos, idx in enumerate(remaining):
        if pos == 0:
            kinds[idx] = "MG"
        else:
            kinds[idx] = f"PART{pos + 2}"
    return kinds


def maybe_mirror(image: Image.Image, mirror: bool) -> Image.Image:
    if mirror:
        return ImageOps.mirror(image)
    return image


def decode_tgv_for_layout(path: Path) -> tuple[Image.Image, str]:
    info = parse_tgv(path)
    mip_idx, offset, size, raw_size = pick_fullres_mip(info)
    raw = decompress_mip(info, offset, size, raw_size)
    role = detect_texture_role(path, info.fmt)
    decoded = decode_tgv_image(info, raw)
    if role == "normal" and np is not None:
        reconstructed, _ = normal_reconstruct_z(decoded.convert("RGB"))
        return reconstructed, role
    return decoded, role


def build_layout_for_group(
    files: list[Path],
    atlas_category: str | None = None,
    aggressive_split: bool = False,
) -> LayoutInfo | None:
    if np is None:
        return None

    # Prefer maps that usually contain clearer packed blocks.
    priority = {"orm": 0, "normal": 1, "generic": 2}
    best: tuple[int, int, int, LayoutInfo] | None = None  # (aux_count, aux_area, priority_neg, layout)

    for path in files:
        try:
            image, role = decode_tgv_for_layout(path)
        except Exception:
            continue

        if is_track_like_source_name(path) and not aggressive_split:
            continue

        main_box, aux_boxes = detect_main_and_aux_bboxes(
            image.convert("RGB"),
            allow_subsplit=bool(aggressive_split),
        )
        main_box, aux_boxes = refine_layout_to_content(image.convert("RGB"), main_box, aux_boxes)
        aux_boxes = filter_aux_boxes_for_parts(aux_boxes, image.size)
        if not should_split_layout(
            image.size,
            main_box,
            aux_boxes,
            atlas_category,
            aggressive=bool(aggressive_split),
        ):
            continue
        if main_box is None and not aux_boxes:
            continue

        layout = LayoutInfo(size=image.size, main_box=main_box, aux_boxes=aux_boxes)
        total_aux_area = sum(box_area(b) for b in aux_boxes)
        score = (
            len(aux_boxes),
            total_aux_area,
            -priority.get(role, 9),
        )
        if best is None or score > (best[0], best[1], best[2]):
            best = (score[0], score[1], score[2], layout)

    if best is None:
        return None
    return best[3]


def scale_box(box: tuple[int, int, int, int], src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[int, int, int, int]:
    sx = dst_size[0] / float(src_size[0])
    sy = dst_size[1] / float(src_size[1])
    x0 = int(round(box[0] * sx))
    y0 = int(round(box[1] * sy))
    x1 = int(round(box[2] * sx))
    y1 = int(round(box[3] * sy))
    return align_bbox_to_grid((x0, y0, x1, y1), dst_size[0], dst_size[1], grid=2)


def box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def dominant_corner_bg_rgb(arr: "np.ndarray") -> "np.ndarray":
    height, width, _ = arr.shape
    corner_sample = np.array(
        [
            arr[0, 0],
            arr[0, width - 1],
            arr[height - 1, 0],
            arr[height - 1, width - 1],
            arr[max(0, height - 2), max(0, width - 2)],
        ],
        dtype=np.uint8,
    )
    colors, counts = np.unique(corner_sample, axis=0, return_counts=True)
    return colors[counts.argmax()]


def tighten_box_to_mask(
    mask: "np.ndarray",
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    grid: int = 2,
    pad: int = 2,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None

    sub = mask[y0:y1, x0:x1]
    if sub.size == 0 or not sub.any():
        return None

    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    row_ratio = sub.mean(axis=1)
    col_ratio = sub.mean(axis=0)

    row_thr = max(0.002, 1.0 / float(w))
    col_thr = max(0.002, 1.0 / float(h))

    rows = np.where(row_ratio >= row_thr)[0]
    cols = np.where(col_ratio >= col_thr)[0]
    if rows.size == 0:
        rows = np.where(sub.any(axis=1))[0]
    if cols.size == 0:
        cols = np.where(sub.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    tx0 = x0 + int(cols[0]) - pad
    ty0 = y0 + int(rows[0]) - pad
    tx1 = x0 + int(cols[-1]) + 1 + pad
    ty1 = y0 + int(rows[-1]) + 1 + pad
    return align_bbox_to_grid((tx0, ty0, tx1, ty1), image_size[0], image_size[1], grid=grid)


def refine_layout_to_content(
    image: Image.Image,
    main_box: tuple[int, int, int, int] | None,
    aux_boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    if np is None:
        return main_box, aux_boxes

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        return main_box, aux_boxes

    bg = dominant_corner_bg_rgb(arr)
    diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = diff > 4

    refined_main = main_box
    if main_box is not None:
        tightened = tighten_box_to_mask(mask, main_box, image.size, grid=2, pad=2)
        if tightened is not None and box_area(tightened) > 0:
            refined_main = snap_box_to_major_grid(tightened, image.size, grid=2)

    refined_aux: list[tuple[int, int, int, int]] = []
    min_area = max(64, (image.size[0] * image.size[1]) // 4000)
    for box in aux_boxes:
        tightened = tighten_box_to_mask(mask, box, image.size, grid=2, pad=2)
        if tightened is None:
            continue
        tightened = snap_box_to_major_grid(tightened, image.size, grid=2)
        if box_area(tightened) < min_area:
            continue
        refined_aux.append(tightened)

    refined_aux = list(dict.fromkeys(refined_aux))
    refined_aux.sort(key=box_area, reverse=True)
    return refined_main, refined_aux


def filter_aux_boxes_for_parts(
    aux_boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    total = float(width * height)
    min_w = max(16, int(width * 0.02))
    min_h = max(16, int(height * 0.02))

    kept: list[tuple[int, int, int, int]] = []
    for box in aux_boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w < min_w or h < min_h:
            continue

        area = float(w * h)
        area_ratio = area / total
        if area_ratio < 0.002:
            continue

        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 12.0 and area_ratio < 0.015:
            continue

        kept.append(box)

    kept.sort(key=box_area, reverse=True)
    return kept


def should_split_layout(
    image_size: tuple[int, int],
    main_box: tuple[int, int, int, int] | None,
    aux_boxes: list[tuple[int, int, int, int]],
    atlas_category: str | None,
    aggressive: bool = False,
) -> bool:
    if main_box is None or not aux_boxes:
        return False

    width, height = image_size
    total = float(width * height)
    main_ratio = box_area(main_box) / total

    if aggressive:
        main_min = 0.20
        main_max = 0.90
        sig_min = 0.012
        aux_min = 0.015
        dense_aux_limit = 4
        dense_aux_total_min = 0.08
    else:
        # Safer defaults for importer workflow: split only on strong signal.
        main_min = 0.28
        main_max = 0.82
        sig_min = 0.018
        aux_min = 0.03
        dense_aux_limit = 3
        dense_aux_total_min = 0.10

    if main_ratio < main_min or main_ratio > main_max:
        return False

    aux_areas = [box_area(b) / total for b in aux_boxes]
    significant = [a for a in aux_areas if a >= sig_min]
    if not significant:
        return False

    total_aux = float(sum(aux_areas))
    if total_aux < aux_min:
        return False

    # Avoid noisy over-splitting on dense atlases with many tiny pieces.
    if len(aux_boxes) > dense_aux_limit and total_aux < dense_aux_total_min:
        return False

    if atlas_category == "unit":
        if aggressive:
            return True
        return len(significant) >= 1 and total_aux >= 0.03
    if atlas_category == "decor":
        if aggressive:
            return total_aux >= 0.025 and len(significant) >= 1
        return total_aux >= 0.05 and len(significant) >= 2

    # Unknown atlas type: require stronger signal to avoid false positives.
    if aggressive:
        return (len(significant) >= 2 and total_aux >= 0.03) or total_aux >= 0.05
    return (len(significant) >= 2 and total_aux >= 0.05) or total_aux >= 0.08


def normal_reconstruct_z(rgb: Image.Image) -> tuple[Image.Image, Image.Image]:
    if np is None:
        raise RuntimeError("NumPy is not installed, cannot reconstruct BC5 normal Z channel")

    arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    x = arr[:, :, 0] / 255.0 * 2.0 - 1.0
    y = arr[:, :, 1] / 255.0 * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))

    out = np.empty_like(arr, dtype=np.uint8)
    out[:, :, 0] = np.clip(arr[:, :, 0], 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(arr[:, :, 1], 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip((z * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    z_gray = np.clip((z * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB"), Image.fromarray(z_gray, mode="L")


def preview_8bit_from_16bit(image: Image.Image) -> Image.Image:
    raw = image.tobytes()
    if image.mode == "I;16B":
        high_bytes = raw[0::2]
    else:
        high_bytes = raw[1::2]
    return Image.frombytes("L", image.size, high_bytes)


def save_auto_channels(image: Image.Image, role: str, out_main: Path) -> list[Path]:
    out_paths: list[Path] = []
    stem = out_main.with_suffix("")
    base_name, canonical_tag = split_base_and_tag(stem.name)

    if role == "orm":
        rgb = image.convert("RGB")
        if canonical_tag == "ORM":
            names = (f"{base_name}_AO.png", f"{base_name}_R.png", f"{base_name}_M.png")
        else:
            names = (
                f"{stem.name}_occlusion.png",
                f"{stem.name}_roughness.png",
                f"{stem.name}_metallic.png",
            )

        for channel, filename in zip(rgb.split(), names):
            path = stem.with_name(filename)
            channel.save(path)
            out_paths.append(path)

    elif role == "combined_da":
        rgba = image.convert("RGBA")
        r, g, b, a = rgba.split()

        if canonical_tag == "DA":
            alpha_path = stem.with_name(f"{base_name}_A.png")
            a.save(alpha_path)
            out_paths.append(alpha_path)
        else:
            diffuse_path = stem.with_name(f"{stem.name}_diffuse.png")
            alpha_path = stem.with_name(f"{stem.name}_alpha.png")

            Image.merge("RGB", (r, g, b)).save(diffuse_path)
            a.save(alpha_path)

            out_paths.extend((diffuse_path, alpha_path))

    elif role == "splat":
        rgba = image.convert("RGBA")
        for channel, suffix in zip(rgba.split(), ("mask_r", "mask_g", "mask_b", "mask_a")):
            path = stem.with_name(f"{stem.name}_{suffix}.png")
            channel.save(path)
            out_paths.append(path)

    elif role == "normal":
        rgb = image.convert("RGB")
        x_chan, y_chan, _ = rgb.split()

        x_path = stem.with_name(f"{stem.name}_normal_x.png")
        y_path = stem.with_name(f"{stem.name}_normal_y.png")
        x_chan.save(x_path)
        y_chan.save(y_path)
        out_paths.extend((x_path, y_path))

        if np is not None:
            _, z_chan = normal_reconstruct_z(rgb)
            z_path = stem.with_name(f"{stem.name}_normal_z.png")
            z_chan.save(z_path)
            out_paths.append(z_path)

    elif role == "height" and image.mode in ("I;16", "I;16L", "I;16B"):
        preview = preview_8bit_from_16bit(image)
        preview_path = stem.with_name(f"{stem.name}_height_preview_8bit.png")
        preview.save(preview_path)
        out_paths.append(preview_path)

    return out_paths


def save_all_channels(image: Image.Image, out_main: Path) -> list[Path]:
    out_paths: list[Path] = []
    stem = out_main.with_suffix("")

    if image.mode in ("RGB", "RGBA"):
        labels = ("r", "g", "b", "a")
        for idx, channel in enumerate(image.split()):
            path = stem.with_name(f"{stem.name}_{labels[idx]}.png")
            channel.save(path)
            out_paths.append(path)
    elif image.mode in ("I;16", "I;16L", "I;16B"):
        preview = preview_8bit_from_16bit(image)
        path = stem.with_name(f"{stem.name}_8bit.png")
        preview.save(path)
        out_paths.append(path)
    return out_paths


def resolve_output_file(in_file: Path, output_arg: str | None, stem_override: str | None = None) -> Path:
    stem = stem_override or in_file.stem
    if output_arg is None:
        return in_file.with_name(f"{stem}.png")

    out = Path(output_arg)
    if out.suffix.lower() == ".png":
        return out
    return out / f"{stem}.png"


def cleanup_stale_outputs(out_main: Path) -> None:
    parent = out_main.parent
    base, tag = split_base_and_tag(out_main.stem)

    candidates: set[Path] = {out_main}
    candidates.update(parent.glob(f"{out_main.stem}_*.png"))

    # Remove old split-part files from previous runs for the same logical texture.
    if tag:
        candidates.update(parent.glob(f"{base}_*_{tag}.png"))
        candidates.update(parent.glob(f"{base}_*_{tag}_*.png"))

    for path in candidates:
        if not path.is_file():
            continue
        try:
            path.unlink()
        except OSError:
            pass


def convert_one(
    in_file: Path,
    out_file: Path,
    split_mode: str,
    mirror: bool,
    aggressive_split: bool = False,
    shared_layout: LayoutInfo | None = None,
    atlas_category: str | None = None,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    info = parse_tgv(in_file)
    mip_idx, offset, size, raw_size = pick_fullres_mip(info)
    raw = decompress_mip(info, offset, size, raw_size)
    role = detect_texture_role(in_file, info.fmt)
    decoded = decode_tgv_image(info, raw)

    base_tag = split_base_and_tag(out_file.stem)[1]
    image_to_save = decoded
    extras: list[Path] = []
    part_images: list[tuple[Image.Image, Path]] = []

    if role == "normal" and np is not None:
        reconstructed, _ = normal_reconstruct_z(decoded.convert("RGB"))
        source_for_split = reconstructed
        image_to_save = reconstructed
    else:
        source_for_split = decoded

    split_candidates = {"D", "NM", "ORM"}
    is_diffuse_like = role == "generic" and "diffuse" in in_file.stem.lower()
    should_split_parts = base_tag in split_candidates or role in ("normal", "orm") or is_diffuse_like
    skip_spatial_split = is_track_like_source_name(in_file) and not bool(aggressive_split)
    if split_mode == "auto" and np is not None and should_split_parts and not skip_spatial_split:
        shared_mode = ""  # "", "full", "aux_only"
        if shared_layout is not None:
            src_w, src_h = shared_layout.size
            dst_w, dst_h = source_for_split.size
            sx = (dst_w / float(src_w)) if src_w > 0 else 1.0
            sy = (dst_h / float(src_h)) if src_h > 0 else 1.0
            # Share layout only when scale is close to uniform.
            if abs(sx - sy) <= 0.06:
                if shared_layout.size == source_for_split.size:
                    main_box = shared_layout.main_box
                    aux_boxes = list(shared_layout.aux_boxes)
                else:
                    main_box = (
                        scale_box(shared_layout.main_box, shared_layout.size, source_for_split.size)
                        if shared_layout.main_box is not None
                        else None
                    )
                    aux_boxes = [scale_box(box, shared_layout.size, source_for_split.size) for box in shared_layout.aux_boxes]
                aux_boxes = filter_aux_boxes_for_parts(aux_boxes, source_for_split.size)
                shared_mode = "full"
            elif is_diffuse_like and shared_layout.aux_boxes:
                # Diffuse maps are often 2048x2048 while NM/ORM are 2048x4096.
                # Reuse only aux boxes from shared layout, but keep diffuse main intact.
                main_box = None
                aux_boxes = [scale_box(box, shared_layout.size, source_for_split.size) for box in shared_layout.aux_boxes]
                aux_boxes = filter_aux_boxes_for_parts(aux_boxes, source_for_split.size)
                shared_mode = "aux_only"

        if not shared_mode:
            main_box, aux_boxes = detect_main_and_aux_bboxes(
                source_for_split.convert("RGB"),
                allow_subsplit=bool(aggressive_split),
            )
            main_box, aux_boxes = refine_layout_to_content(source_for_split.convert("RGB"), main_box, aux_boxes)
            aux_boxes = filter_aux_boxes_for_parts(aux_boxes, source_for_split.size)
            if not should_split_layout(
                source_for_split.size,
                main_box,
                aux_boxes,
                atlas_category,
                aggressive=bool(aggressive_split),
            ):
                main_box = None
                aux_boxes = []

        if main_box is not None:
            image_to_save = source_for_split.crop(main_box)

        part_kinds = assign_part_kinds(aux_boxes, atlas_category)
        for idx, box in enumerate(aux_boxes):
            kind = part_kinds[idx] if idx < len(part_kinds) and part_kinds[idx] else f"PART{idx + 1}"
            part_out = part_output_path(out_file, role, kind)
            part_images.append((source_for_split.crop(box), part_out))

    cleanup_stale_outputs(out_file)

    image_to_save = maybe_mirror(image_to_save, mirror)
    image_to_save.save(out_file)

    for part_image, part_out in part_images:
        part_image = maybe_mirror(part_image, mirror)
        part_image.save(part_out)
        extras.append(part_out)
        if split_mode == "auto":
            if atlas_category == "unit":
                extras.extend(save_auto_channels(part_image, role, part_out))
        elif split_mode == "all":
            extras.extend(save_all_channels(part_image, part_out))

    if split_mode == "auto":
        extras.extend(save_auto_channels(image_to_save, role, out_file))
    elif split_mode == "all":
        extras.extend(save_all_channels(image_to_save, out_file))

    print(
        f"[OK] {in_file.name} -> {out_file.name} | "
        f"fmt={info.fmt} size={info.width}x{info.height} fullMip={mip_idx}/{info.mip_count - 1} role={role}"
    )
    for extra in extras:
        print(f"     + {extra.name}")


def convert_path(
    input_path: Path,
    output_arg: str | None,
    recursive: bool,
    split_mode: str,
    mirror: bool,
    auto_naming: bool,
    aggressive_split: bool = False,
) -> None:
    if input_path.is_file():
        parent = input_path.parent
        unit_name = find_unit_name_in_folder(parent) if auto_naming else None
        atlas_category = detect_atlas_category_in_folder(parent)
        stem = canonical_stem_for_file(input_path, unit_name)
        shared_layout = (
            build_layout_for_group(
                [input_path],
                atlas_category=atlas_category,
                aggressive_split=bool(aggressive_split),
            )
            if split_mode == "auto"
            else None
        )
        convert_one(
            input_path,
            resolve_output_file(input_path, output_arg, stem_override=stem),
            split_mode,
            mirror,
            aggressive_split=aggressive_split,
            shared_layout=shared_layout,
            atlas_category=atlas_category,
        )
        return

    if not input_path.is_dir():
        raise RuntimeError(f"Input path does not exist: {input_path}")

    out_dir = Path(output_arg) if output_arg else input_path / "png_out"
    pattern = "**/*.tgv" if recursive else "*.tgv"
    files = sorted(input_path.glob(pattern))

    if not files:
        print(f"No .tgv files found in {input_path}")
        return

    unit_cache: dict[Path, str | None] = {}
    atlas_category_cache: dict[Path, str | None] = {}
    files_by_parent: dict[Path, list[Path]] = {}
    for file_path in files:
        files_by_parent.setdefault(file_path.parent, []).append(file_path)

    layout_cache: dict[Path, LayoutInfo | None] = {}
    if split_mode == "auto":
        for parent, parent_files in files_by_parent.items():
            if parent not in atlas_category_cache:
                atlas_category_cache[parent] = detect_atlas_category_in_folder(parent)
            layout_cache[parent] = build_layout_for_group(
                parent_files,
                atlas_category=atlas_category_cache.get(parent),
                aggressive_split=bool(aggressive_split),
            )

    for file_path in files:
        rel = file_path.relative_to(input_path)
        parent = file_path.parent
        if auto_naming and parent not in unit_cache:
            unit_cache[parent] = find_unit_name_in_folder(parent)
        if parent not in atlas_category_cache:
            atlas_category_cache[parent] = detect_atlas_category_in_folder(parent)

        stem = canonical_stem_for_file(file_path, unit_cache.get(parent))
        out_file = (out_dir / rel.parent / f"{stem}.png")
        convert_one(
            file_path,
            out_file,
            split_mode,
            mirror,
            aggressive_split=aggressive_split,
            shared_layout=layout_cache.get(parent),
            atlas_category=atlas_category_cache.get(parent),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert WARNO TGV textures directly to PNG, with optional channel extraction."
    )
    parser.add_argument("input", help="Input .tgv file or folder with .tgv files")
    parser.add_argument("output", nargs="?", help="Output .png path (for single file) or output folder")
    parser.add_argument(
        "--split",
        choices=("auto", "all", "none"),
        default="auto",
        help="Channel extraction mode (default: auto)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If input is a folder, search .tgv files recursively",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror textures horizontally before saving",
    )
    parser.add_argument(
        "--auto-naming",
        dest="auto_naming",
        action="store_true",
        default=True,
        help="Use atlas-based canonical names like Unit_D / Unit_NM (default: on)",
    )
    parser.add_argument(
        "--no-auto-naming",
        dest="auto_naming",
        action="store_false",
        help="Keep original source file names",
    )
    parser.add_argument(
        "--aggressive-split",
        action="store_true",
        help="Use more aggressive auto split heuristics",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        convert_path(
            Path(args.input),
            args.output,
            args.recursive,
            args.split,
            args.mirror,
            args.auto_naming,
            args.aggressive_split,
        )
    except Exception as exc:  # keep CLI output user-friendly
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

