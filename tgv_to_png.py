import argparse
import io
import json
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageOps
import zstandard as zstd


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
        dxgi = 99 if "SRGB" in fmt_up else 98
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
    if "BC" in info.fmt.upper():
        return decode_block_compressed(raw, info.width, info.height, info.fmt)
    return decode_uncompressed(raw, info.width, info.height, info.fmt)


def detect_texture_role(path: Path, fmt: str) -> str:
    name = path.stem.lower()
    fmt_up = fmt.upper()
    if "combinedda" in name or "coloralpha" in name:
        return "combined_da"
    if "normal" in name or "tscnm" in name or "BC5" in fmt_up:
        return "normal"
    if "combinedorm" in name or "combinedrm" in name or "_orm" in name or "ormtexture" in name or "rmtexture" in name:
        return "orm"
    return "generic"


def _norm_logical_ref(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    raw = raw.lstrip("/")
    if raw.lower().startswith("gamedata:/"):
        raw = raw[len("gamedata:/") :]
    elif raw.lower().startswith("gamedata:"):
        raw = raw[len("gamedata:") :]
    while "//" in raw:
        raw = raw.replace("//", "/")
    low = raw.lower()
    if low.startswith("pc/atlas/assets/"):
        raw = "Assets/" + raw[len("PC/Atlas/Assets/") :]
    return raw.lower()


def _canonical_channel_name(value: str) -> str:
    low = str(value or "").strip().lower()
    if not low:
        return "generic"
    if low in {"d", "diff", "albedo", "basecolor", "base_color", "diffuse"}:
        return "diffuse"
    if low in {"a", "alpha", "opacity", "mask"}:
        return "alpha"
    if low in {"nm", "normal", "normalmap"}:
        return "normal"
    if low in {"orm", "occlusionroughnessmetallic", "rma", "mrao"}:
        return "orm"
    if low in {"o", "ao", "occlusion", "ambientocclusion"}:
        return "occlusion"
    if low in {"r", "roughness", "rough"}:
        return "roughness"
    if low in {"m", "metal", "metalness", "metallic"}:
        return "metallic"
    if low in {"combined_da", "da"}:
        return "combined_da"
    return low


def _infer_channel_from_names(basename: str, logical_rel: str, source_role: str) -> str:
    for txt in (basename, logical_rel):
        low = str(txt or "").strip().lower()
        if not low:
            continue
        if low.endswith("_nm") or "_nm." in low or "normal" in low or "tscnm" in low:
            return "normal"
        if low.endswith("_orm") or "_orm." in low or "combinedorm" in low or "combinedrm" in low:
            return "orm"
        if low.endswith("_o") or "_o." in low:
            return "occlusion"
        if low.endswith("_ao") or "_ao." in low or "occlusion" in low:
            return "occlusion"
        if low.endswith("_r") or "_r." in low or "rough" in low:
            return "roughness"
        if low.endswith("_m") or "_m." in low or "metal" in low:
            return "metallic"
        if low.endswith("_a") or "_a." in low or "alpha" in low:
            return "alpha"
        if low.endswith("_d") or "_d." in low or "diffuse" in low or "color" in low:
            return "diffuse"
    src = _canonical_channel_name(source_role)
    return src if src != "generic" else "generic"


def _strip_texture_channel_suffix(stem: str) -> str:
    stem_up = str(stem or "")
    suffixes = ("_ORM", "_NM", "_DA", "_AO", "_OS", "_RS", "_D", "_A", "_R", "_M", "_O")
    for tag in suffixes:
        if stem_up.upper().endswith(tag):
            return stem_up[: -len(tag)]
    return stem_up


def _split_base_and_tag(stem: str) -> tuple[str, str | None]:
    stem_up = stem.upper()
    for tag in ("_ORM", "_NM", "_DA", "_AO", "_OS", "_RS", "_D", "_A", "_R", "_M", "_O"):
        if stem_up.endswith(tag):
            return stem[: -len(tag)], tag[1:]
    return stem, None


def _canonical_alias_names(target_basename: str, channel: str) -> list[str]:
    base, tag = _split_base_and_tag(str(target_basename or "").strip())
    canonical_tag_map = {
        "normal": "NM",
        "orm": "ORM",
        "alpha": "A",
        "roughness": "R",
        "metallic": "M",
        "occlusion": "O",
    }
    canonical_tag = canonical_tag_map.get(_canonical_channel_name(channel), "")
    if not canonical_tag:
        return []
    cur_tag = str(tag or "").upper()
    if cur_tag == canonical_tag:
        return []
    alias_base = base or str(target_basename or "").strip()
    if not alias_base:
        return []
    return [f"{alias_base}_{canonical_tag}"]


def _canonicalize_output_basename(target_basename: str, channel: str) -> str:
    base = str(target_basename or "").strip()
    if not base:
        return base
    if _canonical_channel_name(channel) == "diffuse" and base.upper().endswith("_D"):
        return base[:-2]
    if _canonical_channel_name(channel) == "occlusion" and base.upper().endswith("_AO"):
        return base[:-3] + "_O"
    return base


def _load_atlas_map_entries(atlas_map_path: Path, asset_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(Path(atlas_map_path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise RuntimeError(f"atlas map invalid root object: {atlas_map_path}")
    if int(data.get("schema_version", 0) or 0) != 1:
        raise RuntimeError(f"atlas map schema mismatch: {atlas_map_path}")
    textures = data.get("textures")
    if not isinstance(textures, list):
        raise RuntimeError(f"atlas map missing textures[]: {atlas_map_path}")

    asset_norm = _norm_logical_ref(asset_path)
    data_asset = _norm_logical_ref(str(data.get("asset_path", "")))
    if asset_norm and data_asset and data_asset != asset_norm:
        raise RuntimeError(
            f"atlas map asset mismatch: map={data.get('asset_path','')} requested={asset_path}"
        )

    out: list[dict[str, Any]] = []
    for tex in textures:
        if not isinstance(tex, dict):
            continue
        source_tgv_rel = _norm_logical_ref(str(tex.get("source_tgv_rel", "")))
        if not source_tgv_rel:
            continue
        source_role = _canonical_channel_name(str(tex.get("source_role", "")))
        source_asset_path = str(tex.get("source_asset_path", "")).strip()
        rect_raw = tex.get("crop_rect_px")
        if not isinstance(rect_raw, dict):
            continue
        rect = {
            "x": int(rect_raw.get("x", 0) or 0),
            "y": int(rect_raw.get("y", 0) or 0),
            "w": int(rect_raw.get("w", 0) or 0),
            "h": int(rect_raw.get("h", 0) or 0),
        }
        if rect["w"] <= 0 or rect["h"] <= 0:
            continue
        targets = tex.get("targets")
        if not isinstance(targets, list) or not targets:
            continue
        for tgt in targets:
            if not isinstance(tgt, dict):
                continue
            logical_rel = str(tgt.get("logical_rel", "")).strip()
            basename = str(tgt.get("basename", "")).strip()
            if not basename:
                basename = Path(_norm_logical_ref(logical_rel)).stem
            channel = _canonical_channel_name(str(tgt.get("channel", "")))
            if channel == "generic":
                channel = _infer_channel_from_names(basename, logical_rel, source_role)
            out.append(
                {
                    "source_tgv_rel": source_tgv_rel,
                    "source_role": source_role,
                    "source_asset_path": source_asset_path,
                    "crop_rect_px": rect,
                    "target_logical_rel": logical_rel,
                    "target_basename": basename,
                    "target_channel": channel,
                }
            )
    return out, data


def _logical_ref_matches(item_logical: str, requested_ref: str) -> bool:
    req = _norm_logical_ref(requested_ref)
    cur = _norm_logical_ref(item_logical)
    return bool(req and cur and cur == req)


def _clamp_rect_to_image(rect: dict[str, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w_img, h_img = int(size[0]), int(size[1])
    x = max(0, int(rect.get("x", 0)))
    y = max(0, int(rect.get("y", 0)))
    w = max(1, int(rect.get("w", 1)))
    h = max(1, int(rect.get("h", 1)))
    x2 = x + w
    y2 = y + h
    if x >= w_img or y >= h_img or x2 > w_img or y2 > h_img or x2 <= x or y2 <= y:
        raise RuntimeError(
            f"atlas crop is outside source image: rect=({x},{y},{w},{h}) image={w_img}x{h_img}"
        )
    return x, y, x2, y2


def _resolve_source_tgv_for_atlas(
    source_tgv_rel: str,
    source_file: Path | None,
    atlas_map_path: Path,
    search_roots: list[Path],
) -> Path:
    if source_file is not None and source_file.exists() and source_file.is_file():
        return source_file

    source_norm = _norm_logical_ref(source_tgv_rel)
    rel_under_assets = source_norm
    if rel_under_assets.startswith("assets/"):
        rel_under_assets = rel_under_assets[len("assets/") :]
    rel_path = Path(*PurePosixPath(rel_under_assets).parts)

    candidates: list[Path] = []
    for root in search_roots:
        base = Path(root)
        candidates.append(base / rel_path)
        candidates.append(base / "Assets" / rel_path)
        candidates.append(base / "PC" / "Atlas" / "Assets" / rel_path)

    cur = atlas_map_path.parent
    for _ in range(8):
        candidates.append(cur / rel_path)
        candidates.append(cur / "PC" / "Atlas" / "Assets" / rel_path)
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if cand.exists() and cand.is_file():
            return cand
    raise FileNotFoundError(f"atlas source tgv not found: {source_tgv_rel}")


def _out_path_for_target(
    out_dir: Path,
    target_logical_rel: str,
    target_basename: str,
    target_channel: str,
    only_logical_ref: str | None = None,
) -> Path:
    base_raw = str(target_basename or "").strip() or Path(_norm_logical_ref(target_logical_rel)).stem or "Texture"
    base = _canonicalize_output_basename(base_raw, target_channel)
    if str(only_logical_ref or "").strip():
        return out_dir / f"{base}.png"
    logical_norm = _norm_logical_ref(target_logical_rel)
    if logical_norm.startswith("assets/"):
        rel = Path(*PurePosixPath(logical_norm).parts[1:])
        return out_dir / rel.parent / f"{base}.png"
    return out_dir / f"{base}.png"


def _select_atlas_output_image(cropped: Image.Image, channel: str, source_role: str) -> Image.Image:
    chan = _canonical_channel_name(channel)
    src_role = _canonical_channel_name(source_role)

    if chan == "alpha":
        if cropped.mode in {"RGBA", "LA"}:
            return cropped.getchannel("A")
        return cropped.convert("L")

    if src_role == "orm":
        rgb = cropped.convert("RGB")
        r, g, b = rgb.split()
        if chan == "occlusion":
            return r
        if chan == "roughness":
            return g
        if chan == "metallic":
            return b
        if chan == "orm":
            return rgb

    if src_role == "combined_da" and chan == "diffuse":
        return cropped.convert("RGB")

    return cropped


def _invert_normal_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    if image.mode == "RGBA":
        rgb = image.convert("RGB")
        alpha = image.getchannel("A")
        inv = ImageOps.invert(rgb)
        inv.putalpha(alpha)
        return inv
    return ImageOps.invert(image.convert("RGB"))


def _image_has_alpha_data(image: Image.Image) -> bool:
    if image.mode not in {"RGBA", "LA"}:
        return False
    alpha = image.getchannel("A")
    mn, mx = alpha.getextrema()
    return mn < 255


def _save_channel_if_missing(path: Path, image: Image.Image, outputs: list[Path]) -> None:
    keyset = {str(p).lower() for p in outputs}
    if str(path).lower() in keyset:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    outputs.append(path)


def _write_orm_splits(base_name: str, out_main: Path, rgb: Image.Image, outputs: list[Path]) -> None:
    r, g, b = rgb.split()
    ao_path = out_main.with_name(f"{base_name}_O.png")
    rough_path = out_main.with_name(f"{base_name}_R.png")
    metal_path = out_main.with_name(f"{base_name}_M.png")
    _save_channel_if_missing(ao_path, r, outputs)
    _save_channel_if_missing(rough_path, g, outputs)
    _save_channel_if_missing(metal_path, b, outputs)
    legacy_ao = out_main.with_name(f"{base_name}_AO.png")
    try:
        if legacy_ao.exists() and legacy_ao.is_file():
            legacy_ao.unlink()
    except Exception:
        pass


def _write_conversion_manifest(
    out_path: Path,
    asset_path: str,
    atlas_map_path: Path,
    records: list[dict[str, Any]],
) -> Path:
    payload = {
        "schema_version": 1,
        "asset_path": str(asset_path),
        "atlas_map_path": str(atlas_map_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def convert_from_atlas_map(
    atlas_map_path: Path,
    asset_path: str,
    out_dir: Path,
    only_logical_ref: str | None = None,
    manifest_out: Path | None = None,
) -> None:
    entries, atlas_payload = _load_atlas_map_entries(atlas_map_path, asset_path)
    if only_logical_ref:
        entries = [e for e in entries if _logical_ref_matches(e.get("target_logical_rel", ""), str(only_logical_ref))]
    if not entries:
        raise RuntimeError("atlas map has no entries for selected asset/ref")

    out_dir.mkdir(parents=True, exist_ok=True)

    search_roots: list[Path] = []
    atlas_source = str(atlas_payload.get("atlas_source", "") or "").strip()
    if atlas_source:
        atlas_source_path = Path(atlas_source)
        parts = list(atlas_source_path.parts)
        for i in range(0, max(0, len(parts) - 2)):
            if str(parts[i]).lower() == "pc" and str(parts[i + 1]).lower() == "atlas" and str(parts[i + 2]).lower() == "assets":
                search_roots.append(Path(*parts[: i + 3]))
                break
        if atlas_source_path.parent:
            search_roots.append(atlas_source_path.parent)
        if atlas_source_path.parent.parent:
            search_roots.append(atlas_source_path.parent.parent)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in entries:
        source_rel = str(item.get("source_tgv_rel", "")).strip()
        if source_rel:
            grouped.setdefault(source_rel.lower(), []).append(item)
    if not grouped:
        raise RuntimeError("atlas map entries are missing source_tgv_rel")

    explicit_alpha_bases: set[str] = set()
    for it in entries:
        tgt_base = str(it.get("target_basename", "") or "").strip()
        if not tgt_base:
            continue
        base_name = _strip_texture_channel_suffix(tgt_base) or tgt_base
        target_channel = _canonical_channel_name(str(it.get("target_channel", "")))
        if target_channel == "alpha":
            explicit_alpha_bases.add(base_name.lower())

    records: list[dict[str, Any]] = []
    total_saved = 0
    orm_split_done: set[str] = set()

    for _, items in grouped.items():
        source_rel = str(items[0].get("source_tgv_rel", "")).strip()
        src_tgv = _resolve_source_tgv_for_atlas(
            source_tgv_rel=source_rel,
            source_file=None,
            atlas_map_path=atlas_map_path,
            search_roots=search_roots,
        )
        info = parse_tgv(src_tgv)
        mip_idx, offset, size, raw_size = pick_fullres_mip(info)
        raw = decompress_mip(info, offset, size, raw_size)
        decoded = decode_tgv_image(info, raw)
        src_role = detect_texture_role(src_tgv, info.fmt)

        for item in items:
            rect = _clamp_rect_to_image(item.get("crop_rect_px", {}), decoded.size)
            cropped = decoded.crop(rect)
            channel = _canonical_channel_name(str(item.get("target_channel", "")).strip())
            source_role = _canonical_channel_name(str(item.get("source_role", "")).strip())
            if channel == "generic":
                channel = source_role
            if channel == "generic":
                channel = _canonical_channel_name(src_role)

            out_main = _out_path_for_target(
                out_dir=out_dir,
                target_logical_rel=str(item.get("target_logical_rel", "")),
                target_basename=str(item.get("target_basename", "")),
                target_channel=channel,
                only_logical_ref=only_logical_ref,
            )
            out_main.parent.mkdir(parents=True, exist_ok=True)
            out_main_img = _select_atlas_output_image(cropped, channel, source_role if source_role != "generic" else src_role)
            if channel == "normal":
                out_main_img = _invert_normal_rgb(out_main_img)
            out_main_img.save(out_main)

            outputs: list[Path] = [out_main]
            alias_paths: list[Path] = []

            for alias_name in _canonical_alias_names(out_main.stem, channel):
                alias = out_main.with_name(f"{alias_name}{out_main.suffix}")
                _save_channel_if_missing(alias, out_main_img, outputs)
                if str(alias).lower() != str(out_main).lower():
                    alias_paths.append(alias)

            base_name = _strip_texture_channel_suffix(out_main.stem) or out_main.stem
            base_key = base_name.lower()

            if channel == "diffuse" and _image_has_alpha_data(cropped) and base_key not in explicit_alpha_bases:
                alpha = cropped.getchannel("A") if cropped.mode in {"RGBA", "LA"} else cropped.convert("L")
                alpha_path = out_main.with_name(f"{base_name}_A.png")
                _save_channel_if_missing(alpha_path, alpha, outputs)
            if channel == "diffuse":
                legacy_d = out_main.with_name(f"{base_name}_D.png")
                try:
                    if legacy_d.exists() and legacy_d.is_file():
                        legacy_d.unlink()
                except Exception:
                    pass

            if (source_role == "orm" or src_role == "orm") and channel in {"occlusion", "roughness", "metallic", "orm"}:
                split_key = f"{str(out_main.parent).lower()}::{base_key}"
                if split_key not in orm_split_done:
                    orm_split_done.add(split_key)
                    if channel == "orm" and out_main.stem.upper().endswith("_ORM"):
                        orm_main = out_main
                        orm_rgb = out_main_img.convert("RGB")
                    else:
                        orm_main = out_main.with_name(f"{base_name}_ORM.png")
                        orm_rgb = cropped.convert("RGB")
                        _save_channel_if_missing(orm_main, orm_rgb, outputs)
                    _write_orm_splits(base_name, orm_main, orm_rgb, outputs)

            if channel == "occlusion":
                legacy_ao = out_main.with_name(f"{base_name}_AO.png")
                try:
                    if legacy_ao.exists() and legacy_ao.is_file():
                        legacy_ao.unlink()
                except Exception:
                    pass

            total_saved += len(outputs)
            print(
                f"[OK] atlas {Path(str(item.get('target_logical_rel',''))).name or out_main.name} -> {out_main.name} | "
                f"src={src_tgv.name} fmt={info.fmt} fullMip={mip_idx}/{info.mip_count - 1} channel={channel}"
            )
            for p in outputs[1:]:
                print(f"     + {p.name}")

            records.append(
                {
                    "source_tgv_rel": str(item.get("source_tgv_rel", "")),
                    "source_asset_path": str(item.get("source_asset_path", "")),
                    "target_logical_rel": str(item.get("target_logical_rel", "")),
                    "basename": out_main.stem,
                    "channel": channel,
                    "primary": str(out_main),
                    "outputs": [str(p) for p in outputs],
                    "aliases": [str(p) for p in alias_paths],
                }
            )

    manifest_path = _write_conversion_manifest(
        out_path=Path(manifest_out) if manifest_out is not None else (out_dir / "conversion_manifest.json"),
        asset_path=asset_path,
        atlas_map_path=atlas_map_path,
        records=records,
    )
    print(
        f"[OK] atlas conversion complete | entries={len(records)} outputs={total_saved} "
        f"manifest={manifest_path.name}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic WARNO TGV->PNG converter (Atlas JSON mode only)."
    )
    parser.add_argument("--atlas-map", required=True, help="Atlas JSON mapping file")
    parser.add_argument("--asset-path", required=True, help="Asset path in Assets/... form")
    parser.add_argument("--out-dir", required=True, help="Output directory for atlas-driven conversion")
    parser.add_argument("--only-logical-ref", default="", help="Optional exact logical texture ref to convert")
    parser.add_argument("--manifest-out", default="", help="Optional output path for conversion_manifest.json")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        only_logical_ref = str(args.only_logical_ref or "").strip() or None
        manifest_out = str(args.manifest_out or "").strip()
        convert_from_atlas_map(
            atlas_map_path=Path(str(args.atlas_map).strip()),
            asset_path=str(args.asset_path).strip(),
            out_dir=Path(str(args.out_dir).strip()),
            only_logical_ref=only_logical_ref,
            manifest_out=Path(manifest_out) if manifest_out else None,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
