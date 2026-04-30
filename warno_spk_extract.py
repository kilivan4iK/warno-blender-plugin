#!/usr/bin/env python3
"""
WARNO SPK mesh extractor (direct SPK -> OBJ).

This tool parses MESH/PCPC .spk files and exports selected models as OBJ.
If Atlas path is provided, it also resolves texture references and builds OBJ+MTL.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import math
import mmap
import os
import re
import shutil
import subprocess
import struct
import sys
import tempfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence, Tuple

import zstandard


Entry = Tuple[str, Dict[str, Any]]
WARNO_DEV_SCALE = 20.0 / 43.0
WARNO_OFF_MAT_SCALE = WARNO_DEV_SCALE / 100.0


@dataclass
class RawNodeTransform:
    source: str
    block_index: int
    translation: List[float]
    rotation_basis: List[List[float]]
    scale: List[float]
    matrix_local: List[List[float]]
    world_translation: List[float] | None = None


@dataclass
class RawNodeRecord:
    index: int
    raw_name: str
    safe_name: str
    parent_index: int
    off_mat_points: List[List[float]] = field(default_factory=list)
    off_mat_blocks: List[List[float]] = field(default_factory=list)
    own_transform_candidates: List[RawNodeTransform] = field(default_factory=list)
    stream_transform_candidates: List[RawNodeTransform] = field(default_factory=list)
    local_transform: RawNodeTransform | None = None
    world_translation: List[float] | None = None
    deterministic_miss: str = ""


@dataclass
class RawSceneGraph:
    node_index: int
    node_count: int
    stream_start_index: int
    stream_end_index: int
    records: List[RawNodeRecord]


def _norm_low(value: str) -> str:
    return str(value or "").strip().lower()


def _safe_name_text(name: str) -> str:
    raw = SAFE_MATERIAL_RX.sub("_", str(name or "").strip())
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or str(name or "").strip()


def _mat4_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            out[row][col] = sum(float(a[row][k]) * float(b[k][col]) for k in range(4))
    return out


def _mat4_translation(mat: Sequence[Sequence[float]]) -> List[float]:
    return [
        float(mat[0][3]),
        float(mat[1][3]),
        float(mat[2][3]),
    ]


def _vec_len3(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values[:3]))


def _decompose_affine_matrix_rows(local_matrix: Sequence[Sequence[float]]) -> Tuple[List[float], List[List[float]], List[float]] | None:
    if not isinstance(local_matrix, (list, tuple)) or len(local_matrix) < 4:
        return None
    basis_cols = [
        [float(local_matrix[0][0]), float(local_matrix[1][0]), float(local_matrix[2][0])],
        [float(local_matrix[0][1]), float(local_matrix[1][1]), float(local_matrix[2][1])],
        [float(local_matrix[0][2]), float(local_matrix[1][2]), float(local_matrix[2][2])],
    ]
    scale = [_vec_len3(col) for col in basis_cols]
    rotation_cols: List[List[float]] = []
    for axis_i, col in enumerate(basis_cols):
        axis_scale = float(scale[axis_i])
        if axis_scale > 1.0e-9:
            rotation_cols.append([float(v) / axis_scale for v in col])
        else:
            unit = [0.0, 0.0, 0.0]
            unit[axis_i] = 1.0
            rotation_cols.append(unit)
    rotation_rows = [
        [rotation_cols[0][0], rotation_cols[1][0], rotation_cols[2][0]],
        [rotation_cols[0][1], rotation_cols[1][1], rotation_cols[2][1]],
        [rotation_cols[0][2], rotation_cols[1][2], rotation_cols[2][2]],
    ]
    translation = _mat4_translation(local_matrix)
    return translation, rotation_rows, scale


def _compose_affine_matrix_rows(
    translation: Sequence[float],
    rotation_basis: Sequence[Sequence[float]],
    scale: Sequence[float],
) -> List[List[float]]:
    rows = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for row in range(3):
        rows[row][3] = float(translation[row]) if row < len(translation) else 0.0
        for col in range(3):
            basis = 0.0
            try:
                basis = float(rotation_basis[row][col])
            except Exception:
                basis = 1.0 if row == col else 0.0
            scl = float(scale[col]) if col < len(scale) else 1.0
            rows[row][col] = basis * scl
    rows[3][3] = 1.0
    return rows


def _off_mat_change_of_basis_rows(variant_name: str) -> List[List[float]] | None:
    variant_low = _norm_low(variant_name)
    legacy = {
        "current": ("xyz", (-1.0, 1.0, -1.0)),
        "identity": ("xyz", (1.0, 1.0, 1.0)),
        "flip_xy": ("xyz", (-1.0, -1.0, 1.0)),
        "flip_xz": ("xyz", (-1.0, 1.0, 1.0)),
    }
    perm = ""
    signs: Tuple[float, float, float] | None = None
    if variant_low in legacy:
        perm, signs = legacy[variant_low]
    else:
        match = re.fullmatch(r"([xyz]{3}):([+-]1)([+-]1)([+-]1)", variant_low)
        if match:
            perm = str(match.group(1))
            signs = (
                -1.0 if match.group(2).startswith("-") else 1.0,
                -1.0 if match.group(3).startswith("-") else 1.0,
                -1.0 if match.group(4).startswith("-") else 1.0,
            )
    if not perm or signs is None:
        return None
    axis_index = {"x": 0, "y": 1, "z": 2}
    rows = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for out_axis_i, (out_axis, sign) in enumerate(zip(perm, signs)):
        rows[out_axis_i][axis_index[out_axis]] = float(sign) * float(WARNO_OFF_MAT_SCALE)
    rows[3][3] = 1.0
    return rows


def _mat4_inverse_affine(mat: Sequence[Sequence[float]]) -> List[List[float]] | None:
    try:
        r = [
            [float(mat[0][0]), float(mat[0][1]), float(mat[0][2])],
            [float(mat[1][0]), float(mat[1][1]), float(mat[1][2])],
            [float(mat[2][0]), float(mat[2][1]), float(mat[2][2])],
        ]
        t = [float(mat[0][3]), float(mat[1][3]), float(mat[2][3])]
    except Exception:
        return None
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det) <= 1.0e-12:
        return None
    inv_det = 1.0 / det
    rinv = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
        ],
    ]
    tinv = [
        -(rinv[0][0] * t[0] + rinv[0][1] * t[1] + rinv[0][2] * t[2]),
        -(rinv[1][0] * t[0] + rinv[1][1] * t[1] + rinv[1][2] * t[2]),
        -(rinv[2][0] * t[0] + rinv[2][1] * t[1] + rinv[2][2] * t[2]),
    ]
    out = [
        [rinv[0][0], rinv[0][1], rinv[0][2], tinv[0]],
        [rinv[1][0], rinv[1][1], rinv[1][2], tinv[1]],
        [rinv[2][0], rinv[2][1], rinv[2][2], tinv[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return out


def _off_mat_stream_section_bounds(node_count: int) -> Tuple[int, int]:
    if int(node_count) <= 0:
        return 0, 0
    stream_count = int(math.ceil(float(node_count) / 3.0))
    start = stream_count
    end = min(int(node_count), start + stream_count)
    return max(0, int(start)), max(0, int(end))


def _off_mat_stream_target_index(source_idx: int, block_idx: int, node_count: int) -> int:
    if int(node_count) <= 0:
        return -1
    return int((int(source_idx) * 3 + int(block_idx)) % int(node_count))


def _off_mat_block_matrix_rows(block: Sequence[float]) -> List[List[float]] | None:
    if not isinstance(block, (list, tuple)) or len(block) < 12:
        return None
    vals = [float(v) for v in block[:12]]
    return [
        [vals[0], vals[1], vals[2], vals[3]],
        [vals[4], vals[5], vals[6], vals[7]],
        [vals[8], vals[9], vals[10], vals[11]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _off_mat_convert_matrix_rows(raw_matrix: Sequence[Sequence[float]], variant_name: str) -> List[List[float]] | None:
    change = _off_mat_change_of_basis_rows(variant_name)
    if change is None:
        return None
    change_inv = _mat4_inverse_affine(change)
    if change_inv is None:
        return None
    return _mat4_mul(_mat4_mul(change, raw_matrix), change_inv)

LAYOUT_196 = [
    ("vertexFormats", 3),
    ("material", 3),
    ("mesh", 3),
    ("drawCalls", 3),
    ("indexTable", 3),
    ("indexData", 2),
    ("vertexTable", 3),
    ("vertexData", 2),
    ("nodeTable", 3),
    ("nodeData", 2),
]

LAYOUT_244_256 = [
    ("vertexFormats", 3),
    ("material", 3),
    ("block3", 3),
    ("block4", 3),
    ("mesh", 3),
    ("drawCalls", 3),
    ("indexTable", 3),
    ("indexData", 2),
    ("vertexTable", 3),
    ("vertexData", 2),
    ("nodeTable", 3),
    ("nodeData", 2),
    ("block5", 3),
    ("block6", 3),
]

SAFE_SEGMENT_RX = re.compile(r'[<>:"|?*]')
SAFE_MATERIAL_RX = re.compile(r"[^A-Za-z0-9_.-]+")
ATLAS_PATH_RX = re.compile(
    rb"(?:ZZ:)?/PC/Atlas/Assets/[A-Za-z0-9_./\\-]+?\.png",
    re.IGNORECASE,
)
LOD_SUFFIX_RX = re.compile(r"_(LOW|MID|HIGH|LOD[0-9]+)$", re.IGNORECASE)
NODE_NAME_TAIL_RX = re.compile(rb"[A-Za-z0-9_.]{16,}$")
NODE_NAME_TOKEN_START_RX = re.compile(
    r"(?i)(?=(fx_|roue_|wheel_|chenille_|track_|tourelle_|turret_|canon_|gun_|barrel_|axe_|chassis|hull|base_|bloc_|helice_|driver|tireur_|remorque|transport_|window))"
)
NDF_BBOX_BONE_RX = re.compile(r"""GameplayBBoxBoneName\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
NDF_REF_MESH_RX = re.compile(r"""ReferenceMesh\s*=\s*\$/GFX/DepictionResources/([A-Za-z0-9_]+)""", re.IGNORECASE)
NDF_EXPORT_CONST_RX = re.compile(
    r"""^\s*export\s+([A-Za-z0-9_]+)\s+is\s+["'](?:GameData:)?/?([^"']+?\.fbx)["']\s*$""",
    re.IGNORECASE,
)
NDF_EXPORT_TRESOURCE_RX = re.compile(r"""^\s*export\s+([A-Za-z0-9_]+)\s+is\s+TResourceMesh\b""", re.IGNORECASE)
NDF_MESH_ASSIGN_RX = re.compile(r"""^\s*Mesh\s*=\s*(.+?)\s*(?:,)?\s*$""", re.IGNORECASE)
NDF_COATING_RX = re.compile(r"""CoatingName\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE)
NDF_OPERATOR_UNIT_RX = re.compile(r"""^\s*DepictionOperator_([A-Za-z0-9_]+)_Weapon[0-9]+\s+is\b""", re.IGNORECASE)
NDF_STRING_RX = re.compile(r"""["']([^"']+)["']""")
TRACK_TOKEN_RX = re.compile(r"(?i)(?:^|_)(trk|track|chenille)(?:_|$)")
DEPICTION_OPERATOR_NAME_RX = re.compile(r"(DepictionOperator_[A-Za-z0-9_]+)", re.IGNORECASE)
DEPICTION_OPERATOR_ASSIGN_RX = re.compile(r"""^\s*(?:template\s+)?(DepictionOperator_[A-Za-z0-9_]+)\b""", re.IGNORECASE)
DEPICTION_NODE_ASSIGN_RX = re.compile(r"""([A-Za-z0-9_]+)\s*=\s*['"]([^'"]+)['"]""")
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def normalize_asset_path(path: str) -> str:
    path = path.replace("\\", "/")
    path = re.sub(r"/+", "/", path)
    if path.startswith("./"):
        path = path[2:]
    return path.strip("/")


def parse_selection(raw: str, max_index: int) -> List[int]:
    raw = raw.strip().lower()
    if raw in {"all", "a", "*"}:
        return list(range(max_index))

    indices: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            l = int(left)
            r = int(right)
            if l > r:
                l, r = r, l
            for x in range(l, r + 1):
                if x < 1 or x > max_index:
                    raise ValueError(f"Index out of range: {x}")
                indices.add(x - 1)
        else:
            x = int(part)
            if x < 1 or x > max_index:
                raise ValueError(f"Index out of range: {x}")
            indices.add(x - 1)
    if not indices:
        raise ValueError("No valid index selected")
    return sorted(indices)


def print_matches(matches: Sequence[Entry], limit: int | None = None) -> None:
    show = list(matches)
    if limit is not None and limit > 0:
        show = show[:limit]

    for i, (asset, meta) in enumerate(show, start=1):
        print(
            f"{i:4d}. mesh={int(meta.get('meshIndex', -1)):5d} "
            f"node={int(meta.get('nodeIndex', -1)):5d}  {asset}"
        )
    if len(show) < len(matches):
        print(f"... {len(matches) - len(show)} more")


def safe_output_relpath(asset_path: str) -> Path:
    cleaned = normalize_asset_path(asset_path)
    parts = []
    for part in Path(cleaned).parts:
        if part in {"", ".", ".."}:
            continue
        parts.append(SAFE_SEGMENT_RX.sub("_", part))
    if not parts:
        raise ValueError(f"Cannot build output path from asset: {asset_path}")
    rel = Path(*parts)
    return rel.with_suffix(".obj")


def zlib_decode_compat(data: bytes) -> bytes:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            pass
    raise ValueError("Failed to zlib-decompress buffer")


def build_rotation_params(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    mirror_y: bool = False,
) -> Dict[str, float]:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    return {
        "rx_deg": rx_deg,
        "ry_deg": ry_deg,
        "rz_deg": rz_deg,
        "cosx": math.cos(rx),
        "sinx": math.sin(rx),
        "cosy": math.cos(ry),
        "siny": math.sin(ry),
        "cosz": math.cos(rz),
        "sinz": math.sin(rz),
        "mirror_y": 1.0 if mirror_y else 0.0,
    }


def apply_rotation(x: float, y: float, z: float, rot: Dict[str, float]) -> Tuple[float, float, float]:
    # X axis
    y, z = (y * rot["cosx"] - z * rot["sinx"]), (y * rot["sinx"] + z * rot["cosx"])
    # Y axis
    x, z = (x * rot["cosy"] + z * rot["siny"]), (-x * rot["siny"] + z * rot["cosy"])
    # Z axis
    x, y = (x * rot["cosz"] - y * rot["sinz"]), (x * rot["sinz"] + y * rot["cosz"])
    if rot.get("mirror_y", 0.0):
        y = -y
    return x, y, z


def normalize_atlas_ref(path: str) -> str:
    s = path.replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    i = s.lower().find("/assets/")
    if i != -1:
        return normalize_asset_path("Assets" + s[i + 7 :])
    if s.lower().startswith("assets/"):
        return normalize_asset_path(s)
    return normalize_asset_path(s)


def strip_lod_suffix(name: str) -> str:
    out = name
    while True:
        nxt = LOD_SUFFIX_RX.sub("", out)
        if nxt == out:
            return out
        out = nxt


def shared_suffix_score(a: str, b: str) -> int:
    pa = [x.lower() for x in PurePosixPath(normalize_asset_path(a)).parts]
    pb = [x.lower() for x in PurePosixPath(normalize_asset_path(b)).parts]
    i = 1
    score = 0
    while i <= len(pa) and i <= len(pb):
        if pa[-i] != pb[-i]:
            break
        score += 1
        i += 1
    return score


def unique_keep_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def is_track_token(text: str) -> bool:
    return bool(TRACK_TOKEN_RX.search(str(text or "")))


def _classify_track_role_from_text(text: str) -> str:
    low = str(text or "").strip().lower().replace("\\", "/")
    if not low:
        return ""

    # Some assets use droite_2 for the second (left) track material.
    if "chenille_droite_2" in low or "chenille_droite.2" in low:
        return "track_left"

    # Explicit left/right markers from source naming.
    if (
        "chenille_gauche" in low
        or "track_left" in low
        or "left_track" in low
        or "tracks_2" in low
        or "trk_2" in low
    ):
        return "track_left"
    if (
        "chenille_droite" in low
        or "track_right" in low
        or "right_track" in low
    ):
        return "track_right"
    if "tracks" in low or "trk" in low:
        # WARNO convention usually uses base tracks as right and *_2 as left.
        if "tracks_2" in low or "trk_2" in low:
            return "track_left"
        return "track_right"
    return ""


def derive_material_roles_from_source(
    material_name_by_id: Dict[int, str] | None,
    material_texture_names_by_id: Dict[int, Dict[str, str]] | None = None,
) -> Dict[int, str]:
    names_by_id = material_name_by_id or {}
    slots_by_id = material_texture_names_by_id or {}
    out: Dict[int, str] = {}

    mids: set[int] = set()
    mids.update(int(k) for k in names_by_id.keys())
    mids.update(int(k) for k in slots_by_id.keys())

    for mid in sorted(mids):
        role = ""
        name = str(names_by_id.get(int(mid), "")).strip()
        if name:
            role = _classify_track_role_from_text(name)
        if not role:
            slot_map = slots_by_id.get(int(mid), {})
            if isinstance(slot_map, dict):
                for slot_name, slot_value in slot_map.items():
                    role = _classify_track_role_from_text(str(slot_name or ""))
                    if role:
                        break
                    role = _classify_track_role_from_text(str(slot_value or ""))
                    if role:
                        break
        out[int(mid)] = role or "other"
    return out


def read_png_size(path: Path) -> Tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            head = f.read(24)
        if len(head) < 24 or head[:8] != PNG_SIG:
            return None
        if head[12:16] != b"IHDR":
            return None
        w = int.from_bytes(head[16:20], byteorder="big", signed=False)
        h = int.from_bytes(head[20:24], byteorder="big", signed=False)
        if w <= 0 or h <= 0:
            return None
        return (w, h)
    except Exception:
        return None


def split_concatenated_node_names(blob: str) -> List[str]:
    raw = blob.strip("_")
    if not raw:
        return []

    starts = [0]
    for m in NODE_NAME_TOKEN_START_RX.finditer(raw):
        starts.append(m.start())
    starts = sorted({x for x in starts if 0 <= x < len(raw)})
    if starts[0] != 0:
        starts.insert(0, 0)

    out: List[str] = []
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < len(starts) else len(raw)
        token = raw[st:en].strip("_")
        if len(token) < 2:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.]+", token):
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        out.append(token)
    if not out and len(raw) >= 2:
        out = [raw]
    return out


def normalize_gamedata_asset_path(path: str) -> str:
    s = str(path or "").strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    s = re.sub(r"(?i)^gamedata:/", "", s)
    s = re.sub(r"(?i)^zz:", "", s)
    s = s.lstrip("/")
    return normalize_asset_path(s)


def extract_units_country_code(asset_path: str) -> str:
    parts = [p.lower() for p in PurePosixPath(normalize_asset_path(asset_path)).parts]
    for i, part in enumerate(parts):
        if part == "units" and i + 1 < len(parts):
            country = parts[i + 1]
            if re.fullmatch(r"[a-z]{2,4}", country):
                return country
    return ""


def _ndf_extract_quoted_values(raw: str) -> List[str]:
    out: List[str] = []
    for m in NDF_STRING_RX.finditer(str(raw or "")):
        token = m.group(1).strip()
        if token:
            out.append(token)
    return out


class UnitNdfHintsResolver:
    def __init__(self, unit_ndfbin: Path):
        self.unit_ndfbin = unit_ndfbin
        self._loaded = False
        self._load_error = ""
        self._source_files: List[str] = []
        self._asset_bbox_bones: Dict[str, set[str]] = {}
        self._coating_anchors: Dict[str, set[str]] = {}
        self._operator_anchors: Dict[str, set[str]] = {}
        self._depiction_units: Dict[str, Dict[str, Any]] = {}

    @property
    def is_ready(self) -> bool:
        return self._loaded and not self._load_error

    @property
    def source_files(self) -> List[str]:
        return list(self._source_files)

    @property
    def load_error(self) -> str:
        return self._load_error

    def _guess_game_root(self) -> Path | None:
        p = self.unit_ndfbin
        try:
            p = p.resolve()
        except Exception:
            pass
        for parent in [p.parent, *p.parents]:
            if (parent / "Mods").exists():
                return parent
        if len(p.parents) >= 4:
            return p.parents[3]
        return None

    @staticmethod
    def _pick_latest(paths: Sequence[Path]) -> Path | None:
        valid = [p for p in paths if p.exists() and p.is_file()]
        if not valid:
            return None
        return max(valid, key=lambda x: (x.stat().st_mtime, -len(str(x))))

    @staticmethod
    def _find_parent_named(path: Path, name: str) -> Path | None:
        target = name.lower()
        for p in [path, *path.parents]:
            if p.name.lower() == target:
                return p
        return None

    def _discover_unite_descriptor(self, game_root: Path) -> Path | None:
        # Prefer official unpacked base data (requested by user) over modded mirrors.
        direct_candidates = [
            game_root
            / "Mods"
            / "ModData"
            / "base"
            / "GameData"
            / "Generated"
            / "Gameplay"
            / "Gfx"
            / "UniteDescriptor.ndf",
            game_root
            / "ModData"
            / "base"
            / "GameData"
            / "Generated"
            / "Gameplay"
            / "Gfx"
            / "UniteDescriptor.ndf",
            game_root
            / "base"
            / "GameData"
            / "Generated"
            / "Gameplay"
            / "Gfx"
            / "UniteDescriptor.ndf",
            game_root
            / "GameData"
            / "Generated"
            / "Gameplay"
            / "Gfx"
            / "UniteDescriptor.ndf",
            game_root
            / "Generated"
            / "Gameplay"
            / "Gfx"
            / "UniteDescriptor.ndf",
        ]
        for preferred in direct_candidates:
            if preferred.exists() and preferred.is_file():
                return preferred

        cands: List[Path] = []
        for mods_name in ("Mods", "Mods_old"):
            mods_root = game_root / mods_name
            if not mods_root.exists():
                continue
            for p in mods_root.rglob("UniteDescriptor.ndf"):
                low = str(p).replace("\\", "/").lower()
                if "/gamedata/generated/gameplay/gfx/unitedescriptor.ndf" in low:
                    cands.append(p)
        if cands:
            base_like = [
                p for p in cands if "/mods/moddata/base/" in str(p).replace("\\", "/").lower()
            ]
            if base_like:
                return self._pick_latest(base_like)
        if game_root.exists() and game_root.is_dir():
            loose = [
                p
                for p in game_root.rglob("UniteDescriptor.ndf")
                if "/gamedata/generated/gameplay/gfx/unitedescriptor.ndf"
                in str(p).replace("\\", "/").lower()
            ]
            if loose:
                base_like = [
                    p for p in loose if "/mods/moddata/base/" in str(p).replace("\\", "/").lower()
                ]
                if base_like:
                    return self._pick_latest(base_like)
                return self._pick_latest(loose)
        return self._pick_latest(cands)

    def _parse_unite_descriptor(self, path: Path) -> Dict[str, set[str]]:
        refmesh_bbox: Dict[str, set[str]] = {}
        pending_bbox = ""
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_bbox = NDF_BBOX_BONE_RX.search(line)
                if m_bbox:
                    pending_bbox = str(m_bbox.group(1)).strip()
                m_ref = NDF_REF_MESH_RX.search(line)
                if m_ref:
                    ref_name = str(m_ref.group(1)).strip().lower()
                    if not ref_name:
                        continue
                    if pending_bbox:
                        refmesh_bbox.setdefault(ref_name, set()).add(pending_bbox.lower())
        return refmesh_bbox

    @staticmethod
    def _resolve_mesh_expr(expr: str, constants: Dict[str, str]) -> str:
        e = str(expr or "").split("//", 1)[0].strip().rstrip(",")
        if not e:
            return ""
        qm = re.match(r"""^["'](?:GameData:)?/?([^"']+?\.fbx)["']$""", e, re.IGNORECASE)
        if qm:
            return normalize_gamedata_asset_path(qm.group(1)).lower()
        sym = re.match(r"^([A-Za-z0-9_]+)$", e)
        if sym:
            return constants.get(sym.group(1).lower(), "")
        return ""

    def _parse_depiction_resource_file(self, path: Path, out_map: Dict[str, set[str]]) -> None:
        constants: Dict[str, str] = {}
        current_export = ""
        current_mesh_expr = ""
        in_resource = False

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                m_const = NDF_EXPORT_CONST_RX.match(line)
                if m_const:
                    export_name = m_const.group(1).strip().lower()
                    asset = normalize_gamedata_asset_path(m_const.group(2)).lower()
                    if export_name and asset:
                        constants[export_name] = asset
                    continue

                m_res = NDF_EXPORT_TRESOURCE_RX.match(line)
                if m_res:
                    current_export = m_res.group(1).strip().lower()
                    current_mesh_expr = ""
                    in_resource = True
                    continue

                if not in_resource:
                    continue

                m_mesh = NDF_MESH_ASSIGN_RX.match(line)
                if m_mesh:
                    current_mesh_expr = m_mesh.group(1).strip()

                if re.match(r"^\)\s*$", line):
                    if current_export:
                        asset = self._resolve_mesh_expr(current_mesh_expr, constants)
                        if asset:
                            out_map.setdefault(current_export, set()).add(asset)
                    current_export = ""
                    current_mesh_expr = ""
                    in_resource = False

    def _parse_depiction_resources(self, root: Path) -> Dict[str, set[str]]:
        out_map: Dict[str, set[str]] = {}
        if not root.exists() or not root.is_dir():
            return out_map

        for ndf_path in root.rglob("*.ndf"):
            self._parse_depiction_resource_file(ndf_path, out_map)
        return out_map

    def _parse_depictions_anchors(self, depictions_file: Path) -> Tuple[Dict[str, set[str]], Dict[str, set[str]]]:
        coating_anchors: Dict[str, set[str]] = {}
        operator_anchors: Dict[str, set[str]] = {}
        if not depictions_file.exists() or not depictions_file.is_file():
            return coating_anchors, operator_anchors

        current_coating = ""
        current_operator_unit = ""
        in_operator_block = False
        in_anchors_list = False

        def register(anchors: Sequence[str]) -> None:
            clean = [a.strip().lower() for a in anchors if str(a).strip()]
            if not clean:
                return
            if current_coating:
                coating_anchors.setdefault(current_coating, set()).update(clean)
            if in_operator_block and current_operator_unit:
                operator_anchors.setdefault(current_operator_unit, set()).update(clean)

        with depictions_file.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                if line.lower().startswith("unnamed tacticvehicledepictionregistration"):
                    current_coating = ""

                m_op = NDF_OPERATOR_UNIT_RX.match(line)
                if m_op:
                    current_operator_unit = m_op.group(1).strip().lower()
                    in_operator_block = True

                m_coating = NDF_COATING_RX.search(line)
                if m_coating:
                    current_coating = m_coating.group(1).strip().lower()

                anchors_found: List[str] = []
                if "anchor" in line.lower():
                    anchors_found.extend(_ndf_extract_quoted_values(line))

                if "anchors" in line.lower() and "[" in line and "]" not in line:
                    in_anchors_list = True
                if in_anchors_list:
                    anchors_found.extend(_ndf_extract_quoted_values(line))
                    if "]" in line:
                        in_anchors_list = False

                if anchors_found:
                    register(anchors_found)

                if in_operator_block and re.match(r"^\)\s*$", line):
                    in_operator_block = False

        return coating_anchors, operator_anchors

    def _parse_depictions_structured(self, depictions_file: Path) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not depictions_file.exists() or not depictions_file.is_file():
            return out

        pending_registration = False
        in_registration = False
        reg_depth = 0
        current_key = ""
        current_anchors: set[str] = set()
        current_nonfx_anchors: set[str] = set()
        current_operators: List[str] = []

        def flush_registration() -> None:
            nonlocal current_key, current_anchors, current_nonfx_anchors, current_operators
            key = str(current_key or "").strip().lower()
            if not key:
                current_anchors = set()
                current_nonfx_anchors = set()
                current_operators = []
                return
            ops_unique = unique_keep_order(
                [str(x).strip() for x in current_operators if str(x).strip()]
            )
            track_kind = ""
            joined_ops = " ".join(ops_unique).lower()
            if "propulsion_continuoustrack" in joined_ops:
                track_kind = "continuous_track"
            elif "propulsion_wheels" in joined_ops:
                track_kind = "wheels"
            elif "propulsion_" in joined_ops:
                track_kind = "other"
            out[key] = {
                "matched_unit": key,
                "operators": ops_unique,
                "weapon_fx_anchors": unique_keep_order(
                    sorted(a for a in current_anchors if a.startswith("fx_"))
                ),
                "subdepictions": (
                    [{"anchors": unique_keep_order(sorted(current_nonfx_anchors)), "operators": []}]
                    if current_nonfx_anchors
                    else []
                ),
                "track_kind": track_kind,
            }
            current_key = ""
            current_anchors = set()
            current_nonfx_anchors = set()
            current_operators = []

        with depictions_file.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                low = line.lower()
                if not in_registration and low.startswith("unnamed tacticvehicledepictionregistration"):
                    pending_registration = True
                    current_key = ""
                    current_anchors = set()
                    current_nonfx_anchors = set()
                    current_operators = []
                    continue

                if pending_registration and "(" in line:
                    pending_registration = False
                    in_registration = True
                    reg_depth = raw.count("(") - raw.count(")")
                    continue

                if not in_registration:
                    continue

                m_key = re.search(r"""BlackHoleKey\s*=\s*['"]([^'"]+)['"]""", line, re.IGNORECASE)
                if m_key:
                    current_key = str(m_key.group(1)).strip().lower()

                if "anchor" in low:
                    for anchor in _ndf_extract_quoted_values(line):
                        clean = str(anchor).strip().lower()
                        if not clean:
                            continue
                        current_anchors.add(clean)
                        if not clean.startswith("fx_"):
                            current_nonfx_anchors.add(clean)

                current_operators.extend(re.findall(r"DepictionOperator_[A-Za-z0-9_]+", line))

                reg_depth += raw.count("(") - raw.count(")")
                if reg_depth <= 0:
                    flush_registration()
                    in_registration = False
                    pending_registration = False
                    reg_depth = 0

        if in_registration:
            flush_registration()

        return out

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        src = self.unit_ndfbin
        if not src.exists():
            self._load_error = f"NDF hints source not found: {src}"
            return

        unite_ndf: Path | None = None
        game_root: Path | None = None
        depictions_override: Path | None = None
        game_data_root: Path | None = None

        if src.is_file() and src.name.lower() == "unitedescriptor.ndf":
            unite_ndf = src
            game_root = self._guess_game_root()
            game_data_root = self._find_parent_named(src, "GameData")
        elif src.is_file() and src.name.lower() == "depictionvehicles.ndf":
            depictions_override = src
            game_root = self._guess_game_root()
            game_data_root = self._find_parent_named(src, "GameData")
            if game_data_root is not None:
                candidate = game_data_root / "Generated" / "Gameplay" / "Gfx" / "UniteDescriptor.ndf"
                if candidate.exists() and candidate.is_file():
                    unite_ndf = candidate
        elif src.is_dir():
            unite_ndf = self._discover_unite_descriptor(src)
            game_root = src
            if unite_ndf is None:
                guessed = self._guess_game_root()
                if guessed is not None and guessed != src:
                    unite_ndf = self._discover_unite_descriptor(guessed)
                    game_root = guessed
        else:
            game_root = self._guess_game_root()
            if game_root is None:
                self._load_error = f"Cannot infer WARNO root from: {src}"
                return
            unite_ndf = self._discover_unite_descriptor(game_root)

        if unite_ndf is None:
            base = game_root if game_root is not None else src
            self._load_error = f"UniteDescriptor.ndf not found under: {base}"
            return

        self._source_files.append(str(unite_ndf))
        refmesh_bbox = self._parse_unite_descriptor(unite_ndf)
        if game_data_root is None:
            game_data_root = self._find_parent_named(unite_ndf, "GameData")
        if game_data_root is None:
            self._load_error = f"Cannot infer GameData root from: {unite_ndf}"
            return

        depiction_resources_root = game_data_root / "Gameplay" / "Gfx" / "DepictionResources"
        refmesh_assets = self._parse_depiction_resources(depiction_resources_root)
        if depiction_resources_root.exists():
            self._source_files.append(str(depiction_resources_root))

        for ref_name, bbox_bones in refmesh_bbox.items():
            assets = refmesh_assets.get(ref_name, set())
            for asset in assets:
                self._asset_bbox_bones.setdefault(asset, set()).update(bbox_bones)

        depictions_file = depictions_override or (
            game_data_root / "Generated" / "Gameplay" / "Gfx" / "Depictions" / "DepictionVehicles.ndf"
        )
        if depictions_file.exists() and depictions_file.is_file():
            coat, ops = self._parse_depictions_anchors(depictions_file)
            self._coating_anchors = coat
            self._operator_anchors = ops
            self._depiction_units = self._parse_depictions_structured(depictions_file)
            self._source_files.append(str(depictions_file))

    def hints_for_asset(self, asset_path: str) -> Dict[str, Any]:
        self._load()
        out_bones: List[str] = []
        if self._load_error:
            return {
                "bones": out_bones,
                "source": "none",
                "error": self._load_error,
            }

        asset_norm = normalize_asset_path(asset_path).lower()
        asset_stem = Path(asset_norm).stem.lower()
        asset_file = Path(asset_norm).name.lower()
        country = extract_units_country_code(asset_norm)

        bbox_bones: set[str] = set()
        if asset_norm in self._asset_bbox_bones:
            bbox_bones.update(self._asset_bbox_bones[asset_norm])
        if not bbox_bones:
            for mesh_asset, bones in self._asset_bbox_bones.items():
                same_file = Path(mesh_asset).name.lower() == asset_file
                if not same_file:
                    continue
                if asset_stem and asset_stem not in Path(mesh_asset).stem.lower():
                    continue
                if country and f"/units/{country}/" not in mesh_asset:
                    continue
                bbox_bones.update(bones)

        hint_anchors: set[str] = set()
        matched_units: List[str] = []
        matched_operators: List[str] = []
        matched_weapon_fx: List[str] = []
        matched_subdepictions: List[Dict[str, Any]] = []
        track_kind = ""

        for key, payload in self._depiction_units.items():
            if asset_stem and asset_stem not in key:
                continue
            if country and not (key.endswith(f"_{country}") or f"_{country}_" in key):
                continue
            matched_units.append(key)
            matched_operators.extend(
                [str(x).strip() for x in payload.get("operators", []) or [] if str(x).strip()]
            )
            matched_weapon_fx.extend(
                [str(x).strip().lower() for x in payload.get("weapon_fx_anchors", []) or [] if str(x).strip()]
            )
            for row in payload.get("subdepictions", []) or []:
                if not isinstance(row, dict):
                    continue
                anchors = [
                    str(x).strip().lower()
                    for x in row.get("anchors", []) or []
                    if str(x).strip()
                ]
                operators = [
                    str(x).strip()
                    for x in row.get("operators", []) or []
                    if str(x).strip()
                ]
                if anchors or operators:
                    matched_subdepictions.append(
                        {
                            "anchors": unique_keep_order(anchors),
                            "operators": unique_keep_order(operators),
                            "weapon_fx_anchors": [],
                        }
                    )
            if not track_kind:
                track_kind = str(payload.get("track_kind", "") or "").strip().lower()

        for key, anchors in self._operator_anchors.items():
            if asset_stem and asset_stem not in key:
                continue
            if country and not (key.endswith(f"_{country}") or f"_{country}_" in key):
                continue
            hint_anchors.update(anchors)
            matched_weapon_fx.extend(
                [str(x).strip().lower() for x in anchors if str(x).strip().lower().startswith("fx_")]
            )
        for key, anchors in self._coating_anchors.items():
            if asset_stem and asset_stem not in key:
                continue
            if country and not (key.endswith(f"_{country}") or f"_{country}_" in key):
                continue
            hint_anchors.update(anchors)

        out_bones = unique_keep_order(
            [
                *sorted(bbox_bones),
                *sorted(hint_anchors),
                *unique_keep_order(matched_weapon_fx),
                *[
                    anchor
                    for row in matched_subdepictions
                    for anchor in row.get("anchors", []) or []
                ],
            ]
        )
        return {
            "bones": out_bones,
            "source": "ndf_mirror",
            "error": "",
            "matched_units": unique_keep_order([x.lower() for x in matched_units if str(x).strip()]),
            "operators": unique_keep_order(matched_operators),
            "weapon_fx_anchors": unique_keep_order(matched_weapon_fx),
            "subdepictions": [
                row
                for row in unique_keep_order(
                    [
                        json.dumps(row, sort_keys=True, ensure_ascii=False)
                        for row in matched_subdepictions
                    ]
                )
                for row in [json.loads(row)]
            ],
            "track_kind": track_kind,
            "reference_meshes": [],
        }


class DepictionOperatorsResolver:
    def __init__(self, source_path: Path):
        self.source_path = Path(source_path)
        self._loaded = False
        self._load_error = ""
        self._contracts: Dict[str, Dict[str, Any]] = {}

    @property
    def load_error(self) -> str:
        return self._load_error

    def _flush_contract(
        self,
        name: str,
        type_name: str,
        operator_id: str,
        node_fields: Dict[str, str],
    ) -> None:
        clean_name = str(name or "").strip()
        if not clean_name:
            return
        nodes = unique_keep_order(
            [
                str(value).strip().lower()
                for value in node_fields.values()
                if str(value).strip()
            ]
        )
        self._contracts[clean_name] = {
            "name": clean_name,
            "type": str(type_name or "").strip(),
            "operator_id": str(operator_id or "").strip(),
            "node_fields": dict(node_fields),
            "nodes": nodes,
        }

    @staticmethod
    def _looks_like_node_field(key: str) -> bool:
        low = str(key or "").strip().lower()
        if not low:
            return False
        if "physicalproperty" in low:
            return False
        return ("node" in low) or ("bone" in low)

    @staticmethod
    def _extract_operator_name(raw: str) -> str:
        m = DEPICTION_OPERATOR_NAME_RX.search(str(raw or ""))
        return str(m.group(1)).strip() if m else ""

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        src = self.source_path
        if not src.exists() or not src.is_file():
            self._load_error = f"DepictionOperators source not found: {src}"
            return

        pending_name = ""
        in_decl = False
        decl_depth = 0
        saw_open_paren = False
        current_name = ""
        current_type = ""
        current_operator_id = ""
        current_node_fields: Dict[str, str] = {}

        def flush() -> None:
            nonlocal current_name, current_type, current_operator_id, current_node_fields, saw_open_paren
            self._flush_contract(current_name, current_type, current_operator_id, current_node_fields)
            current_name = ""
            current_type = ""
            current_operator_id = ""
            current_node_fields = {}
            saw_open_paren = False

        try:
            lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            self._load_error = f"Failed to read DepictionOperators: {exc}"
            return

        for raw in lines:
            line = str(raw or "").strip()
            if not line or line.startswith("//"):
                continue

            if not in_decl:
                m_assign = DEPICTION_OPERATOR_ASSIGN_RX.match(line)
                if m_assign:
                    pending_name = str(m_assign.group(1)).strip()

                if pending_name and " is " in line:
                    current_name = pending_name
                    pending_name = ""
                    tail = line.split(" is ", 1)[1].strip()
                    current_type = tail.split("(", 1)[0].strip()
                    current_operator_id = ""
                    current_node_fields = {}
                    in_decl = True
                    decl_depth = 0
                    saw_open_paren = False
                elif pending_name and line.lower().startswith("is "):
                    current_name = pending_name
                    pending_name = ""
                    tail = line[3:].strip()
                    current_type = tail.split("(", 1)[0].strip()
                    current_operator_id = ""
                    current_node_fields = {}
                    in_decl = True
                    decl_depth = 0
                    saw_open_paren = False
                else:
                    continue

            for key, value in DEPICTION_NODE_ASSIGN_RX.findall(raw):
                clean_key = str(key or "").strip()
                clean_val = str(value or "").strip()
                if not clean_key or not clean_val:
                    continue
                if clean_key.lower() == "operatorid":
                    current_operator_id = clean_val
                    continue
                if self._looks_like_node_field(clean_key):
                    current_node_fields[clean_key] = clean_val.lower()

            open_count = raw.count("(")
            close_count = raw.count(")")
            if open_count > 0:
                saw_open_paren = True
            decl_depth += open_count - close_count
            if saw_open_paren and decl_depth <= 0:
                flush()
                in_decl = False
                decl_depth = 0

        if in_decl:
            flush()

    def contract_for_name(self, operator_name: str) -> Dict[str, Any] | None:
        self._load()
        if self._load_error:
            return None
        clean = self._extract_operator_name(operator_name)
        if not clean:
            return None
        return self._contracts.get(clean)


class GfxManifestResolver:
    def __init__(
        self,
        *,
        warno_root: Path,
        modding_suite_root: Path,
        cache_dir: Path,
        wrapper_path: Path,
        gfx_cli_path: Path | None = None,
        timeout_sec: int = 180,
        use_gfx_json_manifest: bool = True,
        legacy_ndf_source: Path | None = None,
        legacy_operators_source: Path | None = None,
        enable_operator_semantics: bool = True,
    ):
        self.warno_root = Path(warno_root)
        self.modding_suite_root = Path(modding_suite_root)
        self.cache_dir = Path(cache_dir)
        self.wrapper_path = Path(wrapper_path)
        self.gfx_cli_path = Path(gfx_cli_path) if gfx_cli_path is not None and str(gfx_cli_path).strip() else None
        self.timeout_sec = max(5, int(timeout_sec))
        self.use_gfx_json_manifest = bool(use_gfx_json_manifest)
        self.legacy_ndf_source = Path(legacy_ndf_source) if legacy_ndf_source is not None and str(legacy_ndf_source).strip() else None
        self.legacy_operators_source = Path(legacy_operators_source) if legacy_operators_source is not None and str(legacy_operators_source).strip() else None
        self.enable_operator_semantics = bool(enable_operator_semantics)
        self._legacy_resolver: UnitNdfHintsResolver | None = None
        self._operator_resolver: DepictionOperatorsResolver | None = None

    @staticmethod
    def _empty_manifest(asset_norm: str, source: str, error: str, source_files: Sequence[str] | None = None) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "asset_path": asset_norm,
            "source_files": list(source_files or []),
            "matched_units": [],
            "reference_meshes": [],
            "operators": [],
            "turrets": [],
            "weapon_fx_anchors": [],
            "subdepictions": [],
            "required_nodes": [],
            "fx_nodes": [],
            "operator_contracts": [],
            "subdepiction_nodes": [],
            "role_map": {},
            "track_kind": "",
            "source": source,
            "error": str(error or "").strip(),
            "semantic_bones": [],
            "semantic_nodes": [],
            "transform_debug": [],
        }

    def _resolve_operator_source(self) -> Path | None:
        candidates: List[Path] = []
        if self.legacy_operators_source is not None:
            candidates.append(self.legacy_operators_source)
        if self.legacy_ndf_source is not None:
            game_data_root = UnitNdfHintsResolver._find_parent_named(self.legacy_ndf_source, "GameData")
            if game_data_root is not None:
                candidates.append(game_data_root / "Gameplay" / "Gfx" / "Units" / "DepictionOperators.ndf")
        warno_root = Path(self.warno_root)
        candidates.extend(
            [
                warno_root / "Mods" / "ModData" / "base" / "GameData" / "Gameplay" / "Gfx" / "Units" / "DepictionOperators.ndf",
                warno_root / "ModData" / "base" / "GameData" / "Gameplay" / "Gfx" / "Units" / "DepictionOperators.ndf",
                warno_root / "base" / "GameData" / "Gameplay" / "Gfx" / "Units" / "DepictionOperators.ndf",
                warno_root / "GameData" / "Gameplay" / "Gfx" / "Units" / "DepictionOperators.ndf",
            ]
        )
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None

    def _operator_resolver_or_none(self) -> DepictionOperatorsResolver | None:
        if self._operator_resolver is not None:
            return self._operator_resolver
        source = self._resolve_operator_source()
        if source is None:
            return None
        self._operator_resolver = DepictionOperatorsResolver(source)
        return self._operator_resolver

    @staticmethod
    def _role_from_contract_field(contract: Dict[str, Any], field_name: str) -> str:
        low = str(field_name or "").strip().lower()
        type_low = str(contract.get("type", "") or "").strip().lower()
        operator_id = str(contract.get("operator_id", "") or "").strip().lower()
        if "zaxisnode" in low:
            return "turret_yaw_node"
        if "yaxisnode" in low:
            return "turret_pitch_node"
        if "recoil" in type_low and "node" in low:
            return "turret_recoil_node"
        if operator_id in {"tracks", "wheels"} and "node" in low:
            return "movement_propulsion_node"
        return ""

    @staticmethod
    def _append_role(
        role_map: Dict[str, List[Dict[str, Any]]],
        node_name: str,
        *,
        role: str,
        source: str,
        contract: Dict[str, Any] | None = None,
        field_name: str = "",
    ) -> None:
        clean = str(node_name or "").strip().lower()
        role_clean = str(role or "").strip().lower()
        if not clean or not role_clean:
            return
        entry = {
            "role": role_clean,
            "source": str(source or "").strip().lower(),
            "contract_name": str((contract or {}).get("name", "") or "").strip(),
            "contract_type": str((contract or {}).get("type", "") or "").strip(),
            "operator_id": str((contract or {}).get("operator_id", "") or "").strip().lower(),
            "field_name": str(field_name or "").strip(),
        }
        rows = role_map.setdefault(clean, [])
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        seen = {
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
            for row in rows
            if isinstance(row, dict)
        }
        if key not in seen:
            rows.append(entry)

    def _enrich_manifest_semantics(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(manifest or {})
        resolver = self._operator_resolver_or_none() if bool(self.enable_operator_semantics) else None
        operator_contracts: List[Dict[str, Any]] = []
        required_nodes: List[str] = []
        role_map: Dict[str, List[Dict[str, Any]]] = {}
        fx_nodes = unique_keep_order(
            [
                str(x).strip().lower()
                for x in (out.get("weapon_fx_anchors", []) or [])
                if str(x).strip()
            ]
        )
        subdepiction_nodes: List[str] = []

        def add_required(name: str) -> None:
            clean = str(name or "").strip().lower()
            if clean and clean not in required_nodes:
                required_nodes.append(clean)

        def add_subdepiction(name: str) -> None:
            clean = str(name or "").strip().lower()
            if clean and clean not in subdepiction_nodes:
                subdepiction_nodes.append(clean)

        operator_names: List[str] = [
            str(x).strip()
            for x in (out.get("operators", []) or [])
            if str(x).strip()
        ]
        for row in out.get("subdepictions", []) or []:
            if not isinstance(row, dict):
                continue
            for anchor in row.get("anchors", []) or []:
                add_subdepiction(anchor)
                self._append_role(role_map, anchor, role="subdepiction_anchor", source="subdepiction")
            for anchor in row.get("weapon_fx_anchors", []) or []:
                clean = str(anchor).strip().lower()
                if clean and clean not in fx_nodes:
                    fx_nodes.append(clean)
            operator_names.extend(
                [str(x).strip() for x in row.get("operators", []) or [] if str(x).strip()]
            )

        if resolver is not None:
            seen_contracts: set[str] = set()
            for operator_name in unique_keep_order(operator_names):
                contract = resolver.contract_for_name(operator_name)
                if not contract:
                    continue
                contract_name = str(contract.get("name", "")).strip()
                if not contract_name or contract_name.lower() in seen_contracts:
                    continue
                seen_contracts.add(contract_name.lower())
                operator_contracts.append(contract)
                for node_name in contract.get("nodes", []) or []:
                    add_required(node_name)
                node_fields = contract.get("node_fields", {}) or {}
                if isinstance(node_fields, dict):
                    for field_name, node_name in node_fields.items():
                        role = self._role_from_contract_field(contract, str(field_name or ""))
                        if role:
                            self._append_role(
                                role_map,
                                str(node_name or "").strip().lower(),
                                role=role,
                                source="operator",
                                contract=contract,
                                field_name=str(field_name or ""),
                            )
                if not out.get("track_kind"):
                    operator_id = str(contract.get("operator_id", "") or "").strip().lower()
                    if operator_id == "tracks":
                        out["track_kind"] = "continuous_track"
                    elif operator_id == "wheels":
                        out["track_kind"] = "wheels"
            source_path = getattr(resolver, "source_path", None)
            if source_path is not None:
                source_text = str(source_path)
                src_files = [str(x) for x in (out.get("source_files", []) or []) if str(x).strip()]
                if source_text not in src_files:
                    src_files.append(source_text)
                    out["source_files"] = src_files

        for fx_name in fx_nodes:
            self._append_role(role_map, fx_name, role="weapon_fx_anchor", source="depiction")

        out["required_nodes"] = unique_keep_order(required_nodes)
        out["fx_nodes"] = unique_keep_order(fx_nodes)
        out["operator_contracts"] = list(operator_contracts)
        out["subdepiction_nodes"] = unique_keep_order(subdepiction_nodes)
        out["role_map"] = dict(role_map)
        semantic_bones = unique_keep_order(required_nodes + fx_nodes + subdepiction_nodes)
        manifest_semantic_nodes = [
            row
            for row in (out.get("semantic_nodes", []) or [])
            if isinstance(row, dict)
        ]
        for row in manifest_semantic_nodes:
            name = str(row.get("name", "") or "").strip().lower()
            if name and name not in semantic_bones:
                semantic_bones.append(name)
        out["semantic_bones"] = semantic_bones
        out["semantic_nodes"] = manifest_semantic_nodes
        out["transform_debug"] = [
            row
            for row in (out.get("transform_debug", []) or [])
            if isinstance(row, dict)
        ]
        return out

    def _legacy_manifest(self, asset_norm: str) -> Dict[str, Any]:
        if self.legacy_ndf_source is None:
            return self._empty_manifest(asset_norm, "none", "")

        if self._legacy_resolver is None:
            self._legacy_resolver = UnitNdfHintsResolver(self.legacy_ndf_source)

        hint_payload = self._legacy_resolver.hints_for_asset(asset_norm)
        bones = [
            str(x).strip().lower()
            for x in hint_payload.get("bones", []) or []
            if str(x).strip()
        ]
        manifest = self._empty_manifest(
            asset_norm,
            "legacy_ndf_mirror",
            str(hint_payload.get("error", "") or ""),
            self._legacy_resolver.source_files,
        )
        manifest["matched_units"] = [
            str(x).strip()
            for x in hint_payload.get("matched_units", []) or []
            if str(x).strip()
        ]
        manifest["reference_meshes"] = [
            str(x).strip()
            for x in hint_payload.get("reference_meshes", []) or []
            if str(x).strip()
        ]
        manifest["operators"] = [
            str(x).strip()
            for x in hint_payload.get("operators", []) or []
            if str(x).strip()
        ]
        manifest["weapon_fx_anchors"] = [
            str(x).strip().lower()
            for x in hint_payload.get("weapon_fx_anchors", []) or []
            if str(x).strip()
        ]
        manifest["subdepictions"] = [
            row
            for row in hint_payload.get("subdepictions", []) or []
            if isinstance(row, dict)
        ]
        manifest["track_kind"] = str(hint_payload.get("track_kind", "") or "").strip().lower()
        manifest["semantic_bones"] = unique_keep_order(bones)
        return self._enrich_manifest_semantics(manifest)

    def manifest_for_asset(self, asset_path: str, force_rebuild: bool = False) -> Dict[str, Any]:
        asset_norm = normalize_asset_path(asset_path)
        errors: List[str] = []

        if self.use_gfx_json_manifest:
            try:
                info = build_or_load_gfx_manifest(
                    warno_root=self.warno_root,
                    modding_suite_root=self.modding_suite_root,
                    asset_path=asset_norm,
                    cache_dir=self.cache_dir,
                    wrapper_path=self.wrapper_path,
                    gfx_cli_path=self.gfx_cli_path,
                    force_rebuild=force_rebuild,
                    timeout_sec=self.timeout_sec,
                )
                manifest = dict(info.get("gfx_manifest", {}) or {})
                manifest["source"] = "gfx_manifest"
                manifest["error"] = ""
                return self._enrich_manifest_semantics(manifest)
            except Exception as exc:
                errors.append(str(exc))

        legacy = self._legacy_manifest(asset_norm)
        if legacy.get("source") == "legacy_ndf_mirror":
            legacy_error = str(legacy.get("error", "") or "")
            if errors and not legacy_error:
                legacy["error"] = " | ".join(errors)
            return legacy

        error = " | ".join(x for x in errors if x)
        return self._empty_manifest(asset_norm, "none", error)


def atlas_ref_to_rel_under_assets(ref: str) -> Path:
    norm = normalize_atlas_ref(ref)
    p = PurePosixPath(norm)
    parts = list(p.parts)
    if not parts:
        raise ValueError(f"Invalid atlas ref path: {ref}")
    if parts[0].lower() == "assets":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Atlas ref has no relative part under Assets: {ref}")
    return Path(*parts)


def classify_texture_role(ref: str) -> str:
    n = Path(ref).name.lower()
    if "combinedda" in n or "coloralpha" in n:
        return "combined_da"
    if "normal" in n or "tscnm" in n:
        return "normal"
    if "combinedorm" in n or "combinedrm" in n or "_orm" in n:
        return "orm"
    if "singlechannellinearmap" in n and ("roughness" in n or "metallic" in n or "occlusion" in n or "_ao" in n):
        return "orm"
    if "diffuse" in n or "albedo" in n or "color" in n:
        return "diffuse"
    return "generic"


def detect_part_label(ref: str) -> str:
    s = ref.lower()
    labels = [
        ("tracks", "TRK"),
        ("track", "TRK"),
        ("chenille", "TRK"),
        ("chassis", "CHS"),
        ("hull", "HULL"),
        ("turret", "TURRET"),
        ("wheel", "WHL"),
        ("gun", "GUN"),
        ("barrel", "BARREL"),
    ]
    for needle, label in labels:
        if needle in s:
            return label
    return ""


def extract_texture_small_hints(texture_small_path: Path) -> List[str]:
    if not texture_small_path.exists():
        return []
    try:
        data = texture_small_path.read_bytes().lower()
    except Exception:
        return []
    hint_map = [
        (b"track", "TRK"),
        (b"chenille", "TRK"),
        (b"chassis", "CHS"),
        (b"hull", "HULL"),
        (b"turret", "TURRET"),
        (b"wheel", "WHL"),
    ]
    out: List[str] = []
    for needle, label in hint_map:
        if needle in data and label not in out:
            out.append(label)
    return out


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        pass
    shutil.copy2(src, dst)


def make_unique_texture_name(
    base: str,
    suffix: str,
    part_label: str,
    used: set[str],
) -> str:
    if part_label:
        candidate = f"{base}_{part_label}_{suffix}"
    else:
        candidate = f"{base}_{suffix}"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate

    i = 2
    while True:
        cand = f"{candidate}_{i}"
        if cand.lower() not in used:
            used.add(cand.lower())
            return cand
        i += 1


def sanitize_material_name(name: str) -> str:
    cleaned = SAFE_MATERIAL_RX.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "Material"


def channel_from_token(token: str) -> str | None:
    t = token.lower()
    if "combinedorm" in t or "combinedrm" in t or t.endswith("_orm"):
        return "orm"
    if "normal_x" in t or "normal_y" in t or "normal_z" in t:
        return None
    # "NoAlpha" diffuse maps must never be treated as alpha.
    if "diffusetexturenoalpha" in t or ("noalpha" in t and ("diffuse" in t or "color" in t)):
        return "diffuse"
    if "normal_reconstructed" in t:
        return "normal"
    if "occlusion" in t or "_ao" in t or t.endswith("_o") or "_o." in t or "_track_o" in t:
        return "occlusion"
    if "roughness" in t or t.endswith("_r") or "_track_r" in t:
        return "roughness"
    if "metallic" in t or t.endswith("_m") or "_track_m" in t:
        return "metallic"
    if "normal" in t or "_nm" in t:
        return "normal"
    if "diffuse" in t or t.endswith("_d") or "_track_d" in t:
        return "diffuse"
    if "alpha" in t or t.endswith("_a") or "_track_a" in t:
        return "alpha"
    return None


def part_label_from_token(token: str) -> str:
    t = token.lower()
    if "gauche" in t or "left" in t:
        return "TRK_GAUCHE"
    if "droite" in t or "right" in t:
        return "TRK_DROITE"
    if is_track_token(t):
        return "TRK"
    if re.search(r"(?:^|[\\/_-])mg(?:$|[\\/_-])", t):
        return "MG"
    m_part = re.search(r"(?:^|[\\/_-])part[_-]?([0-9]+)(?:$|[\\/_-])", t)
    if m_part:
        return f"PART{m_part.group(1)}"
    return ""


def material_stats(model: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for part in model["parts"]:
        mid = int(part["material"])
        xyz = part["vertices"]["xyz"]
        if not xyz:
            continue
        st = out.setdefault(
            mid,
            {
                "vertex_count": 0.0,
                "face_count": 0.0,
                "sx": 0.0,
                "sy": 0.0,
                "sz": 0.0,
                "minx": float("inf"),
                "maxx": float("-inf"),
                "miny": float("inf"),
                "maxy": float("-inf"),
                "minz": float("inf"),
                "maxz": float("-inf"),
            },
        )
        vcount = len(xyz) // 3
        st["vertex_count"] += float(vcount)
        st["face_count"] += float(len(part["indices"]) // 3)
        for i in range(0, len(xyz), 3):
            x = float(xyz[i + 0])
            y = float(xyz[i + 1])
            z = float(xyz[i + 2])
            st["sx"] += x
            st["sy"] += y
            st["sz"] += z
            st["minx"] = min(st["minx"], x)
            st["maxx"] = max(st["maxx"], x)
            st["miny"] = min(st["miny"], y)
            st["maxy"] = max(st["maxy"], y)
            st["minz"] = min(st["minz"], z)
            st["maxz"] = max(st["maxz"], z)

    for st in out.values():
        vc = max(1.0, st["vertex_count"])
        st["cx"] = st["sx"] / vc
        st["cy"] = st["sy"] / vc
        st["cz"] = st["sz"] / vc
    return out


def infer_material_names(
    model: Dict[str, Any],
    mirror_y: bool = False,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    stats = material_stats(model)
    mids = sorted(stats.keys())
    if not mids:
        return {}, {}

    names: Dict[int, str] = {}
    roles: Dict[int, str] = {}
    used: set[str] = set()

    def assign(mid: int, raw_name: str, role: str) -> None:
        base = sanitize_material_name(raw_name)
        name = base
        n = 2
        while name.lower() in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name.lower())
        names[mid] = name
        roles[mid] = role

    body_mid = max(mids, key=lambda m: stats[m]["vertex_count"])
    assign(body_mid, "Corps_Principal", "body")

    total_v = sum(stats[m]["vertex_count"] for m in mids)
    if total_v <= 0.0:
        total_v = 1.0
    center_y = sum(stats[m]["cy"] * stats[m]["vertex_count"] for m in mids) / total_v
    mirror_sign = -1.0 if mirror_y else 1.0

    candidates = [m for m in mids if m != body_mid and stats[m]["vertex_count"] >= 32.0]
    pos: List[int] = []
    neg: List[int] = []
    for mid in candidates:
        y_eff = stats[mid]["cy"] * mirror_sign
        if y_eff >= center_y * mirror_sign:
            pos.append(mid)
        else:
            neg.append(mid)

    if pos and neg:
        pos_mid = max(pos, key=lambda m: abs(stats[m]["cy"] - center_y))
        neg_mid = max(neg, key=lambda m: abs(stats[m]["cy"] - center_y))
        assign(pos_mid, "Chenille_Droite", "track_right")
        assign(neg_mid, "Chenille_Gauche", "track_left")

    idx = 1
    for mid in mids:
        if mid in names:
            continue
        assign(mid, f"Element_{idx:02d}", "other")
        idx += 1
    return names, roles


def track_maps_from_named(named_files: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Path]]:
    out = {"generic": {}, "left": {}, "right": {}}
    for item in named_files:
        channel = str(item.get("channel", ""))
        if not channel:
            continue
        p = Path(str(item.get("named", "")))
        stem = p.stem.lower()
        if not is_track_token(stem):
            continue
        if "gauche" in stem or "left" in stem:
            out["left"].setdefault(channel, p)
            continue
        if "droite" in stem or "right" in stem:
            out["right"].setdefault(channel, p)
            continue
        out["generic"].setdefault(channel, p)

    for ch, p in out["generic"].items():
        out["left"].setdefault(ch, p)
        out["right"].setdefault(ch, p)
    return out


def normalize_bone_label(name: str) -> str:
    s = name.strip()
    if not s:
        return ""
    low = s.lower()
    aliases = {
        "chassis": "chassis",
        "base_tourelle_01": "tourelle_base_01",
        "tourelle_01": "tourelle_01",
        "canon_01": "canon_01",
        "axe_canon_01": "axe_canon_01",
        "roue_droite_01": "roue_droite_01",
        "roue_gauche_01": "roue_gauche_01",
        "roue_01": "roue_01",
        "roue_02": "roue_02",
        "chenille_droite": "chenille_droite",
        "chenille_gauche": "chenille_gauche",
    }
    for k, v in aliases.items():
        if low == k:
            return v
    return s


def pretty_part_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return "Part"
    low = raw.lower()

    if low in {"chassis", "chassisarmaturefake", "chassisfake", "hull"}:
        return "Chassis"
    if low.startswith("tourelle") or "turret" in low:
        return "Tourelle_01"
    if low.startswith("axe_canon") or low == "axe" or low.startswith("axe_"):
        return "Axe_Canon_01"
    if low.startswith("canon") or "gun" in low or "barrel" in low:
        return "Canon_01"
    if "chenille_gauche" in low or "track_left" in low:
        return "Chenille_Gauche_01"
    if "chenille_droite" in low or "track_right" in low:
        return "Chenille_Droite_01"

    parts = [p for p in raw.split("_") if p]
    out: List[str] = []
    for p in parts:
        if re.fullmatch(r"[dg]\d+", p.lower()):
            out.append(p.upper())
        elif p.isdigit():
            out.append(p)
        else:
            out.append(p[0].upper() + p[1:])
    return "_".join(out) if out else raw


def map_bone_index_names(names: Sequence[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for i, raw in enumerate(names):
        nm = normalize_bone_label(raw)
        if not nm:
            continue
        out[i] = sanitize_material_name(nm)
    return out


def dominant_bone_indices(verts: Dict[str, List[float]], num_vertices: int) -> List[int]:
    raw_idx = verts.get("bone_idx")
    raw_w = verts.get("bone_w")
    if not raw_idx or not raw_w:
        return []
    if len(raw_idx) < num_vertices * 4 or len(raw_w) < num_vertices * 4:
        return []

    out: List[int] = []
    for v in range(num_vertices):
        base = v * 4
        idxs = [int(raw_idx[base + 0]), int(raw_idx[base + 1]), int(raw_idx[base + 2]), int(raw_idx[base + 3])]
        ws = [float(raw_w[base + 0]), float(raw_w[base + 1]), float(raw_w[base + 2]), float(raw_w[base + 3])]
        best = max(range(4), key=lambda i: ws[i])
        out.append(idxs[best])
    return out


def influential_bone_sets(
    verts: Dict[str, List[float]],
    num_vertices: int,
    min_weight: float = 0.1,
) -> List[set[int]]:
    raw_idx = verts.get("bone_idx")
    raw_w = verts.get("bone_w")
    if not raw_idx or not raw_w:
        return []
    if len(raw_idx) < num_vertices * 4 or len(raw_w) < num_vertices * 4:
        return []

    out: List[set[int]] = []
    for v in range(num_vertices):
        base = v * 4
        idxs = [int(raw_idx[base + 0]), int(raw_idx[base + 1]), int(raw_idx[base + 2]), int(raw_idx[base + 3])]
        ws = [float(raw_w[base + 0]), float(raw_w[base + 1]), float(raw_w[base + 2]), float(raw_w[base + 3])]

        influenced: set[int] = set()
        for i in range(4):
            if ws[i] >= float(min_weight):
                influenced.add(idxs[i])
        if not influenced:
            best = max(range(4), key=lambda i: ws[i])
            influenced.add(idxs[best])
        out.append(influenced)
    return out


def estimate_bone_centers_by_index(
    model: Dict[str, Any],
    bone_name_by_index: Dict[int, str],
    rot: Dict[str, float],
) -> Dict[int, Tuple[float, float, float]]:
    if not bone_name_by_index:
        return {}

    sums: Dict[int, List[float]] = {}
    counts: Dict[int, int] = {}

    for part in model.get("parts", []):
        verts = part.get("vertices", {})
        xyz = verts.get("xyz", [])
        vertex_count = len(xyz) // 3
        if vertex_count <= 0:
            continue

        dominant = dominant_bone_indices(verts, vertex_count)
        if not dominant:
            continue

        limit = min(vertex_count, len(dominant))
        for vi in range(limit):
            bidx = int(dominant[vi])
            if bidx not in bone_name_by_index:
                continue

            x = float(xyz[vi * 3 + 0])
            y = float(xyz[vi * 3 + 1])
            z = float(xyz[vi * 3 + 2])
            x, y, z = apply_rotation(x, y, z, rot)

            acc = sums.setdefault(bidx, [0.0, 0.0, 0.0])
            acc[0] += x
            acc[1] += y
            acc[2] += z
            counts[bidx] = counts.get(bidx, 0) + 1

    out: Dict[int, Tuple[float, float, float]] = {}
    for bidx, acc in sums.items():
        c = counts.get(bidx, 0)
        if c <= 0:
            continue
        out[bidx] = (acc[0] / c, acc[1] / c, acc[2] / c)
    return out


def infer_missing_wheel_bone_names(
    model: Dict[str, Any],
    bone_name_by_index: Dict[int, str],
    rot: Dict[str, float],
    min_vertices: int = 96,
) -> Dict[int, str]:
    # Some meshes use additional wheel bones that are missing from parsed node names.
    # Infer them from dominant-vertex cloud shape and mark as wheel groups.
    stats: Dict[int, Dict[str, float]] = {}

    for part in model.get("parts", []):
        verts = part.get("vertices", {})
        xyz = verts.get("xyz", [])
        vertex_count = len(xyz) // 3
        if vertex_count <= 0:
            continue
        dominant = dominant_bone_indices(verts, vertex_count)
        if not dominant:
            continue

        limit = min(vertex_count, len(dominant))
        for vi in range(limit):
            bidx = int(dominant[vi])
            if bidx in bone_name_by_index:
                continue

            x = float(xyz[vi * 3 + 0])
            y = float(xyz[vi * 3 + 1])
            z = float(xyz[vi * 3 + 2])
            x, y, z = apply_rotation(x, y, z, rot)

            st = stats.get(bidx)
            if st is None:
                stats[bidx] = {
                    "count": 1.0,
                    "minx": x,
                    "maxx": x,
                    "miny": y,
                    "maxy": y,
                    "minz": z,
                    "maxz": z,
                }
                continue

            st["count"] += 1.0
            st["minx"] = min(st["minx"], x)
            st["maxx"] = max(st["maxx"], x)
            st["miny"] = min(st["miny"], y)
            st["maxy"] = max(st["maxy"], y)
            st["minz"] = min(st["minz"], z)
            st["maxz"] = max(st["maxz"], z)

    out: Dict[int, str] = {}
    for bidx, st in stats.items():
        count = int(st["count"])
        if count < int(min_vertices):
            continue

        dims = (
            float(st["maxx"] - st["minx"]),
            float(st["maxy"] - st["miny"]),
            float(st["maxz"] - st["minz"]),
        )
        thin_axis = min(range(3), key=lambda i: dims[i])
        other_axes = [i for i in (0, 1, 2) if i != thin_axis]
        d0 = float(dims[other_axes[0]])
        d1 = float(dims[other_axes[1]])
        if d0 <= 0.35 or d1 <= 0.35:
            continue
        if max(d0, d1) <= 1e-6:
            continue

        round_ratio = max(d0, d1) / max(min(d0, d1), 1e-6)
        if round_ratio > 1.35:
            continue
        if dims[thin_axis] > max(d0, d1) * 0.85:
            continue

        out[int(bidx)] = f"roue_auto_{int(bidx):02d}"

    return out


def _top_level_child_bone_index(
    bone_index: int,
    bone_parent_by_index: Dict[int, int],
) -> int:
    cur = int(bone_index)
    prev = cur
    seen: set[int] = set()
    while True:
        if cur in seen:
            return int(prev)
        seen.add(cur)
        parent = int(bone_parent_by_index.get(cur, -1))
        if parent < 0:
            return int(prev)
        prev = cur
        cur = parent


_WHEEL_BONE_NAME_SHORT_RX = re.compile(r"^roue_(elev_)?([dg])([0-9]+)$", re.IGNORECASE)
_WHEEL_BONE_NAME_WORD_RX = re.compile(
    r"^roue_(elev_)?(droite|gauche)_?([0-9]+)$",
    re.IGNORECASE,
)
_WHEEL_BONE_NAME_UNSIDED_RX = re.compile(r"^roue_([0-9]+)$", re.IGNORECASE)


def _normalize_bone_name_for_tokens(name: str) -> str:
    low = str(name or "").strip().lower()
    if not low:
        return ""
    # Accept multiple separators from different game/build variants:
    # roue_droite_01, roue.droite.01, roue-droite-01.
    return re.sub(r"[^a-z0-9]+", "_", low).strip("_")


def _parse_wheel_bone_name(name: str) -> Tuple[str, str, int] | None:
    low = _normalize_bone_name_for_tokens(name)
    if not low:
        return None
    m = _WHEEL_BONE_NAME_SHORT_RX.match(low)
    if not m:
        m = _WHEEL_BONE_NAME_WORD_RX.match(low)
        if not m:
            m = _WHEEL_BONE_NAME_UNSIDED_RX.match(low)
            if not m:
                return None
            return "wheel", "U", int(m.group(1))
        kind = "elev" if m.group(1) else "wheel"
        side_word = str(m.group(2)).lower()
        side = "D" if side_word.startswith("droi") else "G"
        num = int(m.group(3))
        return kind, side, num
    kind = "elev" if m.group(1) else "wheel"
    side = str(m.group(2)).upper()
    num = int(m.group(3))
    return kind, side, num


def _track_side_from_bone_name(name: str) -> str:
    low = _normalize_bone_name_for_tokens(name)
    if not low:
        return ""
    parsed_wheel = _parse_wheel_bone_name(low)
    if parsed_wheel is not None:
        _, side, _ = parsed_wheel
        if side == "G":
            return "left"
        if side == "D":
            return "right"
        return ""
    if "chenille_gauche" in low or "track_left" in low or "left_track" in low:
        return "left"
    if "chenille_droite" in low or "track_right" in low or "right_track" in low:
        return "right"
    if re.search(r"(?:^|_)g[0-9]+(?:_|$)", low):
        return "left"
    if re.search(r"(?:^|_)d[0-9]+(?:_|$)", low):
        return "right"
    return ""


def _format_preserved_tokens(raw_low: str) -> str:
    parts = [tok for tok in str(raw_low or "").split("_") if tok]
    if not parts:
        return "Chassis"
    out: List[str] = []
    for token in parts:
        tok_low = str(token).lower()
        if token.isdigit():
            out.append(token)
            continue
        m_suffix = re.fullmatch(r"([a-z]+)([0-9]+)", token, flags=re.IGNORECASE)
        if m_suffix:
            stem = str(m_suffix.group(1) or "")
            digits = str(m_suffix.group(2) or "")
            if len(stem) <= 2 and stem.isalpha():
                out.append(f"{stem.upper()}{digits}")
            else:
                out.append(f"{stem[:1].upper()}{stem[1:]}{digits}")
            continue
        if tok_low in {"launcher", "pod"}:
            out.append(tok_low)
            continue
        if len(token) <= 2 and token.isalpha():
            out.append(token.upper())
            continue
        out.append(token[:1].upper() + token[1:])
    return "_".join(out)


def _raw_geometry_helper_like_name(name: str) -> bool:
    low = _normalize_bone_name_for_tokens(name)
    if not low:
        return True
    if low.startswith(("bone_", "fx_", "empty", "cylinder.", "bip01")):
        return True
    if low in {
        "armature",
        "papyrus",
        "fake",
        "pilot",
        "chassis",
        "hull",
        "base",
        "chassisfake",
        "chassisarmaturefake",
    }:
        return True
    if low.startswith("armature_"):
        return True
    return False


def _preserved_raw_geometry_group_name(
    raw_name: str,
    triangle_count: int,
    total_triangles: int,
    classified_group: str,
) -> str:
    low = _normalize_bone_name_for_tokens(raw_name)
    if not low or str(classified_group or "") != "Chassis":
        return ""
    if _raw_geometry_helper_like_name(low):
        return ""
    force_validate = low in {"kokon_launcher", "pods_internes"}
    min_tris = max(48, int(max(1, total_triangles) * 0.012))
    min_tokens = 1 if force_validate else 2
    if int(triangle_count) < min_tris:
        return ""
    if len([tok for tok in low.split("_") if tok]) < min_tokens:
        return ""
    return _format_preserved_tokens(low)


def classify_group_from_bone_name(name: str) -> str:
    low = _normalize_bone_name_for_tokens(name)
    if not low:
        return "Chassis"

    def _fmt2(prefix: str, number: int | None, default_number: int = 1) -> str:
        num = int(default_number if number is None else number)
        if num < 0:
            num = default_number
        return f"{prefix}_{num:02d}"

    wheel = _parse_wheel_bone_name(low)
    if wheel is not None:
        _kind, side, num = wheel
        if side == "U":
            return f"Roue_{num:02d}"
        return f"Roue_{side}{num}"
    if low in {"roue_droite", "roue_gauche"}:
        return "Roue_Droite" if low.endswith("droite") else "Roue_Gauche"

    if "chenille_gauche" in low or "track_left" in low or "left_track" in low:
        return "Chenille_Gauche"
    if "chenille_droite" in low or "track_right" in low or "right_track" in low:
        return "Chenille_Droite"

    if re.fullmatch(r"bloc_moteur(?:_[0-9]+)?", low, flags=re.IGNORECASE):
        parts = [tok for tok in low.split("_") if tok]
        if parts and parts[-1].isdigit():
            return f"Bloc_Moteur_{int(parts[-1])}"
        return "Bloc_Moteur"
    if re.fullmatch(r"hatch(?:_[0-9]+)?", low, flags=re.IGNORECASE):
        parts = [tok for tok in low.split("_") if tok]
        if parts and parts[-1].isdigit():
            return f"Hatch_{int(parts[-1]):02d}"
        return "Hatch_01"
    if re.fullmatch(r"trappe_(?:avant|droite|gauche)", low, flags=re.IGNORECASE):
        tokens = [tok.capitalize() for tok in low.split("_") if tok]
        return "_".join(tokens)
    if low == "window":
        return "Window"
    if re.fullmatch(r"helice_ls_[0-9]+", low, flags=re.IGNORECASE):
        return _format_preserved_tokens(low)
    if re.fullmatch(r"roue_avant[0-9]*", low, flags=re.IGNORECASE):
        return _format_preserved_tokens(low)
    if low == "igla_v_pod":
        return _format_preserved_tokens(low)
    if re.fullmatch(r"kokon_launcher_[lr](?:[0-9]+)?", low, flags=re.IGNORECASE):
        return _format_preserved_tokens(low)
    if re.fullmatch(r"aileron_[dg]", low, flags=re.IGNORECASE):
        side = "D" if low.endswith("_d") else "G"
        return f"Aileron_{side}"
    if re.fullmatch(r"elevator_[dg]", low, flags=re.IGNORECASE):
        side = "D" if low.endswith("_d") else "G"
        return f"Elevator_{side}"
    if low == "rudder":
        return "Rudder"
    if low in {"train_avant", "train_arriere"}:
        return "Train_Avant" if low.endswith("avant") else "Train_Arriere"
    if low == "pilot":
        return "Pilot"

    m = re.match(r"^blindage_([dg])_?([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        side = str(m.group(1)).upper()
        num = int(m.group(2))
        return f"Blindage_{side}{num}"

    m = re.match(r"^(?:base_)?tourelle_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return _fmt2("Tourelle", int(m.group(1)))
    m = re.match(r"^fx_tourelle([0-9]+)(?:_|$)", low, flags=re.IGNORECASE)
    if m:
        return _fmt2("Tourelle", int(m.group(1)))
    if "turret" in low:
        return _fmt2("Tourelle", None)

    m = re.match(r"^axe_canon_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return _fmt2("Axe_Canon", int(m.group(1)))
    if low.startswith("axe_canon"):
        return _fmt2("Axe_Canon", None)

    m = re.match(r"^canon_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return _fmt2("Canon", int(m.group(1)))
    m = re.match(r"^fx_canon([0-9]+)(?:_|$)", low, flags=re.IGNORECASE)
    if m:
        return _fmt2("Canon", int(m.group(1)))
    if low.startswith("canon") or "barrel" in low:
        return _fmt2("Canon", None)

    if low.startswith("chassis") or low in {"hull", "base", "chassisfake", "chassisarmaturefake"}:
        return "Chassis"
    if low.startswith("fx_"):
        return "Chassis"

    return "Chassis"


_RAW_GEOMETRY_PROMOTION_FIXTURES = {
    "kokon_launcher",
    "pods_internes",
}


def _raw_node_is_helper_like_for_geometry_promotion(name: str) -> bool:
    low = _normalize_bone_name_for_tokens(name)
    if not low:
        return True
    if low.startswith("fx_"):
        return True
    if low.startswith("bip01"):
        return True
    if low in {"armature", "papyrus", "fake"}:
        return True
    if low.startswith("armature_"):
        return True
    if re.fullmatch(r"empty(?:\.[0-9]+)?", low, flags=re.IGNORECASE):
        return True
    if low.startswith("cylinder."):
        return True
    return False


def _geometry_promotion_group_name(raw_name: str) -> str:
    return pretty_part_name(str(raw_name or "").strip())


def _should_promote_raw_geometry_group(
    raw_name: str,
    tri_count: int,
    total_triangles: int,
) -> bool:
    low = _normalize_bone_name_for_tokens(raw_name)
    if not low:
        return False
    if _raw_node_is_helper_like_for_geometry_promotion(low):
        return False
    if low in {"chassis", "hull", "base", "chassisfake", "chassisarmaturefake"}:
        return False
    if low in _RAW_GEOMETRY_PROMOTION_FIXTURES:
        return int(tri_count) >= 24
    min_tris = max(96, int(max(1, int(total_triangles)) * 0.015))
    return int(tri_count) >= int(min_tris)


def _build_group_split_diagnostics(
    tri_infos: Sequence[Dict[str, Any]],
    material_role: str = "",
    material_name: str = "",
) -> Dict[str, Any]:
    total = len(tri_infos)
    raw_counts: Dict[str, int] = {}
    raw_index_by_name: Dict[str, int] = {}
    bone_counts: Dict[int, int] = {}
    for info in tri_infos:
        raw_name = str(info.get("raw_bone_name", "") or "").strip()
        raw_low = _normalize_bone_name_for_tokens(raw_name)
        raw_idx = int(info.get("raw_bone_index", -1))
        if raw_low:
            raw_counts[raw_low] = raw_counts.get(raw_low, 0) + 1
            raw_index_by_name.setdefault(raw_low, raw_idx)
        if raw_idx >= 0:
            bone_counts[raw_idx] = bone_counts.get(raw_idx, 0) + 1

    dominant_raw_name = ""
    dominant_raw_triangles = 0
    significant_raw_nodes: List[Dict[str, Any]] = []
    if raw_counts:
        dominant_raw_name = min(raw_counts.keys(), key=lambda key: (-raw_counts[key], key))
        dominant_raw_triangles = int(raw_counts.get(dominant_raw_name, 0))
        min_significant = max(24, int(max(1, total) * 0.10))
        for raw_low in sorted(raw_counts.keys(), key=lambda key: (-raw_counts[key], key)):
            count = int(raw_counts[raw_low])
            if count < min_significant:
                continue
            significant_raw_nodes.append(
                {
                    "raw_node": str(raw_low),
                    "raw_node_index": int(raw_index_by_name.get(raw_low, -1)),
                    "triangles": int(count),
                    "ratio": round(float(count) / float(max(1, total)), 6),
                }
            )

    dominant_raw_ratio = round(float(dominant_raw_triangles) / float(max(1, total)), 6)
    contamination_verdict = "clean"
    if len(significant_raw_nodes) > 1 or (dominant_raw_ratio < 0.85 and total >= 48):
        contamination_verdict = "mixed_raw_nodes"

    dominant_bone_index = -1
    dominant_bone_triangles = 0
    if bone_counts:
        dominant_bone_index = min(bone_counts.keys(), key=lambda idx: (-bone_counts[idx], int(idx)))
        dominant_bone_triangles = int(bone_counts.get(dominant_bone_index, 0))

    return {
        "material_role": str(material_role or ""),
        "material_name": str(material_name or ""),
        "triangles": int(total),
        "dominant_raw_node": str(dominant_raw_name),
        "dominant_raw_node_index": int(raw_index_by_name.get(dominant_raw_name, -1)) if dominant_raw_name else -1,
        "dominant_raw_node_ratio": float(dominant_raw_ratio),
        "dominant_bone_index": int(dominant_bone_index),
        "dominant_bone_ratio": round(float(dominant_bone_triangles) / float(max(1, total)), 6),
        "significant_raw_nodes": significant_raw_nodes,
        "contamination_verdict": str(contamination_verdict),
    }


def _bucket_face_components(
    faces: Sequence[Sequence[int]],
) -> List[List[int]]:
    vert_to_faces: Dict[int, List[int]] = {}
    for face_idx, face in enumerate(faces):
        for vi in face:
            vert_to_faces.setdefault(int(vi), []).append(int(face_idx))

    components: List[List[int]] = []
    visited: set[int] = set()
    for face_idx in range(len(faces)):
        if face_idx in visited:
            continue
        queue = [int(face_idx)]
        visited.add(int(face_idx))
        comp: List[int] = []
        while queue:
            cur = queue.pop()
            comp.append(int(cur))
            for vi in faces[cur]:
                for nei in vert_to_faces.get(int(vi), []):
                    if nei in visited:
                        continue
                    visited.add(int(nei))
                    queue.append(int(nei))
        if comp:
            components.append(comp)
    return components


def _polygon_area_3d(vertices: Sequence[Tuple[float, float, float]], face: Sequence[int]) -> float:
    if len(face) < 3:
        return 0.0
    try:
        p0 = vertices[int(face[0])]
    except Exception:
        return 0.0
    area = 0.0
    for i in range(1, len(face) - 1):
        try:
            p1 = vertices[int(face[i])]
            p2 = vertices[int(face[i + 1])]
        except Exception:
            return 0.0
        ux = float(p1[0]) - float(p0[0])
        uy = float(p1[1]) - float(p0[1])
        uz = float(p1[2]) - float(p0[2])
        vx = float(p2[0]) - float(p0[0])
        vy = float(p2[1]) - float(p0[1])
        vz = float(p2[2]) - float(p0[2])
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    return float(area)


def _conservative_quad_merge(
    vertices: Sequence[Tuple[float, float, float]],
    faces: Sequence[Sequence[int]],
    face_mids: Sequence[int],
) -> Tuple[List[List[int]], List[int], int]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    tri_faces: List[Tuple[int, int, int] | None] = []
    for face_idx, face in enumerate(faces):
        if len(face) != 3:
            tri_faces.append(None)
            continue
        tri = (int(face[0]), int(face[1]), int(face[2]))
        tri_faces.append(tri)
        edges = (
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        )
        for edge in edges:
            edge_to_faces.setdefault(edge, []).append(int(face_idx))

    used: set[int] = set()
    out_faces: List[List[int]] = []
    out_mids: List[int] = []
    merged_pairs = 0

    def _sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
        return (
            float(a[0]) - float(b[0]),
            float(a[1]) - float(b[1]),
            float(a[2]) - float(b[2]),
        )

    def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
        return (
            float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
            float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
            float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
        )

    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])

    def _norm(v: Sequence[float]) -> float:
        return math.sqrt(_dot(v, v))

    def _ordered_quad(face_a: Tuple[int, int, int], face_b: Tuple[int, int, int]) -> List[int] | None:
        verts = sorted({int(v) for v in [*face_a, *face_b]})
        if len(verts) != 4:
            return None
        pts = [vertices[v] for v in verts]
        centroid = (
            sum(float(p[0]) for p in pts) / 4.0,
            sum(float(p[1]) for p in pts) / 4.0,
            sum(float(p[2]) for p in pts) / 4.0,
        )
        n0 = _cross(_sub(vertices[face_a[1]], vertices[face_a[0]]), _sub(vertices[face_a[2]], vertices[face_a[0]]))
        n1 = _cross(_sub(vertices[face_b[1]], vertices[face_b[0]]), _sub(vertices[face_b[2]], vertices[face_b[0]]))
        n = (
            float(n0[0]) + float(n1[0]),
            float(n0[1]) + float(n1[1]),
            float(n0[2]) + float(n1[2]),
        )
        n_len = _norm(n)
        if n_len <= 1.0e-12:
            return None
        n = (n[0] / n_len, n[1] / n_len, n[2] / n_len)

        axis_u = _sub(vertices[verts[0]], centroid)
        axis_u_len = _norm(axis_u)
        if axis_u_len <= 1.0e-12:
            return None
        axis_u = (axis_u[0] / axis_u_len, axis_u[1] / axis_u_len, axis_u[2] / axis_u_len)
        axis_v = _cross(n, axis_u)
        axis_v_len = _norm(axis_v)
        if axis_v_len <= 1.0e-12:
            return None
        axis_v = (axis_v[0] / axis_v_len, axis_v[1] / axis_v_len, axis_v[2] / axis_v_len)

        polar: List[Tuple[float, int]] = []
        for v in verts:
            rel = _sub(vertices[v], centroid)
            x = _dot(rel, axis_u)
            y = _dot(rel, axis_v)
            polar.append((math.atan2(y, x), int(v)))
        ordered = [int(v) for _angle, v in sorted(polar, key=lambda item: item[0])]
        if len(set(ordered)) != 4:
            return None

        boundary_counts: Dict[Tuple[int, int], int] = {}
        for edge in (
            tuple(sorted((face_a[0], face_a[1]))),
            tuple(sorted((face_a[1], face_a[2]))),
            tuple(sorted((face_a[2], face_a[0]))),
            tuple(sorted((face_b[0], face_b[1]))),
            tuple(sorted((face_b[1], face_b[2]))),
            tuple(sorted((face_b[2], face_b[0]))),
        ):
            boundary_counts[edge] = boundary_counts.get(edge, 0) + 1
        boundary_edges = {edge for edge, count in boundary_counts.items() if count == 1}
        quad_edges = {
            tuple(sorted((ordered[0], ordered[1]))),
            tuple(sorted((ordered[1], ordered[2]))),
            tuple(sorted((ordered[2], ordered[3]))),
            tuple(sorted((ordered[3], ordered[0]))),
        }
        if quad_edges != boundary_edges:
            return None

        max_edge = 0.0
        for i in range(4):
            p0 = vertices[ordered[i]]
            p1 = vertices[ordered[(i + 1) % 4]]
            max_edge = max(max_edge, math.dist(p0, p1))
        plane_eps = max(1.0e-5, max_edge * 1.0e-4)
        for v in ordered:
            rel = _sub(vertices[v], vertices[ordered[0]])
            if abs(_dot(rel, n)) > plane_eps:
                return None

        sign = 0.0
        for i in range(4):
            p0 = vertices[ordered[i]]
            p1 = vertices[ordered[(i + 1) % 4]]
            p2 = vertices[ordered[(i + 2) % 4]]
            c = _cross(_sub(p1, p0), _sub(p2, p1))
            s = _dot(c, n)
            if abs(s) <= 1.0e-10:
                continue
            if sign == 0.0:
                sign = 1.0 if s > 0.0 else -1.0
            elif s * sign < 0.0:
                return None
        return ordered

    candidate_pairs: List[Tuple[float, int, int]] = []
    for edge, face_ids in edge_to_faces.items():
        if len(face_ids) != 2:
            continue
        fa, fb = int(face_ids[0]), int(face_ids[1])
        tri_a = tri_faces[fa]
        tri_b = tri_faces[fb]
        if tri_a is None or tri_b is None:
            continue
        if int(face_mids[fa]) != int(face_mids[fb]):
            continue
        n0 = _cross(_sub(vertices[tri_a[1]], vertices[tri_a[0]]), _sub(vertices[tri_a[2]], vertices[tri_a[0]]))
        n1 = _cross(_sub(vertices[tri_b[1]], vertices[tri_b[0]]), _sub(vertices[tri_b[2]], vertices[tri_b[0]]))
        n0_len = _norm(n0)
        n1_len = _norm(n1)
        if n0_len <= 1.0e-12 or n1_len <= 1.0e-12:
            continue
        normal_dot = _dot(n0, n1) / max(1.0e-12, n0_len * n1_len)
        if normal_dot < 0.9995:
            continue
        quad = _ordered_quad(tri_a, tri_b)
        if quad is None:
            continue
        candidate_pairs.append((1.0 - normal_dot, fa, fb))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    pair_face_to_quad: Dict[int, List[int]] = {}
    for _score, fa, fb in candidate_pairs:
        if fa in used or fb in used:
            continue
        tri_a = tri_faces[fa]
        tri_b = tri_faces[fb]
        if tri_a is None or tri_b is None:
            continue
        quad = _ordered_quad(tri_a, tri_b)
        if quad is None:
            continue
        used.add(int(fa))
        used.add(int(fb))
        pair_face_to_quad[int(fa)] = list(quad)
        pair_face_to_quad[int(fb)] = list(quad)
        merged_pairs += 1

    emitted_pairs: set[int] = set()
    for face_idx, face in enumerate(faces):
        if face_idx in used:
            if face_idx in emitted_pairs:
                continue
            quad = pair_face_to_quad.get(int(face_idx))
            if quad is None:
                continue
            out_faces.append(list(quad))
            out_mids.append(int(face_mids[face_idx]))
            for peer_idx, peer_quad in pair_face_to_quad.items():
                if peer_quad == quad:
                    emitted_pairs.add(int(peer_idx))
            continue
        out_faces.append([int(v) for v in face])
        out_mids.append(int(face_mids[face_idx]))

    return out_faces, out_mids, int(merged_pairs)


def conservative_cleanup_bucket_geometry(
    bucket: Dict[str, Any],
) -> Dict[str, Any]:
    vertices_in = [tuple(map(float, row)) for row in list(bucket.get("vertices", []) or [])]
    uvs_in = [tuple(map(float, row)) for row in list(bucket.get("uvs", []) or [])]
    faces_in = [[int(v) for v in face] for face in list(bucket.get("faces", []) or [])]
    mids_in = [int(mid) for mid in list(bucket.get("face_mids", []) or [])]
    refs_in = list(bucket.get("source_refs", []) or [])

    diagnostics = {
        "cleanup_mode": "conservative",
        "pre_vertex_count": int(len(vertices_in)),
        "pre_polygon_count": int(len(faces_in)),
        "pre_component_count": int(len(_bucket_face_components(faces_in))) if faces_in else 0,
        "welded_vertices": 0,
        "degenerate_faces_removed": 0,
        "tiny_components_removed": 0,
        "quad_pairs_merged": 0,
    }
    if not vertices_in or not faces_in:
        diagnostics["post_vertex_count"] = int(len(vertices_in))
        diagnostics["post_polygon_count"] = int(len(faces_in))
        diagnostics["post_component_count"] = int(diagnostics["pre_component_count"])
        return diagnostics

    pos_eps = 1.0e-6
    uv_eps = 1.0e-6
    remap: Dict[int, int] = {}
    new_vertices: List[Tuple[float, float, float]] = []
    new_uvs: List[Tuple[float, float]] = []
    new_refs: List[Any] = []
    key_to_new: Dict[Tuple[int, int, int, int, int], int] = {}
    for src_idx, pos in enumerate(vertices_in):
        uv = uvs_in[src_idx] if src_idx < len(uvs_in) else (0.0, 0.0)
        key = (
            int(round(float(pos[0]) / pos_eps)),
            int(round(float(pos[1]) / pos_eps)),
            int(round(float(pos[2]) / pos_eps)),
            int(round(float(uv[0]) / uv_eps)),
            int(round(float(uv[1]) / uv_eps)),
        )
        mapped = key_to_new.get(key)
        if mapped is None:
            mapped = len(new_vertices)
            key_to_new[key] = int(mapped)
            new_vertices.append(tuple(pos))
            new_uvs.append(tuple(uv))
            new_refs.append(refs_in[src_idx] if src_idx < len(refs_in) else (-1, -1))
        remap[int(src_idx)] = int(mapped)
    diagnostics["welded_vertices"] = max(0, int(len(vertices_in) - len(new_vertices)))

    cleaned_faces: List[List[int]] = []
    cleaned_mids: List[int] = []
    for face_idx, face in enumerate(faces_in):
        mapped = [int(remap.get(int(vi), -1)) for vi in face]
        if len(mapped) < 3 or any(int(vi) < 0 for vi in mapped):
            diagnostics["degenerate_faces_removed"] += 1
            continue
        if len(set(mapped)) < 3:
            diagnostics["degenerate_faces_removed"] += 1
            continue
        if _polygon_area_3d(new_vertices, mapped) <= 1.0e-12:
            diagnostics["degenerate_faces_removed"] += 1
            continue
        cleaned_faces.append(list(mapped))
        cleaned_mids.append(int(mids_in[face_idx]) if face_idx < len(mids_in) else -1)

    kept_face_ids: set[int] = set()
    components = _bucket_face_components(cleaned_faces)
    for comp in components:
        if len(comp) > 2:
            kept_face_ids.update(int(idx) for idx in comp)
            continue
        comp_faces = [cleaned_faces[int(idx)] for idx in comp]
        comp_verts = sorted({int(vi) for face in comp_faces for vi in face})
        if len(comp_verts) < 3:
            diagnostics["tiny_components_removed"] += len(comp)
            continue
        xs = [float(new_vertices[vi][0]) for vi in comp_verts]
        ys = [float(new_vertices[vi][1]) for vi in comp_verts]
        zs = [float(new_vertices[vi][2]) for vi in comp_verts]
        bbox_diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)
        total_area = sum(_polygon_area_3d(new_vertices, face) for face in comp_faces)
        if bbox_diag <= 1.0e-5 or total_area <= 1.0e-10:
            diagnostics["tiny_components_removed"] += len(comp)
            continue
        kept_face_ids.update(int(idx) for idx in comp)

    if len(kept_face_ids) != len(cleaned_faces):
        next_faces: List[List[int]] = []
        next_mids: List[int] = []
        for idx, face in enumerate(cleaned_faces):
            if idx not in kept_face_ids:
                continue
            next_faces.append(face)
            next_mids.append(int(cleaned_mids[idx]))
        cleaned_faces = next_faces
        cleaned_mids = next_mids

    quad_faces, quad_mids, quad_pairs = _conservative_quad_merge(new_vertices, cleaned_faces, cleaned_mids)
    diagnostics["quad_pairs_merged"] = int(quad_pairs)

    bucket["vertices"] = [tuple(v) for v in new_vertices]
    bucket["uvs"] = [tuple(uv) for uv in new_uvs]
    bucket["source_refs"] = list(new_refs)
    bucket["faces"] = [list(face) for face in quad_faces]
    bucket["face_mids"] = [int(mid) for mid in quad_mids]
    bucket["map"] = {}

    diagnostics["post_vertex_count"] = int(len(bucket["vertices"]))
    diagnostics["post_polygon_count"] = int(len(bucket["faces"]))
    diagnostics["post_component_count"] = int(len(_bucket_face_components(bucket["faces"]))) if bucket["faces"] else 0
    return diagnostics


def resolve_center_wheel_bone_index(
    elev_idx: int,
    bone_name_by_index: Dict[int, str],
) -> int:
    cur_name = str(bone_name_by_index.get(int(elev_idx), "")).strip()
    parsed = _parse_wheel_bone_name(cur_name)
    if parsed is None:
        return int(elev_idx)
    kind, side, num = parsed
    if kind != "elev":
        return int(elev_idx)

    target_low = f"roue_{side.lower()}{num}"
    for bidx, raw_name in bone_name_by_index.items():
        if _normalize_bone_name_for_tokens(str(raw_name or "")) == target_low:
            return int(bidx)
    return int(elev_idx)


def split_faces_by_bone_deterministic(
    part: Dict[str, Any],
    bone_name_by_index: Dict[int, str],
    bone_parent_by_index: Dict[int, int],
    material_role: str = "",
    material_name: str = "",
    part_index: int = -1,
) -> List[Dict[str, Any]]:
    idx = part.get("indices", [])
    xyz = part.get("vertices", {}).get("xyz", [])
    vertex_count = len(xyz) // 3

    def all_tris() -> List[Tuple[int, int, int]]:
        out: List[Tuple[int, int, int]] = []
        for i in range(0, len(idx), 3):
            if i + 2 >= len(idx):
                break
            out.append((int(idx[i + 0]), int(idx[i + 1]), int(idx[i + 2])))
        return out

    tris_all = all_tris()
    if not tris_all:
        return []

    dominant = dominant_bone_indices(part.get("vertices", {}), vertex_count)
    if not dominant:
        return [{
            "group_name_raw": "MainBody",
            "group_name_sanitized": sanitize_material_name("MainBody"),
            "group_bone_index": -1,
            "tris": tris_all,
        }]

    out: List[Dict[str, Any]] = []

    role = str(material_role or "").strip().lower()
    mat_low = str(material_name or "").strip().lower()
    default_track_side = ""
    if role == "track_left":
        default_track_side = "left"
    elif role == "track_right":
        default_track_side = "right"
    else:
        hint_side = _track_side_from_bone_name(mat_low)
        default_track_side = hint_side
    is_track_material = role.startswith("track") or ("tracks" in mat_low) or ("chenille" in mat_low)

    chenille_bone_by_side: Dict[str, int] = {}
    for bidx, bone_name in bone_name_by_index.items():
        low = str(bone_name or "").strip().lower()
        if "chenille_" not in low:
            continue
        side = _track_side_from_bone_name(low)
        if side and side not in chenille_bone_by_side:
            chenille_bone_by_side[side] = int(bidx)

    tri_infos: List[Dict[str, Any]] = []
    side_counter: Dict[str, int] = {"left": 0, "right": 0}

    for tri in tris_all:
        a, b, c = tri
        if min(a, b, c) < 0 or max(a, b, c) >= len(dominant):
            continue

        votes = [int(dominant[a]), int(dominant[b]), int(dominant[c])]
        vote_counts: Dict[int, int] = {}
        for v in votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1
        dom_bone = min(vote_counts.keys(), key=lambda x: (-vote_counts[x], int(x)))

        raw_bone_name = str(bone_name_by_index.get(int(dom_bone), f"bone_{int(dom_bone):03d}") or f"bone_{int(dom_bone):03d}")
        group_name = classify_group_from_bone_name(raw_bone_name)
        group_bone_index = int(dom_bone)
        tri_side = _track_side_from_bone_name(raw_bone_name)

        if group_name.startswith("Roue_"):
            center_idx = resolve_center_wheel_bone_index(int(dom_bone), bone_name_by_index)
            group_bone_index = int(center_idx)
            center_name = str(bone_name_by_index.get(int(center_idx), raw_bone_name) or raw_bone_name)
            center_group = classify_group_from_bone_name(center_name)
            if center_group.startswith("Roue_"):
                group_name = center_group

        if tri_side in side_counter:
            side_counter[tri_side] += 1

        tri_infos.append(
            {
                "tri": (int(a), int(b), int(c)),
                "group_name": str(group_name),
                "group_bone_index": int(group_bone_index),
                "tri_side": str(tri_side or ""),
                "raw_bone_name": str(raw_bone_name),
                "raw_bone_name_low": _normalize_bone_name_for_tokens(raw_bone_name),
                "group_name_source": "classifier",
            }
        )

    if not tri_infos:
        return [{
            "group_name_raw": "MainBody",
            "group_name_sanitized": sanitize_material_name("MainBody"),
            "group_bone_index": -1,
            "tris": tris_all,
        }]

    def _pick_track_side() -> str:
        if side_counter["left"] > side_counter["right"]:
            return "left"
        if side_counter["right"] > side_counter["left"]:
            return "right"
        if default_track_side in {"left", "right"}:
            return default_track_side
        return "right"

    part_track_side = _pick_track_side()

    grouped: Dict[str, Dict[str, Any]] = {}
    raw_group_stats: Dict[str, Dict[str, Any]] = {}
    total_triangles = len(tri_infos)
    for info in tri_infos:
        raw_low = str(info.get("raw_bone_name_low", "") or "")
        if not raw_low:
            continue
        row = raw_group_stats.get(raw_low)
        if row is None:
            row = {
                "raw_name": str(info.get("raw_bone_name", "") or raw_low),
                "triangles": 0,
                "groups": {},
            }
            raw_group_stats[raw_low] = row
        row["triangles"] = int(row.get("triangles", 0)) + 1
        groups_map = row["groups"]
        group_name = str(info.get("group_name", "") or "Chassis")
        groups_map[group_name] = int(groups_map.get(group_name, 0)) + 1

    preserved_group_name_by_raw_low: Dict[str, str] = {}
    for raw_low, row in raw_group_stats.items():
        groups_map = dict(row.get("groups", {}) or {})
        dominant_group = min(
            groups_map.keys(),
            key=lambda key: (-int(groups_map.get(key, 0)), str(key).lower()),
        ) if groups_map else "Chassis"
        preserved_group_name = _preserved_raw_geometry_group_name(
            raw_name=str(row.get("raw_name", "") or raw_low),
            triangle_count=int(row.get("triangles", 0)),
            total_triangles=int(total_triangles),
            classified_group=str(dominant_group),
        )
        if preserved_group_name:
            preserved_group_name_by_raw_low[str(raw_low)] = str(preserved_group_name)

    # Secondary deterministic signal for tracks: if caller could not mark track role,
    # but every classified triangle resolves to roue_* and the material name contains
    # common track tokens, force track mode to preserve chenille meshes.
    if (not is_track_material) and ("trk" in mat_low):
        if tri_infos and all(str(it.get("group_name", "")).startswith("Roue_") for it in tri_infos):
            is_track_material = True

    if is_track_material:
        # Strict track policy: all triangles from track materials stay in chenille buckets.
        # This avoids wheel contamination where small track fragments can be weighted to roue_* bones.
        side_left: List[Tuple[int, int, int]] = []
        side_right: List[Tuple[int, int, int]] = []
        fallback_tris: List[Tuple[int, int, int]] = []
        side_bone_hint: Dict[str, int] = {"left": -1, "right": -1}

        for info in tri_infos:
            tri = info["tri"]
            tri_side = str(info.get("tri_side", "") or "")
            gbone = int(info.get("group_bone_index", -1))
            if tri_side == "left":
                side_left.append(tri)
                if side_bone_hint["left"] < 0 and gbone >= 0:
                    side_bone_hint["left"] = gbone
            elif tri_side == "right":
                side_right.append(tri)
                if side_bone_hint["right"] < 0 and gbone >= 0:
                    side_bone_hint["right"] = gbone
            else:
                fallback_tris.append(tri)

        if fallback_tris:
            if part_track_side == "left":
                side_left.extend(fallback_tris)
            else:
                side_right.extend(fallback_tris)

        def _emit_track_bucket(side: str, tris: List[Tuple[int, int, int]]) -> None:
            if not tris:
                return
            gname = "Chenille_Gauche" if side == "left" else "Chenille_Droite"
            gbone = int(
                chenille_bone_by_side.get(
                    side,
                    side_bone_hint.get(side, -1),
                )
            )
            key = sanitize_material_name(gname).lower()
            grouped[key] = {
                "group_name_raw": gname,
                "group_name_sanitized": sanitize_material_name(gname),
                "group_bone_index": gbone,
                "tris": list(tris),
            }

        _emit_track_bucket("left", side_left)
        _emit_track_bucket("right", side_right)

        # Safety fallback: preserve geometry if side detection yielded no buckets.
        if not grouped:
            target_side = part_track_side
            gname = "Chenille_Gauche" if target_side == "left" else "Chenille_Droite"
            gbone = int(chenille_bone_by_side.get(target_side, -1))
            grouped[sanitize_material_name(gname).lower()] = {
                "group_name_raw": gname,
                "group_name_sanitized": sanitize_material_name(gname),
                "group_bone_index": gbone,
                "tris": [it["tri"] for it in tri_infos],
            }
    else:
        for info in tri_infos:
            group_name = str(info.get("group_name", ""))
            group_bone_index = int(info.get("group_bone_index", -1))
            raw_low = str(info.get("raw_bone_name_low", "") or "")
            preserved_group_name = preserved_group_name_by_raw_low.get(raw_low, "")
            group_name_source = str(info.get("group_name_source", "") or "classifier")
            if preserved_group_name:
                group_name = str(preserved_group_name)
                group_name_source = "raw_geometry_node"
            key = sanitize_material_name(group_name).lower()
            payload = grouped.get(key)
            if payload is None:
                payload = {
                    "group_name_raw": group_name,
                    "group_name_sanitized": sanitize_material_name(group_name),
                    "group_bone_index": int(group_bone_index),
                    "tris": [],
                    "_tri_infos": [],
                }
                grouped[key] = payload
            elif int(payload.get("group_bone_index", -1)) < 0 and int(group_bone_index) >= 0:
                payload["group_bone_index"] = int(group_bone_index)
            info_copy = dict(info)
            info_copy["group_name"] = str(group_name)
            info_copy["group_name_source"] = str(group_name_source)
            payload["_tri_infos"].append(info_copy)
            payload["tris"].append(info["tri"])

    for key in sorted(grouped.keys()):
        payload = grouped.get(key)
        if payload is None:
            continue
        tris = payload.get("tris", [])
        if not isinstance(tris, list) or not tris:
            continue
        tri_group_infos = list(payload.pop("_tri_infos", []) or [])
        raw_node_triangles: Dict[str, int] = {}
        raw_name_by_low: Dict[str, str] = {}
        source_counts: Dict[str, int] = {}
        for info in tri_group_infos:
            raw_low = str(info.get("raw_bone_name_low", "") or "")
            raw_name = str(info.get("raw_bone_name", "") or raw_low or "unknown")
            if raw_low:
                raw_node_triangles[raw_low] = int(raw_node_triangles.get(raw_low, 0)) + 1
                raw_name_by_low[raw_low] = raw_name
            src = str(info.get("group_name_source", "") or "classifier")
            source_counts[src] = int(source_counts.get(src, 0)) + 1
        dominant_raw_low = min(
            raw_node_triangles.keys(),
            key=lambda raw_low: (-int(raw_node_triangles.get(raw_low, 0)), str(raw_low)),
        ) if raw_node_triangles else ""
        dominant_raw_triangles = int(raw_node_triangles.get(dominant_raw_low, 0))
        foreign_triangles = max(0, int(len(tris)) - dominant_raw_triangles)
        materialization_source = min(
            source_counts.keys(),
            key=lambda src: (-int(source_counts.get(src, 0)), str(src).lower()),
        ) if source_counts else "classifier"
        payload["diagnostics"] = {
            "part_index": int(part_index),
            "material_role": str(role),
            "material_name": str(material_name or ""),
            "triangle_count": int(len(tris)),
            "group_name_source": str(materialization_source),
            "dominant_raw_node": str(raw_name_by_low.get(dominant_raw_low, dominant_raw_low)),
            "dominant_raw_triangles": int(dominant_raw_triangles),
            "foreign_raw_triangles": int(foreign_triangles),
            "contamination_verdict": "contaminated"
            if foreign_triangles >= max(24, int(len(tris) * 0.08))
            else "stable",
            "raw_node_triangles": {
                str(raw_name_by_low.get(raw_low, raw_low)): int(count)
                for raw_low, count in sorted(
                    raw_node_triangles.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )
            },
        }
        out.append(payload)
    return out


def _resnap_degenerate_uv_faces(
    vertex_rows: List[Tuple[float, float, float]],
    uv_rows: List[Tuple[float, float]],
    src_rows: List[int],
    faces: List[Tuple[int, ...]],
    min_span_threshold: float = 1.0e-3,
    area_threshold: float = 1.0e-5,
    max_iterations: int = 10,
) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[float, float]],
    List[int],
    List[Tuple[int, ...]],
    int,
]:
    """For every face (triangle or quad) whose UV is degenerate, find a
    neighbor face (sharing one edge) with non-degenerate UV and rebuild
    the degenerate face's UV by mirroring the neighbor's far-vertex UV
    across the shared edge midpoint.

    Degenerate is defined as either:
      * min(u_span, v_span) < min_span_threshold — stripe pattern
        (1px on a 1024 atlas at default 1e-3), or
      * u_span * v_span < area_threshold — fully collapsed point.
    The first condition catches the WARNO DECORS striping artifact where
    span looks like (0.041, 0.001); the second catches sub-pixel collapses.

    Why: WARNO composite DECORS assets (e.g. HLM_10_L.fbx aggregating 11
    sibling FBX) and a few unit edge cases ship with faces whose UV span
    is ~0 in one axis. Blender then samples a 1-2 pixel slice and repeats
    it across the entire face, producing visible stripes.

    Algorithm:
      * iterate to a fixed point (max `max_iterations` passes) so that
        degenerate clusters dissolve outward — once an outer face gets
        resnapped, its previously-also-degenerate inner neighbors finally
        have a non-degenerate neighbor to mirror against on the next pass.
      * supports both triangles (3 verts) and quads (4 verts).
      * non-edge vertices of the degenerate face are duplicated (new
        vertex + new UV) so other faces sharing those vertices are not
        disturbed by the UV change.
    Returns (vertices, uvs, src_refs, faces, resnapped_count).
    """
    if not uv_rows or not faces:
        return list(vertex_rows), list(uv_rows), list(src_rows), list(faces), 0

    new_vertices = list(vertex_rows)
    new_uvs = list(uv_rows)
    new_srcs = list(src_rows)
    new_faces = [tuple(face) for face in faces]
    resnapped = 0

    def _is_uv_degenerate(face: Sequence[int]) -> bool:
        if len(face) < 3:
            return False
        try:
            us = [float(new_uvs[v][0]) for v in face]
            vs_ = [float(new_uvs[v][1]) for v in face]
        except Exception:
            return False
        u_span = max(us) - min(us)
        v_span = max(vs_) - min(vs_)
        # 1) Stripe pattern (one axis collapsed to ~1px on a 1024 atlas).
        if min(u_span, v_span) < min_span_threshold:
            return True
        # 2) Sub-pixel area collapse.
        if u_span * v_span < area_threshold:
            return True
        # NOTE: We deliberately do NOT flag butterfly UV on quads as
        # degenerate. Eugen's source FBX in WARNO DECORS (e.g. HLM_10_L
        # rooftops) maps two diagonal corners of a quad to the same texel
        # by design — that is the in-game look the developers chose. Any
        # automatic re-projection here yields a worse, mismatched texture
        # for the modder. Users that want a different roof texture can
        # Smart-UV-Project the face manually in Edit Mode.
        return False

    for _iteration in range(int(max_iterations)):
        # Rebuild edge -> face map each pass so newly-resnapped faces
        # become available as donors for their still-degenerate neighbors.
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi, face in enumerate(new_faces):
            if len(face) < 3:
                continue
            n = len(face)
            for i in range(n):
                a = int(face[i])
                b = int(face[(i + 1) % n])
                edge = (min(a, b), max(a, b))
                edge_to_faces.setdefault(edge, []).append(fi)

        # Detect currently degenerate faces.
        degenerate_indices: List[int] = []
        for fi, face in enumerate(new_faces):
            if 3 <= len(face) <= 4 and _is_uv_degenerate(face):
                degenerate_indices.append(fi)
        if not degenerate_indices:
            break

        progressed = False
        for fi in degenerate_indices:
            face = new_faces[fi]
            face_verts = [int(v) for v in face]
            n = len(face_verts)
            # Find a non-degenerate neighbor via any shared edge.
            chosen: Tuple[int, int, int] | None = None
            for i in range(n):
                a = face_verts[i]
                b = face_verts[(i + 1) % n]
                edge = (min(a, b), max(a, b))
                for nfi in edge_to_faces.get(edge, []):
                    if nfi == fi:
                        continue
                    nface = new_faces[nfi]
                    if len(nface) < 3:
                        continue
                    if _is_uv_degenerate(nface):
                        continue
                    chosen = (nfi, a, b)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            nfi, ea, eb = chosen
            nface = new_faces[nfi]
            # Pick any neighbor vertex that is not on the shared edge.
            c_neighbor = -1
            for v in nface:
                iv = int(v)
                if iv != ea and iv != eb:
                    c_neighbor = iv
                    break
            if c_neighbor < 0:
                continue
            if ea >= len(new_uvs) or eb >= len(new_uvs) or c_neighbor >= len(new_uvs):
                continue
            ua = new_uvs[ea]
            ub = new_uvs[eb]
            uc_n = new_uvs[c_neighbor]
            # Skip donors whose own shared edge is degenerate in UV space —
            # their mirror would just collapse our face to another stripe.
            if abs(float(ua[0]) - float(ub[0])) < 1.0e-6 and abs(float(ua[1]) - float(ub[1])) < 1.0e-6:
                continue
            mid_u = (float(ua[0]) + float(ub[0])) * 0.5
            mid_v = (float(ua[1]) + float(ub[1])) * 0.5
            mirrored_uv = (
                2.0 * mid_u - float(uc_n[0]),
                2.0 * mid_v - float(uc_n[1]),
            )
            # Detect butterfly only by strict diagonal-collapse — same rule
            # as _is_uv_degenerate above. Keep these in sync.
            is_butterfly = False
            if n == 4:
                eps = 1.0e-5
                try:
                    fu = [float(new_uvs[v][0]) for v in face_verts]
                    fv = [float(new_uvs[v][1]) for v in face_verts]
                    d02 = ((fu[0] - fu[2]) ** 2 + (fv[0] - fv[2]) ** 2) ** 0.5
                    d13 = ((fu[1] - fu[3]) ** 2 + (fv[1] - fv[3]) ** 2) ** 0.5
                    if d02 < eps or d13 < eps:
                        is_butterfly = True
                except Exception:
                    is_butterfly = False

            if is_butterfly:
                # For butterfly quads, single-point mirror leaves the whole
                # face as a flat-color patch — visually worse than the source
                # stripe in many cases. Instead reproject our face's 3D
                # positions onto the neighbor's UV bounding box, preserving
                # the size ratio so the texture tiles at the same resolution
                # as on the donor face. This yields a roof that looks like
                # an extension of the neighboring wall pattern instead of a
                # uniform color blob.
                try:
                    nb_us = [float(new_uvs[int(v)][0]) for v in nface]
                    nb_vs_ = [float(new_uvs[int(v)][1]) for v in nface]
                    nb_u_min, nb_u_max = min(nb_us), max(nb_us)
                    nb_v_min, nb_v_max = min(nb_vs_), max(nb_vs_)
                    nb_u_span = max(nb_u_max - nb_u_min, 1.0e-6)
                    nb_v_span = max(nb_v_max - nb_v_min, 1.0e-6)

                    our_pts3d = [tuple(new_vertices[v]) for v in face_verts]
                    nb_pts3d = [tuple(new_vertices[int(v)]) for v in nface]

                    # Pick the two 3D axes with the largest spread on our face
                    # (drops the dominant-normal axis).
                    spans = [
                        max(p[i] for p in our_pts3d) - min(p[i] for p in our_pts3d)
                        for i in range(3)
                    ]
                    axis_priority = sorted(range(3), key=lambda i: -spans[i])
                    u_axis, v_axis = axis_priority[0], axis_priority[1]

                    our_u_min = min(p[u_axis] for p in our_pts3d)
                    our_v_min = min(p[v_axis] for p in our_pts3d)
                    our_u_span = max(spans[u_axis], 1.0e-6)
                    our_v_span = max(spans[v_axis], 1.0e-6)

                    nb_u_min_axis = min(p[u_axis] for p in nb_pts3d)
                    nb_v_min_axis = min(p[v_axis] for p in nb_pts3d)
                    nb_u_span_axis = max(
                        (max(p[u_axis] for p in nb_pts3d) - nb_u_min_axis),
                        1.0e-6,
                    )
                    nb_v_span_axis = max(
                        (max(p[v_axis] for p in nb_pts3d) - nb_v_min_axis),
                        1.0e-6,
                    )

                    # UV-per-3D-unit on the donor; we use the same density
                    # so 1m on our face occupies the same UV distance.
                    u_density = nb_u_span / nb_u_span_axis
                    v_density = nb_v_span / nb_v_span_axis

                    new_uv_list: List[Tuple[float, float]] = []
                    for p in our_pts3d:
                        u = nb_u_min + (p[u_axis] - our_u_min) * u_density
                        v = nb_v_min + (p[v_axis] - our_v_min) * v_density
                        new_uv_list.append((u, v))
                except Exception:
                    new_uv_list = [mirrored_uv] * n

                rebuilt: List[int] = []
                for vi, v in enumerate(face_verts):
                    if v >= len(new_vertices):
                        rebuilt.append(v)
                        continue
                    new_idx = len(new_vertices)
                    new_vertices.append(tuple(new_vertices[v]))
                    new_uvs.append(new_uv_list[vi])
                    new_srcs.append(new_srcs[v] if v < len(new_srcs) else 0)
                    rebuilt.append(new_idx)
                new_faces[fi] = tuple(rebuilt)
                resnapped += 1
                progressed = True
            else:
                # Normal stripe / area-collapse path: keep the shared-edge
                # vertices, duplicate the rest with the mirrored UV.
                rebuilt = []
                for v in face_verts:
                    if v == ea or v == eb:
                        rebuilt.append(v)
                        continue
                    if v >= len(new_vertices):
                        rebuilt.append(v)
                        continue
                    new_idx = len(new_vertices)
                    new_vertices.append(tuple(new_vertices[v]))
                    new_uvs.append(mirrored_uv)
                    new_srcs.append(new_srcs[v] if v < len(new_srcs) else 0)
                    rebuilt.append(new_idx)
                new_faces[fi] = tuple(rebuilt)
                resnapped += 1
                progressed = True

        if not progressed:
            break

    return new_vertices, new_uvs, new_srcs, new_faces, resnapped


def cleanup_bucket_geometry(
    vertices: Sequence[Sequence[float]],
    uvs: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    face_mids: Sequence[int] | None = None,
    source_refs: Sequence[Sequence[int]] | None = None,
    group_name: str = "",
) -> Dict[str, Any]:
    vertex_rows = [
        (
            float(row[0]) if len(row) >= 1 else 0.0,
            float(row[1]) if len(row) >= 2 else 0.0,
            float(row[2]) if len(row) >= 3 else 0.0,
        )
        for row in list(vertices or [])
    ]
    uv_rows = [
        (
            float(row[0]) if len(row) >= 1 else 0.0,
            float(row[1]) if len(row) >= 2 else 0.0,
        )
        for row in list(uvs or [])
    ]
    if len(uv_rows) < len(vertex_rows):
        uv_rows.extend([(0.0, 0.0)] * (len(vertex_rows) - len(uv_rows)))
    src_rows = [
        (
            int(row[0]) if len(row) >= 1 else -1,
            int(row[1]) if len(row) >= 2 else -1,
        )
        for row in list(source_refs or [])
    ]
    if len(src_rows) < len(vertex_rows):
        src_rows.extend([(-1, -1)] * (len(vertex_rows) - len(src_rows)))
    mids_rows = [int(mid) for mid in list(face_mids or [])]
    if len(mids_rows) < len(faces):
        mids_rows.extend([-1] * (len(faces) - len(mids_rows)))

    def _polygon_edges(face: Sequence[int]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        if len(face) < 2:
            return out
        for idx in range(len(face)):
            a = int(face[idx])
            b = int(face[(idx + 1) % len(face)])
            out.append((a, b))
        return out

    def _triangle_area(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        ab = (
            float(b[0]) - float(a[0]),
            float(b[1]) - float(a[1]),
            float(b[2]) - float(a[2]),
        )
        ac = (
            float(c[0]) - float(a[0]),
            float(c[1]) - float(a[1]),
            float(c[2]) - float(a[2]),
        )
        cx = ab[1] * ac[2] - ab[2] * ac[1]
        cy = ab[2] * ac[0] - ab[0] * ac[2]
        cz = ab[0] * ac[1] - ab[1] * ac[0]
        return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

    def _polygon_area(face: Sequence[int]) -> float:
        if len(face) < 3:
            return 0.0
        try:
            anchor = vertex_rows[int(face[0])]
        except Exception:
            return 0.0
        area = 0.0
        for idx in range(1, len(face) - 1):
            try:
                area += _triangle_area(anchor, vertex_rows[int(face[idx])], vertex_rows[int(face[idx + 1])])
            except Exception:
                return 0.0
        return float(area)

    def _component_groups(face_rows: Sequence[Sequence[int]]) -> List[List[int]]:
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for face_i, face in enumerate(face_rows):
            for a, b in _polygon_edges(face):
                edge = (min(int(a), int(b)), max(int(a), int(b)))
                edge_to_faces.setdefault(edge, []).append(int(face_i))
        adjacency: Dict[int, set[int]] = {}
        for face_ids in edge_to_faces.values():
            if len(face_ids) < 2:
                continue
            for face_id in face_ids:
                adjacency.setdefault(int(face_id), set()).update(int(other) for other in face_ids if int(other) != int(face_id))
        visited: set[int] = set()
        groups: List[List[int]] = []
        for face_i in range(len(face_rows)):
            if face_i in visited:
                continue
            queue = [int(face_i)]
            visited.add(int(face_i))
            group: List[int] = []
            while queue:
                cur = queue.pop()
                group.append(int(cur))
                for nxt in sorted(adjacency.get(int(cur), set())):
                    if nxt in visited:
                        continue
                    visited.add(int(nxt))
                    queue.append(int(nxt))
            groups.append(group)
        return groups

    def _compact(
        face_rows: Sequence[Sequence[int]],
        mid_rows: Sequence[int],
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]], List[Tuple[int, int]], List[Tuple[int, ...]], List[int]]:
        used: set[int] = set()
        for face in face_rows:
            for vi in face:
                if 0 <= int(vi) < len(vertex_rows):
                    used.add(int(vi))
        order = sorted(used)
        remap = {old: new for new, old in enumerate(order)}
        out_vertices = [vertex_rows[idx] for idx in order]
        out_uvs = [uv_rows[idx] for idx in order]
        out_refs = [src_rows[idx] for idx in order]
        out_faces: List[Tuple[int, ...]] = []
        out_mids: List[int] = []
        for face_i, face in enumerate(face_rows):
            mapped = tuple(int(remap[int(vi)]) for vi in face if int(vi) in remap)
            if len(mapped) >= 3:
                out_faces.append(mapped)
                out_mids.append(int(mid_rows[face_i]) if face_i < len(mid_rows) else -1)
        return out_vertices, out_uvs, out_refs, out_faces, out_mids

    epsilon = 1.0e-6
    welded_vertices: List[Tuple[float, float, float]] = []
    welded_uvs: List[Tuple[float, float]] = []
    welded_refs: List[Tuple[int, int]] = []
    weld_map: Dict[Tuple[int, int, int, int, int], int] = {}
    old_to_new: Dict[int, int] = {}
    for old_idx, pos in enumerate(vertex_rows):
        uv_row = uv_rows[old_idx] if old_idx < len(uv_rows) else (0.0, 0.0)
        key = (
            int(round(float(pos[0]) / epsilon)),
            int(round(float(pos[1]) / epsilon)),
            int(round(float(pos[2]) / epsilon)),
            int(round(float(uv_row[0]) / epsilon)),
            int(round(float(uv_row[1]) / epsilon)),
        )
        mapped = weld_map.get(key)
        if mapped is None:
            mapped = len(welded_vertices)
            weld_map[key] = mapped
            welded_vertices.append(pos)
            welded_uvs.append(uv_row)
            welded_refs.append(src_rows[old_idx] if old_idx < len(src_rows) else (-1, -1))
        old_to_new[int(old_idx)] = int(mapped)

    face_rows: List[Tuple[int, ...]] = []
    mid_rows: List[int] = []
    degenerate_removed = 0
    for face_i, face in enumerate(list(faces or [])):
        mapped = tuple(int(old_to_new.get(int(vi), -1)) for vi in list(face or []))
        if len(mapped) < 3:
            degenerate_removed += 1
            continue
        if len(set(mapped)) < 3:
            degenerate_removed += 1
            continue
        if any(int(vi) < 0 or int(vi) >= len(welded_vertices) for vi in mapped):
            degenerate_removed += 1
            continue
        area = 0.0
        try:
            anchor = welded_vertices[int(mapped[0])]
            for idx in range(1, len(mapped) - 1):
                area += _triangle_area(anchor, welded_vertices[int(mapped[idx])], welded_vertices[int(mapped[idx + 1])])
        except Exception:
            area = 0.0
        if area <= 1.0e-10:
            degenerate_removed += 1
            continue
        face_rows.append(mapped)
        mid_rows.append(int(mids_rows[face_i]) if face_i < len(mids_rows) else -1)

    vertex_rows = welded_vertices
    uv_rows = welded_uvs
    src_rows = welded_refs
    pre_component_count = len(_component_groups(face_rows))
    total_area = sum(_polygon_area(face) for face in face_rows)
    used_vertices = sorted({int(vi) for face in face_rows for vi in face})
    if used_vertices:
        xs = [vertex_rows[idx][0] for idx in used_vertices]
        ys = [vertex_rows[idx][1] for idx in used_vertices]
        zs = [vertex_rows[idx][2] for idx in used_vertices]
        total_diag = math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    else:
        total_diag = 0.0

    dropped_components = 0
    components = _component_groups(face_rows)
    if len(components) > 1:
        keep_face_ids: set[int] = set()
        area_floor = max(1.0e-8, float(total_area) * 0.0004)
        diag_floor = max(0.0025, float(total_diag) * 0.015)
        for comp in components:
            comp_faces = [face_rows[idx] for idx in comp]
            comp_vertices = sorted({int(vi) for face in comp_faces for vi in face})
            comp_area = sum(_polygon_area(face) for face in comp_faces)
            if comp_vertices:
                xs = [vertex_rows[idx][0] for idx in comp_vertices]
                ys = [vertex_rows[idx][1] for idx in comp_vertices]
                zs = [vertex_rows[idx][2] for idx in comp_vertices]
                comp_diag = math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
            else:
                comp_diag = 0.0
            should_drop = (
                len(comp_faces) <= 2
                and len(comp_vertices) <= 6
                and comp_area <= area_floor
                and comp_diag <= diag_floor
            )
            if should_drop:
                dropped_components += 1
                continue
            keep_face_ids.update(int(idx) for idx in comp)
        if keep_face_ids:
            kept_faces: List[Tuple[int, ...]] = []
            kept_mids: List[int] = []
            for face_i, face in enumerate(face_rows):
                if face_i not in keep_face_ids:
                    continue
                kept_faces.append(face)
                kept_mids.append(mid_rows[face_i])
            face_rows = kept_faces
            mid_rows = kept_mids
            vertex_rows, uv_rows, src_rows, face_rows, mid_rows = _compact(face_rows, mid_rows)

    def _face_normal(face: Sequence[int]) -> Tuple[float, float, float]:
        if len(face) < 3:
            return (0.0, 0.0, 0.0)
        a = vertex_rows[int(face[0])]
        b = vertex_rows[int(face[1])]
        c = vertex_rows[int(face[2])]
        ab = (
            float(b[0]) - float(a[0]),
            float(b[1]) - float(a[1]),
            float(b[2]) - float(a[2]),
        )
        ac = (
            float(c[0]) - float(a[0]),
            float(c[1]) - float(a[1]),
            float(c[2]) - float(a[2]),
        )
        nx = ab[1] * ac[2] - ab[2] * ac[1]
        ny = ab[2] * ac[0] - ab[0] * ac[2]
        nz = ab[0] * ac[1] - ab[1] * ac[0]
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln <= 1.0e-12:
            return (0.0, 0.0, 0.0)
        return (nx / ln, ny / ln, nz / ln)

    def _safe_quad_from_pair(face_a: Sequence[int], face_b: Sequence[int]) -> Tuple[int, int, int, int] | None:
        if len(face_a) != 3 or len(face_b) != 3:
            return None
        shared = sorted(set(int(v) for v in face_a) & set(int(v) for v in face_b))
        if len(shared) != 2:
            return None
        unique = [int(v) for v in list(face_a) + list(face_b) if int(v) not in shared]
        if len(unique) != 2:
            return None
        boundary_counts: Dict[Tuple[int, int], int] = {}
        for edge in _polygon_edges(face_a):
            key = (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
            boundary_counts[key] = int(boundary_counts.get(key, 0)) + 1
        for edge in _polygon_edges(face_b):
            key = (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
            boundary_counts[key] = int(boundary_counts.get(key, 0)) + 1
        boundary_edges = [edge for edge, count in boundary_counts.items() if int(count) == 1]
        if len(boundary_edges) != 4:
            return None
        adjacency: Dict[int, List[int]] = {}
        for a, b in boundary_edges:
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))
        if len(adjacency) != 4 or any(len(nei) != 2 for nei in adjacency.values()):
            return None
        start = min(adjacency.keys())
        cycle = [int(start)]
        prev = None
        cur = int(start)
        for _ in range(3):
            candidates = [n for n in adjacency[cur] if n != prev]
            if not candidates:
                return None
            nxt = min(candidates)
            cycle.append(int(nxt))
            prev, cur = cur, nxt
        if len(set(cycle)) != 4:
            return None
        if start not in adjacency.get(int(cycle[-1]), []):
            return None
        norm_a = _face_normal(face_a)
        norm_b = _face_normal(face_b)
        dot = sum(float(norm_a[i]) * float(norm_b[i]) for i in range(3))
        if dot < 0.9990:
            return None
        avg = (
            norm_a[0] + norm_b[0],
            norm_a[1] + norm_b[1],
            norm_a[2] + norm_b[2],
        )
        avg_len = math.sqrt(avg[0] * avg[0] + avg[1] * avg[1] + avg[2] * avg[2])
        if avg_len <= 1.0e-12:
            return None
        avg = (avg[0] / avg_len, avg[1] / avg_len, avg[2] / avg_len)
        quad = tuple(int(v) for v in cycle)
        for idx in range(4):
            p0 = vertex_rows[int(quad[idx])]
            p1 = vertex_rows[int(quad[(idx + 1) % 4])]
            p2 = vertex_rows[int(quad[(idx + 2) % 4])]
            e1 = (
                float(p1[0]) - float(p0[0]),
                float(p1[1]) - float(p0[1]),
                float(p1[2]) - float(p0[2]),
            )
            e2 = (
                float(p2[0]) - float(p1[0]),
                float(p2[1]) - float(p1[1]),
                float(p2[2]) - float(p1[2]),
            )
            cx = e1[1] * e2[2] - e1[2] * e2[1]
            cy = e1[2] * e2[0] - e1[0] * e2[2]
            cz = e1[0] * e2[1] - e1[1] * e2[0]
            if (cx * avg[0] + cy * avg[1] + cz * avg[2]) <= 1.0e-10:
                return None
        if _polygon_area(quad) <= 1.0e-10:
            return None
        return quad

    used_faces: set[int] = set()
    merged_faces: List[Tuple[int, ...]] = []
    merged_mids: List[int] = []
    quad_merges = 0
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_i, face in enumerate(face_rows):
        if len(face) != 3:
            continue
        for a, b in _polygon_edges(face):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            edge_to_faces.setdefault(edge, []).append(int(face_i))
    for face_i, face in enumerate(face_rows):
        if face_i in used_faces:
            continue
        if len(face) != 3:
            merged_faces.append(tuple(face))
            merged_mids.append(mid_rows[face_i] if face_i < len(mid_rows) else -1)
            used_faces.add(int(face_i))
            continue
        best_pair = -1
        for a, b in _polygon_edges(face):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            candidates = edge_to_faces.get(edge, [])
            if len(candidates) != 2:
                continue
            other = candidates[0] if candidates[1] == face_i else candidates[1]
            if other == face_i or other in used_faces:
                continue
            if other >= len(face_rows) or len(face_rows[other]) != 3:
                continue
            if int(mid_rows[other]) != int(mid_rows[face_i]):
                continue
            best_pair = int(other)
            break
        if best_pair >= 0:
            quad = _safe_quad_from_pair(face, face_rows[best_pair])
            if quad is not None:
                merged_faces.append(tuple(quad))
                merged_mids.append(mid_rows[face_i] if face_i < len(mid_rows) else -1)
                used_faces.add(int(face_i))
                used_faces.add(int(best_pair))
                quad_merges += 1
                continue
        merged_faces.append(tuple(face))
        merged_mids.append(mid_rows[face_i] if face_i < len(mid_rows) else -1)
        used_faces.add(int(face_i))

    vertex_rows, uv_rows, src_rows, merged_faces, merged_mids = _compact(merged_faces, merged_mids)
    # Re-snap any remaining triangles whose UV-rectangle collapsed below the
    # area threshold. This is a Blender-rendering fix for source FBX shipped
    # by Eugen with degenerate UV in WARNO composite DECORS and rare unit
    # corner cases. See _resnap_degenerate_uv_faces docstring for the full
    # rationale; the mirror-from-neighbor strategy preserves geometry and
    # replaces 1-pixel stripes with a copy of the adjacent texture region.
    vertex_rows, uv_rows, src_rows, merged_faces, uv_resnapped_count = _resnap_degenerate_uv_faces(
        list(vertex_rows),
        list(uv_rows),
        list(src_rows),
        list(merged_faces),
    )
    post_component_count = len(_component_groups(merged_faces))
    diagnostics = {
        "group_name": str(group_name or ""),
        "vertex_count_pre": int(len(vertices or [])),
        "polygon_count_pre": int(len(faces or [])),
        "vertex_count_post": int(len(vertex_rows)),
        "polygon_count_post": int(len(merged_faces)),
        "component_count_pre": int(pre_component_count),
        "component_count_post": int(post_component_count),
        "degenerate_polygons_removed": int(degenerate_removed),
        "tiny_components_removed": int(dropped_components),
        "quad_merges": int(quad_merges),
        "uv_degenerate_resnapped": int(uv_resnapped_count),
    }
    return {
        "vertices": list(vertex_rows),
        "uvs": list(uv_rows),
        "source_refs": list(src_rows),
        "faces": list(merged_faces),
        "face_mids": list(merged_mids),
        "diagnostics": diagnostics,
    }


def split_faces_by_bone_top_level(
    part: Dict[str, Any],
    bone_name_by_index: Dict[int, str],
    bone_parent_by_index: Dict[int, int],
    material_role: str = "",
    material_name: str = "",
) -> List[Dict[str, Any]]:
    idx = part.get("indices", [])
    xyz = part.get("vertices", {}).get("xyz", [])
    vertex_count = len(xyz) // 3

    def all_tris() -> List[Tuple[int, int, int]]:
        out: List[Tuple[int, int, int]] = []
        for i in range(0, len(idx), 3):
            if i + 2 >= len(idx):
                break
            out.append((int(idx[i + 0]), int(idx[i + 1]), int(idx[i + 2])))
        return out

    dominant = dominant_bone_indices(part.get("vertices", {}), vertex_count)
    if not dominant:
        return [{
            "group_name_raw": "MainBody",
            "group_name_sanitized": sanitize_material_name("MainBody"),
            "group_bone_index": -1,
            "tris": all_tris(),
        }]

    role = str(material_role or "").strip().lower()
    mat_low = str(material_name or "").strip().lower()

    top_cache: Dict[int, int] = {}

    def top_level_for_bone(bidx: int) -> int:
        key = int(bidx)
        hit = top_cache.get(key)
        if hit is not None:
            return int(hit)
        top = _top_level_child_bone_index(key, bone_parent_by_index)
        top_cache[key] = int(top)
        return int(top)

    def representative_top_level_bone() -> int:
        counts: Dict[int, int] = {}
        for b in dominant:
            bb = int(b)
            counts[bb] = counts.get(bb, 0) + 1
        if not counts:
            return -1
        dom_bone = min(counts.keys(), key=lambda x: (-counts[x], int(x)))
        return top_level_for_bone(dom_bone)

    if role == "track_left" or "chenille_gauche" in mat_low or "track_left" in mat_low:
        return [{
            "group_name_raw": "Chenille_Gauche",
            "group_name_sanitized": sanitize_material_name("Chenille_Gauche"),
            "group_bone_index": int(representative_top_level_bone()),
            "tris": all_tris(),
        }]
    if role == "track_right" or "chenille_droite" in mat_low or "track_right" in mat_low:
        return [{
            "group_name_raw": "Chenille_Droite",
            "group_name_sanitized": sanitize_material_name("Chenille_Droite"),
            "group_bone_index": int(representative_top_level_bone()),
            "tris": all_tris(),
        }]

    grouped: Dict[int, List[Tuple[int, int, int]]] = {}
    for tri in all_tris():
        a, b, c = tri
        if min(a, b, c) < 0 or max(a, b, c) >= len(dominant):
            continue
        votes = [int(dominant[a]), int(dominant[b]), int(dominant[c])]
        counts: Dict[int, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        dom_bone = min(counts.keys(), key=lambda x: (-counts[x], int(x)))
        top = top_level_for_bone(dom_bone)
        grouped.setdefault(int(top), []).append(tri)

    if not grouped:
        return [{
            "group_name_raw": "MainBody",
            "group_name_sanitized": sanitize_material_name("MainBody"),
            "group_bone_index": -1,
            "tris": all_tris(),
        }]

    out: List[Dict[str, Any]] = []
    for bidx in sorted(grouped.keys()):
        tris = grouped.get(int(bidx), [])
        if not tris:
            continue
        raw_name = str(bone_name_by_index.get(int(bidx), f"bone_{int(bidx):03d}") or f"bone_{int(bidx):03d}")
        out.append({
            "group_name_raw": raw_name,
            "group_name_sanitized": sanitize_material_name(raw_name),
            "group_bone_index": int(bidx),
            "tris": tris,
        })
    return out


def split_faces_by_bone(
    part: Dict[str, Any],
    part_fallback_name: str,
    bone_name_by_index: Dict[int, str],
    material_role: str = "",
    material_name: str = "",
    wheel_tuning: Dict[str, Any] | None = None,
) -> List[Tuple[str, List[Tuple[int, int, int]]]]:
    idx = part["indices"]
    xyz = part["vertices"]["xyz"]
    vertex_count = len(xyz) // 3

    def all_tris() -> List[Tuple[int, int, int]]:
        out: List[Tuple[int, int, int]] = []
        for i in range(0, len(idx), 3):
            if i + 2 < len(idx):
                out.append((int(idx[i + 0]), int(idx[i + 1]), int(idx[i + 2])))
        return out

    role = (material_role or "").strip().lower()
    mat_low = (material_name or "").strip().lower()
    if role == "track_left" or "chenille_gauche" in mat_low or "track_left" in mat_low:
        return [("Chenille_Gauche_01", all_tris())]
    if role == "track_right" or "chenille_droite" in mat_low or "track_right" in mat_low:
        return [("Chenille_Droite_01", all_tris())]

    wt = wheel_tuning or {}

    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(val)))

    pair_fix_enabled = bool(wt.get("enabled", True))
    pair_dist_scale = _clamp(float(wt.get("pair_dist_scale", 0.42)), 0.15, 1.25)
    pair_edge_scale = _clamp(float(wt.get("pair_edge_scale", 1.08)), 1.0, 1.8)
    pair_target_ratio = _clamp(float(wt.get("pair_target_ratio", 0.97)), 0.75, 1.0)
    pair_min_pool_ratio = _clamp(float(wt.get("pair_min_pool_ratio", 0.60)), 0.10, 1.0)
    pair_axial_scale = _clamp(float(wt.get("pair_axial_scale", 1.15)), 0.80, 2.0)
    pair_ring_min = _clamp(float(wt.get("pair_ring_min", 0.08)), 0.0, 0.6)
    pair_ring_max = _clamp(float(wt.get("pair_ring_max", 1.08)), 0.8, 1.5)

    dominant = dominant_bone_indices(part["vertices"], vertex_count)
    influence_sets = influential_bone_sets(part["vertices"], vertex_count, min_weight=0.1)
    if not dominant:
        return [(part_fallback_name, all_tris())]

    def classify_group(raw_name: str) -> str:
        n = raw_name.lower()
        if n.startswith("bone_"):
            return "Chassis"
        if "chenille_gauche" in n:
            return "Chenille_Gauche"
        if "chenille_droite" in n:
            return "Chenille_Droite"
        if "track_gauche" in n or "track_left" in n:
            return "Chenille_Gauche"
        if "track_droite" in n or "track_right" in n:
            return "Chenille_Droite"
        if n.startswith("roue_elev_"):
            parts = raw_name.split("_", 2)
            if len(parts) == 3 and parts[2]:
                return pretty_part_name(f"roue_{parts[2]}")
            return "Roue_01"
        if n.startswith("roue_"):
            return pretty_part_name(raw_name)
        if n.startswith("fx_tourelle"):
            return "Tourelle_01"
        if n.startswith("fx_canon"):
            return "Canon_01"
        if n.startswith("fx_chenille"):
            if "gauche" in n or "left" in n:
                return "Chenille_Gauche_01"
            if "droite" in n or "right" in n:
                return "Chenille_Droite_01"
        if n.startswith("tourelle_") or "turret" in n:
            return "Tourelle_01"
        if n.startswith("axe_canon") or n == "axe" or n.startswith("axe_"):
            return "Axe_Canon_01"
        if n.startswith("canon_") or "gun_" in n or "barrel" in n:
            return "Canon_01"
        if n.startswith("fx_"):
            return "Chassis"
        if "chassis" in n or n in {"base", "chassisarmaturefake", "chassisfake"}:
            return "Chassis"
        return "Chassis"

    grouped: Dict[str, List[Tuple[int, int, int]]] = {}
    for i in range(0, len(idx), 3):
        if i + 2 >= len(idx):
            break
        a, b, c = int(idx[i + 0]), int(idx[i + 1]), int(idx[i + 2])
        if a < 0 or b < 0 or c < 0:
            continue
        if a >= vertex_count or b >= vertex_count or c >= vertex_count:
            continue

        bones = [dominant[a], dominant[b], dominant[c]]
        key = max(set(bones), key=bones.count)
        bone_name = bone_name_by_index.get(key, f"bone_{key:03d}")
        grp_name = sanitize_material_name(classify_group(bone_name))
        grouped.setdefault(grp_name, []).append((a, b, c))

    # Some wheel faces in WARNO meshes are weighted/chosen as chassis.
    # Pull near-wheel chassis triangles into wheel objects by spatial proximity.
    if "Chassis" in grouped:
        wheel_names = [name for name in grouped.keys() if name.startswith("Roue_")]
        if wheel_names:
            wheel_metrics: Dict[str, Dict[str, Any]] = {}
            wheel_bones_by_name: Dict[str, set[int]] = {}

            for bidx, bname in bone_name_by_index.items():
                grp = sanitize_material_name(classify_group(bname))
                if grp.startswith("Roue_"):
                    wheel_bones_by_name.setdefault(grp, set()).add(int(bidx))

            for wn in wheel_names:
                tri_list = grouped.get(wn, [])
                if not tri_list:
                    continue

                verts: set[int] = set()
                for a, b, c in tri_list:
                    verts.add(a)
                    verts.add(b)
                    verts.add(c)
                if len(verts) < 8:
                    continue

                wheel_bones = wheel_bones_by_name.get(wn, set())
                if wheel_bones:
                    raw_idx = part["vertices"].get("bone_idx") or []
                    raw_w = part["vertices"].get("bone_w") or []
                    strong_verts: set[int] = set()
                    if len(raw_idx) >= vertex_count * 4 and len(raw_w) >= vertex_count * 4:
                        for v in verts:
                            if v < 0 or v >= vertex_count:
                                continue
                            base = v * 4
                            idxs = (
                                int(raw_idx[base + 0]),
                                int(raw_idx[base + 1]),
                                int(raw_idx[base + 2]),
                                int(raw_idx[base + 3]),
                            )
                            ws = (
                                float(raw_w[base + 0]),
                                float(raw_w[base + 1]),
                                float(raw_w[base + 2]),
                                float(raw_w[base + 3]),
                            )
                            best_i = max(range(4), key=lambda i: ws[i])
                            if idxs[best_i] in wheel_bones and ws[best_i] >= 0.45:
                                strong_verts.add(int(v))
                    elif influence_sets:
                        strong_verts = {
                            int(v)
                            for v in verts
                            if 0 <= int(v) < len(influence_sets) and (influence_sets[int(v)] & wheel_bones)
                        }
                    if len(strong_verts) >= 8:
                        verts = strong_verts

                def percentile(values: Sequence[float], q: float) -> float:
                    seq = sorted(float(v) for v in values)
                    if not seq:
                        return 0.0
                    if q <= 0.0:
                        return seq[0]
                    if q >= 1.0:
                        return seq[-1]
                    pos = (len(seq) - 1) * q
                    lo = int(math.floor(pos))
                    hi = int(math.ceil(pos))
                    if lo == hi:
                        return seq[lo]
                    t = pos - lo
                    return seq[lo] * (1.0 - t) + seq[hi] * t

                xs = [float(xyz[v * 3 + 0]) for v in verts]
                ys = [float(xyz[v * 3 + 1]) for v in verts]
                zs = [float(xyz[v * 3 + 2]) for v in verts]
                center = (
                    percentile(xs, 0.50),
                    percentile(ys, 0.50),
                    percentile(zs, 0.50),
                )
                mins = (percentile(xs, 0.05), percentile(ys, 0.05), percentile(zs, 0.05))
                maxs = (percentile(xs, 0.95), percentile(ys, 0.95), percentile(zs, 0.95))
                dims = (
                    maxs[0] - mins[0],
                    maxs[1] - mins[1],
                    maxs[2] - mins[2],
                )

                thin_axis = min(range(3), key=lambda i: dims[i])
                other_axes = [i for i in (0, 1, 2) if i != thin_axis]
                rr_vals: List[float] = []
                for v in verts:
                    px = float(xyz[v * 3 + 0])
                    py = float(xyz[v * 3 + 1])
                    pz = float(xyz[v * 3 + 2])
                    p = (px, py, pz)
                    rr = math.hypot(
                        p[other_axes[0]] - center[other_axes[0]],
                        p[other_axes[1]] - center[other_axes[1]],
                    )
                    rr_vals.append(rr)
                if not rr_vals:
                    continue

                radial = percentile(rr_vals, 0.85)
                radial_inner = percentile(rr_vals, 0.20)
                thickness = max(float(dims[thin_axis]), radial * 0.12, 1e-6)
                if radial <= 1e-6:
                    continue

                wheel_metrics[wn] = {
                    "center": center,
                    "thin_axis": thin_axis,
                    "other_axes": other_axes,
                    "radial": radial,
                    "radial_inner": radial_inner,
                    "thickness": thickness,
                }

            if wheel_metrics:
                def median_val(vals: Sequence[float]) -> float:
                    seq = sorted(float(v) for v in vals if float(v) > 0.0)
                    if not seq:
                        return 0.0
                    n = len(seq)
                    mid = n // 2
                    if n % 2 == 1:
                        return seq[mid]
                    return 0.5 * (seq[mid - 1] + seq[mid])

                def tri_wheel_support(a: int, b: int, c: int, wheel_bones: set[int]) -> int:
                    if not influence_sets or not wheel_bones:
                        return 3
                    support = 0
                    for vi in (a, b, c):
                        if 0 <= vi < len(influence_sets) and (influence_sets[vi] & wheel_bones):
                            support += 1
                    return support

                def tri_wheel_dominant_support(a: int, b: int, c: int, wheel_bones: set[int]) -> int:
                    if not dominant or not wheel_bones:
                        return 3
                    support = 0
                    for vi in (a, b, c):
                        if 0 <= vi < len(dominant) and dominant[vi] in wheel_bones:
                            support += 1
                    return support

                def tri_wheel_score(
                    a: int,
                    b: int,
                    c: int,
                    m: Dict[str, Any],
                ) -> float | None:
                    center = m["center"]
                    thin_axis = int(m["thin_axis"])
                    other_axes = m["other_axes"]
                    radial = float(m["radial"])
                    radial_inner = float(m.get("radial_inner", radial * 0.25))
                    thickness = float(m["thickness"])

                    tri_center = (
                        (float(xyz[a * 3 + 0]) + float(xyz[b * 3 + 0]) + float(xyz[c * 3 + 0])) / 3.0,
                        (float(xyz[a * 3 + 1]) + float(xyz[b * 3 + 1]) + float(xyz[c * 3 + 1])) / 3.0,
                        (float(xyz[a * 3 + 2]) + float(xyz[b * 3 + 2]) + float(xyz[c * 3 + 2])) / 3.0,
                    )

                    verts = (
                        (float(xyz[a * 3 + 0]), float(xyz[a * 3 + 1]), float(xyz[a * 3 + 2])),
                        (float(xyz[b * 3 + 0]), float(xyz[b * 3 + 1]), float(xyz[b * 3 + 2])),
                        (float(xyz[c * 3 + 0]), float(xyz[c * 3 + 1]), float(xyz[c * 3 + 2])),
                    )
                    axial = [abs(p[thin_axis] - center[thin_axis]) for p in verts]
                    rr_vals = [
                        math.hypot(
                            p[other_axes[0]] - center[other_axes[0]],
                            p[other_axes[1]] - center[other_axes[1]],
                        )
                        for p in verts
                    ]

                    axial_lim = max(thickness * 0.92, radial * 0.12)
                    if max(axial) > axial_lim:
                        return None

                    rr_min = min(rr_vals)
                    rr_max = max(rr_vals)
                    rr_avg = (rr_vals[0] + rr_vals[1] + rr_vals[2]) / 3.0
                    if rr_min < max(radial_inner * 0.45, radial * 0.19):
                        return None
                    if rr_max > radial * 1.08:
                        return None
                    if (rr_max - rr_min) > radial * 0.34:
                        return None

                    dc = abs(tri_center[thin_axis] - center[thin_axis])
                    return abs(rr_avg - radial * 0.74) + dc * 0.24

                def tri_wheel_loose_score(
                    a: int,
                    b: int,
                    c: int,
                    m: Dict[str, Any],
                ) -> float | None:
                    # Loose classifier for wheel hubs/spokes that can sit close
                    # to wheel center and fail ring-focused thresholds.
                    center = m["center"]
                    thin_axis = int(m["thin_axis"])
                    other_axes = m["other_axes"]
                    radial = float(m["radial"])
                    thickness = float(m["thickness"])

                    verts = (
                        (float(xyz[a * 3 + 0]), float(xyz[a * 3 + 1]), float(xyz[a * 3 + 2])),
                        (float(xyz[b * 3 + 0]), float(xyz[b * 3 + 1]), float(xyz[b * 3 + 2])),
                        (float(xyz[c * 3 + 0]), float(xyz[c * 3 + 1]), float(xyz[c * 3 + 2])),
                    )
                    tri_center = (
                        (verts[0][0] + verts[1][0] + verts[2][0]) / 3.0,
                        (verts[0][1] + verts[1][1] + verts[2][1]) / 3.0,
                        (verts[0][2] + verts[1][2] + verts[2][2]) / 3.0,
                    )
                    axial = [abs(p[thin_axis] - center[thin_axis]) for p in verts]
                    rr_vals = [
                        math.hypot(
                            p[other_axes[0]] - center[other_axes[0]],
                            p[other_axes[1]] - center[other_axes[1]],
                        )
                        for p in verts
                    ]

                    axial_lim = max(thickness * max(1.05, pair_axial_scale), radial * 0.20)
                    if max(axial) > axial_lim:
                        return None
                    rr_min = min(rr_vals)
                    rr_max = max(rr_vals)
                    if rr_max > radial * (pair_ring_max * 1.03):
                        return None
                    if rr_min < radial * max(0.0, pair_ring_min * 0.70):
                        return None

                    rr_avg = (rr_vals[0] + rr_vals[1] + rr_vals[2]) / 3.0
                    dc = abs(tri_center[thin_axis] - center[thin_axis])
                    return abs(rr_avg - radial * 0.42) + dc * 0.16

                def tri_max_edge_local(a: int, b: int, c: int) -> float:
                    pa = (
                        float(xyz[a * 3 + 0]),
                        float(xyz[a * 3 + 1]),
                        float(xyz[a * 3 + 2]),
                    )
                    pb = (
                        float(xyz[b * 3 + 0]),
                        float(xyz[b * 3 + 1]),
                        float(xyz[b * 3 + 2]),
                    )
                    pc = (
                        float(xyz[c * 3 + 0]),
                        float(xyz[c * 3 + 1]),
                        float(xyz[c * 3 + 2]),
                    )
                    return max(math.dist(pa, pb), math.dist(pb, pc), math.dist(pc, pa))

                # Remove chassis-like outliers already assigned to wheel groups.
                # This catches triangles that were grouped into Roue_* by dominant bone,
                # but geometrically belong to hull side plates.
                grouped.setdefault("Chassis", [])
                for wn, m in wheel_metrics.items():
                    wheel_bones = wheel_bones_by_name.get(wn, set())
                    keep_wheel: List[Tuple[int, int, int]] = []
                    spill_to_chassis: List[Tuple[int, int, int]] = []
                    for a, b, c in grouped.get(wn, []):
                        dom_support = tri_wheel_dominant_support(a, b, c, wheel_bones)
                        support = tri_wheel_support(a, b, c, wheel_bones)
                        if support < 1:
                            spill_to_chassis.append((a, b, c))
                            continue

                        score = tri_wheel_score(a, b, c, m)
                        if score is None and dom_support >= 2:
                            score = tri_wheel_loose_score(a, b, c, m)
                        if score is None:
                            spill_to_chassis.append((a, b, c))
                        else:
                            keep_wheel.append((a, b, c))
                    grouped[wn] = keep_wheel
                    if spill_to_chassis:
                        grouped["Chassis"].extend(spill_to_chassis)

                new_chassis: List[Tuple[int, int, int]] = []
                for a, b, c in grouped.get("Chassis", []):
                    best_name = ""
                    best_score = float("inf")
                    for wn, m in wheel_metrics.items():
                        wheel_bones = wheel_bones_by_name.get(wn, set())
                        support = tri_wheel_support(a, b, c, wheel_bones)
                        if support < 2:
                            continue
                        dom_support = tri_wheel_dominant_support(a, b, c, wheel_bones)
                        score = tri_wheel_score(a, b, c, m)
                        if score is None and dom_support >= 2:
                            # Recover hub/spoke triangles that belong to the wheel.
                            score = tri_wheel_loose_score(a, b, c, m)
                        if score is None:
                            continue
                        score -= float(dom_support) * 0.015
                        if score < best_score:
                            best_score = score
                            best_name = wn

                    if best_name:
                        grouped[best_name].append((a, b, c))
                    else:
                        new_chassis.append((a, b, c))

                grouped["Chassis"] = new_chassis

                # Second-pass cleanup for abnormally thick wheel groups (typical glued hull strips).
                final_thickness_by_wheel: Dict[str, float] = {}
                for wn, m in wheel_metrics.items():
                    tri_list = grouped.get(wn, [])
                    if not tri_list:
                        continue
                    thin_axis = int(m["thin_axis"])
                    final_verts: set[int] = set()
                    for a, b, c in tri_list:
                        final_verts.add(a)
                        final_verts.add(b)
                        final_verts.add(c)
                    if not final_verts:
                        continue
                    axis_vals = [float(xyz[v * 3 + thin_axis]) for v in final_verts if 0 <= v < vertex_count]
                    if not axis_vals:
                        continue
                    final_thickness_by_wheel[wn] = max(axis_vals) - min(axis_vals)

                final_thickness_med = median_val(final_thickness_by_wheel.values())
                if final_thickness_med > 1e-6:
                    grouped.setdefault("Chassis", [])
                    for wn, m in wheel_metrics.items():
                        tri_list = grouped.get(wn, [])
                        if not tri_list:
                            continue
                        final_thickness = float(final_thickness_by_wheel.get(wn, 0.0))
                        if final_thickness <= final_thickness_med * 2.0:
                            continue

                        thin_axis = int(m["thin_axis"])
                        other_axes = m["other_axes"]
                        center = m["center"]
                        radial = float(m["radial"])
                        wheel_bones = wheel_bones_by_name.get(wn, set())
                        strict_axial = max(final_thickness_med * 1.15, radial * 0.08)

                        keep_wheel: List[Tuple[int, int, int]] = []
                        spill_to_chassis: List[Tuple[int, int, int]] = []
                        for a, b, c in tri_list:
                            verts = (
                                (float(xyz[a * 3 + 0]), float(xyz[a * 3 + 1]), float(xyz[a * 3 + 2])),
                                (float(xyz[b * 3 + 0]), float(xyz[b * 3 + 1]), float(xyz[b * 3 + 2])),
                                (float(xyz[c * 3 + 0]), float(xyz[c * 3 + 1]), float(xyz[c * 3 + 2])),
                            )
                            axial = [abs(p[thin_axis] - center[thin_axis]) for p in verts]
                            rr_vals = [
                                math.hypot(
                                    p[other_axes[0]] - center[other_axes[0]],
                                    p[other_axes[1]] - center[other_axes[1]],
                                )
                                for p in verts
                            ]
                            rr_min = min(rr_vals)
                            rr_max = max(rr_vals)
                            dom_support = tri_wheel_dominant_support(a, b, c, wheel_bones)

                            if max(axial) > strict_axial or rr_max > radial * 1.08:
                                spill_to_chassis.append((a, b, c))
                            elif rr_min < radial * 0.18 and dom_support < 2:
                                spill_to_chassis.append((a, b, c))
                            else:
                                keep_wheel.append((a, b, c))

                        grouped[wn] = keep_wheel
                        if spill_to_chassis:
                            grouped["Chassis"].extend(spill_to_chassis)

    # Final wheel normalization and recovery pass.
    if grouped:
        wheel_name_rx = re.compile(r"^Roue_([DG])(\d+)$", re.IGNORECASE)

        def parse_wheel_name(name: str) -> Tuple[str, int] | None:
            m = wheel_name_rx.match(str(name).strip())
            if not m:
                return None
            return m.group(1).upper(), int(m.group(2))

        def opposite_side(side: str) -> str:
            return "G" if str(side).upper() == "D" else "D"

        def tri_center(a: int, b: int, c: int) -> Tuple[float, float, float]:
            return (
                (float(xyz[a * 3 + 0]) + float(xyz[b * 3 + 0]) + float(xyz[c * 3 + 0])) / 3.0,
                (float(xyz[a * 3 + 1]) + float(xyz[b * 3 + 1]) + float(xyz[c * 3 + 1])) / 3.0,
                (float(xyz[a * 3 + 2]) + float(xyz[b * 3 + 2]) + float(xyz[c * 3 + 2])) / 3.0,
            )

        def tri_max_edge(a: int, b: int, c: int) -> float:
            pa = (
                float(xyz[a * 3 + 0]),
                float(xyz[a * 3 + 1]),
                float(xyz[a * 3 + 2]),
            )
            pb = (
                float(xyz[b * 3 + 0]),
                float(xyz[b * 3 + 1]),
                float(xyz[b * 3 + 2]),
            )
            pc = (
                float(xyz[c * 3 + 0]),
                float(xyz[c * 3 + 1]),
                float(xyz[c * 3 + 2]),
            )
            e0 = math.dist(pa, pb)
            e1 = math.dist(pb, pc)
            e2 = math.dist(pc, pa)
            return max(e0, e1, e2)

        def percentile(values: Sequence[float], q: float) -> float:
            seq = sorted(float(v) for v in values)
            if not seq:
                return 0.0
            if q <= 0.0:
                return seq[0]
            if q >= 1.0:
                return seq[-1]
            pos = (len(seq) - 1) * q
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return seq[lo]
            t = pos - lo
            return seq[lo] * (1.0 - t) + seq[hi] * t

        def wheel_group_metrics(tris: Sequence[Tuple[int, int, int]]) -> Dict[str, Any]:
            verts: set[int] = set()
            for a, b, c in tris:
                verts.add(int(a))
                verts.add(int(b))
                verts.add(int(c))
            if not verts:
                return {
                    "center": (0.0, 0.0, 0.0),
                    "thin_axis": 2,
                    "other_axes": [0, 1],
                    "radial": 0.0,
                    "thickness": 0.0,
                    "x_span": 0.0,
                }

            xs = [float(xyz[v * 3 + 0]) for v in verts if 0 <= v < vertex_count]
            ys = [float(xyz[v * 3 + 1]) for v in verts if 0 <= v < vertex_count]
            zs = [float(xyz[v * 3 + 2]) for v in verts if 0 <= v < vertex_count]
            if not xs or not ys or not zs:
                return {
                    "center": (0.0, 0.0, 0.0),
                    "thin_axis": 2,
                    "other_axes": [0, 1],
                    "radial": 0.0,
                    "thickness": 0.0,
                    "x_span": 0.0,
                }

            center = (
                percentile(xs, 0.5),
                percentile(ys, 0.5),
                percentile(zs, 0.5),
            )
            dims = (
                max(xs) - min(xs),
                max(ys) - min(ys),
                max(zs) - min(zs),
            )
            thin_axis = min(range(3), key=lambda i: dims[i])
            other_axes = [i for i in (0, 1, 2) if i != thin_axis]

            rr_vals: List[float] = []
            for v in verts:
                if v < 0 or v >= vertex_count:
                    continue
                px = float(xyz[v * 3 + 0])
                py = float(xyz[v * 3 + 1])
                pz = float(xyz[v * 3 + 2])
                p = (px, py, pz)
                rr_vals.append(
                    math.hypot(
                        p[other_axes[0]] - center[other_axes[0]],
                        p[other_axes[1]] - center[other_axes[1]],
                    )
                )
            radial = percentile(rr_vals, 0.85) if rr_vals else 0.0
            thickness = max(float(dims[thin_axis]), radial * 0.10, 1e-6)
            return {
                "center": center,
                "thin_axis": thin_axis,
                "other_axes": other_axes,
                "radial": radial,
                "thickness": thickness,
                "x_span": float(max(xs) - min(xs)),
            }

        def split_large_wheel_group(tris: Sequence[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
            if len(tris) < 96:
                return [list(tris)]
            centers: List[Tuple[float, int]] = []
            for i, (a, b, c) in enumerate(tris):
                tc = tri_center(a, b, c)
                centers.append((tc[0], i))
            xs = [x for x, _ in centers]
            if not xs:
                return [list(tris)]
            if (max(xs) - min(xs)) <= 1.90:
                return [list(tris)]

            c1 = min(xs)
            c2 = max(xs)
            assign: List[int] = [0] * len(centers)
            for _ in range(8):
                g1: List[float] = []
                g2: List[float] = []
                for i, (xv, _) in enumerate(centers):
                    if abs(xv - c1) <= abs(xv - c2):
                        assign[i] = 0
                        g1.append(xv)
                    else:
                        assign[i] = 1
                        g2.append(xv)
                if not g1 or not g2:
                    return [list(tris)]
                c1 = sum(g1) / float(len(g1))
                c2 = sum(g2) / float(len(g2))

            out1: List[Tuple[int, int, int]] = []
            out2: List[Tuple[int, int, int]] = []
            for i, (_, tri_idx) in enumerate(centers):
                if assign[i] == 0:
                    out1.append(tuple(tris[tri_idx]))
                else:
                    out2.append(tuple(tris[tri_idx]))
            if len(out1) < 24 or len(out2) < 24:
                return [list(tris)]

            m1 = wheel_group_metrics(out1)
            m2 = wheel_group_metrics(out2)
            if abs(float(m1["center"][0]) - float(m2["center"][0])) < 0.55:
                return [list(tris)]
            return [out1, out2]

        # Split accidentally merged wheels (e.g. Roue_G9 containing two wheel centers).
        split_counter = 1
        for wn in [n for n in list(grouped.keys()) if str(n).startswith("Roue_")]:
            tris = grouped.get(wn, [])
            if not tris:
                continue
            chunks = split_large_wheel_group(tris)
            if len(chunks) <= 1:
                continue

            keep_idx = 0
            parsed = parse_wheel_name(wn)
            if parsed is not None:
                side, num = parsed
                mirror_name = f"Roue_{opposite_side(side)}{num}"
                mirror_tris = grouped.get(mirror_name, [])
                if mirror_tris:
                    mcx = float(wheel_group_metrics(mirror_tris)["center"][0])
                    dists = [abs(float(wheel_group_metrics(ch)["center"][0]) - mcx) for ch in chunks]
                    keep_idx = min(range(len(chunks)), key=lambda i: dists[i])
                else:
                    keep_idx = max(range(len(chunks)), key=lambda i: len(chunks[i]))
            else:
                keep_idx = max(range(len(chunks)), key=lambda i: len(chunks[i]))

            grouped[wn] = list(chunks[keep_idx])
            keep_metrics = wheel_group_metrics(grouped[wn]) if grouped[wn] else {"radial": 0.0}
            for i, ch in enumerate(chunks):
                if i == keep_idx:
                    continue
                if parsed is not None:
                    chunk_metrics = wheel_group_metrics(ch)
                    keep_count = len(grouped[wn])
                    chunk_count = len(ch)
                    keep_radial = float(keep_metrics.get("radial", 0.0))
                    chunk_radial = float(chunk_metrics.get("radial", 0.0))
                    small_chunk = chunk_count < max(24, int(keep_count * 0.55))
                    radial_ratio = (chunk_radial / keep_radial) if keep_radial > 1e-6 else 0.0
                    if small_chunk or radial_ratio < 0.72:
                        grouped.setdefault("Chassis", []).extend(list(ch))
                        continue
                new_name = f"Roue_AutoSplit_{split_counter}"
                split_counter += 1
                grouped[new_name] = list(ch)

        # Estimate geometric side orientation from named wheel groups.
        side_samples: Dict[str, List[Tuple[Tuple[float, float, float], int]]] = {"D": [], "G": []}
        wheel_metrics_now: Dict[str, Dict[str, Any]] = {}
        for name, tris in grouped.items():
            if not tris:
                continue
            if not str(name).startswith("Roue_"):
                continue
            m = wheel_group_metrics(tris)
            wheel_metrics_now[str(name)] = m
            parsed = parse_wheel_name(str(name))
            if parsed is None:
                continue
            side, _ = parsed
            side_samples[side].append((tuple(m["center"]), len(tris)))

        expected_sign = {"D": 1.0, "G": -1.0}
        side_axis = 2
        side_mid = 0.0
        d_samples = side_samples.get("D", [])
        g_samples = side_samples.get("G", [])
        if d_samples or g_samples:
            mean_d = [0.0, 0.0, 0.0]
            mean_g = [0.0, 0.0, 0.0]
            if d_samples:
                d_sum_w = max(1, sum(w for _, w in d_samples))
                for i in range(3):
                    mean_d[i] = sum(c[i] * w for c, w in d_samples) / float(d_sum_w)
            if g_samples:
                g_sum_w = max(1, sum(w for _, w in g_samples))
                for i in range(3):
                    mean_g[i] = sum(c[i] * w for c, w in g_samples) / float(g_sum_w)

            if d_samples and g_samples:
                diffs = [abs(mean_d[i] - mean_g[i]) for i in range(3)]
                side_axis = max(range(3), key=lambda i: diffs[i])
                side_mid = 0.5 * (mean_d[side_axis] + mean_g[side_axis])
                d_dir = mean_d[side_axis] - mean_g[side_axis]
                expected_sign["D"] = 1.0 if d_dir >= 0.0 else -1.0
                expected_sign["G"] = -expected_sign["D"]
            elif d_samples:
                side_axis = max(range(3), key=lambda i: abs(mean_d[i]))
                side_mid = 0.0
                expected_sign["D"] = 1.0 if mean_d[side_axis] >= 0.0 else -1.0
                expected_sign["G"] = -expected_sign["D"]
            elif g_samples:
                side_axis = max(range(3), key=lambda i: abs(mean_g[i]))
                side_mid = 0.0
                expected_sign["G"] = 1.0 if mean_g[side_axis] >= 0.0 else -1.0
                expected_sign["D"] = -expected_sign["G"]

        # Conservative side-fix for named wheels: rename only when destination
        # name is currently free, so we never merge two physical wheels.
        rename_pairs: List[Tuple[str, str]] = []
        for name, tris in grouped.items():
            parsed = parse_wheel_name(str(name))
            if parsed is None or not tris:
                continue
            side, num = parsed
            met = wheel_metrics_now.get(str(name)) or wheel_group_metrics(tris)
            signed = (float(met["center"][side_axis]) - float(side_mid)) * float(expected_sign[side])
            if signed >= 0.0:
                continue
            target = f"Roue_{opposite_side(side)}{num}"
            if target in grouped:
                continue
            rename_pairs.append((str(name), target))

        for old_name, new_name in rename_pairs:
            if old_name not in grouped or new_name in grouped:
                continue
            grouped[new_name] = grouped.pop(old_name)

        # Rename auto wheel groups to missing mirror slots by x/y proximity.
        wheel_metrics_now = {
            name: wheel_group_metrics(tris)
            for name, tris in grouped.items()
            if tris and str(name).startswith("Roue_")
        }
        occupied_named: set[str] = {name for name in wheel_metrics_now.keys() if parse_wheel_name(name) is not None}

        for name in list(grouped.keys()):
            tris = grouped.get(name, [])
            if not tris or not str(name).startswith("Roue_"):
                continue
            if parse_wheel_name(str(name)) is not None:
                continue

            met = wheel_metrics_now.get(str(name)) or wheel_group_metrics(tris)
            center = met["center"]
            signed_here = (float(center[side_axis]) - float(side_mid)) * float(expected_sign["D"])
            geom_side_here = "D" if signed_here >= 0.0 else "G"

            best_target = ""
            best_dist = float("inf")
            plane_axes = [i for i in (0, 1, 2) if i != int(side_axis)]
            for oname, omet in wheel_metrics_now.items():
                parsed = parse_wheel_name(oname)
                if parsed is None:
                    continue
                signed_other = (float(omet["center"][side_axis]) - float(side_mid)) * float(expected_sign["D"])
                geom_side_other = "D" if signed_other >= 0.0 else "G"
                if geom_side_other == geom_side_here:
                    continue
                oside, onum = parsed
                cand_name = f"Roue_{opposite_side(oside)}{onum}"
                if cand_name in occupied_named or cand_name in grouped:
                    continue
                oc = omet["center"]
                d0 = float(center[plane_axes[0]]) - float(oc[plane_axes[0]])
                d1 = float(center[plane_axes[1]]) - float(oc[plane_axes[1]])
                dplane = math.hypot(d0, d1)
                if dplane > 0.85:
                    continue
                if dplane < best_dist:
                    best_dist = dplane
                    best_target = cand_name

            if best_target:
                grouped[best_target] = list(tris)
                occupied_named.add(best_target)
                del grouped[name]

        # Recover missing mirrored road wheel from chassis if one side is absent.
        if pair_fix_enabled and "Chassis" in grouped:
            wheel_metrics_now = {
                name: wheel_group_metrics(tris)
                for name, tris in grouped.items()
                if tris and parse_wheel_name(name) is not None
            }
            chassis_tris = list(grouped.get("Chassis", []))
            alive = [True] * len(chassis_tris)

            for name, met in sorted(wheel_metrics_now.items(), key=lambda kv: kv[0].lower()):
                tris_src = grouped.get(name, [])
                if len(tris_src) < 120:
                    continue
                parsed = parse_wheel_name(name)
                if parsed is None:
                    continue
                side, num = parsed
                mirror_name = f"Roue_{opposite_side(side)}{num}"
                if grouped.get(mirror_name):
                    continue

                center = met["center"]
                if abs(float(center[side_axis]) - float(side_mid)) < 0.2:
                    continue
                target = [float(center[0]), float(center[1]), float(center[2])]
                target[side_axis] = 2.0 * float(side_mid) - float(center[side_axis])
                target = (target[0], target[1], target[2])
                radial = float(met["radial"])
                if radial <= 1e-6:
                    continue

                src_centers_mirrored: List[Tuple[float, float, float]] = []
                src_edge_max = 0.0
                for a, b, c in tris_src:
                    tc = tri_center(a, b, c)
                    mirrored = [float(tc[0]), float(tc[1]), float(tc[2])]
                    mirrored[side_axis] = 2.0 * float(side_mid) - float(tc[side_axis])
                    src_centers_mirrored.append((mirrored[0], mirrored[1], mirrored[2]))
                    src_edge_max = max(src_edge_max, tri_max_edge(a, b, c))
                if not src_centers_mirrored or src_edge_max <= 1e-6:
                    continue

                dist_cap = max(radial * pair_dist_scale, 0.30)
                edge_cap = src_edge_max * pair_edge_scale

                cand_idx: List[Tuple[float, int]] = []
                for i, tri in enumerate(chassis_tris):
                    if not alive[i]:
                        continue
                    a, b, c = tri
                    tc = tri_center(a, b, c)
                    if tri_max_edge(a, b, c) > edge_cap:
                        continue

                    best_dist = float("inf")
                    for mc in src_centers_mirrored:
                        d = math.dist(tc, mc)
                        if d < best_dist:
                            best_dist = d
                    if best_dist > dist_cap:
                        continue
                    cand_idx.append((best_dist, i))

                cand_idx.sort(key=lambda x: x[0])

                # Allow partial mirror recovery when only part of the wheel was
                # glued into chassis, instead of requiring near-complete match.
                min_need = max(48, int(len(tris_src) * 0.35))
                if len(cand_idx) < min_need:
                    continue

                moved: List[Tuple[int, int, int]] = []
                target_count = min(
                    len(cand_idx),
                    max(min_need, int(len(tris_src) * 0.92)),
                )
                if target_count <= 0:
                    continue
                for _, i in cand_idx:
                    if len(moved) >= target_count:
                        break
                    if not alive[i]:
                        continue
                    alive[i] = False
                    moved.append(chassis_tris[i])
                if moved:
                    grouped[mirror_name] = moved

            grouped["Chassis"] = [tri for i, tri in enumerate(chassis_tris) if alive[i]]

        # Pairwise symmetry rebalance for road wheels.
        # Fixes cases where one wheel keeps a hull wedge while missing wheel faces.
        if pair_fix_enabled and "Chassis" in grouped:
            pair_side_axis = int(side_axis)
            pair_side_mid = float(side_mid)
            pair_samples: Dict[str, List[Tuple[Tuple[float, float, float], int]]] = {"D": [], "G": []}
            for name, tris in grouped.items():
                parsed = parse_wheel_name(name)
                if parsed is None or not tris:
                    continue
                pside, _ = parsed
                met_now = wheel_group_metrics(tris)
                pair_samples[pside].append((tuple(met_now["center"]), len(tris)))
            if pair_samples["D"] and pair_samples["G"]:
                mean_d = [0.0, 0.0, 0.0]
                mean_g = [0.0, 0.0, 0.0]
                d_sum = max(1, sum(w for _, w in pair_samples["D"]))
                g_sum = max(1, sum(w for _, w in pair_samples["G"]))
                for ai in range(3):
                    mean_d[ai] = sum(c[ai] * w for c, w in pair_samples["D"]) / float(d_sum)
                    mean_g[ai] = sum(c[ai] * w for c, w in pair_samples["G"]) / float(g_sum)
                diffs = [abs(mean_d[ai] - mean_g[ai]) for ai in range(3)]
                pair_side_axis = max(range(3), key=lambda ai: diffs[ai])
                pair_side_mid = 0.5 * (mean_d[pair_side_axis] + mean_g[pair_side_axis])

            by_num: Dict[int, Dict[str, str]] = {}
            for name, tris in grouped.items():
                if not tris:
                    continue
                parsed = parse_wheel_name(name)
                if parsed is None:
                    continue
                side, num = parsed
                by_num.setdefault(int(num), {})[side] = str(name)

            chassis_tris = list(grouped.get("Chassis", []))
            chassis_alive = [True] * len(chassis_tris)
            for num, sides in sorted(by_num.items()):
                n_d = sides.get("D")
                n_g = sides.get("G")
                if not n_d or not n_g:
                    continue
                tris_d = grouped.get(n_d, [])
                tris_g = grouped.get(n_g, [])
                if not tris_d or not tris_g:
                    continue

                len_d = len(tris_d)
                len_g = len(tris_g)
                if max(len_d, len_g) < 120:
                    continue
                if abs(len_d - len_g) < 8:
                    continue

                if len_d >= len_g:
                    src_name, dst_name = n_d, n_g
                else:
                    src_name, dst_name = n_g, n_d

                src_tris = list(grouped.get(src_name, []))
                dst_tris = list(grouped.get(dst_name, []))
                if not src_tris or not dst_tris:
                    continue

                src_met = wheel_group_metrics(src_tris)
                radial = float(src_met.get("radial", 0.0))
                if radial <= 1e-6:
                    continue
                dst_met = wheel_group_metrics(dst_tris)
                dst_radial = float(dst_met.get("radial", 0.0))
                if dst_radial <= 1e-6:
                    continue
                dst_center = tuple(dst_met.get("center", (0.0, 0.0, 0.0)))
                dst_thin_axis = int(dst_met.get("thin_axis", 2))
                dst_other_axes = dst_met.get("other_axes", [0, 1])
                if not isinstance(dst_other_axes, (list, tuple)) or len(dst_other_axes) != 2:
                    dst_other_axes = [0, 1]
                dst_thickness = float(dst_met.get("thickness", max(dst_radial * 0.10, 1e-6)))

                src_edge_max = 0.0
                src_mirrored_centers: List[Tuple[float, float, float]] = []
                src_side, _ = parse_wheel_name(src_name) or ("D", num)
                dst_side = opposite_side(src_side)
                for a, b, c in src_tris:
                    tc = tri_center(a, b, c)
                    mirrored = [float(tc[0]), float(tc[1]), float(tc[2])]
                    mirrored[pair_side_axis] = 2.0 * float(pair_side_mid) - float(tc[pair_side_axis])
                    src_mirrored_centers.append((mirrored[0], mirrored[1], mirrored[2]))
                    src_edge_max = max(src_edge_max, tri_max_edge(a, b, c))
                if src_edge_max <= 1e-6:
                    continue
                mx = [p[0] for p in src_mirrored_centers]
                my = [p[1] for p in src_mirrored_centers]
                mz = [p[2] for p in src_mirrored_centers]
                mirr_min = (min(mx), min(my), min(mz))
                mirr_max = (max(mx), max(my), max(mz))

                dist_cap = max(radial * pair_dist_scale, 0.30)
                edge_cut = src_edge_max * pair_edge_scale
                bbox_margin = max(radial * 0.28, 0.12)

                def nearest_mirror_dist(tc: Tuple[float, float, float]) -> float:
                    best = float("inf")
                    for mc in src_mirrored_centers:
                        d = math.dist(tc, mc)
                        if d < best:
                            best = d
                    return best

                def inside_mirror_bbox(tc: Tuple[float, float, float]) -> bool:
                    return (
                        (mirr_min[0] - bbox_margin) <= tc[0] <= (mirr_max[0] + bbox_margin)
                        and (mirr_min[1] - bbox_margin) <= tc[1] <= (mirr_max[1] + bbox_margin)
                        and (mirr_min[2] - bbox_margin) <= tc[2] <= (mirr_max[2] + bbox_margin)
                    )

                wheel_bones_dst = wheel_bones_by_name.get(dst_name, set()) if "wheel_bones_by_name" in locals() else set()

                # Strip obvious hull wedges that were mistakenly assigned to wheel.
                dst_keep: List[Tuple[int, int, int]] = []
                dst_spill: List[Tuple[int, int, int]] = []
                for a, b, c in dst_tris:
                    if wheel_bones_dst and tri_wheel_support(a, b, c, wheel_bones_dst) < 1:
                        dst_spill.append((a, b, c))
                        continue
                    if wheel_bones_dst and tri_wheel_dominant_support(a, b, c, wheel_bones_dst) < 1:
                        dst_spill.append((a, b, c))
                        continue
                    tc = tri_center(a, b, c)
                    if not inside_mirror_bbox(tc):
                        dst_spill.append((a, b, c))
                        continue
                    d = nearest_mirror_dist(tc)
                    if tri_max_edge(a, b, c) > edge_cut * 1.02:
                        dst_spill.append((a, b, c))
                        continue
                    if d > dist_cap * 1.35:
                        dst_spill.append((a, b, c))
                        continue
                    dst_keep.append((a, b, c))
                if dst_keep:
                    dst_tris = dst_keep
                if dst_spill:
                    chassis_tris.extend(dst_spill)
                    chassis_alive.extend([True] * len(dst_spill))

                pool_chassis: List[Tuple[float, Tuple[int, int, int], int]] = []
                for i, tri in enumerate(chassis_tris):
                    if not chassis_alive[i]:
                        continue
                    a, b, c = tri
                    if wheel_bones_dst and tri_wheel_support(a, b, c, wheel_bones_dst) < 1:
                        continue
                    if wheel_bones_dst and tri_wheel_dominant_support(a, b, c, wheel_bones_dst) < 1:
                        continue
                    tc = tri_center(a, b, c)
                    if not inside_mirror_bbox(tc):
                        continue
                    d = nearest_mirror_dist(tc)
                    if d > dist_cap:
                        continue
                    if tri_max_edge(a, b, c) > edge_cut:
                        continue
                    tri_verts = (
                        (float(xyz[a * 3 + 0]), float(xyz[a * 3 + 1]), float(xyz[a * 3 + 2])),
                        (float(xyz[b * 3 + 0]), float(xyz[b * 3 + 1]), float(xyz[b * 3 + 2])),
                        (float(xyz[c * 3 + 0]), float(xyz[c * 3 + 1]), float(xyz[c * 3 + 2])),
                    )
                    axial = [abs(p[dst_thin_axis] - float(dst_center[dst_thin_axis])) for p in tri_verts]
                    rr_vals = [
                        math.hypot(
                            p[int(dst_other_axes[0])] - float(dst_center[int(dst_other_axes[0])]),
                            p[int(dst_other_axes[1])] - float(dst_center[int(dst_other_axes[1])]),
                        )
                        for p in tri_verts
                    ]
                    axial_lim = max(dst_thickness * pair_axial_scale, dst_radial * 0.12)
                    if max(axial) > axial_lim:
                        continue
                    rr_min = min(rr_vals)
                    rr_max = max(rr_vals)
                    if rr_max > dst_radial * pair_ring_max:
                        continue
                    if rr_min < dst_radial * pair_ring_min:
                        continue
                    pool_chassis.append((d, (a, b, c), i))

                pool_chassis.sort(key=lambda x: x[0])
                target_count = len(src_tris)
                # Keep existing dst triangles intact (prevents wheel holes),
                # and only fill missing faces from chassis near mirrored source.
                desired_count = max(len(dst_tris), int(target_count * pair_target_ratio))
                desired_count = min(desired_count, len(dst_tris) + len(pool_chassis))
                deficit = desired_count - len(dst_tris)
                if deficit <= 0:
                    continue

                min_need = max(8, int(deficit * pair_min_pool_ratio))
                if len(pool_chassis) < min_need:
                    continue

                selected = list(dst_tris)
                for _, tri, src_idx in pool_chassis:
                    if len(selected) >= desired_count:
                        break
                    selected.append(tri)
                    if src_idx >= 0 and src_idx < len(chassis_alive):
                        chassis_alive[src_idx] = False
                grouped[dst_name] = selected

            grouped["Chassis"] = [tri for i, tri in enumerate(chassis_tris) if chassis_alive[i]]

    if not grouped:
        return [(part_fallback_name, [])]
    return sorted(grouped.items(), key=lambda kv: kv[0].lower())


def score_bone_name_map(
    model: Dict[str, Any],
    bone_name_by_index: Dict[int, str],
    material_roles_by_id: Dict[int, str],
) -> int:
    if not bone_name_by_index:
        return -1

    score = 0
    for part in model["parts"]:
        mid = int(part.get("material", -1))
        role = str(material_roles_by_id.get(mid, ""))
        groups = split_faces_by_bone(
            part,
            "part",
            bone_name_by_index,
            material_role=role,
            material_name="",
        )
        names = {g for g, tris in groups if tris}
        if "Chassis" in names:
            score += 1
        if "Tourelle_01" in names:
            score += 2
        if "Canon_01" in names:
            score += 2
        if "Axe_Canon_01" in names:
            score += 2
        if any(n.startswith("Roue_") for n in names):
            score += 1
        if role == "track_left" and "Chenille_Gauche_01" in names:
            score += 1
        if role == "track_right" and "Chenille_Droite_01" in names:
            score += 1
    return score


def build_named_texture_aliases(
    asset: str,
    model_dir: Path,
    resolved: Sequence[Dict[str, Any]],
    chosen_maps: Dict[str, Path],
) -> Tuple[Dict[str, Path], List[Dict[str, str]]]:
    model_base = Path(asset).stem

    # Clean previous generated aliases for this model to avoid stale TRK/PART files
    # influencing later imports.
    protected_sources: set[str] = set()
    for src in chosen_maps.values():
        try:
            protected_sources.add(str(Path(src).resolve()).lower())
        except Exception:
            protected_sources.add(str(src).lower())
    for old in model_dir.glob(f"{model_base}_*.png"):
        if not old.is_file():
            continue
        try:
            old_key = str(old.resolve()).lower()
        except Exception:
            old_key = str(old).lower()
        if old_key in protected_sources:
            continue
        try:
            old.unlink()
        except OSError:
            pass

    suffix_by_channel = {
        "diffuse": "D",
        "normal": "NM",
        "roughness": "R",
        "metallic": "M",
        "occlusion": "AO",
        "alpha": "A",
        "orm": "ORM",
    }

    # Flatten all available channel sources.
    channel_items: List[Tuple[str, Path, str]] = []
    for item in resolved:
        role = str(item.get("role", ""))
        part = detect_part_label(str(item.get("atlas_ref", "")))
        extras = item.get("extras", {})
        if not isinstance(extras, dict):
            extras = {}
        out_png = item.get("out_png")
        out_path = Path(out_png) if out_png else None

        if role == "combined_da":
            if "diffuse" in extras:
                channel_items.append(("diffuse", Path(extras["diffuse"]), part))
            elif out_path:
                channel_items.append(("diffuse", out_path, part))
            if "alpha" in extras:
                channel_items.append(("alpha", Path(extras["alpha"]), part))
        elif role == "diffuse":
            if out_path:
                channel_items.append(("diffuse", out_path, part))
            if "alpha" in extras:
                channel_items.append(("alpha", Path(extras["alpha"]), part))
            # Explicit packed track block (e.g. *_trk.png) is still diffuse channel.
            trk = extras.get("trk")
            if trk is not None:
                channel_items.append(("diffuse", Path(trk), "TRK"))
            # Other packed diffuse parts (MG/PARTn/...) should remain diffuse channel too.
            for extra_key, extra_value in extras.items():
                if extra_key == "trk":
                    continue
                try:
                    extra_path = Path(extra_value)
                except Exception:
                    continue
                token = f"{extra_key} {extra_path.stem}"
                ch_guess = channel_from_token(token)
                if ch_guess not in {None, "diffuse"}:
                    continue
                part_label = part_label_from_token(token) or part
                channel_items.append(("diffuse", extra_path, part_label))
        elif role == "normal":
            if "normal_reconstructed" in extras:
                channel_items.append(("normal", Path(extras["normal_reconstructed"]), part))
            elif out_path:
                channel_items.append(("normal", out_path, part))
        elif role == "orm":
            if out_path:
                channel_items.append(("orm", out_path, part))
            if "roughness" in extras:
                channel_items.append(("roughness", Path(extras["roughness"]), part))
            if "metallic" in extras:
                channel_items.append(("metallic", Path(extras["metallic"]), part))
            if "occlusion" in extras:
                channel_items.append(("occlusion", Path(extras["occlusion"]), part))

        for extra_key, extra_value in extras.items():
            try:
                extra_path = Path(extra_value)
            except Exception:
                continue
            token = f"{extra_key} {extra_path.stem}"
            ch = channel_from_token(token)
            if ch is None:
                continue
            part_label = part_label_from_token(token) or part
            channel_items.append((ch, extra_path, part_label))

    # Ensure channels from chosen maps are present even if role parser missed some.
    for ch, src in chosen_maps.items():
        if ch not in suffix_by_channel:
            continue
        if not any(it[0] == ch and it[1] == src for it in channel_items):
            channel_items.append((ch, src, ""))

    used_names: set[str] = set()
    named: List[Dict[str, str]] = []
    primary: Dict[str, Path] = {}
    diffuse_secondary: List[Tuple[Path, str]] = []

    for channel, src, part in channel_items:
        if channel not in suffix_by_channel:
            continue
        suffix = suffix_by_channel[channel]
        if channel in primary:
            if channel == "diffuse":
                diffuse_secondary.append((src, part))
                continue
            part_label = part
            if not part_label:
                # Skip unnamed duplicate to avoid noise like <model>_D_2.
                continue
            name = make_unique_texture_name(model_base, suffix, part_label, used_names)
            dst = model_dir / f"{name}.png"
            copy_if_needed(src, dst)
            named.append(
                {
                    "channel": channel,
                    "source": str(src.resolve()),
                    "named": str(dst.resolve()),
                }
            )
            continue

        # Never let part-specific alpha become global alpha map.
        if channel == "alpha" and part:
            name = make_unique_texture_name(model_base, suffix, part, used_names)
            dst = model_dir / f"{name}.png"
            copy_if_needed(src, dst)
            named.append(
                {
                    "channel": channel,
                    "source": str(src.resolve()),
                    "named": str(dst.resolve()),
                }
            )
            continue

        name = make_unique_texture_name(model_base, suffix, "", used_names)
        dst = model_dir / f"{name}.png"
        copy_if_needed(src, dst)
        named.append(
            {
                "channel": channel,
                "source": str(src.resolve()),
                "named": str(dst.resolve()),
            }
        )
        primary[channel] = dst

    def _safe_resolve_str(path: Path) -> str:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

    def _add_named(channel: str, src: Path, part_label: str) -> None:
        suffix = suffix_by_channel[channel]
        name = make_unique_texture_name(model_base, suffix, part_label, used_names)
        dst = model_dir / f"{name}.png"
        copy_if_needed(src, dst)
        named.append(
            {
                "channel": channel,
                "source": _safe_resolve_str(src),
                "named": _safe_resolve_str(dst),
            }
        )

    # Process deferred diffuse duplicates:
    # 1) keep explicit non-track part labels,
    # 2) choose the best track diffuse candidate using size/name scoring.
    track_target_sizes: List[Tuple[int, int]] = []
    for ch, src, part in channel_items:
        if ch not in {"normal", "roughness", "metallic", "occlusion"}:
            continue
        if not (is_track_token(part) or is_track_token(src.stem)):
            continue
        sz = read_png_size(src)
        if sz is not None:
            track_target_sizes.append(sz)
    target_size = max(track_target_sizes, key=lambda t: t[0] * t[1]) if track_target_sizes else None

    trk_candidates: List[Tuple[Path, str]] = []
    for src, part_label in diffuse_secondary:
        if part_label and not is_track_token(part_label):
            _add_named("diffuse", src, part_label)
            continue
        trk_candidates.append((src, part_label))
    primary_diffuse_src = primary.get("diffuse")
    if primary_diffuse_src is not None:
        trk_candidates.append((primary_diffuse_src, "__PRIMARY__"))

    trk_has_non_diffuse = any(
        item.get("channel") in {"normal", "roughness", "metallic", "occlusion"}
        and is_track_token(Path(str(item.get("named", ""))).stem)
        for item in named
    )

    if trk_candidates and trk_has_non_diffuse:
        size_cache: Dict[Path, Tuple[int, int] | None] = {}

        def _sz(path: Path) -> Tuple[int, int] | None:
            if path not in size_cache:
                size_cache[path] = read_png_size(path)
            return size_cache[path]

        def _area(size: Tuple[int, int] | None) -> float:
            if size is None:
                return 0.0
            return float(size[0] * size[1])

        def _mismatch(size: Tuple[int, int] | None, target: Tuple[int, int] | None) -> float | None:
            if size is None or target is None:
                return None
            tw, th = target
            dw = abs(float(size[0] - tw)) / max(1.0, float(tw))
            dh = abs(float(size[1] - th)) / max(1.0, float(th))
            return max(dw, dh)

        combined_candidates: List[Path] = []
        for src, _part_label in trk_candidates:
            low = src.stem.lower()
            if "combinedda" in low or "coloralpha" in low:
                combined_candidates.append(src)
        combined_ref: Path | None = None
        if combined_candidates:
            combined_ref = max(combined_candidates, key=lambda p: _area(_sz(p)))

        def trk_diff_score(src: Path, part_label: str) -> float:
            low = src.stem.lower()
            score = 0.0
            if part_label == "__PRIMARY__":
                score += 14.0
            if part_label and is_track_token(part_label):
                score += 35.0
            if is_track_token(low):
                score += 18.0
            if "combinedda" in low or "coloralpha" in low:
                score += 26.0
                if src == combined_ref:
                    score += 10.0
            if "diffusetexturenoalpha" in low:
                score += 12.0
            if low.endswith("_diffuse") or "diffuse" in low:
                score += 8.0
            if "noalpha" in low and is_track_token(low):
                score += 6.0
            s = _sz(src)
            mismatch = _mismatch(s, target_size)
            if mismatch is not None:
                if mismatch > 0.35:
                    score -= 120.0
                elif mismatch > 0.20:
                    score -= 35.0
                else:
                    tw, th = target_size or (1, 1)
                    dw = abs(float(s[0] - tw)) / max(1.0, float(tw))
                    dh = abs(float(s[1] - th)) / max(1.0, float(th))
                    score += max(-20.0, 56.0 - 70.0 * (dw + dh))

            # Guardrail against cropped *_trk diffuse (common Leopard case):
            # if explicit track NoAlpha patch mismatches TRK NM/ORM size,
            # prefer CombinedDA diffuse candidate.
            if "diffusetexturenoalpha" in low and is_track_token(low) and combined_ref is not None:
                c_size = _sz(combined_ref)
                c_mismatch = _mismatch(c_size, target_size)
                if mismatch is not None and mismatch > 0.22:
                    score -= 60.0
                if mismatch is not None and c_mismatch is not None and (mismatch - c_mismatch) > 0.12:
                    score -= 75.0
                if _area(s) > 0.0 and _area(c_size) > 0.0 and _area(s) < 0.72 * _area(c_size):
                    score -= 50.0
            try:
                score += min(15.0, max(0.0, float(src.stat().st_size) / 65536.0))
            except Exception:
                pass
            return score

        best_src, best_part = max(trk_candidates, key=lambda t: trk_diff_score(t[0], t[1]))
        best_score = trk_diff_score(best_src, best_part)
        if best_score >= 0.0:
            _add_named("diffuse", best_src, "TRK")

    # Fallback: if track material has TRK NM/ORM maps but no TRK diffuse, provide one.
    trk_has_diffuse = any(
        item.get("channel") == "diffuse" and is_track_token(Path(str(item.get("named", ""))).stem)
        for item in named
    )
    trk_has_non_diffuse = any(
        item.get("channel") in {"normal", "roughness", "metallic", "occlusion"}
        and is_track_token(Path(str(item.get("named", ""))).stem)
        for item in named
    )
    if trk_has_non_diffuse and not trk_has_diffuse:
        src = primary.get("diffuse")
        if src is not None and src.exists():
            name = make_unique_texture_name(model_base, suffix_by_channel["diffuse"], "TRK", used_names)
            dst = model_dir / f"{name}.png"
            copy_if_needed(src, dst)
            named.append(
                {
                    "channel": "diffuse",
                    "source": str(src.resolve()),
                    "named": str(dst.resolve()),
                }
            )

    return primary, named


def remap_maps_to_named_sources(
    maps: Dict[str, Path],
    named_files: Sequence[Dict[str, str]],
) -> Dict[str, Path]:
    if not maps or not named_files:
        return dict(maps)

    def key_for(path: Path) -> str:
        try:
            return str(path.resolve()).lower()
        except Exception:
            return str(path).lower()

    src_to_named: Dict[str, Path] = {}
    for item in named_files:
        src_raw = str(item.get("source", "")).strip()
        named_raw = str(item.get("named", "")).strip()
        if not src_raw or not named_raw:
            continue
        src = Path(src_raw)
        dst = Path(named_raw)
        src_to_named[key_for(src)] = dst

    out: Dict[str, Path] = {}
    for ch, src in maps.items():
        p = Path(src) if not isinstance(src, Path) else src
        out[ch] = src_to_named.get(key_for(p), p)
    return out


def pick_material_maps_from_textures(textures: Sequence[Dict[str, Any]]) -> Dict[str, Path]:
    best: Dict[str, Tuple[float, Path]] = {}

    def consider(channel: str, src: Path | None, score: float) -> None:
        if src is None or channel not in {"diffuse", "normal", "roughness", "metallic", "occlusion", "alpha", "orm"}:
            return
        cur = best.get(channel)
        if cur is None or score > cur[0]:
            best[channel] = (float(score), src)

    for item in textures:
        role = str(item.get("role", ""))
        png_raw = item.get("out_png")
        png = Path(png_raw) if png_raw else None
        extras = item.get("extras", {})
        if isinstance(extras, dict):
            extras = {k: Path(v) if not isinstance(v, Path) else v for k, v in extras.items()}
        else:
            extras = {}

        stem_low = str(png.stem if png else "").lower()
        if role == "diffuse":
            consider("diffuse", png, 320.0)
            trk = extras.get("trk")
            if trk is not None:
                # track-specific diffuse is handled later via named aliases; keep base as primary
                consider("diffuse", Path(trk), 280.0)
        elif role == "combined_da":
            consider("diffuse", extras.get("diffuse"), 180.0)
            consider("diffuse", png, 120.0)
            consider("alpha", extras.get("alpha"), 320.0)
        elif role == "normal":
            consider("normal", extras.get("normal_reconstructed"), 330.0)
            consider("normal", png, 300.0)
        elif role == "orm":
            consider("orm", png, 305.0)
            consider("roughness", extras.get("roughness"), 310.0)
            consider("metallic", extras.get("metallic"), 310.0)
            consider("occlusion", extras.get("occlusion"), 310.0)

        # Mild generic backup from extras.
        for ek, ev in extras.items():
            token = f"{ek} {Path(ev).stem}"
            ch = channel_from_token(token)
            if ch is None:
                continue
            bonus = 20.0
            low = token.lower()
            if "combinedda" in low and ch == "diffuse":
                bonus -= 40.0
            if "diffusetexturenoalpha" in low and ch == "diffuse":
                bonus += 60.0
            consider(ch, Path(ev), 120.0 + bonus)

        # Final weak fallback from main file itself.
        if png is not None:
            ch = channel_from_token(stem_low)
            if ch is not None:
                consider(ch, png, 90.0)
            elif "combinedorm" in stem_low or "combinedrm" in stem_low or stem_low.endswith("_orm"):
                consider("orm", png, 95.0)

    out: Dict[str, Path] = {ch: src for ch, (_score, src) in best.items()}
    if "diffuse" not in out:
        fallback: Path | None = None
        for item in textures:
            png = item.get("out_png")
            if not png:
                continue
            role = str(item.get("role", "")).lower()
            stem_low = str(Path(png).stem).lower()
            if role in {"diffuse", "combined_da"} or channel_from_token(stem_low) == "diffuse":
                fallback = Path(png)
                break
        if fallback is not None:
            out["diffuse"] = fallback
    return out


def _atlas_item_is_track(item: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(item.get("atlas_target_basename", "") or ""),
            str(item.get("atlas_target_logical_rel", "") or ""),
            str(item.get("atlas_ref", "") or ""),
        ]
    ).strip()
    low = text.lower()
    if "tracks" in low or "_trk" in low or "chenille" in low:
        return True
    if detect_part_label(text) == "TRK":
        return True
    return is_track_token(text)


def _atlas_item_channel(item: Dict[str, Any]) -> str:
    channel = _canonical_channel(str(item.get("atlas_target_channel", "")))
    token = " ".join(
        [
            str(item.get("atlas_target_basename", "") or ""),
            str(item.get("atlas_target_logical_rel", "") or ""),
            str(item.get("atlas_ref", "") or ""),
            str(Path(str(item.get("out_png", "") or "")).stem),
        ]
    )
    guessed = channel_from_token(token)
    guessed_channel = _canonical_channel(guessed) if guessed is not None else "generic"
    if guessed_channel != "generic":
        if channel == "generic":
            channel = guessed_channel
        elif channel == "diffuse" and guessed_channel in {"alpha", "normal", "roughness", "metallic", "occlusion", "orm"}:
            channel = guessed_channel
    if channel == "generic":
        channel = _canonical_channel(str(item.get("role", "")))
    return channel


def pick_maps_for_material_from_atlas_resolved(
    resolved_items: Sequence[Dict[str, Any]],
    material_role: str,
    asset_stem: str,
) -> Dict[str, Path]:
    wanted = {"diffuse", "normal", "roughness", "metallic", "occlusion", "alpha", "orm"}
    role_low = str(material_role or "").strip().lower()
    want_track = role_low.startswith("track")
    stem_low = str(asset_stem or "").strip().lower()
    best_explicit: Dict[str, Tuple[float, Path]] = {}
    best_extra: Dict[str, Tuple[float, Path]] = {}

    def _prefer_occlusion_short_o(path: Path) -> Path:
        try:
            p = Path(path)
        except Exception:
            return path
        stem = p.stem
        up = stem.upper()
        if up.endswith("_AO"):
            alt = p.with_name(f"{stem[:-3]}_O{p.suffix}")
            if alt.exists() and alt.is_file():
                return alt
        return p

    def _add(bucket: Dict[str, Tuple[float, Path]], channel: str, path: Path | None, is_track: bool, score: float) -> None:
        if path is None or channel not in wanted:
            return
        if channel == "occlusion":
            path = _prefer_occlusion_short_o(path)
        if not path.exists() or not path.is_file():
            return
        current = bucket.get(channel)
        if current is None or float(score) > current[0]:
            bucket[channel] = (float(score), path)

    for idx, item in enumerate(resolved_items):
        out_raw = item.get("out_png")
        out_png = Path(out_raw) if out_raw else None
        if out_png is None:
            continue

        channel = _atlas_item_channel(item)
        if channel not in wanted:
            continue

        is_track = _atlas_item_is_track(item)
        if is_track != want_track:
            continue

        # Deterministic ordering: earlier resolved items win inside same channel.
        score = 1000.0 - float(idx)
        if stem_low:
            base = str(item.get("atlas_target_basename", "") or "").lower()
            if base.startswith(f"{stem_low}_"):
                score += 25.0
            elif base == stem_low:
                score += 20.0

        _add(best_explicit, channel, out_png, is_track, score)

        extras = item.get("extras", {})
        if isinstance(extras, dict):
            for ek, ev in extras.items():
                extra_path = Path(ev)
                token = f"{str(ek or '')} {extra_path.stem}"
                extra_channel = channel_from_token(token)
                if extra_channel is None:
                    continue
                _add(best_extra, _canonical_channel(extra_channel), extra_path, is_track, score - 5.0)

    merged: Dict[str, Path] = {ch: src for ch, (_score, src) in best_explicit.items()}
    for ch, (_score, src) in best_extra.items():
        merged.setdefault(ch, src)
    return merged


def rel_path_for_obj(base_obj: Path, target: Path) -> str:
    try:
        base_parent = base_obj.parent.resolve()
    except Exception:
        base_parent = base_obj.parent
    try:
        target_norm = target.resolve()
    except Exception:
        target_norm = target
    try:
        rel = target_norm.relative_to(base_parent)
    except ValueError:
        rel = target_norm
    return str(rel).replace("\\", "/")


class SpkMeshExtractor:
    def __init__(self, spk_path: Path):
        if not spk_path.exists():
            raise FileNotFoundError(f"SPK not found: {spk_path}")
        self.path = spk_path
        self._fh = spk_path.open("rb")
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

        self.header: Dict[str, Any] = {}
        self.fat: Dict[str, Dict[str, Any]] = {}
        self.vertex_formats: List[str] = []
        self.meshes: List[Dict[str, int]] = []
        self.draw_calls: List[Dict[str, int]] = []
        self.draw_call_stride: int = 8
        self.index_table: List[Dict[str, int]] = []
        self.vertex_table: List[Dict[str, int]] = []
        self.material_texture_refs: List[str] = []
        self.material_refs_by_dir: Dict[str, List[str]] = {}
        self.material_texture_slots_by_id: Dict[int, Dict[str, str]] = {}
        self.material_texture_names_by_id: Dict[int, Dict[str, str]] = {}
        self.material_texture_refs_by_material: Dict[int, List[str]] = {}
        self.material_name_by_id: Dict[int, str] = {}
        self._index_cache: Dict[int, List[int]] = {}
        self._vertex_cache: Dict[int, Dict[str, List[float]]] = {}
        self._node_blob_cache: Dict[int, bytes] = {}
        self._node_names_cache: Dict[int, List[str]] = {}
        self._node_parents_cache: Dict[int, List[int]] = {}
        self._node_off_mat_points_cache: Dict[int, List[List[Tuple[float, float, float]]]] = {}
        self._node_off_mat_blocks_cache: Dict[int, List[List[List[float]]]] = {}

        self._parse_header()
        self._parse_fat()
        self._parse_vertex_formats()
        self._parse_material_texture_refs()
        self._parse_meshes()
        self._parse_draw_calls()
        self._parse_index_table()
        self._parse_vertex_table()

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            self._fh.close()

    def __enter__(self) -> "SpkMeshExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _u8(self, pos: int) -> int:
        return self.mm[pos]

    def _u16(self, pos: int) -> int:
        return struct.unpack_from("<H", self.mm, pos)[0]

    def _i16(self, pos: int) -> int:
        return struct.unpack_from("<h", self.mm, pos)[0]

    def _u32(self, pos: int) -> int:
        return struct.unpack_from("<I", self.mm, pos)[0]

    def _f32(self, pos: int) -> float:
        return struct.unpack_from("<f", self.mm, pos)[0]

    def _read_cstring(self, pos: int) -> Tuple[str, int]:
        end = self.mm.find(b"\x00", pos)
        if end == -1:
            raise ValueError("CString is not null-terminated")
        return self.mm[pos:end].decode("utf-8", errors="replace"), end + 1

    def _parse_header(self) -> None:
        if len(self.mm) < 64:
            raise ValueError("SPK is too small")

        magic, platform, version, filesize = struct.unpack_from("<4s4sII", self.mm, 0)
        magic_s = magic.decode("ascii", errors="replace")
        platform_s = platform.decode("ascii", errors="replace")

        if magic_s != "MESH" or platform_s != "PCPC":
            raise ValueError(f"Unsupported SPK magic/platform: {magic_s}/{platform_s}")

        header_off, header_size = struct.unpack_from("<II", self.mm, 32)
        data_off, data_size, data_num = struct.unpack_from("<III", self.mm, 40)
        ft_off, ft_size, ft_num = struct.unpack_from("<III", self.mm, 52)

        if ft_off < 0 or ft_off + ft_size > len(self.mm):
            raise ValueError(
                "Header filetable points outside file bounds: "
                f"offset={ft_off} size={ft_size} file_size={len(self.mm)}"
            )

        if ft_off == 196:
            layout = LAYOUT_196
        elif ft_off in (244, 256):
            layout = LAYOUT_244_256
        else:
            raise ValueError(f"Unsupported SPK header layout (filetable offset={ft_off})")

        out: Dict[str, Any] = {
            "magic": magic_s,
            "platform": platform_s,
            "version": version,
            "filesize": filesize,
            "md5": self.mm[16:32].hex(),
            "header": {"offset": header_off, "size": header_size},
            "data": {"offset": data_off, "size": data_size, "num": data_num},
            "filetable": {"offset": ft_off, "size": ft_size, "num": ft_num},
        }

        pos = 64
        for name, width in layout:
            if width == 3:
                out[name] = {
                    "offset": self._u32(pos),
                    "size": self._u32(pos + 4),
                    "num": self._u32(pos + 8),
                }
                pos += 12
            elif width == 2:
                out[name] = {"offset": self._u32(pos), "size": self._u32(pos + 4)}
                pos += 8
            else:
                raise ValueError("Invalid descriptor width")

        self.header = out

    def _parse_fat(self) -> None:
        ft = self.header["filetable"]
        start = int(ft["offset"]) + 0x09
        size = int(ft["size"]) - 0x09
        if size < 0:
            raise ValueError("Invalid filetable size")

        fat: Dict[str, Dict[str, Any]] = {}

        def rec(path: str, size_left: int, pos: int) -> int:
            while size_left > 0:
                pos0 = pos
                if pos + 8 > len(self.mm):
                    raise ValueError("FAT parser reached EOF")
                name_size = self._u32(pos)
                pos += 4
                entry_size = self._u32(pos)
                pos += 4

                if name_size != 0:
                    name, pos = self._read_cstring(pos)
                    inpath = path + name
                    if entry_size != 0:
                        already = pos - pos0
                        pos = rec(inpath, entry_size - already, pos)
                        size_left -= entry_size
                    else:
                        path = inpath
                        size_left -= name_size
                else:
                    bbmin = struct.unpack_from("<3f", self.mm, pos)
                    pos += 12
                    bbmax = struct.unpack_from("<3f", self.mm, pos)
                    pos += 12
                    flags = self._u32(pos)
                    pos += 4
                    mesh_index = self._u16(pos)
                    pos += 2
                    node_index = self._u16(pos)
                    pos += 2
                    name, pos = self._read_cstring(pos)
                    full = normalize_asset_path(path + name)
                    fat[full] = {
                        "bbmin": bbmin,
                        "bbmax": bbmax,
                        "flags": int(flags),
                        "meshIndex": int(mesh_index),
                        "nodeIndex": int(node_index),
                    }
                    size_left -= (pos - pos0)
            return pos

        rec("", size, start)
        self.fat = fat

    def _parse_vertex_formats(self) -> None:
        info = self.header.get("vertexFormats")
        if not info:
            self.vertex_formats = []
            return

        offset = int(info["offset"])
        count = int(info["num"])
        if count <= 0:
            self.vertex_formats = []
            return

        size = self._u32(offset)
        pos = offset + 4
        out: List[str] = []
        for _ in range(count):
            raw = self.mm[pos : pos + size]
            pos += size
            out.append(raw.decode("utf-8", errors="replace").rstrip("\x00 ").strip())
        self.vertex_formats = out

    def _decode_material_payload(self) -> bytes:
        info = self.header.get("material")
        if not info:
            return b""
        offset = int(info["offset"])
        size = int(info["size"])
        if size <= 0:
            return b""
        block = bytes(self.mm[offset : offset + size])

        best_payload = b""
        best_score = -1
        for i in range(len(block) - 2):
            b0 = block[i]
            b1 = block[i + 1]
            if b0 != 0x78:
                continue
            if ((b0 << 8) + b1) % 31 != 0:
                continue
            try:
                dec = zlib_decode_compat(block[i:])
            except Exception:
                continue
            score = dec.lower().count(b"/pc/atlas/assets/")
            if score > best_score:
                best_score = score
                best_payload = dec
        return best_payload

    @staticmethod
    def _material_ndf_type_size(ndf_type: int) -> int | None:
        sizes = {
            0x00000000: 1,   # Boolean
            0x00000001: 1,   # Int8
            0x00000018: 2,   # Int16
            0x00000019: 2,   # UInt16
            0x00000002: 4,   # Int32
            0x00000003: 4,   # UInt32
            0x00000005: 4,   # Float32
            0x00000006: 8,   # Float64
            0x00000013: 8,   # Long
            0x00000004: 8,   # Time64
            0x0000001F: 8,   # EugInt2
            0x00000021: 8,   # EugFloat2
            0x00000007: 4,   # TableString
            0x0000001C: 4,   # TableStringFile
            0xAAAAAAAA: 4,   # TransTableReference
            0x0000001D: 8,   # LocalisationHash
            0xBBBBBBBB: 8,   # ObjectReference
            0x0000000D: 4,   # Color32
            0x0000000C: 16,  # Color128
            0x0000000B: 12,  # Vector
            0x0000000E: 12,  # TrippleInt
            0x0000001A: 16,  # Guid
            0x00000025: 16,  # Hash
        }
        return sizes.get(int(ndf_type))

    def _parse_material_ndf_value(
        self,
        payload: bytes,
        pos: int,
        strings: Sequence[str],
        depth: int = 0,
    ) -> Tuple[int, Tuple[str, Any]]:
        if depth > 64 or pos + 4 > len(payload):
            return min(pos, len(payload)), ("unknown", None)

        ndf_type = struct.unpack_from("<I", payload, pos)[0]
        pos += 4

        # Reference wrapper prefix used before object/trans refs.
        if ndf_type == 0x00000009 and pos + 4 <= len(payload):
            ndf_type = struct.unpack_from("<I", payload, pos)[0]
            pos += 4

        # Variable-sized values.
        if ndf_type in {0x00000008, 0x00000011, 0x00000012, 0x00000014, 0x0000001E}:
            if pos + 4 > len(payload):
                return min(pos, len(payload)), ("unknown", ndf_type)
            count_or_len = struct.unpack_from("<I", payload, pos)[0]
            pos += 4
            if ndf_type == 0x0000001E:
                # Zip blob has one additional marker byte in this format.
                if pos + 1 > len(payload):
                    return min(pos, len(payload)), ("unknown", ndf_type)
                pos += 1

            if ndf_type == 0x00000011:  # List
                if count_or_len > 1_000_000:
                    return min(pos, len(payload)), ("list", [])
                items: List[Tuple[str, Any]] = []
                for _ in range(int(count_or_len)):
                    pos, item = self._parse_material_ndf_value(payload, pos, strings, depth + 1)
                    items.append(item)
                return pos, ("list", items)

            if ndf_type == 0x00000012:  # MapList
                if count_or_len > 1_000_000:
                    return min(pos, len(payload)), ("maplist", [])
                pairs: List[Tuple[Tuple[str, Any], Tuple[str, Any]]] = []
                for _ in range(int(count_or_len)):
                    pos, key_val = self._parse_material_ndf_value(payload, pos, strings, depth + 1)
                    pos, map_val = self._parse_material_ndf_value(payload, pos, strings, depth + 1)
                    pairs.append((key_val, map_val))
                return pos, ("maplist", pairs)

            if ndf_type == 0x00000008:  # WideString
                end = min(len(payload), pos + int(count_or_len))
                raw = payload[pos:end]
                pos = end
                try:
                    text = raw.decode("utf-16-le", errors="ignore").rstrip("\x00")
                except Exception:
                    text = ""
                return pos, ("wstr", text)

            # Blob / ZipBlob: keep only length marker.
            end = min(len(payload), pos + int(count_or_len))
            pos = end
            return pos, ("blob", int(count_or_len))

        if ndf_type == 0x00000022:  # Map
            pos, key_val = self._parse_material_ndf_value(payload, pos, strings, depth + 1)
            pos, map_val = self._parse_material_ndf_value(payload, pos, strings, depth + 1)
            return pos, ("map", (key_val, map_val))

        fixed_size = self._material_ndf_type_size(ndf_type)
        if fixed_size is None:
            return min(pos, len(payload)), ("unknown", ndf_type)
        if pos + fixed_size > len(payload):
            return min(pos, len(payload)), ("unknown", ndf_type)

        raw = payload[pos : pos + fixed_size]
        pos += fixed_size

        if ndf_type in {0x00000007, 0x0000001C, 0xAAAAAAAA}:
            sidx = struct.unpack_from("<i", raw, 0)[0]
            if 0 <= sidx < len(strings):
                return pos, ("str", strings[sidx])
            return pos, ("str", f"<str:{sidx}>")

        if ndf_type == 0xBBBBBBBB:
            inst_id = struct.unpack_from("<I", raw, 0)[0]
            cls_id = struct.unpack_from("<I", raw, 4)[0]
            return pos, ("objref", (int(inst_id), int(cls_id)))

        if ndf_type == 0x00000003:
            return pos, ("u32", int(struct.unpack_from("<I", raw, 0)[0]))
        if ndf_type == 0x00000002:
            return pos, ("i32", int(struct.unpack_from("<i", raw, 0)[0]))
        if ndf_type == 0x00000005:
            return pos, ("f32", float(struct.unpack_from("<f", raw, 0)[0]))
        if ndf_type == 0x00000000:
            return pos, ("bool", bool(raw[0]))
        return pos, ("raw", ndf_type)

    @staticmethod
    def _ndf_value_to_string(value: Tuple[str, Any] | Any) -> str:
        if not isinstance(value, tuple) or len(value) < 2:
            return ""
        kind, data = value[0], value[1]
        if kind in {"str", "wstr"}:
            return str(data or "").strip()
        return ""

    def _extract_atlas_refs_from_ndf_value(self, value: Tuple[str, Any] | Any) -> List[str]:
        out: List[str] = []
        if not isinstance(value, tuple) or len(value) < 2:
            return out
        kind, data = value[0], value[1]
        if kind in {"str", "wstr"}:
            s = str(data or "")
            if "/pc/atlas/assets/" in s.lower():
                out.append(normalize_atlas_ref(s))
            return out
        if kind == "list":
            for item in data:
                out.extend(self._extract_atlas_refs_from_ndf_value(item))
            return out
        if kind == "maplist":
            for key_val, map_val in data:
                out.extend(self._extract_atlas_refs_from_ndf_value(key_val))
                out.extend(self._extract_atlas_refs_from_ndf_value(map_val))
            return out
        if kind == "map":
            key_val, map_val = data
            out.extend(self._extract_atlas_refs_from_ndf_value(key_val))
            out.extend(self._extract_atlas_refs_from_ndf_value(map_val))
            return out
        return out

    @staticmethod
    def _extract_object_refs_from_ndf_value(value: Tuple[str, Any] | Any) -> List[int]:
        out: List[int] = []
        if not isinstance(value, tuple) or len(value) < 2:
            return out
        kind, data = value[0], value[1]
        if kind == "objref" and isinstance(data, tuple) and len(data) >= 1:
            try:
                out.append(int(data[0]))
            except Exception:
                pass
            return out
        if kind == "list":
            for item in data:
                out.extend(SpkMeshExtractor._extract_object_refs_from_ndf_value(item))
            return out
        if kind == "maplist":
            for key_val, map_val in data:
                out.extend(SpkMeshExtractor._extract_object_refs_from_ndf_value(key_val))
                out.extend(SpkMeshExtractor._extract_object_refs_from_ndf_value(map_val))
            return out
        if kind == "map":
            key_val, map_val = data
            out.extend(SpkMeshExtractor._extract_object_refs_from_ndf_value(key_val))
            out.extend(SpkMeshExtractor._extract_object_refs_from_ndf_value(map_val))
            return out
        return out

    def _extract_texture_slots_from_value(self, value: Tuple[str, Any] | Any) -> Dict[str, str]:
        raw = self._extract_texture_slots_raw_from_value(value)
        out: Dict[str, str] = {}
        for slot_name, raw_value in raw.items():
            low = str(raw_value or "").strip().lower()
            if "/pc/atlas/assets/" in low or low.startswith("pc/atlas/assets/") or low.startswith("assets/"):
                out.setdefault(slot_name, normalize_atlas_ref(str(raw_value)))
        return out

    def _extract_texture_slots_raw_from_value(self, value: Tuple[str, Any] | Any) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not isinstance(value, tuple) or len(value) < 2:
            return out
        if value[0] != "maplist":
            return out
        entries = value[1]
        if not isinstance(entries, list):
            return out

        for pair in entries:
            if not isinstance(pair, tuple) or len(pair) < 2:
                continue
            key_val, map_val = pair
            slot_name = self._ndf_value_to_string(key_val)
            if not slot_name:
                continue
            raw_value = self._ndf_value_to_string(map_val)
            if raw_value:
                out.setdefault(slot_name, raw_value)
                continue
            refs = self._extract_atlas_refs_from_ndf_value(map_val)
            if refs:
                out.setdefault(slot_name, refs[0])
        return out

    @staticmethod
    def _texture_slot_priority(slot_name: str, ref: str) -> int:
        low = f"{slot_name}_{Path(ref).name}".lower()
        if "combinedda" in low or "coloralpha" in low:
            return 0
        if "diffuse" in low and "noalpha" not in low:
            return 1
        if "diffuse" in low:
            return 2
        if "normal" in low:
            return 3
        if "combinedorm" in low or "combinedrm" in low or "_orm" in low:
            return 4
        return 10

    def _ordered_refs_from_slots(self, slot_map: Dict[str, str]) -> List[str]:
        items = sorted(
            slot_map.items(),
            key=lambda kv: (self._texture_slot_priority(kv[0], kv[1]), kv[0].lower()),
        )
        refs = [normalize_atlas_ref(v) for _, v in items if str(v).strip()]
        return unique_keep_order(refs)

    @staticmethod
    def _material_name_from_raw_slots(slot_map: Dict[str, str]) -> str:
        if not isinstance(slot_map, dict) or not slot_map:
            return ""
        preferred_slots = [
            "DiffuseTextureNoAlpha",
            "DiffuseTexture",
            "BaseColorTexture",
            "ColorTexture",
            "NormalTexture",
            "RoughnessTexture",
            "MetallicTexture",
            "AmbientOcclusionTexture",
            "OpacityTexture",
        ]
        candidates: List[str] = []
        for key in preferred_slots:
            raw = str(slot_map.get(key, "")).strip()
            if raw:
                candidates.append(raw)
        if not candidates:
            for raw in slot_map.values():
                val = str(raw or "").strip()
                if val:
                    candidates.append(val)
        for raw in candidates:
            stem = Path(raw).stem
            if not stem:
                continue
            base = re.sub(r"(?i)_(d|nm|r|m|o|ao|a|orm)$", "", stem).strip("_")
            if not base:
                base = stem.strip("_")
            if base:
                return sanitize_material_name(base)
        return ""

    def _parse_material_texture_refs_structured(
        self,
        payload: bytes,
        toc_index_hint: int | None = None,
        offset_shift: int = 0,
    ) -> Tuple[Dict[int, Dict[str, str]], Dict[int, str], List[str], Dict[int, Dict[str, str]]]:
        if not payload:
            return {}, {}, [], {}

        toc_index = -1
        if toc_index_hint is not None and 0 <= toc_index_hint < len(payload):
            if payload[toc_index_hint : toc_index_hint + 4] == b"TOC0":
                toc_index = toc_index_hint
        if toc_index < 0:
            toc_index = payload.rfind(b"TOC0")
        if toc_index < 0 or toc_index + 8 > len(payload):
            return {}, {}, [], {}

        entry_count = struct.unpack_from("<I", payload, toc_index + 4)[0]
        if entry_count <= 0 or entry_count > 1024:
            return {}, {}, [], {}

        pos = toc_index + 8
        entries: Dict[str, Tuple[int, int]] = {}
        for _ in range(int(entry_count)):
            if pos + 24 > len(payload):
                return {}, {}, [], {}
            name = payload[pos : pos + 8].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            pos += 8
            off = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
            size = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
            adj_off = int(off) - int(offset_shift)
            entries[name] = (adj_off, int(size))

        need = {"OBJE", "CHNK", "CLAS", "PROP", "STRG"}
        if not need.issubset(set(entries.keys())):
            return {}, {}, [], {}

        def read_string_table(offset: int, size: int) -> List[str]:
            out: List[str] = []
            p = int(offset)
            end = min(len(payload), int(offset + size))
            while p + 4 <= end:
                sl = struct.unpack_from("<I", payload, p)[0]
                p += 4
                if sl < 0 or p + int(sl) > end:
                    break
                out.append(payload[p : p + int(sl)].decode("latin1", errors="ignore"))
                p += int(sl)
            return out

        classes = read_string_table(*entries["CLAS"])
        strings = read_string_table(*entries["STRG"])

        prop_names: List[str] = []
        p = int(entries["PROP"][0])
        pend = min(len(payload), int(entries["PROP"][0] + entries["PROP"][1]))
        while p + 8 <= pend:
            sl = struct.unpack_from("<I", payload, p)[0]
            p += 4
            if sl < 0 or p + int(sl) + 4 > pend:
                break
            prop_name = payload[p : p + int(sl)].decode("latin1", errors="ignore")
            p += int(sl)
            _cls = struct.unpack_from("<I", payload, p)[0]
            p += 4
            prop_names.append(prop_name)

        chnk_off, chnk_size = entries["CHNK"]
        if chnk_size < 8 or chnk_off + 8 > len(payload):
            return {}, {}, [], {}
        obj_count = struct.unpack_from("<I", payload, chnk_off + 4)[0]
        if obj_count <= 0 or obj_count > 10_000_000:
            return {}, {}, [], {}

        obje_off, obje_size = entries["OBJE"]
        p = int(obje_off)
        obje_end = min(len(payload), int(obje_off + obje_size))

        slots_by_object_id: Dict[int, Dict[str, str]] = {}
        raw_slots_by_object_id: Dict[int, Dict[str, str]] = {}
        names_by_object_id: Dict[int, str] = {}
        material_object_order: List[int] = []
        refs_all: List[str] = []

        for obj_id in range(int(obj_count)):
            if p + 4 > obje_end:
                break
            cls_id = struct.unpack_from("<I", payload, p)[0]
            p += 4

            cls_name = classes[cls_id] if 0 <= cls_id < len(classes) else ""
            mat_name = ""
            slot_refs: Dict[str, str] = {}

            while p + 4 <= obje_end:
                prop_id = struct.unpack_from("<I", payload, p)[0]
                p += 4
                if prop_id == 0xABABABAB:
                    break
                prop_name = prop_names[prop_id] if 0 <= prop_id < len(prop_names) else f"prop_{prop_id}"
                p, val = self._parse_material_ndf_value(payload, p, strings)
                if cls_name in {"TEugBListPBaseClass", "TMeshPack_MeshMaterialList"} and prop_name in {"Value", "Materials"} and not material_object_order:
                    material_object_order = self._extract_object_refs_from_ndf_value(val)
                elif cls_name == "TMeshMaterial":
                    if prop_name == "MaterialName":
                        mat_name = self._ndf_value_to_string(val)
                    elif prop_name in {"Textures", "TextureDico"}:
                        raw_slots = self._extract_texture_slots_raw_from_value(val)
                        if raw_slots:
                            raw_slots_by_object_id.setdefault(int(obj_id), {}).update(raw_slots)
                        slot_refs.update(self._extract_texture_slots_from_value(val))

            if cls_name != "TMeshMaterial":
                continue

            if mat_name:
                names_by_object_id[int(obj_id)] = str(mat_name)
            if slot_refs:
                clean_slot_refs = {str(k): normalize_atlas_ref(str(v)) for k, v in slot_refs.items() if str(v).strip()}
                if clean_slot_refs:
                    slots_by_object_id[int(obj_id)] = clean_slot_refs
                    refs_all.extend(self._ordered_refs_from_slots(clean_slot_refs))

        slots_by_id: Dict[int, Dict[str, str]] = {}
        raw_slots_by_id: Dict[int, Dict[str, str]] = {}
        names_by_id: Dict[int, str] = {}
        if material_object_order:
            for mat_index, obj_ref_id in enumerate(material_object_order):
                obj_id = int(obj_ref_id)
                slot_map = slots_by_object_id.get(obj_id)
                if slot_map:
                    slots_by_id[int(mat_index)] = slot_map
                raw_slot_map = raw_slots_by_object_id.get(obj_id)
                if raw_slot_map:
                    raw_slots_by_id[int(mat_index)] = dict(raw_slot_map)
                mat_name = names_by_object_id.get(obj_id)
                if mat_name:
                    names_by_id[int(mat_index)] = mat_name
        else:
            slots_by_id = dict(slots_by_object_id)
            raw_slots_by_id = {int(k): dict(v) for k, v in raw_slots_by_object_id.items()}
            names_by_id = dict(names_by_object_id)

        return slots_by_id, names_by_id, unique_keep_order(refs_all), raw_slots_by_id

    def _parse_material_texture_refs(self) -> None:
        self.material_texture_refs = []
        self.material_refs_by_dir = {}
        self.material_texture_slots_by_id = {}
        self.material_texture_names_by_id = {}
        self.material_texture_refs_by_material = {}
        self.material_name_by_id = {}

        payload = self._decode_material_payload()
        refs: List[str] = []

        toc_hint: int | None = None
        offset_shift = 0
        info = self.header.get("material")
        if info:
            try:
                offset = int(info["offset"])
                size = int(info["size"])
                if size > 0:
                    block = bytes(self.mm[offset : offset + size])
                    if len(block) >= 40 and block[:4] == b"EUG0" and block[8:12] == b"CNDF":
                        footer_off = struct.unpack_from("<Q", block, 16)[0]
                        header_size = struct.unpack_from("<Q", block, 24)[0]
                        offset_shift = int(header_size)
                        if footer_off >= header_size:
                            toc_hint = int(footer_off - header_size)
            except Exception:
                toc_hint = None
                offset_shift = 0

        if payload:
            slots_by_id, names_by_id, refs_structured, raw_slots_by_id = self._parse_material_texture_refs_structured(
                payload,
                toc_index_hint=toc_hint,
                offset_shift=offset_shift,
            )
            if slots_by_id:
                self.material_texture_slots_by_id = slots_by_id
                self.material_name_by_id = names_by_id
                refs_by_mid: Dict[int, List[str]] = {}
                refs_ordered: List[str] = []
                for mid in sorted(slots_by_id.keys()):
                    ordered = self._ordered_refs_from_slots(slots_by_id[mid])
                    refs_by_mid[int(mid)] = ordered
                    refs_ordered.extend(ordered)
                self.material_texture_refs_by_material = refs_by_mid
                refs = unique_keep_order(refs_ordered)
                if not refs and refs_structured:
                    refs = unique_keep_order(refs_structured)
            if raw_slots_by_id:
                self.material_texture_names_by_id = {
                    int(mid): {str(k): str(v) for k, v in slots.items()}
                    for mid, slots in raw_slots_by_id.items()
                    if isinstance(slots, dict) and slots
                }
                for mid, slots in self.material_texture_names_by_id.items():
                    if str(self.material_name_by_id.get(int(mid), "")).strip():
                        continue
                    guessed = self._material_name_from_raw_slots(slots)
                    if guessed:
                        self.material_name_by_id[int(mid)] = guessed

        # Fallback for unsupported payload variants: raw regex path scan.
        if not refs and payload:
            raw_paths = [m.group(0).decode("utf-8", errors="replace") for m in ATLAS_PATH_RX.finditer(payload)]
            refs = unique_keep_order([normalize_atlas_ref(p) for p in raw_paths])

        by_dir: Dict[str, List[str]] = {}
        for ref in refs:
            d = normalize_asset_path(str(PurePosixPath(ref).parent))
            by_dir.setdefault(d.lower(), []).append(ref)

        self.material_texture_refs = refs
        self.material_refs_by_dir = by_dir

    def get_texture_refs_for_material(self, material_id: int) -> List[str]:
        mid = int(material_id)
        refs = self.material_texture_refs_by_material.get(mid, [])
        if refs:
            return list(refs)
        slot_map = self.material_texture_slots_by_id.get(mid, {})
        if not slot_map:
            return []
        return self._ordered_refs_from_slots(slot_map)

    def get_texture_refs_for_material_ids(self, material_ids: Sequence[int]) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for mid in material_ids:
            out[int(mid)] = self.get_texture_refs_for_material(int(mid))
        return out

    def find_texture_refs_for_asset(
        self,
        asset_path: str,
        material_ids: Sequence[int] | None = None,
    ) -> List[str]:
        if material_ids:
            refs_by_mid = self.get_texture_refs_for_material_ids(material_ids)
            refs: List[str] = []
            for mid in material_ids:
                refs.extend(refs_by_mid.get(int(mid), []))
            refs = unique_keep_order(refs)
            if refs:
                return refs

        if not self.material_refs_by_dir:
            return []
        asset_norm = normalize_asset_path(asset_path)
        asset_dir = normalize_asset_path(str(PurePosixPath(asset_norm).parent))
        p = PurePosixPath(asset_dir)
        leaf = p.name
        parent = normalize_asset_path(str(p.parent))
        leaf_base = strip_lod_suffix(leaf)

        candidates: List[str] = [asset_dir]
        if leaf_base != leaf:
            candidates.append(normalize_asset_path(str(PurePosixPath(parent) / leaf_base)))
        if parent:
            candidates.append(parent)

        for c in unique_keep_order(candidates):
            refs = self.material_refs_by_dir.get(c.lower())
            if refs:
                return refs

        # Fallback: close siblings like Leopard_1A1_CMD / Leopard_1A1_LOW
        fallback: List[str] = []
        if parent and leaf_base:
            pref = f"{parent.lower()}/{leaf_base.lower()}"
            for d, refs in self.material_refs_by_dir.items():
                if d.startswith(pref):
                    fallback.extend(refs)
        fallback = unique_keep_order(fallback)
        if 0 < len(fallback) <= 24:
            return fallback
        return []

    def find_best_fat_entry_for_asset(self, asset_path: str) -> Tuple[str, Dict[str, Any]] | None:
        if not self.fat:
            return None
        asset_norm = normalize_asset_path(asset_path)
        exact = self.fat.get(asset_norm)
        if exact is not None:
            return asset_norm, exact

        tgt = PurePosixPath(asset_norm)
        tgt_name = tgt.name.lower()
        tgt_stem = strip_lod_suffix(tgt.stem).lower()

        candidates: List[Tuple[int, int, str, Dict[str, Any]]] = []
        for path, meta in self.fat.items():
            p = PurePosixPath(path)
            score = 0
            if p.name.lower() == tgt_name:
                score += 5000
            if strip_lod_suffix(p.stem).lower() == tgt_stem and p.suffix.lower() == tgt.suffix.lower():
                score += 3000
            score += shared_suffix_score(path, asset_norm) * 10
            if score > 0:
                candidates.append((score, len(path), path, meta))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1], x[2].lower()))
        _, _, best_path, best_meta = candidates[0]
        return best_path, best_meta

    def get_node_blob(self, node_index: int) -> bytes:
        cached = self._node_blob_cache.get(node_index)
        if cached is not None:
            return cached

        info = self.header.get("nodeTable")
        data_info = self.header.get("nodeData")
        if not info or not data_info:
            return b""
        count = int(info.get("num", 0))
        if node_index < 0 or node_index >= count:
            return b""

        base = int(info["offset"]) + node_index * 8
        start = self._u32(base)
        size = self._u32(base + 4)
        data_off = int(data_info["offset"]) + int(start)
        blob = bytes(self.mm[data_off : data_off + size])
        self._node_blob_cache[node_index] = blob
        return blob

    @staticmethod
    def _parse_node_blob_layout(blob: bytes) -> Tuple[int, int, int, int] | None:
        if len(blob) < 32:
            return None
        try:
            node_count = int(struct.unpack_from("<I", blob, 4)[0])
            off_mat = int(struct.unpack_from("<I", blob, 8)[0])
            off_parent = int(struct.unpack_from("<I", blob, 12)[0])
            off_names = int(struct.unpack_from("<I", blob, 16)[0])
        except Exception:
            return None

        if node_count <= 0 or node_count > 8192:
            return None
        if off_mat < 0 or off_parent < 0 or off_names < 0:
            return None
        if off_mat > len(blob) or off_parent > len(blob) or off_names > len(blob):
            return None
        if off_parent + node_count * 4 > len(blob):
            return None
        if off_names < off_parent:
            return None
        return node_count, off_mat, off_parent, off_names

    @staticmethod
    def _strip_zero_tail(raw: bytes) -> bytes:
        end = len(raw)
        while end > 0 and raw[end - 1] == 0:
            end -= 1
        return raw[:end]

    @staticmethod
    def _parse_names_from_cstrings(raw: bytes, node_count: int) -> List[str]:
        trimmed = SpkMeshExtractor._strip_zero_tail(raw)
        if not trimmed or b"\x00" not in trimmed:
            return []
        parts = trimmed.split(b"\x00")
        names: List[str] = []
        for part in parts:
            if len(names) >= node_count:
                break
            names.append(part.decode("latin1", errors="ignore").strip())
        if len(names) != node_count:
            return []
        return names

    @staticmethod
    def _select_name_lengths_blob(blob: bytes, node_count: int, off_mat: int, names_len: int) -> List[int]:
        if node_count <= 0 or off_mat <= 0 or names_len <= 0:
            return []
        need = node_count * 2
        if off_mat < need:
            return []

        best_score = -10_000
        best_lens: List[int] = []
        max_start = off_mat - need
        for start in range(0, max_start + 1, 2):
            try:
                lens = list(struct.unpack_from(f"<{node_count}H", blob, start))
            except Exception:
                continue
            total = int(sum(lens))
            if total != names_len:
                continue
            if all(v == 0 for v in lens):
                continue
            if max(lens) > 1024:
                continue
            if sum(1 for v in lens if v > 0) < max(1, node_count // 8):
                continue

            score = 0
            score += 500
            score += sum(1 for v in lens if v > 0)
            score += max(0, start - 24) // 2
            if lens[0] == 0:
                score += 20
            if score > best_score:
                best_score = score
                best_lens = lens
        return best_lens

    @classmethod
    def _parse_names_from_concat_with_lengths(
        cls,
        blob: bytes,
        node_count: int,
        off_mat: int,
        names_raw: bytes,
    ) -> List[str]:
        trimmed = cls._strip_zero_tail(names_raw)
        if not trimmed:
            return []
        lens = cls._select_name_lengths_blob(blob, node_count, off_mat, len(trimmed))
        if not lens:
            return []

        out: List[str] = []
        pos = 0
        for ln in lens:
            if ln <= 0:
                out.append("")
                continue
            chunk = trimmed[pos : pos + ln]
            pos += ln
            out.append(chunk.decode("latin1", errors="ignore").strip())
        if pos != len(trimmed):
            return []
        return out

    def _parse_node_structured(self, node_index: int) -> Tuple[List[str], List[int]]:
        blob = self.get_node_blob(node_index)
        if not blob:
            return [], []
        layout = self._parse_node_blob_layout(blob)
        if layout is None:
            return [], []
        node_count, off_mat, off_parent, off_names = layout

        parents: List[int] = []
        for i in range(node_count):
            try:
                parents.append(int(struct.unpack_from("<i", blob, off_parent + i * 4)[0]))
            except Exception:
                parents = []
                break

        names_raw = blob[off_names:]
        names = self._parse_names_from_cstrings(names_raw, node_count)
        if not names:
            names = self._parse_names_from_concat_with_lengths(blob, node_count, off_mat, names_raw)

        if names:
            if len(names) < node_count:
                names.extend([""] * (node_count - len(names)))
            elif len(names) > node_count:
                names = names[:node_count]
        return names, parents

    def parse_node_parent_indices(self, node_index: int) -> List[int]:
        cached = self._node_parents_cache.get(node_index)
        if cached is not None:
            return cached
        _, parents = self._parse_node_structured(node_index)
        self._node_parents_cache[node_index] = parents
        return parents

    def parse_node_names(self, node_index: int) -> List[str]:
        cached = self._node_names_cache.get(node_index)
        if cached is not None:
            return cached

        names, parents = self._parse_node_structured(node_index)
        if names:
            self._node_names_cache[node_index] = names
            self._node_parents_cache[node_index] = parents
            return names

        blob = self.get_node_blob(node_index)
        if not blob:
            self._node_names_cache[node_index] = []
            return []

        fallback_names: List[str] = []
        m = NODE_NAME_TAIL_RX.search(blob)
        if m:
            tail = m.group(0).decode("ascii", errors="ignore")
            fallback_names.extend(split_concatenated_node_names(tail))

        if not fallback_names:
            best = ""
            for mm in re.finditer(rb"[A-Za-z0-9_.]{12,}", blob):
                candidate = mm.group(0).decode("ascii", errors="ignore")
                if candidate.count("_") >= 2 and len(candidate) > len(best):
                    best = candidate
            if best:
                fallback_names.extend(split_concatenated_node_names(best))

        names = [n for n in fallback_names if n]
        self._node_names_cache[node_index] = names
        return names

    def parse_node_off_mat_points(self, node_index: int) -> List[List[Tuple[float, float, float]]]:
        cached = self._node_off_mat_points_cache.get(node_index)
        if cached is not None:
            return cached

        blob = self.get_node_blob(node_index)
        if not blob:
            self._node_off_mat_points_cache[node_index] = []
            return []

        layout = self._parse_node_blob_layout(blob)
        if layout is None:
            self._node_off_mat_points_cache[node_index] = []
            return []
        node_count, off_mat, off_parent, _off_names = layout

        span = int(off_parent) - int(off_mat)
        if node_count <= 0 or span <= 0 or span % node_count != 0:
            self._node_off_mat_points_cache[node_index] = []
            return []
        stride = span // node_count
        if stride <= 0 or stride % 4 != 0:
            self._node_off_mat_points_cache[node_index] = []
            return []

        floats_per_node = stride // 4
        block_count = floats_per_node // 12
        if block_count <= 0:
            self._node_off_mat_points_cache[node_index] = []
            return []

        out: List[List[Tuple[float, float, float]]] = []
        unpack_fmt = f"<{floats_per_node}f"
        for idx in range(node_count):
            try:
                vals = struct.unpack_from(unpack_fmt, blob, off_mat + idx * stride)
            except Exception:
                out = []
                break
            pts: List[Tuple[float, float, float]] = []
            for block_idx in range(block_count):
                base = block_idx * 12
                try:
                    tx = float(vals[base + 3])
                    ty = float(vals[base + 7])
                    tz = float(vals[base + 11])
                except Exception:
                    continue
                pts.append((tx, ty, tz))
            out.append(pts)

        self._node_off_mat_points_cache[node_index] = out
        return out

    def parse_node_off_mat_blocks(self, node_index: int) -> List[List[List[float]]]:
        cached = self._node_off_mat_blocks_cache.get(node_index)
        if cached is not None:
            return cached

        blob = self.get_node_blob(node_index)
        if not blob:
            self._node_off_mat_blocks_cache[node_index] = []
            return []

        layout = self._parse_node_blob_layout(blob)
        if layout is None:
            self._node_off_mat_blocks_cache[node_index] = []
            return []
        node_count, off_mat, off_parent, _off_names = layout

        span = int(off_parent) - int(off_mat)
        if node_count <= 0 or span <= 0 or span % node_count != 0:
            self._node_off_mat_blocks_cache[node_index] = []
            return []
        stride = span // node_count
        if stride <= 0 or stride % 4 != 0:
            self._node_off_mat_blocks_cache[node_index] = []
            return []

        floats_per_node = stride // 4
        block_count = floats_per_node // 12
        if block_count <= 0:
            self._node_off_mat_blocks_cache[node_index] = []
            return []

        out: List[List[List[float]]] = []
        unpack_fmt = f"<{floats_per_node}f"
        for idx in range(node_count):
            try:
                vals = struct.unpack_from(unpack_fmt, blob, off_mat + idx * stride)
            except Exception:
                out = []
                break
            node_blocks: List[List[float]] = []
            for block_idx in range(block_count):
                base = block_idx * 12
                try:
                    node_blocks.append([float(v) for v in vals[base : base + 12]])
                except Exception:
                    continue
            out.append(node_blocks)

        self._node_off_mat_blocks_cache[node_index] = out
        return out

    def parse_raw_scene_graph(self, node_index: int) -> RawSceneGraph | None:
        names = list(self.parse_node_names(node_index))
        if not names:
            return None
        parents = list(self.parse_node_parent_indices(node_index))
        off_mat_points = list(self.parse_node_off_mat_points(node_index))
        off_mat_blocks = list(self.parse_node_off_mat_blocks(node_index))
        node_count = len(names)
        if len(parents) < node_count:
            parents.extend([-1] * (node_count - len(parents)))
        if len(off_mat_points) < node_count:
            off_mat_points.extend([[] for _ in range(node_count - len(off_mat_points))])
        if len(off_mat_blocks) < node_count:
            off_mat_blocks.extend([[] for _ in range(node_count - len(off_mat_blocks))])

        def _matrix_signature_rows(matrix_rows: Sequence[Sequence[float]] | None, digits: int = 6) -> tuple[tuple[float, ...], ...] | None:
            if not isinstance(matrix_rows, (list, tuple)):
                return None
            rows: list[tuple[float, ...]] = []
            for row in matrix_rows:
                if not isinstance(row, (list, tuple)):
                    return None
                vals: list[float] = []
                for value in row:
                    try:
                        vals.append(round(float(value), digits))
                    except Exception:
                        return None
                rows.append(tuple(vals))
            return tuple(rows)

        def _pick_unique_transform_candidate(candidates: Sequence[RawNodeTransform]) -> RawNodeTransform | None:
            if len(candidates) == 1:
                return candidates[0]
            unique_by_matrix: dict[tuple[tuple[float, ...], ...], RawNodeTransform] = {}
            for candidate in candidates:
                sig = _matrix_signature_rows(getattr(candidate, "matrix_local", None))
                if sig is None:
                    return None
                if sig not in unique_by_matrix:
                    unique_by_matrix[sig] = candidate
            if len(unique_by_matrix) == 1:
                return next(iter(unique_by_matrix.values()))
            return None

        records: List[RawNodeRecord] = []
        for idx in range(node_count):
            raw_name = str(names[idx] or "").strip()
            point_rows: List[List[float]] = []
            for raw_point in off_mat_points[idx] or []:
                if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 3:
                    continue
                try:
                    point_rows.append(
                        [
                            round(-float(raw_point[0]) * float(WARNO_OFF_MAT_SCALE), 9),
                            round(float(raw_point[1]) * float(WARNO_OFF_MAT_SCALE), 9),
                            round(-float(raw_point[2]) * float(WARNO_OFF_MAT_SCALE), 9),
                        ]
                    )
                except Exception:
                    continue
            block_rows: List[List[float]] = []
            own_transform_candidates: List[RawNodeTransform] = []
            for block_idx, raw_block in enumerate(off_mat_blocks[idx] or []):
                if not isinstance(raw_block, (list, tuple)) or len(raw_block) < 12:
                    continue
                try:
                    block_vals = [float(v) for v in raw_block[:12]]
                except Exception:
                    continue
                block_rows.append(block_vals)
                raw_matrix = _off_mat_block_matrix_rows(block_vals)
                if raw_matrix is None:
                    continue
                blender_matrix = _off_mat_convert_matrix_rows(raw_matrix, "xyz:+1-1+1")
                if blender_matrix is None:
                    continue
                decomp = _decompose_affine_matrix_rows(blender_matrix)
                if decomp is None:
                    continue
                translation, rotation_basis, scale = decomp
                own_transform_candidates.append(
                    RawNodeTransform(
                        source="own_block",
                        block_index=int(block_idx),
                        translation=[float(v) for v in translation],
                        rotation_basis=[[float(v) for v in row] for row in rotation_basis],
                        scale=[float(v) for v in scale],
                        matrix_local=[[float(v) for v in row] for row in blender_matrix],
                    )
                )
            records.append(
                RawNodeRecord(
                    index=int(idx),
                    raw_name=raw_name,
                    safe_name=_safe_name_text(raw_name),
                    parent_index=int(parents[idx]) if idx < len(parents) else -1,
                    off_mat_points=point_rows,
                    off_mat_blocks=block_rows,
                    own_transform_candidates=own_transform_candidates,
                )
            )

        stream_start_index, stream_end_index = _off_mat_stream_section_bounds(node_count)
        for source_idx in range(int(stream_start_index), int(stream_end_index)):
            if source_idx < 0 or source_idx >= len(records):
                continue
            source_record = records[int(source_idx)]
            for transform in source_record.own_transform_candidates:
                target_idx = _off_mat_stream_target_index(int(source_idx), int(transform.block_index), node_count)
                if target_idx < 0 or target_idx >= len(records):
                    continue
                records[int(target_idx)].stream_transform_candidates.append(
                    RawNodeTransform(
                        source="stream_local",
                        block_index=int(transform.block_index),
                        translation=list(transform.translation),
                        rotation_basis=[list(row) for row in transform.rotation_basis],
                        scale=list(transform.scale),
                        matrix_local=[list(row) for row in transform.matrix_local],
                    )
                )

        for record in records:
            unique_stream = _pick_unique_transform_candidate(record.stream_transform_candidates)
            if unique_stream is not None:
                record.local_transform = unique_stream

        world_matrix_by_index: Dict[int, List[List[float]]] = {}
        deterministic_world_point_by_index: Dict[int, List[float]] = {}

        for record in records:
            if record.parent_index >= 0:
                continue
            if record.local_transform is not None:
                world_matrix_by_index[int(record.index)] = [list(row) for row in record.local_transform.matrix_local]
                deterministic_world_point_by_index[int(record.index)] = _mat4_translation(record.local_transform.matrix_local)
                continue
            if len(record.off_mat_points) == 1:
                point = [float(v) for v in record.off_mat_points[0][:3]]
                record.world_translation = list(point)
                deterministic_world_point_by_index[int(record.index)] = list(point)
                world_matrix_by_index[int(record.index)] = _compose_affine_matrix_rows(
                    point,
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 1.0, 1.0],
                )
            else:
                record.deterministic_miss = record.deterministic_miss or "missing_root_world"

        changed = True
        for _ in range(max(1, len(records) + 1)):
            if not changed:
                break
            changed = False
            for record in records:
                idx = int(record.index)
                if idx in world_matrix_by_index:
                    continue
                parent_idx = int(record.parent_index)
                parent_world = world_matrix_by_index.get(parent_idx)
                if parent_world is None:
                    continue
                if record.local_transform is not None:
                    world_matrix = _mat4_mul(parent_world, record.local_transform.matrix_local)
                    world_matrix_by_index[idx] = world_matrix
                    deterministic_world_point_by_index[idx] = _mat4_translation(world_matrix)
                    changed = True
                    continue
                if record.own_transform_candidates:
                    local_candidate = record.own_transform_candidates[0]
                    record.local_transform = RawNodeTransform(
                        source="own_block_local_fallback",
                        block_index=int(local_candidate.block_index),
                        translation=[float(v) for v in local_candidate.translation],
                        rotation_basis=[[float(v) for v in row] for row in local_candidate.rotation_basis],
                        scale=[float(v) for v in local_candidate.scale],
                        matrix_local=[[float(v) for v in row] for row in local_candidate.matrix_local],
                    )
                    world_matrix = _mat4_mul(parent_world, record.local_transform.matrix_local)
                    world_matrix_by_index[idx] = world_matrix
                    deterministic_world_point_by_index[idx] = _mat4_translation(world_matrix)
                    changed = True
                    continue
                if len(record.off_mat_points) == 1:
                    child_world = _compose_affine_matrix_rows(
                        record.off_mat_points[0],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [1.0, 1.0, 1.0],
                    )
                    parent_world_inv = _mat4_inverse_affine(parent_world)
                    if parent_world_inv is None:
                        record.deterministic_miss = record.deterministic_miss or "parent_inverse_failed"
                        continue
                    local_matrix = _mat4_mul(parent_world_inv, child_world)
                    decomp = _decompose_affine_matrix_rows(local_matrix)
                    if decomp is None:
                        record.deterministic_miss = record.deterministic_miss or "local_decompose_failed"
                        continue
                    translation, rotation_basis, scale = decomp
                    record.local_transform = RawNodeTransform(
                        source="single_world_point",
                        block_index=0,
                        translation=[float(v) for v in translation],
                        rotation_basis=[[float(v) for v in row] for row in rotation_basis],
                        scale=[float(v) for v in scale],
                        matrix_local=[[float(v) for v in row] for row in local_matrix],
                        world_translation=[float(v) for v in record.off_mat_points[0][:3]],
                    )
                    world_matrix_by_index[idx] = child_world
                    deterministic_world_point_by_index[idx] = [float(v) for v in record.off_mat_points[0][:3]]
                    changed = True

        for record in records:
            idx = int(record.index)
            world_point = deterministic_world_point_by_index.get(idx)
            if world_point is not None:
                record.world_translation = [float(v) for v in world_point[:3]]
                if record.local_transform is not None and record.local_transform.world_translation is None:
                    record.local_transform.world_translation = [float(v) for v in world_point[:3]]
            elif not record.deterministic_miss:
                if record.local_transform is None:
                    record.deterministic_miss = "missing_local_transform"
                else:
                    record.deterministic_miss = "missing_world_transform"

        return RawSceneGraph(
            node_index=int(node_index),
            node_count=int(node_count),
            stream_start_index=int(stream_start_index),
            stream_end_index=int(stream_end_index),
            records=records,
        )

    def find_node_names_for_asset(self, asset_path: str) -> List[str]:
        hit = self.find_best_fat_entry_for_asset(asset_path)
        if hit is None:
            return []
        _, meta = hit
        node_index = int(meta.get("nodeIndex", -1))
        if node_index < 0:
            return []
        return self.parse_node_names(node_index)

    def _parse_meshes(self) -> None:
        info = self.header["mesh"]
        pos = int(info["offset"])
        count = int(info["num"])
        out: List[Dict[str, int]] = []
        for _ in range(count):
            out.append({"drawCall": self._u16(pos), "num": self._u16(pos + 2)})
            pos += 4
        self.meshes = out

    def _parse_draw_calls(self) -> None:
        info = self.header["drawCalls"]
        pos = int(info["offset"])
        count = int(info["num"])
        size = int(info.get("size", 0))
        stride = 8
        if count > 0 and size > 0 and size % count == 0:
            cand = size // count
            if cand >= 8:
                # WARNO layout-256 packs (e.g. Fulda/CommonSet) use 12-byte draw calls:
                # file:u16 material:u16 index:u16 vertex:u16 unknown08:u32.
                stride = cand
        self.draw_call_stride = stride
        out: List[Dict[str, int]] = []
        for _ in range(count):
            dc: Dict[str, int] = {
                "file": self._u16(pos),
                "material": self._u16(pos + 2),
                "index": self._u16(pos + 4),
                "vertex": self._u16(pos + 6),
            }
            if stride >= 12:
                dc["unknown08"] = self._u32(pos + 8)
            out.append(dc)
            pos += stride
        self.draw_calls = out

    def _parse_index_table(self) -> None:
        info = self.header["indexTable"]
        pos = int(info["offset"])
        count = int(info["num"])
        out: List[Dict[str, int]] = []
        for _ in range(count):
            out.append(
                {
                    "offset": self._u32(pos),
                    "size": self._u32(pos + 4),
                    "num": self._u32(pos + 8),
                    "format": self._u16(pos + 12),
                    "unknown": self._u8(pos + 14),
                    "packed": self._u8(pos + 15),
                }
            )
            pos += 16
        self.index_table = out

    def _parse_vertex_table(self) -> None:
        info = self.header["vertexTable"]
        pos = int(info["offset"])
        count = int(info["num"])
        out: List[Dict[str, int]] = []
        for _ in range(count):
            out.append(
                {
                    "offset": self._u32(pos),
                    "size": self._u32(pos + 4),
                    "num": self._u32(pos + 8),
                    "format": self._u16(pos + 12),
                    "unknown": self._u8(pos + 14),
                    "packed": self._u8(pos + 15),
                }
            )
            pos += 16
        self.vertex_table = out

    def list_entries(self) -> List[Entry]:
        out = [(k, v) for k, v in self.fat.items()]
        out.sort(key=lambda it: it[0].lower())
        return out

    def find_matches(self, query: str | None, asset: str | None) -> List[Entry]:
        entries = self.list_entries()
        if asset:
            target = normalize_asset_path(asset).lower()
            return [it for it in entries if it[0].lower() == target]
        if query:
            needle = query.lower()
            return [it for it in entries if needle in it[0].lower()]
        return entries

    def read_indices(self, table_index: int) -> List[int]:
        cached = self._index_cache.get(table_index)
        if cached is not None:
            return cached

        entry = self.index_table[table_index]
        start = int(self.header["indexData"]["offset"]) + int(entry["offset"])
        size = int(entry["size"])
        num = int(entry["num"])
        fmt = int(entry["format"])
        packed = int(entry["packed"])

        if fmt != 0x01:
            raise ValueError(f"Unsupported index format: {fmt}")

        if packed == 0xC0:
            length = self._u32(start)
            comp = self.mm[start + 4 : start + size]
            raw = zlib_decode_compat(bytes(comp))
            if length > 0 and len(raw) < length:
                raise ValueError("Corrupted compressed index buffer")
            vals = list(struct.unpack_from(f"<{num}h", raw, 0))
            idx = vals[0] if vals else 0
            out: List[int] = []
            for v in vals:
                vv = v + idx
                out.append(vv)
                idx = vv
        else:
            raw = self.mm[start : start + size]
            out = list(struct.unpack_from(f"<{num}H", raw, 0))

        # flip winding
        for i in range(0, len(out) - 2, 3):
            out[i], out[i + 2] = out[i + 2], out[i]

        self._index_cache[table_index] = out
        return out

    def _parse_vertices_by_format(
        self, fmt: str, raw: bytes, num_vertices: int
    ) -> Dict[str, List[float]]:
        # Known formats from SpkFile.php logic.
        patterns = {
            "$/M3D/System/VertexType/TVertex__Position_3f__TexCoord0_2wn__TangentIn01_4ubn__BinormalIn01_4ubn",
            "$/M3D/System/VertexType/TVertex__Position_3f__TexCoord0_2wn__TangentIn01_4ubn__BinormalIn01_4ubn__TexPackedAtlas0_4ubn__TexPackedAtlas1_4ubn__TexPackedAtlas2_4ubn",
            "$/M3D/System/VertexType/TVertex__Position_3f__NormalIn01_4ubn__TexCoord0_2wn__TexPackedAtlas0_4ubn__TexPackedAtlas1_4ubn__TexPackedAtlas2_4ubn",
            "$/M3D/System/VertexType/TVertex__Position_3f__NormalIn01_4ubn__TexCoord0_2wn__TexPackedAtlas0_4ubn",
            "$/M3D/System/VertexType/TVertex__Position_3f__BlW_4ubn__BlIdx_4ub__TexCoord0_2wn__TangentIn01_4ubn__BinormalIn01_4ubn",
            "$/M3D/System/VertexType/TVertex__Position_3f__BlW_4ubn__BlIdx_4ub__TexCoord0_2wn__TangentIn01_4ubn__BinormalAndChenilleIndexIn01_4ubn",
            "$/M3D/System/VertexType/TVertex__Position_3f__NormalIn01_4ubn__BlW_4ubn__BlIdx_4ub__TexCoord0_2wn",
        }
        if fmt not in patterns:
            raise ValueError(f"Unsupported vertex format: {fmt}")

        pos = 0
        xyz: List[float] = []
        uv: List[float] = []
        bone_idx: List[int] = []
        bone_w: List[float] = []

        def rf32() -> float:
            nonlocal pos
            v = struct.unpack_from("<f", raw, pos)[0]
            pos += 4
            return v / 100.0

        def ru16_uv() -> float:
            nonlocal pos
            v = struct.unpack_from("<H", raw, pos)[0]
            pos += 2
            return v / 8192.0

        def skip(n: int) -> None:
            nonlocal pos
            pos += n

        def ru8() -> int:
            nonlocal pos
            v = raw[pos]
            pos += 1
            return int(v)

        for _ in range(num_vertices):
            xyz.extend([rf32(), rf32(), rf32()])

            if fmt == "$/M3D/System/VertexType/TVertex__Position_3f__TexCoord0_2wn__TangentIn01_4ubn__BinormalIn01_4ubn":
                uv.extend([ru16_uv(), ru16_uv()])
                # Tangent + Binormal (4 + 4 bytes).
                skip(0x08)
            elif fmt == "$/M3D/System/VertexType/TVertex__Position_3f__TexCoord0_2wn__TangentIn01_4ubn__BinormalIn01_4ubn__TexPackedAtlas0_4ubn__TexPackedAtlas1_4ubn__TexPackedAtlas2_4ubn":
                uv.extend([ru16_uv(), ru16_uv()])
                skip(0x14)
            elif fmt == "$/M3D/System/VertexType/TVertex__Position_3f__NormalIn01_4ubn__TexCoord0_2wn__TexPackedAtlas0_4ubn__TexPackedAtlas1_4ubn__TexPackedAtlas2_4ubn":
                skip(0x04)
                uv.extend([ru16_uv(), ru16_uv()])
                skip(0x0C)
            elif fmt == "$/M3D/System/VertexType/TVertex__Position_3f__NormalIn01_4ubn__TexCoord0_2wn__TexPackedAtlas0_4ubn":
                skip(0x04)
                uv.extend([ru16_uv(), ru16_uv()])
                skip(0x04)
            elif fmt in {
                "$/M3D/System/VertexType/TVertex__Position_3f__BlW_4ubn__BlIdx_4ub__TexCoord0_2wn__TangentIn01_4ubn__BinormalIn01_4ubn",
                "$/M3D/System/VertexType/TVertex__Position_3f__BlW_4ubn__BlIdx_4ub__TexCoord0_2wn__TangentIn01_4ubn__BinormalAndChenilleIndexIn01_4ubn",
            }:
                w = [ru8(), ru8(), ru8(), ru8()]
                bi = [ru8(), ru8(), ru8(), ru8()]
                ws = [float(x) / 255.0 for x in w]
                total = sum(ws)
                if total > 0.0:
                    ws = [x / total for x in ws]
                bone_w.extend(ws)
                bone_idx.extend(bi)
                uv.extend([ru16_uv(), ru16_uv()])
                skip(0x08)
            elif fmt == "$/M3D/System/VertexType/TVertex__Position_3f__NormalIn01_4ubn__BlW_4ubn__BlIdx_4ub__TexCoord0_2wn":
                skip(0x04)
                w = [ru8(), ru8(), ru8(), ru8()]
                bi = [ru8(), ru8(), ru8(), ru8()]
                ws = [float(x) / 255.0 for x in w]
                total = sum(ws)
                if total > 0.0:
                    ws = [x / total for x in ws]
                bone_w.extend(ws)
                bone_idx.extend(bi)
                uv.extend([ru16_uv(), ru16_uv()])
            else:
                raise ValueError(f"Unhandled vertex format: {fmt}")

        out: Dict[str, List[float]] = {"xyz": xyz, "uv": uv}
        if bone_idx:
            out["bone_idx"] = [float(x) for x in bone_idx]
            out["bone_w"] = bone_w
        return out

    def read_vertices(self, table_index: int) -> Dict[str, List[float]]:
        cached = self._vertex_cache.get(table_index)
        if cached is not None:
            return cached

        entry = self.vertex_table[table_index]
        start = int(self.header["vertexData"]["offset"]) + int(entry["offset"])
        size = int(entry["size"])
        num = int(entry["num"])
        fmt_index = int(entry["format"])
        packed = int(entry["packed"])

        if fmt_index < 0 or fmt_index >= len(self.vertex_formats):
            raise ValueError(f"Vertex format index out of range: {fmt_index}")
        fmt = self.vertex_formats[fmt_index]

        if packed == 0xC0:
            magic = self.mm[start : start + 4]
            if magic != b"VBUF":
                raise ValueError(f"Unknown packed vertex buffer magic: {magic!r}")
            pack_type = self._u16(start + 10)
            raw_size = self._u32(start + 12)
            comp = bytes(self.mm[start + 16 : start + size])
            if pack_type == 0x0100:
                raw = zlib_decode_compat(comp)
            elif pack_type == 0x0200:
                raw = zstandard.ZstdDecompressor().decompress(comp)
            else:
                raise ValueError(f"Unsupported vertex compression type: 0x{pack_type:04X}")
            if raw_size > 0 and len(raw) < raw_size:
                raise ValueError("Corrupted packed vertex buffer")
        else:
            raw = bytes(self.mm[start : start + size])

        verts = self._parse_vertices_by_format(fmt, raw, num)
        self._vertex_cache[table_index] = verts
        return verts

    def get_model_geometry(self, asset_path: str) -> Dict[str, Any]:
        key = normalize_asset_path(asset_path)
        entry = self.fat.get(key)
        if not entry:
            raise KeyError(f"Model not found in FAT: {asset_path}")

        mesh_index = int(entry["meshIndex"])
        if mesh_index < 0 or mesh_index >= len(self.meshes):
            raise ValueError(f"meshIndex out of range: {mesh_index}")
        mesh = self.meshes[mesh_index]

        parts: List[Dict[str, Any]] = []
        skipped_drawcalls: List[Dict[str, Any]] = []
        dc_start = int(mesh["drawCall"])
        dc_num = int(mesh["num"])

        for i in range(dc_num):
            dc_idx = dc_start + i
            if dc_idx < 0 or dc_idx >= len(self.draw_calls):
                skipped_drawcalls.append(
                    {"drawCallIndex": dc_idx, "reason": "drawCall_out_of_range"}
                )
                continue

            dc = self.draw_calls[dc_idx]
            index_idx = int(dc.get("index", -1))
            vertex_idx = int(dc.get("vertex", -1))
            material_idx = int(dc.get("material", -1))

            # Some SPK packs (e.g. decor packs) can contain placeholder draw calls
            # with sentinel indices; skip them instead of aborting whole asset import.
            if index_idx in {0xFFFF, -1} or vertex_idx in {0xFFFF, -1}:
                skipped_drawcalls.append(
                    {
                        "drawCallIndex": dc_idx,
                        "index": index_idx,
                        "vertex": vertex_idx,
                        "material": material_idx,
                        "reason": "drawCall_sentinel_indices",
                    }
                )
                continue
            if index_idx < 0 or index_idx >= len(self.index_table):
                skipped_drawcalls.append(
                    {
                        "drawCallIndex": dc_idx,
                        "index": index_idx,
                        "vertex": vertex_idx,
                        "material": material_idx,
                        "reason": "indexTable_out_of_range",
                    }
                )
                continue
            if vertex_idx < 0 or vertex_idx >= len(self.vertex_table):
                skipped_drawcalls.append(
                    {
                        "drawCallIndex": dc_idx,
                        "index": index_idx,
                        "vertex": vertex_idx,
                        "material": material_idx,
                        "reason": "vertexTable_out_of_range",
                    }
                )
                continue

            try:
                indices = self.read_indices(index_idx)
                vertices = self.read_vertices(vertex_idx)
            except Exception as exc:
                skipped_drawcalls.append(
                    {
                        "drawCallIndex": dc_idx,
                        "index": index_idx,
                        "vertex": vertex_idx,
                        "material": material_idx,
                        "reason": f"decode_error:{exc}",
                    }
                )
                continue

            if not indices:
                skipped_drawcalls.append(
                    {
                        "drawCallIndex": dc_idx,
                        "index": index_idx,
                        "vertex": vertex_idx,
                        "material": material_idx,
                        "reason": "empty_indices",
                    }
                )
                continue
            xyz = vertices.get("xyz", []) if isinstance(vertices, dict) else []
            if not xyz:
                skipped_drawcalls.append(
                    {
                        "drawCallIndex": dc_idx,
                        "index": index_idx,
                        "vertex": vertex_idx,
                        "material": material_idx,
                        "reason": "empty_vertices",
                    }
                )
                continue

            parts.append(
                {
                    "drawCallIndex": dc_idx,
                    "material": material_idx,
                    "indices": indices,
                    "vertices": vertices,
                }
            )

        if not parts:
            first_reason = skipped_drawcalls[0]["reason"] if skipped_drawcalls else "no_drawcalls"
            raise ValueError(
                f"No valid draw calls for asset {key} "
                f"(meshIndex={mesh_index}, drawCall={dc_start}, num={dc_num}, reason={first_reason})"
            )

        return {
            "asset": key,
            "meta": entry,
            "parts": parts,
            "skippedDrawCalls": skipped_drawcalls,
        }


def resolve_atlas_assets_root(path: Path) -> Path:
    p = path
    if p.name.lower() == "atlas":
        p = p / "Assets"
    elif p.name.lower() != "assets" and (p / "Assets").exists():
        p = p / "Assets"
    return p


def _is_numbered_zz_dat_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low.endswith(".dat"):
        return False
    stem = Path(low).stem
    if stem == "zz":
        return True
    if not stem.startswith("zz_"):
        return False
    suffix = stem[3:]
    return bool(suffix) and suffix.isdigit()


def _zz_dat_sort_key(path: Path) -> int:
    stem = path.stem.lower()
    if stem == "zz":
        return -1
    if stem.startswith("zz_"):
        suffix = stem[3:]
        if suffix.isdigit():
            return int(suffix)
    # Non-ZZ packages (e.g. *_Assets.dat) should have lower precedence than ZZ packs.
    return -2


def find_warno_zz_dat_files(warno_root: Path) -> List[Path]:
    root = Path(warno_root)
    if not root.exists() or not root.is_dir():
        return []
    scan_root = root / "Data"
    if not scan_root.exists() or not scan_root.is_dir():
        scan_root = root
    out: List[Path] = []
    for p in scan_root.rglob("ZZ*.dat"):
        try:
            if p.is_file() and _is_numbered_zz_dat_name(p.name):
                out.append(p)
        except Exception:
            continue
    out.sort(key=lambda p: (_zz_dat_sort_key(p), str(p).lower()))
    return out


def find_warno_texture_dat_files(warno_root: Path) -> List[Path]:
    root = Path(warno_root)
    if not root.exists() or not root.is_dir():
        return []
    scan_root = root / "Data"
    if not scan_root.exists() or not scan_root.is_dir():
        scan_root = root

    ordered: List[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        try:
            if not path.exists() or not path.is_file():
                return
            key = str(path.resolve()).lower()
        except Exception:
            return
        if key in seen:
            return
        seen.add(key)
        ordered.append(path)

    for p in find_warno_zz_dat_files(root):
        _push(p)

    assets_dats: List[Path] = []
    for p in scan_root.rglob("*_Assets.dat"):
        try:
            if p.is_file() and p.name.lower().endswith("_assets.dat"):
                assets_dats.append(p)
        except Exception:
            continue
    assets_dats.sort(key=lambda p: str(p).lower())
    for p in assets_dats:
        _push(p)

    return ordered


def _read_u32_le(buf: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", buf, offset)[0])


def _read_cstring_in_dict(fh, dict_end: int) -> str:
    parts: List[bytes] = []
    while fh.tell() < dict_end:
        b = fh.read(1)
        if not b:
            raise EOFError("Unexpected EOF while reading EDAT dictionary string")
        if b == b"\x00":
            return b"".join(parts).decode("ascii", errors="ignore")
        parts.append(b)
    raise ValueError("Unterminated string in EDAT dictionary")


def _parse_zz_dat_header(dat_path: Path) -> Dict[str, int]:
    with dat_path.open("rb") as fh:
        head = fh.read(64)
    if len(head) < 41:
        raise ValueError(f"EDAT header too short: {dat_path}")
    if head[:4] != b"edat":
        raise ValueError(f"Invalid EDAT magic in {dat_path}: {head[:4]!r}")
    version = _read_u32_le(head, 4)
    if version not in {1, 2}:
        raise ValueError(f"Unsupported EDAT version {version} in {dat_path}")
    dict_offset = _read_u32_le(head, 25)
    dict_length = _read_u32_le(head, 29)
    file_offset = _read_u32_le(head, 33)
    file_length = _read_u32_le(head, 37)
    return {
        "version": int(version),
        "dict_offset": int(dict_offset),
        "dict_length": int(dict_length),
        "file_offset": int(file_offset),
        "file_length": int(file_length),
    }


def _parse_zz_dat_v2_entries(dat_path: Path, header: Dict[str, int], legacy_padding: bool) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    dirs: List[str] = []
    endings: List[int] = []
    dict_offset = int(header["dict_offset"])
    dict_length = int(header["dict_length"])
    file_offset = int(header["file_offset"])
    dict_end = dict_offset + dict_length
    package_len = dat_path.stat().st_size

    with dat_path.open("rb") as fh:
        fh.seek(dict_offset)
        while fh.tell() < dict_end:
            block_start = fh.tell()
            raw = fh.read(4)
            if len(raw) != 4:
                raise EOFError("Unexpected EOF while reading V2 fileGroupId")
            file_group_id = int(struct.unpack("<i", raw)[0])

            if file_group_id == 0:
                file_entry_size = _read_u32_le(fh.read(4), 0)
                raw_offset = int(struct.unpack("<q", fh.read(8))[0])
                raw_size = int(struct.unpack("<q", fh.read(8))[0])
                checksum = fh.read(16)
                if len(checksum) != 16:
                    raise EOFError("Unexpected EOF while reading V2 checksum")
                name = _read_cstring_in_dict(fh, dict_end)
                path = "".join(dirs) + name

                if legacy_padding and len(name) % 2 == 0:
                    if fh.tell() >= dict_end:
                        raise ValueError("Expected legacy V2 padding byte but reached dictionary end")
                    fh.seek(1, os.SEEK_CUR)

                abs_offset = int(file_offset) + int(raw_offset)
                if raw_offset < 0 or raw_size < 0:
                    raise ValueError(f"Invalid V2 file entry (offset/size < 0) in {dat_path}: {path}")
                if abs_offset < 0 or abs_offset > package_len:
                    raise ValueError(f"Invalid V2 file offset in {dat_path}: {path}")
                if raw_size > package_len - abs_offset:
                    raise ValueError(f"Invalid V2 file size in {dat_path}: {path}")
                entries.append(
                    {
                        "path": normalize_asset_path(path),
                        "offset": int(raw_offset),
                        "size": int(raw_size),
                        "abs_offset": int(abs_offset),
                        "file_entry_size": int(file_entry_size),
                    }
                )

                while endings and fh.tell() == endings[-1]:
                    dirs.pop()
                    endings.pop()
            elif file_group_id > 0:
                file_entry_size = _read_u32_le(fh.read(4), 0)
                if file_entry_size != 0:
                    endings.append(int(block_start + file_entry_size))
                elif endings:
                    endings.append(endings[-1])
                name = _read_cstring_in_dict(fh, dict_end)
                if legacy_padding and len(name) % 2 == 0:
                    if fh.tell() >= dict_end:
                        raise ValueError("Expected legacy V2 dir padding byte but reached dictionary end")
                    fh.seek(1, os.SEEK_CUR)
                dirs.append(name)
            else:
                raise ValueError(f"Invalid V2 fileGroupId {file_group_id} in {dat_path}")
    return entries


def _parse_zz_dat_v1_entries(dat_path: Path, header: Dict[str, int]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    dirs: List[str] = []
    endings: List[int] = []
    dict_offset = int(header["dict_offset"])
    dict_length = int(header["dict_length"])
    file_offset = int(header["file_offset"])
    dict_end = dict_offset + dict_length
    package_len = dat_path.stat().st_size

    with dat_path.open("rb") as fh:
        fh.seek(dict_offset)
        while fh.tell() < dict_end:
            block_start = fh.tell()
            raw = fh.read(4)
            if len(raw) != 4:
                raise EOFError("Unexpected EOF while reading V1 fileGroupId")
            file_group_id = int(struct.unpack("<i", raw)[0])

            if file_group_id == 0:
                file_entry_size = _read_u32_le(fh.read(4), 0)
                raw_offset = _read_u32_le(fh.read(4), 0)
                raw_size = _read_u32_le(fh.read(4), 0)
                fh.seek(1, os.SEEK_CUR)
                name = _read_cstring_in_dict(fh, dict_end)
                path = "".join(dirs) + name
                if (len(name) + 1) % 2 == 0:
                    if fh.tell() < dict_end:
                        fh.seek(1, os.SEEK_CUR)

                abs_offset = int(file_offset) + int(raw_offset)
                if abs_offset < 0 or abs_offset > package_len:
                    continue
                if raw_size > package_len - abs_offset:
                    continue
                entries.append(
                    {
                        "path": normalize_asset_path(path),
                        "offset": int(raw_offset),
                        "size": int(raw_size),
                        "abs_offset": int(abs_offset),
                        "file_entry_size": int(file_entry_size),
                    }
                )
                while endings and fh.tell() == endings[-1]:
                    dirs.pop()
                    endings.pop()
            elif file_group_id > 0:
                file_entry_size = _read_u32_le(fh.read(4), 0)
                if file_entry_size != 0:
                    endings.append(int(block_start + file_entry_size))
                elif endings:
                    endings.append(endings[-1])
                name = _read_cstring_in_dict(fh, dict_end)
                if (len(name) + 1) % 2 == 1 and fh.tell() < dict_end:
                    fh.seek(1, os.SEEK_CUR)
                dirs.append(name)
            else:
                break
    return entries


def _scan_zz_dat_entries(dat_path: Path) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    header = _parse_zz_dat_header(dat_path)
    version = int(header["version"])
    if version == 1:
        return header, _parse_zz_dat_v1_entries(dat_path, header)

    try:
        return header, _parse_zz_dat_v2_entries(dat_path, header, legacy_padding=True)
    except Exception:
        return header, _parse_zz_dat_v2_entries(dat_path, header, legacy_padding=False)


class ZZDatResolver:
    def __init__(self, dat_files: Sequence[Path]):
        self.dat_files = [Path(p) for p in dat_files if Path(p).exists() and Path(p).is_file()]
        self._index_ready = False
        self._assets: Dict[str, Dict[str, Any]] = {}
        self._basename_to_keys: Dict[str, List[str]] = {}
        self._header_by_dat: Dict[str, Dict[str, int]] = {}
        self._generic_parent_hint: Dict[str, str] = {}

    def _build_index(self) -> None:
        if self._index_ready:
            return
        ordered = sorted(self.dat_files, key=lambda p: (_zz_dat_sort_key(p), str(p).lower()), reverse=True)
        for dat in ordered:
            dat_key = str(dat.resolve())
            try:
                header, entries = _scan_zz_dat_entries(dat)
            except Exception:
                continue
            self._header_by_dat[dat_key] = header
            for entry in entries:
                path_raw = str(entry.get("path", "")).strip()
                if not path_raw:
                    continue
                norm = normalize_asset_path(path_raw).lower()
                if norm and norm not in self._assets:
                    payload = dict(entry)
                    payload["dat_path"] = dat
                    self._assets[norm] = payload
                    base = Path(norm).name.lower()
                    if base:
                        self._basename_to_keys.setdefault(base, []).append(norm)
        self._index_ready = True

    def all_asset_keys(self) -> List[str]:
        self._build_index()
        return list(self._assets.keys())

    def _find_exact_norm(self, norm: str) -> Dict[str, Any] | None:
        if not norm:
            return None
        hit = self._assets.get(norm)
        if hit is not None:
            return hit

        # Atlas refs can come as Assets/... or PC/Atlas/Assets/...
        if norm.startswith("assets/"):
            alt = normalize_asset_path(f"PC/Atlas/{norm}").lower()
            hit = self._assets.get(alt)
            if hit is not None:
                return hit
        if norm.startswith("pc/atlas/assets/"):
            alt = normalize_asset_path(norm[len("pc/atlas/") :]).lower()
            hit = self._assets.get(alt)
            if hit is not None:
                return hit
        return None

    def find_exact(self, asset_path: str) -> Dict[str, Any] | None:
        self._build_index()
        norm = normalize_asset_path(str(asset_path or "")).lower()
        return self._find_exact_norm(norm)

    def find(self, asset_path: str) -> Dict[str, Any] | None:
        self._build_index()
        norm = normalize_asset_path(str(asset_path or "")).lower()
        hit = self._find_exact_norm(norm)
        if hit is not None:
            return hit

        base = Path(norm).name.lower()
        if base:
            generic_marks = (
                "tsccolor_diffusetexturenoalpha",
                "tscnm_normaltexture",
                "tscorm_combinedrmtexture",
                "tscorm_combinedormtexture",
                "tsccoloralpha_combineddatexture",
            )
            is_generic_base = any(mark in base for mark in generic_marks)
            cands = self._basename_to_keys.get(base, [])
            req_low_for_score = norm
            # Common atlas refs often request .png while payload is only .tgv (or reverse).
            if not cands:
                if base.endswith(".png"):
                    cands = self._basename_to_keys.get(base[:-4] + ".tgv", [])
                    if cands:
                        req_low_for_score = norm[:-4] + ".tgv"
                elif base.endswith(".tgv"):
                    cands = self._basename_to_keys.get(base[:-4] + ".png", [])
                    if cands:
                        req_low_for_score = norm[:-4] + ".png"

            # ORM alias support: some assets use CombinedRM in refs while payload is CombinedORM (or reverse).
            orm_alt_bases: List[str] = []
            if "combinedrmtexture" in base:
                orm_alt_bases.append(base.replace("combinedrmtexture", "combinedormtexture"))
            if "combinedormtexture" in base:
                orm_alt_bases.append(base.replace("combinedormtexture", "combinedrmtexture"))
            if orm_alt_bases:
                merged = list(cands)
                for alt_base in orm_alt_bases:
                    alt = self._basename_to_keys.get(alt_base, [])
                    if not alt and alt_base.endswith(".png"):
                        alt = self._basename_to_keys.get(alt_base[:-4] + ".tgv", [])
                    elif not alt and alt_base.endswith(".tgv"):
                        alt = self._basename_to_keys.get(alt_base[:-4] + ".png", [])
                    if alt:
                        merged.extend(alt)
                if merged:
                    # Preserve order while removing duplicates.
                    cands = list(dict.fromkeys(merged))
            req_dir_key = normalize_asset_path(str(PurePosixPath(req_low_for_score).parent)).lower()
            hint_parent = self._generic_parent_hint.get(req_dir_key, "") if is_generic_base else ""
            if len(cands) == 1:
                if is_generic_base and req_dir_key:
                    only_key = cands[0]
                    only_parent = normalize_asset_path(str(PurePosixPath(only_key).parent)).lower()
                    req_is_decors = "/decors/" in req_low_for_score
                    if only_parent and (not req_is_decors or "/decors/" in only_parent):
                        self._generic_parent_hint[req_dir_key] = only_parent
                return self._assets.get(cands[0])
            if cands:
                req_low = req_low_for_score
                req_parent = normalize_asset_path(str(PurePosixPath(req_low).parent)).lower()
                req_parts = [p for p in PurePosixPath(req_parent).parts if p]
                generic = {"pc", "atlas", "assets", "3d", "2d", "output", "mods", "moddata", "base"}
                req_tokens = [p.lower() for p in req_parts if p.lower() not in generic]
                req_base = Path(req_low).name.lower()

                def _score_candidate(key: str) -> float:
                    key_low = str(key).lower()
                    score = float(shared_suffix_score(key_low, req_low) * 100)
                    key_parent = normalize_asset_path(str(PurePosixPath(key_low).parent)).lower()
                    if req_parent and key_parent.endswith(req_parent):
                        score += 160.0
                    for tok in req_tokens:
                        if tok and f"/{tok}/" in f"/{key_low}/":
                            score += 12.0
                    if "/decors/" in req_low and "/decors/" not in key_low:
                        score -= 45.0
                    if "/units/" in req_low and "/units/" not in key_low:
                        score -= 45.0
                    if "/decors/" in req_low and "/units/" in key_low:
                        score -= 20.0
                    if "/fx/" in key_low and "/fx/" not in req_low:
                        score -= 35.0
                    if "/units_tests/" in key_low:
                        score -= 90.0
                    if "/units_tests_autos/" in key_low:
                        score -= 110.0
                    if "/tests/" in key_low:
                        score -= 18.0
                    if "/editor/" in key_low:
                        score -= 18.0

                    if hint_parent:
                        if key_parent == hint_parent:
                            score += 220.0
                        elif key_parent.startswith(hint_parent + "/"):
                            score += 48.0

                    # Generic TSC* atlas refs are often ambiguous across many folders.
                    # Prefer candidate folders that provide a coherent D/NM/ORM set.
                    if "tsc" in req_base and "texture01" in req_base:
                        has_diff = (
                            f"{key_parent}/tsccolor_diffusetexturenoalpha01.tgv" in self._assets
                            or f"{key_parent}/tsccolor_diffusetexturenoalpha01.png" in self._assets
                        )
                        has_nm = (
                            f"{key_parent}/tscnm_normaltexture01.tgv" in self._assets
                            or f"{key_parent}/tscnm_normaltexture01.png" in self._assets
                        )
                        has_orm = (
                            f"{key_parent}/tscorm_combinedrmtexture01.tgv" in self._assets
                            or f"{key_parent}/tscorm_combinedrmtexture01.png" in self._assets
                            or f"{key_parent}/tscorm_combinedormtexture01.tgv" in self._assets
                            or f"{key_parent}/tscorm_combinedormtexture01.png" in self._assets
                        )
                        if "tsccolor_diffusetexturenoalpha" in req_base:
                            if has_nm:
                                score += 18.0
                            if has_orm:
                                score += 18.0
                        elif "tscnm_normaltexture" in req_base:
                            if has_diff:
                                score += 18.0
                            if has_orm:
                                score += 14.0
                        elif "tscorm_combinedrmtexture" in req_base or "tscorm_combinedormtexture" in req_base:
                            if has_diff:
                                score += 26.0
                            if has_nm:
                                score += 16.0

                    try:
                        size_bytes = int(self._assets.get(key_low, {}).get("size", 0) or 0)
                    except Exception:
                        size_bytes = 0
                    # Avoid tiny placeholder atlas textures when resolving ambiguous generic names.
                    if size_bytes > 0:
                        if size_bytes <= 256:
                            score -= 120.0
                        elif size_bytes <= 1024:
                            score -= 60.0
                        elif size_bytes <= 4096:
                            score -= 24.0
                        elif size_bytes >= 1_000_000:
                            score += 14.0
                        elif size_bytes >= 200_000:
                            score += 8.0

                    score -= min(20.0, float(len(key_low)) * 0.01)
                    return score

                scored = sorted(((_score_candidate(k), k) for k in cands), reverse=True)
                req_is_decors = "/decors/" in req_low
                if is_generic_base and req_is_decors:
                    decors_scored_strict = [pair for pair in scored if "/decors/" in pair[1]]
                    if decors_scored_strict:
                        if req_dir_key:
                            top_parent = normalize_asset_path(str(PurePosixPath(decors_scored_strict[0][1]).parent)).lower()
                            if top_parent:
                                self._generic_parent_hint[req_dir_key] = top_parent
                        return self._assets.get(decors_scored_strict[0][1])
                    # Guardrail: for decors refs, never fallback to units-style generic TSC textures.
                    return None
                if scored and scored[0][0] >= 90.0:
                    if is_generic_base and req_dir_key:
                        top_parent = normalize_asset_path(str(PurePosixPath(scored[0][1]).parent)).lower()
                        req_is_decors = "/decors/" in req_low
                        if top_parent and (not req_is_decors or "/decors/" in top_parent):
                            self._generic_parent_hint[req_dir_key] = top_parent
                    return self._assets.get(scored[0][1])
                if scored:
                    req_is_decors = "/decors/" in req_low
                    # Soft fallback for generic shared texture names where strict path
                    # similarity cannot work (e.g. decors refs mapped to atlas templates).
                    is_generic = is_generic_base
                    if is_generic:
                        # Prefer non-test candidates for generic TSC* texture names.
                        non_test_scored = [
                            pair
                            for pair in scored
                            if "/units_tests/" not in pair[1] and "/units_tests_autos/" not in pair[1]
                        ]
                        if req_is_decors:
                            decors_scored = [pair for pair in non_test_scored if "/decors/" in pair[1]]
                            if decors_scored:
                                if req_dir_key:
                                    top_parent = normalize_asset_path(str(PurePosixPath(decors_scored[0][1]).parent)).lower()
                                    if top_parent:
                                        self._generic_parent_hint[req_dir_key] = top_parent
                                return self._assets.get(decors_scored[0][1])
                            # Guardrail: for decors refs, never fallback to units-style generic TSC textures.
                            return None
                        if non_test_scored and non_test_scored[0][0] >= 35.0:
                            if req_dir_key:
                                top_parent = normalize_asset_path(str(PurePosixPath(non_test_scored[0][1]).parent)).lower()
                                if top_parent:
                                    self._generic_parent_hint[req_dir_key] = top_parent
                            return self._assets.get(non_test_scored[0][1])
                        if scored[0][0] >= 40.0:
                            if req_dir_key:
                                top_parent = normalize_asset_path(str(PurePosixPath(scored[0][1]).parent)).lower()
                                if top_parent:
                                    self._generic_parent_hint[req_dir_key] = top_parent
                            return self._assets.get(scored[0][1])
                    # Non-generic fallback: still allow strong-but-not-perfect match.
                    if scored[0][0] >= 70.0:
                        return self._assets.get(scored[0][1])
        return None

    def find_first_by_suffix(self, suffixes: Sequence[str], must_contain: Sequence[str] | None = None) -> Dict[str, Any] | None:
        self._build_index()
        clean_suffixes = [str(s).strip().lower().lstrip("/") for s in suffixes if str(s).strip()]
        if not clean_suffixes:
            return None
        contains = [str(s).strip().lower() for s in (must_contain or []) if str(s).strip()]

        for key, payload in self._assets.items():
            if contains and not all(tok in key for tok in contains):
                continue
            if any(key.endswith(suf) for suf in clean_suffixes):
                return payload
        return None

    def find_all_by_suffix(
        self,
        suffixes: Sequence[str],
        must_contain: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        self._build_index()
        clean_suffixes = [str(s).strip().lower().lstrip("/") for s in suffixes if str(s).strip()]
        if not clean_suffixes:
            return []
        contains = [str(s).strip().lower() for s in (must_contain or []) if str(s).strip()]

        out: List[Dict[str, Any]] = []
        for key in sorted(self._assets.keys()):
            if contains and not all(tok in key for tok in contains):
                continue
            if any(key.endswith(suf) for suf in clean_suffixes):
                out.append(self._assets[key])
        return out

    def extract_hit_to_runtime(self, hit: Dict[str, Any], runtime_root: Path) -> Path:
        dat_path = Path(hit["dat_path"])
        rel = normalize_asset_path(str(hit.get("path", "")))
        if not rel:
            raise ValueError("Cannot extract ZZ entry with empty path")
        out_path = Path(runtime_root) / Path(*rel.split("/"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        need_write = True
        size = int(hit.get("size", 0))
        if out_path.exists() and out_path.is_file():
            try:
                if size > 0 and out_path.stat().st_size == size:
                    need_write = False
            except Exception:
                pass
        if need_write:
            abs_offset = int(hit["abs_offset"])
            with dat_path.open("rb") as fh:
                fh.seek(abs_offset)
                data = fh.read(size)
            if len(data) != size:
                raise IOError(f"Failed to read full ZZ entry: {rel} from {dat_path}")
            out_path.write_bytes(data)
        return out_path

    def extract_asset_to_runtime(self, asset_path: str, runtime_root: Path, exact_only: bool = False) -> Path | None:
        hit = self.find_exact(asset_path) if exact_only else self.find(asset_path)
        if hit is None:
            return None
        return self.extract_hit_to_runtime(hit, runtime_root)


_ZZ_RESOLVER_CACHE: Dict[Tuple[Tuple[str, int, int], ...], ZZDatResolver] = {}
_ATLAS_JSON_CACHE: Dict[Tuple[str, str, int, int, str], Dict[str, Any]] = {}
_GFX_JSON_CACHE: Dict[Tuple[str, str, int, int, str], Dict[str, Any]] = {}


def get_zz_runtime_resolver(warno_root: Path) -> ZZDatResolver:
    dat_files = find_warno_texture_dat_files(warno_root)
    if not dat_files:
        raise FileNotFoundError(f"No texture DAT packages found under WARNO folder: {warno_root}")

    key_rows: List[Tuple[str, int, int]] = []
    for p in dat_files:
        st = p.stat()
        key_rows.append((str(p.resolve()).lower(), int(st.st_mtime_ns), int(st.st_size)))
    key = tuple(key_rows)
    cached = _ZZ_RESOLVER_CACHE.get(key)
    if cached is not None:
        return cached
    resolver = ZZDatResolver(dat_files)
    _ZZ_RESOLVER_CACHE.clear()
    _ZZ_RESOLVER_CACHE[key] = resolver
    return resolver


def clear_atlas_json_cache() -> None:
    _ATLAS_JSON_CACHE.clear()


def clear_gfx_json_cache() -> None:
    _GFX_JSON_CACHE.clear()


def _normalize_logical_ref(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low.startswith("combinedtexture:/"):
        raw = raw[len("combinedtexture:/") :]
    if raw.lower().startswith("gamedata:/"):
        raw = raw[len("gamedata:/") :]
    if raw.lower().startswith("gamedata:"):
        raw = raw[len("gamedata:") :]
    raw = raw.lstrip("/")
    raw = normalize_asset_path(raw)
    low = raw.lower()
    if low.startswith("pc/atlas/assets/"):
        return normalize_asset_path("Assets/" + raw[len("PC/Atlas/Assets/") :]).lower()
    return low


def _canonical_channel(value: str) -> str:
    low = str(value or "").strip().lower()
    if not low:
        return "generic"
    if low in {"d", "diff", "albedo", "basecolor", "base_color", "diffuse"}:
        return "diffuse"
    if low in {"nm", "normal", "normalmap"}:
        return "normal"
    if low in {"orm", "occlusionroughnessmetallic", "rma"}:
        return "orm"
    if low in {"r", "rough", "roughness"}:
        return "roughness"
    if low in {"m", "metal", "metallic"}:
        return "metallic"
    if low in {"o", "ao", "occlusion", "ambientocclusion"}:
        return "occlusion"
    if low in {"a", "alpha", "opacity"}:
        return "alpha"
    if low in {"da", "combined_da", "coloralpha"}:
        return "combined_da"
    return low


def _atlas_target_output_rel(target_logical_rel: str, target_basename: str) -> Path:
    logical = _normalize_logical_ref(target_logical_rel)
    if logical.startswith("assets/"):
        rel = Path(*PurePosixPath(logical).parts[1:])
        parent = rel.parent
    else:
        parent = Path()
    stem = str(target_basename or "").strip() or Path(logical).stem or "Texture"
    return parent / f"{stem}.png"


def _build_atlas_json_index(data: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    entries: List[Dict[str, Any]] = []
    textures = data.get("textures", [])
    if not isinstance(textures, list):
        return index, entries

    def add_key(key: str, item: Dict[str, Any]) -> None:
        norm = _normalize_logical_ref(key)
        if not norm:
            return
        bucket = index.setdefault(norm, [])
        if item not in bucket:
            bucket.append(item)

    for tex in textures:
        if not isinstance(tex, dict):
            continue
        source_tgv_rel = _normalize_logical_ref(str(tex.get("source_tgv_rel", "")).strip())
        if not source_tgv_rel:
            continue
        source_role = _canonical_channel(str(tex.get("source_role", "generic")))
        # Originating FBX path of the texture. WARNO's DECORS atlases pack
        # textures from many sibling FBX into a single composite atlas
        # (e.g. HLM_10_L.fbx aggregates 11 building-block FBX). Keeping
        # source_asset_path on every entry lets the importer filter atlas
        # targets down to those actually referenced by SPK materials.
        source_asset_path = str(tex.get("source_asset_path", "")).strip()
        rect = tex.get("crop_rect_px") if isinstance(tex.get("crop_rect_px"), dict) else {}
        crop_rect_px = {
            "x": int(rect.get("x", 0) or 0),
            "y": int(rect.get("y", 0) or 0),
            "w": int(rect.get("w", 0) or 0),
            "h": int(rect.get("h", 0) or 0),
        }
        targets = tex.get("targets") if isinstance(tex.get("targets"), list) else []
        if not targets:
            targets = [{}]
        for tgt in targets:
            if not isinstance(tgt, dict):
                tgt = {}
            logical_rel = str(tgt.get("logical_rel", "")).strip()
            basename = str(tgt.get("basename", "")).strip()
            channel = _canonical_channel(str(tgt.get("channel", "")))
            entry = {
                "source_tgv_rel": source_tgv_rel,
                "source_role": source_role,
                "source_asset_path": source_asset_path,
                "crop_rect_px": dict(crop_rect_px),
                "target_logical_rel": logical_rel,
                "target_basename": basename or Path(_normalize_logical_ref(logical_rel)).stem,
                "target_channel": channel if channel != "generic" else source_role,
            }
            entries.append(entry)
            add_key(logical_rel, entry)
            logical_norm = _normalize_logical_ref(logical_rel)
            if logical_norm.endswith(".png"):
                add_key(logical_norm[:-4] + ".tgv", entry)
            elif logical_norm.endswith(".tgv"):
                add_key(logical_norm[:-4] + ".png", entry)
    return index, entries


def _atlas_target_refs_from_entries(entries: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for entry in entries:
        logical = str(entry.get("target_logical_rel", "")).strip()
        norm = _normalize_logical_ref(logical)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(logical or norm)
    return out


def _atlas_map_signature(path: Path) -> Tuple[str, int, int]:
    p = Path(path)
    st = p.stat()
    return (str(p.resolve()).lower(), int(st.st_mtime_ns), int(st.st_size))


def _python_cmd_candidates_for_wrapper() -> List[List[str]]:
    if sys.executable:
        return [[sys.executable]]
    env_py = os.environ.get("PYTHON", "").strip()
    if env_py:
        return [[env_py]]
    return []


def _resolve_atlas_wrapper(wrapper_path: Path) -> Path:
    p = Path(wrapper_path)
    if p.exists() and p.is_file():
        return p
    alt = Path(__file__).with_name("modding_suite_atlas_export.py")
    if alt.exists() and alt.is_file():
        return alt
    return p


def _resolve_gfx_wrapper(wrapper_path: Path) -> Path:
    p = Path(wrapper_path)
    if p.exists() and p.is_file():
        return p
    alt = Path(__file__).with_name("modding_suite_gfx_export.py")
    if alt.exists() and alt.is_file():
        return alt
    return p


def build_or_load_atlas_texture_map(
    warno_root: Path,
    modding_suite_root: Path,
    asset_path: str,
    cache_dir: Path,
    wrapper_path: Path,
    atlas_cli_path: Path | None = None,
    force_rebuild: bool = False,
    timeout_sec: int = 45,
) -> Dict[str, Any]:
    asset_norm = normalize_asset_path(str(asset_path or "")).strip()
    if not asset_norm:
        raise RuntimeError("Atlas JSON build failed: empty asset path")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    rel = Path(*PurePosixPath(asset_norm).parts).with_suffix(".atlas_map.json")
    out_json = cache_root / rel
    out_json.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _resolve_atlas_wrapper(wrapper_path)
    if not wrapper.exists() or not wrapper.is_file():
        raise RuntimeError(f"Atlas JSON wrapper not found: {wrapper}")

    need_export = force_rebuild or not out_json.exists()
    if need_export:
        temp_out = out_json.with_suffix(".tmp.json")
        cmd_tail = [
            str(wrapper),
            "--warno-root",
            str(warno_root),
            "--modding-suite-root",
            str(modding_suite_root),
            "--asset-path",
            asset_norm,
            "--out-json",
            str(temp_out),
            "--cache-dir",
            str(cache_root),
            "--timeout-sec",
            str(max(5, int(timeout_sec))),
        ]
        if atlas_cli_path is not None and str(atlas_cli_path).strip():
            cmd_tail.extend(["--atlas-cli", str(atlas_cli_path)])
        errors: List[str] = []
        for py_cmd in _python_cmd_candidates_for_wrapper():
            cmd = [*py_cmd, *cmd_tail]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(10, int(timeout_sec) + 10),
                )
            except subprocess.TimeoutExpired:
                errors.append(f"[{' '.join(py_cmd)}] timeout>{int(timeout_sec) + 10}s")
                continue
            if proc.returncode == 0 and temp_out.exists():
                temp_out.replace(out_json)
                errors = []
                break
            msg = (proc.stderr or proc.stdout or "").strip()
            if proc.returncode == 2:
                errors.append(f"[{' '.join(py_cmd)}] rc=2: atlas data has no entries for asset {asset_norm}")
                continue
            errors.append(f"[{' '.join(py_cmd)}] rc={proc.returncode}: {msg}")
        if errors:
            raise RuntimeError(
                "Atlas JSON export failed: " + " | ".join(errors[:2])
            )

    sig = _atlas_map_signature(out_json)
    try:
        data = json.loads(out_json.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Atlas JSON parse failed: {out_json}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Atlas JSON has invalid root object: {out_json}")
    if int(data.get("schema_version", 0) or 0) != 1:
        raise RuntimeError(f"Atlas JSON schema mismatch: expected schema_version=1, got {data.get('schema_version')}")
    if not isinstance(data.get("textures"), list):
        raise RuntimeError("Atlas JSON schema mismatch: textures[] is missing")

    cli_version = str(data.get("cli_version", "") or "").strip() or "unknown"
    cache_key = (asset_norm.lower(), sig[0], sig[1], sig[2], cli_version)
    cached = _ATLAS_JSON_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    index, entries = _build_atlas_json_index(data)
    atlas_targets = _atlas_target_refs_from_entries(entries)
    result = {
        "asset_path": asset_norm,
        "atlas_map_path": str(out_json),
        "atlas_map_entries": int(len(entries)),
        "atlas_map_targets": list(atlas_targets),
        "atlas_map_index": index,
        "atlas_cli_version": cli_version,
        "atlas_signature": {
            "path": sig[0],
            "mtime_ns": int(sig[1]),
            "size": int(sig[2]),
        },
    }
    _ATLAS_JSON_CACHE.clear()
    _ATLAS_JSON_CACHE[cache_key] = dict(result)
    return dict(result)


def build_or_load_gfx_manifest(
    warno_root: Path,
    modding_suite_root: Path,
    asset_path: str,
    cache_dir: Path,
    wrapper_path: Path,
    gfx_cli_path: Path | None = None,
    force_rebuild: bool = False,
    timeout_sec: int = 180,
) -> Dict[str, Any]:
    asset_norm = normalize_asset_path(str(asset_path or "")).strip()
    if not asset_norm:
        raise RuntimeError("GFX manifest build failed: empty asset path")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    rel = Path(*PurePosixPath(asset_norm).parts).with_suffix(".gfx_manifest.json")
    out_json = cache_root / rel
    out_json.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _resolve_gfx_wrapper(wrapper_path)
    if not wrapper.exists() or not wrapper.is_file():
        raise RuntimeError(f"GFX manifest wrapper not found: {wrapper}")

    need_export = force_rebuild or not out_json.exists()
    if need_export:
        temp_out = out_json.with_suffix(".tmp.json")
        cmd_tail = [
            str(wrapper),
            "--warno-root",
            str(warno_root),
            "--modding-suite-root",
            str(modding_suite_root),
            "--asset-path",
            asset_norm,
            "--out-json",
            str(temp_out),
            "--cache-dir",
            str(cache_root),
            "--timeout-sec",
            str(max(5, int(timeout_sec))),
        ]
        if gfx_cli_path is not None and str(gfx_cli_path).strip():
            cmd_tail.extend(["--gfx-cli", str(gfx_cli_path)])
        errors: List[str] = []
        for py_cmd in _python_cmd_candidates_for_wrapper():
            cmd = [*py_cmd, *cmd_tail]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(10, int(timeout_sec) + 10),
                )
            except subprocess.TimeoutExpired:
                errors.append(f"[{' '.join(py_cmd)}] timeout>{int(timeout_sec) + 10}s")
                continue
            if proc.returncode == 0 and temp_out.exists():
                temp_out.replace(out_json)
                errors = []
                break
            msg = (proc.stderr or proc.stdout or "").strip()
            errors.append(f"[{' '.join(py_cmd)}] rc={proc.returncode}: {msg}")
        if errors:
            raise RuntimeError("GFX manifest export failed: " + " | ".join(errors[:2]))

    sig = _atlas_map_signature(out_json)
    try:
        data = json.loads(out_json.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"GFX manifest parse failed: {out_json}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"GFX manifest has invalid root object: {out_json}")
    schema_version = int(data.get("schema_version", 0) or 0)
    if schema_version not in {1, 2}:
        raise RuntimeError(
            f"GFX manifest schema mismatch: expected schema_version in {{1,2}}, got {data.get('schema_version')}"
        )
    for req_key in (
        "asset_path",
        "source_files",
        "matched_units",
        "reference_meshes",
        "operators",
        "turrets",
        "weapon_fx_anchors",
        "subdepictions",
        "track_kind",
    ):
        if req_key not in data:
            raise RuntimeError(f"GFX manifest schema mismatch: missing {req_key}")
    if schema_version >= 2:
        if "semantic_nodes" not in data:
            raise RuntimeError("GFX manifest schema mismatch: missing semantic_nodes")
        if "transform_debug" not in data:
            raise RuntimeError("GFX manifest schema mismatch: missing transform_debug")

    cli_version = str(data.get("cli_version", "") or "").strip() or "unknown"
    cache_key = (asset_norm.lower(), sig[0], sig[1], sig[2], cli_version)
    cached = _GFX_JSON_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    result = {
        "asset_path": asset_norm,
        "gfx_manifest_path": str(out_json),
        "gfx_manifest": data,
        "gfx_cli_version": cli_version,
        "gfx_signature": {
            "path": sig[0],
            "mtime_ns": int(sig[1]),
            "size": int(sig[2]),
        },
        "source_files": list(data.get("source_files", []) or []),
        "matched_units": list(data.get("matched_units", []) or []),
        "reference_meshes": list(data.get("reference_meshes", []) or []),
        "operators": list(data.get("operators", []) or []),
        "turrets": list(data.get("turrets", []) or []),
        "weapon_fx_anchors": list(data.get("weapon_fx_anchors", []) or []),
        "subdepictions": list(data.get("subdepictions", []) or []),
        "track_kind": str(data.get("track_kind", "") or ""),
        "semantic_nodes": list(data.get("semantic_nodes", []) or []),
        "transform_debug": list(data.get("transform_debug", []) or []),
    }
    _GFX_JSON_CACHE.clear()
    _GFX_JSON_CACHE[cache_key] = dict(result)
    return dict(result)


def prepare_runtime_sources_from_zz(warno_root: Path, runtime_root: Path) -> Dict[str, Any]:
    root = Path(warno_root)
    runtime = Path(runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    resolver = get_zz_runtime_resolver(root)

    # WARNO DLC delta packs (v188908+) renamed the dictionary entry from the
    # legacy "PC/Mesh/Pack/Base.spk" to a short "MeshPack/Base.spk", which our
    # old strict prefix filter dropped silently. We accept both formats by
    # matching any dictionary path that contains both "mesh" and "pack" tokens
    # and ends with ".spk", and we de-duplicate by raw path so legacy and DLC
    # variants of the same SPK both get extracted side-by-side.
    seen_paths: set[str] = set()
    all_pack_spk_hits: List[Dict[str, Any]] = []
    for must in (["pc/mesh/pack/"], ["meshpack/"], ["mesh", "pack"]):
        for hit in resolver.find_all_by_suffix(
            suffixes=[".spk"],
            must_contain=must,
        ):
            key = str(hit.get("path", "")).strip().lower()
            if not key or key in seen_paths:
                continue
            seen_paths.add(key)
            all_pack_spk_hits.append(hit)
    if not all_pack_spk_hits:
        raise FileNotFoundError(
            "No SPK files were found under pc/mesh/pack/ or meshpack/ in WARNO ZZ.dat"
        )

    extracted_spk_paths: List[Path] = []
    for hit in all_pack_spk_hits:
        try:
            extracted_spk_paths.append(resolver.extract_hit_to_runtime(hit, runtime))
        except Exception:
            continue
    if not extracted_spk_paths:
        raise FileNotFoundError("Failed to extract any SPK files from WARNO ZZ.dat")

    def _pick_preferred(paths: Sequence[Path], preferred_tokens: Sequence[str]) -> Path | None:
        if not paths:
            return None
        lowered = [(p, normalize_asset_path(str(p)).lower()) for p in paths]
        for token in preferred_tokens:
            tok = token.lower()
            for p, low in lowered:
                if low.endswith(tok) or tok in Path(low).name:
                    return p
        return paths[0]

    mesh_spk_files: List[Path] = []
    skeleton_spk_files: List[Path] = []
    for p in extracted_spk_paths:
        name_low = p.name.lower()
        if "skeleton" in name_low:
            skeleton_spk_files.append(p)
        else:
            mesh_spk_files.append(p)
    mesh_spk_files = sorted(mesh_spk_files, key=lambda p: str(p).lower())
    skeleton_spk_files = sorted(skeleton_spk_files, key=lambda p: str(p).lower())

    if not mesh_spk_files:
        raise FileNotFoundError("No mesh SPK files were extracted from WARNO ZZ.dat")

    mesh_spk = _pick_preferred(
        mesh_spk_files,
        preferred_tokens=["mesh_all.spk", "gfxdescriptor/mesh_all.spk"],
    )
    skeleton_spk = _pick_preferred(
        skeleton_spk_files,
        preferred_tokens=["skeleton_all.spk", "gfxdescriptor/skeleton_all.spk"],
    )

    mesh_spk_dir = runtime / "PC" / "mesh" / "pack"
    if not mesh_spk_dir.exists():
        mesh_spk_dir = mesh_spk.parent if mesh_spk is not None else runtime
    skeleton_spk_dir = skeleton_spk.parent if skeleton_spk is not None else mesh_spk_dir

    unit_ndfbin_hit = resolver.find_first_by_suffix(
        suffixes=["unit.ndfbin"],
        must_contain=["gfx"],
    )
    unite_desc_hit = resolver.find_first_by_suffix(
        suffixes=["unitedescriptor.ndf"],
        must_contain=["gamedata", "gameplay", "gfx"],
    )
    unit_ndfbin_path = resolver.extract_hit_to_runtime(unit_ndfbin_hit, runtime) if unit_ndfbin_hit is not None else None
    unite_desc_path = resolver.extract_hit_to_runtime(unite_desc_hit, runtime) if unite_desc_hit is not None else None

    atlas_assets_root = runtime / "PC" / "Atlas" / "Assets"
    atlas_assets_root.mkdir(parents=True, exist_ok=True)

    ndf_hint_source = unite_desc_path or unit_ndfbin_path
    return {
        "warno_root": str(root),
        "runtime_root": str(runtime),
        "source_policy": "zz_runtime_only",
        "atlas_assets_root": str(atlas_assets_root),
        "mesh_spk": str(mesh_spk) if mesh_spk is not None else "",
        "mesh_spk_dir": str(mesh_spk_dir),
        "mesh_spk_files": [str(p) for p in mesh_spk_files],
        "skeleton_spk": str(skeleton_spk) if skeleton_spk is not None else "",
        "skeleton_spk_dir": str(skeleton_spk_dir),
        "skeleton_spk_files": [str(p) for p in skeleton_spk_files],
        "unit_ndfbin": str(unit_ndfbin_path) if unit_ndfbin_path is not None else "",
        "unite_descriptor": str(unite_desc_path) if unite_desc_path is not None else "",
        "ndf_hint_source": str(ndf_hint_source) if ndf_hint_source is not None else "",
        "zz_dat_files": [str(p) for p in resolver.dat_files],
    }


def find_generated_extra_maps(out_png: Path, strict_atlas: bool = False) -> Dict[str, Path]:
    stem = out_png.with_suffix("")
    out: Dict[str, Path] = {}
    alpha_canonical = stem.with_name(f"{stem.name}_A.png")
    alpha_legacy = stem.with_name(f"{stem.name}_alpha.png")
    candidates = {
        "diffuse": stem.with_name(f"{stem.name}_diffuse.png"),
        "diffuse_d": stem.with_name(f"{stem.name}_D.png"),
        "occlusion": stem.with_name(f"{stem.name}_occlusion.png"),
        "roughness": stem.with_name(f"{stem.name}_roughness.png"),
        "metallic": stem.with_name(f"{stem.name}_metallic.png"),
        "normal_reconstructed": stem.with_name(f"{stem.name}_normal_reconstructed.png"),
    }
    if alpha_canonical.exists():
        out["alpha"] = alpha_canonical
    elif alpha_legacy.exists():
        out["alpha"] = alpha_legacy
    for k, v in candidates.items():
        if v.exists():
            out[k] = v

    # Strict atlas mode only allows exact sibling extras for the same stem.
    # Non-strict mode may still discover dynamic extras from broad converter outputs.
    if not bool(strict_atlas):
        prefix = f"{stem.name}_"
        for p in stem.parent.glob(f"{stem.name}_*.png"):
            suffix = p.stem[len(prefix) :]
            if suffix:
                out.setdefault(suffix.lower(), p)

    # Canonical aliases derived from the same logical base.
    m = re.match(r"(?i)^(.+?)_(NM|ORM|DA|D|A|AO|O|R|M|OS|RS)$", stem.name)
    if m:
        base = m.group(1)
        tag = m.group(2).upper()
        if tag == "D":
            canonical_alpha = stem.with_name(f"{base}_A.png")
            if canonical_alpha.exists():
                out.setdefault("alpha", canonical_alpha)
        if tag == "ORM":
            canonical_o = stem.with_name(f"{base}_O.png")
            canonical_ao = stem.with_name(f"{base}_AO.png")
            canonical_r = stem.with_name(f"{base}_R.png")
            canonical_m = stem.with_name(f"{base}_M.png")
            if canonical_o.exists():
                out.setdefault("occlusion", canonical_o)
            elif canonical_ao.exists():
                out.setdefault("occlusion", canonical_ao)
            if canonical_r.exists():
                out.setdefault("roughness", canonical_r)
            if canonical_m.exists():
                out.setdefault("metallic", canonical_m)
        if tag in {"O", "AO"}:
            canonical_o = stem.with_name(f"{base}_O.png")
            canonical_ao = stem.with_name(f"{base}_AO.png")
            if canonical_o.exists():
                out.setdefault("occlusion", canonical_o)
            elif canonical_ao.exists():
                out.setdefault("occlusion", canonical_ao)
        if not bool(strict_atlas):
            for p in stem.parent.glob(f"{base}_TRK_*.png"):
                out.setdefault(p.stem.lower(), p)
    return out


_DEPS_READY_CACHE: Dict[Tuple[str, str], bool] = {}
_DEPS_AUTO_INSTALL_COUNT: int = 0


def _safe_resolved_str(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _python_cmd_key(py_cmd: Sequence[str]) -> str:
    return " ".join(str(x).strip().lower() for x in py_cmd if str(x).strip())


def _is_missing_py_dependency_error(stderr_text: str) -> set[str]:
    low = str(stderr_text or "").lower()
    found: set[str] = set()
    if ("no module named" not in low) and ("modulenotfounderror" not in low):
        return found
    if "pil" in low:
        found.add("Pillow")
    if "zstandard" in low:
        found.add("zstandard")
    return found


def _converter_env_with_deps(target_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    deps = str(target_dir)
    cur = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = deps if not cur else deps + os.pathsep + cur
    return env


def _scoped_deps_dir(base_dir: Path) -> Path:
    major = int(getattr(sys.version_info, "major", 3))
    minor = int(getattr(sys.version_info, "minor", 0))
    return Path(base_dir) / f"py{major}{minor}"


def _ensure_converter_python_deps(py_cmd: Sequence[str], target_dir: Path) -> Tuple[bool, str]:
    global _DEPS_AUTO_INSTALL_COUNT
    key = (_python_cmd_key(py_cmd), _safe_resolved_str(target_dir))
    if _DEPS_READY_CACHE.get(key):
        return True, f"deps already ready in {target_dir}"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return False, f"cannot create deps dir {target_dir}: {exc}"

    ensure_cmd = [*py_cmd, "-m", "ensurepip", "--upgrade"]
    ensure_cmd_text = " ".join(ensure_cmd)
    ensure_proc = subprocess.run(ensure_cmd, capture_output=True, text=True)
    ensure_msg = (ensure_proc.stderr or ensure_proc.stdout or "").strip()

    pip_cmd = [
        *py_cmd,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--target",
        str(target_dir),
        "Pillow",
        "zstandard",
    ]
    pip_cmd_text = " ".join(pip_cmd)
    pip_proc = subprocess.run(pip_cmd, capture_output=True, text=True)
    pip_msg = (pip_proc.stderr or pip_proc.stdout or "").strip()
    if pip_proc.returncode == 0:
        _DEPS_READY_CACHE[key] = True
        _DEPS_AUTO_INSTALL_COUNT += 1
        if ensure_proc.returncode != 0 and ensure_msg:
            return True, f"pip ok ({pip_cmd_text}), ensurepip warning ({ensure_cmd_text}): {ensure_msg[:220]}"
        return True, f"installed to {target_dir}"

    if ensure_proc.returncode != 0 and ensure_msg:
        return (
            False,
            f"ensurepip failed [{ensure_cmd_text}]: {ensure_msg[:220]} | "
            f"pip failed [{pip_cmd_text}]: {pip_msg[:320]}",
        )
    return False, f"pip failed [{pip_cmd_text}]: {pip_msg[:420]}"


def get_tgv_deps_auto_install_count() -> int:
    return int(_DEPS_AUTO_INSTALL_COUNT)


def install_tgv_converter_deps(converter: Path, deps_dir: Path | None = None) -> Tuple[bool, str]:
    if Path(converter).suffix.lower() != ".py":
        return True, "converter is not a python script; deps bootstrap is not required"
    if not sys.executable:
        return False, "python runtime unavailable (sys.executable is empty)"
    base_target = Path(deps_dir) if deps_dir is not None else (Path(converter).parent / ".warno_pydeps")
    target_dir = _scoped_deps_dir(base_target)
    py_cmd = [sys.executable]
    ok, msg = _ensure_converter_python_deps(py_cmd, target_dir)
    if ok:
        return True, f"[{' '.join(py_cmd)}] {msg}"
    return False, f"[{' '.join(py_cmd)}] {msg}"


def run_tgv_converter(
    converter: Path,
    src_tgv: Path,
    dst_png: Path,
    auto_install_deps: bool = True,
    deps_dir: Path | None = None,
    subprocess_timeout_sec: int = 120,
    atlas_map_path: Path | None = None,
    atlas_asset_path: str = "",
    atlas_out_dir: Path | None = None,
    only_logical_ref: str | None = None,
) -> None:
    atlas_mode = atlas_map_path is not None and str(atlas_asset_path or "").strip()
    if not atlas_mode:
        raise RuntimeError("converter_failed_other: legacy converter mode removed; atlas-map mode is required")
    if converter.suffix.lower() != ".py":
        raise RuntimeError("converter_failed_other: atlas-map mode requires python converter script")

    timeout_value: int | None
    try:
        timeout_value = int(subprocess_timeout_sec)
    except Exception:
        timeout_value = 120
    if timeout_value is not None and timeout_value <= 0:
        timeout_value = None

    if not sys.executable:
        raise RuntimeError("converter_failed_other: Python runtime not available (sys.executable is empty)")
    py_cmd: List[str] = [sys.executable]
    who = " ".join(py_cmd)
    base_deps_target = Path(deps_dir) if deps_dir is not None else (converter.parent / ".warno_pydeps")
    deps_target = _scoped_deps_dir(base_deps_target)
    runtime_env: Dict[str, str] | None = None

    if auto_install_deps:
        ok, dep_msg = _ensure_converter_python_deps(py_cmd, deps_target)
        if not ok:
            raise RuntimeError(
                "converter_failed_missing_dep: "
                f"Failed to prepare converter deps for {src_tgv.name}: [{who}] {dep_msg}"
            )
        runtime_env = _converter_env_with_deps(deps_target)

    out_dir = Path(atlas_out_dir) if atlas_out_dir is not None else dst_png.parent
    manifest_out = out_dir / "conversion_manifest.json"
    cmd = [
        *py_cmd,
        str(converter),
        "--atlas-map",
        str(Path(atlas_map_path)),
        "--asset-path",
        str(atlas_asset_path),
        "--out-dir",
        str(out_dir),
        "--manifest-out",
        str(manifest_out),
    ]
    if only_logical_ref:
        cmd.extend(["--only-logical-ref", str(only_logical_ref)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=runtime_env,
            timeout=timeout_value,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"converter_timeout: TGV converter timed out after {timeout_value}s "
            f"for {src_tgv.name}"
        )

    if proc.returncode == 0:
        return

    msg = (proc.stderr or proc.stdout or "").strip()
    missing = _is_missing_py_dependency_error(msg)
    if missing:
        modules = ",".join(sorted(missing))
        raise RuntimeError(
            "converter_failed_missing_dep: "
            f"TGV converter failed for {src_tgv.name} (missing {modules}): [{who}] {msg}. "
            "Install Pillow/zstandard for Blender Python runtime."
        )
    raise RuntimeError(f"converter_failed_other: TGV converter failed for {src_tgv.name}: [{who}] {msg}")


def _atlas_map_lookup_entries(
    atlas_map_index: Dict[str, List[Dict[str, Any]]] | None,
    ref: str,
) -> List[Dict[str, Any]]:
    if not isinstance(atlas_map_index, dict):
        return []
    ref_norm = _normalize_logical_ref(ref)
    if not ref_norm:
        return []
    out = atlas_map_index.get(ref_norm, [])
    return [item for item in out if isinstance(item, dict)]


def _resolve_source_from_atlas_entry(
    entry: Dict[str, Any],
    atlas_assets_root: Path,
    extra_roots: Sequence[Path],
    zz_resolver: ZZDatResolver | None,
    zz_runtime_root: Path | None,
) -> Tuple[Path, Path, str]:
    source_rel_norm = _normalize_logical_ref(str(entry.get("source_tgv_rel", "")))
    if not source_rel_norm:
        raise FileNotFoundError("missing_source: atlas map entry has empty source_tgv_rel")

    rel_under_assets = source_rel_norm
    if rel_under_assets.startswith("assets/"):
        rel_under_assets = rel_under_assets[len("assets/") :]
    rel_path = Path(*PurePosixPath(rel_under_assets).parts)

    src_png = atlas_assets_root / rel_path.with_suffix(".png")
    src_tgv = atlas_assets_root / rel_path.with_suffix(".tgv")
    atlas_source = "manual_path"

    if not src_tgv.exists() and not src_png.exists():
        for alt_root in extra_roots:
            alt_png = alt_root / rel_path.with_suffix(".png")
            alt_tgv = alt_root / rel_path.with_suffix(".tgv")
            if alt_tgv.exists() or alt_png.exists():
                src_tgv = alt_tgv
                src_png = alt_png
                atlas_source = "fallback"
                break

    if not src_tgv.exists() and not src_png.exists() and zz_resolver is not None:
        runtime_root = Path(zz_runtime_root) if zz_runtime_root is not None else None
        if runtime_root is None:
            runtime_root = atlas_assets_root
            for _ in range(3):
                runtime_root = runtime_root.parent
        candidates = [
            normalize_asset_path(f"PC/Atlas/Assets/{rel_path.with_suffix('.tgv').as_posix()}"),
            normalize_asset_path(f"PC/Atlas/Assets/{rel_path.with_suffix('.png').as_posix()}"),
            normalize_asset_path(f"Assets/{rel_path.with_suffix('.tgv').as_posix()}"),
            normalize_asset_path(f"Assets/{rel_path.with_suffix('.png').as_posix()}"),
        ]
        for cand in candidates:
            try:
                extracted = zz_resolver.extract_asset_to_runtime(cand, runtime_root, exact_only=True)
            except Exception:
                extracted = None
            if extracted is None:
                continue
            if extracted.suffix.lower() == ".tgv":
                src_tgv = extracted
                src_png = extracted.with_suffix(".png")
            else:
                src_png = extracted
                src_tgv = extracted.with_suffix(".tgv")
            atlas_source = "zz_runtime"
            break
    return src_tgv, src_png, atlas_source


def resolve_texture_from_atlas_ref(
    ref: str,
    atlas_assets_root: Path,
    out_model_dir: Path,
    converter: Path,
    texture_subdir: str,
    zz_resolver: ZZDatResolver | None = None,
    zz_runtime_root: Path | None = None,
    fallback_atlas_roots: Sequence[Path] | None = None,
    auto_install_deps: bool = True,
    deps_dir: Path | None = None,
    subprocess_timeout_sec: int = 120,
    atlas_map_path: Path | None = None,
    atlas_map_index: Dict[str, List[Dict[str, Any]]] | None = None,
    asset_path: str = "",
    atlas_json_strict: bool = True,
) -> Dict[str, Any]:
    rel = atlas_ref_to_rel_under_assets(ref)
    role_hint = classify_texture_role(ref)
    src_png = atlas_assets_root / rel
    src_tgv = src_png.with_suffix(".tgv")
    atlas_source = "manual_path"
    atlas_mode = ""
    atlas_entry: Dict[str, Any] | None = None

    extra_roots: List[Path] = []
    for root in fallback_atlas_roots or []:
        try:
            p = resolve_atlas_assets_root(Path(root))
        except Exception:
            p = Path(root)
        if p.exists() and p.is_dir():
            extra_roots.append(p)

    if isinstance(atlas_map_index, dict) and atlas_map_index:
        atlas_mode = "json_export"
        matched = _atlas_map_lookup_entries(atlas_map_index, ref)
        if matched:
            if len(matched) == 1:
                atlas_entry = matched[0]
            else:
                role_hint = classify_texture_role(ref)
                ranked = sorted(
                    matched,
                    key=lambda item: (
                        0
                        if _canonical_channel(str(item.get("target_channel", ""))) == role_hint
                        else 1,
                        str(item.get("target_logical_rel", "")).lower(),
                        str(item.get("source_tgv_rel", "")).lower(),
                    ),
                )
                atlas_entry = ranked[0]
        if atlas_entry is None and bool(atlas_json_strict):
            raise FileNotFoundError(
                f"missing_source: atlas_json_strict no mapping for ref {ref}"
            )

    if atlas_entry is not None:
        out_rel = _atlas_target_output_rel(
            target_logical_rel=str(atlas_entry.get("target_logical_rel", "")),
            target_basename=str(atlas_entry.get("target_basename", "")),
        )
        out_png = out_model_dir / texture_subdir / out_rel
    else:
        out_png = out_model_dir / texture_subdir / rel
    out_png.parent.mkdir(parents=True, exist_ok=True)

    source_type = ""
    if atlas_entry is None and bool(atlas_json_strict):
        raise FileNotFoundError(f"missing_source: atlas_json_strict no mapping for ref {ref}")

    # Deterministic cache hit: if target texture already exists, reuse it and skip converter.
    try:
        if out_png.exists() and out_png.is_file() and out_png.stat().st_size > 0:
            extras = find_generated_extra_maps(
                out_png,
                strict_atlas=bool(atlas_entry is not None and atlas_json_strict),
            )
            return {
                "atlas_ref": ref,
                "role": classify_texture_role(ref),
                "atlas_source": atlas_source,
                "source_type": "cached_png",
                "source_tgv": str(src_tgv) if src_tgv.exists() else None,
                "source_png": str(src_png) if src_png.exists() else None,
                "out_png": out_png,
                "extras": extras,
                "deps_auto_installed": False,
                "atlas_mode": atlas_mode,
                "atlas_target_logical_rel": str(atlas_entry.get("target_logical_rel", "")).strip() if atlas_entry else "",
                "atlas_target_basename": str(atlas_entry.get("target_basename", "")).strip() if atlas_entry else "",
                "atlas_target_channel": _canonical_channel(str(atlas_entry.get("target_channel", ""))) if atlas_entry else "",
                "cache_hit": True,
            }
    except OSError:
        pass

    if atlas_entry is not None:
        src_tgv, src_png, atlas_source = _resolve_source_from_atlas_entry(
            atlas_entry,
            atlas_assets_root=atlas_assets_root,
            extra_roots=extra_roots,
            zz_resolver=zz_resolver,
            zz_runtime_root=Path(zz_runtime_root) if zz_runtime_root is not None else None,
        )

    if src_tgv.exists():
        deps_before = get_tgv_deps_auto_install_count()
        try:
            run_tgv_converter(
                converter,
                src_tgv,
                out_png,
                auto_install_deps=auto_install_deps,
                deps_dir=deps_dir,
                subprocess_timeout_sec=subprocess_timeout_sec,
                atlas_map_path=Path(atlas_map_path) if atlas_map_path is not None else None,
                atlas_asset_path=str(asset_path or ""),
                atlas_out_dir=out_png.parent,
                only_logical_ref=(
                    str(atlas_entry.get("target_logical_rel", "")).strip()
                    if atlas_entry is not None
                    else None
                ),
            )
            source_type = "tgv"
        except Exception:
            raise
        deps_auto_installed = get_tgv_deps_auto_install_count() > deps_before
    elif src_png.exists():
        shutil.copy2(src_png, out_png)
        source_type = "png"
        deps_auto_installed = False
    else:
        raise FileNotFoundError(
            f"missing_source: Missing source texture for ref {ref}: expected {src_tgv} or {src_png}"
        )

    extras = find_generated_extra_maps(
        out_png,
        strict_atlas=bool(atlas_entry is not None and atlas_json_strict),
    )
    return {
        "atlas_ref": ref,
        "role": classify_texture_role(ref),
        "atlas_source": atlas_source,
        "source_type": source_type,
        "source_tgv": str(src_tgv) if src_tgv.exists() else None,
        "source_png": str(src_png) if src_png.exists() else None,
        "out_png": out_png,
        "extras": extras,
        "deps_auto_installed": bool(deps_auto_installed),
        "atlas_mode": atlas_mode,
        "atlas_target_logical_rel": str(atlas_entry.get("target_logical_rel", "")).strip() if atlas_entry else "",
        "atlas_target_basename": str(atlas_entry.get("target_basename", "")).strip() if atlas_entry else "",
        "atlas_target_channel": _canonical_channel(str(atlas_entry.get("target_channel", ""))) if atlas_entry else "",
        "cache_hit": False,
    }


def write_mtl(
    mtl_path: Path,
    material_names: Sequence[str],
    material_maps: Dict[str, Dict[str, Path]],
    obj_path: Path,
) -> None:
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    with mtl_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# WARNO extracted materials\n")
        for name in material_names:
            maps = material_maps.get(name, {})
            f.write(f"\nnewmtl {name}\n")
            f.write("Ka 1.000000 1.000000 1.000000\n")
            f.write("Kd 1.000000 1.000000 1.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("Ns 10.000000\n")
            f.write("illum 2\n")

            if "diffuse" in maps:
                f.write(f"map_Kd {rel_path_for_obj(obj_path, maps['diffuse'])}\n")
            if "alpha" in maps:
                f.write(f"map_d {rel_path_for_obj(obj_path, maps['alpha'])}\n")
            if "normal" in maps:
                f.write(f"map_Bump {rel_path_for_obj(obj_path, maps['normal'])}\n")
            if "roughness" in maps:
                f.write(f"map_Pr {rel_path_for_obj(obj_path, maps['roughness'])}\n")
            if "metallic" in maps:
                f.write(f"map_Pm {rel_path_for_obj(obj_path, maps['metallic'])}\n")


def write_obj(
    path: Path,
    model: Dict[str, Any],
    rot: Dict[str, float],
    material_names_by_id: Dict[int, str] | None = None,
    material_roles_by_id: Dict[int, str] | None = None,
    material_maps: Dict[str, Dict[str, Path]] | None = None,
    bone_name_by_index: Dict[int, str] | None = None,
    split_by_bone: bool = False,
) -> Dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    v_offset = 1
    vt_offset = 1
    num_faces = 0
    num_vertices = 0
    num_objects = 0
    material_maps = material_maps or {}
    material_names_by_id = material_names_by_id or {}
    material_roles_by_id = material_roles_by_id or {}
    bone_name_by_index = bone_name_by_index or {}
    used_materials: List[str] = []
    used_object_names: set[str] = set()
    mtl_path = path.with_suffix(".mtl")

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# WARNO extracted mesh\n")
        f.write(f"# asset: {model['asset']}\n")
        f.write(f"mtllib {mtl_path.name}\n")

        grouped_face_lines: Dict[str, List[Tuple[str, str]]] = {}
        grouped_order: List[str] = []
        grouped_name_by_key: Dict[str, str] = {}

        for part_i, part in enumerate(model["parts"]):
            xyz = part["vertices"]["xyz"]
            uv = part["vertices"]["uv"]
            mid = int(part["material"])
            mat_name = material_names_by_id.get(mid, f"Material_{mid}")
            mat_safe = sanitize_material_name(mat_name)
            mat_role = str(material_roles_by_id.get(mid, ""))

            vertex_count = len(xyz) // 3
            uv_count = len(uv) // 2
            num_vertices += vertex_count

            for i in range(vertex_count):
                x = xyz[i * 3 + 0]
                y = xyz[i * 3 + 1]
                z = xyz[i * 3 + 2]
                x, y, z = apply_rotation(x, y, z, rot)
                f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

            for i in range(uv_count):
                u = uv[i * 2 + 0]
                v = 1.0 - uv[i * 2 + 1]
                f.write(f"vt {u:.8f} {v:.8f}\n")

            part_label = f"part_{part_i:03d}_{mat_safe}"
            if split_by_bone:
                face_groups = split_faces_by_bone(
                    part,
                    part_label,
                    bone_name_by_index,
                    material_role=mat_role,
                    material_name=mat_safe,
                )
            else:
                tris: List[Tuple[int, int, int]] = []
                idx = part["indices"]
                for i in range(0, len(idx), 3):
                    if i + 2 < len(idx):
                        tris.append((int(idx[i + 0]), int(idx[i + 1]), int(idx[i + 2])))
                face_groups = [(part_label, tris)]

            if mat_safe not in used_materials:
                used_materials.append(mat_safe)

            for grp_name, tris in face_groups:
                if split_by_bone:
                    base_obj = sanitize_material_name(grp_name)
                    if base_obj.lower().startswith("bone_"):
                        base_obj = sanitize_material_name(f"{mat_safe}_{base_obj}")
                    key = base_obj.lower()
                    obj_name = grouped_name_by_key.get(key)
                    if obj_name is None:
                        obj_name = base_obj
                        grouped_name_by_key[key] = obj_name
                        grouped_order.append(obj_name)
                        grouped_face_lines[obj_name] = []
                else:
                    base_obj = sanitize_material_name(part_label)
                    obj_name = base_obj
                    n = 2
                    while obj_name.lower() in used_object_names:
                        obj_name = f"{base_obj}_{n}"
                        n += 1
                    used_object_names.add(obj_name.lower())
                    f.write(f"\no {obj_name}\n")
                    f.write(f"g {obj_name}\n")
                    f.write(f"usemtl {mat_safe}\n")
                    num_objects += 1

                for a, b, c in tris:
                    if a < 0 or b < 0 or c < 0:
                        continue
                    if a >= vertex_count or b >= vertex_count or c >= vertex_count:
                        continue

                    va = v_offset + a
                    vb = v_offset + b
                    vc = v_offset + c

                    if uv_count >= vertex_count:
                        ta = vt_offset + a
                        tb = vt_offset + b
                        tc = vt_offset + c
                        face_line = f"f {va}/{ta} {vb}/{tb} {vc}/{tc}"
                    else:
                        face_line = f"f {va} {vb} {vc}"
                    if split_by_bone:
                        grouped_face_lines[obj_name].append((mat_safe, face_line))
                    else:
                        f.write(face_line + "\n")
                    num_faces += 1

            v_offset += vertex_count
            vt_offset += uv_count

        if split_by_bone:
            for obj_name in grouped_order:
                rows = grouped_face_lines.get(obj_name, [])
                if not rows:
                    continue
                f.write(f"\no {obj_name}\n")
                f.write(f"g {obj_name}\n")
                num_objects += 1
                last_mat: str | None = None
                for mat_safe, face_line in rows:
                    if mat_safe != last_mat:
                        f.write(f"usemtl {mat_safe}\n")
                        last_mat = mat_safe
                    f.write(face_line + "\n")

    write_mtl(mtl_path, used_materials, material_maps, path)

    return {
        "vertices": num_vertices,
        "faces": num_faces,
        "parts": len(model["parts"]),
        "materials": len(used_materials),
        "objects": num_objects,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Direct extractor from WARNO .spk to .obj")
    ap.add_argument("--spk", required=True, type=Path, help="Input SPK file (MESH/PCPC)")
    ap.add_argument("--query", help="Case-insensitive substring in asset path")
    ap.add_argument("--asset", help="Exact asset path to extract")
    ap.add_argument("--list", action="store_true", help="Only list matches and exit")
    ap.add_argument("--limit", type=int, default=50, help="List limit (default: 50)")

    ap.add_argument("--pick", action="store_true", help="Interactive selection")
    ap.add_argument("--all", action="store_true", help="Select all matches")

    ap.add_argument("--out", type=Path, help="Output folder for OBJ files")
    ap.add_argument("--flat", action="store_true", help="Do not keep source folder structure")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path (default: <out>/manifest.json)",
    )
    ap.add_argument(
        "--write-fat",
        type=Path,
        default=None,
        help="Save parsed FAT json to this path",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite OBJ if file already exists",
    )
    ap.add_argument(
        "--rotate-x",
        type=float,
        default=0.0,
        help="Rotate vertices around X axis in degrees (default: 0)",
    )
    ap.add_argument(
        "--rotate-y",
        type=float,
        default=0.0,
        help="Rotate vertices around Y axis in degrees (default: 0)",
    )
    ap.add_argument(
        "--rotate-z",
        type=float,
        default=0.0,
        help="Rotate vertices around Z axis in degrees (default: 0)",
    )
    ap.add_argument(
        "--mirror-y",
        action="store_true",
        help="Mirror vertices on Y axis after rotation (fixes mirrored models in Blender)",
    )
    ap.add_argument(
        "--atlas-root",
        type=Path,
        default=None,
        help="Path to WARNO Atlas Assets folder for auto texture lookup/conversion",
    )
    ap.add_argument(
        "--tgv-converter",
        type=Path,
        default=Path("tgv_to_png.py"),
        help="Path to tgv_to_png converter script/exe (default: tgv_to_png.py)",
    )
    ap.add_argument(
        "--texture-subdir",
        default="textures",
        help="Subfolder name (inside each model output folder) for converted textures",
    )
    ap.add_argument(
        "--skeleton-spk",
        type=Path,
        default=None,
        help="Optional Skeleton_All.spk path for bone names / split parts",
    )
    ap.add_argument(
        "--unit-ndfbin",
        type=Path,
        default=None,
        help="Optional NDF hints source: Unit.ndfbin, UniteDescriptor.ndf, or WARNO/ModData base folder",
    )
    ap.add_argument(
        "--split-bone-parts",
        action="store_true",
        help="Split OBJ objects by dominant bone index (best with --skeleton-spk)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    with ExitStack() as stack:
        spk = stack.enter_context(SpkMeshExtractor(args.spk))
        skeleton_spk: SpkMeshExtractor | None = None
        if args.skeleton_spk is not None:
            if not args.skeleton_spk.exists() or not args.skeleton_spk.is_file():
                raise SystemExit(f"Skeleton SPK not found: {args.skeleton_spk}")
            skeleton_spk = stack.enter_context(SpkMeshExtractor(args.skeleton_spk))
        unit_ndf_hints: UnitNdfHintsResolver | None = None
        if args.unit_ndfbin is not None:
            if args.unit_ndfbin.exists():
                unit_ndf_hints = UnitNdfHintsResolver(args.unit_ndfbin)
            else:
                print(f"[WARN] NDF hints source not found, NDF bone hints disabled: {args.unit_ndfbin}")
        rot = build_rotation_params(args.rotate_x, args.rotate_y, args.rotate_z, mirror_y=args.mirror_y)
        atlas_root: Path | None = None
        if args.atlas_root is not None:
            atlas_root = resolve_atlas_assets_root(args.atlas_root)
            if not atlas_root.exists() or not atlas_root.is_dir():
                raise SystemExit(f"Atlas Assets folder not found: {atlas_root}")
            if not (atlas_root / "3D").exists() and not (atlas_root / "2D").exists():
                raise SystemExit(
                    f"Atlas root does not look like Assets folder (no 2D/3D subfolders): {atlas_root}"
                )
            if not args.tgv_converter.exists() or not args.tgv_converter.is_file():
                raise SystemExit(f"TGV converter not found: {args.tgv_converter}")
            # Fast sanity check: this should be Assets tree with texture files.
            has_texture = any(atlas_root.rglob("*.tgv")) or any(atlas_root.rglob("*.png"))
            if not has_texture:
                raise SystemExit(
                    f"No .tgv/.png textures found under atlas root: {atlas_root}"
                )

        if args.write_fat:
            payload = {"header": spk.header, "fat": spk.fat}
            args.write_fat.parent.mkdir(parents=True, exist_ok=True)
            args.write_fat.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Wrote FAT json: {args.write_fat}")

        matches = spk.find_matches(args.query, args.asset)

        if args.list:
            print(f"[INFO] Source: {args.spk}")
            print(f"[INFO] Total FAT entries: {len(spk.fat)}")
            print(f"[INFO] Material texture refs: {len(spk.material_texture_refs)}")
            print(f"[INFO] Matches: {len(matches)}")
            print_matches(matches, limit=args.limit)
            return 0

        if args.out is None:
            raise SystemExit("For extraction, --out is required")
        if not matches:
            raise SystemExit("No matches found")

        if args.all:
            selected = matches
        elif len(matches) == 1 and not args.pick:
            selected = matches
        elif args.pick:
            if len(matches) > 400:
                raise SystemExit(
                    f"Too many matches for --pick ({len(matches)}). Use a narrower --query."
                )
            print_matches(matches)
            raw = input("Pick index (e.g. 1,3-5 or all): ").strip()
            selected = [matches[i] for i in parse_selection(raw, len(matches))]
        else:
            raise SystemExit(
                f"Found {len(matches)} matches. Use --pick, --all, or exact --asset."
            )

        args.out.mkdir(parents=True, exist_ok=True)
        manifest_path = args.manifest or (args.out / "manifest.json")
        manifest: Dict[str, Any] = {
            "source_spk": str(args.spk),
            "header": spk.header,
            "rotation": {
                "x_deg": args.rotate_x,
                "y_deg": args.rotate_y,
                "z_deg": args.rotate_z,
                "mirror_y": bool(args.mirror_y),
            },
            "textures": {
                "atlas_root": str(atlas_root) if atlas_root else None,
                "tgv_converter": str(args.tgv_converter) if atlas_root else None,
                "material_ref_count": len(spk.material_texture_refs),
            },
            "skeleton": {
                "source_spk": str(args.skeleton_spk) if args.skeleton_spk else None,
                "source_unit_ndfbin": str(args.unit_ndfbin) if args.unit_ndfbin else None,
                "split_bone_parts": bool(args.split_bone_parts),
            },
            "query": args.query,
            "asset": args.asset,
            "items": [],
            "errors": [],
        }

        extracted = 0
        texture_errors_total = 0
        texture_files_total = 0
        for asset, meta in selected:
            rel = safe_output_relpath(asset)
            dst = args.out / rel.name if args.flat else (args.out / rel)
            if args.skip_existing and dst.exists():
                manifest["items"].append(
                    {
                        "asset": asset,
                        "meshIndex": int(meta["meshIndex"]),
                        "nodeIndex": int(meta["nodeIndex"]),
                        "obj": str(dst),
                        "status": "skipped_existing",
                    }
                )
                continue

            try:
                model = spk.get_model_geometry(asset)
                material_name_by_id, material_role_by_id = infer_material_names(
                    model,
                    mirror_y=bool(args.mirror_y),
                )
                material_ids = sorted({int(part["material"]) for part in model["parts"]})
                skeleton_hit: Tuple[str, Dict[str, Any]] | None = None
                mesh_node_index = int(meta.get("nodeIndex", -1))
                mesh_bone_names: List[str] = []
                external_bone_names: List[str] = []
                source_name_lists: Dict[str, List[str]] = {}
                external_sets: List[Tuple[str, List[str]]] = []
                bone_names: List[str] = []
                inferred_wheel_names: List[str] = []
                bone_name_by_index: Dict[int, str] = {}
                bone_name_source = "none"
                if mesh_node_index >= 0:
                    try:
                        mesh_bone_names = spk.parse_node_names(mesh_node_index)
                    except Exception:
                        mesh_bone_names = []
                if mesh_bone_names:
                    source_name_lists["mesh"] = list(mesh_bone_names)
                if skeleton_spk is not None:
                    used_ext_indices: set[int] = set()
                    used_ext_signatures: set[str] = set()

                    def add_external_set(source_name: str, node_idx: int) -> None:
                        if node_idx < 0:
                            return
                        nidx = int(node_idx)
                        if nidx in used_ext_indices:
                            return
                        used_ext_indices.add(nidx)
                        try:
                            names = skeleton_spk.parse_node_names(nidx)
                        except Exception:
                            names = []
                        if not names:
                            return
                        sig = "\x1f".join(str(n).strip().lower() for n in names if str(n).strip())
                        if not sig or sig in used_ext_signatures:
                            return
                        used_ext_signatures.add(sig)
                        external_sets.append((source_name, list(names)))
                        source_name_lists[source_name] = list(names)

                    # Primary: same hierarchical node index as mesh FAT entry.
                    add_external_set("external_same_index", mesh_node_index)

                    # Secondary: nearest asset-path hit in Skeleton SPK FAT.
                    skeleton_hit = skeleton_spk.find_best_fat_entry_for_asset(asset)
                    if skeleton_hit is not None:
                        _, sk_meta = skeleton_hit
                        sk_node_idx = int(sk_meta.get("nodeIndex", -1))
                        add_external_set("external_asset_match", sk_node_idx)

                    for _, names in external_sets:
                        external_bone_names.extend([str(n) for n in names if str(n).strip()])
                    external_bone_names = unique_keep_order(external_bone_names)

                candidates: List[Tuple[str, Dict[int, str]]] = []
                if mesh_bone_names:
                    candidates.append(("mesh", map_bone_index_names(mesh_bone_names)))
                for src_name, names in external_sets:
                    if names:
                        candidates.append((src_name, map_bone_index_names(names)))

                best_score = -10_000
                for src, cmap in candidates:
                    sc = score_bone_name_map(model, cmap, material_role_by_id)
                    if sc > best_score:
                        best_score = sc
                        bone_name_source = src
                        bone_name_by_index = cmap

                # Keep best candidate names, then fill only missing indices from
                # other maps. This prevents wheel bones from becoming anonymous
                # bone_### when one source is truncated.
                if bone_name_by_index and candidates:
                    merged = dict(bone_name_by_index)
                    for src, cmap in candidates:
                        if src == bone_name_source:
                            continue
                        for bidx, nm in cmap.items():
                            if bidx not in merged and nm:
                                merged[int(bidx)] = str(nm)
                    bone_name_by_index = merged

                if bone_name_by_index:
                    # Fallback for unresolved dominant wheel bones.
                    # Assign synthetic roue_* labels so wheel geometry does not
                    # collapse back into Chassis.
                    inferred_wheels = infer_missing_wheel_bone_names(
                        model,
                        bone_name_by_index,
                        rot,
                    )
                    if inferred_wheels:
                        bone_name_by_index.update(inferred_wheels)
                        inferred_wheel_names = unique_keep_order(
                            [str(v) for v in inferred_wheels.values() if str(v).strip()]
                        )

                if bone_name_source == "mesh":
                    bone_names = mesh_bone_names
                elif bone_name_source in source_name_lists:
                    bone_names = source_name_lists[bone_name_source]
                else:
                    bone_names = unique_keep_order([*mesh_bone_names, *external_bone_names])
                if inferred_wheel_names:
                    bone_names = unique_keep_order([*bone_names, *inferred_wheel_names])

                ndf_hint_bones: List[str] = []
                ndf_hint_source = "none"
                ndf_hint_error = ""
                if unit_ndf_hints is not None:
                    hint_payload = unit_ndf_hints.hints_for_asset(asset)
                    ndf_hint_source = str(hint_payload.get("source", "none")).strip() or "none"
                    ndf_hint_error = str(hint_payload.get("error", "")).strip()
                    raw_hint_bones = hint_payload.get("bones", [])
                    if isinstance(raw_hint_bones, list):
                        for raw_hint in raw_hint_bones:
                            nm = normalize_bone_label(str(raw_hint))
                            if not nm:
                                continue
                            ndf_hint_bones.append(sanitize_material_name(nm))
                    ndf_hint_bones = unique_keep_order(ndf_hint_bones)

                    if ndf_hint_bones and not bone_names:
                        bone_names = ndf_hint_bones
                        bone_name_source = "ndf"
                    elif ndf_hint_bones:
                        critical = [
                            n
                            for n in ndf_hint_bones
                            if (
                                n.lower() in {"chassis", "hull"}
                                or n.lower().startswith("base_tourelle")
                                or n.lower().startswith("fx_tourelle")
                            )
                        ]
                        if critical:
                            bone_names = unique_keep_order([*bone_names, *critical])

                bone_centers_by_index = estimate_bone_centers_by_index(model, bone_name_by_index, rot)
                raw_names_for_centers = (
                    list(source_name_lists.get(bone_name_source, []))
                    if source_name_lists.get(bone_name_source, [])
                    else bone_names
                )
                bone_positions: Dict[str, List[float]] = {}

                def _register_bone_position(key_name: str, pos: Tuple[float, float, float]) -> None:
                    k = str(key_name).strip().lower()
                    if not k:
                        return
                    if k not in bone_positions:
                        bone_positions[k] = [float(pos[0]), float(pos[1]), float(pos[2])]
                    tok = re.sub(r"[^a-z0-9]+", "", k)
                    if tok and tok not in bone_positions:
                        bone_positions[tok] = [float(pos[0]), float(pos[1]), float(pos[2])]

                for bidx, pos in bone_centers_by_index.items():
                    mapped_name = bone_name_by_index.get(int(bidx))
                    if mapped_name:
                        _register_bone_position(mapped_name, pos)
                    if 0 <= int(bidx) < len(raw_names_for_centers):
                        _register_bone_position(raw_names_for_centers[int(bidx)], pos)

                if ndf_hint_bones and "chassis" in {n.lower() for n in ndf_hint_bones} and "chassis" not in bone_positions:
                    for alias in ("chassisfake", "chassisarmaturefake", "base"):
                        hit = bone_positions.get(alias)
                        if hit:
                            bone_positions["chassis"] = list(hit)
                            break

                model_dir = dst.parent
                material_maps_by_name: Dict[str, Dict[str, Path]] = {}
                texture_payload: Dict[str, Any] = {
                    "refs": [],
                    "resolved": [],
                    "errors": [],
                }

                if atlas_root is not None:
                    refs_by_material = spk.get_texture_refs_for_material_ids(material_ids)
                    refs: List[str] = []
                    for mid in material_ids:
                        refs.extend(refs_by_material.get(int(mid), []))
                    refs = unique_keep_order(refs)
                    if not refs:
                        refs = spk.find_texture_refs_for_asset(asset, material_ids=material_ids)
                    texture_payload["refs"] = refs
                    texture_payload["refsByMaterial"] = {
                        str(int(mid)): list(refs_by_material.get(int(mid), []))
                        for mid in material_ids
                    }
                    resolved: List[Dict[str, Any]] = []
                    errors: List[Dict[str, str]] = []
                    resolved_by_ref: Dict[str, Dict[str, Any]] = {}
                    for ref in refs:
                        try:
                            item = resolve_texture_from_atlas_ref(
                                ref=ref,
                                atlas_assets_root=atlas_root,
                                out_model_dir=model_dir,
                                converter=args.tgv_converter,
                                texture_subdir=args.texture_subdir,
                            )
                            resolved.append(item)
                            resolved_by_ref[str(item["atlas_ref"])] = item
                        except Exception as tex_exc:
                            errors.append({"atlas_ref": ref, "error": str(tex_exc)})

                    chosen_maps = pick_material_maps_from_textures(resolved)
                    named_maps: Dict[str, Path] = {}
                    named_files: List[Dict[str, str]] = []
                    if chosen_maps:
                        named_maps, named_files = build_named_texture_aliases(
                            asset=asset,
                            model_dir=model_dir,
                            resolved=resolved,
                            chosen_maps=chosen_maps,
                        )
                        map_for_materials = named_maps or chosen_maps
                        track_named_maps = track_maps_from_named(named_files)
                        for mid in material_ids:
                            refs_mid = refs_by_material.get(int(mid), [])
                            resolved_mid = [
                                resolved_by_ref[ref]
                                for ref in refs_mid
                                if ref in resolved_by_ref
                            ]
                            maps = pick_material_maps_from_textures(resolved_mid)
                            if maps and named_files:
                                maps = remap_maps_to_named_sources(maps, named_files)
                            if not maps:
                                maps = dict(map_for_materials)
                            else:
                                for ch, src in map_for_materials.items():
                                    maps.setdefault(ch, src)

                            mname = sanitize_material_name(
                                material_name_by_id.get(mid, f"Material_{mid}")
                            )
                            role = material_role_by_id.get(mid, "other")
                            if role == "track_left":
                                maps.update(track_named_maps.get("left", {}))
                            elif role == "track_right":
                                maps.update(track_named_maps.get("right", {}))
                            elif role.startswith("track"):
                                maps.update(track_named_maps.get("generic", {}))
                            material_maps_by_name[mname] = maps

                    texture_payload["resolved"] = [
                        {
                            "atlas_ref": item["atlas_ref"],
                            "role": item["role"],
                            "source_type": item["source_type"],
                            "source_tgv": item["source_tgv"],
                            "source_png": item["source_png"],
                            "out_png": str(item["out_png"].resolve()),
                            "extras": {k: str(v.resolve()) for k, v in item["extras"].items()},
                        }
                        for item in resolved
                    ]
                    texture_payload["errors"] = errors
                    texture_payload["named"] = named_files
                    texture_errors_total += len(errors)
                    texture_files_total += len(resolved)

                stats = write_obj(
                    dst,
                    model,
                    rot,
                    material_names_by_id=material_name_by_id,
                    material_roles_by_id=material_role_by_id,
                    material_maps=material_maps_by_name,
                    bone_name_by_index=bone_name_by_index,
                    split_by_bone=bool(args.split_bone_parts),
                )
                manifest["items"].append(
                    {
                        "asset": asset,
                        "meshIndex": int(meta["meshIndex"]),
                        "nodeIndex": int(meta["nodeIndex"]),
                        "obj": str(dst),
                        "mtl": str(dst.with_suffix(".mtl")),
                        "materialTextures": {
                            mname: {
                                k: str(v.resolve()) if isinstance(v, Path) else str(v)
                                for k, v in maps.items()
                            }
                            for mname, maps in material_maps_by_name.items()
                        },
                        "materials": [
                            {
                                "id": int(mid),
                                "name": sanitize_material_name(
                                    material_name_by_id.get(mid, f"Material_{mid}")
                                ),
                                "role": material_role_by_id.get(mid, "other"),
                            }
                            for mid in material_ids
                        ],
                        "skeleton": {
                            "bone_name_source": bone_name_source,
                            "mesh_nodeIndex": mesh_node_index,
                            "external_matched_asset": skeleton_hit[0] if skeleton_hit else None,
                            "external_nodeIndex": int(skeleton_hit[1].get("nodeIndex", -1)) if skeleton_hit else -1,
                            "ndf_hint_source": ndf_hint_source,
                            "ndf_hint_error": ndf_hint_error or None,
                            "ndf_hint_bones": ndf_hint_bones,
                            "ndf_hint_files": (unit_ndf_hints.source_files if unit_ndf_hints else []),
                            "mesh_bone_names": mesh_bone_names,
                            "external_bone_names": external_bone_names,
                            "bone_names": bone_names,
                            "bone_positions": bone_positions,
                        },
                        "textures": texture_payload,
                        "status": "ok",
                        "stats": stats,
                    }
                )
                extracted += 1
            except Exception as exc:
                manifest["errors"].append(
                    {
                        "asset": asset,
                        "meshIndex": int(meta.get("meshIndex", -1)),
                        "error": str(exc),
                    }
                )

        if unit_ndf_hints is not None:
            manifest["skeleton"]["ndf_hint_files"] = unit_ndf_hints.source_files
            manifest["skeleton"]["ndf_hint_error"] = unit_ndf_hints.load_error or None

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[OK] Source FAT entries: {len(spk.fat)}")
        print(f"[OK] Matched: {len(matches)} | Selected: {len(selected)}")
        print(f"[OK] Extracted OBJ: {extracted}")
        if atlas_root is not None:
            print(
                f"[OK] Texture refs resolved: {texture_files_total}"
                + (f" | texture errors: {texture_errors_total}" if texture_errors_total else "")
            )
        if unit_ndf_hints is not None:
            if unit_ndf_hints.load_error:
                print(f"[WARN] NDF bone hints disabled: {unit_ndf_hints.load_error}")
            elif unit_ndf_hints.source_files:
                print(f"[OK] NDF bone hints source: {unit_ndf_hints.source_files[0]}")
        if manifest["errors"]:
            print(f"[WARN] Extraction errors: {len(manifest['errors'])}")
        print(f"[OK] Manifest: {manifest_path}")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("[ERROR] Interrupted by user")
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
