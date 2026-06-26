bl_info = {
    "name": "WARNO Importer",
    "author": "Kilivanchik",
    "version": (1, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > WARNO",
    "description": "Direct WARNO SPK importer with textures and optional helper bones",
    "category": "Import-Export",
}

import importlib.util
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from itertools import combinations, permutations, product
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence, Tuple

import bmesh
import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty
from bpy.props import CollectionProperty
from bpy.props import EnumProperty
from bpy.props import FloatProperty
from bpy.props import IntProperty
from bpy.props import PointerProperty
from bpy.props import StringProperty
from bpy.types import Operator
from bpy.types import Panel
from bpy.types import PropertyGroup
from bpy.types import UIList
from mathutils import Matrix
from mathutils import Vector


MODULE_CACHE: dict[tuple[str, str], tuple[int, Any]] = {}
ZZ_RUNTIME_SESSION_CACHE: dict[tuple[str, str, str, tuple[tuple[str, int, int], ...]], Dict[str, Any]] = {}
ASSET_INDEX_SESSION_CACHE: dict[str, Dict[str, Any]] = {}
ASSET_PICKER_VIEW_CACHE: dict[str, Dict[str, Any]] = {}
SAFE_NAME_RX = re.compile(r"[^A-Za-z0-9_.-]+")
LOD_SUFFIX_LOCAL_RX = re.compile(r"_(LOW|MID|HIGH|LOD[0-9]+)$", re.IGNORECASE)
CONTROL_CHAR_RX = re.compile(r"[\x00-\x1f\x7f]+")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff"}
WARNO_BUILD_STAMP = "v1.6.0"
WARNO_DEV_SCALE = 20.0 / 43.0
WARNO_OFF_MAT_SCALE = WARNO_DEV_SCALE / 100.0
WARNO_HELPER_BONE_LENGTH = 0.4
WARNO_SCENE_COLLECTION_SENTINEL = "__scene_collection__"
WARNO_DEV_CHILD_COLLECTION_THRESHOLD = 32


def _addon_state_path() -> Path:
    try:
        cfg_root = Path(str(bpy.utils.user_resource("CONFIG") or "")).expanduser()
        if not str(cfg_root):
            raise RuntimeError("empty config root")
    except Exception:
        cfg_root = Path.home()
    return cfg_root / "warno_importer_state.json"


def _read_addon_state() -> Dict[str, Any]:
    path = _addon_state_path()
    try:
        if not path.exists() or not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _read_saved_project_root() -> str:
    raw = _read_addon_state()
    txt = str(raw.get("project_root", "") or "").strip()
    if not txt:
        return ""
    p = Path(txt)
    try:
        if p.exists() and p.is_dir():
            return str(p)
    except Exception:
        return ""
    return ""


def _guess_default_project_root() -> str:
    remembered = _read_saved_project_root()
    if remembered:
        return remembered
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "warno_spk_extract.py").exists():
            return str(candidate)
    return str(here)


DEFAULT_PROJECT_ROOT = _guess_default_project_root()


def _safe_name(name: str, fallback: str) -> str:
    raw = SAFE_NAME_RX.sub("_", str(name or "").strip())
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or fallback


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _norm_low(value: str) -> str:
    return str(value or "").strip().lower()


def _warno_dev_scaled_xyz(
    x: float, y: float, z: float, scale: "float | None" = None
) -> Tuple[float, float, float]:
    s = float(WARNO_DEV_SCALE if scale is None else scale)
    return x * s, y * s, z * s


def _warno_off_mat_point_xyz(
    x: float, y: float, z: float, scale: "float | None" = None
) -> Tuple[float, float, float]:
    s = float(WARNO_OFF_MAT_SCALE if scale is None else scale)
    return -x * s, y * s, -z * s


def _active_game_scales(settings) -> Tuple[float, float]:
    """Return (dev_scale, off_mat_scale) for the active game, falling back to the
    WARNO constants if the extractor/profile is unavailable. Resolve ONCE per
    import (never per-vertex) and pass the value into the scale helpers."""
    try:
        mod = _extractor_module(settings)
        prof = mod.get_game_profile(getattr(settings, "game_preset", "WARNO"))
        return float(prof["dev_scale"]), float(prof["off_mat_scale"])
    except Exception:
        return float(WARNO_DEV_SCALE), float(WARNO_OFF_MAT_SCALE)


def _off_mat_axis_permutations(x: float, y: float, z: float) -> Dict[str, Tuple[float, float, float]]:
    return {
        "xyz": (float(x), float(y), float(z)),
        "xzy": (float(x), float(z), float(y)),
        "yxz": (float(y), float(x), float(z)),
        "yzx": (float(y), float(z), float(x)),
        "zxy": (float(z), float(x), float(y)),
        "zyx": (float(z), float(y), float(x)),
    }


def _off_mat_blender_remaps(x: float, y: float, z: float) -> Dict[str, Tuple[float, float, float]]:
    scale = float(WARNO_OFF_MAT_SCALE)
    out = {
        "current": _warno_off_mat_point_xyz(x, y, z),
        "identity": (float(x) * scale, float(y) * scale, float(z) * scale),
        "flip_xy": (-float(x) * scale, -float(y) * scale, float(z) * scale),
        "flip_xz": (-float(x) * scale, float(y) * scale, float(z) * scale),
    }
    for perm_name, (a, b, c) in _off_mat_axis_permutations(x, y, z).items():
        for sx, sy, sz in product((-1.0, 1.0), repeat=3):
            sign_key = f"{'+' if sx > 0 else '-'}1{'+' if sy > 0 else '-'}1{'+' if sz > 0 else '-'}1"
            key = f"{perm_name}:{sign_key}"
            out.setdefault(
                key,
                (float(sx) * float(a) * scale, float(sy) * float(b) * scale, float(sz) * float(c) * scale),
            )
    return out


def _off_mat_change_of_basis_matrix(variant_name: str) -> Matrix | None:
    variant_low = str(variant_name or "").strip().lower()
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
        m = re.fullmatch(r"([xyz]{3}):([+-]1)([+-]1)([+-]1)", variant_low)
        if m:
            perm = str(m.group(1))
            signs = (
                -1.0 if m.group(2).startswith("-") else 1.0,
                -1.0 if m.group(3).startswith("-") else 1.0,
                -1.0 if m.group(4).startswith("-") else 1.0,
            )
    if not perm or signs is None:
        return None
    axis_index = {"x": 0, "y": 1, "z": 2}
    scale = float(WARNO_OFF_MAT_SCALE)
    rows: List[List[float]] = []
    for out_axis, sign in zip(perm, signs):
        row = [0.0, 0.0, 0.0]
        row[axis_index[out_axis]] = float(sign) * scale
        rows.append(row)
    return Matrix(
        (
            (rows[0][0], rows[0][1], rows[0][2], 0.0),
            (rows[1][0], rows[1][1], rows[1][2], 0.0),
            (rows[2][0], rows[2][1], rows[2][2], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _off_mat_convert_matrix_to_blender(raw_matrix: Sequence[Sequence[float]], variant_name: str) -> Matrix | None:
    try:
        raw = Matrix(raw_matrix)
    except Exception:
        return None
    change = _off_mat_change_of_basis_matrix(variant_name)
    if change is None:
        return None
    try:
        change_inv = change.inverted()
    except Exception:
        return None
    try:
        return change @ raw @ change_inv
    except Exception:
        return None


def _decompose_affine_components(local_matrix: Matrix) -> Tuple[Vector, List[List[float]], Vector] | None:
    if not isinstance(local_matrix, Matrix):
        return None
    try:
        basis = local_matrix.to_3x3()
    except Exception:
        return None
    scale = Vector((basis.col[0].length, basis.col[1].length, basis.col[2].length))
    rot = basis.copy()
    for axis_i in range(3):
        axis_scale = float(scale[axis_i])
        if axis_scale > 1.0e-9:
            rot.col[axis_i] = rot.col[axis_i] / axis_scale
    rot_rows = [
        [float(rot[row][0]), float(rot[row][1]), float(rot[row][2])]
        for row in range(3)
    ]
    return local_matrix.translation.copy(), rot_rows, scale.copy()


def _compose_affine_components(
    translation: Sequence[float],
    rotation_basis: Sequence[Sequence[float]] | None,
    scale: Sequence[float] | None,
) -> Matrix:
    tx = float(translation[0]) if len(translation) >= 1 else 0.0
    ty = float(translation[1]) if len(translation) >= 2 else 0.0
    tz = float(translation[2]) if len(translation) >= 3 else 0.0
    rot = Matrix.Identity(3)
    if isinstance(rotation_basis, (list, tuple)) and len(rotation_basis) >= 3:
        try:
            rot = Matrix(
                (
                    (
                        float(rotation_basis[0][0]),
                        float(rotation_basis[0][1]),
                        float(rotation_basis[0][2]),
                    ),
                    (
                        float(rotation_basis[1][0]),
                        float(rotation_basis[1][1]),
                        float(rotation_basis[1][2]),
                    ),
                    (
                        float(rotation_basis[2][0]),
                        float(rotation_basis[2][1]),
                        float(rotation_basis[2][2]),
                    ),
                )
            )
        except Exception:
            rot = Matrix.Identity(3)
    scl = Vector((1.0, 1.0, 1.0))
    if isinstance(scale, (list, tuple)) and len(scale) >= 3:
        try:
            scl = Vector((float(scale[0]), float(scale[1]), float(scale[2])))
        except Exception:
            scl = Vector((1.0, 1.0, 1.0))
    basis = rot.copy()
    for axis_i in range(3):
        basis.col[axis_i] = basis.col[axis_i] * float(scl[axis_i])
    out = basis.to_4x4()
    out.translation = Vector((tx, ty, tz))
    return out


def _mat4_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(float(a[i][k]) * float(b[k][j]) for k in range(4))
    return out


def _mat3_det(m: Sequence[Sequence[float]]) -> float:
    return (
        float(m[0][0]) * (float(m[1][1]) * float(m[2][2]) - float(m[1][2]) * float(m[2][1]))
        - float(m[0][1]) * (float(m[1][0]) * float(m[2][2]) - float(m[1][2]) * float(m[2][0]))
        + float(m[0][2]) * (float(m[1][0]) * float(m[2][1]) - float(m[1][1]) * float(m[2][0]))
    )


def _mat3_inv(m: Sequence[Sequence[float]]) -> List[List[float]] | None:
    det = _mat3_det(m)
    if abs(float(det)) <= 1.0e-12:
        return None
    inv_det = 1.0 / float(det)
    return [
        [
            (float(m[1][1]) * float(m[2][2]) - float(m[1][2]) * float(m[2][1])) * inv_det,
            (float(m[0][2]) * float(m[2][1]) - float(m[0][1]) * float(m[2][2])) * inv_det,
            (float(m[0][1]) * float(m[1][2]) - float(m[0][2]) * float(m[1][1])) * inv_det,
        ],
        [
            (float(m[1][2]) * float(m[2][0]) - float(m[1][0]) * float(m[2][2])) * inv_det,
            (float(m[0][0]) * float(m[2][2]) - float(m[0][2]) * float(m[2][0])) * inv_det,
            (float(m[0][2]) * float(m[1][0]) - float(m[0][0]) * float(m[1][2])) * inv_det,
        ],
        [
            (float(m[1][0]) * float(m[2][1]) - float(m[1][1]) * float(m[2][0])) * inv_det,
            (float(m[0][1]) * float(m[2][0]) - float(m[0][0]) * float(m[2][1])) * inv_det,
            (float(m[0][0]) * float(m[1][1]) - float(m[0][1]) * float(m[1][0])) * inv_det,
        ],
    ]


def _mat_affine_inverse(m: Sequence[Sequence[float]]) -> List[List[float]] | None:
    rot = [
        [float(m[0][0]), float(m[0][1]), float(m[0][2])],
        [float(m[1][0]), float(m[1][1]), float(m[1][2])],
        [float(m[2][0]), float(m[2][1]), float(m[2][2])],
    ]
    trans = [float(m[0][3]), float(m[1][3]), float(m[2][3])]
    inv_rot = _mat3_inv(rot)
    if inv_rot is None:
        return None
    inv_trans = [
        -(float(inv_rot[0][0]) * trans[0] + float(inv_rot[0][1]) * trans[1] + float(inv_rot[0][2]) * trans[2]),
        -(float(inv_rot[1][0]) * trans[0] + float(inv_rot[1][1]) * trans[1] + float(inv_rot[1][2]) * trans[2]),
        -(float(inv_rot[2][0]) * trans[0] + float(inv_rot[2][1]) * trans[1] + float(inv_rot[2][2]) * trans[2]),
    ]
    return [
        [float(inv_rot[0][0]), float(inv_rot[0][1]), float(inv_rot[0][2]), float(inv_trans[0])],
        [float(inv_rot[1][0]), float(inv_rot[1][1]), float(inv_rot[1][2]), float(inv_trans[1])],
        [float(inv_rot[2][0]), float(inv_rot[2][1]), float(inv_rot[2][2]), float(inv_trans[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _off_mat_block_matrix(block: Sequence[float], layout_name: str) -> List[List[float]] | None:
    if not isinstance(block, (list, tuple)) or len(block) < 12:
        return None
    vals = [float(x) for x in block[:12]]
    if str(layout_name).strip().lower() == "row":
        return [
            [vals[0], vals[1], vals[2], vals[3]],
            [vals[4], vals[5], vals[6], vals[7]],
            [vals[8], vals[9], vals[10], vals[11]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    if str(layout_name).strip().lower() == "col":
        return [
            [vals[0], vals[4], vals[8], vals[3]],
            [vals[1], vals[5], vals[9], vals[7]],
            [vals[2], vals[6], vals[10], vals[11]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    return None


def _off_mat_pair_local_candidates(
    child_block: Sequence[float],
    parent_block: Sequence[float],
) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    for child_layout in ("row", "col"):
        child_mat = _off_mat_block_matrix(child_block, child_layout)
        if child_mat is None:
            continue
        for parent_layout in ("row", "col"):
            parent_mat = _off_mat_block_matrix(parent_block, parent_layout)
            if parent_mat is None:
                continue
            parent_inv = _mat_affine_inverse(parent_mat)
            if parent_inv is None:
                continue
            local_mat = _mat4_mul(parent_inv, child_mat)
            raw_xyz = (
                float(local_mat[0][3]),
                float(local_mat[1][3]),
                float(local_mat[2][3]),
            )
            for remap_name, remap_xyz in _off_mat_blender_remaps(*raw_xyz).items():
                out[f"{child_layout}_from_{parent_layout}:{remap_name}"] = remap_xyz
    return out


def _off_mat_stream_section_bounds(node_count: int) -> Tuple[int, int]:
    total = max(0, int(node_count))
    if total <= 0:
        return 0, 0
    start = int((total + 2) // 3)
    count = int((total + 2) // 3)
    end = min(total, start + count)
    return start, end


def _off_mat_stream_target_index(source_idx: int, block_idx: int, node_count: int) -> int:
    total = max(0, int(node_count))
    if total <= 0:
        return -1
    return int((int(source_idx) * 3 + int(block_idx)) % total)


def _helper_empty_display_preset(raw_name: str) -> Tuple[str, float]:
    low = _norm_low(raw_name)
    if low in {"armature", "papyrus", "fake"}:
        return "PLAIN_AXES", 0.2
    if low == "fx_tir":
        return "PLAIN_AXES", 0.06465
    if low == "fx_tourelle1_tir_01":
        return "PLAIN_AXES", 0.1
    if re.fullmatch(r"aa_[0-9]+_[0-9]+", low, flags=re.IGNORECASE):
        return "PLAIN_AXES", 0.41
    if re.fullmatch(r"as_1_[0-9]+", low, flags=re.IGNORECASE):
        return "CUBE", 0.28
    if re.fullmatch(r"as_2_[0-9]+", low, flags=re.IGNORECASE):
        return "PLAIN_AXES", 0.41
    if re.fullmatch(r"fx_tourelle[23]_tir_[0-9]+", low, flags=re.IGNORECASE):
        return "PLAIN_AXES", 0.41
    if re.fullmatch(r"pilot2?|train_avant", low, flags=re.IGNORECASE):
        return "PLAIN_AXES", 0.2
    if re.fullmatch(r"train_arriere|fx_leurre(?:_2)?", low, flags=re.IGNORECASE):
        return "PLAIN_AXES", 0.4
    if low.startswith("fx_fumee_chenille_"):
        return "CUBE", 0.125
    if low.startswith("fx_tourelle") and "_tir_" in low:
        return "PLAIN_AXES", 0.25
    if low.startswith("fx_"):
        return "PLAIN_AXES", 0.5
    if low.startswith("roue_elev_"):
        return "CUBE", 0.25
    if low == "tourelle_04":
        return "CUBE", 0.12
    if re.fullmatch(r"(tourelle|axe_canon|canon)_0[23]", low, flags=re.IGNORECASE):
        return "PLAIN_AXES", 0.41
    if low.startswith("tourelle_") or low.startswith("axe_canon_") or low.startswith("canon_"):
        return "CUBE", 0.08
    return "PLAIN_AXES", 1.0


def _is_preserved_runtime_fx_anchor(raw_name: str) -> bool:
    low = _norm_low(raw_name)
    if not low or not low.startswith("fx_"):
        return False
    if low.startswith("fx_tourelle"):
        return False
    if low.startswith("fx_fumee_chenille_"):
        return True
    if low.startswith("fx_stress_"):
        return True
    if low.startswith("fx_chaleur_"):
        return True
    if low.startswith("fx_incendie"):
        return True
    if low.startswith("fx_moteur"):
        return True
    if low.startswith("fx_munition"):
        return True
    return False


def _skip_default_group_assignment(low_name: str) -> bool:
    return bool(re.fullmatch(r"canon_0[23]", str(low_name or "")))


def _raw_scene_looks_vehicle_like(raw_name_values: Sequence[str]) -> bool:
    for raw_name in raw_name_values:
        low = _norm_low(str(raw_name))
        if not low:
            continue
        if low in {"chassis", "window"}:
            return True
        if low.startswith(("tourelle_", "canon_", "axe_canon_", "chenille_", "bloc_moteur", "trappe_")):
            return True
        if low.startswith("fx_fumee_chenille_"):
            return True
        if re.fullmatch(r"roue(_elev)?_[a-z0-9_]+", low, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"armature_[dg][12]", low, flags=re.IGNORECASE):
            return True
    return False


def _pretty_character_bip_name(raw_name: str) -> str:
    tokens = [tok for tok in re.split(r"[\s_]+", str(raw_name or "").strip()) if tok]
    if not tokens:
        return str(raw_name or "").strip()
    pretty_tokens: List[str] = []
    for token in tokens:
        low = str(token).strip().lower()
        if low == "bip01":
            pretty_tokens.append("Bip01")
        elif len(low) == 1 and low.isalpha():
            pretty_tokens.append(low.upper())
        elif low == "upperarm":
            pretty_tokens.append("UpperArm")
        else:
            pretty_tokens.append(low.capitalize())
    return " ".join(pretty_tokens)


def _needs_named_auto_smooth(low_name: str) -> bool:
    return bool(re.fullmatch(r"canon_0[23]", str(low_name or "")))


def _has_lod_suffix(stem: str) -> bool:
    return bool(LOD_SUFFIX_LOCAL_RX.search(str(stem or "")))


def _strip_lod_suffix_local(extractor_mod, stem: str) -> str:
    if hasattr(extractor_mod, "strip_lod_suffix"):
        try:
            return str(extractor_mod.strip_lod_suffix(stem))
        except Exception:
            pass
    return LOD_SUFFIX_LOCAL_RX.sub("", str(stem or ""))


def _asset_picker_sort_key(extractor_mod, asset: str) -> tuple[int, int, int, int, str]:
    path = str(asset or "").replace("\\", "/")
    low = path.lower()
    pp = PurePosixPath(path)
    stem = str(pp.stem or "")
    lod_suffix = _has_lod_suffix(stem)
    in_lods_folder = "/lods/" in low
    has_dest = "dest" in low
    has_anim = "anim" in low

    lod_rank = 2 if lod_suffix else 0
    if in_lods_folder:
        lod_rank += 1
    dest_rank = 1 if has_dest else 0
    anim_rank = 1 if has_anim else 0
    return (lod_rank, dest_rank, anim_rank, len(path), low)


def _asset_display_name(asset: str) -> str:
    p = str(asset or "").replace("\\", "/").strip()
    pp = PurePosixPath(p)
    stem = str(pp.stem or pp.name or "").strip()
    stem = CONTROL_CHAR_RX.sub("", stem).replace("\u200b", "")
    stem = re.sub(r"\s+", " ", stem).strip()
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "", stem)
    return cleaned or stem or str(pp.name or "asset")


def _asset_context_token(asset: str) -> str:
    p = str(asset or "").replace("\\", "/")
    pp = PurePosixPath(p)
    parts = [str(x) for x in pp.parts]
    low = [x.lower() for x in parts]

    token = ""
    if "units" in low:
        idx = low.index("units")
        if idx + 1 < len(parts):
            token = parts[idx + 1]
    elif "decors" in low:
        idx = low.index("decors")
        if idx + 1 < len(parts):
            token = parts[idx + 1]
    elif len(parts) >= 2:
        token = parts[-2]

    token = re.sub(r"[^0-9A-Za-z_.-]+", "", str(token or "").strip())
    return token or "ctx"


def _unique_asset_labels(assets: Sequence[str]) -> Dict[str, str]:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for a in assets:
        by_name[_asset_display_name(a)].append(a)

    out: Dict[str, str] = {}
    for name, vals in by_name.items():
        if len(vals) == 1:
            out[vals[0]] = name
            continue

        by_ctx: Dict[str, List[str]] = defaultdict(list)
        for a in vals:
            by_ctx[_asset_context_token(a)].append(a)

        for ctx, ctx_vals in by_ctx.items():
            if len(ctx_vals) == 1:
                out[ctx_vals[0]] = f"{name} · {ctx}"
                continue
            for i, a in enumerate(ctx_vals, start=1):
                out[a] = f"{name} · {ctx} #{i}"
    return out


def _model_valid_face_stats(model: Dict[str, Any]) -> tuple[int, int, float]:
    good_faces = 0
    total_faces = 0
    for part in model.get("parts", []):
        verts = part.get("vertices", {})
        xyz = verts.get("xyz", []) if isinstance(verts, dict) else []
        vc = len(xyz) // 3
        if vc <= 0:
            continue
        idx = part.get("indices", [])
        for i in range(0, len(idx), 3):
            if i + 2 >= len(idx):
                break
            total_faces += 1
            a = int(idx[i + 0])
            b = int(idx[i + 1])
            c = int(idx[i + 2])
            if min(a, b, c) >= 0 and max(a, b, c) < vc:
                good_faces += 1
    ratio = float(good_faces) / float(max(1, total_faces))
    return good_faces, total_faces, ratio


def _choose_asset_variant_for_import(extractor_mod, spk, requested_asset: str) -> tuple[str, str]:
    req_norm = extractor_mod.normalize_asset_path(str(requested_asset or ""))
    if not req_norm:
        return requested_asset, ""
    req_path = PurePosixPath(req_norm)
    req_stem = str(req_path.stem or "")
    req_base = _strip_lod_suffix_local(extractor_mod, req_stem).lower()
    req_ext = str(req_path.suffix or "").lower()
    req_low = req_norm.lower()
    req_dest = "dest" in req_low
    requested_is_lod = _has_lod_suffix(req_stem) or "/lods/" in req_low
    # If user explicitly picked an LOD variant, keep it as-is.
    # This avoids expensive candidate probing and prevents accidental fallback to base mesh.
    if requested_is_lod:
        return req_norm, "explicit_lod_selected"
    prefer_non_lod = not requested_is_lod

    candidates: List[str] = []
    try:
        entries = spk.list_entries()
    except Exception:
        entries = []
    for path, _meta in entries:
        pnorm = extractor_mod.normalize_asset_path(str(path or ""))
        if not pnorm:
            continue
        pp = PurePosixPath(pnorm)
        if req_ext and str(pp.suffix).lower() != req_ext:
            continue
        base = _strip_lod_suffix_local(extractor_mod, str(pp.stem)).lower()
        if base != req_base:
            continue
        candidates.append(pnorm)

    candidates = list(dict.fromkeys(candidates))
    if req_norm not in candidates:
        candidates.append(req_norm)
    if len(candidates) <= 1:
        return req_norm, ""
    try:
        requested_model = spk.get_model_geometry(req_norm)
    except Exception:
        requested_model = None
    if isinstance(requested_model, dict):
        req_good, req_total, req_ratio = _model_valid_face_stats(requested_model)
        if req_good > 0 and req_ratio >= 0.98:
            return req_norm, f"requested_exact good={req_good} total={req_total} ratio={req_ratio:.3f}"

    best_asset = req_norm
    best_score = -1.0e18
    best_note = ""
    for cand in candidates:
        try:
            model = spk.get_model_geometry(cand)
        except Exception:
            continue
        good, total, ratio = _model_valid_face_stats(model)
        if good <= 0:
            continue
        low = cand.lower()
        stem = PurePosixPath(cand).stem
        has_lod = _has_lod_suffix(stem)
        in_lods = "/lods/" in low
        has_dest = "dest" in low
        has_anim = "anim" in low

        score = float(good) + ratio * 1200.0
        if has_lod:
            score -= 350.0
        if in_lods:
            score -= 280.0
        if has_anim:
            score -= 220.0
        if req_dest == has_dest:
            score += 100.0
        else:
            score -= 140.0
        if "staticmesh" in low:
            score += 90.0
        if not has_dest:
            score += 50.0
        if prefer_non_lod and (not has_lod) and (not in_lods):
            score += 240.0
        if cand == req_norm:
            score += 60.0

        if score > best_score:
            best_score = score
            best_asset = cand
            best_note = f"good={good} total={total} ratio={ratio:.3f}"

    return best_asset, best_note


def _resolve_path(project_root: Path, raw: str) -> Path:
    txt = str(raw or "").strip().strip('"')
    if not txt:
        return Path()
    p = Path(bpy.path.abspath(txt))
    if not p.is_absolute():
        p = project_root / p
    return p


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in paths:
        try:
            p = Path(raw)
        except Exception:
            continue
        key = str(p).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _candidate_paths_from_raw(project_root: Path, raw: str) -> List[Path]:
    txt = str(raw or "").strip().strip('"')
    if not txt:
        return []
    p = Path(txt).expanduser()
    if p.is_absolute():
        return [p]
    bases = [project_root, *list(project_root.parents)[:3]]
    return _dedupe_paths([base / p for base in bases])


def _looks_like_modding_suite_root(path: Path) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    markers = [
        p / "moddingSuite.exe",            # unified single-exe (current)
        p / "moddingSuite" / "moddingSuite.exe",
        p / "moddingSuite.GfxCli",         # legacy multi-project layout
        p / "moddingSuite.AtlasCli",
        p / "atlas_cli",
        p / "tgv_to_png.py",
    ]
    return any(marker.exists() for marker in markers)


def _autodetect_modding_suite_root(project_root: Path, raw: str) -> Path | None:
    candidates: List[Path] = []
    candidates.extend(_candidate_paths_from_raw(project_root, raw))
    for name in ("moddingSuite", "moddingSuite-master"):
        for base in [project_root, *list(project_root.parents)[:3]]:
            candidates.append(base / name)
    for cand in _dedupe_paths(candidates):
        if _looks_like_modding_suite_root(cand):
            return cand
    return None


def _iter_modding_suite_cli_candidates(
    *,
    project_root: Path,
    modding_suite_root: Path | None,
    configured_path: str,
    exe_name: str,
    legacy_subdir: str,
    project_name: str,
) -> List[Path]:
    candidates: List[Path] = []
    candidates.extend(_candidate_paths_from_raw(project_root, configured_path))
    # Preferred: unified moddingSuite.exe (one binary, subcommand chosen by
    # the wrapper scripts via name detection).
    for base in [project_root, *list(project_root.parents)[:3]]:
        candidates.append(base / "moddingSuite" / "moddingSuite.exe")
        candidates.append(base / "moddingSuite-master" / "moddingSuite.exe")
    if modding_suite_root is not None:
        candidates.append(Path(modding_suite_root) / "moddingSuite.exe")
    # Legacy: split CLI exes (kept as fallback for older installs).
    for base in [project_root, *list(project_root.parents)[:3]]:
        candidates.append(base / "moddingSuite" / legacy_subdir / exe_name)
        candidates.append(base / "moddingSuite" / exe_name)
        candidates.append(base / "moddingSuite-master" / legacy_subdir / exe_name)
    if modding_suite_root is not None:
        root = Path(modding_suite_root)
        candidates.append(root / legacy_subdir / exe_name)
        candidates.append(root / exe_name)
        for config_name in ("Release", "Debug"):
            bin_root = root / project_name / "bin" / config_name
            if not bin_root.exists() or not bin_root.is_dir():
                continue
            for child in sorted(bin_root.iterdir(), key=lambda item: str(item).lower()):
                if child.is_dir():
                    candidates.append(child / exe_name)
    return _dedupe_paths(candidates)


def _autodetect_modding_suite_cli(
    *,
    project_root: Path,
    modding_suite_root: Path | None,
    configured_path: str,
    exe_name: str,
    legacy_subdir: str,
    project_name: str,
) -> Path | None:
    for cand in _iter_modding_suite_cli_candidates(
        project_root=project_root,
        modding_suite_root=modding_suite_root,
        configured_path=configured_path,
        exe_name=exe_name,
        legacy_subdir=legacy_subdir,
        project_name=project_name,
    ):
        if cand.exists() and cand.is_file():
            return cand
    return None


def _apply_modding_suite_runtime_defaults(settings: "WARNOImporterSettings") -> None:
    try:
        project_root = _project_root(settings)
    except Exception:
        return
    modding_suite_root = _autodetect_modding_suite_root(project_root, str(getattr(settings, "modding_suite_root", "") or ""))
    if modding_suite_root is not None:
        settings.modding_suite_root = str(modding_suite_root)

    atlas_cli = _autodetect_modding_suite_cli(
        project_root=project_root,
        modding_suite_root=modding_suite_root,
        configured_path=str(getattr(settings, "modding_suite_atlas_cli", "") or ""),
        exe_name="moddingSuite.AtlasCli.exe",
        legacy_subdir="atlas_cli",
        project_name="moddingSuite.AtlasCli",
    )
    if atlas_cli is not None:
        settings.modding_suite_atlas_cli = str(atlas_cli)

    gfx_cli = _autodetect_modding_suite_cli(
        project_root=project_root,
        modding_suite_root=modding_suite_root,
        configured_path=str(getattr(settings, "modding_suite_gfx_cli", "") or ""),
        exe_name="moddingSuite.GfxCli.exe",
        legacy_subdir="gfx_cli",
        project_name="moddingSuite.GfxCli",
    )
    if gfx_cli is not None:
        settings.modding_suite_gfx_cli = str(gfx_cli)


def _project_root(settings: "WARNOImporterSettings") -> Path:
    txt = str(settings.project_root or "").strip()
    if not txt:
        return Path(DEFAULT_PROJECT_ROOT)
    p = Path(bpy.path.abspath(txt))
    if not p.is_absolute():
        p = Path(DEFAULT_PROJECT_ROOT) / p
    return p


def _load_local_module(module_key: str, path: Path):
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Missing module file: {path}")
    key = (module_key, str(path.resolve()))
    mtime = path.stat().st_mtime_ns
    cached = MODULE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    module_name = f"warno_{module_key}_{abs(hash(key[1]))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    MODULE_CACHE[key] = (mtime, module)
    return module


def _extractor_module(settings: "WARNOImporterSettings"):
    root = _project_root(settings)
    return _load_local_module("extractor", root / "warno_spk_extract.py")


def _config_path(settings: "WARNOImporterSettings") -> Path:
    root = _project_root(settings)
    return root / "config.json"


def _cache_asset_dir(extractor_mod, settings: "WARNOImporterSettings", asset: str) -> Path:
    project_root = _project_root(settings)
    rel = extractor_mod.safe_output_relpath(asset)
    cache_raw = str(settings.cache_dir or "").strip()
    if cache_raw:
        cache_base = _resolve_path(project_root, cache_raw)
    else:
        cache_base = project_root / "output_blender"
    return cache_base / rel.parent


def _zz_runtime_root(settings: "WARNOImporterSettings") -> Path:
    project_root = _project_root(settings)
    raw = str(settings.zz_runtime_dir or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    cache_raw = str(settings.cache_dir or "").strip()
    if cache_raw:
        return _resolve_path(project_root, cache_raw) / "_zz_runtime"
    return project_root / "out_blender_runtime" / "zz_runtime"


def _atlas_json_cache_root(settings: "WARNOImporterSettings") -> Path:
    project_root = _project_root(settings)
    cache_raw = str(settings.cache_dir or "").strip()
    if cache_raw:
        base = _resolve_path(project_root, cache_raw)
    else:
        base = project_root / "output_blender"
    sub = str(settings.atlas_json_cache_subdir or "").strip() or "atlas_json_cache"
    return base / sub


def _gfx_json_cache_root(settings: "WARNOImporterSettings") -> Path:
    project_root = _project_root(settings)
    cache_raw = str(settings.cache_dir or "").strip()
    if cache_raw:
        base = _resolve_path(project_root, cache_raw)
    else:
        base = project_root / "output_blender"
    sub = str(settings.gfx_json_cache_subdir or "").strip() or "gfx_json_cache"
    return base / sub


def _log_file_path(settings: "WARNOImporterSettings") -> Path:
    root = _project_root(settings)
    name = str(settings.log_file_name or "").strip() or "warno_import.log"
    safe = _safe_name(name, "warno_import.log")
    if "." not in safe:
        safe += ".log"
    return root / safe


def _save_project_root_state(settings: "WARNOImporterSettings") -> tuple[bool, str]:
    try:
        root = _project_root(settings)
        if not root.exists() or not root.is_dir():
            return False, f"Project root not found: {root}"
        state = _read_addon_state()
        state["project_root"] = str(root.resolve())
        state["saved_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        path = _addon_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
        return True, f"Project root remembered: {root}"
    except Exception as exc:
        return False, f"Cannot remember project root: {exc}"


def _restore_project_root_and_auto_config(settings: "WARNOImporterSettings") -> None:
    if settings is None:
        return
    try:
        remembered = _read_saved_project_root()
        if remembered:
            settings.project_root = remembered
    except Exception:
        pass
    try:
        cfg = _config_path(settings)
        if cfg.exists() and cfg.is_file():
            _load_config_into_settings(settings, cfg)
        else:
            # First run (no config.json yet): auto-load the bundled example so the
            # moddingSuite root, wrappers and deps are correct without manual setup.
            for ex in (
                _project_root(settings) / "config.example.json",
                Path(__file__).resolve().parent / "config.example.json",
            ):
                if ex.exists() and ex.is_file():
                    _load_config_into_settings(settings, ex)
                    break
    except Exception:
        pass
    _enforce_fixed_runtime_defaults(settings)


def _warno_log(settings: Any, message: str, level: str = "INFO", stage: str = "") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lvl = str(level or "INFO").upper().strip() or "INFO"
    stg = str(stage or "").strip()
    prefix = f"[{ts}] [{lvl}]"
    if stg:
        prefix += f" [{stg}]"
    line = f"{prefix} {message}"
    print(line)
    if settings is None or not bool(getattr(settings, "log_to_file", False)):
        return
    try:
        path = _log_file_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _set_status(settings: Any, text: str, log_level: str = "INFO", stage: str = "") -> None:
    settings.status = str(text or "")
    _warno_log(settings, settings.status, level=log_level, stage=stage)


def _log_import_progress(
    settings: Any,
    *,
    step_idx: int,
    total_steps: int,
    label: str,
    t0: float,
    extra: str = "",
) -> None:
    safe_total = max(1, int(total_steps))
    safe_step = max(0, min(int(step_idx), safe_total))
    pct = int(round((float(safe_step) / float(safe_total)) * 100.0))
    label_s = str(label or "").strip()
    msg = f"progress {safe_step}/{safe_total} ({pct}%) {label_s}"
    if extra:
        msg += f" | {extra}"
    elapsed = time.monotonic() - float(t0)
    msg += f" | elapsed:{elapsed:.1f}s"
    _set_status(settings, msg, stage="import_progress")
    # Visible progress so the user can tell the import is working (not frozen):
    # (1) an ASCII bar in the system console, (2) the cursor % counter via wm.progress
    # (updates even during this blocking operator), (3) the bottom status-bar text.
    try:
        cells = 24
        filled = max(0, min(cells, int(round(cells * safe_step / safe_total))))
        bar = "█" * filled + "░" * (cells - filled)
        print(f"[WARNO] {bar} {pct:3d}% | {safe_step}/{safe_total} {label_s} | {elapsed:.1f}s", flush=True)
    except Exception:
        pass
    try:
        wm = bpy.context.window_manager
        if wm is not None:
            wm.progress_update(int(pct))
            wm.status_text_set(f"WARNO import {pct}% — {label_s} ({safe_step}/{safe_total})")
    except Exception:
        pass


def _toggle_import_console(settings: Any, *, open_console: bool, active: bool = True) -> bool:
    if not open_console:
        # Every import exit routes through the close call -> tear down the cursor progress
        # counter + clear the status-bar text here, regardless of console state.
        try:
            wm = bpy.context.window_manager
            if wm is not None:
                wm.progress_end()
                wm.status_text_set(None)
        except Exception:
            pass
    if not active:
        return False
    try:
        if bool(getattr(bpy.app, "background", False)):
            return False
    except Exception:
        pass
    if not hasattr(bpy.ops.wm, "console_toggle"):
        if open_console:
            _warno_log(settings, "System console is not available on this platform/build.", level="WARNING", stage="ui")
        return False
    try:
        bpy.ops.wm.console_toggle()
        _warno_log(
            settings,
            "System console auto-opened for import." if open_console else "System console auto-closed after import.",
            stage="ui",
        )
        return True
    except Exception as exc:
        _warno_log(
            settings,
            f"System console auto-toggle failed: {exc}",
            level="WARNING",
            stage="ui",
        )
        return False


def _dat_signature_for_cache(
    extractor_mod,
    warno_root: Path,
    game: "str | None" = None,
) -> tuple[tuple[str, int, int], ...]:
    rows: List[tuple[str, int, int]] = []
    try:
        dat_files = extractor_mod.find_warno_texture_dat_files(warno_root, game=game)
    except Exception:
        dat_files = []
    for p in dat_files:
        try:
            st = p.stat()
            rows.append((str(Path(p).resolve()).lower(), int(st.st_mtime_ns), int(st.st_size)))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0])
    return tuple(rows)


def _candidate_tgv_converter_from_modding_suite(settings: "WARNOImporterSettings") -> Path | None:
    project_root = _project_root(settings)
    mod_root = _autodetect_modding_suite_root(project_root, str(settings.modding_suite_root or "").strip())
    if mod_root is None:
        return None
    root = Path(mod_root)
    if not root.exists():
        return None

    cands = [root / "tgv_to_png.py"]
    try:
        for p in root.rglob("tgv_to_png.py"):
            cands.append(p)
    except Exception:
        pass
    for c in cands:
        if c.exists() and c.is_file():
            return c
    return None


def _iter_scenes_safe() -> List[Any]:
    data = getattr(bpy, "data", None)
    if data is None:
        return []
    scenes = getattr(data, "scenes", None)
    if scenes is None:
        return []
    try:
        return list(scenes)
    except Exception:
        return []


def _prepare_zz_runtime_sources(
    extractor_mod,
    settings: "WARNOImporterSettings",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    project_root = _project_root(settings)
    warno_raw = str(settings.warno_root or "").strip()
    if not warno_raw:
        raise RuntimeError("WARNO Folder is empty while ZZ.dat source is enabled.")
    warno_root = _resolve_path(project_root, warno_raw)
    if not warno_root.exists() or not warno_root.is_dir():
        raise RuntimeError(f"WARNO Folder not found: {warno_root}")

    game = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))
    runtime_root = _zz_runtime_root(settings)
    runtime_root.mkdir(parents=True, exist_ok=True)
    warno_key = str(warno_root.resolve()).lower()
    runtime_key = str(runtime_root.resolve()).lower()
    dat_signature = _dat_signature_for_cache(extractor_mod, warno_root, game=game)
    cache_key = (game, warno_key, runtime_key, dat_signature)
    if force_rebuild:
        for old_key in list(ZZ_RUNTIME_SESSION_CACHE.keys()):
            if old_key[1] == warno_key and old_key[2] == runtime_key:
                ZZ_RUNTIME_SESSION_CACHE.pop(old_key, None)
    else:
        cached = ZZ_RUNTIME_SESSION_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)

    info = extractor_mod.prepare_runtime_sources_from_zz(
        warno_root=warno_root,
        runtime_root=runtime_root,
        game=game,
    )
    info["runtime_root"] = str(runtime_root)
    info["source_policy"] = "zz_runtime_only"

    try:
        resolver = extractor_mod.get_zz_runtime_resolver(warno_root, game=game)
    except Exception:
        resolver = None
    if resolver is not None:
        info["zz_resolver"] = resolver
    ZZ_RUNTIME_SESSION_CACHE[cache_key] = dict(info)
    return dict(info)


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _expand_spk_source_paths(project_root: Path, raw: str) -> List[Path]:
    src = _resolve_path(project_root, raw)
    if not src.exists():
        return []
    if src.is_file():
        if src.suffix.lower() == ".spk":
            return [src]
        return []
    if not src.is_dir():
        return []

    files: List[Path] = []
    try:
        files.extend(sorted(src.rglob("*.spk"), key=lambda p: str(p).lower()))
    except Exception:
        files = []
    return [p for p in files if p.is_file()]


def _spk_paths_from_runtime_info(runtime_info: Dict[str, Any], list_key: str, single_key: str, dir_key: str) -> List[Path]:
    out: List[Path] = []
    raw_list = runtime_info.get(list_key)
    if isinstance(raw_list, list):
        for item in raw_list:
            txt = str(item or "").strip()
            if txt:
                out.append(Path(txt))
    single = str(runtime_info.get(single_key, "")).strip()
    if single:
        out.append(Path(single))
    source_dir = str(runtime_info.get(dir_key, "")).strip()
    if source_dir:
        d = Path(source_dir)
        if d.exists() and d.is_dir():
            try:
                out.extend(sorted(d.rglob("*.spk"), key=lambda p: str(p).lower()))
            except Exception:
                pass
    return [p for p in _dedupe_paths(out) if p.exists() and p.is_file()]


def _resolve_mesh_spk_paths(
    project_root: Path,
    settings: "WARNOImporterSettings",
    runtime_info: Dict[str, Any] | None = None,
) -> List[Path]:
    _ = project_root, settings
    runtime_info = runtime_info or {}
    out = _spk_paths_from_runtime_info(runtime_info, "mesh_spk_files", "mesh_spk", "mesh_spk_dir")
    return [p for p in out if "skeleton" not in p.name.lower()]


def _resolve_skeleton_spk_paths(
    project_root: Path,
    settings: "WARNOImporterSettings",
    runtime_info: Dict[str, Any] | None = None,
) -> List[Path]:
    _ = project_root, settings
    runtime_info = runtime_info or {}
    out = _spk_paths_from_runtime_info(runtime_info, "skeleton_spk_files", "skeleton_spk", "skeleton_spk_dir")
    return [p for p in out if "skeleton" in p.name.lower()]


def _pick_best_asset_spk_path(
    extractor_mod,
    spk_paths: Sequence[Path],
    asset: str,
    game: "str | None" = None,
) -> tuple[Path, str] | None:
    target = extractor_mod.normalize_asset_path(str(asset or "")).lower()
    best: tuple[int, int, str, str, Path, str] | None = None
    for spk_path in spk_paths:
        try:
            with extractor_mod.SpkMeshExtractor(spk_path, game=game) as spk:
                hit = spk.find_best_fat_entry_for_asset(asset)
        except Exception:
            continue
        if hit is None:
            continue
        asset_real, _ = hit
        asset_low = extractor_mod.normalize_asset_path(asset_real).lower()

        score = 0
        if asset_low == target:
            score += 1_000_000
        try:
            score += int(extractor_mod.shared_suffix_score(asset_low, target) * 100)
        except Exception:
            pass
        score -= len(asset_low)
        key = (score, -len(asset_low), asset_low, str(spk_path).lower(), spk_path, asset_real)
        if best is None or key > best:
            best = key

    if best is None:
        return None
    _, _, _, _, out_path, out_asset = best
    return out_path, out_asset


def _picker_view_cache_key(settings: "WARNOImporterSettings") -> str:
    try:
        return f"ptr:{int(settings.as_pointer())}"
    except Exception:
        return f"id:{id(settings)}"


def _get_picker_view_cache(settings: "WARNOImporterSettings") -> Dict[str, Any]:
    return ASSET_PICKER_VIEW_CACHE.get(_picker_view_cache_key(settings), {})


def _set_picker_view_cache(
    settings: "WARNOImporterSettings",
    *,
    assets: Sequence[str],
    groups: Sequence[Dict[str, Any]],
    folders: Sequence[Dict[str, Any]],
    folder_tree: Sequence[Dict[str, Any]],
    source: str,
    query: str,
) -> None:
    key = _picker_view_cache_key(settings)
    ASSET_PICKER_VIEW_CACHE[key] = {
        "assets": [str(a).strip() for a in assets if str(a).strip()],
        "groups": [dict(g) for g in groups if isinstance(g, dict)],
        "folders": [dict(f) for f in folders if isinstance(f, dict)],
        "folder_tree": [dict(n) for n in folder_tree if isinstance(n, dict)],
        "source": str(source or "").strip(),
        "query": str(query or "").strip(),
    }
    # Keep legacy Scene-string caches lightweight for backward compatibility.
    settings.match_cache_json = "[]"
    settings.asset_group_cache_json = "[]"
    settings.asset_folder_cache_json = "[]"


def _enum_keys(items: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for it in items:
        if isinstance(it, (tuple, list)) and len(it) >= 1:
            out.append(str(it[0]))
    return out


def _safe_set_selected_asset(settings: "WARNOImporterSettings", value: str) -> str:
    wanted = str(value or "").strip()
    settings.selected_asset = wanted
    return wanted


def _safe_set_selected_asset_group(settings: "WARNOImporterSettings", value: str) -> str:
    wanted = str(value or "").strip()
    settings.selected_asset_group = wanted
    return wanted


def _safe_set_selected_asset_lod(settings: "WARNOImporterSettings", value: str) -> str:
    wanted = str(value or "").strip() or "__base__"
    settings.selected_asset_lod = wanted
    return wanted


def _asset_index_path(settings: "WARNOImporterSettings") -> Path:
    project_root = _project_root(settings)
    cache_raw = str(settings.cache_dir or "").strip()
    cache_base = _resolve_path(project_root, cache_raw) if cache_raw else (project_root / "output_blender")
    return cache_base / "asset_index" / "mesh_assets_index.v2.json"


def _asset_index_signature(runtime_info: Dict[str, Any], mesh_spk_paths: Sequence[Path]) -> Dict[str, Any]:
    runtime_root = str(runtime_info.get("runtime_root", "") or "").strip()
    runtime_root_norm = ""
    if runtime_root:
        try:
            runtime_root_norm = str(Path(runtime_root).resolve()).lower()
        except Exception:
            runtime_root_norm = str(runtime_root).lower()

    spk_rows: List[Dict[str, Any]] = []
    for p in mesh_spk_paths:
        try:
            st = p.stat()
            path_key = str(p.resolve()).lower()
            spk_rows.append({"path": path_key, "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)})
        except Exception:
            continue
    spk_rows.sort(key=lambda r: str(r.get("path", "")))

    zz_rows: List[Dict[str, Any]] = []
    raw_dats = runtime_info.get("zz_dat_files", [])
    if isinstance(raw_dats, list):
        for raw in raw_dats:
            p = Path(str(raw))
            try:
                st = p.stat()
                zz_rows.append(
                    {
                        "path": str(p.resolve()).lower(),
                        "mtime_ns": int(st.st_mtime_ns),
                        "size": int(st.st_size),
                    }
                )
            except Exception:
                continue
    zz_rows.sort(key=lambda r: str(r.get("path", "")))
    return {
        "runtime_root": runtime_root_norm,
        "mesh_spk": spk_rows,
        "zz_dat": zz_rows,
    }


def _asset_index_signature_matches(index_data: Dict[str, Any], signature: Dict[str, Any]) -> bool:
    if not isinstance(index_data, dict):
        return False
    return index_data.get("signature", {}) == signature


def _load_asset_index_file(index_path: Path) -> Dict[str, Any] | None:
    try:
        if not index_path.exists() or not index_path.is_file():
            return None
        st = index_path.stat()
        key = str(index_path.resolve()).lower()
        cached = ASSET_INDEX_SESSION_CACHE.get(key)
        if isinstance(cached, dict) and int(cached.get("mtime_ns", -1)) == int(st.st_mtime_ns):
            data = cached.get("data")
            if isinstance(data, dict):
                return data
        data = json.loads(index_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            return None
        if int(data.get("schema_version", 0) or 0) != 3:
            return None
        ASSET_INDEX_SESSION_CACHE[key] = {"mtime_ns": int(st.st_mtime_ns), "data": data}
        return data
    except Exception:
        return None


def _save_asset_index_file(index_path: Path, payload: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp = index_path.with_suffix(index_path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(index_path)
    try:
        st = index_path.stat()
        key = str(index_path.resolve()).lower()
        ASSET_INDEX_SESSION_CACHE[key] = {"mtime_ns": int(st.st_mtime_ns), "data": payload}
    except Exception:
        pass


def _asset_enum_items(self: "WARNOImporterSettings", _context):
    view = _get_picker_view_cache(self)
    assets = view.get("assets", []) if isinstance(view, dict) else []
    if not isinstance(assets, list) or not assets:
        raw = str(self.match_cache_json or "").strip()
        if raw:
            try:
                assets = json.loads(raw)
            except Exception:
                assets = []
    if not isinstance(assets, list) or not assets:
        return [("__none__", "<no matches>", "No matches", 0)]

    out_assets = [str(a) for a in assets if str(a).strip()]
    selected = str(getattr(self, "selected_asset", "") or "").strip()
    if selected and selected != "__none__" and selected not in out_assets:
        out_assets.insert(0, selected)
    labels = _unique_asset_labels(out_assets)
    out = []
    for i, asset in enumerate(out_assets):
        asset_txt = str(asset)
        label = labels.get(asset_txt, _asset_display_name(asset_txt))
        out.append((asset_txt, label, asset_txt, i))
    return out


def _asset_popup_enum_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__none__", "<no scene settings>", "No settings", 0)]
    return _asset_enum_items(settings, context)


def _asset_groups_from_cache(settings: "WARNOImporterSettings") -> List[Dict[str, Any]]:
    view = _get_picker_view_cache(settings)
    groups_view = view.get("groups", []) if isinstance(view, dict) else []
    if isinstance(groups_view, list) and groups_view:
        out_view: List[Dict[str, Any]] = []
        for g in groups_view:
            if isinstance(g, dict):
                out_view.append(dict(g))
        if out_view:
            return out_view
    raw = str(getattr(settings, "asset_group_cache_json", "") or "").strip()
    if not raw:
        return []
    try:
        groups = json.loads(raw)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    if not isinstance(groups, list):
        return out
    for g in groups:
        if not isinstance(g, dict):
            continue
        primary = str(g.get("primary", "")).strip()
        label = str(g.get("label", "")).strip() or primary
        lods_raw = g.get("lods", [])
        lods: List[str] = []
        if isinstance(lods_raw, list):
            for it in lods_raw:
                txt = str(it or "").strip()
                if txt and txt != primary:
                    lods.append(txt)
        if not primary:
            continue
        out.append({"primary": primary, "label": label, "lods": lods})
    return out


def _asset_folders_from_cache(settings: "WARNOImporterSettings") -> List[Dict[str, Any]]:
    view = _get_picker_view_cache(settings)
    folders_view = view.get("folders", []) if isinstance(view, dict) else []
    if isinstance(folders_view, list) and folders_view:
        out_view: List[Dict[str, Any]] = []
        for f in folders_view:
            if isinstance(f, dict):
                out_view.append(dict(f))
        if out_view:
            return out_view
    raw = str(getattr(settings, "asset_folder_cache_json", "") or "").strip()
    if not raw:
        # Backward compatibility: derive folders from group cache if folder cache is absent.
        groups = _asset_groups_from_cache(settings)
        if not groups:
            return []
        buckets: Dict[str, Dict[str, Any]] = {}
        for g in groups:
            primary = str(g.get("primary", "")).strip()
            if not primary:
                continue
            pp = PurePosixPath(primary.replace("\\", "/"))
            parts = list(pp.parent.parts)
            if parts and str(parts[-1]).lower() == "lods":
                parts = parts[:-1]
            if parts:
                parts[-1] = LOD_SUFFIX_LOCAL_RX.sub("", str(parts[-1]))
            key = "/".join([str(p).lower() for p in parts])
            label = "/".join([str(p) for p in parts]) if parts else "."
            item = buckets.get(key)
            if item is None:
                item = {"key": key, "label": label, "category": _asset_category_from_folder_key(key), "models": []}
                buckets[key] = item
            lods = [str(v).strip() for v in g.get("lods", []) if str(v).strip()]
            item["models"].append({"primary": primary, "lods": lods})
        return list(buckets.values())
    try:
        folders = json.loads(raw)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    if not isinstance(folders, list):
        return out
    for f in folders:
        if not isinstance(f, dict):
            continue
        key = str(f.get("key", "")).strip()
        label = str(f.get("label", "")).strip() or key
        category = str(f.get("category", "")).strip() or _asset_category_from_folder_key(key)
        models_raw = f.get("models", [])
        models: List[Dict[str, Any]] = []
        if isinstance(models_raw, list):
            for m in models_raw:
                if not isinstance(m, dict):
                    continue
                primary = str(m.get("primary", "")).strip()
                if not primary:
                    continue
                lods_raw = m.get("lods", [])
                lods: List[str] = []
                if isinstance(lods_raw, list):
                    lods = [str(v).strip() for v in lods_raw if str(v).strip()]
                models.append({"primary": primary, "lods": lods})
        if not key:
            continue
        out.append({"key": key, "label": label, "category": category, "models": models})
    return out


def _browser_folder_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__none__", "<scan assets first>", "No folders", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__none__", "<scan assets first>", "No folders", 0)]
    root_key = str(getattr(self, "root_key", "") or "").strip().lower()
    category_key = str(getattr(self, "category_key", "") or "").strip().lower()
    out = []
    idx = 0
    filtered = []
    for folder in folders:
        key = str(folder.get("key", "")).strip().lower()
        if not key:
            continue
        if root_key and root_key != "__none__":
            if key != root_key and not key.startswith(root_key + "/"):
                continue
        folder_cat = str(folder.get("category", "")).strip().lower() or _asset_category_from_folder_key(key).lower()
        if category_key and category_key not in {"__all__", "__none__"} and folder_cat != category_key:
            continue
        filtered.append(folder)
    filtered.sort(key=lambda f: str(f.get("key", "")).lower())
    for folder in filtered:
        key = str(folder.get("key", "")).strip()
        key_low = key.lower()
        if root_key and root_key != "__none__" and key_low.startswith(root_key):
            rel = key_low[len(root_key) :].lstrip("/")
        else:
            rel = key_low
        parts = [p for p in rel.split("/") if p]
        if not parts:
            label = _folder_root_label(root_key if root_key and root_key != "__none__" else key)
        else:
            indent = "  " * max(0, len(parts) - 1)
            label = f"{indent}{parts[-1]}"
        out.append((key, label, key, idx))
        idx += 1
    if not out:
        return [("__none__", "<no folders>", "No folders for this root", 0)]
    return out


def _browser_model_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__none__", "<no models>", "No models", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__none__", "<scan assets first>", "No models", 0)]
    folder_key = str(getattr(self, "folder_key", "") or "").strip()
    root_key = str(getattr(self, "root_key", "") or "").strip().lower()
    category_key = str(getattr(self, "category_key", "") or "").strip().lower()
    filtered_folders: List[Dict[str, Any]] = []
    for f in folders:
        key = str(f.get("key", "")).strip().lower()
        if not key:
            continue
        if root_key and root_key != "__none__":
            if key != root_key and not key.startswith(root_key + "/"):
                continue
        folder_cat = str(f.get("category", "")).strip().lower() or _asset_category_from_folder_key(key).lower()
        if category_key and category_key not in {"__all__", "__none__"} and folder_cat != category_key:
            continue
        filtered_folders.append(f)
    if not filtered_folders:
        filtered_folders = folders

    target = None
    for f in filtered_folders:
        if str(f.get("key", "")).strip() == folder_key:
            target = f
            break
    if target is None:
        target = filtered_folders[0]
    models = target.get("models", [])
    if not isinstance(models, list):
        models = []

    tokens = _normalize_search_tokens(str(getattr(self, "search_text", "") or ""))
    primaries: List[str] = []
    for m in models:
        p = str(m.get("primary", "")).strip()
        if not p:
            continue
        blob = f"{p.lower()} {_asset_display_name(p).lower()}"
        if tokens and not all(tok in blob for tok in tokens):
            continue
        primaries.append(p)
    if not primaries:
        return [("__none__", "<no models>", "No models for this folder/filter", 0)]
    labels = _unique_asset_labels(primaries)
    out = []
    for i, p in enumerate(primaries):
        out.append((p, labels.get(p, _asset_display_name(p)), p, i))
    return out


def _browser_lod_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__base__", "<base model>", "Use base model", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__base__", "<base model>", "Use base model", 0)]
    model = str(getattr(self, "model_key", "") or "").strip()
    target_lods: List[str] = []
    for f in folders:
        models = f.get("models", [])
        if not isinstance(models, list):
            continue
        for m in models:
            primary = str(m.get("primary", "")).strip()
            if primary != model:
                continue
            lods_raw = m.get("lods", [])
            if isinstance(lods_raw, list):
                target_lods = [str(v).strip() for v in lods_raw if str(v).strip()]
            break
    out = [("__base__", "<base model>", "Use base model", 0)]
    if not target_lods:
        return out
    labels = _unique_asset_labels(target_lods)
    for i, lod in enumerate(target_lods, start=1):
        out.append((lod, labels.get(lod, _asset_display_name(lod)), lod, i))
    return out


def _is_lod_asset_path(asset: str) -> bool:
    p = str(asset or "").replace("\\", "/")
    low = p.lower()
    stem = PurePosixPath(p).stem
    return _has_lod_suffix(stem) or "/lods/" in low


def _lod_rank_for_asset(asset: str) -> tuple[int, int, str]:
    p = str(asset or "").replace("\\", "/")
    pp = PurePosixPath(p)
    stem = str(pp.stem or "")
    m = LOD_SUFFIX_LOCAL_RX.search(stem)
    if m:
        token = str(m.group(1) or "").upper()
        if token == "LOW":
            return (0, 0, p.lower())
        if token == "MID":
            return (1, 0, p.lower())
        if token == "HIGH":
            return (2, 0, p.lower())
        if token.startswith("LOD"):
            try:
                return (3, int(token[3:]), p.lower())
            except Exception:
                return (3, 999, p.lower())
    if "/lods/" in p.lower():
        return (4, 999, p.lower())
    return (99, 999, p.lower())


def _asset_group_key(extractor_mod, asset: str) -> str:
    norm = str(asset or "").replace("\\", "/")
    if hasattr(extractor_mod, "normalize_asset_path"):
        try:
            norm = str(extractor_mod.normalize_asset_path(norm))
        except Exception:
            pass
    pp = PurePosixPath(norm)
    parts = list(pp.parent.parts)
    if parts and str(parts[-1]).lower() == "lods":
        parts = parts[:-1]
    if parts:
        parts[-1] = _strip_lod_suffix_local(extractor_mod, str(parts[-1]))
    base = _strip_lod_suffix_local(extractor_mod, pp.stem)
    return "/".join([str(p).lower() for p in parts] + [str(base).lower()])


def _unit_lod_match_key(extractor_mod, asset: str) -> tuple:
    """(faction, lod-stripped stem) — used to re-attach a shared-folder LOD to its unit.
    Both `units/<faction>/<unit>/<unit>.ase2ndfbin` and the shared
    `units/<faction>/lods/<unit>_low.ase2ndfbin` resolve to the SAME (faction, unit)."""
    norm = str(asset or "").replace("\\", "/")
    pp = PurePosixPath(norm)
    parts = [str(p).lower() for p in pp.parts]
    faction = ""
    if "units" in parts:
        i = parts.index("units")
        if i + 1 < len(parts) and parts[i + 1] != "lods":
            faction = parts[i + 1]
    stem = _strip_lod_suffix_local(extractor_mod, pp.stem).lower()
    return (faction, stem)


def _build_asset_groups(extractor_mod, assets: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for asset in assets:
        txt = str(asset or "").strip()
        if not txt:
            continue
        buckets[_asset_group_key(extractor_mod, txt)].append(txt)

    groups: List[Dict[str, Any]] = []
    for _key, arr in buckets.items():
        uniq: List[str] = list(dict.fromkeys(arr))
        uniq = sorted(uniq, key=lambda a: _asset_picker_sort_key(extractor_mod, a))
        non_lod = [a for a in uniq if not _is_lod_asset_path(a)]
        primary = non_lod[0] if non_lod else uniq[0]
        lods = [a for a in uniq if a != primary and _is_lod_asset_path(a)]
        lods = sorted(lods, key=_lod_rank_for_asset)
        pp = PurePosixPath(primary.replace("\\", "/"))
        label = _asset_display_name(primary)
        groups.append({"primary": primary, "label": label, "lods": lods})

    # Re-attach LOD-only "orphan" groups (assets under a SHARED `<faction>/lods/` folder, whose
    # group key resolves to the faction not the unit -> they don't merge with their base unit)
    # to their real unit group, matched by (faction, lod-stripped stem). Without this the Asset
    # Browser shows scattered *_low models not bound to any unit. No-op when there are no such
    # orphans (e.g. per-unit LODs already merge), so it's safe for every game/preset.
    unit_groups = [g for g in groups if not _is_lod_asset_path(str(g.get("primary", "")))]
    orphan_groups = [g for g in groups if _is_lod_asset_path(str(g.get("primary", "")))]
    if orphan_groups and unit_groups:
        by_key: Dict[tuple, Dict[str, Any]] = {}
        for g in unit_groups:
            by_key.setdefault(_unit_lod_match_key(extractor_mod, str(g.get("primary", ""))), g)
        leftover: List[Dict[str, Any]] = []
        for og in orphan_groups:
            host = by_key.get(_unit_lod_match_key(extractor_mod, str(og.get("primary", ""))))
            if host is None:
                leftover.append(og)
                continue
            merged = list(host.get("lods", [])) + [str(og.get("primary", ""))] + list(og.get("lods", []))
            host["lods"] = sorted(dict.fromkeys([m for m in merged if m]), key=_lod_rank_for_asset)
        groups = unit_groups + leftover

    groups.sort(key=lambda g: _asset_picker_sort_key(extractor_mod, str(g.get("primary", ""))))
    return groups


def _asset_folder_key_for_group(extractor_mod, primary_asset: str) -> tuple[str, str]:
    norm = str(primary_asset or "").replace("\\", "/")
    if hasattr(extractor_mod, "normalize_asset_path"):
        try:
            norm = str(extractor_mod.normalize_asset_path(norm))
        except Exception:
            pass
    pp = PurePosixPath(norm)
    parts = list(pp.parent.parts)
    if parts and str(parts[-1]).lower() == "lods":
        parts = parts[:-1]
    if parts:
        parts[-1] = _strip_lod_suffix_local(extractor_mod, str(parts[-1]))
    key = "/".join([str(p).lower() for p in parts])
    display = "/".join([str(p) for p in parts]) if parts else "."
    return key, display


def _asset_category_from_folder_key(folder_key: str) -> str:
    parts = [p for p in str(folder_key or "").strip().lower().split("/") if p]
    if not parts:
        return "Other"

    domain = ""
    if len(parts) >= 3 and parts[0] == "assets" and parts[1] == "3d":
        domain = parts[2]
    else:
        domain = parts[0]

    if domain == "units":
        branch = parts[4] if len(parts) >= 5 else ""
        if branch in {"char", "veh", "vehicle", "tank", "ifv", "apc"}:
            return "Tanks & Vehicles"
        if branch in {"inf", "infantry", "soldier"}:
            return "Infantry"
        if branch in {"heli", "helo", "helicopter", "air", "plane"}:
            return "Air"
        if branch in {"art", "artillery"}:
            return "Artillery"
        return "Units Other"
    if domain == "decors":
        return "Decor"
    if domain in {"props", "prop"}:
        return "Props"
    if domain == "ammo":
        return "Ammo"
    if domain in {"fx", "effects"}:
        return "FX"
    return domain.replace("_", " ").title() or "Other"


def _asset_category_sort_key(name: str) -> tuple[int, str]:
    low = str(name or "").strip().lower()
    rank = {
        "tanks & vehicles": 0,
        "infantry": 1,
        "air": 2,
        "artillery": 3,
        "decor": 4,
        "props": 5,
        "ammo": 6,
        "fx": 7,
        "units other": 8,
    }.get(low, 50)
    return (rank, low)


def _build_asset_folders_cache(extractor_mod, groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        primary = str(g.get("primary", "")).strip()
        if not primary:
            continue
        lods = [str(v).strip() for v in g.get("lods", []) if str(v).strip()]
        folder_key, folder_display = _asset_folder_key_for_group(extractor_mod, primary)
        bucket = buckets.get(folder_key)
        if bucket is None:
            bucket = {
                "key": folder_key,
                "label": folder_display,
                "category": _asset_category_from_folder_key(folder_key),
                "models": [],
            }
            buckets[folder_key] = bucket
        bucket["models"].append(
            {
                "primary": primary,
                "lods": lods,
            }
        )

    out = list(buckets.values())
    out.sort(key=lambda x: str(x.get("key", "")))
    for item in out:
        models = item.get("models", [])
        if isinstance(models, list):
            models.sort(key=lambda m: _asset_picker_sort_key(extractor_mod, str(m.get("primary", ""))))
    return out


def _build_folder_tree_from_folders(folders: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_by_key: Dict[str, Dict[str, Any]] = {}
    roots: List[Dict[str, Any]] = []

    def ensure_node(key: str, name: str) -> Dict[str, Any]:
        node = node_by_key.get(key)
        if node is None:
            node = {"name": name, "key": key, "children": [], "models": []}
            node_by_key[key] = node
        return node

    for f in folders:
        if not isinstance(f, dict):
            continue
        key = str(f.get("key", "")).strip().lower()
        if not key:
            continue
        models = f.get("models", [])
        if not isinstance(models, list):
            models = []
        parts = [p for p in key.split("/") if p]
        parent: Dict[str, Any] | None = None
        for i, part in enumerate(parts):
            cur_key = "/".join(parts[: i + 1])
            cur = ensure_node(cur_key, part)
            if parent is None:
                if cur not in roots:
                    roots.append(cur)
            else:
                children = parent.get("children", [])
                if isinstance(children, list) and cur not in children:
                    children.append(cur)
            parent = cur
        if parent is not None:
            parent["models"] = [
                {"primary": str(m.get("primary", "")).strip(), "lods": [str(v).strip() for v in m.get("lods", []) if str(v).strip()]}
                for m in models
                if isinstance(m, dict) and str(m.get("primary", "")).strip()
            ]

    def norm_node(node: Dict[str, Any]) -> Dict[str, Any]:
        children_raw = node.get("children", [])
        children: List[Dict[str, Any]] = []
        if isinstance(children_raw, list):
            children = [norm_node(ch) for ch in children_raw if isinstance(ch, dict)]
            children.sort(key=lambda x: str(x.get("key", "")))
        models_raw = node.get("models", [])
        models: List[Dict[str, Any]] = []
        if isinstance(models_raw, list):
            for m in models_raw:
                if not isinstance(m, dict):
                    continue
                p = str(m.get("primary", "")).strip()
                if not p:
                    continue
                lods = [str(v).strip() for v in m.get("lods", []) if str(v).strip()]
                models.append({"primary": p, "lods": lods})
            models.sort(key=lambda m: str(m.get("primary", "")).lower())
        return {
            "name": str(node.get("name", "")).strip(),
            "key": str(node.get("key", "")).strip(),
            "children": children,
            "models": models,
        }

    out = [norm_node(n) for n in roots if isinstance(n, dict)]
    out.sort(key=lambda n: str(n.get("key", "")))
    return out


def _scan_assets_from_spk_paths(
    extractor_mod,
    spk_paths: Sequence[Path],
    query: str | None,
    game: "str | None" = None,
) -> List[str]:
    query_val = str(query or "").strip()
    query_for_scan = query_val if query_val else None
    seen_assets: set[str] = set()
    out: List[str] = []
    for spk_path in spk_paths:
        try:
            with extractor_mod.SpkMeshExtractor(spk_path, game=game) as spk:
                for asset, _meta in spk.find_matches(query_for_scan, None):
                    txt = str(asset).strip()
                    key = txt.lower()
                    if not txt or key in seen_assets:
                        continue
                    seen_assets.add(key)
                    out.append(txt)
        except Exception:
            continue
    out.sort(key=lambda a: _asset_picker_sort_key(extractor_mod, a))
    return out


def _build_asset_index_payload(extractor_mod, assets: Sequence[str], signature: Dict[str, Any], spk_count: int) -> Dict[str, Any]:
    groups = _build_asset_groups(extractor_mod, assets)
    folders = _build_asset_folders_cache(extractor_mod, groups)
    folder_tree = _build_folder_tree_from_folders(folders)
    return {
        "schema_version": 3,  # bumped: LOD orphan re-attach (group shared /lods/ with units)
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "signature": signature,
        "assets": [str(a).strip() for a in assets if str(a).strip()],
        "groups": groups,
        "folders": folders,
        "folder_tree": folder_tree,
        "counts": {
            "assets": len([a for a in assets if str(a).strip()]),
            "groups": len(groups),
            "folders": len(folders),
            "spk_count": int(spk_count),
        },
    }


def _normalize_search_tokens(query: str) -> List[str]:
    cleaned = CONTROL_CHAR_RX.sub(" ", str(query or "").strip().lower())
    cleaned = cleaned.replace("\\", "/")
    tokens = [t for t in re.split(r"[^a-z0-9]+", cleaned) if t]
    return tokens


def _filter_assets_from_index(index_data: Dict[str, Any], query: str) -> List[str]:
    assets_raw = index_data.get("assets", []) if isinstance(index_data, dict) else []
    assets = [str(a).strip() for a in assets_raw if str(a).strip()]
    tokens = _normalize_search_tokens(query)
    if not tokens:
        return assets
    out: List[str] = []
    for asset in assets:
        blob = f"{str(asset).lower()} {_asset_display_name(asset).lower()}"
        if all(tok in blob for tok in tokens):
            out.append(asset)
    return out


def _apply_picker_view_from_assets(
    settings: "WARNOImporterSettings",
    extractor_mod,
    assets: Sequence[str],
    *,
    source: str,
    query: str,
) -> None:
    groups = _build_asset_groups(extractor_mod, assets)
    folders = _build_asset_folders_cache(extractor_mod, groups)
    folder_tree = _build_folder_tree_from_folders(folders)
    _set_picker_view_cache(
        settings,
        assets=assets,
        groups=groups,
        folders=folders,
        folder_tree=folder_tree,
        source=source,
        query=query,
    )
    settings.asset_sync_lock = True
    try:
        if groups:
            primary = str(groups[0].get("primary", "")).strip()
            _safe_set_selected_asset_group(settings, primary)
            _safe_set_selected_asset_lod(settings, "__base__")
            _safe_set_selected_asset(settings, primary)
        elif assets:
            _safe_set_selected_asset(settings, str(assets[0]).strip())
            _safe_set_selected_asset_group(settings, "__none__")
            _safe_set_selected_asset_lod(settings, "__base__")
        else:
            _safe_set_selected_asset(settings, "")
            _safe_set_selected_asset_group(settings, "__none__")
            _safe_set_selected_asset_lod(settings, "__base__")
    finally:
        settings.asset_sync_lock = False
    _sync_group_lod_from_selected(settings)


def _apply_picker_view_from_index(
    settings: "WARNOImporterSettings",
    extractor_mod,
    index_data: Dict[str, Any],
    *,
    source: str,
    query: str,
) -> List[str]:
    assets_all = [str(a).strip() for a in index_data.get("assets", []) if str(a).strip()]
    filtered_assets = _filter_assets_from_index(index_data, query)
    tokens = _normalize_search_tokens(query)
    if not tokens:
        groups = index_data.get("groups", []) if isinstance(index_data.get("groups", []), list) else []
        folders = index_data.get("folders", []) if isinstance(index_data.get("folders", []), list) else []
        folder_tree = index_data.get("folder_tree", []) if isinstance(index_data.get("folder_tree", []), list) else []
        if not groups:
            groups = _build_asset_groups(extractor_mod, assets_all)
        if not folders:
            folders = _build_asset_folders_cache(extractor_mod, groups)
        if not folder_tree:
            folder_tree = _build_folder_tree_from_folders(folders)
        _set_picker_view_cache(
            settings,
            assets=assets_all,
            groups=groups,
            folders=folders,
            folder_tree=folder_tree,
            source=source,
            query="",
        )
    else:
        _apply_picker_view_from_assets(
            settings,
            extractor_mod,
            filtered_assets,
            source=source,
            query=query,
        )
    settings.asset_sync_lock = True
    try:
        if filtered_assets:
            if str(settings.selected_asset or "").strip() not in set(filtered_assets):
                _safe_set_selected_asset(settings, str(filtered_assets[0]).strip())
        elif assets_all:
            _safe_set_selected_asset(settings, str(assets_all[0]).strip())
        else:
            _safe_set_selected_asset(settings, "")
            _safe_set_selected_asset_group(settings, "__none__")
            _safe_set_selected_asset_lod(settings, "__base__")
    finally:
        settings.asset_sync_lock = False
    _sync_group_lod_from_selected(settings)
    return filtered_assets


def _ensure_asset_index_sync(
    settings: "WARNOImporterSettings",
    *,
    force_rebuild: bool = False,
) -> tuple[Dict[str, Any], str, Dict[str, Any], List[Path]]:
    extractor_mod = _extractor_module(settings)
    runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
    project_root = _project_root(settings)
    mesh_spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
    if not mesh_spk_paths:
        raise RuntimeError("No mesh SPK files found in prepared ZZ runtime.")

    signature = _asset_index_signature(runtime_info, mesh_spk_paths)
    index_path = _asset_index_path(settings)
    if not force_rebuild:
        cached = _load_asset_index_file(index_path)
        if cached is not None and _asset_index_signature_matches(cached, signature):
            return cached, "cache", runtime_info, mesh_spk_paths

    assets = _scan_assets_from_spk_paths(
        extractor_mod,
        mesh_spk_paths,
        query=None,
        game=_normalize_game_id(getattr(settings, "game_preset", "WARNO")),
    )
    payload = _build_asset_index_payload(extractor_mod, assets, signature, len(mesh_spk_paths))
    _save_asset_index_file(index_path, payload)
    return payload, "rebuilt", runtime_info, mesh_spk_paths


def _folder_root_key(folder_key: str) -> str:
    parts = [p for p in str(folder_key or "").strip().lower().split("/") if p]
    if len(parts) >= 3 and parts[0] == "assets" and parts[1] == "3d":
        return "/".join(parts[:3])
    if parts:
        return parts[0]
    return "__none__"


def _folder_root_label(root_key: str) -> str:
    parts = [p for p in str(root_key or "").split("/") if p]
    if not parts:
        return "<root>"
    tail = parts[-1]
    return tail.replace("_", " ").title()


def _browser_root_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__none__", "<scan assets first>", "No roots", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__none__", "<scan assets first>", "No roots", 0)]
    roots: List[str] = []
    seen: set[str] = set()
    for f in folders:
        key = _folder_root_key(str(f.get("key", "")).strip())
        if not key or key in seen:
            continue
        seen.add(key)
        roots.append(key)
    roots.sort()
    out = []
    for i, root in enumerate(roots):
        out.append((root, _folder_root_label(root), root, i))
    return out or [("__none__", "<scan assets first>", "No roots", 0)]


def _browser_category_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__all__", "<all categories>", "No categories", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__all__", "<all categories>", "No categories", 0)]
    root_key = str(getattr(self, "root_key", "") or "").strip().lower()
    categories: List[str] = []
    seen: set[str] = set()
    for folder in folders:
        key = str(folder.get("key", "")).strip().lower()
        if not key:
            continue
        if root_key and root_key != "__none__":
            if key != root_key and not key.startswith(root_key + "/"):
                continue
        cat = str(folder.get("category", "")).strip() or _asset_category_from_folder_key(key)
        if not cat:
            continue
        cat_low = cat.lower()
        if cat_low in seen:
            continue
        seen.add(cat_low)
        categories.append(cat)
    categories.sort(key=_asset_category_sort_key)
    out = [("__all__", "<all categories>", "Show all categories", 0)]
    for i, cat in enumerate(categories, start=1):
        out.append((cat, cat, cat, i))
    return out


def _update_browser_root_key(self, context):
    cat_items = _browser_category_items(self, context)
    cat_valid = {str(it[0]) for it in cat_items if isinstance(it, (tuple, list)) and len(it) >= 1}
    if str(getattr(self, "category_key", "") or "").strip() not in cat_valid:
        self.category_key = "__all__"
    items = _browser_folder_items(self, context)
    if not items:
        self.folder_key = "__none__"
        return
    valid = {str(it[0]) for it in items if isinstance(it, (tuple, list)) and len(it) >= 1}
    if str(getattr(self, "folder_key", "") or "").strip() not in valid:
        self.folder_key = str(items[0][0])


def _update_browser_category_key(self, context):
    items = _browser_folder_items(self, context)
    if not items:
        self.folder_key = "__none__"
        return
    valid = {str(it[0]) for it in items if isinstance(it, (tuple, list)) and len(it) >= 1}
    if str(getattr(self, "folder_key", "") or "").strip() not in valid:
        self.folder_key = str(items[0][0])


def _update_browser_folder_key(self, context):
    items = _browser_model_items(self, context)
    if not items:
        self.model_key = "__none__"
        return
    valid = {str(it[0]) for it in items if isinstance(it, (tuple, list)) and len(it) >= 1}
    if str(getattr(self, "model_key", "") or "").strip() not in valid:
        self.model_key = str(items[0][0])


def _update_browser_search_text(self, context):
    _update_browser_folder_key(self, context)


def _asset_group_enum_items(self: "WARNOImporterSettings", _context):
    groups = _asset_groups_from_cache(self)
    if not groups:
        return [("__none__", "<scan assets first>", "No scanned groups", 0)]
    assets = [str(g.get("primary", "")).strip() for g in groups if str(g.get("primary", "")).strip()]
    labels_map = _unique_asset_labels(assets)
    out = []
    for i, g in enumerate(groups):
        primary = str(g.get("primary", "")).strip()
        label = labels_map.get(primary, _asset_display_name(primary))
        out.append((primary, label, primary, i))
    return out


def _asset_lod_enum_items(self: "WARNOImporterSettings", _context):
    groups = _asset_groups_from_cache(self)
    if not groups:
        return [("__base__", "<no lods>", "No scanned LODs", 0)]
    group_key = str(getattr(self, "selected_asset_group", "") or "").strip()
    target = None
    for g in groups:
        if str(g.get("primary", "")).strip() == group_key:
            target = g
            break
    if target is None:
        target = groups[0]

    lod_assets = [str(v).strip() for v in target.get("lods", []) if str(v).strip()]
    labels_map = _unique_asset_labels(lod_assets)
    out = [("__base__", "<base model>", "Use base model (not LOD)", 0)]
    for i, lod in enumerate(lod_assets, start=1):
        lod_txt = str(lod or "").strip()
        if not lod_txt:
            continue
        label = labels_map.get(lod_txt, _asset_display_name(lod_txt))
        out.append((lod_txt, label, lod_txt, i))
    return out


def _sync_group_lod_from_selected(settings: "WARNOImporterSettings"):
    groups = _asset_groups_from_cache(settings)
    if not groups:
        return
    sel = str(settings.selected_asset or "").strip()
    for g in groups:
        primary = str(g.get("primary", "")).strip()
        lods = [str(v).strip() for v in g.get("lods", []) if str(v).strip()]
        if sel == primary:
            _safe_set_selected_asset_group(settings, primary)
            _safe_set_selected_asset_lod(settings, "__base__")
            return
        if sel in lods:
            _safe_set_selected_asset_group(settings, primary)
            _safe_set_selected_asset_lod(settings, sel)
            return
    _safe_set_selected_asset_group(settings, str(groups[0].get("primary", "")).strip())
    _safe_set_selected_asset_lod(settings, "__base__")


def _update_selected_asset(self: "WARNOImporterSettings", _context):
    if bool(getattr(self, "asset_sync_lock", False)):
        return
    self.asset_sync_lock = True
    try:
        _sync_group_lod_from_selected(self)
    finally:
        self.asset_sync_lock = False


def _update_selected_asset_group(self: "WARNOImporterSettings", _context):
    if bool(getattr(self, "asset_sync_lock", False)):
        return
    self.asset_sync_lock = True
    try:
        primary = str(self.selected_asset_group or "").strip()
        if primary and primary != "__none__":
            _safe_set_selected_asset(self, primary)
            _safe_set_selected_asset_lod(self, "__base__")
    finally:
        self.asset_sync_lock = False


def _update_selected_asset_lod(self: "WARNOImporterSettings", _context):
    if bool(getattr(self, "asset_sync_lock", False)):
        return
    self.asset_sync_lock = True
    try:
        lod = str(self.selected_asset_lod or "").strip()
        primary = str(self.selected_asset_group or "").strip()
        if lod and lod != "__base__":
            _safe_set_selected_asset(self, lod)
        elif primary and primary != "__none__":
            _safe_set_selected_asset(self, primary)
    finally:
        self.asset_sync_lock = False


# =====================================================================
# Tree-view Asset Browser
# =====================================================================
# A flat-list UIList that visually renders the asset folder tree from
# mesh_assets_index.v2.json with expand/collapse, depth-based indent,
# folder counters and a live search filter. Replaces the legacy 5-dropdown
# WARNO_OT_PickAssetBrowser (kept registered for backward compatibility).


class WARNOBrowserNode(PropertyGroup):
    """One row in the tree-view UIList. The full tree is materialised as a
    flat collection so Blender's UIList virtual rendering can scroll it
    efficiently even for 7000+ entries."""
    display_name: StringProperty()
    path_key: StringProperty()
    is_folder: BoolProperty()
    is_model: BoolProperty()
    is_lod: BoolProperty()
    is_expanded: BoolProperty()
    depth: IntProperty(default=0)
    parent_key: StringProperty()
    model_primary: StringProperty()
    lod_variant: StringProperty()
    asset_count: IntProperty(default=0)


def _browser_count_models(node: Dict[str, Any]) -> int:
    """Recursively count model entries under a folder_tree node."""
    if not isinstance(node, dict):
        return 0
    count = 0
    models = node.get("models", [])
    if isinstance(models, list):
        count += len(models)
    children = node.get("children", [])
    if isinstance(children, list):
        for ch in children:
            count += _browser_count_models(ch)
    return count


def _browser_node_matches_search(node: Dict[str, Any], search_low: str) -> bool:
    """True if this node, any of its descendants, or any of its model
    primaries contains the lowercased search token."""
    if not search_low:
        return True
    if not isinstance(node, dict):
        return False
    name_low = str(node.get("name", "")).lower()
    key_low = str(node.get("key", "")).lower()
    if search_low in name_low or search_low in key_low:
        return True
    models = node.get("models", [])
    if isinstance(models, list):
        for m in models:
            if not isinstance(m, dict):
                continue
            if search_low in str(m.get("primary", "")).lower():
                return True
            for lod in m.get("lods", []) or []:
                if search_low in str(lod).lower():
                    return True
    children = node.get("children", [])
    if isinstance(children, list):
        for ch in children:
            if _browser_node_matches_search(ch, search_low):
                return True
    return False


def _browser_auto_expand_keys(folder_tree: Sequence[Dict[str, Any]]) -> set[str]:
    """Folder keys to auto-expand on first open: every root, plus any linear
    single-child folder chain, descending until the first folder that branches
    or holds models. This makes deeply-nested single-root layouts (Wargame Red
    Dragon / Steel Division 2: assets/3d/units/...) reveal their first useful
    level instead of opening as one collapsed folder. WARNO trees that branch at
    the root are unaffected (only their roots get expanded, as before)."""
    out: set[str] = set()

    def walk(node: Dict[str, Any]) -> None:
        if not isinstance(node, dict):
            return
        key = str(node.get("key", "")).strip()
        if not key:
            return
        out.add(key)
        children = [c for c in (node.get("children", []) or []) if isinstance(c, dict)]
        models = node.get("models", []) or []
        if len(children) == 1 and not models:
            walk(children[0])

    for root in folder_tree or []:
        walk(root)
    return out


def _browser_serialize_expanded(settings) -> set[str]:
    raw = str(getattr(settings, "browser_expanded_keys", "") or "[]")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {str(s) for s in data if isinstance(s, str)}
    except Exception:
        pass
    return set()


def _browser_save_expanded(settings, expanded: set[str]) -> None:
    try:
        settings.browser_expanded_keys = json.dumps(sorted(expanded))
    except Exception:
        pass


def _browser_flatten_tree(
    tree: Sequence[Dict[str, Any]],
    expanded: set[str],
    search_low: str,
    show_lods: bool,
) -> List[Dict[str, Any]]:
    """Linearise the recursive folder_tree into a flat list of dicts ready to
    populate the CollectionProperty. When `search_low` is non-empty, branches
    that have no descendant match are pruned and matching branches are
    auto-expanded so results are immediately visible."""
    out: List[Dict[str, Any]] = []
    auto_expand = bool(search_low)

    def _node_model_matches(model_primary: str, lods: Sequence[str]) -> bool:
        if not search_low:
            return True
        if search_low in model_primary.lower():
            return True
        for lod in lods:
            if search_low in str(lod).lower():
                return True
        return False

    def visit(node: Dict[str, Any], depth: int, parent_key: str) -> None:
        if not isinstance(node, dict):
            return
        if search_low and not _browser_node_matches_search(node, search_low):
            return
        path_key = str(node.get("key", "")).strip()
        name = str(node.get("name", "")).strip() or path_key.rsplit("/", 1)[-1]
        is_expanded = auto_expand or (path_key in expanded)
        children = node.get("children", []) if isinstance(node.get("children", []), list) else []
        models = node.get("models", []) if isinstance(node.get("models", []), list) else []
        out.append({
            "display_name": name,
            "path_key": path_key,
            "is_folder": True,
            "is_model": False,
            "is_lod": False,
            "is_expanded": is_expanded,
            "depth": depth,
            "parent_key": parent_key,
            "model_primary": "",
            "lod_variant": "",
            "asset_count": _browser_count_models(node),
        })
        if not is_expanded:
            return
        for ch in children:
            visit(ch, depth + 1, path_key)
        for m in models:
            if not isinstance(m, dict):
                continue
            primary = str(m.get("primary", "")).strip()
            if not primary:
                continue
            lods = [str(v).strip() for v in m.get("lods", []) if str(v).strip()]
            if not _node_model_matches(primary, lods):
                continue
            label = primary.rsplit("/", 1)[-1]
            if label.lower().endswith(".fbx"):
                label = label[:-4]
            out.append({
                "display_name": label,
                "path_key": primary,
                "is_folder": False,
                "is_model": True,
                "is_lod": False,
                "is_expanded": False,
                "depth": depth + 1,
                "parent_key": path_key,
                "model_primary": primary,
                "lod_variant": "",
                "asset_count": len(lods) + 1,
            })
            if show_lods and lods:
                for lod in lods:
                    lod_label = lod.rsplit("/", 1)[-1]
                    if lod_label.lower().endswith(".fbx"):
                        lod_label = lod_label[:-4]
                    out.append({
                        "display_name": lod_label,
                        "path_key": lod,
                        "is_folder": False,
                        "is_model": False,
                        "is_lod": True,
                        "is_expanded": False,
                        "depth": depth + 2,
                        "parent_key": primary,
                        "model_primary": primary,
                        "lod_variant": lod,
                        "asset_count": 0,
                    })

    for root in tree or []:
        visit(root, 0, "")
    return out


def _browser_populate_nodes(settings, index_data: Dict[str, Any]) -> None:
    """Refresh `settings.browser_nodes` from the cached folder_tree payload."""
    tree = []
    if isinstance(index_data, dict):
        tree_raw = index_data.get("folder_tree", [])
        if isinstance(tree_raw, list):
            tree = tree_raw
    expanded = _browser_serialize_expanded(settings)
    search = str(getattr(settings, "browser_search", "") or "").strip().lower()
    show_lods = bool(getattr(settings, "browser_show_lods", True))
    flat = _browser_flatten_tree(tree, expanded, search, show_lods)
    nodes = settings.browser_nodes
    nodes.clear()
    for item in flat:
        n = nodes.add()
        for key, value in item.items():
            try:
                setattr(n, key, value)
            except Exception:
                pass
    n_total = len(nodes)
    if settings.browser_active_index >= n_total:
        settings.browser_active_index = max(0, n_total - 1)


def _browser_refresh_from_settings(settings) -> None:
    """Try to refresh browser_nodes from the on-disk asset index without
    triggering a full rescan; safe no-op when nothing is cached yet."""
    try:
        index_path = _asset_index_path(settings)
        cached = _load_asset_index_file(index_path)
        if cached is None:
            return
        _browser_populate_nodes(settings, cached)
    except Exception:
        pass


def _browser_update_search(self, _context):
    """PropertyUpdate callback: re-flatten tree on every keystroke."""
    _browser_refresh_from_settings(self)


def _browser_update_show_lods(self, _context):
    _browser_refresh_from_settings(self)


# ---------------------------------------------------------------------------
# Multi-game preset support (WARNO / Wargame Red Dragon / Steel Division 2).
#
# Each game keeps its own install folder + isolated extraction/cache dirs in a
# ``game_presets`` section of config.json. Tooling (moddingSuite, CLIs, wrappers)
# and import options stay shared at the top level. The live PropertyGroup always
# reflects the currently selected game; switching the preset flushes the outgoing
# game's fields to the section, loads the incoming section, and persists.
# A legacy flat config with no ``game_presets`` is treated as the WARNO preset,
# so WARNO behaviour is unchanged.
# ---------------------------------------------------------------------------
VALID_GAME_IDS = ("WARNO", "WARGAME_RD", "STEEL_DIVISION_2")

# Per-game (isolated) fields. Everything else in config.json is shared.
_PER_GAME_FIELDS = (
    "warno_root",
    "cache_dir",
    "zz_runtime_dir",
    "atlas_assets_dir",
    "spk_path",
    "skeleton_spk",
)

# Known default install paths, used to pre-seed a game section on first switch.
_BUILTIN_DEFAULT_INSTALL = {
    "WARNO": r"F:\SteamLibrary\steamapps\common\WARNO",
    "WARGAME_RD": r"C:\Program Files (x86)\Steam\steamapps\common\Wargame Red Dragon",
    "STEEL_DIVISION_2": r"C:\Program Files (x86)\Steam\steamapps\common\Steel Division 2",
}


def _normalize_game_id(game_id) -> str:
    gid = str(game_id or "WARNO").strip().upper()
    return gid if gid in VALID_GAME_IDS else "WARNO"


def _default_game_section(game_id: str) -> Dict[str, Any]:
    """Seed defaults for a game section. WARNO keeps the historical flat defaults
    (so a fresh WARNO setup is byte-for-byte unchanged); other games get isolated
    cache/runtime subfolders so their extracted data never collides with WARNO."""
    gid = _normalize_game_id(game_id)
    install = _BUILTIN_DEFAULT_INSTALL.get(gid, "")
    if gid == "WARNO":
        cache_dir = "output_blender"
        zz_runtime_dir = "out_blender_runtime/zz_runtime"
    else:
        slug = gid.lower()
        cache_dir = f"output_blender/{slug}"
        zz_runtime_dir = f"out_blender_runtime/zz_runtime/{slug}"
    return {
        "warno_root": install,
        "cache_dir": cache_dir,
        "zz_runtime_dir": zz_runtime_dir,
        "atlas_assets_dir": "",
        "spk_path": "spk/Mesh_All.spk",
        "skeleton_spk": "skeletonsspk/Skeleton_All.spk",
    }


def _collect_live_game_section(settings) -> Dict[str, Any]:
    return {k: str(getattr(settings, k, "") or "") for k in _PER_GAME_FIELDS}


def _apply_game_section(settings, section: Dict[str, Any]) -> None:
    if not isinstance(section, dict):
        return
    for k in _PER_GAME_FIELDS:
        if k in section:
            try:
                setattr(settings, k, str(section.get(k, "") or ""))
            except Exception:
                pass


def _read_config_dict(path: Path) -> Dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_config_dict(path: Path, data: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _switch_game_preset(settings) -> None:
    """EnumProperty flow: flush the outgoing game's live fields into its section,
    load the incoming section into the live fields, and persist to config.json."""
    cfg = _config_path(settings)
    raw = _read_config_dict(cfg)
    presets = raw.get("game_presets")
    if not isinstance(presets, dict):
        presets = {}
    outgoing = _normalize_game_id(getattr(settings, "active_game_loaded", "WARNO"))
    incoming = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))

    # 1) flush current live per-game fields into the outgoing section
    presets[outgoing] = _collect_live_game_section(settings)
    # 2) ensure incoming section exists (seed defaults the first time)
    if not isinstance(presets.get(incoming), dict):
        presets[incoming] = _default_game_section(incoming)
    # 3) load incoming section into live fields
    _apply_game_section(settings, presets[incoming])
    settings.active_game_loaded = incoming
    # 4) persist: active game + all sections + mirror active section to flat keys
    raw["active_game"] = incoming
    raw["game_presets"] = presets
    for k, v in presets[incoming].items():
        raw[k] = v
    _write_config_dict(cfg, raw)
    _enforce_fixed_runtime_defaults(settings)


def _on_game_preset_changed(self, _context):
    """Update callback for game_preset; guarded so config-driven assignment
    (during load) does not trigger a spurious flush/load."""
    if bool(getattr(self, "game_switch_lock", False)):
        return
    self.game_switch_lock = True
    try:
        _switch_game_preset(self)
    except Exception:
        pass
    finally:
        self.game_switch_lock = False


class WARNOImporterSettings(PropertyGroup):
    game_preset: EnumProperty(
        name="Game",
        description="Which Eugen Systems game to import from (each remembers its own folder + cache)",
        items=[
            ("WARNO", "WARNO", "Import from WARNO"),
            ("WARGAME_RD", "Wargame Red Dragon", "Import from Wargame Red Dragon"),
            ("STEEL_DIVISION_2", "Steel Division 2 (soon)", "Import from Steel Division 2"),
        ],
        default="WARNO",
        update=_on_game_preset_changed,
    )
    game_switch_lock: BoolProperty(default=False, options={"HIDDEN"})
    active_game_loaded: StringProperty(default="WARNO", options={"HIDDEN"})
    project_root: StringProperty(
        name="Project Root",
        description="Folder that contains warno_spk_extract.py and config.json",
        subtype="DIR_PATH",
        default=DEFAULT_PROJECT_ROOT,
    )
    spk_path: StringProperty(
        name="Mesh SPK / Folder",
        subtype="FILE_PATH",
        default="spk/Mesh_All.spk",
        description="Path to one SPK file or folder with many .spk files",
    )
    skeleton_spk: StringProperty(
        name="Skeleton SPK / Folder",
        subtype="FILE_PATH",
        default="skeletonsspk/Skeleton_All.spk",
        description="Path to one skeleton SPK file or folder with many skeleton .spk files",
    )
    # Deprecated: NDF hints are intentionally disabled (heuristic naming was unreliable).
    unit_ndfbin: StringProperty(name="Unit NDF", subtype="FILE_PATH", default="", options={"HIDDEN"})
    atlas_assets_dir: StringProperty(
        name="Texture Assets",
        description="Folder that contains extracted WARNO texture assets",
        subtype="DIR_PATH",
        default="",
    )
    tgv_converter: StringProperty(name="TGV Converter", subtype="FILE_PATH", default="tgv_to_png.py")
    modding_suite_atlas_wrapper: StringProperty(
        name="ModdingSuite Atlas Wrapper",
        subtype="FILE_PATH",
        default="modding_suite_atlas_export.py",
        description="Wrapper script that exports Atlas metadata JSON via ModdingSuite",
        options={"HIDDEN"},
    )
    modding_suite_atlas_cli: StringProperty(
        name="ModdingSuite Atlas CLI",
        subtype="FILE_PATH",
        default="moddingSuite/moddingSuite.exe",
        description="Path to headless Atlas CLI executable (moddingSuite.AtlasCli.exe)",
        options={"HIDDEN"},
    )
    modding_suite_gfx_wrapper: StringProperty(
        name="ModdingSuite GFX Wrapper",
        subtype="FILE_PATH",
        default="modding_suite_gfx_export.py",
        description="Wrapper script that exports GFX semantic JSON via ModdingSuite",
        options={"HIDDEN"},
    )
    modding_suite_gfx_cli: StringProperty(
        name="ModdingSuite GFX CLI",
        subtype="FILE_PATH",
        default="moddingSuite/moddingSuite.exe",
        description="Path to headless GFX CLI executable (moddingSuite.GfxCli.exe)",
        options={"HIDDEN"},
    )
    depiction_vehicles_ndf: StringProperty(
        name="DepictionVehicles NDF",
        subtype="FILE_PATH",
        default="",
        description="Optional explicit text DepictionVehicles.ndf override used for semantic bones/FX filtering",
        options={"HIDDEN"},
    )
    depiction_operators_ndf: StringProperty(
        name="DepictionOperators NDF",
        subtype="FILE_PATH",
        default="",
        description="Optional explicit text DepictionOperators.ndf override used to resolve exact operator node names",
        options={"HIDDEN"},
    )
    import_semantic_mode: EnumProperty(
        name="Semantic Mode",
        items=[
            ("REFERENCE", "Reference", "Only create nodes backed by exact semantic contracts or proven deformation"),
            ("RAW_DEBUG", "Raw debug", "Create the full raw SPK helper tree for debugging"),
        ],
        default="REFERENCE",
        description="Controls whether import prefers reference parity or raw SPK debugging output",
    )
    use_atlas_json_mapping: BoolProperty(
        name="Use Atlas JSON (ModdingSuite)",
        default=True,
        description="Resolve crop + naming from Atlas JSON exported by ModdingSuite wrapper",
        options={"HIDDEN"},
    )
    use_gfx_json_manifest: BoolProperty(
        name="Use GFX JSON (ModdingSuite)",
        default=True,
        description="Resolve turret/fx/operator semantics from compiled GFX JSON exported by ModdingSuite wrapper",
        options={"HIDDEN"},
    )
    atlas_json_strict: BoolProperty(
        name="Atlas JSON strict mode",
        default=True,
        description="If enabled, unresolved Atlas JSON mappings fail without fallback guessing",
        options={"HIDDEN"},
    )
    atlas_json_cache_subdir: StringProperty(
        name="Atlas JSON cache",
        default="atlas_json_cache",
        description="Subfolder inside cache dir to store atlas JSON maps",
        options={"HIDDEN"},
    )
    gfx_json_cache_subdir: StringProperty(
        name="GFX JSON cache",
        default="gfx_json_cache",
        description="Subfolder inside cache dir to store GFX semantic JSON manifests",
        options={"HIDDEN"},
    )
    texture_subdir: StringProperty(name="Texture Subdir", default="textures")
    cache_dir: StringProperty(name="Cache Dir", subtype="DIR_PATH", default="output_blender")
    use_zz_dat_source: BoolProperty(
        name="Use ZZ.dat source",
        default=False,
        description="Auto-read Mesh/Skeleton/NDF/textures from WARNO ZZ.dat packages",
    )
    warno_root: StringProperty(
        name="WARNO Folder",
        subtype="DIR_PATH",
        default="",
        description="WARNO game root folder that contains ZZ*.dat packages",
    )
    modding_suite_root: StringProperty(
        name="moddingSuite Root",
        subtype="DIR_PATH",
        default="moddingSuite",
        description="Folder with the bundled moddingSuite.exe (auto-detected; always the 'moddingSuite' folder next to the plugin)",
    )
    zz_runtime_dir: StringProperty(
        name="ZZ Runtime Cache",
        subtype="DIR_PATH",
        default="out_blender_runtime/zz_runtime",
        description="Local cache where files extracted from ZZ.dat are stored",
    )

    query: StringProperty(name="Query", default="Leopard_1A1")
    match_limit: IntProperty(name="Match Limit", default=200, min=1, max=2000)
    match_cache_json: StringProperty(default="[]", options={"HIDDEN"})
    asset_group_cache_json: StringProperty(default="[]", options={"HIDDEN"})
    asset_folder_cache_json: StringProperty(default="[]", options={"HIDDEN"})
    asset_sync_lock: BoolProperty(default=False, options={"HIDDEN"})

    # ---- Tree-view Asset Browser state (persisted per-scene) ----
    browser_nodes: CollectionProperty(type=WARNOBrowserNode, options={"HIDDEN"})
    browser_active_index: IntProperty(default=0, options={"HIDDEN"})
    browser_expanded_keys: StringProperty(
        default="[]",
        options={"HIDDEN"},
        description="JSON list of folder path_keys currently expanded in the tree browser",
    )
    browser_search: StringProperty(
        name="Search",
        default="",
        description="Filter the asset tree by name (case-insensitive). Auto-expands matching branches.",
        update=_browser_update_search,
    )
    browser_show_lods: BoolProperty(
        name="Show LODs",
        default=False,
        description="Show LOW/MID LOD variants under each model in the tree",
        update=_browser_update_show_lods,
    )
    selected_asset: StringProperty(
        name="Asset",
        description="Asset path to import",
        default="",
        update=_update_selected_asset,
    )
    selected_asset_group: StringProperty(
        name="Main Asset",
        description="Main (base) asset",
        default="__none__",
        update=_update_selected_asset_group,
    )
    selected_asset_lod: StringProperty(
        name="LOD Asset",
        description="LOD variant for selected main asset",
        default="__base__",
        update=_update_selected_asset_lod,
    )
    show_asset_lods: BoolProperty(
        name="Show LODs",
        default=False,
        description="Expand to pick LOD variant under selected main asset",
    )
    show_first_setup_logs: BoolProperty(
        name="First Setup / Logs",
        default=False,
        description="Show initial setup (sources/deps) and logging controls",
    )
    show_project_section: BoolProperty(
        name="Project",
        default=True,
        description="Show/hide project settings section",
    )
    show_textures_section: BoolProperty(
        name="Textures",
        default=True,
        description="Show/hide textures section",
    )
    show_asset_picker_section: BoolProperty(
        name="Asset Picker",
        default=True,
        description="Show/hide asset picker section",
    )
    show_import_options: BoolProperty(
        name="Import Options",
        default=True,
        description="Show/hide import options",
    )

    auto_textures: BoolProperty(name="Auto textures", default=True)
    fast_exact_texture_resolve: BoolProperty(
        name="Fast exact texture resolve",
        default=True,
        description="Resolve only exact refs and exact companion maps (faster, avoids long folder scans)",
    )
    texture_process_timeout_sec: IntProperty(
        name="Converter timeout (sec)",
        default=120,
        min=10,
        max=1800,
        description="Timeout for each converter process",
    )
    atlas_cli_timeout_sec: IntProperty(
        name="Atlas CLI timeout (sec)",
        default=45,
        min=5,
        max=600,
        description="Timeout for headless atlas metadata export",
        options={"HIDDEN"},
    )
    gfx_cli_timeout_sec: IntProperty(
        name="GFX CLI timeout (sec)",
        default=180,
        min=5,
        max=600,
        description="Timeout for headless GFX semantic export",
        options={"HIDDEN"},
    )
    texture_stage_timeout_sec: IntProperty(
        name="Texture stage timeout (sec)",
        default=240,
        min=30,
        max=3600,
        description="Global timeout for full texture resolving stage",
    )
    auto_install_tgv_deps: BoolProperty(
        name="Auto-install TGV deps",
        default=True,
        description="Auto-install Pillow/zstandard for converter on first missing dependency error",
    )
    tgv_deps_dir: StringProperty(
        name="TGV deps dir",
        subtype="DIR_PATH",
        default=".warno_pydeps",
        description="Project-local folder for converter Python deps",
    )
    auto_split_main_parts: BoolProperty(
        name="Auto split main parts",
        default=True,
        description="Split model by top-level bone groups, including wheels",
    )
    auto_name_parts: BoolProperty(
        name="Auto part naming",
        default=False,
        description="Use detected group names for imported objects",
    )
    auto_name_materials: BoolProperty(name="Auto material naming", default=True)
    auto_pull_bones: BoolProperty(
        name="Auto pull bones",
        default=False,
        description="Build helper armature from parsed bone names and parent imported parts to bones",
    )

    # Deprecated: TGV mirroring is always OFF.
    tgv_mirror: BoolProperty(name="Mirror TGV", default=False, options={"HIDDEN"})
    auto_rename_textures: BoolProperty(
        name="Auto texture naming",
        default=True,
        description="Rename resolved textures to model-style names (Unit_D, Unit_NM, ...)",
    )
    use_ao_multiply: BoolProperty(
        name="AO multiply with diffuse",
        default=True,
        description="If enabled, AO is multiplied into Base Color; otherwise AO texture stays unconnected",
    )
    normal_invert_mode: EnumProperty(
        name="Normal invert",
        items=(
            ("none", "None", "Use normal map as-is"),
            ("invert_green", "Invert Green (Y)", "Invert green channel before Normal Map node"),
            ("invert_rgb", "Invert RGB", "Invert all RGB channels before Normal Map node"),
        ),
        default="none",
    )
    log_to_file: BoolProperty(
        name="Log to file",
        default=True,
        description="Write import/texture stages into project log file",
    )
    log_file_name: StringProperty(
        name="Log file",
        default="warno_import.log",
        description="Log file name in project root",
    )
    # Deprecated: rotations are fixed to zero in import flow.
    rotate_x: FloatProperty(name="Rotate X", default=0.0, options={"HIDDEN"})
    rotate_y: FloatProperty(name="Rotate Y", default=0.0, options={"HIDDEN"})
    rotate_z: FloatProperty(name="Rotate Z", default=0.0, options={"HIDDEN"})
    # Deprecated: Mirror Y is always ON.
    mirror_y: BoolProperty(name="Mirror Y", default=True, options={"HIDDEN"})

    use_merge_by_distance: BoolProperty(name="Merge by distance", default=False)
    merge_distance: FloatProperty(
        name="Merge distance (m)",
        default=0.0001,
        min=0.0,
        precision=6,
        description="Distance threshold in meters for merge-by-distance",
    )
    auto_smooth_mode: EnumProperty(
        name="Auto Smooth",
        items=[
            ("MODIFIER", "Shade auto smooth as modifier", "Add Auto Smooth modifier and keep it editable"),
            ("OFF", "No auto smooth", "Do not apply auto smooth"),
            ("APPLY", "Shade auto smooth and apply effect", "Apply auto smooth effect directly to imported meshes"),
        ],
        default="OFF",
        description="Choose how auto smooth should behave after import",
    )
    auto_smooth_angle: FloatProperty(name="Smooth angle", default=30.0, min=0.0, max=180.0)
    last_texture_dir: StringProperty(name="Last texture dir", subtype="DIR_PATH", default="", options={"HIDDEN"})
    last_import_collection: StringProperty(name="Last import collection", default="", options={"HIDDEN"})
    startup_state_restored: BoolProperty(default=False, options={"HIDDEN"})

    status: StringProperty(name="Status", default="")


FIXED_MERGE_DISTANCE = 0.0001
FIXED_AUTO_SMOOTH_ANGLE = 30.0


def _enforce_fixed_runtime_defaults(settings: WARNOImporterSettings) -> None:
    settings.import_semantic_mode = "REFERENCE" if str(settings.import_semantic_mode or "").upper() not in {"REFERENCE", "RAW_DEBUG"} else settings.import_semantic_mode
    if str(getattr(settings, "auto_smooth_mode", "OFF") or "").upper() not in {"MODIFIER", "OFF", "APPLY"}:
        settings.auto_smooth_mode = "OFF"
    settings.auto_textures = True
    settings.auto_rename_textures = True
    settings.fast_exact_texture_resolve = True
    settings.use_zz_dat_source = True
    # WARNO atlas + GFX pipelines are always on (restores 1.5.0 behavior). The
    # RD/SD2 texture/mesh paths early-return BEFORE the atlas gate and never read
    # these flags, so forcing them True is WARNO-only in effect and RD-safe.
    settings.use_atlas_json_mapping = True
    settings.use_gfx_json_manifest = True
    settings.atlas_json_strict = True
    settings.use_ao_multiply = False
    settings.normal_invert_mode = "none"
    try:
        settings.merge_distance = max(0.0, float(settings.merge_distance))
    except Exception:
        settings.merge_distance = float(FIXED_MERGE_DISTANCE)
    settings.auto_smooth_angle = float(FIXED_AUTO_SMOOTH_ANGLE)
    try:
        settings.gfx_cli_timeout_sec = max(180, int(settings.gfx_cli_timeout_sec))
    except Exception:
        settings.gfx_cli_timeout_sec = 180
    _apply_modding_suite_runtime_defaults(settings)


def _load_config_into_settings(settings: WARNOImporterSettings, path: Path) -> tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, f"Config not found: {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return False, f"Config parse failed: {exc}"
    if not isinstance(raw, dict):
        return False, f"Config has invalid format: {path}"

    def get_bool(key: str, default: bool) -> bool:
        val = raw.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            low = val.strip().lower()
            if low in {"1", "true", "yes", "on", "y"}:
                return True
            if low in {"0", "false", "no", "off", "n"}:
                return False
        return default

    def get_float(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except Exception:
            return default

    def get_text(key: str, default: str = "") -> str:
        return str(raw.get(key, default) or "").strip()

    settings.spk_path = get_text("spk_path", settings.spk_path)
    settings.skeleton_spk = get_text("skeleton_spk", settings.skeleton_spk)
    settings.project_root = get_text("project_root", settings.project_root) or settings.project_root
    settings.atlas_assets_dir = get_text("atlas_assets_dir", settings.atlas_assets_dir)
    settings.tgv_converter = get_text("tgv_converter", settings.tgv_converter) or "tgv_to_png.py"
    settings.modding_suite_atlas_wrapper = "modding_suite_atlas_export.py"
    settings.modding_suite_atlas_cli = "moddingSuite/moddingSuite.exe"
    settings.modding_suite_gfx_wrapper = "modding_suite_gfx_export.py"
    settings.modding_suite_gfx_cli = "moddingSuite/moddingSuite.exe"
    settings.depiction_vehicles_ndf = ""
    settings.depiction_operators_ndf = ""
    semantic_mode = get_text("import_semantic_mode", settings.import_semantic_mode).upper()
    settings.import_semantic_mode = semantic_mode if semantic_mode in {"REFERENCE", "RAW_DEBUG"} else "REFERENCE"
    settings.use_atlas_json_mapping = True
    settings.use_gfx_json_manifest = True
    settings.atlas_json_strict = True
    settings.atlas_json_cache_subdir = "atlas_json_cache"
    settings.gfx_json_cache_subdir = "gfx_json_cache"
    settings.texture_subdir = "textures"
    settings.cache_dir = get_text("cache_dir", settings.cache_dir) or settings.cache_dir
    settings.use_zz_dat_source = get_bool("use_zz_dat_source", settings.use_zz_dat_source)
    settings.warno_root = get_text("warno_root", settings.warno_root)
    settings.modding_suite_root = get_text("modding_suite_root", settings.modding_suite_root)
    settings.zz_runtime_dir = get_text("zz_runtime_dir", settings.zz_runtime_dir)

    settings.auto_textures = get_bool("auto_textures", settings.auto_textures)
    settings.fast_exact_texture_resolve = get_bool("fast_exact_texture_resolve", settings.fast_exact_texture_resolve)
    try:
        settings.texture_process_timeout_sec = int(raw.get("texture_process_timeout_sec", settings.texture_process_timeout_sec))
    except Exception:
        pass
    try:
        settings.atlas_cli_timeout_sec = int(raw.get("atlas_cli_timeout_sec", settings.atlas_cli_timeout_sec))
    except Exception:
        pass
    try:
        settings.gfx_cli_timeout_sec = int(raw.get("gfx_cli_timeout_sec", settings.gfx_cli_timeout_sec))
    except Exception:
        pass
    try:
        settings.texture_stage_timeout_sec = int(raw.get("texture_stage_timeout_sec", settings.texture_stage_timeout_sec))
    except Exception:
        pass
    settings.auto_install_tgv_deps = get_bool("auto_install_tgv_deps", settings.auto_install_tgv_deps)
    settings.tgv_deps_dir = get_text("tgv_deps_dir", settings.tgv_deps_dir) or ".warno_pydeps"
    settings.auto_split_main_parts = get_bool("auto_split_main_parts", get_bool("split_bone_parts", settings.auto_split_main_parts))
    settings.auto_name_parts = get_bool("auto_name_parts", settings.auto_name_parts)
    settings.auto_name_materials = get_bool("auto_name_materials", settings.auto_name_materials)
    settings.auto_pull_bones = get_bool("auto_pull_bones", settings.auto_pull_bones)
    settings.use_merge_by_distance = get_bool("fbx_use_merge_by_distance", settings.use_merge_by_distance)
    settings.merge_distance = get_float("fbx_merge_distance", settings.merge_distance)
    auto_smooth_mode = get_text("fbx_auto_smooth_mode", "")
    if auto_smooth_mode:
        settings.auto_smooth_mode = str(auto_smooth_mode).upper()
    else:
        settings.auto_smooth_mode = "APPLY" if get_bool("fbx_use_auto_smooth", False) else "OFF"
    settings.auto_smooth_angle = get_float("fbx_auto_smooth_angle", settings.auto_smooth_angle)
    settings.tgv_mirror = False
    settings.auto_rename_textures = get_bool("auto_rename_textures", settings.auto_rename_textures)
    settings.use_ao_multiply = get_bool("ao_multiply_diffuse", settings.use_ao_multiply)
    settings.normal_invert_mode = get_text("normal_invert_mode", settings.normal_invert_mode) or settings.normal_invert_mode
    if settings.normal_invert_mode not in {"none", "invert_green", "invert_rgb"}:
        settings.normal_invert_mode = "none"
    settings.log_to_file = get_bool("log_to_file", settings.log_to_file)
    settings.log_file_name = get_text("log_file_name", settings.log_file_name) or "warno_import.log"
    settings.rotate_x = 0.0
    settings.rotate_y = 0.0
    settings.rotate_z = 0.0
    settings.mirror_y = True

    # --- Multi-game preset layer (backward compatible) ---
    presets = raw.get("game_presets")
    active = _normalize_game_id(raw.get("active_game", ""))
    if not isinstance(presets, dict) or not presets:
        # Legacy flat config: the flat per-game values ARE the WARNO preset, so
        # an old WARNO config loads exactly as before.
        presets = {"WARNO": {k: str(getattr(settings, k, "") or "") for k in _PER_GAME_FIELDS}}
        active = "WARNO"
    # ensure every game is selectable (seed missing sections with defaults)
    for gid in VALID_GAME_IDS:
        if not isinstance(presets.get(gid), dict):
            presets[gid] = _default_game_section(gid)
    # assign the active game without firing the switch callback
    settings.game_switch_lock = True
    try:
        settings.game_preset = active
    finally:
        settings.game_switch_lock = False
    settings.active_game_loaded = active
    _apply_game_section(settings, presets[active])

    _enforce_fixed_runtime_defaults(settings)
    return True, f"Loaded: {path}"


def _save_settings_to_config(settings: WARNOImporterSettings, path: Path) -> tuple[bool, str]:
    _enforce_fixed_runtime_defaults(settings)
    raw: dict[str, Any] = {}
    if path.exists() and path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    raw["project_root"] = settings.project_root
    raw["spk_path"] = settings.spk_path
    raw["skeleton_spk"] = settings.skeleton_spk
    raw["atlas_assets_dir"] = settings.atlas_assets_dir
    raw["tgv_converter"] = str(settings.tgv_converter or "tgv_to_png.py")
    raw["import_semantic_mode"] = str(settings.import_semantic_mode or "REFERENCE")
    raw["texture_subdir"] = "textures"
    raw["cache_dir"] = settings.cache_dir
    raw["use_zz_dat_source"] = bool(settings.use_zz_dat_source)
    raw["warno_root"] = settings.warno_root
    raw["modding_suite_root"] = settings.modding_suite_root
    raw["zz_runtime_dir"] = settings.zz_runtime_dir

    raw["auto_textures"] = bool(settings.auto_textures)
    raw["fast_exact_texture_resolve"] = bool(settings.fast_exact_texture_resolve)
    raw["texture_process_timeout_sec"] = int(settings.texture_process_timeout_sec)
    raw["texture_stage_timeout_sec"] = int(settings.texture_stage_timeout_sec)
    raw["auto_install_tgv_deps"] = bool(settings.auto_install_tgv_deps)
    raw["tgv_deps_dir"] = str(settings.tgv_deps_dir or ".warno_pydeps")
    raw["auto_split_main_parts"] = bool(settings.auto_split_main_parts)
    raw["auto_name_parts"] = bool(settings.auto_name_parts)
    raw["auto_name_materials"] = bool(settings.auto_name_materials)
    raw["auto_pull_bones"] = bool(settings.auto_pull_bones)
    raw["auto_rename_textures"] = bool(settings.auto_rename_textures)
    raw["ao_multiply_diffuse"] = bool(settings.use_ao_multiply)
    raw["normal_invert_mode"] = str(settings.normal_invert_mode or "none")
    raw["log_to_file"] = bool(settings.log_to_file)
    raw["log_file_name"] = str(settings.log_file_name or "warno_import.log")
    raw["rotate_x"] = 0.0
    raw["rotate_y"] = 0.0
    raw["rotate_z"] = 0.0
    raw["mirror_y"] = True
    raw["fbx_use_merge_by_distance"] = bool(settings.use_merge_by_distance)
    raw["fbx_merge_distance"] = float(settings.merge_distance)
    raw["fbx_auto_smooth_mode"] = str(settings.auto_smooth_mode or "OFF").upper()
    raw["fbx_use_auto_smooth"] = str(settings.auto_smooth_mode or "OFF").upper() == "APPLY"
    raw["fbx_auto_smooth_angle"] = float(settings.auto_smooth_angle)

    # --- Multi-game preset layer ---
    # The active game's per-game values are written both to the flat keys above
    # (old-reader-safe) and into game_presets[active]; other games keep their
    # remembered sections untouched.
    presets = raw.get("game_presets")
    if not isinstance(presets, dict):
        presets = {}
    active = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))
    presets[active] = _collect_live_game_section(settings)
    for gid in VALID_GAME_IDS:
        if not isinstance(presets.get(gid), dict):
            presets[gid] = _default_game_section(gid)
    raw["active_game"] = active
    raw["game_presets"] = presets

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        return False, f"Save failed: {exc}"
    return True, f"Saved: {path}"


def _all_tris(indices: Sequence[int]) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for i in range(0, len(indices), 3):
        if i + 2 >= len(indices):
            break
        out.append((int(indices[i + 0]), int(indices[i + 1]), int(indices[i + 2])))
    return out


def _build_bone_payload(
    extractor_mod,
    spk,
    model: dict[str, Any],
    asset: str,
    meta: dict[str, Any],
    material_role_by_id: dict[int, str],
    rot: dict[str, float],
    skeleton_spks,
    gfx_manifest_info,
    dev_scale: float = WARNO_DEV_SCALE,
    off_mat_scale: float = WARNO_OFF_MAT_SCALE,
) -> dict[str, Any]:
    gfx_manifest = dict(gfx_manifest_info or {}) if isinstance(gfx_manifest_info, dict) else {}
    gfx_semantic_bones = [
        _norm_low(str(x))
        for x in (gfx_manifest.get("semantic_bones", []) or [])
        if str(x).strip()
    ]
    gfx_required_nodes = [
        _norm_low(str(x))
        for x in (gfx_manifest.get("required_nodes", []) or [])
        if str(x).strip()
    ]
    gfx_fx_nodes = [
        _norm_low(str(x))
        for x in (gfx_manifest.get("fx_nodes", []) or [])
        if str(x).strip()
    ]
    gfx_subdepiction_nodes = [
        _norm_low(str(x))
        for x in (gfx_manifest.get("subdepiction_nodes", []) or [])
        if str(x).strip()
    ]
    gfx_operator_contracts = [
        row
        for row in (gfx_manifest.get("operator_contracts", []) or [])
        if isinstance(row, dict)
    ]
    gfx_semantic_nodes = [
        row
        for row in (gfx_manifest.get("semantic_nodes", []) or [])
        if isinstance(row, dict)
    ]
    gfx_transform_debug = [
        row
        for row in (gfx_manifest.get("transform_debug", []) or [])
        if isinstance(row, dict)
    ]
    gfx_role_map_raw = gfx_manifest.get("role_map", {}) or {}
    gfx_role_map: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(gfx_role_map_raw, dict):
        for key, rows in gfx_role_map_raw.items():
            low = _norm_low(str(key))
            if not low:
                continue
            if isinstance(rows, list):
                clean_rows = [dict(row) for row in rows if isinstance(row, dict)]
                if clean_rows:
                    gfx_role_map[low] = clean_rows
    gfx_manifest_source = str(gfx_manifest.get("source", "none") or "none").strip() or "none"
    gfx_manifest_error = str(gfx_manifest.get("error", "") or "").strip()
    gfx_track_kind = str(gfx_manifest.get("track_kind", "") or "").strip().lower()
    mesh_node_index = int(meta.get("nodeIndex", -1))
    mesh_bone_names: List[str] = []
    external_bone_names: List[str] = []
    bone_names: List[str] = []
    raw_bone_names: List[str] = []
    bone_name_by_index: Dict[int, str] = {}
    raw_bone_name_by_index: Dict[int, str] = {}
    bone_parent_by_index: Dict[int, int] = {}
    off_mat_points_by_index: Dict[int, List[List[float]]] = {}
    off_mat_blocks_by_index: Dict[int, List[List[float]]] = {}
    raw_scene_graph = None
    bone_name_source = "none"

    candidates: List[Dict[str, Any]] = []

    def _add_candidate(source: str, parser_obj: Any, node_idx: int) -> None:
        if int(node_idx) < 0:
            return
        try:
            names = list(parser_obj.parse_node_names(int(node_idx)))
        except Exception:
            names = []
        if not names:
            return
        try:
            parents = list(parser_obj.parse_node_parent_indices(int(node_idx)))
        except Exception:
            parents = []
        try:
            off_mat_points = list(parser_obj.parse_node_off_mat_points(int(node_idx)))
        except Exception:
            off_mat_points = []
        try:
            off_mat_blocks = list(parser_obj.parse_node_off_mat_blocks(int(node_idx)))
        except Exception:
            off_mat_blocks = []
        try:
            scene_graph = parser_obj.parse_raw_scene_graph(int(node_idx))
        except Exception:
            scene_graph = None
        if len(parents) < len(names):
            parents = [int(x) for x in parents] + [-1] * (len(names) - len(parents))
        elif len(parents) > len(names):
            parents = [int(x) for x in parents[: len(names)]]
        if len(off_mat_points) < len(names):
            off_mat_points = list(off_mat_points) + [[] for _ in range(len(names) - len(off_mat_points))]
        elif len(off_mat_points) > len(names):
            off_mat_points = list(off_mat_points[: len(names)])
        if len(off_mat_blocks) < len(names):
            off_mat_blocks = list(off_mat_blocks) + [[] for _ in range(len(names) - len(off_mat_blocks))]
        elif len(off_mat_blocks) > len(names):
            off_mat_blocks = list(off_mat_blocks[: len(names)])
        candidates.append(
            {
                "source": str(source),
                "names": [str(x) for x in names],
                "parents": [int(x) for x in parents],
                "off_mat_points": off_mat_points,
                "off_mat_blocks": off_mat_blocks,
                "raw_scene_graph": scene_graph,
            }
        )

    if mesh_node_index >= 0:
        _add_candidate("mesh", spk, mesh_node_index)
    if candidates:
        mesh_bone_names = list(candidates[0].get("names", []))

    if skeleton_spks is None:
        skeleton_spk_list: List[Any] = []
    elif isinstance(skeleton_spks, (list, tuple)):
        skeleton_spk_list = [sp for sp in skeleton_spks if sp is not None]
    else:
        skeleton_spk_list = [skeleton_spks]

    for sk_i, skeleton_spk in enumerate(skeleton_spk_list):
        try:
            sk_hit = skeleton_spk.find_best_fat_entry_for_asset(asset)
        except Exception:
            sk_hit = None
        if sk_hit is not None:
            _, sk_meta = sk_hit
            _add_candidate(f"external_asset_match_{sk_i}", skeleton_spk, int(sk_meta.get("nodeIndex", -1)))
        _add_candidate(f"external_same_index_{sk_i}", skeleton_spk, mesh_node_index)

    if len(candidates) > 1:
        ext_vals: List[str] = []
        for cand in candidates[1:]:
            ext_vals.extend([str(n) for n in cand.get("names", []) if str(n).strip()])
        if ext_vals:
            external_bone_names = extractor_mod.unique_keep_order(ext_vals)

    selected: Dict[str, Any] | None = None
    for cand in candidates:
        names = [str(n) for n in cand.get("names", [])]
        if any(str(n).strip() for n in names):
            selected = cand
            break

    if selected is not None:
        bone_name_source = str(selected.get("source", "none"))
        raw_scene_graph = selected.get("raw_scene_graph")
        primary_raw_names = [str(n) for n in selected.get("names", [])]
        raw_names = list(primary_raw_names)
        raw_bone_names = [str(n).strip() for n in primary_raw_names if str(n).strip()]
        bone_names = [str(n).strip() for n in primary_raw_names if str(n).strip()]
        for bidx, raw_name in enumerate(raw_names):
            name_text = str(raw_name or "").strip()
            if not name_text:
                continue
            raw_bone_name_by_index[int(bidx)] = name_text
            if hasattr(extractor_mod, "sanitize_material_name"):
                try:
                    safe_name = str(extractor_mod.sanitize_material_name(name_text) or name_text).strip()
                except Exception:
                    safe_name = name_text
            else:
                safe_name = name_text
            if not safe_name:
                continue
            bone_name_by_index[int(bidx)] = safe_name
        primary_parents = [int(x) for x in selected.get("parents", [])]

        # Secondary source merge: fill only missing names/parents on same indices.
        for cand in candidates:
            if cand is selected:
                continue
            cand_names = [str(n) for n in cand.get("names", [])]
            for bidx, raw_name in enumerate(cand_names):
                if int(bidx) in bone_name_by_index:
                    continue
                name_text = str(raw_name or "").strip()
                if not name_text:
                    continue
                raw_bone_name_by_index[int(bidx)] = name_text
                raw_bone_names.append(name_text)
                if hasattr(extractor_mod, "sanitize_material_name"):
                    try:
                        safe_name = str(extractor_mod.sanitize_material_name(name_text) or name_text).strip()
                    except Exception:
                        safe_name = name_text
                else:
                    safe_name = name_text
                if not safe_name:
                    continue
                bone_name_by_index[int(bidx)] = safe_name
                bone_names.append(name_text)

        def _parent_for_index(bidx: int) -> int:
            if 0 <= int(bidx) < len(primary_parents):
                parent_idx = int(primary_parents[int(bidx)])
                if parent_idx >= 0:
                    return parent_idx
            for cand in candidates:
                if cand is selected:
                    continue
                cand_parents = [int(x) for x in cand.get("parents", [])]
                if 0 <= int(bidx) < len(cand_parents):
                    parent_idx = int(cand_parents[int(bidx)])
                    if parent_idx >= 0:
                        return parent_idx
            return -1

        for bidx in bone_name_by_index.keys():
            bone_parent_by_index[int(bidx)] = _parent_for_index(int(bidx))

        def _normalized_off_mat_points(raw_points: Any) -> List[List[float]]:
            out: List[List[float]] = []
            if not isinstance(raw_points, (list, tuple)):
                return out
            seen: set[Tuple[int, int, int]] = set()
            for raw_point in raw_points:
                if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 3:
                    continue
                try:
                    px, py, pz = _warno_off_mat_point_xyz(
                        float(raw_point[0]),
                        float(raw_point[1]),
                        float(raw_point[2]),
                        scale=off_mat_scale,
                    )
                except Exception:
                    continue
                if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                    continue
                key = (round(px, 6), round(py, 6), round(pz, 6))
                if key in seen:
                    continue
                seen.add(key)
                if abs(px) <= 1e-9 and abs(py) <= 1e-9 and abs(pz) <= 1e-9:
                    continue
                out.append([float(px), float(py), float(pz)])
            return out

        def _off_mat_for_index(bidx: int) -> List[List[float]]:
            if selected is not None:
                primary_points = list(selected.get("off_mat_points", []) or [])
                if 0 <= int(bidx) < len(primary_points):
                    got = _normalized_off_mat_points(primary_points[int(bidx)])
                    if got:
                        return got
            for cand in candidates:
                if cand is selected:
                    continue
                cand_points = list(cand.get("off_mat_points", []) or [])
                if 0 <= int(bidx) < len(cand_points):
                    got = _normalized_off_mat_points(cand_points[int(bidx)])
                    if got:
                        return got
            return []

        def _normalized_off_mat_blocks(raw_blocks: Any) -> List[List[float]]:
            out: List[List[float]] = []
            if not isinstance(raw_blocks, (list, tuple)):
                return out
            for block in raw_blocks:
                if not isinstance(block, (list, tuple)) or len(block) < 12:
                    continue
                vals: List[float] = []
                ok = True
                for item in block[:12]:
                    try:
                        val = float(item)
                    except Exception:
                        ok = False
                        break
                    if not math.isfinite(val):
                        ok = False
                        break
                    vals.append(val)
                if ok and len(vals) == 12:
                    out.append(vals)
            return out

        def _off_mat_blocks_for_index(bidx: int) -> List[List[float]]:
            if selected is not None:
                primary_blocks = list(selected.get("off_mat_blocks", []) or [])
                if 0 <= int(bidx) < len(primary_blocks):
                    got = _normalized_off_mat_blocks(primary_blocks[int(bidx)])
                    if got:
                        return got
            for cand in candidates:
                if cand is selected:
                    continue
                cand_blocks = list(cand.get("off_mat_blocks", []) or [])
                if 0 <= int(bidx) < len(cand_blocks):
                    got = _normalized_off_mat_blocks(cand_blocks[int(bidx)])
                    if got:
                        return got
            return []

        for bidx in bone_name_by_index.keys():
            got = _off_mat_for_index(int(bidx))
            if got:
                off_mat_points_by_index[int(bidx)] = got
            raw_blocks = _off_mat_blocks_for_index(int(bidx))
            if raw_blocks:
                off_mat_blocks_by_index[int(bidx)] = raw_blocks

        if hasattr(extractor_mod, "unique_keep_order"):
            try:
                bone_names = [str(x) for x in extractor_mod.unique_keep_order(bone_names)]
            except Exception:
                pass
            try:
                raw_bone_names = [str(x) for x in extractor_mod.unique_keep_order(raw_bone_names)]
            except Exception:
                pass

    raw_bone_centers_by_index = extractor_mod.estimate_bone_centers_by_index(model, bone_name_by_index, rot)
    bone_centers_by_index: Dict[int, Tuple[float, float, float]] = {}
    for bidx, pos in raw_bone_centers_by_index.items():
        try:
            px, py, pz = _warno_dev_scaled_xyz(float(pos[0]), float(pos[1]), float(pos[2]), scale=dev_scale)
            bone_centers_by_index[int(bidx)] = (px, py, pz)
        except Exception:
            continue
    bone_positions: Dict[str, List[float]] = {}

    def register(name: str, pos: Tuple[float, float, float], *, overwrite: bool = False) -> None:
        low = _norm_low(name)
        if not low:
            return
        if overwrite or low not in bone_positions:
            bone_positions[low] = [float(pos[0]), float(pos[1]), float(pos[2])]
        tok = _norm_token(low)
        if tok and (overwrite or tok not in bone_positions):
            bone_positions[tok] = [float(pos[0]), float(pos[1]), float(pos[2])]

    for bidx, pos in bone_centers_by_index.items():
        mapped_name = bone_name_by_index.get(int(bidx))
        if mapped_name:
            register(mapped_name, pos)
        raw_name = raw_bone_name_by_index.get(int(bidx))
        if raw_name:
            register(raw_name, pos)

    for bidx, pts in off_mat_points_by_index.items():
        if not isinstance(pts, (list, tuple)) or not pts:
            continue
        raw_pt = pts[0]
        if not isinstance(raw_pt, (list, tuple)) or len(raw_pt) < 3:
            continue
        try:
            pos = (float(raw_pt[0]), float(raw_pt[1]), float(raw_pt[2]))
        except Exception:
            continue
        mapped_name = bone_name_by_index.get(int(bidx))
        if mapped_name:
            register(
                mapped_name,
                pos,
                overwrite=_norm_low(str(mapped_name)).startswith("roue_elev_"),
            )
        raw_name = raw_bone_name_by_index.get(int(bidx))
        if raw_name:
            register(
                raw_name,
                pos,
                overwrite=_norm_low(str(raw_name)).startswith("roue_elev_"),
            )

    return {
        "bone_name_by_index": bone_name_by_index,
        "raw_bone_name_by_index": raw_bone_name_by_index,
        "bone_parent_by_index": bone_parent_by_index,
        "bone_names": bone_names,
        "raw_bone_names": raw_bone_names,
        "bone_positions": bone_positions,
        "off_mat_points_by_index": off_mat_points_by_index,
        "off_mat_blocks_by_index": off_mat_blocks_by_index,
        "raw_scene_graph": raw_scene_graph,
        "bone_name_source": bone_name_source,
        "mesh_bone_names": mesh_bone_names,
        "external_bone_names": external_bone_names,
        "gfx_manifest": gfx_manifest,
        "gfx_manifest_source": gfx_manifest_source,
        "gfx_manifest_error": gfx_manifest_error,
        "gfx_track_kind": gfx_track_kind,
        "gfx_semantic_bones": gfx_semantic_bones,
        "gfx_required_nodes": gfx_required_nodes,
        "gfx_fx_nodes": gfx_fx_nodes,
        "gfx_subdepiction_nodes": gfx_subdepiction_nodes,
        "gfx_operator_contracts": gfx_operator_contracts,
        "gfx_role_map": gfx_role_map,
        "gfx_semantic_nodes": gfx_semantic_nodes,
        "gfx_transform_debug": gfx_transform_debug,
    }


def _channel_hint_from_stem(stem: str) -> str | None:
    low = _norm_low(stem)
    if not low:
        return None
    if "combinedorm" in low or "combinedrm" in low or low.endswith("_orm"):
        return "orm"
    if "diffusetexturenoalpha" in low or ("noalpha" in low and ("diffuse" in low or "color" in low)):
        return "diffuse"
    if "normal_x" in low or "normal_y" in low or "normal_z" in low:
        return None
    # NOTE: the loose "<channel> in name" substring tests must NOT fire when the channel word
    # is the unit-name PREFIX (e.g. "Alpha_Jet" -> the DIFFUSE texture, not an alpha map; "Metal_*"
    # etc.). Matching a prefix mis-routes the bare diffuse to that channel, so the material ends up
    # with no Base Color. A channel keyword as a real channel sits at/after a separator, never at
    # the very start. (This is the "all textures present except the diffuse" bug.)
    if "normal_reconstructed" in low or low.endswith("_nm") or ("normal" in low and not low.startswith("normal")):
        return "normal"
    if low.endswith("_o") or low.endswith("_ao") or ("occlusion" in low and not low.startswith("occlusion")):
        return "occlusion"
    if low.endswith("_roughness") or low.endswith("_r") or ("roughness" in low and not low.startswith("roughness")):
        return "roughness"
    if low.endswith("_metallic") or low.endswith("_m") or ("metallic" in low and not low.startswith("metallic")):
        return "metallic"
    if low.endswith("_alpha") or low.endswith("_a") or ("alpha" in low and not low.startswith("alpha")):
        return "alpha"
    if low.endswith("_d") or "diffuse" in low or "albedo" in low or "color" in low:
        return "diffuse"
    return None


def _asset_atlas_ref_dir_candidates(extractor_mod, asset: str) -> List[str]:
    norm = extractor_mod.normalize_asset_path(str(asset or ""))
    if not norm:
        return []
    p = Path(norm)
    parent = str(p.parent).replace("\\", "/").strip("/")
    if not parent:
        return []
    out: List[str] = [parent]
    last = p.parent.name
    stem = p.stem
    stem_no_lod = extractor_mod.strip_lod_suffix(stem) if hasattr(extractor_mod, "strip_lod_suffix") else stem
    if stem_no_lod and last and stem_no_lod.lower() != last.lower():
        parent2 = str(p.parent.parent / stem_no_lod).replace("\\", "/").strip("/")
        if parent2:
            out.append(parent2)
    return list(dict.fromkeys(out))


def _atlas_map_no_entries_error(text: str) -> bool:
    low = _norm_low(text)
    if not low:
        return False
    return ("atlas_cli_no_entries" in low) or ("no entries for asset" in low)


def _atlas_asset_variant_candidates(asset: str) -> List[str]:
    raw = str(asset or "").replace("\\", "/").strip()
    if not raw:
        return []
    p = PurePosixPath(raw)
    parent = str(p.parent).strip("/")
    stem = str(p.stem or "").strip()
    suffix = str(p.suffix or ".fbx")
    if not stem:
        return [raw]
    tokens = [t for t in stem.split("_") if t]
    if not tokens:
        return [raw]

    candidates: List[str] = [raw]
    known_tail = {"l", "r", "left", "right", "dest", "destroyed", "damaged", "wreck"}
    cur = list(tokens)
    while len(cur) > 1 and _norm_low(cur[-1]) in known_tail:
        cur = cur[:-1]
        cand_stem = "_".join(cur)
        if parent:
            candidates.append(f"{parent}/{cand_stem}{suffix}")
        else:
            candidates.append(f"{cand_stem}{suffix}")

    return list(dict.fromkeys(candidates))


def _build_strict_grouped_maps(
    resolved_items: Sequence[Dict[str, Any]],
    extractor_mod: Any | None = None,
) -> Dict[str, Dict[str, Any]]:
    wanted = {"diffuse", "normal", "roughness", "metallic", "occlusion", "alpha", "orm"}
    groups: Dict[str, Dict[str, Any]] = {}

    def _group_key(item: Dict[str, Any], out_png: Path) -> str:
        raw = str(item.get("atlas_target_basename", "") or "").strip() or out_png.stem
        return _norm_low(_strip_texture_channel_suffix(raw) or raw)

    def _add(group: Dict[str, Any], channel: str, path: Path | None, score: float) -> None:
        if channel not in wanted or path is None:
            return
        if not path.exists() or not path.is_file():
            return
        best = group.setdefault("best", {})
        cur = best.get(channel)
        if cur is None or float(score) > float(cur[0]):
            best[channel] = (float(score), path)

    for idx, item in enumerate(resolved_items):
        out_raw = item.get("out_png")
        out_png = Path(out_raw) if out_raw else None
        if out_png is None:
            continue
        gk = _group_key(item, out_png)
        if not gk:
            continue
        group = groups.setdefault(
            gk,
            {
                "is_track": _strict_atlas_item_is_track(item, extractor_mod=extractor_mod),
                "best": {},
                "count": 0,
                "sample_name": str(item.get("atlas_target_basename", "") or out_png.stem),
            },
        )
        group["count"] = int(group.get("count", 0)) + 1
        channel = _strict_atlas_item_channel(item, extractor_mod=extractor_mod)
        base_score = 1000.0 - float(idx)
        _add(group, channel, out_png, base_score)
        extras = item.get("extras", {})
        if isinstance(extras, dict):
            for ek, ev in extras.items():
                p = Path(ev)
                token = f"{str(ek or '')} {p.stem}"
                hint = _channel_hint_from_stem(token)
                if hint is None and extractor_mod is not None and hasattr(extractor_mod, "channel_from_token"):
                    try:
                        ext_hint = extractor_mod.channel_from_token(token)
                    except Exception:
                        ext_hint = None
                    hint = _canonical_map_channel_name(str(ext_hint)) if ext_hint else None
                ch = _canonical_map_channel_name(str(hint or ""))
                _add(group, ch, p, base_score - 5.0)

    out: Dict[str, Dict[str, Any]] = {}
    for key, payload in groups.items():
        best = payload.get("best", {})
        out[key] = {
            "is_track": bool(payload.get("is_track", False)),
            "count": int(payload.get("count", 0)),
            "sample_name": str(payload.get("sample_name", key)),
            "maps": {ch: src for ch, (_score, src) in best.items()},
        }
    return out


def _norm_ref_like(ref: str) -> str:
    return str(ref or "").strip().replace("\\", "/").lower()


def _score_guessed_ref(ref: str) -> int:
    low = _norm_ref_like(ref)
    score = 0
    if "/tsccolor_diffusetexturenoalpha" in low:
        score += 140
    if "/tscnm_normaltexture" in low:
        score += 120
    if "/tscorm_combinedormtexture" in low:
        score += 120
    if "singlechannellinearmap_roughness" in low or "roughnesstexture" in low:
        score += 116
    if "singlechannellinearmap_metallic" in low or "metallictexture" in low:
        score += 116
    if "singlechannellinearmap_occlusion" in low or "occlusiontexture" in low:
        score += 112
    if "/tsccoloralpha_combineddatexture" in low:
        score += 100
    if "_trk" in low:
        score += 40
    if "/ui/" in low or "/icons/" in low:
        score -= 80
    if low.endswith(".png"):
        score += 4
    return score


def _guess_texture_refs_for_asset(
    extractor_mod,
    asset: str,
    atlas_assets_root: Path,
    runtime_info: Dict[str, Any] | None = None,
    max_refs: int = 24,
    include_zz_scan: bool = True,
) -> List[str]:
    runtime_info = runtime_info or {}
    out: List[str] = []
    dirs = _asset_atlas_ref_dir_candidates(extractor_mod, asset)
    if not dirs:
        return out

    zz_resolver = runtime_info.get("zz_resolver")
    limit = max(1, int(max_refs))
    if include_zz_scan and zz_resolver is not None:
        try:
            keys = zz_resolver.all_asset_keys()
        except Exception:
            keys = []
        for d in dirs:
            prefix = f"pc/atlas/{d.lower().strip('/')}/"
            for key in keys:
                low = _norm_ref_like(key)
                if not (low.startswith(prefix) and (low.endswith(".tgv") or low.endswith(".png"))):
                    continue
                if low.endswith(".tgv"):
                    low = low[:-4] + ".png"
                out.append(low)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break

    for d in dirs:
        rel_under_assets = d
        if rel_under_assets.lower().startswith("assets/"):
            rel_under_assets = rel_under_assets[7:]
        folder = atlas_assets_root / Path(rel_under_assets)
        if not folder.exists() or not folder.is_dir():
            continue
        try:
            files = list(folder.glob("*.tgv")) + list(folder.glob("*.png"))
        except Exception:
            files = []
        for fp in files:
            try:
                rel = fp.relative_to(atlas_assets_root).with_suffix(".png").as_posix()
            except Exception:
                continue
            out.append(f"PC/Atlas/Assets/{rel}")
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break

    if not out:
        return []
    uniq = list(dict.fromkeys(_norm_ref_like(v) for v in out if str(v).strip()))
    uniq.sort(key=lambda v: (-_score_guessed_ref(v), len(v), v))
    return uniq[:limit]


def _augment_maps_from_existing_files(
    model_dir: Path,
    asset: str,
    chosen_maps: Dict[str, Path],
) -> Dict[str, Path]:
    out = dict(chosen_maps)
    files = sorted(model_dir.rglob("*.png"))
    if not files:
        return out

    wanted = {"diffuse", "normal", "roughness", "metallic", "occlusion", "alpha", "orm"}

    asset_stem = _norm_low(Path(asset).stem)
    best_by_channel: Dict[str, tuple[int, Path]] = {}
    for p in files:
        hint = _channel_hint_from_stem(p.stem)
        if hint is None or hint not in wanted:
            continue
        score = 0
        low_stem = _norm_low(p.stem)
        if asset_stem and low_stem.startswith(f"{asset_stem}_"):
            score += 40
        if "_trk_" in low_stem:
            score -= 8
        if "part" in low_stem or "_mg" in low_stem:
            score -= 10
        if "diffuse" in low_stem and hint == "diffuse":
            score += 8
        if "normal_reconstructed" in low_stem and hint == "normal":
            score += 8
        if low_stem.endswith("_ao") and hint == "occlusion":
            score += 8
        if low_stem.endswith("_r") and hint == "roughness":
            score += 6
        if low_stem.endswith("_m") and hint == "metallic":
            score += 6
        if low_stem.endswith("_a") and hint == "alpha":
            score += 6

        current = best_by_channel.get(hint)
        if current is None or score > current[0]:
            best_by_channel[hint] = (score, p)

    for channel, pair in best_by_channel.items():
        if channel not in out:
            out[channel] = pair[1]
    return out


def _cleanup_strict_legacy_alias_pngs(model_dir: Path, asset: str) -> int:
    stem = str(Path(str(asset or "")).stem or "").strip()
    if not stem:
        return 0
    removed = 0
    patterns = [f"{stem}.png", f"{stem}_*.png"]
    for pat in patterns:
        for p in model_dir.glob(pat):
            if not p.exists() or not p.is_file():
                continue
            try:
                p.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def _canonical_map_channel_name(value: str) -> str:
    low = _norm_low(value)
    if not low:
        return "generic"
    if low in {"d", "diff", "albedo", "basecolor", "base_color", "diffuse"}:
        return "diffuse"
    if low in {"a", "alpha", "opacity"}:
        return "alpha"
    if low in {"nm", "normal", "normalmap"}:
        return "normal"
    if low in {"orm", "rma", "mrao"}:
        return "orm"
    if low in {"ao", "o", "occlusion", "ambientocclusion"}:
        return "occlusion"
    if low in {"r", "rough", "roughness"}:
        return "roughness"
    if low in {"m", "metal", "metallic", "metalness"}:
        return "metallic"
    return low


def _strict_atlas_item_is_track(item: Dict[str, Any], extractor_mod: Any | None = None) -> bool:
    parts = [
        str(item.get("atlas_target_basename", "") or ""),
        str(item.get("atlas_target_logical_rel", "") or ""),
        str(item.get("atlas_ref", "") or ""),
    ]
    out_raw = str(item.get("out_png", "") or "").strip()
    if out_raw:
        parts.append(Path(out_raw).stem)
    text = " ".join(parts)
    low = _norm_low(text)
    if "tracks" in low or "_trk" in low or "chenille" in low:
        return True
    if extractor_mod is not None and hasattr(extractor_mod, "is_track_token"):
        try:
            if bool(extractor_mod.is_track_token(text)):
                return True
        except Exception:
            pass
    return False


def _strict_atlas_item_channel(item: Dict[str, Any], extractor_mod: Any | None = None) -> str:
    channel = _canonical_map_channel_name(str(item.get("atlas_target_channel", "")))
    guesses: List[str] = []
    if channel == "generic":
        role_guess = _canonical_map_channel_name(str(item.get("role", "")))
        if role_guess != "generic":
            guesses.append(role_guess)

    tokens = [
        str(item.get("atlas_target_basename", "") or ""),
        str(item.get("atlas_target_logical_rel", "") or ""),
        str(item.get("atlas_ref", "") or ""),
    ]
    out_raw = str(item.get("out_png", "") or "").strip()
    if out_raw:
        tokens.append(Path(out_raw).stem)

    for tok in tokens:
        stem = Path(str(tok)).stem
        hint = _channel_hint_from_stem(stem) or _channel_hint_from_stem(str(tok))
        if hint:
            guesses.append(_canonical_map_channel_name(hint))
    if extractor_mod is not None and hasattr(extractor_mod, "channel_from_token"):
        merged = " ".join(tokens)
        try:
            ext_hint = extractor_mod.channel_from_token(merged)
            if ext_hint:
                guesses.append(_canonical_map_channel_name(str(ext_hint)))
        except Exception:
            pass

    for g in guesses:
        if g != "generic":
            if channel == "generic":
                channel = g
            elif channel == "diffuse" and g in {"alpha", "normal", "roughness", "metallic", "occlusion", "orm"}:
                channel = g
    return channel


def _pick_maps_for_material_strict_local(
    resolved_items: Sequence[Dict[str, Any]],
    material_role: str,
    asset_stem: str,
    extractor_mod: Any | None = None,
) -> Dict[str, Path]:
    wanted = {"diffuse", "normal", "roughness", "metallic", "occlusion", "alpha", "orm"}
    role_low = _norm_low(material_role)
    want_track = role_low.startswith("track")
    stem_low = _norm_low(asset_stem)
    best_explicit: Dict[str, tuple[float, Path]] = {}
    best_extra: Dict[str, tuple[float, Path]] = {}

    def _prefer_occlusion_short_o(path: Path) -> Path:
        try:
            p = Path(path)
        except Exception:
            return path
        stem = p.stem
        if stem.upper().endswith("_AO"):
            alt = p.with_name(f"{stem[:-3]}_O{p.suffix}")
            if alt.exists() and alt.is_file():
                return alt
        return p

    def _add(bucket: Dict[str, tuple[float, Path]], channel: str, path: Path | None, score: float) -> None:
        if path is None or channel not in wanted:
            return
        if channel == "occlusion":
            path = _prefer_occlusion_short_o(path)
        if not path.exists() or not path.is_file():
            return
        cur = bucket.get(channel)
        if cur is None or float(score) > cur[0]:
            bucket[channel] = (float(score), path)

    for idx, item in enumerate(resolved_items):
        out_raw = item.get("out_png")
        out_png = Path(out_raw) if out_raw else None
        if out_png is None:
            continue
        is_track = _strict_atlas_item_is_track(item, extractor_mod=extractor_mod)
        if is_track != want_track:
            continue
        channel = _strict_atlas_item_channel(item, extractor_mod=extractor_mod)
        if channel not in wanted:
            continue
        score = 1000.0 - float(idx)
        base = _norm_low(str(item.get("atlas_target_basename", "") or ""))
        if stem_low:
            if base.startswith(f"{stem_low}_"):
                score += 25.0
            elif base == stem_low:
                score += 20.0
        _add(best_explicit, channel, out_png, score)

        extras = item.get("extras", {})
        if not isinstance(extras, dict):
            continue
        for ek, ev in extras.items():
            p = Path(ev)
            token = f"{str(ek or '')} {p.stem}"
            hint = _channel_hint_from_stem(token)
            if hint is None and extractor_mod is not None and hasattr(extractor_mod, "channel_from_token"):
                try:
                    ext_hint = extractor_mod.channel_from_token(token)
                except Exception:
                    ext_hint = None
                hint = _canonical_map_channel_name(str(ext_hint)) if ext_hint else None
            ch = _canonical_map_channel_name(str(hint or ""))
            if ch in wanted:
                _add(best_extra, ch, p, score - 5.0)

    merged: Dict[str, Path] = {ch: src for ch, (_score, src) in best_explicit.items()}
    for ch, (_score, src) in best_extra.items():
        merged.setdefault(ch, src)
    return merged


def _ensure_occlusion_from_same_stem(maps: Dict[str, Path]) -> Dict[str, Path]:
    if not isinstance(maps, dict):
        return {}
    if maps.get("occlusion") is not None:
        return maps
    for key in ("orm", "roughness", "metallic", "normal", "diffuse"):
        src = maps.get(key)
        if src is None:
            continue
        try:
            p = Path(src)
        except Exception:
            continue
        base = _strip_texture_channel_suffix(p.stem) or p.stem
        cand_o = p.with_name(f"{base}_O{p.suffix}")
        if cand_o.exists() and cand_o.is_file():
            maps["occlusion"] = cand_o
            return maps
        cand_ao = p.with_name(f"{base}_AO{p.suffix}")
        if cand_ao.exists() and cand_ao.is_file():
            maps["occlusion"] = cand_ao
            return maps
    return maps


def _atlas_ref_exact_exists(
    extractor_mod,
    ref: str,
    atlas_root: Path,
    fallback_roots: Sequence[Path],
    zz_resolver: Any | None,
) -> bool:
    try:
        rel = extractor_mod.atlas_ref_to_rel_under_assets(ref)
    except Exception:
        return False
    rel_png = rel.with_suffix(".png")
    rel_tgv = rel.with_suffix(".tgv")

    roots: List[Path] = [atlas_root]
    for r in fallback_roots:
        if r not in roots:
            roots.append(r)
    for root in roots:
        try:
            if (root / rel_png).exists() or (root / rel_tgv).exists():
                return True
        except Exception:
            continue

    if zz_resolver is None:
        return False
    candidates = [
        f"PC/Atlas/Assets/{rel_png.as_posix()}",
        f"PC/Atlas/Assets/{rel_tgv.as_posix()}",
        f"Assets/{rel_png.as_posix()}",
        f"Assets/{rel_tgv.as_posix()}",
    ]
    for cand in candidates:
        try:
            if zz_resolver.find_exact(cand) is not None:
                return True
        except Exception:
            continue
    return False


def _build_fast_companion_refs(extractor_mod, refs: Sequence[str], cap: int = 24) -> List[str]:
    if not refs:
        return []
    tokens = [
        "tsccolor_diffusetexturenoalpha",
        "tscnm_normaltexture",
        "tscorm_combinedormtexture",
        "tscorm_combinedrmtexture",
        "tsccoloralpha_combineddatexture",
        "singlechannellinearmap_roughnesstexture",
        "singlechannellinearmap_metallictexture",
        "singlechannellinearmap_occlusiontexture",
        "singlechannellinearmap_alphatexture",
    ]
    out: List[str] = []
    max_refs = max(1, int(cap))
    for ref in refs:
        try:
            rel = extractor_mod.atlas_ref_to_rel_under_assets(ref)
        except Exception:
            continue
        stem = str(rel.stem or "")
        stem_low = stem.lower()
        source_token = ""
        for tok in tokens:
            if tok in stem_low:
                source_token = tok
                break
        if not source_token:
            continue
        for target_token in tokens:
            if target_token == source_token:
                continue
            cand_stem = re.sub(source_token, target_token, stem, flags=re.IGNORECASE)
            cand_rel = rel.with_name(f"{cand_stem}.png")
            out.append(f"PC/Atlas/Assets/{cand_rel.as_posix()}")
            if len(out) >= max_refs * 3:
                break
        if len(out) >= max_refs * 3:
            break
    uniq = list(dict.fromkeys(out))
    return uniq[: max_refs * 3]


def _size_area(size: tuple[int, int] | None) -> float:
    if size is None:
        return 0.0
    return float(size[0] * size[1])


def _size_mismatch(size: tuple[int, int] | None, target: tuple[int, int] | None) -> float | None:
    if size is None or target is None:
        return None
    tw, th = target
    dw = abs(float(size[0] - tw)) / max(1.0, float(tw))
    dh = abs(float(size[1] - th)) / max(1.0, float(th))
    return max(dw, dh)


def _pick_track_diffuse_override(
    extractor_mod,
    resolved_items: Sequence[Dict[str, Any]],
    current_diffuse: Path | None,
) -> Path | None:
    size_fn = getattr(extractor_mod, "read_png_size", None)
    if not callable(size_fn):
        return None

    def _sz(p: Path) -> tuple[int, int] | None:
        try:
            return size_fn(Path(p))
        except Exception:
            return None

    ch_from_token = getattr(extractor_mod, "channel_from_token", None)
    track_targets: List[tuple[int, int]] = []
    all_targets: List[tuple[int, int]] = []
    diffuse_candidates: List[tuple[Path, str, str, str]] = []

    for item in resolved_items:
        role = str(item.get("role", ""))
        atlas_ref = str(item.get("atlas_ref", ""))
        extras = item.get("extras", {})
        out_png_raw = item.get("out_png")
        out_png = Path(out_png_raw) if out_png_raw else None
        if not isinstance(extras, dict):
            extras = {}

        if role == "diffuse" and out_png is not None:
            diffuse_candidates.append((out_png, role, "", atlas_ref))
        if role == "combined_da":
            diff = extras.get("diffuse")
            if diff:
                diffuse_candidates.append((Path(diff), role, "diffuse", atlas_ref))
            elif out_png is not None:
                diffuse_candidates.append((out_png, role, "", atlas_ref))

        for ek, ev in extras.items():
            p = Path(ev)
            token = f"{ek} {p.stem}"
            ch = ch_from_token(token) if callable(ch_from_token) else _channel_hint_from_stem(token)
            if ch == "diffuse":
                diffuse_candidates.append((p, role, str(ek), atlas_ref))
            if ch in {"normal", "roughness", "metallic", "occlusion"}:
                s = _sz(p)
                if s is not None:
                    all_targets.append(s)
                    low = _norm_low(token)
                    if extractor_mod.is_track_token(low) or extractor_mod.is_track_token(_norm_low(atlas_ref)):
                        track_targets.append(s)

    if not diffuse_candidates:
        return None

    target = None
    if track_targets:
        target = max(track_targets, key=lambda t: t[0] * t[1])
    elif all_targets:
        target = max(all_targets, key=lambda t: t[0] * t[1])

    cur = Path(current_diffuse) if current_diffuse else None
    cur_key = _norm_ref_like(str(cur)) if cur is not None else ""

    def _cand_score(path: Path, role: str, extra_key: str, atlas_ref: str) -> float:
        low = _norm_low(path.stem)
        low_ref = _norm_low(atlas_ref)
        low_key = _norm_low(extra_key)
        score = 0.0
        if role == "combined_da":
            score += 40.0
        if "combinedda" in low or "coloralpha" in low:
            score += 35.0
        if "diffusetexturenoalpha" in low:
            score += 22.0
        if extractor_mod.is_track_token(low) or extractor_mod.is_track_token(low_ref) or extractor_mod.is_track_token(low_key):
            score += 28.0
        if low_key == "diffuse":
            score += 8.0
        sz = _sz(path)
        mm = _size_mismatch(sz, target)
        if mm is not None:
            if mm > 0.38:
                score -= 120.0
            elif mm > 0.22:
                score -= 36.0
            else:
                score += max(0.0, 34.0 - 70.0 * mm)
        if cur is not None and _norm_ref_like(str(path)) == cur_key:
            score += 12.0
        try:
            score += min(12.0, max(0.0, float(path.stat().st_size) / 65536.0))
        except Exception:
            pass
        return score

    ranked = sorted(
        (( _cand_score(p, r, ek, ar), p) for p, r, ek, ar in diffuse_candidates),
        key=lambda it: (it[0], _size_area(_sz(it[1]))),
        reverse=True,
    )
    if not ranked:
        return None
    best_score, best_path = ranked[0]
    if cur is None:
        return best_path if best_score >= 0.0 else None

    cur_score = None
    for score, path in ranked:
        if _norm_ref_like(str(path)) == cur_key:
            cur_score = score
            break
    if cur_score is None:
        cur_score = -10.0

    # Replace only when candidate is clearly better.
    if best_score >= cur_score + 8.0:
        return best_path
    return None


def _material_key(name: str) -> str:
    low = _norm_low(name)
    return re.sub(r"\.[0-9]{3}$", "", low)


def _collect_zz_candidates_for_ref(zz_resolver, ref: str) -> List[str]:
    norm = str(ref or "").strip().replace("\\", "/").lower()
    base = Path(norm).name.lower()
    if not base:
        return []
    names = [base]
    if base.endswith(".png"):
        names.append(base[:-4] + ".tgv")
    elif base.endswith(".tgv"):
        names.append(base[:-4] + ".png")
    if "combinedrmtexture" in base:
        names.append(base.replace("combinedrmtexture", "combinedormtexture"))
    if "combinedormtexture" in base:
        names.append(base.replace("combinedormtexture", "combinedrmtexture"))
    out: List[str] = []
    for nm in names:
        out.extend(zz_resolver._basename_to_keys.get(nm, []))
    return list(dict.fromkeys(out))


def _filter_atlas_targets_by_materials(
    atlas_map_index: Dict[str, List[Dict[str, Any]]] | None,
    material_name_by_id: Dict[int, str] | None,
    asset_hint: str = "",
) -> List[str] | None:
    """For composite assets (typically WARNO DECORS like HLM_10_L.fbx) the
    atlas JSON aggregates textures from many sibling FBX (e.g. 11 building
    blocks for one HLM building), so the legacy 'union all atlas targets'
    behavior of _resolve_material_maps explodes 5 real refs into 45.

    This helper returns the subset of atlas target refs whose owning
    texture entry's source_asset_path matches one of the SPK material
    names by stem. Returns None when no filtering signal is available
    (caller falls back to the legacy 'use everything' behavior so we
    never accidentally regress UNITS where SPK already provides full refs).
    """
    if not atlas_map_index or not material_name_by_id:
        return None

    wanted_stems: set[str] = set()
    for mat_name in material_name_by_id.values():
        s = str(mat_name or "").strip().lower()
        if not s:
            continue
        # Strip our per-asset namespace prefix if present
        s = _warno_strip_namespace(s)
        # Strip in-import dedup suffix '_2' / '_3'
        s = re.sub(r"_\d+$", "", s)
        if s:
            wanted_stems.add(s)
    if not wanted_stems:
        return None

    # Also keep the asset's own stem as a wanted token so its primary
    # texture set is never filtered out even when SPK material name
    # doesn't reflect it (rare path).
    asset_stem = _warno_asset_namespace(asset_hint).lower()
    if asset_stem:
        wanted_stems.add(asset_stem)

    seen_entries: set[int] = set()
    relevant_targets: List[str] = []
    target_seen: set[str] = set()
    matched_any = False
    for entries_list in atlas_map_index.values():
        for entry in entries_list:
            eid = id(entry)
            if eid in seen_entries:
                continue
            seen_entries.add(eid)
            src = str(entry.get("source_asset_path", "")).strip()
            if not src:
                # Older atlas exports without source_asset_path; can't filter.
                return None
            stem = src.rsplit("/", 1)[-1]
            if "." in stem:
                stem = stem.rsplit(".", 1)[0]
            stem_low = stem.lower()
            matched = any(
                w == stem_low or w in stem_low or stem_low in w
                for w in wanted_stems
            )
            tgt = str(entry.get("target_logical_rel", "")).strip()
            # Keep TRACK textures ('<Unit>_tracks') when the unit has a track material
            # (Chenille_*), even if the texture's source-FBX stem doesn't match a material stem —
            # otherwise the track stays untextured (T80BK: target 'T80B_tracks' vs material
            # 'Chenille_Droite' never matches by stem, so the track refs get filtered away).
            _tl = tgt.lower()
            _is_track_tgt = ("track" in _tl) or ("chenille" in _tl) or ("_trk" in _tl)
            _has_track_mat = any(("chenille" in w) or ("track" in w) or ("trk" in w) for w in wanted_stems)
            if not matched and not (_is_track_tgt and _has_track_mat):
                continue
            matched_any = True
            if tgt and tgt not in target_seen:
                target_seen.add(tgt)
                relevant_targets.append(tgt)

    if not matched_any:
        # No SPK material maps to any atlas source. Fall back to legacy
        # behavior so a corner case can't silently leave the asset bare.
        return None
    return relevant_targets


def _rd_alpha_is_cutout(png_path) -> bool:
    """True when a combined texture's alpha is a transparency CUTOUT mask (chains /
    track-teeth gaps — large fully-transparent regions) rather than a smooth spec
    mask. Cutout -> wire alpha to opacity; spec -> wire alpha (inverted) to roughness.
    Decided by content (fraction of near-zero alpha) so it works regardless of the
    texture's DS/DAS naming. PIL is on sys.path by the time this is called."""
    # Eugen's CombinedD-A-S name has an 'A' slot, but that name is NOT sufficient: the
    # body atlas and the track/chains atlas are BOTH "CombinedDAS", yet only the latter
    # packs a real transparency CUTOUT in its alpha. The body's CombinedDAS alpha is a
    # packed SPEC mask (mostly mid-grey, ZERO fully-transparent pixels) — wiring it to
    # BSDF Alpha makes the hull see-through (the ikv_103/strv_103 'Chains' bug). So the
    # decision MUST be made by alpha CONTENT, with the name only used to reject the
    # CombinedDS family (which never carries a cutout) early.
    name = Path(str(png_path)).name.lower()
    if ("combds" in name) or ("combinedds" in name) or ("ddstexture" in name):
        return False
    # A genuine CUTOUT has real, mostly-binary transparency: a meaningful fraction of
    # FULLY-transparent pixels (the gaps between chain links / track teeth). A packed
    # SPEC mask has ~no transparent pixels (it is a grey gradient). Require real
    # transparency regardless of the DAS/DS naming. PIL is on sys.path by now.
    try:
        from PIL import Image
        im = Image.open(str(png_path))
        if im.mode not in ("RGBA", "LA"):
            return False
        a = im.getchannel("A").resize((64, 128))
        data = list(a.getdata())
        n = len(data) or 1
        fz = sum(1 for v in data if v < 32) / n           # fully-transparent fraction
        # Opaque region: a CombinedDAS cutout's "solid" links peak at ~195 (0.76), NOT
        # 255, so threshold at 150 (a flat spec mask sits at ~128 and never reaches it).
        fop = sum(1 for v in data if v > 150) / n          # opaque-ish fraction
        # Cutout == real transparency (>=5% transparent) AND a clear opaque region
        # (>=5% opaque) -> a binary-ish mask. A packed SPEC mask (the body atlas) has
        # ~0 transparent pixels (fz fails) and sits flat at mid-grey (fop fails too).
        return fz >= 0.05 and fop >= 0.05
    except Exception:
        return False


def _resolve_rd_textures(
    extractor_mod,
    spk,
    settings: "WARNOImporterSettings",
    asset: str,
    model_dir: Path,
    material_ids: Sequence[int],
    material_name_by_id: Dict[int, str],
    runtime_info: Dict[str, Any] | None,
    report: dict,
) -> tuple[Dict[str, Dict[str, Path]], dict]:
    """Resolve Wargame RD / SD2 unit textures: material ref (ZZ:/PC/TextureGroup/.../X.png)
    -> ZZ .tgv (TextureGroup->texture, .png->.tgv via ZZDatResolver.find) -> PNG
    (tgv_to_png, ZIPO/zlib + DXTn) -> {role: Path} per material. No atlas/C# needed."""
    import importlib.util
    maps_by_name: Dict[str, Dict[str, Path]] = {}
    runtime_info = runtime_info or {}
    project_root = _project_root(settings)

    # Make PIL + zstandard importable in-process (Blender's Python lacks them; the
    # plugin bundles them under .warno_pydeps/py<ver>, same as the WARNO converter).
    _pyver = "py%d%d" % (sys.version_info[0], sys.version_info[1])
    _deps = _resolve_path(project_root, str(getattr(settings, "tgv_deps_dir", "") or ".warno_pydeps").strip()) / _pyver
    if _deps.is_dir() and str(_deps) not in sys.path:
        sys.path.insert(0, str(_deps))

    tgv_path = project_root / "tgv_to_png.py"
    spec = importlib.util.spec_from_file_location("warno_tgv_conv", str(tgv_path))
    tgvm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgvm)

    zz = runtime_info.get("zz_resolver")
    warno_root_text = str(runtime_info.get("warno_root", "") or getattr(settings, "warno_root", "") or "").strip()
    if zz is None:
        zz = extractor_mod.get_zz_runtime_resolver(Path(warno_root_text), game=settings.game_preset)
    runtime_root = Path(str(runtime_info.get("runtime_root", "")).strip() or (project_root / "out_blender_runtime" / "zz_runtime"))
    out_tex = model_dir / "textures"
    try:
        out_tex.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        refs_by_material = spk.get_texture_refs_for_material_ids(material_ids)
    except Exception:
        refs_by_material = {}

    png_cache: Dict[str, Path] = {}

    def convert_ref(ref: str):
        q = str(ref).split(":", 1)[-1]
        try:
            q = extractor_mod.normalize_asset_path(q)
        except Exception:
            pass
        hit = zz.find(q)
        if not hit:
            return None
        key = str(hit.get("path", ""))
        if key in png_cache:
            return png_cache[key]
        out_png = out_tex / (Path(key).stem + ".png")
        if not out_png.exists():
            try:
                tgv_file = zz.extract_hit_to_runtime(hit, runtime_root)
                tgvm.convert_single_tgv(tgv_file, out_png)
            except Exception:
                return None
        if out_png.exists():
            png_cache[key] = out_png
            return out_png
        return None

    # Prefer the combined diffuse+spec texture as the base (it carries the spec
    # mask in its alpha); fall back to a plain diffuse, then anything else. The
    # "combined" check is by name, not classify role, so CombinedDS (which the
    # role classifier reports as generic) still wins and gets its alpha wired.
    base_priority = {"combined_da": 1, "diffuse": 2, "generic": 3, "orm": 5}
    for mid in material_ids:
        name = material_name_by_id.get(int(mid)) or f"Material_{int(mid):03d}"
        refs = refs_by_material.get(int(mid)) or []
        maps: Dict[str, Path] = {}
        best_base = None
        best_base_combined = False
        best_pri = 99
        for ref in refs:
            role = extractor_mod.classify_texture_role(ref)
            png = convert_ref(ref)
            if not png:
                continue
            if role == "normal":
                maps.setdefault("normal", png)
            elif role == "orm":
                maps.setdefault("orm", png)
            # CombinedDS/DAS keep the specular mask in their alpha channel
            # (DiffuseTextureNoAlpha does not — its alpha is flat 255).
            is_combined = "combined" in Path(ref).name.lower()
            pri = 0 if is_combined else base_priority.get(role, 4)
            if role != "normal" and pri < best_pri:
                best_pri = pri
                best_base = png
                best_base_combined = is_combined
        if best_base is not None:
            maps["diffuse"] = best_base
            if best_base_combined:
                # CombinedDAS-style alpha is a transparency cutout (chains/teeth);
                # CombinedDS-style alpha is a spec mask. Decide by content — EXCEPT for
                # TRACK (chenille) materials: the chenille CombinedDAS alpha IS a real cutout
                # (sprocket-hole / link gaps) but only ~4% of the atlas is transparent, just
                # under the content-cutout threshold, so it gets misclassified as a spec mask
                # -> the diffuse is baked to RGB (alpha dropped) -> the track loses its cutout
                # (the "white alpha" + solid black gaps the user saw). The TGV name "combdas"
                # already meant cutout; the content check over-corrected. Force cutout for tracks
                # AND chains (the ball-chain curtain cards are alpha cutouts too — same ~4%
                # transparent-atlas misclassification, e.g. merkava LODs where the split doesn't run).
                _is_track_mat = "chenille" in str(name).lower() or "chain" in str(name).lower()
                if _is_track_mat or _rd_alpha_is_cutout(best_base):
                    maps["alpha_cutout"] = best_base
                else:
                    maps["spec_alpha"] = best_base
        if maps:
            # Clean texture names: copy each resolved map to "<material>_<channel>.png"
            # so the Blender node shows a sensible name instead of the raw eugen
            # TextureGroup stem ("tsccombds_combineddstexture01.png"). Copy (not move)
            # so the shared png_cache originals stay valid; repoint maps at the copies.
            try:
                _safe = extractor_mod.sanitize_material_name(name)
            except Exception:
                _safe = ""
            if not _safe:
                _safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or f"Material_{int(mid):03d}"
            import shutil as _sh
            _renamed_src: Dict[str, Path] = {}
            for _role, _suf in (("diffuse", "D"), ("normal", "NM"), ("orm", "ORM")):
                _src = maps.get(_role)
                if not isinstance(_src, Path) or not _src.exists():
                    continue
                _dst = out_tex / f"{_safe}_{_suf}.png"
                try:
                    if _role == "diffuse" and ("spec_alpha" in maps) and ("alpha_cutout" not in maps):
                        # Body CombinedDS alpha is a SPEC mask (mostly ~0 -> the file
                        # reads as ~98% transparent). Bake a NORMAL opaque diffuse at the
                        # FILE level (the in-game round-trip reads the texture file, not
                        # the Blender node graph, so an in-Blender invert never helps):
                        # drop the spec alpha -> a clean opaque colour texture.
                        from PIL import Image as _PILImage
                        _PILImage.open(str(_src)).convert("RGB").save(str(_dst))
                    elif _dst != _src and (not _dst.exists() or (_role == "diffuse" and "alpha_cutout" in maps)):
                        # cutout diffuse MUST keep its alpha -> overwrite any stale RGB copy
                        # (a previous build baked it to RGB and the not-exists guard would
                        # otherwise keep the stale alpha-less file).
                        _sh.copyfile(str(_src), str(_dst))
                    if _dst.exists():
                        _renamed_src[str(_src)] = _dst
                        maps[_role] = _dst
                except Exception:
                    pass
            # spec_alpha / alpha_cutout point at the same combined file as diffuse —
            # repoint them at the renamed copy too.
            for _ak in ("spec_alpha", "alpha_cutout"):
                _av = maps.get(_ak)
                if isinstance(_av, Path) and str(_av) in _renamed_src:
                    maps[_ak] = _renamed_src[str(_av)]
            # Mark this as a Wargame-RD/SD2 material so the importer uses the
            # RD-specific node graph (spec-in-alpha -> roughness) instead of the
            # WARNO atlas graph. WARNO materials never carry this key.
            maps["rd_material"] = True
            maps_by_name[name] = maps
            report["resolved"].append(name)
    report["atlas_mode"] = "rd_texturegroup"
    report["atlas_source"] = "zz_texturegroup"
    _warno_log(settings, f"RD textures resolved: {len(maps_by_name)} materials, {len(png_cache)} unique pngs", stage="texture")
    return maps_by_name, report


def _repair_track_uv_seams(mesh, uv_layer, bucket_uvs, thresh=0.4):
    """Un-stretch the few RD-track triangles that straddle the texture wrap, KEEPING the
    exact decoded game UV on every other triangle (1:1). The codec's `& mask` collapses
    the texture wrap edge to 0, so a handful of triangles get one corner on the opposite
    side of the tile from the others; mapped onto the CombinedDAS sub-rect they smear the
    texture diagonally across the triangle ("трикутники летять"). They can't be made
    contiguous inside a single sub-rect copy (shifting a corner by a tile would leave the
    rect and sample the chains). So collapse each such triangle's UV to its centroid: it
    then samples ONE texel (flat) instead of a smear. The track diffuse is near-uniform
    dark scaly, so a flat triangle is invisible — only the stretched artefact is removed.
    Every non-straddling triangle keeps its exact decoded UV. RD/SD2 track only."""
    n = len(bucket_uvs)
    for poly in mesh.polygons:
        loops = list(poly.loop_indices)
        vis = [mesh.loops[li].vertex_index for li in loops]
        us = [bucket_uvs[vi][0] if vi < n else 0.0 for vi in vis]
        vs = [bucket_uvs[vi][1] if vi < n else 0.0 for vi in vis]
        if (max(us) - min(us) > thresh) or (max(vs) - min(vs) > thresh):
            cu = sum(us) / len(us); cv = sum(vs) / len(vs)
            for li in loops:
                uv_layer.data[li].uv = (cu, cv)
        else:
            for li, vi in zip(loops, vis):
                uv_layer.data[li].uv = bucket_uvs[vi] if vi < n else (0.0, 0.0)


def _crop_scaly_for_track(combined_png):
    """Crop the scaly-armour chunk out of a unit's CombinedDAS for use as a TILING track
    tread texture. CombinedDAS = scaly armour (image top-left) + sawtooth teeth (right
    edge) + ball-chains (bottom); the track has no dedicated tread texture, so the scaly
    armour is the closest usable one. Crop a clean inner chunk (no teeth, no chains) that
    tiles under REPEAT along the tube-unwrapped belt. Returns a Path or None."""
    try:
        from PIL import Image
        p = Path(str(combined_png))
        im = Image.open(str(p)).convert("RGB")
        W, H = im.size
        crop = im.crop((int(0.06 * W), int(0.04 * H), int(0.62 * W), int(0.42 * H)))
        out = p.with_name(p.stem + "_trackscaly.png")
        crop.save(str(out))
        return out
    except Exception:
        return None


def _cube_project_track_uv(mesh, uv_layer, tiles=40.0):
    """Per-face triplanar (cube) UV for an RD track (chenille) belt. The belt has faces in
    several orientations — the radial TREAD faces AND the flat ±side faces you see from the
    side. A single tube/cylindrical unwrap gives the side faces a DEGENERATE UV (constant
    across one axis -> the chevron/herringbone smear). Triplanar projects EACH face onto
    its own dominant-axis plane at a uniform world scale, so every face — tread or side —
    gets a clean, non-stretched, tiling map. Paired with the REPEAT scaly-crop material.
    `tiles` sets texel density (bbox span / tiles per repeat). RD/SD2 track only."""
    verts = mesh.vertices
    if len(verts) == 0:
        return
    xs = [v.co.x for v in verts]; ys = [v.co.y for v in verts]; zs = [v.co.z for v in verts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) or 1.0
    S = (span / float(tiles)) or 1.0
    for poly in mesh.polygons:
        nm = poly.normal
        ax, ay, az = abs(nm.x), abs(nm.y), abs(nm.z)
        if ax >= ay and ax >= az:
            comp = 0
        elif ay >= az:
            comp = 1
        else:
            comp = 2
        for li, vi in zip(poly.loop_indices, poly.vertices):
            co = verts[vi].co
            if comp == 0:
                uu, vv = co.y, co.z
            elif comp == 1:
                uu, vv = co.x, co.z
            else:
                uu, vv = co.x, co.y
            uv_layer.data[li].uv = (uu / S, vv / S)


def _tube_unwrap_track(verts):
    """Tube/cylindrical UV unwrap for an RD track (chenille) belt — fixes the texture
    stretch at the CURVED ends that the decoded file UV has (the game regenerates a
    per-link UV at runtime via IndexedChenilles; that runtime UV is not in the file, so
    the file UV maps the belt as if flat and smears around the idler/sprocket).

    The belt is a closed loop band. We unwrap it as a tube: U = arc length around the
    loop, V = position across the belt width — giving uniform, square texels everywhere
    INCLUDING the curves. Robust to the model's orientation and to the part holding BOTH
    tracks (left+right): PCA finds the principal axes; the most BIMODAL axis is the
    width/separation axis (the gap between the two tracks); the loop lies in the other
    two axes. Each track is unwrapped independently around its own centre.

    `verts` is a flat list of (x,y,z) in the SAME space the mesh is built in (bucket
    space). Returns a per-vertex list of (u,v); U tiles along the belt (≈ perimeter/width
    so texels stay square), V in [0,1] across the width. Caller folds V into the texture
    sub-rect. RD/SD2 track only — never called for WARNO geometry."""
    import numpy as np
    import math as _m
    P = np.asarray(verts, dtype=float).reshape(-1, 3)
    n = len(P)
    out = [(0.0, 0.0)] * n
    if n < 6:
        return out
    c = P.mean(0)
    Q = P - c
    # PCA (eigh -> ascending eigenvalues; columns are axes)
    try:
        w, v = np.linalg.eigh(Q.T @ Q)
    except Exception:
        return out
    A = Q @ v   # projections onto principal axes
    # Width/separation axis = the most bimodal projection (largest empty central gap).
    def _gap(vals):
        if vals.max() - vals.min() < 1e-6:
            return 0
        h, _ = np.histogram(vals, bins=12)
        return int((h[3:9] == 0).sum())
    waxis = int(np.argmax([_gap(A[:, i]) for i in range(3)]))
    other = [i for i in range(3) if i != waxis]
    wv = A[:, waxis]
    split = (wv.max() + wv.min()) / 2.0   # between the two track clusters
    res = np.zeros((n, 2))
    N = 240
    for side in (0, 1):
        mask = (wv < split) if side == 0 else (wv >= split)
        idx = np.where(mask)[0]
        if len(idx) < 3:
            continue
        a = A[idx][:, other[0]]
        b = A[idx][:, other[1]]
        ws = A[idx][:, waxis]
        ca = a.mean(); cb = b.mean()
        th = np.arctan2(b - cb, a - ca)               # angle around the loop
        binix = (((th + _m.pi) / (2 * _m.pi)) * N).astype(int) % N
        cent = np.zeros((N, 2)); cnt = np.zeros(N)
        for k in range(len(idx)):
            bi = binix[k]; cent[bi, 0] += a[k]; cent[bi, 1] += b[k]; cnt[bi] += 1
        filled = [bi for bi in range(N) if cnt[bi] > 0]
        if len(filled) < 3:
            continue
        for bi in filled:
            cent[bi] /= cnt[bi]
        # cumulative TRUE arc length around the ordered loop (not raw angle -> no end
        # compression on the elongated stadium loop)
        arclen = np.zeros(N); cum = 0.0
        for i in range(1, len(filled)):
            p = cent[filled[i]]; qq = cent[filled[i - 1]]
            cum += _m.hypot(p[0] - qq[0], p[1] - qq[1]); arclen[filled[i]] = cum
        p = cent[filled[0]]; qq = cent[filled[-1]]
        perim = cum + _m.hypot(p[0] - qq[0], p[1] - qq[1])
        if perim < 1e-6:
            perim = 1.0
        width = float(ws.max() - ws.min()) or 1.0
        tiles = perim / width                          # square texels
        wmin = float(ws.min())
        for k in range(len(idx)):
            bi = binix[k]
            al = arclen[bi]
            if cnt[bi] == 0:
                bi = min(filled, key=lambda x: abs(x - bi)); al = arclen[bi]
            res[idx[k], 0] = al / perim * tiles
            res[idx[k], 1] = (ws[k] - wmin) / width
    return [(float(res[i, 0]), float(res[i, 1])) for i in range(n)]


def _split_track_chains_textures(model, material_name_by_id, maps_by_name):
    """RD/SD2: split the shared CombinedDAS atlas into a separate TRACKS texture (the tread
    = top half) and a CHAINS texture (ball-chains = bottom half), each written as its own
    PNG with a generated BINARY ALPHA map (WARNO-style — the alpha PNG is a clean 0/255
    cutout that plugs straight into BSDF Alpha, so the shader needs no GreaterThan node).

    The decoded UV is kept exactly as the game has it: the track samples the top of the
    atlas, the chains the bottom. So each split file keeps its own half and BLANKS the
    other half (RGB->black, alpha->0); that way each material owns a texture with only its
    content while the existing UV still lands on the right pixels (no UV remap needed).

    Files are named from the source atlas texture: <stem>_tracks.png / _tracks_alpha.png
    and <stem>_chains.png / _chains_alpha.png. Track = the chenille part's material; chains
    = a non-track material whose name contains 'chain'. Gated on the chenille flag, so
    WARNO never enters this path."""
    try:
        from PIL import Image
        parts = model.get("parts") or []

        def mname(mid):
            return str((material_name_by_id or {}).get(int(mid), "") or "")

        track_mids = {
            int(p["material"]) for p in parts
            if isinstance(p.get("vertices"), dict) and p["vertices"].get("chenille")
            and p.get("material") is not None
        }
        # Track parts baked from a POST-VS getter carry tread_full_uv: the game binds the
        # FULL CombinedDAS tread strip (256x256) and the UV spans its full height. For those
        # use the WHOLE texture (no top-half crop) and DON'T remap the UV into a crop sub-rect
        # — that half-crop+remap was the stretch/seam bug. File-only/legacy tracks keep the crop.
        track_full_uv = {
            int(p["material"]) for p in parts
            if isinstance(p.get("vertices"), dict) and p["vertices"].get("chenille")
            and p["vertices"].get("tread_full_uv") and p.get("material") is not None
        }
        # WARNO-safety: this whole split (track AND chains) is an RD/SD2-only feature.
        # Gate it on the chenille flag — if the model has no track part, do NOT touch any
        # material (the chain-name match alone is NOT a reliable RD signal, and WARNO must
        # stay byte-for-byte unchanged). Only RD/SD2 track units have a chenille part.
        if not track_mids:
            return
        chain_mids = {
            int(p["material"]) for p in parts
            if p.get("material") is not None and int(p["material"]) not in track_mids
            and "chain" in mname(p["material"]).lower()
        }

        def _make(combined_png, region, full=False):
            # full=True: the WHOLE CombinedDAS (the complete tread strip, game-exact). Else a
            # TRUE 1:1 crop of the part's half of the atlas (256x256), not a blanked copy.
            p = Path(str(combined_png))
            im = Image.open(str(p)).convert("RGBA")
            W, H = im.size
            if full:
                crop = im
            else:
                half = H // 2
                box = (0, 0, W, half) if region == "tracks" else (0, half, W, H)
                crop = im.crop(box)
            r, g, b, a = crop.split()
            rgb = Image.merge("RGB", (r, g, b))
            abin = a.point(lambda v: 255 if v >= 76 else 0).convert("L")  # binary cutout
            # WARNO-style clean texture names: "<base>_D.png" (diffuse) + "<base>_A.png"
            # (alpha) — the source diffuse is already "<base>_D.png" (e.g. Droite_D.png).
            # (was the verbose "<stem>_tracks_full[_alpha].png".)
            _base = p.stem[:-2] if p.stem.lower().endswith("_d") else p.stem
            rgb_p = p.with_name(_base + "_D.png"); rgb.save(str(rgb_p))
            # If the source atlas alpha for this region is FULLY OPAQUE, the binary mask is a
            # useless blank-WHITE PNG (e.g. the leopard tread strip alpha = all 255). Wiring it
            # into BSDF Alpha then renders the belt WHITE/washed in Material-Preview/Solid on
            # some GPUs (user: "track alpha texture is completely white"). A solid belt needs no
            # cutout -> emit no alpha file; the caller renders it opaque.
            if abin.getextrema()[0] >= 255:
                return rgb_p, None
            al_p = p.with_name(_base + "_A.png"); abin.save(str(al_p))
            return rgb_p, al_p

        cache = {}

        def _apply(mid, region):
            nm = (material_name_by_id or {}).get(int(mid))
            m = (maps_by_name or {}).get(nm) if nm else None
            if not isinstance(m, dict):
                return
            comb = m.get("diffuse")
            if not comb:
                return
            full = (region == "tracks" and int(mid) in track_full_uv)
            key = (str(comb), region, full)
            if key not in cache:
                cache[key] = _make(comb, region, full)
            rgb_p, al_p = cache[key]
            for k in ("alpha_cutout", "spec_alpha", "track_repeat", "force_opaque", "uv_crop"):
                m.pop(k, None)
            m["diffuse"] = rgb_p
            if al_p is not None:
                m["alpha"] = al_p             # separate binary alpha (WARNO-style)
            else:
                # No cutout in the source alpha -> render the belt OPAQUE (do NOT wire a
                # blank-white alpha). track_repeat = opaque + REPEAT so the tread UV tails
                # outside [0,1] still tile seamlessly.
                m.pop("alpha", None)
                m["track_repeat"] = True
            if full:
                # FULL game tread strip + the post-VS UV used 1:1 (no crop remap). REPEAT (set
                # via the alpha path) tiles the few V values just outside [0,1]. This is the
                # game-exact path (texture pixel-identical to CombinedDAS, UV = post-VS).
                pass
            else:
                # The crop covers this Blender-V region of the full atlas; the importer remaps
                # the decoded UV into [0,1] of the crop so it fills the texture like the get.
                m["uv_crop"] = (0.5, 1.0) if region == "tracks" else (0.0, 0.5)

        for mid in track_mids:
            _apply(mid, "tracks")
        for mid in chain_mids:
            _apply(mid, "chains")
    except Exception:
        pass


# Back-compat alias for existing call sites.
_force_opaque_baked_tracks = _split_track_chains_textures


def _resolve_material_maps(
    extractor_mod,
    spk,
    settings: WARNOImporterSettings,
    asset: str,
    model_dir: Path,
    material_ids: Sequence[int],
    material_name_by_id: Dict[int, str],
    material_role_by_id: Dict[int, str],
    runtime_info: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Dict[str, Path]], dict[str, Any]]:
    maps_by_name: Dict[str, Dict[str, Path]] = {}
    stage_t0 = time.monotonic()
    report: dict[str, Any] = {
        "refs": [],
        "refs_by_material": {},
        "resolved": [],
        "errors": [],
        "named": [],
        "channels": [],
        "atlas_source": "",
        "converter_source": "",
        "deps_auto_installed": False,
        "atlas_mode": "legacy",
        "atlas_map_entries": 0,
        "atlas_map_targets": 0,
        "exact_hits": 0,
        "strict_misses": 0,
    }
    if not settings.auto_textures:
        return maps_by_name, report

    # Wargame RD / SD2 store unit textures as standalone .tgv in the ZZ TextureGroup
    # (no WARNO-style atlas). Resolve them directly and return BEFORE the atlas-path
    # validation below (which requires a WARNO atlas folder RD does not have).
    try:
        _gid = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))
    except Exception:
        _gid = "WARNO"
    if _gid in ("WARGAME_RD", "STEEL_DIVISION_2"):
        try:
            return _resolve_rd_textures(
                extractor_mod, spk, settings, asset, model_dir,
                material_ids, material_name_by_id, runtime_info, report,
            )
        except Exception as exc:  # never block geometry import on texture failure
            _warno_log(settings, f"RD texture resolve failed: {exc}", stage="texture")
            report["errors"].append(f"rd_texture:{exc}")
            return maps_by_name, report

    _warno_log(settings, f"texture resolve start asset={asset}", stage="texture")

    project_root = _project_root(settings)
    runtime_info = runtime_info or {}
    atlas_override = str(runtime_info.get("atlas_assets_root", "")).strip()
    zz_resolver = runtime_info.get("zz_resolver")
    zz_runtime_root_text = str(runtime_info.get("runtime_root", "")).strip()
    warno_root_text = str(runtime_info.get("warno_root", "")).strip()

    atlas_raw_text = atlas_override or str(settings.atlas_assets_dir or "").strip()
    atlas_source = "zz_runtime" if atlas_override else "manual_path"
    report["atlas_source"] = atlas_source
    bundled_converter = project_root / "tgv_to_png.py"
    converter = bundled_converter
    converter_text = str(bundled_converter)
    converter_source = "bundled"

    if not atlas_raw_text:
        raise RuntimeError("Atlas Assets path is empty.")
    atlas_raw = _resolve_path(project_root, atlas_raw_text)
    if not atlas_raw.exists() or not atlas_raw.is_dir():
        raise RuntimeError(f"Atlas Assets folder not found: {atlas_raw}")
    if not converter.exists() or not converter.is_file():
        raise RuntimeError(f"Bundled TGV converter not found: {converter}")
    report["converter_source"] = converter_source

    atlas_root = extractor_mod.resolve_atlas_assets_root(atlas_raw)
    fallback_atlas_roots: List[Path] = []
    if warno_root_text:
        warno_root_path = Path(warno_root_text)
        for rel in (
            Path("Output") / "PC" / "Atlas" / "Assets",
            Path("Mods") / "ModData" / "base" / "PC" / "Atlas" / "Assets",
        ):
            cand = warno_root_path / rel
            try:
                cand_resolved = extractor_mod.resolve_atlas_assets_root(cand)
            except Exception:
                cand_resolved = cand
            if cand_resolved.exists() and cand_resolved.is_dir():
                try:
                    if cand_resolved.resolve() != atlas_root.resolve():
                        fallback_atlas_roots.append(cand_resolved)
                except Exception:
                    if str(cand_resolved) != str(atlas_root):
                        fallback_atlas_roots.append(cand_resolved)
    refs_by_material: Dict[int, List[str]] = {}
    try:
        refs_by_material = spk.get_texture_refs_for_material_ids(material_ids)
    except Exception:
        refs_by_material = {}

    atlas_map_index: Dict[str, List[Dict[str, Any]]] | None = None
    atlas_map_path: Path | None = None
    atlas_map_targets: List[str] = []
    atlas_map_asset_path = str(asset)
    atlas_init_error = ""
    if bool(settings.use_atlas_json_mapping):
        _warno_log(settings, "stage: atlas_map_build", stage="atlas_map_build")
        report["atlas_mode"] = "json_export"
        warno_root_for_atlas = Path(warno_root_text) if warno_root_text else _resolve_path(project_root, str(settings.warno_root or "").strip())
        modsuite_root = _resolve_path(project_root, str(settings.modding_suite_root or "").strip())
        wrapper_path = _resolve_path(project_root, str(settings.modding_suite_atlas_wrapper or "").strip() or "modding_suite_atlas_export.py")
        atlas_cli_path_text = str(settings.modding_suite_atlas_cli or "").strip()
        atlas_cli_path = _resolve_path(project_root, atlas_cli_path_text) if atlas_cli_path_text else None
        atlas_cache_root = _atlas_json_cache_root(settings)
        atlas_candidates = _atlas_asset_variant_candidates(asset)
        if not warno_root_for_atlas.exists() or not warno_root_for_atlas.is_dir():
            atlas_init_error = f"WARNO folder for Atlas JSON not found: {warno_root_for_atlas}"
        elif not modsuite_root.exists() or not modsuite_root.is_dir():
            atlas_init_error = f"moddingSuite root for Atlas JSON not found: {modsuite_root}"
        else:
            for idx, atlas_asset_candidate in enumerate(atlas_candidates):
                try:
                    atlas_map_info = extractor_mod.build_or_load_atlas_texture_map(
                        warno_root=warno_root_for_atlas,
                        modding_suite_root=modsuite_root,
                        asset_path=atlas_asset_candidate,
                        cache_dir=atlas_cache_root,
                        wrapper_path=wrapper_path,
                        atlas_cli_path=atlas_cli_path,
                        force_rebuild=False,
                        timeout_sec=max(5, int(settings.atlas_cli_timeout_sec)),
                        game=_normalize_game_id(getattr(settings, "game_preset", "WARNO")),
                    )
                    atlas_map_path_text = str(atlas_map_info.get("atlas_map_path", "")).strip()
                    atlas_map_path = Path(atlas_map_path_text) if atlas_map_path_text else None
                    atlas_map_index_raw = atlas_map_info.get("atlas_map_index")
                    if isinstance(atlas_map_index_raw, dict):
                        atlas_map_index = atlas_map_index_raw
                    atlas_targets_raw = atlas_map_info.get("atlas_map_targets")
                    if isinstance(atlas_targets_raw, list):
                        atlas_map_targets = [str(x).strip() for x in atlas_targets_raw if str(x).strip()]
                    atlas_map_asset_path = str(atlas_asset_candidate or asset)
                    report["atlas_map_entries"] = int(atlas_map_info.get("atlas_map_entries", 0) or 0)
                    report["atlas_map_targets"] = len(atlas_map_targets)
                    extra_note = ""
                    if _norm_ref_like(atlas_map_asset_path) != _norm_ref_like(asset):
                        extra_note = f" asset_fallback={atlas_map_asset_path}"
                    _warno_log(
                        settings,
                        f"atlas map ready entries={report['atlas_map_entries']} path={atlas_map_path_text}{extra_note}",
                        stage="atlas_map_build",
                    )
                    atlas_init_error = ""
                    break
                except Exception as exc:
                    atlas_init_error = str(exc)
                    if (
                        idx + 1 < len(atlas_candidates)
                        and _atlas_map_no_entries_error(atlas_init_error)
                    ):
                        _warno_log(
                            settings,
                            f"atlas map no entries for {atlas_asset_candidate}; trying variant fallback",
                            level="WARNING",
                            stage="atlas_map_build",
                        )
                        continue
                    break
        if atlas_init_error:
            _warno_log(settings, f"atlas map build failed: {atlas_init_error}", level="ERROR", stage="atlas_map_build")
            report["errors"].append(
                {
                    "atlas_ref": "__atlas_json__",
                    "error": f"missing_source: atlas json mapping unavailable: {atlas_init_error}",
                }
            )
            report["atlas_mode"] = "json_export_failed"
            if bool(settings.atlas_json_strict):
                _warno_log(settings, f"atlas json strict stop: {atlas_init_error}", level="WARNING", stage="texture")
                return maps_by_name, report

    refs: List[str] = []
    for mid in material_ids:
        refs.extend(refs_by_material.get(int(mid), []))
    refs = extractor_mod.unique_keep_order(refs) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(refs))
    if not refs:
        refs = spk.find_texture_refs_for_asset(asset, material_ids=material_ids)
    refs = extractor_mod.unique_keep_order(refs) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(refs))

    atlas_json_enabled = bool(settings.use_atlas_json_mapping and atlas_map_index)
    strict_atlas_mode = bool(atlas_json_enabled and settings.atlas_json_strict)
    if strict_atlas_mode:
        removed_legacy = _cleanup_strict_legacy_alias_pngs(model_dir=model_dir, asset=asset)
        if removed_legacy > 0:
            _warno_log(
                settings,
                f"strict atlas cleanup removed legacy top-level png: {removed_legacy}",
                stage="texture",
            )
        if atlas_map_targets:
            # WARNO DECORS atlases pack textures for many sibling FBX into one
            # composite atlas. Without filtering, strict mode would extract all
            # of them (e.g. 45 PNG for HLM_10_L.fbx where only ~5 are actually
            # referenced by the imported geometry). Try to narrow the target
            # list to those whose source FBX matches an SPK material name; on
            # any uncertainty fall back to the full target list so we never
            # accidentally drop required textures for a regular UNIT asset.
            filtered_targets = _filter_atlas_targets_by_materials(
                atlas_map_index=atlas_map_index,
                material_name_by_id=material_name_by_id,
                asset_hint=asset,
            )
            targets_to_use = (
                filtered_targets
                if filtered_targets is not None and filtered_targets
                else atlas_map_targets
            )
            refs_before = len(refs)
            refs = extractor_mod.unique_keep_order([*refs, *targets_to_use]) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys([*refs, *targets_to_use]))
            refs_added = len(refs) - refs_before
            if refs_added > 0:
                if filtered_targets is not None and len(filtered_targets) < len(atlas_map_targets):
                    note = f" (filtered {len(atlas_map_targets)} -> {len(filtered_targets)} by material match)"
                else:
                    note = ""
                _warno_log(
                    settings,
                    f"strict atlas refs extended from atlas targets: +{refs_added}{note}",
                    stage="refs_resolve",
                )
    if atlas_json_enabled and not refs:
        refs = list(atlas_map_targets)
        if refs:
            _warno_log(
                settings,
                f"refs fallback from atlas map targets: {len(refs)}",
                stage="refs_resolve",
            )
    guessed_refs: List[str] = []
    if not atlas_json_enabled:
        if refs and bool(settings.use_zz_dat_source) and bool(settings.fast_exact_texture_resolve):
            companion = _build_fast_companion_refs(extractor_mod, refs, cap=24)
            companion_exact = [
                ref
                for ref in companion
                if _atlas_ref_exact_exists(
                    extractor_mod=extractor_mod,
                    ref=ref,
                    atlas_root=atlas_root,
                    fallback_roots=fallback_atlas_roots,
                    zz_resolver=zz_resolver,
                )
            ]
            if companion_exact:
                refs = extractor_mod.unique_keep_order(list(refs) + list(companion_exact)) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(list(refs) + list(companion_exact)))
        elif refs:
            try:
                guessed_refs = _guess_texture_refs_for_asset(
                    extractor_mod=extractor_mod,
                    asset=asset,
                    atlas_assets_root=atlas_root,
                    runtime_info=runtime_info,
                    max_refs=48,
                    include_zz_scan=bool(settings.use_zz_dat_source),
                )
            except Exception:
                guessed_refs = []
            if guessed_refs:
                low_refs = [str(r).lower() for r in refs]
                has_nm = any(("normaltexture" in r) or ("_nm" in r) or ("/tscnm_" in r) for r in low_refs)
                has_orm = any(
                    ("combinedorm" in r)
                    or ("combinedrm" in r)
                    or ("roughnesstexture" in r)
                    or ("singlechannellinearmap_roughness" in r)
                    or ("singlechannellinearmap_metallic" in r)
                    for r in low_refs
                )
                if (not has_nm) or (not has_orm) or len(refs) <= 2:
                    refs = extractor_mod.unique_keep_order(list(refs) + list(guessed_refs)) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(list(refs) + list(guessed_refs)))
        elif not refs:
            try:
                guessed_refs = _guess_texture_refs_for_asset(
                    extractor_mod=extractor_mod,
                    asset=asset,
                    atlas_assets_root=atlas_root,
                    runtime_info=runtime_info,
                    max_refs=24,
                    include_zz_scan=bool(settings.use_zz_dat_source),
                )
            except Exception:
                guessed_refs = []
            refs = guessed_refs

    _warno_log(settings, f"stage: refs_resolve refs={len(refs)}", stage="refs_resolve")
    report["refs"] = list(refs)
    report["refs_by_material"] = {
        str(int(mid)): list(refs_by_material.get(int(mid), []))
        for mid in material_ids
    }
    if refs and not any(report["refs_by_material"].values()):
        report["refs_by_material"]["__guessed__"] = list(refs)
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    resolved_by_ref: Dict[str, dict[str, Any]] = {}
    used_bundled_fallback = False
    allow_bundled_fallback = (
        converter_source == "custom"
        and bundled_converter.exists()
        and bundled_converter.is_file()
    )
    deps_dir = _resolve_path(project_root, str(settings.tgv_deps_dir or "").strip() or ".warno_pydeps")
    stage_start = time.monotonic()
    stage_timeout_sec = max(30, int(settings.texture_stage_timeout_sec))
    process_timeout_sec = max(10, int(settings.texture_process_timeout_sec))
    total_refs = len(refs)
    for idx, ref in enumerate(refs, start=1):
        if (time.monotonic() - stage_start) > float(stage_timeout_sec):
            errors.append(
                {
                    "atlas_ref": str(ref),
                    "error": f"texture_stage_timeout: exceeded {stage_timeout_sec}s while resolving refs",
                }
            )
            _set_status(
                settings,
                f"stage: resolving textures ({idx}/{total_refs}) timeout>{stage_timeout_sec}s",
                log_level="WARNING",
                stage="texture",
            )
            break
        if idx == 1 or idx % 4 == 0 or idx == total_refs:
            _set_status(
                settings,
                f"stage: resolving textures ({idx}/{total_refs})",
                log_level="INFO",
                stage="texture",
            )
            # Move the visible progress bar/cursor during the (slow) per-texture TGV decode
            # so this phase doesn't look frozen. Textures span ~the 50%->60% band.
            try:
                _tpct = int(round(50.0 + 10.0 * (float(idx) / float(max(1, total_refs)))))
                _tc = 24
                _tf = max(0, min(_tc, int(round(_tc * _tpct / 100.0))))
                print(f"[WARNO] {'█' * _tf}{'░' * (_tc - _tf)} {_tpct:3d}% | "
                      f"resolving textures {idx}/{total_refs}", flush=True)
                _wm = bpy.context.window_manager
                if _wm is not None:
                    _wm.progress_update(int(_tpct))
                    _wm.status_text_set(f"WARNO import {_tpct}% — resolving textures ({idx}/{total_refs})")
            except Exception:
                pass

        def _resolve_with_converter(conv_path: Path):
            return extractor_mod.resolve_texture_from_atlas_ref(
                ref=ref,
                atlas_assets_root=atlas_root,
                out_model_dir=model_dir,
                converter=conv_path,
                texture_subdir="textures",
                zz_resolver=zz_resolver,
                zz_runtime_root=Path(zz_runtime_root_text) if zz_runtime_root_text else None,
                fallback_atlas_roots=fallback_atlas_roots,
                auto_install_deps=bool(settings.auto_install_tgv_deps),
                deps_dir=deps_dir,
                subprocess_timeout_sec=process_timeout_sec,
                atlas_map_path=atlas_map_path,
                atlas_map_index=atlas_map_index,
                asset_path=atlas_map_asset_path,
                atlas_json_strict=bool(settings.atlas_json_strict),
            )

        try:
            _warno_log(
                settings,
                f"converter ref {idx}/{total_refs}: {ref}",
                stage="converter",
            )
            item = _resolve_with_converter(converter)
            resolved.append(item)
            resolved_by_ref[str(item.get("atlas_ref", ""))] = item
        except Exception as tex_exc:
            if allow_bundled_fallback:
                try:
                    item = _resolve_with_converter(bundled_converter)
                    resolved.append(item)
                    resolved_by_ref[str(item.get("atlas_ref", ""))] = item
                    used_bundled_fallback = True
                    continue
                except Exception as tex_exc2:
                    tex_exc = tex_exc2
            errors.append({"atlas_ref": ref, "error": str(tex_exc)})

    # Strict resolve policy:
    # Do not auto-convert parent ZZ folders when direct refs failed. It can pull wrong textures.
    report["errors"] = errors
    if used_bundled_fallback:
        report["converter_source"] = "bundled"

    strict_picker = None
    if strict_atlas_mode:
        if hasattr(extractor_mod, "pick_maps_for_material_from_atlas_resolved"):
            strict_picker = lambda items, role: extractor_mod.pick_maps_for_material_from_atlas_resolved(
                resolved_items=items,
                material_role=role,
                asset_stem=Path(asset).stem,
            )
        else:
            _warno_log(
                settings,
                "strict atlas picker fallback: extractor helper missing, using local deterministic picker",
                level="WARNING",
                stage="texture",
            )
            strict_picker = lambda items, role: _pick_maps_for_material_strict_local(
                resolved_items=items,
                material_role=role,
                asset_stem=Path(asset).stem,
                extractor_mod=extractor_mod,
            )

    if strict_atlas_mode and strict_picker is not None:
        chosen_maps = strict_picker(resolved, "other")
    else:
        chosen_maps = extractor_mod.pick_material_maps_from_textures(resolved)
        chosen_maps = _augment_maps_from_existing_files(model_dir=model_dir, asset=asset, chosen_maps=chosen_maps)
    named_maps: Dict[str, Path] = {}
    named_files: list[dict[str, str]] = []
    if chosen_maps:
        if strict_atlas_mode and strict_picker is not None:
            strict_groups = _build_strict_grouped_maps(resolved, extractor_mod=extractor_mod)
            asset_stem_low = _norm_low(Path(asset).stem)
            non_track_groups = [(k, v) for k, v in strict_groups.items() if not bool(v.get("is_track", False))]
            track_groups = [(k, v) for k, v in strict_groups.items() if bool(v.get("is_track", False))]

            def _pick_primary_non_track() -> str:
                if not non_track_groups:
                    return ""
                asset_tokens = [t for t in re.split(r"[^a-z0-9]+", asset_stem_low) if t]
                best_key = ""
                best_score = float("-inf")
                for key, payload in non_track_groups:
                    maps_count = len(payload.get("maps", {}))
                    score = float(maps_count) * 20.0 + float(payload.get("count", 0))
                    if asset_stem_low and (key == asset_stem_low or key.startswith(f"{asset_stem_low}_")):
                        score += 200.0
                    if asset_tokens:
                        key_tokens = [t for t in re.split(r"[^a-z0-9]+", _norm_low(key)) if t]
                        overlap = len(set(asset_tokens).intersection(set(key_tokens)))
                        score += float(overlap) * 35.0
                    if score > best_score:
                        best_score = score
                        best_key = key
                return best_key

            def _pick_primary_track() -> str:
                if not track_groups:
                    return ""
                best_key = ""
                best_score = float("-inf")
                for key, payload in track_groups:
                    maps_count = len(payload.get("maps", {}))
                    score = float(maps_count) * 20.0 + float(payload.get("count", 0))
                    k_low = _norm_low(key)
                    if "tracks" in k_low or "_trk" in k_low or "chenille" in k_low:
                        score += 60.0
                    if score > best_score:
                        best_score = score
                        best_key = key
                return best_key

            primary_non_track_key = _pick_primary_non_track()
            primary_track_key = _pick_primary_track()
            secondary_non_track_keys = [
                key
                for key, payload in sorted(
                    non_track_groups,
                    key=lambda kv: (
                        len(kv[1].get("maps", {})),
                        int(kv[1].get("count", 0)),
                        kv[0],
                    ),
                    reverse=True,
                )
                if key != primary_non_track_key
            ]
            used_non_track_group_keys: set[str] = set()
            raw_slot_hints_by_mid = getattr(spk, "material_texture_names_by_id", {}) or {}

            def _maps_from_group_key(group_key: str, want_track: bool) -> tuple[Dict[str, Path], str]:
                key = _norm_low(group_key)
                if not key:
                    return {}, ""
                payload = strict_groups.get(key)
                if not isinstance(payload, dict):
                    return {}, ""
                if bool(payload.get("is_track", False)) != bool(want_track):
                    return {}, ""
                maps = payload.get("maps", {})
                if isinstance(maps, dict) and maps:
                    return dict(maps), key
                return {}, ""

            def _exact_material_key(raw_name: str) -> str:
                name = str(raw_name or "").strip()
                if not name:
                    return ""
                # Blender may append .001/.002 for duplicated material names.
                name = re.sub(r"\.\d{3}$", "", name)
                base = _strip_texture_channel_suffix(name) or name
                key = _norm_low(base)
                return key

            def _is_autogen_material_name(raw_name: str) -> bool:
                low = _norm_low(raw_name)
                if not low:
                    return True
                return (
                    low.startswith("material_")
                    or low.startswith("element_")
                    or low.startswith("part_")
                    or low in {"corps_principal", "body", "other"}
                )

            def _slot_priority(slot_name: str) -> int:
                low = _norm_low(slot_name).replace(" ", "")
                if "diffusetexturenoalpha" in low or "diffusetexture" in low:
                    return 0
                if "opacitytexture" in low or low.endswith("alpha"):
                    return 1
                if "normaltexture" in low or "normal" in low:
                    return 2
                if "roughnesstexture" in low or "roughness" in low:
                    return 3
                if "metallictexture" in low or "metallic" in low:
                    return 4
                if "ambientocclusiontexture" in low or "occlusion" in low:
                    return 5
                return 10

            def _maps_from_material_name_or_slot_hints(mid: int, material_name: str, want_track: bool) -> tuple[Dict[str, Path], str]:
                by_name, by_name_key = _maps_from_group_key(_exact_material_key(material_name), want_track)
                if by_name:
                    return by_name, by_name_key

                slot_map = raw_slot_hints_by_mid.get(int(mid), {})
                if isinstance(slot_map, dict) and slot_map:
                    slot_items = sorted(slot_map.items(), key=lambda kv: (_slot_priority(str(kv[0])), str(kv[0]).lower()))
                    for _slot_name, raw_value in slot_items:
                        stem = Path(str(raw_value or "")).stem
                        key = _exact_material_key(stem)
                        if not key:
                            continue
                        by_slot, by_slot_key = _maps_from_group_key(key, want_track)
                        if by_slot:
                            return by_slot, by_slot_key
                return {}, ""

            for mid in material_ids:
                mname = material_name_by_id.get(int(mid), f"Material_{int(mid):03d}")
                role = str(material_role_by_id.get(int(mid), "other"))
                role_for_pick = role
                if not role_for_pick.startswith("track"):
                    try:
                        if extractor_mod.is_track_token(str(mname)):
                            role_for_pick = "track"
                    except Exception:
                        pass
                if not role_for_pick.startswith("track"):
                    slot_map_probe = raw_slot_hints_by_mid.get(int(mid), {})
                    if isinstance(slot_map_probe, dict):
                        for raw_slot_value in slot_map_probe.values():
                            stem_low = _norm_low(Path(str(raw_slot_value or "")).stem)
                            if "tracks" in stem_low or "_trk" in stem_low or "chenille" in stem_low:
                                role_for_pick = "track"
                                break
                refs_mid = refs_by_material.get(int(mid), [])
                resolved_mid = [resolved_by_ref[ref] for ref in refs_mid if ref in resolved_by_ref]
                basis = resolved_mid if resolved_mid else resolved
                maps: Dict[str, Path] = {}
                used_group_key = ""
                want_track = role_for_pick.startswith("track")

                maps, used_group_key = _maps_from_material_name_or_slot_hints(int(mid), str(mname), want_track)
                if used_group_key and not want_track:
                    used_non_track_group_keys.add(used_group_key)

                if not refs_mid:
                    if not maps and role_for_pick.startswith("track") and primary_track_key:
                        maps, used_group_key = _maps_from_group_key(primary_track_key, True)
                    elif not maps and role_for_pick == "body" and primary_non_track_key:
                        maps, used_group_key = _maps_from_group_key(primary_non_track_key, False)
                        if used_group_key:
                            used_non_track_group_keys.add(used_group_key)
                    elif (
                        not maps
                        and role_for_pick == "other"
                        and _is_autogen_material_name(str(mname))
                    ):
                        for group_key in secondary_non_track_keys:
                            if group_key in used_non_track_group_keys:
                                continue
                            maps, used_group_key = _maps_from_group_key(group_key, False)
                            if maps:
                                used_non_track_group_keys.add(group_key)
                                break

                if not maps:
                    maps = strict_picker(basis, role_for_pick)
                if not maps and basis is not resolved:
                    maps = strict_picker(resolved, role_for_pick)
                if not isinstance(maps, dict):
                    maps = {}
                # Only use broad strict role fill when we do not have an exact
                # atlas group match from material name/slot hints. This prevents
                # cross-group contamination (e.g. foreign diffuse/alpha on HLM).
                if not used_group_key:
                    fill = strict_picker(basis, role_for_pick)
                    if isinstance(fill, dict):
                        for ch, src in fill.items():
                            maps.setdefault(ch, src)
                    if basis is not resolved:
                        fill_all = strict_picker(resolved, role_for_pick)
                        if isinstance(fill_all, dict):
                            for ch, src in fill_all.items():
                                maps.setdefault(ch, src)
                maps_by_name[mname] = _ensure_occlusion_from_same_stem(maps)
        else:
            if settings.auto_rename_textures:
                named_maps, named_files = extractor_mod.build_named_texture_aliases(
                    asset=asset,
                    model_dir=model_dir,
                    resolved=resolved,
                    chosen_maps=chosen_maps,
                )
            map_for_materials = named_maps or chosen_maps
            track_named_maps = extractor_mod.track_maps_from_named(named_files) if named_files else {"generic": {}, "left": {}, "right": {}}
            for mid in material_ids:
                mname = material_name_by_id.get(int(mid), f"Material_{int(mid):03d}")
                role = str(material_role_by_id.get(int(mid), "other"))
                refs_mid = refs_by_material.get(int(mid), [])
                resolved_mid = [resolved_by_ref[ref] for ref in refs_mid if ref in resolved_by_ref]
                maps = extractor_mod.pick_material_maps_from_textures(resolved_mid) if resolved_mid else {}
                is_track_role = role.startswith("track")
                if maps and named_files:
                    if hasattr(extractor_mod, "remap_maps_to_named_sources"):
                        maps = extractor_mod.remap_maps_to_named_sources(maps, named_files)
                    else:
                        def _path_key(path_value: Path) -> str:
                            try:
                                return str(path_value.resolve()).lower()
                            except Exception:
                                return str(path_value).lower()

                        src_to_named: Dict[str, Path] = {}
                        for item in named_files:
                            src_raw = str(item.get("source", "")).strip()
                            named_raw = str(item.get("named", "")).strip()
                            if not src_raw or not named_raw:
                                continue
                            src_to_named[_path_key(Path(src_raw))] = Path(named_raw)
                        remapped: Dict[str, Path] = {}
                        for ch, src in maps.items():
                            p = Path(src) if not isinstance(src, Path) else src
                            remapped[ch] = src_to_named.get(_path_key(p), p)
                        maps = remapped

                if not maps:
                    maps = {}
                    for ch, src in map_for_materials.items():
                        if is_track_role and ch == "diffuse":
                            continue
                        maps[ch] = src
                else:
                    for ch, src in map_for_materials.items():
                        if is_track_role and ch == "diffuse":
                            continue
                        maps.setdefault(ch, src)

                if role == "track_left":
                    maps.update(track_named_maps.get("left", {}))
                elif role == "track_right":
                    maps.update(track_named_maps.get("right", {}))
                elif role.startswith("track"):
                    maps.update(track_named_maps.get("generic", {}))

                if is_track_role:
                    override = _pick_track_diffuse_override(
                        extractor_mod=extractor_mod,
                        resolved_items=resolved_mid or resolved,
                        current_diffuse=maps.get("diffuse"),
                    )
                    if override is not None:
                        maps["diffuse"] = override
                    elif "diffuse" not in maps and "diffuse" in map_for_materials:
                        # Last-resort fallback to avoid untextured tracks.
                        maps["diffuse"] = map_for_materials["diffuse"]
                maps_by_name[mname] = _ensure_occlusion_from_same_stem(maps)

    channel_set: set[str] = set(chosen_maps.keys())
    for mm in maps_by_name.values():
        channel_set.update(mm.keys())
    report["channels"] = sorted(channel_set)

    report["resolved"] = [
        {
            "atlas_ref": item["atlas_ref"],
            "role": item["role"],
            "atlas_source": str(item.get("atlas_source", "")),
            "source_type": item["source_type"],
            "cache_hit": bool(item.get("cache_hit", False)),
            "source_tgv": item["source_tgv"],
            "source_png": item["source_png"],
            "out_png": str(item["out_png"]),
            "extras": {k: str(v) for k, v in item.get("extras", {}).items()},
            "deps_auto_installed": bool(item.get("deps_auto_installed", False)),
            "atlas_mode": str(item.get("atlas_mode", "")),
            "atlas_target_channel": str(item.get("atlas_target_channel", "")),
        }
        for item in resolved
    ]
    report["exact_hits"] = sum(1 for item in resolved if str(item.get("atlas_mode", "")).strip().lower() == "json_export")
    report["strict_misses"] = sum(
        1
        for err in errors
        if "atlas_json_strict" in str(err.get("error", "")).lower()
        or str(err.get("error", "")).lower().startswith("missing_source")
    )
    report["cache_hits"] = sum(1 for item in resolved if bool(item.get("cache_hit", False)))
    if any(str(item.get("atlas_source", "")).strip().lower() == "fallback" for item in resolved):
        report["atlas_source"] = "fallback"
    report["deps_auto_installed"] = any(bool(item.get("deps_auto_installed", False)) for item in resolved)
    report["named"] = list(named_files)
    _warno_log(
        settings,
        (
            f"texture resolve done refs={len(report.get('refs', []))} "
            f"resolved={len(report.get('resolved', []))} errors={len(report.get('errors', []))} "
            f"cache_hits={int(report.get('cache_hits', 0) or 0)} "
            f"channels={','.join(report.get('channels', [])) if report.get('channels') else '-'} "
            f"elapsed={time.monotonic()-stage_t0:.1f}s"
        ),
        stage="texture",
    )
    return maps_by_name, report


def _split_track_by_side(
    tris: Sequence[Tuple[int, int, int]],
    rotated_vertices: Sequence[Tuple[float, float, float]],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Split a single RD track mesh (which spans BOTH belts) into the left and right
    belt by geometry. RD tracks have no usable bone names, so the only robust signal
    is position: each belt is a separate connected component sitting on one side of
    the Y=0 mid-plane (left = -Y / Gauche, right = +Y / Droite). Group whole connected
    components by their centroid-Y sign; if connectivity yields a single welded blob
    (or all on one side), fall back to a per-triangle centroid-Y partition. Returns
    {"left": [...], "right": [...]}. Pure geometry, used only for the RD chenille
    branch — WARNO tracks already split per-material and never reach this."""
    out: Dict[str, List[Tuple[int, int, int]]] = {"left": [], "right": []}
    tri_list = [tuple(int(v) for v in t) for t in tris if len(t) == 3]
    vcount = len(rotated_vertices)
    if not tri_list or vcount <= 0:
        return out

    def _comp_cy(tri_indices: Sequence[int]) -> float:
        ys: List[float] = []
        for ti in tri_indices:
            for v in tri_list[ti]:
                if 0 <= v < vcount:
                    ys.append(float(rotated_vertices[v][1]))
        return (sum(ys) / len(ys)) if ys else 0.0

    vert_to_tris: Dict[int, List[int]] = defaultdict(list)
    for ti, tri in enumerate(tri_list):
        a, b, c = tri
        if min(a, b, c) < 0 or max(a, b, c) >= vcount:
            continue
        vert_to_tris[a].append(ti)
        vert_to_tris[b].append(ti)
        vert_to_tris[c].append(ti)

    visited: set[int] = set()
    components: List[List[int]] = []
    for start in range(len(tri_list)):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        comp: List[int] = []
        while queue:
            cur = queue.pop()
            comp.append(cur)
            for v in tri_list[cur]:
                for nei in vert_to_tris.get(v, []):
                    if nei not in visited:
                        visited.add(nei)
                        queue.append(nei)
        if comp:
            components.append(comp)

    if len(components) >= 2:
        for comp in components:
            side = "right" if _comp_cy(comp) >= 0.0 else "left"
            out[side].extend(tri_list[ti] for ti in comp)

    # Fallback: a single welded component (or everything landed on one side) ->
    # partition per-triangle by its own centroid Y.
    if not out["left"] or not out["right"]:
        out = {"left": [], "right": []}
        for tri in tri_list:
            ys = [float(rotated_vertices[v][1]) for v in tri if 0 <= v < vcount]
            cy = (sum(ys) / len(ys)) if ys else 0.0
            out["right" if cy >= 0.0 else "left"].append(tri)
    return out


def _chassis_face_components(nverts: int, faces) -> List[List[int]]:
    """Connected components of a triangle soup by SHARED VERTEX INDEX. Returns a list of
    face-index lists (one per disconnected mesh island). Union-find with path halving."""
    if nverts <= 0 or not faces:
        return []
    parent = list(range(nverts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[find(rc)] = ra
    comp: Dict[int, List[int]] = {}
    for fi, f in enumerate(faces):
        comp.setdefault(find(int(f[0])), []).append(fi)
    return list(comp.values())


def _subbucket_from_faces(b: dict, face_indices, name: str, *, is_hull: bool) -> dict:
    """Build a new bucket holding only `face_indices` of bucket `b`, re-indexing vertices.
    Copies b's metadata (group_bone_index, diagnostics, ...). Peeled (non-hull) subparts get
    a centroid origin via warno_rd_origin."""
    verts = b.get("vertices") or []
    uvs = b.get("uvs") or []
    srcs = b.get("source_refs") or []
    fmids = b.get("face_mids") or []
    faces = b.get("faces") or []
    remap: Dict[int, int] = {}
    nv: List[Any] = []
    nuv: List[Any] = []
    nsrc: List[Any] = []
    nf: List[Any] = []
    nfm: List[Any] = []
    for fi in face_indices:
        f = faces[int(fi)]
        tri: List[int] = []
        for vi in (int(f[0]), int(f[1]), int(f[2])):
            mapped = remap.get(vi)
            if mapped is None:
                mapped = len(nv)
                remap[vi] = mapped
                nv.append(verts[vi])
                nuv.append(uvs[vi] if vi < len(uvs) else (0.0, 0.0))
                nsrc.append(srcs[vi] if vi < len(srcs) else (0, 0))
            tri.append(mapped)
        nf.append((tri[0], tri[1], tri[2]))
        nfm.append(fmids[int(fi)] if int(fi) < len(fmids) else 0)
    nb = dict(b)
    nb["group_name"] = str(name)
    nb["vertices"] = nv
    nb["uvs"] = nuv
    nb["source_refs"] = nsrc
    nb["faces"] = nf
    nb["face_mids"] = nfm
    if isinstance(b.get("diagnostics"), dict):
        d = dict(b["diagnostics"])
        d["group_name"] = str(name)
        nb["diagnostics"] = d
    if not is_hull:
        nb["warno_rd_origin"] = True  # peeled detail: object builder gives it a centroid origin
        nb["warno_peeled_name"] = str(name)  # name the object by this, not the inherited Chassis bone
    return nb


def _peel_chassis_welded_subparts(
    buckets: List[dict],
    *,
    max_ratio: float = 0.06,
    min_tris: int = 8,
    max_subparts: int = 16,
) -> List[dict]:
    """RD/SD2 only: peel the obvious WELDED ANTENNAS/AERIALS/MASTS out of the Chassis/MainBody
    bucket into their own movable objects. These thin vertical rods share the chassis bone, so
    the name-based split can't separate them — only geometry can. We peel ONLY clearly
    antenna-like islands (tall + thin: height >= 2.5x its widest horizontal extent) so the rest
    of the hull (boxes, fenders, lights, the bolts) is NEVER shattered into junk objects (an
    earlier all-islands peel produced 64 fragments). Caller gates on model_is_rd -> WARNO
    untouched."""
    result: List[dict] = []
    for b in buckets:
        gname = _norm_low(str(b.get("group_name", "")))
        faces = b.get("faces") or []
        if gname not in {"chassis", "mainbody"} or len(faces) < max(36, min_tris * 3):
            result.append(b)
            continue
        comps = _chassis_face_components(len(b.get("vertices") or []), faces)
        if len(comps) <= 1:
            result.append(b)
            continue
        comps.sort(key=lambda c: -len(c))
        total = len(faces)
        verts = b.get("vertices") or []

        def _comp_extent(face_list):
            vs = set()
            for fi in face_list:
                f = faces[int(fi)]
                vs.update((int(f[0]), int(f[1]), int(f[2])))
            xs = [float(verts[v][0]) for v in vs]
            ys = [float(verts[v][1]) for v in vs]
            zs = [float(verts[v][2]) for v in vs]
            return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs),
                    (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)))

        keep: set = set(comps[0])  # largest component is always the hull
        # Unit scale = bbox diagonal of the WHOLE chassis bucket (not just the largest
        # component — a multi-shell hull like the S-tank fragments, which would understate the
        # scale and let slivers through). A REAL antenna/mast is tall relative to the unit;
        # small thin slivers (a row of smoke launchers, tiny fittings) are not. Without this,
        # stridsvagn peeled 15 sliver objects and ikv peeled tiny 0.1m bits.
        allx = [float(v[0]) for v in verts]
        ally = [float(v[1]) for v in verts]
        allz = [float(v[2]) for v in verts]
        hull_diag = max((
            (max(allx) - min(allx)) ** 2
            + (max(ally) - min(ally)) ** 2
            + (max(allz) - min(allz)) ** 2
        ) ** 0.5, 1.0e-3) if verts else 1.0
        subs: List[List[int]] = []
        for c in comps[1:]:
            ntri = len(c)
            if len(subs) >= max_subparts or ntri < int(min_tris) or ntri > int(total * max_ratio):
                keep.update(c)
                continue
            dx, dy, dz, _cen = _comp_extent(c)
            tall_thin = dz >= 2.5 * max(dx, dy, 1.0e-6)
            tall_abs = dz >= 0.10 * hull_diag
            if tall_thin and tall_abs:  # tall + thin + tall-relative-to-unit -> antenna/mast ONLY
                subs.append(c)
            else:
                keep.update(c)  # boxes / hatches / fenders / slivers stay welded (no junk objects)
        if not subs:
            result.append(b)
            continue
        # deterministic numbering by centroid
        subs.sort(key=lambda c: tuple(round(v, 3) for v in _comp_extent(c)[3]))
        result.append(_subbucket_from_faces(b, sorted(keep), str(b.get("group_name", "Chassis")), is_hull=True))
        for n_ant, c in enumerate(subs, start=1):
            result.append(_subbucket_from_faces(b, c, f"Antenna_{n_ant:02d}", is_hull=False))
    return result


def _collect_mesh_buckets(
    extractor_mod,
    model: dict[str, Any],
    rot: dict[str, float],
    material_name_by_id: Dict[int, str],
    material_role_by_id: Dict[int, str],
    bone_name_by_index: Dict[int, str],
    bone_parent_by_index: Dict[int, int],
    bone_positions: Dict[str, List[float]],
    split_main_parts: bool,
    dev_scale: float = WARNO_DEV_SCALE,
    material_maps_by_name: Dict[str, Any] = None,
    is_rd_asset: bool = False,
) -> list[dict[str, Any]]:
    buckets: Dict[str, dict[str, Any]] = {}
    order: List[str] = []
    default_root_bone_index = -1
    if bone_name_by_index:
        root_candidates = [
            int(i)
            for i in bone_name_by_index.keys()
            if int(bone_parent_by_index.get(int(i), -1)) < 0
        ]
        if root_candidates:
            default_root_bone_index = min(root_candidates)
        else:
            default_root_bone_index = min(int(i) for i in bone_name_by_index.keys())

    def _parse_wheel_group_label(raw_name: str) -> Tuple[str, str] | None:
        low = _norm_low(raw_name)
        if not low:
            return None
        m = re.match(r"^roue_(elev_)?([dg])([0-9]+)$", low, flags=re.IGNORECASE)
        if m:
            kind = "elev" if m.group(1) else "wheel"
            side = str(m.group(2)).upper()
            num = int(m.group(3))
            return (f"Roue_{side}{num}", kind)
        m = re.match(r"^roue_(elev_)?(droite|gauche)_?([0-9]+)$", low, flags=re.IGNORECASE)
        if m:
            kind = "elev" if m.group(1) else "wheel"
            side = "D" if str(m.group(2)).lower().startswith("droi") else "G"
            num = int(m.group(3))
            return (f"Roue_{side}{num}", kind)
        m = re.match(r"^roue_([0-9]+)$", low, flags=re.IGNORECASE)
        if m:
            num = int(m.group(1))
            return (f"Roue_{num:02d}", "wheel")
        return None

    def _position_for_bone_name(raw_name: str) -> Vector | None:
        low = _norm_low(raw_name)
        if not low:
            return None
        keys = [low]
        tok = _norm_token(low)
        if tok:
            keys.append(tok)
        for key in keys:
            hit = bone_positions.get(key)
            if isinstance(hit, (list, tuple)) and len(hit) >= 3:
                try:
                    return Vector((float(hit[0]), float(hit[1]), float(hit[2])))
                except Exception:
                    continue
        return None

    wheel_anchor_by_group: Dict[str, Dict[str, Any]] = {}
    for bidx, raw_name in bone_name_by_index.items():
        parsed = _parse_wheel_group_label(str(raw_name or ""))
        if parsed is None:
            continue
        group_name, kind = parsed
        pos = _position_for_bone_name(str(raw_name or ""))
        if pos is None:
            continue
        score = 2 if kind == "wheel" else 1
        cur = wheel_anchor_by_group.get(group_name)
        if cur is None or int(cur.get("_score", 0)) < score:
            wheel_anchor_by_group[group_name] = {
                "bone_index": int(bidx),
                "position": pos,
                "_score": score,
            }

    def _split_chassis_tris_by_wheel_anchors(
        tris: Sequence[Tuple[int, int, int]],
        rotated_vertices: Sequence[Tuple[float, float, float]],
    ) -> Tuple[List[Tuple[int, int, int]], Dict[str, List[Tuple[int, int, int]]]] | None:
        if len(wheel_anchor_by_group) < 2:
            return None
        tri_list = [tuple(map(int, t)) for t in tris if len(t) == 3]
        if len(tri_list) < 12:
            return None
        vcount = len(rotated_vertices)
        if vcount <= 0:
            return None

        vert_to_tris: Dict[int, List[int]] = defaultdict(list)
        for ti, tri in enumerate(tri_list):
            a, b, c = tri
            if min(a, b, c) < 0 or max(a, b, c) >= vcount:
                continue
            vert_to_tris[a].append(ti)
            vert_to_tris[b].append(ti)
            vert_to_tris[c].append(ti)

        visited: set[int] = set()
        components: List[List[int]] = []
        for start in range(len(tri_list)):
            if start in visited:
                continue
            queue = [start]
            visited.add(start)
            comp: List[int] = []
            while queue:
                cur = queue.pop()
                comp.append(cur)
                a, b, c = tri_list[cur]
                for v in (a, b, c):
                    for nei in vert_to_tris.get(v, []):
                        if nei in visited:
                            continue
                        visited.add(nei)
                        queue.append(nei)
            if comp:
                components.append(comp)

        if len(components) <= 1:
            return None

        anchors = [
            (str(gname), Vector(payload["position"]))
            for gname, payload in wheel_anchor_by_group.items()
            if isinstance(payload.get("position"), Vector)
        ]
        if len(anchors) < 2:
            return None

        min_anchor_gap = None
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                d = (anchors[i][1] - anchors[j][1]).length
                if d <= 1e-6:
                    continue
                if min_anchor_gap is None or d < min_anchor_gap:
                    min_anchor_gap = d
        if min_anchor_gap is None:
            min_anchor_gap = 1.0
        max_assign_dist = max(0.28, float(min_anchor_gap) * 0.48)

        assignments: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        keep_chassis: List[Tuple[int, int, int]] = []
        total_triangles = len(tri_list)
        max_component_triangles = max(120, int(total_triangles * 0.35))

        for comp in components:
            comp_tris = [tri_list[i] for i in comp]
            if not comp_tris:
                continue

            verts: set[int] = set()
            for a, b, c in comp_tris:
                if min(a, b, c) < 0 or max(a, b, c) >= vcount:
                    continue
                verts.update((a, b, c))
            if not verts:
                keep_chassis.extend(comp_tris)
                continue

            xs = [float(rotated_vertices[v][0]) for v in verts]
            ys = [float(rotated_vertices[v][1]) for v in verts]
            zs = [float(rotated_vertices[v][2]) for v in verts]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            dz = max(zs) - min(zs)
            dims = sorted([dx, dy, dz], reverse=True)
            centroid = Vector((
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(zs) / len(zs),
            ))

            # Conservative deterministic gate: keep only compact wheel-like components.
            wheel_like = True
            if len(comp_tris) > max_component_triangles:
                wheel_like = False
            if dims[0] < 0.12 or dims[1] < 0.12:
                wheel_like = False
            if dims[1] > 1e-6 and (dims[0] / dims[1]) > 2.5:
                wheel_like = False
            if dims[2] > dims[0] * 0.92:
                wheel_like = False

            nearest_group = ""
            nearest_dist = float("inf")
            for gname, gpos in anchors:
                dist = (centroid - gpos).length
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_group = gname

            if wheel_like and nearest_group and nearest_dist <= max_assign_dist:
                assignments[nearest_group].extend(comp_tris)
            else:
                keep_chassis.extend(comp_tris)

        if not assignments:
            return None
        return keep_chassis, dict(assignments)

    # Wargame RD / SD2 skinned units: the skeleton has no usable bone names, so a
    # per-bone split shatters the body into messy named/"Part_NNN" fragments, each
    # with a tiny sub-region UV (looks "squished"). Detect such a model by its
    # chenille (track) part and keep all non-track geometry as ONE MainBody with a
    # full UV; only the tracks are split out (below). WARNO never sets chenille.
    model_is_rd = any(
        bool(p.get("vertices", {}).get("chenille"))
        for p in model.get("parts", [])
    )
    # Wargame RD / SD2 used to fall back to a geometric body split because its
    # skeleton bone names decoded to control-byte garbage. The node-blob name
    # table is now decoded correctly (see SpkMeshExtractor._parse_concat_names_fixed),
    # so RD bodies carry real bone names (chassis / tourelle_01 / canon_01 /
    # roue_d* / roue_g* ...). When those names are present, route the RD BODY through
    # the exact same named-bone split WARNO uses, so RD imports gain real part names,
    # vertex groups + weights, bone-pivot origins and the empty-bone hierarchy — i.e.
    # parity with WARNO. The chenille (track) parts are still handled by the RD-only
    # branch above (geometric belt split + track UV), and the geometric rd_split_body_part
    # remains the fallback when no usable names were resolved.
    _model_has_named_bones = any(str(v).strip() for v in (bone_name_by_index or {}).values())

    # Map each track belt side -> its real skeleton bone index (chenille_droite_01 /
    # chenille_gauche_01). The RD belt split is geometric, but tagging each belt with
    # its own chenille bone lets the downstream builder name it Chenille_Droite_01 /
    # Chenille_Gauche_01, give it the belt's bone-pivot origin and parent it under the
    # chassis — i.e. WARNO parity. Without this the belts inherited the chassis bone
    # (default_root_bone_index) and, once real RD names were decoded, both collided
    # into "Chassis_2 / Chassis_3". Empty when names are unavailable -> belts keep the
    # old chassis-rooted fallback.
    _chenille_bone_idx_by_side: Dict[str, int] = {}
    for _bidx, _bnm in (bone_name_by_index or {}).items():
        _low = _norm_low(str(_bnm or ""))
        if _low.startswith("chenille_droite") and "right" not in _chenille_bone_idx_by_side:
            _chenille_bone_idx_by_side["right"] = int(_bidx)
        elif _low.startswith("chenille_gauche") and "left" not in _chenille_bone_idx_by_side:
            _chenille_bone_idx_by_side["left"] = int(_bidx)

    for part_i, part in enumerate(model.get("parts", [])):
        indices = part.get("indices", [])
        xyz = part.get("vertices", {}).get("xyz", [])
        uv = part.get("vertices", {}).get("uv", [])
        vertex_count = len(xyz) // 3
        if vertex_count <= 0:
            continue

        rotated: List[Tuple[float, float, float]] = []
        uv_data: List[Tuple[float, float]] = []
        for vi in range(vertex_count):
            x = float(xyz[vi * 3 + 0])
            y = float(xyz[vi * 3 + 1])
            z = float(xyz[vi * 3 + 2])
            rx, ry, rz = extractor_mod.apply_rotation(x, y, z, rot)
            rx, ry, rz = _warno_dev_scaled_xyz(float(rx), float(ry), float(rz), scale=dev_scale)
            rotated.append((rx, ry, rz))
            if vi * 2 + 1 < len(uv):
                uu = float(uv[vi * 2 + 0])
                vv = 1.0 - float(uv[vi * 2 + 1])
                uv_data.append((uu, vv))
            else:
                uv_data.append((0.0, 0.0))

        mid = int(part.get("material", -1))
        role = material_role_by_id.get(mid, "")
        mat_name = material_name_by_id.get(mid, f"Material_{mid:03d}")
        fallback = f"Part_{part_i:03d}"

        # Wargame RD / SD2 track geometry is flagged by the IRISZoom decoder
        # (the vertex type carries a "NormalAndChenilleIndex" field). Keep the whole
        # chenille part as ONE separate "Chenille" object instead of folding it into
        # MainBody (its big tiling-strip UVs were what made the body UV look cramped)
        # — and do NOT bone-split it (its per-link bones would shatter it into pieces).
        # WARNO geometry never sets this flag, so WARNO bucketing is unchanged.
        is_chenille = bool(part.get("vertices", {}).get("chenille"))

        # RD ball-chains ("цепури"): a NON-chenille part whose material is a cutout
        # CombinedDAS (the ball-chain belt shares the track's atlas). Peel it into its OWN
        # "Chains" object — like the track's "Chenille" object — so its UV + alpha cutout
        # never conflict with the merged MainBody. Identify it by the material maps
        # (alpha_cutout / uv_crop are RD-only keys, set only for CombinedDAS cutouts), not
        # by the unreliable material name. WARNO never sets these keys, so it's unaffected.
        _ch_maps = (material_maps_by_name or {}).get(mat_name) or {}
        is_chains = (not is_chenille) and bool(
            _ch_maps.get("alpha_cutout") or _ch_maps.get("uv_crop")
        )

        group_payloads: List[Dict[str, Any]] = []
        if is_chenille:
            # RD track is ONE mesh spanning BOTH belts. Split it into the left
            # (Chenille_Gauche, -Y) and right (Chenille_Droite, +Y) belt so each is a
            # separate object + per-side material, mirroring WARNO. Pure geometry (RD
            # has no usable bone names). Falls back to one "Chenille" if degenerate.
            _track_sides = _split_track_by_side(_all_tris(indices), rotated)
            group_payloads = []
            for _side_key, _side_label in (("left", "Chenille_Gauche"), ("right", "Chenille_Droite")):
                _side_tris = _track_sides.get(_side_key) or []
                if _side_tris:
                    _side_bone = int(_chenille_bone_idx_by_side.get(_side_key, default_root_bone_index))
                    group_payloads.append({
                        "group_name_raw": _side_label,
                        "group_name_sanitized": _side_label,
                        "group_bone_index": _side_bone,
                        "tris": list(_side_tris),
                    })
            if not group_payloads:
                group_payloads = [{
                    "group_name_raw": "Chenille",
                    "group_name_sanitized": "Chenille",
                    "group_bone_index": int(default_root_bone_index),
                    "tris": _all_tris(indices),
                }]
        elif is_chains:
            group_payloads = [{
                "group_name_raw": "Chains",
                "group_name_sanitized": "Chains",
                "group_bone_index": int(default_root_bone_index),
                "tris": _all_tris(indices),
            }]
        elif split_main_parts and (not model_is_rd or _model_has_named_bones):
            # Bone-split when the skeleton yielded usable (non-empty) bone names.
            # WARNO always does; Wargame-RD now does too (real names decoded from the
            # skeleton node blob). Tracks were already separated via the chenille
            # branch above. Falls back to a single MainBody if the split yields nothing.
            _has_named_bones = any(str(v).strip() for v in (bone_name_by_index or {}).values())
            if _has_named_bones and hasattr(extractor_mod, "split_faces_by_bone_deterministic"):
                try:
                    group_payloads = extractor_mod.split_faces_by_bone_deterministic(
                        part=part,
                        bone_name_by_index=bone_name_by_index,
                        bone_parent_by_index=bone_parent_by_index or {},
                        material_role=role,
                        material_name=mat_name,
                        part_index=int(part_i),
                    )
                except Exception:
                    group_payloads = []
            elif _has_named_bones and hasattr(extractor_mod, "split_faces_by_bone_top_level"):
                # Backward compatibility with older extractor builds.
                try:
                    group_payloads = extractor_mod.split_faces_by_bone_top_level(
                        part=part,
                        bone_name_by_index=bone_name_by_index,
                        bone_parent_by_index=bone_parent_by_index or {},
                        material_role=role,
                        material_name=mat_name,
                    )
                except Exception:
                    group_payloads = []
            if not group_payloads:
                group_payloads = [{
                    "group_name_raw": fallback,
                    "group_name_sanitized": fallback,
                    "group_bone_index": int(default_root_bone_index),
                    "tris": _all_tris(indices),
                }]

            # Deterministic wheel fallback for wheeled vehicles where wheel faces have no bone weights
            # and were merged into chassis/mainbody by split step.
            if wheel_anchor_by_group:
                has_wheel_group = any(
                    _norm_low(str(g.get("group_name_raw", "") or g.get("group_name_sanitized", ""))).startswith("roue_")
                    for g in group_payloads
                )
                if not has_wheel_group:
                    patched_groups: List[Dict[str, Any]] = []
                    for g in group_payloads:
                        gname_low = _norm_low(str(g.get("group_name_raw", "") or g.get("group_name_sanitized", "")))
                        gtris = g.get("tris", [])
                        if gname_low in {"chassis", "mainbody"} or gname_low.startswith("part_"):
                            split_result = _split_chassis_tris_by_wheel_anchors(gtris, rotated)
                            if split_result is not None:
                                chassis_tris, wheel_tris_by_group = split_result
                                if chassis_tris:
                                    g2 = dict(g)
                                    g2["tris"] = list(chassis_tris)
                                    patched_groups.append(g2)
                                for wheel_group_name in sorted(wheel_tris_by_group.keys()):
                                    tris_list = wheel_tris_by_group.get(wheel_group_name, [])
                                    if not tris_list:
                                        continue
                                    anchor = wheel_anchor_by_group.get(wheel_group_name, {})
                                    patched_groups.append(
                                        {
                                            "group_name_raw": wheel_group_name,
                                            "group_name_sanitized": wheel_group_name,
                                            "group_bone_index": int(anchor.get("bone_index", -1)),
                                            "tris": list(tris_list),
                                        }
                                    )
                                continue
                        patched_groups.append(g)
                    if patched_groups:
                        group_payloads = patched_groups
        elif model_is_rd and split_main_parts and hasattr(extractor_mod, "rd_split_body_part"):
            # RD body has no usable bone names, but the IRISZoom per-vertex dominant
            # bone index + geometry let us auto-split it into MainBody / Canon (gun) /
            # Tourelle (gun mount or turret) / Roue_D*/Roue_G* (roadwheels), mirroring
            # WARNO. rd_split_body_part never shatters (single MainBody on ambiguity);
            # each group carries an "origin" (cluster centroid) applied per-object later.
            try:
                group_payloads = extractor_mod.rd_split_body_part(
                    part.get("vertices", {}) or {}, rotated, _all_tris(indices),
                ) or []
            except Exception:
                group_payloads = []
            if not group_payloads:
                group_payloads = [{
                    "group_name_raw": "MainBody",
                    "group_name_sanitized": "MainBody",
                    "group_bone_index": int(default_root_bone_index),
                    "tris": _all_tris(indices),
                }]
        else:
            group_payloads = [{
                "group_name_raw": "MainBody",
                "group_name_sanitized": "MainBody",
                "group_bone_index": int(default_root_bone_index),
                "tris": _all_tris(indices),
            }]

        for group in group_payloads:
            group_name = str(group.get("group_name_raw", "") or group.get("group_name_sanitized", "") or fallback)
            tris = group.get("tris", [])
            group_bone_index = int(group.get("group_bone_index", -1))
            key = _norm_low(group_name) or "mainbody"
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {
                    "group_name": str(group_name),
                    "group_bone_index": int(group_bone_index),
                    "vertices": [],
                    "uvs": [],
                    "source_refs": [],
                    "faces": [],
                    "face_mids": [],
                    "map": {},
                    "_diag_part_indices": set(),
                    "_diag_material_names": {},
                    "_diag_material_roles": set(),
                    "_diag_raw_node_triangles": {},
                    "_diag_group_name_sources": {},
                }
                buckets[key] = bucket
                order.append(key)
            elif int(bucket.get("group_bone_index", -1)) < 0 and int(group_bone_index) >= 0:
                bucket["group_bone_index"] = int(group_bone_index)
            # RD body-split groups carry an "origin" -> flag the bucket so the object
            # builder sets its origin to the part's centroid (RD has no bone pivots).
            if "origin" in group:
                bucket["warno_rd_origin"] = True
            # RD wheels: their skeleton roue_* bone pivots routinely resolve to the world
            # origin (the wheel mesh carries its own world position), which would leave the
            # wheel pivot at (0,0,0). Flag them so the object builder gives each a
            # geometric-centroid origin (the wheel centre). The armature pass runs AFTER but
            # skips objects carrying this centroid (so it isn't clobbered back to 0,0,0).
            # Body/turret/gun keep their resolved bone-pivot origin; tracks (chenille) keep
            # the native track-deform setup untouched. RD-only; WARNO never reaches this.
            if model_is_rd:
                _gn_low_origin = _norm_low(str(group_name or ""))
                if _gn_low_origin.startswith("roue"):
                    bucket["warno_rd_origin"] = True

            group_diag = group.get("diagnostics", {})
            if not isinstance(group_diag, dict):
                group_diag = {}
            try:
                bucket["_diag_part_indices"].add(int(group_diag.get("part_index", part_i)))
            except Exception:
                bucket["_diag_part_indices"].add(int(part_i))
            mat_diag_name = str(group_diag.get("material_name", "") or mat_name)
            if mat_diag_name:
                bucket["_diag_material_names"][mat_diag_name] = True
            role_diag = str(group_diag.get("material_role", "") or role)
            if role_diag:
                bucket["_diag_material_roles"].add(role_diag)
            group_source = str(group_diag.get("group_name_source", "") or "classifier")
            group_source_counts = bucket["_diag_group_name_sources"]
            group_source_counts[group_source] = int(group_source_counts.get(group_source, 0)) + max(1, int(len(tris)))
            raw_diag = group_diag.get("raw_node_triangles", {})
            if isinstance(raw_diag, dict):
                raw_counts = bucket["_diag_raw_node_triangles"]
                for raw_name, tri_count in raw_diag.items():
                    raw_key = str(raw_name or "").strip()
                    if not raw_key:
                        continue
                    try:
                        raw_counts[raw_key] = int(raw_counts.get(raw_key, 0)) + int(tri_count)
                    except Exception:
                        continue

            vmap: Dict[Tuple[int, int], int] = bucket["map"]
            for a, b, c in tris:
                tri = (int(a), int(b), int(c))
                if min(tri) < 0 or max(tri) >= vertex_count:
                    continue
                face_idx: List[int] = []
                for vi in tri:
                    map_key = (part_i, vi)
                    mapped = vmap.get(map_key)
                    if mapped is None:
                        mapped = len(bucket["vertices"])
                        vmap[map_key] = mapped
                        bucket["vertices"].append(rotated[vi])
                        bucket["uvs"].append(uv_data[vi])
                        bucket["source_refs"].append((int(part_i), int(vi)))
                    face_idx.append(mapped)
                if len(face_idx) == 3:
                    bucket["faces"].append((face_idx[0], face_idx[1], face_idx[2]))
                    bucket["face_mids"].append(mid)

    out: List[dict[str, Any]] = []
    for key in order:
        b = buckets.get(key)
        if b is None:
            continue
        if not b["faces"]:
            continue
        cleanup_diag: Dict[str, Any] = {}
        if hasattr(extractor_mod, "cleanup_bucket_geometry"):
            try:
                cleanup_payload = extractor_mod.cleanup_bucket_geometry(
                    vertices=b.get("vertices", []) or [],
                    uvs=b.get("uvs", []) or [],
                    faces=b.get("faces", []) or [],
                    face_mids=b.get("face_mids", []) or [],
                    source_refs=b.get("source_refs", []) or [],
                    group_name=str(b.get("group_name", "") or ""),
                    # RD/SD2: skip the tri->quad merge. For RD geometry the merge emits
                    # quads that render as HOLES (the real cause of the leopard hull holes,
                    # proven by ON/OFF frozen-camera renders). WARNO keeps the merge ->
                    # byte-identical. RD keeps all triangles (watertight, verified).
                    merge_quads=(not bool(model_is_rd)),
                )
            except Exception:
                cleanup_payload = {}
            if isinstance(cleanup_payload, dict) and list(cleanup_payload.get("faces", []) or []):
                for field in ("vertices", "uvs", "source_refs", "faces", "face_mids"):
                    if field in cleanup_payload:
                        b[field] = list(cleanup_payload.get(field, []) or [])
                cleanup_diag = dict(cleanup_payload.get("diagnostics", {}) or {})
        raw_node_triangles = dict(b.get("_diag_raw_node_triangles", {}) or {})
        dominant_raw_name = min(
            raw_node_triangles.keys(),
            key=lambda raw_name: (-int(raw_node_triangles.get(raw_name, 0)), str(raw_name).lower()),
        ) if raw_node_triangles else ""
        dominant_raw_triangles = int(raw_node_triangles.get(dominant_raw_name, 0))
        total_tris = sum(int(v) for v in raw_node_triangles.values())
        if total_tris <= 0:
            total_tris = int(len(b.get("faces", []) or []))
        material_roles = sorted(str(v) for v in list(b.get("_diag_material_roles", set()) or []))
        is_track_bucket = any(str(role).lower().startswith("track") for role in material_roles)
        foreign_triangles = max(0, int(total_tris) - int(dominant_raw_triangles))
        contamination_verdict = "contaminated" if foreign_triangles >= max(24, int(total_tris * 0.08)) else "stable"
        if is_track_bucket and not raw_node_triangles:
            foreign_triangles = 0
            contamination_verdict = "stable"
        group_source_counts = dict(b.get("_diag_group_name_sources", {}) or {})
        group_name_source = min(
            group_source_counts.keys(),
            key=lambda source: (-int(group_source_counts.get(source, 0)), str(source).lower()),
        ) if group_source_counts else "classifier"
        diagnostics = {
            "group_name": str(b.get("group_name", "") or ""),
            "group_bone_index": int(b.get("group_bone_index", -1)),
            "has_geometry": True,
            "part_indices": sorted(int(v) for v in list(b.get("_diag_part_indices", set()) or [])),
            "material_names": sorted(str(v) for v in list((b.get("_diag_material_names", {}) or {}).keys())),
            "material_roles": material_roles,
            "group_name_source": str(group_name_source),
            "dominant_raw_node": str(dominant_raw_name),
            "dominant_raw_triangles": int(dominant_raw_triangles),
            "foreign_raw_triangles": int(foreign_triangles),
            "contamination_verdict": str(contamination_verdict),
            "raw_node_triangles": {
                str(name): int(count)
                for name, count in sorted(
                    raw_node_triangles.items(),
                    key=lambda item: (-int(item[1]), str(item[0]).lower()),
                )
            },
        }
        diagnostics.update(cleanup_diag)
        b["diagnostics"] = diagnostics
        for temp_key in (
            "_diag_part_indices",
            "_diag_material_names",
            "_diag_material_roles",
            "_diag_raw_node_triangles",
            "_diag_group_name_sources",
            "map",
        ):
            b.pop(temp_key, None)
        out.append(b)
    # RD/SD2 only: un-weld antennas/hatches/fittings that share the chassis bone (the
    # name-based split can't separate them; geometry islands can). Gated on the RD/SD2
    # game preset (NOT chenille) so it also runs on TRACKLESS RD units (boats / wheeled,
    # e.g. fremantle's masts) whose model_is_rd would be False; WARNO is byte-for-byte
    # unchanged. Falls back to `out` on any error.
    if model_is_rd or is_rd_asset:
        try:
            out = _peel_chassis_welded_subparts(out)
        except Exception:
            pass
    return out


def _load_image(path: Path):
    if not path.exists() or not path.is_file():
        return None
    try:
        return bpy.data.images.load(str(path), check_existing=True)
    except Exception:
        return None


def _texture_node(nodes, path: Path, x: int, y: int, non_color: bool = False):
    img = _load_image(path)
    if img is None:
        return None
    node = nodes.new("ShaderNodeTexImage")
    node.location = (x, y)
    node.image = img
    if non_color:
        try:
            node.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
    return node


def _normal_invert_mode_text(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in {"none", "invert_green", "invert_rgb"}:
        return "none"
    return mode


def _effective_normal_invert_mode(settings: WARNOImporterSettings) -> str:
    # Atlas JSON conversion already writes inverted normal textures on disk.
    if bool(getattr(settings, "use_atlas_json_mapping", False)):
        return "none"
    return _normal_invert_mode_text(str(getattr(settings, "normal_invert_mode", "none") or "none"))


def _wire_normal_color_to_normal_map(nodes, links, color_socket, normal_map_node, invert_mode: str) -> None:
    mode = _normal_invert_mode_text(invert_mode)
    try:
        while normal_map_node.inputs["Color"].is_linked:
            links.remove(normal_map_node.inputs["Color"].links[0])
    except Exception:
        pass

    if mode == "invert_rgb":
        inv = nodes.new("ShaderNodeInvert")
        inv.location = (normal_map_node.location.x - 190, normal_map_node.location.y)
        links.new(color_socket, inv.inputs["Color"])
        links.new(inv.outputs["Color"], normal_map_node.inputs["Color"])
        return

    if mode == "invert_green":
        sep = nodes.new("ShaderNodeSeparateRGB")
        sep.location = (normal_map_node.location.x - 300, normal_map_node.location.y)
        inv_g = nodes.new("ShaderNodeInvert")
        inv_g.location = (normal_map_node.location.x - 80, normal_map_node.location.y - 130)
        comb = nodes.new("ShaderNodeCombineRGB")
        comb.location = (normal_map_node.location.x + 120, normal_map_node.location.y)
        links.new(color_socket, sep.inputs["Image"])
        links.new(sep.outputs["R"], comb.inputs["R"])
        links.new(sep.outputs["B"], comb.inputs["B"])
        links.new(sep.outputs["G"], inv_g.inputs["Color"])
        links.new(inv_g.outputs["Color"], comb.inputs["G"])
        links.new(comb.outputs["Image"], normal_map_node.inputs["Color"])
        return

    links.new(color_socket, normal_map_node.inputs["Color"])


def _apply_rd_material_nodes(
    mat: bpy.types.Material,
    maps: dict[str, Path],
    role: str,
) -> None:
    """Build the node graph for a Wargame Red Dragon / Steel Division 2 material.

    RD-era units do not ship separate normal / roughness / metallic textures the
    way WARNO does — the shading data is packed into one texture's alpha channel.
    Two cases, decided by alpha content in `_resolve_rd_textures`:
      * CombinedDAS-style = a transparency CUTOUT mask (chains, track-teeth gaps):
        alpha -> BSDF Alpha, blend = CLIP so the black gaps become see-through.
      * CombinedDS-style = a specular mask: alpha (inverted) -> Roughness.
    RGB always -> Base Color. DiffuseTextureNoAlpha has a flat alpha -> matte
    default roughness. Fully isolated from the WARNO graph (WARNO unchanged)."""
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (560, 40)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (240, 40)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    base = maps.get("diffuse")
    spec_alpha = maps.get("spec_alpha")
    alpha_cutout = maps.get("alpha_cutout")
    alpha_tex = maps.get("alpha")            # separate WARNO-style binary alpha map
    track_repeat = bool(maps.get("track_repeat"))
    force_opaque = bool(maps.get("force_opaque")) or track_repeat
    normal = maps.get("normal")

    base_node = None
    if base is not None:
        base_node = _texture_node(nodes, base, -560, 200, non_color=False)
        if base_node is not None:
            if track_repeat:
                # RD track: the cropped scaly-armour tread tiles along the tube-unwrapped
                # belt (UV set at object build). REPEAT so it tiles seamlessly.
                try:
                    base_node.extension = "REPEAT"
                except Exception:
                    pass
            links.new(base_node.outputs["Color"], bsdf.inputs["Base Color"])

    if alpha_tex is not None:
        # WARNO-style SEPARATE alpha map: a clean binary (0/255) cutout texture wired
        # straight into BSDF Alpha — no GreaterThan/threshold node needed in the shader.
        # Used by the split track/chains textures (_split_track_chains_textures).
        a_node = _texture_node(nodes, alpha_tex, -560, 20, non_color=True)
        if a_node is not None:
            links.new(a_node.outputs["Color"], bsdf.inputs["Alpha"])
        # The track UV seam fix (get_model_geometry -> _floodfill_track_uv_seam) leaves the
        # tread V continuous, running slightly outside [0,1]; REPEAT makes those tails tile
        # seamlessly (no edge clamp). Set it explicitly on BOTH the diffuse and alpha nodes.
        for _n in (base_node, a_node):
            if _n is not None:
                try:
                    _n.extension = "REPEAT"
                except Exception:
                    pass
        for attr, val in (("blend_method", "CLIP"), ("shadow_method", "CLIP")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
        if hasattr(mat, "alpha_threshold"):
            try:
                mat.alpha_threshold = 0.5
            except Exception:
                pass
        if hasattr(mat, "surface_render_method"):
            try:
                mat.surface_render_method = "DITHERED"
            except Exception:
                pass
        try:
            bsdf.inputs["Roughness"].default_value = 0.6
        except Exception:
            pass
    elif force_opaque:
        # Solid (no alpha link), keeps RGB -> Base Color from above.
        for attr, val in (("blend_method", "OPAQUE"), ("shadow_method", "OPAQUE")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
        try:
            bsdf.inputs["Roughness"].default_value = 0.6
        except Exception:
            pass
    elif alpha_cutout is not None:
        # Transparency cutout: the texture's alpha is a binary-ish mask (chain
        # links / track teeth opaque, gaps == 0). Drive BSDF Alpha and CLIP so the
        # gaps vanish instead of rendering as black. The opaque alpha peaks ~0.76
        # (not 1.0), so a low clip threshold keeps the shapes solid.
        cut_node = base_node if (base is not None and str(alpha_cutout) == str(base)) \
            else _texture_node(nodes, alpha_cutout, -560, 200, non_color=False)
        if cut_node is not None and "Alpha" in cut_node.outputs:
            # Wire the texture alpha STRAIGHT into BSDF Alpha (no GreaterThan/threshold
            # node) — same clean path as the WARNO-style separate alpha map. The crisp
            # binary cutout comes from CLIP + alpha_threshold below, which works under
            # both legacy blend_method (<4.2) and EEVEE-Next surface_render_method (5.1).
            links.new(cut_node.outputs["Alpha"], bsdf.inputs["Alpha"])
        for attr, val in (("blend_method", "CLIP"), ("shadow_method", "CLIP")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
        if hasattr(mat, "alpha_threshold"):
            try:
                mat.alpha_threshold = 0.3
            except Exception:
                pass
        if hasattr(mat, "surface_render_method"):
            try:
                mat.surface_render_method = "DITHERED"
            except Exception:
                pass
        try:
            bsdf.inputs["Roughness"].default_value = 0.6
        except Exception:
            pass
    elif spec_alpha is not None:
        # The CombinedDS alpha is a packed spec mask, NOT opacity. Per user preference
        # we do NOT wire it at all (no Invert -> Roughness node): the body is just the
        # opaque diffuse texture with a matte default roughness — clean node graph.
        if hasattr(mat, "blend_method"):
            mat.blend_method = "OPAQUE"
        try:
            bsdf.inputs["Roughness"].default_value = 0.7
        except Exception:
            pass
    elif base_node is not None:
        try:
            bsdf.inputs["Roughness"].default_value = 0.7   # matte painted-metal default
        except Exception:
            pass

    if normal is not None:
        nrm_node = _texture_node(nodes, normal, -880, -360, non_color=True)
        if nrm_node is not None:
            nm = nodes.new("ShaderNodeNormalMap")
            nm.location = (-560, -360)
            links.new(nrm_node.outputs["Color"], nm.inputs["Color"])
            links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])


def _apply_material_nodes(
    mat: bpy.types.Material,
    maps: dict[str, Path],
    role: str,
    ao_multiply_diffuse: bool = True,
    normal_invert_mode: str = "none",
) -> None:
    # Wargame RD / SD2 materials carry an "rd_material" marker and use a separate
    # node graph (spec-in-alpha). WARNO maps never set it, so WARNO is unaffected.
    if maps.get("rd_material"):
        _apply_rd_material_nodes(mat, maps, role)
        return
    alpha = maps.get("alpha")
    alpha_track_like = False
    if alpha is not None:
        low_alpha = _norm_low(alpha.stem)
        alpha_track_like = bool(re.search(r"(?:^|_)(trk|track|chenille)(?:_|$)", low_alpha))
    allow_alpha_link = alpha is not None and (str(role).startswith("track") or not alpha_track_like)

    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "HASHED" if allow_alpha_link else "OPAQUE"
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = "HASHED" if allow_alpha_link else "OPAQUE"
        except Exception:
            pass

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (520, 40)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (240, 40)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    diffuse = maps.get("diffuse")
    normal = maps.get("normal")
    roughness = maps.get("roughness")
    metallic = maps.get("metallic")
    occlusion = maps.get("occlusion")

    diffuse_node = None
    if diffuse is not None:
        diffuse_node = _texture_node(nodes, diffuse, -620, 190, non_color=False)

    ao_node = _texture_node(nodes, occlusion, -620, -20, non_color=True) if occlusion is not None else None

    if diffuse_node is not None and ao_multiply_diffuse:
        if ao_node is not None:
            mul = nodes.new("ShaderNodeMixRGB")
            mul.location = (-260, 120)
            mul.blend_type = "MULTIPLY"
            mul.inputs["Fac"].default_value = 1.0
            links.new(diffuse_node.outputs["Color"], mul.inputs["Color1"])
            links.new(ao_node.outputs["Color"], mul.inputs["Color2"])
            links.new(mul.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            links.new(diffuse_node.outputs["Color"], bsdf.inputs["Base Color"])
    elif diffuse_node is not None:
        links.new(diffuse_node.outputs["Color"], bsdf.inputs["Base Color"])

    if alpha is not None:
        alpha_node = _texture_node(nodes, alpha, -620, -200, non_color=True)
        if alpha_node is not None and allow_alpha_link:
            links.new(alpha_node.outputs["Color"], bsdf.inputs["Alpha"])
    elif diffuse_node is not None and "Alpha" in diffuse_node.outputs and str(role).startswith("track"):
        links.new(diffuse_node.outputs["Alpha"], bsdf.inputs["Alpha"])

    if normal is not None:
        normal_node = _texture_node(nodes, normal, -940, -340, non_color=True)
        if normal_node is not None:
            nm = nodes.new("ShaderNodeNormalMap")
            nm.location = (-670, -340)
            _wire_normal_color_to_normal_map(
                nodes=nodes,
                links=links,
                color_socket=normal_node.outputs["Color"],
                normal_map_node=nm,
                invert_mode=normal_invert_mode,
            )
            links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    if roughness is not None:
        rough_node = _texture_node(nodes, roughness, -620, -470, non_color=True)
        if rough_node is not None:
            links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])

    if metallic is not None:
        metal_node = _texture_node(nodes, metallic, -620, -620, non_color=True)
        if metal_node is not None:
            links.new(metal_node.outputs["Color"], bsdf.inputs["Metallic"])

    try:
        if occlusion is not None:
            occ_name = str(Path(occlusion).name)
            if occ_name.lower().endswith("_o.png"):
                mat["EugAO"] = occ_name
            elif "EugAO" in mat:
                del mat["EugAO"]
        elif "EugAO" in mat:
            del mat["EugAO"]
    except Exception:
        pass

def _ensure_collection(scene: bpy.types.Scene, base_name: str) -> bpy.types.Collection:
    name = base_name
    idx = 2
    while bpy.data.collections.get(name) is not None:
        name = f"{base_name}_{idx}"
        idx += 1
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


def _collection_from_setting(scene: bpy.types.Scene, col_name: str) -> bpy.types.Collection | None:
    name = str(col_name or "").strip()
    if not name:
        return None
    if name == WARNO_SCENE_COLLECTION_SENTINEL:
        return scene.collection
    col = bpy.data.collections.get(name)
    if col is not None:
        return col
    if scene.collection is not None and str(scene.collection.name) == name:
        return scene.collection
    return None


def _apply_dev_collection_layout(
    scene: bpy.types.Scene,
    new_objects: Sequence[bpy.types.Object],
) -> str:
    if scene is None:
        return WARNO_SCENE_COLLECTION_SENTINEL
    unique_objects: List[bpy.types.Object] = []
    seen_ptrs: set[int] = set()
    for obj in new_objects:
        if obj is None:
            continue
        try:
            ptr = int(obj.as_pointer())
        except Exception:
            continue
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        unique_objects.append(obj)
    if not unique_objects:
        return WARNO_SCENE_COLLECTION_SENTINEL
    if len(unique_objects) > int(WARNO_DEV_CHILD_COLLECTION_THRESHOLD):
        return WARNO_SCENE_COLLECTION_SENTINEL
    col = bpy.data.collections.get("Collection")
    if col is None:
        col = bpy.data.collections.new("Collection")
        scene.collection.children.link(col)
    for obj in unique_objects:
        if col not in obj.users_collection:
            try:
                col.objects.link(obj)
            except Exception:
                pass
        for user_col in list(obj.users_collection):
            if user_col == col:
                continue
            try:
                user_col.objects.unlink(obj)
            except Exception:
                continue
    return str(col.name)


def _merge_by_distance(objects: Sequence[bpy.types.Object], distance: float) -> None:
    if distance <= 0.0:
        return
    for obj in objects:
        if obj.type != "MESH":
            continue
        try:
            # NEVER merge-by-distance a track (chenille) / chains belt: its tread tiles via a
            # per-link UV sawtooth whose tile boundaries are DUPLICATE verts at the same
            # position with different UVs; merging collapses them -> the per-link tiling
            # smears into a stretched, seamed tread (the user-reported bug). Check the flag
            # AND the object name, because the REFERENCE-mode cleanup deletes warno_is_track_mesh
            # before this merge runs (so the flag alone is unreliable here).
            _nlow = _norm_low(getattr(obj, "name", ""))
            if (bool(obj.get("warno_is_track_mesh", False))
                    or bool(obj.get("warno_is_chains_mesh", False))
                    or _nlow.startswith("chenille")
                    or _nlow.startswith("chains")
                    or "track" in _nlow):
                continue
        except Exception:
            pass
        try:
            non_default_groups = [
                str(vg.name)
                for vg in getattr(obj, "vertex_groups", [])
                if _norm_low(getattr(vg, "name", "")) != "group"
            ]
        except Exception:
            non_default_groups = []
        if len(non_default_groups) > 1:
            continue
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        if bm.verts:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()


def _fill_rd_mesh_holes(obj: bpy.types.Object, *, weld_dist: float = 0.0006, max_sides: int = 24) -> int:
    """RD/SD2 unit meshes ship as OPEN shells — the hull has many open boundary edges (areas
    the game never shows: under hatches/grilles, the open bottom, inner walls). In Blender they
    read as holes. Weld near-coincident boundary verts into closed loops, then fill holes up to
    `max_sides` edges (so the large intended bottom opening is left open). RD-only; skips
    track/chains belts (their open edges are intentional UV-seam duplicates). Returns faces added."""
    if obj.type != "MESH" or obj.data is None:
        return 0
    nlow = _norm_low(getattr(obj, "name", ""))
    if (bool(obj.get("warno_is_track_mesh", False)) or bool(obj.get("warno_is_chains_mesh", False))
            or nlow.startswith("chenille") or nlow.startswith("chains") or "track" in nlow):
        return 0
    mesh = obj.data
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        uvl = bm.loops.layers.uv.active
        f0 = len(bm.faces)
        if weld_dist > 0 and bm.verts:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(weld_dist))
        boundary = [e for e in bm.edges if e.is_boundary]
        if not boundary:
            bm.free()
            return 0
        # representative UV per boundary vert (from its existing loops) so the new fill faces
        # get sensible UVs instead of (0,0) -> the v1.7.4 smeared-texture bug.
        vert_uv = {}
        if uvl is not None:
            for f in bm.faces:
                for lp in f.loops:
                    if lp.vert not in vert_uv:
                        vert_uv[lp.vert] = lp[uvl].uv.copy()
        res = bmesh.ops.holes_fill(bm, edges=boundary, sides=int(max_sides))
        new_faces = [f for f in res.get("faces", []) or [] if f.is_valid]
        if uvl is not None:
            for f in new_faces:
                for lp in f.loops:
                    uv = vert_uv.get(lp.vert)
                    if uv is not None:
                        lp[uvl].uv = uv
        if new_faces:
            bmesh.ops.recalc_face_normals(bm, faces=new_faces)
        added = len(bm.faces) - f0
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        return max(0, int(added))
    except Exception:
        return 0


def _assign_uniform_vertex_group(obj: bpy.types.Object, name: str, weight: float = 1.0) -> bool:
    if obj.type != "MESH" or obj.data is None or not getattr(obj.data, "vertices", None):
        return False
    if obj.vertex_groups.get(str(name)) is not None:
        return False
    vg = obj.vertex_groups.new(name=str(name))
    indices = [int(v.index) for v in obj.data.vertices]
    if not indices:
        return False
    vg.add(indices, float(weight), "REPLACE")
    return True


def _wrap_track_uv_layer(mesh: bpy.types.Mesh, uv_layer: bpy.types.MeshUVLoopLayer) -> bool:
    try:
        uv_data = uv_layer.data
    except Exception:
        return False
    if uv_data is None:
        return False

    max_abs = 0.0
    for loop in uv_data:
        max_abs = max(max_abs, abs(float(loop.uv.x)), abs(float(loop.uv.y)))
    if max_abs <= 4.0:
        return False

    # WARNO cooker rejects UVs outside [-4, 4]. Track strips can carry large
    # integer tile offsets, while repeat sampling only depends on the fractional part.
    # Remove integer offsets but preserve the local tile shape.
    for loop in uv_data:
        u = float(loop.uv.x)
        v = float(loop.uv.y)
        loop.uv = (u - math.floor(u), v - math.floor(v))
    return True


def _track_group_sort_key(name: str) -> Tuple[int, str]:
    text = str(name or "").strip()
    m = re.search(r"([0-9]+)$", text)
    if m:
        try:
            return int(m.group(1)), text.lower()
        except Exception:
            pass
    return 10**9, text.lower()


def _collect_track_helper_candidates(
    side_token: str,
    bone_name_by_index: Dict[int, str],
    bone_positions: Dict[str, Any] | None,
) -> List[Tuple[str, Vector]]:
    side = str(side_token or "").strip().lower()
    if side not in {"d", "g"}:
        return []
    bone_positions = bone_positions or {}
    side_candidates: List[Tuple[str, Vector]] = []
    wheel_rx = re.compile(rf"^roue_elev_{side}[0-9]+$", re.IGNORECASE)
    for _bidx, raw_name in bone_name_by_index.items():
        raw = str(raw_name or "").strip()
        low = _norm_low(raw)
        if not wheel_rx.fullmatch(low):
            continue
        pos = None
        m = re.match(r"^roue_elev_([dg])([0-9]+)$", low, re.IGNORECASE)
        if m:
            wheel_side = str(m.group(1)).lower()
            wheel_num = int(m.group(2))
            for wheel_ref in (
                f"roue_{wheel_side}{wheel_num}",
                f"Roue_{wheel_side.upper()}{wheel_num}",
            ):
                pos = _pick_position_from_payload(wheel_ref, bone_positions)
                if pos is not None:
                    break
        if pos is None:
            pos = _pick_position_from_payload(raw, bone_positions)
        if pos is None:
            continue
        side_candidates.append((_pretty_warno_node_name(raw), pos.copy()))
    side_candidates.sort(key=lambda item: _track_group_sort_key(item[0]))
    return side_candidates


def _mesh_vertex_neighbors(mesh: bpy.types.Mesh) -> List[set[int]]:
    neighbors: List[set[int]] = [set() for _ in range(len(getattr(mesh, "vertices", []) or []))]
    try:
        for edge in getattr(mesh, "edges", []) or []:
            verts = list(getattr(edge, "vertices", []) or [])
            if len(verts) < 2:
                continue
            a = int(verts[0])
            b = int(verts[1])
            if 0 <= a < len(neighbors) and 0 <= b < len(neighbors):
                neighbors[a].add(b)
                neighbors[b].add(a)
    except Exception:
        pass
    if any(neighbors):
        return neighbors
    try:
        for poly in getattr(mesh, "polygons", []) or []:
            verts = [int(v) for v in list(getattr(poly, "vertices", []) or [])]
            for i, a in enumerate(verts):
                b = verts[(i + 1) % len(verts)]
                if 0 <= a < len(neighbors) and 0 <= b < len(neighbors):
                    neighbors[a].add(b)
                    neighbors[b].add(a)
    except Exception:
        pass
    return neighbors


def _track_group_weight_totals(
    obj: bpy.types.Object,
    track_group_names: Sequence[str],
) -> Dict[int, float]:
    mesh = getattr(obj, "data", None)
    if obj.type != "MESH" or mesh is None:
        return {}
    valid_indices: set[int] = set()
    for group_name in track_group_names:
        vg = obj.vertex_groups.get(str(group_name))
        if vg is None:
            continue
        valid_indices.add(int(vg.index))
    totals: Dict[int, float] = {}
    for vert in getattr(mesh, "vertices", []) or []:
        total = 0.0
        for slot in getattr(vert, "groups", []) or []:
            if int(slot.group) not in valid_indices:
                continue
            weight = float(slot.weight)
            if weight > 1.0e-8:
                total += weight
        totals[int(vert.index)] = total
    return totals


def _assign_track_group_patch(
    obj: bpy.types.Object,
    group_name: str,
    vert_indices: Sequence[int],
    track_group_names: Sequence[str],
    *,
    exclusive: bool = True,
) -> int:
    if obj.type != "MESH" or obj.data is None:
        return 0
    unique_indices = sorted({int(idx) for idx in vert_indices if int(idx) >= 0})
    if not unique_indices:
        return 0
    vg = obj.vertex_groups.get(str(group_name))
    if vg is None:
        vg = obj.vertex_groups.new(name=str(group_name))
    other_groups = []
    if exclusive:
        other_groups = [
            other_vg
            for other_name in track_group_names
            if _norm_low(str(other_name)) != _norm_low(str(group_name))
            for other_vg in [obj.vertex_groups.get(str(other_name))]
            if other_vg is not None
        ]
    for vert_index in unique_indices:
        for other_vg in other_groups:
            try:
                other_vg.remove([int(vert_index)])
            except Exception:
                continue
        vg.add([int(vert_index)], 1.0, "REPLACE")
    return len(unique_indices)


def _overlay_track_group_weights(
    obj: bpy.types.Object,
    group_name: str,
    vert_weights: Dict[int, float],
) -> int:
    if obj.type != "MESH" or obj.data is None:
        return 0
    cleaned: Dict[int, float] = {}
    for vert_index, weight in dict(vert_weights or {}).items():
        idx = int(vert_index)
        w = float(weight)
        if idx < 0 or w <= 1.0e-8:
            continue
        prev = float(cleaned.get(idx, 0.0))
        if w > prev:
            cleaned[idx] = w
    if not cleaned:
        return 0
    vg = obj.vertex_groups.get(str(group_name))
    if vg is None:
        vg = obj.vertex_groups.new(name=str(group_name))
    for vert_index, weight in sorted(cleaned.items()):
        try:
            vg.add([int(vert_index)], float(weight), "REPLACE")
        except Exception:
            continue
    return len(cleaned)


def _track_helper_height_threshold(helper_positions_by_group: Dict[str, Vector] | None) -> float | None:
    zs = sorted(
        float(pos.z)
        for pos in dict(helper_positions_by_group or {}).values()
        if isinstance(pos, Vector)
    )
    if len(zs) < 2:
        return None
    best_gap = -1.0
    threshold = None
    for prev_z, next_z in zip(zs, zs[1:]):
        gap = float(next_z) - float(prev_z)
        if gap > best_gap:
            best_gap = gap
            threshold = (float(prev_z) + float(next_z)) * 0.5
    if threshold is None:
        threshold = sum(zs) / float(len(zs))
    return float(threshold)


def _synthesize_zero_weight_track_lane(
    obj: bpy.types.Object,
    zero_vertices: Sequence[int],
    track_group_names: Sequence[str],
    group_source_by_name: Dict[str, str] | None = None,
    helper_positions_by_group: Dict[str, Vector] | None = None,
) -> int:
    if obj.type != "MESH" or obj.data is None:
        return 0
    mesh = obj.data
    zero_indices = sorted({int(idx) for idx in zero_vertices if int(idx) >= 0})
    if not zero_indices:
        return 0

    group_source_by_name = {
        str(key): str(value)
        for key, value in dict(group_source_by_name or {}).items()
        if str(key).strip()
    }
    helper_positions_by_group = {
        str(name): pos.copy()
        for name, pos in dict(helper_positions_by_group or {}).items()
        if str(name).strip() and isinstance(pos, Vector)
    }
    synthetic_items = [
        (str(name), helper_positions_by_group[str(name)].copy())
        for name, source in group_source_by_name.items()
        if _norm_low(str(source)) == "synthetic" and str(name) in helper_positions_by_group
    ]
    if not synthetic_items:
        return 0

    sorted_by_x = sorted(
        synthetic_items,
        key=lambda item: (-float(item[1].x), _track_group_sort_key(str(item[0]))),
    )
    front_end_name = str(sorted_by_x[0][0])
    rear_end_name = str(sorted_by_x[-1][0])
    upper_threshold = _track_helper_height_threshold(helper_positions_by_group)
    upper_names: List[str] = []
    if upper_threshold is not None:
        upper_names = [
            str(name)
            for name, pos in sorted_by_x
            if float(pos.z) >= float(upper_threshold) - 1.0e-6
        ]
    if len(upper_names) < 2:
        upper_names = [str(name) for name, _pos in sorted_by_x]

    filled = 0
    upper_zero = [
        int(vert_index)
        for vert_index in zero_indices
        if upper_threshold is None
        or float(mesh.vertices[int(vert_index)].co.z) >= float(upper_threshold) - 1.0e-6
    ]
    remaining_zero = [
        int(vert_index)
        for vert_index in zero_indices
        if int(vert_index) not in set(upper_zero)
    ]
    # Voronoi-in-X: assign every belt vertex to its NEAREST synthetic wheel helper in X.
    # The old code split the upper return lane by ONE midpoint and the lower (ground) lane
    # between only the two endcaps, so each endcap group owned ~half the belt -> the
    # weight-bleed bug (a single Roue_Elev_* group spanning ~half the belt; e.g. gepard
    # Roue_Elev_D1 diag=4.7, raketenjagd D2/G2 diag=3.7). Nearest-helper bucketing localizes
    # each group to roughly one inter-wheel pitch.
    upper_pool = [(str(n), helper_positions_by_group.get(str(n))) for n in upper_names]
    upper_pool = [(n, p) for n, p in upper_pool if isinstance(p, Vector)]
    all_pool = [(str(n), p) for n, p in synthetic_items if isinstance(p, Vector)]

    def _bucket_by_nearest_x(verts, pool):
        buckets: Dict[str, List[int]] = defaultdict(list)
        for vert_index in verts:
            vx = float(mesh.vertices[int(vert_index)].co.x)
            best = min(pool, key=lambda np: abs(vx - float(np[1].x)))[0]
            buckets[str(best)].append(int(vert_index))
        return buckets

    # upper return lane -> nearest UPPER (return-roller) helper; lower ground-contact lane
    # -> nearest of ALL wheel helpers. Each lane painted exclusively (weights stay normalized).
    for lane_verts, lane_pool in (
        (upper_zero, upper_pool if len(upper_pool) >= 2 else all_pool),
        (remaining_zero, all_pool),
    ):
        if not lane_verts:
            continue
        if not lane_pool:
            filled += _assign_track_group_patch(
                obj=obj, group_name=str(front_end_name), vert_indices=list(lane_verts),
                track_group_names=track_group_names, exclusive=True)
            continue
        for group_name, vert_indices in _bucket_by_nearest_x(lane_verts, lane_pool).items():
            filled += _assign_track_group_patch(
                obj=obj, group_name=str(group_name), vert_indices=vert_indices,
                track_group_names=track_group_names, exclusive=True)
    return filled


def _refine_track_weights_from_mesh_positions(
    imported_objects: Sequence[bpy.types.Object],
    settings: Any | None = None,
) -> Dict[str, Dict[str, int]]:
    def _object_center_world(obj: bpy.types.Object) -> Vector:
        try:
            corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            if corners:
                acc = Vector((0.0, 0.0, 0.0))
                for corner in corners:
                    acc += corner
                return acc / float(len(corners))
        except Exception:
            pass
        try:
            return obj.matrix_world.to_translation().copy()
        except Exception:
            return Vector((0.0, 0.0, 0.0))

    mesh_by_name_low: Dict[str, bpy.types.Object] = {}
    for obj in imported_objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        mesh_by_name_low[_norm_low(str(obj.name))] = obj

    reports: Dict[str, Dict[str, int]] = {}
    wheel_rx = re.compile(r"^roue_elev_([dg])([0-9]+)$", re.IGNORECASE)
    for obj in imported_objects:
        if getattr(obj, "type", "") != "MESH" or not bool(obj.get("warno_is_track_mesh", False)):
            continue
        try:
            group_sources = json.loads(str(obj.get("warno_track_group_sources_json", "") or "") or "{}")
        except Exception:
            group_sources = {}
        group_sources = {
            str(name): str(source)
            for name, source in dict(group_sources or {}).items()
            if str(name).strip()
        }
        synthetic_names = [
            str(name)
            for name, source in group_sources.items()
            if _norm_low(str(source)) == "synthetic"
        ]
        if not synthetic_names:
            continue

        helper_positions_by_group: Dict[str, Vector] = {}
        for group_name in sorted(group_sources.keys(), key=_track_group_sort_key):
            m = wheel_rx.match(_norm_low(str(group_name)))
            if not m:
                continue
            wheel_name = f"Roue_{str(m.group(1)).upper()}{int(m.group(2))}"
            wheel_obj = mesh_by_name_low.get(_norm_low(wheel_name))
            if wheel_obj is None:
                continue
            helper_positions_by_group[str(group_name)] = _object_center_world(wheel_obj)
        if len(helper_positions_by_group) < 2:
            continue

        all_indices = [int(v.index) for v in getattr(obj.data, "vertices", []) or []]
        for group_name in synthetic_names:
            vg = obj.vertex_groups.get(str(group_name))
            if vg is None or not all_indices:
                continue
            try:
                vg.remove(all_indices)
            except Exception:
                continue

        backfill_report = _backfill_zero_weight_track_vertices(
            obj=obj,
            track_group_names=sorted(group_sources.keys(), key=_track_group_sort_key),
            group_source_by_name=group_sources,
            helper_positions_by_group=helper_positions_by_group,
        )
        reports[str(obj.name)] = {
            "zero_before": int(backfill_report.get("zero_before", 0)),
            "zero_after": int(backfill_report.get("zero_after", 0)),
            "filled": int(backfill_report.get("filled", 0)),
        }
        if settings is not None:
            _warno_log(
                settings,
                (
                    f"track weight refine obj={obj.name} "
                    f"zero_before={int(backfill_report.get('zero_before', 0))} "
                    f"zero_after={int(backfill_report.get('zero_after', 0))} "
                    f"filled={int(backfill_report.get('filled', 0))}"
                ),
                stage="track_weights",
            )
    return reports


def _backfill_zero_weight_track_vertices(
    obj: bpy.types.Object,
    track_group_names: Sequence[str],
    group_source_by_name: Dict[str, str] | None = None,
    helper_positions_by_group: Dict[str, Vector] | None = None,
) -> Dict[str, int]:
    if obj.type != "MESH" or obj.data is None:
        return {"filled": 0, "zero_before": 0, "zero_after": 0}
    mesh = obj.data
    if not getattr(mesh, "vertices", None):
        return {"filled": 0, "zero_before": 0, "zero_after": 0}

    valid_group_names: List[str] = []
    valid_group_indices: set[int] = set()
    group_name_by_index: Dict[int, str] = {}
    group_source_by_name = {
        str(key): str(value)
        for key, value in dict(group_source_by_name or {}).items()
        if str(key).strip()
    }
    for group_name in track_group_names:
        vg = obj.vertex_groups.get(str(group_name))
        if vg is None:
            continue
        vg_name = str(vg.name)
        valid_group_names.append(vg_name)
        valid_group_indices.add(int(vg.index))
        group_name_by_index[int(vg.index)] = vg_name
    if not valid_group_names:
        return {"filled": 0, "zero_before": 0, "zero_after": 0}

    helper_positions_by_group = {
        str(name): pos.copy()
        for name, pos in dict(helper_positions_by_group or {}).items()
        if str(name).strip() and isinstance(pos, Vector)
    }
    xs = [float(v.co.x) for v in mesh.vertices]
    ys = [float(v.co.y) for v in mesh.vertices]
    zs = [float(v.co.z) for v in mesh.vertices]
    span_x = max(xs) - min(xs) if xs else 0.0
    span_y = max(ys) - min(ys) if ys else 0.0
    span_z = max(zs) - min(zs) if zs else 0.0
    neighbors = _mesh_vertex_neighbors(mesh)

    def _scan_state() -> Tuple[Dict[int, str], List[int], Dict[str, List[int]]]:
        dominant_group_by_vertex: Dict[int, str] = {}
        zero_vertices_local: List[int] = []
        group_vertices: Dict[str, List[int]] = defaultdict(list)
        for vert in mesh.vertices:
            best_name = ""
            best_weight = -1.0
            total = 0.0
            for slot in vert.groups:
                if int(slot.group) not in valid_group_indices:
                    continue
                vg_name = group_name_by_index.get(int(slot.group), "")
                if not vg_name:
                    continue
                weight = float(slot.weight)
                if weight <= 1.0e-8:
                    continue
                total += weight
                if weight > best_weight:
                    best_weight = weight
                    best_name = vg_name
            if total > 1.0e-8 and best_name:
                dominant_group_by_vertex[int(vert.index)] = best_name
                group_vertices[str(best_name)].append(int(vert.index))
            else:
                zero_vertices_local.append(int(vert.index))
        return dominant_group_by_vertex, zero_vertices_local, group_vertices

    def _group_assignments_by_bfs(
        source_group_by_vertex: Dict[int, str],
        target_vertices: set[int],
        allow_group_vertex: Any | None = None,
        max_hops: int | None = None,
    ) -> Dict[int, str]:
        if not source_group_by_vertex or not target_vertices:
            return {}
        assigned: Dict[int, str] = {}
        seen: set[Tuple[int, str]] = set()
        queue: List[Tuple[int, int, str]] = []
        for vert_index, group_name in source_group_by_vertex.items():
            queue.append((0, int(vert_index), str(group_name)))
            seen.add((int(vert_index), str(group_name)))
        queue.sort(key=lambda item: (int(item[0]), int(item[1]), str(item[2]).lower()))
        head = 0
        while head < len(queue):
            hops, vert_index, group_name = queue[head]
            head += 1
            if max_hops is not None and int(hops) >= int(max_hops):
                continue
            for neigh in sorted(neighbors[int(vert_index)]):
                neigh_idx = int(neigh)
                state_key = (neigh_idx, str(group_name))
                if state_key in seen:
                    continue
                seen.add(state_key)
                if allow_group_vertex is not None and not bool(allow_group_vertex(str(group_name), neigh_idx)):
                    continue
                if neigh_idx in target_vertices and neigh_idx not in assigned:
                    assigned[neigh_idx] = str(group_name)
                queue.append((int(hops) + 1, neigh_idx, str(group_name)))
        return assigned

    dominant_group_by_vertex, zero_vertices, group_vertices = _scan_state()
    zero_before = len(zero_vertices)
    if not zero_vertices or not dominant_group_by_vertex:
        return {"filled": 0, "zero_before": zero_before, "zero_after": zero_before}

    filled = 0

    # First resolve the upper return lane / end-cap patches that are missing from raw
    # Leopard-like track data. Those vertices should belong to synthetic helper groups,
    # not get soaked up by lower raw wheel groups.
    if helper_positions_by_group:
        filled += _synthesize_zero_weight_track_lane(
            obj=obj,
            zero_vertices=zero_vertices,
            track_group_names=track_group_names,
            group_source_by_name=group_source_by_name,
            helper_positions_by_group=helper_positions_by_group,
        )

    dominant_group_by_vertex, zero_vertices, _group_vertices = _scan_state()
    remaining_zero = set(int(vi) for vi in zero_vertices)
    if remaining_zero:
        raw_sources = {
            int(vert_index): str(group_name)
            for vert_index, group_name in dominant_group_by_vertex.items()
            if _norm_low(group_source_by_name.get(str(group_name), "raw")) != "synthetic"
        }
        if not raw_sources:
            raw_sources = {
                int(vert_index): str(group_name)
                for vert_index, group_name in dominant_group_by_vertex.items()
            }
        raw_assignments = _group_assignments_by_bfs(
            source_group_by_vertex=raw_sources,
            target_vertices=remaining_zero,
        )
        raw_group_assignments: Dict[str, List[int]] = defaultdict(list)
        for vert_index, group_name in raw_assignments.items():
            raw_group_assignments[str(group_name)].append(int(vert_index))
        for group_name, group_verts in raw_group_assignments.items():
            filled += _assign_track_group_patch(
                obj=obj,
                group_name=str(group_name),
                vert_indices=group_verts,
                track_group_names=track_group_names,
            )

    _dominant_final, zero_vertices_final, _group_vertices_final = _scan_state()
    zero_after = len(zero_vertices_final)
    return {"filled": filled, "zero_before": zero_before, "zero_after": zero_after}


def _synthesize_missing_track_vertex_groups(
    obj: bpy.types.Object,
    side_token: str,
    bone_name_by_index: Dict[int, str],
    bone_positions: Dict[str, Any] | None,
    existing_group_names: Sequence[str] | None = None,
) -> Dict[str, int]:
    if obj.type != "MESH" or obj.data is None:
        return {}
    side = str(side_token or "").strip().lower()
    if side not in {"d", "g"}:
        return {}
    mesh = obj.data
    if not getattr(mesh, "vertices", None):
        return {}

    side_candidates = _collect_track_helper_candidates(
        side_token=side,
        bone_name_by_index=bone_name_by_index,
        bone_positions=bone_positions,
    )
    if len(side_candidates) < 2:
        return {}

    existing_group_names = [str(name) for name in list(existing_group_names or []) if str(name).strip()]
    created_or_filled: Dict[str, int] = {}
    for vg_name, _target_pos in side_candidates:
        vg = obj.vertex_groups.get(str(vg_name))
        if vg is None:
            obj.vertex_groups.new(name=str(vg_name))
            created_or_filled[str(vg_name)] = 0
    return created_or_filled


def _relabel_synthetic_track_groups(
    obj: bpy.types.Object,
    group_source_by_name: Dict[str, str] | None,
    helper_positions_by_group: Dict[str, Vector] | None,
) -> int:
    if obj.type != "MESH" or obj.data is None:
        return 0
    mesh = obj.data
    group_source_by_name = {
        str(name): str(source)
        for name, source in dict(group_source_by_name or {}).items()
        if str(name).strip()
    }
    helper_positions_by_group = {
        str(name): pos.copy()
        for name, pos in dict(helper_positions_by_group or {}).items()
        if str(name).strip() and isinstance(pos, Vector)
    }
    synthetic_names = [
        str(name)
        for name, source_name in group_source_by_name.items()
        if _norm_low(str(source_name)) == "synthetic" and str(name) in helper_positions_by_group
    ]
    if len(synthetic_names) < 2:
        return 0

    occupied: Dict[str, Dict[str, Any]] = {}
    for group_name in synthetic_names:
        vg = obj.vertex_groups.get(str(group_name))
        if vg is None:
            continue
        entries: List[Tuple[int, float]] = []
        sx = sy = sz = sw = 0.0
        vg_index = int(vg.index)
        for vert in mesh.vertices:
            weight = 0.0
            for slot in vert.groups:
                if int(slot.group) == vg_index and float(slot.weight) > 1.0e-8:
                    weight = float(slot.weight)
                    break
            if weight <= 1.0e-8:
                continue
            entries.append((int(vert.index), float(weight)))
            sx += float(vert.co.x) * float(weight)
            sy += float(vert.co.y) * float(weight)
            sz += float(vert.co.z) * float(weight)
            sw += float(weight)
        if not entries or sw <= 1.0e-8:
            continue
        occupied[str(group_name)] = {
            "entries": entries,
            "centroid": Vector((sx / sw, sy / sw, sz / sw)),
        }
    if not occupied:
        return 0

    occupied_names = sorted(occupied.keys(), key=_track_group_sort_key)
    target_names = sorted(synthetic_names, key=_track_group_sort_key)
    best_cost = float("inf")
    best_mapping: Dict[str, str] = {}
    for target_subset in combinations(target_names, len(occupied_names)):
        for target_perm in permutations(target_subset):
            cost = 0.0
            for source_name, target_name in zip(occupied_names, target_perm):
                centroid = occupied[str(source_name)]["centroid"]
                helper_pos = helper_positions_by_group[str(target_name)]
                cost += (
                    abs(float(centroid.x) - float(helper_pos.x)) * 4.0
                    + abs(float(centroid.y) - float(helper_pos.y)) * 0.5
                    + abs(float(centroid.z) - float(helper_pos.z)) * 1.0
                )
            if cost < best_cost:
                best_cost = float(cost)
                best_mapping = {
                    str(source_name): str(target_name)
                    for source_name, target_name in zip(occupied_names, target_perm)
                }
    if not best_mapping:
        return 0

    changed = 0
    target_entries: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    source_groups: Dict[str, bpy.types.VertexGroup] = {}
    target_groups: Dict[str, bpy.types.VertexGroup] = {}
    for source_name, target_name in best_mapping.items():
        source_vg = obj.vertex_groups.get(str(source_name))
        if source_vg is None:
            continue
        if obj.vertex_groups.get(str(target_name)) is None:
            obj.vertex_groups.new(name=str(target_name))
        target_vg = obj.vertex_groups.get(str(target_name))
        if target_vg is None:
            continue
        source_groups[str(source_name)] = source_vg
        target_groups[str(target_name)] = target_vg
        target_entries[str(target_name)].extend(list(occupied[str(source_name)]["entries"]))
        if _norm_low(str(source_name)) != _norm_low(str(target_name)):
            changed += 1
    if changed <= 0:
        return 0

    for source_name, source_vg in source_groups.items():
        indices = [int(vert_index) for vert_index, _weight in occupied[str(source_name)]["entries"]]
        if not indices:
            continue
        try:
            source_vg.remove(indices)
        except Exception:
            pass
    for target_name, entries in target_entries.items():
        target_vg = target_groups.get(str(target_name))
        if target_vg is None:
            continue
        for vert_index, weight in entries:
            try:
                target_vg.add([int(vert_index)], float(weight), "REPLACE")
            except Exception:
                continue
    return changed


def _rebalance_rd_track_weights(obj, track_group_names, helper_positions_by_group) -> int:
    """RD chenille ONLY: re-assign EVERY belt vertex exclusively to its NEAREST wheel helper
    in X. The game's RAW belt weights are coarse — one wheel (e.g. gepard Roue_Elev_D1) can
    span half the belt, which paints ugly and deforms badly. This Voronoi-in-X partition gives
    a clean per-wheel rig consistent with the synthetic backfill. WARNO tracks (no chenille
    flag) never reach this. Returns the number of re-bucketed vertices."""
    if obj.type != "MESH" or obj.data is None:
        return 0
    pool = [(str(n), p) for n, p in (helper_positions_by_group or {}).items() if isinstance(p, Vector)]
    if len(pool) < 2:
        return 0
    gi = {g.name: g.index for g in obj.vertex_groups}
    track_idx = {n: gi[n] for n in track_group_names if n in gi}
    if len(track_idx) < 2:
        return 0
    mesh = obj.data
    buckets: Dict[str, List[int]] = defaultdict(list)
    for v in mesh.vertices:
        vx = float(v.co.x)
        best = min(pool, key=lambda np: abs(vx - float(np[1].x)))[0]
        if str(best) in track_idx:
            buckets[str(best)].append(int(v.index))
    moved = 0
    vg_by_name = {n: obj.vertex_groups[i] for n, i in track_idx.items()}
    for gname, verts in buckets.items():
        if not verts:
            continue
        for other_name, vg in vg_by_name.items():
            if other_name != gname:
                try:
                    vg.remove(verts)
                except Exception:
                    pass
        try:
            vg_by_name[gname].add(verts, 1.0, "REPLACE")
            moved += len(verts)
        except Exception:
            pass
    return moved


def _apply_vertex_groups_from_bucket(
    obj: bpy.types.Object,
    bucket: Dict[str, Any],
    model: Dict[str, Any],
    bone_name_by_index: Dict[int, str],
    raw_bone_name_by_index: Dict[int, str] | None = None,
    bone_positions: Dict[int, Any] | None = None,
) -> Dict[str, int]:
    result = {
        "group_added": 0,
        "track_groups_added": 0,
        "track_groups_synth": 0,
        "track_zero_weight_filled": 0,
        "track_zero_weight_before": 0,
        "track_zero_weight_after": 0,
        "track_groups_explicit": 0,
        "track_group_sources": {},
    }
    if obj.type != "MESH" or obj.data is None:
        return result

    source_refs = list(bucket.get("source_refs", []) or [])
    if not source_refs:
        return result

    low_name = _norm_low(obj.name)
    is_track = low_name.startswith("chenille_") or "track" in low_name
    side_token = ""
    if "gauche" in low_name or "left" in low_name:
        side_token = "g"
    elif "droite" in low_name or "right" in low_name:
        side_token = "d"

    # RD chenille (IRISZoom flag) vs WARNO track (also named chenille, but NO flag). Only the
    # RD belt gets the per-wheel weight rebalance; WARNO tracks keep their real raw weights.
    is_rd_track = False
    if is_track and source_refs:
        try:
            _p0 = int(source_refs[0][0])
            _parts = model.get("parts", []) or []
            is_rd_track = bool(0 <= _p0 < len(_parts) and (_parts[_p0].get("vertices", {}) or {}).get("chenille"))
        except Exception:
            is_rd_track = False

    def _raw_bone_payload(part_i: int) -> Tuple[List[float], List[float]]:
        parts = list(model.get("parts", []) or [])
        if not (0 <= int(part_i) < len(parts)):
            return [], []
        verts = parts[int(part_i)].get("vertices", {}) or {}
        return list(verts.get("bone_idx", []) or []), list(verts.get("bone_w", []) or [])

    track_weights: Dict[str, Dict[int, float]] = {}
    deform_weights: Dict[str, Dict[int, float]] = {}
    dom_hits = 0
    dom_total = 0
    group_bone_index = int(bucket.get("group_bone_index", -1))
    raw_bone_name_by_index = raw_bone_name_by_index or {}

    def _vertex_group_name_for_bone_index(bidx: int) -> str:
        raw_name = str(raw_bone_name_by_index.get(int(bidx), bone_name_by_index.get(int(bidx), "")) or "").strip()
        raw_low = _norm_low(raw_name)
        if raw_low.startswith("bip01"):
            return _pretty_character_bip_name(raw_name)
        pretty = _pretty_warno_node_name(raw_name)
        return str(pretty or raw_name or f"bone_{int(bidx):03d}")

    for mapped_idx, ref in enumerate(source_refs):
        if not isinstance(ref, (list, tuple)) or len(ref) < 2:
            continue
        part_i = int(ref[0])
        src_vi = int(ref[1])
        raw_idx, raw_w = _raw_bone_payload(part_i)
        base = src_vi * 4
        if base + 3 >= len(raw_idx) or base + 3 >= len(raw_w):
            continue

        idxs = [
            int(raw_idx[base + 0]),
            int(raw_idx[base + 1]),
            int(raw_idx[base + 2]),
            int(raw_idx[base + 3]),
        ]
        ws = [
            float(raw_w[base + 0]),
            float(raw_w[base + 1]),
            float(raw_w[base + 2]),
            float(raw_w[base + 3]),
        ]
        best_slot = max(range(4), key=lambda i: ws[i])
        if group_bone_index >= 0:
            dom_total += 1
            if idxs[best_slot] == group_bone_index:
                dom_hits += 1

        for slot_idx in range(4):
            weight = float(ws[slot_idx])
            if weight <= 1.0e-4:
                continue
            bidx = int(idxs[slot_idx])
            raw_name = str(bone_name_by_index.get(int(bidx), "") or "").strip()
            low = _norm_low(raw_name)
            if is_track:
                if not low.startswith("roue_elev_"):
                    continue
                if side_token and not re.match(rf"^roue_elev_{side_token}[0-9]+$", low, re.IGNORECASE):
                    continue
                vg_name = _pretty_warno_node_name(raw_name)
                weights_for_group = track_weights.setdefault(vg_name, {})
                prev = float(weights_for_group.get(int(mapped_idx), 0.0))
                if weight > prev:
                    weights_for_group[int(mapped_idx)] = weight
                continue
            if not low.startswith("bip01"):
                continue
            vg_name = _vertex_group_name_for_bone_index(int(bidx))
            weights_for_group = deform_weights.setdefault(vg_name, {})
            prev = float(weights_for_group.get(int(mapped_idx), 0.0))
            if weight > prev:
                weights_for_group[int(mapped_idx)] = weight

    if is_track:
        created_track_group_names: List[str] = []
        group_source_by_name: Dict[str, str] = {}
        for vg_name in sorted(track_weights.keys(), key=str.lower):
            if obj.vertex_groups.get(vg_name) is None:
                vg = obj.vertex_groups.new(name=vg_name)
            else:
                vg = obj.vertex_groups[vg_name]
            for mapped_idx, weight in sorted(track_weights[vg_name].items()):
                vg.add([int(mapped_idx)], float(weight), "REPLACE")
            result["track_groups_added"] += 1
            result["track_groups_explicit"] += 1
            created_track_group_names.append(str(vg_name))
            group_source_by_name[str(vg_name)] = "raw"
        result["group_added"] += len(track_weights)
        synth_groups = _synthesize_missing_track_vertex_groups(
            obj=obj,
            side_token=side_token,
            bone_name_by_index=bone_name_by_index,
            bone_positions=bone_positions,
            existing_group_names=created_track_group_names,
        )
        if synth_groups:
            for synth_name in sorted(synth_groups.keys(), key=str.lower):
                if synth_name not in created_track_group_names:
                    created_track_group_names.append(str(synth_name))
                group_source_by_name[str(synth_name)] = "synthetic"
            result["track_groups_synth"] += len(list(synth_groups.keys()))
        helper_positions_by_group = {
            str(name): pos.copy()
            for name, pos in _collect_track_helper_candidates(
                side_token=side_token,
                bone_name_by_index=bone_name_by_index,
                bone_positions=bone_positions,
            )
            if str(name) in created_track_group_names
        }
        backfill_report = _backfill_zero_weight_track_vertices(
            obj,
            created_track_group_names,
            group_source_by_name=group_source_by_name,
            helper_positions_by_group=helper_positions_by_group,
        )
        result["track_zero_weight_filled"] += int(backfill_report.get("filled", 0))
        result["track_zero_weight_before"] += int(backfill_report.get("zero_before", 0))
        result["track_zero_weight_after"] += int(backfill_report.get("zero_after", 0))
        # RD chenille: rebalance EVERY belt vertex to its nearest wheel (the raw belt weights
        # are coarse -> one wheel spans half the belt). Clean per-wheel rig; RD-only.
        if is_rd_track:
            try:
                result["track_rebalanced"] = _rebalance_rd_track_weights(
                    obj, created_track_group_names, helper_positions_by_group)
            except Exception:
                pass
        result["track_group_sources"] = {
            str(name): str(group_source_by_name.get(str(name), "raw"))
            for name in sorted(created_track_group_names, key=_track_group_sort_key)
        }
        return result

    if deform_weights:
        for vg_name in sorted(deform_weights.keys(), key=str.lower):
            vg = obj.vertex_groups.get(str(vg_name))
            if vg is None:
                vg = obj.vertex_groups.new(name=str(vg_name))
            for mapped_idx, weight in sorted(deform_weights[vg_name].items()):
                vg.add([int(mapped_idx)], float(weight), "REPLACE")
        result["group_added"] += len(deform_weights)
        return result

    if _skip_default_group_assignment(low_name):
        return result
    return result


def _ensure_armature_modifier(
    obj: bpy.types.Object,
    name: str,
    armature_obj: bpy.types.Object,
) -> bool:
    if obj.type != "MESH" or armature_obj is None or armature_obj.type != "ARMATURE":
        return False
    mod = obj.modifiers.get(str(name))
    if mod is None or mod.type != "ARMATURE":
        mod = obj.modifiers.new(name=str(name), type="ARMATURE")
    changed = False
    if getattr(mod, "object", None) != armature_obj:
        mod.object = armature_obj
        changed = True
    if hasattr(mod, "use_vertex_groups") and not bool(mod.use_vertex_groups):
        mod.use_vertex_groups = True
        changed = True
    if hasattr(mod, "use_bone_envelopes") and bool(mod.use_bone_envelopes):
        mod.use_bone_envelopes = False
        changed = True
    return changed


def _apply_auto_smooth_modifier(objects: Sequence[bpy.types.Object], angle_deg: float) -> int:
    if angle_deg <= 0.0:
        return 0
    angle_rad = math.radians(angle_deg)
    mesh_objs = [o for o in objects if o.type == "MESH"]
    if not mesh_objs:
        return 0

    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    count = 0
    for obj in mesh_objs:
        for o in bpy.context.scene.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=angle_rad)
        except Exception:
            try:
                bpy.ops.object.shade_smooth_by_angle(angle=angle_rad, keep_sharp_edges=True)
            except Exception:
                continue

        # Accept/apply generated smooth-by-angle modifier.
        for mod in list(obj.modifiers):
            if mod.type == "NODES" and "smooth by angle" in mod.name.lower():
                try:
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception:
                    pass
        count += 1
    return count


def _ensure_named_auto_smooth_modifiers(objects: Sequence[bpy.types.Object], angle_deg: float) -> int:
    if angle_deg <= 0.0:
        return 0
    angle_rad = math.radians(angle_deg)
    targets = [obj for obj in objects if obj.type == "MESH"]
    if not targets:
        return 0

    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    count = 0
    for obj in targets:
        for scene_obj in bpy.context.scene.objects:
            scene_obj.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=angle_rad)
        except Exception:
            try:
                bpy.ops.object.shade_smooth_by_angle(angle=angle_rad, keep_sharp_edges=True)
            except Exception:
                continue
        for mod in obj.modifiers:
            if mod.type != "NODES":
                continue
            low = _norm_low(mod.name)
            if "smooth" not in low or "angle" not in low:
                continue
            try:
                mod.name = "Auto Smooth"
            except Exception:
                pass
            count += 1
            break
    return count


def _mesh_bounds(objects: Sequence[bpy.types.Object]) -> tuple[Vector, float]:
    pts: list[Vector] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        if not obj.bound_box:
            pts.append(obj.matrix_world.translation.copy())
            continue
        for c in obj.bound_box:
            pts.append(obj.matrix_world @ Vector((c[0], c[1], c[2])))
    if not pts:
        return Vector((0.0, 0.0, 0.0)), 0.0
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (min_v + max_v) * 0.5
    diag = (max_v - min_v).length
    return center, diag


def _pick_position_from_payload(
    raw_name: str,
    bone_positions: dict[str, list[float]],
) -> Vector | None:
    low = _norm_low(raw_name)
    keys = [low, _norm_token(low)]
    for key in keys:
        if not key:
            continue
        hit = bone_positions.get(key)
        if isinstance(hit, list) and len(hit) >= 3:
            try:
                return Vector((float(hit[0]), float(hit[1]), float(hit[2])))
            except Exception:
                pass
    return None


def _pretty_warno_node_name(raw_name: str) -> str:
    low = _norm_low(raw_name)
    if not low:
        return "Empty"
    m = re.match(r"^roue_(elev_)?([dg])([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        prefix = "Roue_Elev" if m.group(1) else "Roue"
        side = str(m.group(2)).upper()
        num = int(m.group(3))
        return f"{prefix}_{side}{num}"
    m = re.match(r"^roue_(elev_)?(droite|gauche)_?([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        prefix = "Roue_Elev" if m.group(1) else "Roue"
        side = "D" if str(m.group(2)).lower().startswith("droi") else "G"
        num = int(m.group(3))
        return f"{prefix}_{side}{num}"
    m = re.match(r"^roue_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return f"Roue_{int(m.group(1)):02d}"
    m = re.match(r"^armature_([dg])([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        side = str(m.group(1)).upper()
        num = int(m.group(2))
        return f"Armature_{side}{num}"
    m = re.match(r"^tourelle_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return f"Tourelle_{int(m.group(1)):02d}"
    m = re.match(r"^axe_canon_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return f"Axe_Canon_{int(m.group(1)):02d}"
    m = re.match(r"^canon_([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return f"Canon_{int(m.group(1)):02d}"
    m = re.match(r"^blindage_([dg])([0-9]+)$", low, flags=re.IGNORECASE)
    if m:
        return f"Blindage_{str(m.group(1)).upper()}{int(m.group(2))}"
    m = re.match(r"^chenille_(droite|gauche)(?:_([0-9]+))?$", low, flags=re.IGNORECASE)
    if m:
        side = "Droite" if m.group(1).lower() == "droite" else "Gauche"
        n = m.group(2)
        if n is None:
            return f"Chenille_{side}"
        return f"Chenille_{side}_{int(n):02d}"
    parts = [p for p in re.split(r"_+", str(raw_name or "").strip()) if p]
    if not parts:
        return "Empty"
    out_parts: List[str] = []
    for token in parts:
        if token.isdigit():
            out_parts.append(token)
        elif len(token) == 1 and token.lower() in {"d", "g"}:
            out_parts.append(token.upper())
        else:
            out_parts.append(token[:1].upper() + token[1:])
    return "_".join(out_parts)


def _display_warno_node_name(raw_name: str, asset_hint: str = "") -> str:
    raw = str(raw_name or "").strip()
    low = _norm_low(raw)
    if not low:
        return "Empty"
    asset_low = _norm_low(asset_hint).replace("\\", "/")
    if "/ammo/armes/" in asset_low:
        return raw or low
    if low in {"papyrus", "props", "soldat", "trappe_avant"}:
        return low
    if low == "fx_tir":
        return low
    if "/helico/" in asset_low and low.startswith("fx_tourelle2_tir_"):
        return low
    if "/canon/" in asset_low and low.startswith("fx_tourelle"):
        return f"Fx_{low[3:]}"

    parts = [p for p in re.split(r"_+", raw) if p]
    if not parts:
        return _pretty_warno_node_name(raw)

    def _fmt_token(token: str) -> str:
        tok = str(token or "")
        tok_low = tok.lower()
        if not tok:
            return tok
        if tok.isdigit():
            return tok
        if tok_low in {"launcher", "pod"}:
            return tok_low
        if len(tok) == 1 and tok_low in {"d", "g", "l", "r", "v"}:
            return tok.upper()
        if len(tok) <= 2 and tok.isalpha():
            return tok.upper()
        return tok[:1].upper() + tok[1:]

    if low.startswith("fx_helice_") or low.startswith("fx_fumee_roue_"):
        return "FX_" + "_".join(_fmt_token(tok) for tok in parts[1:])
    if low.startswith("fx_"):
        return "Fx_" + "_".join(_fmt_token(tok) for tok in parts[1:])

    return "_".join(_fmt_token(tok) for tok in parts)


def _is_aircraft_asset(asset_hint: str) -> bool:
    low = _norm_low(asset_hint).replace("\\", "/")
    return "/avion/" in low or "/plane/" in low or "/aircraft/" in low


def _asset_prefers_synthetic_default_group(asset_hint: str) -> bool:
    low = _norm_low(asset_hint).replace("\\", "/")
    if not low:
        return True
    if "/ammo/armes/" in low or "/weapon/" in low or "/weapons/" in low:
        return False
    if "/helico/" in low or _is_aircraft_asset(low):
        return False
    return True


def _is_aircraft_control_surface_name(raw_name: str) -> bool:
    low = _norm_low(raw_name)
    return low == "rudder" or low.startswith("aileron_") or low.startswith("elevator_")


# ===== Per-asset Blender datablock namespacing =====
# Different WARNO units historically reused the same Blender material name
# (e.g. "Vitre" for any glass, "Mi_24V" for the body). Blender's data API
# resolves identical names to the same datablock, which caused freshly
# imported units to silently inherit textures/nodes from a previously
# imported (and possibly already-deleted) unit. We avoid that by prefixing
# every material with a sanitized asset namespace so each unit owns its
# own copy of "Vitre", "Mi_24V" etc.
_WARNO_MATERIAL_NS_SEP = "__"

# Material names that the WARNO asset cooker requires verbatim — namespacing
# them with our '<asset>__' prefix breaks the in-game export. These materials
# are always created with their raw name globally; latest import wins for
# the node-tree contents (acceptable trade-off documented to the user).
_WARNO_RESERVED_MATERIAL_NAMES: set[str] = {"vitre"}


def _warno_asset_namespace(asset_hint: str) -> str:
    """Derive a sanitized per-asset namespace string from an asset path,
    e.g. 'Assets/3D/Units/US/Vehicule/M1038_Humvee/M1038_Humvee.fbx'
    -> 'M1038_Humvee'. Returns '' when the hint is empty so callers can
    fall back to the legacy unnamespaced behavior."""
    raw = str(asset_hint or "").strip().replace("\\", "/")
    if not raw:
        return ""
    stem = raw.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return cleaned


def _warno_namespaced_material_name(namespace: str, raw_name: str) -> str:
    """Combine an asset namespace with a raw material name, e.g.
    ('M1038_Humvee', 'Vitre') -> 'M1038_Humvee__Vitre'.

    Special cases (no namespace prefix is added):
      - raw is already prefixed with our namespace separator (idempotent).
      - raw equals the asset namespace itself (would dup the stem,
        e.g. 'Infanterie_Test__Infanterie_Test').

    WARNO-reserved names like 'Vitre' DO get the namespace at this stage —
    the post-import pass `_warno_finalize_reserved_material_names` then
    renames the populated material back to 'Vitre' once its node-tree
    holds the right asset's textures. This avoids accidental texture
    sharing while still complying with the cooker's name requirement.

    Returns raw_name unchanged when namespace is empty.
    """
    raw = str(raw_name or "").strip() or "Material"
    ns = str(namespace or "").strip()
    if not ns:
        return raw
    if raw.startswith(ns + _WARNO_MATERIAL_NS_SEP):
        return raw
    if raw.lower() == ns.lower():
        return raw
    return f"{ns}{_WARNO_MATERIAL_NS_SEP}{raw}"


def _warno_finalize_track_material_names(asset_hint: str) -> int:
    """Post-import: strip the asset namespace from each track mesh's material and
    rename it to the bare OBJECT name, e.g. '<asset>__Chenille_Gauche' on object
    'Chenille_Gauche_01' -> 'Chenille_Gauche_01'. The namespace is kept DURING the
    build to avoid cross-unit material collisions; stripping it afterwards gives the
    clean per-track names that match the objects. If a same-named material already
    exists (a previously-imported unit), Blender auto-suffixes '.001' — the intended
    collision avoidance. Each track object has its own material (one side per part),
    so this is a 1:1 rename."""
    namespace = _warno_asset_namespace(asset_hint)
    if not namespace:
        return 0
    prefix = namespace + _WARNO_MATERIAL_NS_SEP
    renamed = 0
    for obj in list(bpy.data.objects):
        if getattr(obj, "type", "") != "MESH":
            continue
        if not bool(obj.get("warno_is_track_mesh", False)):
            continue
        if str(obj.get("warno_asset", "") or "") != str(asset_hint):
            continue
        data = getattr(obj, "data", None)
        mats = getattr(data, "materials", None) if data is not None else None
        if not mats:
            continue
        for mat in mats:
            if mat is None or not str(mat.name).startswith(prefix):
                continue
            target = str(obj.name)
            if mat.name == target:
                continue
            try:
                mat.name = target
                renamed += 1
            except Exception:
                continue
    return renamed


def _warno_finalize_rd_material_names(asset_hint: str) -> int:
    """Post-import: strip the asset namespace from EVERY material of this asset so it gets the
    clean BARE name ('AIFV_B__Chenille_Droite' -> 'Chenille_Droite', 'merkava_1__Chassis' ->
    'Chassis'). Applies to BOTH RD and WARNO (the cooker wants the bare game name; the
    '<unit>__' prefix is only a Blender-side collision guard). Runs AFTER the reserved-name
    finalize (so '<unit>__Vitre' is already 'Vitre' and won't be touched) and BEFORE the track
    finalize (so tracks end up bare, not the object name). A collision with a leftover material
    from a previous import auto-suffixes ('.001'); re-importing the same unit purges the orphan
    first so the name stays clean."""
    namespace = _warno_asset_namespace(asset_hint)
    if not namespace:
        return 0
    prefix = namespace + _WARNO_MATERIAL_NS_SEP
    renamed = 0
    for mat in list(bpy.data.materials):
        if mat is None:
            continue
        nm = str(mat.name or "")
        if not nm.startswith(prefix):
            continue
        bare = _warno_strip_namespace(nm)
        if not bare or bare == nm:
            continue
        try:
            mat.name = bare
            renamed += 1
        except Exception:
            continue
    return renamed


def _warno_finalize_reserved_material_names(asset_hint: str) -> int:
    """Post-import pass: rename namespaced reserved materials (e.g.
    'M1038_Humvee__Vitre') back to their canonical name ('Vitre') so the
    WARNO asset cooker accepts the exported FBX.

    Existing global material with the canonical name (left over from a
    previously-imported unit) is "parked" under '<old_asset>__<canonical>'
    so its references on already-placed objects stay intact and that
    material's node-tree (with the OTHER unit's textures) is preserved.

    Returns the number of materials successfully renamed.
    """
    namespace = _warno_asset_namespace(asset_hint)
    if not namespace:
        return 0
    sep = _WARNO_MATERIAL_NS_SEP
    renamed = 0
    for reserved_low in _WARNO_RESERVED_MATERIAL_NAMES:
        cap = reserved_low.capitalize() if reserved_low else ""
        candidates_for_new = [
            f"{namespace}{sep}{cap}",
            f"{namespace}{sep}{reserved_low}",
            f"{namespace}{sep}{reserved_low.upper()}",
        ]
        new_mat = None
        for cand in candidates_for_new:
            new_mat = bpy.data.materials.get(cand)
            if new_mat is not None:
                break
        if new_mat is None:
            continue

        canonical_name = cap or reserved_low
        existing = bpy.data.materials.get(canonical_name)
        if existing is not None and existing is not new_mat:
            old_asset = str(existing.get("warno_asset", "") or "").strip() or "previous"
            parked_base = f"{old_asset}{sep}{canonical_name}"
            parked_name = parked_base
            counter = 1
            while bpy.data.materials.get(parked_name) is not None and counter < 1000:
                parked_name = f"{parked_base}_{counter}"
                counter += 1
            try:
                existing.name = parked_name
            except Exception:
                continue

        try:
            new_mat.name = canonical_name
            new_mat["warno_asset"] = namespace
            renamed += 1
        except Exception:
            continue
    return renamed


def _warno_strip_namespace(name: str) -> str:
    """Reverse of _warno_namespaced_material_name: if the given Blender
    datablock name carries the asset-namespace prefix (e.g. 'M1038_Humvee__Vitre'),
    return the raw part after the separator ('Vitre'). Otherwise return the
    name unchanged. Used by Apply/Reapply Textures to match against the
    raw material-name keys produced by _build_material_name_map."""
    raw = str(name or "")
    if not raw:
        return ""
    sep = _WARNO_MATERIAL_NS_SEP
    idx = raw.find(sep)
    if idx > 0:
        return raw[idx + len(sep):]
    return raw


def _warno_purge_orphan_datablocks() -> Dict[str, int]:
    """Purge orphan materials/images/meshes/node_groups (users == 0 and no
    fake user) from bpy.data BEFORE a new import so that a stale 'Vitre'
    or 'Mi_24V' material left over from a previously deleted unit cannot
    be re-bound by name to the new one. Idempotent."""
    purged = {"materials": 0, "images": 0, "meshes": 0, "node_groups": 0}
    try:
        for m in list(bpy.data.materials):
            try:
                if m.users == 0 and not m.use_fake_user:
                    bpy.data.materials.remove(m, do_unlink=True)
                    purged["materials"] += 1
            except Exception:
                continue
        for img in list(bpy.data.images):
            try:
                if img.users == 0 and not img.use_fake_user:
                    bpy.data.images.remove(img, do_unlink=True)
                    purged["images"] += 1
            except Exception:
                continue
        for me in list(bpy.data.meshes):
            try:
                if me.users == 0 and not me.use_fake_user:
                    bpy.data.meshes.remove(me, do_unlink=True)
                    purged["meshes"] += 1
            except Exception:
                continue
        for ng in list(bpy.data.node_groups):
            try:
                if ng.users == 0 and not ng.use_fake_user:
                    bpy.data.node_groups.remove(ng, do_unlink=True)
                    purged["node_groups"] += 1
            except Exception:
                continue
    except Exception:
        pass
    return purged


def _finalize_glass_materials() -> int:
    """RD ships the canopy/glass material (e.g. 'Vitres') OPAQUE, but the game renders it as
    see-through tinted glass. Detect glass by name (vitre/vitres/verre/glass), make it transparent
    (low BSDF alpha + BLEND) and normalize the name to the cooker-reserved 'Vitre'. Only an OPAQUE
    glass material has its alpha lowered, so an already-transparent one is left as-is."""
    glass_rx = re.compile(r"^(vitres?|verre|glass)", re.IGNORECASE)
    count = 0
    for mat in list(bpy.data.materials):
        if mat is None:
            continue
        bare = _norm_low(_warno_strip_namespace(str(getattr(mat, "name", ""))))
        if not glass_rx.match(bare):
            continue
        try:
            mat.use_nodes = True
            bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        except Exception:
            bsdf = None
        if bsdf is not None:
            ai = bsdf.inputs.get("Alpha")
            if ai is not None and not ai.is_linked and float(ai.default_value) >= 0.99:
                ai.default_value = 0.2  # see-through tinted glass
            ri = bsdf.inputs.get("Roughness")
            if ri is not None and not ri.is_linked:
                ri.default_value = 0.05
        for attr, val in (("blend_method", "BLEND"), ("shadow_method", "HASHED"),
                          ("show_transparent_back", True), ("use_backface_culling", False)):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
        if hasattr(mat, "surface_render_method"):
            try:
                mat.surface_render_method = "BLENDED"
            except Exception:
                pass
        if bare != "vitre":
            try:
                mat.name = "Vitre"
            except Exception:
                pass
        count += 1
    return count


def _fix_unwired_diffuse() -> int:
    """A WARNO CombinedDA/diffuse texture for a unit whose NAME contains a channel keyword
    (Alpha_Jet, Metal_*, ...) gets mis-routed by the atlas resolver: the diffuse image node ends
    up wired to BSDF Alpha while Base Color stays empty -> the model renders flat/untextured ("all
    textures present except the diffuse"). Repair at the node level: for any Principled material
    whose Base Color is UNLINKED, wire the diffuse image node (the TEX_IMAGE whose stem carries NO
    channel suffix, i.e. the bare '<Unit>.png' or CombinedDA) into Base Color. No-op for materials
    whose Base Color is already wired (RD + healthy WARNO), so it only repairs the broken case."""
    suffix_rx = re.compile(r"_(a|d|nm|n|m|o|ao|r|s|orm|os|rs|alpha|normal|metallic|roughness|occlusion|diffuse)$", re.I)
    cnt = 0
    for m in list(bpy.data.materials):
        if m is None or not getattr(m, "use_nodes", False) or m.node_tree is None:
            continue
        low = _norm_low(getattr(m, "name", ""))
        if low.startswith("vitre") or "verre" in low:
            continue  # glass intentionally has no diffuse (transparency only)
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        bc = bsdf.inputs.get("Base Color")
        if bc is None or bc.is_linked:
            continue
        diff = None
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and getattr(n, "image", None):
                stem = _norm_low(Path(str(n.image.name)).stem)
                if stem and not suffix_rx.search(stem):
                    diff = n
                    break
        if diff is not None:
            try:
                m.node_tree.links.new(diff.outputs["Color"], bc)
                cnt += 1
            except Exception:
                pass
    return cnt


def _normalize_diffuse_colorspace_and_alpha() -> int:
    """Universal (WARNO + RD) node hygiene for every Principled material:
      * the DIFFUSE (image wired into Base Color) must be sRGB;
      * every OTHER image map (normal/metallic/roughness/occlusion/alpha mask) must be Non-Color;
      * BSDF Alpha must NEVER be driven by the diffuse's own COLOR output (a 3-channel colour
        mis-wired onto 1-channel alpha — happens on keyword-named units like Alpha_Jet). Disconnect
        it so Alpha falls back to opaque. A SEPARATE alpha-mask texture, or the diffuse's true ALPHA
        channel, driving Alpha is left intact (cutout chains/tracks keep working)."""
    # A texture is a DATA map (Non-Color) iff its stem ends in a non-diffuse channel suffix; anything
    # else (bare '<Unit>.png', '<Unit>_E.png' variant, CombinedDA, '*_D'/'*_diffuse') is the diffuse.
    nondiff_rx = re.compile(r"_(a|nm|n|m|o|ao|r|s|orm|os|rs|alpha|normal|metallic|roughness|occlusion|specular|spec|mask)$", re.I)

    def _is_data_map(node):
        img = getattr(node, "image", None)
        if img is None:
            return False
        return bool(nondiff_rx.search(_norm_low(Path(str(img.name)).stem)))

    cnt = 0
    for m in list(bpy.data.materials):
        if m is None or not getattr(m, "use_nodes", False) or m.node_tree is None:
            continue
        nt = m.node_tree
        # 1. colour space by texture ROLE (from its name) — global + deterministic, no flip-flop when
        #    the same image is shared across materials: data maps -> Non-Color, diffuse -> sRGB.
        for n in nt.nodes:
            if getattr(n, "type", "") != "TEX_IMAGE" or not getattr(n, "image", None):
                continue
            want = "Non-Color" if _is_data_map(n) else "sRGB"
            try:
                cs = n.image.colorspace_settings
                if cs.name != want:
                    cs.name = want
                    cnt += 1
            except Exception:
                pass
        # 2. BSDF Alpha must NEVER be driven by a DIFFUSE texture's COLOR output (3-channel colour
        #    mis-wired onto 1-channel alpha). A real alpha-mask (*_A) texture driving Alpha stays.
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        # Cutout materials (chains / tank tracks) legitimately key their see-through gaps off a
        # texture and are individually tuned — never strip their Alpha; only the colour space above
        # is normalised for them. Glass (Vitre) is NOT cutout (it uses a flat 0.2 value) so it is
        # still cleaned below.
        m_low = _norm_low(getattr(m, "name", ""))
        if any(k in m_low for k in ("chenille", "track", "chain", "chaine", "_trk")):
            continue
        ai = bsdf.inputs.get("Alpha")
        if ai is not None and ai.is_linked:
            lk = ai.links[0]
            fn = lk.from_node
            if (getattr(fn, "type", "") == "TEX_IMAGE"
                    and getattr(lk.from_socket, "name", "") == "Color"
                    and not _is_data_map(fn)):
                try:
                    nt.links.remove(lk)
                    cnt += 1
                except Exception:
                    pass
    return cnt


def _name_chain_objects() -> int:
    """Name a chain-card object after its material instead of the generic peel name: a mesh whose
    only material is a chain material but named 'Chassis_N' / 'Part_N' becomes 'Chains', so the
    anti-RPG ball-chain curtain reads as its own object (not a chassis fragment)."""
    cnt = 0
    for o in list(bpy.data.objects):
        if getattr(o, "type", "") != "MESH":
            continue
        if not re.match(r"^(chassis|mainbody|part)_\d+$", _norm_low(getattr(o, "name", ""))):
            continue
        mats = [sl.material for sl in getattr(o, "material_slots", []) if sl.material]
        if mats and all("chain" in str(getattr(m, "name", "")).lower() for m in mats):
            try:
                o.name = "Chains"
                cnt += 1
            except Exception:
                pass
    return cnt


def _ensure_window_material_name(obj: bpy.types.Object, asset_hint: str = "") -> bool:
    if obj is None or obj.type != "MESH":
        return False
    if _norm_low(getattr(obj, "name", "")) != "window":
        return False
    namespace = _warno_asset_namespace(asset_hint)
    target_name = _warno_namespaced_material_name(namespace, "Vitre")
    slots = list(getattr(obj, "material_slots", []) or [])
    src = getattr(slots[0], "material", None) if slots else None
    # Look up by the namespaced name so that distinct units never collide on
    # a shared "Vitre" datablock. If a legacy un-namespaced "Vitre" exists in
    # the .blend, leave it untouched — orphan_purge at the next import start
    # will remove it once nothing references it.
    target = bpy.data.materials.get(target_name)
    if target is None:
        if src is not None:
            target = src.copy()
            target.name = target_name
        else:
            target = bpy.data.materials.new(name=target_name)
    if src is None:
        try:
            target.use_nodes = True
        except Exception:
            pass
        for attr, value in (
            ("blend_method", "BLEND"),
            ("shadow_method", "HASHED"),
            ("use_backface_culling", False),
            ("show_transparent_back", True),
        ):
            if hasattr(target, attr):
                try:
                    setattr(target, attr, value)
                except Exception:
                    pass
    elif src is not None and target != src:
        try:
            target.use_nodes = bool(getattr(src, "use_nodes", False))
        except Exception:
            pass
        if getattr(src, "node_tree", None) is not None and getattr(target, "node_tree", None) is not None:
            try:
                target.node_tree = src.node_tree.copy()
            except Exception:
                pass
        for attr in ("blend_method", "shadow_method", "use_backface_culling", "show_transparent_back"):
            if hasattr(src, attr) and hasattr(target, attr):
                try:
                    setattr(target, attr, getattr(src, attr))
                except Exception:
                    pass
    changed = False
    if getattr(obj, "data", None) is not None and hasattr(obj.data, "materials") and len(obj.data.materials) == 0:
        try:
            obj.data.materials.append(target)
            changed = True
        except Exception:
            pass
    slots = list(getattr(obj, "material_slots", []) or [])
    for slot in slots:
        if getattr(slot, "material", None) != target:
            slot.material = target
            changed = True
    return changed


def _set_object_origin_world(obj: bpy.types.Object, world_point: Vector) -> None:
    if obj.type != "MESH":
        return
    mesh = obj.data
    mw = obj.matrix_world.copy()
    local = mw.inverted() @ world_point
    mesh.transform(Matrix.Translation(-local))
    obj.matrix_world.translation = world_point


def _fix_outlier_wheels(imported_objects: Sequence[bpy.types.Object], _fix_collection=None) -> int:
    """Snap any wheel that sits far from its own Roue_Elev_<side><n> elevator empty back onto it.
    A wheel can be flung to the wrong side by a duplicate/bogus bone (Leopard_1V Roue_D1) or a
    whole-side source mirror defect (m48a5_mid: 7 of 9 "D" wheels on the +Y/left side). The
    elevator empty is the reliable reference — it's placed from the resolver + the
    resolved-positions mirror-fix. PRIMARY: for each wheel, if its centre is >1.0m from its own
    elevator (and that elevator is consistent with its side's elevators, i.e. not itself an
    outlier), translate the wheel so its centre lands on the elevator. FALLBACK (no elevator):
    mirror a far same-side Y-outlier from its opposite-side counterpart. Correctly-placed wheels
    (centre already on their elevator) are never moved -> safe for every normal unit."""
    import statistics
    # Force parent transforms to propagate first — right after the armature build a parented
    # wheel's matrix_world is still stale, so the displacement wouldn't be seen.
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    def _ctr(o: bpy.types.Object) -> Vector:
        wb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        return Vector((sum(p.x for p in wb) / 8.0, sum(p.y for p in wb) / 8.0, sum(p.z for p in wb) / 8.0))
    wheels: Dict[Tuple[str, int], bpy.types.Object] = {}
    elevs: Dict[Tuple[str, int], Vector] = {}
    # Roue_Elev_* are EMPTIES created inside _build_helper_armature and linked to the import
    # collection, NOT present in imported_objects (the mesh buckets) — gather them from the
    # collection so the elevator reference is available.
    elev_pool = list(getattr(_fix_collection, "all_objects", []) or []) if _fix_collection is not None else list(imported_objects)
    for o in imported_objects:
        nl = _norm_low(getattr(o, "name", ""))
        m = re.fullmatch(r"roue_([dg])(\d+)", nl)
        if m and o.type == "MESH":
            wheels[(m.group(1), int(m.group(2)))] = o
    for o in elev_pool:
        me = re.fullmatch(r"roue_elev_([dg])(\d+)", _norm_low(getattr(o, "name", "")))
        if me:
            elevs[(me.group(1), int(me.group(2)))] = o.matrix_world.translation.copy()
    elev_med: Dict[str, float] = {}
    for side in ("d", "g"):
        ys = [p.y for (sd, _n), p in elevs.items() if sd == side]
        if len(ys) >= 2:
            elev_med[side] = statistics.median(ys)
    fixed = 0
    snapped: set = set()
    # PRIMARY: snap each displaced wheel onto its own (trustworthy) elevator.
    for (side, num), o in list(wheels.items()):
        elev = elevs.get((side, num))
        if elev is None:
            continue
        emed = elev_med.get(side)
        if emed is not None and abs(elev.y - emed) > 1.0:
            continue  # elevator itself is an outlier -> don't trust it as a target
        c = _ctr(o)
        if (c - elev).length <= 1.0:
            continue  # wheel already sits on its elevator
        try:
            mw = o.matrix_world.copy()
            mw.translation = mw.translation + (elev - c)
            o.matrix_world = mw
            fixed += 1
            snapped.add((side, num))
        except Exception:
            continue
    # FALLBACK: opposite-side mirror for wheels that have no usable elevator.
    medians: Dict[str, float] = {}
    for side in ("d", "g"):
        ys = [_ctr(o).y for (sd, _n), o in wheels.items() if sd == side]
        if len(ys) >= 3:
            medians[side] = statistics.median(ys)
    for (side, num), o in list(wheels.items()):
        if (side, num) in snapped or (side, num) in elevs:
            continue
        med = medians.get(side)
        if med is None:
            continue
        c = _ctr(o)
        if abs(c.y - med) <= 1.0:
            continue
        opp = wheels.get(("g" if side == "d" else "d", num))
        omed = medians.get("g" if side == "d" else "d")
        if opp is None or omed is None:
            continue
        oc = _ctr(opp)
        if abs(oc.y - omed) > 1.0:
            continue
        target = Vector((oc.x, -oc.y, oc.z))
        try:
            mw = o.matrix_world.copy()
            mw.translation = mw.translation + (target - c)
            o.matrix_world = mw
            fixed += 1
        except Exception:
            continue
    return fixed


def _fix_flyaway_rotors(collection) -> int:
    """A rotor/propeller (Helice/Rotor/Pale) whose geometry decodes FAR from its object origin
    (the bone pivot failed, so the object sits at the unit but the mesh is ~90 m away, e.g.
    mil_mi-2d_przetacznik) flies off and blows up the scene bounds (the heli renders as a dot).
    Re-center such a far-outlier rotor's geometry onto its own origin so it sits back at the unit."""
    objs = list(getattr(collection, "all_objects", []) or [])
    meshes = [o for o in objs if getattr(o, "type", "") == "MESH" and o.data is not None and len(o.data.vertices) > 0]
    rotor_rx = re.compile(r"(helice|rotor|\bpale|propeller|rotopale)", re.IGNORECASE)
    rotors = [o for o in meshes if rotor_rx.search(getattr(o, "name", ""))]
    body = [o for o in meshes if o not in rotors]
    if not rotors or not body:
        return 0

    def _wcenter(o):
        c = Vector((0.0, 0.0, 0.0))
        mw = o.matrix_world
        for v in o.data.vertices:
            c += mw @ v.co
        return c / len(o.data.vertices)

    # Body centroid + spread from ACTUAL world vertices (object bound_box is unreliable here).
    bcen = Vector((0.0, 0.0, 0.0)); npts = 0
    for o in body:
        mw = o.matrix_world
        for v in o.data.vertices:
            bcen += mw @ v.co; npts += 1
    bcen /= max(1, npts)
    spread = 0.0
    for o in body:
        mw = o.matrix_world
        step = max(1, len(o.data.vertices) // 200)
        for i in range(0, len(o.data.vertices), step):
            spread = max(spread, (mw @ o.data.vertices[i].co - bcen).length)
    thresh = 2.0 * spread + 10.0  # a real rotor sits within the unit; >this = flown off

    cnt = 0
    for o in rotors:
        if (_wcenter(o) - bcen).length <= thresh:
            continue
        lc = Vector((0.0, 0.0, 0.0))
        for v in o.data.vertices:
            lc += v.co
        lc /= len(o.data.vertices)
        for v in o.data.vertices:
            v.co = v.co - lc
        o.data.update()
        cnt += 1
    return cnt


def _align_coaxial_rotors(collection) -> int:
    """Coaxial helicopters (Ka-50/52/27 ...) carry TWO large flat HORIZONTAL rotor discs that must
    be stacked on ONE mast (same X,Y, different Z). When the bone pivots decode to different X,Y the
    two blade-sets fan from offset centres -> a messy crisscross (Ka_50_AA). Snap the coaxial discs
    to their common X,Y centroid, keeping each disc's own Z. Single-rotor helis (one big disc + a
    small tail rotor) have <2 large horizontal discs, so they are never touched."""
    objs = list(getattr(collection, "all_objects", []) or [])
    meshes = [o for o in objs if getattr(o, "type", "") == "MESH" and o.data is not None and len(o.data.vertices) > 0]
    rotor_rx = re.compile(r"(helice|rotor|\bpale|propeller|rotopale)", re.IGNORECASE)
    discs = []
    for o in meshes:
        if not rotor_rx.search(getattr(o, "name", "")):
            continue
        pts = [o.matrix_world @ v.co for v in o.data.vertices]
        mnv = Vector((min(p[i] for p in pts) for i in range(3)))
        mxv = Vector((max(p[i] for p in pts) for i in range(3)))
        d = mxv - mnv
        # large flat HORIZONTAL disc: Z is the thin axis, both X and Y span a real rotor
        if d.z < 0.30 * min(d.x, d.y) and d.x > 5.0 and d.y > 5.0:
            cen = Vector((0.0, 0.0, 0.0))
            for p in pts:
                cen += p
            cen /= len(pts)
            discs.append((o, cen))
    if len(discs) < 2:
        return 0
    avg_x = sum(c.x for _, c in discs) / len(discs)
    avg_y = sum(c.y for _, c in discs) / len(discs)
    cnt = 0
    for o, cen in discs:
        delta = Vector((avg_x - cen.x, avg_y - cen.y, 0.0))
        if delta.length < 0.05:
            continue
        local_delta = o.matrix_world.to_3x3().inverted() @ delta
        for v in o.data.vertices:
            v.co = v.co + local_delta
        o.data.update()
        cnt += 1
    return cnt


def _split_and_weight_merged_track(imported_objects, collection) -> int:
    """Normalize track belts so Gauche=LEFT, Droite=RIGHT always match the Roue_G/Roue_D wheels,
    each belt is weighted to its own side's wheels, bound to its side armature, and carries a
    side-named material. Two cases:
      * WARNO merged belt (one mesh 'Chenille_Droite' with verts on BOTH +Y and -Y, the other
        side just an empty): split by world-Y into Chenille_Droite/Chenille_Gauche.
      * RD / already-split single-side belt: relabel it when its Droite/Gauche name disagrees with
        the wheel side (RD's geometric belt split is backwards vs the wheels on most units, e.g.
        hafiz/loggim/olifant/telak, but correct on a few like sexton), and weight it if it has none.
    The RIGHT side is read from the wheels, never a hardcoded Y sign."""
    import bmesh
    pool = list(getattr(collection, "all_objects", []) or []) if collection is not None else list(imported_objects)
    elevs: Dict[str, List[Tuple[str, Vector]]] = {"d": [], "g": []}
    arms: Dict[str, Any] = {"d": None, "g": None}
    empties: Dict[str, Any] = {}
    for o in pool:
        nl = _norm_low(getattr(o, "name", ""))
        me = re.fullmatch(r"roue_elev_([dg])(\d+)", nl)
        if me:
            elevs[me.group(1)].append((o.name, o.matrix_world.translation.copy()))
        ma = re.fullmatch(r"armature_([dg])\d", nl)
        if ma and getattr(o, "type", "") == "ARMATURE" and arms[ma.group(1)] is None:
            arms[ma.group(1)] = o
        mt = re.fullmatch(r"chenille_(droite|gauche)(_\d+)?", nl)
        if mt and getattr(o, "type", "") == "EMPTY":
            empties[mt.group(1)] = o

    # Which world-Y sign is the RIGHT (Droite) side? Read it from the wheels (Roue_D side), not a
    # fixed convention. Fall back to the right-hand rule (Droite = -Y) only with no elevators.
    dY = (sum(float(p.y) for _, p in elevs["d"]) / len(elevs["d"])) if elevs["d"] else None
    gY = (sum(float(p.y) for _, p in elevs["g"]) / len(elevs["g"])) if elevs["g"] else None
    if dY is not None and gY is not None:
        d_is_neg = dY < gY
    elif dY is not None:
        d_is_neg = dY < 0.0
    elif gY is not None:
        d_is_neg = gY > 0.0
    else:
        d_is_neg = True

    def _side_for_meanY(my):
        return "d" if ((my < 0.0) == d_is_neg) else "g"

    track_meshes = [o for o in pool
                    if getattr(o, "type", "") == "MESH" and o.data is not None
                    and re.match(r"chenille_(droite|gauche)", _norm_low(getattr(o, "name", "")))]

    def _delete_side(mesh, mwx, keep_neg):
        bm = bmesh.new(); bm.from_mesh(mesh)
        kill = [f for f in bm.faces
                if ((sum((mwx @ v.co).y for v in f.verts) / max(1, len(f.verts))) < 0) != keep_neg]
        if kill:
            bmesh.ops.delete(bm, geom=kill, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(mesh); bm.free(); mesh.update()

    def _track_maxw(t):
        vl = t.data.vertices
        mw = 0.0
        for vi in range(0, len(vl), max(1, len(vl) // 100)):
            for g in vl[vi].groups:
                if g.weight > mw:
                    mw = g.weight
        return mw

    def _weight_bind(obj, side):
        ev = elevs.get(side, [])
        if not ev or not obj.data.vertices:
            return
        for g in list(obj.vertex_groups):
            obj.vertex_groups.remove(g)
        vg: Dict[str, Any] = {}
        mw2 = obj.matrix_world
        for v in obj.data.vertices:
            wx = (mw2 @ v.co).x
            best = min(ev, key=lambda e: abs(float(e[1].x) - wx))
            grp = vg.get(best[0])
            if grp is None:
                grp = obj.vertex_groups.new(name=best[0]); vg[best[0]] = grp
            grp.add([v.index], 1.0, "REPLACE")
        arm = arms.get(side)
        if arm is not None:
            # The side carrier may have been built with use_deform=False (a merged belt carrying
            # only one side's vgroups -> the other side's wheels were non-deform), which makes the
            # modifier silently do nothing. Force the bones this belt rides deforming.
            arm_bones = getattr(getattr(arm, "data", None), "bones", None)
            if arm_bones is not None:
                for bname in vg.keys():
                    b = arm_bones.get(bname)
                    if b is not None:
                        b.use_deform = True
            # Bind EXACTLY this side's armature (replace a wrong/old one left from a swapped belt).
            existing = [m for m in obj.modifiers if m.type == "ARMATURE"]
            if not (len(existing) == 1 and existing[0].object is arm):
                for m in existing:
                    obj.modifiers.remove(m)
                m = obj.modifiers.new(name="Armature", type="ARMATURE"); m.object = arm

    def _ensure_side_material(half, want, other, base_asset):
        hd = getattr(half, "data", None)
        hmats = getattr(hd, "materials", None)
        if not hmats or not any(m is not None for m in hmats):
            return
        src = next(m for m in hmats if m is not None)
        if want.lower() in str(src.name).lower():
            return  # already the right side
        target = (str(src.name).replace(other, want)
                  .replace(other.lower(), want.lower())
                  .replace(other.upper(), want.upper()))
        if target == str(src.name):
            _ns = _warno_asset_namespace(base_asset) if base_asset else ""
            target = (_ns + _WARNO_MATERIAL_NS_SEP + "Chenille_" + want) if _ns else ("Chenille_" + want)
        m = bpy.data.materials.get(target)
        if m is None:
            m = src.copy()
            try:
                m.name = target
            except Exception:
                pass
            if base_asset:
                m["warno_asset"] = base_asset
        for i in range(len(hmats)):
            if hmats[i] is not None:
                hmats[i] = m

    done = 0
    single_side = []
    for o in track_meshes:
        mw = o.matrix_world.copy()
        ys = [(mw @ v.co).y for v in o.data.vertices]
        if not ys:
            continue
        nL = sum(1 for y in ys if y > 0.4)
        nR = sum(1 for y in ys if y < -0.4)
        if min(nL, nR) < 0.2 * max(1, nL + nR):
            single_side.append(o)
            continue
        # Split geometry by world-Y: o keeps Y<0 (neg_half), a fresh copy keeps Y>0 (pos_half).
        if o.data.users > 1:
            o.data = o.data.copy()
        md = o.data.copy()
        gauche = bpy.data.objects.new("__trackG", md)
        gauche.matrix_world = mw.copy()
        collection.objects.link(gauche)
        _delete_side(gauche.data, mw, keep_neg=False)   # keep Y>0
        _delete_side(o.data, mw, keep_neg=True)          # keep Y<0
        neg_half, pos_half = o, gauche
        droite_half = neg_half if d_is_neg else pos_half
        gauche_half = pos_half if d_is_neg else neg_half
        _weight_bind(droite_half, "d")
        _weight_bind(gauche_half, "g")
        base_asset = o.get("warno_asset", "") if hasattr(o, "get") else ""
        _ensure_side_material(droite_half, "Droite", "Gauche", base_asset)
        _ensure_side_material(gauche_half, "Gauche", "Droite", base_asset)
        # The left belt replaces the unused Chenille_Gauche empty (remove it if childless).
        ge = empties.get("gauche")
        if ge is not None:
            try:
                if not list(getattr(ge, "children", []) or []):
                    bpy.data.objects.remove(ge, do_unlink=True)
                else:
                    ge.name = ge.name + "_bone"
            except Exception:
                pass
        droite_half.name = "Chenille_Droite_01"
        gauche_half.name = "Chenille_Gauche_01"
        droite_half["warno_is_track_mesh"] = True
        gauche_half["warno_is_track_mesh"] = True
        if base_asset:
            droite_half["warno_asset"] = base_asset
            gauche_half["warno_asset"] = base_asset
        try:
            imported_objects.append(gauche)
        except Exception:
            pass
        done += 1

    # Single-side belts. First weight any that have NO weights yet (m48a5/sexton), by their
    # existing name side. Then fix the systematic RD mislabel ONLY in the clean case: exactly one
    # Droite + one Gauche belt, on OPPOSITE Y sides, with the Droite-named belt sitting on the
    # Roue_G side. Degenerate layouts (both belts on the same side, e.g. sexton) are left ALONE so
    # we never collapse two belts into one name.
    named: Dict[str, Tuple[Any, float]] = {}
    for t in single_side:
        ys = [(t.matrix_world @ v.co).y for v in t.data.vertices]
        if not ys:
            continue
        my = sum(ys) / len(ys)
        nlow = _norm_low(getattr(t, "name", ""))
        key = "droite" if "droite" in nlow else ("gauche" if "gauche" in nlow else "")
        if key and key not in named:
            named[key] = (t, my)
        if key and _track_maxw(t) < 0.01:
            _weight_bind(t, "d" if key == "droite" else "g")
            done += 1
    if "droite" in named and "gauche" in named:
        td, myd = named["droite"]
        tg, myg = named["gauche"]
        if (myd < 0.0) != (myg < 0.0) and _side_for_meanY(myd) == "g":
            # Clean opposite-side swap: exchange Chenille_Droite/Chenille_Gauche (name + material
            # + weights + armature) so each belt matches its wheels. Temp-rename first.
            for i, t in enumerate((td, tg)):
                try:
                    t.name = "__rdtrk_%d" % i
                except Exception:
                    pass
            for t, side in ((td, "g"), (tg, "d")):
                ba = t.get("warno_asset", "") if hasattr(t, "get") else ""
                _weight_bind(t, side)
                if side == "d":
                    _ensure_side_material(t, "Droite", "Gauche", ba)
                else:
                    _ensure_side_material(t, "Gauche", "Droite", ba)
                t.name = "Chenille_Droite_01" if side == "d" else "Chenille_Gauche_01"
                t["warno_is_track_mesh"] = True
                done += 1
    return done


def _build_helper_armature(
    imported_objects: Sequence[bpy.types.Object],
    bone_payload: dict[str, Any],
    collection: bpy.types.Collection,
    settings: WARNOImporterSettings | None = None,
) -> bpy.types.Object | None:
    asset_hint = ""
    for obj in imported_objects:
        try:
            asset_hint = _norm_low(str(obj.get("warno_asset", "") or ""))
        except Exception:
            asset_hint = ""
        if asset_hint:
            break

    bone_name_by_index_raw = bone_payload.get("bone_name_by_index", {}) or {}
    raw_bone_name_by_index_raw = bone_payload.get("raw_bone_name_by_index", {}) or {}
    if not isinstance(bone_name_by_index_raw, dict) or not bone_name_by_index_raw:
        return None
    bone_name_by_index: Dict[int, str] = {}
    for k, v in bone_name_by_index_raw.items():
        try:
            idx = int(k)
        except Exception:
            continue
        name = str(v or "").strip()
        if not name:
            continue
        bone_name_by_index[idx] = name
    if not bone_name_by_index:
        return None
    display_name_by_index: Dict[int, str] = dict(bone_name_by_index)
    if isinstance(raw_bone_name_by_index_raw, dict):
        for k, v in raw_bone_name_by_index_raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            name = str(v or "").strip()
            if not name:
                continue
            display_name_by_index[int(idx)] = name
    display_name_values_low = {
        _norm_low(str(name))
        for name in display_name_by_index.values()
        if str(name).strip()
    }
    papyrus_character_proxy = (
        "papyrus" in display_name_values_low
        and "soldat" in display_name_values_low
        and any(name.startswith("bip01") for name in display_name_values_low)
    )
    papyrus_root_index = next(
        (int(idx) for idx, name in display_name_by_index.items() if _norm_low(str(name)) == "papyrus"),
        -1,
    )
    soldat_root_index = next(
        (int(idx) for idx, name in display_name_by_index.items() if _norm_low(str(name)) == "soldat"),
        -1,
    )
    character_bip_indices: set[int] = {
        int(idx)
        for idx, name in display_name_by_index.items()
        if _norm_low(str(name)).startswith("bip01")
    }
    bone_parent_by_index_raw = bone_payload.get("bone_parent_by_index", {}) or {}
    bone_parent_by_index: Dict[int, int] = {}
    if isinstance(bone_parent_by_index_raw, dict):
        for k, v in bone_parent_by_index_raw.items():
            try:
                bone_parent_by_index[int(k)] = int(v)
            except Exception:
                continue

    _scene_center, diag = _mesh_bounds(imported_objects)
    bone_positions = bone_payload.get("bone_positions", {}) or {}
    off_mat_points_raw = bone_payload.get("off_mat_points_by_index", {}) or {}
    off_mat_blocks_raw = bone_payload.get("off_mat_blocks_by_index", {}) or {}
    raw_scene_graph = bone_payload.get("raw_scene_graph")
    raw_scene_record_by_index: Dict[int, Any] = {}
    if raw_scene_graph is not None:
        for record in list(getattr(raw_scene_graph, "records", []) or []):
            try:
                raw_scene_record_by_index[int(getattr(record, "index", -1))] = record
            except Exception:
                continue
    use_deterministic_raw_scene = raw_scene_graph is not None
    # The parent-anchor / fallen-empty fallbacks below are an RD/SD2-only net (the IRISZoom
    # raw scene leaves a few helpers at origin). use_deterministic_raw_scene is NOT RD-specific
    # — WARNO scenes also carry a raw_scene_graph — so without this extra game gate those
    # fallbacks fired on WARNO too and REGRESSED it: smoke empties (Fx_Fumee_Chenille_D1/G1)
    # collapsed onto their shared parent, and wheels/empties jumped to the parent anchor
    # (Leopard_1A1/Leopard_1V). Gate RD-only so WARNO's empty + wheel placement is byte-for-byte
    # as before (these blocks are purely additive; off for WARNO == the original code path).
    _armature_is_rd = bool(
        settings is not None
        and str(getattr(settings, "game_preset", "") or "").upper() in ("WARGAME_RD", "STEEL_DIVISION_2")
    )
    ordered_indices = sorted(int(i) for i in bone_name_by_index.keys())
    gfx_track_kind = str(bone_payload.get("gfx_track_kind", "") or "").strip().lower()
    gfx_manifest_source = str(bone_payload.get("gfx_manifest_source", "none") or "none").strip() or "none"
    semantic_mode = str(getattr(settings, "import_semantic_mode", "REFERENCE") or "REFERENCE").strip().upper()
    if semantic_mode not in {"REFERENCE", "RAW_DEBUG"}:
        semantic_mode = "REFERENCE"
    gfx_required_nodes = {
        _norm_low(str(x))
        for x in (bone_payload.get("gfx_required_nodes", []) or [])
        if str(x).strip()
    }
    gfx_fx_nodes = {
        _norm_low(str(x))
        for x in (bone_payload.get("gfx_fx_nodes", []) or [])
        if str(x).strip()
    }
    gfx_subdepiction_nodes = {
        _norm_low(str(x))
        for x in (bone_payload.get("gfx_subdepiction_nodes", []) or [])
        if str(x).strip()
    }
    gfx_operator_contracts = [
        row
        for row in (bone_payload.get("gfx_operator_contracts", []) or [])
        if isinstance(row, dict)
    ]
    gfx_semantic_nodes_raw = [
        row
        for row in (bone_payload.get("gfx_semantic_nodes", []) or [])
        if isinstance(row, dict)
    ]
    gfx_role_map_raw = bone_payload.get("gfx_role_map", {}) or {}
    gfx_role_map: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(gfx_role_map_raw, dict):
        for key, rows in gfx_role_map_raw.items():
            low = _norm_low(str(key))
            if not low:
                continue
            if isinstance(rows, list):
                clean_rows = [dict(row) for row in rows if isinstance(row, dict)]
                if clean_rows:
                    gfx_role_map[low] = clean_rows
    role_priority = {
        "subdepiction_anchor": 0,
        "weapon_fx_anchor": 1,
        "turret_recoil_node": 2,
        "turret_pitch_node": 3,
        "turret_yaw_node": 4,
        "movement_propulsion_node": 5,
    }
    primary_role_by_index: Dict[int, str] = {}
    raw_index_by_name: Dict[str, int] = {}
    for bidx in ordered_indices:
        raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "") or ""))
        if raw_low and raw_low not in raw_index_by_name:
            raw_index_by_name[raw_low] = int(bidx)
    preserved_runtime_fx_nodes = {
        raw_low
        for raw_low in raw_index_by_name.keys()
        if _is_preserved_runtime_fx_anchor(raw_low)
    }
    gfx_semantic_nodes_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in gfx_semantic_nodes_raw:
        name_low = _norm_low(str(row.get("name", "") or ""))
        source_kind = _norm_low(str(row.get("source_kind", "") or ""))
        if not name_low or source_kind not in {"gfx_exact", "spk_exact"}:
            continue
        parent_low = _norm_low(str(row.get("parent_name", "") or ""))
        role_low = _norm_low(str(row.get("role", "") or ""))
        local_translation = row.get("local_translation", []) or []
        local_rotation_basis = row.get("local_rotation_basis", []) or []
        local_scale = row.get("local_scale", []) or []
        try:
            clean_translation = [
                float(local_translation[0]),
                float(local_translation[1]),
                float(local_translation[2]),
            ]
        except Exception:
            continue
        clean_basis: List[List[float]] = []
        if isinstance(local_rotation_basis, (list, tuple)) and len(local_rotation_basis) >= 3:
            try:
                clean_basis = [
                    [
                        float(local_rotation_basis[0][0]),
                        float(local_rotation_basis[0][1]),
                        float(local_rotation_basis[0][2]),
                    ],
                    [
                        float(local_rotation_basis[1][0]),
                        float(local_rotation_basis[1][1]),
                        float(local_rotation_basis[1][2]),
                    ],
                    [
                        float(local_rotation_basis[2][0]),
                        float(local_rotation_basis[2][1]),
                        float(local_rotation_basis[2][2]),
                    ],
                ]
            except Exception:
                clean_basis = []
        clean_scale = [1.0, 1.0, 1.0]
        if isinstance(local_scale, (list, tuple)) and len(local_scale) >= 3:
            try:
                clean_scale = [
                    float(local_scale[0]),
                    float(local_scale[1]),
                    float(local_scale[2]),
                ]
            except Exception:
                clean_scale = [1.0, 1.0, 1.0]
        gfx_semantic_nodes_by_name[name_low].append(
            {
                "name": name_low,
                "parent_name": parent_low,
                "role": role_low,
                "source_kind": source_kind,
                "local_translation": clean_translation,
                "local_rotation_basis": clean_basis,
                "local_scale": clean_scale,
                "provenance": dict(row.get("provenance", {}) or {}),
            }
        )
    for raw_low, rows in gfx_role_map.items():
        bidx = raw_index_by_name.get(str(raw_low))
        if bidx is None:
            continue
        best_role = ""
        best_rank = 999
        for row in rows:
            role = _norm_low(str(row.get("role", "")))
            rank = int(role_priority.get(role, 999))
            if rank < best_rank:
                best_rank = rank
            best_role = role
        if best_role:
            primary_role_by_index[int(bidx)] = best_role
    semantic_scene_nodes = set(gfx_required_nodes | gfx_fx_nodes | gfx_subdepiction_nodes)
    semantic_support_nodes: set[str] = set()
    for semantic_low in sorted(semantic_scene_nodes):
        semantic_idx = raw_index_by_name.get(str(semantic_low))
        if semantic_idx is None:
            continue
        cur = int(bone_parent_by_index.get(int(semantic_idx), -1))
        while cur >= 0:
            cur_low = _norm_low(str(bone_name_by_index.get(int(cur), "")))
            if not cur_low:
                break
            if cur_low in {"armature", "papyrus", "fake"} or cur_low.startswith("armature_"):
                break
            if cur_low in semantic_scene_nodes:
                break
            semantic_support_nodes.add(cur_low)
            cur = int(bone_parent_by_index.get(int(cur), -1))
    wheel_pattern = re.compile(r"^roue_elev_([dg])([0-9]+)$", re.IGNORECASE)
    geometry_bone_indices: set[int] = set()
    for obj in imported_objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        try:
            bidx = int(obj.get("warno_group_bone_index", -1))
        except Exception:
            bidx = -1
        if bidx >= 0:
            geometry_bone_indices.add(int(bidx))

    def _defer_semantic_helper_resolution(idx: int) -> bool:
        bidx = int(idx)
        raw_low = _norm_low(str(bone_name_by_index.get(bidx, "") or ""))
        if not raw_low:
            return False
        if bidx in geometry_bone_indices:
            return False
        if raw_low in {"armature", "papyrus", "fake"}:
            return False
        if raw_low.startswith("armature_"):
            return False
        if wheel_pattern.fullmatch(raw_low):
            return False
        if raw_low in semantic_scene_nodes or raw_low in semantic_support_nodes:
            return True
        return False

    bone_len = float(WARNO_HELPER_BONE_LENGTH)

    weighted_positions_by_index: Dict[int, Vector] = {}
    if not use_deterministic_raw_scene:
        for bidx in ordered_indices:
            raw = str(bone_name_by_index.get(int(bidx), "")).strip()
            if not raw:
                continue
            pos = _pick_position_from_payload(raw, bone_positions)
            if pos is None:
                continue
            weighted_positions_by_index[int(bidx)] = pos

    child_by_parent: Dict[int, List[int]] = {}
    for child_idx, parent_idx_raw in bone_parent_by_index.items():
        child = int(child_idx)
        parent = int(parent_idx_raw)
        if child not in bone_name_by_index:
            continue
        child_by_parent.setdefault(parent, []).append(child)

    resolved_positions: Dict[int, Vector] = (
        {}
        if use_deterministic_raw_scene
        else {int(k): v.copy() for k, v in weighted_positions_by_index.items()}
    )
    pending: set[int] = {int(i) for i in ordered_indices if int(i) not in resolved_positions}
    fallback_from_children = 0
    fallback_from_parent = 0
    fallback_line_placed = 0
    missing_parent_anchor = 0

    if not use_deterministic_raw_scene:
        # Fallback A: pull position from already-resolved children.
        for _ in range(max(1, len(pending) + 1)):
            if not pending:
                break
            changed = False
            for bidx in list(pending):
                if _defer_semantic_helper_resolution(int(bidx)):
                    continue
                child_positions: List[Vector] = []
                for ch in child_by_parent.get(int(bidx), []):
                    pos = resolved_positions.get(int(ch))
                    if pos is not None:
                        child_positions.append(pos)
                if not child_positions:
                    continue
                acc = Vector((0.0, 0.0, 0.0))
                for pos in child_positions:
                    acc += pos
                resolved_positions[int(bidx)] = acc / float(len(child_positions))
                pending.remove(int(bidx))
                fallback_from_children += 1
                changed = True
            if not changed:
                break

        # Fallback B: parent anchor + deterministic local offset.
        def _parent_depth(idx: int) -> int:
            depth = 0
            cur = int(idx)
            seen: set[int] = set()
            while True:
                if cur in seen:
                    break
                seen.add(cur)
                p = int(bone_parent_by_index.get(cur, -1))
                if p < 0:
                    break
                depth += 1
                cur = p
                if depth > 256:
                    break
            return depth

        for _ in range(max(1, len(pending) + 1)):
            if not pending:
                break
            changed = False
            for bidx in sorted(list(pending)):
                if _defer_semantic_helper_resolution(int(bidx)):
                    continue
                parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
                parent_pos = resolved_positions.get(parent_idx)
                if parent_pos is None:
                    continue
                depth = _parent_depth(int(bidx))
                name_low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
                side_sign = 0.0
                if re.search(r"(?:^|_)g[0-9]+(?:_|$)", name_low) or "gauche" in name_low:
                    side_sign = -1.0
                elif re.search(r"(?:^|_)d[0-9]+(?:_|$)", name_low) or "droite" in name_low:
                    side_sign = 1.0
                offset = Vector((0.003 * side_sign, 0.0, 0.008 + 0.0015 * float(depth % 5)))
                resolved_positions[int(bidx)] = parent_pos + offset
                pending.remove(int(bidx))
                fallback_from_parent += 1
                changed = True
            if not changed:
                break

        # Fallback C: deterministic parking line near roots for unresolved nodes.
        roots = [
            int(i)
            for i in ordered_indices
            if int(bone_parent_by_index.get(int(i), -1)) < 0
        ]
        root_positions = [resolved_positions[int(i)] for i in roots if int(i) in resolved_positions]
        if root_positions:
            anchor = Vector((0.0, 0.0, 0.0))
            for pos in root_positions:
                anchor += pos
            anchor = anchor / float(len(root_positions))
        elif resolved_positions:
            anchor = Vector((0.0, 0.0, 0.0))
            for pos in resolved_positions.values():
                anchor += pos
            anchor = anchor / float(len(resolved_positions))
        else:
            anchor = Vector((0.0, 0.0, 0.0))

        for line_idx, bidx in enumerate(sorted(list(pending))):
            if _defer_semantic_helper_resolution(int(bidx)):
                continue
            jitter = -1.0 if (line_idx % 2 == 0) else 1.0
            resolved_positions[int(bidx)] = anchor + Vector((0.0025 * jitter, 0.0, 0.02 + float(line_idx) * 0.004))
            pending.remove(int(bidx))
            fallback_line_placed += 1
            missing_parent_anchor += 1

    def _to_pretty_name(raw_name: str) -> str:
        return _display_warno_node_name(raw_name, asset_hint)

    def _set_parent_keep_world(child: bpy.types.Object, parent: bpy.types.Object) -> None:
        wm = child.matrix_world.copy()
        child.parent = parent
        child.parent_type = "OBJECT"
        child.parent_bone = ""
        child.matrix_parent_inverse = parent.matrix_world.inverted()
        child.matrix_world = wm

    def _set_bone_parent_keep_world(
        child: bpy.types.Object,
        armature_obj: bpy.types.Object,
        bone_name: str,
    ) -> None:
        wm = child.matrix_world.copy()
        child.parent = armature_obj
        child.parent_type = "BONE"
        child.parent_bone = str(bone_name or "")
        child.matrix_parent_inverse = armature_obj.matrix_world.inverted()
        child.matrix_world = wm

    def _character_anchor_position(idx: int) -> Vector | None:
        raw_name = str(display_name_by_index.get(int(idx), bone_name_by_index.get(int(idx), "")) or "").strip()
        if raw_name:
            pos = _pick_position_from_payload(raw_name, bone_positions)
            if isinstance(pos, Vector):
                return pos.copy()
        pos = resolved_positions.get(int(idx))
        if isinstance(pos, Vector):
            return pos.copy()
        record = raw_scene_record_by_index.get(int(idx))
        world_translation = getattr(record, "world_translation", None) if record is not None else None
        if isinstance(world_translation, (list, tuple)) and len(world_translation) >= 3:
            try:
                return Vector(
                    (
                        float(world_translation[0]),
                        float(world_translation[1]),
                        float(world_translation[2]),
                    )
                )
            except Exception:
                return None
        return None

    mesh_by_bone_index: Dict[int, List[bpy.types.Object]] = {}
    for obj in imported_objects:
        try:
            bidx = int(obj.get("warno_group_bone_index", -1))
        except Exception:
            bidx = -1
        if bidx < 0:
            continue
        mesh_by_bone_index.setdefault(int(bidx), []).append(obj)

    mesh_by_name_low: Dict[str, bpy.types.Object] = {}
    for obj in imported_objects:
        key = _norm_low(obj.name)
        if key and key not in mesh_by_name_low:
            mesh_by_name_low[key] = obj

    off_mat_points_by_index: Dict[int, List[Tuple[int, Vector]]] = {}
    for k, raw_points in (off_mat_points_raw.items() if isinstance(off_mat_points_raw, dict) else []):
        try:
            bidx = int(k)
        except Exception:
            continue
        pts_out: List[Tuple[int, Vector]] = []
        if isinstance(raw_points, (list, tuple)):
            for point_idx, raw_point in enumerate(raw_points):
                if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 3:
                    continue
                try:
                    vec = Vector((float(raw_point[0]), float(raw_point[1]), float(raw_point[2])))
                except Exception:
                    continue
                if not math.isfinite(vec.x) or not math.isfinite(vec.y) or not math.isfinite(vec.z):
                    continue
                if vec.length <= 1.0e-9:
                    continue
                pts_out.append((int(point_idx), vec))
        if pts_out:
            off_mat_points_by_index[int(bidx)] = pts_out
    off_mat_blocks_by_index: Dict[int, List[List[float]]] = {}
    for k, raw_blocks in (off_mat_blocks_raw.items() if isinstance(off_mat_blocks_raw, dict) else []):
        try:
            bidx = int(k)
        except Exception:
            continue
        out_blocks: List[List[float]] = []
        if isinstance(raw_blocks, (list, tuple)):
            for raw_block in raw_blocks:
                if not isinstance(raw_block, (list, tuple)) or len(raw_block) < 12:
                    continue
                try:
                    block_vals = [float(x) for x in raw_block[:12]]
                except Exception:
                    continue
                if not all(math.isfinite(v) for v in block_vals):
                    continue
                out_blocks.append(block_vals)
        if out_blocks:
            off_mat_blocks_by_index[int(bidx)] = out_blocks
    off_mat_global_translation_candidates: List[Dict[str, Any]] = []
    for source_idx, source_blocks in off_mat_blocks_by_index.items():
        source_name = _norm_low(str(bone_name_by_index.get(int(source_idx), "")))
        for block_idx, block_vals in enumerate(source_blocks):
            if not isinstance(block_vals, list) or len(block_vals) < 12:
                continue
            raw_variants = {
                "row_major_last_col": (
                    float(block_vals[3]),
                    float(block_vals[7]),
                    float(block_vals[11]),
                ),
                "column_major_last_col": (
                    float(block_vals[9]),
                    float(block_vals[10]),
                    float(block_vals[11]),
                ),
            }
            for basis_name, raw_xyz in raw_variants.items():
                for remap_name, world_xyz in _off_mat_blender_remaps(*raw_xyz).items():
                    try:
                        world_point = Vector((float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])))
                    except Exception:
                        continue
                    if not math.isfinite(float(world_point.x)) or not math.isfinite(float(world_point.y)) or not math.isfinite(float(world_point.z)):
                        continue
                    if float(world_point.length) <= 1.0e-9:
                        continue
                    off_mat_global_translation_candidates.append(
                        {
                            "source_idx": int(source_idx),
                            "source_name": str(source_name),
                            "block_idx": int(block_idx),
                            "variant": f"{basis_name}:{remap_name}",
                            "world": world_point.copy(),
                        }
                    )
    off_mat_stream_local_candidates_by_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    node_stream_count = len(ordered_indices)
    stream_start_idx, stream_end_idx = _off_mat_stream_section_bounds(node_stream_count)
    for source_idx in range(int(stream_start_idx), int(stream_end_idx)):
        source_blocks = off_mat_blocks_by_index.get(int(source_idx), [])
        if not source_blocks:
            continue
        source_name = _norm_low(str(bone_name_by_index.get(int(source_idx), "")))
        for block_idx, block_vals in enumerate(source_blocks):
            raw_matrix = _off_mat_block_matrix(block_vals, "row")
            if raw_matrix is None:
                continue
            blender_matrix = _off_mat_convert_matrix_to_blender(raw_matrix, "xyz:+1-1+1")
            if blender_matrix is None:
                continue
            decomp = _decompose_affine_components(blender_matrix)
            if decomp is None:
                continue
            local_translation, local_rotation_basis, local_scale = decomp
            if not (
                math.isfinite(float(local_translation.x))
                and math.isfinite(float(local_translation.y))
                and math.isfinite(float(local_translation.z))
            ):
                continue
            target_idx = _off_mat_stream_target_index(int(source_idx), int(block_idx), node_stream_count)
            if target_idx < 0:
                continue
            off_mat_stream_local_candidates_by_index[int(target_idx)].append(
                {
                    "source": "stream_local",
                    "source_idx": int(source_idx),
                    "source_name": str(source_name),
                    "block_idx": int(block_idx),
                    "layout": "row",
                    "remap": "xyz:+1-1+1",
                    "local_matrix": blender_matrix.copy(),
                    "local": local_translation.copy(),
                    "local_rotation_basis": [list(row_vals) for row_vals in local_rotation_basis],
                    "local_scale": [float(local_scale[0]), float(local_scale[1]), float(local_scale[2])],
                }
            )

    reserved_off_mat_points: set[Tuple[int, int]] = set()
    off_mat_chain_resolved = 0
    off_mat_pair_resolved = 0
    off_mat_fx_resolved = 0
    semantic_helper_source_by_index: Dict[int, str] = {}
    semantic_helper_reason_by_index: Dict[int, str] = {}
    support_helper_source_by_index: Dict[int, str] = {}
    support_helper_reason_by_index: Dict[int, str] = {}
    semantic_helpers_exact = 0
    semantic_helpers_approx = 0
    semantic_helpers_skipped = 0
    support_helpers_exact = 0
    support_helpers_approx = 0
    support_helpers_skipped = 0
    exact_local_transform_by_index: Dict[int, Dict[str, Any]] = {}
    resolved_world_matrix_by_index: Dict[int, Matrix] = {}
    for bidx in ordered_indices:
        mesh_nodes = mesh_by_bone_index.get(int(bidx), [])
        obj = mesh_nodes[0] if mesh_nodes else None
        if obj is not None:
            try:
                resolved_world_matrix_by_index[int(bidx)] = obj.matrix_world.copy()
                continue
            except Exception:
                pass
        pos = resolved_positions.get(int(bidx))
        if isinstance(pos, Vector):
            resolved_world_matrix_by_index[int(bidx)] = Matrix.Translation(pos.copy())

    def _manifest_exact_transform_for_index(idx: int) -> Dict[str, Any] | None:
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))
        if not raw_low:
            return None
        parent_idx = int(bone_parent_by_index.get(int(idx), -1))
        parent_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "") or ""))
        rows = gfx_semantic_nodes_by_name.get(raw_low, [])
        if not rows:
            return None
        for row in rows:
            expected_parent = _norm_low(str(row.get("parent_name", "") or ""))
            if expected_parent and expected_parent != parent_low:
                continue
            return dict(row)
        return None

    def _register_exact_local_transform(
        idx: int,
        *,
        role: str,
        source_kind: str,
        local_translation: Vector | Sequence[float] | None,
        local_rotation_basis: Sequence[Sequence[float]] | None = None,
        local_scale: Sequence[float] | None = None,
        provenance: Dict[str, Any] | None = None,
    ) -> None:
        source_kind_low = str(source_kind or "").strip().lower()
        if use_deterministic_raw_scene and source_kind_low != "spk_exact":
            return
        translation_vec = None
        if isinstance(local_translation, Vector):
            translation_vec = local_translation.copy()
        elif isinstance(local_translation, (list, tuple)) and len(local_translation) >= 3:
            try:
                translation_vec = Vector(
                    (
                        float(local_translation[0]),
                        float(local_translation[1]),
                        float(local_translation[2]),
                    )
                )
            except Exception:
                translation_vec = None
        if translation_vec is None:
            return
        if isinstance(local_rotation_basis, Matrix):
            basis_rows = [
                [float(local_rotation_basis[row][col]) for col in range(3)]
                for row in range(3)
            ]
        elif isinstance(local_rotation_basis, (list, tuple)) and len(local_rotation_basis) >= 3:
            basis_rows = [
                [float(local_rotation_basis[0][0]), float(local_rotation_basis[0][1]), float(local_rotation_basis[0][2])],
                [float(local_rotation_basis[1][0]), float(local_rotation_basis[1][1]), float(local_rotation_basis[1][2])],
                [float(local_rotation_basis[2][0]), float(local_rotation_basis[2][1]), float(local_rotation_basis[2][2])],
            ]
        else:
            basis_rows = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        if isinstance(local_scale, (list, tuple)) and len(local_scale) >= 3:
            scale_vals = [
                float(local_scale[0]),
                float(local_scale[1]),
                float(local_scale[2]),
            ]
        else:
            scale_vals = [1.0, 1.0, 1.0]
        exact_local_transform_by_index[int(idx)] = {
            "role": str(role or "").strip().lower(),
            "source_kind": source_kind_low,
            "local_translation": translation_vec.copy(),
            "local_rotation_basis": basis_rows,
            "local_scale": scale_vals,
            "provenance": dict(provenance or {}),
        }

    def _set_resolved_world_point(idx: int, world_point: Vector | None) -> None:
        if not isinstance(world_point, Vector):
            return
        resolved_positions[int(idx)] = world_point.copy()
        resolved_world_matrix_by_index[int(idx)] = Matrix.Translation(world_point.copy())

    def _set_resolved_world_from_local_matrix(idx: int, local_matrix: Matrix) -> Vector | None:
        if not isinstance(local_matrix, Matrix):
            return None
        parent_idx = int(bone_parent_by_index.get(int(idx), -1))
        parent_world = resolved_world_matrix_by_index.get(int(parent_idx))
        if isinstance(parent_world, Matrix):
            try:
                world_matrix = parent_world @ local_matrix
                resolved_world_matrix_by_index[int(idx)] = world_matrix.copy()
                world_point = world_matrix.translation.copy()
                resolved_positions[int(idx)] = world_point.copy()
                return world_point.copy()
            except Exception:
                pass
        parent_anchor = resolved_positions.get(int(parent_idx))
        if isinstance(parent_anchor, Vector):
            world_point = parent_anchor.copy() + local_matrix.translation.copy()
            resolved_world_matrix_by_index[int(idx)] = Matrix.Translation(world_point.copy())
            resolved_positions[int(idx)] = world_point.copy()
            return world_point.copy()
        return None

    def _stream_local_exact_candidate(
        idx: int,
        *,
        role_override: str = "",
        require_parent: bool = True,
    ) -> Dict[str, Any] | None:
        rows = off_mat_stream_local_candidates_by_index.get(int(idx), [])
        if not rows:
            return None
        parent_idx = int(bone_parent_by_index.get(int(idx), -1))
        if bool(require_parent) and parent_idx < 0:
            return None
        parent_anchor = resolved_positions.get(int(parent_idx))
        if bool(require_parent) and not isinstance(parent_anchor, Vector):
            return None
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "") or ""))
        role_name = str(
            role_override
            or _semantic_role_for_index(int(idx))
            or _support_role_for_index(int(idx))
            or "semantic_support"
        )
        current_pos = resolved_positions.get(int(idx))
        parent_bounds_full = _nearest_ancestor_mesh_bounds(int(idx))
        parent_bounds = None
        if parent_bounds_full is not None:
            parent_bounds = (parent_bounds_full[0].copy(), parent_bounds_full[1].copy())
        best_row: Dict[str, Any] | None = None
        best_score = float("inf")
        for row in rows:
            local_matrix = row.get("local_matrix")
            if not isinstance(local_matrix, Matrix):
                continue
            local_point = row.get("local")
            if not isinstance(local_point, Vector):
                continue
            world_point = None
            parent_world = resolved_world_matrix_by_index.get(int(parent_idx))
            if isinstance(parent_world, Matrix):
                try:
                    world_point = (parent_world @ local_matrix).translation.copy()
                except Exception:
                    world_point = None
            if world_point is None and isinstance(parent_anchor, Vector):
                world_point = parent_anchor.copy() + local_point.copy()
            if not _is_usable_world_point(world_point, parent_anchor=parent_anchor if isinstance(parent_anchor, Vector) else None):
                continue
            score = _score_semantic_candidate(
                role_name,
                raw_low,
                parent_raw_low,
                world_point.copy(),
                parent_anchor.copy() if isinstance(parent_anchor, Vector) else Vector((0.0, 0.0, 0.0)),
                parent_bounds,
                current_pos.copy() if isinstance(current_pos, Vector) else None,
                "stream_local",
                local_point.copy(),
            )
            score -= 2.25
            if score < best_score:
                best_score = float(score)
                best_row = dict(row)
                best_row["world"] = world_point.copy()
                best_row["score"] = float(score)
                best_row["role"] = str(role_name)
                best_row["exact"] = True
                best_row["confidence"] = "high"
        return best_row

    def _support_local_template_penalty(
        idx: int,
        parent_idx: int,
        local_vec: Vector | None,
    ) -> float | None:
        if not isinstance(local_vec, Vector):
            return None
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "") or ""))
        role_low = _support_role_for_index(int(idx))
        penalty = _semantic_local_template_penalty(
            role_low,
            raw_low,
            parent_raw_low,
            local_vec.copy(),
        )
        if penalty is not None:
            return float(penalty)
        if (
            role_low == "turret_yaw_node"
            and parent_raw_low == "tourelle_01"
            and raw_low.startswith("tourelle_")
        ):
            abs_x = abs(float(local_vec.x))
            abs_y = abs(float(local_vec.y))
            z = float(local_vec.z)
            if abs_x <= 1.2 and abs_y <= 0.75 and -0.2 <= z <= 1.45:
                stream_penalty = 0.35
                stream_penalty += max(0.0, abs_x - 0.25) * 0.35
                stream_penalty += abs_y * 0.25
                stream_penalty += abs(z - 0.95) * 0.35
                return float(stream_penalty)
        return None

    def _support_local_template_ok(
        idx: int,
        parent_idx: int,
        local_vec: Vector | None,
    ) -> bool:
        return _support_local_template_penalty(int(idx), int(parent_idx), local_vec) is not None

    def _support_chain_needs_stream_repair(
        tourelle_idx: int,
        axe_idx: int,
        canon_idx: int,
    ) -> bool:
        chain_rows = (
            (int(tourelle_idx), int(bone_parent_by_index.get(int(tourelle_idx), -1))),
            (int(axe_idx), int(tourelle_idx)),
            (int(canon_idx), int(axe_idx)),
        )
        for idx, parent_idx in chain_rows:
            if parent_idx < 0:
                return False
            prev_source = str(support_helper_source_by_index.get(int(idx), "") or "")
            exact_row = exact_local_transform_by_index.get(int(idx))
            local_vec = None
            if isinstance(exact_row, dict):
                local_val = exact_row.get("local_translation")
                if isinstance(local_val, Vector):
                    local_vec = local_val.copy()
                elif isinstance(local_val, (list, tuple)) and len(local_val) >= 3:
                    try:
                        local_vec = Vector((float(local_val[0]), float(local_val[1]), float(local_val[2])))
                    except Exception:
                        local_vec = None
            if local_vec is None:
                parent_anchor = resolved_positions.get(int(parent_idx))
                current_pos = resolved_positions.get(int(idx))
                if isinstance(parent_anchor, Vector) and isinstance(current_pos, Vector):
                    local_vec = current_pos.copy() - parent_anchor.copy()
            if prev_source not in {"gfx_exact", "spk_exact"}:
                return True
            if not _support_local_template_ok(int(idx), int(parent_idx), local_vec):
                return True
        return False

    def _best_stream_support_chain_solution(
        tourelle_idx: int,
        axe_idx: int,
        canon_idx: int,
    ) -> Dict[str, Any] | None:
        parent_idx = int(bone_parent_by_index.get(int(tourelle_idx), -1))
        if parent_idx < 0:
            return None
        parent_anchor = resolved_positions.get(int(parent_idx))
        if not isinstance(parent_anchor, Vector):
            return None
        parent_world = resolved_world_matrix_by_index.get(int(parent_idx))
        if not isinstance(parent_world, Matrix):
            parent_world = Matrix.Translation(parent_anchor.copy())

        def _evaluate_stream_rows(
            source_idx: int,
            t_row: Dict[str, Any],
            a_row: Dict[str, Any],
            c_row: Dict[str, Any],
        ) -> Dict[str, Any] | None:
            t_matrix = t_row.get("local_matrix")
            a_matrix = a_row.get("local_matrix")
            c_matrix = c_row.get("local_matrix")
            t_local = t_row.get("local")
            a_local = a_row.get("local")
            c_local = c_row.get("local")
            if not isinstance(t_local, Vector) or not isinstance(a_local, Vector) or not isinstance(c_local, Vector):
                return None
            if not isinstance(t_matrix, Matrix):
                t_matrix = Matrix.Translation(t_local.copy())
            if not isinstance(a_matrix, Matrix):
                a_matrix = Matrix.Translation(a_local.copy())
            if not isinstance(c_matrix, Matrix):
                c_matrix = Matrix.Translation(c_local.copy())
            try:
                t_world_matrix = parent_world @ t_matrix
                a_world_matrix = t_world_matrix @ a_matrix
                c_world_matrix = a_world_matrix @ c_matrix
            except Exception:
                return None
            if not _support_local_template_ok(int(tourelle_idx), int(parent_idx), t_local.copy()):
                return None
            if not _support_local_template_ok(int(axe_idx), int(tourelle_idx), a_local.copy()):
                return None
            if not _support_local_template_ok(int(canon_idx), int(axe_idx), c_local.copy()):
                return None
            current_t = resolved_positions.get(int(tourelle_idx))
            current_a = resolved_positions.get(int(axe_idx))
            current_c = resolved_positions.get(int(canon_idx))
            score = 0.0
            for idx, parent_for_idx, local_vec in (
                (int(tourelle_idx), int(parent_idx), t_local.copy()),
                (int(axe_idx), int(tourelle_idx), a_local.copy()),
                (int(canon_idx), int(axe_idx), c_local.copy()),
            ):
                penalty = _support_local_template_penalty(
                    int(idx),
                    int(parent_for_idx),
                    local_vec.copy(),
                )
                if penalty is None:
                    return None
                score += float(penalty)
            if int(t_row.get("block_idx", -1)) == 0:
                score -= 0.35
            if int(a_row.get("block_idx", -1)) == 1:
                score -= 0.35
            if int(c_row.get("block_idx", -1)) == 2:
                score -= 0.35
            if isinstance(current_t, Vector):
                score += float((t_world_matrix.translation.copy() - current_t.copy()).length) * 0.02
            if isinstance(current_a, Vector):
                score += float((a_world_matrix.translation.copy() - current_a.copy()).length) * 0.02
            if isinstance(current_c, Vector):
                score += float((c_world_matrix.translation.copy() - current_c.copy()).length) * 0.02
            return {
                "source_idx": int(source_idx),
                "score": float(score),
                "tourelle": {
                    **dict(t_row),
                    "world_matrix": t_world_matrix.copy(),
                    "world": t_world_matrix.translation.copy(),
                },
                "axe": {
                    **dict(a_row),
                    "world_matrix": a_world_matrix.copy(),
                    "world": a_world_matrix.translation.copy(),
                },
                "canon": {
                    **dict(c_row),
                    "world_matrix": c_world_matrix.copy(),
                    "world": c_world_matrix.translation.copy(),
                },
            }

        best_solution: Dict[str, Any] | None = None
        best_score = float("inf")
        node_stream_count = len(ordered_indices)
        stream_start_idx, stream_end_idx = _off_mat_stream_section_bounds(node_stream_count)
        for source_idx in range(int(stream_start_idx), int(stream_end_idx)):
            if _off_mat_stream_target_index(int(source_idx), 0, node_stream_count) != int(tourelle_idx):
                continue
            if _off_mat_stream_target_index(int(source_idx), 1, node_stream_count) != int(axe_idx):
                continue
            if _off_mat_stream_target_index(int(source_idx), 2, node_stream_count) != int(canon_idx):
                continue
            source_blocks = off_mat_blocks_by_index.get(int(source_idx), [])
            if len(source_blocks) < 3:
                continue
            source_name = _norm_low(str(bone_name_by_index.get(int(source_idx), "") or ""))
            direct_rows: Dict[str, Dict[str, Any]] = {}
            valid_direct = True
            for key, block_idx in (("tourelle", 0), ("axe", 1), ("canon", 2)):
                raw_matrix = _off_mat_block_matrix(source_blocks[int(block_idx)], "row")
                if raw_matrix is None:
                    valid_direct = False
                    break
                blender_matrix = _off_mat_convert_matrix_to_blender(raw_matrix, "xyz:+1-1+1")
                if blender_matrix is None:
                    valid_direct = False
                    break
                decomp = _decompose_affine_components(blender_matrix)
                if decomp is None:
                    valid_direct = False
                    break
                local_translation, local_rotation_basis, local_scale = decomp
                direct_rows[str(key)] = {
                    "source": "stream_local",
                    "source_idx": int(source_idx),
                    "source_name": str(source_name),
                    "block_idx": int(block_idx),
                    "layout": "row",
                    "remap": "xyz:+1-1+1",
                    "local_matrix": blender_matrix.copy(),
                    "local": local_translation.copy(),
                    "local_rotation_basis": [list(row_vals) for row_vals in local_rotation_basis],
                    "local_scale": [float(local_scale[0]), float(local_scale[1]), float(local_scale[2])],
                }
            if not valid_direct:
                continue
            candidate = _evaluate_stream_rows(
                int(source_idx),
                direct_rows["tourelle"],
                direct_rows["axe"],
                direct_rows["canon"],
            )
            if candidate is None:
                continue
            if float(candidate.get("score", 999999.0)) < best_score:
                best_score = float(candidate.get("score", 999999.0))
                best_solution = dict(candidate)

        source_map: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for target_idx in (int(tourelle_idx), int(axe_idx), int(canon_idx)):
            for row in off_mat_stream_local_candidates_by_index.get(int(target_idx), []):
                source_idx = int(row.get("source_idx", -1))
                if source_idx < 0:
                    continue
                source_map[int(target_idx)][int(source_idx)] = dict(row)

        shared_sources = (
            set(source_map.get(int(tourelle_idx), {}).keys())
            & set(source_map.get(int(axe_idx), {}).keys())
            & set(source_map.get(int(canon_idx), {}).keys())
        )
        for source_idx in sorted(shared_sources):
            t_row = source_map[int(tourelle_idx)].get(int(source_idx))
            a_row = source_map[int(axe_idx)].get(int(source_idx))
            c_row = source_map[int(canon_idx)].get(int(source_idx))
            if not isinstance(t_row, dict) or not isinstance(a_row, dict) or not isinstance(c_row, dict):
                continue
            candidate = _evaluate_stream_rows(int(source_idx), t_row, a_row, c_row)
            if candidate is None:
                continue
            if float(candidate.get("score", 999999.0)) < best_score:
                best_score = float(candidate.get("score", 999999.0))
                best_solution = dict(candidate)
        return best_solution

    def _promote_matching_off_mat_exact(idx: int) -> bool:
        nonlocal semantic_helpers_exact, support_helpers_exact
        idx = int(idx)
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))
        if not raw_low:
            return False
        if mesh_by_bone_index.get(int(idx)):
            return False
        if raw_low in {"armature", "papyrus", "fake"}:
            return False
        parent_idx = int(bone_parent_by_index.get(int(idx), -1))
        if parent_idx < 0:
            return False
        parent_anchor = resolved_positions.get(int(parent_idx))
        current_pos = resolved_positions.get(int(idx))
        if not _is_usable_world_point(parent_anchor):
            return False
        if not _is_usable_world_point(current_pos, parent_anchor=parent_anchor):
            return False

        if raw_low in semantic_scene_nodes:
            if semantic_helper_source_by_index.get(int(idx), "") in {"gfx_exact", "spk_exact"}:
                return True
            role_name = str(
                _semantic_role_for_index(int(idx))
                or ("weapon_fx_anchor" if raw_low in gfx_fx_nodes else "subdepiction_anchor" if raw_low in gfx_subdepiction_nodes else "semantic_helper")
            )
        elif raw_low in semantic_support_nodes:
            if support_helper_source_by_index.get(int(idx), "") in {"gfx_exact", "spk_exact"}:
                return True
            role_name = str(_support_role_for_index(int(idx)) or "semantic_support")
        else:
            return False

        best_row: Dict[str, Any] | None = None
        best_dist = float("inf")
        for row in _own_semantic_candidates(int(idx), int(parent_idx), parent_anchor.copy()):
            source_name = str(row.get("source", "") or "")
            if source_name != "own_local" and not source_name.startswith("pair_local:"):
                continue
            world_point = row.get("world")
            local_point = row.get("local")
            if not isinstance(world_point, Vector) or not isinstance(local_point, Vector):
                continue
            dist = float((world_point - current_pos).length)
            if dist < best_dist:
                best_dist = float(dist)
                best_row = dict(row)

        if best_row is None or best_dist > 1.0e-4:
            return False

        provenance = {
            "source_node": str(bone_name_by_index.get(int(best_row.get("source_idx", idx)), raw_low) or raw_low),
            "source_index": int(best_row.get("source_idx", idx)),
            "block_idx": int(best_row.get("block_idx", -1)),
            "layout": "",
            "remap": "",
            "composition": str(best_row.get("source", "") or ""),
            "parent_block_idx": int(best_row.get("parent_block_idx", -1)),
        }

        if raw_low in semantic_scene_nodes:
            semantic_helper_source_by_index[int(idx)] = "spk_exact"
            semantic_helper_reason_by_index[int(idx)] = str(role_name or "semantic_helper")
            semantic_helpers_exact += 1
        else:
            support_helper_source_by_index[int(idx)] = "spk_exact"
            support_helper_reason_by_index[int(idx)] = str(role_name or "semantic_support")
            support_helpers_exact += 1

        _register_exact_local_transform(
            int(idx),
            role=str(role_name or "semantic_support"),
            source_kind="spk_exact",
            local_translation=best_row.get("local"),
            provenance=provenance,
        )
        return True

    def _promote_explicit_exact_world(
        idx: int,
        *,
        source_idx: int,
        block_idx: int,
        composition: str,
        parent_block_idx: int = -1,
        role_override: str = "",
    ) -> bool:
        nonlocal semantic_helpers_exact, support_helpers_exact
        idx = int(idx)
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))
        if not raw_low:
            return False
        parent_idx = int(bone_parent_by_index.get(int(idx), -1))
        if parent_idx < 0:
            return False
        parent_anchor = resolved_positions.get(int(parent_idx))
        world_point = resolved_positions.get(int(idx))
        if not _is_usable_world_point(parent_anchor):
            return False
        if not _is_usable_world_point(world_point, parent_anchor=parent_anchor):
            return False
        local_vec = world_point.copy() - parent_anchor.copy()
        if float(local_vec.length) <= 1.0e-9 and raw_low.startswith("axe_canon_"):
            # Zero delta is valid for some support chains, e.g. recoil carried directly by the parent.
            pass

        if raw_low in semantic_scene_nodes:
            if semantic_helper_source_by_index.get(int(idx), "") in {"gfx_exact", "spk_exact"}:
                return True
            role_name = str(
                role_override
                or _semantic_role_for_index(int(idx))
                or ("weapon_fx_anchor" if raw_low in gfx_fx_nodes else "subdepiction_anchor" if raw_low in gfx_subdepiction_nodes else "semantic_helper")
            )
            semantic_helper_source_by_index[int(idx)] = "spk_exact"
            semantic_helper_reason_by_index[int(idx)] = str(role_name or "semantic_helper")
            semantic_helpers_exact += 1
        elif raw_low in semantic_support_nodes:
            if support_helper_source_by_index.get(int(idx), "") in {"gfx_exact", "spk_exact"}:
                return True
            role_name = str(role_override or _support_role_for_index(int(idx)) or "semantic_support")
            support_helper_source_by_index[int(idx)] = "spk_exact"
            support_helper_reason_by_index[int(idx)] = str(role_name or "semantic_support")
            support_helpers_exact += 1
        else:
            return False

        _register_exact_local_transform(
            int(idx),
            role=str(role_name or "semantic_support"),
            source_kind="spk_exact",
            local_translation=local_vec.copy(),
            provenance={
                "source_node": str(bone_name_by_index.get(int(source_idx), raw_low) or raw_low),
                "source_index": int(source_idx),
                "block_idx": int(block_idx),
                "layout": "",
                "remap": "",
                "composition": str(composition or ""),
                "parent_block_idx": int(parent_block_idx),
            },
        )
        return True

    def _single_local_off_candidate(
        idx: int,
        parent_anchor: Vector,
    ) -> Dict[str, Any] | None:
        point_candidates = [
            (int(block_idx), point.copy())
            for block_idx, point in off_mat_points_by_index.get(int(idx), [])
            if isinstance(point, Vector) and float(point.length) > 1.0e-9
        ]
        if not point_candidates:
            return None
        if any(float(local_point.length) >= 0.25 for _bi, local_point in point_candidates):
            filtered_candidates = [
                (int(block_idx), local_point.copy())
                for block_idx, local_point in point_candidates
                if float(local_point.length) >= 0.15
                and (
                    abs(float(local_point.x)) >= 0.08
                    or abs(float(local_point.y)) >= 0.08
                    or abs(float(local_point.z)) >= 0.08
                )
            ]
            if filtered_candidates:
                point_candidates = filtered_candidates
        if len(point_candidates) != 1:
            return None
        block_idx, local_point = point_candidates[0]
        return {
            "block_idx": int(block_idx),
            "local": local_point.copy(),
            "world": parent_anchor.copy() + local_point.copy(),
            "source_idx": int(idx),
        }

    def _bounds_world(obj: bpy.types.Object) -> Tuple[Vector, Vector, Vector]:
        pts: List[Vector] = []
        if obj.type == "MESH" and obj.bound_box:
            for c in obj.bound_box:
                pts.append(obj.matrix_world @ Vector((c[0], c[1], c[2])))
        if not pts:
            p = obj.matrix_world.translation.copy()
            return p.copy(), p.copy(), p.copy()
        min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        center = (min_v + max_v) * 0.5
        return min_v, max_v, center

    def _is_usable_world_point(
        pos: Any,
        *,
        parent_anchor: Vector | None = None,
        min_len: float = 1.0e-4,
    ) -> bool:
        if not isinstance(pos, Vector):
            return False
        if not (math.isfinite(float(pos.x)) and math.isfinite(float(pos.y)) and math.isfinite(float(pos.z))):
            return False
        pos_len = float(pos.length)
        if pos_len <= float(min_len):
            if parent_anchor is not None and isinstance(parent_anchor, Vector) and float(parent_anchor.length) <= 0.05:
                return True
            return False
        return True

    def _band_centroid_world(
        obj: bpy.types.Object,
        *,
        axis: str,
        fraction: float,
        mode: str = "min",
    ) -> Vector | None:
        if obj.type != "MESH" or obj.data is None or not getattr(obj.data, "vertices", None):
            return None
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_map.get(str(axis).lower())
        if axis_idx is None:
            return None
        pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if not pts:
            return None
        vals = [float(p[axis_idx]) for p in pts]
        lo = min(vals)
        hi = max(vals)
        span = max(0.0, hi - lo)
        frac = max(0.0, min(1.0, float(fraction)))
        if mode == "max":
            threshold = hi - span * frac
            band = [p for p in pts if float(p[axis_idx]) >= threshold]
        else:
            threshold = lo + span * frac
            band = [p for p in pts if float(p[axis_idx]) <= threshold]
        if not band:
            band = pts
        acc = Vector((0.0, 0.0, 0.0))
        for p in band:
            acc += p
        return acc / float(len(band))

    def _point_to_bbox_distance(point: Vector, bbox_min: Vector, bbox_max: Vector) -> float:
        dx = 0.0
        if point.x < bbox_min.x:
            dx = float(bbox_min.x - point.x)
        elif point.x > bbox_max.x:
            dx = float(point.x - bbox_max.x)
        dy = 0.0
        if point.y < bbox_min.y:
            dy = float(bbox_min.y - point.y)
        elif point.y > bbox_max.y:
            dy = float(point.y - bbox_max.y)
        dz = 0.0
        if point.z < bbox_min.z:
            dz = float(bbox_min.z - point.z)
        elif point.z > bbox_max.z:
            dz = float(point.z - bbox_max.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _point_inside_bbox(point: Vector, bbox_min: Vector, bbox_max: Vector, margin: float = 0.0) -> bool:
        return (
            float(point.x) >= float(bbox_min.x) - float(margin)
            and float(point.x) <= float(bbox_max.x) + float(margin)
            and float(point.y) >= float(bbox_min.y) - float(margin)
            and float(point.y) <= float(bbox_max.y) + float(margin)
            and float(point.z) >= float(bbox_min.z) - float(margin)
            and float(point.z) <= float(bbox_max.z) + float(margin)
        )

    def _point_to_bbox_shell_distance(point: Vector, bbox_min: Vector, bbox_max: Vector) -> float:
        if not _point_inside_bbox(point, bbox_min, bbox_max, margin=0.0):
            return _point_to_bbox_distance(point, bbox_min, bbox_max)
        return min(
            float(point.x - bbox_min.x),
            float(bbox_max.x - point.x),
            float(point.y - bbox_min.y),
            float(bbox_max.y - point.y),
            float(point.z - bbox_min.z),
            float(bbox_max.z - point.z),
        )

    def _child_bone_indices(parent_idx: int) -> List[int]:
        return [int(ch) for ch in child_by_parent.get(int(parent_idx), []) if int(ch) in bone_name_by_index]

    def _find_child_with_prefix(parent_idx: int, prefix: str, suffix_num: int) -> int:
        target = f"{str(prefix).lower()}_{int(suffix_num):02d}"
        for ch_idx in _child_bone_indices(int(parent_idx)):
            if _norm_low(str(bone_name_by_index.get(int(ch_idx), ""))) == target:
                return int(ch_idx)
        return -1

    def _ancestor_chain_indices(start_idx: int) -> List[int]:
        out: List[int] = []
        cur = int(start_idx)
        seen: set[int] = set()
        while cur >= 0 and cur not in seen:
            seen.add(cur)
            out.append(int(cur))
            cur = int(bone_parent_by_index.get(int(cur), -1))
        return out

    def _descendant_indices(root_idx: int) -> List[int]:
        out: List[int] = []
        stack = [int(root_idx)]
        seen: set[int] = set()
        while stack:
            cur = int(stack.pop())
            if cur in seen:
                continue
            seen.add(cur)
            out.append(int(cur))
            stack.extend(int(ch) for ch in child_by_parent.get(int(cur), []))
        return out

    def _semantic_closure_indices(node_idx: int, parent_idx: int = -1) -> set[int]:
        closure = set(_ancestor_chain_indices(int(node_idx)))
        if int(parent_idx) >= 0:
            closure.update(_ancestor_chain_indices(int(parent_idx)))
        closure.update(_descendant_indices(int(node_idx)))
        return closure

    def _off_mat_triplet_candidates() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for source_idx in sorted(off_mat_points_by_index.keys()):
            points = off_mat_points_by_index.get(int(source_idx), [])
            if not points:
                continue
            out.append(
                {
                    "source_index": int(source_idx),
                    "source_name": _norm_low(str(bone_name_by_index.get(int(source_idx), ""))),
                    "points": [(int(block_idx), point.copy()) for block_idx, point in points],
                }
            )
        return out

    def _line_deviation(a: Vector, b: Vector, c: Vector) -> float:
        ac = c - a
        denom = ac.length
        if denom <= 1.0e-9:
            return float("inf")
        return float((b - a).cross(ac).length / denom)

    def _claim_off_mat_point(source_idx: int, block_idx: int) -> None:
        reserved_off_mat_points.add((int(source_idx), int(block_idx)))

    def _is_off_mat_point_reserved(source_idx: int, block_idx: int) -> bool:
        return (int(source_idx), int(block_idx)) in reserved_off_mat_points

    def _best_off_mat_chain_triplet(
        parent_ref: Vector,
        parent_bounds: Tuple[Vector, Vector] | None,
        current_points: Sequence[Vector],
        *,
        parent_anchor: Vector | None = None,
        tourelle_idx: int = -1,
        axe_idx: int = -1,
        canon_idx: int = -1,
        allowed_source_indices: set[int] | None = None,
    ) -> List[Tuple[int, int, Vector]]:
        best_score = float("inf")
        best_points: List[Tuple[int, int, Vector]] = []
        for cand in _off_mat_triplet_candidates():
            source_idx = int(cand.get("source_index", -1))
            if allowed_source_indices is not None and int(source_idx) not in allowed_source_indices:
                continue
            raw_points = [
                (int(block_idx), point.copy())
                for block_idx, point in cand.get("points", [])
                if not _is_off_mat_point_reserved(source_idx, int(block_idx))
            ]
            if len(raw_points) < 3:
                continue
            uniq_keys = {
                (round(point.x, 6), round(point.y, 6), round(point.z, 6))
                for _block_idx, point in raw_points
            }
            if len(uniq_keys) < 3:
                continue
            ordered = sorted(
                raw_points,
                key=lambda it: (
                    (it[1] - parent_ref).length,
                    it[1].x,
                    it[1].y,
                    it[1].z,
                ),
            )[:3]
            pts_only = [point for _block_idx, point in ordered]
            gaps = [(pts_only[1] - pts_only[0]).length, (pts_only[2] - pts_only[1]).length]
            if min(gaps) <= 1.0e-4:
                continue
            score = 0.0
            score += _line_deviation(pts_only[0], pts_only[1], pts_only[2]) * 12.0
            spans = [
                max(point[i] for point in pts_only) - min(point[i] for point in pts_only)
                for i in range(3)
            ]
            dominant_span = max(spans)
            if dominant_span <= 0.02:
                continue
            secondary_span = sum(spans) - dominant_span
            score += float(secondary_span) * 5.0
            if (
                isinstance(parent_anchor, Vector)
                and min(int(tourelle_idx), int(axe_idx), int(canon_idx)) >= 0
            ):
                chain_rows = (
                    (
                        int(tourelle_idx),
                        _support_role_for_index(int(tourelle_idx)),
                        _norm_low(str(bone_name_by_index.get(int(tourelle_idx), "") or "")),
                        _norm_low(str(bone_name_by_index.get(int(bone_parent_by_index.get(int(tourelle_idx), -1)), "") or "")),
                        pts_only[0].copy() - parent_anchor.copy(),
                    ),
                    (
                        int(axe_idx),
                        _support_role_for_index(int(axe_idx)),
                        _norm_low(str(bone_name_by_index.get(int(axe_idx), "") or "")),
                        _norm_low(str(bone_name_by_index.get(int(tourelle_idx), "") or "")),
                        pts_only[1].copy() - pts_only[0].copy(),
                    ),
                    (
                        int(canon_idx),
                        _support_role_for_index(int(canon_idx)),
                        _norm_low(str(bone_name_by_index.get(int(canon_idx), "") or "")),
                        _norm_low(str(bone_name_by_index.get(int(axe_idx), "") or "")),
                        pts_only[2].copy() - pts_only[1].copy(),
                    ),
                )
                template_penalty_total = 0.0
                valid_locals = True
                for _node_idx, role_name, raw_low, parent_raw_low, local_vec in chain_rows:
                    template_penalty = _semantic_local_template_penalty(
                        role_name,
                        raw_low,
                        parent_raw_low,
                        local_vec.copy(),
                    )
                    if template_penalty is None:
                        valid_locals = False
                        break
                    template_penalty_total += float(template_penalty)
                if not valid_locals:
                    continue
                score += float(template_penalty_total) * 1.75
            if parent_bounds is not None:
                bbox_min, bbox_max = parent_bounds
                score += sum(_point_to_bbox_distance(point, bbox_min, bbox_max) for point in pts_only) * 3.5
            if current_points:
                score += sum((pts_only[i] - current_points[i]).length for i in range(min(3, len(current_points)))) * 0.5
            score += abs(gaps[0] - gaps[1]) * 1.5
            if score < best_score:
                best_score = float(score)
                best_points = [(source_idx, int(block_idx), point.copy()) for block_idx, point in ordered]
        return best_points

    def _best_off_mat_pair(
        parent_ref: Vector,
        current_points: Sequence[Vector],
        allowed_source_indices: set[int] | None = None,
    ) -> List[Tuple[int, int, Vector]]:
        best_score = float("inf")
        best_pair: List[Tuple[int, int, Vector]] = []
        triplets = _off_mat_triplet_candidates()
        for cand in triplets:
            source_idx = int(cand.get("source_index", -1))
            if allowed_source_indices is not None and int(source_idx) not in allowed_source_indices:
                continue
            raw_points = [
                (int(block_idx), point.copy())
                for block_idx, point in cand.get("points", [])
                if not _is_off_mat_point_reserved(source_idx, int(block_idx))
            ]
            if len(raw_points) < 2:
                continue
            for i in range(len(raw_points)):
                for j in range(i + 1, len(raw_points)):
                    pair = [raw_points[i], raw_points[j]]
                    pts_only = [pair[0][1], pair[1][1]]
                    pair_center = (pts_only[0] + pts_only[1]) * 0.5
                    pair_delta = pts_only[1] - pts_only[0]
                    spans = [abs(float(pair_delta[k])) for k in range(3)]
                    dominant_span = max(spans)
                    if dominant_span <= 0.02:
                        continue
                    secondary_span = sum(spans) - dominant_span
                    score = 0.0
                    score += (pair_center - parent_ref).length * 1.5
                    score += secondary_span * 4.0
                    score += abs((pts_only[0] - pair_center).length - (pts_only[1] - pair_center).length) * 2.0
                    if current_points:
                        if len(current_points) >= 2:
                            ordered_by_y = sorted(pts_only, key=lambda p: (-p.y, p.x, p.z))
                            score += (ordered_by_y[0] - current_points[0]).length * 0.35
                            score += (ordered_by_y[1] - current_points[1]).length * 0.35
                        else:
                            score += min((point - current_points[0]).length for point in pts_only) * 0.35
                    if score < best_score:
                        best_score = float(score)
                        ordered_raw = sorted(pair, key=lambda it: (-it[1].y, it[1].x, it[1].z))
                        best_pair = [(source_idx, int(block_idx), point.copy()) for block_idx, point in ordered_raw]
        return best_pair

    def _best_nearest_off_mat_point(current_pos: Vector) -> Tuple[int, int, Vector] | None:
        best_score = float("inf")
        best_match: Tuple[int, int, Vector] | None = None
        for source_idx, points in off_mat_points_by_index.items():
            for block_idx, point in points:
                if _is_off_mat_point_reserved(int(source_idx), int(block_idx)):
                    continue
                dist = float((point - current_pos).length)
                if dist < best_score:
                    best_score = dist
                    best_match = (int(source_idx), int(block_idx), point.copy())
        return best_match

    def _nearest_resolved_ancestor_anchor(start_idx: int) -> Tuple[int, Vector]:
        cur = int(bone_parent_by_index.get(int(start_idx), -1))
        while cur >= 0:
            pos = resolved_positions.get(int(cur))
            if pos is not None:
                return int(cur), pos.copy()
            cur = int(bone_parent_by_index.get(int(cur), -1))
        return -1, Vector((0.0, 0.0, 0.0))

    def _own_local_off_candidates(
        bidx: int,
        anchor: Vector,
        ignore_reserved: bool = False,
    ) -> List[Tuple[int, int, Vector]]:
        out: List[Tuple[int, int, Vector]] = []
        for block_idx, point in off_mat_points_by_index.get(int(bidx), []):
            if not ignore_reserved and _is_off_mat_point_reserved(int(bidx), int(block_idx)):
                continue
            out.append((int(bidx), int(block_idx), anchor + point.copy()))
        return out

    def _max_pairwise_distance(points: Sequence[Vector]) -> float:
        if len(points) < 2:
            return 0.0
        best = 0.0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                best = max(best, float((points[i] - points[j]).length))
        return best

    def _mean_point(points: Sequence[Vector]) -> Vector:
        if not points:
            return Vector((0.0, 0.0, 0.0))
        acc = Vector((0.0, 0.0, 0.0))
        for point in points:
            acc += point
        return acc / float(len(points))

    def _best_local_helper_chain_solution(
        parent_anchor: Vector,
        parent_bounds: Tuple[Vector, Vector] | None,
        tourelle_idx: int,
        axe_idx: int,
        canon_idx: int,
        fx_children: Sequence[int],
        ignore_reserved: bool = False,
    ) -> Dict[str, Any] | None:
        t_cands = _own_local_off_candidates(
            int(tourelle_idx),
            parent_anchor.copy(),
            ignore_reserved=bool(ignore_reserved),
        )
        a_local = [
            (int(block_idx), point.copy())
            for block_idx, point in off_mat_points_by_index.get(int(axe_idx), [])
            if bool(ignore_reserved) or not _is_off_mat_point_reserved(int(axe_idx), int(block_idx))
        ]
        c_local = [
            (int(block_idx), point.copy())
            for block_idx, point in off_mat_points_by_index.get(int(canon_idx), [])
            if bool(ignore_reserved) or not _is_off_mat_point_reserved(int(canon_idx), int(block_idx))
        ]
        if not t_cands or not a_local or not c_local:
            return None

        fx_raw_points: List[Vector] = []
        for fx_idx in fx_children:
            for _block_idx, point in off_mat_points_by_index.get(int(fx_idx), []):
                fx_raw_points.append(point.copy())
        side_sign = 0.0
        if fx_raw_points:
            avg_y = sum(float(p.y) for p in fx_raw_points) / float(len(fx_raw_points))
            if avg_y > 0.05:
                side_sign = 1.0
            elif avg_y < -0.05:
                side_sign = -1.0

        best_score = float("inf")
        best_solution: Dict[str, Any] | None = None
        parent_mn = parent_bounds[0].copy() if parent_bounds is not None else None
        parent_mx = parent_bounds[1].copy() if parent_bounds is not None else None

        current_t = resolved_positions.get(int(tourelle_idx))
        current_a = resolved_positions.get(int(axe_idx))
        current_c = resolved_positions.get(int(canon_idx))

        def _score_fx_candidate(
            world_point: Vector,
            local_point: Vector,
            current_fx: Vector | None,
        ) -> float:
            score = 0.0
            local_len = float(local_point.length)
            if local_len <= 1.0e-6:
                return 1.0e9
            if parent_mn is not None and parent_mx is not None:
                shell = _point_to_bbox_shell_distance(world_point, parent_mn, parent_mx)
                if _point_inside_bbox(world_point, parent_mn, parent_mx, margin=0.0):
                    score += 4.0 + shell * 10.0
                else:
                    score += shell * 1.5
            score += max(0.0, 0.12 - abs(float(local_point.y))) * 16.0
            score += max(0.0, 0.08 - abs(float(local_point.z))) * 10.0
            score += max(0.0, 0.05 - local_len) * 30.0
            if current_fx is not None:
                score += float((world_point - current_fx).length) * 0.025
            return score

        for t_src_idx, t_block_idx, t_point in t_cands:
            dt = t_point - parent_anchor
            if dt.length <= 0.01:
                continue
            score_t = 0.0
            if side_sign != 0.0:
                side_delta = float(t_point.y - parent_anchor.y)
                if side_delta * side_sign <= 0.0:
                    score_t += 4.0
                score_t += max(0.0, 0.20 - abs(side_delta)) * 24.0
            if parent_mn is not None and parent_mx is not None:
                score_t += _point_to_bbox_distance(t_point, parent_mn, parent_mx) * 1.5
            if current_t is not None:
                score_t += float((t_point - current_t).length) * 0.02

            for a_block_idx, a_point_local in a_local:
                a_point = t_point + a_point_local
                da = a_point - t_point
                score_a = score_t
                if da.length <= 0.02:
                    score_a += 5.0
                if da.length > 1.75:
                    score_a += (float(da.length) - 1.75) * 4.0
                if side_sign != 0.0:
                    side_delta_a = float(a_point.y - parent_anchor.y)
                    if side_delta_a * side_sign <= -0.05:
                        score_a += 2.0
                    score_a += max(0.0, 0.10 - abs(side_delta_a)) * 8.0
                if current_a is not None:
                    score_a += float((a_point - current_a).length) * 0.01

                for c_block_idx, c_point_local in c_local:
                    c_point = a_point + c_point_local
                    dc = c_point - a_point
                    score_c = score_a
                    if dc.length <= 0.02:
                        score_c += 5.0
                    if side_sign != 0.0:
                        side_delta_c = float(c_point.y - parent_anchor.y)
                        if side_delta_c * side_sign <= -0.05:
                            score_c += 2.0
                        score_c += max(0.0, 0.10 - abs(side_delta_c)) * 8.0
                    if current_c is not None:
                        score_c += float((c_point - current_c).length) * 0.01
                    if dc.length > 1.75:
                        score_c += (float(dc.length) - 1.75) * 4.0

                    fx_solution: List[Tuple[int, int, int, Vector]] = []
                    fx_score = score_c
                    if fx_children:
                        for fx_idx in fx_children:
                            current_fx = resolved_positions.get(int(fx_idx))
                            point_candidates = [
                                (int(block_idx), point.copy())
                                for block_idx, point in off_mat_points_by_index.get(int(fx_idx), [])
                                if point.length > 1.0e-9
                            ]
                            if not point_candidates:
                                continue
                            if any(float(local_point.length) >= 0.25 for _bi, local_point in point_candidates):
                                filtered_candidates = [
                                    (int(block_idx), local_point.copy())
                                    for block_idx, local_point in point_candidates
                                    if float(local_point.length) >= 0.15
                                    and (
                                        abs(float(local_point.x)) >= 0.08
                                        or abs(float(local_point.y)) >= 0.08
                                        or abs(float(local_point.z)) >= 0.08
                                    )
                                ]
                                if filtered_candidates:
                                    point_candidates = filtered_candidates
                            best_fx: Tuple[int, Vector, float] | None = None
                            for fx_block_idx, local_point in point_candidates:
                                fx_point = c_point + local_point
                                score_fx = _score_fx_candidate(fx_point, local_point, current_fx)
                                dfx = fx_point - c_point
                                score_fx += abs(float(dfx.x)) * 0.15
                                if side_sign != 0.0:
                                    if float(dfx.y) * side_sign <= 0.0:
                                        score_fx += 1.5
                                cand = (int(fx_block_idx), fx_point.copy(), float(score_fx))
                                if best_fx is None or cand[2] < best_fx[2]:
                                    best_fx = cand
                            if best_fx is None:
                                continue
                            fx_block_idx, fx_point, score_fx = best_fx
                            fx_solution.append((int(fx_idx), int(fx_block_idx), int(fx_idx), fx_point.copy()))
                            fx_score += float(score_fx)
                        if len(fx_solution) >= 2:
                            fx_score += 1.0 / max(0.02, _max_pairwise_distance([item[3] for item in fx_solution]))

                    if fx_score < best_score:
                        best_score = fx_score
                        best_solution = {
                            "tourelle": (int(t_src_idx), int(t_block_idx), t_point.copy()),
                            "axe": (int(axe_idx), int(a_block_idx), a_point.copy()),
                            "canon": (int(canon_idx), int(c_block_idx), c_point.copy()),
                            "fx": list(fx_solution),
                        }

        return best_solution

    def _nearest_resolved_semantic_parent(idx: int) -> Tuple[int, Vector | None]:
        cur = int(bone_parent_by_index.get(int(idx), -1))
        while cur >= 0:
            pos = resolved_positions.get(int(cur))
            if _is_usable_world_point(pos):
                return int(cur), pos.copy()
            cur = int(bone_parent_by_index.get(int(cur), -1))
        return -1, None

    def _nearest_ancestor_mesh_bounds(idx: int) -> Tuple[Vector, Vector, Vector] | None:
        cur = int(bone_parent_by_index.get(int(idx), -1))
        while cur >= 0:
            mesh_nodes = mesh_by_bone_index.get(int(cur), [])
            obj = mesh_nodes[0] if mesh_nodes else None
            if obj is None:
                obj = mesh_by_name_low.get(_norm_low(str(bone_name_by_index.get(int(cur), ""))))
            if obj is not None and getattr(obj, "type", "") == "MESH":
                return _bounds_world(obj)
            cur = int(bone_parent_by_index.get(int(cur), -1))
        return None

    def _semantic_role_for_index(idx: int) -> str:
        return _norm_low(str(primary_role_by_index.get(int(idx), "") or ""))

    def _support_role_for_index(idx: int) -> str:
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))
        explicit = _semantic_role_for_index(int(idx))
        if explicit:
            return explicit
        if raw_low.startswith("tourelle_"):
            return "turret_yaw_node"
        if raw_low.startswith("axe_canon_"):
            return "turret_pitch_node"
        if raw_low.startswith("canon_"):
            return "turret_recoil_node"
        if raw_low.startswith("fx_"):
            return "weapon_fx_anchor"
        return "semantic_support"

    def _own_semantic_candidates(
        idx: int,
        parent_idx: int,
        parent_anchor: Vector,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen_keys: set[Tuple[float, float, float, str]] = set()

        def _append(source: str, world_point: Vector, local_point: Vector | None, block_idx: int, extra: Dict[str, Any] | None = None) -> None:
            if not isinstance(world_point, Vector):
                return
            if not math.isfinite(float(world_point.x)) or not math.isfinite(float(world_point.y)) or not math.isfinite(float(world_point.z)):
                return
            if float(world_point.length) <= 1.0e-9:
                return
            key = (
                round(float(world_point.x), 6),
                round(float(world_point.y), 6),
                round(float(world_point.z), 6),
                str(source),
            )
            if key in seen_keys:
                return
            seen_keys.add(key)
            row = {
                "source": str(source),
                "world": world_point.copy(),
                "block_idx": int(block_idx),
                "source_idx": int(idx),
            }
            if isinstance(local_point, Vector):
                row["local"] = local_point.copy()
            if isinstance(extra, dict):
                row.update(extra)
            candidates.append(row)

        for block_idx, point in off_mat_points_by_index.get(int(idx), []):
            _append("own_world", point.copy(), None, int(block_idx))
            local_point = point.copy()
            _append("own_local", parent_anchor + local_point, local_point, int(block_idx))

        child_blocks = off_mat_blocks_by_index.get(int(idx), [])
        parent_blocks = off_mat_blocks_by_index.get(int(parent_idx), []) if int(parent_idx) >= 0 else []
        for child_block_idx, child_block in enumerate(child_blocks):
            for parent_block_idx, parent_block in enumerate(parent_blocks):
                for rule_name, local_xyz in _off_mat_pair_local_candidates(child_block, parent_block).items():
                    try:
                        local_point = Vector((float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2])))
                    except Exception:
                        continue
                    if not math.isfinite(float(local_point.x)) or not math.isfinite(float(local_point.y)) or not math.isfinite(float(local_point.z)):
                        continue
                    if float(local_point.length) <= 1.0e-9:
                        continue
                    _append(
                        f"pair_local:{rule_name}",
                        parent_anchor + local_point,
                        local_point,
                        int(child_block_idx),
                        {"parent_block_idx": int(parent_block_idx)},
                    )
        return candidates

    def _semantic_local_template(
        role: str,
        raw_low: str,
        parent_raw_low: str,
    ) -> Dict[str, float] | None:
        role_low = _norm_low(role)
        raw_low = _norm_low(raw_low)
        parent_raw_low = _norm_low(parent_raw_low)
        if role_low == "weapon_fx_anchor":
            if parent_raw_low == "canon_01":
                return {
                    "x_min": 1.0,
                    "x_max": 2.6,
                    "y_min": -0.12,
                    "y_max": 0.12,
                    "z_min": -0.18,
                    "z_max": 0.18,
                    "target_x": 1.7,
                    "target_y": 0.0,
                    "target_z": 0.0,
                }
            if parent_raw_low == "canon_02":
                return {
                    "x_min": 0.55,
                    "x_max": 1.5,
                    "y_min": -0.12,
                    "y_max": 0.12,
                    "z_min": -0.18,
                    "z_max": 0.18,
                    "target_x": 1.0,
                    "target_y": 0.0,
                    "target_z": 0.03,
                }
            if parent_raw_low in {"canon_03", "canon_04"}:
                return {
                    "x_min": 0.2,
                    "x_max": 0.95,
                    "y_min": -0.12,
                    "y_max": 0.12,
                    "z_min": -0.18,
                    "z_max": 0.18,
                    "target_x": 0.5 if parent_raw_low == "canon_03" else 0.55,
                    "target_y": 0.0,
                    "target_z": 0.0,
                }
            if parent_raw_low == "canon_05":
                return {
                    "x_min": -0.55,
                    "x_max": 0.55,
                    "abs_y_min": 0.85,
                    "abs_y_max": 2.35,
                    "z_min": -0.4,
                    "z_max": 1.25,
                    "target_x": 0.0,
                    "target_abs_y": 1.45,
                    "target_z": 0.35,
                }
        if role_low == "subdepiction_anchor" and raw_low.startswith("base_tourelle_") and parent_raw_low.startswith("tourelle_01"):
            return {
                "x_min": -0.55,
                "x_max": 0.55,
                "abs_y_min": 0.25,
                "abs_y_max": 0.95,
                "z_min": 0.55,
                "z_max": 1.25,
                "target_x": 0.0,
                "target_abs_y": 0.5,
                "target_z": 0.9,
            }
        if role_low == "turret_yaw_node":
            return {
                "x_min": -0.65,
                "x_max": 0.65,
                "y_min": -1.35,
                "y_max": 1.35,
                "z_min": -0.35,
                "z_max": 1.35,
                "target_x": 0.0,
                "target_y": 0.0,
            }
        if role_low == "turret_pitch_node" and parent_raw_low.startswith("tourelle_"):
            return {
                "abs_x_min": 0.03,
                "abs_x_max": 0.8,
                "y_min": -0.15,
                "y_max": 0.15,
                "z_min": -0.15,
                "z_max": 0.15,
                "target_abs_x": 0.2,
                "target_y": 0.0,
                "target_z": 0.0,
            }
        if role_low == "turret_recoil_node" and parent_raw_low.startswith("axe_canon_"):
            return {
                "abs_x_min": 0.02,
                "abs_x_max": 1.0,
                "y_min": -0.18,
                "y_max": 0.18,
                "z_min": -0.18,
                "z_max": 0.18,
                "target_abs_x": 0.25,
                "target_y": 0.0,
                "target_z": 0.0,
            }
        return None

    def _semantic_local_template_penalty(
        role: str,
        raw_low: str,
        parent_raw_low: str,
        local_point: Vector | None,
    ) -> float | None:
        if not isinstance(local_point, Vector):
            return 0.0
        template = _semantic_local_template(role, raw_low, parent_raw_low)
        if not template:
            return 0.0
        x = float(local_point.x)
        y = float(local_point.y)
        z = float(local_point.z)
        abs_x = abs(x)
        abs_y = abs(y)
        for bound_key, bound_value in template.items():
            if not str(bound_key).endswith(("_min", "_max")):
                continue
            current_value = None
            if str(bound_key).startswith("x_"):
                current_value = x
            elif str(bound_key).startswith("abs_x_"):
                current_value = abs_x
            elif str(bound_key).startswith("y_"):
                current_value = y
            elif str(bound_key).startswith("z_"):
                current_value = z
            elif str(bound_key).startswith("abs_y_"):
                current_value = abs_y
            if current_value is None:
                continue
            if str(bound_key).endswith("_min") and float(current_value) < float(bound_value):
                return None
            if str(bound_key).endswith("_max") and float(current_value) > float(bound_value):
                return None
        penalty = 0.0
        if "target_x" in template:
            penalty += abs(x - float(template["target_x"])) * 0.75
        if "target_abs_x" in template:
            penalty += abs(abs_x - float(template["target_abs_x"])) * 0.75
        if "target_y" in template:
            penalty += abs(y - float(template["target_y"])) * 1.25
        if "target_z" in template:
            penalty += abs(z - float(template["target_z"])) * 1.25
        if "target_abs_y" in template:
            penalty += abs(abs_y - float(template["target_abs_y"])) * 1.0
        return float(penalty)

    def _score_semantic_candidate(
        role: str,
        raw_low: str,
        parent_raw_low: str,
        world_point: Vector,
        parent_anchor: Vector,
        parent_bounds: Tuple[Vector, Vector] | None,
        current_pos: Vector | None,
        source_name: str,
        local_point: Vector | None,
    ) -> float:
        score = 0.0
        source_low = str(source_name or "").strip().lower()
        delta = world_point - parent_anchor
        delta_len = float(delta.length)
        template_penalty = _semantic_local_template_penalty(
            role,
            raw_low,
            parent_raw_low,
            local_point if isinstance(local_point, Vector) else None,
        )
        if template_penalty is None:
            score += 50.0
        else:
            score += float(template_penalty)
        if source_name.startswith("pair_local:"):
            score -= 1.5
        elif source_name == "own_local":
            score += 0.6
        elif source_name.startswith("global_world:"):
            score += 0.2
        else:
            score += 2.0
        # Global candidate quality is highly variant-dependent. Current stable hits across
        # Abrams/Leopard are row-major translation with current/identity remaps; other
        # remaps are debug material and should lose unless nothing else exists.
        if "column_major_last_col" in source_low:
            score += 2.5
        if ":flip_xy" in source_low or ":flip_xz" in source_low:
            score += 3.5
        if role in {"weapon_fx_anchor", "subdepiction_anchor"}:
            if source_name.startswith("pair_local:"):
                score += 1.5
            elif source_name == "own_local":
                score += 1.4
            elif source_name == "own_world":
                score += 2.5
        if current_pos is not None:
            score += float((world_point - current_pos).length) * 0.03
        if parent_bounds is not None:
            bbox_min, bbox_max = parent_bounds
            outside_dist = _point_to_bbox_distance(world_point, bbox_min, bbox_max)
            inside = _point_inside_bbox(world_point, bbox_min, bbox_max, margin=0.0)
            shell = _point_to_bbox_shell_distance(world_point, bbox_min, bbox_max)
        else:
            outside_dist = 0.0
            inside = False
            shell = 0.0

        if role == "weapon_fx_anchor":
            if parent_bounds is not None:
                if inside:
                    score += 12.0 + max(0.0, float(shell)) * 18.0
                else:
                    score += float(outside_dist) * 1.2
            score += max(0.0, 0.05 - abs(float(delta.x))) * 10.0
            score += max(0.0, 0.05 - abs(float(delta.z))) * 8.0
            if isinstance(local_point, Vector):
                score += max(0.0, 0.04 - float(local_point.length)) * 25.0
                if raw_low.startswith("fx_tourelle5_") and str(parent_raw_low or "").startswith("canon_05"):
                    score += max(0.0, 0.65 - abs(float(local_point.y))) * 10.0
                    score += max(0.0, abs(float(local_point.y)) - 2.6) * 2.0
                    score += max(0.0, abs(float(local_point.x)) - 0.8) * 5.0
                    score += max(0.0, abs(float(local_point.z)) - 1.2) * 2.0
                elif raw_low.startswith("fx_tourelle") and str(parent_raw_low or "").startswith("canon_"):
                    score += max(0.0, abs(float(local_point.y)) - 0.25) * 6.0
                    score += max(0.0, abs(float(local_point.z)) - 0.25) * 6.0
                    score += max(0.0, 0.10 - abs(float(local_point.x))) * 10.0
                    score += max(0.0, abs(float(local_point.x)) - 1.25) * 4.0
        elif role == "subdepiction_anchor":
            if parent_bounds is not None:
                bbox_min, bbox_max = parent_bounds
                score += abs(float(world_point.z - bbox_max.z)) * 1.2
                if inside:
                    score += max(0.0, float(shell)) * 8.0
                else:
                    score += float(outside_dist) * 2.4
            score += delta_len * 0.12
            if isinstance(local_point, Vector):
                score += max(0.0, 0.25 - float(local_point.z)) * 8.0
                score += max(0.0, abs(float(local_point.x)) - 0.75) * 3.0
                score += max(0.0, abs(float(local_point.y)) - 1.25) * 2.0
        elif role == "turret_yaw_node":
            if parent_bounds is not None:
                score += float(outside_dist) * 1.2
                if inside:
                    score += max(0.0, float(shell)) * 3.0
            score += abs(float(delta.y)) * 0.15
        elif role == "turret_pitch_node":
            if parent_bounds is not None:
                score += float(outside_dist) * 1.2
                if inside:
                    score += max(0.0, float(shell)) * 2.0
            score += abs(float(delta.y)) * 0.1
        elif role == "turret_recoil_node":
            if parent_bounds is not None:
                score += float(outside_dist) * 1.0
            score += max(0.0, 0.02 - abs(float(delta.x))) * 12.0
        else:
            score += delta_len * 0.15
            if parent_bounds is not None:
                score += float(outside_dist) * 1.25
        return float(score)

    def _is_preferred_global_variant(role: str, variant_name: str) -> bool:
        low = str(variant_name or "").strip().lower()
        if not low.startswith("row_major_last_col:"):
            return False
        remap = low.split(":", 1)[1] if ":" in low else low
        if remap not in {"current", "identity"}:
            return False
        if role in {"weapon_fx_anchor", "subdepiction_anchor", "turret_yaw_node", "turret_pitch_node"}:
            return True
        return False

    def _global_role_band_adjustment(
        role: str,
        raw_low: str,
        parent_raw_low: str,
        local_point: Vector | None,
    ) -> float | None:
        if not isinstance(local_point, Vector):
            return 0.0
        template_penalty = _semantic_local_template_penalty(role, raw_low, parent_raw_low, local_point)
        if template_penalty is None:
            return None
        return -min(1.5, max(0.0, 1.5 - float(template_penalty)))

    def _global_semantic_candidates(
        idx: int,
        *,
        role_override: str | None = None,
    ) -> List[Dict[str, Any]]:
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "")))
        role = str(role_override or _semantic_role_for_index(int(idx)) or "")
        if not role:
            if raw_low in gfx_fx_nodes:
                role = "weapon_fx_anchor"
            elif raw_low in gfx_subdepiction_nodes:
                role = "subdepiction_anchor"
            else:
                role = "semantic_helper"
        if role not in {"weapon_fx_anchor", "subdepiction_anchor", "turret_yaw_node", "turret_pitch_node"}:
            return []
        parent_idx, parent_anchor = _nearest_resolved_semantic_parent(int(idx))
        if parent_anchor is None:
            return []
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        parent_bounds_full = _nearest_ancestor_mesh_bounds(int(idx))
        parent_bounds = None
        if parent_bounds_full is not None:
            parent_bounds = (parent_bounds_full[0].copy(), parent_bounds_full[1].copy())
        current_pos = resolved_positions.get(int(idx))
        rows: List[Dict[str, Any]] = []
        max_parent_delta = 3.5
        if parent_bounds is not None:
            try:
                max_parent_delta = max(1.25, float((parent_bounds[1] - parent_bounds[0]).length) * 1.6)
            except Exception:
                max_parent_delta = 3.5
        for row in off_mat_global_translation_candidates:
            source_idx = int(row.get("source_idx", -1))
            block_idx = int(row.get("block_idx", -1))
            variant_name = str(row.get("variant", "") or "")
            if not _is_preferred_global_variant(role, variant_name):
                continue
            world_point = row.get("world")
            if not isinstance(world_point, Vector):
                continue
            if _is_off_mat_point_reserved(source_idx, block_idx):
                continue
            if not _is_usable_world_point(world_point, parent_anchor=parent_anchor):
                continue
            if float((world_point - parent_anchor).length) > max_parent_delta:
                continue
            local_point = world_point.copy() - parent_anchor.copy()
            band_adjust = _global_role_band_adjustment(
                role,
                raw_low,
                parent_raw_low,
                local_point.copy(),
            )
            if band_adjust is None:
                continue
            score = _score_semantic_candidate(
                role,
                raw_low,
                parent_raw_low,
                world_point.copy(),
                parent_anchor.copy(),
                parent_bounds,
                current_pos.copy() if isinstance(current_pos, Vector) else None,
                f"global_world:{str(row.get('source_name', ''))}:{str(row.get('variant', ''))}",
                local_point.copy(),
            )
            score += float(band_adjust)
            rows.append(
                {
                    "world": world_point.copy(),
                    "score": float(score),
                    "source": "global_world",
                    "role": str(role),
                    "local": local_point.copy(),
                    "block_idx": int(block_idx),
                    "source_idx": int(source_idx),
                    "source_name": str(row.get("source_name", "")),
                    "variant": variant_name,
                }
            )
        rows.sort(key=lambda item: float(item.get("score", 999999.0)))
        return rows

    def _best_global_semantic_candidate(
        idx: int,
        *,
        role_override: str | None = None,
    ) -> Dict[str, Any] | None:
        rows = _global_semantic_candidates(int(idx), role_override=role_override)
        if not rows:
            return None
        best_row = dict(rows[0])
        best_score = float(best_row.get("score", 999.0))
        second_best_score = float(rows[1].get("score", 999.0)) if len(rows) >= 2 else float("inf")
        if best_row is None:
            return None
        best_score_val = float(best_row.get("score", 999.0))
        margin = float(second_best_score - best_score) if math.isfinite(second_best_score) else 999.0
        exact = False
        if role == "weapon_fx_anchor":
            exact = bool(best_score_val <= 1.05)
        elif role == "subdepiction_anchor":
            exact = bool(best_score_val <= 0.95)
        elif role in {"turret_yaw_node", "turret_pitch_node", "turret_recoil_node"}:
            exact = bool(best_score_val <= 0.9)
        else:
            exact = bool(best_score_val <= 0.8 and margin >= 0.05)
        best_row["exact"] = exact
        best_row["second_score"] = float(second_best_score) if math.isfinite(second_best_score) else None
        best_row["margin"] = float(margin)
        best_row["confidence"] = "high" if exact else "low"
        return best_row

    def _current_semantic_position_score(
        idx: int,
        role: str,
    ) -> Dict[str, Any] | None:
        parent_idx, parent_anchor = _nearest_resolved_semantic_parent(int(idx))
        if parent_anchor is None:
            return None
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        current_pos = resolved_positions.get(int(idx))
        if not _is_usable_world_point(current_pos, parent_anchor=parent_anchor):
            return None
        parent_bounds_full = _nearest_ancestor_mesh_bounds(int(idx))
        parent_bounds = None
        if parent_bounds_full is not None:
            parent_bounds = (parent_bounds_full[0].copy(), parent_bounds_full[1].copy())
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "")))
        score = _score_semantic_candidate(
            str(role or "semantic_helper"),
            raw_low,
            parent_raw_low,
            current_pos.copy(),
            parent_anchor.copy(),
            parent_bounds,
            None,
            "current_world",
            current_pos.copy() - parent_anchor.copy(),
        )
        return {
            "world": current_pos.copy(),
            "local": current_pos.copy() - parent_anchor.copy(),
            "score": float(score),
            "source": "current_world",
            "role": str(role or "semantic_helper"),
            "confidence": "low",
        }

    def _mesh_current_semantic_candidate(
        idx: int,
        role: str,
    ) -> Dict[str, Any] | None:
        parent_idx, _parent_anchor = _nearest_resolved_semantic_parent(int(idx))
        if parent_idx < 0:
            return None
        parent_obj = None
        parent_mesh_nodes = mesh_by_bone_index.get(int(parent_idx), [])
        if parent_mesh_nodes:
            parent_obj = parent_mesh_nodes[0]
        if parent_obj is None:
            parent_obj = mesh_by_name_low.get(_norm_low(str(bone_name_by_index.get(int(parent_idx), ""))))
        if parent_obj is None or getattr(parent_obj, "type", "") != "MESH":
            return None
        current_row = _current_semantic_position_score(int(idx), str(role or "semantic_helper"))
        if current_row is None:
            return None
        local_vec = current_row.get("local")
        if not isinstance(local_vec, Vector):
            return None
        out_row = dict(current_row)
        out_row["source"] = "mesh_current"
        out_row["local"] = local_vec.copy()
        out_row["confidence"] = "fallback"
        out_row["exact"] = False
        return out_row

    def _semantic_template_candidate(
        idx: int,
        parent_anchor: Vector,
        role: str,
    ) -> Dict[str, Any] | None:
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "")))
        parent_idx = int(bone_parent_by_index.get(int(idx), -1))
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        template = _semantic_local_template(str(role or "semantic_helper"), raw_low, parent_raw_low)
        if not template:
            return None
        local_x = float(template.get("target_x", 0.0))
        local_y = 0.0
        if "target_y" in template:
            local_y = float(template.get("target_y", 0.0))
        elif "target_abs_y" in template:
            sign = 0.0
            own_points = [
                point.copy()
                for _block_idx, point in off_mat_points_by_index.get(int(idx), [])
                if isinstance(point, Vector) and float(point.length) > 1.0e-9
            ]
            if own_points:
                avg_y = sum(float(point.y) for point in own_points) / float(len(own_points))
                if abs(avg_y) > 1.0e-5:
                    sign = 1.0 if avg_y > 0.0 else -1.0
            if sign == 0.0:
                current_pos = resolved_positions.get(int(idx))
                if _is_usable_world_point(current_pos, parent_anchor=parent_anchor):
                    current_local = current_pos.copy() - parent_anchor.copy()
                    if abs(float(current_local.y)) > 1.0e-5:
                        sign = 1.0 if float(current_local.y) > 0.0 else -1.0
            if sign == 0.0:
                sign = 1.0
            local_y = sign * float(template.get("target_abs_y", 0.0))
        local_z = float(template.get("target_z", 0.0))
        local_vec = Vector((local_x, local_y, local_z))
        return {
            "world": parent_anchor.copy() + local_vec.copy(),
            "local": local_vec.copy(),
            "score": 0.0,
            "source": "role_template",
            "role": str(role or "semantic_helper"),
            "confidence": "fallback",
            "exact": False,
        }

    def _semantic_fallback_candidate(
        idx: int,
        parent_anchor: Vector,
        role: str,
    ) -> Dict[str, Any] | None:
        role_low = _norm_low(str(role or "semantic_helper"))
        if role_low == "subdepiction_anchor":
            template_candidate = _semantic_template_candidate(int(idx), parent_anchor.copy(), role_low)
            if template_candidate is not None:
                return template_candidate
        if role_low in {"weapon_fx_anchor", "subdepiction_anchor"}:
            global_candidate = _best_global_semantic_candidate(
                int(idx),
                role_override=role_low,
            )
            if global_candidate is not None:
                out_row = dict(global_candidate)
                out_row["confidence"] = "fallback"
                out_row["exact"] = False
                return out_row
        if role_low == "weapon_fx_anchor":
            mesh_candidate = _mesh_current_semantic_candidate(int(idx), role_low)
            if mesh_candidate is not None:
                return mesh_candidate
        return _semantic_template_candidate(int(idx), parent_anchor.copy(), role_low)

    def _best_semantic_helper_candidate(idx: int) -> Dict[str, Any] | None:
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "")))
        role = _semantic_role_for_index(int(idx))
        if not role:
            if raw_low in gfx_fx_nodes:
                role = "weapon_fx_anchor"
            elif raw_low in gfx_subdepiction_nodes:
                role = "subdepiction_anchor"
        parent_idx, parent_anchor = _nearest_resolved_semantic_parent(int(idx))
        if parent_anchor is None:
            return None
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        parent_bounds_full = _nearest_ancestor_mesh_bounds(int(idx))
        parent_bounds = None
        if parent_bounds_full is not None:
            parent_bounds = (parent_bounds_full[0].copy(), parent_bounds_full[1].copy())
        current_pos = resolved_positions.get(int(idx))
        candidates = _own_semantic_candidates(int(idx), int(parent_idx), parent_anchor.copy())
        if not candidates:
            return None
        best_row: Dict[str, Any] | None = None
        best_score = float("inf")
        second_best_score = float("inf")
        for row in candidates:
            world_point = row.get("world")
            if not isinstance(world_point, Vector):
                continue
            score = _score_semantic_candidate(
                role,
                raw_low,
                parent_raw_low,
                world_point.copy(),
                parent_anchor.copy(),
                parent_bounds,
                current_pos.copy() if isinstance(current_pos, Vector) else None,
                str(row.get("source", "")),
                row.get("local") if isinstance(row.get("local"), Vector) else None,
            )
            if score < best_score:
                second_best_score = float(best_score)
                best_score = float(score)
                best_row = {
                    "world": world_point.copy(),
                    "score": float(score),
                    "source": str(row.get("source", "")),
                    "role": str(role or "semantic_helper"),
                }
                if isinstance(row.get("local"), Vector):
                    best_row["local"] = row.get("local").copy()
                if "block_idx" in row:
                    best_row["block_idx"] = int(row.get("block_idx"))
                if "source_idx" in row:
                    best_row["source_idx"] = int(row.get("source_idx"))
                if "parent_block_idx" in row:
                    best_row["parent_block_idx"] = int(row.get("parent_block_idx"))
            elif score < second_best_score:
                second_best_score = float(score)
        if best_row is None:
            return None
        source_name = str(best_row.get("source", ""))
        best_score_val = float(best_row.get("score", 999.0))
        margin = float(second_best_score - best_score) if math.isfinite(second_best_score) else 999.0
        exact = False
        role_name = str(best_row.get("role", "") or "")
        helper_only_canon_parent = (
            role_name == "weapon_fx_anchor"
            and parent_raw_low.startswith("canon_")
            and not mesh_by_bone_index.get(int(parent_idx))
        )
        if role_name == "weapon_fx_anchor":
            if not helper_only_canon_parent and source_name.startswith("pair_local:") and best_score_val <= 1.5 and margin >= 0.0:
                exact = True
            elif not helper_only_canon_parent and source_name == "own_local" and best_score_val <= 1.5 and margin >= 0.0:
                exact = True
        elif role_name == "subdepiction_anchor":
            if source_name.startswith("pair_local:") and best_score_val <= 1.2 and margin >= 0.0:
                exact = True
            elif source_name == "own_local" and best_score_val <= 1.2 and margin >= 0.0:
                exact = True
        else:
            if source_name.startswith("pair_local:") and best_score_val <= 0.45 and margin >= 0.2:
                exact = True
            elif source_name == "own_local" and best_score_val <= 0.25 and margin >= 0.12:
                exact = True
        best_row["exact"] = bool(exact)
        best_row["second_score"] = float(second_best_score) if math.isfinite(second_best_score) else None
        best_row["margin"] = float(margin)
        best_row["confidence"] = "high" if exact else "low"
        return best_row

    def _best_support_helper_candidate(idx: int) -> Dict[str, Any] | None:
        parent_idx, parent_anchor = _nearest_resolved_semantic_parent(int(idx))
        if parent_anchor is None:
            return None
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        role = _support_role_for_index(int(idx))
        current_pos = resolved_positions.get(int(idx))
        parent_bounds_full = _nearest_ancestor_mesh_bounds(int(idx))
        parent_bounds = None
        if parent_bounds_full is not None:
            parent_bounds = (parent_bounds_full[0].copy(), parent_bounds_full[1].copy())
        raw_low = _norm_low(str(bone_name_by_index.get(int(idx), "")))
        candidates = _own_semantic_candidates(int(idx), int(parent_idx), parent_anchor.copy())
        if not candidates:
            return None
        best_row: Dict[str, Any] | None = None
        best_score = float("inf")
        second_best_score = float("inf")
        for row in candidates:
            world_point = row.get("world")
            if not isinstance(world_point, Vector):
                continue
            if not _is_usable_world_point(world_point, parent_anchor=parent_anchor):
                continue
            score = _score_semantic_candidate(
                role,
                raw_low,
                parent_raw_low,
                world_point.copy(),
                parent_anchor.copy(),
                parent_bounds,
                current_pos.copy() if isinstance(current_pos, Vector) else None,
                str(row.get("source", "")),
                row.get("local") if isinstance(row.get("local"), Vector) else None,
            )
            if score < best_score:
                second_best_score = float(best_score)
                best_score = float(score)
                best_row = {
                    "world": world_point.copy(),
                    "score": float(score),
                    "source": str(row.get("source", "")),
                    "role": str(role or "semantic_support"),
                }
                if isinstance(row.get("local"), Vector):
                    best_row["local"] = row.get("local").copy()
                if "block_idx" in row:
                    best_row["block_idx"] = int(row.get("block_idx"))
                if "source_idx" in row:
                    best_row["source_idx"] = int(row.get("source_idx"))
                if "parent_block_idx" in row:
                    best_row["parent_block_idx"] = int(row.get("parent_block_idx"))
            elif score < second_best_score:
                second_best_score = float(score)
        if best_row is None:
            return None
        best_score_val = float(best_row.get("score", 999.0))
        margin = float(second_best_score - best_score) if math.isfinite(second_best_score) else 999.0
        source_name = str(best_row.get("source", ""))
        exact = False
        if source_name.startswith("pair_local:") and best_score_val <= 1.2 and margin >= 0.0:
            exact = True
        elif source_name == "own_local" and best_score_val <= 1.2 and margin >= 0.0:
            exact = True
        best_row["exact"] = exact
        best_row["second_score"] = float(second_best_score) if math.isfinite(second_best_score) else None
        best_row["margin"] = float(margin)
        best_row["confidence"] = "high" if exact else "low"
        return best_row

    # Override Roue_Elev anchors from corresponding wheel mesh centers.
    roue_re = re.compile(r"^roue_elev_([dg])([0-9]+)$", re.IGNORECASE)
    for bidx, raw_name in bone_name_by_index.items():
        low = _norm_low(raw_name)
        m = roue_re.match(low)
        if not m:
            continue
        wheel_name = f"roue_{str(m.group(1)).upper()}{int(m.group(2))}"
        wheel_obj = mesh_by_name_low.get(_norm_low(wheel_name))
        if wheel_obj is None:
            continue
        _mn, _mx, ctr = _bounds_world(wheel_obj)
        resolved_positions[int(bidx)] = ctr.copy()

    # Deterministic FX anchors from source names and mesh bounds.
    chassis_obj = mesh_by_name_low.get("chassis")
    tourelle_obj = mesh_by_name_low.get("tourelle_01")
    canon_by_number: Dict[int, bpy.types.Object] = {}
    for name_low, obj in mesh_by_name_low.items():
        m = re.match(r"^canon_([0-9]+)$", name_low, re.IGNORECASE)
        if m:
            canon_by_number[int(m.group(1))] = obj

    track_right_obj = None
    track_left_obj = None
    stress_pair_indices: set[int] = set()
    for obj in imported_objects:
        low = _norm_low(obj.name)
        if low.startswith("chenille_droite"):
            track_right_obj = obj
        elif low.startswith("chenille_gauche"):
            track_left_obj = obj

    # Dev FBX pivots are much closer to mesh-space anchors than to weighted bone centers.
    for bidx, raw_name in bone_name_by_index.items():
        obj = None
        mesh_nodes = mesh_by_bone_index.get(int(bidx), [])
        if mesh_nodes:
            obj = mesh_nodes[0]
        if obj is None:
            obj = mesh_by_name_low.get(_norm_low(raw_name))
        if obj is None or obj.type != "MESH":
            continue

        low = _norm_low(raw_name)
        mn, mx, ctr = _bounds_world(obj)
        if low == "chassis":
            resolved_positions[int(bidx)] = Vector((0.0, 0.0, 0.0))
            continue
        if low.startswith("tourelle_"):
            base_ctr = _band_centroid_world(obj, axis="z", fraction=0.01, mode="min") or ctr.copy()
            dz = max(0.0, mx.z - mn.z)
            base_ctr.z = mn.z + min(0.08, dz * 0.028)
            resolved_positions[int(bidx)] = base_ctr
            continue
        resolved_positions[int(bidx)] = ctr.copy()

    canon_empty_re = re.compile(r"^axe_canon_([0-9]+)$", re.IGNORECASE)
    launcher_empty_re = re.compile(r"^tourelle_([0-9]+)$", re.IGNORECASE)
    for bidx, raw_name in bone_name_by_index.items():
        low = _norm_low(raw_name)
        m_axe = canon_empty_re.match(low)
        if m_axe and not mesh_by_bone_index.get(int(bidx)):
            canon_obj = canon_by_number.get(int(m_axe.group(1)))
            if canon_obj is not None and canon_obj.type == "MESH":
                mn, mx, ctr = _bounds_world(canon_obj)
                resolved_positions[int(bidx)] = Vector((mn.x, ctr.y, ctr.z))
                continue
        m_tourelle = launcher_empty_re.match(low)
        if m_tourelle and not mesh_by_bone_index.get(int(bidx)):
            canon_obj = canon_by_number.get(int(m_tourelle.group(1)))
            if canon_obj is not None and canon_obj.type == "MESH":
                mn, mx, ctr = _bounds_world(canon_obj)
                dx = max(0.01, (mx.x - mn.x) * 0.122)
                resolved_positions[int(bidx)] = Vector((mn.x - dx, ctr.y, ctr.z))

    empty_chain_groups: List[Dict[str, Any]] = []
    for tourelle_idx in ordered_indices:
        low = _norm_low(str(bone_name_by_index.get(int(tourelle_idx), "")))
        m = launcher_empty_re.match(low)
        if not m or mesh_by_bone_index.get(int(tourelle_idx)):
            continue
        suffix_num = int(m.group(1))
        if suffix_num <= 1:
            continue
        axe_idx = _find_child_with_prefix(int(tourelle_idx), "axe_canon", suffix_num)
        canon_idx = _find_child_with_prefix(int(axe_idx), "canon", suffix_num) if axe_idx >= 0 else -1
        if axe_idx < 0 or canon_idx < 0:
            continue
        if mesh_by_bone_index.get(int(axe_idx)) or mesh_by_bone_index.get(int(canon_idx)):
            continue
        parent_idx = int(bone_parent_by_index.get(int(tourelle_idx), -1))
        parent_obj = None
        if parent_idx >= 0:
            parent_nodes = mesh_by_bone_index.get(int(parent_idx), [])
            if parent_nodes:
                parent_obj = parent_nodes[0]
            elif int(parent_idx) in bone_name_by_index:
                parent_obj = mesh_by_name_low.get(_norm_low(str(bone_name_by_index.get(int(parent_idx), ""))))
        if parent_obj is None:
            continue
        parent_mn, parent_mx, parent_ctr = _bounds_world(parent_obj)
        current_points = [
            resolved_positions.get(int(tourelle_idx), parent_ctr.copy()),
            resolved_positions.get(int(axe_idx), parent_ctr.copy()),
            resolved_positions.get(int(canon_idx), parent_ctr.copy()),
        ]
        semantic_fx_children = []
        for child_idx in _child_bone_indices(int(canon_idx)):
            child_low = _norm_low(str(bone_name_by_index.get(int(child_idx), "")))
            m_fx = re.match(r"^fx_tourelle([0-9]+)_tir_([0-9]+)$", child_low, re.IGNORECASE)
            if not m_fx or int(m_fx.group(1)) != int(suffix_num):
                continue
            if child_low not in semantic_scene_nodes:
                continue
            semantic_fx_children.append(int(child_idx))
        allowed_sources = None
        if len(semantic_fx_children) >= 2:
            allowed_sources = _semantic_closure_indices(int(tourelle_idx), int(parent_idx))
        match = _best_off_mat_chain_triplet(
            parent_ctr.copy(),
            (parent_mn, parent_mx),
            current_points,
            parent_anchor=resolved_positions.get(int(parent_idx)).copy() if isinstance(resolved_positions.get(int(parent_idx)), Vector) else None,
            tourelle_idx=int(tourelle_idx),
            axe_idx=int(axe_idx),
            canon_idx=int(canon_idx),
            allowed_source_indices=allowed_sources,
        )
        if len(match) != 3:
            continue
        empty_chain_groups.append(
            {
                "suffix_num": int(suffix_num),
                "tourelle_idx": int(tourelle_idx),
                "axe_idx": int(axe_idx),
                "canon_idx": int(canon_idx),
                "parent_center": parent_ctr.copy(),
                "match": match,
            }
        )

    for group in sorted(empty_chain_groups, key=lambda item: int(item.get("suffix_num", 0))):
        match = list(group.get("match", []) or [])
        if len(match) != 3:
            continue
        t_idx = int(group.get("tourelle_idx", -1))
        a_idx = int(group.get("axe_idx", -1))
        c_idx = int(group.get("canon_idx", -1))
        if min(t_idx, a_idx, c_idx) < 0:
            continue
        resolved_positions[int(t_idx)] = match[0][2].copy()
        resolved_positions[int(a_idx)] = match[1][2].copy()
        resolved_positions[int(c_idx)] = match[2][2].copy()
        for source_idx, block_idx, _point in match:
            _claim_off_mat_point(int(source_idx), int(block_idx))
        off_mat_chain_resolved += 1
        _promote_explicit_exact_world(int(t_idx), source_idx=int(match[0][0]), block_idx=int(match[0][1]), composition="chain_triplet")
        _promote_explicit_exact_world(int(a_idx), source_idx=int(match[1][0]), block_idx=int(match[1][1]), composition="chain_triplet")
        _promote_explicit_exact_world(int(c_idx), source_idx=int(match[2][0]), block_idx=int(match[2][1]), composition="chain_triplet")

    fx_tir_re = re.compile(r"^fx_tourelle([0-9]+)_tir_([0-9]+)$", re.IGNORECASE)
    fx_tir_count_by_canon: Dict[int, int] = {}
    for raw_name in bone_name_by_index.values():
        mt = fx_tir_re.match(_norm_low(raw_name))
        if not mt:
            continue
        canon_num = int(mt.group(1))
        fx_tir_count_by_canon[canon_num] = fx_tir_count_by_canon.get(canon_num, 0) + 1

    for bidx, raw_name in bone_name_by_index.items():
        low = _norm_low(raw_name)
        if not low.startswith("fx_"):
            continue
        if int(bidx) in stress_pair_indices:
            continue

        if low.startswith("fx_fumee_chenille_d"):
            smoke_points = [
                point.copy()
                for _block_idx, point in off_mat_points_by_index.get(int(bidx), [])
                if float(abs(point.z)) >= 0.05
            ]
            if smoke_points:
                chosen_point = max(
                    smoke_points,
                    key=lambda point: abs(float(point.y)) + abs(float(point.z)),
                )
                _set_resolved_world_point(int(bidx), chosen_point.copy())
                if int(bone_parent_by_index.get(int(bidx), -1)) < 0:
                    exact_local_transform_by_index.pop(int(bidx), None)
                continue
            if use_deterministic_raw_scene and int(bidx) in exact_local_transform_by_index:
                continue
            if track_right_obj is not None:
                mn, mx, ctr = _bounds_world(track_right_obj)
                resolved_positions[int(bidx)] = Vector((mn.x + (mx.x - mn.x) * 0.12, ctr.y, mx.z + max(0.08, (mx.z - mn.z) * 0.08)))
            continue
        if low.startswith("fx_fumee_chenille_g"):
            smoke_points = [
                point.copy()
                for _block_idx, point in off_mat_points_by_index.get(int(bidx), [])
                if float(abs(point.z)) >= 0.05
            ]
            if smoke_points:
                chosen_point = max(
                    smoke_points,
                    key=lambda point: abs(float(point.y)) + abs(float(point.z)),
                )
                _set_resolved_world_point(int(bidx), chosen_point.copy())
                if int(bone_parent_by_index.get(int(bidx), -1)) < 0:
                    exact_local_transform_by_index.pop(int(bidx), None)
                continue
            if use_deterministic_raw_scene and int(bidx) in exact_local_transform_by_index:
                continue
            mirror_d_idx = -1
            for other_idx, other_name in bone_name_by_index.items():
                if _norm_low(str(other_name)).startswith("fx_fumee_chenille_d"):
                    mirror_d_idx = int(other_idx)
                    break
            if mirror_d_idx >= 0 and resolved_positions.get(int(mirror_d_idx)) is not None:
                p_d = resolved_positions[int(mirror_d_idx)].copy()
                _set_resolved_world_point(int(bidx), Vector((p_d.x, -p_d.y, p_d.z)))
                if int(bone_parent_by_index.get(int(bidx), -1)) < 0:
                    exact_local_transform_by_index.pop(int(bidx), None)
            elif track_left_obj is not None:
                mn, mx, ctr = _bounds_world(track_left_obj)
                resolved_positions[int(bidx)] = Vector((mn.x + (mx.x - mn.x) * 0.12, ctr.y, mx.z + max(0.08, (mx.z - mn.z) * 0.08)))
            continue

        mt = fx_tir_re.match(low)
        if mt:
            canon_num = int(mt.group(1))
            tir_num = int(mt.group(2))
            canon_obj = canon_by_number.get(canon_num)
            if canon_obj is not None:
                mn, mx, ctr = _bounds_world(canon_obj)
                lx = max(0.01, (mx.x - mn.x) * 0.02)
                tir_count = int(fx_tir_count_by_canon.get(canon_num, 1))
                y_val = ctr.y
                if tir_count >= 2 and tir_num == 2:
                    y_val = mn.y
                elif tir_count >= 2 and tir_num == 1:
                    y_val = mx.y
                resolved_positions[int(bidx)] = Vector((mx.x + lx, y_val, ctr.z))
            continue

        if chassis_obj is not None:
            cmn, cmx, cctr = _bounds_world(chassis_obj)
            sx = (cmx.x - cmn.x)
            sy = (cmx.y - cmn.y)
            sz = (cmx.z - cmn.z)
            if low == "fx_moteur":
                resolved_positions[int(bidx)] = Vector((cmn.x + sx * 0.04, cctr.y - sy * 0.06, cmx.z + sz * 0.05))
            elif low == "fx_chaleur_01":
                resolved_positions[int(bidx)] = Vector((cmn.x + sx * 0.04, cctr.y + sy * 0.02, cmx.z + sz * 0.05))
            elif low == "fx_incendie":
                resolved_positions[int(bidx)] = Vector((cmn.x + sx * 0.10, cctr.y + sy * 0.14, cmx.z + sz * 0.16))
            elif low == "fx_stress_01":
                resolved_positions[int(bidx)] = Vector((cmn.x + sx * 0.17, cmx.y - sy * 0.10, cmx.z + sz * 0.07))
            elif low == "fx_stress_02":
                resolved_positions[int(bidx)] = Vector((cmn.x + sx * 0.17, cmn.y + sy * 0.10, cmx.z + sz * 0.07))
            elif low == "fx_munition" and tourelle_obj is not None:
                tmn, tmx, tctr = _bounds_world(tourelle_obj)
                tx = tmn.x + (tmx.x - tmn.x) * 0.30
                tz = tmn.z + (tmx.z - tmn.z) * 0.42
                resolved_positions[int(bidx)] = Vector((tx, tctr.y, tz))

    if chassis_obj is not None:
        _cmn, _cmx, chassis_ctr = _bounds_world(chassis_obj)
        sx = float(_cmx.x - _cmn.x)
        sy = float(_cmx.y - _cmn.y)
        sz = float(_cmx.z - _cmn.z)
        stress_children = []
        stress_re = re.compile(r"^fx_stress_([0-9]+)$", re.IGNORECASE)
        for bidx in ordered_indices:
            low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
            ms = stress_re.match(low)
            if not ms:
                continue
            stress_children.append((int(bidx), int(ms.group(1))))
        stress_children = sorted(stress_children, key=lambda item: int(item[1]))
        if len(stress_children) >= 2:
            target_pos = Vector((
                _cmx.x - sx * 0.265,
                chassis_ctr.y + sy * 0.115,
                _cmn.z + sz * 0.77,
            ))
            target_neg = Vector((
                _cmn.x + sx * 0.215,
                chassis_ctr.y - sy * 0.355,
                _cmn.z + sz * 0.82,
            ))
            min_z = _cmn.z + sz * 0.45
            best_score = float("inf")
            best_pair: List[Tuple[int, int, Vector]] = []
            for source_idx, raw_points in off_mat_points_by_index.items():
                avail = [
                    (int(block_idx), point.copy())
                    for block_idx, point in raw_points
                    if not _is_off_mat_point_reserved(int(source_idx), int(block_idx))
                ]
                if len(avail) < 2:
                    continue
                pos_cands = [
                    (block_idx, point)
                    for block_idx, point in avail
                    if point.y > 0.05 and point.z >= min_z
                ]
                neg_cands = [
                    (block_idx, point)
                    for block_idx, point in avail
                    if point.y < -0.05 and point.z >= min_z
                ]
                if not pos_cands or not neg_cands:
                    continue
                for pos_block_idx, pos_point in pos_cands:
                    for neg_block_idx, neg_point in neg_cands:
                        if pos_block_idx == neg_block_idx:
                            continue
                        score = 0.0
                        score += float((pos_point - target_pos).length)
                        score += float((neg_point - target_neg).length)
                        score += _point_to_bbox_distance(pos_point, _cmn, _cmx) * 2.0
                        score += _point_to_bbox_distance(neg_point, _cmn, _cmx) * 2.0
                        score += abs(float(pos_point.z - neg_point.z)) * 0.75
                        if pos_point.x <= chassis_ctr.x:
                            score += 0.4
                        if neg_point.x >= chassis_ctr.x:
                            score += 0.4
                        if score < best_score:
                            best_score = float(score)
                            best_pair = [
                                (int(source_idx), int(pos_block_idx), pos_point.copy()),
                                (int(source_idx), int(neg_block_idx), neg_point.copy()),
                            ]
            if len(best_pair) >= 2:
                for child_i, (child_idx, _ord) in enumerate(stress_children[:2]):
                    resolved_positions[int(child_idx)] = best_pair[child_i][2].copy()
                    stress_pair_indices.add(int(child_idx))
                for source_idx, block_idx, _point in best_pair[:2]:
                    _claim_off_mat_point(int(source_idx), int(block_idx))
                off_mat_pair_resolved += 1

    fx_pair_groups: List[Dict[str, Any]] = []
    paired_fx_indices: set[int] = set()
    fx_tir_suffix_re = re.compile(r"^fx_tourelle([0-9]+)_tir_([0-9]+)$", re.IGNORECASE)
    for canon_idx in ordered_indices:
        canon_low = _norm_low(str(bone_name_by_index.get(int(canon_idx), "")))
        m_canon = re.match(r"^canon_([0-9]+)$", canon_low, re.IGNORECASE)
        if not m_canon:
            continue
        suffix_num = int(m_canon.group(1))
        fx_children: List[Tuple[int, int]] = []
        for child_idx in _child_bone_indices(int(canon_idx)):
            child_low = _norm_low(str(bone_name_by_index.get(int(child_idx), "")))
            m_fx = fx_tir_suffix_re.match(child_low)
            if not m_fx or int(m_fx.group(1)) != suffix_num:
                continue
            fx_children.append((int(child_idx), int(m_fx.group(2))))
        if len(fx_children) < 2:
            continue
        parent_idx = int(bone_parent_by_index.get(int(canon_idx), -1))
        grandparent_idx = int(bone_parent_by_index.get(int(parent_idx), -1))
        if (
            parent_idx >= 0
            and grandparent_idx >= 0
            and not mesh_by_bone_index.get(int(canon_idx))
            and not mesh_by_bone_index.get(int(parent_idx))
            and not mesh_by_bone_index.get(int(grandparent_idx))
        ):
            continue
        parent_ref = resolved_positions.get(int(canon_idx))
        if parent_ref is None:
            continue
        current_points = [
            resolved_positions.get(int(child_idx), parent_ref.copy())
            for child_idx, _tir_num in sorted(fx_children, key=lambda item: int(item[1]))
        ]
        allowed_sources = _semantic_closure_indices(int(canon_idx), int(bone_parent_by_index.get(int(canon_idx), -1)))
        match = _best_off_mat_pair(
            parent_ref.copy(),
            current_points,
            allowed_source_indices=allowed_sources,
        )
        if len(match) < 2:
            continue
        fx_pair_groups.append(
            {
                "canon_idx": int(canon_idx),
                "fx_children": sorted(fx_children, key=lambda item: int(item[1])),
                "match": match[: len(fx_children)],
            }
        )

    for group in fx_pair_groups:
        match = list(group.get("match", []) or [])
        fx_children = list(group.get("fx_children", []) or [])
        if len(match) < len(fx_children):
            continue
        for child_i, (child_idx, _tir_num) in enumerate(fx_children):
            resolved_positions[int(child_idx)] = match[child_i][2].copy()
            paired_fx_indices.add(int(child_idx))
            _promote_explicit_exact_world(
                int(child_idx),
                source_idx=int(match[child_i][0]),
                block_idx=int(match[child_i][1]),
                composition="fx_pair_local",
            )
        for source_idx, block_idx, _point in match[: len(fx_children)]:
            _claim_off_mat_point(int(source_idx), int(block_idx))
        off_mat_pair_resolved += 1

    if chassis_obj is not None:
        chassis_mn, chassis_mx, _chassis_ctr = _bounds_world(chassis_obj)
        chassis_fx_indices: List[int] = []
        for bidx in ordered_indices:
            low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
            if not low.startswith("fx_"):
                continue
            if low.startswith("fx_fumee_chenille_") or low.startswith("fx_tourelle"):
                continue
            if mesh_by_bone_index.get(int(bidx)):
                continue
            parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
            if _norm_low(str(bone_name_by_index.get(int(parent_idx), ""))) != "chassis":
                continue
            chassis_fx_indices.append(int(bidx))

        for bidx in chassis_fx_indices:
            if int(bidx) in stress_pair_indices:
                continue
            current_pos = resolved_positions.get(int(bidx))
            if current_pos is None:
                continue
            best_match = _best_nearest_off_mat_point(current_pos.copy())
            if best_match is None:
                continue
            source_idx, block_idx, point = best_match
            dist = float((point - current_pos).length)
            bbox_dist = _point_to_bbox_distance(point, chassis_mn, chassis_mx)
            if dist > max(1.25, float(diag) * 0.12):
                continue
            if bbox_dist > 0.8:
                continue
            resolved_positions[int(bidx)] = point.copy()
            _claim_off_mat_point(int(source_idx), int(block_idx))
            off_mat_fx_resolved += 1

    mesh_pivot_candidates: List[Tuple[float, int, int, Vector]] = []
    sec_chain_re = re.compile(r"^(tourelle|axe_canon|canon)_([0-9]+)$", re.IGNORECASE)
    for bidx in ordered_indices:
        mesh_nodes = mesh_by_bone_index.get(int(bidx), [])
        if not mesh_nodes:
            continue
        low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        eligible = False
        if low.startswith("blindage_") or low == "fx_munition":
            eligible = True
        else:
            mc = sec_chain_re.match(low)
            if mc and int(mc.group(2)) >= 2:
                eligible = True
        if not eligible:
            continue
        current_pos = resolved_positions.get(int(bidx))
        if current_pos is None:
            continue
        best_match = _best_nearest_off_mat_point(current_pos.copy())
        if best_match is None:
            continue
        source_idx, block_idx, point = best_match
        dist = float((point - current_pos).length)
        if dist > 0.85:
            continue
        mesh_pivot_candidates.append((dist, int(bidx), int(source_idx), point.copy()))

    for _dist, bidx, source_idx, point in sorted(mesh_pivot_candidates, key=lambda item: (item[0], item[1])):
        best_match = _best_nearest_off_mat_point(resolved_positions.get(int(bidx), point.copy()).copy())
        if best_match is None:
            continue
        src_idx2, block_idx2, point2 = best_match
        if int(src_idx2) != int(source_idx):
            point = point2.copy()
            source_idx = int(src_idx2)
            block_idx = int(block_idx2)
        else:
            block_idx = int(block_idx2)
        if float((point2 - resolved_positions.get(int(bidx), point.copy())).length) > 0.85:
            continue
        resolved_positions[int(bidx)] = point2.copy()
        _claim_off_mat_point(int(src_idx2), int(block_idx2))

    chain_members_by_suffix: Dict[int, Dict[str, List[int]]] = {}
    for bidx in ordered_indices:
        low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        mc = sec_chain_re.match(low)
        if mc:
            suffix_num = int(mc.group(2))
            chain_members_by_suffix.setdefault(suffix_num, {}).setdefault(str(mc.group(1)).lower(), []).append(int(bidx))
            continue
        mt = fx_tir_re.match(low)
        if mt:
            suffix_num = int(mt.group(1))
            chain_members_by_suffix.setdefault(suffix_num, {}).setdefault("fx", []).append(int(bidx))

    # Some WARNO helper anchors form an ordered barrel-line sequence across
    # multiple source nodes. Solving them as a monotonic X-chain is more stable
    # than snapping each node independently.
    for suffix_num in sorted(chain_members_by_suffix.keys()):
        members = chain_members_by_suffix.get(int(suffix_num), {})
        fx_nodes = [int(idx) for idx in members.get("fx", []) if int(idx) not in paired_fx_indices]
        if len(fx_nodes) != 1:
            continue

        chain_nodes = []
        for key in ("tourelle", "axe_canon", "canon"):
            chain_nodes.extend(int(idx) for idx in members.get(key, []))
        chain_nodes.extend(fx_nodes)
        chain_nodes = [int(idx) for idx in chain_nodes if int(idx) in resolved_positions]
        if len(chain_nodes) != 4:
            continue

        ordered_chain = sorted(
            chain_nodes,
            key=lambda idx: (
                resolved_positions[int(idx)].x,
                resolved_positions[int(idx)].y,
                resolved_positions[int(idx)].z,
            ),
        )

        candidate_points: List[Tuple[int, int, Vector]] = []
        seen_coords: set[Tuple[float, float, float]] = set()
        for source_idx, raw_points in off_mat_points_by_index.items():
            for block_idx, point in raw_points:
                if min((point - resolved_positions[int(idx)]).length for idx in ordered_chain) > 0.12:
                    continue
                coord_key = (round(point.x, 6), round(point.y, 6), round(point.z, 6))
                if coord_key in seen_coords:
                    continue
                seen_coords.add(coord_key)
                candidate_points.append((int(source_idx), int(block_idx), point.copy()))

        if len(candidate_points) < len(ordered_chain):
            continue

        candidate_points = sorted(
            candidate_points,
            key=lambda item: (item[2].x, item[2].y, item[2].z),
        )

        best_score = float("inf")
        best_seq: List[Tuple[int, int, Vector]] = []
        for combo in combinations(range(len(candidate_points)), len(ordered_chain)):
            seq = [candidate_points[i] for i in combo]
            score = 0.0
            valid = True
            for chain_i, idx in enumerate(ordered_chain):
                point = seq[chain_i][2]
                dist = float((point - resolved_positions[int(idx)]).length)
                if dist > 0.12:
                    valid = False
                    break
                score += dist
            if not valid:
                continue
            if score < best_score:
                best_score = float(score)
                best_seq = [(src_idx, block_idx, point.copy()) for src_idx, block_idx, point in seq]

        if not best_seq:
            continue

        for chain_i, idx in enumerate(ordered_chain):
            src_idx, block_idx, point = best_seq[chain_i]
            resolved_positions[int(idx)] = point.copy()
            _claim_off_mat_point(int(src_idx), int(block_idx))

    # Helper-only turret/smoke chains can expose only local off_mat deltas.
    # If their current placement collapses near the centerline, rebuild them
    # from own-node off_mat offsets relative to the resolved parent anchor.
    for suffix_num in sorted(chain_members_by_suffix.keys()):
        members = chain_members_by_suffix.get(int(suffix_num), {})
        tourelle_nodes = [int(idx) for idx in members.get("tourelle", [])]
        axe_nodes = [int(idx) for idx in members.get("axe_canon", [])]
        canon_nodes = [int(idx) for idx in members.get("canon", [])]
        fx_nodes = [int(idx) for idx in members.get("fx", [])]
        if len(tourelle_nodes) != 1 or len(axe_nodes) != 1 or len(canon_nodes) != 1 or not fx_nodes:
            continue
        t_idx = int(tourelle_nodes[0])
        a_idx = int(axe_nodes[0])
        c_idx = int(canon_nodes[0])
        helper_nodes = [t_idx, a_idx, c_idx, *fx_nodes]
        if any(mesh_by_bone_index.get(int(idx)) for idx in helper_nodes):
            continue
        parent_idx = int(bone_parent_by_index.get(int(t_idx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if parent_anchor is None:
            continue

        current_fx_positions = [
            resolved_positions.get(int(idx))
            for idx in fx_nodes
            if resolved_positions.get(int(idx)) is not None
        ]
        collapsed_fx = len(current_fx_positions) >= 2 and _max_pairwise_distance(current_fx_positions) <= 0.04
        unresolved_chain = False
        for helper_idx in helper_nodes:
            helper_pos = resolved_positions.get(int(helper_idx))
            if helper_pos is None:
                unresolved_chain = True
                break
            if parent_anchor.length > 0.05 and float(helper_pos.length) <= 1.0e-6:
                unresolved_chain = True
                break
        side_hint_from_fx = False
        for fx_idx in fx_nodes:
            for _block_idx, point in off_mat_points_by_index.get(int(fx_idx), []):
                if abs(float(point.y)) > 0.15:
                    side_hint_from_fx = True
                    break
            if side_hint_from_fx:
                break
        current_t = resolved_positions.get(int(t_idx))
        centerline_t = (
            current_t is not None
            and abs(float(current_t.y - parent_anchor.y)) <= 0.05
            and side_hint_from_fx
        )
        if not collapsed_fx and not centerline_t and not unresolved_chain:
            continue

        parent_obj = None
        parent_mesh_nodes = mesh_by_bone_index.get(int(parent_idx), [])
        if parent_mesh_nodes:
            parent_obj = parent_mesh_nodes[0]
        if parent_obj is None and int(parent_idx) in bone_name_by_index:
            parent_obj = mesh_by_name_low.get(_norm_low(str(bone_name_by_index.get(int(parent_idx), ""))))
        parent_bounds = None
        if parent_obj is not None:
            parent_mn, parent_mx, _parent_ctr = _bounds_world(parent_obj)
            parent_bounds = (parent_mn.copy(), parent_mx.copy())

        local_solution = _best_local_helper_chain_solution(
            parent_anchor.copy(),
            parent_bounds,
            t_idx,
            a_idx,
            c_idx,
            fx_nodes,
            ignore_reserved=False,
        )
        if not local_solution:
            continue

        t_src_idx, t_block_idx, t_point = local_solution["tourelle"]
        a_src_idx, a_block_idx, a_point = local_solution["axe"]
        c_src_idx, c_block_idx, c_point = local_solution["canon"]
        resolved_positions[int(t_idx)] = t_point.copy()
        resolved_positions[int(a_idx)] = a_point.copy()
        resolved_positions[int(c_idx)] = c_point.copy()
        _claim_off_mat_point(int(t_src_idx), int(t_block_idx))
        _claim_off_mat_point(int(a_src_idx), int(a_block_idx))
        _claim_off_mat_point(int(c_src_idx), int(c_block_idx))
        _promote_explicit_exact_world(int(t_idx), source_idx=int(t_src_idx), block_idx=int(t_block_idx), composition="helper_chain_local")
        _promote_explicit_exact_world(int(a_idx), source_idx=int(a_src_idx), block_idx=int(a_block_idx), composition="helper_chain_local")
        _promote_explicit_exact_world(int(c_idx), source_idx=int(c_src_idx), block_idx=int(c_block_idx), composition="helper_chain_local")
        for fx_src_idx, fx_block_idx, fx_idx, fx_point in local_solution.get("fx", []):
            resolved_positions[int(fx_idx)] = fx_point.copy()
            _claim_off_mat_point(int(fx_src_idx), int(fx_block_idx))
            _promote_explicit_exact_world(int(fx_idx), source_idx=int(fx_src_idx), block_idx=int(fx_block_idx), composition="helper_chain_local")
        off_mat_chain_resolved += 1

    # Late raw-local recovery for helper-only turret chains that still remained
    # unresolved at world origin after the main passes above.
    for suffix_num in sorted(chain_members_by_suffix.keys()):
        members = chain_members_by_suffix.get(int(suffix_num), {})
        tourelle_nodes = [int(idx) for idx in members.get("tourelle", [])]
        axe_nodes = [int(idx) for idx in members.get("axe_canon", [])]
        canon_nodes = [int(idx) for idx in members.get("canon", [])]
        fx_nodes = [int(idx) for idx in members.get("fx", [])]
        if len(tourelle_nodes) != 1 or len(axe_nodes) != 1 or len(canon_nodes) != 1 or not fx_nodes:
            continue
        t_idx = int(tourelle_nodes[0])
        a_idx = int(axe_nodes[0])
        c_idx = int(canon_nodes[0])
        helper_nodes = [t_idx, a_idx, c_idx, *fx_nodes]
        if any(mesh_by_bone_index.get(int(idx)) for idx in helper_nodes):
            continue
        parent_idx = int(bone_parent_by_index.get(int(t_idx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if parent_anchor is None:
            continue
        unresolved_chain = False
        for helper_idx in helper_nodes:
            helper_pos = resolved_positions.get(int(helper_idx))
            if helper_pos is None:
                unresolved_chain = True
                break
            if parent_anchor.length > 0.05 and float(helper_pos.length) <= 1.0e-6:
                unresolved_chain = True
                break
        if not unresolved_chain:
            continue
        parent_obj = None
        parent_mesh_nodes = mesh_by_bone_index.get(int(parent_idx), [])
        if parent_mesh_nodes:
            parent_obj = parent_mesh_nodes[0]
        if parent_obj is None and int(parent_idx) in bone_name_by_index:
            parent_obj = mesh_by_name_low.get(_norm_low(str(bone_name_by_index.get(int(parent_idx), ""))))
        parent_bounds = None
        if parent_obj is not None:
            parent_mn, parent_mx, _parent_ctr = _bounds_world(parent_obj)
            parent_bounds = (parent_mn.copy(), parent_mx.copy())
        local_solution = _best_local_helper_chain_solution(
            parent_anchor.copy(),
            parent_bounds,
            t_idx,
            a_idx,
            c_idx,
            fx_nodes,
            ignore_reserved=True,
        )
        if not local_solution:
            continue
        t_src_idx, t_block_idx, t_point = local_solution["tourelle"]
        a_src_idx, a_block_idx, a_point = local_solution["axe"]
        c_src_idx, c_block_idx, c_point = local_solution["canon"]
        resolved_positions[int(t_idx)] = t_point.copy()
        resolved_positions[int(a_idx)] = a_point.copy()
        resolved_positions[int(c_idx)] = c_point.copy()
        _claim_off_mat_point(int(t_src_idx), int(t_block_idx))
        _claim_off_mat_point(int(a_src_idx), int(a_block_idx))
        _claim_off_mat_point(int(c_src_idx), int(c_block_idx))
        _promote_explicit_exact_world(int(t_idx), source_idx=int(t_src_idx), block_idx=int(t_block_idx), composition="helper_chain_local_late")
        _promote_explicit_exact_world(int(a_idx), source_idx=int(a_src_idx), block_idx=int(a_block_idx), composition="helper_chain_local_late")
        _promote_explicit_exact_world(int(c_idx), source_idx=int(c_src_idx), block_idx=int(c_block_idx), composition="helper_chain_local_late")
        for fx_src_idx, fx_block_idx, fx_idx, fx_point in local_solution.get("fx", []):
            resolved_positions[int(fx_idx)] = fx_point.copy()
            _claim_off_mat_point(int(fx_src_idx), int(fx_block_idx))
            _promote_explicit_exact_world(int(fx_idx), source_idx=int(fx_src_idx), block_idx=int(fx_block_idx), composition="helper_chain_local_late")
        off_mat_chain_resolved += 1

    # Late raw-local recovery for single FX anchors that still stayed at origin.
    for bidx in ordered_indices:
        raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if not raw_low.startswith("fx_tourelle"):
            continue
        if mesh_by_bone_index.get(int(bidx)):
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if parent_anchor is None:
            continue
        current_pos = resolved_positions.get(int(bidx))
        if current_pos is not None and not (parent_anchor.length > 0.05 and float(current_pos.length) <= 1.0e-6):
            continue
        point_candidates = [
            point.copy()
            for _block_idx, point in off_mat_points_by_index.get(int(bidx), [])
            if point.length > 1.0e-9
        ]
        if not point_candidates:
            continue
        if any(float(point.length) >= 0.25 for point in point_candidates):
            filtered_candidates = [
                point.copy()
                for point in point_candidates
                if float(point.length) >= 0.15
                and (
                    abs(float(point.x)) >= 0.08
                    or abs(float(point.y)) >= 0.08
                    or abs(float(point.z)) >= 0.08
                )
            ]
            if filtered_candidates:
                point_candidates = filtered_candidates
        if len(point_candidates) == 1:
            local_point = point_candidates[0].copy()
        else:
            acc = Vector((0.0, 0.0, 0.0))
            for point in point_candidates:
                acc += point
            local_point = acc / float(len(point_candidates))
        resolved_positions[int(bidx)] = parent_anchor + local_point

    if settings is not None:
        for suffix_num in sorted(chain_members_by_suffix.keys()):
            members = chain_members_by_suffix.get(int(suffix_num), {})
            helper_nodes: List[int] = []
            for key in ("tourelle", "axe_canon", "canon", "fx"):
                helper_nodes.extend(int(idx) for idx in members.get(key, []))
            if not helper_nodes:
                continue
            if any(mesh_by_bone_index.get(int(idx)) for idx in helper_nodes):
                continue
            parent_idx = -1
            tou_nodes = [int(idx) for idx in members.get("tourelle", [])]
            if tou_nodes:
                parent_idx = int(bone_parent_by_index.get(int(tou_nodes[0]), -1))
            parent_anchor = resolved_positions.get(int(parent_idx))
            if parent_anchor is None or parent_anchor.length <= 0.05:
                continue
            unresolved = []
            for idx in helper_nodes:
                pos = resolved_positions.get(int(idx))
                if pos is None or float(pos.length) <= 1.0e-6:
                    unresolved.append(str(bone_name_by_index.get(int(idx), idx)))
            if unresolved:
                _warno_log(
                    settings,
                    (
                        f"helper chain unresolved suffix={int(suffix_num)} "
                        f"parent={str(bone_name_by_index.get(int(parent_idx), parent_idx))} "
                        f"nodes={','.join(unresolved)}"
                    ),
                    stage="armature",
                )

    standalone_anchor_targets = {"fx_munition"}
    for bidx in ordered_indices:
        low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if low not in standalone_anchor_targets:
            continue
        current_pos = resolved_positions.get(int(bidx))
        if current_pos is None:
            continue
        best_match = _best_nearest_off_mat_point(current_pos.copy())
        if best_match is None:
            continue
        src_idx2, block_idx2, point2 = best_match
        if float((point2 - current_pos).length) > 0.08:
            continue
        resolved_positions[int(bidx)] = point2.copy()
        _claim_off_mat_point(int(src_idx2), int(block_idx2))

    # If one track smoke exists and the mirrored side is unresolved, mirror by Y.
    def _find_bidx(prefix: str) -> int:
        for bidx, name in bone_name_by_index.items():
            if _norm_low(name).startswith(prefix):
                return int(bidx)
        return -1

    b_d = _find_bidx("fx_fumee_chenille_d")
    b_g = _find_bidx("fx_fumee_chenille_g")
    if b_d >= 0 and b_g >= 0:
        p_d = resolved_positions.get(int(b_d))
        p_g = resolved_positions.get(int(b_g))
        if p_d is not None and p_g is None:
            resolved_positions[int(b_g)] = Vector((p_d.x, -p_d.y, p_d.z))
        elif p_g is not None and p_d is None:
            resolved_positions[int(b_d)] = Vector((p_g.x, -p_g.y, p_g.z))

    # Some semantic/support helpers are already placed exactly by earlier off_mat
    # local recovery passes. Promote those placements before the strict semantic
    # resolver starts filtering unresolved helpers away.
    for bidx in ordered_indices:
        _promote_matching_off_mat_exact(int(bidx))

    for bidx in ordered_indices:
        transform_row = exact_local_transform_by_index.get(int(bidx))
        if isinstance(transform_row, dict):
            try:
                local_matrix = _compose_affine_components(
                    transform_row.get("local_translation", [0.0, 0.0, 0.0]),
                    transform_row.get("local_rotation_basis"),
                    transform_row.get("local_scale"),
                )
                if _set_resolved_world_from_local_matrix(int(bidx), local_matrix) is not None:
                    continue
            except Exception:
                pass
        pos = resolved_positions.get(int(bidx))
        if isinstance(pos, Vector):
            resolved_world_matrix_by_index[int(bidx)] = Matrix.Translation(pos.copy())

    for bidx in ordered_indices:
        raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if raw_low not in semantic_scene_nodes and raw_low not in semantic_support_nodes:
            continue
        pos = resolved_positions.get(int(bidx))
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if not _is_usable_world_point(pos, parent_anchor=parent_anchor):
            resolved_positions.pop(int(bidx), None)

    for bidx in ordered_indices:
        raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if raw_low not in semantic_scene_nodes:
            continue
        if mesh_by_bone_index.get(int(bidx)):
            continue
        if raw_low in {"armature", "papyrus", "fake"}:
            continue
        if raw_low.startswith("armature_"):
            continue
        if wheel_pattern.fullmatch(raw_low):
            continue
        role_name = str(_semantic_role_for_index(int(bidx)) or ("weapon_fx_anchor" if raw_low in gfx_fx_nodes else "subdepiction_anchor" if raw_low in gfx_subdepiction_nodes else "semantic_helper"))
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if semantic_helper_source_by_index.get(int(bidx), "") in {"gfx_exact", "spk_exact"}:
            if _is_usable_world_point(resolved_positions.get(int(bidx)), parent_anchor=parent_anchor):
                continue
        manifest_exact = _manifest_exact_transform_for_index(int(bidx))
        if manifest_exact is not None:
            if not isinstance(parent_anchor, Vector):
                semantic_helper_source_by_index[int(bidx)] = "skipped"
                semantic_helper_reason_by_index[int(bidx)] = "gfx_parent_unresolved"
                semantic_helpers_skipped += 1
                resolved_positions.pop(int(bidx), None)
                continue
            manifest_local = Vector(tuple(float(v) for v in manifest_exact.get("local_translation", [0.0, 0.0, 0.0])))
            _set_resolved_world_point(int(bidx), parent_anchor.copy() + manifest_local.copy())
            semantic_helper_source_by_index[int(bidx)] = "gfx_exact"
            semantic_helper_reason_by_index[int(bidx)] = str(manifest_exact.get("role", "") or role_name or "semantic_helper")
            semantic_helpers_exact += 1
            _register_exact_local_transform(
                int(bidx),
                role=str(manifest_exact.get("role", "") or role_name or "semantic_helper"),
                source_kind="gfx_exact",
                local_translation=manifest_local.copy(),
                local_rotation_basis=manifest_exact.get("local_rotation_basis"),
                local_scale=manifest_exact.get("local_scale"),
                provenance=dict(manifest_exact.get("provenance", {}) or {}),
            )
            try:
                manifest_local_matrix = _compose_affine_components(
                    manifest_local.copy(),
                    manifest_exact.get("local_rotation_basis"),
                    manifest_exact.get("local_scale"),
                )
                _set_resolved_world_from_local_matrix(int(bidx), manifest_local_matrix)
            except Exception:
                pass
            continue
        if isinstance(parent_anchor, Vector):
            single_local = _single_local_off_candidate(int(bidx), parent_anchor.copy())
            if single_local is not None:
                _set_resolved_world_point(int(bidx), single_local["world"].copy())
                _claim_off_mat_point(int(single_local.get("source_idx", int(bidx))), int(single_local.get("block_idx", -1)))
                semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                semantic_helper_reason_by_index[int(bidx)] = str(role_name or "semantic_helper")
                semantic_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(role_name or "semantic_helper"),
                    source_kind="spk_exact",
                    local_translation=single_local.get("local"),
                    provenance={
                        "source_node": str(bone_name_by_index.get(int(single_local.get("source_idx", int(bidx))), raw_low) or raw_low),
                        "source_index": int(single_local.get("source_idx", int(bidx))),
                        "block_idx": int(single_local.get("block_idx", -1)),
                        "layout": "",
                        "remap": "",
                        "composition": "single_local",
                        "parent_block_idx": -1,
                    },
                )
                continue
        best_candidate = _best_semantic_helper_candidate(int(bidx))
        if best_candidate is None:
            stream_exact = None
            if isinstance(parent_anchor, Vector):
                stream_exact = _stream_local_exact_candidate(
                    int(bidx),
                    role_override=str(role_name or "semantic_helper"),
                )
            if stream_exact is not None:
                local_matrix = stream_exact.get("local_matrix")
                world_point = None
                if isinstance(local_matrix, Matrix):
                    world_point = _set_resolved_world_from_local_matrix(int(bidx), local_matrix.copy())
                if world_point is None:
                    _set_resolved_world_point(int(bidx), stream_exact.get("world"))
                semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                semantic_helper_reason_by_index[int(bidx)] = str(stream_exact.get("role", role_name or "semantic_helper"))
                semantic_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(stream_exact.get("role", role_name or "semantic_helper")),
                    source_kind="spk_exact",
                    local_translation=stream_exact.get("local"),
                    local_rotation_basis=stream_exact.get("local_rotation_basis"),
                    local_scale=stream_exact.get("local_scale"),
                    provenance={
                        "source_node": str(stream_exact.get("source_name", "") or raw_low),
                        "source_index": int(stream_exact.get("source_idx", int(bidx))),
                        "block_idx": int(stream_exact.get("block_idx", -1)),
                        "layout": str(stream_exact.get("layout", "") or ""),
                        "remap": str(stream_exact.get("remap", "") or ""),
                        "composition": "stream_local",
                        "parent_block_idx": -1,
                    },
                )
                continue
            fallback_candidate = None
            if isinstance(parent_anchor, Vector):
                fallback_candidate = _semantic_fallback_candidate(
                    int(bidx),
                    parent_anchor.copy(),
                    role_name,
                )
            if fallback_candidate is not None:
                resolved_positions[int(bidx)] = fallback_candidate["world"].copy()
                semantic_helper_source_by_index[int(bidx)] = str(fallback_candidate.get("source", "") or "semantic_fallback")
                semantic_helper_reason_by_index[int(bidx)] = str(fallback_candidate.get("role", role_name or "semantic_helper"))
                semantic_helpers_approx += 1
                continue
            # Parent-anchor fallback: a helper that fails ALL transform sources still lands at
            # its parent (the antenna base sits on the hull) instead of collapsing to world
            # origin (the ikv_103 Antenne_01/02 "fallen empty" bug). Only fills nodes that
            # currently resolve to NOTHING, so it never moves an already-placed node.
            _pa = parent_anchor
            if not isinstance(_pa, Vector):
                _walk = int(bone_parent_by_index.get(int(bidx), -1)); _g = 0
                while _walk >= 0 and _g < 256:
                    _anc = resolved_positions.get(int(_walk))
                    if isinstance(_anc, Vector):
                        _pa = _anc; break
                    _walk = int(bone_parent_by_index.get(int(_walk), -1)); _g += 1
            if use_deterministic_raw_scene and _armature_is_rd and isinstance(_pa, Vector):
                resolved_positions[int(bidx)] = _pa.copy()
                semantic_helper_source_by_index[int(bidx)] = "parent_anchor_fallback"
                semantic_helper_reason_by_index[int(bidx)] = "semantic_no_own_candidate_parent_anchor"
                semantic_helpers_approx += 1
                continue
            semantic_helper_source_by_index[int(bidx)] = "skipped"
            semantic_helper_reason_by_index[int(bidx)] = "semantic_no_own_candidate"
            semantic_helpers_skipped += 1
            resolved_positions.pop(int(bidx), None)
            continue
        if not bool(best_candidate.get("exact")):
            stream_exact = None
            if isinstance(parent_anchor, Vector):
                stream_exact = _stream_local_exact_candidate(
                    int(bidx),
                    role_override=str(role_name or "semantic_helper"),
                )
            if stream_exact is not None:
                local_matrix = stream_exact.get("local_matrix")
                world_point = None
                if isinstance(local_matrix, Matrix):
                    world_point = _set_resolved_world_from_local_matrix(int(bidx), local_matrix.copy())
                if world_point is None:
                    _set_resolved_world_point(int(bidx), stream_exact.get("world"))
                semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                semantic_helper_reason_by_index[int(bidx)] = str(stream_exact.get("role", role_name or "semantic_helper"))
                semantic_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(stream_exact.get("role", role_name or "semantic_helper")),
                    source_kind="spk_exact",
                    local_translation=stream_exact.get("local"),
                    local_rotation_basis=stream_exact.get("local_rotation_basis"),
                    local_scale=stream_exact.get("local_scale"),
                    provenance={
                        "source_node": str(stream_exact.get("source_name", "") or raw_low),
                        "source_index": int(stream_exact.get("source_idx", int(bidx))),
                        "block_idx": int(stream_exact.get("block_idx", -1)),
                        "layout": str(stream_exact.get("layout", "") or ""),
                        "remap": str(stream_exact.get("remap", "") or ""),
                        "composition": "stream_local",
                        "parent_block_idx": -1,
                    },
                )
                continue
            fallback_candidate = None
            if isinstance(parent_anchor, Vector):
                fallback_candidate = _semantic_fallback_candidate(
                    int(bidx),
                    parent_anchor.copy(),
                    role_name,
                )
            if fallback_candidate is not None:
                resolved_positions[int(bidx)] = fallback_candidate["world"].copy()
                semantic_helper_source_by_index[int(bidx)] = str(fallback_candidate.get("source", "") or "semantic_fallback")
                semantic_helper_reason_by_index[int(bidx)] = str(fallback_candidate.get("role", role_name or "semantic_helper"))
                semantic_helpers_approx += 1
                continue
            # Parent-anchor fallback (same as the no-own-candidate branch): land at the
            # nearest resolved ancestor instead of world origin.
            _pa = parent_anchor
            if not isinstance(_pa, Vector):
                _walk = int(bone_parent_by_index.get(int(bidx), -1)); _g = 0
                while _walk >= 0 and _g < 256:
                    _anc = resolved_positions.get(int(_walk))
                    if isinstance(_anc, Vector):
                        _pa = _anc; break
                    _walk = int(bone_parent_by_index.get(int(_walk), -1)); _g += 1
            if use_deterministic_raw_scene and _armature_is_rd and isinstance(_pa, Vector):
                resolved_positions[int(bidx)] = _pa.copy()
                semantic_helper_source_by_index[int(bidx)] = "parent_anchor_fallback"
                semantic_helper_reason_by_index[int(bidx)] = "semantic_non_exact_parent_anchor"
                semantic_helpers_approx += 1
                continue
            semantic_helper_source_by_index[int(bidx)] = "skipped"
            semantic_helper_reason_by_index[int(bidx)] = "semantic_non_exact"
            semantic_helpers_skipped += 1
            resolved_positions.pop(int(bidx), None)
            continue
        if str(best_candidate.get("source", "")) in {"current_world", "global_world"}:
            semantic_helper_source_by_index[int(bidx)] = "skipped"
            semantic_helper_reason_by_index[int(bidx)] = f"forbidden_{str(best_candidate.get('source', '') or 'source')}"
            semantic_helpers_skipped += 1
            resolved_positions.pop(int(bidx), None)
            continue
        _set_resolved_world_point(int(bidx), best_candidate["world"].copy())
        if int(best_candidate.get("source_idx", -1)) >= 0 and int(best_candidate.get("block_idx", -1)) >= 0:
            _claim_off_mat_point(int(best_candidate.get("source_idx", -1)), int(best_candidate.get("block_idx", -1)))
        semantic_helper_source_by_index[int(bidx)] = "spk_exact"
        semantic_helpers_exact += 1
        semantic_helper_reason_by_index[int(bidx)] = str(best_candidate.get("role", "semantic_helper"))
        local_vec = best_candidate.get("local")
        if not isinstance(local_vec, Vector):
            if isinstance(parent_anchor, Vector):
                local_vec = best_candidate["world"].copy() - parent_anchor.copy()
            else:
                local_vec = Vector((0.0, 0.0, 0.0))
        _register_exact_local_transform(
            int(bidx),
            role=str(best_candidate.get("role", role_name or "semantic_helper")),
            source_kind="spk_exact",
            local_translation=local_vec.copy(),
            provenance={
                "source_node": str(best_candidate.get("source_name", "") or raw_low),
                "source_index": int(best_candidate.get("source_idx", int(bidx))),
                "block_idx": int(best_candidate.get("block_idx", -1)),
                "layout": "",
                "remap": "",
                "composition": str(best_candidate.get("source", "") or ""),
                "parent_block_idx": int(best_candidate.get("parent_block_idx", -1)),
            },
        )
        if settings is not None:
            _warno_log(
                settings,
                (
                    f"semantic_helper node={str(bone_name_by_index.get(int(bidx), ''))} "
                    f"role={str(best_candidate.get('role', 'semantic_helper'))} "
                    f"source={str(semantic_helper_source_by_index.get(int(bidx), ''))} "
                    f"score={float(best_candidate.get('score', 0.0)):.6f} "
                    f"margin={float(best_candidate.get('margin', 0.0)):.6f} "
                    f"confidence={str(best_candidate.get('confidence', 'low'))} "
                    f"source_node={str(best_candidate.get('source_name', ''))} "
                    f"variant={str(best_candidate.get('variant', ''))} "
                    f"world={tuple(round(float(v), 6) for v in best_candidate['world'])} "
                    f"local={tuple(round(float(v), 6) for v in best_candidate.get('local', Vector((0.0, 0.0, 0.0)))) if isinstance(best_candidate.get('local'), Vector) else None}"
                ),
                stage="semantic_solver",
            )

    for bidx in ordered_indices:
        raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if raw_low not in semantic_support_nodes:
            continue
        if mesh_by_bone_index.get(int(bidx)):
            continue
        if raw_low in {"armature", "papyrus", "fake"}:
            continue
        if raw_low.startswith("armature_"):
            continue
        if wheel_pattern.fullmatch(raw_low):
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if support_helper_source_by_index.get(int(bidx), "") in {"gfx_exact", "spk_exact"}:
            if _is_usable_world_point(resolved_positions.get(int(bidx)), parent_anchor=parent_anchor):
                continue
        manifest_exact = _manifest_exact_transform_for_index(int(bidx))
        if manifest_exact is not None:
            if not isinstance(parent_anchor, Vector):
                support_helper_source_by_index[int(bidx)] = "skipped"
                support_helper_reason_by_index[int(bidx)] = "gfx_parent_unresolved"
                support_helpers_skipped += 1
                resolved_positions.pop(int(bidx), None)
                continue
            manifest_local = Vector(tuple(float(v) for v in manifest_exact.get("local_translation", [0.0, 0.0, 0.0])))
            _set_resolved_world_point(int(bidx), parent_anchor.copy() + manifest_local.copy())
            support_helper_source_by_index[int(bidx)] = "gfx_exact"
            support_helper_reason_by_index[int(bidx)] = str(manifest_exact.get("role", "") or _support_role_for_index(int(bidx)) or "semantic_support")
            support_helpers_exact += 1
            _register_exact_local_transform(
                int(bidx),
                role=str(manifest_exact.get("role", "") or _support_role_for_index(int(bidx)) or "semantic_support"),
                source_kind="gfx_exact",
                local_translation=manifest_local.copy(),
                local_rotation_basis=manifest_exact.get("local_rotation_basis"),
                local_scale=manifest_exact.get("local_scale"),
                provenance=dict(manifest_exact.get("provenance", {}) or {}),
            )
            try:
                manifest_local_matrix = _compose_affine_components(
                    manifest_local.copy(),
                    manifest_exact.get("local_rotation_basis"),
                    manifest_exact.get("local_scale"),
                )
                _set_resolved_world_from_local_matrix(int(bidx), manifest_local_matrix)
            except Exception:
                pass
            continue
        if isinstance(parent_anchor, Vector):
            single_local = _single_local_off_candidate(int(bidx), parent_anchor.copy())
            if single_local is not None:
                _set_resolved_world_point(int(bidx), single_local["world"].copy())
                _claim_off_mat_point(int(single_local.get("source_idx", int(bidx))), int(single_local.get("block_idx", -1)))
                support_helper_source_by_index[int(bidx)] = "spk_exact"
                support_helper_reason_by_index[int(bidx)] = str(_support_role_for_index(int(bidx)) or "semantic_support")
                support_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(_support_role_for_index(int(bidx)) or "semantic_support"),
                    source_kind="spk_exact",
                    local_translation=single_local.get("local"),
                    provenance={
                        "source_node": str(bone_name_by_index.get(int(single_local.get("source_idx", int(bidx))), raw_low) or raw_low),
                        "source_index": int(single_local.get("source_idx", int(bidx))),
                        "block_idx": int(single_local.get("block_idx", -1)),
                        "layout": "",
                        "remap": "",
                        "composition": "single_local",
                        "parent_block_idx": -1,
                    },
                )
                continue
        best_candidate = _best_support_helper_candidate(int(bidx))
        if best_candidate is None:
            stream_exact = None
            if isinstance(parent_anchor, Vector):
                stream_exact = _stream_local_exact_candidate(
                    int(bidx),
                    role_override=str(_support_role_for_index(int(bidx)) or "semantic_support"),
                )
            if stream_exact is not None:
                local_matrix = stream_exact.get("local_matrix")
                world_point = None
                if isinstance(local_matrix, Matrix):
                    world_point = _set_resolved_world_from_local_matrix(int(bidx), local_matrix.copy())
                if world_point is None:
                    _set_resolved_world_point(int(bidx), stream_exact.get("world"))
                support_helper_source_by_index[int(bidx)] = "spk_exact"
                support_helper_reason_by_index[int(bidx)] = str(stream_exact.get("role", _support_role_for_index(int(bidx)) or "semantic_support"))
                support_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(stream_exact.get("role", _support_role_for_index(int(bidx)) or "semantic_support")),
                    source_kind="spk_exact",
                    local_translation=stream_exact.get("local"),
                    local_rotation_basis=stream_exact.get("local_rotation_basis"),
                    local_scale=stream_exact.get("local_scale"),
                    provenance={
                        "source_node": str(stream_exact.get("source_name", "") or raw_low),
                        "source_index": int(stream_exact.get("source_idx", int(bidx))),
                        "block_idx": int(stream_exact.get("block_idx", -1)),
                        "layout": str(stream_exact.get("layout", "") or ""),
                        "remap": str(stream_exact.get("remap", "") or ""),
                        "composition": "stream_local",
                        "parent_block_idx": -1,
                    },
                )
                continue
            support_helper_source_by_index[int(bidx)] = "skipped"
            support_helper_reason_by_index[int(bidx)] = "support_no_own_candidate"
            support_helpers_skipped += 1
            resolved_positions.pop(int(bidx), None)
            continue
        if not bool(best_candidate.get("exact")):
            stream_exact = None
            if isinstance(parent_anchor, Vector):
                stream_exact = _stream_local_exact_candidate(
                    int(bidx),
                    role_override=str(_support_role_for_index(int(bidx)) or "semantic_support"),
                )
            if stream_exact is not None:
                local_matrix = stream_exact.get("local_matrix")
                world_point = None
                if isinstance(local_matrix, Matrix):
                    world_point = _set_resolved_world_from_local_matrix(int(bidx), local_matrix.copy())
                if world_point is None:
                    _set_resolved_world_point(int(bidx), stream_exact.get("world"))
                support_helper_source_by_index[int(bidx)] = "spk_exact"
                support_helper_reason_by_index[int(bidx)] = str(stream_exact.get("role", _support_role_for_index(int(bidx)) or "semantic_support"))
                support_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(stream_exact.get("role", _support_role_for_index(int(bidx)) or "semantic_support")),
                    source_kind="spk_exact",
                    local_translation=stream_exact.get("local"),
                    local_rotation_basis=stream_exact.get("local_rotation_basis"),
                    local_scale=stream_exact.get("local_scale"),
                    provenance={
                        "source_node": str(stream_exact.get("source_name", "") or raw_low),
                        "source_index": int(stream_exact.get("source_idx", int(bidx))),
                        "block_idx": int(stream_exact.get("block_idx", -1)),
                        "layout": str(stream_exact.get("layout", "") or ""),
                        "remap": str(stream_exact.get("remap", "") or ""),
                        "composition": "stream_local",
                        "parent_block_idx": -1,
                    },
                )
                continue
            support_helper_source_by_index[int(bidx)] = "skipped"
            support_helper_reason_by_index[int(bidx)] = "support_non_exact"
            support_helpers_skipped += 1
            resolved_positions.pop(int(bidx), None)
            continue
        if str(best_candidate.get("source", "")) in {"current_world", "global_world"}:
            support_helper_source_by_index[int(bidx)] = "skipped"
            support_helper_reason_by_index[int(bidx)] = f"forbidden_{str(best_candidate.get('source', '') or 'source')}"
            support_helpers_skipped += 1
            resolved_positions.pop(int(bidx), None)
            continue
        _set_resolved_world_point(int(bidx), best_candidate["world"].copy())
        if int(best_candidate.get("source_idx", -1)) >= 0 and int(best_candidate.get("block_idx", -1)) >= 0:
            _claim_off_mat_point(int(best_candidate.get("source_idx", -1)), int(best_candidate.get("block_idx", -1)))
        support_helper_source_by_index[int(bidx)] = "spk_exact"
        support_helpers_exact += 1
        support_helper_reason_by_index[int(bidx)] = str(best_candidate.get("role", "semantic_support"))
        local_vec = best_candidate.get("local")
        if not isinstance(local_vec, Vector):
            if isinstance(parent_anchor, Vector):
                local_vec = best_candidate["world"].copy() - parent_anchor.copy()
            else:
                local_vec = Vector((0.0, 0.0, 0.0))
        _register_exact_local_transform(
            int(bidx),
            role=str(best_candidate.get("role", _support_role_for_index(int(bidx)) or "semantic_support")),
            source_kind="spk_exact",
            local_translation=local_vec.copy(),
            provenance={
                "source_node": str(best_candidate.get("source_name", "") or raw_low),
                "source_index": int(best_candidate.get("source_idx", int(bidx))),
                "block_idx": int(best_candidate.get("block_idx", -1)),
                "layout": "",
                "remap": "",
                "composition": str(best_candidate.get("source", "") or ""),
                "parent_block_idx": int(best_candidate.get("parent_block_idx", -1)),
            },
        )
        if settings is not None:
            _warno_log(
                settings,
                (
                    f"support_helper node={str(bone_name_by_index.get(int(bidx), ''))} "
                    f"role={str(best_candidate.get('role', 'semantic_support'))} "
                    f"source={str(support_helper_source_by_index.get(int(bidx), ''))} "
                    f"score={float(best_candidate.get('score', 0.0)):.6f} "
                    f"margin={float(best_candidate.get('margin', 0.0)):.6f} "
                    f"confidence={str(best_candidate.get('confidence', 'low'))} "
                    f"source_node={str(best_candidate.get('source_name', ''))} "
                    f"variant={str(best_candidate.get('variant', ''))} "
                    f"world={tuple(round(float(v), 6) for v in best_candidate['world'])} "
                    f"local={tuple(round(float(v), 6) for v in best_candidate.get('local', Vector((0.0, 0.0, 0.0)))) if isinstance(best_candidate.get('local'), Vector) else None}"
                ),
                stage="semantic_solver",
            )

    def _best_helper_chain_own_local(
        idx: int,
        mode: str,
    ) -> Tuple[int, Vector] | None:
        candidates = [
            (int(block_idx), point.copy())
            for block_idx, point in off_mat_points_by_index.get(int(idx), [])
            if isinstance(point, Vector) and float(point.length) > 1.0e-9
        ]
        if not candidates:
            return None
        best_row: Tuple[float, int, Vector] | None = None
        for block_idx, local_point in candidates:
            x = float(local_point.x)
            y = float(local_point.y)
            z = float(local_point.z)
            abs_x = abs(x)
            abs_y = abs(y)
            abs_z = abs(z)
            if mode == "yaw":
                if abs_y < 0.2 or abs_x > 0.75 or abs_z > 1.35:
                    continue
                score = abs(abs_y - 0.45) + abs_x * 0.6 + abs_z * 0.35
            else:
                if abs_x < 0.02 or abs_x > 1.25 or abs_y > 0.12 or abs_z > 0.12:
                    continue
                score = abs(abs_x - 0.22) + abs_y * 4.0 + abs_z * 4.0
            cand = (float(score), int(block_idx), local_point.copy())
            if best_row is None or cand[0] < best_row[0]:
                best_row = cand
        if best_row is None:
            return None
        return int(best_row[1]), best_row[2].copy()

    for suffix_num in sorted(chain_members_by_suffix.keys()):
        members = chain_members_by_suffix.get(int(suffix_num), {})
        tourelle_nodes = [int(idx) for idx in members.get("tourelle", [])]
        axe_nodes = [int(idx) for idx in members.get("axe_canon", [])]
        canon_nodes = [int(idx) for idx in members.get("canon", [])]
        fx_nodes = [int(idx) for idx in members.get("fx", [])]
        if len(tourelle_nodes) != 1 or len(axe_nodes) != 1 or len(canon_nodes) != 1 or not fx_nodes:
            continue
        t_idx = int(tourelle_nodes[0])
        a_idx = int(axe_nodes[0])
        c_idx = int(canon_nodes[0])
        if mesh_by_bone_index.get(int(t_idx)) or mesh_by_bone_index.get(int(a_idx)) or mesh_by_bone_index.get(int(c_idx)):
            continue
        parent_idx = int(bone_parent_by_index.get(int(t_idx), -1))
        parent_anchor = resolved_positions.get(int(parent_idx))
        if not isinstance(parent_anchor, Vector):
            continue
        current_t = resolved_positions.get(int(t_idx))
        if isinstance(current_t, Vector):
            current_t_local = current_t.copy() - parent_anchor.copy()
            if abs(float(current_t_local.y)) >= 0.25:
                continue
        else:
            current_t_local = None
        if current_t_local is None and support_helper_source_by_index.get(int(t_idx), "") in {"gfx_exact", "spk_exact"}:
            continue
        t_pick = _best_helper_chain_own_local(int(t_idx), "yaw")
        a_pick = _best_helper_chain_own_local(int(a_idx), "linear")
        c_pick = _best_helper_chain_own_local(int(c_idx), "linear")
        if t_pick is None or a_pick is None or c_pick is None:
            continue
        t_block_idx, t_local = t_pick
        a_block_idx, a_local = a_pick
        c_block_idx, c_local = c_pick
        resolved_positions[int(t_idx)] = parent_anchor.copy() + t_local.copy()
        resolved_positions[int(a_idx)] = resolved_positions[int(t_idx)].copy() + a_local.copy()
        resolved_positions[int(c_idx)] = resolved_positions[int(a_idx)].copy() + c_local.copy()
        _claim_off_mat_point(int(t_idx), int(t_block_idx))
        _claim_off_mat_point(int(a_idx), int(a_block_idx))
        _claim_off_mat_point(int(c_idx), int(c_block_idx))
        support_helper_source_by_index[int(t_idx)] = "spk_exact"
        support_helper_source_by_index[int(a_idx)] = "spk_exact"
        support_helper_source_by_index[int(c_idx)] = "spk_exact"
        support_helper_reason_by_index[int(t_idx)] = "turret_yaw_node"
        support_helper_reason_by_index[int(a_idx)] = "turret_pitch_node"
        support_helper_reason_by_index[int(c_idx)] = "turret_recoil_node"
        _register_exact_local_transform(
            int(t_idx),
            role="turret_yaw_node",
            source_kind="spk_exact",
            local_translation=t_local.copy(),
            provenance={
                "source_node": str(bone_name_by_index.get(int(t_idx), "") or ""),
                "source_index": int(t_idx),
                "block_idx": int(t_block_idx),
                "layout": "",
                "remap": "",
                "composition": "own_local_chain:yaw",
                "parent_block_idx": -1,
            },
        )
        _register_exact_local_transform(
            int(a_idx),
            role="turret_pitch_node",
            source_kind="spk_exact",
            local_translation=a_local.copy(),
            provenance={
                "source_node": str(bone_name_by_index.get(int(a_idx), "") or ""),
                "source_index": int(a_idx),
                "block_idx": int(a_block_idx),
                "layout": "",
                "remap": "",
                "composition": "own_local_chain:linear",
                "parent_block_idx": -1,
            },
        )
        _register_exact_local_transform(
            int(c_idx),
            role="turret_recoil_node",
            source_kind="spk_exact",
            local_translation=c_local.copy(),
            provenance={
                "source_node": str(bone_name_by_index.get(int(c_idx), "") or ""),
                "source_index": int(c_idx),
                "block_idx": int(c_block_idx),
                "layout": "",
                "remap": "",
                "composition": "own_local_chain:linear",
                "parent_block_idx": -1,
            },
        )

    for suffix_num in sorted(chain_members_by_suffix.keys()):
        members = chain_members_by_suffix.get(int(suffix_num), {})
        tourelle_nodes = [int(idx) for idx in members.get("tourelle", [])]
        axe_nodes = [int(idx) for idx in members.get("axe_canon", [])]
        canon_nodes = [int(idx) for idx in members.get("canon", [])]
        if len(tourelle_nodes) != 1 or len(axe_nodes) != 1 or len(canon_nodes) != 1:
            continue
        t_idx = int(tourelle_nodes[0])
        a_idx = int(axe_nodes[0])
        c_idx = int(canon_nodes[0])
        if mesh_by_bone_index.get(int(t_idx)) or mesh_by_bone_index.get(int(a_idx)) or mesh_by_bone_index.get(int(c_idx)):
            continue
        if not _support_chain_needs_stream_repair(int(t_idx), int(a_idx), int(c_idx)):
            continue
        stream_chain = _best_stream_support_chain_solution(int(t_idx), int(a_idx), int(c_idx))
        if not isinstance(stream_chain, dict):
            continue
        for idx, key, role_name in (
            (int(t_idx), "tourelle", "turret_yaw_node"),
            (int(a_idx), "axe", "turret_pitch_node"),
            (int(c_idx), "canon", "turret_recoil_node"),
        ):
            row = stream_chain.get(key)
            if not isinstance(row, dict):
                continue
            prev_source = str(support_helper_source_by_index.get(int(idx), "") or "")
            if prev_source == "skipped":
                support_helpers_skipped = max(0, support_helpers_skipped - 1)
                support_helpers_exact += 1
            elif prev_source not in {"gfx_exact", "spk_exact"}:
                support_helpers_exact += 1
            world_point = row.get("world")
            world_matrix = row.get("world_matrix")
            if isinstance(world_point, Vector):
                resolved_positions[int(idx)] = world_point.copy()
            if isinstance(world_matrix, Matrix):
                resolved_world_matrix_by_index[int(idx)] = world_matrix.copy()
            support_helper_source_by_index[int(idx)] = "spk_exact"
            support_helper_reason_by_index[int(idx)] = str(role_name)
            _register_exact_local_transform(
                int(idx),
                role=str(role_name),
                source_kind="spk_exact",
                local_translation=row.get("local"),
                local_rotation_basis=row.get("local_rotation_basis"),
                local_scale=row.get("local_scale"),
                provenance={
                    "source_node": str(row.get("source_name", "") or _norm_low(str(bone_name_by_index.get(int(idx), "") or ""))),
                    "source_index": int(row.get("source_idx", int(idx))),
                    "block_idx": int(row.get("block_idx", -1)),
                    "layout": str(row.get("layout", "") or ""),
                    "remap": str(row.get("remap", "") or ""),
                    "composition": "stream_local",
                    "parent_block_idx": -1,
                },
            )
            _claim_off_mat_point(int(row.get("source_idx", int(idx))), int(row.get("block_idx", -1)))

    for bidx in ordered_indices:
        raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if raw_low not in semantic_scene_nodes:
            continue
        if mesh_by_bone_index.get(int(bidx)):
            continue
        role_name = str(
            _semantic_role_for_index(int(bidx))
            or ("weapon_fx_anchor" if raw_low in gfx_fx_nodes else "subdepiction_anchor" if raw_low in gfx_subdepiction_nodes else "semantic_helper")
        )
        if role_name != "weapon_fx_anchor":
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent_raw_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        if parent_raw_low not in semantic_support_nodes:
            continue
        if support_helper_source_by_index.get(int(parent_idx), "") not in {"gfx_exact", "spk_exact"}:
            continue
        parent_anchor = resolved_positions.get(int(parent_idx))
        if not isinstance(parent_anchor, Vector):
            continue
        parent_exact_row = exact_local_transform_by_index.get(int(parent_idx))
        parent_composition = ""
        if isinstance(parent_exact_row, dict):
            parent_provenance = parent_exact_row.get("provenance")
            if isinstance(parent_provenance, dict):
                parent_composition = str(parent_provenance.get("composition", "") or "").strip().lower()
        current_exact_row = exact_local_transform_by_index.get(int(bidx))
        current_source_kind = str(semantic_helper_source_by_index.get(int(bidx), "") or "")
        current_composition = ""
        if isinstance(current_exact_row, dict):
            current_provenance = current_exact_row.get("provenance")
            if isinstance(current_provenance, dict):
                current_composition = str(current_provenance.get("composition", "") or "").strip().lower()
        if (
            current_source_kind in {"gfx_exact", "spk_exact"}
            and isinstance(current_exact_row, dict)
            and _is_usable_world_point(resolved_positions.get(int(bidx)), parent_anchor=parent_anchor)
            and not (
                parent_composition == "stream_local"
                and current_composition != "stream_local"
            )
        ):
            continue

        prev_source = str(semantic_helper_source_by_index.get(int(bidx), "") or "")
        if prev_source in {"gfx_exact", "spk_exact"}:
            semantic_helpers_exact = max(0, semantic_helpers_exact - 1)
        elif prev_source in {"global_world", "mesh_current", "role_template"}:
            semantic_helpers_approx = max(0, semantic_helpers_approx - 1)
        elif prev_source == "skipped":
            semantic_helpers_skipped = max(0, semantic_helpers_skipped - 1)
        exact_local_transform_by_index.pop(int(bidx), None)

        if parent_composition == "stream_local":
            stream_exact = _stream_local_exact_candidate(
                int(bidx),
                role_override=str(role_name or "weapon_fx_anchor"),
            )
            if stream_exact is not None:
                local_matrix = stream_exact.get("local_matrix")
                world_point = None
                if isinstance(local_matrix, Matrix):
                    world_point = _set_resolved_world_from_local_matrix(int(bidx), local_matrix.copy())
                if world_point is None:
                    _set_resolved_world_point(int(bidx), stream_exact.get("world"))
                semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                semantic_helper_reason_by_index[int(bidx)] = str(stream_exact.get("role", role_name or "weapon_fx_anchor"))
                semantic_helpers_exact += 1
                _register_exact_local_transform(
                    int(bidx),
                    role=str(stream_exact.get("role", role_name or "weapon_fx_anchor")),
                    source_kind="spk_exact",
                    local_translation=stream_exact.get("local"),
                    local_rotation_basis=stream_exact.get("local_rotation_basis"),
                    local_scale=stream_exact.get("local_scale"),
                    provenance={
                        "source_node": str(stream_exact.get("source_name", "") or raw_low),
                        "source_index": int(stream_exact.get("source_idx", int(bidx))),
                        "block_idx": int(stream_exact.get("block_idx", -1)),
                        "layout": str(stream_exact.get("layout", "") or ""),
                        "remap": str(stream_exact.get("remap", "") or ""),
                        "composition": "stream_local",
                        "parent_block_idx": -1,
                    },
                )
                continue

        best_candidate = _best_semantic_helper_candidate(int(bidx))
        if best_candidate is not None and bool(best_candidate.get("exact")):
            if str(best_candidate.get("source", "")) in {"current_world", "global_world"}:
                best_candidate = None
            else:
                resolved_positions[int(bidx)] = best_candidate["world"].copy()
                if int(best_candidate.get("source_idx", -1)) >= 0 and int(best_candidate.get("block_idx", -1)) >= 0:
                    _claim_off_mat_point(int(best_candidate.get("source_idx", -1)), int(best_candidate.get("block_idx", -1)))
                semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                semantic_helper_reason_by_index[int(bidx)] = str(best_candidate.get("role", role_name or "semantic_helper"))
                semantic_helpers_exact += 1
                local_vec = best_candidate.get("local")
                if not isinstance(local_vec, Vector):
                    local_vec = best_candidate["world"].copy() - parent_anchor.copy()
                _register_exact_local_transform(
                    int(bidx),
                    role=str(best_candidate.get("role", role_name or "semantic_helper")),
                    source_kind="spk_exact",
                    local_translation=local_vec.copy(),
                    provenance={
                        "source_node": str(best_candidate.get("source_name", "") or raw_low),
                        "source_index": int(best_candidate.get("source_idx", int(bidx))),
                        "block_idx": int(best_candidate.get("block_idx", -1)),
                        "layout": "",
                        "remap": "",
                        "composition": str(best_candidate.get("source", "") or ""),
                        "parent_block_idx": int(best_candidate.get("parent_block_idx", -1)),
                    },
                )
                continue

        fallback_candidate = _semantic_fallback_candidate(
            int(bidx),
            parent_anchor.copy(),
            role_name,
        )
        if fallback_candidate is not None:
            resolved_positions[int(bidx)] = fallback_candidate["world"].copy()
            semantic_helper_source_by_index[int(bidx)] = str(fallback_candidate.get("source", "") or "semantic_fallback")
            semantic_helper_reason_by_index[int(bidx)] = str(fallback_candidate.get("role", role_name or "semantic_helper"))
            semantic_helpers_approx += 1
            continue

        semantic_helper_source_by_index[int(bidx)] = "skipped"
        semantic_helper_reason_by_index[int(bidx)] = "semantic_support_parent_refresh_failed"
        semantic_helpers_skipped += 1
        resolved_positions.pop(int(bidx), None)

    for canon_idx in ordered_indices:
        canon_low = _norm_low(str(bone_name_by_index.get(int(canon_idx), "")))
        if not re.fullmatch(r"canon_[0-9]+", canon_low):
            continue
        canon_exact_row = exact_local_transform_by_index.get(int(canon_idx))
        canon_composition = ""
        if isinstance(canon_exact_row, dict):
            canon_provenance = canon_exact_row.get("provenance")
            if isinstance(canon_provenance, dict):
                canon_composition = str(canon_provenance.get("composition", "") or "").strip().lower()
        if canon_composition == "stream_local":
            continue
        parent_idx = int(bone_parent_by_index.get(int(canon_idx), -1))
        grandparent_idx = int(bone_parent_by_index.get(int(parent_idx), -1))
        if (
            parent_idx < 0
            or grandparent_idx < 0
            or mesh_by_bone_index.get(int(canon_idx))
            or mesh_by_bone_index.get(int(parent_idx))
            or mesh_by_bone_index.get(int(grandparent_idx))
        ):
            continue
        fx_children: List[Tuple[int, int]] = []
        for child_idx in _child_bone_indices(int(canon_idx)):
            child_low = _norm_low(str(bone_name_by_index.get(int(child_idx), "")))
            m_fx = re.match(r"^fx_tourelle([0-9]+)_tir_([0-9]+)$", child_low, re.IGNORECASE)
            if not m_fx:
                continue
            if child_low not in semantic_scene_nodes:
                continue
            fx_children.append((int(child_idx), int(m_fx.group(2))))
        if len(fx_children) < 2:
            continue
        ordered_fx_children = sorted(fx_children, key=lambda item: int(item[1]))
        rows_by_child: Dict[int, List[Dict[str, Any]]] = {}
        for child_idx, _tir_num in ordered_fx_children:
            rows = _global_semantic_candidates(int(child_idx), role_override="weapon_fx_anchor")
            if rows:
                rows_by_child[int(child_idx)] = rows[:8]
        if len(rows_by_child) != len(ordered_fx_children):
            continue
        best_combo: List[Tuple[int, Dict[str, Any]]] | None = None
        best_score = float("inf")
        if len(ordered_fx_children) == 2:
            left_idx = int(ordered_fx_children[0][0])
            right_idx = int(ordered_fx_children[1][0])
            for left_row in rows_by_child.get(int(left_idx), []):
                for right_row in rows_by_child.get(int(right_idx), []):
                    world_points = [left_row["world"].copy(), right_row["world"].copy()]
                    sep = _max_pairwise_distance(world_points)
                    combo_score = float(left_row.get("score", 999.0)) + float(right_row.get("score", 999.0))
                    if (
                        int(left_row.get("source_idx", -1)) == int(right_row.get("source_idx", -2))
                        and int(left_row.get("block_idx", -1)) == int(right_row.get("block_idx", -2))
                    ):
                        combo_score += 10.0
                    if sep < 0.08:
                        combo_score += (0.08 - sep) * 80.0
                    if combo_score < best_score:
                        best_score = float(combo_score)
                        best_combo = [
                            (int(left_idx), dict(left_row)),
                            (int(right_idx), dict(right_row)),
                        ]
        if not best_combo:
            continue
        combo_points = [row["world"].copy() for _child_idx, row in best_combo]
        combo_sep = _max_pairwise_distance(combo_points)
        if combo_sep < 0.08 and len(best_combo) == 2:
            parent_anchor = resolved_positions.get(int(canon_idx))
            if isinstance(parent_anchor, Vector):
                shared_local_y = sum(float(row.get("local", Vector((0.0, 0.0, 0.0))).y) for _child_idx, row in best_combo) / 2.0
                shared_local_z = sum(float(row.get("local", Vector((0.0, 0.0, 0.0))).z) for _child_idx, row in best_combo) / 2.0
                for combo_i, (child_idx, row) in enumerate(best_combo):
                    own_x_candidates = []
                    for _block_idx, point in off_mat_points_by_index.get(int(child_idx), []):
                        if not isinstance(point, Vector) or float(point.length) <= 1.0e-9:
                            continue
                        if abs(float(point.y)) < 0.8:
                            continue
                        own_x_candidates.append(float(point.x))
                    if not own_x_candidates:
                        continue
                    pos_x = [value for value in own_x_candidates if value > 0.02]
                    neg_x = [value for value in own_x_candidates if value < -0.02]
                    if pos_x and not neg_x:
                        x_hint = sum(pos_x) / float(len(pos_x))
                    elif neg_x and not pos_x:
                        x_hint = sum(neg_x) / float(len(neg_x))
                    elif len(pos_x) > len(neg_x):
                        x_hint = sum(pos_x) / float(len(pos_x))
                    elif len(neg_x) > len(pos_x):
                        x_hint = sum(neg_x) / float(len(neg_x))
                    else:
                        x_hint = sum(own_x_candidates) / float(len(own_x_candidates))
                    local_vec = Vector((float(x_hint), float(shared_local_y), float(shared_local_z)))
                    row["local"] = local_vec.copy()
                    row["world"] = parent_anchor.copy() + local_vec.copy()
        for child_idx, chosen_row in best_combo:
            prev_source = str(semantic_helper_source_by_index.get(int(child_idx), "") or "")
            if prev_source in {"gfx_exact", "spk_exact"}:
                semantic_helpers_exact = max(0, semantic_helpers_exact - 1)
            elif prev_source in {"global_world", "mesh_current", "role_template"}:
                semantic_helpers_approx = max(0, semantic_helpers_approx - 1)
            elif prev_source == "skipped":
                semantic_helpers_skipped = max(0, semantic_helpers_skipped - 1)
            exact_local_transform_by_index.pop(int(child_idx), None)
            resolved_positions[int(child_idx)] = chosen_row["world"].copy()
            semantic_helper_source_by_index[int(child_idx)] = "global_world"
            semantic_helper_reason_by_index[int(child_idx)] = str(chosen_row.get("role", "weapon_fx_anchor"))
            semantic_helpers_approx += 1

    track_vertex_groups_by_side: Dict[str, set[str]] = {"left": set(), "right": set()}
    for obj in imported_objects:
        if obj.type != "MESH":
            continue
        low = _norm_low(obj.name)
        side = ""
        if "chenille_gauche" in low:
            side = "left"
        elif "chenille_droite" in low:
            side = "right"
        if not side:
            continue
        try:
            for vg in obj.vertex_groups:
                vg_low = _norm_low(getattr(vg, "name", ""))
                if vg_low.startswith("roue_"):
                    track_vertex_groups_by_side.setdefault(side, set()).add(vg_low)
        except Exception:
            continue

    deterministic_exact_indices: set[int] = set()
    deterministic_reason_by_index: Dict[int, str] = {}
    if raw_scene_graph is not None:
        raw_records = getattr(raw_scene_graph, "records", None)
        if isinstance(raw_records, list):
            pending_local_rows: Dict[int, Dict[str, Any]] = {}
            for record in raw_records:
                try:
                    bidx = int(getattr(record, "index", -1))
                except Exception:
                    continue
                if bidx not in bone_name_by_index:
                    continue
                parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
                local_transform = getattr(record, "local_transform", None)
                world_translation = getattr(record, "world_translation", None)
                if local_transform is not None:
                    try:
                        local_translation = [float(v) for v in list(getattr(local_transform, "translation", []) or [])[:3]]
                        local_basis = [
                            [float(v) for v in row[:3]]
                            for row in list(getattr(local_transform, "rotation_basis", []) or [])[:3]
                        ]
                        local_scale = [float(v) for v in list(getattr(local_transform, "scale", []) or [])[:3]]
                        pending_local_rows[int(bidx)] = {
                            "role": "raw_scene_graph",
                            "source_kind": "spk_exact",
                            "local_translation": local_translation,
                            "local_rotation_basis": local_basis,
                            "local_scale": local_scale,
                            "provenance": {
                                "source": str(getattr(local_transform, "source", "") or ""),
                                "block_index": int(getattr(local_transform, "block_index", -1)),
                            },
                        }
                        deterministic_reason_by_index[int(bidx)] = str(getattr(local_transform, "source", "") or "spk_exact")
                    except Exception:
                        pass
                if isinstance(world_translation, (list, tuple)) and len(world_translation) >= 3:
                    try:
                        world_point = Vector(
                            (
                                float(world_translation[0]),
                                float(world_translation[1]),
                                float(world_translation[2]),
                            )
                        )
                    except Exception:
                        world_point = None
                    if isinstance(world_point, Vector) and all(math.isfinite(float(v)) for v in world_point):
                        _set_resolved_world_point(int(bidx), world_point.copy())
                        deterministic_exact_indices.add(int(bidx))
                        semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                        support_helper_source_by_index[int(bidx)] = "spk_exact"
                        deterministic_reason_by_index.setdefault(int(bidx), "world_point")
            for _ in range(max(1, len(pending_local_rows) + 1)):
                changed = False
                for bidx, row in list(pending_local_rows.items()):
                    parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
                    if parent_idx >= 0 and int(parent_idx) not in resolved_world_matrix_by_index:
                        continue
                    try:
                        local_matrix = _compose_affine_components(
                            row.get("local_translation", [0.0, 0.0, 0.0]),
                            row.get("local_rotation_basis"),
                            row.get("local_scale"),
                        )
                    except Exception:
                        pending_local_rows.pop(int(bidx), None)
                        continue
                    world_point = _set_resolved_world_from_local_matrix(int(bidx), local_matrix)
                    if world_point is None and parent_idx < 0:
                        try:
                            _set_resolved_world_point(int(bidx), local_matrix.translation.copy())
                            world_point = local_matrix.translation.copy()
                        except Exception:
                            world_point = None
                    if world_point is None:
                        continue
                    _register_exact_local_transform(
                        int(bidx),
                        role="raw_scene_graph",
                        source_kind="spk_exact",
                        local_translation=row.get("local_translation", [0.0, 0.0, 0.0]),
                        local_rotation_basis=row.get("local_rotation_basis"),
                        local_scale=row.get("local_scale"),
                        provenance=row.get("provenance"),
                    )
                    deterministic_exact_indices.add(int(bidx))
                    semantic_helper_source_by_index[int(bidx)] = "spk_exact"
                    support_helper_source_by_index[int(bidx)] = "spk_exact"
                    changed = True
                    pending_local_rows.pop(int(bidx), None)
                if not changed:
                    break

    track_deform_nodes = {
        _norm_low(name)
        for side_names in track_vertex_groups_by_side.values()
        for name in side_names
        if str(name).strip()
    }
    wheel_pattern = re.compile(r"^roue_elev_([dg])([0-9]+)$", re.IGNORECASE)

    def _node_policy(bidx: int, has_mesh: bool) -> Tuple[str, str]:
        raw_low = _norm_low(str(display_name_by_index.get(int(bidx), "")))
        if not raw_low:
            return "skipped_unresolved", "blank_name"
        if has_mesh:
            return "visible_mesh", "mesh_geometry"
        if raw_low in {"armature", "papyrus", "fake"}:
            if semantic_mode == "RAW_DEBUG":
                return "visible_helper", "raw_debug_carrier"
            return "ignored_raw_carrier", "raw_carrier"
        if semantic_mode != "RAW_DEBUG":
            if re.fullmatch(r"^empty(?:\.[0-9]+)?$", raw_low, flags=re.IGNORECASE):
                return "skipped_unresolved", "raw_empty_omitted"
            if raw_low.startswith("cylinder."):
                return "skipped_unresolved", "raw_cylinder_omitted"
            if raw_low.startswith("bip01"):
                return "skipped_unresolved", "character_bone_omitted"
            if papyrus_character_proxy and raw_low == "soldat":
                return "skipped_unresolved", "character_mesh_proxy"
        if raw_low.startswith("armature_"):
            return "hidden_deform_helper", "track_carrier"
        if wheel_pattern.fullmatch(raw_low):
            if semantic_mode == "RAW_DEBUG":
                return "visible_helper", "raw_debug_wheel_helper"
            return "visible_helper", "track_wheel_helper"
        if semantic_mode == "RAW_DEBUG":
            return "visible_helper", "raw_debug"
        if int(bidx) in deterministic_exact_indices:
            return "visible_helper", deterministic_reason_by_index.get(int(bidx), "spk_exact")
        return "skipped_unresolved", "deterministic_miss"

    node_policy_by_index: Dict[int, Tuple[str, str]] = {}
    for bidx in ordered_indices:
        node_policy_by_index[int(bidx)] = _node_policy(
            int(bidx),
            bool(mesh_by_bone_index.get(int(bidx), [])),
        )
    node_policy_counts: Dict[str, int] = defaultdict(int)
    for policy, _reason in node_policy_by_index.values():
        node_policy_counts[str(policy)] += 1
    if settings is not None:
        for bidx in ordered_indices:
            raw_name = str(display_name_by_index.get(int(bidx), "") or "").strip()
            if not raw_name:
                continue
            policy, reason = node_policy_by_index.get(int(bidx), ("skipped_unresolved", "policy_missing"))
            raw_low = _norm_low(raw_name)
            if raw_low in {"armature", "papyrus", "fake"}:
                source_tag = "spk"
            elif raw_low in track_deform_nodes:
                source_tag = "weights"
            elif int(bidx) in deterministic_exact_indices:
                source_tag = deterministic_reason_by_index.get(int(bidx), "spk_exact")
            else:
                source_tag = "skipped"
            _warno_log(
                settings,
                f"node={raw_name} policy={policy} source={source_tag} reason={reason}",
                stage="node_policy",
            )

    used_object_names: set[str] = {_norm_low(o.name) for o in bpy.data.objects}
    node_by_bone_index: Dict[int, bpy.types.Object] = {}
    created_empties = 0

    def _character_bone_name(raw_name: str) -> str:
        tokens = [tok for tok in re.split(r"[\s_]+", str(raw_name or "").strip()) if tok]
        if not tokens:
            return "Bip01"
        out: List[str] = []
        for token in tokens:
            low = token.lower()
            if low == "bip01":
                out.append("Bip01")
            elif len(token) == 1 and low in {"l", "r"}:
                out.append(token.upper())
            elif low == "upperarm":
                out.append("UpperArm")
            else:
                out.append(token[:1].upper() + token[1:])
        return " ".join(out)

    papyrus_bone_name_by_index: Dict[int, str] = {}
    papyrus_armature_obj: bpy.types.Object | None = None
    if papyrus_character_proxy and papyrus_root_index >= 0 and character_bip_indices:
        arm_name = "papyrus"
        if _norm_low(arm_name) in used_object_names:
            arm_name = "papyrus"
        used_object_names.add(_norm_low(arm_name))
        arm_data = bpy.data.armatures.new(arm_name)
        papyrus_armature_obj = bpy.data.objects.new(arm_name, arm_data)
        collection.objects.link(papyrus_armature_obj)
        if hasattr(arm_data, "display_type"):
            arm_data.display_type = "OCTAHEDRAL"
        if hasattr(arm_obj := papyrus_armature_obj, "show_in_front"):
            arm_obj.show_in_front = False
        if hasattr(arm_obj, "display_type"):
            arm_obj.display_type = "TEXTURED"
        node_by_bone_index[int(papyrus_root_index)] = papyrus_armature_obj

        bip_head_by_index: Dict[int, Vector] = {}
        weighted_head_by_index: Dict[int, Vector] = {}
        papyrus_root_record = raw_scene_record_by_index.get(int(papyrus_root_index))
        papyrus_root_off_points = list(getattr(papyrus_root_record, "off_mat_points", []) or [])
        for bidx in sorted(character_bip_indices):
            raw_name = str(display_name_by_index.get(int(bidx), "") or "").strip()
            raw_low = _norm_low(raw_name)
            weighted = _pick_position_from_payload(raw_name, bone_positions)
            if isinstance(weighted, Vector) and "finger" in raw_low and float(weighted.z) < 0.4:
                weighted = None
            resolved = resolved_positions.get(int(bidx))
            parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
            if isinstance(weighted, Vector):
                weighted_head_by_index[int(bidx)] = weighted.copy()
            if raw_low == "bip01 footsteps" and len(papyrus_root_off_points) >= 3:
                try:
                    raw_point = papyrus_root_off_points[2]
                    if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 3:
                        head = Vector((float(raw_point[0]), float(raw_point[1]), float(raw_point[2])))
                    else:
                        head = resolved.copy() if isinstance(resolved, Vector) else (weighted.copy() if isinstance(weighted, Vector) else None)
                except Exception:
                    head = resolved.copy() if isinstance(resolved, Vector) else (weighted.copy() if isinstance(weighted, Vector) else None)
            elif int(bidx) == 1 or parent_idx == int(papyrus_root_index):
                head = resolved.copy() if isinstance(resolved, Vector) else (weighted.copy() if isinstance(weighted, Vector) else None)
            else:
                head = weighted.copy() if isinstance(weighted, Vector) else (resolved.copy() if isinstance(resolved, Vector) else None)
            if head is None:
                continue
            bip_head_by_index[int(bidx)] = head
            papyrus_bone_name_by_index[int(bidx)] = _character_bone_name(raw_name)

        semantic_child_by_name = {
            "bip01 l thigh": "bip01 l calf",
            "bip01 l calf": "bip01 l foot",
            "bip01 l foot": "bip01 l toe0",
            "bip01 r thigh": "bip01 r calf",
            "bip01 r calf": "bip01 r foot",
            "bip01 r foot": "bip01 r toe0",
        }
        bip_index_by_raw_low = {
            _norm_low(str(display_name_by_index.get(int(idx), "") or "")): int(idx)
            for idx in character_bip_indices
        }
        for bidx in sorted(character_bip_indices):
            if int(bidx) in weighted_head_by_index:
                continue
            head = bip_head_by_index.get(int(bidx))
            if head is None:
                continue
            parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
            parent_head = bip_head_by_index.get(int(parent_idx))
            child_heads = [
                bip_head_by_index.get(int(ch_idx))
                for ch_idx in child_by_parent.get(int(bidx), [])
                if bip_head_by_index.get(int(ch_idx)) is not None
            ]
            child_heads = [pt.copy() for pt in child_heads if pt is not None]
            if parent_head is not None and child_heads:
                bip_head_by_index[int(bidx)] = (parent_head.copy() + child_heads[0].copy()) * 0.5
                continue
            if parent_head is not None and (float(head.length) <= 0.05 or float(head.z) <= 0.05):
                bip_head_by_index[int(bidx)] = parent_head.copy()

        for o in bpy.context.scene.objects:
            o.select_set(False)
        papyrus_armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = papyrus_armature_obj
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit_bones_by_index: Dict[int, Any] = {}
            for bidx in sorted(bip_head_by_index.keys()):
                bone_name = papyrus_bone_name_by_index.get(int(bidx))
                if not bone_name:
                    continue
                eb = arm_data.edit_bones.new(bone_name)
                head = bip_head_by_index[int(bidx)].copy()
                child_heads: List[Vector] = []
                for ch_idx in child_by_parent.get(int(bidx), []):
                    ch_head = bip_head_by_index.get(int(ch_idx))
                    if ch_head is not None:
                        child_heads.append(ch_head.copy())
                raw_low = _norm_low(str(display_name_by_index.get(int(bidx), "") or ""))
                semantic_next_idx = bip_index_by_raw_low.get(semantic_child_by_name.get(raw_low, ""))
                if semantic_next_idx is not None:
                    semantic_head = bip_head_by_index.get(int(semantic_next_idx))
                    if semantic_head is not None:
                        child_heads.append(semantic_head.copy())
                tail = None
                if raw_low == "bip01":
                    tail = head + Vector((0.3, 0.0, 0.0))
                elif raw_low == "bip01 footsteps":
                    tail = head + Vector((0.458033, 0.0, 0.0))
                else:
                    child_heads = [pt for pt in child_heads if (pt - head).length > 1.0e-5]
                    if child_heads:
                        tail = min(child_heads, key=lambda pt: (pt - head).length_squared)
                    parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
                    parent_head = bip_head_by_index.get(int(parent_idx))
                    if tail is None and parent_head is not None and (head - parent_head).length > 1.0e-5:
                        direction = (head - parent_head).normalized()
                        tail = head + direction * max(0.02, min(0.2, (head - parent_head).length * 0.65))
                if tail is None or (tail - head).length <= 1.0e-5:
                    tail = head + Vector((0.0, 0.0, bone_len))
                eb.head = head
                eb.tail = tail
                eb.use_deform = True
                edit_bones_by_index[int(bidx)] = eb

            for bidx, eb in edit_bones_by_index.items():
                parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
                if parent_idx in edit_bones_by_index:
                    try:
                        eb.parent = edit_bones_by_index[int(parent_idx)]
                    except Exception:
                        continue
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    for bidx in ordered_indices:
        mesh_nodes = mesh_by_bone_index.get(int(bidx), [])
        if mesh_nodes:
            node_by_bone_index[int(bidx)] = mesh_nodes[0]
            continue

        raw = str(display_name_by_index.get(int(bidx), "")).strip()
        if not raw:
            continue
        raw_low = _norm_low(raw)
        policy, _reason = node_policy_by_index.get(int(bidx), ("skipped_unresolved", "policy_missing"))
        if policy != "visible_helper":
            continue

        base_name = _to_pretty_name(raw)
        name = base_name[:63] if base_name else f"Empty_{int(bidx):03d}"
        if not name:
            name = f"Empty_{int(bidx):03d}"
        suffix = 2
        while _norm_low(name) in used_object_names:
            tail = f"_{suffix}"
            cut = max(1, 63 - len(tail))
            name = f"{base_name[:cut]}{tail}"
            suffix += 1
        used_object_names.add(_norm_low(name))

        empty = bpy.data.objects.new(name, None)
        display_type, display_size = _helper_empty_display_preset(raw)
        if "/helico/" in asset_hint.replace("\\", "/").lower():
            if re.fullmatch(r"(tourelle|axe_canon|canon)_[0-9]+", raw_low, flags=re.IGNORECASE):
                display_type, display_size = "PLAIN_AXES", 0.41
        empty.empty_display_type = str(display_type)
        empty.empty_display_size = float(display_size)
        collection.objects.link(empty)
        node_by_bone_index[int(bidx)] = empty
        created_empties += 1

    # Mirror-fix outlier wheel/elevator bones in resolved_positions BEFORE anything is placed.
    # A duplicate/bogus bone or a bad weights-derived position can fling a single wheel + its
    # elevator empty + its track deform bone to the wrong side (Leopard_1V roue_elev_d1 ->
    # (0.79,2.72,-1.38), dragging Roue_D1). Detect a roue_[dg]#/roue_elev_[dg]# whose resolved Y
    # is a far outlier from its same-side siblings and snap it to the MIRROR of its opposite-side
    # counterpart. Fixing resolved_positions here fixes the empty (placed just below), the wheel
    # origin, AND the track deform bone (head=resolved_positions, built later) all at once.
    # Universal; only clear outliers WITH a good counterpart are touched (no-op otherwise).
    try:
        import statistics as _wst
        _wgrp: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for _bi in ordered_indices:
            if not isinstance(resolved_positions.get(int(_bi)), Vector):
                continue
            _wm = re.fullmatch(r"(roue_elev_|roue_)([dg])(\d+)", _norm_low(str(display_name_by_index.get(int(_bi), ""))))
            if _wm:
                _wgrp.setdefault((_wm.group(1), _wm.group(2)), []).append((int(_wm.group(3)), int(_bi)))
        for (_pfx, _sd) in list(_wgrp.keys()):
            _entries = _wgrp.get((_pfx, _sd), [])
            _ys = [resolved_positions[_bi].y for _, _bi in _entries]
            if len(_ys) < 3:
                continue
            _med = _wst.median(_ys)
            _oentries = _wgrp.get((_pfx, "g" if _sd == "d" else "d"), [])
            _oys = [resolved_positions[_bi].y for _, _bi in _oentries]
            if len(_oys) < 3:
                continue
            _omed = _wst.median(_oys)
            _good_opp: Dict[int, Vector] = {}
            for _n, _bi in _oentries:
                _p = resolved_positions[_bi]
                if abs(_p.y - _omed) <= 1.0:
                    _good_opp[_n] = _p
            for _n, _bi in _entries:
                _p = resolved_positions[_bi]
                if abs(_p.y - _med) <= 1.0:
                    continue
                _op = _good_opp.get(_n)
                if _op is not None:
                    resolved_positions[int(_bi)] = Vector((_op.x, -_op.y, _op.z))
    except Exception:
        pass

    # Place object origins/empties at resolved deterministic positions.
    for bidx, node in node_by_bone_index.items():
        pos = resolved_positions.get(int(bidx))
        if node.type == "MESH":
            if pos is None:
                continue
            # RD wheels/tracks intentionally use a geometric-centroid origin (their
            # roue_*/chenille_* bone pivot resolves to the world origin); don't clobber
            # it with the bone pivot. WARNO/other meshes never set this marker.
            try:
                if node.get("warno_rd_centroid_origin"):
                    continue
            except Exception:
                pass
            _set_object_origin_world(node, pos)
        else:
            # "Fallen empty" net: a helper with NO resolved position, or one collapsed to
            # ~world-origin while its parent sits elsewhere (a degenerate stream_local, e.g.
            # the ikv_103 Antenne_* / gepard 'Particle view' bug), snaps to its nearest
            # non-origin ancestor (antenna base on the hull) instead of dropping to (0,0,0).
            # WARNO empties resolve to real positions; also gate on the RD deterministic
            # scene path so the WARNO (weighted-positions) path is byte-for-byte untouched.
            if use_deterministic_raw_scene and _armature_is_rd and (pos is None or float(pos.length) < 0.01):
                _walk = int(bone_parent_by_index.get(int(bidx), -1))
                _g = 0
                while _walk >= 0 and _g < 256:
                    _anc = resolved_positions.get(int(_walk))
                    if isinstance(_anc, Vector) and float(_anc.length) >= 0.01:
                        pos = _anc.copy()
                        break
                    _walk = int(bone_parent_by_index.get(int(_walk), -1))
                    _g += 1
            if pos is None:
                continue
            node.location = pos

    def _ensure_track_edge_groups_from_resolved(
        obj: bpy.types.Object,
        side_key: str,
    ) -> List[str]:
        if obj.type != "MESH" or obj.data is None:
            return []
        mesh = obj.data
        if not getattr(mesh, "vertices", None):
            return []
        side = "d" if str(side_key or "").strip().lower() == "right" else "g"
        wheel_rx = re.compile(rf"^roue_elev_{side}[0-9]+$", re.IGNORECASE)
        candidates: List[Tuple[str, Vector]] = []
        for bidx in ordered_indices:
            raw_name = str(display_name_by_index.get(int(bidx), "") or "").strip()
            low = _norm_low(raw_name)
            if not wheel_rx.fullmatch(low):
                continue
            pos = resolved_positions.get(int(bidx))
            if pos is None:
                continue
            candidates.append((_pretty_warno_node_name(raw_name), pos.copy()))
        if len(candidates) < 2:
            return []

        edge_candidates: Dict[str, Vector] = {}
        min_name, min_pos = min(candidates, key=lambda item: float(item[1].x))
        max_name, max_pos = max(candidates, key=lambda item: float(item[1].x))
        edge_candidates[min_name] = min_pos
        edge_candidates[max_name] = max_pos

        verts = list(mesh.vertices)
        if not verts:
            return []
        span_x = max(float(v.co.x) for v in verts) - min(float(v.co.x) for v in verts)
        band_x = max(0.28, span_x * 0.08)
        created: List[str] = []
        for vg_name, target_pos in edge_candidates.items():
            vg = obj.vertex_groups.get(str(vg_name))
            if vg is None:
                vg = obj.vertex_groups.new(name=str(vg_name))
            vg_index = int(vg.index)
            has_weights = False
            for vert in verts:
                for slot in vert.groups:
                    if int(slot.group) == vg_index and float(slot.weight) > 1.0e-6:
                        has_weights = True
                        break
                if has_weights:
                    break
            if has_weights:
                continue

            band_verts = [v for v in verts if abs(float(v.co.x) - float(target_pos.x)) <= band_x]
            if len(band_verts) < 4:
                band_verts = verts
            ranked = sorted(
                band_verts,
                key=lambda vert: (
                    -float(vert.co.z),
                    abs(float(vert.co.x) - float(target_pos.x)),
                    (vert.co - target_pos).length_squared,
                ),
            )
            nearest = list(ranked[:2])
            if not nearest:
                continue
            for vert in nearest:
                vg.add([int(vert.index)], 0.02, "REPLACE")
            created.append(str(vg_name))
        return created

    synthesized_track_edge_groups = 0
    if semantic_mode == "RAW_DEBUG":
        for obj in imported_objects:
            if obj.type != "MESH":
                continue
            low = _norm_low(obj.name)
            side = ""
            if "chenille_gauche" in low:
                side = "left"
            elif "chenille_droite" in low:
                side = "right"
            if not side:
                continue
            try:
                created_groups = _ensure_track_edge_groups_from_resolved(obj, side)
                synthesized_track_edge_groups += len(created_groups)
                for vg_name in created_groups:
                    vg_low = _norm_low(vg_name)
                    if vg_low.startswith("roue_"):
                        track_vertex_groups_by_side.setdefault(side, set()).add(vg_low)
                        track_deform_nodes.add(vg_low)
            except Exception:
                continue

    tracked_vehicle = gfx_track_kind == "continuous_track"
    if gfx_track_kind:
        build_track_helpers = tracked_vehicle
    else:
        build_track_helpers = any(track_vertex_groups_by_side.values())

    raw_indices_by_name: Dict[str, List[int]] = defaultdict(list)
    for raw_idx, raw_name in bone_name_by_index.items():
        raw_low = _norm_low(str(raw_name))
        if raw_low:
            raw_indices_by_name[raw_low].append(int(raw_idx))

    def _raw_cluster_key(start_idx: int) -> str:
        cur = int(start_idx)
        seen: set[int] = set()
        while cur >= 0 and cur not in seen:
            seen.add(cur)
            parent_idx = int(bone_parent_by_index.get(int(cur), -1))
            if parent_idx < 0:
                return "__root__"
            parent_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
            if re.fullmatch(r"^armature_[dg][12]$", parent_low, flags=re.IGNORECASE):
                return parent_low
            if parent_low in {"chassis", "armature", "papyrus", "fake"}:
                return parent_low
            cur = int(parent_idx)
        return "__root__"

    def _nearest_raw_track_carrier(start_idx: int) -> str:
        cur = int(start_idx)
        seen: set[int] = set()
        while cur >= 0 and cur not in seen:
            seen.add(cur)
            parent_idx = int(bone_parent_by_index.get(int(cur), -1))
            if parent_idx < 0:
                break
            parent_low = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
            if re.fullmatch(r"^armature_[dg][12]$", parent_low, flags=re.IGNORECASE):
                return parent_low
            if parent_low in {"armature", "papyrus", "fake"}:
                return parent_low
            cur = int(parent_idx)
        return ""

    def _fallback_side_track_carrier(side_low: str, wheel_idx: int) -> str:
        side = str(side_low or "").strip().lower()
        if side not in {"d", "g"}:
            return ""
        wheel_cluster = _raw_cluster_key(int(wheel_idx))
        carrier_keys = [f"armature_{side}1", f"armature_{side}2"]
        carriers: List[Tuple[str, str]] = []
        for carrier_key in carrier_keys:
            carrier_indices = raw_indices_by_name.get(carrier_key, [])
            if not carrier_indices:
                continue
            carriers.append((carrier_key, _raw_cluster_key(int(carrier_indices[0]))))
        if not carriers:
            return ""

        exact = [name for name, cluster in carriers if cluster == wheel_cluster]
        if exact:
            return exact[0]
        if wheel_cluster == "chassis":
            for name, _cluster in carriers:
                if name.endswith("1"):
                    return name
        if wheel_cluster == "__root__":
            for name, _cluster in carriers:
                if name.endswith("2"):
                    return name
        return carriers[0][0]

    def _wheel_armature_group_name(side_low: str, wheel_idx: int) -> str:
        carrier_low = _nearest_raw_track_carrier(int(wheel_idx))
        if re.fullmatch(r"^armature_[dg][12]$", carrier_low, flags=re.IGNORECASE):
            return carrier_low
        if carrier_low in {"armature", "fake"}:
            return f"armature_{side_low}1"
        if carrier_low == "papyrus":
            return f"armature_{side_low}2"
        return _fallback_side_track_carrier(side_low, int(wheel_idx))

    wheel_groups: Dict[str, List[Tuple[int, bool]]] = {}
    wheel_pattern = re.compile(r"^roue_elev_([dg])([0-9]+)$", re.IGNORECASE)
    track_edge_helper_indices: set[int] = set()
    if tracked_vehicle and chassis_obj is not None:
        wheel_rows_by_side: Dict[str, List[Tuple[int, Vector]]] = defaultdict(list)
        for bidx in ordered_indices:
            raw_low = _norm_low(str(display_name_by_index.get(int(bidx), "") or ""))
            m = wheel_pattern.match(raw_low)
            if not m:
                continue
            pos = resolved_positions.get(int(bidx))
            if not isinstance(pos, Vector):
                continue
            wheel_rows_by_side[str(m.group(1)).lower()].append((int(bidx), pos.copy()))
        for rows in wheel_rows_by_side.values():
            if not rows:
                continue
            front_idx, _front_pos = max(rows, key=lambda item: (float(item[1].x), -abs(float(item[1].z)), int(item[0])))
            rear_idx, _rear_pos = min(rows, key=lambda item: (float(item[1].x), abs(float(item[1].z)), int(item[0])))
            track_edge_helper_indices.add(int(front_idx))
            track_edge_helper_indices.add(int(rear_idx))
    wheel_missing_carrier = 0
    for bidx in ordered_indices:
        raw = str(display_name_by_index.get(int(bidx), "")).strip()
        raw_low = _norm_low(raw)
        m = wheel_pattern.match(raw_low)
        if not m:
            continue
        side = str(m.group(1)).lower()
        side_key = "right" if side == "d" else "left"
        side_track_groups = track_vertex_groups_by_side.get(side_key, set())
        is_deform = raw_low in side_track_groups
        if not build_track_helpers and not is_deform:
            continue
        grp = _wheel_armature_group_name(side, int(bidx))
        if not grp:
            # A1-B: the skeleton has Roue_Elev_<side> wheels but NO Armature_<side><n> carrier
            # bone (loggim_sa / hafiz / olifant), so no track armature was ever built. Synthesize
            # a side carrier so the rig IS created — exactly what units that have carrier bones
            # get for free. Only fires when the real carrier is genuinely absent.
            wheel_missing_carrier += 1
            grp = "armature_%s1" % side
        wheel_groups.setdefault(grp, []).append((int(bidx), bool(is_deform)))

    created_armatures: List[bpy.types.Object] = []
    armatures_by_side: Dict[str, List[bpy.types.Object]] = {"left": [], "right": []}
    wheel_helper_bones_total = 0
    wheel_helper_deform_bones = 0

    for grp_low in sorted(wheel_groups.keys()):
        entries = sorted(wheel_groups.get(grp_low, []), key=lambda item: int(item[0]))
        if not entries:
            continue
        arm_name = _to_pretty_name(grp_low)
        arm_data = bpy.data.armatures.new(arm_name)
        arm_obj = bpy.data.objects.new(arm_name, arm_data)
        collection.objects.link(arm_obj)
        if hasattr(arm_data, "display_type"):
            arm_data.display_type = "OCTAHEDRAL"
        if hasattr(arm_obj, "show_in_front"):
            arm_obj.show_in_front = True
        if hasattr(arm_obj, "display_type"):
            arm_obj.display_type = "TEXTURED"

        for o in bpy.context.scene.objects:
            o.select_set(False)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")
        grp_has_deform = False
        for bidx, is_deform in entries:
            braw = str(display_name_by_index.get(int(bidx), "")).strip()
            bname = _to_pretty_name(braw)[:63] if braw else f"Bone_{int(bidx):03d}"
            eb = arm_data.edit_bones.new(bname)
            pos = resolved_positions.get(int(bidx), Vector((0.0, 0.0, 0.0)))
            eb.head = pos
            eb.tail = pos + Vector((0.0, 0.0, bone_len))
            eb.use_deform = bool(is_deform)
            wheel_helper_bones_total += 1
            if is_deform:
                wheel_helper_deform_bones += 1
                grp_has_deform = True
        bpy.ops.object.mode_set(mode="OBJECT")

        created_armatures.append(arm_obj)
        if grp_low.endswith("1") and chassis_obj is not None:
            _set_parent_keep_world(arm_obj, chassis_obj)
        m = re.match(r"^armature_([DG])([12])$", arm_name, flags=re.IGNORECASE)
        if m and grp_has_deform:
            side = "right" if m.group(1).upper() == "D" else "left"
            armatures_by_side.setdefault(side, []).append(arm_obj)

    # Hierarchical OBJECT parenting from bone tree.
    parented_nodes = 0
    for bidx in ordered_indices:
        child = node_by_bone_index.get(int(bidx))
        if child is None:
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        if parent_idx < 0:
            continue
        parent_raw = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        child_raw = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
        if parent_raw.startswith("armature_"):
            continue
        if parent_raw in {"armature", "papyrus", "fake"}:
            continue
        if child_raw.startswith("fx_fumee_chenille_"):
            continue
        parent = node_by_bone_index.get(int(parent_idx))
        if parent is None or parent == child:
            continue
        _set_parent_keep_world(child, parent)
        parented_nodes += 1

    # Deterministic raw-scene parenting pass: re-assert parent links for every resolved raw node
    # before local matrices are applied. This is intentionally redundant with the legacy pass
    # because the old heuristic path may skip nodes that now have exact SPK transforms.
    deterministic_parent_links = 0
    for bidx in sorted(deterministic_exact_indices):
        child = node_by_bone_index.get(int(bidx))
        if child is None:
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        if parent_idx < 0:
            continue
        parent_raw = _norm_low(str(bone_name_by_index.get(int(parent_idx), "")))
        if parent_raw.startswith("armature_") or parent_raw in {"armature", "papyrus", "fake"}:
            continue
        parent = node_by_bone_index.get(int(parent_idx))
        if parent is None or parent == child:
            continue
        if child.parent != parent:
            _set_parent_keep_world(child, parent)
            deterministic_parent_links += 1

    # Front idlers / rear sprockets should stay under Chassis on tracked vehicles.
    # Use same-side min/max X Roue_Elev helpers so this applies automatically across tanks
    # with different wheel counts, while leaving interior support rollers unchanged.
    if chassis_obj is not None and track_edge_helper_indices:
        for bidx in sorted(track_edge_helper_indices):
            child = node_by_bone_index.get(int(bidx))
            if child is None or child == chassis_obj:
                continue
            child_raw = _norm_low(str(bone_name_by_index.get(int(bidx), "")))
            if not wheel_pattern.fullmatch(child_raw):
                continue
            if child.parent != chassis_obj:
                _set_parent_keep_world(child, chassis_obj)

    exact_local_applied = 0
    for bidx, transform_row in exact_local_transform_by_index.items():
        node = node_by_bone_index.get(int(bidx))
        if node is None:
            continue
        if _norm_low(str(transform_row.get("source_kind", "") or "")) != "spk_exact":
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent = node_by_bone_index.get(int(parent_idx))
        if parent is None or node.parent != parent:
            continue
        try:
            raw_name = str(bone_name_by_index.get(int(bidx), "") or "")
            local_rotation_basis = transform_row.get("local_rotation_basis")
            if (
                node.type == "MESH"
                and _is_aircraft_asset(asset_hint)
                and _is_aircraft_control_surface_name(raw_name)
            ):
                local_rotation_basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            # Wheel meshes (Roue_D*, Roue_G*, but NOT Roue_Elev_* helpers) are
            # already oriented horizontally in Eugen's source FBX — DEV blend
            # for Leopard_1A1 confirms rotation_euler is (0,0,0). On some tanks
            # (e.g. Leopard_2A1) SPK off_mat carries a non-identity basis that
            # we then double-apply, rotating the cylinder onto its side.
            # Force identity for wheel mesh nodes so the DEV behaviour matches
            # universally without affecting tanks that already worked.
            if node.type == "MESH":
                _raw_low = _norm_low(raw_name)
                if _raw_low.startswith("roue_") and not _raw_low.startswith("roue_elev_"):
                    local_rotation_basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            local_matrix = _compose_affine_components(
                transform_row.get("local_translation", [0.0, 0.0, 0.0]),
                local_rotation_basis,
                transform_row.get("local_scale"),
            )
            node.matrix_parent_inverse = Matrix.Identity(4)
            loc, rot, scale = local_matrix.decompose()
            node.location = loc
            node.rotation_mode = "XYZ"
            node.rotation_euler = rot.to_euler("XYZ")
            node.scale = scale
            node["warno_semantic_role"] = str(transform_row.get("role", "") or "")
            node["warno_semantic_source_kind"] = str(transform_row.get("source_kind", "") or "")
            node["warno_semantic_parent_name"] = str(bone_name_by_index.get(int(parent_idx), "") or "")
            node["warno_semantic_local_translation"] = json.dumps(
                [round(float(v), 9) for v in transform_row.get("local_translation", Vector((0.0, 0.0, 0.0)))],
                ensure_ascii=False,
            )
            node["warno_semantic_local_rotation_basis"] = json.dumps(
                local_rotation_basis if local_rotation_basis is not None else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ensure_ascii=False,
            )
            node["warno_semantic_local_scale"] = json.dumps(
                transform_row.get("local_scale", [1.0, 1.0, 1.0]),
                ensure_ascii=False,
            )
            node["warno_semantic_provenance"] = json.dumps(
                dict(transform_row.get("provenance", {}) or {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            exact_local_applied += 1
        except Exception:
            continue

    if papyrus_armature_obj is not None:
        for bidx in sorted(character_bip_indices):
            bone_name = papyrus_bone_name_by_index.get(int(bidx), "")
            if not bone_name or bone_name not in papyrus_armature_obj.data.bones:
                continue
        for obj in imported_objects:
            raw_low = _norm_low(str(obj.name))
            if raw_low == "soldat":
                if obj.type == "MESH":
                    try:
                        _set_object_origin_world(obj, Vector((0.0, 0.0, 0.0)))
                    except Exception:
                        pass
                wm = obj.matrix_world.copy()
                obj.parent = papyrus_armature_obj
                obj.parent_type = "OBJECT"
                obj.parent_bone = ""
                obj.matrix_parent_inverse = papyrus_armature_obj.matrix_world.inverted()
                obj.matrix_world = wm
                _ensure_armature_modifier(obj, "papyrus", papyrus_armature_obj)
                continue
        for bidx in ordered_indices:
            child = node_by_bone_index.get(int(bidx))
            if child is None or child == papyrus_armature_obj:
                continue
            raw_low = _norm_low(str(getattr(child, "name", "") or ""))
            parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
            bone_name = papyrus_bone_name_by_index.get(int(parent_idx), "")
            if not bone_name:
                continue
            transform_row = exact_local_transform_by_index.get(int(bidx))
            keep_child_world = child.type == "MESH" and raw_low == "soldat"
            if child.parent != papyrus_armature_obj or child.parent_type != "BONE" or child.parent_bone != bone_name:
                preserved_world = child.matrix_world.copy() if keep_child_world else None
                child.parent = papyrus_armature_obj
                child.parent_type = "BONE"
                child.parent_bone = bone_name
                child.matrix_parent_inverse = Matrix.Identity(4)
                if preserved_world is not None:
                    child.matrix_world = preserved_world
            if keep_child_world:
                continue
            if isinstance(transform_row, dict):
                try:
                    local_matrix = _compose_affine_components(
                        transform_row.get("local_translation", [0.0, 0.0, 0.0]),
                        transform_row.get("local_rotation_basis"),
                        transform_row.get("local_scale"),
                    )
                    loc, rot, scale = local_matrix.decompose()
                    child.location = loc
                    child.rotation_mode = "XYZ"
                    child.rotation_euler = rot.to_euler("XYZ")
                    child.scale = scale
                except Exception:
                    pass

    deterministic_world_applied = 0
    for bidx in sorted(deterministic_exact_indices):
        node = node_by_bone_index.get(int(bidx))
        pos = resolved_positions.get(int(bidx))
        if node is None or not isinstance(pos, Vector):
            continue
        if node.parent is not None and int(bidx) in exact_local_transform_by_index:
            continue
        try:
            if node.type == "MESH":
                _set_object_origin_world(node, pos.copy())
            else:
                node.location = pos.copy()
            deterministic_world_applied += 1
        except Exception:
            continue

    smoke_world_overrides = 0
    # Track-smoke (Fx_Fumee_Chenille_D/G) empties belong BEHIND the belt at ground level, not at
    # their raw bone pivot (mid-hull, "under the tracks"). Move them just behind the track rear
    # edge (X) at ground (Z). The side (Y) comes from each empty's OWN resolved position — the
    # WARNO track BASE mesh is centred at Y~=0 so the track-mesh bbox center_y is useless there
    # (it collapsed both empties to the centre line); the resolved Y keeps D/G on their correct
    # sides for WARNO + RD. Runs for both games (this is the intended smoke placement).
    if use_deterministic_raw_scene:
        def _track_dust_anchor_metrics(track_obj: bpy.types.Object | None) -> Dict[str, float] | None:
            if track_obj is None:
                return None
            try:
                mn, mx, ctr = _bounds_world(track_obj)
                rear_x = max(
                    abs(float(mx.x) - float(ctr.x)),
                    abs(float(ctr.x) - float(mn.x)),
                )
                return {
                    "rear_x": float(-rear_x),
                    "center_y": float(ctr.y),
                    "ground_z": float(mn.z),
                }
            except Exception:
                return None

        smoke_d_idx = -1
        smoke_g_idx = -1
        for bidx in ordered_indices:
            raw_low = _norm_low(str(bone_name_by_index.get(int(bidx), "") or ""))
            if raw_low.startswith("fx_fumee_chenille_d"):
                smoke_d_idx = int(bidx)
            elif raw_low.startswith("fx_fumee_chenille_g"):
                smoke_g_idx = int(bidx)

        right_metrics = _track_dust_anchor_metrics(track_right_obj)
        left_metrics = _track_dust_anchor_metrics(track_left_obj)
        dust_x_values = [
            abs(float(metrics.get("rear_x", 0.0)))
            for metrics in (right_metrics, left_metrics)
            if isinstance(metrics, dict)
        ]
        dust_y_values = [
            abs(float(metrics.get("center_y", 0.0)))
            for metrics in (right_metrics, left_metrics)
            if isinstance(metrics, dict)
        ]
        dust_z_values = [
            float(metrics.get("ground_z", 0.0))
            for metrics in (right_metrics, left_metrics)
            if isinstance(metrics, dict)
        ]
        # Anchor the smoke just behind the TRACK belt's rear (the idler — where track dust kicks
        # up), at ground; close to the hull, not floating behind the rear plate. Universal:
        # derived from the track object bounds + a small fixed margin (works for any unit).
        _track_rear_x = None
        for _t in (track_right_obj, track_left_obj):
            if _t is None:
                continue
            try:
                _mn, _mx, _c = _bounds_world(_t)
                _v = float(_mn.x)
                if _track_rear_x is None or _v < _track_rear_x:
                    _track_rear_x = _v
            except Exception:
                continue
        if _track_rear_x is not None:
            shared_rear_x = _track_rear_x - 0.15
        else:
            shared_rear_x = -max(dust_x_values) * 1.12 if dust_x_values else 0.0
        shared_abs_y = min(dust_y_values) if dust_y_values else 0.0
        shared_ground_z = min(dust_z_values) if dust_z_values else 0.0

        def _smoke_side_y(idx: int, fallback_y: float) -> float:
            # Prefer the empty's OWN resolved (spk_exact) side position; fall back to the
            # track-mesh center_y only when the resolved Y is missing/collapsed (|y|<=0.1,
            # e.g. the centred WARNO belt base mesh) so D/G never collapse to the centre line.
            rp = resolved_positions.get(int(idx))
            if isinstance(rp, Vector) and abs(float(rp.y)) > 0.1:
                return float(rp.y)
            return float(fallback_y)

        if right_metrics is not None or left_metrics is not None:
            smoke_d_point = Vector((shared_rear_x, _smoke_side_y(smoke_d_idx, -shared_abs_y), shared_ground_z))
            smoke_g_point = Vector((shared_rear_x, _smoke_side_y(smoke_g_idx, shared_abs_y), shared_ground_z))
        else:
            smoke_d_point = None
            smoke_g_point = None

        for smoke_idx, smoke_point in ((smoke_d_idx, smoke_d_point), (smoke_g_idx, smoke_g_point)):
            if smoke_idx < 0 or not isinstance(smoke_point, Vector):
                continue
            node = node_by_bone_index.get(int(smoke_idx))
            if node is None:
                continue
            try:
                if node.type == "MESH":
                    _set_object_origin_world(node, smoke_point.copy())
                else:
                    node.location = smoke_point.copy()
                resolved_positions[int(smoke_idx)] = smoke_point.copy()
                resolved_world_matrix_by_index[int(smoke_idx)] = Matrix.Translation(smoke_point.copy())
                smoke_world_overrides += 1
            except Exception:
                continue

    # Track meshes use armature modifiers and wheel vertex groups, not object parenting.
    track_modifier_links = 0
    for obj in imported_objects:
        low = _norm_low(obj.name)
        side = ""
        if "chenille_gauche" in low:
            side = "left"
        elif "chenille_droite" in low:
            side = "right"
        if not side:
            continue
        if obj.parent is not None and obj.parent.type == "ARMATURE":
            wm = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = wm
        for arm in sorted(armatures_by_side.get(side, []), key=lambda item: item.name.lower()):
            if _ensure_armature_modifier(obj, arm.name, arm):
                track_modifier_links += 1

    if semantic_mode == "REFERENCE":
        cleanup_objects: List[bpy.types.Object] = []
        seen_cleanup_ids: set[int] = set()
        for obj in list(imported_objects) + list(node_by_bone_index.values()) + list(created_armatures):
            if obj is None:
                continue
            obj_id = id(obj)
            if obj_id in seen_cleanup_ids:
                continue
            seen_cleanup_ids.add(obj_id)
            cleanup_objects.append(obj)
        if papyrus_armature_obj is not None and id(papyrus_armature_obj) not in seen_cleanup_ids:
            cleanup_objects.append(papyrus_armature_obj)
        for obj in cleanup_objects:
            try:
                for key in list(obj.keys()):
                    if str(key).startswith("warno_semantic_"):
                        del obj[key]
                for key in ("warno_group", "warno_group_bone_index", "warno_is_track_mesh"):
                    if key in obj.keys():
                        del obj[key]
                if obj.type != "MESH" and "warno_asset" in obj.keys():
                    del obj["warno_asset"]
            except Exception:
                continue

    if settings is not None:
        bones_total = len(ordered_indices)
        weighted_count = len(weighted_positions_by_index)
        fallback_count = max(0, bones_total - weighted_count)
        _warno_log(
            settings,
            (
                f"armature bones: total={bones_total} "
                f"bones_created_weighted={weighted_count} "
                f"bones_created_fallback={fallback_count} "
                f"bones_missing_parent_anchor={missing_parent_anchor} "
                f"(fallback_children={fallback_from_children}, "
                f"fallback_parent={fallback_from_parent}, "
                f"fallback_line={fallback_line_placed}) "
                f"| off_mat_sources={len(off_mat_points_by_index)} "
                f"| off_mat_chain_resolved={off_mat_chain_resolved} "
                f"| off_mat_pair_resolved={off_mat_pair_resolved} "
                f"| off_mat_fx_resolved={off_mat_fx_resolved} "
                f"| empties_created={created_empties} "
                f"| wheel_armatures_created={len(created_armatures)} "
                f"| wheel_helper_bones_total={wheel_helper_bones_total} "
                f"| wheel_helper_bones_deform={wheel_helper_deform_bones} "
                f"| wheel_helpers_missing_carrier={wheel_missing_carrier} "
                f"| object_parent_links={parented_nodes} "
                f"| object_parent_links_det={deterministic_parent_links} "
                f"| exact_local_applied={exact_local_applied} "
                f"| deterministic_world_applied={deterministic_world_applied} "
                f"| smoke_world_overrides={smoke_world_overrides} "
                f"| track_modifier_links={track_modifier_links} "
                f"| track_edge_groups_synth={synthesized_track_edge_groups} "
                f"| semantic_mode={semantic_mode.lower()} "
                f"| helpers=node_policy "
                f"| fx={gfx_manifest_source} "
                f"| track_deform=mesh_weights "
                f"| track_kind={gfx_track_kind or 'unknown'} "
                f"| semantic_nodes={len(semantic_scene_nodes)} "
                f"| support_nodes={len(semantic_support_nodes)} "
                f"| required_nodes={len(gfx_required_nodes)} "
                f"| fx_nodes={len(gfx_fx_nodes)} "
                f"| subdepiction_nodes={len(gfx_subdepiction_nodes)} "
                f"| runtime_fx_nodes={len(preserved_runtime_fx_nodes)} "
                f"| operator_contracts={len(gfx_operator_contracts)} "
                f"| semantic_helpers_exact={semantic_helpers_exact} "
                f"| semantic_helpers_approx={semantic_helpers_approx} "
                f"| semantic_helpers_skipped={semantic_helpers_skipped} "
                f"| support_helpers_exact={support_helpers_exact} "
                f"| support_helpers_approx={support_helpers_approx} "
                f"| support_helpers_skipped={support_helpers_skipped} "
                f"| policy_visible_mesh={node_policy_counts.get('visible_mesh', 0)} "
                f"| policy_visible_helper={node_policy_counts.get('visible_helper', 0)} "
                f"| policy_hidden_deform={node_policy_counts.get('hidden_deform_helper', 0)} "
                f"| policy_ignored_carrier={node_policy_counts.get('ignored_raw_carrier', 0)} "
                f"| policy_skipped={node_policy_counts.get('skipped_unresolved', 0)}"
            ),
            stage="armature",
        )

    if created_armatures:
        return created_armatures[0]
    return None


class WARNO_OT_LoadConfig(Operator):
    bl_idname = "warno.load_config"
    bl_label = "Load My config"
    bl_description = "Load plugin settings from config.json"

    def execute(self, context):
        settings = context.scene.warno_import
        path = _config_path(settings)
        ok, msg = _load_config_into_settings(settings, path)
        _enforce_fixed_runtime_defaults(settings)
        if ok:
            _save_project_root_state(settings)
        settings.status = msg
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED" if ok else "CANCELLED"}


class WARNO_OT_LoadExampleConfig(Operator):
    bl_idname = "warno.load_example_config"
    bl_label = "Load example config"
    bl_description = "Load plugin settings from config.example.json"

    def execute(self, context):
        settings = context.scene.warno_import
        project_candidate = _project_root(settings) / "config.example.json"
        addon_candidate = Path(__file__).resolve().parent / "config.example.json"
        path = project_candidate if project_candidate.exists() else addon_candidate
        if not path.exists() or not path.is_file():
            msg = "config.example.json not found in project/addon folder."
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        ok, msg = _load_config_into_settings(settings, path)
        _enforce_fixed_runtime_defaults(settings)
        if ok:
            _save_project_root_state(settings)
        settings.status = msg
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED" if ok else "CANCELLED"}


class WARNO_OT_SaveConfig(Operator):
    bl_idname = "warno.save_config"
    bl_label = "Save Config"
    bl_description = "Save plugin settings to config.json"

    def execute(self, context):
        settings = context.scene.warno_import
        path = _config_path(settings)
        ok, msg = _save_settings_to_config(settings, path)
        remember_ok, remember_msg = _save_project_root_state(settings)
        if remember_ok:
            msg = f"{msg} | {remember_msg}"
        else:
            msg = f"{msg} | {remember_msg}"
        settings.status = msg
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED" if ok else "CANCELLED"}


class WARNO_OT_RememberProjectRoot(Operator):
    bl_idname = "warno.remember_project_root"
    bl_label = "Remember Project Root"
    bl_description = "Save Project Root and auto-load this root config on next Blender start"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        ok, msg = _save_project_root_state(settings)
        if not ok:
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        cfg = _config_path(settings)
        if cfg.exists() and cfg.is_file():
            cfg_ok, cfg_msg = _load_config_into_settings(settings, cfg)
            _enforce_fixed_runtime_defaults(settings)
            status = f"{msg} | {cfg_msg}"
            settings.status = status
            self.report({"INFO" if cfg_ok else "WARNING"}, status)
            return {"FINISHED" if cfg_ok else "CANCELLED"}

        settings.status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _scan_assets_impl(self, context, scan_all: bool):
    settings = context.scene.warno_import
    _enforce_fixed_runtime_defaults(settings)
    t0 = time.monotonic()
    query = str(settings.query or "").strip()
    _warno_log(settings, f"scan start mode={'ALL' if scan_all else 'query'} query='{query}'", stage="scan")
    if scan_all:
        return bpy.ops.warno.build_asset_index(force_rebuild=False)

    try:
        _warno_log(settings, "stage: asset_index_prepare", stage="asset_index_prepare")
        index_data, source, _runtime_info, spk_paths = _ensure_asset_index_sync(settings, force_rebuild=False)
        extractor_mod = _extractor_module(settings)
        filtered_assets = _apply_picker_view_from_index(
            settings,
            extractor_mod,
            index_data,
            source=source,
            query=query,
        )
        groups = _asset_groups_from_cache(settings)
        folders = _asset_folders_from_cache(settings)
        msg = (
            f"Matches: {len(filtered_assets)} "
            f"(index source:{source} | groups:{len(groups)} | folders:{len(folders)} | mode:query | spk:{len(spk_paths)}) "
            f"| elapsed:{time.monotonic()-t0:.1f}s"
        )
        settings.status = msg
        _warno_log(settings, msg, stage="asset_index_query_filter")
        self.report({"INFO"}, msg)
        return {"FINISHED"}
    except Exception as exc:
        msg = f"Scan failed: {exc}"
        settings.status = msg
        _warno_log(settings, msg, level="ERROR", stage="scan")
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}


def _load_extractor_module_for_worker(extractor_path: Path):
    spec = importlib.util.spec_from_file_location(f"warno_asset_index_worker_{time.time_ns()}", str(extractor_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extractor module: {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_asset_index_payload_worker(
    extractor_path: Path,
    spk_paths: Sequence[Path],
    signature: Dict[str, Any],
    index_path: Path,
    game: "str | None" = None,
) -> Dict[str, Any]:
    extractor_mod = _load_extractor_module_for_worker(extractor_path)
    assets = _scan_assets_from_spk_paths(extractor_mod, spk_paths, query=None, game=game)
    payload = _build_asset_index_payload(extractor_mod, assets, signature, len(spk_paths))
    _save_asset_index_file(index_path, payload)
    return payload


class WARNO_OT_BuildAssetIndex(Operator):
    bl_idname = "warno.build_asset_index"
    bl_label = "Build Asset Index"
    bl_description = "Build full cached asset index from prepared ZZ runtime mesh SPK files"
    bl_options = {"INTERNAL"}

    force_rebuild: BoolProperty(default=False, options={"HIDDEN"})

    _timer = None
    _thread = None
    _job: Dict[str, Any] | None = None
    _started_at: float = 0.0

    def _cleanup_timer(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        t0 = time.monotonic()
        _warno_log(settings, "stage: asset_index_prepare", stage="asset_index_prepare")
        try:
            extractor_mod = _extractor_module(settings)
            runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
            project_root = _project_root(settings)
            spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
            if not spk_paths:
                raise RuntimeError("No mesh SPK files found in prepared ZZ runtime.")
            signature = _asset_index_signature(runtime_info, spk_paths)
            index_path = _asset_index_path(settings)
            if not bool(self.force_rebuild):
                cached = _load_asset_index_file(index_path)
                if cached is not None and _asset_index_signature_matches(cached, signature):
                    assets = _apply_picker_view_from_index(
                        settings,
                        extractor_mod,
                        cached,
                        source="cache",
                        query="",
                    )
                    counts = cached.get("counts", {}) if isinstance(cached.get("counts", {}), dict) else {}
                    msg = (
                        "Scan ALL ready from cache: "
                        f"assets={int(counts.get('assets', len(assets)) or len(assets))} "
                        f"groups={int(counts.get('groups', len(_asset_groups_from_cache(settings))) or len(_asset_groups_from_cache(settings)))} "
                        f"folders={int(counts.get('folders', len(_asset_folders_from_cache(settings))) or len(_asset_folders_from_cache(settings)))} "
                        f"| spk={len(spk_paths)} | elapsed:{time.monotonic()-t0:.1f}s"
                    )
                    settings.status = msg
                    _warno_log(settings, msg, stage="asset_index_load")
                    self.report({"INFO"}, msg)
                    return {"FINISHED"}

            extractor_path = project_root / "warno_spk_extract.py"
            self._job = {
                "done": False,
                "error": "",
                "payload": None,
                "spk_paths": [Path(p) for p in spk_paths],
                "index_path": index_path,
                "signature": signature,
                "extractor_path": extractor_path,
                "game": _normalize_game_id(getattr(settings, "game_preset", "WARNO")),
            }

            def worker():
                try:
                    payload = _build_asset_index_payload_worker(
                        extractor_path=Path(self._job["extractor_path"]),
                        spk_paths=[Path(p) for p in self._job["spk_paths"]],
                        signature=dict(self._job["signature"]),
                        index_path=Path(self._job["index_path"]),
                        game=self._job["game"],
                    )
                    self._job["payload"] = payload
                except Exception as exc:
                    self._job["error"] = str(exc)
                finally:
                    self._job["done"] = True

            self._thread = threading.Thread(target=worker, name="WARNO_AssetIndexBuilder", daemon=True)
            self._thread.start()
            self._started_at = time.monotonic()
            self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
            context.window_manager.modal_handler_add(self)
            settings.status = "Scan ALL started: building index in background..."
            _warno_log(settings, "Scan ALL started: background index build", stage="asset_index_build")
            return {"RUNNING_MODAL"}
        except Exception as exc:
            msg = f"Scan ALL failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="asset_index_prepare")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

    def modal(self, context, event):
        settings = context.scene.warno_import
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread is not None and self._thread.is_alive():
            elapsed = time.monotonic() - float(self._started_at or time.monotonic())
            settings.status = f"Scan ALL running... elapsed:{elapsed:.1f}s"
            return {"RUNNING_MODAL"}

        self._cleanup_timer(context)
        if not isinstance(self._job, dict):
            settings.status = "Scan ALL cancelled: internal job state is missing."
            self.report({"ERROR"}, settings.status)
            return {"CANCELLED"}

        err = str(self._job.get("error", "")).strip()
        if err:
            msg = f"Scan ALL failed: {err}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="asset_index_build")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        payload = self._job.get("payload")
        if not isinstance(payload, dict):
            msg = "Scan ALL failed: index payload is empty."
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="asset_index_build")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        try:
            extractor_mod = _extractor_module(settings)
            assets = _apply_picker_view_from_index(
                settings,
                extractor_mod,
                payload,
                source="rebuilt",
                query="",
            )
            counts = payload.get("counts", {}) if isinstance(payload.get("counts", {}), dict) else {}
            msg = (
                "Scan ALL done: "
                f"assets={int(counts.get('assets', len(assets)) or len(assets))} "
                f"groups={int(counts.get('groups', len(_asset_groups_from_cache(settings))) or len(_asset_groups_from_cache(settings)))} "
                f"folders={int(counts.get('folders', len(_asset_folders_from_cache(settings))) or len(_asset_folders_from_cache(settings)))} "
                f"| spk={int(counts.get('spk_count', 0) or 0)} "
                f"| source=rebuilt | elapsed:{time.monotonic()-float(self._started_at or time.monotonic()):.1f}s"
            )
            settings.status = msg
            _warno_log(settings, msg, stage="asset_index_build")
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        except Exception as exc:
            msg = f"Scan ALL finalize failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="asset_index_build")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}


class WARNO_OT_ScanAssets(Operator):
    bl_idname = "warno.scan_assets"
    bl_label = "Scan Assets"
    bl_description = "Find matching assets in Mesh SPK for current query"

    def execute(self, context):
        return _scan_assets_impl(self, context, scan_all=False)


class WARNO_OT_ScanAssetsAll(Operator):
    bl_idname = "warno.scan_assets_all"
    bl_label = "Scan ALL Assets (uses cache)"
    bl_description = "Load full asset index from cache or rebuild it in background when stale"

    def execute(self, context):
        return _scan_assets_impl(self, context, scan_all=True)


class WARNO_OT_PickAssetBrowser(Operator):
    bl_idname = "warno.pick_asset_browser"
    bl_label = "Pick Asset Browser"
    bl_description = "Pick asset by folder/model/LOD in a browser-like dialog"

    root_key: EnumProperty(name="Root", items=_browser_root_items, update=_update_browser_root_key)
    category_key: EnumProperty(name="Category", items=_browser_category_items, update=_update_browser_category_key)
    folder_key: EnumProperty(name="Folder", items=_browser_folder_items, update=_update_browser_folder_key)
    search_text: StringProperty(name="Search", default="", update=_update_browser_search_text)
    model_key: EnumProperty(name="Model", items=_browser_model_items)
    lod_key: EnumProperty(name="LOD", items=_browser_lod_items)

    def invoke(self, context, _event):
        settings = context.scene.warno_import
        folders = _asset_folders_from_cache(settings)
        if not folders:
            try:
                index_data, source, _runtime_info, _spk_paths = _ensure_asset_index_sync(settings, force_rebuild=False)
                extractor_mod = _extractor_module(settings)
                _apply_picker_view_from_index(
                    settings,
                    extractor_mod,
                    index_data,
                    source=source,
                    query="",
                )
                folders = _asset_folders_from_cache(settings)
            except Exception:
                pass
        if not folders:
            self.report({"WARNING"}, "No scanned assets. Run Scan ALL Assets first.")
            return {"CANCELLED"}

        selected_primary = str(settings.selected_asset_group or "").strip()
        selected_lod = str(settings.selected_asset_lod or "").strip()
        found_folder_key = ""
        found_model_key = ""
        for folder in folders:
            fkey = str(folder.get("key", "")).strip()
            models = folder.get("models", [])
            if not isinstance(models, list):
                continue
            for m in models:
                primary = str(m.get("primary", "")).strip()
                if primary != selected_primary:
                    continue
                found_folder_key = fkey
                found_model_key = primary
                break
            if found_folder_key:
                break

        if not found_folder_key:
            first = folders[0]
            found_folder_key = str(first.get("key", "")).strip()
            first_models = first.get("models", []) if isinstance(first.get("models", []), list) else []
            found_model_key = str(first_models[0].get("primary", "")).strip() if first_models else "__none__"

        root_key = _folder_root_key(found_folder_key)
        self.root_key = root_key
        found_category = ""
        found_folder_key_low = str(found_folder_key or "").strip().lower()
        for folder in folders:
            fkey = str(folder.get("key", "")).strip().lower()
            if fkey != found_folder_key_low:
                continue
            found_category = str(folder.get("category", "")).strip() or _asset_category_from_folder_key(fkey)
            break
        self.category_key = found_category or "__all__"
        self.folder_key = found_folder_key or "__none__"
        self.search_text = ""
        self.model_key = found_model_key or "__none__"
        if selected_lod and selected_lod != "__base__":
            self.lod_key = selected_lod
        else:
            self.lod_key = "__base__"

        return context.window_manager.invoke_props_dialog(self, width=1100)

    def draw(self, context):
        layout = self.layout
        top = layout.row(align=True)
        top.prop(self, "root_key")
        top.prop(self, "category_key")
        top.prop(self, "search_text")

        row = layout.row(align=True)
        c1 = row.column(align=True)
        c1.label(text="Folder")
        c1.prop(self, "folder_key", text="")

        c2 = row.column(align=True)
        c2.label(text="Model")
        c2.prop(self, "model_key", text="")

        c3 = row.column(align=True)
        c3.label(text="LOD")
        c3.prop(self, "lod_key", text="")

        chosen = str(self.model_key or "").strip()
        lod = str(self.lod_key or "").strip()
        if lod and lod != "__base__":
            chosen = lod
        layout.separator()
        layout.label(text=f"Will use: {chosen}", icon="OBJECT_DATA")

    def execute(self, context):
        settings = context.scene.warno_import
        model = str(self.model_key or "").strip()
        if not model or model == "__none__":
            return {"CANCELLED"}
        lod = str(self.lod_key or "").strip()
        final_asset = lod if lod and lod != "__base__" else model

        settings.asset_sync_lock = True
        try:
            _safe_set_selected_asset_group(settings, model)
            _safe_set_selected_asset_lod(settings, lod if lod else "__base__")
            _safe_set_selected_asset(settings, final_asset)
        finally:
            settings.asset_sync_lock = False
        _sync_group_lod_from_selected(settings)
        settings.status = f"Picked: {final_asset}"
        return {"FINISHED"}


class WARNO_OT_PickAssetPopup(Operator):
    bl_idname = "warno.pick_asset_popup"
    bl_label = "Pick Asset (Popup)"
    bl_description = "Popup search list for scanned assets"
    bl_options = {"INTERNAL"}

    asset_pick: EnumProperty(name="Asset", items=_asset_popup_enum_items)

    def invoke(self, context, _event):
        # Blender 5.x may return None here in some builds; operator invoke must return a set.
        result = context.window_manager.invoke_search_popup(self)
        return result if isinstance(result, set) else {"RUNNING_MODAL"}

    def execute(self, context):
        settings = context.scene.warno_import
        pick = str(self.asset_pick or "").strip()
        if not pick or pick == "__none__":
            return {"CANCELLED"}
        settings.asset_sync_lock = True
        try:
            _safe_set_selected_asset(settings, pick)
        finally:
            settings.asset_sync_lock = False
        _sync_group_lod_from_selected(settings)
        settings.status = f"Picked: {pick}"
        return {"FINISHED"}


# =====================================================================
# Tree-view Asset Browser — UIList + operators
# =====================================================================


class WARNO_UL_AssetBrowser(UIList):
    """UIList renderer for the flat-list materialised tree.

    Each row draws:
      - a depth-based indent
      - either a folder triangle (▶/▼) that toggles expansion, or a
        leaf icon (model / LOD)
      - the display name with a counter for folders
    """

    bl_idname = "WARNO_UL_asset_browser"

    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_propname,
        index,
    ):
        row = layout.row(align=True)
        # Depth-based indent. Two BLANK1 icons per level give a reasonably
        # tight tree while still being visually clear.
        for _ in range(int(item.depth)):
            row.label(text="", icon="BLANK1")

        if item.is_folder:
            tri_icon = "DOWNARROW_HLT" if item.is_expanded else "RIGHTARROW"
            op = row.operator(
                "warno.browser_toggle_folder",
                text="",
                icon=tri_icon,
                emboss=False,
            )
            op.path_key = item.path_key
            row.label(text=item.display_name, icon="FILE_FOLDER")
            if item.asset_count > 0:
                row.label(text=f"({item.asset_count})")
        elif item.is_lod:
            row.label(text="", icon="BLANK1")
            pick_op = row.operator(
                "warno.browser_pick_model",
                text=item.display_name,
                icon="MOD_DECIM",
                emboss=False,
            )
            pick_op.model_primary = item.model_primary
            pick_op.lod_variant = item.lod_variant
            pick_op.row_index = index
        else:  # model
            row.label(text="", icon="BLANK1")
            pick_op = row.operator(
                "warno.browser_pick_model",
                text=item.display_name,
                icon="OBJECT_DATA",
                emboss=False,
            )
            pick_op.model_primary = item.model_primary
            pick_op.lod_variant = ""
            pick_op.row_index = index

    def filter_items(self, context, data, propname):
        # We do our own filtering during _browser_populate_nodes via search
        # text + expand state, so let the UIList show every row that's in
        # the collection. Returning empty arrays preserves default order.
        return [], []


class WARNO_OT_BrowserToggleFolder(Operator):
    """Toggle expansion of a folder node and refresh the tree."""

    bl_idname = "warno.browser_toggle_folder"
    bl_label = "Toggle Folder"
    bl_options = {"INTERNAL"}

    path_key: StringProperty()

    def execute(self, context):
        settings = context.scene.warno_import
        key = str(self.path_key or "").strip()
        if not key:
            return {"CANCELLED"}
        expanded = _browser_serialize_expanded(settings)
        if key in expanded:
            expanded.discard(key)
        else:
            expanded.add(key)
        _browser_save_expanded(settings, expanded)
        _browser_refresh_from_settings(settings)
        return {"FINISHED"}


class WARNO_OT_BrowserClearSearch(Operator):
    """Clear the tree-browser search box."""

    bl_idname = "warno.browser_clear_search"
    bl_label = "Clear Search"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        settings = context.scene.warno_import
        settings.browser_search = ""
        _browser_refresh_from_settings(settings)
        return {"FINISHED"}


class WARNO_OT_BrowserPickModel(Operator):
    """Single-click handler on a model / LOD row in the tree browser.
    Immediately sets the selected asset (group + lod + final path), tags
    every visible UI area for redraw so the panel's 'Current:' label and
    the picker dropdown reflect the new selection right away. The popup
    stays open so the user can keep browsing; clicking OK closes it."""

    bl_idname = "warno.browser_pick_model"
    bl_label = "Pick Model"
    bl_options = {"INTERNAL"}

    model_primary: StringProperty()
    lod_variant: StringProperty()
    row_index: IntProperty(default=-1)

    def execute(self, context):
        settings = context.scene.warno_import
        primary = str(self.model_primary or "").strip()
        lod = str(self.lod_variant or "").strip()
        if not primary and not lod:
            return {"CANCELLED"}
        final_asset = lod if lod else primary
        # Sync the active row index too so OK / footer match the click.
        if self.row_index >= 0:
            try:
                settings.browser_active_index = int(self.row_index)
            except Exception:
                pass
        settings.asset_sync_lock = True
        try:
            _safe_set_selected_asset_group(settings, primary or final_asset)
            _safe_set_selected_asset_lod(settings, lod if lod else "__base__")
            _safe_set_selected_asset(settings, final_asset)
        finally:
            settings.asset_sync_lock = False
        try:
            _sync_group_lod_from_selected(settings)
        except Exception:
            pass
        # Re-assert the user's pick in case _sync_group_lod_from_selected's
        # fallback overwrote it (older path could trip when groups cache is
        # not in sync with the freshly clicked asset).
        if str(settings.selected_asset or "").strip() != final_asset:
            settings.asset_sync_lock = True
            try:
                _safe_set_selected_asset(settings, final_asset)
            finally:
                settings.asset_sync_lock = False
        settings.status = f"Picked: {final_asset}"
        # Force UI redraw across all areas so the panel and popup update now.
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        except Exception:
            pass
        return {"FINISHED"}


class WARNO_OT_BrowserExpandAll(Operator):
    """Expand every folder in the tree (or collapse to top-level only when
    already fully expanded — works as a 'Reset View' too)."""

    bl_idname = "warno.browser_expand_all"
    bl_label = "Expand / Collapse All"
    bl_options = {"INTERNAL"}

    collapse: BoolProperty(default=False)

    def execute(self, context):
        settings = context.scene.warno_import
        try:
            index_path = _asset_index_path(settings)
            cached = _load_asset_index_file(index_path) or {}
        except Exception:
            cached = {}
        if self.collapse:
            _browser_save_expanded(settings, set())
        else:
            tree = cached.get("folder_tree", []) if isinstance(cached, dict) else []
            keys: set[str] = set()

            def collect(n):
                if not isinstance(n, dict):
                    return
                k = str(n.get("key", "")).strip()
                if k:
                    keys.add(k)
                for ch in n.get("children", []) or []:
                    collect(ch)

            for root in tree:
                collect(root)
            _browser_save_expanded(settings, keys)
        _browser_refresh_from_settings(settings)
        return {"FINISHED"}


class WARNO_OT_PickAssetBrowserTree(Operator):
    """Tree-view popup picker (replaces the legacy 5-dropdown dialog)."""

    bl_idname = "warno.pick_asset_browser_tree"
    bl_label = "Asset Browser"
    bl_description = "Browse WARNO assets in a tree view (folders, models, LODs)"

    def invoke(self, context, _event):
        settings = context.scene.warno_import
        try:
            index_data, _source, _runtime, _spk = _ensure_asset_index_sync(
                settings, force_rebuild=False
            )
        except Exception as exc:
            self.report(
                {"WARNING"},
                f"No scanned asset index ({exc}). Run 'Scan ALL Assets' first.",
            )
            return {"CANCELLED"}

        # On first open, auto-expand the root(s) and any linear single-child
        # folder chain so the tree is not empty-looking — important for deep
        # single-root layouts (Wargame RD / Steel Division 2). Keep any deeper
        # expand state from a previous session.
        expanded = _browser_serialize_expanded(settings)
        auto = _browser_auto_expand_keys((index_data or {}).get("folder_tree", []) or [])
        new_keys = auto - expanded
        if new_keys:
            expanded |= new_keys
            _browser_save_expanded(settings, expanded)

        _browser_populate_nodes(settings, index_data)

        # Try to position the active row on the currently selected asset.
        sel_primary = str(settings.selected_asset_group or "").strip()
        sel_lod = str(settings.selected_asset_lod or "").strip()
        if sel_primary and sel_primary != "__none__":
            best_idx = -1
            for i, n in enumerate(settings.browser_nodes):
                if n.is_lod and sel_lod and sel_lod != "__base__":
                    if n.lod_variant == sel_lod:
                        best_idx = i
                        break
                if n.is_model and n.model_primary == sel_primary:
                    best_idx = i  # keep looking in case of LOD
            if best_idx >= 0:
                settings.browser_active_index = best_idx

        return context.window_manager.invoke_props_dialog(self, width=720)

    def draw(self, context):
        layout = self.layout
        settings = context.scene.warno_import

        # Toolbar row: search + show-LODs + expand/collapse-all
        bar = layout.row(align=True)
        bar.prop(settings, "browser_search", text="", icon="VIEWZOOM")
        if str(settings.browser_search or "").strip():
            bar.operator("warno.browser_clear_search", text="", icon="X")
        bar.separator()
        bar.prop(settings, "browser_show_lods", text="LODs", toggle=True)
        bar.separator()
        op_expand = bar.operator(
            "warno.browser_expand_all", text="", icon="DISCLOSURE_TRI_DOWN"
        )
        op_expand.collapse = False
        op_collapse = bar.operator(
            "warno.browser_expand_all", text="", icon="DISCLOSURE_TRI_RIGHT"
        )
        op_collapse.collapse = True

        # Counter line
        n = len(settings.browser_nodes)
        info = layout.row(align=True)
        info.label(text=f"{n} rows visible", icon="OUTLINER")

        # The tree itself
        layout.template_list(
            "WARNO_UL_asset_browser",
            "",
            settings,
            "browser_nodes",
            settings,
            "browser_active_index",
            rows=20,
        )

        # Selection footer
        if 0 <= settings.browser_active_index < n:
            sel = settings.browser_nodes[settings.browser_active_index]
            if sel.is_lod:
                layout.label(text=f"Selected LOD: {sel.lod_variant}", icon="MOD_DECIM")
            elif sel.is_model:
                layout.label(text=f"Selected model: {sel.model_primary}", icon="OBJECT_DATA")
            else:
                layout.label(
                    text=f"Folder: {sel.path_key}  ({sel.asset_count} models)",
                    icon="FILE_FOLDER",
                )
        else:
            layout.label(text="Click a model row to select it.", icon="INFO")

    def execute(self, context):
        settings = context.scene.warno_import
        n = len(settings.browser_nodes)
        if not (0 <= settings.browser_active_index < n):
            self.report({"WARNING"}, "No row selected.")
            return {"CANCELLED"}
        sel = settings.browser_nodes[settings.browser_active_index]
        if sel.is_folder:
            self.report({"WARNING"}, "Select a model (not a folder) before clicking OK.")
            return {"CANCELLED"}
        primary = str(sel.model_primary or "").strip()
        lod = str(sel.lod_variant or "").strip() if sel.is_lod else ""
        if not primary and not lod:
            return {"CANCELLED"}
        # Delegate to the same single-click handler so OK and direct row
        # clicks share one set-asset code path and one redraw guarantee.
        try:
            bpy.ops.warno.browser_pick_model(
                model_primary=primary,
                lod_variant=lod,
                row_index=int(settings.browser_active_index),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Pick failed: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class WARNO_OT_PrepareZZRuntime(Operator):
    bl_idname = "warno.prepare_zz_runtime"
    bl_label = "Prepare ZZ Runtime"
    bl_description = "Extract runtime Mesh/Skeleton/Atlas stubs from WARNO ZZ.dat into local cache"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        t0 = time.monotonic()
        try:
            _set_status(settings, "stage: preparing runtime", stage="runtime")
            extractor_mod = _extractor_module(settings)
            info = _prepare_zz_runtime_sources(extractor_mod, settings, force_rebuild=True)

            mesh_spk = str(info.get("mesh_spk", "")).strip()
            mesh_spk_dir = str(info.get("mesh_spk_dir", "")).strip()
            mesh_spk_files = info.get("mesh_spk_files", []) if isinstance(info.get("mesh_spk_files", []), list) else []
            skeleton_spk = str(info.get("skeleton_spk", "")).strip()
            skeleton_spk_dir = str(info.get("skeleton_spk_dir", "")).strip()
            skeleton_spk_files = info.get("skeleton_spk_files", []) if isinstance(info.get("skeleton_spk_files", []), list) else []
            atlas_root = str(info.get("atlas_assets_root", "")).strip()
            dat_count = len(info.get("zz_dat_files", []) or [])

            if mesh_spk_dir:
                settings.spk_path = mesh_spk_dir
            elif mesh_spk:
                settings.spk_path = mesh_spk
            if skeleton_spk_dir:
                settings.skeleton_spk = skeleton_spk_dir
            elif skeleton_spk:
                settings.skeleton_spk = skeleton_spk
            if atlas_root:
                settings.atlas_assets_dir = atlas_root
            settings.tgv_converter = "tgv_to_png.py"

            index_note = "asset index: pending"
            try:
                project_root = _project_root(settings)
                spk_paths = _resolve_mesh_spk_paths(project_root, settings, info)
                signature = _asset_index_signature(info, spk_paths)
                idx = _load_asset_index_file(_asset_index_path(settings))
                if idx is not None and _asset_index_signature_matches(idx, signature):
                    _apply_picker_view_from_index(
                        settings,
                        extractor_mod,
                        idx,
                        source="cache",
                        query="",
                    )
                    index_note = "asset index: cache"
                else:
                    index_note = "asset index: stale or missing (run Scan ALL Assets)"
            except Exception:
                index_note = "asset index: status unavailable"

            msg = (
                f"ZZ runtime ready: dat={dat_count}, "
                f"mesh_spk={len(mesh_spk_files) if mesh_spk_files else (1 if mesh_spk else 0)}, "
                f"skeleton_spk={len(skeleton_spk_files) if skeleton_spk_files else (1 if skeleton_spk else 0)}"
            )
            msg += f" | {index_note}"
            msg += f" | elapsed:{time.monotonic()-t0:.1f}s"
            settings.status = msg
            _warno_log(settings, msg, stage="runtime")
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        except Exception as exc:
            msg = f"ZZ runtime prepare failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="runtime")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}


class WARNO_OT_OpenTextureFolder(Operator):
    bl_idname = "warno.open_texture_folder"
    bl_label = "Open Texture Folder"
    bl_description = "Open folder where textures were saved on last import"

    @staticmethod
    def _pick_best_texture_folder(root: Path) -> Path:
        # Prefer deterministic atlas output leaf: <asset>/textures/<logical_path>/...
        textures_root = root / "textures"
        search_root = textures_root if textures_root.exists() and textures_root.is_dir() else root

        dirs_with_images: List[Path] = []
        try:
            for p in search_root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    dirs_with_images.append(p.parent.resolve())
                except Exception:
                    dirs_with_images.append(p.parent)
        except Exception:
            return search_root

        if not dirs_with_images:
            return search_root

        # Pick deepest directory with images (usually textures/<3d/.../asset_name>).
        dirs_unique = sorted(set(dirs_with_images), key=lambda d: (len(d.parts), str(d).lower()), reverse=True)
        return dirs_unique[0]

    def execute(self, context):
        settings = context.scene.warno_import
        raw = str(settings.last_texture_dir or "").strip()
        if not raw:
            self.report({"WARNING"}, "No texture folder yet. Import model with textures first.")
            return {"CANCELLED"}
        folder = Path(raw)
        if not folder.exists() or not folder.is_dir():
            self.report({"WARNING"}, f"Texture folder not found: {folder}")
            return {"CANCELLED"}
        folder = self._pick_best_texture_folder(folder)
        try:
            bpy.ops.wm.path_open(filepath=str(folder))
        except Exception as exc:
            self.report({"ERROR"}, f"Cannot open folder: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class WARNO_OT_ClearTextureFolder(Operator):
    bl_idname = "warno.clear_texture_folder"
    bl_label = "Clear texture folder from old files"
    bl_description = "Delete old generated texture images for last imported asset"

    def execute(self, context):
        settings = context.scene.warno_import
        raw = str(settings.last_texture_dir or "").strip()
        if not raw:
            self.report({"WARNING"}, "No texture folder yet. Import model with textures first.")
            return {"CANCELLED"}
        root = Path(raw)
        if not root.exists() or not root.is_dir():
            self.report({"WARNING"}, f"Texture folder not found: {root}")
            return {"CANCELLED"}

        # Remove generated texture files only (keep mesh/manifests unrelated to textures).
        deleted = 0
        failed = 0
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            low_ext = p.suffix.lower()
            if low_ext in IMAGE_EXTENSIONS:
                try:
                    p.unlink()
                    deleted += 1
                except Exception:
                    failed += 1

        # Remove empty directories after cleanup.
        for d in sorted((x for x in root.rglob("*") if x.is_dir()), key=lambda x: len(str(x)), reverse=True):
            try:
                d.rmdir()
            except Exception:
                pass

        msg = f"Texture cleanup done: deleted={deleted}"
        if failed:
            msg += f", failed={failed}"
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, msg)
        settings.status = msg
        return {"FINISHED"}


class WARNO_OT_InstallTGVDeps(Operator):
    bl_idname = "warno.install_tgv_deps"
    bl_label = "Install/Check TGV deps"
    bl_description = "Install Pillow and zstandard for the configured TGV converter Python runtime"

    def execute(self, context):
        settings = context.scene.warno_import
        try:
            extractor_mod = _extractor_module(settings)
            project_root = _project_root(settings)
            converter = project_root / "tgv_to_png.py"
            if not converter.exists() or not converter.is_file():
                raise RuntimeError(f"Bundled TGV converter not found: {converter}")

            deps_dir = _resolve_path(project_root, str(settings.tgv_deps_dir or "").strip() or ".warno_pydeps")
            ok, msg = extractor_mod.install_tgv_converter_deps(converter=converter, deps_dir=deps_dir)
            if ok:
                status = f"TGV deps ready: {msg}"
                settings.status = status
                self.report({"INFO"}, status)
                return {"FINISHED"}
            status = f"TGV deps install failed: {msg}"
            settings.status = status
            self.report({"ERROR"}, status)
            return {"CANCELLED"}
        except Exception as exc:
            status = f"TGV deps check failed: {exc}"
            settings.status = status
            self.report({"ERROR"}, status)
            return {"CANCELLED"}


class WARNO_OT_OpenSystemConsole(Operator):
    bl_idname = "warno.open_system_console"
    bl_label = "Open System Console"
    bl_description = "Open Blender system console window for live logs"

    def execute(self, context):
        settings = context.scene.warno_import
        try:
            if hasattr(bpy.ops.wm, "console_toggle"):
                bpy.ops.wm.console_toggle()
                _warno_log(settings, "System console toggled.", stage="ui")
                return {"FINISHED"}
            msg = "System console is not available on this platform/build."
            self.report({"WARNING"}, msg)
            _warno_log(settings, msg, level="WARNING", stage="ui")
            return {"CANCELLED"}
        except Exception as exc:
            msg = f"Failed to open system console: {exc}"
            self.report({"ERROR"}, msg)
            _warno_log(settings, msg, level="ERROR", stage="ui")
            return {"CANCELLED"}


class WARNO_OT_OpenLogFile(Operator):
    bl_idname = "warno.open_log_file"
    bl_label = "Open Log File"
    bl_description = "Open warno import log file from project root"

    def execute(self, context):
        settings = context.scene.warno_import
        path = _log_file_path(settings)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("", encoding="utf-8")
            bpy.ops.wm.path_open(filepath=str(path))
            _warno_log(settings, f"Opened log file: {path}", stage="ui")
            return {"FINISHED"}
        except Exception as exc:
            msg = f"Cannot open log file: {exc}"
            self.report({"ERROR"}, msg)
            _warno_log(settings, msg, level="ERROR", stage="ui")
            return {"CANCELLED"}


def _strip_texture_channel_suffix(stem: str) -> str:
    raw = str(stem or "").strip()
    if not raw:
        return ""
    out = re.sub(r"(?i)_(diffuse|albedo|color|normal|roughness|metallic|occlusion|alpha)$", "", raw)
    out = re.sub(r"(?i)_(d|nm|r|m|o|ao|a|orm|os|rs)$", "", out)
    out = re.sub(r"(?i)_part[0-9]+$", "", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out


def _forced_track_material_name(raw_name: str) -> str:
    low = _norm_low(str(raw_name or ""))
    if not low:
        return ""
    low = re.sub(r"\.\d{3}$", "", low)
    low = low.replace(" ", "_")
    low = re.sub(r"_+", "_", low).strip("_")
    if re.search(r"(?:^|_)(tracks|trk)_2$", low):
        return "Chenille_Gauche"
    if re.search(r"(?:^|_)(tracks|trk)$", low):
        return "Chenille_Droite"
    return ""


def _forced_track_material_name_for_role(raw_name: str, material_role: str) -> str:
    role_low = _norm_low(material_role)
    if role_low == "track_left":
        return "Chenille_Gauche"
    if role_low == "track_right":
        return "Chenille_Droite"

    forced = _forced_track_material_name(raw_name)
    if forced:
        return forced

    low = _norm_low(str(raw_name or ""))
    if "chenille_gauche" in low or "track_left" in low:
        return "Chenille_Gauche"
    if "chenille_droite" in low or "track_right" in low:
        return "Chenille_Droite"
    return ""


def _derive_material_roles_runtime(
    extractor_mod,
    material_ids: Sequence[int],
    raw_spk_names: Dict[int, str],
    raw_slot_names: Dict[int, Dict[str, str]],
) -> Dict[int, str]:
    def _local_track_role(text: str) -> str:
        low = _norm_low(text).replace("\\", "/")
        if not low:
            return ""
        if "chenille_droite_2" in low or "chenille_droite.2" in low:
            return "track_left"
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
            or ("tracks" in low and "tracks_2" not in low)
            or ("trk" in low and "trk_2" not in low)
        ):
            return "track_right"
        return ""

    roles: Dict[int, str] = {}
    if hasattr(extractor_mod, "derive_material_roles_from_source"):
        try:
            src_roles = extractor_mod.derive_material_roles_from_source(raw_spk_names, raw_slot_names)
        except Exception:
            src_roles = {}
    else:
        src_roles = {}

    for mid in material_ids:
        role = str(src_roles.get(int(mid), "")).strip().lower()
        if role not in {"track_left", "track_right"}:
            role = _local_track_role(str(raw_spk_names.get(int(mid), "")))
        if role not in {"track_left", "track_right"}:
            slot_map = raw_slot_names.get(int(mid), {})
            if isinstance(slot_map, dict):
                for key, value in slot_map.items():
                    role = _local_track_role(str(key or "")) or _local_track_role(str(value or ""))
                    if role:
                        break
        if role not in {"track_left", "track_right"}:
            role = "other"
        roles[int(mid)] = role
    return roles


def _build_material_name_map(
    extractor_mod,
    material_ids: Sequence[int],
    raw_spk_names: Dict[int, str],
    material_role_by_id: Dict[int, str],
    auto_name_materials: bool,
) -> Dict[int, str]:
    if not auto_name_materials:
        return {int(mid): f"Material_{int(mid):03d}" for mid in material_ids}

    used_names: set[str] = set()
    out: Dict[int, str] = {}
    for mid in material_ids:
        role_name = str(material_role_by_id.get(int(mid), "other"))
        raw_name = str(raw_spk_names.get(int(mid), "")).strip()
        if not raw_name:
            raw_name = f"Material_{int(mid):03d}"
        if hasattr(extractor_mod, "sanitize_material_name"):
            try:
                base_name = str(extractor_mod.sanitize_material_name(raw_name) or raw_name).strip()
            except Exception:
                base_name = raw_name
        else:
            base_name = raw_name
        if not base_name:
            base_name = f"Material_{int(mid):03d}"

        forced_track_name = _forced_track_material_name_for_role(base_name, role_name)
        if forced_track_name:
            base_name = forced_track_name

        unique_name = base_name
        dup_idx = 2
        while _norm_low(unique_name) in used_names:
            unique_name = f"{base_name}_{dup_idx}"
            dup_idx += 1
        used_names.add(_norm_low(unique_name))
        out[int(mid)] = unique_name
    return out


def _target_model_meshes(context, settings: WARNOImporterSettings) -> List[bpy.types.Object]:
    col_name = str(settings.last_import_collection or "").strip()
    if col_name:
        col = _collection_from_setting(context.scene, col_name)
        if col is not None:
            objs = [o for o in col.all_objects if o.type == "MESH"]
            if objs:
                return objs
    selected = [o for o in context.selected_objects if o.type == "MESH"]
    if selected:
        return selected
    return [o for o in context.scene.objects if o.type == "MESH" and str(o.get("warno_asset", "")).strip()]


def _collection_asset_from_name(col_name: str) -> str:
    name = str(col_name or "").strip()
    if not name:
        return ""
    scene = bpy.context.scene
    col = _collection_from_setting(scene, name) if scene is not None else bpy.data.collections.get(name)
    if col is None:
        return ""
    for obj in col.all_objects:
        if obj.type != "MESH":
            continue
        asset = str(obj.get("warno_asset", "")).strip()
        if asset:
            return asset
    return ""


class WARNO_OT_ApplyTextures(Operator):
    bl_idname = "warno.apply_textures"
    bl_label = "Apply/Reapply Textures"
    bl_description = "Resolve and reapply textures to the last imported or selected WARNO model without reimporting geometry"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        t0 = time.monotonic()
        _warno_log(settings, "Apply/Reapply textures started.", stage="apply_textures")
        col_name = str(settings.last_import_collection or "").strip()
        asset = _collection_asset_from_name(col_name) if col_name else ""
        if not asset:
            asset = str(settings.selected_asset or "").strip()
        if not asset or asset == "__none__":
            self.report({"WARNING"}, "No WARNO asset selected/found for texture apply.")
            return {"CANCELLED"}

        try:
            extractor_mod = _extractor_module(settings)
            project_root = _project_root(settings)
            runtime_info: Dict[str, Any] = {}
            _set_status(settings, "stage: preparing runtime", stage="apply_textures")
            runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
            _warno_log(settings, "runtime source policy: zz_runtime_only", stage="runtime")
            game = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))
            mesh_spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
            if not mesh_spk_paths:
                raise RuntimeError("No mesh SPK files found.")
            pick = _pick_best_asset_spk_path(extractor_mod, mesh_spk_paths, asset, game=game)
            if pick is None:
                raise RuntimeError(f"Asset not found in SPK sources: {asset}")
            mesh_spk_path, asset_hint = pick

            with ExitStack() as stack:
                spk = stack.enter_context(extractor_mod.SpkMeshExtractor(mesh_spk_path, game=game))
                hit = spk.find_best_fat_entry_for_asset(asset)
                if hit is None and asset_hint and asset_hint != asset:
                    hit = spk.find_best_fat_entry_for_asset(asset_hint)
                if hit is None:
                    raise RuntimeError(f"Asset not found in SPK: {asset}")
                asset_real, _meta = hit
                model = spk.get_model_geometry(asset_real)
                material_ids = sorted({int(part["material"]) for part in model["parts"]})
                raw_spk_names = getattr(spk, "material_name_by_id", {}) or {}
                raw_slot_names = getattr(spk, "material_texture_names_by_id", {}) or {}
                material_role_by_id = _derive_material_roles_runtime(
                    extractor_mod=extractor_mod,
                    material_ids=material_ids,
                    raw_spk_names=raw_spk_names,
                    raw_slot_names=raw_slot_names,
                )
                material_name_by_id = _build_material_name_map(
                    extractor_mod=extractor_mod,
                    material_ids=material_ids,
                    raw_spk_names=raw_spk_names,
                    material_role_by_id=material_role_by_id,
                    auto_name_materials=bool(settings.auto_name_materials),
                )

                model_dir = _cache_asset_dir(extractor_mod, settings, asset_real)
                model_dir.mkdir(parents=True, exist_ok=True)
                _set_status(settings, "stage: resolving textures", stage="apply_textures")
                material_maps_by_name, texture_report = _resolve_material_maps(
                    extractor_mod=extractor_mod,
                    spk=spk,
                    settings=settings,
                    asset=asset_real,
                    model_dir=model_dir,
                    material_ids=material_ids,
                    material_name_by_id=material_name_by_id,
                    material_role_by_id=material_role_by_id,
                    runtime_info=runtime_info,
                )
                _force_opaque_baked_tracks(model, material_name_by_id, material_maps_by_name)
        except Exception as exc:
            msg = f"Texture apply failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="apply_textures")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        target_meshes = _target_model_meshes(context, settings)
        if col_name:
            col = _collection_from_setting(context.scene, col_name)
            if col is not None:
                mesh_in_col = [o for o in col.all_objects if o.type == "MESH"]
                if mesh_in_col:
                    target_meshes = mesh_in_col
        if not target_meshes:
            self.report({"WARNING"}, "No target mesh objects found for texture apply.")
            return {"CANCELLED"}

        _set_status(settings, "stage: applying materials", stage="apply_textures")
        role_by_mat_key: Dict[str, str] = {}
        for mid, mname in material_name_by_id.items():
            role_by_mat_key[_material_key(mname)] = str(material_role_by_id.get(int(mid), "other"))
        maps_by_mat_key: Dict[str, Dict[str, Path]] = {}
        for mname, maps in material_maps_by_name.items():
            maps_by_mat_key[_material_key(mname)] = dict(maps)

        touched = 0
        for obj in target_meshes:
            if obj.type != "MESH" or obj.data is None:
                continue
            for mat in obj.data.materials:
                if mat is None:
                    continue
                # The datablock name may carry our per-asset namespace prefix
                # (e.g. 'M1038_Humvee__Vitre'). Try the namespaced key first
                # for correctness, then fall back to the stgot raw key so
                # legacy un-namespaced materials in older .blend files still
                # match the SPK-derived map dictionaries.
                key_full = _material_key(mat.name)
                key_raw = _material_key(_warno_strip_namespace(mat.name))
                maps = maps_by_mat_key.get(key_full) or maps_by_mat_key.get(key_raw)
                if not maps:
                    maps = (
                        material_maps_by_name.get(mat.name)
                        or material_maps_by_name.get(_warno_strip_namespace(mat.name))
                    )
                if not maps:
                    continue
                role = (
                    role_by_mat_key.get(key_full)
                    or role_by_mat_key.get(key_raw)
                    or "other"
                )
                _apply_material_nodes(
                    mat,
                    maps,
                    role,
                    ao_multiply_diffuse=False,
                    normal_invert_mode="none",
                )
                touched += 1

        tex_refs = len(texture_report.get("refs", []))
        tex_resolved = len(texture_report.get("resolved", []))
        tex_errors = len(texture_report.get("errors", []))
        atlas_source = str(texture_report.get("atlas_source", "")).strip()
        converter_source = str(texture_report.get("converter_source", "")).strip()
        deps_auto_installed = bool(texture_report.get("deps_auto_installed", False))
        msg = f"Textures reapplied: mats={touched} tex:{tex_resolved}/{tex_refs}"
        if tex_errors:
            msg += f" err:{tex_errors}"
            first = texture_report.get("errors", [{}])[0]
            first_msg = str(first.get("error", "")).strip()
            if first_msg:
                msg += f" | {first_msg[:120]}"
        if atlas_source:
            msg += f" | atlas:{atlas_source}"
        if converter_source:
            msg += f" | converter:{converter_source}"
        if deps_auto_installed:
            msg += " | deps:auto-installed"
        msg += f" | elapsed:{time.monotonic()-t0:.1f}s"
        settings.status = msg
        _warno_log(settings, msg, stage="apply_textures")
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class WARNO_OT_ManualAutoSmoothApply(Operator):
    bl_idname = "warno.manual_auto_smooth_apply"
    bl_label = "Manual Auto Smooth Apply"
    bl_description = "Apply auto smooth to all mesh objects from last imported WARNO collection"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        t0 = time.monotonic()
        col_name = str(settings.last_import_collection or "").strip()
        if not col_name:
            msg = "No imported WARNO collection found yet."
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="manual_auto_smooth")
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        collection = _collection_from_setting(context.scene, col_name)
        if collection is None:
            msg = f"Last import collection not found: {col_name}"
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="manual_auto_smooth")
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        meshes = [obj for obj in collection.objects if obj.type == "MESH"]
        if not meshes:
            msg = f"No mesh objects in collection: {col_name}"
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="manual_auto_smooth")
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        try:
            touched = _apply_auto_smooth_modifier(meshes, float(FIXED_AUTO_SMOOTH_ANGLE))
            msg = f"Manual auto smooth applied: meshes={touched}/{len(meshes)} | elapsed:{time.monotonic()-t0:.1f}s"
            settings.status = msg
            _warno_log(settings, msg, stage="manual_auto_smooth")
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        except Exception as exc:
            msg = f"Manual auto smooth failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="manual_auto_smooth")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}


class WARNO_OT_GetUVFromGame(Operator):
    bl_idname = "warno.get_uv_from_game"
    bl_label = "Get UV from game"
    bl_description = (
        "Launch the bundled in-game mesh getter (and the game, if found). In the game's "
        "Armory, show the unit you want, then press INSERT to capture its exact GPU UVs "
        "into the plugin cache. Re-import the unit afterwards — its track UV is taken 1:1 "
        "from the get. Captures only read your own offline Armory view; the game is not modified"
    )

    _EXE_BY_GAME = {"WARGAME_RD": "WarGame3.exe", "WARNO": "WARNO.exe", "STEEL_DIVISION_2": "SteelDivision2.exe"}
    _ROOT_DEFAULT = {
        "WARGAME_RD": r"C:\Program Files (x86)\Steam\steamapps\common\Wargame Red Dragon",
        "WARNO": r"F:\SteamLibrary\steamapps\common\WARNO",
        "STEEL_DIVISION_2": r"C:\Program Files (x86)\Steam\steamapps\common\Steel Division 2",
    }
    _NEW_CONSOLE = 0x00000010  # CREATE_NEW_CONSOLE

    def execute(self, context):
        import os
        import subprocess
        settings = context.scene.warno_import
        game = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))
        exe_name = self._EXE_BY_GAME.get(game, "WARNO.exe")
        addon_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        # CRITICAL: the getter must write get_cache where the EXTRACTOR reads it back.
        # The extractor module is loaded from the PROJECT ROOT (config project_root), and
        # _try_apply_get_bake reads `<extractor_dir>/get_cache` — i.e. project_root/get_cache,
        # NOT the addon dir. Writing to the addon dir (as before) meant a capture was never
        # found on re-import. Write to project_root/get_cache; fall back to the addon dir.
        try:
            cache_dir = _project_root(settings) / "get_cache"
        except Exception:
            cache_dir = addon_dir / "get_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        getter_exe = addon_dir / "game_getter" / "warno_game_getter.exe"
        getter_py = addon_dir / "game_getter" / "warno_game_getter.py"

        # 1) launch the GETTER FIRST. The post-VS capture must hook the D3D11 device
        #    BEFORE the game creates its shaders (at load), so the getter has to be
        #    poll-attaching while the game loads. The standalone exe is preferred only if
        #    it is newer than the bundled .py (a stale exe = the old pre-VS getter); else
        #    use a Python that has frida.
        args = ["--exe", exe_name, "--cache", str(cache_dir)]
        try:
            use_exe = getter_exe.exists()
            try:
                if use_exe and getter_py.exists() and getter_py.stat().st_mtime > getter_exe.stat().st_mtime + 1:
                    use_exe = False  # .py is newer than the built exe -> exe is stale
            except Exception:
                pass
            if use_exe:
                subprocess.Popen([str(getter_exe)] + args, creationflags=self._NEW_CONSOLE)
                _warno_log(settings, f"launched getter exe: {getter_exe}", stage="get")
            else:
                py = self._python_with_frida()
                if not (py and getter_py.exists()):
                    msg = ("Getter not found. Build it (tools/game_getter) or `pip install frida` "
                           "for a Python on PATH, then re-sync the addon.")
                    settings.status = msg
                    self.report({"ERROR"}, msg)
                    return {"CANCELLED"}
                subprocess.Popen([py, str(getter_py)] + args, creationflags=self._NEW_CONSOLE)
                _warno_log(settings, f"launched getter (python+frida): {getter_py}", stage="get")
        except Exception as exc:
            self.report({"ERROR"}, f"Getter launch failed: {exc}")
            return {"CANCELLED"}

        # 2) then launch the game (a moment later) if it isn't already running, so the
        #    getter — now polling — attaches during loading and wins the VS-bytecode race.
        try:
            import time as _t
            _t.sleep(1.5)
            if not self._is_running(exe_name):
                root = self._game_root(settings, game)
                exe_path = self._find_game_exe(root, exe_name) if root else None
                if exe_path:
                    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
                    _warno_log(settings, f"launched game: {exe_path}", stage="get")
                else:
                    _warno_log(settings, f"game exe '{exe_name}' not found — launch it yourself NOW (getter is polling).",
                               level="WARNING", stage="get")
            else:
                _warno_log(settings, "game already running — getter attached late (still captures, but for the cleanest "
                                     "post-VS result close the game and let the button launch it).", level="WARNING", stage="get")
        except Exception as exc:
            _warno_log(settings, f"(game auto-launch skipped: {exc})", stage="get")

        msg = ("Getter running (watch its console: VS_captured should climb on the loading screen). "
               "In the Armory show the unit, press INSERT to capture, then re-import it.")
        settings.status = msg
        _warno_log(settings, msg, stage="get")
        self.report({"INFO"}, msg)
        return {"FINISHED"}

    @staticmethod
    def _is_running(exe_name):
        try:
            import subprocess
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {exe_name}"],
                                 capture_output=True, text=True, timeout=10)
            return exe_name.lower() in (out.stdout or "").lower()
        except Exception:
            return False

    @staticmethod
    def _python_with_frida():
        import shutil
        import subprocess
        for cand in (shutil.which("python"), shutil.which("py"), shutil.which("python3")):
            if not cand:
                continue
            try:
                if subprocess.run([cand, "-c", "import frida"], capture_output=True, timeout=20).returncode == 0:
                    return cand
            except Exception:
                continue
        return None

    def _game_root(self, settings, game):
        for raw in (str(getattr(settings, "warno_root", "") or "").strip(), self._ROOT_DEFAULT.get(game, "")):
            if raw and Path(raw).is_dir():
                return Path(raw)
        return None

    @staticmethod
    def _find_game_exe(root, exe_name):
        direct = root / exe_name
        if direct.is_file():
            return direct
        try:
            for p in root.rglob(exe_name):
                return p
        except Exception:
            pass
        return None


class WARNO_OT_ImportAsset(Operator):
    bl_idname = "warno.import_asset"
    bl_label = "Import To Blender"
    bl_description = "Import selected WARNO asset directly into current scene"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        t0 = time.monotonic()
        progress_total_steps = 8
        console_toggled = _toggle_import_console(settings, open_console=True)
        # Start the cursor progress counter (visible % near the mouse, updates even while
        # this blocking operator runs) so the user can see the import is working. Torn down
        # in _toggle_import_console(open_console=False), which every import exit routes through.
        try:
            _wm = bpy.context.window_manager
            if _wm is not None:
                _wm.progress_begin(0, 100)
                _wm.progress_update(0)
        except Exception:
            pass
        _warno_log(settings, "Import started.", stage="import")
        _log_import_progress(
            settings,
            step_idx=1,
            total_steps=progress_total_steps,
            label="import start",
            t0=t0,
        )
        preexisting_object_ptrs = {int(obj.as_pointer()) for obj in bpy.data.objects}
        asset = str(settings.selected_asset or "").strip()
        if not asset or asset == "__none__":
            msg = "Pick asset first (Scan Assets)."
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="import")
            self.report({"WARNING"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}

        try:
            extractor_mod = _extractor_module(settings)
        except Exception as exc:
            msg = f"Backend load failed: {exc}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}

        project_root = _project_root(settings)
        runtime_info: Dict[str, Any] = {}
        # Purge orphan Blender datablocks (materials/images/meshes left behind
        # by previous imports the user has since deleted from the scene). This
        # prevents a stale 'Vitre' or 'Mi_24V' material from being silently
        # reused by name in the new asset's import. Safe no-op on a clean .blend.
        try:
            purged = _warno_purge_orphan_datablocks()
            if any(purged.values()):
                _warno_log(
                    settings,
                    "pre-import orphan purge: "
                    + ", ".join(f"{k}={v}" for k, v in purged.items() if v),
                    stage="import",
                )
        except Exception:
            pass
        try:
            _set_status(settings, "stage: preparing runtime", stage="import")
            runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
            _warno_log(settings, "runtime source policy: zz_runtime_only", stage="runtime")
        except Exception as exc:
            msg = f"ZZ runtime prepare failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}
        mesh_spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
        if not mesh_spk_paths:
            msg = "No mesh SPK files found in prepared ZZ runtime."
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}
        _log_import_progress(
            settings,
            step_idx=2,
            total_steps=progress_total_steps,
            label="runtime prepared",
            t0=t0,
            extra=f"mesh_spk_sources={len(mesh_spk_paths)}",
        )
        game = _normalize_game_id(getattr(settings, "game_preset", "WARNO"))
        dev_scale, off_mat_scale = _active_game_scales(settings)
        pick = _pick_best_asset_spk_path(extractor_mod, mesh_spk_paths, asset, game=game)
        if pick is None:
            msg = f"Asset not found in selected SPK sources: {asset}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}
        mesh_spk_path, asset_hint = pick
        _log_import_progress(
            settings,
            step_idx=3,
            total_steps=progress_total_steps,
            label="asset source resolved",
            t0=t0,
            extra=f"spk={Path(mesh_spk_path).name}",
        )

        need_bone_map = bool(settings.auto_split_main_parts or settings.auto_pull_bones)
        rot = extractor_mod.build_rotation_params(
            0.0,
            0.0,
            0.0,
            mirror_y=True,
        )

        material_name_by_id: Dict[int, str] = {}
        material_role_by_id: Dict[int, str] = {}
        material_maps_by_name: Dict[str, Dict[str, Path]] = {}
        texture_report: dict[str, Any] = {"refs": [], "resolved": [], "errors": [], "named": [], "channels": []}
        texture_dir_to_open = ""
        bone_payload: dict[str, Any] = {
            "bone_name_by_index": {},
            "raw_bone_name_by_index": {},
            "bone_parent_by_index": {},
            "bone_names": [],
            "raw_bone_names": [],
            "bone_positions": {},
        }

        try:
            with ExitStack() as stack:
                _set_status(settings, "stage: reading SPK/model", stage="import")
                spk = stack.enter_context(extractor_mod.SpkMeshExtractor(mesh_spk_path, game=game))
                hit = spk.find_best_fat_entry_for_asset(asset)
                if hit is None and asset_hint and asset_hint != asset:
                    hit = spk.find_best_fat_entry_for_asset(asset_hint)
                if hit is None:
                    raise RuntimeError(f"Asset not found in SPK: {asset} ({mesh_spk_path})")
                asset_real, meta = hit

                variant_note = ""
                try:
                    picked_asset, picked_note = _choose_asset_variant_for_import(extractor_mod, spk, asset_real)
                    if picked_asset and picked_asset != asset_real:
                        hit2 = spk.find_best_fat_entry_for_asset(picked_asset)
                        if hit2 is not None:
                            asset_real, meta = hit2
                            variant_note = f"variant={Path(asset_real).name} ({picked_note})"
                except Exception:
                    pass

                # Decoding the mesh is the longest blocking phase (per-part IRISZoom decode,
                # track UV, bone names). Feed a per-part callback so the progress bar/cursor
                # MOVE during it (otherwise it sat at 38% for ~20s and looked frozen). The
                # decode spans the 38%->60% band of the overall 8-step bar; throttle to int %.
                _decode_last_pct = [-1]

                def _decode_progress(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    pct = int(round(38.0 + 22.0 * (float(done) / float(total))))
                    if pct == _decode_last_pct[0]:
                        return
                    _decode_last_pct[0] = pct
                    try:
                        cells = 24
                        filled = max(0, min(cells, int(round(cells * pct / 100.0))))
                        print(f"[WARNO] {'█' * filled}{'░' * (cells - filled)} {pct:3d}% | "
                              f"decoding mesh part {done + 1}/{total}", flush=True)
                    except Exception:
                        pass
                    try:
                        wm = bpy.context.window_manager
                        if wm is not None:
                            wm.progress_update(int(pct))
                            wm.status_text_set(f"WARNO import {pct}% — decoding mesh ({done + 1}/{total})")
                    except Exception:
                        pass

                model = spk.get_model_geometry(asset_real, progress_cb=_decode_progress)
                material_ids = sorted({int(part["material"]) for part in model["parts"]})
                raw_spk_names = getattr(spk, "material_name_by_id", {}) or {}
                raw_slot_names = getattr(spk, "material_texture_names_by_id", {}) or {}
                material_role_by_id = _derive_material_roles_runtime(
                    extractor_mod=extractor_mod,
                    material_ids=material_ids,
                    raw_spk_names=raw_spk_names,
                    raw_slot_names=raw_slot_names,
                )
                material_name_by_id = _build_material_name_map(
                    extractor_mod=extractor_mod,
                    material_ids=material_ids,
                    raw_spk_names=raw_spk_names,
                    material_role_by_id=material_role_by_id,
                    auto_name_materials=bool(settings.auto_name_materials),
                )

                skeleton_spks: List[Any] = []
                if need_bone_map:
                    for skeleton_path in _resolve_skeleton_spk_paths(project_root, settings, runtime_info):
                        try:
                            skeleton_spks.append(stack.enter_context(extractor_mod.SpkMeshExtractor(skeleton_path, game=game)))
                        except Exception:
                            continue

                warno_root_path = _resolve_path(project_root, str(settings.warno_root or "").strip())
                modding_suite_root_path = _resolve_path(project_root, str(settings.modding_suite_root or "").strip())
                gfx_manifest_info: Dict[str, Any] = {}
                if need_bone_map:
                    try:
                        gfx_resolver = extractor_mod.GfxManifestResolver(
                            warno_root=warno_root_path,
                            modding_suite_root=modding_suite_root_path,
                            cache_dir=_gfx_json_cache_root(settings),
                            wrapper_path=_resolve_path(
                                project_root,
                                str(settings.modding_suite_gfx_wrapper or "").strip() or "modding_suite_gfx_export.py",
                            ),
                            gfx_cli_path=_resolve_path(
                                project_root,
                                str(settings.modding_suite_gfx_cli or "").strip() or "moddingSuite/moddingSuite.exe",
                            ),
                            timeout_sec=int(settings.gfx_cli_timeout_sec),
                            use_gfx_json_manifest=bool(settings.use_gfx_json_manifest),
                            legacy_ndf_source=None,
                            legacy_operators_source=None,
                            enable_operator_semantics=False,
                            game=game,
                        )
                        gfx_manifest_info = gfx_resolver.manifest_for_asset(asset_real)
                    except Exception as exc:
                        gfx_manifest_info = {
                            "source": "none",
                            "error": str(exc),
                            "semantic_bones": [],
                            "track_kind": "",
                            "matched_units": [],
                            "weapon_fx_anchors": [],
                            "turrets": [],
                            "subdepictions": [],
                            "source_files": [],
                        }
                    _warno_log(
                        settings,
                        (
                            "gfx manifest: "
                            f"source={str(gfx_manifest_info.get('source', 'none'))} "
                            f"track_kind={str(gfx_manifest_info.get('track_kind', '') or 'none')} "
                            f"matched_units={len(gfx_manifest_info.get('matched_units', []) or [])} "
                            f"turrets={len(gfx_manifest_info.get('turrets', []) or [])} "
                            f"fx={len(gfx_manifest_info.get('weapon_fx_anchors', []) or [])} "
                            f"subdepictions={len(gfx_manifest_info.get('subdepictions', []) or [])}"
                        ),
                        level="WARNING" if str(gfx_manifest_info.get("source", "none")) == "legacy_ndf_mirror" else "INFO",
                        stage="gfx",
                    )

                if need_bone_map:
                    bone_payload = _build_bone_payload(
                        extractor_mod=extractor_mod,
                        spk=spk,
                        model=model,
                        asset=asset_real,
                        meta=meta,
                        material_role_by_id=material_role_by_id,
                        rot=rot,
                        skeleton_spks=skeleton_spks,
                        gfx_manifest_info=gfx_manifest_info,
                        dev_scale=dev_scale,
                        off_mat_scale=off_mat_scale,
                    )
                    _warno_log(
                        settings,
                        (
                            "bone payload: "
                            f"source={bone_payload.get('bone_name_source', 'none')} "
                            f"names={len(bone_payload.get('bone_names', []) or [])} "
                            f"raw_names={len(bone_payload.get('raw_bone_names', []) or [])} "
                            f"indexed={len(bone_payload.get('bone_name_by_index', {}) or {})} "
                            f"raw_indexed={len(bone_payload.get('raw_bone_name_by_index', {}) or {})} "
                            f"positions={len(bone_payload.get('bone_positions', {}) or {})} "
                            f"gfx={str(bone_payload.get('gfx_manifest_source', 'none'))}"
                        ),
                        stage="import",
                    )

                model_dir = _cache_asset_dir(extractor_mod, settings, asset_real)
                model_dir.mkdir(parents=True, exist_ok=True)
                texture_dir_to_open = str(model_dir.resolve())
                _set_status(settings, "stage: resolving textures", stage="import")
                material_maps_by_name, texture_report = _resolve_material_maps(
                    extractor_mod=extractor_mod,
                    spk=spk,
                    settings=settings,
                    asset=asset_real,
                    model_dir=model_dir,
                    material_ids=material_ids,
                    material_name_by_id=material_name_by_id,
                    material_role_by_id=material_role_by_id,
                    runtime_info=runtime_info,
                )
                _force_opaque_baked_tracks(model, material_name_by_id, material_maps_by_name)
                _log_import_progress(
                    settings,
                    step_idx=5,
                    total_steps=progress_total_steps,
                    label="textures resolved",
                    t0=t0,
                    extra=(
                        f"refs={len(texture_report.get('refs', []) or [])} "
                        f"resolved={len(texture_report.get('resolved', []) or [])} "
                        f"errors={len(texture_report.get('errors', []) or [])}"
                    ),
                )

                buckets = _collect_mesh_buckets(
                    extractor_mod=extractor_mod,
                    model=model,
                    rot=rot,
                    material_name_by_id=material_name_by_id,
                    material_role_by_id=material_role_by_id,
                    bone_name_by_index=bone_payload.get("bone_name_by_index", {}),
                    bone_parent_by_index=bone_payload.get("bone_parent_by_index", {}),
                    bone_positions=bone_payload.get("bone_positions", {}) or {},
                    split_main_parts=bool(settings.auto_split_main_parts),
                    dev_scale=dev_scale,
                    material_maps_by_name=material_maps_by_name,
                    is_rd_asset=str(getattr(settings, "game_preset", "") or "").upper() in ("WARGAME_RD", "STEEL_DIVISION_2"),
                )
                # Diagnostic: shows whether the RD track-split / body-whole path ran.
                # If you see many "Part_NNN" groups here for an RD unit, the running
                # code is stale (see the build-stamp line above) — restart Blender.
                _rd_model = any(bool(p.get("vertices", {}).get("chenille")) for p in model.get("parts", []))
                _warno_log(
                    settings,
                    "mesh buckets: %d objects [rd_model=%s] -> %s"
                    % (len(buckets), _rd_model, ", ".join(str(b.get("group_name", "?")) for b in buckets)),
                    stage="import",
                )
                # Per-part belt diagnostic (RD): nverts, chenille/cutout flags, whether the
                # within-tile V reconstruction fired (belt_vfixed) and the raw V span
                # (belt_vmax) — tells us if a belt's UV is file-reconstructable (vmax~8) or
                # flat/constant (vmax<2 -> needs a post-VS capture). Helps diagnose chains.
                try:
                    for _pi, _p in enumerate(model.get("parts", []) or []):
                        _v = _p.get("vertices", {}) or {}
                        if not isinstance(_v, dict):
                            continue
                        _nv = len(_v.get("xyz", []) or []) // 3
                        _mn = material_name_by_id.get(int(_p.get("material", -1)), "")
                        _mm = (material_maps_by_name or {}).get(_mn) or {}
                        _warno_log(
                            settings,
                            "part %d mat=%s nverts=%d chenille=%s cutout=%s belt_vfixed=%s belt_vmax=%.3f"
                            % (_pi, _mn, _nv, bool(_v.get("chenille")),
                               bool(_mm.get("alpha_cutout")), bool(_v.get("belt_vfixed")),
                               float(_v.get("belt_vmax") or 0.0)),
                            stage="import",
                        )
                except Exception:
                    pass
                wheel_groups = 0
                track_groups = 0
                turret_chain_groups = 0
                for b in buckets:
                    gname_low = _norm_low(str(b.get("group_name", "")))
                    if gname_low.startswith("roue_") or "wheel" in gname_low:
                        wheel_groups += 1
                    if gname_low.startswith("chenille_") or "track" in gname_low:
                        track_groups += 1
                    if gname_low in {"tourelle_01", "axe_canon_01", "canon_01"}:
                        turret_chain_groups += 1
                _warno_log(
                    settings,
                    (
                        "split_mode=bone_deterministic "
                        f"groups_total={len(buckets)} "
                        f"groups_wheels={wheel_groups} "
                        f"groups_tracks={track_groups} "
                        f"groups_turret_chain={turret_chain_groups}"
                    ),
                    stage="import",
                )
                _log_import_progress(
                    settings,
                    step_idx=6,
                    total_steps=progress_total_steps,
                    label="mesh groups prepared",
                    t0=t0,
                    extra=f"groups_total={len(buckets)}",
                )
        except Exception as exc:
            msg = f"Import prep failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}

        if not buckets:
            msg = "No mesh geometry to import."
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="import")
            self.report({"WARNING"}, msg)
            _toggle_import_console(settings, open_console=False, active=console_toggled)
            return {"CANCELLED"}

        collection = context.scene.collection
        settings.last_import_collection = WARNO_SCENE_COLLECTION_SENTINEL
        imported_objects: List[bpy.types.Object] = []
        material_cache: Dict[str, bpy.types.Material] = {}
        mesh_count = 0
        mesh_group_vertex_groups_added = 0
        track_vertex_groups_added = 0
        track_vertex_groups_synth = 0
        track_zero_weight_before = 0
        track_zero_weight_after = 0
        used_object_names: set[str] = set()
        hierarchy_root_obj: bpy.types.Object | None = None
        scene_bucket_diagnostics: List[Dict[str, Any]] = []
        try:
            if "warno_bucket_diagnostics_json" in context.scene.keys():
                del context.scene["warno_bucket_diagnostics_json"]
        except Exception:
            pass

        # RD/SD2 model? (any part carries the IRISZoom chenille flag). Gates the track
        # tube-unwrap to RD only — WARNO tracks are also named chenille_* but must keep
        # their own UV untouched (WARNO never sets the chenille flag).
        _model_is_rd = any(bool(p.get("vertices", {}).get("chenille")) for p in model.get("parts", []))
        for bucket_i, bucket in enumerate(buckets, start=1):
            group_name = str(bucket.get("group_name", f"Part_{bucket_i:03d}"))
            group_bone_index = int(bucket.get("group_bone_index", -1))
            node_name_map = bone_payload.get("raw_bone_name_by_index", {}) or bone_payload.get("bone_name_by_index", {}) or {}
            node_name_raw = str(node_name_map.get(group_bone_index, "")).strip()
            raw_name_values = {
                _norm_low(str(v))
                for v in list(node_name_map.values())
                if str(v).strip()
            }
            vehicle_like_asset = _raw_scene_looks_vehicle_like(sorted(raw_name_values))
            papyrus_character_mesh_proxy = (
                "papyrus" in raw_name_values
                and "soldat" in raw_name_values
                and _norm_low(node_name_raw).startswith("bip01")
            )
            _peeled_name = str(bucket.get("warno_peeled_name", "") or "").strip()
            if _peeled_name:
                # geometry-peeled antenna/detail: name it by its own label, NOT the inherited
                # chassis bone (which made antennas import as "Chassis_2/Chassis_3").
                preferred_group_name = _peeled_name
            elif papyrus_character_mesh_proxy:
                preferred_group_name = _display_warno_node_name("soldat", asset_real)
                soldat_proxy_index = next(
                    (
                        int(idx)
                        for idx, name in (node_name_map.items() if isinstance(node_name_map, dict) else [])
                        if _norm_low(str(name)) == "soldat"
                    ),
                    group_bone_index,
                )
                group_bone_index = int(soldat_proxy_index)
            else:
                _disp = _display_warno_node_name(node_name_raw, asset_real) if node_name_raw else ""
                # RD/SD2 skeleton nodes store no readable names (control bytes), so the
                # display name sanitises to empty -> fall back to the bucket's semantic
                # group name ("Chenille"/"MainBody") instead of a meaningless "Part_NNN".
                preferred_group_name = _disp if _safe_name(_disp, "").strip() else group_name
            if settings.auto_name_parts:
                base_name = _safe_name(preferred_group_name, f"Part_{bucket_i:03d}")[:63]
            else:
                base_name = f"Part_{bucket_i:03d}"
            obj_name = base_name
            suffix = 2
            while _norm_low(obj_name) in used_object_names:
                tail = f"_{suffix}"
                cut = max(1, 63 - len(tail))
                obj_name = f"{base_name[:cut]}{tail}"
                suffix += 1
            used_object_names.add(_norm_low(obj_name))

            mesh = bpy.data.meshes.new(obj_name)
            mesh.from_pydata(bucket["vertices"], [], bucket["faces"])
            mesh.update()
            uv_layer = mesh.uv_layers.new(name="UVMap")
            if uv_layer is not None:
                # Every part uses its GAME-decoded TexCoord0 UV (1:1). For RD track/chains
                # materials whose atlas half was split into a 1:1 crop
                # (_split_track_chains_textures sets maps["uv_crop"]=(v0,v1)), remap that
                # face's V into [0,1] of the crop so the UV fills the cropped texture like
                # the Ninja-Rip — v' = (v - v0)/(v1 - v0). All other faces keep the exact UV.
                _bu = bucket["uvs"]; _fmids = bucket["face_mids"]
                _crop_by_mid: Dict[int, Any] = {}
                for _mid in {int(m) for m in _fmids}:
                    _mm = material_maps_by_name.get(material_name_by_id.get(_mid))
                    if isinstance(_mm, dict) and _mm.get("uv_crop"):
                        _crop_by_mid[_mid] = _mm["uv_crop"]
                for fi, poly in enumerate(mesh.polygons):
                    vc = _crop_by_mid.get(int(_fmids[fi])) if fi < len(_fmids) else None
                    for li, vi in zip(poly.loop_indices, poly.vertices):
                        if 0 <= vi < len(_bu):
                            u, v = _bu[vi]
                            if vc is not None:
                                v0, v1 = vc
                                if (v1 - v0) > 1e-6:
                                    v = (v - v0) / (v1 - v0)
                            uv_layer.data[li].uv = (u, v)
                        else:
                            uv_layer.data[li].uv = (0.0, 0.0)

            mids_order: List[int] = []
            for mid in bucket["face_mids"]:
                if int(mid) not in mids_order:
                    mids_order.append(int(mid))

            mat_slot_by_mid: Dict[int, int] = {}
            asset_namespace = _warno_asset_namespace(asset)
            for mid in mids_order:
                raw_mat_name = material_name_by_id.get(mid, f"Material_{mid:03d}")
                role = str(material_role_by_id.get(mid, "other"))
                maps = material_maps_by_name.get(raw_mat_name, {})
                # Track meshes: name the material after the per-SIDE bucket
                # (Chenille_Gauche/Droite/Chains) so each split belt gets its own
                # distinctly-named material even when both RD belts share one material
                # id. Textures still come from the real material (maps unchanged); the
                # post-import finalize later strips the namespace to the object name.
                display_mat_name = raw_mat_name
                _gn_low = _norm_low(str(group_name or ""))
                if _gn_low.startswith("chenille") or _gn_low.startswith("chains"):
                    display_mat_name = str(group_name)
                # Per-asset cache key keeps the in-import dedup intact while
                # the Blender datablock name itself is namespaced.
                cache_key = f"{display_mat_name.lower()}::{role}::{raw_mat_name.lower()}"
                mat = material_cache.get(cache_key)
                if mat is None:
                    mat_name = _warno_namespaced_material_name(asset_namespace, display_mat_name)
                    mat = bpy.data.materials.get(mat_name)
                    if mat is None:
                        mat = bpy.data.materials.new(name=mat_name)
                    _apply_material_nodes(
                        mat,
                        maps,
                        role,
                        ao_multiply_diffuse=False,
                        normal_invert_mode="none",
                    )
                    # Flag RD/SD2 materials so the post-import pass can de-namespace them
                    # to clean WARNO-style bare names (WARNO materials never carry this).
                    if maps.get("rd_material"):
                        try:
                            mat["warno_rd_material"] = True
                        except Exception:
                            pass
                    material_cache[cache_key] = mat
                mesh.materials.append(mat)
                mat_slot_by_mid[mid] = len(mesh.materials) - 1

            for face_i, poly in enumerate(mesh.polygons):
                if face_i < len(bucket["face_mids"]):
                    mid = int(bucket["face_mids"][face_i])
                    poly.material_index = int(mat_slot_by_mid.get(mid, 0))

            obj = bpy.data.objects.new(obj_name, mesh)
            obj["warno_group"] = preferred_group_name
            obj["warno_asset"] = asset
            obj["warno_group_bone_index"] = int(group_bone_index)
            # Track meshes skip merge-by-distance (keep the clean game vertex topology so
            # the seam UV isn't collapsed). Match WARNO 'chenille_droite/gauche' AND the RD
            # 'Chenille' object (exact, no underscore) — the old startswith('chenille_')
            # missed the RD name, so RD tracks were wrongly merged 908->412.
            obj["warno_is_track_mesh"] = bool(_norm_low(obj_name).startswith("chenille") or "track" in _norm_low(obj_name))
            # RD ball-chains object: skip merge-by-distance too (collapsing verts would
            # destroy the cutout silhouette), same as the track mesh.
            obj["warno_is_chains_mesh"] = bool(_norm_low(obj_name) == "chains")
            collection.objects.link(obj)
            # RD auto-split parts have no bone pivots (RD bone names are unusable), so
            # give each its OWN origin = the part's geometric centroid (the wheel pivot
            # for wheels, the body/gun/turret centre otherwise). WARNO/non-RD buckets
            # never set warno_rd_origin and keep their bone-driven origins untouched.
            if bool(bucket.get("warno_rd_origin")) and getattr(mesh, "vertices", None):
                try:
                    _vn = len(mesh.vertices)
                    if _vn > 0:
                        _cx = sum(v.co.x for v in mesh.vertices) / _vn
                        _cy = sum(v.co.y for v in mesh.vertices) / _vn
                        _cz = sum(v.co.z for v in mesh.vertices) / _vn
                        _set_object_origin_world(obj, Vector((_cx, _cy, _cz)))
                        # Mark so the later armature pass does not clobber this centroid
                        # origin with the bone pivot (RD wheel/track bones sit at 0,0,0).
                        obj["warno_rd_centroid_origin"] = True
                except Exception:
                    pass
            bucket_diag = dict(bucket.get("diagnostics", {}) or {})
            if bucket_diag:
                bucket_diag["object_name"] = str(obj_name)
                bucket_diag["preferred_group_name"] = str(preferred_group_name)
                scene_bucket_diagnostics.append(bucket_diag)
            _ensure_window_material_name(obj, asset_hint=asset)
            vg_report = _apply_vertex_groups_from_bucket(
                obj=obj,
                bucket=bucket,
                model=model,
                bone_name_by_index=bone_payload.get("bone_name_by_index", {}) or {},
                raw_bone_name_by_index=bone_payload.get("raw_bone_name_by_index", {}) or {},
                bone_positions=bone_payload.get("bone_positions", {}) or {},
            )
            if papyrus_character_mesh_proxy and len(getattr(obj, "vertex_groups", [])) == 0:
                raw_name_map = bone_payload.get("raw_bone_name_by_index", {}) or {}
                fallback_bip_names = sorted(
                    {
                        _pretty_character_bip_name(str(raw_name))
                        for raw_name in raw_name_map.values()
                        if _norm_low(str(raw_name)).startswith("bip01")
                    },
                    key=str.lower,
                )
                for vg_name in fallback_bip_names:
                    if not vg_name or obj.vertex_groups.get(str(vg_name)) is not None:
                        continue
                    obj.vertex_groups.new(name=str(vg_name))
                if fallback_bip_names:
                    vg_report["group_added"] = int(vg_report.get("group_added", 0)) + len(fallback_bip_names)
            if (
                vehicle_like_asset
                and not papyrus_character_mesh_proxy
                and _asset_prefers_synthetic_default_group(asset_real)
                and len(getattr(obj, "vertex_groups", [])) == 0
                and not bool(obj.get("warno_is_track_mesh", False))
                and not _skip_default_group_assignment(_norm_low(obj.name))
            ):
                obj.vertex_groups.new(name="Group")
                vg_report["group_added"] = int(vg_report.get("group_added", 0)) + 1
            mesh_group_vertex_groups_added += int(vg_report.get("group_added", 0))
            track_vertex_groups_added += int(vg_report.get("track_groups_added", 0))
            track_vertex_groups_synth += int(vg_report.get("track_groups_synth", 0))
            track_zero_weight_before += int(vg_report.get("track_zero_weight_before", 0))
            track_zero_weight_after += int(vg_report.get("track_zero_weight_after", 0))
            if bool(obj.get("warno_is_track_mesh", False)):
                group_sources = dict(vg_report.get("track_group_sources", {}) or {})
                if bucket_diag and group_sources:
                    bucket_diag["track_group_sources"] = group_sources
                    bucket_diag["track_groups_explicit"] = int(vg_report.get("track_groups_explicit", 0))
                    bucket_diag["track_groups_synth"] = int(vg_report.get("track_groups_synth", 0))
                    bucket_diag["track_zero_weight_before"] = int(vg_report.get("track_zero_weight_before", 0))
                    bucket_diag["track_zero_weight_after"] = int(vg_report.get("track_zero_weight_after", 0))
                if group_sources:
                    try:
                        obj["warno_track_group_sources_json"] = json.dumps(
                            group_sources,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    except Exception:
                        pass
                    source_chunks = [
                        f"{name}:{group_sources.get(name, 'raw')}"
                        for name in sorted(group_sources.keys(), key=_track_group_sort_key)
                    ]
                    _warno_log(
                        settings,
                        (
                            f"track weight groups obj={obj_name} "
                            f"explicit={int(vg_report.get('track_groups_explicit', 0))} "
                            f"synth={int(vg_report.get('track_groups_synth', 0))} "
                            f"zero_before={int(vg_report.get('track_zero_weight_before', 0))} "
                            f"zero_after={int(vg_report.get('track_zero_weight_after', 0))} "
                            f"sources={','.join(source_chunks)}"
                        ),
                        stage="track_weights",
                    )
            imported_objects.append(obj)
            mesh_count += 1
            if bucket_i == 1 or bucket_i == len(buckets) or (bucket_i % 5) == 0:
                _warno_log(
                    settings,
                    (
                        f"mesh build progress: {bucket_i}/{len(buckets)} "
                        f"({int(round((float(bucket_i) / float(max(1, len(buckets)))) * 100.0))}%) "
                        f"obj={obj_name} verts={len(bucket.get('vertices', []) or [])} "
                        f"faces={len(bucket.get('faces', []) or [])}"
                    ),
                    stage="import_progress",
                )

        _warno_log(
            settings,
            (
                f"mesh vertex groups: default_group={mesh_group_vertex_groups_added} "
                f"track_groups={track_vertex_groups_added} "
                f"track_groups_synth={track_vertex_groups_synth} "
                f"track_zero_before={track_zero_weight_before} "
                f"track_zero_after={track_zero_weight_after}"
            ),
            stage="import",
        )
        if scene_bucket_diagnostics:
            try:
                context.scene["warno_bucket_diagnostics_json"] = json.dumps(
                    scene_bucket_diagnostics,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except Exception:
                pass

        # Geometry cleanup on import: merge by distance first.
        _set_status(settings, "stage: building scene objects", stage="import")
        if bool(settings.use_merge_by_distance):
            _merge_by_distance(imported_objects, float(settings.merge_distance))
        # NOTE: hole-fill (v1.7.4/v1.7.6) is DISABLED — filling the game's open-shell gaps
        # made flat sheets across large gaps that look like folds/creases. The open edges are
        # the game's own mesh; leave them. (_fill_rd_mesh_holes kept but not called.)
        auto_smooth_mode = str(settings.auto_smooth_mode or "OFF").upper()
        if auto_smooth_mode in {"MODIFIER", "APPLY"}:
            try:
                if auto_smooth_mode == "MODIFIER":
                    _ensure_named_auto_smooth_modifiers(imported_objects, float(FIXED_AUTO_SMOOTH_ANGLE))
                else:
                    _apply_auto_smooth_modifier(imported_objects, float(FIXED_AUTO_SMOOTH_ANGLE))
            except Exception as exc:
                _warno_log(settings, f"auto_smooth warning: {exc}", level="WARNING", stage="import")
        # Track weights are already synthesized from raw helper positions during mesh build.
        # A second scene-space rewrite pass tends to rename upper/endcap patches incorrectly
        # on vehicles like Leopard where missing helper ordinals must stay segment-aware.
        # Keep the first-pass weights authoritative.
        # _refine_track_weights_from_mesh_positions(imported_objects, settings=settings)

        # Record pre-rig WORLD VERTEX POSITIONS of rotor/engine parts (Helice, Bloc_Moteur, ...) so a
        # later pass can detect & undo a BAD bone-pull that displaces OR re-orients them off their
        # correct FBX pose (Ka-50: the rig drags the gearbox up AND tips it 90 deg onto its side).
        _rotor_engine_rx = re.compile(r"(helice|rotor|\bpale|propeller|rotopale|moteur)", re.IGNORECASE)
        _prepull_world = {}
        for _o in imported_objects:
            if getattr(_o, "type", "") == "MESH" and _o.data and len(_o.data.vertices) and _rotor_engine_rx.search(getattr(_o, "name", "")):
                _prepull_world[_o.name] = [(_o.matrix_world @ _v.co).copy() for _v in _o.data.vertices]

        if settings.auto_pull_bones:
            try:
                hierarchy_root_obj = _build_helper_armature(imported_objects, bone_payload, collection, settings=settings)
            except Exception as exc:
                self.report({"WARNING"}, f"Hierarchy build failed: {exc}")
                hierarchy_root_obj = None
            # Snap any wheel flung to the wrong side by a duplicate/bogus bone (Leopard_1V
            # Roue_D1) back to the mirror of its opposite-side counterpart. No-op when all
            # wheels are already on their rows.
            try:
                _nfix = _fix_outlier_wheels(imported_objects, collection)
                if _nfix:
                    _warno_log(settings, f"outlier wheel fix: {_nfix} wheel(s) snapped to mirror", stage="import")
            except Exception:
                pass
            # A2 + A1-A: split a merged WARNO track belt (both sides in Chenille_Droite) into
            # Droite/Gauche and weight each belt to its side wheels. No-op for single-side
            # (RD / already-split) tracks.
            try:
                _nsplit = _split_and_weight_merged_track(imported_objects, collection)
                if _nsplit:
                    _warno_log(settings, f"track split+weight: {_nsplit} merged belt(s) split into Droite/Gauche", stage="import")
            except Exception as _exc:
                _warno_log(settings, f"track split+weight failed: {_exc}", level="WARNING", stage="import")

        # Undo a BAD bone-pull: if the rig moved a rotor/engine part >0.4 m OR re-oriented it
        # (Ka-50: the Bloc_Moteur gearbox floats up AND tips 90 deg onto its side), restore its
        # EXACT pre-rig world geometry (position + orientation). No-op when bones are off or the
        # pull was small and axis-aligned.
        try:
            _nrev = 0
            for _o in imported_objects:
                _pw = _prepull_world.get(getattr(_o, "name", ""))
                if not _pw or getattr(_o, "type", "") != "MESH" or not _o.data or len(_pw) != len(_o.data.vertices):
                    continue
                _cur = [_o.matrix_world @ _v.co for _v in _o.data.vertices]
                _n = len(_cur)
                _pcen = Vector((0.0, 0.0, 0.0)); _ccen = Vector((0.0, 0.0, 0.0))
                for _p in _pw:
                    _pcen = _pcen + _p
                for _p in _cur:
                    _ccen = _ccen + _p
                _pcen = _pcen / _n; _ccen = _ccen / _n
                # per-axis world-bbox dims catch a re-orientation even when the centroid is unchanged
                _pd = [max(p[i] for p in _pw) - min(p[i] for p in _pw) for i in range(3)]
                _cd = [max(p[i] for p in _cur) - min(p[i] for p in _cur) for i in range(3)]
                _moved = (_pcen - _ccen).length > 0.4
                _tipped = max(abs(_pd[i] - _cd[i]) for i in range(3)) > 0.4
                if _moved or _tipped:
                    _mwi = _o.matrix_world.inverted()
                    for _i, _v in enumerate(_o.data.vertices):
                        _v.co = _mwi @ _pw[_i]
                    _o.data.update()
                    _nrev += 1
            if _nrev:
                _warno_log(settings, f"rotor/engine de-displace: {_nrev} part(s) restored to FBX pose", stage="import")
        except Exception as _exc:
            _warno_log(settings, f"rotor/engine de-displace failed: {_exc}", level="WARNING", stage="import")

        # Rotor geometry fixes — pure mesh repairs, so run them OUTSIDE the auto_pull_bones gate
        # (they don't need an armature; otherwise Ka_50_AA etc. stay broken when bones are off).
        try:
            _nrot = _fix_flyaway_rotors(collection)
            if _nrot:
                _warno_log(settings, f"rotor fly-away fix: {_nrot} rotor(s) re-centered", stage="import")
        except Exception as _exc:
            _warno_log(settings, f"rotor fly-away fix failed: {_exc}", level="WARNING", stage="import")
        # Coaxial rotors (Ka-50/52/27): stack the two discs on one mast (same X,Y).
        try:
            _ncox = _align_coaxial_rotors(collection)
            if _ncox:
                _warno_log(settings, f"coaxial rotor align: {_ncox} disc(s) stacked on mast", stage="import")
        except Exception as _exc:
            _warno_log(settings, f"coaxial rotor align failed: {_exc}", level="WARNING", stage="import")

        # Finalize WARNO-reserved material names (e.g. 'Vitre'): rename the
        # namespaced material we just built ('<asset>__Vitre') back to the
        # canonical 'Vitre' that the in-game cooker accepts. Any existing
        # canonical material left over from a previous import is parked
        # under '<old_asset>__Vitre' so its references on previously-imported
        # objects continue to work without picking up our textures.
        try:
            renamed_reserved = _warno_finalize_reserved_material_names(asset)
            if renamed_reserved > 0:
                _warno_log(
                    settings,
                    f"reserved material rename: {renamed_reserved} canonical names finalized",
                    stage="import",
                )
        except Exception as exc:
            _warno_log(
                settings,
                f"reserved material rename failed: {exc}",
                level="WARNING",
                stage="import",
            )

        # Both games: de-namespace ALL materials of this asset to clean bare names
        # ('Chenille_Droite', 'Chassis'). Runs after the reserved-name finalize (Vitre already
        # canonical) and before the track finalize. The '<unit>__' prefix was only a Blender
        # collision guard; the cooker wants the bare game name.
        try:
            renamed_rd = _warno_finalize_rd_material_names(asset)
            if renamed_rd > 0:
                _warno_log(
                    settings,
                    f"material rename: {renamed_rd} de-namespaced to bare names",
                    stage="import",
                )
        except Exception as exc:
            _warno_log(
                settings,
                f"RD material rename failed: {exc}",
                level="WARNING",
                stage="import",
            )

        # Tracks: strip the namespace and rename the material to its object name
        # (Chenille_Gauche_01 / Chenille_Droite_01) now that the build is done.
        try:
            renamed_tracks = _warno_finalize_track_material_names(asset)
            if renamed_tracks > 0:
                _warno_log(
                    settings,
                    f"track material rename: {renamed_tracks} track material(s) renamed to object names",
                    stage="import",
                )
        except Exception as exc:
            _warno_log(
                settings,
                f"track material rename failed: {exc}",
                level="WARNING",
                stage="import",
            )

        # Glass: RD ships the canopy material ('Vitres') opaque; make glass see-through and
        # normalize its name to the cooker-reserved 'Vitre'.
        try:
            n_glass = _finalize_glass_materials()
            if n_glass > 0:
                _warno_log(settings, f"glass: {n_glass} material(s) made transparent + named Vitre", stage="import")
        except Exception as exc:
            _warno_log(settings, f"glass finalize failed: {exc}", level="WARNING", stage="import")

        # Chains: name the ball-chain curtain object 'Chains' (not the generic 'Chassis_2' peel name).
        try:
            n_chain = _name_chain_objects()
            if n_chain > 0:
                _warno_log(settings, f"chains: {n_chain} chain object(s) named 'Chains'", stage="import")
        except Exception:
            pass

        # Diffuse repair: re-wire a diffuse texture left unlinked from Base Color (keyword-named
        # units like Alpha_Jet whose diffuse got mis-routed to the alpha input).
        try:
            n_diff = _fix_unwired_diffuse()
            if n_diff > 0:
                _warno_log(settings, f"diffuse repair: {n_diff} material(s) re-wired to Base Color", stage="import")
        except Exception as _exc:
            _warno_log(settings, f"diffuse repair failed: {_exc}", level="WARNING", stage="import")

        # Universal node hygiene: diffuse -> sRGB, other maps -> Non-Color, and the diffuse COLOR
        # must never drive BSDF Alpha (alpha is always a separate map). WARNO + RD.
        try:
            n_norm = _normalize_diffuse_colorspace_and_alpha()
            if n_norm > 0:
                _warno_log(settings, f"colorspace/alpha hygiene: {n_norm} fix(es) (diffuse sRGB, maps Non-Color, alpha unlinked)", stage="import")
        except Exception as _exc:
            _warno_log(settings, f"colorspace/alpha hygiene failed: {_exc}", level="WARNING", stage="import")

        new_scene_objects = [
            obj for obj in bpy.data.objects
            if int(obj.as_pointer()) not in preexisting_object_ptrs
        ]
        settings.last_import_collection = _apply_dev_collection_layout(context.scene, new_scene_objects)
        _log_import_progress(
            settings,
            step_idx=7,
            total_steps=progress_total_steps,
            label="scene objects created",
            t0=t0,
            extra=f"objects={mesh_count}",
        )

        for obj in context.scene.objects:
            obj.select_set(False)
        for obj in imported_objects:
            obj.select_set(True)
        if imported_objects:
            context.view_layer.objects.active = imported_objects[0]
        if hierarchy_root_obj is not None:
            hierarchy_root_obj.select_set(True)

        tex_refs = len(texture_report.get("refs", []))
        tex_resolved = len(texture_report.get("resolved", []))
        tex_errors = len(texture_report.get("errors", []))
        tex_channels = texture_report.get("channels", [])
        atlas_source = str(texture_report.get("atlas_source", "")).strip()
        converter_source = str(texture_report.get("converter_source", "")).strip()
        deps_auto_installed = bool(texture_report.get("deps_auto_installed", False))
        if settings.auto_textures:
            if tex_refs == 0:
                self.report({"WARNING"}, "No texture refs found in SPK for this asset.")
            elif tex_resolved == 0 and tex_refs > 0:
                self.report({"WARNING"}, f"Textures unresolved ({tex_errors} errors). Check Atlas/TGV paths.")
            elif tex_errors > 0:
                self.report({"WARNING"}, f"Textures partial: {tex_resolved} resolved, {tex_errors} errors.")
            elif tex_channels:
                self.report({"INFO"}, f"Texture channels: {', '.join(tex_channels)}")
            if tex_errors:
                first = texture_report.get("errors", [{}])[0]
                first_msg = str(first.get("error", "")).strip()
                first_ref = str(first.get("atlas_ref", "")).strip()
                if first_msg:
                    short_msg = first_msg[:260]
                    if first_ref:
                        self.report({"WARNING"}, f"{Path(first_ref).name}: {short_msg}")
                    else:
                        self.report({"WARNING"}, short_msg)

        msg = f"Imported: {asset} | objects={mesh_count}"
        if variant_note:
            msg += f" | {variant_note}"
        if hierarchy_root_obj is not None:
            msg += " + hierarchy"
        if settings.auto_textures:
            msg += f" | tex:{tex_resolved}/{tex_refs}"
            if tex_errors:
                msg += f" err:{tex_errors}"
                first = texture_report.get("errors", [{}])[0]
                first_msg = str(first.get("error", "")).strip()
                if first_msg:
                    msg += f" | {first_msg[:120]}"
            if atlas_source:
                msg += f" | atlas:{atlas_source}"
            if converter_source:
                msg += f" | converter:{converter_source}"
            if deps_auto_installed:
                msg += " | deps:auto-installed"
        msg += f" | elapsed:{time.monotonic()-t0:.1f}s"
        if texture_dir_to_open:
            settings.last_texture_dir = texture_dir_to_open
        settings.status = msg
        _warno_log(settings, msg, stage="import")
        self.report({"INFO"}, msg)
        _log_import_progress(
            settings,
            step_idx=8,
            total_steps=progress_total_steps,
            label="import complete",
            t0=t0,
            extra=f"objects={mesh_count}",
        )
        _toggle_import_console(settings, open_console=False, active=console_toggled)
        return {"FINISHED"}


class WARNO_PT_ImporterPanel(Panel):
    bl_label = "WARNO Import"
    bl_idname = "WARNO_PT_import_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WARNO"

    def draw(self, context):
        layout = self.layout
        s = context.scene.warno_import

        game_box = layout.box()
        game_box.prop(s, "game_preset", text="Game")

        root_box = layout.box()
        root_hdr = root_box.row(align=True)
        root_icon = "TRIA_DOWN" if bool(s.show_project_section) else "TRIA_RIGHT"
        root_hdr.prop(s, "show_project_section", text="Project", icon=root_icon, emboss=False)
        if s.show_project_section:
            root_box.prop(s, "project_root")
            row = root_box.row(align=True)
            row.operator("warno.load_config", text="Load My config")
            row.operator("warno.load_example_config", text="Load example config")
            row.operator("warno.save_config", text="Save Config")
            root_box.operator("warno.remember_project_root", text="Remember Project Root", icon="BOOKMARKS")

        setup = layout.box()
        header = setup.row(align=True)
        setup_icon = "TRIA_DOWN" if bool(s.show_first_setup_logs) else "TRIA_RIGHT"
        header.prop(s, "show_first_setup_logs", text="First Setup / Logs", icon=setup_icon, emboss=False)
        if s.show_first_setup_logs:
            src = setup.column(align=True)
            _game_folder_label = {
                "WARNO": "WARNO Folder",
                "WARGAME_RD": "Wargame RD Folder",
                "STEEL_DIVISION_2": "Steel Division 2 Folder",
            }.get(_normalize_game_id(getattr(s, "game_preset", "WARNO")), "Game Folder")
            src.prop(s, "warno_root", text=_game_folder_label)
            src.prop(s, "modding_suite_root")
            src.prop(s, "zz_runtime_dir")
            src.operator("warno.prepare_zz_runtime", text="Prepare ZZ Runtime", icon="FILE_REFRESH")
            src.prop(s, "cache_dir")
            src.separator()
            src.prop(s, "auto_install_tgv_deps")
            src.prop(s, "tgv_deps_dir")
            src.operator("warno.install_tgv_deps", text="Install/Check TGV deps", icon="CONSOLE")
            row = src.row(align=True)
            row.operator("warno.open_system_console", text="Open System Console", icon="CONSOLE")
            row.operator("warno.open_log_file", text="Open Log File", icon="TEXT")
            src.prop(s, "log_to_file")
            src.prop(s, "log_file_name")

        tex = layout.box()
        tex_hdr = tex.row(align=True)
        tex_icon = "TRIA_DOWN" if bool(s.show_textures_section) else "TRIA_RIGHT"
        tex_hdr.prop(s, "show_textures_section", text="Textures", icon=tex_icon, emboss=False)
        if s.show_textures_section:
            tex.prop(s, "atlas_assets_dir")
            row = tex.row(align=True)
            row.operator("warno.open_texture_folder", text="Open Texture Folder", icon="FILE_FOLDER")
            row.operator("warno.clear_texture_folder", text="Clear texture folder from old files", icon="TRASH")
            tex.operator("warno.apply_textures", text="Apply/Reapply Textures", icon="SHADING_TEXTURE")

        qry = layout.box()
        qry_hdr = qry.row(align=True)
        qry_icon = "TRIA_DOWN" if bool(s.show_asset_picker_section) else "TRIA_RIGHT"
        qry_hdr.prop(s, "show_asset_picker_section", text="Asset Picker", icon=qry_icon, emboss=False)
        if s.show_asset_picker_section:
            qry.prop(s, "query")
            row = qry.row(align=True)
            row.operator("warno.scan_assets", text="Scan Assets")
            row.operator("warno.scan_assets_all", text="Scan ALL Assets (uses cache)", icon="TIME")
            qry.operator(
                "warno.pick_asset_browser_tree",
                text="Browse Assets",
                icon="OUTLINER",
            )
            qry.label(text=f"Current: {str(s.selected_asset or '').strip()}", icon="OBJECT_DATA")
            lod_row = qry.row(align=True)
            lod_icon = "TRIA_DOWN" if bool(s.show_asset_lods) else "TRIA_RIGHT"
            lod_row.prop(s, "show_asset_lods", text="LOD", icon=lod_icon, emboss=False)
            if s.show_asset_lods:
                qry.label(text=f"Main: {str(s.selected_asset_group or '__none__').strip()}", icon="MESH_CUBE")
                picked_lod = str(s.selected_asset_lod or "__base__").strip()
                if picked_lod == "__base__":
                    qry.label(text="LOD: <base model>", icon="MOD_DECIM")
                else:
                    qry.label(text=f"LOD: {picked_lod}", icon="MOD_DECIM")
                qry.label(text="Use Browse Assets to choose exact LOD variant.", icon="INFO")

        opts = layout.box()
        hdr = opts.row(align=True)
        opt_icon = "TRIA_DOWN" if bool(s.show_import_options) else "TRIA_RIGHT"
        hdr.prop(s, "show_import_options", text="Import Options", icon=opt_icon, emboss=False)
        if s.show_import_options:
            opt_grid = opts.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=False, align=True)
            opt_grid.prop(s, "auto_split_main_parts")
            opt_grid.prop(s, "auto_name_parts")
            opt_grid.prop(s, "auto_pull_bones")
            opt_grid.prop(s, "auto_name_materials")
            opt_grid.prop(s, "use_merge_by_distance")
            if bool(s.use_merge_by_distance):
                opts.prop(s, "merge_distance")
            auto_smooth_row = opts.row(align=True)
            auto_smooth_row.prop(s, "auto_smooth_mode", expand=True)

        import_row = layout.row(align=True)
        import_row.operator("warno.import_asset", text="Import To Blender", icon="IMPORT")
        import_row.operator("warno.manual_auto_smooth_apply", text="Manual Auto Smooth Apply", icon="MOD_SMOOTH")
        layout.label(text="Auto Smooth mode: modifier | off | apply.", icon="INFO")
        get_row = layout.row(align=True)
        get_row.operator("warno.get_uv_from_game", text="Get UV from game (Experimental)", icon="RESTRICT_RENDER_OFF")
        layout.label(text="Armory: show unit, press INSERT to capture, then re-import.", icon="INFO")
        if s.status:
            layout.label(text=s.status)


CLASSES = [
    # WARNOBrowserNode must be registered BEFORE WARNOImporterSettings
    # because the latter declares a CollectionProperty(type=WARNOBrowserNode).
    WARNOBrowserNode,
    WARNOImporterSettings,
    WARNO_OT_LoadConfig,
    WARNO_OT_LoadExampleConfig,
    WARNO_OT_SaveConfig,
    WARNO_OT_RememberProjectRoot,
    WARNO_OT_BuildAssetIndex,
    WARNO_OT_ScanAssets,
    WARNO_OT_ScanAssetsAll,
    WARNO_OT_PickAssetBrowser,
    WARNO_OT_PickAssetPopup,
    # New tree-view browser
    WARNO_UL_AssetBrowser,
    WARNO_OT_BrowserToggleFolder,
    WARNO_OT_BrowserClearSearch,
    WARNO_OT_BrowserExpandAll,
    WARNO_OT_BrowserPickModel,
    WARNO_OT_PickAssetBrowserTree,
    WARNO_OT_PrepareZZRuntime,
    WARNO_OT_InstallTGVDeps,
    WARNO_OT_OpenSystemConsole,
    WARNO_OT_OpenLogFile,
    WARNO_OT_OpenTextureFolder,
    WARNO_OT_ClearTextureFolder,
    WARNO_OT_ApplyTextures,
    WARNO_OT_ManualAutoSmoothApply,
    WARNO_OT_GetUVFromGame,
    WARNO_OT_ImportAsset,
    WARNO_PT_ImporterPanel,
]


@persistent
def _warno_load_post(_dummy):
    for scene in _iter_scenes_safe():
        settings = getattr(scene, "warno_import", None)
        if settings is None:
            continue
        try:
            _restore_project_root_and_auto_config(settings)
        except Exception:
            continue


def register():
    ASSET_PICKER_VIEW_CACHE.clear()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.warno_import = PointerProperty(type=WARNOImporterSettings)
    if _warno_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_warno_load_post)
    for scene in _iter_scenes_safe():
        settings = getattr(scene, "warno_import", None)
        if settings is None:
            continue
        try:
            _restore_project_root_and_auto_config(settings)
        except Exception:
            continue


def unregister():
    if hasattr(bpy.types.Scene, "warno_import"):
        del bpy.types.Scene.warno_import
    if _warno_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_warno_load_post)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    ASSET_PICKER_VIEW_CACHE.clear()
    ASSET_INDEX_SESSION_CACHE.clear()


if __name__ == "__main__":
    register()
