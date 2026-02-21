bl_info = {
    "name": "WARNO Importer",
    "author": "Kilivanchik",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > WARNO",
    "description": "Direct WARNO SPK importer with textures and optional helper bones",
    "category": "Import-Export",
}

import importlib.util
import json
import math
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import bmesh
import bpy
from bpy.props import BoolProperty
from bpy.props import EnumProperty
from bpy.props import FloatProperty
from bpy.props import IntProperty
from bpy.props import PointerProperty
from bpy.props import StringProperty
from bpy.types import Operator
from bpy.types import Panel
from bpy.types import PropertyGroup
from mathutils import Matrix
from mathutils import Vector


MODULE_CACHE: dict[tuple[str, str], tuple[int, Any]] = {}
SAFE_NAME_RX = re.compile(r"[^A-Za-z0-9_.-]+")


def _guess_default_project_root() -> str:
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


def _resolve_path(project_root: Path, raw: str) -> Path:
    txt = str(raw or "").strip().strip('"')
    if not txt:
        return Path()
    p = Path(bpy.path.abspath(txt))
    if not p.is_absolute():
        p = project_root / p
    return p


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

    spec = importlib.util.spec_from_file_location(f"warno_{module_key}_{abs(hash(key[1]))}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODULE_CACHE[key] = (mtime, module)
    return module


def _extractor_module(settings: "WARNOImporterSettings"):
    root = _project_root(settings)
    return _load_local_module("extractor", root / "warno_spk_extract.py")


def _config_path(settings: "WARNOImporterSettings") -> Path:
    root = _project_root(settings)
    return root / "config.json"


def _wheel_preset_path(settings: "WARNOImporterSettings") -> Path:
    root = _project_root(settings)
    return root / "track_wheel_presets.json"


def _cache_asset_dir(extractor_mod, settings: "WARNOImporterSettings", asset: str) -> Path:
    project_root = _project_root(settings)
    rel = extractor_mod.safe_output_relpath(asset)
    cache_raw = str(settings.cache_dir or "").strip()
    if cache_raw:
        cache_base = _resolve_path(project_root, cache_raw)
    else:
        cache_base = project_root / "output_blender"
    return cache_base / rel.parent


def _asset_enum_items(self: "WARNOImporterSettings", _context):
    raw = str(self.match_cache_json or "").strip()
    if not raw:
        return [("__none__", "<scan assets first>", "No scanned assets", 0)]
    try:
        assets = json.loads(raw)
    except Exception:
        assets = []
    if not isinstance(assets, list) or not assets:
        return [("__none__", "<no matches>", "No matches", 0)]

    out = []
    for i, asset in enumerate(assets):
        label = str(asset)
        if len(label) > 120:
            label = "..." + label[-117:]
        out.append((str(asset), label, str(asset), i))
    return out


def _wheel_preset_enum_items(self: "WARNOImporterSettings", _context):
    raw = str(self.track_preset_cache_json or "").strip()
    if not raw:
        return [("__none__", "<no presets>", "No saved presets", 0)]
    try:
        names = json.loads(raw)
    except Exception:
        names = []
    if not isinstance(names, list) or not names:
        return [("__none__", "<no presets>", "No saved presets", 0)]
    out = []
    for i, name in enumerate(names):
        txt = str(name).strip()
        if not txt:
            continue
        out.append((txt, txt, f"Track/wheel preset: {txt}", i))
    if not out:
        return [("__none__", "<no presets>", "No saved presets", 0)]
    return out


def _wheel_default_tuning() -> Dict[str, float]:
    return {
        "pair_dist_scale": 0.42,
        "pair_edge_scale": 1.08,
        "pair_target_ratio": 0.97,
        "pair_min_pool_ratio": 0.60,
        "pair_axial_scale": 1.15,
        "pair_ring_min": 0.08,
        "pair_ring_max": 1.08,
    }


class WARNOImporterSettings(PropertyGroup):
    project_root: StringProperty(
        name="Project Root",
        description="Folder that contains warno_spk_extract.py and config.json",
        subtype="DIR_PATH",
        default=DEFAULT_PROJECT_ROOT,
    )
    spk_path: StringProperty(name="Mesh SPK", subtype="FILE_PATH", default="spk/Mesh_All.spk")
    skeleton_spk: StringProperty(name="Skeleton SPK", subtype="FILE_PATH", default="skeletonsspk/Skeleton_All.spk")
    unit_ndfbin: StringProperty(
        name="Unit NDF",
        subtype="FILE_PATH",
        default="",
        description="Unit.ndfbin, UniteDescriptor.ndf, or WARNO ModData base folder",
    )
    atlas_assets_dir: StringProperty(name="Atlas Assets", subtype="DIR_PATH", default="")
    tgv_converter: StringProperty(name="TGV Converter", subtype="FILE_PATH", default="tgv_to_png.py")
    texture_subdir: StringProperty(name="Texture Subdir", default="textures")
    cache_dir: StringProperty(name="Cache Dir", subtype="DIR_PATH", default="output_blender")

    query: StringProperty(name="Query", default="Leopard_1A1")
    match_limit: IntProperty(name="Match Limit", default=200, min=1, max=2000)
    match_cache_json: StringProperty(default="[]", options={"HIDDEN"})
    selected_asset: EnumProperty(name="Asset", description="Asset path to import", items=_asset_enum_items)

    auto_textures: BoolProperty(name="Auto textures", default=True)
    auto_split_main_parts: BoolProperty(
        name="Auto split main parts",
        default=True,
        description="Split chassis/track/turret/weapon (without road wheels)",
    )
    auto_split_wheels: BoolProperty(
        name="Auto split wheels",
        default=True,
        description="Split road wheels into Roue_* objects",
    )
    auto_track_wheel_correction: BoolProperty(
        name="Auto track/wheel correction",
        default=True,
        description="Auto-fix glued wheel faces using mirrored wheel heuristics",
    )
    track_fix_distance_scale: FloatProperty(
        name="Distance scale",
        default=0.42,
        min=0.15,
        max=1.25,
        precision=3,
        description="How far mirrored candidates can be from target wheel",
    )
    track_fix_edge_scale: FloatProperty(
        name="Edge scale",
        default=1.08,
        min=1.0,
        max=1.8,
        precision=3,
        description="Max allowed triangle edge size for correction candidates",
    )
    track_fix_target_ratio: FloatProperty(
        name="Target fill",
        default=0.97,
        min=0.75,
        max=1.0,
        precision=3,
        description="Desired wheel face count ratio versus mirrored side",
    )
    track_fix_min_pool_ratio: FloatProperty(
        name="Min pool ratio",
        default=0.60,
        min=0.10,
        max=1.0,
        precision=3,
        description="Minimum fraction of candidate faces required to apply correction",
    )
    track_fix_axial_scale: FloatProperty(
        name="Axial limit",
        default=1.15,
        min=0.80,
        max=2.0,
        precision=3,
        description="Thickness tolerance for moved faces around wheel axis",
    )
    track_fix_ring_min: FloatProperty(
        name="Ring min",
        default=0.08,
        min=0.0,
        max=0.6,
        precision=3,
        description="Minimum radial ratio (blocks hull-center wedges)",
    )
    track_fix_ring_max: FloatProperty(
        name="Ring max",
        default=1.08,
        min=0.8,
        max=1.5,
        precision=3,
        description="Maximum radial ratio for moved faces",
    )
    track_preset_name: StringProperty(
        name="Preset Name",
        default="LeopardFix",
        description="Name for saving current track/wheel correction preset",
    )
    track_preset_cache_json: StringProperty(default="[]", options={"HIDDEN"})
    selected_track_preset: EnumProperty(
        name="Preset",
        description="Saved track/wheel correction preset",
        items=_wheel_preset_enum_items,
    )
    auto_name_parts: BoolProperty(name="Auto part naming", default=True)
    auto_name_materials: BoolProperty(name="Auto material naming", default=True)
    auto_pull_bones: BoolProperty(name="Auto pull bones", default=True)

    tgv_split_mode: EnumProperty(
        name="TGV split",
        items=(
            ("auto", "Auto", "Auto split channels"),
            ("all", "All", "Save all channels"),
            ("none", "None", "No channel split"),
        ),
        default="auto",
    )
    tgv_mirror: BoolProperty(name="Mirror TGV", default=False)
    tgv_aggressive_split: BoolProperty(
        name="Aggressive split",
        default=False,
        description="Use stronger packed-part split detection (can over-split track strips)",
    )
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

    rotate_x: FloatProperty(name="Rotate X", default=0.0)
    rotate_y: FloatProperty(name="Rotate Y", default=0.0)
    rotate_z: FloatProperty(name="Rotate Z", default=0.0)
    mirror_y: BoolProperty(name="Mirror Y", default=False)

    use_merge_by_distance: BoolProperty(name="Merge by distance", default=False)
    merge_distance: FloatProperty(name="Merge distance", default=0.0001, min=0.0, precision=6)
    auto_smooth_angle: FloatProperty(name="Smooth angle", default=30.0, min=0.0, max=180.0)
    last_texture_dir: StringProperty(name="Last texture dir", subtype="DIR_PATH", default="", options={"HIDDEN"})
    last_import_collection: StringProperty(name="Last import collection", default="", options={"HIDDEN"})

    status: StringProperty(name="Status", default="")


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
    settings.unit_ndfbin = get_text("unit_ndfbin", settings.unit_ndfbin)
    settings.atlas_assets_dir = get_text("atlas_assets_dir", settings.atlas_assets_dir)
    settings.tgv_converter = get_text("tgv_converter", settings.tgv_converter)
    settings.texture_subdir = get_text("texture_subdir", settings.texture_subdir) or "textures"
    settings.cache_dir = get_text("cache_dir", settings.cache_dir) or settings.cache_dir

    settings.auto_textures = get_bool("auto_textures", settings.auto_textures)
    settings.auto_split_main_parts = get_bool("auto_split_main_parts", get_bool("split_bone_parts", settings.auto_split_main_parts))
    settings.auto_split_wheels = get_bool("auto_split_wheels", settings.auto_split_wheels)
    settings.auto_track_wheel_correction = get_bool("auto_track_wheel_correction", settings.auto_track_wheel_correction)
    settings.track_fix_distance_scale = get_float("track_fix_distance_scale", settings.track_fix_distance_scale)
    settings.track_fix_edge_scale = get_float("track_fix_edge_scale", settings.track_fix_edge_scale)
    settings.track_fix_target_ratio = get_float("track_fix_target_ratio", settings.track_fix_target_ratio)
    settings.track_fix_min_pool_ratio = get_float("track_fix_min_pool_ratio", settings.track_fix_min_pool_ratio)
    settings.track_fix_axial_scale = get_float("track_fix_axial_scale", settings.track_fix_axial_scale)
    settings.track_fix_ring_min = get_float("track_fix_ring_min", settings.track_fix_ring_min)
    settings.track_fix_ring_max = get_float("track_fix_ring_max", settings.track_fix_ring_max)
    settings.track_preset_name = get_text("track_preset_name", settings.track_preset_name) or settings.track_preset_name
    selected_preset = get_text("track_selected_preset", settings.selected_track_preset)
    if selected_preset:
        try:
            settings.selected_track_preset = selected_preset
        except Exception:
            pass
    settings.auto_name_parts = get_bool("auto_name_parts", settings.auto_name_parts)
    settings.auto_name_materials = get_bool("auto_name_materials", settings.auto_name_materials)
    settings.auto_pull_bones = get_bool("auto_pull_bones", settings.auto_pull_bones)
    settings.use_merge_by_distance = get_bool("fbx_use_merge_by_distance", settings.use_merge_by_distance)
    settings.merge_distance = get_float("fbx_merge_distance", settings.merge_distance)
    settings.auto_smooth_angle = get_float("fbx_auto_smooth_angle", settings.auto_smooth_angle)

    settings.tgv_split_mode = get_text("tgv_split_mode", settings.tgv_split_mode) or settings.tgv_split_mode
    if settings.tgv_split_mode not in {"auto", "all", "none"}:
        settings.tgv_split_mode = "auto"
    settings.tgv_mirror = get_bool("tgv_mirror", settings.tgv_mirror)
    settings.tgv_aggressive_split = get_bool("tgv_aggressive_split", settings.tgv_aggressive_split)
    settings.auto_rename_textures = get_bool("auto_rename_textures", settings.auto_rename_textures)
    settings.use_ao_multiply = get_bool("ao_multiply_diffuse", settings.use_ao_multiply)

    settings.rotate_x = get_float("rotate_x", settings.rotate_x)
    settings.rotate_y = get_float("rotate_y", settings.rotate_y)
    settings.rotate_z = get_float("rotate_z", settings.rotate_z)
    settings.mirror_y = get_bool("mirror_y", settings.mirror_y)
    return True, f"Loaded: {path}"


def _save_settings_to_config(settings: WARNOImporterSettings, path: Path) -> tuple[bool, str]:
    raw: dict[str, Any] = {}
    if path.exists() and path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    raw["spk_path"] = settings.spk_path
    raw["skeleton_spk"] = settings.skeleton_spk
    raw["unit_ndfbin"] = settings.unit_ndfbin
    raw["atlas_assets_dir"] = settings.atlas_assets_dir
    raw["tgv_converter"] = settings.tgv_converter
    raw["texture_subdir"] = settings.texture_subdir
    raw["cache_dir"] = settings.cache_dir

    raw["auto_textures"] = bool(settings.auto_textures)
    raw["auto_split_main_parts"] = bool(settings.auto_split_main_parts)
    raw["auto_split_wheels"] = bool(settings.auto_split_wheels)
    raw["auto_track_wheel_correction"] = bool(settings.auto_track_wheel_correction)
    raw["track_fix_distance_scale"] = float(settings.track_fix_distance_scale)
    raw["track_fix_edge_scale"] = float(settings.track_fix_edge_scale)
    raw["track_fix_target_ratio"] = float(settings.track_fix_target_ratio)
    raw["track_fix_min_pool_ratio"] = float(settings.track_fix_min_pool_ratio)
    raw["track_fix_axial_scale"] = float(settings.track_fix_axial_scale)
    raw["track_fix_ring_min"] = float(settings.track_fix_ring_min)
    raw["track_fix_ring_max"] = float(settings.track_fix_ring_max)
    raw["track_preset_name"] = str(settings.track_preset_name or "").strip()
    raw["track_selected_preset"] = str(settings.selected_track_preset or "").strip()
    raw["auto_name_parts"] = bool(settings.auto_name_parts)
    raw["auto_name_materials"] = bool(settings.auto_name_materials)
    raw["auto_pull_bones"] = bool(settings.auto_pull_bones)
    raw["tgv_split_mode"] = str(settings.tgv_split_mode)
    raw["tgv_mirror"] = bool(settings.tgv_mirror)
    raw["tgv_aggressive_split"] = bool(settings.tgv_aggressive_split)
    raw["auto_rename_textures"] = bool(settings.auto_rename_textures)
    raw["ao_multiply_diffuse"] = bool(settings.use_ao_multiply)

    raw["rotate_x"] = float(settings.rotate_x)
    raw["rotate_y"] = float(settings.rotate_y)
    raw["rotate_z"] = float(settings.rotate_z)
    raw["mirror_y"] = bool(settings.mirror_y)
    raw["fbx_use_merge_by_distance"] = bool(settings.use_merge_by_distance)
    raw["fbx_merge_distance"] = float(settings.merge_distance)
    raw["fbx_auto_smooth_angle"] = float(settings.auto_smooth_angle)

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


def _is_wheel_name(name: str) -> bool:
    return _norm_low(name).startswith("roue_")


def _split_groups_for_options(
    groups: Sequence[Tuple[str, List[Tuple[int, int, int]]]],
    fallback_name: str,
    split_main_parts: bool,
    split_wheels: bool,
) -> List[Tuple[str, List[Tuple[int, int, int]]]]:
    merged: Dict[str, List[Tuple[int, int, int]]] = {}

    if not groups:
        return [(fallback_name, [])]
    if not split_main_parts and not split_wheels:
        all_faces: List[Tuple[int, int, int]] = []
        for _, tris in groups:
            all_faces.extend(tris)
        return [("MainBody", all_faces)]

    for raw_name, tris in groups:
        if not tris:
            continue
        name = str(raw_name or fallback_name).strip() or fallback_name
        wheel = _is_wheel_name(name)
        target = name
        if split_main_parts and not split_wheels and wheel:
            target = "Chassis"
        elif split_wheels and not split_main_parts and not wheel:
            target = "MainBody"
        merged.setdefault(target, []).extend(tris)

    if not merged:
        return [(fallback_name, [])]
    return sorted(merged.items(), key=lambda kv: kv[0].lower())


def _wheel_tuning_payload(settings: WARNOImporterSettings) -> Dict[str, Any]:
    return {
        "enabled": bool(settings.auto_track_wheel_correction),
        "pair_dist_scale": float(settings.track_fix_distance_scale),
        "pair_edge_scale": float(settings.track_fix_edge_scale),
        "pair_target_ratio": float(settings.track_fix_target_ratio),
        "pair_min_pool_ratio": float(settings.track_fix_min_pool_ratio),
        "pair_axial_scale": float(settings.track_fix_axial_scale),
        "pair_ring_min": float(settings.track_fix_ring_min),
        "pair_ring_max": float(settings.track_fix_ring_max),
    }


def _apply_wheel_tuning_to_settings(settings: WARNOImporterSettings, tuning: Dict[str, Any]) -> None:
    defaults = _wheel_default_tuning()
    settings.track_fix_distance_scale = float(tuning.get("pair_dist_scale", defaults["pair_dist_scale"]))
    settings.track_fix_edge_scale = float(tuning.get("pair_edge_scale", defaults["pair_edge_scale"]))
    settings.track_fix_target_ratio = float(tuning.get("pair_target_ratio", defaults["pair_target_ratio"]))
    settings.track_fix_min_pool_ratio = float(tuning.get("pair_min_pool_ratio", defaults["pair_min_pool_ratio"]))
    settings.track_fix_axial_scale = float(tuning.get("pair_axial_scale", defaults["pair_axial_scale"]))
    settings.track_fix_ring_min = float(tuning.get("pair_ring_min", defaults["pair_ring_min"]))
    settings.track_fix_ring_max = float(tuning.get("pair_ring_max", defaults["pair_ring_max"]))


def _read_wheel_presets(settings: WARNOImporterSettings) -> Dict[str, Dict[str, float]]:
    path = _wheel_preset_path(settings)
    if not path.exists() or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    presets = raw.get("presets", raw)
    if not isinstance(presets, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for key, val in presets.items():
        name = str(key).strip()
        if not name or not isinstance(val, dict):
            continue
        out[name] = {
            "pair_dist_scale": float(val.get("pair_dist_scale", _wheel_default_tuning()["pair_dist_scale"])),
            "pair_edge_scale": float(val.get("pair_edge_scale", _wheel_default_tuning()["pair_edge_scale"])),
            "pair_target_ratio": float(val.get("pair_target_ratio", _wheel_default_tuning()["pair_target_ratio"])),
            "pair_min_pool_ratio": float(val.get("pair_min_pool_ratio", _wheel_default_tuning()["pair_min_pool_ratio"])),
            "pair_axial_scale": float(val.get("pair_axial_scale", _wheel_default_tuning()["pair_axial_scale"])),
            "pair_ring_min": float(val.get("pair_ring_min", _wheel_default_tuning()["pair_ring_min"])),
            "pair_ring_max": float(val.get("pair_ring_max", _wheel_default_tuning()["pair_ring_max"])),
        }
    return out


def _write_wheel_presets(settings: WARNOImporterSettings, presets: Dict[str, Dict[str, float]]) -> tuple[bool, str]:
    path = _wheel_preset_path(settings)
    payload = {
        "version": 1,
        "presets": presets,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        return False, f"Preset save failed: {exc}"
    return True, f"Presets saved: {path}"


def _refresh_wheel_preset_cache(settings: WARNOImporterSettings) -> None:
    presets = _read_wheel_presets(settings)
    names = sorted(presets.keys(), key=lambda x: x.lower())
    settings.track_preset_cache_json = json.dumps(names, ensure_ascii=False)
    cur = str(settings.selected_track_preset or "").strip()
    if names:
        if cur not in names:
            settings.selected_track_preset = names[0]
    else:
        settings.selected_track_preset = "__none__"


def _build_bone_payload(
    extractor_mod,
    spk,
    model: dict[str, Any],
    asset: str,
    meta: dict[str, Any],
    material_role_by_id: dict[int, str],
    rot: dict[str, float],
    skeleton_spk,
    unit_ndf_hints,
) -> dict[str, Any]:
    mesh_node_index = int(meta.get("nodeIndex", -1))
    mesh_bone_names: List[str] = []
    external_bone_names: List[str] = []
    source_name_lists: Dict[str, List[str]] = {}
    external_sets: List[Tuple[str, List[str]]] = []
    bone_names: List[str] = []
    inferred_wheel_names: List[str] = []
    bone_name_by_index: Dict[int, str] = {}
    bone_name_source = "none"
    ndf_hint_bones: List[str] = []
    ndf_hint_source = "none"
    ndf_hint_error = ""

    if mesh_node_index >= 0:
        try:
            mesh_bone_names = spk.parse_node_names(mesh_node_index)
        except Exception:
            mesh_bone_names = []
    if mesh_bone_names:
        source_name_lists["mesh"] = list(mesh_bone_names)

    skeleton_hit = None
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

        # Primary: same hierarchical node index from mesh dictionary.
        add_external_set("external_same_index", mesh_node_index)

        # Secondary fallback: best path match in external Skeleton SPK FAT.
        skeleton_hit = skeleton_spk.find_best_fat_entry_for_asset(asset)
        if skeleton_hit is not None:
            _, sk_meta = skeleton_hit
            sk_node_idx = int(sk_meta.get("nodeIndex", -1))
            add_external_set("external_asset_match", sk_node_idx)

        for _, names in external_sets:
            external_bone_names.extend([str(n) for n in names if str(n).strip()])
        if external_bone_names:
            external_bone_names = extractor_mod.unique_keep_order(external_bone_names)

    candidates: List[Tuple[str, Dict[int, str]]] = []
    if mesh_bone_names:
        candidates.append(("mesh", extractor_mod.map_bone_index_names(mesh_bone_names)))
    for src_name, names in external_sets:
        if names:
            candidates.append((src_name, extractor_mod.map_bone_index_names(names)))

    best_score = -10_000
    for src, cmap in candidates:
        try:
            score = extractor_mod.score_bone_name_map(model, cmap, material_role_by_id)
        except Exception:
            score = -10_000
        if score > best_score:
            best_score = score
            bone_name_source = src
            bone_name_by_index = dict(cmap)

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
        try:
            inferred = extractor_mod.infer_missing_wheel_bone_names(model, bone_name_by_index, rot)
        except Exception:
            inferred = {}
        if inferred:
            bone_name_by_index.update(inferred)
            inferred_wheel_names = extractor_mod.unique_keep_order([str(v) for v in inferred.values() if str(v).strip()])

    if bone_name_source == "mesh":
        bone_names = list(mesh_bone_names)
    elif bone_name_source in source_name_lists:
        bone_names = list(source_name_lists[bone_name_source])
    else:
        bone_names = extractor_mod.unique_keep_order([*mesh_bone_names, *external_bone_names])
    if inferred_wheel_names:
        bone_names = extractor_mod.unique_keep_order([*bone_names, *inferred_wheel_names])

    if unit_ndf_hints is not None:
        try:
            hint_payload = unit_ndf_hints.hints_for_asset(asset)
        except Exception as exc:
            hint_payload = {"source": "none", "error": str(exc), "bones": []}
        ndf_hint_source = str(hint_payload.get("source", "none")).strip() or "none"
        ndf_hint_error = str(hint_payload.get("error", "")).strip()
        raw_hint_bones = hint_payload.get("bones", [])
        if isinstance(raw_hint_bones, list):
            for raw_hint in raw_hint_bones:
                nm = extractor_mod.normalize_bone_label(str(raw_hint))
                if nm:
                    ndf_hint_bones.append(extractor_mod.sanitize_material_name(nm))
        ndf_hint_bones = extractor_mod.unique_keep_order(ndf_hint_bones)

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
                bone_names = extractor_mod.unique_keep_order([*bone_names, *critical])

    bone_centers_by_index = extractor_mod.estimate_bone_centers_by_index(model, bone_name_by_index, rot)
    raw_names_for_centers = (
        list(source_name_lists.get(bone_name_source, []))
        if source_name_lists.get(bone_name_source, [])
        else bone_names
    )
    bone_positions: Dict[str, List[float]] = {}

    def register(name: str, pos: Tuple[float, float, float]) -> None:
        low = _norm_low(name)
        if not low:
            return
        if low not in bone_positions:
            bone_positions[low] = [float(pos[0]), float(pos[1]), float(pos[2])]
        tok = _norm_token(low)
        if tok and tok not in bone_positions:
            bone_positions[tok] = [float(pos[0]), float(pos[1]), float(pos[2])]

    for bidx, pos in bone_centers_by_index.items():
        mapped_name = bone_name_by_index.get(int(bidx))
        if mapped_name:
            register(mapped_name, pos)
        if 0 <= int(bidx) < len(raw_names_for_centers):
            register(raw_names_for_centers[int(bidx)], pos)

    if ndf_hint_bones and "chassis" in {n.lower() for n in ndf_hint_bones} and "chassis" not in bone_positions:
        for alias in ("chassisfake", "chassisarmaturefake", "base"):
            hit = bone_positions.get(alias)
            if hit:
                bone_positions["chassis"] = list(hit)
                break

    return {
        "bone_name_by_index": bone_name_by_index,
        "bone_names": bone_names,
        "bone_positions": bone_positions,
        "bone_name_source": bone_name_source,
        "mesh_bone_names": mesh_bone_names,
        "external_bone_names": external_bone_names,
        "ndf_hint_bones": ndf_hint_bones,
        "ndf_hint_source": ndf_hint_source,
        "ndf_hint_error": ndf_hint_error,
    }


def _channel_hint_from_stem(stem: str) -> str | None:
    low = _norm_low(stem)
    if not low:
        return None
    if "normal_x" in low or "normal_y" in low or "normal_z" in low:
        return None
    if "normal_reconstructed" in low or low.endswith("_nm") or "normal" in low:
        return "normal"
    if low.endswith("_ao") or "occlusion" in low:
        return "occlusion"
    if low.endswith("_roughness") or low.endswith("_r") or "roughness" in low:
        return "roughness"
    if low.endswith("_metallic") or low.endswith("_m") or "metallic" in low:
        return "metallic"
    if low.endswith("_alpha") or low.endswith("_a") or "alpha" in low:
        return "alpha"
    if low.endswith("_d") or "diffuse" in low or "albedo" in low or "color" in low:
        return "diffuse"
    return None


def _augment_maps_from_existing_files(
    model_dir: Path,
    asset: str,
    chosen_maps: Dict[str, Path],
) -> Dict[str, Path]:
    out = dict(chosen_maps)
    files = sorted(model_dir.rglob("*.png"))
    if not files:
        return out

    wanted = {"diffuse", "normal", "roughness", "metallic", "occlusion", "alpha"}
    missing = [ch for ch in wanted if ch not in out]
    if not missing:
        return out

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


def _resolve_material_maps(
    extractor_mod,
    spk,
    settings: WARNOImporterSettings,
    asset: str,
    model_dir: Path,
    material_ids: Sequence[int],
    material_name_by_id: Dict[int, str],
    material_role_by_id: Dict[int, str],
) -> tuple[Dict[str, Dict[str, Path]], dict[str, Any]]:
    maps_by_name: Dict[str, Dict[str, Path]] = {}
    report: dict[str, Any] = {
        "refs": [],
        "refs_by_material": {},
        "resolved": [],
        "errors": [],
        "named": [],
        "channels": [],
    }
    if not settings.auto_textures:
        return maps_by_name, report

    project_root = _project_root(settings)
    atlas_raw_text = str(settings.atlas_assets_dir or "").strip()
    converter_text = str(settings.tgv_converter or "").strip()
    if not atlas_raw_text:
        raise RuntimeError("Atlas Assets path is empty.")
    if not converter_text:
        raise RuntimeError("TGV converter path is empty.")
    atlas_raw = _resolve_path(project_root, atlas_raw_text)
    converter = _resolve_path(project_root, converter_text)
    if not atlas_raw.exists() or not atlas_raw.is_dir():
        raise RuntimeError(f"Atlas Assets folder not found: {atlas_raw}")
    if not converter.exists() or not converter.is_file():
        raise RuntimeError(f"TGV converter not found: {converter}")

    atlas_root = extractor_mod.resolve_atlas_assets_root(atlas_raw)
    refs_by_material: Dict[int, List[str]] = {}
    try:
        refs_by_material = spk.get_texture_refs_for_material_ids(material_ids)
    except Exception:
        refs_by_material = {}

    refs: List[str] = []
    for mid in material_ids:
        refs.extend(refs_by_material.get(int(mid), []))
    refs = extractor_mod.unique_keep_order(refs) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(refs))
    if not refs:
        refs = spk.find_texture_refs_for_asset(asset, material_ids=material_ids)

    report["refs"] = list(refs)
    report["refs_by_material"] = {
        str(int(mid)): list(refs_by_material.get(int(mid), []))
        for mid in material_ids
    }
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    resolved_by_ref: Dict[str, dict[str, Any]] = {}
    for ref in refs:
        try:
            item = extractor_mod.resolve_texture_from_atlas_ref(
                ref=ref,
                atlas_assets_root=atlas_root,
                out_model_dir=model_dir,
                converter=converter,
                texture_subdir=settings.texture_subdir or "textures",
                tgv_split_mode=settings.tgv_split_mode,
                tgv_mirror=bool(settings.tgv_mirror),
                tgv_aggressive_split=bool(settings.tgv_aggressive_split),
            )
            resolved.append(item)
            resolved_by_ref[str(item.get("atlas_ref", ""))] = item
        except Exception as tex_exc:
            errors.append({"atlas_ref": ref, "error": str(tex_exc)})
    report["errors"] = errors

    chosen_maps = extractor_mod.pick_material_maps_from_textures(resolved)
    chosen_maps = _augment_maps_from_existing_files(model_dir=model_dir, asset=asset, chosen_maps=chosen_maps)
    report["channels"] = sorted(chosen_maps.keys())
    named_maps: Dict[str, Path] = {}
    named_files: list[dict[str, str]] = []
    if chosen_maps:
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
                maps = dict(map_for_materials)
            else:
                for ch, src in map_for_materials.items():
                    maps.setdefault(ch, src)

            if role == "track_left":
                maps.update(track_named_maps.get("left", {}))
            elif role == "track_right":
                maps.update(track_named_maps.get("right", {}))
            elif role.startswith("track"):
                maps.update(track_named_maps.get("generic", {}))
            maps_by_name[mname] = maps

    report["resolved"] = [
        {
            "atlas_ref": item["atlas_ref"],
            "role": item["role"],
            "source_type": item["source_type"],
            "source_tgv": item["source_tgv"],
            "source_png": item["source_png"],
            "out_png": str(item["out_png"]),
            "extras": {k: str(v) for k, v in item.get("extras", {}).items()},
        }
        for item in resolved
    ]
    report["named"] = list(named_files)
    return maps_by_name, report


def _collect_mesh_buckets(
    extractor_mod,
    model: dict[str, Any],
    rot: dict[str, float],
    material_name_by_id: Dict[int, str],
    material_role_by_id: Dict[int, str],
    bone_name_by_index: Dict[int, str],
    split_main_parts: bool,
    split_wheels: bool,
    wheel_tuning: Dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    buckets: Dict[str, dict[str, Any]] = {}
    order: List[str] = []

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

        if split_main_parts or split_wheels:
            if bone_name_by_index:
                groups = extractor_mod.split_faces_by_bone(
                    part,
                    fallback,
                    bone_name_by_index,
                    material_role=role,
                    material_name=mat_name,
                    wheel_tuning=wheel_tuning or None,
                )
            else:
                groups = [(fallback, _all_tris(indices))]
            groups = _split_groups_for_options(
                groups=groups,
                fallback_name=fallback,
                split_main_parts=split_main_parts,
                split_wheels=split_wheels,
            )
        else:
            groups = [("MainBody", _all_tris(indices))]

        for group_name, tris in groups:
            key = _norm_low(group_name) or "mainbody"
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {
                    "group_name": str(group_name),
                    "vertices": [],
                    "uvs": [],
                    "faces": [],
                    "face_mids": [],
                    "map": {},
                }
                buckets[key] = bucket
                order.append(key)

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
        out.append(b)
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


def _apply_material_nodes(
    mat: bpy.types.Material,
    maps: dict[str, Path],
    role: str,
    ao_multiply_diffuse: bool = True,
) -> None:
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
    if diffuse_node is not None and ao_node is not None and ao_multiply_diffuse:
        if ao_node is not None:
            mul = nodes.new("ShaderNodeMixRGB")
            mul.location = (-260, 120)
            mul.blend_type = "MULTIPLY"
            mul.inputs["Fac"].default_value = 1.0
            links.new(diffuse_node.outputs["Color"], mul.inputs["Color1"])
            links.new(ao_node.outputs["Color"], mul.inputs["Color2"])
            links.new(mul.outputs["Color"], bsdf.inputs["Base Color"])
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
            links.new(normal_node.outputs["Color"], nm.inputs["Color"])
            links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    if roughness is not None:
        rough_node = _texture_node(nodes, roughness, -620, -470, non_color=True)
        if rough_node is not None:
            links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])

    if metallic is not None:
        metal_node = _texture_node(nodes, metallic, -620, -620, non_color=True)
        if metal_node is not None:
            links.new(metal_node.outputs["Color"], bsdf.inputs["Metallic"])

def _ensure_collection(scene: bpy.types.Scene, base_name: str) -> bpy.types.Collection:
    name = base_name
    idx = 2
    while bpy.data.collections.get(name) is not None:
        name = f"{base_name}_{idx}"
        idx += 1
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


def _merge_by_distance(objects: Sequence[bpy.types.Object], distance: float) -> None:
    if distance <= 0.0:
        return
    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        if bm.verts:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()


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


def _object_world_center(obj: bpy.types.Object) -> Vector:
    if obj.type != "MESH" or not obj.bound_box:
        return obj.matrix_world.translation.copy()
    acc = Vector((0.0, 0.0, 0.0))
    for c in obj.bound_box:
        acc += Vector((c[0], c[1], c[2]))
    return obj.matrix_world @ (acc / 8.0)


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


def _pick_root_bone_name(bones: Sequence[str]) -> str:
    for name in bones:
        low = _norm_low(name)
        if "chassis" in low or "hull" in low or low.startswith("base"):
            return name
    return bones[0] if bones else "Chassis"


def _match_object_for_bone(raw_name: str, objects: Sequence[bpy.types.Object]) -> bpy.types.Object | None:
    low = _norm_low(raw_name)
    tok = _norm_token(raw_name)
    if not low:
        return None
    by_low = {_norm_low(o.get("warno_group", o.name)): o for o in objects}
    if low in by_low:
        return by_low[low]
    for obj in objects:
        if _norm_token(str(obj.get("warno_group", obj.name))) == tok:
            return obj
    if low.startswith("roue_elev_"):
        alt = low.replace("roue_elev_", "roue_")
        if alt in by_low:
            return by_low[alt]
    if "chassis" in low:
        for obj in objects:
            if "chassis" in _norm_low(str(obj.get("warno_group", obj.name))):
                return obj
    return None


def _pick_position_for_bone(
    raw_name: str,
    bone_positions: dict[str, list[float]],
    objects: Sequence[bpy.types.Object],
    scene_center: Vector,
) -> Vector:
    low = _norm_low(raw_name)
    keys = [low, _norm_token(low)]
    if low.startswith("fx_"):
        keys.append(low[3:])
        keys.append(_norm_token(low[3:]))
    if low.startswith("bone_"):
        keys.append("chassis")
    for key in keys:
        if not key:
            continue
        hit = bone_positions.get(key)
        if isinstance(hit, list) and len(hit) >= 3:
            try:
                return Vector((float(hit[0]), float(hit[1]), float(hit[2])))
            except Exception:
                pass
    src_obj = _match_object_for_bone(raw_name, objects)
    if src_obj is not None:
        return _object_world_center(src_obj)
    return scene_center.copy()


def _pick_parent_name(raw_name: str, root_name: str, present_low: set[str]) -> str | None:
    low = _norm_low(raw_name)
    root_low = _norm_low(root_name)
    if not low or low == root_low:
        return None

    turret = next((n for n in sorted(present_low) if ("tourelle" in n or "turret" in n) and not n.startswith("fx_")), None)
    axe = next((n for n in sorted(present_low) if ("axe_canon" in n or n == "axe" or n.startswith("axe_")) and not n.startswith("fx_")), None)

    if "canon" in low or "gun" in low or "barrel" in low:
        if axe:
            return axe
        if turret:
            return turret
        return root_low
    if "axe_canon" in low or low == "axe" or low.startswith("axe_"):
        if turret:
            return turret
        return root_low
    if "tourelle" in low or "turret" in low:
        return root_low
    if low.startswith("roue_") or low.startswith("chenille_") or low.startswith("fx_"):
        return root_low
    return root_low


def _pick_bone_for_object(obj: bpy.types.Object, bone_map_low_to_actual: dict[str, str], root_low: str) -> str | None:
    names = [str(obj.get("warno_group", "")), obj.name]
    for nm in names:
        low = _norm_low(nm)
        tok = _norm_token(nm)
        if low in bone_map_low_to_actual:
            return low
        for raw_low in bone_map_low_to_actual.keys():
            if _norm_token(raw_low) == tok:
                return raw_low
    group_low = _norm_low(str(obj.get("warno_group", obj.name)))
    if "chassis" in group_low:
        return root_low
    return root_low if root_low in bone_map_low_to_actual else next(iter(bone_map_low_to_actual.keys()), None)


def _set_object_origin_world(obj: bpy.types.Object, world_point: Vector) -> None:
    if obj.type != "MESH":
        return
    mesh = obj.data
    mw = obj.matrix_world.copy()
    local = mw.inverted() @ world_point
    mesh.transform(Matrix.Translation(-local))
    obj.matrix_world.translation = world_point


def _build_helper_armature(
    imported_objects: Sequence[bpy.types.Object],
    bone_payload: dict[str, Any],
    collection: bpy.types.Collection,
) -> bpy.types.Object | None:
    bone_names = [str(x).strip() for x in bone_payload.get("bone_names", []) if str(x).strip()]
    if not bone_names:
        return None

    scene_center, diag = _mesh_bounds(imported_objects)
    bone_positions = bone_payload.get("bone_positions", {}) or {}
    root_name = _pick_root_bone_name(bone_names)
    ordered = [root_name] + [n for n in bone_names if _norm_low(n) != _norm_low(root_name)]

    arm_data = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    collection.objects.link(arm_obj)
    if hasattr(arm_obj, "show_in_front"):
        arm_obj.show_in_front = True
    if hasattr(arm_data, "display_type"):
        arm_data.display_type = "BBONE"
    if hasattr(arm_obj, "display_type"):
        arm_obj.display_type = "WIRE"
    if hasattr(arm_obj, "color"):
        arm_obj.color = (0.55, 0.85, 1.0, 0.35)

    for o in bpy.context.scene.objects:
        o.select_set(False)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    bone_len = max(0.003, diag * 0.00035) if diag > 0.0 else 0.003
    bone_len = min(bone_len, 0.03)
    by_raw_low: dict[str, str] = {}
    used_actual_low: set[str] = set()

    for raw in ordered:
        base = raw[:63] if raw else "Bone"
        if not base:
            base = "Bone"
        actual = base
        suffix = 2
        while _norm_low(actual) in used_actual_low:
            cut = max(1, 63 - len(str(suffix)) - 1)
            actual = f"{base[:cut]}_{suffix}"
            suffix += 1
        used_actual_low.add(_norm_low(actual))

        eb = arm_data.edit_bones.new(actual)
        pos = _pick_position_for_bone(raw, bone_positions, imported_objects, scene_center)
        eb.head = pos
        eb.tail = pos + Vector((0.0, 0.0, bone_len))
        by_raw_low[_norm_low(raw)] = actual

    present = set(by_raw_low.keys())
    for raw in ordered:
        raw_low = _norm_low(raw)
        parent_low = _pick_parent_name(raw, root_name, present)
        if parent_low is None:
            continue
        child_name = by_raw_low.get(raw_low)
        parent_name = by_raw_low.get(_norm_low(parent_low))
        if not child_name or not parent_name:
            continue
        child = arm_data.edit_bones.get(child_name)
        parent = arm_data.edit_bones.get(parent_name)
        if child is not None and parent is not None and child != parent:
            child.parent = parent

    bpy.ops.object.mode_set(mode="OBJECT")

    root_low = _norm_low(root_name)
    for obj in imported_objects:
        target_low = _pick_bone_for_object(obj, by_raw_low, root_low)
        if not target_low:
            continue
        bone_name = by_raw_low.get(target_low)
        if not bone_name:
            continue
        bone = arm_obj.data.bones.get(bone_name)
        if bone is None:
            continue
        head_world = arm_obj.matrix_world @ bone.head_local
        _set_object_origin_world(obj, head_world)
        world_matrix = obj.matrix_world.copy()
        obj.parent = arm_obj
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        obj.matrix_parent_inverse = (arm_obj.matrix_world @ bone.matrix_local).inverted()
        obj.matrix_world = world_matrix

    return arm_obj


class WARNO_OT_LoadConfig(Operator):
    bl_idname = "warno.load_config"
    bl_label = "Load Config"
    bl_description = "Load plugin settings from config.json"

    def execute(self, context):
        settings = context.scene.warno_import
        path = _config_path(settings)
        ok, msg = _load_config_into_settings(settings, path)
        _refresh_wheel_preset_cache(settings)
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
        _refresh_wheel_preset_cache(settings)
        settings.status = msg
        self.report({"INFO" if ok else "WARNING"}, msg)
        return {"FINISHED" if ok else "CANCELLED"}


class WARNO_OT_ResetTrackTuning(Operator):
    bl_idname = "warno.reset_track_tuning"
    bl_label = "Reset Track Params"
    bl_description = "Reset track/wheel correction sliders to built-in defaults"

    def execute(self, context):
        settings = context.scene.warno_import
        _apply_wheel_tuning_to_settings(settings, _wheel_default_tuning())
        settings.status = "Track/wheel correction params reset to defaults."
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class WARNO_OT_SaveTrackPreset(Operator):
    bl_idname = "warno.save_track_preset"
    bl_label = "Save Track Preset"
    bl_description = "Save current track/wheel correction sliders as named preset"

    def execute(self, context):
        settings = context.scene.warno_import
        name = str(settings.track_preset_name or "").strip()
        if not name:
            self.report({"WARNING"}, "Preset name is empty.")
            return {"CANCELLED"}

        presets = _read_wheel_presets(settings)
        presets[name] = {
            "pair_dist_scale": float(settings.track_fix_distance_scale),
            "pair_edge_scale": float(settings.track_fix_edge_scale),
            "pair_target_ratio": float(settings.track_fix_target_ratio),
            "pair_min_pool_ratio": float(settings.track_fix_min_pool_ratio),
            "pair_axial_scale": float(settings.track_fix_axial_scale),
            "pair_ring_min": float(settings.track_fix_ring_min),
            "pair_ring_max": float(settings.track_fix_ring_max),
        }
        ok, msg = _write_wheel_presets(settings, presets)
        if not ok:
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        _refresh_wheel_preset_cache(settings)
        try:
            settings.selected_track_preset = name
        except Exception:
            pass
        settings.status = f"Track preset saved: {name}"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class WARNO_OT_LoadTrackPreset(Operator):
    bl_idname = "warno.load_track_preset"
    bl_label = "Load Track Preset"
    bl_description = "Load selected track/wheel correction preset to sliders"

    def execute(self, context):
        settings = context.scene.warno_import
        _refresh_wheel_preset_cache(settings)
        key = str(settings.selected_track_preset or "").strip()
        if not key or key == "__none__":
            self.report({"WARNING"}, "No preset selected.")
            return {"CANCELLED"}

        presets = _read_wheel_presets(settings)
        tuning = presets.get(key)
        if not tuning:
            self.report({"WARNING"}, f"Preset not found: {key}")
            return {"CANCELLED"}

        _apply_wheel_tuning_to_settings(settings, tuning)
        settings.track_preset_name = key
        settings.status = f"Track preset loaded: {key}"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class WARNO_OT_ScanAssets(Operator):
    bl_idname = "warno.scan_assets"
    bl_label = "Scan Assets"
    bl_description = "Find matching assets in Mesh SPK"

    def execute(self, context):
        settings = context.scene.warno_import
        project_root = _project_root(settings)
        spk_path = _resolve_path(project_root, settings.spk_path)
        if not spk_path.exists() or not spk_path.is_file():
            msg = f"SPK not found: {spk_path}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        try:
            extractor_mod = _extractor_module(settings)
            with extractor_mod.SpkMeshExtractor(spk_path) as spk:
                query = str(settings.query or "").strip()
                matches = spk.find_matches(query if query else None, None)
        except Exception as exc:
            msg = f"Scan failed: {exc}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        assets = [asset for asset, _ in matches[: int(settings.match_limit)]]
        settings.match_cache_json = json.dumps(assets, ensure_ascii=False)
        if assets:
            settings.selected_asset = assets[0]
        msg = f"Matches: {len(matches)} (cached: {len(assets)})"
        settings.status = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class WARNO_OT_OpenTextureFolder(Operator):
    bl_idname = "warno.open_texture_folder"
    bl_label = "Open Texture Folder"
    bl_description = "Open folder where textures were saved on last import"

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
        try:
            bpy.ops.wm.path_open(filepath=str(folder))
        except Exception as exc:
            self.report({"ERROR"}, f"Cannot open folder: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


def _target_model_meshes(context, settings: WARNOImporterSettings) -> List[bpy.types.Object]:
    col_name = str(settings.last_import_collection or "").strip()
    if col_name:
        col = bpy.data.collections.get(col_name)
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
    col = bpy.data.collections.get(name)
    if col is None:
        return ""
    for obj in col.all_objects:
        if obj.type != "MESH":
            continue
        asset = str(obj.get("warno_asset", "")).strip()
        if asset:
            return asset
    return ""


def _remove_collection_with_objects(col_name: str) -> int:
    name = str(col_name or "").strip()
    if not name:
        return 0
    col = bpy.data.collections.get(name)
    if col is None:
        return 0
    all_objs = list(col.all_objects)
    for obj in all_objs:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    try:
        bpy.data.collections.remove(col)
    except Exception:
        pass
    return len(all_objs)


class WARNO_OT_ApplyAutoSmooth(Operator):
    bl_idname = "warno.apply_auto_smooth"
    bl_label = "Apply Auto Smooth To Model"
    bl_description = "Apply Smooth by Angle to imported model and accept modifier"

    def execute(self, context):
        settings = context.scene.warno_import
        targets = _target_model_meshes(context, settings)
        if not targets:
            self.report({"WARNING"}, "No imported mesh objects found.")
            return {"CANCELLED"}

        try:
            count = _apply_auto_smooth_modifier(targets, float(settings.auto_smooth_angle))
        except Exception as exc:
            self.report({"ERROR"}, f"Auto smooth failed: {exc}")
            return {"CANCELLED"}

        if count <= 0:
            self.report({"WARNING"}, "Auto smooth was not applied.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Auto smooth applied to {count} object(s).")
        return {"FINISHED"}


class WARNO_OT_ApplyTrackCorrection(Operator):
    bl_idname = "warno.apply_track_correction"
    bl_label = "Apply Track/Wheel Correction"
    bl_description = "Rebuild last imported WARNO model with current track/wheel tuning, then replace old collection"

    def execute(self, context):
        settings = context.scene.warno_import
        old_col_name = str(settings.last_import_collection or "").strip()
        if not old_col_name:
            self.report({"WARNING"}, "No last import collection found.")
            return {"CANCELLED"}

        asset = _collection_asset_from_name(old_col_name)
        if not asset:
            self.report({"WARNING"}, f"Collection has no WARNO asset: {old_col_name}")
            return {"CANCELLED"}

        settings.selected_asset = asset
        result = bpy.ops.warno.import_asset()
        if "FINISHED" not in result:
            self.report({"ERROR"}, "Track correction import failed.")
            return {"CANCELLED"}

        new_col_name = str(settings.last_import_collection or "").strip()
        if old_col_name and old_col_name != new_col_name:
            removed = _remove_collection_with_objects(old_col_name)
            self.report({"INFO"}, f"Track correction applied. Rebuilt '{new_col_name}', removed {removed} old object(s).")
        else:
            self.report({"INFO"}, "Track correction applied.")
        return {"FINISHED"}


class WARNO_OT_ImportAsset(Operator):
    bl_idname = "warno.import_asset"
    bl_label = "Import To Blender"
    bl_description = "Import selected WARNO asset directly into current scene"

    def execute(self, context):
        settings = context.scene.warno_import
        asset = str(settings.selected_asset or "").strip()
        if not asset or asset == "__none__":
            msg = "Pick asset first (Scan Assets)."
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        try:
            extractor_mod = _extractor_module(settings)
        except Exception as exc:
            msg = f"Backend load failed: {exc}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        project_root = _project_root(settings)
        spk_path = _resolve_path(project_root, settings.spk_path)
        if not spk_path.exists() or not spk_path.is_file():
            msg = f"SPK not found: {spk_path}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        need_bone_map = bool(settings.auto_pull_bones or settings.auto_split_main_parts or settings.auto_split_wheels)
        rot = extractor_mod.build_rotation_params(
            float(settings.rotate_x),
            float(settings.rotate_y),
            float(settings.rotate_z),
            mirror_y=bool(settings.mirror_y),
        )

        material_name_by_id: Dict[int, str] = {}
        material_role_by_id: Dict[int, str] = {}
        material_maps_by_name: Dict[str, Dict[str, Path]] = {}
        texture_report: dict[str, Any] = {"refs": [], "resolved": [], "errors": [], "named": [], "channels": []}
        texture_dir_to_open = ""
        bone_payload: dict[str, Any] = {
            "bone_name_by_index": {},
            "bone_names": [],
            "bone_positions": {},
        }

        try:
            with ExitStack() as stack:
                spk = stack.enter_context(extractor_mod.SpkMeshExtractor(spk_path))
                hit = spk.find_best_fat_entry_for_asset(asset)
                if hit is None:
                    raise RuntimeError(f"Asset not found in SPK: {asset}")
                asset_real, meta = hit
                model = spk.get_model_geometry(asset_real)

                infer_names, material_role_by_id = extractor_mod.infer_material_names(
                    model,
                    mirror_y=bool(settings.mirror_y),
                )
                material_ids = sorted({int(part["material"]) for part in model["parts"]})
                if settings.auto_name_materials:
                    material_name_by_id = dict(infer_names)
                else:
                    material_name_by_id = {int(mid): f"Material_{int(mid):03d}" for mid in material_ids}

                skeleton_spk = None
                if need_bone_map:
                    skeleton_path = _resolve_path(project_root, settings.skeleton_spk)
                    if skeleton_path.exists() and skeleton_path.is_file():
                        skeleton_spk = stack.enter_context(extractor_mod.SpkMeshExtractor(skeleton_path))

                ndf_hints = None
                if need_bone_map:
                    ndf_path = _resolve_path(project_root, settings.unit_ndfbin)
                    if ndf_path.exists():
                        ndf_hints = extractor_mod.UnitNdfHintsResolver(ndf_path)

                if need_bone_map:
                    bone_payload = _build_bone_payload(
                        extractor_mod=extractor_mod,
                        spk=spk,
                        model=model,
                        asset=asset_real,
                        meta=meta,
                        material_role_by_id=material_role_by_id,
                        rot=rot,
                        skeleton_spk=skeleton_spk,
                        unit_ndf_hints=ndf_hints,
                    )

                model_dir = _cache_asset_dir(extractor_mod, settings, asset_real)
                model_dir.mkdir(parents=True, exist_ok=True)
                texture_dir_to_open = str(model_dir.resolve())
                material_maps_by_name, texture_report = _resolve_material_maps(
                    extractor_mod=extractor_mod,
                    spk=spk,
                    settings=settings,
                    asset=asset_real,
                    model_dir=model_dir,
                    material_ids=material_ids,
                    material_name_by_id=material_name_by_id,
                    material_role_by_id=material_role_by_id,
                )

                buckets = _collect_mesh_buckets(
                    extractor_mod=extractor_mod,
                    model=model,
                    rot=rot,
                    material_name_by_id=material_name_by_id,
                    material_role_by_id=material_role_by_id,
                    bone_name_by_index=bone_payload.get("bone_name_by_index", {}),
                    split_main_parts=bool(settings.auto_split_main_parts),
                    split_wheels=bool(settings.auto_split_wheels),
                    wheel_tuning=_wheel_tuning_payload(settings),
                )
        except Exception as exc:
            msg = f"Import prep failed: {exc}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if not buckets:
            msg = "No mesh geometry to import."
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        collection = _ensure_collection(context.scene, f"WARNO_{Path(asset).stem}")
        settings.last_import_collection = collection.name
        imported_objects: List[bpy.types.Object] = []
        material_cache: Dict[str, bpy.types.Material] = {}
        mesh_count = 0

        for bucket_i, bucket in enumerate(buckets, start=1):
            group_name = str(bucket.get("group_name", f"Part_{bucket_i:03d}"))
            if settings.auto_name_parts:
                obj_name = _safe_name(group_name, f"Part_{bucket_i:03d}")
            else:
                obj_name = f"Part_{bucket_i:03d}"

            mesh = bpy.data.meshes.new(obj_name)
            mesh.from_pydata(bucket["vertices"], [], bucket["faces"])
            mesh.update()
            uv_layer = mesh.uv_layers.new(name="UVMap")
            if uv_layer is not None:
                for poly in mesh.polygons:
                    for li, vi in zip(poly.loop_indices, poly.vertices):
                        if 0 <= vi < len(bucket["uvs"]):
                            uv_layer.data[li].uv = bucket["uvs"][vi]
                        else:
                            uv_layer.data[li].uv = (0.0, 0.0)

            mids_order: List[int] = []
            for mid in bucket["face_mids"]:
                if int(mid) not in mids_order:
                    mids_order.append(int(mid))

            mat_slot_by_mid: Dict[int, int] = {}
            for mid in mids_order:
                mat_name = material_name_by_id.get(mid, f"Material_{mid:03d}")
                role = str(material_role_by_id.get(mid, "other"))
                maps = material_maps_by_name.get(mat_name, {})
                cache_key = f"{mat_name.lower()}::{role}"
                mat = material_cache.get(cache_key)
                if mat is None:
                    mat = bpy.data.materials.get(mat_name)
                    if mat is None:
                        mat = bpy.data.materials.new(name=mat_name)
                    _apply_material_nodes(mat, maps, role, bool(settings.use_ao_multiply))
                    material_cache[cache_key] = mat
                mesh.materials.append(mat)
                mat_slot_by_mid[mid] = len(mesh.materials) - 1

            for face_i, poly in enumerate(mesh.polygons):
                if face_i < len(bucket["face_mids"]):
                    mid = int(bucket["face_mids"][face_i])
                    poly.material_index = int(mat_slot_by_mid.get(mid, 0))

            obj = bpy.data.objects.new(obj_name, mesh)
            obj["warno_group"] = group_name
            obj["warno_asset"] = asset
            collection.objects.link(obj)
            imported_objects.append(obj)
            mesh_count += 1

        # Geometry cleanup on import: merge by distance first.
        if settings.use_merge_by_distance:
            _merge_by_distance(imported_objects, float(settings.merge_distance))

        arm_obj = None
        if settings.auto_pull_bones:
            try:
                arm_obj = _build_helper_armature(imported_objects, bone_payload, collection)
                if arm_obj is not None:
                    imported_objects.append(arm_obj)
            except Exception as exc:
                self.report({"WARNING"}, f"Armature build skipped: {exc}")

        for obj in context.scene.objects:
            obj.select_set(False)
        for obj in imported_objects:
            obj.select_set(True)
        if imported_objects:
            context.view_layer.objects.active = imported_objects[0]

        tex_refs = len(texture_report.get("refs", []))
        tex_resolved = len(texture_report.get("resolved", []))
        tex_errors = len(texture_report.get("errors", []))
        tex_channels = texture_report.get("channels", [])
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

        msg = f"Imported: {asset} | objects={mesh_count}" + (" + armature" if arm_obj is not None else "")
        if settings.auto_textures:
            msg += f" | tex:{tex_resolved}/{tex_refs}"
            if tex_errors:
                msg += f" err:{tex_errors}"
                first = texture_report.get("errors", [{}])[0]
                first_msg = str(first.get("error", "")).strip()
                if first_msg:
                    msg += f" | {first_msg[:120]}"
        if texture_dir_to_open:
            settings.last_texture_dir = texture_dir_to_open
        settings.status = msg
        self.report({"INFO"}, msg)
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
        if not str(s.track_preset_cache_json or "").strip():
            _refresh_wheel_preset_cache(s)

        root_box = layout.box()
        root_box.label(text="Project")
        root_box.prop(s, "project_root")
        row = root_box.row(align=True)
        row.operator("warno.load_config", text="Load Config")
        row.operator("warno.save_config", text="Save Config")

        src = layout.box()
        src.label(text="Sources")
        src.prop(s, "spk_path")
        src.prop(s, "skeleton_spk")
        src.prop(s, "unit_ndfbin")
        src.prop(s, "cache_dir")

        tex = layout.box()
        tex.label(text="Textures")
        tex.prop(s, "auto_textures")
        tex.prop(s, "atlas_assets_dir")
        tex.prop(s, "tgv_converter")
        tex.prop(s, "texture_subdir")
        tex.prop(s, "tgv_split_mode")
        tex.prop(s, "tgv_mirror")
        tex.prop(s, "tgv_aggressive_split")
        tex.prop(s, "auto_rename_textures")
        tex.prop(s, "use_ao_multiply")
        tex.operator("warno.open_texture_folder", text="Open Texture Folder", icon="FILE_FOLDER")

        qry = layout.box()
        qry.label(text="Asset Picker")
        qry.prop(s, "query")
        qry.prop(s, "match_limit")
        qry.operator("warno.scan_assets", text="Scan Assets")
        qry.prop(s, "selected_asset")

        opts = layout.box()
        opts.label(text="Import Options")
        opts.prop(s, "auto_split_main_parts")
        opts.prop(s, "auto_split_wheels")
        opts.prop(s, "auto_track_wheel_correction")
        if s.auto_track_wheel_correction:
            corr = opts.column(align=True)
            corr.prop(s, "track_fix_distance_scale")
            corr.prop(s, "track_fix_edge_scale")
            corr.prop(s, "track_fix_target_ratio")
            corr.prop(s, "track_fix_min_pool_ratio")
            corr.prop(s, "track_fix_axial_scale")
            row = corr.row(align=True)
            row.prop(s, "track_fix_ring_min")
            row.prop(s, "track_fix_ring_max")
            corr.separator()
            corr.prop(s, "track_preset_name")
            corr.prop(s, "selected_track_preset")
            row = corr.row(align=True)
            row.operator("warno.reset_track_tuning", text="Reset Defaults", icon="LOOP_BACK")
            row.operator("warno.load_track_preset", text="Load Preset", icon="IMPORT")
            corr.operator("warno.save_track_preset", text="Save Preset", icon="ADD")
            corr.operator("warno.apply_track_correction", text="Apply Correction Now", icon="MODIFIER")
        opts.prop(s, "auto_name_parts")
        opts.prop(s, "auto_name_materials")
        opts.prop(s, "auto_pull_bones")
        opts.prop(s, "rotate_x")
        opts.prop(s, "rotate_y")
        opts.prop(s, "rotate_z")
        opts.prop(s, "mirror_y")

        geo = layout.box()
        geo.label(text="Geometry Cleanup")
        geo.prop(s, "use_merge_by_distance")
        geo.prop(s, "merge_distance")
        geo.prop(s, "auto_smooth_angle")
        geo.operator("warno.apply_auto_smooth", text="Apply Auto smooth to the model", icon="MOD_SMOOTH")

        layout.operator("warno.import_asset", text="Import To Blender", icon="IMPORT")
        if s.status:
            layout.label(text=s.status)


CLASSES = [
    WARNOImporterSettings,
    WARNO_OT_LoadConfig,
    WARNO_OT_SaveConfig,
    WARNO_OT_ResetTrackTuning,
    WARNO_OT_SaveTrackPreset,
    WARNO_OT_LoadTrackPreset,
    WARNO_OT_ScanAssets,
    WARNO_OT_OpenTextureFolder,
    WARNO_OT_ApplyAutoSmooth,
    WARNO_OT_ApplyTrackCorrection,
    WARNO_OT_ImportAsset,
    WARNO_PT_ImporterPanel,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.warno_import = PointerProperty(type=WARNOImporterSettings)


def unregister():
    if hasattr(bpy.types.Scene, "warno_import"):
        del bpy.types.Scene.warno_import
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
