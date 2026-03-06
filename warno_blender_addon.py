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
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence, Tuple

import bmesh
import bpy
from bpy.app.handlers import persistent
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
ZZ_RUNTIME_SESSION_CACHE: dict[tuple[str, str, tuple[tuple[str, int, int], ...]], Dict[str, Any]] = {}
ASSET_INDEX_SESSION_CACHE: dict[str, Dict[str, Any]] = {}
ASSET_PICKER_VIEW_CACHE: dict[str, Dict[str, Any]] = {}
SAFE_NAME_RX = re.compile(r"[^A-Za-z0-9_.-]+")
LOD_SUFFIX_LOCAL_RX = re.compile(r"_(LOW|MID|HIGH|LOD[0-9]+)$", re.IGNORECASE)
CONTROL_CHAR_RX = re.compile(r"[\x00-\x1f\x7f]+")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff"}


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
    prefer_non_lod = _has_lod_suffix(req_stem) or "/lods/" in req_low

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


def _dat_signature_for_cache(extractor_mod, warno_root: Path) -> tuple[tuple[str, int, int], ...]:
    rows: List[tuple[str, int, int]] = []
    try:
        dat_files = extractor_mod.find_warno_texture_dat_files(warno_root)
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
    raw = str(settings.modding_suite_root or "").strip()
    if not raw:
        return None
    root = _resolve_path(project_root, raw)
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

    runtime_root = _zz_runtime_root(settings)
    runtime_root.mkdir(parents=True, exist_ok=True)
    warno_key = str(warno_root.resolve()).lower()
    runtime_key = str(runtime_root.resolve()).lower()
    dat_signature = _dat_signature_for_cache(extractor_mod, warno_root)
    cache_key = (warno_key, runtime_key, dat_signature)
    if force_rebuild:
        for old_key in list(ZZ_RUNTIME_SESSION_CACHE.keys()):
            if old_key[0] == warno_key and old_key[1] == runtime_key:
                ZZ_RUNTIME_SESSION_CACHE.pop(old_key, None)
    else:
        cached = ZZ_RUNTIME_SESSION_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)

    info = extractor_mod.prepare_runtime_sources_from_zz(warno_root=warno_root, runtime_root=runtime_root)
    info["runtime_root"] = str(runtime_root)
    info["source_policy"] = "zz_runtime_only"

    try:
        resolver = extractor_mod.get_zz_runtime_resolver(warno_root)
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
) -> tuple[Path, str] | None:
    target = extractor_mod.normalize_asset_path(str(asset or "")).lower()
    best: tuple[int, int, str, str, Path, str] | None = None
    for spk_path in spk_paths:
        try:
            with extractor_mod.SpkMeshExtractor(spk_path) as spk:
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
        if int(data.get("schema_version", 0) or 0) != 2:
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


def _scan_assets_from_spk_paths(extractor_mod, spk_paths: Sequence[Path], query: str | None) -> List[str]:
    query_val = str(query or "").strip()
    query_for_scan = query_val if query_val else None
    seen_assets: set[str] = set()
    out: List[str] = []
    for spk_path in spk_paths:
        try:
            with extractor_mod.SpkMeshExtractor(spk_path) as spk:
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
        "schema_version": 2,
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

    assets = _scan_assets_from_spk_paths(extractor_mod, mesh_spk_paths, query=None)
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


class WARNOImporterSettings(PropertyGroup):
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
    atlas_assets_dir: StringProperty(name="Atlas Assets", subtype="DIR_PATH", default="")
    tgv_converter: StringProperty(name="TGV Converter", subtype="FILE_PATH", default="tgv_to_png.py")
    modding_suite_atlas_wrapper: StringProperty(
        name="ModdingSuite Atlas Wrapper",
        subtype="FILE_PATH",
        default="modding_suite_atlas_export.py",
        description="Wrapper script that exports Atlas metadata JSON via ModdingSuite",
    )
    modding_suite_atlas_cli: StringProperty(
        name="ModdingSuite Atlas CLI",
        subtype="FILE_PATH",
        default="moddingSuite/atlas_cli/moddingSuite.AtlasCli.exe",
        description="Path to headless Atlas CLI executable (moddingSuite.AtlasCli.exe)",
    )
    use_atlas_json_mapping: BoolProperty(
        name="Use Atlas JSON (ModdingSuite)",
        default=True,
        description="Resolve crop + naming from Atlas JSON exported by ModdingSuite wrapper",
    )
    atlas_json_strict: BoolProperty(
        name="Atlas JSON strict mode",
        default=True,
        description="If enabled, unresolved Atlas JSON mappings fail without fallback guessing",
    )
    atlas_json_cache_subdir: StringProperty(
        name="Atlas JSON cache",
        default="atlas_json_cache",
        description="Subfolder inside cache dir to store atlas JSON maps",
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
        default="moddingSuite-master",
        description="Folder with moddingSuite (used to auto-detect tgv_to_png.py)",
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
        name="Auto pull bones (experimental)",
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
    merge_distance: FloatProperty(name="Merge distance", default=0.0001, min=0.0, precision=6)
    auto_smooth_angle: FloatProperty(name="Smooth angle", default=30.0, min=0.0, max=180.0)
    last_texture_dir: StringProperty(name="Last texture dir", subtype="DIR_PATH", default="", options={"HIDDEN"})
    last_import_collection: StringProperty(name="Last import collection", default="", options={"HIDDEN"})
    startup_state_restored: BoolProperty(default=False, options={"HIDDEN"})

    status: StringProperty(name="Status", default="")


FIXED_MERGE_DISTANCE = 0.0001
FIXED_AUTO_SMOOTH_ANGLE = 30.0


def _enforce_fixed_runtime_defaults(settings: WARNOImporterSettings) -> None:
    settings.use_atlas_json_mapping = True
    settings.atlas_json_strict = True
    settings.auto_textures = True
    settings.auto_rename_textures = True
    settings.fast_exact_texture_resolve = True
    settings.use_zz_dat_source = True
    settings.use_merge_by_distance = True
    settings.use_ao_multiply = False
    settings.normal_invert_mode = "none"
    settings.merge_distance = float(FIXED_MERGE_DISTANCE)
    settings.auto_smooth_angle = float(FIXED_AUTO_SMOOTH_ANGLE)


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
    settings.modding_suite_atlas_wrapper = get_text(
        "modding_suite_atlas_wrapper",
        settings.modding_suite_atlas_wrapper,
    ) or "modding_suite_atlas_export.py"
    settings.modding_suite_atlas_cli = get_text(
        "modding_suite_atlas_cli",
        settings.modding_suite_atlas_cli,
    ) or "moddingSuite/atlas_cli/moddingSuite.AtlasCli.exe"
    settings.use_atlas_json_mapping = get_bool("use_atlas_json_mapping", settings.use_atlas_json_mapping)
    settings.atlas_json_strict = get_bool("atlas_json_strict", settings.atlas_json_strict)
    settings.atlas_json_cache_subdir = (
        get_text("atlas_json_cache_subdir", settings.atlas_json_cache_subdir) or "atlas_json_cache"
    )
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
    raw["modding_suite_atlas_wrapper"] = str(settings.modding_suite_atlas_wrapper or "modding_suite_atlas_export.py")
    raw["modding_suite_atlas_cli"] = str(settings.modding_suite_atlas_cli or "moddingSuite/atlas_cli/moddingSuite.AtlasCli.exe")
    raw["use_atlas_json_mapping"] = bool(settings.use_atlas_json_mapping)
    raw["atlas_json_strict"] = bool(settings.atlas_json_strict)
    raw["atlas_json_cache_subdir"] = str(settings.atlas_json_cache_subdir or "atlas_json_cache")
    raw["texture_subdir"] = "textures"
    raw["cache_dir"] = settings.cache_dir
    raw["use_zz_dat_source"] = bool(settings.use_zz_dat_source)
    raw["warno_root"] = settings.warno_root
    raw["modding_suite_root"] = settings.modding_suite_root
    raw["zz_runtime_dir"] = settings.zz_runtime_dir

    raw["auto_textures"] = bool(settings.auto_textures)
    raw["fast_exact_texture_resolve"] = bool(settings.fast_exact_texture_resolve)
    raw["texture_process_timeout_sec"] = int(settings.texture_process_timeout_sec)
    raw["atlas_cli_timeout_sec"] = int(settings.atlas_cli_timeout_sec)
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


def _build_bone_payload(
    extractor_mod,
    spk,
    model: dict[str, Any],
    asset: str,
    meta: dict[str, Any],
    material_role_by_id: dict[int, str],
    rot: dict[str, float],
    skeleton_spks,
    unit_ndf_hints,
) -> dict[str, Any]:
    _ = unit_ndf_hints
    mesh_node_index = int(meta.get("nodeIndex", -1))
    mesh_bone_names: List[str] = []
    external_bone_names: List[str] = []
    bone_names: List[str] = []
    bone_name_by_index: Dict[int, str] = {}
    bone_parent_by_index: Dict[int, int] = {}
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
        if len(parents) < len(names):
            parents = [int(x) for x in parents] + [-1] * (len(names) - len(parents))
        elif len(parents) > len(names):
            parents = [int(x) for x in parents[: len(names)]]
        candidates.append(
            {
                "source": str(source),
                "names": [str(x) for x in names],
                "parents": [int(x) for x in parents],
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
        primary_raw_names = [str(n) for n in selected.get("names", [])]
        raw_names = list(primary_raw_names)
        bone_names = [str(n).strip() for n in primary_raw_names if str(n).strip()]
        for bidx, raw_name in enumerate(raw_names):
            name_text = str(raw_name or "").strip()
            if not name_text:
                continue
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

        if hasattr(extractor_mod, "unique_keep_order"):
            try:
                bone_names = [str(x) for x in extractor_mod.unique_keep_order(bone_names)]
            except Exception:
                pass

    bone_centers_by_index = extractor_mod.estimate_bone_centers_by_index(model, bone_name_by_index, rot)
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

    return {
        "bone_name_by_index": bone_name_by_index,
        "bone_parent_by_index": bone_parent_by_index,
        "bone_names": bone_names,
        "bone_positions": bone_positions,
        "bone_name_source": bone_name_source,
        "mesh_bone_names": mesh_bone_names,
        "external_bone_names": external_bone_names,
        "ndf_hint_bones": [],
        "ndf_hint_source": "disabled",
        "ndf_hint_error": "",
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
    if "normal_reconstructed" in low or low.endswith("_nm") or "normal" in low:
        return "normal"
    if low.endswith("_o") or low.endswith("_ao") or "occlusion" in low:
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
            refs_before = len(refs)
            refs = extractor_mod.unique_keep_order([*refs, *atlas_map_targets]) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys([*refs, *atlas_map_targets]))
            refs_added = len(refs) - refs_before
            if refs_added > 0:
                _warno_log(
                    settings,
                    f"strict atlas refs extended from atlas targets: +{refs_added}",
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

        group_payloads: List[Dict[str, Any]] = []
        if split_main_parts:
            if bone_name_by_index and hasattr(extractor_mod, "split_faces_by_bone_deterministic"):
                try:
                    group_payloads = extractor_mod.split_faces_by_bone_deterministic(
                        part=part,
                        bone_name_by_index=bone_name_by_index,
                        bone_parent_by_index=bone_parent_by_index or {},
                        material_role=role,
                        material_name=mat_name,
                    )
                except Exception:
                    group_payloads = []
            elif bone_name_by_index and hasattr(extractor_mod, "split_faces_by_bone_top_level"):
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
                    "faces": [],
                    "face_mids": [],
                    "map": {},
                }
                buckets[key] = bucket
                order.append(key)
            elif int(bucket.get("group_bone_index", -1)) < 0 and int(group_bone_index) >= 0:
                bucket["group_bone_index"] = int(group_bone_index)

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


def _apply_material_nodes(
    mat: bpy.types.Material,
    maps: dict[str, Path],
    role: str,
    ao_multiply_diffuse: bool = True,
    normal_invert_mode: str = "none",
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
    settings: WARNOImporterSettings | None = None,
) -> bpy.types.Object | None:
    bone_name_by_index_raw = bone_payload.get("bone_name_by_index", {}) or {}
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
    ordered_indices = sorted(int(i) for i in bone_name_by_index.keys())

    bone_len = max(0.003, diag * 0.00035) if diag > 0.0 else 0.003
    bone_len = min(bone_len, 0.03)

    weighted_positions_by_index: Dict[int, Vector] = {}
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

    resolved_positions: Dict[int, Vector] = {int(k): v.copy() for k, v in weighted_positions_by_index.items()}
    pending: set[int] = {int(i) for i in ordered_indices if int(i) not in resolved_positions}
    fallback_from_children = 0
    fallback_from_parent = 0
    fallback_line_placed = 0
    missing_parent_anchor = 0

    # Fallback A: pull position from already-resolved children.
    for _ in range(max(1, len(pending) + 1)):
        if not pending:
            break
        changed = False
        for bidx in list(pending):
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
        jitter = -1.0 if (line_idx % 2 == 0) else 1.0
        resolved_positions[int(bidx)] = anchor + Vector((0.0025 * jitter, 0.0, 0.02 + float(line_idx) * 0.004))
        pending.remove(int(bidx))
        fallback_line_placed += 1
        missing_parent_anchor += 1

    def _to_pretty_name(raw_name: str) -> str:
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

    def _set_parent_keep_world(child: bpy.types.Object, parent: bpy.types.Object) -> None:
        wm = child.matrix_world.copy()
        child.parent = parent
        child.parent_type = "OBJECT"
        child.parent_bone = ""
        child.matrix_parent_inverse = parent.matrix_world.inverted()
        child.matrix_world = wm

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
    for obj in imported_objects:
        low = _norm_low(obj.name)
        if low.startswith("chenille_droite"):
            track_right_obj = obj
        elif low.startswith("chenille_gauche"):
            track_left_obj = obj

    fx_tir_re = re.compile(r"^fx_tourelle([0-9]+)_tir_([0-9]+)$", re.IGNORECASE)
    for bidx, raw_name in bone_name_by_index.items():
        low = _norm_low(raw_name)
        if not low.startswith("fx_"):
            continue

        if low.startswith("fx_fumee_chenille_d"):
            if track_right_obj is not None:
                mn, mx, _ctr = _bounds_world(track_right_obj)
                dx = max(0.05, (mx.x - mn.x) * 0.05)
                dz = (mx.z - mn.z) * 0.18
                resolved_positions[int(bidx)] = Vector((mn.x - dx, mn.y, mn.z + dz))
            continue
        if low.startswith("fx_fumee_chenille_g"):
            if track_left_obj is not None:
                mn, mx, _ctr = _bounds_world(track_left_obj)
                dx = max(0.05, (mx.x - mn.x) * 0.05)
                dz = (mx.z - mn.z) * 0.18
                resolved_positions[int(bidx)] = Vector((mn.x - dx, mx.y, mn.z + dz))
            continue

        mt = fx_tir_re.match(low)
        if mt:
            canon_num = int(mt.group(1))
            tir_num = int(mt.group(2))
            canon_obj = canon_by_number.get(canon_num)
            if canon_obj is not None:
                mn, mx, ctr = _bounds_world(canon_obj)
                lx = max(0.01, (mx.x - mn.x) * 0.02)
                y_val = ctr.y
                if tir_num == 2:
                    y_val = mn.y
                elif tir_num == 1:
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
                resolved_positions[int(bidx)] = Vector((tctr.x, tctr.y, tmx.z + (tmx.z - tmn.z) * 0.12))

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

    used_object_names: set[str] = {_norm_low(o.name) for o in bpy.data.objects}
    node_by_bone_index: Dict[int, bpy.types.Object] = {}
    created_empties = 0

    for bidx in ordered_indices:
        mesh_nodes = mesh_by_bone_index.get(int(bidx), [])
        if mesh_nodes:
            node_by_bone_index[int(bidx)] = mesh_nodes[0]
            continue

        raw = str(bone_name_by_index.get(int(bidx), "")).strip()
        if not raw:
            continue
        raw_low = _norm_low(raw)
        if re.fullmatch(r"empty(?:\.[0-9]+)?", raw_low):
            continue
        if raw_low.startswith("cylinder."):
            continue
        if raw_low.startswith("armature_"):
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
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 1.0
        collection.objects.link(empty)
        node_by_bone_index[int(bidx)] = empty
        created_empties += 1

    # Place object origins/empties at resolved deterministic positions.
    for bidx, node in node_by_bone_index.items():
        pos = resolved_positions.get(int(bidx))
        if pos is None:
            continue
        if node.type == "MESH":
            _set_object_origin_world(node, pos)
        else:
            node.location = pos

    # Build small wheel-elev armatures (dev-like) and bind track meshes to side root armatures.
    elev_groups: Dict[str, List[int]] = {}
    elev_pattern = re.compile(r"^roue_elev_([dg])([0-9]+)$", re.IGNORECASE)
    for bidx in ordered_indices:
        raw = str(bone_name_by_index.get(int(bidx), "")).strip()
        m = elev_pattern.match(_norm_low(raw))
        if not m:
            continue
        parent_idx = int(bone_parent_by_index.get(int(bidx), -1))
        parent_raw = _norm_low(str(bone_name_by_index.get(parent_idx, "")))
        if parent_raw.startswith("armature_"):
            grp = parent_raw
        else:
            side = str(m.group(1)).lower()
            num = int(m.group(2))
            grp_num = 1 if num in {1, 9} else 2
            grp = f"armature_{side}{grp_num}"
        elev_groups.setdefault(grp, []).append(int(bidx))

    created_armatures: List[bpy.types.Object] = []
    armature_by_side: Dict[str, bpy.types.Object] = {}

    for grp_low in sorted(elev_groups.keys()):
        bone_indices = sorted(elev_groups.get(grp_low, []))
        if not bone_indices:
            continue
        arm_name = _to_pretty_name(grp_low)
        arm_data = bpy.data.armatures.new(arm_name)
        arm_obj = bpy.data.objects.new(arm_name, arm_data)
        collection.objects.link(arm_obj)
        if hasattr(arm_data, "display_type"):
            arm_data.display_type = "OCTAHEDRAL"
        if hasattr(arm_obj, "show_in_front"):
            arm_obj.show_in_front = False
        if hasattr(arm_obj, "display_type"):
            arm_obj.display_type = "TEXTURED"

        for o in bpy.context.scene.objects:
            o.select_set(False)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")
        for bidx in bone_indices:
            braw = str(bone_name_by_index.get(int(bidx), "")).strip()
            bname = _to_pretty_name(braw)[:63] if braw else f"Bone_{int(bidx):03d}"
            eb = arm_data.edit_bones.new(bname)
            pos = resolved_positions.get(int(bidx), Vector((0.0, 0.0, 0.0)))
            eb.head = pos
            eb.tail = pos + Vector((0.0, 0.0, bone_len))
        bpy.ops.object.mode_set(mode="OBJECT")

        created_armatures.append(arm_obj)
        m = re.match(r"^armature_([DG])1$", arm_name, flags=re.IGNORECASE)
        if m:
            side = "right" if m.group(1).upper() == "D" else "left"
            armature_by_side[side] = arm_obj

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
        if child_raw.startswith("fx_fumee_chenille_"):
            continue
        parent = node_by_bone_index.get(int(parent_idx))
        if parent is None or parent == child:
            continue
        _set_parent_keep_world(child, parent)
        parented_nodes += 1

    # Track meshes parent to side armature object (dev-like).
    track_parented = 0
    for obj in imported_objects:
        low = _norm_low(obj.name)
        side = ""
        if "chenille_gauche" in low:
            side = "left"
        elif "chenille_droite" in low:
            side = "right"
        if not side:
            continue
        arm = armature_by_side.get(side)
        if arm is None:
            continue
        _set_parent_keep_world(obj, arm)
        track_parented += 1

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
                f"| empties_created={created_empties} "
                f"| wheel_armatures_created={len(created_armatures)} "
                f"| object_parent_links={parented_nodes} "
                f"| track_parent_links={track_parented}"
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
) -> Dict[str, Any]:
    extractor_mod = _load_extractor_module_for_worker(extractor_path)
    assets = _scan_assets_from_spk_paths(extractor_mod, spk_paths, query=None)
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
            }

            def worker():
                try:
                    payload = _build_asset_index_payload_worker(
                        extractor_path=Path(self._job["extractor_path"]),
                        spk_paths=[Path(p) for p in self._job["spk_paths"]],
                        signature=dict(self._job["signature"]),
                        index_path=Path(self._job["index_path"]),
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


class WARNO_OT_RebuildAtlasJsonCache(Operator):
    bl_idname = "warno.rebuild_atlas_json_cache"
    bl_label = "Rebuild Atlas JSON Cache"
    bl_description = "Force rebuild Atlas JSON map for current selected asset"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)

        asset = str(settings.selected_asset or "").strip()
        if not asset:
            msg = "No asset selected. Scan and pick an asset first."
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        project_root = _project_root(settings)
        warno_root = _resolve_path(project_root, str(settings.warno_root or "").strip())
        modsuite_root = _resolve_path(project_root, str(settings.modding_suite_root or "").strip())
        wrapper_path = _resolve_path(
            project_root,
            str(settings.modding_suite_atlas_wrapper or "").strip() or "modding_suite_atlas_export.py",
        )
        atlas_cli_text = str(settings.modding_suite_atlas_cli or "").strip()
        atlas_cli_path = _resolve_path(project_root, atlas_cli_text) if atlas_cli_text else None
        cache_root = _atlas_json_cache_root(settings)

        if not warno_root.exists() or not warno_root.is_dir():
            msg = f"WARNO folder not found: {warno_root}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        if not modsuite_root.exists() or not modsuite_root.is_dir():
            msg = f"moddingSuite root not found: {modsuite_root}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        try:
            extractor_mod = _extractor_module(settings)
            if hasattr(extractor_mod, "clear_atlas_json_cache"):
                extractor_mod.clear_atlas_json_cache()
            info = extractor_mod.build_or_load_atlas_texture_map(
                warno_root=warno_root,
                modding_suite_root=modsuite_root,
                asset_path=asset,
                cache_dir=cache_root,
                wrapper_path=wrapper_path,
                atlas_cli_path=atlas_cli_path,
                force_rebuild=True,
                timeout_sec=max(5, int(settings.atlas_cli_timeout_sec)),
            )
            entries = int(info.get("atlas_map_entries", 0) or 0)
            map_path = str(info.get("atlas_map_path", "")).strip()
            msg = f"Atlas JSON rebuilt: entries={entries} | {map_path or asset}"
            settings.status = msg
            _warno_log(settings, msg, stage="atlas")
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        except Exception as exc:
            msg = f"Atlas JSON rebuild failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="atlas")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}


def _strip_texture_channel_suffix(stem: str) -> str:
    raw = str(stem or "").strip()
    if not raw:
        return ""
    out = re.sub(r"(?i)_(diffuse|albedo|color|normal|roughness|metallic|occlusion|alpha)$", "", raw)
    out = re.sub(r"(?i)_(d|nm|r|m|ao|a|orm)$", "", out)
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
            mesh_spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
            if not mesh_spk_paths:
                raise RuntimeError("No mesh SPK files found.")
            pick = _pick_best_asset_spk_path(extractor_mod, mesh_spk_paths, asset)
            if pick is None:
                raise RuntimeError(f"Asset not found in SPK sources: {asset}")
            mesh_spk_path, asset_hint = pick

            with ExitStack() as stack:
                spk = stack.enter_context(extractor_mod.SpkMeshExtractor(mesh_spk_path))
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
        except Exception as exc:
            msg = f"Texture apply failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="apply_textures")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        target_meshes = _target_model_meshes(context, settings)
        if col_name:
            col = bpy.data.collections.get(col_name)
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
                key = _material_key(mat.name)
                maps = maps_by_mat_key.get(key)
                if not maps:
                    maps = material_maps_by_name.get(mat.name)
                if not maps:
                    continue
                role = role_by_mat_key.get(key, "other")
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
        collection = bpy.data.collections.get(col_name)
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


class WARNO_OT_ImportAsset(Operator):
    bl_idname = "warno.import_asset"
    bl_label = "Import To Blender"
    bl_description = "Import selected WARNO asset directly into current scene"

    def execute(self, context):
        settings = context.scene.warno_import
        _enforce_fixed_runtime_defaults(settings)
        t0 = time.monotonic()
        _warno_log(settings, "Import started.", stage="import")
        asset = str(settings.selected_asset or "").strip()
        if not asset or asset == "__none__":
            msg = "Pick asset first (Scan Assets)."
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="import")
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
        runtime_info: Dict[str, Any] = {}
        try:
            _set_status(settings, "stage: preparing runtime", stage="import")
            runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
            _warno_log(settings, "runtime source policy: zz_runtime_only", stage="runtime")
        except Exception as exc:
            msg = f"ZZ runtime prepare failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        mesh_spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
        if not mesh_spk_paths:
            msg = "No mesh SPK files found in prepared ZZ runtime."
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        pick = _pick_best_asset_spk_path(extractor_mod, mesh_spk_paths, asset)
        if pick is None:
            msg = f"Asset not found in selected SPK sources: {asset}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        mesh_spk_path, asset_hint = pick

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
            "bone_parent_by_index": {},
            "bone_names": [],
            "bone_positions": {},
        }

        try:
            with ExitStack() as stack:
                _set_status(settings, "stage: reading SPK/model", stage="import")
                spk = stack.enter_context(extractor_mod.SpkMeshExtractor(mesh_spk_path))
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

                skeleton_spks: List[Any] = []
                if need_bone_map:
                    for skeleton_path in _resolve_skeleton_spk_paths(project_root, settings, runtime_info):
                        try:
                            skeleton_spks.append(stack.enter_context(extractor_mod.SpkMeshExtractor(skeleton_path)))
                        except Exception:
                            continue

                ndf_hints = None

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
                        unit_ndf_hints=ndf_hints,
                    )
                    _warno_log(
                        settings,
                        (
                            "bone payload: "
                            f"source={bone_payload.get('bone_name_source', 'none')} "
                            f"names={len(bone_payload.get('bone_names', []) or [])} "
                            f"indexed={len(bone_payload.get('bone_name_by_index', {}) or {})} "
                            f"positions={len(bone_payload.get('bone_positions', {}) or {})}"
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
                )
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
        except Exception as exc:
            msg = f"Import prep failed: {exc}"
            settings.status = msg
            _warno_log(settings, msg, level="ERROR", stage="import")
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        if not buckets:
            msg = "No mesh geometry to import."
            settings.status = msg
            _warno_log(settings, msg, level="WARNING", stage="import")
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        collection = _ensure_collection(context.scene, f"WARNO_{Path(asset).stem}")
        settings.last_import_collection = collection.name
        imported_objects: List[bpy.types.Object] = []
        material_cache: Dict[str, bpy.types.Material] = {}
        mesh_count = 0
        used_object_names: set[str] = set()
        armature_obj: bpy.types.Object | None = None

        for bucket_i, bucket in enumerate(buckets, start=1):
            group_name = str(bucket.get("group_name", f"Part_{bucket_i:03d}"))
            if settings.auto_name_parts:
                base_name = _safe_name(group_name, f"Part_{bucket_i:03d}")[:63]
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
                    _apply_material_nodes(
                        mat,
                        maps,
                        role,
                        ao_multiply_diffuse=False,
                        normal_invert_mode="none",
                    )
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
            obj["warno_group_bone_index"] = int(bucket.get("group_bone_index", -1))
            collection.objects.link(obj)
            imported_objects.append(obj)
            mesh_count += 1

        # Geometry cleanup on import: merge by distance first.
        _set_status(settings, "stage: building scene objects", stage="import")
        _merge_by_distance(imported_objects, float(FIXED_MERGE_DISTANCE))
        try:
            _apply_auto_smooth_modifier(imported_objects, float(FIXED_AUTO_SMOOTH_ANGLE))
        except Exception as exc:
            _warno_log(settings, f"auto_smooth warning: {exc}", level="WARNING", stage="import")

        if settings.auto_pull_bones:
            try:
                armature_obj = _build_helper_armature(imported_objects, bone_payload, collection, settings=settings)
            except Exception as exc:
                self.report({"WARNING"}, f"Armature build failed: {exc}")
                armature_obj = None

        for obj in context.scene.objects:
            obj.select_set(False)
        for obj in imported_objects:
            obj.select_set(True)
        if imported_objects:
            context.view_layer.objects.active = imported_objects[0]
        if armature_obj is not None:
            armature_obj.select_set(True)

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
        if armature_obj is not None:
            msg += " + armature"
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
            src.prop(s, "warno_root")
            src.prop(s, "modding_suite_root")
            src.prop(s, "modding_suite_atlas_cli")
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
            row.operator("warno.rebuild_atlas_json_cache", text="Rebuild Atlas JSON Cache", icon="FILE_REFRESH")
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
            qry.operator("warno.pick_asset_browser", text="Pick Asset Browser", icon="FILEBROWSER")
            qry.label(text=f"Current: {str(s.selected_asset or '').strip()}", icon="OBJECT_DATA")

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

        import_row = layout.row(align=True)
        import_row.operator("warno.import_asset", text="Import To Blender", icon="IMPORT")
        import_row.operator("warno.manual_auto_smooth_apply", text="Manual Auto Smooth Apply", icon="MOD_SMOOTH")
        layout.label(
            text="Auto smooth use only if your model looks unsmoothed and polygons are clearly visible",
            icon="INFO",
        )
        if s.status:
            layout.label(text=s.status)


CLASSES = [
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
    WARNO_OT_PrepareZZRuntime,
    WARNO_OT_InstallTGVDeps,
    WARNO_OT_OpenSystemConsole,
    WARNO_OT_OpenLogFile,
    WARNO_OT_RebuildAtlasJsonCache,
    WARNO_OT_OpenTextureFolder,
    WARNO_OT_ClearTextureFolder,
    WARNO_OT_ApplyTextures,
    WARNO_OT_ManualAutoSmoothApply,
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
