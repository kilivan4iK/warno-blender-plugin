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
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
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
LOD_SUFFIX_LOCAL_RX = re.compile(r"_(LOW|MID|HIGH|LOD[0-9]+)$", re.IGNORECASE)
CONTROL_CHAR_RX = re.compile(r"[\x00-\x1f\x7f]+")


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


def _zz_runtime_root(settings: "WARNOImporterSettings") -> Path:
    project_root = _project_root(settings)
    raw = str(settings.zz_runtime_dir or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    cache_raw = str(settings.cache_dir or "").strip()
    if cache_raw:
        return _resolve_path(project_root, cache_raw) / "_zz_runtime"
    return project_root / "out_blender_runtime" / "zz_runtime"


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


def _prepare_zz_runtime_sources(extractor_mod, settings: "WARNOImporterSettings") -> Dict[str, Any]:
    if not bool(settings.use_zz_dat_source):
        return {}
    project_root = _project_root(settings)
    warno_raw = str(settings.warno_root or "").strip()
    if not warno_raw:
        raise RuntimeError("WARNO Folder is empty while ZZ.dat source is enabled.")
    warno_root = _resolve_path(project_root, warno_raw)
    if not warno_root.exists() or not warno_root.is_dir():
        raise RuntimeError(f"WARNO Folder not found: {warno_root}")

    runtime_root = _zz_runtime_root(settings)
    runtime_root.mkdir(parents=True, exist_ok=True)
    info = extractor_mod.prepare_runtime_sources_from_zz(warno_root=warno_root, runtime_root=runtime_root)
    info["runtime_root"] = str(runtime_root)
    cand_tgv = _candidate_tgv_converter_from_modding_suite(settings)
    if cand_tgv is not None:
        info["tgv_converter"] = str(cand_tgv)

    try:
        resolver = extractor_mod.get_zz_runtime_resolver(warno_root)
    except Exception:
        resolver = None
    if resolver is not None:
        info["zz_resolver"] = resolver
    return info


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
    runtime_info = runtime_info or {}
    if bool(settings.use_zz_dat_source):
        out = _spk_paths_from_runtime_info(runtime_info, "mesh_spk_files", "mesh_spk", "mesh_spk_dir")
        out = [p for p in out if "skeleton" not in p.name.lower()]
        if out:
            return out
    out = _dedupe_paths(_expand_spk_source_paths(project_root, settings.spk_path))
    return [p for p in out if "skeleton" not in p.name.lower()]


def _resolve_skeleton_spk_paths(
    project_root: Path,
    settings: "WARNOImporterSettings",
    runtime_info: Dict[str, Any] | None = None,
) -> List[Path]:
    runtime_info = runtime_info or {}
    if bool(settings.use_zz_dat_source):
        out = _spk_paths_from_runtime_info(runtime_info, "skeleton_spk_files", "skeleton_spk", "skeleton_spk_dir")
        out = [p for p in out if "skeleton" in p.name.lower()]
        if out:
            return out
    out = _dedupe_paths(_expand_spk_source_paths(project_root, settings.skeleton_spk))
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

    labels = _unique_asset_labels([str(a) for a in assets])
    out = []
    for i, asset in enumerate(assets):
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
                item = {"key": key, "label": label, "models": []}
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
        out.append({"key": key, "label": label, "models": models})
    return out


def _browser_folder_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__none__", "<scan assets first>", "No folders", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__none__", "<scan assets first>", "No folders", 0)]
    out = []
    for i, folder in enumerate(folders):
        key = str(folder.get("key", "")).strip()
        label = str(folder.get("label", "")).strip() or key
        out.append((key, label, key, i))
    return out


def _browser_model_items(self, context):
    settings = getattr(getattr(context, "scene", None), "warno_import", None)
    if settings is None:
        return [("__none__", "<no models>", "No models", 0)]
    folders = _asset_folders_from_cache(settings)
    if not folders:
        return [("__none__", "<scan assets first>", "No models", 0)]
    folder_key = str(getattr(self, "folder_key", "") or "").strip()
    target = None
    for f in folders:
        if str(f.get("key", "")).strip() == folder_key:
            target = f
            break
    if target is None:
        target = folders[0]
    models = target.get("models", [])
    if not isinstance(models, list):
        models = []

    search = str(getattr(self, "search_text", "") or "").strip().lower()
    primaries: List[str] = []
    for m in models:
        p = str(m.get("primary", "")).strip()
        if not p:
            continue
        name = _asset_display_name(p).lower()
        if search and search not in name:
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
            settings.selected_asset_group = primary
            settings.selected_asset_lod = "__base__"
            return
        if sel in lods:
            settings.selected_asset_group = primary
            settings.selected_asset_lod = sel
            return
    settings.selected_asset_group = str(groups[0].get("primary", "")).strip()
    settings.selected_asset_lod = "__base__"


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
            self.selected_asset = primary
            self.selected_asset_lod = "__base__"
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
            self.selected_asset = lod
        elif primary and primary != "__none__":
            self.selected_asset = primary
    finally:
        self.asset_sync_lock = False


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
    manual_texture_tool: StringProperty(
        name="Manual Texture Tool",
        subtype="FILE_PATH",
        default="manual_texture_corrector_cpp/build/Release/manual_texture_corrector.exe",
        description="Path to external C++ manual texture correction tool",
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
    selected_asset: EnumProperty(
        name="Asset",
        description="Asset path to import",
        items=_asset_enum_items,
        update=_update_selected_asset,
    )
    selected_asset_group: EnumProperty(
        name="Main Asset",
        description="Main (base) asset",
        items=_asset_group_enum_items,
        update=_update_selected_asset_group,
    )
    selected_asset_lod: EnumProperty(
        name="LOD Asset",
        description="LOD variant for selected main asset",
        items=_asset_lod_enum_items,
        update=_update_selected_asset_lod,
    )
    show_asset_lods: BoolProperty(
        name="Show LODs",
        default=False,
        description="Expand to pick LOD variant under selected main asset",
    )

    auto_textures: BoolProperty(name="Auto textures", default=True)
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

    tgv_split_mode: EnumProperty(
        name="TGV split",
        items=(
            ("auto", "Auto", "Auto split channels"),
            ("all", "All", "Save all channels"),
            ("none", "None (better use with manual correcting)", "No channel split"),
        ),
        default="auto",
    )
    # Deprecated: TGV mirroring is always OFF.
    tgv_mirror: BoolProperty(name="Mirror TGV", default=False, options={"HIDDEN"})
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
    # Deprecated: Mirror Y is always ON.
    mirror_y: BoolProperty(name="Mirror Y", default=True, options={"HIDDEN"})

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
    settings.project_root = get_text("project_root", settings.project_root) or settings.project_root
    settings.atlas_assets_dir = get_text("atlas_assets_dir", settings.atlas_assets_dir)
    settings.tgv_converter = get_text("tgv_converter", settings.tgv_converter)
    settings.manual_texture_tool = get_text("manual_texture_tool", settings.manual_texture_tool) or settings.manual_texture_tool
    settings.texture_subdir = get_text("texture_subdir", settings.texture_subdir) or "textures"
    settings.cache_dir = get_text("cache_dir", settings.cache_dir) or settings.cache_dir
    settings.use_zz_dat_source = get_bool("use_zz_dat_source", settings.use_zz_dat_source)
    settings.warno_root = get_text("warno_root", settings.warno_root)
    settings.modding_suite_root = get_text("modding_suite_root", settings.modding_suite_root)
    settings.zz_runtime_dir = get_text("zz_runtime_dir", settings.zz_runtime_dir)

    settings.auto_textures = get_bool("auto_textures", settings.auto_textures)
    settings.auto_install_tgv_deps = get_bool("auto_install_tgv_deps", settings.auto_install_tgv_deps)
    settings.tgv_deps_dir = get_text("tgv_deps_dir", settings.tgv_deps_dir) or ".warno_pydeps"
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
    settings.tgv_mirror = False
    settings.tgv_aggressive_split = get_bool("tgv_aggressive_split", settings.tgv_aggressive_split)
    settings.auto_rename_textures = get_bool("auto_rename_textures", settings.auto_rename_textures)
    settings.use_ao_multiply = get_bool("ao_multiply_diffuse", settings.use_ao_multiply)

    settings.rotate_x = get_float("rotate_x", settings.rotate_x)
    settings.rotate_y = get_float("rotate_y", settings.rotate_y)
    settings.rotate_z = get_float("rotate_z", settings.rotate_z)
    settings.mirror_y = True
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

    raw["project_root"] = settings.project_root
    raw["spk_path"] = settings.spk_path
    raw["skeleton_spk"] = settings.skeleton_spk
    raw["atlas_assets_dir"] = settings.atlas_assets_dir
    raw["tgv_converter"] = settings.tgv_converter
    raw["manual_texture_tool"] = settings.manual_texture_tool
    raw["texture_subdir"] = settings.texture_subdir
    raw["cache_dir"] = settings.cache_dir
    raw["use_zz_dat_source"] = bool(settings.use_zz_dat_source)
    raw["warno_root"] = settings.warno_root
    raw["modding_suite_root"] = settings.modding_suite_root
    raw["zz_runtime_dir"] = settings.zz_runtime_dir

    raw["auto_textures"] = bool(settings.auto_textures)
    raw["auto_install_tgv_deps"] = bool(settings.auto_install_tgv_deps)
    raw["tgv_deps_dir"] = str(settings.tgv_deps_dir or ".warno_pydeps")
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
    raw.pop("tgv_mirror", None)
    raw["tgv_aggressive_split"] = bool(settings.tgv_aggressive_split)
    raw["auto_rename_textures"] = bool(settings.auto_rename_textures)
    raw["ao_multiply_diffuse"] = bool(settings.use_ao_multiply)

    raw["rotate_x"] = float(settings.rotate_x)
    raw["rotate_y"] = float(settings.rotate_y)
    raw["rotate_z"] = float(settings.rotate_z)
    raw["mirror_y"] = True
    raw.pop("unit_ndfbin", None)
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
    skeleton_spks,
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

    if skeleton_spks is None:
        skeleton_spk_list: List[Any] = []
    elif isinstance(skeleton_spks, (list, tuple)):
        skeleton_spk_list = [sp for sp in skeleton_spks if sp is not None]
    else:
        skeleton_spk_list = [skeleton_spks]

    if skeleton_spk_list:
        used_ext_indices: set[str] = set()
        used_ext_signatures: set[tuple[str, str]] = set()

        def add_external_set(source_name: str, spk_key: str, node_idx: int, skeleton_spk) -> None:
            if node_idx < 0:
                return
            nidx = int(node_idx)
            uniq_idx = f"{spk_key}:{nidx}"
            if uniq_idx in used_ext_indices:
                return
            used_ext_indices.add(uniq_idx)
            try:
                names = skeleton_spk.parse_node_names(nidx)
            except Exception:
                names = []
            if not names:
                return
            sig = "\x1f".join(str(n).strip().lower() for n in names if str(n).strip())
            sig_key = (spk_key, sig)
            if not sig or sig_key in used_ext_signatures:
                return
            used_ext_signatures.add(sig_key)
            external_sets.append((source_name, list(names)))
            source_name_lists[source_name] = list(names)

        for sk_i, skeleton_spk in enumerate(skeleton_spk_list):
            try:
                spk_key = str(getattr(skeleton_spk, "path", f"skeleton_{sk_i}")).lower()
            except Exception:
                spk_key = f"skeleton_{sk_i}"

            # Primary: same hierarchical node index from mesh dictionary.
            add_external_set(f"external_same_index_{sk_i}", spk_key, mesh_node_index, skeleton_spk)

            # Secondary fallback: best path match in external Skeleton SPK FAT.
            skeleton_hit = skeleton_spk.find_best_fat_entry_for_asset(asset)
            if skeleton_hit is not None:
                _, sk_meta = skeleton_hit
                sk_node_idx = int(sk_meta.get("nodeIndex", -1))
                add_external_set(f"external_asset_match_{sk_i}", spk_key, sk_node_idx, skeleton_spk)

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
    if "combinedorm" in low or "combinedrm" in low or low.endswith("_orm"):
        return "orm"
    if "diffusetexturenoalpha" in low or ("noalpha" in low and ("diffuse" in low or "color" in low)):
        return "diffuse"
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
) -> List[str]:
    runtime_info = runtime_info or {}
    out: List[str] = []
    dirs = _asset_atlas_ref_dir_candidates(extractor_mod, asset)
    if not dirs:
        return out

    zz_resolver = runtime_info.get("zz_resolver")
    if zz_resolver is not None:
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

    if not out:
        return []
    uniq = list(dict.fromkeys(_norm_ref_like(v) for v in out if str(v).strip()))
    uniq.sort(key=lambda v: (-_score_guessed_ref(v), len(v), v))
    return uniq


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


def _pick_zz_generic_tsc_parent(extractor_mod, refs: Sequence[str], zz_resolver) -> str | None:
    if zz_resolver is None or not refs:
        return None
    try:
        zz_resolver.all_asset_keys()
    except Exception:
        return None

    parent_score: Dict[str, float] = defaultdict(float)
    parent_roles: Dict[str, set[str]] = defaultdict(set)
    need_roles: set[str] = set()

    for ref in refs:
        role = str(getattr(extractor_mod, "classify_texture_role")(ref))
        if role in {"combined_da"}:
            need_roles.add("diffuse")
        elif role in {"diffuse", "normal", "orm"}:
            need_roles.add(role)
        cands = _collect_zz_candidates_for_ref(zz_resolver, ref)
        for key in cands:
            low = str(key).lower()
            if "/units_tests/" in low or "/units_tests_autos/" in low or "/editor/" in low:
                continue
            if "/fx/" in low:
                continue
            parent = str(PurePosixPath(low).parent)
            role_weight = 20.0
            if role == "diffuse":
                role_weight = 40.0
            elif role == "normal":
                role_weight = 32.0
            elif role == "orm":
                role_weight = 34.0
            size = 0
            try:
                size = int(zz_resolver._assets.get(low, {}).get("size", 0) or 0)
            except Exception:
                size = 0
            size_bonus = min(30.0, float(size) / 120000.0)
            parent_score[parent] += role_weight + size_bonus
            if role == "combined_da":
                parent_roles[parent].add("diffuse")
            elif role in {"diffuse", "normal", "orm"}:
                parent_roles[parent].add(role)

    ranked = sorted(parent_score.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None
    for parent, _score in ranked:
        roles = parent_roles.get(parent, set())
        if need_roles and not need_roles.issubset(roles):
            continue
        return parent
    return ranked[0][0]


def _fallback_convert_zz_parent_folder(
    extractor_mod,
    refs: Sequence[str],
    zz_resolver,
    runtime_root: Path,
    converter: Path,
    model_dir: Path,
    texture_subdir: str,
    tgv_split_mode: str,
    tgv_aggressive_split: bool,
    auto_install_deps: bool,
    deps_dir: Path | None,
) -> int:
    parent = _pick_zz_generic_tsc_parent(extractor_mod, refs, zz_resolver)
    if not parent:
        return 0
    all_keys = zz_resolver.all_asset_keys()
    src_keys = [k for k in all_keys if str(k).startswith(parent + "/") and (str(k).endswith(".tgv") or str(k).endswith(".png"))]
    if not src_keys:
        return 0
    extracted = 0
    for key in src_keys:
        try:
            if str(key).endswith(".tgv"):
                zz_resolver.extract_asset_to_runtime(key, runtime_root)
                extracted += 1
        except Exception:
            continue
    if extracted <= 0:
        return 0

    src_dir = runtime_root / Path(*parent.split("/"))
    if not src_dir.exists() or not src_dir.is_dir():
        return 0

    first_ref = str(refs[0])
    rel_parent = extractor_mod.atlas_ref_to_rel_under_assets(first_ref).parent
    dst_dir = model_dir / (texture_subdir or "textures") / rel_parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    extractor_mod.run_tgv_converter_for_folder_once(
        converter=converter,
        src_dir=src_dir,
        dst_dir=dst_dir,
        split_mode=tgv_split_mode,
        mirror=False,
        aggressive_split=bool(tgv_aggressive_split),
        auto_naming=False,
        auto_install_deps=bool(auto_install_deps),
        deps_dir=deps_dir,
    )
    return extracted


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
    }
    if not settings.auto_textures:
        return maps_by_name, report

    project_root = _project_root(settings)
    runtime_info = runtime_info or {}
    atlas_override = str(runtime_info.get("atlas_assets_root", "")).strip()
    converter_override = str(runtime_info.get("tgv_converter", "")).strip()
    zz_resolver = runtime_info.get("zz_resolver")
    zz_runtime_root_text = str(runtime_info.get("runtime_root", "")).strip()
    warno_root_text = str(runtime_info.get("warno_root", "")).strip()

    atlas_raw_text = atlas_override or str(settings.atlas_assets_dir or "").strip()
    converter_text = converter_override or str(settings.tgv_converter or "").strip()
    atlas_source = "zz_runtime" if atlas_override else "manual_path"
    report["atlas_source"] = atlas_source
    converter = Path()
    converter_source = "custom"
    if converter_text:
        converter = _resolve_path(project_root, converter_text)
    bundled_converter = project_root / "tgv_to_png.py"
    if converter_text and bundled_converter.exists():
        try:
            if converter.resolve() == bundled_converter.resolve():
                converter_source = "bundled"
        except Exception:
            pass
    modsuite_root = _resolve_path(project_root, str(settings.modding_suite_root or "").strip()) if str(settings.modding_suite_root or "").strip() else None
    if converter_text and converter_source != "bundled" and modsuite_root is not None:
        try:
            if str(converter.resolve()).lower().startswith(str(modsuite_root.resolve()).lower()):
                converter_source = "moddingSuite"
        except Exception:
            pass
    if (not converter_text or not converter.exists() or not converter.is_file()) and bool(settings.use_zz_dat_source):
        cand = _candidate_tgv_converter_from_modding_suite(settings)
        if cand is not None:
            converter = cand
            converter_text = str(cand)
            converter_source = "moddingSuite"
    if (not converter_text or not converter.exists() or not converter.is_file()) and bundled_converter.exists() and bundled_converter.is_file():
        converter = bundled_converter
        converter_text = str(bundled_converter)
        converter_source = "bundled"

    if not atlas_raw_text:
        raise RuntimeError("Atlas Assets path is empty.")
    if not converter_text:
        raise RuntimeError("TGV converter path is empty.")
    atlas_raw = _resolve_path(project_root, atlas_raw_text)
    if not atlas_raw.exists() or not atlas_raw.is_dir():
        raise RuntimeError(f"Atlas Assets folder not found: {atlas_raw}")
    if not converter.exists() or not converter.is_file():
        raise RuntimeError(f"TGV converter not found: {converter}")
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

    refs: List[str] = []
    for mid in material_ids:
        refs.extend(refs_by_material.get(int(mid), []))
    refs = extractor_mod.unique_keep_order(refs) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(refs))
    guessed_refs: List[str] = []
    try:
        guessed_refs = _guess_texture_refs_for_asset(
            extractor_mod=extractor_mod,
            asset=asset,
            atlas_assets_root=atlas_root,
            runtime_info=runtime_info,
        )
    except Exception:
        guessed_refs = []
    if not refs:
        refs = spk.find_texture_refs_for_asset(asset, material_ids=material_ids)
    # Some packs expose sparse material refs (often only diffuse). Merge conservative
    # directory guesses to recover missing companion maps (normal/orm/single-channel).
    if guessed_refs:
        if bool(settings.use_zz_dat_source):
            # ZZ mode: always merge all folder refs to convert full unit texture set from runtime.
            refs = extractor_mod.unique_keep_order(list(refs) + list(guessed_refs)) if hasattr(extractor_mod, "unique_keep_order") else list(dict.fromkeys(list(refs) + list(guessed_refs)))
        else:
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
    if not refs:
        refs = guessed_refs

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
    for ref in refs:
        def _resolve_with_converter(conv_path: Path):
            return extractor_mod.resolve_texture_from_atlas_ref(
                ref=ref,
                atlas_assets_root=atlas_root,
                out_model_dir=model_dir,
                converter=conv_path,
                texture_subdir=settings.texture_subdir or "textures",
                tgv_split_mode=settings.tgv_split_mode,
                tgv_mirror=False,
                tgv_aggressive_split=bool(settings.tgv_aggressive_split),
                zz_resolver=zz_resolver,
                zz_runtime_root=Path(zz_runtime_root_text) if zz_runtime_root_text else None,
                fallback_atlas_roots=fallback_atlas_roots,
                auto_install_deps=bool(settings.auto_install_tgv_deps),
                deps_dir=deps_dir,
            )

        try:
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

    if (not resolved) and errors and zz_resolver is not None and zz_runtime_root_text:
        try:
            extracted_count = _fallback_convert_zz_parent_folder(
                extractor_mod=extractor_mod,
                refs=refs,
                zz_resolver=zz_resolver,
                runtime_root=Path(zz_runtime_root_text),
                converter=converter,
                model_dir=model_dir,
                texture_subdir=(settings.texture_subdir or "textures"),
                tgv_split_mode=str(settings.tgv_split_mode or "auto"),
                tgv_aggressive_split=bool(settings.tgv_aggressive_split),
                auto_install_deps=bool(settings.auto_install_tgv_deps),
                deps_dir=deps_dir,
            )
            if extracted_count > 0:
                for ref in refs:
                    try:
                        rel = extractor_mod.atlas_ref_to_rel_under_assets(ref)
                        out_png = model_dir / (settings.texture_subdir or "textures") / rel
                        if not out_png.exists():
                            continue
                        item = {
                            "atlas_ref": ref,
                            "role": extractor_mod.classify_texture_role(ref),
                            "atlas_source": "zz_runtime",
                            "source_type": "zz_parent_folder",
                            "source_tgv": None,
                            "source_png": None,
                            "out_png": out_png,
                            "extras": extractor_mod.find_generated_extra_maps(out_png),
                            "deps_auto_installed": False,
                        }
                        resolved.append(item)
                        resolved_by_ref[str(ref)] = item
                    except Exception:
                        continue
                if resolved_by_ref:
                    errors = [e for e in errors if str(e.get("atlas_ref", "")) not in resolved_by_ref]
        except Exception:
            pass
    report["errors"] = errors
    if used_bundled_fallback:
        report["converter_source"] = "bundled"

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
            maps_by_name[mname] = maps

    report["resolved"] = [
        {
            "atlas_ref": item["atlas_ref"],
            "role": item["role"],
            "atlas_source": str(item.get("atlas_source", "")),
            "source_type": item["source_type"],
            "source_tgv": item["source_tgv"],
            "source_png": item["source_png"],
            "out_png": str(item["out_png"]),
            "extras": {k: str(v) for k, v in item.get("extras", {}).items()},
            "deps_auto_installed": bool(item.get("deps_auto_installed", False)),
        }
        for item in resolved
    ]
    if any(str(item.get("atlas_source", "")).strip().lower() == "fallback" for item in resolved):
        report["atlas_source"] = "fallback"
    report["deps_auto_installed"] = any(bool(item.get("deps_auto_installed", False)) for item in resolved)
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


def _scan_assets_impl(self, context, scan_all: bool):
    settings = context.scene.warno_import
    project_root = _project_root(settings)

    try:
        extractor_mod = _extractor_module(settings)
        runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings) if settings.use_zz_dat_source else {}
        spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
        if not spk_paths:
            msg = "No mesh SPK files found. Set Mesh SPK/Folder or prepare ZZ runtime."
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        query = str(settings.query or "").strip()
        query_for_scan = None if scan_all else (query if query else None)
        seen_assets: set[str] = set()
        matches: List[tuple[str, Dict[str, Any]]] = []
        for spk_path in spk_paths:
            try:
                with extractor_mod.SpkMeshExtractor(spk_path) as spk:
                    for asset, meta in spk.find_matches(query_for_scan, None):
                        key = str(asset).strip().lower()
                        if not key or key in seen_assets:
                            continue
                        seen_assets.add(key)
                        matches.append((asset, meta))
            except Exception:
                continue
        matches.sort(key=lambda it: _asset_picker_sort_key(extractor_mod, str(it[0])))
    except Exception as exc:
        msg = f"Scan failed: {exc}"
        settings.status = msg
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}

    assets_all = [str(asset) for asset, _ in matches]
    assets = assets_all if scan_all else assets_all[: int(settings.match_limit)]
    groups = _build_asset_groups(extractor_mod, assets)
    folders = _build_asset_folders_cache(extractor_mod, groups)

    settings.match_cache_json = json.dumps(assets, ensure_ascii=False)
    settings.asset_group_cache_json = json.dumps(groups, ensure_ascii=False)
    settings.asset_folder_cache_json = json.dumps(folders, ensure_ascii=False)
    settings.asset_sync_lock = True
    try:
        if groups:
            settings.selected_asset_group = str(groups[0].get("primary", "")).strip()
            settings.selected_asset_lod = "__base__"
            settings.selected_asset = settings.selected_asset_group
        elif assets:
            settings.selected_asset = assets[0]
    finally:
        settings.asset_sync_lock = False
    _sync_group_lod_from_selected(settings)

    mode_txt = "ALL" if scan_all else "query"
    msg = (
        f"Matches: {len(matches)} "
        f"(cached: {len(assets)} | groups:{len(groups)} | folders:{len(folders)} | mode:{mode_txt} | spk:{len(spk_paths)})"
    )
    settings.status = msg
    self.report({"INFO"}, msg)
    return {"FINISHED"}


class WARNO_OT_ScanAssets(Operator):
    bl_idname = "warno.scan_assets"
    bl_label = "Scan Assets"
    bl_description = "Find matching assets in Mesh SPK for current query"

    def execute(self, context):
        return _scan_assets_impl(self, context, scan_all=False)


class WARNO_OT_ScanAssetsAll(Operator):
    bl_idname = "warno.scan_assets_all"
    bl_label = "Scan ALL Assets (takes a long time)"
    bl_description = "Scan all assets in all connected Mesh SPK files (can take a long time)"

    def execute(self, context):
        return _scan_assets_impl(self, context, scan_all=True)


class WARNO_OT_PickAssetBrowser(Operator):
    bl_idname = "warno.pick_asset_browser"
    bl_label = "Pick Asset Browser"
    bl_description = "Pick asset by folder/model/LOD in a browser-like dialog"

    folder_key: EnumProperty(name="Folder", items=_browser_folder_items)
    search_text: StringProperty(name="Search", default="")
    model_key: EnumProperty(name="Model", items=_browser_model_items)
    lod_key: EnumProperty(name="LOD", items=_browser_lod_items)

    def invoke(self, context, _event):
        settings = context.scene.warno_import
        folders = _asset_folders_from_cache(settings)
        if not folders:
            self.report({"WARNING"}, "No scanned assets. Run Scan Assets first.")
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

        self.folder_key = found_folder_key or "__none__"
        self.search_text = ""
        self.model_key = found_model_key or "__none__"
        if selected_lod and selected_lod != "__base__":
            self.lod_key = selected_lod
        else:
            self.lod_key = "__base__"

        return context.window_manager.invoke_props_dialog(self, width=760)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(self, "folder_key")
        col.prop(self, "search_text")
        col.prop(self, "model_key")
        col.prop(self, "lod_key")
        chosen = str(self.model_key or "").strip()
        lod = str(self.lod_key or "").strip()
        if lod and lod != "__base__":
            chosen = lod
        col.separator()
        col.label(text=f"Will use: {chosen}", icon="OBJECT_DATA")

    def execute(self, context):
        settings = context.scene.warno_import
        model = str(self.model_key or "").strip()
        if not model or model == "__none__":
            return {"CANCELLED"}
        lod = str(self.lod_key or "").strip()
        final_asset = lod if lod and lod != "__base__" else model

        settings.asset_sync_lock = True
        try:
            settings.selected_asset_group = model
            settings.selected_asset_lod = lod if lod else "__base__"
            settings.selected_asset = final_asset
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
        settings.selected_asset = pick
        _sync_group_lod_from_selected(settings)
        settings.status = f"Picked: {pick}"
        return {"FINISHED"}


class WARNO_OT_PrepareZZRuntime(Operator):
    bl_idname = "warno.prepare_zz_runtime"
    bl_label = "Prepare ZZ Runtime"
    bl_description = "Extract runtime Mesh/Skeleton/Atlas stubs from WARNO ZZ.dat into local cache"

    def execute(self, context):
        settings = context.scene.warno_import
        try:
            extractor_mod = _extractor_module(settings)
            info = _prepare_zz_runtime_sources(extractor_mod, settings)
            if not info:
                msg = "ZZ runtime source is disabled."
                settings.status = msg
                self.report({"WARNING"}, msg)
                return {"CANCELLED"}

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
            if atlas_root and (not str(settings.atlas_assets_dir or "").strip() or settings.use_zz_dat_source):
                settings.atlas_assets_dir = atlas_root
            if bool(settings.use_zz_dat_source):
                cand = _candidate_tgv_converter_from_modding_suite(settings)
                if cand is not None:
                    settings.tgv_converter = str(cand)

            msg = (
                f"ZZ runtime ready: dat={dat_count}, "
                f"mesh_spk={len(mesh_spk_files) if mesh_spk_files else (1 if mesh_spk else 0)}, "
                f"skeleton_spk={len(skeleton_spk_files) if skeleton_spk_files else (1 if skeleton_spk else 0)}"
            )
            settings.status = msg
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        except Exception as exc:
            msg = f"ZZ runtime prepare failed: {exc}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}


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
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff"}
        deleted = 0
        failed = 0
        extra_files = {"manual_texture_manifest.json"}

        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            low_name = p.name.lower()
            low_ext = p.suffix.lower()
            if low_ext in image_exts or low_name in extra_files:
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

            converter_text = str(settings.tgv_converter or "").strip()
            converter = _resolve_path(project_root, converter_text) if converter_text else Path()
            if (not converter_text or not converter.exists() or not converter.is_file()) and bool(settings.use_zz_dat_source):
                cand = _candidate_tgv_converter_from_modding_suite(settings)
                if cand is not None:
                    converter = cand
            if not converter.exists() or not converter.is_file():
                bundled = project_root / "tgv_to_png.py"
                if bundled.exists() and bundled.is_file():
                    converter = bundled
            if not converter.exists() or not converter.is_file():
                raise RuntimeError(f"TGV converter not found: {converter}")

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


def _manual_channel_from_suffix_token(token: str) -> str | None:
    low = _norm_low(token)
    if low in {"d", "diff", "diffuse", "albedo", "color"}:
        return "diffuse"
    if low in {"nm", "normal", "n"}:
        return "normal"
    if low in {"r", "rough", "roughness"}:
        return "roughness"
    if low in {"m", "metal", "metallic"}:
        return "metallic"
    if low in {"ao", "occlusion"}:
        return "occlusion"
    if low in {"a", "alpha"}:
        return "alpha"
    if low in {"orm"}:
        return "roughness"
    return None


def _strip_texture_channel_suffix(stem: str) -> str:
    raw = str(stem or "").strip()
    if not raw:
        return ""
    out = re.sub(r"(?i)_(diffuse|albedo|color|normal|roughness|metallic|occlusion|alpha)$", "", raw)
    out = re.sub(r"(?i)_(d|nm|r|m|ao|a|orm)$", "", out)
    out = re.sub(r"(?i)_part[0-9]+$", "", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out


def _name_similarity_score(target_base: str, cand_base: str, mat_name: str) -> float:
    ta = _norm_low(_strip_texture_channel_suffix(target_base))
    ca = _norm_low(_strip_texture_channel_suffix(cand_base))
    ma = _norm_low(_strip_texture_channel_suffix(mat_name))
    if not ta and not ca:
        return 0.0
    score = 0.0
    if ta and ca:
        if ta == ca:
            score += 120.0
        if ta and ta in ca:
            score += 40.0
        if ca and ca in ta:
            score += 30.0
        pref = 0
        for a, b in zip(ta, ca):
            if a != b:
                break
            pref += 1
        score += min(20.0, pref * 0.8)
        tok_t = {t for t in re.split(r"[^a-z0-9]+", ta) if t}
        tok_c = {t for t in re.split(r"[^a-z0-9]+", ca) if t}
        score += float(len(tok_t & tok_c)) * 12.0
    if ma and ca:
        if ma in ca or ca in ma:
            score += 16.0
        tok_m = {t for t in re.split(r"[^a-z0-9]+", ma) if t}
        tok_c = {t for t in re.split(r"[^a-z0-9]+", ca) if t}
        score += float(len(tok_m & tok_c)) * 6.0
    return score


def _first_image_upstream_socket(socket, max_depth: int = 5):
    visited: set[tuple[int, str, int]] = set()

    def walk(sock, depth: int):
        if sock is None or depth < 0:
            return None
        links = getattr(sock, "links", [])
        for link_i, link in enumerate(links):
            src_node = getattr(link, "from_node", None)
            src_sock = getattr(link, "from_socket", None)
            key = (id(src_node), str(getattr(src_sock, "name", "")), depth)
            if key in visited:
                continue
            visited.add(key)
            if src_node is None:
                continue
            if getattr(src_node, "type", "") == "TEX_IMAGE":
                return src_node
            if depth <= 0:
                continue
            for in_sock in getattr(src_node, "inputs", []):
                hit = walk(in_sock, depth - 1)
                if hit is not None:
                    return hit
        return None

    return walk(socket, max_depth)


def _guess_material_base_from_nodes(mat: bpy.types.Material) -> str:
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return _strip_texture_channel_suffix(mat.name if mat else "")
    nt = mat.node_tree
    bsdf = None
    for node in nt.nodes:
        if getattr(node, "type", "") == "BSDF_PRINCIPLED":
            bsdf = node
            break
    image_node = None
    if bsdf is not None and "Base Color" in bsdf.inputs:
        image_node = _first_image_upstream_socket(bsdf.inputs["Base Color"], max_depth=6)
    if image_node is None:
        for node in nt.nodes:
            if getattr(node, "type", "") == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                image_node = node
                break
    if image_node is not None and image_node.image is not None:
        fp = str(image_node.image.filepath_raw or image_node.image.filepath or "").strip()
        if fp:
            try:
                stem = Path(bpy.path.abspath(fp)).stem
            except Exception:
                stem = Path(fp).stem
            return _strip_texture_channel_suffix(stem)
    return _strip_texture_channel_suffix(mat.name)


def _assign_image_to_material_channel(mat: bpy.types.Material, channel: str, image_path: Path) -> bool:
    if mat is None or not image_path.exists() or not image_path.is_file():
        return False
    if not mat.use_nodes:
        mat.use_nodes = True
    if mat.node_tree is None:
        return False
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = None
    for node in nodes:
        if getattr(node, "type", "") == "BSDF_PRINCIPLED":
            bsdf = node
            break
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    img = _load_image(image_path)
    if img is None:
        return False

    def ensure_tex_node(non_color: bool, x: float, y: float):
        node = nodes.new("ShaderNodeTexImage")
        node.location = (x, y)
        node.image = img
        if non_color:
            try:
                node.image.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        return node

    ch = str(channel or "").lower()
    if ch == "diffuse":
        sock = bsdf.inputs.get("Base Color")
        if sock is None:
            return False
        tex_node = _first_image_upstream_socket(sock, max_depth=6)
        if tex_node is not None:
            tex_node.image = img
            try:
                tex_node.image.colorspace_settings.name = "sRGB"
            except Exception:
                pass
            return True
        tex_node = ensure_tex_node(non_color=False, x=bsdf.location.x - 620, y=bsdf.location.y + 170)
        links.new(tex_node.outputs["Color"], sock)
        return True

    if ch == "normal":
        sock = bsdf.inputs.get("Normal")
        if sock is None:
            return False
        normal_map = None
        if sock.is_linked:
            src = sock.links[0].from_node
            if getattr(src, "type", "") == "NORMAL_MAP":
                normal_map = src
        if normal_map is None:
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.location = (bsdf.location.x - 360, bsdf.location.y - 300)
            links.new(normal_map.outputs["Normal"], sock)
        tex_node = _first_image_upstream_socket(normal_map.inputs.get("Color"), max_depth=4)
        if tex_node is not None:
            tex_node.image = img
            try:
                tex_node.image.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            return True
        tex_node = ensure_tex_node(non_color=True, x=normal_map.location.x - 280, y=normal_map.location.y)
        links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
        return True

    scalar_inputs = {
        "roughness": "Roughness",
        "metallic": "Metallic",
        "alpha": "Alpha",
        "occlusion": None,  # keep AO unconnected by default
    }
    target_input = scalar_inputs.get(ch)
    if ch == "occlusion":
        ensure_tex_node(non_color=True, x=bsdf.location.x - 860, y=bsdf.location.y - 40)
        return True
    if target_input is None:
        return False
    sock = bsdf.inputs.get(target_input)
    if sock is None:
        return False
    tex_node = _first_image_upstream_socket(sock, max_depth=4)
    if tex_node is not None:
        tex_node.image = img
        try:
            tex_node.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        return True
    tex_node = ensure_tex_node(non_color=True, x=bsdf.location.x - 620, y=bsdf.location.y - (250 if ch == "roughness" else 400))
    links.new(tex_node.outputs["Color"], sock)
    return True


def _materials_from_last_import(context, settings: WARNOImporterSettings) -> List[bpy.types.Material]:
    out: List[bpy.types.Material] = []
    seen: set[str] = set()
    for obj in _target_model_meshes(context, settings):
        for slot in getattr(obj, "material_slots", []):
            mat = getattr(slot, "material", None)
            if mat is None:
                continue
            key = _norm_low(mat.name)
            if key in seen:
                continue
            seen.add(key)
            out.append(mat)
    return out


def _apply_manual_manifest_to_materials(context, settings: WARNOImporterSettings, manifest_path: Path) -> int:
    if not manifest_path.exists() or not manifest_path.is_file():
        return 0
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return 0
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    if not isinstance(entries, list):
        return 0

    parsed: List[Dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        primary = str(item.get("primary", "")).strip()
        outs = item.get("outputs", [])
        suffix = str(item.get("suffix", "")).strip()
        base = str(item.get("base", "")).strip()
        channel = str(item.get("channel", "")).strip().lower()
        if not channel:
            channel = str(_manual_channel_from_suffix_token(suffix) or "")
        if not channel:
            continue
        paths_raw: List[str] = []
        if primary:
            paths_raw.append(primary)
        if isinstance(outs, list):
            for op in outs:
                if isinstance(op, str) and op.strip():
                    paths_raw.append(op.strip())
        if not paths_raw:
            continue

        seen_paths: set[str] = set()
        for raw_path in paths_raw:
            p = Path(raw_path)
            if not p.is_absolute():
                p = manifest_path.parent / p
            try:
                key = str(p.resolve()).lower()
            except Exception:
                key = str(p).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            if not p.exists() or not p.is_file():
                continue
            parsed.append({"path": p, "channel": channel, "base": base, "suffix": suffix})
    if not parsed:
        return 0

    mats = _materials_from_last_import(context, settings)
    if not mats:
        return 0

    applied = 0
    for mat in mats:
        target_base = _guess_material_base_from_nodes(mat)
        for channel in ("diffuse", "normal", "roughness", "metallic", "occlusion", "alpha"):
            cands = [it for it in parsed if str(it.get("channel", "")) == channel]
            if not cands:
                continue
            cands = sorted(
                cands,
                key=lambda it: (
                    _name_similarity_score(target_base, str(it.get("base", "")), mat.name),
                    it["path"].stat().st_mtime if it["path"].exists() else 0.0,
                ),
                reverse=True,
            )
            if _assign_image_to_material_channel(mat, channel, cands[0]["path"]):
                applied += 1
    return applied


def _find_latest_manual_manifest(tex_dir: Path, preferred: Path) -> Path | None:
    cand: List[Path] = []
    if preferred.exists() and preferred.is_file():
        cand.append(preferred)
    try:
        for p in tex_dir.rglob("manual_texture_manifest.json"):
            if p.is_file():
                cand.append(p)
    except Exception:
        pass
    if not cand:
        return None
    cand = list(dict.fromkeys(cand))
    cand.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return cand[0]


class WARNO_OT_ManualTextureCorrecting(Operator):
    bl_idname = "warno.manual_texture_correcting"
    bl_label = "Manual texture correcting"
    bl_description = "Launch external C++ tool for manual texture split/correction and apply manifest back to Blender"

    def execute(self, context):
        settings = context.scene.warno_import
        tex_dir_raw = str(settings.last_texture_dir or "").strip()
        if not tex_dir_raw:
            msg = "No texture folder yet. Import model with textures first."
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        tex_dir = Path(tex_dir_raw)
        if not tex_dir.exists() or not tex_dir.is_dir():
            msg = f"Texture folder not found: {tex_dir}"
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}

        project_root = _project_root(settings)
        tool_raw = str(settings.manual_texture_tool or "").strip()
        if not tool_raw:
            msg = "Manual texture tool path is empty."
            settings.status = msg
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        tool_path = _resolve_path(project_root, tool_raw)
        if not tool_path.exists() or not tool_path.is_file():
            msg = f"Tool not found: {tool_path}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        out_dir = tex_dir / "manual_corrected"
        cmd = [str(tool_path), "--input", str(tex_dir), "--output", str(out_dir)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as exc:
            msg = f"Tool launch failed: {exc}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        tool_error_msg = ""
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if len(err) > 240:
                err = err[:240] + "..."
            code = int(proc.returncode)
            crash_note = ""
            if code in (3221225477, -1073741819):
                crash_note = " (access violation crash)"
            tool_error_msg = f"Manual tool failed (code {code}{crash_note}). {err}".strip()

        preferred_manifest = out_dir / "manual_texture_manifest.json"
        manifest = _find_latest_manual_manifest(tex_dir, preferred_manifest)

        # Reload images from known roots to pick file changes.
        roots_low = []
        roots = [tex_dir, out_dir]
        if manifest is not None:
            roots.append(manifest.parent)
        for root in roots:
            try:
                roots_low.append(str(root.resolve()).lower())
            except Exception:
                roots_low.append(str(root).lower())
        for img in bpy.data.images:
            fp = str(img.filepath_raw or img.filepath or "").strip()
            if not fp:
                continue
            try:
                abs_fp = str(Path(bpy.path.abspath(fp)).resolve()).lower()
            except Exception:
                abs_fp = str(Path(fp)).lower()
            if any(abs_fp.startswith(root) for root in roots_low):
                try:
                    img.reload()
                except Exception:
                    pass

        applied = _apply_manual_manifest_to_materials(context, settings, manifest) if manifest is not None else 0
        if tool_error_msg:
            if manifest is not None and applied > 0:
                msg = f"{tool_error_msg} Recovered from manifest: {manifest}. Applied channels: {applied}."
                settings.status = msg
                self.report({"WARNING"}, msg)
                return {"FINISHED"}
            msg = tool_error_msg
            if manifest is None:
                msg += " Manifest not found."
            else:
                msg += f" Manifest found ({manifest}) but nothing applied."
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        msg = f"Manual texture correction complete. Applied channels: {applied}. Output: {out_dir}"
        settings.status = msg
        self.report({"INFO"}, msg)
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


class WARNO_OT_ApplyTextures(Operator):
    bl_idname = "warno.apply_textures"
    bl_label = "Apply/Reapply Textures"
    bl_description = "Resolve and reapply textures to the last imported or selected WARNO model without reimporting geometry"

    def execute(self, context):
        settings = context.scene.warno_import
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
            if settings.use_zz_dat_source:
                runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
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
                infer_names, material_role_by_id = extractor_mod.infer_material_names(model, mirror_y=True)
                material_ids = sorted({int(part["material"]) for part in model["parts"]})
                if settings.auto_name_materials:
                    material_name_by_id = dict(infer_names)
                else:
                    material_name_by_id = {int(mid): f"Material_{int(mid):03d}" for mid in material_ids}

                model_dir = _cache_asset_dir(extractor_mod, settings, asset_real)
                model_dir.mkdir(parents=True, exist_ok=True)
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
                _apply_material_nodes(mat, maps, role, bool(settings.use_ao_multiply))
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
        settings.status = msg
        self.report({"INFO"}, msg)
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
        runtime_info: Dict[str, Any] = {}
        if settings.use_zz_dat_source:
            try:
                runtime_info = _prepare_zz_runtime_sources(extractor_mod, settings)
            except Exception as exc:
                msg = f"ZZ runtime prepare failed: {exc}"
                settings.status = msg
                self.report({"ERROR"}, msg)
                return {"CANCELLED"}
        mesh_spk_paths = _resolve_mesh_spk_paths(project_root, settings, runtime_info)
        if not mesh_spk_paths:
            msg = "No mesh SPK files found. Set Mesh SPK/Folder or prepare ZZ runtime."
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        pick = _pick_best_asset_spk_path(extractor_mod, mesh_spk_paths, asset)
        if pick is None:
            msg = f"Asset not found in selected SPK sources: {asset}"
            settings.status = msg
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        mesh_spk_path, asset_hint = pick

        need_bone_map = bool(settings.auto_split_main_parts or settings.auto_split_wheels or settings.auto_pull_bones)
        rot = extractor_mod.build_rotation_params(
            float(settings.rotate_x),
            float(settings.rotate_y),
            float(settings.rotate_z),
            mirror_y=True,
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

                infer_names, material_role_by_id = extractor_mod.infer_material_names(
                    model,
                    mirror_y=True,
                )
                material_ids = sorted({int(part["material"]) for part in model["parts"]})
                if settings.auto_name_materials:
                    material_name_by_id = dict(infer_names)
                else:
                    material_name_by_id = {int(mid): f"Material_{int(mid):03d}" for mid in material_ids}

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
                    runtime_info=runtime_info,
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
        used_object_names: set[str] = set()
        armature_obj: bpy.types.Object | None = None

        for bucket_i, bucket in enumerate(buckets, start=1):
            group_name = str(bucket.get("group_name", f"Part_{bucket_i:03d}"))
            if settings.auto_name_parts:
                pretty_name = group_name
                if hasattr(extractor_mod, "pretty_part_name"):
                    try:
                        pretty_name = str(extractor_mod.pretty_part_name(group_name) or group_name)
                    except Exception:
                        pretty_name = group_name
                base_name = _safe_name(pretty_name, f"Part_{bucket_i:03d}")[:63]
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

        if settings.auto_pull_bones:
            try:
                armature_obj = _build_helper_armature(imported_objects, bone_payload, collection)
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
        src.prop(s, "use_zz_dat_source")
        if s.use_zz_dat_source:
            src.prop(s, "warno_root")
            src.prop(s, "modding_suite_root")
            src.prop(s, "zz_runtime_dir")
            src.operator("warno.prepare_zz_runtime", text="Prepare ZZ Runtime", icon="FILE_REFRESH")
        src.prop(s, "spk_path")
        src.prop(s, "skeleton_spk")
        src.prop(s, "cache_dir")

        tex = layout.box()
        tex.label(text="Textures")
        tex.prop(s, "auto_textures")
        tex.prop(s, "atlas_assets_dir")
        tex.prop(s, "tgv_converter")
        tex.prop(s, "auto_install_tgv_deps")
        tex.prop(s, "tgv_deps_dir")
        tex.prop(s, "manual_texture_tool")
        tex.prop(s, "texture_subdir")
        tex.prop(s, "tgv_split_mode")
        tex.prop(s, "tgv_aggressive_split")
        tex.prop(s, "auto_rename_textures")
        tex.prop(s, "use_ao_multiply")
        tex.operator("warno.install_tgv_deps", text="Install/Check TGV deps", icon="CONSOLE")
        tex.operator("warno.open_texture_folder", text="Open Texture Folder", icon="FILE_FOLDER")
        tex.operator("warno.clear_texture_folder", text="Clear texture folder from old files", icon="TRASH")
        tex.operator("warno.manual_texture_correcting", text="Manual texture correcting", icon="PREFERENCES")

        qry = layout.box()
        qry.label(text="Asset Picker")
        qry.prop(s, "query")
        qry.prop(s, "match_limit")
        row = qry.row(align=True)
        row.operator("warno.scan_assets", text="Scan Assets")
        row.operator("warno.scan_assets_all", text="Scan ALL Assets (takes a long time)", icon="TIME")
        qry.operator("warno.pick_asset_browser", text="Pick Asset Browser", icon="FILEBROWSER")

        main_row = qry.row(align=True)
        lod_icon = "TRIA_DOWN" if bool(s.show_asset_lods) else "TRIA_RIGHT"
        main_row.prop(s, "selected_asset_group", text="Main")
        main_row.prop(s, "show_asset_lods", text="", icon=lod_icon, emboss=False)
        if s.show_asset_lods:
            qry.prop(s, "selected_asset_lod", text="LOD")
        qry.label(text=f"Current: {str(s.selected_asset or '').strip()}", icon="OBJECT_DATA")

        opts = layout.box()
        opts.label(text="Import Options")
        opts.prop(s, "auto_split_main_parts")
        opts.prop(s, "auto_split_wheels")
        opts.prop(s, "auto_name_parts")
        opts.prop(s, "auto_pull_bones")
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
        opts.prop(s, "auto_name_materials")
        opts.prop(s, "rotate_x")
        opts.prop(s, "rotate_y")
        opts.prop(s, "rotate_z")

        geo = layout.box()
        geo.label(text="Geometry Cleanup")
        geo.prop(s, "use_merge_by_distance")
        geo.prop(s, "merge_distance")
        geo.prop(s, "auto_smooth_angle")
        geo.operator("warno.apply_auto_smooth", text="Apply Auto smooth to the model", icon="MOD_SMOOTH")
        layout.operator("warno.apply_textures", text="Apply/Reapply Textures", icon="SHADING_TEXTURE")

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
    WARNO_OT_ScanAssetsAll,
    WARNO_OT_PickAssetBrowser,
    WARNO_OT_PickAssetPopup,
    WARNO_OT_PrepareZZRuntime,
    WARNO_OT_InstallTGVDeps,
    WARNO_OT_OpenTextureFolder,
    WARNO_OT_ClearTextureFolder,
    WARNO_OT_ManualTextureCorrecting,
    WARNO_OT_ApplyAutoSmooth,
    WARNO_OT_ApplyTrackCorrection,
    WARNO_OT_ApplyTextures,
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
