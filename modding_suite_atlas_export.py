#!/usr/bin/env python3
"""
WARNO Atlas JSON export wrapper (strict headless mode).

This wrapper only runs the dedicated moddingSuite.AtlasCli executable.
No GUI fallback is allowed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _norm_asset(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    raw = raw.lstrip("/")
    while "//" in raw:
        raw = raw.replace("//", "/")
    return raw


def _resolve_output(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    return path / "atlas_map.json"


def _has_atlas_tree(root: Path) -> bool:
    try:
        return (root / "PC" / "Atlas").exists()
    except Exception:
        return False


def _resolve_lookup_cache_dir(requested_cache_dir: Path) -> Path:
    """
    AtlasCli uses --cache-dir only for Atlas lookup (expects <cache>/PC/Atlas/...).
    In strict ZZ runtime flow atlas_json_cache usually does not contain atlas binaries,
    so we auto-point lookup to prepared ZZ runtime when available.
    """
    script_root = Path(__file__).resolve().parent
    candidates: List[Path] = []

    req = requested_cache_dir
    candidates.append(req)
    candidates.append(script_root / "out_blender_runtime" / "zz_runtime")
    candidates.append(script_root / "output_blender" / "_zz_runtime")
    candidates.append(script_root / "output_blender")

    seen: set[str] = set()
    uniq: List[Path] = []
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    for c in uniq:
        if _has_atlas_tree(c):
            return c
    return req


def _load_extractor_module(script_root: Path):
    extractor_path = script_root / "warno_spk_extract.py"
    if not extractor_path.exists() or not extractor_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("warno_spk_extract_runtime", str(extractor_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atlas_rel_candidates_for_asset(asset_path: str) -> List[str]:
    norm = _norm_asset(asset_path)
    parts = [p for p in norm.split("/") if p]
    if len(parts) < 2:
        return []
    dir_parts = parts[:-1]  # drop .fbx
    # Try exact dir first, then parents up to Assets.
    rels: List[str] = []
    for i in range(len(dir_parts), 0, -1):
        cur = "/".join(dir_parts[:i]).strip("/")
        if not cur:
            continue
        if not cur.lower().startswith("assets/"):
            continue
        rels.append(f"PC/Atlas/{cur}/TextureSmall.atlas")
        if cur.lower() == "assets":
            break
    # De-dup preserve order.
    out: List[str] = []
    seen: set[str] = set()
    for r in rels:
        k = r.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _ensure_atlas_for_asset_from_zz(
    *,
    warno_root: Path,
    lookup_cache_dir: Path,
    asset_path: str,
    verbose: bool,
) -> bool:
    rel_candidates = _atlas_rel_candidates_for_asset(asset_path)
    if not rel_candidates:
        return False

    # Fast path: already present.
    for rel in rel_candidates:
        p = lookup_cache_dir / Path(*rel.split("/"))
        if p.exists() and p.is_file():
            return True

    script_root = Path(__file__).resolve().parent
    extractor = _load_extractor_module(script_root)
    if extractor is None:
        return False

    try:
        resolver = extractor.get_zz_runtime_resolver(Path(warno_root))
    except Exception:
        return False

    extracted = False
    for rel in rel_candidates:
        try:
            out = resolver.extract_asset_to_runtime(rel, Path(lookup_cache_dir), exact_only=True)
            if out is None:
                out = resolver.extract_asset_to_runtime(rel, Path(lookup_cache_dir), exact_only=False)
            if out is not None and Path(out).exists():
                extracted = True
                if verbose:
                    print(f"[atlas-wrapper] extracted atlas from zz: {out}", file=sys.stderr)
                break
        except Exception:
            continue

    # Confirm file exists where AtlasCli expects it.
    if extracted:
        for rel in rel_candidates:
            p = lookup_cache_dir / Path(*rel.split("/"))
            if p.exists() and p.is_file():
                return True
    return False


def _tail_text(text: str, max_chars: int = 1600) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def _validate_v1_schema(data: Dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "root is not object"
    if int(data.get("schema_version", 0) or 0) != 1:
        return False, f"unsupported schema_version={data.get('schema_version')}"
    if not isinstance(data.get("textures"), list):
        return False, "missing textures[]"

    included = data.get("included_asset_paths")
    if included is not None:
        if not isinstance(included, list):
            return False, "included_asset_paths must be list when present"
        for i, it in enumerate(included):
            if not str(it or "").strip():
                return False, f"included_asset_paths[{i}] is empty"

    textures = data.get("textures") or []
    for i, tex in enumerate(textures):
        if not isinstance(tex, dict):
            return False, f"textures[{i}] is not object"
        if not str(tex.get("source_tgv_rel", "")).strip():
            return False, f"textures[{i}].source_tgv_rel missing"
        src_asset = tex.get("source_asset_path")
        if src_asset is not None and not str(src_asset or "").strip():
            return False, f"textures[{i}].source_asset_path empty"
        rect = tex.get("crop_rect_px")
        if not isinstance(rect, dict):
            return False, f"textures[{i}].crop_rect_px missing"
        for key in ("x", "y", "w", "h"):
            if key not in rect:
                return False, f"textures[{i}].crop_rect_px.{key} missing"
        if not isinstance(tex.get("targets"), list) or not tex.get("targets"):
            return False, f"textures[{i}].targets missing"
    return True, ""


def _load_and_validate(path: Path, asset_path: str, atlas_source: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Atlas export JSON is invalid (root type): {path}")

    # Keep exporter metadata but enforce required root fields.
    data["schema_version"] = int(data.get("schema_version", 1) or 1)
    data["asset_path"] = str(data.get("asset_path", asset_path)).strip() or asset_path
    data["atlas_source"] = str(data.get("atlas_source", atlas_source)).strip() or atlas_source

    ok, msg = _validate_v1_schema(data)
    if not ok:
        raise RuntimeError(f"Atlas export JSON schema invalid: {msg}")
    return data


def _resolve_cli_exe(modding_suite_root: Path, atlas_cli_override: str) -> tuple[Path | None, List[Path]]:
    script_root = Path(__file__).resolve().parent
    primary = script_root / "moddingSuite" / "atlas_cli" / "moddingSuite.AtlasCli.exe"
    secondary = modding_suite_root / "atlas_cli" / "moddingSuite.AtlasCli.exe"
    override = Path(atlas_cli_override).expanduser() if atlas_cli_override.strip() else None

    candidates: List[Path] = [primary, secondary]
    if override is not None:
        candidates.append(override if override.is_absolute() else (script_root / override))

    seen: set[str] = set()
    uniq: List[Path] = []
    for p in candidates:
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)

    for p in uniq:
        if p.exists() and p.is_file():
            return p, uniq
    return None, uniq


def _run_cli(cmd: List[str], timeout_sec: int) -> tuple[int, str, str, float, bool]:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_sec)),
        )
        elapsed = time.monotonic() - t0
        return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or ""), elapsed, False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        out = str(getattr(exc, "stdout", "") or "")
        err = str(getattr(exc, "stderr", "") or "")
        return 124, out, err, elapsed, True


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export WARNO Atlas crop/name map via headless Atlas CLI")
    ap.add_argument("--warno-root", required=True)
    ap.add_argument("--modding-suite-root", required=True)
    ap.add_argument("--asset-path", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--atlas-cli", default="", help="Optional explicit path to moddingSuite.AtlasCli.exe")
    ap.add_argument("--timeout-sec", type=int, default=45)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    warno_root = Path(args.warno_root)
    modding_suite_root = Path(args.modding_suite_root)
    asset_path = _norm_asset(args.asset_path)
    out_json = _resolve_output(Path(args.out_json))
    cache_dir = Path(args.cache_dir)
    lookup_cache_dir = _resolve_lookup_cache_dir(cache_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not warno_root.exists() or not warno_root.is_dir():
        print(f"Atlas export failed: WARNO root not found: {warno_root}", file=sys.stderr)
        return 3

    cli_exe, tried = _resolve_cli_exe(modding_suite_root=modding_suite_root, atlas_cli_override=str(args.atlas_cli or ""))
    if cli_exe is None:
        tried_text = ", ".join(str(p) for p in tried)
        print(
            "Atlas export failed: headless Atlas CLI was not found. "
            f"Tried: {tried_text}",
            file=sys.stderr,
        )
        return 3

    cmd = [
        str(cli_exe),
        "--warno-root",
        str(warno_root),
        "--asset-path",
        asset_path,
        "--out-json",
        str(out_json),
        "--cache-dir",
        str(lookup_cache_dir),
        "--include-sibling-assets",
    ]
    if bool(args.verbose):
        cmd.append("--verbose")

    quoted = " ".join(shlex.quote(x) for x in cmd)
    print(f"[atlas-wrapper] cmd: {quoted}", file=sys.stderr)
    if str(lookup_cache_dir) != str(cache_dir):
        print(f"[atlas-wrapper] cache_dir_for_lookup: {lookup_cache_dir}", file=sys.stderr)

    # On clean PCs atlas files may be missing outside ZZ.dat; pre-extract expected atlas on demand.
    _ensure_atlas_for_asset_from_zz(
        warno_root=warno_root,
        lookup_cache_dir=lookup_cache_dir,
        asset_path=asset_path,
        verbose=bool(args.verbose),
    )

    rc, out, err, elapsed, timed_out = _run_cli(cmd, timeout_sec=max(5, int(args.timeout_sec or 45)))
    print(f"[atlas-wrapper] elapsed: {elapsed:.2f}s", file=sys.stderr)
    print(f"[atlas-wrapper] exit_code: {rc}", file=sys.stderr)
    out_tail = _tail_text(out)
    err_tail = _tail_text(err)
    if out_tail:
        print(f"[atlas-wrapper] stdout_tail: {out_tail}", file=sys.stderr)
    if err_tail:
        print(f"[atlas-wrapper] stderr_tail: {err_tail}", file=sys.stderr)

    if timed_out:
        print(f"atlas_cli_timeout: exceeded {int(args.timeout_sec or 45)}s", file=sys.stderr)
        return 3

    # Preserve "no entries" contract.
    if rc == 2:
        return 2
    if rc != 0:
        print("Atlas export failed: headless Atlas CLI returned non-zero exit code.", file=sys.stderr)
        return 3

    if not out_json.exists() or not out_json.is_file():
        print(f"Atlas export failed: output JSON missing: {out_json}", file=sys.stderr)
        return 3

    try:
        data = _load_and_validate(out_json, asset_path=asset_path, atlas_source="")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 4

    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Atlas JSON exported: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
