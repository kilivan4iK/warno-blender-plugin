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


ATLAS_FALLBACK_MODES = ("ambient", "none")


def atlas_rel_for_asset(asset_path: str) -> str:
    """The atlas the CLI reads for an asset: PC/Atlas/<asset folder>/TextureSmall.atlas,
    under --cache-dir first, then (fallback "ambient", the default) under
    Mods/ModData/base and Output. "--atlas-file PATH" names the file outright."""
    parts = [p for p in _norm_asset(asset_path).split("/") if p]
    if len(parts) < 2:
        return ""
    return "PC/Atlas/" + "/".join(parts[:-1]) + "/TextureSmall.atlas"


def _load_extractor_module(script_root: Path):
    extractor_path = script_root / "warno_spk_extract.py"
    if not extractor_path.exists() or not extractor_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("warno_spk_extract_runtime", str(extractor_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(spec.name, None)
        raise


def _ensure_atlas_for_asset_from_zz(*, warno_root: Path, lookup_cache_dir: Path, asset_path: str, game: str) -> str:
    """Standalone use only: extract the asset's TextureSmall.atlas from the game packs
    into <lookup_cache_dir>/PC/Atlas/... (checksum-validated, so a patched atlas
    replaces the old copy). The Blender add-on does this in-process instead."""
    rel = atlas_rel_for_asset(asset_path)
    if not rel:
        return "no atlas path for asset"
    extractor = _load_extractor_module(Path(__file__).resolve().parent)
    if extractor is None:
        return "warno_spk_extract.py not found"
    try:
        resolver = extractor.get_zz_runtime_resolver(Path(warno_root), game=game)
        out = resolver.extract_asset_to_runtime(rel, Path(lookup_cache_dir), exact_only=True)
    except Exception as exc:
        return f"zz extract failed: {exc}"
    return f"extracted {out}" if out is not None else f"{rel} is not in the game packs"


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
    candidates: List[Path] = []
    # An explicit path from the settings wins; it used to be tried LAST, so a
    # self-built CLI was silently ignored whenever a bundled one existed.
    if atlas_cli_override.strip():
        override = Path(atlas_cli_override).expanduser()
        candidates.append(override if override.is_absolute() else (script_root / override))
    candidates += [
        # Unified moddingSuite.exe (the subcommand routes the call).
        script_root / "moddingSuite" / "moddingSuite.exe",
        modding_suite_root / "moddingSuite.exe",
        # Dedicated AtlasCli.exe from older moddingSuite builds.
        script_root / "moddingSuite" / "atlas_cli" / "moddingSuite.AtlasCli.exe",
        script_root / "moddingSuite" / "moddingSuite.AtlasCli.exe",
        modding_suite_root / "atlas_cli" / "moddingSuite.AtlasCli.exe",
        modding_suite_root / "moddingSuite.AtlasCli.exe",
    ]

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


def _build_cli_cmd(cli_exe: Path, base_args: List[str]) -> List[str]:
    """Construct argv for the resolved CLI. Unified moddingSuite.exe needs the
    'atlas' subcommand prepended; legacy AtlasCli.exe is invoked directly."""
    cmd = [str(cli_exe)]
    if cli_exe.name.lower() == "moddingsuite.exe":
        cmd.append("atlas")
    cmd.extend(base_args)
    return cmd


def _run_cli(cmd: List[str], timeout_sec: int) -> tuple[int, str, str, float, bool]:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5, int(timeout_sec)),
            # No console window flashing up over Blender for the dotnet host.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        elapsed = time.monotonic() - t0
        return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or ""), elapsed, False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return 124, out, err, elapsed, True
    except OSError as exc:
        return 127, "", f"cannot start {cmd[0]}: {exc}", time.monotonic() - t0, False


def _build_dotnet_fallback_cmd(cli_exe: Path) -> List[str] | None:
    exe = Path(cli_exe)
    dll = exe.with_suffix(".dll")
    runtimeconfig = dll.with_suffix(".runtimeconfig.json")
    if dll.exists() and dll.is_file() and runtimeconfig.exists() and runtimeconfig.is_file():
        return ["dotnet", str(dll)]
    return None


def export_atlas_json(
    *,
    warno_root: Path,
    modding_suite_root: Path,
    asset_path: str,
    out_json: Path,
    lookup_cache_dir: Path,
    atlas_cli_override: str = "",
    timeout_sec: int = 45,
    verbose: bool = False,
    atlas_file: "str | Path | None" = None,
    fallback: "str | None" = None,
) -> tuple[int, str]:
    """Run the Atlas CLI for one asset and validate its JSON.

    The atlas must already sit at <lookup_cache_dir>/PC/Atlas/<asset folder>/TextureSmall.atlas,
    unless atlas_file names it ("--atlas-file PATH": exactly that file, nothing else is
    searched). fallback "none" ("--fallback none") stops the CLI from falling back to the
    Mods/ModData/base and Output copies, which keep pre-patch atlases; "ambient" is the
    CLI's default. Both are passed only when set, as "--key value" pairs, which CLIs
    older than the moddingSuite branch blender-plugin-interop skip.
    Returns (exit code, log text): 0 ok, 2 the atlas has no entries for the asset,
    3 hard failure, 4 the CLI wrote an invalid JSON. Called in-process by the add-on,
    which spares a Python start-up and a full re-index of the game packs per asset.
    """
    log: List[str] = []
    asset_path = _norm_asset(asset_path)
    out_json = _resolve_output(Path(out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    timeout_sec = max(5, int(timeout_sec or 45))

    if not Path(warno_root).is_dir():
        return 3, f"Atlas export failed: WARNO root not found: {warno_root}"
    fallback_mode = str(fallback or "").strip().lower()
    if fallback_mode and fallback_mode not in ATLAS_FALLBACK_MODES:
        return 3, f"Atlas export failed: invalid fallback {fallback!r} (expected one of {', '.join(ATLAS_FALLBACK_MODES)})"

    cli_exe, tried = _resolve_cli_exe(modding_suite_root=Path(modding_suite_root), atlas_cli_override=str(atlas_cli_override or ""))
    if cli_exe is None:
        return 3, "Atlas export failed: headless Atlas CLI was not found. Tried: " + ", ".join(str(p) for p in tried)

    base_args = [
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
    if atlas_file is not None and str(atlas_file).strip():
        base_args += ["--atlas-file", str(atlas_file)]
    if fallback_mode:
        base_args += ["--fallback", fallback_mode]
    if verbose:
        base_args.append("--verbose")

    def _attempt(cmd: List[str], tag: str) -> tuple[int, bool]:
        log.append(f"[atlas] {tag}cmd: " + " ".join(shlex.quote(x) for x in cmd))
        rc, out, err, elapsed, timed_out = _run_cli(cmd, timeout_sec=timeout_sec)
        log.append(f"[atlas] {tag}exit_code={rc} elapsed={elapsed:.2f}s")
        if _tail_text(out):
            log.append(f"[atlas] {tag}stdout: {_tail_text(out)}")
        if _tail_text(err):
            log.append(f"[atlas] {tag}stderr: {_tail_text(err)}")
        return rc, timed_out

    rc, timed_out = _attempt(_build_cli_cmd(cli_exe, base_args), "")
    # Some machines crash launching the apphost EXE directly (CLR assert) but work
    # through the dotnet host + DLL; retry that once.
    if not timed_out and rc not in {0, 2}:
        dotnet_exec = _build_dotnet_fallback_cmd(cli_exe)
        if dotnet_exec is not None:
            retry = list(dotnet_exec)
            if cli_exe.name.lower() == "moddingsuite.exe":
                retry.append("atlas")
            rc, timed_out = _attempt(retry + base_args, "retry_")

    if timed_out:
        log.append(f"atlas_cli_timeout: exceeded {timeout_sec}s")
        return 3, "\n".join(log)
    if rc == 2:
        log.append(f"atlas_cli_no_entries: atlas data has no entries for asset {asset_path}")
        return 2, "\n".join(log)
    if rc != 0:
        log.append("Atlas export failed: headless Atlas CLI returned non-zero exit code.")
        return 3, "\n".join(log)
    if not out_json.is_file():
        log.append(f"Atlas export failed: output JSON missing: {out_json}")
        return 3, "\n".join(log)
    try:
        data = _load_and_validate(out_json, asset_path=asset_path, atlas_source="")
    except Exception as exc:
        log.append(str(exc))
        return 4, "\n".join(log)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0, "\n".join(log)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export WARNO Atlas crop/name map via headless Atlas CLI")
    ap.add_argument("--warno-root", required=True)
    ap.add_argument("--modding-suite-root", required=True)
    ap.add_argument("--asset-path", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument(
        "--cache-dir",
        required=True,
        help="Folder holding PC/Atlas/<asset folder>/TextureSmall.atlas (extracted from the game packs if missing)",
    )
    ap.add_argument("--atlas-cli", default="", help="Optional explicit path to moddingSuite.exe / moddingSuite.AtlasCli.exe")
    ap.add_argument(
        "--atlas-file",
        default="",
        help="Read exactly this TextureSmall.atlas (needs a CLI with --atlas-file; older ones ignore it)",
    )
    ap.add_argument(
        "--fallback",
        choices=ATLAS_FALLBACK_MODES,
        default=None,
        help="none: only <cache-dir>/PC/Atlas/<asset folder>/TextureSmall.atlas, no Mods/Output copies "
        "(CLI default: ambient)",
    )
    ap.add_argument("--timeout-sec", type=int, default=45)
    ap.add_argument("--game", default="WARNO", help="Active Eugen game id (WARNO|WARGAME_RD|STEEL_DIVISION_2)")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    note = _ensure_atlas_for_asset_from_zz(
        warno_root=Path(args.warno_root),
        lookup_cache_dir=cache_dir,
        asset_path=args.asset_path,
        game=str(args.game or "WARNO"),
    )
    print(f"[atlas-wrapper] {note}", file=sys.stderr)
    rc, log = export_atlas_json(
        warno_root=Path(args.warno_root),
        modding_suite_root=Path(args.modding_suite_root),
        asset_path=args.asset_path,
        out_json=Path(args.out_json),
        lookup_cache_dir=cache_dir,
        atlas_cli_override=str(args.atlas_cli or ""),
        timeout_sec=int(args.timeout_sec or 45),
        verbose=bool(args.verbose),
        atlas_file=str(args.atlas_file or "").strip() or None,
        fallback=args.fallback,
    )
    if log:
        print(log, file=sys.stderr)
    if rc == 0:
        print(f"[OK] Atlas JSON exported: {_resolve_output(Path(args.out_json))}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
