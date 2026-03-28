#!/usr/bin/env python3
"""
WARNO GFX JSON export wrapper (strict headless mode).

This wrapper runs the dedicated moddingSuite.GfxCli executable against
compiled Output/AllPlatforms/NDF/GFX/*.ndfbin files.
"""
from __future__ import annotations

import argparse
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
    return path / "gfx_manifest.json"


def _tail_text(text: str, max_chars: int = 1600) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def _validate_schema(data: Dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "root is not object"
    schema_version = int(data.get("schema_version", 0) or 0)
    if schema_version not in {1, 2}:
        return False, f"unsupported schema_version={data.get('schema_version')}"
    for key in (
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
        if key not in data:
            return False, f"missing {key}"
    if not isinstance(data.get("source_files"), list):
        return False, "source_files must be list"
    if not isinstance(data.get("matched_units"), list):
        return False, "matched_units must be list"
    if not isinstance(data.get("reference_meshes"), list):
        return False, "reference_meshes must be list"
    if not isinstance(data.get("operators"), list):
        return False, "operators must be list"
    if not isinstance(data.get("turrets"), list):
        return False, "turrets must be list"
    if not isinstance(data.get("weapon_fx_anchors"), list):
        return False, "weapon_fx_anchors must be list"
    if not isinstance(data.get("subdepictions"), list):
        return False, "subdepictions must be list"
    if schema_version >= 2:
        if not isinstance(data.get("semantic_nodes", []), list):
            return False, "semantic_nodes must be list"
        if not isinstance(data.get("transform_debug", []), list):
            return False, "transform_debug must be list"
    return True, ""


def _load_and_validate(path: Path, asset_path: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise RuntimeError(f"GFX export JSON is invalid (root type): {path}")
    data["schema_version"] = int(data.get("schema_version", 1) or 1)
    data["asset_path"] = str(data.get("asset_path", asset_path)).strip() or asset_path
    ok, msg = _validate_schema(data)
    if not ok:
        raise RuntimeError(f"GFX export JSON schema invalid: {msg}")
    if data["schema_version"] >= 2:
        data["semantic_nodes"] = [
            row for row in (data.get("semantic_nodes", []) or []) if isinstance(row, dict)
        ]
        data["transform_debug"] = [
            row for row in (data.get("transform_debug", []) or []) if isinstance(row, dict)
        ]
    return data


def _resolve_cli_exe(modding_suite_root: Path, gfx_cli_override: str) -> tuple[Path | None, List[Path]]:
    script_root = Path(__file__).resolve().parent
    sibling_modding_suite = script_root.parent / "moddingSuite"
    override = Path(gfx_cli_override).expanduser() if gfx_cli_override.strip() else None
    candidates: List[Path] = [
        script_root / "moddingSuite" / "gfx_cli" / "moddingSuite.GfxCli.exe",
        sibling_modding_suite / "gfx_cli" / "moddingSuite.GfxCli.exe",
        modding_suite_root / "gfx_cli" / "moddingSuite.GfxCli.exe",
        sibling_modding_suite / "moddingSuite.GfxCli" / "bin" / "Release" / "net9.0-windows10.0.19041" / "moddingSuite.GfxCli.exe",
        sibling_modding_suite / "moddingSuite.GfxCli" / "bin" / "Debug" / "net9.0-windows10.0.19041" / "moddingSuite.GfxCli.exe",
        modding_suite_root / "moddingSuite.GfxCli" / "bin" / "Release" / "net9.0-windows10.0.19041" / "moddingSuite.GfxCli.exe",
        modding_suite_root / "moddingSuite.GfxCli" / "bin" / "Debug" / "net9.0-windows10.0.19041" / "moddingSuite.GfxCli.exe",
    ]
    if override is not None:
        candidates.append(override if override.is_absolute() else (script_root / override))

    seen: set[str] = set()
    uniq: List[Path] = []
    for p in candidates:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    for p in uniq:
        if p.exists() and p.is_file():
            return p, uniq
    return None, uniq


def _resolve_cli_project(modding_suite_root: Path) -> Path | None:
    script_root = Path(__file__).resolve().parent
    for root in (modding_suite_root, script_root.parent / "moddingSuite"):
        project = root / "moddingSuite.GfxCli" / "moddingSuite.GfxCli.csproj"
        if project.exists() and project.is_file():
            return project
    return None


def _build_cli_exec_cmd(cli_exe: Path) -> List[str]:
    exe = Path(cli_exe)
    runtimeconfig = exe.with_suffix(".runtimeconfig.json")
    # Framework-dependent DLLs need the dotnet host.
    # Framework-dependent EXEs already include the apphost and must be launched directly,
    # otherwise dotnet sees the EXE and sibling DLL as duplicate assemblies.
    if exe.suffix.lower() == ".dll" and runtimeconfig.exists():
        return ["dotnet", str(exe)]
    return [str(exe)]


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
    ap = argparse.ArgumentParser(description="Export WARNO GFX semantic manifest via headless Gfx CLI")
    ap.add_argument("--warno-root", required=True)
    ap.add_argument("--modding-suite-root", required=True)
    ap.add_argument("--asset-path", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gfx-cli", default="", help="Optional explicit path to moddingSuite.GfxCli.exe")
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    warno_root = Path(args.warno_root)
    modding_suite_root = Path(args.modding_suite_root)
    asset_path = _norm_asset(args.asset_path)
    out_json = _resolve_output(Path(args.out_json))
    cache_dir = Path(args.cache_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not warno_root.exists() or not warno_root.is_dir():
        print(f"GFX export failed: WARNO root not found: {warno_root}", file=sys.stderr)
        return 3

    cli_exe, tried = _resolve_cli_exe(modding_suite_root=modding_suite_root, gfx_cli_override=str(args.gfx_cli or ""))
    project = _resolve_cli_project(modding_suite_root)

    cmd: List[str]
    if cli_exe is not None:
        cmd = [
            *_build_cli_exec_cmd(cli_exe),
            "--warno-root",
            str(warno_root),
            "--asset-path",
            asset_path,
            "--out-json",
            str(out_json),
            "--cache-dir",
            str(cache_dir),
        ]
    elif project is not None:
        cmd = [
            "dotnet",
            "run",
            "--project",
            str(project),
            "--configuration",
            "Release",
            "--",
            "--warno-root",
            str(warno_root),
            "--asset-path",
            asset_path,
            "--out-json",
            str(out_json),
            "--cache-dir",
            str(cache_dir),
        ]
    else:
        tried_text = ", ".join(str(p) for p in tried)
        print(
            "GFX export failed: headless Gfx CLI was not found. "
            f"Tried: {tried_text}",
            file=sys.stderr,
        )
        return 3

    if bool(args.verbose):
        cmd.append("--verbose")

    quoted = " ".join(shlex.quote(x) for x in cmd)
    print(f"[gfx-wrapper] cmd: {quoted}", file=sys.stderr)

    rc, out, err, elapsed, timed_out = _run_cli(cmd, timeout_sec=max(5, int(args.timeout_sec or 45)))
    print(f"[gfx-wrapper] elapsed: {elapsed:.2f}s", file=sys.stderr)
    print(f"[gfx-wrapper] exit_code: {rc}", file=sys.stderr)
    out_tail = _tail_text(out)
    err_tail = _tail_text(err)
    if out_tail:
        print(f"[gfx-wrapper] stdout_tail: {out_tail}", file=sys.stderr)
    if err_tail:
        print(f"[gfx-wrapper] stderr_tail: {err_tail}", file=sys.stderr)

    if timed_out:
        print(f"gfx_cli_timeout: exceeded {int(args.timeout_sec or 45)}s", file=sys.stderr)
        return 3
    if rc != 0:
        print("GFX export failed: headless Gfx CLI returned non-zero exit code.", file=sys.stderr)
        return 3
    if not out_json.exists() or not out_json.is_file():
        print(f"GFX export failed: output JSON missing: {out_json}", file=sys.stderr)
        return 3

    try:
        data = _load_and_validate(out_json, asset_path=asset_path)
    except Exception as exc:
        print(f"GFX export validation failed: {exc}", file=sys.stderr)
        return 3

    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[gfx-wrapper] ok: asset={asset_path} matched_units={len(data.get('matched_units', []))}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
