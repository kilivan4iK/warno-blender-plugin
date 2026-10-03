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
    candidates: List[Path] = []
    # An explicit path from the settings wins (it used to be tried last).
    if gfx_cli_override.strip():
        override = Path(gfx_cli_override).expanduser()
        candidates.append(override if override.is_absolute() else (script_root / override))
    candidates += [
        # Unified single exe (the subcommand routes the call).
        script_root / "moddingSuite" / "moddingSuite.exe",
        sibling_modding_suite / "moddingSuite.exe",
        modding_suite_root / "moddingSuite.exe",
        # Dedicated GfxCli.exe from older moddingSuite builds.
        script_root / "moddingSuite" / "gfx_cli" / "moddingSuite.GfxCli.exe",
        script_root / "moddingSuite" / "moddingSuite.GfxCli.exe",
        sibling_modding_suite / "gfx_cli" / "moddingSuite.GfxCli.exe",
        modding_suite_root / "gfx_cli" / "moddingSuite.GfxCli.exe",
        modding_suite_root / "moddingSuite.GfxCli.exe",
    ]

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


def _build_cli_exec_cmd(cli_exe: Path) -> List[str]:
    exe = Path(cli_exe)
    runtimeconfig = exe.with_suffix(".runtimeconfig.json")
    # Framework-dependent DLLs need the dotnet host.
    # Framework-dependent EXEs already include the apphost and must be launched directly,
    # otherwise dotnet sees the EXE and sibling DLL as duplicate assemblies.
    if exe.suffix.lower() == ".dll" and runtimeconfig.exists():
        base = ["dotnet", str(exe)]
    else:
        base = [str(exe)]
    # Unified moddingSuite[.exe|.dll] dispatches by subcommand — first arg.
    name_stem = exe.stem.lower()
    if name_stem == "moddingsuite":
        base.append("gfx")
    return base


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


def export_gfx_json(
    *,
    warno_root: Path,
    modding_suite_root: Path,
    asset_path: str,
    out_json: Path,
    cache_dir: Path,
    gfx_cli_override: str = "",
    timeout_sec: int = 180,
    verbose: bool = False,
) -> tuple[int, str]:
    """Run the GFX CLI for one asset and validate its manifest. Returns (exit code, log)."""
    log: List[str] = []
    asset_path = _norm_asset(asset_path)
    out_json = _resolve_output(Path(out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if not Path(warno_root).is_dir():
        return 3, f"GFX export failed: WARNO root not found: {warno_root}"

    cli_exe, tried = _resolve_cli_exe(modding_suite_root=Path(modding_suite_root), gfx_cli_override=str(gfx_cli_override or ""))
    if cli_exe is None:
        return 3, "GFX export failed: headless Gfx CLI was not found. Tried: " + ", ".join(str(p) for p in tried)
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
    if verbose:
        cmd.append("--verbose")
    log.append("[gfx] cmd: " + " ".join(shlex.quote(x) for x in cmd))
    rc, out, err, elapsed, timed_out = _run_cli(cmd, timeout_sec=max(5, int(timeout_sec or 180)))
    log.append(f"[gfx] exit_code={rc} elapsed={elapsed:.2f}s")
    if _tail_text(out):
        log.append(f"[gfx] stdout: {_tail_text(out)}")
    if _tail_text(err):
        log.append(f"[gfx] stderr: {_tail_text(err)}")
    if timed_out:
        log.append(f"gfx_cli_timeout: exceeded {int(timeout_sec or 180)}s")
        return 3, "\n".join(log)
    if rc != 0:
        log.append("GFX export failed: headless Gfx CLI returned non-zero exit code.")
        return 3, "\n".join(log)
    if not out_json.is_file():
        log.append(f"GFX export failed: output JSON missing: {out_json}")
        return 3, "\n".join(log)
    try:
        data = _load_and_validate(out_json, asset_path=asset_path)
    except Exception as exc:
        log.append(f"GFX export validation failed: {exc}")
        return 3, "\n".join(log)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.append(f"[gfx] ok: asset={asset_path} matched_units={len(data.get('matched_units', []))}")
    return 0, "\n".join(log)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export WARNO GFX semantic manifest via headless Gfx CLI")
    ap.add_argument("--warno-root", required=True)
    ap.add_argument("--modding-suite-root", required=True)
    ap.add_argument("--asset-path", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gfx-cli", default="", help="Optional explicit path to moddingSuite.GfxCli.exe")
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument(
        "--game",
        default="WARNO",
        help="Active Eugen game id (WARNO|WARGAME_RD|STEEL_DIVISION_2); informational, passed via --warno-root",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    rc, log = export_gfx_json(
        warno_root=Path(args.warno_root),
        modding_suite_root=Path(args.modding_suite_root),
        asset_path=args.asset_path,
        out_json=Path(args.out_json),
        cache_dir=Path(args.cache_dir),
        gfx_cli_override=str(args.gfx_cli or ""),
        timeout_sec=int(args.timeout_sec or 180),
        verbose=bool(args.verbose),
    )
    if log:
        print(log, file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
