"""Bridge between the in-game getter (tools/game_getter/warno_game_getter.py) and the
Blender importer: turn a captured POST-VS stream-out into the per-vertex track UV the
decoder substitutes (the same baked_track_uv/*.json format used for the hand-baked
merkava_3b, now produced automatically from any unit's in-game get).

The getter captures the game's POST-vertex-shader output (a stream-out replay), so its
TexCoord0 is the GAME-EXACT track UV — a single clean tile (U[0,1], V[-1,0]) — with none
of the file decoder's seam-collapse ambiguity. We read TexCoord0 out of the SO records,
map each record back to a file vertex through the captured input index buffer, and key the
bake by (vertex_count, position checksum) so the decoder's _match_baked_track picks it up.

Per-draw layout the getter writes into get_cache/<name>/manifest.json:
  draws[]: so_file (post-VS blob) + so_stride + so_entries[{name,semIdx,off,cnt}]
           ib_file (input index buffer) + isz + index_count + min_index
           vb_file (input vertex buffer, file stride-32, for XYZ ordering) + nverts + stride

Bake space = file-only space (U=U_rip, V=0.5+V_rip/2 -> [0,0.5]); the importer's V-flip +
atlas-crop then reproduce V_rip on the tread crop (identical to file-only / merkava_3b).
Legacy PRE-VS captures (vb TexCoord0 u16, no so_file) are still read as a fallback.
"""
import json
import math
import os
import struct
from pathlib import Path

TC_OFFSET = 24          # TexCoord0 byte offset in the legacy stride-32 (pre-VS) track vertex
UV_DIVISOR = 8192.0     # Eugen custom word-normalized scale (legacy pre-VS path only)
TRACK_STRIDE = 32

_DBG = bool(os.environ.get("WARNO_GET_DEBUG"))


def _dbg(msg):
    if _DBG:
        print(f"[get-bake] {msg}")


def _track_pos_checksum(xyz, num):
    """Identical to warno_iriszoom._track_pos_checksum (decode-xyz space, metres*1000)."""
    chk = 0
    for i in range(num * 3):
        chk = (chk + int(round(xyz[i] * 1000))) & 0xffffffff
    return chk


def decode_get_xyz(vb_bytes, stride, nverts):
    """Parse a captured input vertex buffer -> xyz_flat (buffer's own units; for ordering)."""
    xyz = []
    for i in range(nverts):
        o = i * stride
        if o + 12 > len(vb_bytes):
            break
        x, y, z = struct.unpack_from("<3f", vb_bytes, o)
        xyz.append(x); xyz.append(y); xyz.append(z)
    return xyz


def decode_get_draw(vb_bytes, stride, nverts, tc_off=TC_OFFSET, div=UV_DIVISOR):
    """LEGACY pre-VS path: xyz + TexCoord0 (u16/div) straight from the input VB."""
    xyz = []
    uv = []
    for i in range(nverts):
        o = i * stride
        if o + tc_off + 4 > len(vb_bytes):
            break
        x, y, z = struct.unpack_from("<3f", vb_bytes, o)
        u, v = struct.unpack_from("<2H", vb_bytes, o + tc_off)
        xyz.append(x); xyz.append(y); xyz.append(z)
        uv.append(u / div); uv.append(v / div)
    return xyz, uv


def decode_so_uv(draw):
    """POST-VS path: read TexCoord0 (2 floats) out of the SO records and map each record to
    a file vertex through the input index buffer. Returns a flat per-vertex uv list (2*nverts)
    or None. SO record k corresponds to draw index k -> file vertex IB[k]-min_index."""
    so = draw.get("so") or b""
    sost = int(draw.get("so_stride", 0))
    ents = draw.get("so_entries") or []
    ib = draw.get("ib") or b""
    isz = int(draw.get("isz", 2)) or 2
    ic = int(draw.get("index_count", 0))
    minidx = int(draw.get("min_index", 0))
    nverts = int(draw.get("nverts", 0))
    if not so or not ib or sost <= 0 or ic <= 0 or nverts <= 0:
        return None
    tc = next((e for e in ents if str(e.get("name")) == "TEXCOORD" and int(e.get("semIdx", -1)) == 0), None)
    if tc is None:
        _dbg("SO has no TEXCOORD0 entry")
        return None
    off = int(tc.get("off", 0))
    ifmt = "<H" if isz == 2 else "<I"
    uv = [0.0] * (2 * nverts)
    seen = 0
    covered = [False] * nverts
    for k in range(ic):
        rec = k * sost
        ipos = k * isz
        if rec + off + 8 > len(so) or ipos + isz > len(ib):
            break
        vidx = struct.unpack_from(ifmt, ib, ipos)[0] - minidx
        if not (0 <= vidx < nverts):
            continue
        u, v = struct.unpack_from("<2f", so, rec + off)
        uv[2 * vidx] = u
        uv[2 * vidx + 1] = v
        if not covered[vidx]:
            covered[vidx] = True
            seen += 1
    if seen < nverts * 0.5:
        _dbg(f"SO covered only {seen}/{nverts} verts")
        return None
    return uv


def _corr(a, b):
    n = min(len(a), len(b))
    if n < 4:
        return 0.0
    ma = sum(a[:n]) / n
    mb = sum(b[:n]) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da * db > 1e-9 else 0.0


def _order_matches(decode_xyz, get_xyz, nv):
    """The get's vertex order == the decode order. Verify by per-axis correlation
    (scale/offset-invariant) so we can transfer UV by index."""
    if len(get_xyz) < 3 * nv:
        return False
    dx = [decode_xyz[3 * i] for i in range(nv)]
    rx = [get_xyz[3 * i] for i in range(nv)]
    dy = [decode_xyz[3 * i + 1] for i in range(nv)]
    ry = [get_xyz[3 * i + 1] for i in range(nv)]
    dz = [decode_xyz[3 * i + 2] for i in range(nv)]
    rz = [get_xyz[3 * i + 2] for i in range(nv)]
    raxes = [rx, ry, rz]
    ok = 0
    for d in (dx, dy, dz):
        if max(abs(_corr(d, r)) for r in raxes) > 0.95:
            ok += 1
    return ok >= 3


def load_get_cache(cache_dir):
    """Yield captured draws from the getter's cache. Each: nverts, stride, vb, +(post-VS:
    so, so_stride, so_entries, ib, isz, index_count, min_index)."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return
    for sub in sorted(cache_dir.iterdir()):
        man = sub / "manifest.json"
        if not man.is_file():
            continue
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
        except Exception:
            continue
        for d in m.get("draws", []):
            vb_file = sub / str(d.get("vb_file", ""))
            if not vb_file.is_file():
                continue
            out = {
                "name": f"{sub.name}/{d.get('vb_file')}",
                "nverts": int(d.get("nverts", 0)),
                "stride": int(d.get("stride", 0)),
                "vb": vb_file.read_bytes(),
                "index_count": int(d.get("index_count", 0)),
                "min_index": int(d.get("min_index", 0)),
                "isz": int(d.get("isz", 2)),
            }
            so_file = d.get("so_file")
            if so_file and (sub / str(so_file)).is_file():
                out["so"] = (sub / str(so_file)).read_bytes()
                out["so_stride"] = int(d.get("so_stride", 0))
                out["so_entries"] = d.get("so_entries", [])
            ib_file = d.get("ib_file")
            if ib_file and (sub / str(ib_file)).is_file():
                out["ib"] = (sub / str(ib_file)).read_bytes()
            yield out


def bake_track_from_get(decode_xyz, decode_nv, cache_dir, baked_dir, asset_hint=""):
    """Find a cached get draw matching this decoded track part (vertex count + order),
    transfer its POST-VS TexCoord0 as the exact game UV, write baked_track_uv/*.json keyed
    by (num, checksum). Returns the written bake dict or None.

    Bake space = (U_rip, 0.5 + V_rip/2): post-VS V is one tile in [-1,0]; 0.5+V_rip/2 lands
    in [0,0.5] (the file-only/merkava_3b bake space) so the importer's flip+crop reproduce it.
    """
    candidates = [d for d in load_get_cache(cache_dir)
                  if d["nverts"] == decode_nv and d["stride"] == TRACK_STRIDE]
    if not candidates:
        _dbg(f"no candidate draw with nverts=={decode_nv} stride=={TRACK_STRIDE} in {cache_dir}")
        return None
    for d in candidates:
        get_xyz = decode_get_xyz(d["vb"], d["stride"], d["nverts"])
        # POST-VS UV (preferred) else legacy pre-VS input-buffer UV
        if d.get("so"):
            get_uv = decode_so_uv(d)
            post_vs = True
        else:
            _, get_uv = decode_get_draw(d["vb"], d["stride"], d["nverts"])
            post_vs = False
        if not get_uv or len(get_uv) < 2 * decode_nv:
            _dbg(f"{d['name']}: short/empty UV")
            continue
        vv = get_uv[1::2]
        vspan = (max(vv) - min(vv)) if vv else 99.0
        if vspan > 1.5:
            _dbg(f"{d['name']}: V span {vspan:.3f} > 1.5 (pre-VS multi-tile? captured wrong draw)")
            continue
        if not _order_matches(decode_xyz, get_xyz, decode_nv):
            _dbg(f"{d['name']}: vertex order does not match the decoded part")
            continue
        bake = []
        for i in range(decode_nv):
            u = get_uv[2 * i]
            v = get_uv[2 * i + 1]
            if post_vs:
                # post-VS V in [-1,0] -> FULL tile [0,1] on the FULL CombinedDAS (no half-crop).
                # The game binds the whole 256x256 tread strip and the UV spans its full height
                # (verified: tread links sit at V levels across the whole tile). Blender V is
                # flipped vs D3D, so negate; REPEAT wraps the few values just outside [0,1].
                bv = -v
            else:
                # legacy pre-VS path kept the old transform
                bv = 1.0 - v / 2.0
            bake.append(round(u, 6))
            bake.append(round(bv, 6))
        chk = _track_pos_checksum(decode_xyz, decode_nv)
        out = {
            "asset": str(asset_hint or "got"),
            "num": int(decode_nv),
            "checksum": int(chk),
            "uv": bake,
            "source": ("post-vs " if post_vs else "pre-vs ") + d["name"],
            # post-VS bakes use the FULL tread tile -> importer must use the FULL CombinedDAS
            # (no top-half crop) and NOT remap the UV into a crop sub-rect.
            "tread_full_uv": bool(post_vs),
        }
        try:
            baked_dir = Path(baked_dir)
            baked_dir.mkdir(parents=True, exist_ok=True)
            (baked_dir / f"rip_{chk:08x}.json").write_text(json.dumps(out), encoding="utf-8")
        except Exception:
            pass
        _dbg(f"BAKED from {d['name']} (post_vs={post_vs}) num={decode_nv} chk={chk:08x}")
        return out
    return None
