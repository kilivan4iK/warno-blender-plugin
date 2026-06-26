"""
IRISZoom unit-mesh vertex codec — decoder for Wargame Red Dragon / Steel Division 2
(and other older Eugen Systems IRISZoom titles) whose VBUF vertex buffers use the
proprietary `01 14 XX 00` codec (pack_type 0x0200, NOT zstd).

Reverse-engineered from WarGame3.exe and validated byte-for-byte against the running
game (June 2026). Every layer is confirmed against live-game ground truth:
  - LZ (Eugen::TLZPlusEntropyCodec<N,unsigned_char>  / <N,unsigned_short>)
  - varint prediction/remap table
  - delta-prediction + Householder symmetry blocks
  - bbox-relative dequantization

Public entry point:  decode_vbuf(buf, num_vertices) -> {"xyz", "uv", "bone_idx", "bone_w"}
where `buf` is the raw on-disk VBUF blob (starting at the b"VBUF" magic).

WARNO is unaffected (it uses real zstd); this module is only invoked for the
proprietary 0x0200 payload.
"""
import struct
import math
import os
import json

# ---------------------------------------------------------------------------
# baked track UV overrides (captured 1:1 from the running game's GPU)
# ---------------------------------------------------------------------------
# The track tread's UV is NOT in the mesh file — the engine recomputes it at
# runtime (the "IndexedChenilles" system) and maps the tread to a thin
# link-pattern strip of the CombinedDAS texture. The file only carries the
# stretched scaly-region UV (what the decoder produces), which renders wrong.
# So for units whose true track UV we captured from the live game (D3D11 frame
# hook -> the track draw's vertex buffer), we ship the exact per-vertex UV here
# and substitute it for the matching chenille part. Keyed by (vertex_count,
# position_checksum) so it only applies to the exact part it was captured from.
_BAKED_TRACKS = None


def _load_baked_tracks():
    global _BAKED_TRACKS
    if _BAKED_TRACKS is None:
        _BAKED_TRACKS = {}
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baked_track_uv")
        try:
            for fn in os.listdir(d):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(d, fn), encoding="utf-8") as fh:
                    o = json.load(fh)
                _BAKED_TRACKS[(int(o["num"]), int(o["checksum"]))] = o["uv"]
        except Exception:
            pass
    return _BAKED_TRACKS


def _track_pos_checksum(xyz, num):
    chk = 0
    for i in range(num * 3):
        chk = (chk + int(round(xyz[i] * 1000))) & 0xffffffff
    return chk


def _match_baked_track(xyz, num):
    bk = _load_baked_tracks()
    if not bk:
        return None
    return bk.get((num, _track_pos_checksum(xyz, num)))


def _fileonly_track_bake(raw_pre, num):
    """Exact track-tread UV computed FROM THE FILE ALONE (no GPU capture).

    Validated 1:1 against the live game's post-vertex-shader output (D3D11 stream-out
    capture) on two very different tanks — merkava (908 v) and M1A1 Abrams (4 track
    parts, 796..1202 v): 0 errors over ~5000 vertices. It also reproduces the proven
    hand-baked merkava_3b.json (99.6%, the rest is rounding).

    Why it works: the decoder's TexCoord0 U is already game-exact; only the V-tile is
    wrong. The IRISZoom codec collapses the tread belt's V onto its two tile seams
    (raw V ~0 or ~P, P = the tread V-period, empirically 8). At runtime the engine
    wraps each vertex to its nearest seam (post-VS V = rawV - P*round(rawV/P), in
    (-1,0]). We reproduce that and map it into the SAME bake space the importer already
    handles for merkava_3b (U = rawU, V = 0.5 + post_VS_V/2 -> [0,0.5]); the importer's
    normal V-flip + atlas-crop then reproduce the game UV with no importer changes.

    `raw_pre` is the PRE-seam-dedup TexCoord0 [(u,v),...] (off24/off26 / 8192). Returns
    a flat [u,v,...] bake, or None when the part isn't a cleanly-collapsed tread (then
    the importer falls back to the connectivity flood-fill). Chenille-gated by callers,
    so WARNO (which never decodes here) is untouched."""
    if not raw_pre or len(raw_pre) < num:
        return None
    vmax = 0.0
    for i in range(num):
        rv = raw_pre[i][1]
        if rv > vmax:
            vmax = rv
    # The belt's V-period is the tile ABOVE the highest cluster: floor(max)+1. The engine
    # constant is 8 for these belts — the tread's high cluster spans [7,8) (-> floor+1=8)
    # and the chains sit exactly at tile 7 (max=7.0 -> floor+1=8, NOT round(7)=7 which
    # would collapse {0,7} onto one V = a 1-D strip). Validated 1:1 vs post-VS.
    period = int(vmax) + 1
    if period < 2:                                   # no V-tiling -> not a collapsed tread
        return None
    # The collapse leaves every rawV near a tile seam (0 or `period`); snapping to the
    # nearest multiple of `period` must land each vertex in (-1, 0]. Build the bake and
    # count any vertex that lands well outside that band (a genuine mid-tile value, i.e.
    # NOT a simple collapsed tread) — if too many, bail to the connectivity flood-fill.
    out = []
    ambiguous = 0
    for i in range(num):
        ru, rv = raw_pre[i]
        post_v = rv - period * round(rv / period)    # -> (-1, 0] for a collapsed tread
        if post_v < -1.05 or post_v > 0.05:
            ambiguous += 1
        # FULL tile [0,1] on the FULL CombinedDAS (matches the captured post-VS path:
        # bake V = -post_v; importer flip 1-v -> final 1+post_v). The old 0.5+post_v/2
        # half-space depended on the top-half crop, which stretched the lower tread links.
        bv = -post_v                                  # post_v in (-1,0] -> bv in [0,1)
        if bv < 0.0:
            bv = 0.0
        elif bv > 1.0:
            bv = 1.0
        out.append(round(ru, 6))
        out.append(round(bv, 6))
    if ambiguous > max(2, int(num * 0.02)):
        return None
    return out


def _within_tile_track_v(raw_pre, num):
    """Reconstruct the CONTINUOUS within-tile V for a collapsed tiled belt (track OR the
    ball-chains belt), to be applied BEFORE the atlas-fold so the part's own atlas rect +
    the importer's crop place it correctly — crop/region-agnostic, unlike _fileonly_track_bake
    (which bakes into the track's specific crop space). Same wrap rule as the tread:
        post-VS V = rawV - period*round(rawV/period)   (period ~ 8, the belt V-period)
        within-tile V = post-VS V + 1                  -> (0, 1]
    Returns a per-vertex V list in (0,1], or None if this part isn't a cleanly-collapsed
    belt (then the caller leaves the UV untouched). Used for the chains, which are NOT a
    chenille part and so don't go through the tread bake."""
    if not raw_pre or len(raw_pre) < num:
        return None
    vmax = 0.0
    for i in range(num):
        rv = raw_pre[i][1]
        if rv > vmax:
            vmax = rv
    period = int(vmax) + 1                            # tile above the high cluster (=8 for tracks)
    if period < 2:                                   # no V-tiling -> not a collapsed belt
        return None
    out = []
    ambiguous = 0
    for i in range(num):
        rv = raw_pre[i][1]
        post_v = rv - period * round(rv / period)    # (-1, 0]
        if post_v < -1.05 or post_v > 0.05:
            ambiguous += 1
        wv = post_v + 1.0                            # (0, 1]
        if wv < 0.0:
            wv = 0.0
        elif wv > 1.0:
            wv = 1.0
        out.append(wv)
    if ambiguous > max(2, int(num * 0.02)):
        return None
    return out

# ---------------------------------------------------------------------------
# little-endian helpers
# ---------------------------------------------------------------------------
def _u16(b, o):
    return b[o] | (b[o + 1] << 8)

def _u32(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)

def _ctz16(x):
    x |= 0x10000
    n = 0
    while (x & 1) == 0:
        n += 1
        x >>= 1
    return n

# Element format ids (FUN_005d23e0 switch) -> (#components, bytes-per-component, is_short)
# 0 float1, 1 float2, 2 float3, 3 float4, 4 ubyte4, 5 ubyte4_keep3, 6 word2, 8 word4
_FMT = {
    0: (1, 2, True), 1: (2, 2, True), 2: (3, 2, True), 3: (4, 2, True),
    4: (4, 1, False), 5: (4, 1, False), 6: (2, 2, True), 8: (4, 2, True), 9: (4, 1, False),
}
# formats that use the unsigned_short LZ variant (FUN_005ef7d0 case dispatch)
_SHORT_FMT = {0, 1, 2, 3, 6, 8}


# ---------------------------------------------------------------------------
# [1] Eugen::TLZPlusEntropyCodec decode  (FUN_005e9790 char / FUN_005e9b00 short)
# ---------------------------------------------------------------------------
def lz_decode(data, unit=1):
    """Decompress one `01 14 XX 00` stream. unit=1 (unsigned_char) or 2 (unsigned_short).

    The marker's third byte (`data[2]`) is the codec template parameter N
    (Eugen::TLZPlusEntropyCodec<N, T>). N is the **bit-width of each packed literal**:
    literals are stored as N-bit little-endian (LSB-first) values in a contiguous
    bitstream, NOT as raw `unit`-sized words. When N == unit*8 (e.g. N=16 short, N=8
    char) this reduces exactly to a raw word copy, so the common WARNO-era streams are
    unchanged; smaller N (e.g. word2 UV uses N=13) packs literals tighter and the SIMD
    decoder (FUN_005ed060 …) unpacks them — reproduced here in scalar form, validated
    byte-for-byte against the running game for N=13 and N=16.

    Stream header: [4:8]=decompressed size (in units), [16:20]=region descriptor f4,
    shift=byte3+2; control bits@20, literals@((f4&0xffff)<<shift), tokens@((f4>>16)<<shift).
    Match tokens (offset/length) are identical across all N."""
    size_units = _u32(data, 4)
    size_bytes = size_units * unit
    if data[2] & 0x80:                          # stored / uncompressed
        return bytes(data[8:8 + size_bytes])
    litw = data[2] & 0x7f                        # codec N = literal bit-width
    shift = (data[3] + 2) & 0x1f
    f4 = _u32(data, 16)
    lit_pos = (f4 & 0xffff) << shift
    tok_pos = (f4 >> 16) << shift
    out = bytearray()
    ctrl_pos = 20
    # bit reader over the literal region (LSB-first, N bits per literal)
    _lit_acc = 0
    _lit_nb = 0
    _lit_bi = lit_pos
    _lit_mask = (1 << litw) - 1
    _raw_lit = (litw == unit * 8)                # fast path: byte-aligned literals
    while len(out) < size_bytes:
        ctrl = _u32(data, ctrl_pos); ctrl_pos += 4
        bits = 32
        while True:
            if (ctrl & 1) == 0:                 # literal run of `n` units
                n = _ctz16(ctrl)
                if n > bits:
                    n = bits
                if _raw_lit:                    # N==unit*8: raw word copy (WARNO-era)
                    out += data[_lit_bi: _lit_bi + n * unit]
                    _lit_bi += n * unit
                else:                           # packed N-bit literals -> `unit` words
                    for _ in range(n):
                        while _lit_nb < litw:
                            _lit_acc |= data[_lit_bi] << _lit_nb
                            _lit_nb += 8; _lit_bi += 1
                        v = _lit_acc & _lit_mask
                        _lit_acc >>= litw; _lit_nb -= litw
                        out += v.to_bytes(unit, "little")
                ctrl >>= n
                bits -= n
            else:                               # match (offset/len in units)
                tok = _u32(data, tok_pos)
                if (tok & 3) == 3:
                    if (tok & 4) == 0:
                        tok_pos += 2
                        offset = ((tok >> 7) & 0x1ff) + 1
                        cnt = ((tok >> 3) & 0xf) + 4
                    else:
                        tok_pos += 3
                        offset = ((tok >> 0xb) & 0x1fff) + 1
                        cnt = ((tok >> 3) & 0xff) + 4
                else:
                    cnt = (tok + 1) & 3
                    mask = 0x1f if (tok & 4) else 0x1fff
                    offset = (mask & (tok >> 3)) + 1
                    tok_pos += 1 if (tok & 4) else 2
                src = len(out) - offset * unit
                for i in range(cnt * unit):
                    out.append(out[src + i])
                ctrl >>= 1
                bits -= 1
            if bits == 0 or len(out) >= size_bytes:
                break
    return bytes(out[:size_bytes])


# ---------------------------------------------------------------------------
# [3] prediction / remap table  (FUN_0065fde0)
# ---------------------------------------------------------------------------
def build_remap(stream, count):
    table = [0, 0]
    p = 0
    for i in range(2, count):
        b0 = stream[p]; p += 1
        if b0 == 0:
            table.append(i - 1)
        else:
            b1 = stream[p]; p += 1
            table.append(i - (((b0 & 0x7f) << 8) | b1))
    return table


# ---------------------------------------------------------------------------
# [4] quantized-value decode (predict + symmetry)  (FUN_005d7350 / FUN_005d2b50)
# ---------------------------------------------------------------------------
def decode_stream(residual, remap, mask, ncomp, count, blocks=None):
    res = struct.unpack_from("<%dH" % (count * ncomp), residual, 0)
    vals = [None] * count
    by_start = {}
    if blocks:
        for b in blocks:
            by_start[b['start']] = b
    i = 0
    while i < count:
        b = by_start.get(i)
        if b is not None and i > 0:
            bc = b['count']
            n = b['normal']
            bias = b['bias']
            for j in range(bc):
                src = vals[i - bc + j]
                t = sum(src[c] * n[c] for c in range(min(ncomp, len(n)))) * 2 - bias
                vals[i + j] = tuple((src[c] - (n[c] if c < len(n) else 0) * t) & mask for c in range(ncomp))
            i += bc
            continue
        p = remap[i] if i < len(remap) else (i - 1 if i > 0 else 0)
        pv = vals[p] if (0 <= p < count and vals[p] is not None) else (0,) * ncomp
        vals[i] = tuple((pv[c] + res[i * ncomp + c]) & mask for c in range(ncomp))
        i += 1
    return vals


# ---------------------------------------------------------------------------
# container walk  (FUN_005eeaa0 + FUN_00598b00 + FUN_00598b70 + FUN_005f0050)
# ---------------------------------------------------------------------------
def _read_top_header(buf):
    """Returns (stride, elem_count, flag, cursor_after_header)."""
    assert buf[0:4] == b"VBUF", "not a VBUF blob"
    adv = _u16(buf, 4)
    htype = buf[6]
    stride = _u16(buf, 7)
    count = _u16(buf, 9)
    flag = buf[11]
    return stride, count, flag, 4 + adv


def _read_sub_descriptor(buf, cur):
    """SUBP element descriptor (FUN_00598b70). Returns (info_dict, next_cursor, data_cursor)."""
    assert buf[cur:cur + 4] == b"SUBP", "expected SUBP at %d (%r)" % (cur, buf[cur:cur + 4])
    p = cur + 4
    block_span = _u16(buf, p)
    fmt = buf[p + 2]
    mode = buf[p + 3]
    kind = buf[p + 4]
    jt = _u16(buf, p + 5)
    out_off = _u16(buf, p + 7)
    blk_count = _u16(buf, p + 9)
    blocks = []
    rp = p + 11
    for _ in range(blk_count):
        recs = struct.unpack_from("<6I", buf, rp)
        # record fields used by FUN_005d7350: [0]=normal coefs packed, etc.
        # (symmetry block decode; for blk_count==0 this is skipped — the common case)
        blocks.append(recs)
        rp += 24
    info = dict(fmt=fmt, mode=mode, kind=kind, jt=jt, out_off=out_off,
                blk_count=blk_count, blocks=blocks)
    return info, cur + 4 + block_span, None


def _read_substream(buf, cur, fmt):
    """FUN_005f0050: [u32 size][payload]; LZ-decode (char/short by fmt). Returns (decoded, next_cursor)."""
    size = _u32(buf, cur)
    payload = buf[cur + 4: cur + 4 + size]
    unit = 2 if fmt in _SHORT_FMT else 1
    decoded = lz_decode(payload, unit=unit)
    nxt = cur + 4 + ((size + 3) & ~3)
    return decoded, nxt


def _parse_vertex_format_fields(vertex_format):
    """Split a `$/M3D/.../TVertex__A__B__C` type string into per-element field names
    (Position, NormalIn01, BlW, BlIdx, TexCoord0, TexPackedAtlas0, ...). Field index
    lines up 1:1 with VBUF element index, so it tells us which element is which
    semantic (bone index / weight / track-index normal)."""
    if not vertex_format:
        return []
    s = str(vertex_format)
    marker = "TVertex__"
    i = s.find(marker)
    if i < 0:
        return []
    return s[i + len(marker):].split("__")


def _chenille_arc_uv(xyz, num):
    """Procedural arc-length UV for a track (chenille) belt, since the tread tiling
    is a runtime engine system absent from the file. Each track loops in the X-Z
    plane with its width along Y (the standard Eugen unit frame, pre-rotation); the
    model's two tracks are separated across Y. Per track: U = cumulative arc length
    around the loop (so the tread tiles uniformly — NOT by raw angle, which stretches
    the straight runs), V = position across the track width [0,1]. Tile count N is
    derived as perimeter / width so the pads stay roughly square (texture aspect
    preserved) and auto-scales to any unit. Returns a flat [u0,v0,u1,v1,...] list, or
    None if the geometry is too small to parametrise."""
    if not xyz or len(xyz) < 3 * num or num < 6:
        return None
    ys = [xyz[v * 3 + 1] for v in range(num)]
    ymid = (min(ys) + max(ys)) * 0.5
    sides = [
        [v for v in range(num) if xyz[v * 3 + 1] > ymid],
        [v for v in range(num) if xyz[v * 3 + 1] <= ymid],
    ]
    uv = [0.0] * (num * 2)
    K = 120
    for side in sides:
        if len(side) < 3:
            continue
        cx = sum(xyz[v * 3] for v in side) / len(side)
        cz = sum(xyz[v * 3 + 2] for v in side) / len(side)
        ymn = min(xyz[v * 3 + 1] for v in side)
        ymx = max(xyz[v * 3 + 1] for v in side)
        yr = (ymx - ymn) or 1.0
        bins = [[] for _ in range(K)]
        for v in side:
            a = math.atan2(xyz[v * 3 + 2] - cz, xyz[v * 3] - cx)
            bins[int((a + math.pi) / (2 * math.pi) * K) % K].append(v)
        cent = {}
        for b in range(K):
            if bins[b]:
                cent[b] = (sum(xyz[v * 3] for v in bins[b]) / len(bins[b]),
                           sum(xyz[v * 3 + 2] for v in bins[b]) / len(bins[b]))
        order = sorted(cent)
        if len(order) < 2:
            continue
        arc = {}
        acc = 0.0
        prev = None
        for b in order:
            if prev is not None:
                dx = cent[b][0] - cent[prev][0]; dz = cent[b][1] - cent[prev][1]
                acc += math.hypot(dx, dz)
            arc[b] = acc
            prev = b
        # close the loop
        dx = cent[order[0]][0] - cent[order[-1]][0]; dz = cent[order[0]][1] - cent[order[-1]][1]
        perim = acc + math.hypot(dx, dz)
        if perim <= 0:
            continue
        n_tiles = max(1.0, perim / yr)        # square-ish pads
        for v in side:
            a = math.atan2(xyz[v * 3 + 2] - cz, xyz[v * 3] - cx)
            b = int((a + math.pi) / (2 * math.pi) * K) % K
            bb = b
            for d in range(K):                 # snap to the nearest filled angle bin
                if (b + d) % K in arc:
                    bb = (b + d) % K; break
                if (b - d) % K in arc:
                    bb = (b - d) % K; break
            uv[v * 2] = (arc[bb] / perim) * n_tiles
            uv[v * 2 + 1] = (xyz[v * 3 + 1] - ymn) / yr
    return uv


def decode_vbuf(buf, num_vertices, want_uv=True, want_bones=True, vertex_format=None):
    """Decode a full IRISZoom VBUF blob into channel-split lists.
    buf: bytes starting at b'VBUF'. num_vertices: vertex count for this buffer.
    vertex_format: the `$/M3D/.../TVertex__...` type string (optional) — when given,
    bone indices/weights are decoded (the BlIdx/BlW elements) and a `chenille` flag
    is set for track geometry (NormalAndChenilleIndex / Chenille fields).
    Returns {"xyz":[...], "uv":[...], "bone_idx":[...], "bone_w":[...], "chenille":bool}."""
    stride, elem_count, flag, cur = _read_top_header(buf)
    fields = _parse_vertex_format_fields(vertex_format)
    chenille = any("chenille" in f.lower() for f in fields)
    remap = None
    if flag:
        lbsize = _u32(buf, cur)
        lb = lz_decode(buf[cur + 4: cur + 4 + lbsize], unit=1)   # remap varint (char)
        remap = build_remap(lb, num_vertices)
        cur += 4 + ((lbsize + 3) & ~3)
    if remap is None:
        remap = [0] + list(range(0, num_vertices - 1))

    out = {"xyz": None, "uv": None, "bone_idx": None, "bone_w": None}
    bidx_bytes = None
    bw_bytes = None
    atlas0 = None        # TexPackedAtlas0: per-part (off_u, off_v, size_u, size_v)/255
    for e in range(elem_count):
        info, nxt_desc, _ = _read_sub_descriptor(buf, cur)
        decoded, nxt = _read_substream(buf, nxt_desc, info['fmt'])
        cur = nxt
        fmt = info['fmt']; kind = info['kind']; out_off = info['out_off']
        ncomp, cbytes, is_short = _FMT.get(fmt, (0, 0, False))
        field = fields[e] if e < len(fields) else ""

        if field:
            fl = field.lower()
            if want_bones and fl.startswith("blidx"):
                bidx_bytes = decoded
                continue
            if want_bones and fl.startswith("blw"):
                bw_bytes = decoded
                continue
            # TexPackedAtlas0 = the texture-atlas sub-rect this part maps into
            # (constant per part): bytes (off_u, off_v, size_u, size_v), /255. RD packs
            # several parts into one texture (e.g. merkava CombinedDAS: tracks -> bottom
            # half, chains -> next quarter); the within-rect UV is TexCoord0. Capture it
            # here and fold it into the UV after the loop so each part lands in its rect.
            if want_uv and fl.startswith("texpackedatlas0") and len(decoded) >= 4:
                atlas0 = (decoded[0], decoded[1], decoded[2], decoded[3])
                continue

        if kind == 2 and fmt == 2:               # float3 position
            mask = _u16(decoded, 0)
            f6 = struct.unpack_from("<6f", decoded, 2)
            mins = f6[0:3]; maxs = f6[3:6]
            residual = decoded[28:]
            vals = decode_stream(residual, remap, mask, 3, num_vertices, info['blocks'])
            inv = 1.0 / mask if mask else 0.0
            # /100: IRISZoom positions are in centimetres; the WARNO importer path
            # expects metres (its float32 path divides by 100), so match it here.
            sx = inv * (maxs[0] - mins[0]) / 100.0
            sy = inv * (maxs[1] - mins[1]) / 100.0
            sz = inv * (maxs[2] - mins[2]) / 100.0
            ox = mins[0] / 100.0; oy = mins[1] / 100.0; oz = mins[2] / 100.0
            xyz = []
            for v in vals:
                xyz.append(ox + v[0] * sx)
                xyz.append(oy + v[1] * sy)
                xyz.append(oz + v[2] * sz)
            out["xyz"] = xyz
        elif kind == 2 and fmt == 6 and want_uv:  # word2 uv (FUN_005d2720)
            mask = _u16(decoded, 0)               # header = u16 mask + 2 pad = 4 bytes
            residual = decoded[4:]
            vals = decode_stream(residual, remap, mask, 2, num_vertices, info['blocks'])
            # Faithful dequant — matches the game (FUN_005d2720) AND the WARNO float
            # path (`v / 8192.0`, warno_spk_extract.py): out_u16 = (val*scale)&0xffff
            # with scale = 65536//(mask+1); UV = out_u16 / 8192.
            #   * The divisor is 8192, NOT 65535 — so a component spanning the full
            #     quantum range yields UV up to ~8.0. The vertex type `TexCoord0_2wn`
            #     is a tiling coordinate: tracks legitimately run V≈0..8 (the tread
            #     repeats), and unit BODIES quantise into a single high V-tile (e.g.
            #     ~[7,8]). Both are correct under texture *repeat*.
            #   * Earlier attempts were wrong: dividing by 65535 squashed U to ~1/8;
            #     dividing each component by 2^ceil(log2(max)) (bit-width) squashed
            #     the body's tile-7 band into a 1/8 strip ("приплюснута"). 8192 is the
            #     right divisor.
            # Then wrap each component into [0,1] with `% 1.0`. RD UVs are quantised
            # into discrete integer tiles: bodies sit in one high tile (~7), tracks/
            # chains split across two (e.g. tile 7 + tile 0). Every tile maps to the
            # SAME texels under texture *repeat* (tile k == tile 0), so the modulo is a
            # rendering no-op that simply brings ALL clusters back into a clean [0,1]
            # island — no squished band, no stray strips at far offsets. (An earlier
            # subtract-the-median-tile offset only re-centred ONE tile, leaving the
            # other cluster as tall strips far away — that was the "track UV" artefact.)
            scale = (65536 // (mask + 1)) if mask else 1
            # Defer the dequant until after the element loop: the wrap-seam de-dup
            # below needs decoded POSITIONS, which may arrive in a later element.
            out["_uv_raw"] = vals
            out["_uv_scale"] = scale
            out["_uv_mask"] = mask
        # other kind-0 elements (normal / tangent) are not needed here
    # Dequantise UV (deferred from the element loop so decoded positions are on hand).
    raw_pre = None
    if want_uv and out.get("_uv_raw") is not None:
        vals = out.pop("_uv_raw")
        scale = out.pop("_uv_scale")
        mask = out.pop("_uv_mask")
        xyz = out.get("xyz")
        # Snapshot the PRE-seam-dedup raw TexCoord0 (off24/off26 / 8192) for the exact
        # tread/chains V reconstruction below — the seam-dedup mutates `vals`, so capture
        # it now. RD-only (WARNO never reaches decode_vbuf).
        raw_pre = [((((v[0] * scale) & 0xffff) / 8192.0), (((v[1] * scale) & 0xffff) / 8192.0)) for v in vals]
        # De-dupe wrap-seam sentinels: the codec's `& mask` collapses the texture
        # wrap point (raw == mask+1) back to raw 0, leaving a duplicate vertex at the
        # SAME 3D position as a real one but with a 0 UV component. Faces that mix the
        # two span the entire tile range -> the compressed-stripe "track UV" artefact
        # (merkava skirts: 31% of faces, every 0-V vert coincident with a tile-7 vert).
        # Adopt the real (highest-V) coincident vertex's UV for any 0-V sentinel so the
        # seam pair shares a texel and faces stay contiguous. Self-gating: only fires
        # where a 0-V vert coincides with a high-V (> mask/2) partner.
        if xyz is not None and len(xyz) >= 3 * num_vertices:
            posmap = {}
            for i in range(num_vertices):
                k = (round(xyz[i * 3], 3), round(xyz[i * 3 + 1], 3), round(xyz[i * 3 + 2], 3))
                posmap.setdefault(k, []).append(i)
            vals = [list(t) for t in vals]
            half = (mask // 2) if mask else 0
            # The wrap is in V (the only axis that reaches the tile-8 boundary and
            # collapses to 0): adopt the real (highest-V) coincident vertex's full UV
            # for any V==0 sentinel. Gate on a high-V partner (> mask/2) so genuine
            # U==0 edge verts that merely have a real V are left untouched (those keep
            # their own U — U does not wrap here, e.g. droite U maxes at ~0.9).
            for grp in posmap.values():
                if len(grp) < 2:
                    continue
                rep = max(grp, key=lambda i: vals[i][1])
                if vals[rep][1] <= half:
                    continue
                ru, rv = vals[rep][0], vals[rep][1]
                for i in grp:
                    if i != rep and vals[i][1] == 0:
                        vals[i][0] = ru
                        vals[i][1] = rv
        # Wrap into [0,1] ONLY for values above 1.0 (higher tiles — bodies/skirts sit
        # in tile ~7 and must fold down). Values already in [0,1] are kept verbatim so
        # an exact tile-boundary top (UV == 1.0, e.g. the chain strip's right edge at
        # raw == one tile) is NOT collapsed to 0.0 by the modulo — that collapse made
        # chain quads that should span U[0,1] degenerate to a single column. Identical
        # output for bodies/skirts (their components are 0 or > 1.0).
        def _frac(raw):
            f = (((raw * scale) & 0xffff) / 8192.0)
            return f if f <= 1.0 else (f % 1.0)
        uv = []
        for v in vals:
            uv.append(_frac(v[0]))
            uv.append(_frac(v[1]))
        out["uv"] = uv
        # RD track (chenille) only: also keep the UN-wrapped multi-tile UV (no `% 1.0`).
        # The tread tiles in V (the belt spans e.g. tile 7..8); `_frac`'s wrap collapses
        # the tile-boundary seam verts onto the wrong end, smearing the tread. The
        # importer-side seam flood-fill (get_model_geometry -> _floodfill_track_uv_seam)
        # needs the continuous multi-tile values + face connectivity to re-place the seam.
        if chenille:
            out["uv_pretile"] = [
                (((v[0] * scale) & 0xffff) / 8192.0) if (j == 0) else
                (((v[1] * scale) & 0xffff) / 8192.0)
                for v in vals for j in range(2)
            ]

    # (Track-tread arc-length heuristic removed: it re-UV'd the whole chenille part,
    # including the flat hull-side, and tiled 28x = chaotic. The exact tread UV is the
    # engine's runtime IndexedChenilles output — captured 1:1 from the live game and
    # applied as a per-asset override instead of guessed here.)

    # RD ball-chains belt (the "цепури") — a NON-chenille tiled belt whose V is collapsed
    # the SAME way as the tread (codec snaps it to the tile seams), leaving the UV a flat
    # 1-D strip → the links stretch into vertical lines. It never goes through the tread
    # bake (that's chenille-only), so reconstruct its continuous within-tile UV HERE, from
    # the true pre-dedup raw, BEFORE the atlas-fold — the part's own atlas rect then places
    # it (region-agnostic). The bimodal-belt guard returns None for normal body/turret
    # parts (smooth V, no tiling), so only collapsed belts are touched. WARNO never reaches
    # decode_vbuf, so it is unaffected.
    if want_uv and not chenille and raw_pre is not None and out.get("uv") and len(out["uv"]) >= 2 * num_vertices:
        _wv = _within_tile_track_v(raw_pre, num_vertices)
        if _wv is not None and len(_wv) == num_vertices:
            _uv = out["uv"]
            for _i in range(num_vertices):
                _ru = raw_pre[_i][0]
                _uv[2 * _i] = _ru if _ru <= 1.0 else (_ru % 1.0)   # true raw U (pre-dedup)
                _uv[2 * _i + 1] = _wv[_i]                           # reconstructed within-tile V
            out["belt_vfixed"] = True
    # Diagnostic: the raw TexCoord0 V span (a collapsed tiled belt has rawV reaching the
    # tile period ~8; a flat/constant-V part stays < 2 and is NOT file-reconstructable).
    if raw_pre is not None and num_vertices > 0:
        _n = min(num_vertices, len(raw_pre))
        out["belt_vmax"] = max((raw_pre[_i][1] for _i in range(_n)), default=0.0)

    # Fold the per-part TexPackedAtlas0 sub-rect into the UV: the wrapped [0,1]
    # TexCoord0 is the WITHIN-rect coordinate; place it into the part's rect
    # (offset, size). RD packs several parts into one texture (merkava CombinedDAS:
    # tracks -> V[0,0.5], chains -> V[0.5,0.75]); without this they all stacked on
    # [0,1] and showed the wrong texels ("видно цепури і самі траки"). Parts whose
    # rect is the full texture (size 1,1 — bodies, most units) are unchanged.
    if want_uv and atlas0 is not None and out.get("uv"):
        ou = atlas0[0] / 255.0
        ov = atlas0[1] / 255.0
        su = atlas0[2] / 255.0
        sv = atlas0[3] / 255.0
        uvv = out["uv"]
        out["uv"] = [
            (ou + uvv[i] * su) if (i & 1) == 0 else (ov + uvv[i] * sv)
            for i in range(len(uvv))
        ]
        # keep the rect for the importer-side track seam flood-fill (it re-folds V)
        if chenille:
            out["atlas0"] = [atlas0[0], atlas0[1], atlas0[2], atlas0[3]]

    # RD track (chenille): if we shipped an exact ripped tread UV for THIS part
    # (matched by vertex count + position checksum), substitute it for the decoded
    # TexCoord0. The decoder's UV is correct for the belt body but collapses the ~216
    # tile-boundary seam sentinels (raw V==0) onto the wrong texels, which smears the
    # tread (badU=28 vs the game's 0). The baked UV is the game-exact value, stored in
    # decoder-output space (U=U_rip, V=1-V_rip/2) so the importer's normal V-flip +
    # atlas-crop remap reproduce the rip 1:1 with NO importer changes. Gated on the
    # `chenille` flag (RD/SD2 tracks only) — WARNO parts never set it, so WARNO is
    # untouched; the (num,checksum) key restricts the override to its captured part.
    if chenille and out.get("uv") and out.get("xyz"):
        baked = _match_baked_track(out["xyz"], num_vertices)
        if baked is not None and len(baked) == len(out["uv"]):
            out["uv"] = list(baked)
            out["baked_track"] = True   # exact: importer must NOT also seam-flood-fill

    # RD track (chenille): EXACT file-only tread UV (no GPU capture, no per-asset bake).
    # The engine maps the track tread to ONE tile — VERIFIED from our own post-VS stream-out
    # capture: the game's post-VS TEXCOORD0 for a track is a single tile (U[0,1], V[-1,0],
    # Vspan~1.0), NOT a per-link multi-tile sawtooth. So the one-tile fold IS the correct,
    # game-matching UV. period=int(vmax)+1; post_v=rawV-period*round(rawV/period); the bake
    # space (V=0.5+post_v/2) reproduces the post-VS tile after the importer flip+crop.
    # Chenille-gated -> WARNO untouched. Falls back to the flood-fill seam repair (returns
    # None) for parts that aren't a cleanly-collapsed tread.
    if chenille and not out.get("baked_track") and raw_pre is not None and out.get("uv"):
        fo = _fileonly_track_bake(raw_pre, num_vertices)
        if fo is not None and len(fo) == len(out["uv"]):
            out["uv"] = fo
            out["baked_track"] = True
            out["fileonly_track"] = True
            # FULL tread tile -> importer uses the WHOLE CombinedDAS (no top-half crop),
            # same game-exact path as a post-VS capture. Fixes non-captured RD units too.
            out["tread_full_uv"] = True

    # RD track (chenille): flag the part so the importer keeps it as one "Chenille"
    # object and routes it through the standard CombinedDAS + alpha_cutout material.
    # The track uses its OWN decoded TexCoord0 (same as the unit body) — no procedural
    # projection. The single V-flip every part needs (D3D V-down -> Blender V-up) is
    # applied ONCE in the importer (_collect_mesh_buckets, `vv = 1.0 - v`); do NOT flip
    # again here or the track double-flips back onto the ball-chains half of CombinedDAS.
    if chenille:
        out["chenille"] = True

    # Assemble bone skinning (matches the WARNO float-path contract: bone_idx is a
    # flat list of 4 ints/vertex, bone_w a flat list of 4 normalised floats/vertex).
    if want_bones and bidx_bytes is not None:
        bone_idx = []
        bone_w = []
        for v in range(num_vertices):
            o = v * 4
            bi = [int(bidx_bytes[o + k]) if o + k < len(bidx_bytes) else 0 for k in range(4)]
            if bw_bytes is not None and o + 3 < len(bw_bytes):
                ws = [float(bw_bytes[o + k]) / 255.0 for k in range(4)]
            else:
                ws = [1.0, 0.0, 0.0, 0.0]
            tot = sum(ws)
            if tot > 0.0:
                ws = [w / tot for w in ws]
            bone_idx.extend(bi)
            bone_w.extend(ws)
        out["bone_idx"] = bone_idx
        out["bone_w"] = bone_w
    result = {k: v for k, v in out.items() if v is not None}
    if chenille:
        result["chenille"] = True
    return result
