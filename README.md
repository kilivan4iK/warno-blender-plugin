# WARNO Blender Importer

> A Blender add-on that imports 3D models, hierarchy, materials and textures
> directly out of WARNO's `Base.spk` / `ZZ_*.dat` packages — vehicles,
> helicopters, infantry, props and decors.
> Built for modders who want to edit WARNO assets without bouncing through
> external converters.

- **Author:** Kilivanchik
- **Companion tool:** [moddingSuite (ZZ.dat support fork)](https://github.com/kilivan4iK/moddingSuite/releases)
- **Status:** Beta — solid for most vehicles, edge cases documented in
  [Known Issues](#-known-issues).
- **Tested with:** Blender 4.2 LTS and 5.0+. WARNO build v189668 (April 2026 patch).

![preview](https://github.com/user-attachments/assets/8f0005b0-0681-4870-8728-a7c3e6320d28)

---

## ✨ Features

### Asset extraction pipeline
- Parses Eugen's `EDAT v1/v2` containers (`ZZ_*.dat`, `*_Assets.dat`) directly,
  no QuickBMS dance required.
- Handles **chained delta packs** — every WARNO update ships a new
  `Data/PC/<old_build>/<new_build>/ZZ_*.dat` and the resolver merges all
  layers (newest wins) automatically. New DLC content is picked up the
  moment Steam finishes downloading.
- Handles both legacy `PC/Mesh/Pack/Base.spk` and the new short
  `MeshPack/Base.spk` dictionary path scheme that Eugen introduced in
  v188908+.
- Pulls SPK files out of `ZZ.dat` into a local cache (`out_blender_runtime/`),
  re-uses cached extracts on subsequent imports.

### Geometry & hierarchy
- Reconstructs the WARNO node tree as a Blender scene graph: `Chassis`,
  `Tourelle_*`, `Canon_*`, `Roue_*`, `Roue_Elev_*`, `Chenille_*`,
  `Fx_Tourelle*_Tir_*`, smoke anchors, FX locators, hatches.
- Bone-parented helper armatures for each track side (`Armature_D1`,
  `Armature_D2`, `Armature_G1`, `Armature_G2`) with deform-only weights.
- Geometry cleanup: removes degenerate triangles, welds duplicate vertices
  (position + UV), conservative triangle-pair → quad merging.
- UV repair for tracked-vehicle UV-tile artefacts (the cooker rejects
  out-of-range UVs).
- Per-asset `WARNO_DEV_SCALE = 20/43` so imported bounds match the
  developer reference `.blend` exactly.
- Wheel-mesh rotation override on tanks where SPK ships a non-identity
  off_mat that would double-rotate the cylinder onto its side
  (matches the dev `.blend` `rot=(0, 0, 0)` for `Roue_*`).

### UI
- **Tree-view Asset Browser** — replaces the legacy 5-dropdown popup.
  Expand/collapse folders, live search, optional LOD rows, single-click
  to pick. Persists expand state across sessions.
- **Asset Picker** with query field, scan-cache, current asset display,
  per-group LOD selector.
- **Apply / Reapply Textures** button to re-resolve textures on an
  already-imported asset without rebuilding geometry.
- **Auto Smooth modes:** `as modifier` (editable), `off`, `apply effect`.
- **Logs to file:** every import writes structured progress lines to
  `warno_import.log` for diffing and bug reports.

---

## 📦 Installation

### Prerequisites
- **Blender 4.2 LTS or 5.0+**
- **Windows 10/11 x64.** Linux/macOS are technically supported by the
  Python side, but the bundled moddingSuite CLIs are .NET binaries
  shipped for `win-x64`.
- **WARNO** installed via Steam (the importer reads directly from
  `<Steam>\steamapps\common\WARNO\Data\PC\...\ZZ_*.dat`).
- **.NET 8 runtime** (https://builds.dotnet.microsoft.com/dotnet/WindowsDesktop/9.0.14/windowsdesktop-runtime-9.0.14-win-x64.exe).

### Step-by-step

1. Clone or download this repository to any location, e.g.
   `E:\warno-blender\`.
2. In Blender: `Edit → Preferences → Add-ons → Install…`, select
   `warno_blender_addon.py` from this folder, then enable the checkbox.
3. Open any 3D Viewport, press `N`, switch to the **WARNO** tab.
4. Fill in the **Project** section:
   - **Project Root:** the absolute path to this folder.
   - Click **Load example config** (then **Save Config** to remember it).
5. Open **First Setup / Logs** and confirm:
   - **WARNO Folder:** `<Steam>\steamapps\common\WARNO`
   - **moddingSuite Root:** path to a moddingSuite checkout
     (the repo includes a working sibling `moddingSuite/` folder).
   - **ModdingSuite Atlas CLI / GFX CLI:** auto-resolve to the bundled
     `.exe` files; only override if you compiled your own.
6. Click **Install / Check TGV deps**. The plugin downloads two Python
   packages (`Pillow`, `zstandard`) into `.warno_pydeps/py311/` so it
   doesn't pollute Blender's global site-packages.
7. Click **Prepare ZZ Runtime**. First run takes ~30 seconds (extracts
   `Base.spk` and a handful of decor packs). Subsequent runs are
   near-instant unless WARNO updated.
8. Click **Scan ALL Assets** to build the asset index (~7000 entries,
   one-time ~20 second build, then cached).

---

## 🚀 Quick start

After installation:

1. **Browse Assets** (in the *Asset Picker* section) — opens the
   tree-view popup. Expand `Units → US → Char → M1_Abrams`, click the
   model row.
2. The picker dropdowns and the *Current* label update with the picked
   asset's path.
3. Tweak **Import Options** as needed:
   - `Auto split main parts` — splits chassis/turrets/wheels by raw
     bone names (recommended for most units).
   - `Auto pull bones` — builds helper armatures and parents the wheel
     hierarchy. Off only if you want a flat scene.
   - `Auto material naming` — uses Eugen's material names from SPK
     instead of `Material_001`-style placeholders.
   - `Merge by distance` — extra weld pass after import.
   - `Shade auto smooth` modes — see UI tooltips.
4. Click **Import To Blender**. First import of a unit takes 5-30 s
   (texture conversion is the bottleneck); subsequent re-imports of the
   same unit are <5 s thanks to the cache.
5. Use **Apply / Reapply Textures** if you only want to refresh the
   texture set on the already-imported geometry (handy after a WARNO
   update).

---

## ⚙️ Configuration

The plugin stores all paths and toggles in `config.json` next to
`warno_blender_addon.py`. Click **Save Config** in the panel to write the
current settings out, **Load My config** to restore them. Key fields:

| Key | Purpose |
|---|---|
| `warno_root` | Absolute path to your WARNO Steam folder. |
| `modding_suite_root` | moddingSuite checkout (sibling `..\moddingSuite\` works). |
| `modding_suite_atlas_cli` | Atlas CLI executable (auto-detected). |
| `modding_suite_gfx_cli` | GFX manifest CLI executable (auto-detected). |
| `zz_runtime_dir` | Where extracted SPK / atlas live (`out_blender_runtime/zz_runtime` by default). |
| `cache_dir` | Asset index + atlas JSON cache (`output_blender` by default). |
| `auto_textures` | Master toggle for texture extraction. |
| `use_atlas_json_mapping` / `atlas_json_strict` | Use atlas JSON (recommended) and reject ambiguous mappings (recommended). |
| `fast_exact_texture_resolve` | Take the cheap exact-name path before scanning all atlas roots. |
| `tgv_deps_dir` / `auto_install_tgv_deps` | Where to install Pillow/zstandard, and whether to do it automatically. |
| `auto_pull_bones`, `auto_split_main_parts`, `auto_name_materials` | Default checkbox states. |
| `fbx_auto_smooth_mode` | `MODIFIER` / `OFF` / `APPLY`. |
| `import_semantic_mode` | `REFERENCE` matches the developer reference `.blend` layout. |

---

## 🧠 How it works (architecture)

```
warno_blender_addon.py    Blender UI, operators, scene build, materials.
warno_spk_extract.py      SPK / EDAT parser, geometry extraction, atlas
                          mapping, texture resolution.
modding_suite_atlas_export.py
                          Wrapper for moddingSuite.AtlasCli.exe
                          (atlas → JSON mapping per asset).
modding_suite_gfx_export.py
                          Wrapper for moddingSuite.GfxCli.exe
                          (turret/FX/sub-depiction manifest per asset).
tgv_to_png.py             Eugen's TGV format → PNG converter.
moddingSuite/             Bundled .NET CLIs + native dependencies.
out_blender_runtime/      Extracted SPK files + atlas mirror (cache).
output_blender/           Per-asset PNGs, atlas JSON cache, asset index.
tools/                    Headless Blender scripts for regression
                          testing (dump_blend_state, compare_blend_dumps,
                          import_asset_blend, ...).
```


## 🐞 Known Issues

This list is honest, not marketing. Some items are *fundamental* limits
of the SPK/FBX pipeline; others are open work.

### Visual / cosmetic

- **Striped roof on composite DECORS buildings** (e.g. `HLM_10_L.fbx`).
  Eugen's source FBX ships some quads with butterfly UV (two diagonal
  corners share the same texel), which Blender renders as visible
  stripes. The dev `.blend` doesn't show it because it was authored
  by hand with ngons. The cooker is fine with it. Manual fix:
  Edit Mode → select the face → `U` → `Smart UV Project`.

- **Not every model is 100% quad/ngon**. The dev reference `.blend` for
  units like Leopard_1A1 has ~95% quads and 24 ngons; we get ~65% quads
  / 35% triangles after import. This is FBX-format limitation: ngons
  are triangulated on Eugen's export, our quad-merge pass recovers
  most pairs but never the 5+ vertex faces.

- **Hatches (`Trappe_*`) appear as empty markers** instead of separate
  meshes on tanks like Leopard_2A1. The hatch geometry is welded into
  the chassis bone vertex group in SPK, even though Eugen's authoring
  `.blend` keeps each hatch as its own mesh parented to the turret.
  Manual fix: in Edit Mode on `Chassis`, border-select the hatch verts,
  press `P → Selection`, then `Ctrl+P` on the new object → parent to
  `Tourelle_01`.

---

## 🛠️ Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ZZ runtime prepare failed: No texture DAT packages found under WARNO folder` | WARNO folder path wrong, or Steam is mid-update. | Verify `<WARNO>/Data/PC/.../ZZ_*.dat` exists; wait for Steam if downloading. |
| `Asset not found in SPK` after WARNO updated | Cache stale. | Click **Prepare ZZ Runtime** then **Scan ALL Assets**. |
| Newly added DLC unit doesn't appear in Browse | Asset index cache from before the update. | Same as above. |
| Window glass shows the wrong unit's texture | Two units in the scene; latest import owns `Vitre`. | Delete the older unit, or rename `<old>__Vitre` back to `Vitre` before exporting. |
| Wheel cylinders lying on their side | Affects some tanks where SPK off_mat carries non-identity rotation. Should be auto-fixed for `Roue_*`. | If you still see it: open an issue with the unit name. |

For anything not covered here:

1. Open **System Console** (button in *First Setup / Logs*).
2. Reproduce the issue.
3. Open `warno_import.log` (button next to it) and grep for `[ERROR]`
   or `[WARNING]`.
4. Open an issue with the log excerpt + the asset path you tried.

---

> If you imported a unit that doesn't match this README's promises —
> open an issue with the asset path, a screenshot, and the relevant
> excerpt from `warno_import.log`.