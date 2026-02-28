# WARNO Blender Importer

Author: Kilivanchik
ModdingSuite for ZZ.dat support: https://github.com/kilivan4iK/moddingSuite/releases
## Included files
- `warno_blender_addon.py` - Blender addon panel/importer
- `warno_spk_extract.py` - core mesh/skeleton/material/texture logic
- `tgv_to_png.py` - TGV -> PNG converter
- `config.example.json` - safe config template
- `track_wheel_presets.example.json` - sample track/wheel presets

## Quick start
1. Copy this folder anywhere.
2. In Blender: `Edit > Preferences > Add-ons > Install...` and select `warno_blender_addon.py`.
3. Open the `WARNO` tab in 3D View sidebar.
4. Set `Project Root` to this folder.
5. Set paths for `Mesh SPK`, `Atlas Assets`, and `TGV Converter`.
6. Scan assets and import.

## Clean PC first run
- `tgv_converter` default is `tgv_to_png.py`.
- `auto_install_tgv_deps` default is `true`.
- `tgv_deps_dir` default is `.warno_pydeps` (local project folder, no admin rights needed).
- First import on a clean machine may auto-install converter deps (`Pillow`, `zstandard`).
- Texture pipeline uses ZZ runtime / Atlas paths and does not require legacy `output/`.
- Resolver policy is strict: if exact texture ref is not found, addon keeps it unresolved (no unsafe "wrong texture" substitution).
- Default performance profile is `Fast exact texture resolve` with hard timeouts (prevents long UI freezes on large shared folders).
- Live diagnostics: use `Open System Console` and `Open Log File` (`warno_import.log` in project root).

## Notes
- This is Alpha version and some tools not correctly work for all WARNO models, currently tested on tanks
- Track/wheel correction supports live presets and post-import `Apply Correction Now`.
- Save your own presets to `track_wheel_presets.json`.
