# WARNO Blender Importer

Author: Kilivanchik

## Included files
- `warno_blender_addon.py` - Blender addon panel/importer
- `warno_spk_extract.py` - core mesh/skeleton/material/texture logic
- `tgv_to_png.py` - TGV -> PNG converter
- `manual_texture_corrector_cpp/` - manual texture correction tool (source + ready `build/Release/manual_texture_corrector.exe`)
- `config.example.json` - safe config template
- `track_wheel_presets.example.json` - sample track/wheel presets
- `spk/.gitkeep`, `skeletonsspk/.gitkeep` - placeholders for optional local SPK folders

## Not included
- `moddingSuite/` and `moddingSuite-master/` are intentionally excluded from this package.

## Quick start
1. In Blender: `Edit > Preferences > Add-ons > Install...` and select `warno_blender_addon.py` from this folder.
2. Open the `WARNO` tab in 3D View sidebar.
3. Set `Project Root` to this folder.
4. Configure paths (`Mesh SPK/Folder`, `Atlas Assets`, `TGV Converter`, WARNO folder for ZZ.dat mode).
5. Click `Scan Assets` (or `Scan ALL Assets (takes a long time)`) and import.

## Notes
- Tool is focused on WARNO extraction/import workflows; behavior can vary by model type.
- Manual texture correction is available via the included C++ tool.
