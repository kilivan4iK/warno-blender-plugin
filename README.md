# WARNO Blender Importer

Author: Kilivanchik

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

## Notes
- This is Alpha version and some tools not correctly work for all WARNO models, currently tested on tanks
- Track/wheel correction supports live presets and post-import `Apply Correction Now`.
- Save your own presets to `track_wheel_presets.json`.
