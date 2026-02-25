# Manual Texture Corrector (C++)

Standalone Win32 GUI tool for manual texture splitting/correction.

## What it does

- Opens all image files from a texture folder.
- Shows each texture one-by-one.
- Lets you define split grid via:
  - `Cols`, `Rows` (uniform grid), or
  - click guides with `Add V guide` / `Add H guide`.
- Numeric fields use spinner arrows (up/down) and update preview grid immediately.
- Guide clicks can snap to pixel step (`Guide snap step (px)`), e.g. 512, 1024.
- Lets you set:
  - base name (without suffix),
  - suffix (`_D`, `_NM`, `_R`, `_M`, `_AO`, `_A`, `_ORM`, custom),
  - extension,
  - tile selection (`Save all tiles` or one tile index),
  - expand in pixels.
- Saves outputs to target folder (`manual_corrected` by default).
- Writes `manual_texture_manifest.json` used by Blender addon for auto-apply.

## Build (Visual Studio / MSVC)

```powershell
cd manual_texture_corrector_cpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Result:

- `manual_texture_corrector_cpp/build/Release/manual_texture_corrector.exe`

## Run manually

```powershell
manual_texture_corrector.exe --input "E:\path\to\textures\folder" --output "E:\path\to\textures\folder\manual_corrected"
```

`--output` is optional.
