@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% -c "import zstandard" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python module "zstandard" not found.
  echo Run: %PY% -m pip install zstandard
  pause
  exit /b 1
)

%PY% -c "from PIL import Image" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python module "Pillow" not found.
  echo Run: %PY% -m pip install pillow
  pause
  exit /b 1
)

if not exist "png_out" mkdir "png_out"

set "SPLIT_MODE=auto"
set /p "SPLIT_INPUT=Enable auto texture slicing? [Y/n]: "
if /I "!SPLIT_INPUT!"=="n" set "SPLIT_MODE=none"
if /I "!SPLIT_INPUT!"=="no" set "SPLIT_MODE=none"

set "NAMING_FLAG="
set /p "NAMING_INPUT=Enable auto texture naming? [Y/n]: "
if /I "!NAMING_INPUT!"=="n" set "NAMING_FLAG=--no-auto-naming"
if /I "!NAMING_INPUT!"=="no" set "NAMING_FLAG=--no-auto-naming"

echo Converting all .tgv in: %CD%
echo Auto split mode: !SPLIT_MODE!
if defined NAMING_FLAG (
  echo Auto naming: off
) else (
  echo Auto naming: on
)
echo.

%PY% tgv_to_png.py "." "png_out" --split !SPLIT_MODE! --recursive !NAMING_FLAG!

echo.
echo Done! PNG files are in: png_out
pause
popd
