@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0New-Release.ps1"
if errorlevel 1 (
  echo Release packaging failed. See the errors above.
  pause
  exit /b 1
)
echo Release ZIP and checksum are in the releases folder.
pause
