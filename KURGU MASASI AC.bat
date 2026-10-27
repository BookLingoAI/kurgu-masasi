@echo off
cd /d "%~dp0"
rem Restart the server so new code is loaded - but never while an edit job is running (it would be lost).
rem Only the main server is stopped; errors about already-closed processes are hidden.
if exist "isler\sonuc\*.calisiyor.json" (
  echo Bir kurgu isi calisiyor: sunucu yeniden baslatilmadi, yalnizca pencere aciliyor.
) else (
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | Where-Object { $_.CommandLine -match '-m kurgu\s*$' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
  timeout /t 2 /nobreak >nul
)
rem kurgu.desktop starts the server (if not running), then opens the app window
start "" ".venv\Scripts\pythonw.exe" -m kurgu.desktop
