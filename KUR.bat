@echo off
rem Kurgu Masasi - kurulum. Eksikleri bulur ve ucretsiz olanlari kurar (Python, FFmpeg, yapay zeka paketleri, LM Studio).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0kurulum_kaynak\depodan_kur.ps1"
if errorlevel 1 pause
