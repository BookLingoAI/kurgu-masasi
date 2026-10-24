# Kurgu Masası - GitHub'dan indirilen klasörden kurulum
# KUR.bat bunu çalıştırır: depodaki dosyaları kurulum sihirbazının beklediği düzene koyar (%TEMP%) ve sihirbazı açar.
# Sihirbaz eksikleri bulur ve ücretsiz olanları kurar: Python 3.12, FFmpeg, yapay zekâ paketleri, LM Studio + dil modeli.
param([switch]$SadeceHazirla)   # testler için: dosyaları hazırla, sihirbazı açma
$ErrorActionPreference = 'Stop'
$Depo = Split-Path -Parent $PSScriptRoot
$Sahne = Join-Path $env:TEMP 'KurguMasasiKurulum'
$Uyg = Join-Path $Sahne 'uygulama'
if (Test-Path $Sahne) { Remove-Item $Sahne -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Uyg 'kurulum') | Out-Null

function Kopyala([string]$kaynak, [string]$hedef) {
    $ra = @("`"$kaynak`"", "`"$hedef`"", '/E', '/R:1', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/XD', '__pycache__')
    $rc = (Start-Process -FilePath "$env:SystemRoot\System32\robocopy.exe" -ArgumentList $ra -PassThru -Wait -WindowStyle Hidden).ExitCode
    if ($rc -ge 8) { throw "Kopyalanamadı: $kaynak (robocopy $rc)" }
}
foreach ($k in @('kurgu', 'assets', 'modeller', 'premiere')) {
    if (Test-Path (Join-Path $Depo $k)) { Kopyala (Join-Path $Depo $k) (Join-Path $Uyg $k) }
}
if (Test-Path (Join-Path $Depo 'araclar\hyperframes')) { Kopyala (Join-Path $Depo 'araclar\hyperframes') (Join-Path $Uyg 'araclar\hyperframes') }
foreach ($f in @('design_config.json', 'dictionary.json', 'KURGU MASASI AC.bat')) {
    if (Test-Path (Join-Path $Depo $f)) { Copy-Item (Join-Path $Depo $f) $Uyg }
}
Set-Content -Path (Join-Path $Uyg 'user_settings.json') -Value '{"camera_files": [], "screen_files": [], "project_name": "", "subtitles_enabled": true}' -Encoding UTF8
$K = Join-Path $Depo 'kurulum_kaynak'
foreach ($f in @('kaldir.ps1', 'dogrula.py', 'gereksinimler.txt', 'gereksinimler-gpu.txt', 'gereksinimler-istege-bagli.txt', 'simge.ico', 'surum.txt')) {
    Copy-Item (Join-Path $K $f) (Join-Path $Uyg 'kurulum')
}
Kopyala (Join-Path $K 'premiere-kopru') (Join-Path $Uyg 'kurulum\premiere-kopru')
Copy-Item (Join-Path $K 'kurulum.ps1') $Sahne
Copy-Item (Join-Path $K 'simge.ico') $Sahne
if ($SadeceHazirla) { Write-Output $Sahne; exit 0 }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File (Join-Path $Sahne 'kurulum.ps1') -Kaynak $Sahne
