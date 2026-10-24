# Kurgu Masası - kaldırma ("Uygulamalar ve özellikler" listesinden çalışır)
# Videolar ve projeler istenirse korunur. Python, FFmpeg, LM Studio gibi ayrı kurulan programlara dokunulmaz.
Add-Type -AssemblyName System.Windows.Forms
$Hedef = Split-Path -Parent $PSScriptRoot
$Ad = 'Kurgu Masası'

$r = [System.Windows.Forms.MessageBox]::Show("$Ad kaldırılsın mı?`n`nKlasör: $Hedef", $Ad, 'YesNo', 'Question')
if ($r -ne 'Yes') { exit }
$veri = [System.Windows.Forms.MessageBox]::Show("Projeler ve videolar (projeler, ham, Videolarım, muzik klasörleri) da silinsin mi?`n`n'Hayır' derseniz bu klasörler yerinde kalır.",
    $Ad, 'YesNo', 'Warning')

# çalışan sunucuyu kapat
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($Hedef, [StringComparison]::OrdinalIgnoreCase) } |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
Start-Sleep -Seconds 1

foreach ($y in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    Remove-Item (Join-Path $y 'Kurgu Masası.lnk') -Force -ErrorAction SilentlyContinue
}
# Premiere köprü eklentisi yalnızca bu kuruluma bağlıysa kaldırılır
$cep = "$env:APPDATA\Adobe\CEP\extensions\MCPBridgeCEP"
$bd = Join-Path $cep 'bridge_dir.txt'
if ((Test-Path $bd) -and ((Get-Content $bd -Raw).Trim().StartsWith($Hedef, [StringComparison]::OrdinalIgnoreCase))) {
    Remove-Item $cep -Recurse -Force -ErrorAction SilentlyContinue
}
Remove-Item 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\KurguMasasi' -Recurse -Force -ErrorAction SilentlyContinue

$koru = @()
if ($veri -ne 'Yes') { $koru = @('projeler', 'ham', 'Videolarım', 'muzik', 'ses_efektleri') }
Get-ChildItem $Hedef -Force -ErrorAction SilentlyContinue | Where-Object { $koru -notcontains $_.Name } | ForEach-Object {
    try { Remove-Item $_.FullName -Recurse -Force -ErrorAction Stop } catch {}
}
if (-not $koru) { Remove-Item $Hedef -Recurse -Force -ErrorAction SilentlyContinue }

$msg = "$Ad kaldırıldı."
if ($koru) { $msg += "`n`nProjeleriniz ve videolarınız şurada duruyor: $Hedef" }
[void][System.Windows.Forms.MessageBox]::Show($msg, $Ad, 'OK', 'Information')
