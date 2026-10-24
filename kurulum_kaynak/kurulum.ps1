# Kurgu Masası - kurulum sihirbazı (Windows PowerShell 5.1 + Windows Forms)
# Setup.exe bu betiği geçici klasörden çalıştırır: -Kaynak = uygulama dosyalarının bulunduğu klasör.
# Eksikleri bulur ve ücretsiz olanları indirir: Python 3.12, FFmpeg, yapay zekâ paketleri, LM Studio (+ dil modeli).
# Premiere Pro ücretli olduğu için indirilmez; kuruluysa köprü eklentisi kurulur, değilse uygulama kendi motoruyla çalışır.
param([string]$Kaynak = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$AppAdi = 'Kurgu Masası'
$PyUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
$FfUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
$ModelRepolari = @('lmstudio-community/Qwen3-8B-GGUF', 'Qwen/Qwen3-8B-GGUF')
$WhisperModel = 'large-v3-turbo'

# ------------------------------------------------------------------ algılama
function Python312-Bul([string]$hedef) {
    $adaylar = @("$hedef\python\python.exe")
    foreach ($kok in @('HKCU', 'HKLM')) {
        try {
            $ip = (Get-ItemProperty "$($kok):\Software\Python\PythonCore\3.12\InstallPath" -ErrorAction Stop).'(default)'
            if ($ip) { $adaylar += (Join-Path $ip 'python.exe') }
        } catch {}
    }
    $adaylar += @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")
    foreach ($a in $adaylar) { if ($a -and (Test-Path $a)) { return $a } }
    try {
        $p = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path $p.Trim())) { return $p.Trim() }
    } catch {}
    return $null
}

function FFmpeg-Bul([string]$hedef) {
    foreach ($a in @("$hedef\ffmpeg\bin\ffmpeg.exe", 'C:\ffmpeg\bin\ffmpeg.exe', "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ffmpeg.exe")) {
        if (Test-Path $a) { return $a }
    }
    $c = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $w = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter ffmpeg.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($w) { return $w.FullName }
    return $null
}

function Nvidia-Var {
    if (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe") { return $true }
    try { return [bool](Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }) } catch { return $false }
}

function LMStudio-Bul {
    foreach ($a in @("$env:LOCALAPPDATA\Programs\LM Studio\LM Studio.exe", "$env:LOCALAPPDATA\Programs\lm-studio\LM Studio.exe",
                     "$env:ProgramFiles\LM Studio\LM Studio.exe")) { if (Test-Path $a) { return $a } }
    return $null
}

function Model-Var {
    $d = "$env:USERPROFILE\.lmstudio\models"
    if (-not (Test-Path $d)) { return $false }
    return [bool](Get-ChildItem $d -Filter *.gguf -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '(?i)qwen' } | Select-Object -First 1)
}

function Premiere-Bul {
    try {
        $p = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Adobe Premiere Pro.exe' -ErrorAction Stop).'(default)'
        if ($p -and (Test-Path $p)) { return $p }
    } catch {}
    $g = Get-ChildItem "$env:ProgramFiles\Adobe" -Directory -Filter 'Adobe Premiere Pro*' -ErrorAction SilentlyContinue |
         Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'Adobe Premiere Pro.exe' } | Where-Object { Test-Path $_ } | Select-Object -First 1
    return $g
}

function Kopru-Baska-Yerde([string]$hedef) {
    # Premiere köprü eklentisi zaten başka bir Kurgu Masası klasörüne bağlıysa (ör. geliştiricinin kendi kurulumu)
    $f = "$env:APPDATA\Adobe\CEP\extensions\MCPBridgeCEP\bridge_dir.txt"
    if (-not (Test-Path $f)) { return $null }
    $d = (Get-Content $f -Raw -Encoding UTF8).Trim()
    if ($d -and (Test-Path $d) -and -not $d.StartsWith($hedef, [StringComparison]::OrdinalIgnoreCase)) { return $d }
    return $null
}

# ------------------------------------------------------------------ arayüz
$font = New-Object System.Drawing.Font('Segoe UI', 10)
$form = New-Object System.Windows.Forms.Form
$form.Text = "$AppAdi - Kurulum"
$form.Size = New-Object System.Drawing.Size(820, 720)
$form.StartPosition = 'CenterScreen'
$form.Font = $font
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$ico = Join-Path $PSScriptRoot 'simge.ico'
if (Test-Path $ico) { $form.Icon = New-Object System.Drawing.Icon($ico) }

$baslik = New-Object System.Windows.Forms.Label
$baslik.Text = "$AppAdi kurulumu"
$baslik.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 16)
$baslik.Location = '20,14'; $baslik.AutoSize = $true
$form.Controls.Add($baslik)

$aciklama = New-Object System.Windows.Forms.Label
$aciklama.Text = "Bilgisayarındaki eksikleri buldum. 'Kur'a basınca eksik ve ücretsiz olanlar internetten indirilip kurulur. " +
                 "İnternet hızına göre 10-60 dakika sürebilir; bu sırada bilgisayarı kapatma."
$aciklama.Location = '22,52'; $aciklama.Size = '770,40'
$form.Controls.Add($aciklama)

$lblYer = New-Object System.Windows.Forms.Label
$lblYer.Text = 'Kurulum klasörü:'; $lblYer.Location = '22,100'; $lblYer.AutoSize = $true
$form.Controls.Add($lblYer)
$txtYer = New-Object System.Windows.Forms.TextBox
$txtYer.Location = '140,97'; $txtYer.Size = '540,26'
$txtYer.Text = (Join-Path $env:USERPROFILE 'Kurgu Masasi')
$form.Controls.Add($txtYer)
$btnYer = New-Object System.Windows.Forms.Button
$btnYer.Text = 'Değiştir'; $btnYer.Location = '690,95'; $btnYer.Size = '100,30'
$form.Controls.Add($btnYer)

$liste = New-Object System.Windows.Forms.ListView
$liste.View = 'Details'; $liste.FullRowSelect = $true; $liste.HeaderStyle = 'Nonclickable'
$liste.Location = '22,136'; $liste.Size = '768,206'
[void]$liste.Columns.Add('Bileşen', 210)
[void]$liste.Columns.Add('Durum', 190)
[void]$liste.Columns.Add('Ne yapılacak', 360)
$form.Controls.Add($liste)

$chkLm = New-Object System.Windows.Forms.CheckBox
$chkLm.Text = 'LM Studio (ücretsiz yerel yapay zekâ: Shorts özeti, yazılar) kurulsun'
$chkLm.Location = '22,350'; $chkLm.Size = '770,24'; $chkLm.Checked = $true
$form.Controls.Add($chkLm)
$chkModel = New-Object System.Windows.Forms.CheckBox
$chkModel.Text = 'Dil modelini de indir (Qwen3 8B, yaklaşık 5 GB)'
$chkModel.Location = '22,376'; $chkModel.Size = '770,24'; $chkModel.Checked = $true
$form.Controls.Add($chkModel)
$chkKopru = New-Object System.Windows.Forms.CheckBox
$chkKopru.Text = "Premiere Pro köprüsünü bu kuruluma bağla"
$chkKopru.Location = '22,402'; $chkKopru.Size = '770,24'; $chkKopru.Checked = $true
$form.Controls.Add($chkKopru)
$chkKisayol = New-Object System.Windows.Forms.CheckBox
$chkKisayol.Text = 'Masaüstüne kısayol koy'
$chkKisayol.Location = '22,428'; $chkKisayol.Size = '770,24'; $chkKisayol.Checked = $true
$form.Controls.Add($chkKisayol)

$lnkAdobe = New-Object System.Windows.Forms.LinkLabel
$lnkAdobe.Text = "Premiere Pro'yu denemek istersen (ücretli, 7 gün deneme): adobe.com"
$lnkAdobe.Location = '22,456'; $lnkAdobe.Size = '770,22'; $lnkAdobe.Visible = $false
$lnkAdobe.add_LinkClicked({ Start-Process 'https://www.adobe.com/products/premiere.html' })
$form.Controls.Add($lnkAdobe)

$bar = New-Object System.Windows.Forms.ProgressBar
$bar.Location = '22,484'; $bar.Size = '768,22'; $bar.Minimum = 0; $bar.Maximum = 1000
$form.Controls.Add($bar)
$lblAdim = New-Object System.Windows.Forms.Label
$lblAdim.Location = '22,510'; $lblAdim.Size = '768,22'; $lblAdim.Text = ''
$form.Controls.Add($lblAdim)
$kayit = New-Object System.Windows.Forms.TextBox
$kayit.Multiline = $true; $kayit.ScrollBars = 'Vertical'; $kayit.ReadOnly = $true
$kayit.Location = '22,536'; $kayit.Size = '768,92'
$kayit.Font = New-Object System.Drawing.Font('Consolas', 9)
$form.Controls.Add($kayit)

$btnKur = New-Object System.Windows.Forms.Button
$btnKur.Text = 'Kur'; $btnKur.Location = '560,638'; $btnKur.Size = '110,34'
$form.Controls.Add($btnKur)
$btnKapat = New-Object System.Windows.Forms.Button
$btnKapat.Text = 'Kapat'; $btnKapat.Location = '680,638'; $btnKapat.Size = '110,34'
$form.Controls.Add($btnKapat)
$btnAc = New-Object System.Windows.Forms.Button
$btnAc.Text = "$AppAdi'nı aç"; $btnAc.Location = '380,638'; $btnAc.Size = '170,34'; $btnAc.Visible = $false
$form.Controls.Add($btnAc)
$form.AcceptButton = $btnKur

$durum = @{}
function Tara {
    $hedef = $txtYer.Text.Trim()
    $durum.py = Python312-Bul $hedef
    $durum.ff = FFmpeg-Bul $hedef
    $durum.nv = Nvidia-Var
    $durum.lm = LMStudio-Bul
    $durum.model = Model-Var
    $durum.pr = Premiere-Bul
    $durum.venv = Test-Path "$hedef\.venv\Scripts\python.exe"
    $durum.kopruBaska = Kopru-Baska-Yerde $hedef
    try { $durum.bosGB = [math]::Round((Get-PSDrive ($hedef.Substring(0, 1))).Free / 1GB, 1) } catch { $durum.bosGB = $null }

    $liste.Items.Clear()
    function Satir($a, $b, $c) { $i = New-Object System.Windows.Forms.ListViewItem($a); [void]$i.SubItems.Add($b); [void]$i.SubItems.Add($c); [void]$liste.Items.Add($i) }
    Satir 'Python 3.12' ($(if ($durum.py) { 'Var' } else { 'Yok' })) ($(if ($durum.py) { 'Kullanılacak' } else { 'İndirilip kurulacak (ücretsiz, ~25 MB)' }))
    Satir 'FFmpeg (video işleme)' ($(if ($durum.ff) { 'Var' } else { 'Yok' })) ($(if ($durum.ff) { 'Kullanılacak' } else { 'İndirilecek (ücretsiz, ~100 MB)' }))
    Satir 'Yapay zekâ paketleri' ($(if ($durum.venv) { 'Var (onarılacak)' } else { 'Yok' })) 'Whisper, MediaPipe, OpenCV... (~2-3 GB)'
    Satir 'NVIDIA ekran kartı' ($(if ($durum.nv) { 'Var' } else { 'Yok' })) ($(if ($durum.nv) { 'Hızlı GPU kitaplıkları kurulacak' } else { 'Yazıya dökme işlemci ile (daha yavaş)' }))
    Satir 'Whisper konuşma modeli' 'İlk kurulumda yok' 'İndirilecek (ücretsiz, ~1,6 GB)'
    Satir 'LM Studio' ($(if ($durum.lm) { 'Var' } else { 'Yok' })) ($(if ($durum.lm) { 'Kullanılacak' } else { 'İsteğe bağlı: indirilecek (ücretsiz)' }))
    Satir 'Dil modeli (Qwen3 8B)' ($(if ($durum.model) { 'Var' } else { 'Yok' })) ($(if ($durum.model) { 'Kullanılacak' } else { 'İsteğe bağlı: indirilecek (~5 GB)' }))
    Satir 'Adobe Premiere Pro' ($(if ($durum.pr) { 'Var' } else { 'Yok (ücretli)' })) ($(if ($durum.pr) { 'Köprü eklentisi kurulacak' } else { 'Gerekmez: uygulama kendi kurgu motoruyla çalışır' }))
    if ($durum.bosGB -ne $null) { Satir 'Boş disk alanı' "$($durum.bosGB) GB" ($(if ($durum.bosGB -lt 20) { 'AZ: en az 20 GB önerilir' } else { 'Yeterli' })) }

    $chkLm.Enabled = -not $durum.lm; if ($durum.lm) { $chkLm.Checked = $false; $chkLm.Text = 'LM Studio zaten kurulu' }
    $chkModel.Enabled = -not $durum.model; if ($durum.model) { $chkModel.Checked = $false; $chkModel.Text = 'Dil modeli zaten var' }
    $chkKopru.Visible = [bool]$durum.pr
    if ($durum.kopruBaska) {
        $chkKopru.Checked = $false
        $chkKopru.Text = "Premiere köprüsünü bu kuruluma bağla (şu an bağlı olduğu yer: $($durum.kopruBaska))"
    }
    $lnkAdobe.Visible = -not $durum.pr
}

$btnYer.add_Click({
    $d = New-Object System.Windows.Forms.FolderBrowserDialog
    $d.Description = 'Kurgu Masası nereye kurulsun? (Projeler ve videolar da bu klasörde durur.)'
    if ($d.ShowDialog() -eq 'OK') { $txtYer.Text = (Join-Path $d.SelectedPath 'Kurgu Masasi'); Tara }
})

# ------------------------------------------------------------------ arka planda kurulum
$sync = [hashtable]::Synchronized(@{ Satirlar = [System.Collections.ArrayList]::Synchronized((New-Object System.Collections.ArrayList)); Yuzde = 0.0; Adim = ''; Bitti = $false; Tamam = $false
                                     Uyarilar = [System.Collections.ArrayList]::Synchronized((New-Object System.Collections.ArrayList)); LmSayfa = $false })

$is = {
    param($sync, $p)
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
    $hedef = $p.Hedef
    $logDosya = $null

    function Yaz([string]$t) {
        [void]$sync.Satirlar.Add($t)
        if ($script:logDosya) { try { Add-Content -Path $script:logDosya -Value $t -Encoding UTF8 } catch {} }
    }
    function Adim([string]$t, [double]$y) { $sync.Adim = $t; $sync.Yuzde = $y; Yaz ""; Yaz "=== $t ===" }
    function Uyar([string]$t) { [void]$sync.Uyarilar.Add($t); Yaz "  [!] $t" }

    function Indir([string]$url, [string]$cikis, [string]$ad, [double]$y0, [double]$y1, [long]$toplam = 0) {
        New-Item -ItemType Directory -Force -Path (Split-Path $cikis) | Out-Null
        Yaz "  İndiriliyor: $ad"
        $curl = "$env:SystemRoot\System32\curl.exe"
        if (Test-Path $curl) {
            $pr = Start-Process -FilePath $curl -ArgumentList @('-L', '-f', '-sS', '--retry', '5', '--retry-delay', '3', '-C', '-', '-o', "`"$cikis`"", "`"$url`"") -PassThru -WindowStyle Hidden
            $null = $pr.Handle   # ExitCode is only kept when the handle was opened
            while (-not $pr.HasExited) {
                Start-Sleep -Milliseconds 700
                if ($toplam -gt 0 -and (Test-Path $cikis)) {
                    $n = (Get-Item $cikis).Length
                    $sync.Yuzde = $y0 + ($y1 - $y0) * [math]::Min(1.0, $n / $toplam)
                    $sync.Adim = "$ad - $([math]::Round($n / 1MB)) / $([math]::Round($toplam / 1MB)) MB"
                }
            }
            if ($pr.ExitCode -ne 0) { throw "$ad indirilemedi (curl kodu $($pr.ExitCode))" }
        } else {
            Invoke-WebRequest -Uri $url -OutFile $cikis -UseBasicParsing
        }
        if (-not (Test-Path $cikis) -or (Get-Item $cikis).Length -lt 1000) { throw "$ad indirilemedi" }
        $sync.Yuzde = $y1
    }

    function Calistir([string]$exe, [string[]]$argv) {
        # komutun çıktısını satır satır kayda yazar; çıkış kodunu döndürür (stderr satırları hata sayılmaz)
        $ErrorActionPreference = 'Continue'
        & $exe @argv 2>&1 | ForEach-Object { $s = "$_"; if ($s.Trim()) { Yaz "    $s" } }
        return $LASTEXITCODE
    }

    try {
        # 1. uygulama dosyaları (projeler, ham videolar ve ayarlar korunur)
        Adim 'Uygulama dosyaları kopyalanıyor' 0.02
        New-Item -ItemType Directory -Force -Path $hedef | Out-Null
        $script:logDosya = Join-Path $hedef 'kurulum.log'
        Set-Content -Path $script:logDosya -Value "Kurgu Masası kurulumu $(Get-Date)" -Encoding UTF8
        # güncelleme: bu klasörden çalışan eski uygulamayı kapat (dosyalar ve paketler değişecek)
        Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($hedef, [StringComparison]::OrdinalIgnoreCase) } |
            ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force; Yaz '  Çalışan eski sürüm kapatıldı.' } catch {} }
        if (Test-Path "$hedef\kurgu") { Remove-Item "$hedef\kurgu" -Recurse -Force }   # eski kod dosyaları kalmasın
        $haric = @('user_settings.json', 'dictionary.json') | Where-Object { Test-Path (Join-Path $hedef $_) }   # kullanıcının ayarları korunur
        $ra = @("`"$($p.Kaynak)`"", "`"$hedef`"", '/E', '/R:2', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS', '/NP')
        if ($haric) { $ra += '/XF'; $ra += $haric }
        $rc = (Start-Process -FilePath "$env:SystemRoot\System32\robocopy.exe" -ArgumentList $ra -PassThru -Wait -WindowStyle Hidden).ExitCode
        if ($rc -ge 8) { throw "Uygulama dosyaları kopyalanamadı (robocopy $rc)" }
        foreach ($k in @('ham', 'muzik', 'ses_efektleri', 'kopru', 'logs', 'Videolarım')) { New-Item -ItemType Directory -Force -Path (Join-Path $hedef $k) | Out-Null }
        Yaz "  Klasör: $hedef"

        # 2. Python 3.12
        Adim 'Python 3.12' 0.05
        $py = $p.Python
        if (-not $py) {
            $kur = Join-Path $env:TEMP 'python-3.12.10-amd64.exe'
            try {
                Remove-Item $kur -Force -ErrorAction SilentlyContinue
                Indir $p.PyUrl $kur 'Python 3.12.10' 0.05 0.09 26000000
                Yaz '  Python kuruluyor (sessiz, yalnızca bu kullanıcıya)...'
                $a = @('/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0', 'Include_test=0', 'Include_doc=0',
                       'Shortcuts=0', 'AssociateFiles=0', 'Include_tcltk=1', 'Include_pip=1', "TargetDir=`"$hedef\python`"")
                $pr = Start-Process -FilePath $kur -ArgumentList $a -PassThru -Wait
                if ($pr.ExitCode -ne 0) { Yaz "  Python kurulum kodu: $($pr.ExitCode)" }
            } catch { Uyar "Python indirilemedi: $($_.Exception.Message)" }
            foreach ($a in @("$hedef\python\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")) { if (Test-Path $a) { $py = $a; break } }
            if (-not $py -and (Get-Command winget -ErrorAction SilentlyContinue)) {
                Yaz '  winget ile deneniyor...'
                [void](Calistir 'winget' @('install', '--id', 'Python.Python.3.12', '-e', '--scope', 'user', '--silent', '--accept-package-agreements', '--accept-source-agreements'))
                if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") { $py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
            }
            if (-not $py) { throw 'Python 3.12 kurulamadı. İnternet bağlantısını kontrol edip kurulumu yeniden çalıştırın.' }
        }
        Yaz "  Python: $py"

        # 3. FFmpeg
        Adim 'FFmpeg' 0.10
        if (-not $p.FFmpeg) {
            $zip = Join-Path $env:TEMP 'ffmpeg-release-essentials.zip'
            try {
                Remove-Item $zip -Force -ErrorAction SilentlyContinue
                Indir $p.FfUrl $zip 'FFmpeg' 0.10 0.16 105000000
                $tmp = Join-Path $env:TEMP ("ffmpeg_ac_" + [guid]::NewGuid().ToString('N'))
                Expand-Archive -Path $zip -DestinationPath $tmp -Force
                $bin = Get-ChildItem $tmp -Filter ffmpeg.exe -Recurse | Select-Object -First 1
                New-Item -ItemType Directory -Force -Path "$hedef\ffmpeg\bin" | Out-Null
                Copy-Item (Join-Path $bin.DirectoryName '*') "$hedef\ffmpeg\bin" -Force
                Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
                Remove-Item $zip -Force -ErrorAction SilentlyContinue
                Yaz "  FFmpeg: $hedef\ffmpeg\bin"
            } catch {
                Uyar "FFmpeg indirilemedi: $($_.Exception.Message)"
                if (Get-Command winget -ErrorAction SilentlyContinue) {
                    Yaz '  winget ile deneniyor...'
                    [void](Calistir 'winget' @('install', '--id', 'Gyan.FFmpeg', '-e', '--silent', '--accept-package-agreements', '--accept-source-agreements'))
                }
            }
        } else { Yaz "  Var: $($p.FFmpeg)" }

        # 4. Python ortamı ve paketler
        Adim 'Yapay zekâ paketleri (en uzun adım)' 0.18
        $vpy = "$hedef\.venv\Scripts\python.exe"
        if (-not (Test-Path $vpy)) {
            if ((Calistir $py @('-m', 'venv', "$hedef\.venv")) -ne 0 -or -not (Test-Path $vpy)) { throw 'Python ortamı (.venv) oluşturulamadı.' }
        }
        $pipOrtak = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', '--prefer-binary')
        [void](Calistir $vpy (@('-m', 'pip', 'install', '--upgrade', '--disable-pip-version-check', 'pip', 'wheel')))
        $sync.Yuzde = 0.22
        if ((Calistir $vpy ($pipOrtak + @('-r', "$hedef\kurulum\gereksinimler.txt"))) -ne 0) {
            Yaz '  Bir kez daha deneniyor...'
            if ((Calistir $vpy ($pipOrtak + @('-r', "$hedef\kurulum\gereksinimler.txt"))) -ne 0) { throw 'Python paketleri kurulamadı (ayrıntı: kurulum.log).' }
        }
        $sync.Yuzde = 0.45
        if ($p.Nvidia) {
            Yaz '  NVIDIA kitaplıkları kuruluyor...'
            if ((Calistir $vpy ($pipOrtak + @('-r', "$hedef\kurulum\gereksinimler-gpu.txt"))) -ne 0) { Uyar 'GPU kitaplıkları kurulamadı; Whisper işlemciyle çalışır.' }
        }
        $sync.Yuzde = 0.55

        # 5. Whisper modeli (ilk videoda beklememek için şimdi)
        Adim 'Whisper konuşma modeli' 0.56
        $kod = "from faster_whisper import download_model; print(download_model('$($p.Whisper)'))"
        if ((Calistir $vpy @('-c', $kod)) -ne 0) { Uyar 'Whisper modeli şimdi indirilemedi; ilk videoda kendiliğinden indirilir.' }
        $sync.Yuzde = 0.62

        # 6. LM Studio
        if ($p.LmKur) {
            Adim 'LM Studio' 0.63
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                [void](Calistir 'winget' @('install', '--id', 'ElementLabs.LMStudio', '-e', '--silent', '--accept-package-agreements', '--accept-source-agreements'))
            }
            $lmv = $false
            foreach ($a in @("$env:LOCALAPPDATA\Programs\LM Studio\LM Studio.exe", "$env:LOCALAPPDATA\Programs\lm-studio\LM Studio.exe", "$env:ProgramFiles\LM Studio\LM Studio.exe")) { if (Test-Path $a) { $lmv = $true } }
            if ($lmv) { Yaz '  LM Studio kuruldu.' } else {
                Uyar 'LM Studio otomatik kurulamadı. Kurulum bitince açılan sayfadan ücretsiz indirip kurun (isteğe bağlı).'
                $sync.LmSayfa = $true
            }
        }

        # 7. Dil modeli (GGUF, LM Studio'nun model klasörüne)
        if ($p.ModelKur) {
            Adim 'Dil modeli (Qwen3 8B)' 0.66
            $indi = $false
            foreach ($repo in $p.Repolar) {
                try {
                    $api = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$($repo)?blobs=true" -UseBasicParsing
                    $dosya = $api.siblings | Where-Object { $_.rfilename -match '(?i)Q4_K_M\.gguf$' } | Select-Object -First 1
                    if (-not $dosya) { continue }
                    $cikis = Join-Path "$env:USERPROFILE\.lmstudio\models\$($repo -replace '/', '\')" $dosya.rfilename
                    $boyut = [long]($dosya.size)
                    if ((Test-Path $cikis) -and $boyut -gt 0 -and (Get-Item $cikis).Length -eq $boyut) { Yaz '  Zaten indirilmiş.'; $indi = $true; break }
                    Indir "https://huggingface.co/$repo/resolve/main/$($dosya.rfilename)" $cikis "Qwen3 8B ($($dosya.rfilename))" 0.66 0.94 $boyut
                    $indi = $true; break
                } catch { Yaz "  $repo : $($_.Exception.Message)" }
            }
            if (-not $indi) { Uyar 'Dil modeli indirilemedi. LM Studio içinden "Qwen3 8B" aratıp indirebilirsiniz (isteğe bağlı).' }
        }

        # 8. Premiere köprüsü
        if ($p.Premiere -and $p.KopruBagla) {
            Adim 'Premiere Pro köprü eklentisi' 0.95
            $cep = "$env:APPDATA\Adobe\CEP\extensions\MCPBridgeCEP"
            New-Item -ItemType Directory -Force -Path $cep | Out-Null
            Copy-Item (Join-Path $hedef 'kurulum\premiere-kopru\*') $cep -Recurse -Force
            [IO.File]::WriteAllText((Join-Path $cep 'bridge_dir.txt'), (Join-Path $hedef 'kopru'), (New-Object Text.UTF8Encoding $false))
            foreach ($n in 9..16) {
                $k = "HKCU:\Software\Adobe\CSXS.$n"
                New-Item -Path $k -Force | Out-Null
                New-ItemProperty -Path $k -Name 'PlayerDebugMode' -Value '1' -PropertyType String -Force | Out-Null
            }
            [IO.File]::WriteAllText((Join-Path $hedef 'premiere_yolu.txt'), $p.Premiere, (New-Object Text.UTF8Encoding $false))
            Yaz "  Eklenti: $cep"
        }

        # 9. kısayollar ve Programlar listesi
        Adim 'Kısayollar' 0.97
        $pyw = "$hedef\.venv\Scripts\pythonw.exe"
        $ws = New-Object -ComObject WScript.Shell
        $yerler = @([Environment]::GetFolderPath('Programs'))
        if ($p.Kisayol) { $yerler += [Environment]::GetFolderPath('Desktop') }
        foreach ($y in $yerler) {
            $l = $ws.CreateShortcut((Join-Path $y 'Kurgu Masası.lnk'))
            $l.TargetPath = $pyw; $l.Arguments = '-m kurgu.desktop'; $l.WorkingDirectory = $hedef
            $l.IconLocation = "$hedef\kurulum\simge.ico"; $l.Description = 'Kurgu Masası - otomatik video kurgu'
            $l.Save()
        }
        $u = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\KurguMasasi'
        New-Item -Path $u -Force | Out-Null
        $deg = @{ DisplayName = 'Kurgu Masası'; DisplayIcon = "$hedef\kurulum\simge.ico"; Publisher = 'Kurgu Masası'; DisplayVersion = $p.Surum
                  InstallLocation = $hedef
                  UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File `"$hedef\kurulum\kaldir.ps1`"" }
        foreach ($k in $deg.Keys) { New-ItemProperty -Path $u -Name $k -Value $deg[$k] -PropertyType String -Force | Out-Null }
        New-ItemProperty -Path $u -Name 'NoModify' -Value 1 -PropertyType DWord -Force | Out-Null
        New-ItemProperty -Path $u -Name 'NoRepair' -Value 1 -PropertyType DWord -Force | Out-Null

        # 10. doğrulama
        Adim 'Doğrulama' 0.98
        $kod = Calistir $vpy @("$hedef\kurulum\dogrula.py")
        if ($kod -ne 0) { throw 'Doğrulamada eksik bileşen var (yukarıdaki [X] satırlarına bakın).' }
        $sync.Yuzde = 1.0
        $sync.Adim = 'Kurulum tamamlandı'
        $sync.Tamam = $true
    } catch {
        Yaz ""
        Yaz "HATA: $($_.Exception.Message)"
        $sync.Adim = "Hata: $($_.Exception.Message)"
    } finally {
        $sync.Bitti = $true
    }
}

$zamanlayici = New-Object System.Windows.Forms.Timer
$zamanlayici.Interval = 300
$script:ps = $null
$zamanlayici.add_Tick({
    while ($sync.Satirlar.Count -gt 0) {
        $s = $sync.Satirlar[0]; $sync.Satirlar.RemoveAt(0)
        $kayit.AppendText($s + "`r`n")
    }
    $bar.Value = [int]([math]::Min(1.0, [math]::Max(0.0, $sync.Yuzde)) * 1000)
    $lblAdim.Text = $sync.Adim
    if ($sync.Bitti) {
        $zamanlayici.Stop()
        $btnKapat.Enabled = $true
        if ($sync.Tamam) {
            $btnAc.Visible = $true
            $msg = "$AppAdi kuruldu."
            if ($sync.Uyarilar.Count) { $msg += "`n`nUyarılar:`n- " + ($sync.Uyarilar -join "`n- ") }
            if ($durum.pr -and $chkKopru.Checked) { $msg += "`n`nPremiere ilk açılışta: Pencere (Window) > Uzantılar (Extensions) > MCP Bridge panelini bir kez açın; sonra kendiliğinden çalışır." }
            if ($sync.LmSayfa) { Start-Process 'https://lmstudio.ai' }
            [void][System.Windows.Forms.MessageBox]::Show($msg, $AppAdi, 'OK', 'Information')
        } else {
            $btnKur.Enabled = $true; $btnKur.Text = 'Yeniden dene'
            [void][System.Windows.Forms.MessageBox]::Show("Kurulum tamamlanamadı:`n$($sync.Adim)`n`nAyrıntı: $($txtYer.Text)\kurulum.log", $AppAdi, 'OK', 'Error')
        }
    }
})

$btnKur.add_Click({
    $hedef = $txtYer.Text.Trim()
    if (-not $hedef) { return }
    Tara
    if ($durum.bosGB -ne $null -and $durum.bosGB -lt 12) {
        $r = [System.Windows.Forms.MessageBox]::Show("Diskte yalnızca $($durum.bosGB) GB boş yer var. Kurulum ve videolar için en az 20 GB önerilir. Yine de devam edilsin mi?", $AppAdi, 'YesNo', 'Warning')
        if ($r -ne 'Yes') { return }
    }
    $btnKur.Enabled = $false; $btnKapat.Enabled = $false; $btnYer.Enabled = $false; $txtYer.Enabled = $false
    $chkLm.Enabled = $false; $chkModel.Enabled = $false; $chkKopru.Enabled = $false; $chkKisayol.Enabled = $false
    $sync.Bitti = $false; $sync.Tamam = $false; $sync.Uyarilar.Clear()
    $prm = @{
        Hedef = $hedef; Kaynak = (Join-Path $Kaynak 'uygulama'); Python = $durum.py; FFmpeg = $durum.ff; Nvidia = $durum.nv
        PyUrl = $PyUrl; FfUrl = $FfUrl; Whisper = $WhisperModel; Repolar = $ModelRepolari
        LmKur = ($chkLm.Checked -and -not $durum.lm); ModelKur = ($chkModel.Checked -and -not $durum.model)
        Premiere = $durum.pr; KopruBagla = $chkKopru.Checked; Kisayol = $chkKisayol.Checked
        Surum = (Get-Content (Join-Path $Kaynak 'uygulama\kurulum\surum.txt') -ErrorAction SilentlyContinue | Select-Object -First 1)
    }
    $rs = [runspacefactory]::CreateRunspace()
    $rs.ApartmentState = 'STA'; $rs.Open()
    $script:ps = [powershell]::Create()
    $script:ps.Runspace = $rs
    [void]$script:ps.AddScript($is).AddArgument($sync).AddArgument($prm)
    [void]$script:ps.BeginInvoke()
    $zamanlayici.Start()
})

$btnAc.add_Click({
    $hedef = $txtYer.Text.Trim()
    Start-Process -FilePath "$hedef\.venv\Scripts\pythonw.exe" -ArgumentList '-m', 'kurgu.desktop' -WorkingDirectory $hedef
    $form.Close()
})
$btnKapat.add_Click({ $form.Close() })
$form.add_FormClosing({
    param($s, $e)
    if ($script:ps -and -not $sync.Bitti) {
        $r = [System.Windows.Forms.MessageBox]::Show('Kurulum sürüyor. Yarıda kesilsin mi? (Sonra yeniden çalıştırıp devam edebilirsiniz.)', $AppAdi, 'YesNo', 'Warning')
        if ($r -ne 'Yes') { $e.Cancel = $true } else { try { $script:ps.Stop() } catch {} }
    }
})
$txtYer.add_Leave({ Tara })

Tara
[void]$form.ShowDialog()
