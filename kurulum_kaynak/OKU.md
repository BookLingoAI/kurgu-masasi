# Kurulum kaynakları (Kurgu Masası Setup.exe)
- `kurulum.ps1`: kurulum sihirbazı (PowerShell 5.1 + WinForms). Eksikleri bulur; Python 3.12.10 (python.org),
  FFmpeg (gyan.dev essentials, yoksa winget), pip paketleri (gereksinimler*.txt), Whisper large-v3-turbo,
  LM Studio (winget ElementLabs.LMStudio), Qwen3 8B Q4_K_M GGUF (Hugging Face API ile dosya adı bulunur),
  Premiere varsa CEP köprüsü (+ bridge_dir.txt, PlayerDebugMode CSXS.9-16). Kısayol: `.venv\Scripts\pythonw.exe -m kurgu.desktop`.
- `main.go`: Setup.exe. payload.zip'i (kök: kurulum.ps1, simge.ico, uygulama\...) gömer, %TEMP%'e açar, sihirbazı çalıştırır.
  Derleme (Linux'ta): `python mksyso.py simge.ico rsrc_windows_amd64.syso` ve
  `GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags "-H windowsgui -s -w"`.
- payload\uygulama = kurgu\, assets\, modeller\, design_config.json, dictionary.json, user_settings.json (boş),
  kurulum\ (kaldir.ps1, dogrula.py, gereksinimler, simge.ico, surum.txt, premiere-kopru\ = araclar\premiere-pro-mcp\cep-plugin).
