"""Native Windows file picker, run as a separate process by the server (the app is local, so the user's own files
can be used where they are instead of being uploaded/copied through the browser).

usage: python picker.py <out.json> <kind> <initial_dir>
Writes {"files": [...]} (empty list when cancelled) or {"error": "..."}.
"""
from __future__ import annotations

import json
import sys

PATTERNS = {
    "video": [("Video", "*.mp4 *.mov *.mkv *.m4v *.avi *.mts *.MP4 *.MOV")],
    "ek": [("Video ve fotoğraf", "*.mp4 *.mov *.mkv *.m4v *.avi *.mts *.jpg *.jpeg *.png *.webp *.heic *.MP4 *.MOV *.JPG")],
    "ses": [("Ses", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg")],
}
TITLES = {"video": "Videoları seç", "ek": "Ek görüntüleri seç", "ses": "Müzik / ses seç"}


def _powershell(kind: str, init: str) -> dict | None:
    """Fallback when tkinter is missing: the standard Windows dialog through PowerShell (STA)."""
    import os
    import subprocess
    if os.name != "nt":
        return None
    flt = {"ek": "Video ve fotoğraf|*.mp4;*.mov;*.mkv;*.m4v;*.avi;*.mts;*.jpg;*.jpeg;*.png;*.webp",
           "ses": "Ses|*.mp3;*.wav;*.m4a;*.aac;*.flac;*.ogg"}.get(kind, "Video|*.mp4;*.mov;*.mkv;*.m4v;*.avi;*.mts")
    ps = ("Add-Type -AssemblyName System.Windows.Forms;"
          "$d=New-Object System.Windows.Forms.OpenFileDialog;$d.Multiselect=$true;"
          f"$d.Filter='{flt}|Tüm dosyalar|*.*';$d.InitialDirectory='{init.replace(chr(39), '')}';"
          "$f=New-Object System.Windows.Forms.Form;$f.TopMost=$true;"
          "if($d.ShowDialog($f) -eq 'OK'){[Console]::OutputEncoding=[Text.Encoding]::UTF8;$d.FileNames -join \"`n\"}")
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, timeout=1800,
                           creationflags=0x08000000)
        out = p.stdout.decode("utf-8", "replace").strip()
        return {"files": [x.strip() for x in out.splitlines() if x.strip()]}
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    out, kind, init = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        root.lift()
        root.focus_force()
        files = filedialog.askopenfilenames(parent=root, title=TITLES.get(kind, "Dosya seç"),
                                            initialdir=init or None,
                                            filetypes=PATTERNS.get(kind, PATTERNS["video"]) + [("Tüm dosyalar", "*.*")])
        root.destroy()
        import os
        res = {"files": [os.path.normpath(str(f)) for f in (files or ())]}
    except Exception as e:  # noqa: BLE001
        res = _powershell(kind, init) or {"error": f"{type(e).__name__}: {e}"}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
