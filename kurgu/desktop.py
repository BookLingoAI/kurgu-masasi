"""Desktop launcher: `pythonw -m kurgu.desktop` (what the Start menu / desktop shortcut runs).

Starts the local server in the background if it is not running yet (jobs keep running when the window is closed),
then shows the app in its own window: Microsoft Edge (or Chrome) in "app" mode - no tabs, no address bar - with a
separate profile, so it looks and behaves like a normal desktop program. Falls back to the default browser.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8770
URL = f"http://localhost:{PORT}/"
NOWIN = 0x08000000 if os.name == "nt" else 0


def server_up(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(URL + "api/projects", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_server() -> None:
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe") if exe.name.lower() == "python.exe" else exe
    flags = (0x00000008 | 0x00000200 | NOWIN) if os.name == "nt" else 0   # detached, own group, no window
    subprocess.Popen([str(pyw), "-m", "kurgu"], cwd=str(ROOT), close_fds=True, creationflags=flags,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _browser_app() -> str | None:
    pf = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
    # Chrome first (user, 23.10.2026: "chromu kullan her zaman"); Edge only when Chrome is not installed
    rel = [r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe"]
    for r in rel:
        for base in pf:
            if base and (Path(base) / r).exists():
                return str(Path(base) / r)
    return None


def open_window() -> None:
    exe = _browser_app()
    if exe:
        profile = ROOT / "logs" / "pencere"
        profile.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([exe, f"--app={URL}", f"--user-data-dir={profile}", "--window-size=1440,920",
                          "--no-first-run", "--no-default-browser-check", "--disable-features=Translate"],
                         close_fds=True)
    else:
        webbrowser.open(URL)


def _message(text: str) -> None:
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "Kurgu Masası", 0x10)
    else:
        print(text)


def main() -> int:
    if not server_up():
        start_server()
        for _ in range(120):      # first start can take a while (antivirus scans the new Python files)
            time.sleep(0.5)
            if server_up():
                break
        else:
            log = sorted((ROOT / "logs").glob("sunucu-*.log"))
            _message("Kurgu Masası başlatılamadı.\n\nAyrıntı: " + (str(log[-1]) if log else "logs klasörü") +
                     "\n\nSorun sürerse kurulumu yeniden çalıştırın (Onar).")
            return 1
    start_room()
    open_window()
    return 0


def start_room() -> None:
    """The board room (kurgu/board/room.py, port 8771) runs beside the app: it starts a meeting by itself when a new
    project's video is ready and Premiere is open, and opens its own window for it."""
    try:
        from .board.room import server_up as room_up
        if room_up():
            return
        exe = Path(sys.executable)
        pyw = exe.with_name("pythonw.exe") if exe.name.lower() == "python.exe" else exe
        flags = (0x00000008 | 0x00000200 | NOWIN) if os.name == "nt" else 0
        args = dict(cwd=str(ROOT), close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
        try:    # leave the starter's job object, or the room dies with whatever started it (seen 22.10.2026)
            subprocess.Popen([str(pyw), "-m", "kurgu.board.room"], creationflags=flags | 0x01000000, **args)
        except OSError:   # the job does not allow breaking away: start it the plain way
            subprocess.Popen([str(pyw), "-m", "kurgu.board.room"], creationflags=flags, **args)
    except Exception:  # noqa: BLE001 - the app itself must open even if the room cannot
        pass


if __name__ == "__main__":
    sys.exit(main())
