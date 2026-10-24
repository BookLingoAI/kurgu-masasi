"""Start Premiere Pro when the bridge is needed and close it again (saving the project) if we opened it."""
from __future__ import annotations

import os
import subprocess
import time

from .premiere import BridgeNotRunning, PremiereError


def premiere_exe() -> str | None:
    from .premiere_edit import premiere_exe as _exe
    return _exe()


def ensure_premiere(pr, progress, wait_s: float = 240) -> bool:
    """True if we had to start Premiere."""
    try:
        pr.ping()
        return False
    except (BridgeNotRunning, PremiereError):
        pass
    exe = premiere_exe()
    if not exe:
        raise PremiereError("Premiere Pro bulunamadı")
    progress(0.01, "Premiere açılıyor (köprü paneli kendiliğinden başlar)")
    subprocess.Popen([exe], creationflags=0x00000008 if os.name == "nt" else 0)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(5)
        try:
            pr.ping()
            return True
        except (BridgeNotRunning, PremiereError):
            continue
    raise PremiereError("Premiere açıldı ama MCP Bridge paneli başlamadı (Window > Extensions > MCP Bridge bir kez açılmalı)")


def quit_premiere(pr) -> None:
    try:
        pr.run("app.project.save(); app.quit(); return KL.ok(true);", "quit", 60)
    except Exception:  # noqa: BLE001
        pass
