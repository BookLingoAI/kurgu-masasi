"""Make the pip-installed NVIDIA DLLs (cuBLAS, cuDNN) visible on Windows before CTranslate2 loads.

The .venv activate scripts add them to PATH, but the app may be started without activation (bridge, scheduler).
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_done = False


def suppress_error_dialogs() -> None:
    """Windows: make DLL load failures (e.g. Smart App Control blocking cublas64_12.dll, 0xc0e90002) return an
    error to the caller instead of showing a modal 'Bad Image' dialog that freezes the process."""
    if os.name != "nt":
        return
    try:
        import ctypes

        SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SEM_NOOPENFILEERRORBOX = 0x0001, 0x0002, 0x8000
        k = ctypes.windll.kernel32
        k.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)
        try:
            k.SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX, None)
        except AttributeError:
            pass
    except Exception:  # noqa: BLE001
        pass


def ensure_cuda_dlls() -> list[str]:
    global _done
    suppress_error_dialogs()
    if _done or os.name != "nt":
        return []
    _done = True
    added = []
    roots = {Path(sysconfig.get_paths()["purelib"]), Path(sys.prefix) / "Lib" / "site-packages"}
    for root in roots:
        nv = root / "nvidia"
        if not nv.exists():
            continue
        for sub in nv.iterdir():
            b = sub / "bin"
            if b.is_dir():
                added.append(str(b))
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        for d in added:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
    return added
