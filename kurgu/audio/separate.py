"""Voice / background-music separation with Demucs (Meta, MIT licence) - optional.

Phone and camera microphones pick up venue music almost as loud as the voice; a denoiser cannot remove music under
speech (YouTube Content ID would still match it). Demucs splits the recording into "vocals" and the rest; the vocals
file is then used everywhere audio is read: energy/pause analysis, Whisper, and the final mix.

- Installed on request (UI button -> job "install_separator": PyTorch CUDA + demucs into the app's .venv).
- Runs in a separate Python process (GPU memory and DLL problems cannot take the server down), in 60 s windows
  with overlap, so a one-hour recording needs little RAM.
- Results are cached per source file (path + size + mtime) in <root>/cache/vokal/ and reused by every project.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "Videolarım" / "_ortak" / "vokal"   # shared by the projects, inside Videolarım so one delete clears all
NOWIN = 0x08000000 if os.name == "nt" else 0
_state = {"enabled": False}


def installed() -> bool:
    return importlib.util.find_spec("demucs") is not None and importlib.util.find_spec("torch") is not None


def configure(enabled: bool) -> None:
    """Called at the start of every job: use separated voice files (when they exist) or the original audio."""
    _state["enabled"] = bool(enabled) and installed()


def enabled() -> bool:
    return _state["enabled"]


def _key(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|htdemucs-v1".encode()).hexdigest()[:16]


def cached_voice(path: str | Path) -> Path | None:
    try:
        f = CACHE / f"{_key(path)}.wav"
    except OSError:
        return None
    return f if f.exists() and f.stat().st_size > 1000 else None


def voice_file(path: str | Path) -> str:
    """The separated voice for this recording if separation is on and done, else the recording itself."""
    if not _state["enabled"]:
        return str(path)
    f = cached_voice(path)
    return str(f) if f else str(path)


def separate(paths: list[str], progress=None) -> dict:
    """Make sure every recording has a separated voice file. Returns {"done": n, "skipped": n, "device": ...}."""
    todo = [p for p in dict.fromkeys(paths) if Path(p).exists() and not cached_voice(p)]
    if not todo:
        return {"done": 0, "cached": len(paths), "device": None}
    CACHE.mkdir(parents=True, exist_ok=True)
    from ..integrations import ffmpeg as ff
    jobs = [{"src": p, "dst": str(CACHE / f"{_key(p)}.wav"), "ffmpeg": ff.ffmpeg_bin()} for p in todo]
    spec = CACHE / f"is_{os.getpid()}.json"
    spec.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":       # console python for the worker (output is read through a pipe)
        exe = exe.with_name("python.exe")
    proc = subprocess.Popen([str(exe), "-m", "kurgu.audio.separate_worker", str(spec)], cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=NOWIN, text=True, encoding="utf-8", errors="replace")
    device, tail = None, []
    for line in proc.stdout:
        line = line.strip()
        tail = (tail + [line])[-15:]
        if line.startswith("DEVICE "):
            device = line[7:]
        elif line.startswith("PROGRESS ") and progress:
            try:
                f, msg = line[9:].split(" ", 1)
                progress(float(f), msg)
            except ValueError:
                pass
    proc.wait()
    spec.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError("Müzik ayırma başarısız: " + " | ".join(tail[-4:]))
    return {"done": len(todo), "cached": len(paths) - len(todo), "device": device}
