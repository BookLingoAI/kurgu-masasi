"""Subprocess: Demucs voice separation for a list of recordings (see separate.py).

python -m kurgu.audio.separate_worker jobs.json
jobs.json = [{"src": video/audio path, "dst": voice .wav, "ffmpeg": ffmpeg.exe}, ...]
Prints "DEVICE x" and "PROGRESS <0..1> <message>" lines.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

NOWIN = 0x08000000 if os.name == "nt" else 0
WIN_S, CTX_S = 60.0, 3.0


def say(*a):
    print(*a, flush=True)


def main(spec: str) -> int:
    jobs = json.loads(Path(spec).read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from kurgu.integrations.cuda_env import suppress_error_dialogs
        suppress_error_dialogs()
    except Exception:  # noqa: BLE001
        pass
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say("DEVICE", dev)
    say("PROGRESS 0.0 Müzik ayırma modeli yükleniyor")
    model = get_model("htdemucs")
    model.to(dev).eval()
    sr = model.samplerate
    vi = model.sources.index("vocals")
    # GPU: one random time shift averaged in (cleaner voice, fewer "underwater" artefacts); CPU stays fast
    for j, job in enumerate(jobs):
        name = Path(job["src"]).name
        tmp = Path(tempfile.gettempdir()) / f"kurgu_sep_{os.getpid()}_{j}.f32"
        # decode once to a raw float file and memory-map it: an hour of audio needs almost no RAM
        p = subprocess.run([job["ffmpeg"], "-v", "error", "-nostdin", "-y", "-i", job["src"], "-vn", "-ac", "2",
                            "-ar", str(sr), "-f", "f32le", str(tmp)], creationflags=NOWIN)
        if p.returncode != 0 or not tmp.exists():
            say(f"HATA ses okunamadı: {name}")
            return 2
        x = np.memmap(tmp, dtype=np.float32, mode="r").reshape(-1, 2)
        n = len(x)
        if n < sr:
            del x
            tmp.unlink(missing_ok=True)
            continue
        mono = x.mean(axis=1)
        mean, std = float(mono.mean()), float(mono.std()) or 1.0
        del mono
        out_tmp = Path(job["dst"] + ".part")
        with wave.open(str(out_tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            L, C = int(WIN_S * sr), int(CTX_S * sr)
            s = 0
            while s < n:
                a, b = max(0, s - C), min(n, s + L + C)
                seg = torch.from_numpy(np.ascontiguousarray(x[a:b].T)).float()
                seg = (seg - mean) / std
                with torch.no_grad():
                    est = apply_model(model, seg[None].to(dev), shifts=1 if dev == "cuda" else 0, split=True, overlap=0.25,
                                      progress=False, device=dev)[0, vi]
                v = (est * std + mean).mean(dim=0).cpu().numpy()
                v = v[s - a: s - a + min(L, n - s)]
                w.writeframes((np.clip(v, -1, 1) * 32767).astype("<i2").tobytes())
                s += L
                f = (j + min(1.0, s / n)) / len(jobs)
                say(f"PROGRESS {f:.4f} Müzik ayrılıyor: {name} ({j + 1}/{len(jobs)}) %{min(100, int(s / n * 100))}")
        del x
        tmp.unlink(missing_ok=True)
        os.replace(out_tmp, job["dst"])
        if dev == "cuda":
            torch.cuda.empty_cache()
    say("PROGRESS 1.0 Müzik ayırma tamam")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
