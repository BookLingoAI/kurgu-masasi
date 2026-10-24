"""Voices of the board members - all local, nothing leaves the computer.

Main engine: XTTS-v2 (Coqui, `modeller/xtts_v2`, Coqui Public Model License = non-commercial; the user's channel is
not commercial and the voices are only heard in the meetings, never published). Natural Turkish speech from its
built-in studio speakers - no real person's voice is cloned. During a meeting the GPU belongs to the language model,
so XTTS runs on the CPU (~1.5x real time) unless `ajanlar/sesler.json` says "cihaz": "cuda".
Fallback when XTTS cannot load: Piper `tr_TR-dfki-medium` + Windows "Microsoft Tolga", made distinct by pitch/pace.

The user picks the speakers in `ajanlar/sesler.json` ({"cihaz": "cpu", "sesler": {"hikaye": "Asya Anara", ...}}).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import wave
from pathlib import Path

from ..integrations import ffmpeg as ff

ROOT = Path(__file__).resolve().parents[2]
PIPER_MODEL = ROOT / "modeller" / "piper" / "tr_TR-dfki-medium.onnx"
XTTS_DIR = ROOT / "modeller" / "xtts_v2"
TOLGA_PS1 = Path(__file__).with_name("tolga.ps1")
SETTINGS = ROOT / "ajanlar" / "sesler.json"

# agent -> name, XTTS speaker and pace; the fallback voice (engine, pitch factor, pace, colour)
VOICES = {
    "hikaye":    {"isim": "Derya",  "xtts": "Asya Anara",       "x_hiz": 1.0,  "motor": "piper", "perde": 1.30, "hiz": 1.00, "renk": "highpass=f=140,treble=g=2"},
    "grafik":    {"isim": "Zeynep", "xtts": "Ana Florence",     "x_hiz": 1.0,  "motor": "tolga", "perde": 1.36, "hiz": 1.04, "renk": "highpass=f=160,treble=g=3"},
    "animasyon": {"isim": "Selin",  "xtts": "Gracie Wise",      "x_hiz": 1.08, "motor": "piper", "perde": 1.42, "hiz": 1.10, "renk": "highpass=f=180,treble=g=4"},
    "gecis":     {"isim": "Can",    "xtts": "Ilkin Urbano",     "x_hiz": 1.05, "motor": "piper", "perde": 1.00, "hiz": 1.12, "renk": "treble=g=1"},
    "ses":       {"isim": "Emre",   "xtts": "Eugenio Mataracı", "x_hiz": 0.97, "motor": "tolga", "perde": 0.90, "hiz": 0.95, "renk": "bass=g=3"},
    "renk":      {"isim": "Mert",   "xtts": "Damien Black",     "x_hiz": 1.0,  "motor": "tolga", "perde": 1.00, "hiz": 1.00, "renk": "anull"},
    "hakem":     {"isim": "Hakem",  "xtts": "Viktor Menelaos",  "x_hiz": 0.98, "motor": "tolga", "perde": 0.97, "hiz": 0.98, "renk": "anull"},
}
_piper = None
_xtts = None
_xtts_error: str | None = None
_lock = threading.Lock()


def settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def speaker_of(agent: str) -> str:
    return (settings().get("sesler") or {}).get(agent) or VOICES[agent]["xtts"]


def _load_xtts():
    """XTTS-v2 from modeller/xtts_v2. Two import workarounds, both for parts XTTS does not use here:
    coqui-tts refuses to import without torchcodec on PyTorch >= 2.9 (it is only for reading/writing audio FILES; we
    use built-in speakers and write the samples ourselves), and transformers pulls scikit-learn only for assisted
    generation, whose DLL Smart App Control blocks on this computer."""
    global _xtts, _xtts_error
    if _xtts is not None or _xtts_error:
        return _xtts
    try:
        import importlib.machinery
        import importlib.metadata
        import importlib.util
        import sys
        import types
        if "TTS" not in sys.modules:
            spec = importlib.util.find_spec("TTS")
            pkg = types.ModuleType("TTS")
            pkg.__path__, pkg.__spec__ = list(spec.submodule_search_locations), spec
            pkg.__version__ = importlib.metadata.version("coqui-tts")
            sys.modules["TTS"] = pkg
        try:
            import sklearn.metrics  # noqa: F401
        except Exception:  # noqa: BLE001 - blocked DLL: a stand-in for the unused assisted generation
            sk = types.ModuleType("sklearn.metrics")
            sk.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", None)
            sk.roc_curve = None
            sys.modules["sklearn.metrics"] = sk
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        cfg = XttsConfig()
        cfg.load_json(str(XTTS_DIR / "config.json"))
        model = Xtts.init_from_config(cfg)
        model.load_checkpoint(cfg, checkpoint_dir=str(XTTS_DIR), use_deepspeed=False)
        model.eval()
        dev = settings().get("cihaz", "cpu")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        model.to(dev)
        _xtts = (model, dev)
    except Exception as e:  # noqa: BLE001 - the meeting goes on with the fallback voices
        _xtts_error = f"{type(e).__name__}: {e}"[:300]
    return _xtts


def engine() -> str:
    return "xtts" if (XTTS_DIR / "model.pth").exists() and _load_xtts() else "yedek"


def _xtts_wav(agent: str, text: str, out: Path) -> None:
    import numpy as np
    import soundfile as sf
    model, dev = _xtts
    s = model.speaker_manager.speakers[speaker_of(agent)]
    with _lock:
        r = model.inference(text, "tr", s["gpt_cond_latent"].to(dev), s["speaker_embedding"].to(dev),
                            temperature=0.7, speed=VOICES[agent]["x_hiz"], enable_text_splitting=True)
    sf.write(str(out), np.asarray(r["wav"], np.float32), 24000)


def _piper_voice():
    global _piper
    if _piper is None:
        from piper import PiperVoice
        _piper = PiperVoice.load(str(PIPER_MODEL))
    return _piper


def _fallback_raw(text: str, engine_name: str, out: Path) -> None:
    if engine_name == "piper":
        with _lock, wave.open(str(out), "wb") as w:
            _piper_voice().synthesize_wav(text, w)
    else:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TOLGA_PS1),
                        "-Text", text, "-Out", str(out)], check=True, capture_output=True, timeout=60,
                       creationflags=ff._CREATE_NO_WINDOW)


def say(agent: str, text: str, out_dir: Path) -> Path:
    """WAV of `agent` saying `text` (cached by content), levelled to -16 LUFS so every voice is equally loud."""
    v = VOICES[agent]
    use_xtts = engine() == "xtts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"x|{speaker_of(agent)}|{v['x_hiz']}" if use_xtts else f"y|{v['motor']}|{v['perde']}|{v['hiz']}|{v['renk']}"
    key = hashlib.sha1(f"{agent}|{tag}|{text}".encode("utf-8")).hexdigest()[:16]
    out = out_dir / f"{agent}_{key}.wav"
    if out.exists():
        return out
    raw = out_dir / f"_{key}_raw.wav"
    if use_xtts:
        _xtts_wav(agent, text, raw)
        chain = ["highpass=f=70", "loudnorm=I=-16:TP=-1.5:LRA=11", "aresample=24000"]
    else:
        _fallback_raw(text, v["motor"], raw)
        with wave.open(str(raw), "rb") as w:
            sr = w.getframerate()
        p, s = v["perde"], v["hiz"]
        tempo = s / p                      # asetrate speeds up by p; bring the pace to s
        chain = [f"asetrate={int(sr * p)}", f"aresample={sr}"]
        while tempo < 0.5 or tempo > 2.0:  # atempo works in 0.5..2 steps
            step = 0.5 if tempo < 0.5 else 2.0
            chain.append(f"atempo={step}")
            tempo /= step
        chain += [f"atempo={tempo:.4f}", v["renk"], "loudnorm=I=-16:TP=-1.5:LRA=11", "aresample=24000"]
    ff.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-i", str(raw), "-af", ",".join(chain), "-ac", "1", str(out)])
    raw.unlink(missing_ok=True)
    return out
