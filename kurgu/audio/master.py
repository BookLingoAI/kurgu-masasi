"""Voice mastering like a YouTube editor does it, plus built-in sound effects.

Chain (ffmpeg): rumble cut -> RNNoise denoise (model from rnnoise-models, FFT denoise as fallback) -> EQ
(less mud, more presence/air) -> de-esser -> gentle compressor -> [music bed ducked under the voice] ->
two-pass loudness normalisation to YouTube's -14 LUFS / -1.5 dBTP.
"""
from __future__ import annotations

import os
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..integrations import ffmpeg as ff

NOWIN = 0x08000000 if os.name == "nt" else 0
SR = 48000
ASSETS = Path(__file__).resolve().parents[2] / "assets"


@lru_cache(maxsize=1)
def _filters() -> str:
    try:
        return subprocess.run([ff.ffmpeg_bin(), "-hide_banner", "-filters"], capture_output=True, text=True,
                              creationflags=NOWIN, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return ""


def _esc(p: Path) -> str:
    return str(p).replace("\\", "/").replace(":", "\\:")


def voice_chain(cfg: dict | None = None, separated: bool | None = None) -> tuple[str, str]:
    """ffmpeg filter string for the voice + a short description of what is applied.

    Natural first: a phone / DJI wireless mic already cleans the voice, and a strong neural denoiser on top (or on
    a voice with music under it) makes it sound robotic and thin. So RNNoise is blended with the dry voice, and when
    Demucs already separated the voice from the music it is skipped. Compression is gentle; loudness is set later
    by the two-pass normaliser, not by squashing."""
    if separated is None:
        try:
            from .separate import enabled
            separated = enabled()
        except Exception:  # noqa: BLE001
            separated = False
    c = {"denoise": True, "highpass": 70, "mud_db": -1.5, "presence_db": 2.0, "air_db": 1.0, "deess": 0.25,
         "comp_threshold_db": -24, "comp_ratio": 2.2, "rnn_mix": 0.55}
    c.update(cfg or {})
    have = _filters()
    parts, desc = [f"highpass=f={c['highpass']}"], ["uğultu kesme"]
    if " adeclip " in have:   # repair clipped peaks (a loud voice into a small camera mic)
        parts.append("adeclip=w=55:o=75")
        desc.append("patlama onarımı")
    model = ASSETS / "rnnoise" / "cb.rnnn"
    if separated:
        desc.append("müzik ve mekan sesi ayrıldı (Demucs)")
    elif c["denoise"]:
        if " arnndn " in have and model.exists():
            parts.append(f"arnndn=m='{_esc(model)}':mix={c['rnn_mix']}")
            desc.append("hafif gürültü giderme (RNNoise, doğal ses korunur)")
        elif " afftdn " in have:
            parts.append("afftdn=nr=8:nf=-40:tn=1")
            desc.append("hafif gürültü giderme (FFT)")
    parts += [f"equalizer=f=280:t=q:w=1.1:g={c['mud_db']}", f"equalizer=f=3800:t=q:w=1.0:g={c['presence_db']}",
              f"highshelf=f=10000:g={c['air_db']}"]
    desc.append("ses rengi (EQ)")
    if " deesser " in have:
        parts.append(f"deesser=i={c['deess']}")
        desc.append("ıslık sesi azaltma")
    parts.append(f"acompressor=threshold={c['comp_threshold_db']}dB:ratio={c['comp_ratio']}:attack=15:release=200"
                 f":knee=6:makeup=1")
    desc.append("yumuşak dinamik sıkıştırma")
    return ",".join(parts), ", ".join(desc)


def mix_sfx(path: str, sfx: list[dict]) -> int:
    """Add sound effects into a wav in place (after the voice chain, so the denoiser never eats them)."""
    from ..motion.render import _pcm
    if not sfx:
        return 0
    with wave.open(path) as w:
        n, ch, rate = w.getnframes(), w.getnchannels(), w.getframerate()
        audio = np.frombuffer(w.readframes(n), "<i2").reshape(-1, ch).astype(np.float32) / 32768.0
    if ch == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio.copy()
    used = 0
    for c in sfx:
        try:
            a = _pcm(c["file"], c.get("offset", 0.0), c.get("dur", 2.5))
        except Exception:  # noqa: BLE001
            continue
        if c.get("dur"):   # music: keep length, fade in/out
            fi, fo = int(c.get("fade_in", 0) * rate), int(c.get("fade_out", 0) * rate)
            if fi > 1:
                a[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
            if fo > 1:
                a[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
        else:
            nz = np.nonzero(np.abs(a).max(axis=1) > 1e-4)[0]
            a = a[: nz[-1] + 1] if len(nz) else a[:0]
        i0 = int(c["t"] * rate)
        k = min(len(a), len(audio) - i0)
        if k > 0 and i0 >= 0:
            peak = float(np.abs(a).max()) or 1.0
            audio[i0:i0 + k] += a[:k] / peak * (10 ** (c.get("gain_db", -17.0) / 20))
            used += 1
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return used


def duck_breaths(path: str, speech: list[tuple[float, float]], depth_db: float = -9.0) -> float:
    """Turn down breaths, lip smacks and room tone BETWEEN words (in place). A region is ducked only when it is
    outside every (padded) word AND clearly quieter than the voice, so drifting word times cannot eat speech.
    Returns ducked seconds."""
    with wave.open(path) as w:
        n, ch, rate = w.getnframes(), w.getnchannels(), w.getframerate()
        a = np.frombuffer(w.readframes(n), "<i2").reshape(-1, ch).astype(np.float32) / 32768.0
    hop = int(rate * 0.01)
    m = a.mean(axis=1)
    frames = len(m) // hop
    if frames < 10 or not speech:
        return 0.0
    rms = np.sqrt((m[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    in_word = np.zeros(frames, bool)
    for t0, t1 in speech:
        in_word[max(0, int((t0 - 0.15) * 100)): min(frames, int((t1 + 0.32) * 100))] = True
    if not in_word.any():
        return 0.0
    level = np.percentile(db[in_word], 85)
    quiet = (~in_word) & (db < level - 18.0)
    # only gaps of at least 0.12 s
    duck = np.zeros(frames, bool)
    i = 0
    while i < frames:
        if quiet[i]:
            j = i
            while j < frames and quiet[j]:
                j += 1
            if j - i >= 20:
                duck[i:j] = True
            i = j
        else:
            i += 1
    if not duck.any():
        return 0.0
    g = np.where(duck, 10 ** (depth_db / 20), 1.0).astype(np.float32)
    g = np.convolve(g, np.ones(9, np.float32) / 9, mode="same")        # 90 ms soft edges
    gain = np.repeat(g, hop)
    gain = np.concatenate([gain, np.full(len(a) - len(gain), gain[-1] if len(gain) else 1.0, np.float32)])
    a = a * gain[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    return float(duck.sum()) / 100.0


def master(src: str, dst: str, target: str = "I=-14:TP=-1.5:LRA=11", music: str | None = None,
           music_db: float = -24.0, cfg: dict | None = None, sfx: list[dict] | None = None,
           speech: list[tuple[float, float]] | None = None) -> dict:
    """Voice chain (+ ducked music bed) + sound effects -> -14 LUFS. Returns what was done and the loudness."""
    from ..motion.render import loudnorm_wav
    chain, desc = voice_chain(cfg)
    tmp = str(Path(dst).with_suffix(".islenen.wav"))
    if music and Path(music).exists():
        fc = (f"[0:a]{chain},asplit=2[v][key];[1:a]aformat=channel_layouts=stereo,volume={music_db}dB,"
              f"afade=t=in:d=2[m];[m][key]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[md];"
              f"[v][md]amix=inputs=2:duration=first:normalize=0,afade=t=out:st=0:d=0[out]")
        # fade the music out over the last 3 s
        dur = _duration(src)
        fc = fc.replace("afade=t=out:st=0:d=0", f"afade=t=out:st={max(0.0, dur - 3):.2f}:d=3")
        cmd = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", src, "-stream_loop", "-1", "-i", music,
               "-filter_complex", fc, "-map", "[out]", "-ar", str(SR), "-ac", "2", tmp]
        desc += ", müzik yatağı (konuşmada kısılır)"
    else:
        cmd = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", src, "-af", chain, "-ar", str(SR), "-ac", "2", tmp]
    p = subprocess.run(cmd, capture_output=True, creationflags=NOWIN)
    if p.returncode != 0:  # never lose the audio because of a filter problem: fall back to loudness only
        import shutil
        shutil.copyfile(src, tmp)
        mix_sfx(tmp, sfx or [])
        loud = loudnorm_wav(tmp, dst, target)
        loud["mastering"] = "uygulanamadı: " + p.stderr.decode("utf-8", "replace")[-300:]
        return loud
    ducked = duck_breaths(tmp, speech) if speech else 0.0
    if ducked:
        desc += f", nefes ve ağız sesleri hafifçe kısıldı ({ducked:.0f} sn)"
    n_sfx = mix_sfx(tmp, sfx or [])
    loud = loudnorm_wav(tmp, dst, target)
    try:
        os.remove(tmp)
    except OSError:
        pass
    loud["mastering"] = desc + (f", {n_sfx} ses efekti" if n_sfx else "") + ", YouTube ses seviyesi (-14 LUFS)"
    return loud


def _duration(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


# ------------------------------------------------------------------ built-in sound effects ----------------------
def _write(path: Path, x: np.ndarray) -> str:
    x = x / (np.abs(x).max() or 1.0) * 0.9
    st = np.stack([x, x], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    return str(path)


def _env(n: int, attack: float, release: float) -> np.ndarray:
    t = np.linspace(0, 1, n)
    a = np.clip(t / max(attack, 1e-3), 0, 1)
    r = np.clip((1 - t) / max(release, 1e-3), 0, 1)
    return (a * r) ** 1.5


def _bandpass_sweep(noise: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """Noise through a moving resonant band-pass (simple state-variable filter)."""
    out = np.zeros_like(noise)
    low = band = 0.0
    q = 0.35
    freqs = np.geomspace(f0, f1, len(noise))
    for i, s in enumerate(noise):
        f = 2 * np.sin(np.pi * freqs[i] / SR)
        low += f * band
        high = s - low - q * band
        band += f * high
        out[i] = band
    return out


def _bell(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 2093 * t) + 0.7 * np.sin(2 * np.pi * 2637 * t) * np.exp(-t * 1.5)
            + 0.2 * np.sin(2 * np.pi * 5230 * t) * np.exp(-t * 9))
    return tone * np.exp(-t * 5.0) * np.minimum(1.0, t / 0.002)


def _strikes(rng, times, dur: float) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n, np.float32)
    for k, t0 in enumerate(times):
        i0 = int(t0 * SR)
        b = _bell(n - i0) * (1.0 - 0.18 * k)
        out[i0:] += b[: n - i0]
    return out


def _sparkle(rng, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for k, f in enumerate((1568.0, 2093.0, 2637.0, 3136.0, 4186.0)):
        t0 = 0.045 * k
        tt = np.clip(t - t0, 0, None)
        env = np.where(t >= t0, np.exp(-tt * 9.0), 0.0)
        out += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.01 * tt)) * env * (0.8 - 0.1 * k)
    return (out * _env(n, 0.005, 0.6)).astype(np.float32)


def builtin_sfx(folder: Path | None = None) -> dict[str, str]:
    """Synthesised effects (no licence questions): whoosh, swish, pop, click, riser, impact, ding, key."""
    d = folder or (ASSETS / "sfx_hazir")
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    out = {}
    specs = {
        "whoosh": lambda: _bandpass_sweep(rng.standard_normal(int(0.55 * SR)), 300, 4500) * _env(int(0.55 * SR), 0.55, 0.45),
        "swish": lambda: _bandpass_sweep(rng.standard_normal(int(0.3 * SR)), 2500, 600) * _env(int(0.3 * SR), 0.2, 0.7),
        "pop": lambda: np.sin(2 * np.pi * np.cumsum(np.geomspace(900, 300, int(0.09 * SR))) / SR) * _env(int(0.09 * SR), 0.05, 0.9),
        "click": lambda: rng.standard_normal(int(0.02 * SR)) * _env(int(0.02 * SR), 0.02, 0.95),
        "riser": lambda: _bandpass_sweep(rng.standard_normal(int(1.0 * SR)), 200, 6000) * _env(int(1.0 * SR), 0.95, 0.08),
        "impact": lambda: (np.sin(2 * np.pi * np.cumsum(np.geomspace(120, 40, int(0.5 * SR))) / SR)
                           + 0.3 * rng.standard_normal(int(0.5 * SR)) * _env(int(0.5 * SR), 0.01, 0.97) ** 3) * _env(int(0.5 * SR), 0.005, 0.95),
        # cash-register bell for prices: two bright partials + a faint inharmonic one, fast attack, ringing decay
        "ding": lambda: _bell(int(0.9 * SR)) + 0.25 * rng.standard_normal(int(0.9 * SR)) * _env(int(0.9 * SR), 0.002, 0.995) ** 40,
        # notification bell: three quick strikes of a small bright bell
        "bell": lambda: _strikes(rng, (0.0, 0.11, 0.22), 0.9),
        # sparkle: rising glassy pings with a little shimmer (likes, highlights)
        "sparkle": lambda: _sparkle(rng, 0.6),
        # one soft typewriter key: a short filtered tick on a tiny low thump
        "key": lambda: (_bandpass_sweep(rng.standard_normal(int(0.035 * SR)), 3500, 2500) * 2.5
                        + np.sin(2 * np.pi * 170 * np.arange(int(0.035 * SR)) / SR) * 0.5) * _env(int(0.035 * SR), 0.03, 0.9) ** 2,
    }
    for name, fn in specs.items():
        p = d / f"{name}.wav"
        out[name] = str(p) if p.exists() else _write(p, fn())
    bank = SfxBank(out)
    for name, k, fn in _variants():
        p = d / f"{name}_{k}.wav"
        bank.add(name, str(p) if p.exists() else _write(p, fn()))
    try:
        for name, f in library_sfx(d / "kutuphane"):
            bank.pool.setdefault(name, []).insert(0, f)   # the downloaded sounds lead the rotation
    except Exception:  # noqa: BLE001 - a broken library file must never stop a render; the synth set remains
        pass
    return bank


# downloaded effects (paketler\efekt, user-approved sources: Mixkit Free License, Kenney CC0) -> bank names.
# (source folder prefix, file name prefix): the folder holds what the package manager stored with its licence record
LIBRARY = {"whoosh": [("mixkit-", "whoosh_")], "swish": [("mixkit-", "swoosh_")],
           "pop": [("mixkit-", "pop_"), ("kenney-interface-sounds", "drop_")],
           "impact": [("kenney-impact-sounds", "impactPunch_heavy"), ("kenney-impact-sounds", "impactSoft_heavy")],
           "ding": [("mixkit-", "coin_")], "sparkle": [("mixkit-", "magic_")],
           "click": [("mixkit-", "click_"), ("kenney-interface-sounds", "click_")], "bell": [("mixkit-", "bell_")]}
LIB_MAX_S = {"whoosh": 1.6, "swish": 1.0, "pop": 0.5, "impact": 1.2, "ding": 1.5, "sparkle": 1.5, "click": 0.35, "bell": 1.5}


def library_sfx(out_dir: Path, root: Path | None = None) -> list[tuple[str, str]]:
    """Downloaded effects trimmed to where the sound is (from the first audible sample), faded out and peak-matched
    to the synthesised set, cached as wav. Returns [(bank name, file)]."""
    from ..integrations import ffmpeg as ff
    lib = (root or ASSETS.parent) / "paketler" / "efekt"
    if not lib.exists():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for name, rules in LIBRARY.items():
        for folder, prefix in rules:
            for src in sorted(p for d_ in lib.glob(folder + "*") if d_.is_dir() for p in d_.iterdir()
                              if p.suffix.lower() in (".wav", ".ogg") and p.name.startswith(prefix)):
                dst = out_dir / f"{name}__{src.parent.name}__{src.stem}.wav"
                if not dst.exists():
                    x = ff.load_audio(src, sr=SR)
                    if x.size < SR // 50 or np.abs(x).max() < 1e-4:
                        continue
                    a = int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))
                    x = x[max(0, a - SR // 200): a + int(LIB_MAX_S[name] * SR)]
                    n = min(len(x), int(0.12 * SR))
                    x[-n:] *= np.linspace(1.0, 0.0, n)
                    _write(dst, x)
                found.append((name, str(dst)))
    return found


class SfxBank(dict):
    """{name: file} whose lookups rotate through the variants of each sound (whoosh, whoosh_2, ...), so the same
    effect is not heard identically every time. fixed(name) always gives the first one (the CTA keeps its sound)."""

    def __init__(self, base: dict):
        super().__init__(base)
        self.pool = {k: [v] for k, v in base.items()}
        self.turn: dict[str, int] = {}

    def add(self, name: str, file: str) -> None:
        self.pool.setdefault(name, []).append(file)

    def fixed(self, name: str) -> str:
        return dict.__getitem__(self, name)

    def __getitem__(self, name: str) -> str:
        files = self.pool.get(name)
        if not files:
            return dict.__getitem__(self, name)
        k = self.turn.get(name, 0)
        self.turn[name] = k + 1
        return files[k % len(files)]


def _variants():
    """Extra takes of the busiest effects: other sweep ranges, lengths, pitches and noise seeds."""
    def whoosh(seed, f0, f1, dur, a, r):
        g = np.random.default_rng(seed)
        n = int(dur * SR)
        return lambda: _bandpass_sweep(g.standard_normal(n), f0, f1) * _env(n, a, r)

    def pop(f0, f1, dur, noise, seed):
        g = np.random.default_rng(seed)
        n = int(dur * SR)
        return lambda: (np.sin(2 * np.pi * np.cumsum(np.geomspace(f0, f1, n)) / SR)
                        + noise * g.standard_normal(n) * _env(n, 0.01, 0.98) ** 6) * _env(n, 0.05, 0.9)

    def impact(f0, f1, dur, noise, seed):
        g = np.random.default_rng(seed)
        n = int(dur * SR)
        return lambda: (np.sin(2 * np.pi * np.cumsum(np.geomspace(f0, f1, n)) / SR)
                        + noise * g.standard_normal(n) * _env(n, 0.01, 0.97) ** 3) * _env(n, 0.005, 0.95)

    return [("whoosh", 2, whoosh(11, 200, 3200, 0.7, 0.6, 0.4)),
            ("whoosh", 3, whoosh(12, 500, 6500, 0.42, 0.5, 0.5)),
            ("whoosh", 4, whoosh(13, 4200, 350, 0.6, 0.35, 0.6)),
            ("whoosh", 5, whoosh(14, 250, 5200, 0.85, 0.7, 0.3)),
            ("swish", 2, whoosh(21, 3500, 900, 0.24, 0.15, 0.75)),
            ("swish", 3, whoosh(22, 1800, 400, 0.38, 0.25, 0.65)),
            ("swish", 4, whoosh(23, 900, 3000, 0.28, 0.3, 0.6)),
            ("pop", 2, pop(1200, 420, 0.08, 0.15, 31)),
            ("pop", 3, pop(700, 250, 0.11, 0.1, 32)),
            ("pop", 4, pop(1500, 700, 0.06, 0.2, 33)),
            ("impact", 2, impact(90, 35, 0.6, 0.4, 41)),
            ("impact", 3, impact(160, 55, 0.4, 0.2, 42))]
