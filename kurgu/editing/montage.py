"""Music-driven venue montage: after the intro talk, the extra footage is cut to the beat of the user's music
(like a promo), then the talk continues.

- beats: onset envelope (spectral flux) -> tempo by autocorrelation (70-170 BPM) -> best beat phase.
- shots: every clip is scanned at 3 fps; windows that are sharp and steady (no whip pans, not frozen) win.
- order: clips in shooting order; each shot lasts a whole number of beats (~2 s), cuts land on beats.
"""
from __future__ import annotations

import subprocess

import numpy as np

from ..integrations import ffmpeg as ff

SR = 22050
HOP = 512
VENUE_WORDS = {"dükkan", "dükkân", "dükkanım", "dükkânım", "dükkanımı", "dükkânımı", "mekan", "mekân", "mekanı",
               "mekânı", "salon", "salonu", "içeri", "içerisi", "içerisini", "gezdireceğim", "gezdirelim", "gezelim",
               "burası", "burayı", "yerimiz", "yerimizi", "işyeri", "işyerimi", "berber", "berberim", "kuaför"}


def _decode_mono(path: str) -> np.ndarray:
    p = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True,
                       creationflags=0x08000000 if __import__("os").name == "nt" else 0)
    return np.frombuffer(p.stdout, np.float32)


def beat_grid(path: str, max_s: float = 240.0) -> dict:
    y = _decode_mono(path)[: int(max_s * SR)]
    if len(y) < SR * 4:
        raise ValueError("Müzik çok kısa")
    n = 2048
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::HOP] * np.hanning(n).astype(np.float32)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    env = (flux - flux.mean()) / (flux.std() + 1e-9)
    env = np.maximum(env, 0)
    fr = SR / HOP                                             # envelope rate (~43 Hz)
    lags = np.arange(int(fr * 60 / 170), int(fr * 60 / 70) + 1)
    ac = np.array([np.dot(env[:-l], env[l:]) for l in lags])
    bpm_w = np.exp(-0.5 * (np.log2((fr * 60 / lags) / 120.0) / 0.9) ** 2)   # mild preference around 120 BPM
    lag = int(lags[np.argmax(ac * bpm_w)])
    # refine tempo and phase finely (the envelope grid is ~23 ms; small errors drift over many beats)
    tt = np.arange(len(env)) / fr
    best = (-1.0, lag / fr, 0.0)
    for p in np.linspace(0.97, 1.03, 61) * (lag / fr):
        for o in np.arange(0, p, 0.01):
            grid_t = np.arange(o, tt[-1], p)
            v = float(np.interp(grid_t, tt, env).sum()) / max(1, len(grid_t))
            if v > best[0]:
                best = (v, float(p), float(o))
    period, off = best[1], best[2] + 1024 / SR   # flux frame time -> window centre
    beats = np.arange(off, len(y) / SR - 0.05, period)
    # where the music "starts" (skip a quiet intro): first beat with energy near the track's typical level
    rms = np.sqrt(np.convolve(y ** 2, np.ones(SR // 2) / (SR // 2), mode="same"))
    lvl = np.percentile(rms, 75)
    start = next((b for b in beats if rms[min(len(rms) - 1, int(b * SR))] >= 0.6 * lvl), beats[0])
    return {"bpm": round(60 / period, 1), "period": period, "beats": beats.tolist(), "start": float(start),
            "duration": len(y) / SR}


def shot_scores(path: str, fps: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """(times, score) per sample: sharp + steady, penalising whip pans and frozen frames."""
    import cv2
    ts, sharp, motion = [], [], []
    prev = None
    for t, g in ff.gray_frames(path, fps, 320):
        sharp.append(cv2.Laplacian(g, cv2.CV_32F).var())
        motion.append(float(np.abs(g.astype(np.int16) - prev).mean()) if prev is not None else 0.0)
        ts.append(t)
        prev = g.astype(np.int16)
    if not ts:
        return np.zeros(0), np.zeros(0)
    sharp = np.array(sharp)
    motion = np.array(motion)
    s = sharp / (np.percentile(sharp, 90) + 1e-6)
    shake = np.clip((motion - 12.0) / 12.0, 0, 2)              # big frame-to-frame change = whip pan / shake
    frozen = (motion < 0.4).astype(float) * 0.3
    return np.array(ts), np.clip(s, 0, 1.5) - shake - frozen


def best_windows(ts: np.ndarray, score: np.ndarray, length: float, k: int, dur: float) -> list[float]:
    """Start times of up to k non-overlapping windows with the best mean score (in time order)."""
    if not len(ts) or dur < length + 0.2:
        return [0.0] if dur >= length else []
    starts = np.arange(0.1, dur - length - 0.05, 0.33)
    vals = []
    for a in starts:
        m = (ts >= a) & (ts <= a + length)
        vals.append(score[m].mean() if m.any() else -9)
    order = np.argsort(vals)[::-1]
    picked: list[float] = []
    for i in order:
        a = float(starts[i])
        if all(abs(a - b) >= length + 0.3 for b in picked):
            picked.append(a)
        if len(picked) >= k:
            break
    return sorted(picked)


def plan_montage(items: list[dict], music: str, durations: dict, target_s: float | None = None) -> dict:
    """items: extra footage (videos only, in shooting order). Returns shots + music cue info."""
    grid = beat_grid(music)
    period = grid["period"]
    per_cut = max(1, int(round(2.0 / period)))               # ~2 s per shot, a whole number of beats
    slot = per_cut * period
    vids = [it for it in items if (durations.get(it["file"]) or 0) >= slot + 0.3]
    if not vids:
        raise ValueError("Montaj için yeterince uzun ek video yok")
    # use the material: up to one shot per ~2 shot-lengths of every clip (max 4 per clip), 30-75 s in total
    shots = []
    for it in vids:
        k = int(np.clip(durations[it["file"]] // (2 * slot), 1, 4))
        ts, sc = shot_scores(it["file"])
        for a in best_windows(ts, sc, slot, k, durations[it["file"]]):
            shots.append({"file": it["file"], "in": round(a, 3), "out": round(a + slot, 3)})
    cap = int((target_s or 75.0) // slot)
    if len(shots) > cap:   # too much: keep every clip, drop extra windows evenly
        step = len(shots) / cap
        shots = [shots[int(i * step)] for i in range(cap)]
    for i, sh in enumerate(shots):
        sh["move"] = 1 if i % 2 == 0 else -1     # alternate push-in / pull-out
    # music: start on a strong beat, end exactly on the beat grid
    b0 = grid["start"]
    total = len(shots) * slot
    return {"shots": shots, "slot": slot, "beats_per_cut": per_cut, "bpm": grid["bpm"], "music_offset": round(b0, 3),
            "duration": round(total, 3), "music_file": music}


def insert_point(sents: list[dict], notes: list[str] | None = None, segments: list[dict] | None = None) -> float:
    """End of the intro: after the first COMPLETE sentence (12 s or later) that talks about the place - never in the
    middle of a sentence. The montage goes in at the edit cut right after it (a natural pause)."""
    import re as _re

    from ..analysis.text import norm_word
    words = set(VENUE_WORDS) | {norm_word(w) for n in (notes or []) for w in n.split() if len(w) > 3}

    def complete(s):
        t = s["text"].rstrip()
        return bool(_re.search(r"[.!?]$", t)) and not t.endswith("...")
    pick = None
    for i, s in enumerate(sents):
        if s["end"] >= 12 and any(norm_word(w["word"]) in words for w in s["words"]):
            pick = next((x for x in sents[i:i + 4] if complete(x)), s)
            break
    if pick is None:
        pick = next((s for s in sents if s["end"] >= 25 and complete(s)), None) or \
            next((s for s in sents if s["end"] >= 25), sents[-1] if sents else None)
    if pick is None:
        return 0.0
    te = pick["end"]
    if segments:   # snap to the next edit cut (already a cleaned pause), if it is close
        cuts = [sg["timeline_end"] for sg in segments if sg["timeline_end"] >= te - 0.05]
        if cuts and min(cuts) - te <= 1.5:
            return min(cuts)
    return te + 0.15


def insert_into_plan(pl: dict, mont: dict, at: float) -> dict:
    """Shift everything after `at` by the montage length and add the montage shots as segments."""
    M = mont["duration"]
    segs = []
    for s in pl["segments"]:
        s = dict(s)
        s.setdefault("cam_tl", s["timeline_start"])            # position in the camera-only timeline
        if s["timeline_start"] >= at - 1e-3:
            s["timeline_start"] += M
            s["timeline_end"] += M
        elif s["timeline_end"] > at + 1e-3:                    # the insert point falls inside: split
            first = dict(s, timeline_end=at, out=s["in"] + (at - s["timeline_start"]) * float(s.get("speed", 1.0)))
            rest = dict(s, timeline_start=at + M, timeline_end=s["timeline_end"] + M,
                        **{"in": first["out"], "cam_tl": s["cam_tl"] + (at - s["timeline_start"])})
            segs += [first, rest]
            continue
        segs.append(s)
    t = at
    shots = []
    for sh in mont["shots"]:
        d = sh["out"] - sh["in"]
        shots.append({"kind": "montage", "file": sh["file"], "in": sh["in"], "out": sh["out"], "timeline_start": t,
                      "timeline_end": t + d, "screen": None, "move": sh.get("move", 1)})
        t += d
    k = next((i for i, s in enumerate(segs) if s["timeline_start"] >= at + M - 1e-3), len(segs))
    segs = segs[:k] + shots + segs[k:]
    words = []
    for w in pl["transcript_cut"]:
        w = dict(w)
        if w["t_start"] >= at - 1e-3:
            w["t_start"] += M
            w["t_end"] += M
        words.append(w)
    out = dict(pl, segments=segs, transcript_cut=words)
    out["stats"] = dict(pl["stats"], timeline_s=pl["stats"]["timeline_s"] + M)
    out["montage"] = dict(mont, at=round(at, 3))
    return out
