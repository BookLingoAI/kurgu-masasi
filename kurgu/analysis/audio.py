"""Energy-based speech/silence analysis (section 5.3: silence is measured from signal energy, not word gaps)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyProfile:
    db: np.ndarray          # frame energy in dBFS
    frame_s: float
    floor_db: float
    speech_db: float
    threshold_db: float

    def time(self, i: int) -> float:
        return i * self.frame_s


def frame_db(audio: np.ndarray, sr: int, frame_s: float = 0.01) -> np.ndarray:
    n = max(1, int(round(sr * frame_s)))
    usable = len(audio) // n * n
    if usable == 0:
        return np.full(1, -120.0, dtype=np.float32)
    fr = audio[:usable].reshape(-1, n).astype(np.float64)
    rms = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


def energy_profile(audio: np.ndarray, sr: int, cfg: dict) -> EnergyProfile:
    db = frame_db(audio, sr, cfg["frame_s"])
    floor = float(np.percentile(db, cfg["noise_percentile"]))
    speech = float(np.percentile(db, cfg["speech_percentile"]))
    thr = floor + max(cfg["min_threshold_above_floor_db"], cfg["threshold_fraction"] * (speech - floor))
    return EnergyProfile(db=db, frame_s=cfg["frame_s"], floor_db=floor, speech_db=speech, threshold_db=thr)


def active_regions(profile: EnergyProfile, cfg: dict) -> list[tuple[float, float]]:
    """Regions with sound above threshold (hysteresis), as (start_s, end_s). Includes speech AND noises."""
    hi = profile.threshold_db
    lo = hi - cfg["hysteresis_db"]
    db = profile.db
    regions = []
    on = False
    start = 0
    for i, v in enumerate(db):
        if not on and v >= hi:
            on = True
            # walk back while above the low threshold
            s = i
            while s > 0 and db[s - 1] >= lo:
                s -= 1
            start = s
        elif on and v < lo:
            on = False
            regions.append((start, i))
    if on:
        regions.append((start, len(db)))
    fs = profile.frame_s
    guard = cfg["edge_guard_s"]
    min_len = cfg["min_speech_island_s"]
    soft = profile.floor_db + cfg.get("edge_extend_db", 4.0)
    ext = int(round(cfg.get("edge_extend_max_s", 0.15) / fs))
    out = []
    total = len(db) * fs
    for s, e in regions:
        # soft onsets (plosives like 'b', 'p') and breathy endings sit below the threshold: extend while the
        # energy stays clearly above the noise floor, at most edge_extend_max_s on each side
        k = 0
        while k < ext and s > 0 and db[s - 1] >= soft:
            s -= 1
            k += 1
        k = 0
        while k < ext and e < len(db) and db[e] >= soft:
            e += 1
            k += 1
        a, b = s * fs, e * fs
        if b - a < min_len:
            continue
        out.append((max(0.0, a - guard), min(total, b + guard)))
    return merge_regions(out, gap=0.0)


def merge_regions(regions: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    if not regions:
        return []
    regions = sorted(regions)
    out = [list(regions[0])]
    for a, b in regions[1:]:
        if a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(float(a), float(b)) for a, b in out]


def invert_regions(regions: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    out = []
    t = 0.0
    for a, b in sorted(regions):
        if a > t:
            out.append((t, a))
        t = max(t, b)
    if t < total:
        out.append((t, total))
    return out


def pauses(profile: EnergyProfile, cfg: dict, min_len: float) -> list[tuple[float, float]]:
    total = len(profile.db) * profile.frame_s
    return [(a, b) for a, b in invert_regions(active_regions(profile, cfg), total) if b - a >= min_len]


def peak_dbfs(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.max(np.abs(audio)) + 1e-12))
