"""Gentle automatic colour correction (what an editor does first in Lumetri): neutral white balance, black and
white points, a soft S-curve and a little saturation. Conservative limits so skin never looks processed."""
from __future__ import annotations

import cv2
import numpy as np


GRADE_VERSION = 3   # 3: one grade per recording (shot matching), 2: one per project


def compute_grade(frames: list[np.ndarray], strength: float = 1.0) -> dict:
    """frames: a few RGB frames of the camera. Returns per-channel 256-entry LUTs + saturation.

    "Do no harm" (version 2): a modern phone already exposes and colours well, so the picture must never come out
    darker, crushed or clipped. Only: a slight white-balance nudge, lifted blacks pulled down a little (only when
    they really are lifted), dim footage brightened in the mid-tones, a soft highlight roll-off so nothing clips,
    and a touch of saturation. No S-curve (it darkened shadows: jackets, hair and beards turned to black blobs)."""
    px = np.concatenate([cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))).reshape(-1, 3) for f in frames])
    px = px.astype(np.float32)
    luma = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mid = px[(luma > np.percentile(luma, 20)) & (luma < np.percentile(luma, 95))]
    means = mid.mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1), 0.96, 1.04)       # gray world, at most ±4 %
    gains = 1 + (gains - 1) * 0.5 * strength                               # keep most of the room's warmth
    gains = gains / float(gains @ np.array([0.2126, 0.7152, 0.0722]))     # colour shift only, same brightness
    p1 = float(np.percentile(luma, 1.0))
    lo = float(np.clip((p1 - 16.0) * 0.5, 0, 8)) * strength               # only clearly lifted blacks
    x = np.arange(256, dtype=np.float32)
    med = float(np.median(luma))
    # dim footage (restaurants, evenings): lift the mid-tones, never darken bright footage
    gamma = float(np.clip(np.log(0.42) / np.log(max(0.05, med / 255.0)), 0.82, 1.0)) if med < 0.42 * 255 else 1.0
    gamma = 1 + (gamma - 1) * strength
    luts = []
    for c in range(3):
        xs = x * gains[c]
        xs = xs - lo * np.clip(1 - xs / 64.0, 0, 1)           # lifted blacks: only the toe moves, mid-tones stay
        v = np.clip(xs / 255.0, 0, None) ** gamma
        top, k = float(v[-1]), 0.88
        if top > 1.0:   # a gain > 1 would clip the brightest values: compress just the top instead
            v = np.where(v > k, k + (1 - k) * (v - k) / (top - k), v)
        luts.append(np.clip(np.round(v * 255), 0, 255).astype(np.uint8).tolist())
    return {"version": GRADE_VERSION, "luts": luts, "saturation": round(1 + 0.05 * strength, 3),
            "gains": [round(float(g), 3) for g in gains], "black": round(lo, 1), "gamma": round(gamma, 3)}


def grade_for(g: dict | None, file: str | None = None) -> dict | None:
    """The grade of one recording from a project grade ({"files": {path: grade}, "default": grade}) or the grade
    itself (older single-grade form)."""
    if not g:
        return None
    if "files" in g:
        return g["files"].get(str(file)) or g.get("default")
    return g


def apply_grade(rgb: np.ndarray, g: dict | None, file: str | None = None) -> np.ndarray:
    g = grade_for(g, file)
    if not g:
        return rgb
    lut = np.stack([np.array(l, np.uint8) for l in g["luts"]], axis=1).reshape(256, 1, 3)
    out = cv2.LUT(np.ascontiguousarray(rgb), lut)
    s = g.get("saturation", 1.0)
    if abs(s - 1) > 1e-3:
        gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        out = cv2.addWeighted(out, s, gray, 1 - s, 0)
    return out
