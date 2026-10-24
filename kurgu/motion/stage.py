"""Talking-head "stage" (Anlatım / masa başı mode): the camera picture becomes a rounded card on a dark set and
moves around it - full screen while he just talks, to the right while a title or a list is written on the left,
to the left for the other way round, smaller in the centre for a quote. Moves are eased (smootherstep, 0.6 s) and
the card carries a soft shadow and a faint glow in the accent colour.

layout: [{"t0", "t1", "state"}] covering the timeline; states: full | right | left | center | none (pass-through,
e.g. while a screen recording is shown by the compositor).
"""
from __future__ import annotations

import cv2
import numpy as np

EASE = 0.6
# x, y, width as fractions of the output frame (height follows the frame's aspect ratio)
RECTS = {"full": (0.0, 0.0, 1.0), "right": (0.525, 0.20, 0.425), "left": (0.05, 0.20, 0.425),
         "center": (0.25, 0.12, 0.50)}


def _smooth(k: float) -> float:
    k = min(1.0, max(0.0, k))
    return k * k * k * (k * (6 * k - 15) + 10)


def background(w: int, h: int, accent: tuple[int, int, int]) -> np.ndarray:
    """Near-black set: a soft lighter centre, dark corners, a hint of the accent colour and fine grain."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - 0.42 * w) / w) ** 2 + ((yy - 0.45 * h) / h) ** 2)
    base = 26 - 16 * np.clip(r / 0.75, 0, 1) ** 1.6
    img = np.stack([base, base, base * 1.05], axis=2)
    warm = np.exp(-(((xx - 0.75 * w) / (0.35 * w)) ** 2 + ((yy - 0.42 * h) / (0.35 * h)) ** 2))
    img += warm[..., None] * np.array(accent, np.float32)[None, None, :] * 0.035
    rng = np.random.default_rng(3)
    img += rng.normal(0, 1.4, (h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


class Stage:
    def __init__(self, layout: list[dict], w: int, h: int, accent: tuple[int, int, int]):
        self.layout = sorted(layout, key=lambda s: s["t0"])
        self.w, self.h = w, h
        self.accent = accent
        self.bg = background(w, h, accent)
        self._fx_cache: dict = {}
        self._bg_small = None

    def state_at(self, t: float) -> tuple[str, str, float]:
        """(previous state, current state, 0..1 progress of the move into the current one)."""
        prev = "full"
        for i, s in enumerate(self.layout):
            if s["t0"] <= t < s["t1"]:
                prev = self.layout[i - 1]["state"] if i > 0 and abs(self.layout[i - 1]["t1"] - s["t0"]) < 0.05 else "full"
                return prev, s["state"], _smooth((t - s["t0"]) / EASE)
        return "full", "full", 1.0

    def rect(self, state: str) -> tuple[float, float, float, float, float]:
        x, y, fw = RECTS.get(state, RECTS["full"])
        hw = fw * self.w
        hh = hw * self.h / self.w
        rad = 0.0 if state == "full" else 0.034 * hw
        return x * self.w, y * self.h, hw, hh, rad

    def _set(self, x: int, y: int, cw: int, ch: int) -> np.ndarray:
        """The set with the card's shadow and glow for a card at this place (computed small, cached)."""
        key = (x, y, cw, ch)
        if key in self._fx_cache:
            return self._fx_cache[key]
        s = 8
        sw, sh = self.w // s, self.h // s
        m = np.zeros((sh, sw), np.float32)
        cv2.rectangle(m, (x // s, (y + int(0.02 * ch)) // s), ((x + cw) // s, (y + ch + int(0.02 * ch)) // s), 1.0, -1)
        shadow = cv2.GaussianBlur(m, (0, 0), max(2.0, cw / s * 0.06))
        g = np.zeros((sh, sw), np.float32)
        cv2.rectangle(g, (x // s, y // s), ((x + cw) // s, (y + ch) // s), 1.0, -1)
        glow = cv2.GaussianBlur(g, (0, 0), max(3.0, cw / s * 0.12))
        if self._bg_small is None:
            self._bg_small = cv2.resize(self.bg, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        delta = self._bg_small * (-0.55 * shadow)[..., None] + glow[..., None] * np.array(self.accent, np.float32)[None, None, :] * 0.10
        delta = cv2.resize(np.clip(delta + 128, 0, 255).astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # bg + (delta - 128) in integer maths (cv2 saturates)
        out = cv2.subtract(cv2.add(self.bg, delta), np.full_like(self.bg, 128))
        if len(self._fx_cache) > 24:
            self._fx_cache.clear()
        self._fx_cache[key] = out
        return out

    def _mask(self, cw: int, ch: int, rad: float) -> np.ndarray:
        key = ("m", cw, ch, int(rad))
        if key in self._fx_cache:
            return self._fx_cache[key]
        mask = np.zeros((ch * 2, cw * 2), np.uint8)
        r2 = int(rad * 2)
        if r2 > 0:
            cv2.rectangle(mask, (r2, 0), (cw * 2 - 1 - r2, ch * 2 - 1), 255, -1)
            cv2.rectangle(mask, (0, r2), (cw * 2 - 1, ch * 2 - 1 - r2), 255, -1)
            for cx, cy in ((r2, r2), (cw * 2 - 1 - r2, r2), (r2, ch * 2 - 1 - r2), (cw * 2 - 1 - r2, ch * 2 - 1 - r2)):
                cv2.circle(mask, (cx, cy), r2, 255, -1, lineType=cv2.LINE_AA)
        else:
            mask[:] = 255
        m = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_AREA)
        self._fx_cache[key] = m
        return m

    def apply(self, rgb: np.ndarray, t: float) -> np.ndarray:
        prev, cur, k = self.state_at(t)
        if cur == "none" or (cur == "full" and (prev in ("full", "none") or k >= 1.0)):
            return rgb
        a, b = self.rect(prev if prev != "none" else "full"), self.rect(cur)
        x, y, cw, ch, rad = (a[i] + (b[i] - a[i]) * k for i in range(5))
        x, y, cw, ch = int(round(x)), int(round(y)), max(8, int(round(cw))), max(8, int(round(ch)))
        if cw >= self.w - 2 and ch >= self.h - 2:
            return rgb
        out = self._set(x, y, cw, ch).copy()
        card = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.w, x + cw), min(self.h, y + ch)
        if x1 <= x0 or y1 <= y0:
            return out
        sub = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
        out[y0:y1, x0:x1] = card[sub]
        r = int(np.ceil(rad)) + 1
        if r > 1:     # only the four corners need the rounded (anti-aliased) mask
            m = self._mask(cw, ch, rad)
            for cy0, cx0 in ((0, 0), (0, cw - r), (ch - r, 0), (ch - r, cw - r)):
                ya, yb = max(cy0, y0 - y), min(cy0 + r, y1 - y)
                xa, xb = max(cx0, x0 - x), min(cx0 + r, x1 - x)
                if ya >= yb or xa >= xb:
                    continue
                mf = m[ya:yb, xa:xb].astype(np.float32)[..., None] / 255.0
                base = self._set(x, y, cw, ch)[y + ya:y + yb, x + xa:x + xb].astype(np.float32)
                out[y + ya:y + yb, x + xa:x + xb] = (base * (1 - mf) + card[ya:yb, xa:xb].astype(np.float32) * mf).astype(np.uint8)
        return out


def plan_layout(windows: list[dict], duration: float, screen: list[tuple[float, float]] | None = None,
                min_hold: float = 3.0, join_gap: float = 4.0) -> list[dict]:
    """windows: graphics that need the free side [{"t0", "t1", "side"}] (side = where the TEXT is). The camera goes
    to the other side for each, stays there across short gaps (no ping-pong), and returns to full screen between.
    Screen-recording spans become "none" (the compositor shows the screen with the camera in a corner)."""
    want = []
    for w in sorted(windows, key=lambda w: w["t0"]):
        state = {"left": "right", "right": "left", "center": "center"}.get(w.get("side", "left"), "right")
        t0, t1 = max(0.0, w["t0"] - 0.35), min(duration, max(w["t1"] + 0.3, w["t0"] + min_hold))
        if want and want[-1]["state"] == state and t0 - want[-1]["t1"] <= join_gap:
            want[-1]["t1"] = max(want[-1]["t1"], t1)
        elif want and t0 < want[-1]["t1"]:
            want[-1]["t1"] = t0
            want.append({"t0": t0, "t1": t1, "state": state})
        else:
            want.append({"t0": t0, "t1": t1, "state": state})
    for a, b in screen or []:
        want = [dict(s, t1=min(s["t1"], a)) if s["t0"] < a < s["t1"] else s for s in want if not (a <= s["t0"] and s["t1"] <= b)]
        want = [dict(s, t0=max(s["t0"], b)) if s["t0"] < b < s["t1"] else s for s in want]
        want.append({"t0": a, "t1": b, "state": "none"})
    want = sorted((s for s in want if s["t1"] - s["t0"] > 0.2), key=lambda s: s["t0"])
    out, t = [], 0.0
    for s in want:
        if s["t0"] > t + 1e-3:
            out.append({"t0": t, "t1": s["t0"], "state": "full"})
        out.append(s)
        t = s["t1"]
    if t < duration:
        out.append({"t0": t, "t1": duration, "state": "full"})
    return out
