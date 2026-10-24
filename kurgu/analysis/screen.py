"""Screen-recording activity analysis (section 5.5) at low fps.

- Cursor is found from MOTION (three-frame difference), not by shape matching. A small, compact moving blob that is
  present in both consecutive differences is the cursor; when nothing moves the last position is held. If the motion
  evidence is ambiguous, the position is marked unreliable (no position is invented).
- Events: typing (many small changes in a text-sized band), scroll (large vertical shift), window change (most of
  the frame changes at once), video playing (sustained change over a sizeable area).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _blobs(mask: np.ndarray, min_area: int):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area),
                        "cx": float(cents[i][0]), "cy": float(cents[i][1])})
    return out


def analyze_frames(frames, cfg: dict, cursor_max_frac: float = 0.012) -> dict:
    """frames: iterable of (t, gray HxW uint8). Returns per-sample activity + cursor track + events."""
    prev2 = prev1 = None
    samples = []
    cursor = []
    last = None
    H = W = 0
    for t, g in frames:
        H, W = g.shape
        if prev1 is not None and prev2 is not None:
            d1 = cv2.absdiff(prev1, prev2) > 18
            d2 = cv2.absdiff(g, prev1) > 18
            both = d1 & d2  # three-frame difference: moving object at prev1's position
            frac = float(d2.mean())
            rows = d2.mean(axis=1)
            entry = {"t": round(t, 3), "changed": round(frac, 5)}
            # scroll: global vertical shift
            if frac > 0.15:
                shift = _vertical_shift(prev1, g)
                if shift and abs(shift) >= 3:
                    entry["scroll"] = int(shift)
            if frac > 0.6:
                entry["window_change"] = True
            blobs = _blobs(both, 6)
            small = [b for b in blobs if b["w"] * b["h"] <= cursor_max_frac * W * H and b["w"] < W * 0.08 and b["h"] < H * 0.08]
            big_area = sum(b["area"] for b in blobs if b not in small) / float(W * H)
            entry["big_area"] = round(big_area, 5)
            # typing: several small changes in a horizontal band
            band = rows > 0.002
            if frac < 0.05 and band.sum() < H * 0.06 and band.sum() > 0 and len(small) >= 2:
                entry["typing"] = True
            if len(small) == 1 and big_area < 0.02:
                b = small[0]
                last = (b["cx"] / W, b["cy"] / H)
                cursor.append({"t": round(t, 3), "x": round(last[0], 4), "y": round(last[1], 4), "moving": True, "reliable": True})
            elif not blobs and last is not None:
                cursor.append({"t": round(t, 3), "x": round(last[0], 4), "y": round(last[1], 4), "moving": False, "reliable": True})
            else:
                cursor.append({"t": round(t, 3), "x": None, "y": None, "moving": bool(blobs), "reliable": False})
            samples.append(entry)
        prev2, prev1 = prev1, g
    return {"size": [W, H], "samples": samples, "cursor": cursor,
            "video_playing": video_playing_spans(samples, cfg)}


def _vertical_shift(a: np.ndarray, b: np.ndarray) -> int:
    ra = a.mean(axis=1).astype(np.float32)
    rb = b.mean(axis=1).astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    c = np.correlate(rb, ra, mode="full")
    lag = int(np.argmax(c)) - (len(ra) - 1)
    return lag


def video_playing_spans(samples: list[dict], cfg: dict, min_len: float = 2.0) -> list[tuple[float, float]]:
    """Sustained large-area change (not scroll / window switch) = a video is playing on screen."""
    area = cfg["video_playing_area_fraction"]
    ratio = cfg["video_playing_min_ratio"]
    flags = [(s["t"], s.get("big_area", 0) >= area and "scroll" not in s and not s.get("window_change")) for s in samples]
    spans = []
    i = 0
    while i < len(flags):
        if not flags[i][1]:
            i += 1
            continue
        j = i
        on = 0
        k = i
        while k < len(flags):
            on += flags[k][1]
            if flags[k][1]:
                j = k
            elif (k - i + 1) and on / (k - i + 1) < ratio:
                break
            k += 1
        a, b = flags[i][0], flags[j][0]
        if b - a >= min_len:
            spans.append((a, b))
        i = j + 1
    return spans
