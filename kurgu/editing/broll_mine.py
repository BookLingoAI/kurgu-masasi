"""Intro material from the talk recordings themselves: the moments where the camera shows the food or the place.

Candidates
  - every stretch of the talk recordings where the speaker's face is NOT in the picture (keyframe face timeline,
    ~1 check per second, fast): the camera turned to the food, the room, the entrance;
  - whole files the user marked as extra footage / "İkisi de".

Per 3 fps sample (320 px): faces (MediaPipe), person mask (selfie segmenter), sharpness, brightness, colourfulness,
and the global camera motion between samples (phase correlation). A window is rejected when
  - a face is large (the speaker talking, or a stranger close up),
  - a body is in frame without a face (feet, legs, hands, pocket),
  - the camera moves jerkily (shake, spin, phone being picked up),
  - it is dark, blurred, or empty (floor, ceiling, plain wall: no colour, no structure).
A SMOOTH, steady camera move (slow tilt up to the decor, pan across the room) is kept and preferred as the opening
shot: speed may be high, but direction stays the same and the velocity changes slowly. A move that starts on the feet
is trimmed to the first clean sample instead of being dropped.

Repeats are removed three ways: one shot per scene (same file within 20 s), colour fingerprint of the food itself
(white rice / dark seaweed / grey ignored, so two sushi plates with different fillings differ) plus ORB keypoint
matching of the middle frames (the same plate or the same view filmed again), and - only for "showing" moments - the
words said at that moment ("somon" vs "yengeç"): different names keep both, the same name drops the second.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

from ..analysis.text import norm_word
from ..integrations import ffmpeg as ff

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

NOWIN = 0x08000000 if os.name == "nt" else 0
FPS = 3.0
W = 320
SHOW_WORDS = {"bakın", "bakalım", "işte", "geldi", "geliyor", "göstereyim", "gösteriyim", "gösteriyorum", "gösterdim",
              "burası", "burda", "burada", "şöyle", "görüyorsunuz", "gördüğünüz", "tabak", "tabağımız", "yemeğimiz",
              "geldiler", "getirdi", "mekan", "mekân", "giriş", "girişi", "gezdiriyim", "gezdireyim"}
NAME_STOP = {"bakın", "bakalım", "işte", "geldi", "geliyor", "burası", "burada", "burda", "dostlar", "dostlarım",
             "şimdi", "şöyle", "böyle", "bunlar", "bunu", "şunu", "şuna", "gösteriyim", "göstereyim", "gördüğünüz",
             "görüyorsunuz", "arkadaşlar", "evet", "tamam", "yani", "biraz", "güzel", "yeni", "size", "sizlere"}


# ------------------------------------------------------------------ frames + measurements ----------------------
_PROBES: dict = {}


def _rgb_frames(path: str, start: float, dur: float, fps: float = FPS, width: int = W):
    if path not in _PROBES:   # one ffprobe per recording, not per window
        _PROBES[path] = ff.probe(path)
    info = _PROBES[path]
    h = int(round(info.height * width / max(1, info.width) / 2) * 2) or width * 9 // 16
    p = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{max(0.0, start):.3f}", "-i", path,
                        "-t", f"{dur:.3f}", "-vf", f"fps={fps},scale={width}:{h}", "-f", "rawvideo", "-pix_fmt",
                        "rgb24", "-"], capture_output=True, creationflags=NOWIN)
    n = width * h * 3
    buf = p.stdout
    return [np.frombuffer(buf[i * n:(i + 1) * n], np.uint8).reshape(h, width, 3) for i in range(len(buf) // n)]


class _Vision:
    """Face detector + person segmenter (the MediaPipe models, run by OpenCV), or nothing when the models are
    missing."""

    def __init__(self):
        from ..vision.faces import FaceDetector, backend, person_mask  # noqa: F401
        self.faces = FaceDetector()
        self.seg = backend() != "yok"

    def person(self, rgb: np.ndarray) -> tuple[float, float, float, float]:
        """(share of the frame covered by a person, share of the TOP third covered, share of picture rows that
        contain the person, share of the top third's MIDDLE half covered). A person spanning (almost) the full height or reaching into the top third is a
        head/torso - someone in the picture, maybe a turned face the detector missed; a hand holding a plate
        comes in from the bottom or the side and covers far fewer rows."""
        if not self.seg:
            return 0.0, 0.0, 0.0, 0.0
        from ..vision.faces import _cv_ok, _shared_selfie, person_mask
        if _cv_ok():
            m = _shared_selfie().mask(rgb)
        else:
            m = person_mask(rgb)
            if m is None:
                return 0.0, 0.0, 0.0, 0.0
            m = m * 0.3 + 0.35      # undo the soft-mask stretch: back to probabilities
        on = m > 0.5
        h, w = on.shape
        return (float(on.mean()), float(on[: h // 3].mean()), float(on.any(axis=1).mean()),
                float(on[: h // 3, w // 4: 3 * w // 4].mean()))


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler-Suesstrunk colourfulness (0 grey .. ~100 very colourful)."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def measure(frames: list[np.ndarray], vis: _Vision | None) -> list[dict]:
    """Per sample: face (largest face height, 0..1), body (person mask share), sharp, bright, colour, structure,
    and the camera motion (dx, dy in % of the width per sample) from the previous sample."""
    out = []
    prev = None
    for rgb in frames:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gf = g.astype(np.float32)
        faces = vis.faces.detect(rgb) if vis is not None else []
        face = max((f["h"] for f in faces if f.get("score", 1) >= 0.6), default=0.0)
        body, body_top, body_rows, body_top_c = vis.person(rgb) if vis is not None else (0.0, 0.0, 0.0, 0.0)
        edges = cv2.Canny(g, 60, 140)
        m = {"face": float(face), "body": body, "body_top": body_top, "body_rows": body_rows, "body_top_c": body_top_c, "sharp": float(cv2.Laplacian(g, cv2.CV_32F).var()),
             "bright": float(g.mean()), "colour": colourfulness(rgb), "structure": float((edges > 0).mean()),
             "dx": 0.0, "dy": 0.0}
        if prev is not None:
            win = cv2.createHanningWindow(gf.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(prev, gf, win)
            if resp < 0.03:      # nothing matched: a cut, a spin or a huge jump
                dx, dy = 40.0, 40.0
            m["dx"], m["dy"] = float(dx) / gf.shape[1] * 100, float(dy) / gf.shape[1] * 100
        prev = gf
        out.append(m)
    return out


def clean(m: dict) -> bool:
    """One sample is usable as b-roll."""
    if m["face"] > 0.18:                     # a face fills the frame: the speaker / someone close
        return False
    if m["face"] < 0.05:
        # a body without a face: legs/feet/trousers fill the frame, or a dull body part (shoe, pocket, floor);
        # a hand holding a colourful plate or dessert is fine - that is the classic food shot (the segmenter
        # counts a dessert held up to the lens as "person", so a big share alone is not enough to reject)
        body_top_c = m.get("body_top_c", m.get("body_top", 0.0))
        if m["body"] > 0.60 or (m["body"] > 0.40 and m["colour"] < 30) or (m["body"] > 0.10 and m["colour"] < 25):
            return False
        if body_top_c > 0.08 or (m["body"] > 0.12 and m.get("body_rows", 0.0) >= 0.88):
            return False      # head / torso in the middle, or a person top to bottom: someone in the picture
    if not (35 <= m["bright"] <= 225):
        return False
    if m["sharp"] < 40:
        return False
    return m["colour"] >= 12 or m["structure"] >= 0.06   # not an empty floor / ceiling / wall


def motion_kind(ms: list[dict]) -> str:
    """steady / smooth (a deliberate slow move) / shaky."""
    if len(ms) < 3:
        return "steady"
    v = np.array([[m["dx"], m["dy"]] for m in ms[1:]], np.float32)
    speed = np.linalg.norm(v, axis=1)
    if speed.max() >= 35:                     # lost track: spin / jump
        return "shaky"
    if speed.mean() < 1.2:
        return "steady"
    jerk = np.linalg.norm(np.diff(v, axis=0), axis=1).mean() if len(v) > 1 else 0.0
    mean_dir = v.mean(axis=0)
    consistency = float(np.linalg.norm(mean_dir) / (speed.mean() + 1e-6))   # 1 = always the same direction
    if consistency > 0.75 and jerk < 0.6 * speed.mean() + 1.0 and speed.mean() < 12:
        return "smooth"
    return "shaky"


# ------------------------------------------------------------------ fingerprints for repeats --------------------
def food_hist(rgb: np.ndarray) -> np.ndarray:
    """Hue/saturation histogram of the coloured pixels in the middle of the picture (where the plate is), with rice
    white, seaweed black and greys ignored."""
    h0, w0 = rgb.shape[:2]
    rgb = rgb[int(h0 * 0.2):int(h0 * 0.8), int(w0 * 0.25):int(w0 * 0.75)]   # the subject, not the room around it
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 50) & (hsv[..., 2] > 45)).astype(np.uint8)
    if mask.mean() < 0.02:
        mask = np.ones_like(mask)
    h = cv2.calcHist([hsv], [0, 1], mask, [18, 6], [0, 180, 40, 256])
    h = cv2.GaussianBlur(h, (0, 0), 2.0, borderType=cv2.BORDER_REPLICATE)   # tolerant to small colour shifts
    return cv2.normalize(h, h).flatten()


def same_view(a: np.ndarray, b: np.ndarray) -> bool:
    """ORB keypoints + homography: the same plate / the same part of the room filmed again."""
    def centre(x):   # the subject, not the room: the same restaurant background must not count as "the same"
        h0, w0 = x.shape[:2]
        return x[int(h0 * 0.15):int(h0 * 0.85), int(w0 * 0.2):int(w0 * 0.8)]
    orb = cv2.ORB_create(500)
    ga, gb = cv2.cvtColor(centre(a), cv2.COLOR_RGB2GRAY), cv2.cvtColor(centre(b), cv2.COLOR_RGB2GRAY)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return False
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    good = [m for m in matches if m.distance < 48]
    if len(good) < 18:
        return False
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return inl is not None and int(inl.sum()) >= 35


def spoken_names(words: list[dict], t0: float, t1: float) -> set[str]:
    """What the speaker names while SHOWING something: nouns said within 4 s of the shot, only if a showing word
    ("bakın", "işte", "geldi"...) is among them. Otherwise nothing (talk while eating says nothing about the shot)."""
    near = [norm_word(w["word"]) for w in words if t0 - 4.0 <= w.get("start", -99) <= t1 + 4.0]
    if not any(w in SHOW_WORDS for w in near):
        return set()
    return {w for w in near if len(w) >= 4 and w not in NAME_STOP and w not in SHOW_WORDS and w.isalpha()}


# ------------------------------------------------------------------ candidates ----------------------------------
def _locate(parts: list[dict], t: float) -> tuple[str, float, float] | None:
    """Virtual camera time -> (file, local time, file duration)."""
    for p in parts:
        if p["start"] <= t < p["start"] + p["duration"]:
            return p["path"], t - p["start"], p["duration"]
    return None


def face_timeline(path: str, vis: "_Vision") -> list[tuple[float, float]]:
    """(time, largest face height 0..1) on every keyframe (~1 per second): decoding only keyframes is fast, so a
    whole recording is checked in seconds."""
    p = subprocess.run([ff.ffmpeg_bin(), "-v", "info", "-nostdin", "-skip_frame", "nokey", "-i", path, "-fps_mode", "vfr",
                        "-vf", "scale=192:108,showinfo", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, creationflags=NOWIN)
    err = p.stderr.decode("utf-8", "replace")
    times = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", err)]
    if not p.stdout or not times:      # never cache "no keyframes" when ffmpeg itself failed
        raise RuntimeError(f"Anahtar kareler okunamadı ({Path(path).name}): {err.strip()[-300:]}")
    n = 192 * 108 * 3
    out = []
    for i, t in enumerate(times[: len(p.stdout) // n]):
        rgb = np.frombuffer(p.stdout[i * n:(i + 1) * n], np.uint8).reshape(108, 192, 3)
        faces = vis.faces.detect(rgb) if vis is not None else []
        out.append((t, max((f["h"] for f in faces if f.get("score", 1) >= 0.6), default=0.0)))
    return out


def candidate_windows(pl: dict, faces: dict, min_len: float = 2.2, max_windows: int = 240,
                      extra_files: list[str] | None = None, durations: dict | None = None) -> list[dict]:
    """Stretches worth a close look: where the speaker's face is NOT in the picture (the camera is turned to the
    food, the room, the entrance), from the keyframe face timeline of every talk recording, plus whole files the
    user marked as intro material. faces: {file: [(t, face height)]}."""
    out = []
    for prt in pl["camera"]["parts"]:
        tl = faces.get(prt["path"]) or []
        run = []
        for t, fh in tl + [(1e9, 1.0)]:
            if fh < 0.12:
                run.append(t)
                continue
            if run:
                a, b = max(0.0, run[0] - 0.4), min(prt["duration"], run[-1] + 0.6)
                while b - a >= min_len:
                    e = min(b, a + 8.0)
                    out.append({"file": prt["path"], "in": round(a, 3), "out": round(e, 3),
                                "vt0": prt["start"] + a, "vt1": prt["start"] + e, "why": "gösterme"})
                    a = e
            run = []
    out = out[:max_windows]
    talk = {str(Path(p["path"])) for p in pl["camera"]["parts"]}
    for f in dict.fromkeys(extra_files or []):
        if str(Path(f)) in talk:     # a talk recording marked "İkisi de" is already covered by its face-free stretches
            continue
        d = (durations or {}).get(f) or 0.0
        a = 0.0
        while d - a >= min_len and len(out) < max_windows + 80:
            b = min(d, a + 8.0)
            out.append({"file": f, "in": round(a, 3), "out": round(b, 3), "vt0": -1e6, "vt1": -1e6, "why": "ek"})
            a = b
    return out


def scan(cands: list[dict], slot: float, progress=None, vis: _Vision | None = None) -> list[dict]:
    """Measure every candidate; keep the clean sub-window of length `slot` with the best score in each."""
    from concurrent.futures import ThreadPoolExecutor
    vis = vis if vis is not None else _Vision()
    n_need = max(2, int(round(slot * FPS)) + 1)
    shots = []
    # the windows are decoded six at a time (separate ffmpeg processes); measuring stays in order here.
    # Measured 23.10.2026: 0.9 s decode + 0.4 s measure per window, ~250 windows one by one = ~5 minutes
    pool = ThreadPoolExecutor(max_workers=6)
    decoded = pool.map(lambda c: _rgb_frames(c["file"], c["in"], c["out"] - c["in"]), cands)
    for k, (c, frames) in enumerate(zip(cands, decoded)):
        if progress and k % 5 == 0:
            progress(k / max(1, len(cands)), f"İntro için görüntü aranıyor {k}/{len(cands)}")
        if len(frames) < n_need:
            continue
        ms = measure(frames, vis)
        ok = [clean(m) for m in ms]
        best = None
        for i in range(0, len(ms) - n_need + 1):
            if not all(ok[i:i + n_need]):
                continue
            seg = ms[i:i + n_need]
            kind = motion_kind(seg)
            if kind == "shaky":
                continue
            sharp = np.mean([min(1.0, m["sharp"] / 400) for m in seg])
            colour = np.mean([min(1.0, m["colour"] / 60) for m in seg])
            struct = np.mean([min(1.0, m["structure"] / 0.15) for m in seg])
            score = 0.4 * sharp + 0.35 * colour + 0.25 * struct + (0.25 if kind == "smooth" else 0.0)
            # a deliberate move that starts on the feet: the window begins at the first clean sample (i > 0
            # after unclean ones) - same scoring, nothing extra needed; prefer the start of a reveal
            if best is None or score > best["score"]:
                best = {"score": float(score), "i": i, "kind": kind}
        if best is None:
            continue
        a = c["in"] + best["i"] / FPS
        mid = frames[best["i"] + n_need // 2]
        vt0 = c["vt0"] + best["i"] / FPS if c["vt0"] > -1e5 else -1.0
        shots.append({"file": c["file"], "in": round(a, 3), "out": round(a + slot, 3), "score": round(best["score"], 3),
                      "motion": best["kind"], "why": c["why"], "vt0": vt0,
                      "_mid": mid, "_hist": food_hist(mid)})
    pool.shutdown(wait=False)
    return shots


def pick(shots: list[dict], words: list[dict], max_shots: int, per_file: int = 4) -> tuple[list[dict], list[str]]:
    """Diverse, non-repeating selection (best first), returned in shooting order with the best smooth reveal first.
    Also returns notes on what was dropped as a repeat."""
    chosen: list[dict] = []
    notes = []
    for s in sorted(shots, key=lambda x: -x["score"]):
        if len(chosen) >= max_shots:
            break
        if sum(1 for c in chosen if c["file"] == s["file"]) >= per_file:
            continue
        s["_names"] = spoken_names(words, s["vt0"] - 0.5, s["vt0"] + (s["out"] - s["in"])) if s["vt0"] > 0 else set()
        dup = None
        for c in chosen:
            if c["file"] == s["file"] and abs(c["in"] - s["in"]) < 12.0:
                dup = "aynı sahne"
                break
            if s["_names"] and c.get("_names"):
                if s["_names"] & c["_names"]:
                    dup = "aynı yemek adı"
                    break
                continue          # named differently by the speaker ("somon" vs "yengeç"): different dishes
            if cv2.compareHist(s["_hist"], c["_hist"], cv2.HISTCMP_CORREL) > 0.95 or same_view(s["_mid"], c["_mid"]):
                dup = "aynı görüntü"
                break
        if dup:
            notes.append(f"{Path(s['file']).name} {s['in']:.1f} sn: {dup}")
            continue
        chosen.append(s)
    chosen.sort(key=lambda s: s["vt0"])
    reveal = next((s for s in chosen if s["motion"] == "smooth"), None)
    if reveal is not None and chosen and chosen[0] is not reveal and reveal["vt0"] - chosen[0]["vt0"] < 120:
        chosen.remove(reveal)
        chosen.insert(0, reveal)      # a slow reveal makes the best opening shot
    for c in chosen:
        for k in ("_mid", "_hist", "_names"):
            c.pop(k, None)
    return chosen, notes


def preview(shots: list[dict], out_dir: Path) -> list[str]:
    """Small JPEG of every chosen shot (middle frame) for the review strip in the app."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("intro_*.jpg"):
        old.unlink()
    files = []
    for k, s in enumerate(shots):
        fr = _rgb_frames(s["file"], (s["in"] + s["out"]) / 2, 0.4, fps=2.0, width=480)
        if fr:
            p = out_dir / f"intro_{k:02d}.jpg"
            Image.fromarray(fr[0]).save(p, "JPEG", quality=85)
            files.append(str(p))
        else:
            files.append("")
    return files


def key_of(shot: dict) -> str:
    """Stable id of a shot (for the user's "do not use" list)."""
    return f"{Path(shot['file']).name}@{round(float(shot['in']), 1)}"


__all__ = ["candidate_windows", "scan", "pick", "preview", "key_of", "clean", "motion_kind", "food_hist", "same_view",
           "spoken_names"]

