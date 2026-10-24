"""Our own render engine (no Premiere needed): cut pieces -> retouch -> zoom/crop -> overlays -> GPU encode.

Video: every piece is decoded with ffmpeg at the output frame rate, frames go through Python (retouch, zoom,
overlays) and are piped into one ffmpeg encoder. Audio: pieces are decoded to PCM, joined with 4 ms crossfades
(no clicks), loudness-normalised to YouTube level and muxed.
"""
from __future__ import annotations

import os
import math
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..integrations import ffmpeg as ff
from ..config import replace_file

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

NOWIN = 0x08000000 if os.name == "nt" else 0
SR = 48000


@dataclass
class Piece:
    file: str
    t_in: float
    t_out: float
    zoom: float = 1.0          # punch-in factor around the face
    cx: float = 0.5            # face centre (0..1) used for zoom / vertical crop
    cy: float = 0.42
    kind: str = "main"         # "hold" = freeze the previous frame (blurred) for an end card, silent
    screen: dict | None = None  # synced screen recording {"file", "in"} for this piece
    trans_in: float = 0.0      # transition at the start of the piece (zoom: extra zoom, e.g. 0.25 = 125%)
    trans: str = "zoom"        # zoom | whip (fast pan with motion blur) | flash (warm light leak)
    move: int = 1              # montage shots: +1 push in, -1 pull out
    a_file: str | None = None  # take the sound from this file instead (e.g. the original recording when the
    a_in: float = 0.0          # picture comes from an already-retouched timeline file)
    speed: float = 1.0         # >1: played faster (t_in..t_out are source times; dur is the output length)
    pitch: float = 1.0         # voice pitch factor while sped up ("helium" counting)

    @property
    def dur(self) -> float:
        return (self.t_out - self.t_in) / (self.speed or 1.0)

    @property
    def src_dur(self) -> float:
        return self.t_out - self.t_in


@dataclass
class Overlay:
    t0: float                  # output timeline seconds
    t1: float
    rgba: np.ndarray           # HxWx4 uint8, full output frame size
    fade: float = 0.25
    box: tuple = field(default=(0, 0, 0, 0))  # x0, y0, x1, y1 of non-transparent area
    anim: str = "fade"         # fade | pop (scale with overshoot) | rise (slide up) | push (slow scale, full frame)
                               # | wipe (title: an accent bar sweeps in from the left and uncovers it, then wipes out)
                               # | sweep (same without the light bar: a highlighter being drawn)
                               # | type (typewriter: meta["cuts"] = x offsets after each character, then fades out)
                               # | words (kinetic sentence: meta boxes/times/ends, word by word as it is spoken)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.box == (0, 0, 0, 0):
            ys, xs = np.nonzero(self.rgba[..., 3])
            self.box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 1, 1)


def image_filters(image: dict | None) -> str:
    """Clean-up applied while decoding: temporal+spatial denoise (low-light camera grain, ugly colour noise) and a
    light unsharp mask, so the downscaled picture looks crisp. Strengths come from design_config "image"."""
    if not image:
        return ""
    from ..audio.master import _filters
    have = _filters()
    out = ""
    d = float(image.get("denoise", 0) or 0)
    if d > 0 and " hqdn3d " in have:   # GPL filter: missing from some ffmpeg builds
        out += f",hqdn3d={1.5 * d:.2f}:{2.5 * d:.2f}:{3.0 * d:.2f}:{4.5 * d:.2f}"
    s = float(image.get("sharpen", 0) or 0)
    if s > 0 and " unsharp " in have:
        out += f",unsharp=5:5:{0.8 * s:.2f}:5:5:0"
    return out


def _crop_box(w: int, h: int, z: float, cx: float, cy: float) -> tuple[float, float, float, float]:
    """The part of a w x h frame a zoom of z around (cx, cy) shows: x0, y0, width, height (same rule as zoom_frame)."""
    cw, ch = w / z, h / z
    return min(max(cx * w - cw / 2, 0), w - cw), min(max(cy * h - ch / 2, 0), h - ch), cw, ch


def _decode(path: str, t0: float, dur: float, fps: float, w: int, h: int, image: dict | None = None,
            speed: float = 1.0, zoom: tuple[float, float, float] | None = None):
    """dur: output seconds (with speed > 1 the source span is dur * speed). zoom=(z, cx, cy): the punch-in is cut
    out by ffmpeg while decoding (multi-threaded, Lanczos) instead of per frame in Python (23.10.2026: 10 ms/frame)."""
    sp = f"setpts=(PTS-STARTPTS)/{speed:.4f}," if abs(speed - 1.0) > 1e-3 else ""
    crop = ""
    if zoom and zoom[0] > 1.001:
        x0, y0, cw, ch = _crop_box(w, h, *zoom)
        crop = f"crop={cw:.3f}*iw/{w}:{ch:.3f}*ih/{h}:{x0:.3f}*iw/{w}:{y0:.3f}*ih/{h},"
    args = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{t0:.4f}", "-i", path, "-t", f"{dur:.4f}",
            "-vf", f"{sp}fps={fps},{crop}scale={w}:{h}:flags=lanczos" + image_filters(image),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NOWIN)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def _decode_size(pc: "Piece", out_h: int, vertical: bool) -> tuple[int, int]:
    """Decode only as big as needed (4K -> ~1.35x the output): much faster retouch and zoom."""
    info = _probe(pc.file)
    src_w, src_h = info.width, info.height
    sc = min(1.0, (out_h * (1.15 if vertical else 1.35)) / max(1, src_h))
    if sc < 0.999:
        src_w, src_h = int(src_w * sc) // 2 * 2, int(src_h * sc) // 2 * 2
    return src_w, src_h


_PROBES: dict = {}


def _probe(path: str):
    """One ffprobe per recording, not one per piece (a 35-minute edit has ~570 pieces from 13 files)."""
    if path not in _PROBES:
        _PROBES[path] = ff.probe(path)
    return _PROBES[path]


def prezoomed(pc: "Piece", vertical: bool) -> bool:
    """Pieces whose steady punch-in is done by the decoder (not montage/hold, not under a screen recording whose
    speaker box needs the whole camera frame, not the vertical crop)."""
    return not vertical and pc.kind == "main" and not pc.screen and pc.zoom > 1.001


class _Ahead:
    """Decodes the pieces one after another on a background thread while the main loop draws the frames: the
    renderer no longer waits for ffmpeg at every cut (profiled 23.10.2026: 43 % of the time was waiting)."""

    def __init__(self, pieces, fps: float, out_h: int, vertical: bool, image, depth: int = 48):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, args=(pieces, fps, out_h, vertical, image), daemon=True)
        self.t.start()

    def _run(self, pieces, fps, out_h, vertical, image):
        try:
            for k, pc in enumerate(pieces):
                if pc.kind == "hold":
                    continue
                w, h = _decode_size(pc, out_h, vertical)
                zoom = (pc.zoom, pc.cx, pc.cy) if prezoomed(pc, vertical) else None
                for fr in _decode(pc.file, pc.t_in, pc.dur, fps, w, h, image, pc.speed, zoom):
                    if self.stop.is_set():
                        return
                    self.q.put((k, fr))
                self.q.put((k, None))
        except Exception as e:  # noqa: BLE001 - handed to the main loop
            self.q.put((-1, e))

    def frames(self, k: int):
        while True:
            kk, fr = self.q.get()
            if kk == -1:
                raise fr
            if fr is None:
                return
            yield fr

    def close(self):
        self.stop.set()
        while self.t.is_alive():
            try:
                self.q.get(timeout=0.2)
            except Exception:  # noqa: BLE001
                pass


def _parallel(frames, frame_fn, pc, gi0: int, fps: float, pool, want: int):
    """Retouch frames on several CPU cores, keep their order. frame_fn.prepare (face landmarks; must run in order)
    stays on this thread, frame_fn.apply (colour + skin + teeth: OpenCV/NumPy, releases the GIL) runs in the pool."""
    from collections import deque
    q = deque()
    depth = getattr(pool, "_max_workers", 4) * 2
    k = 0
    for rgb in frames:
        if k >= want:
            break
        prep = frame_fn.prepare(rgb, pc, gi0 + k, (gi0 + k) / fps)
        q.append(pool.submit(frame_fn.apply, rgb, prep))
        k += 1
        if len(q) >= depth:
            yield q.popleft().result()
    while q:
        yield q.popleft().result()


TRANS_S = {"zoom": 0.32, "whip": 0.30, "flash": 0.55}
_leak_cache: dict = {}


def _leak(w: int, h: int) -> np.ndarray:
    """Warm light-leak layer (orange core, pink fringe) from the upper left, 0..1 float RGB."""
    key = (w, h)
    if key not in _leak_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - 0.08 * w) / (0.75 * w)) ** 2 + ((yy - 0.15 * h) / (0.85 * h)) ** 2)
        core = np.clip(1 - d, 0, 1) ** 1.6
        col = np.stack([np.full_like(core, 1.0), 0.62 + 0.18 * core, 0.30 + 0.25 * (1 - core)], axis=-1)
        _leak_cache.clear()
        _leak_cache[key] = (core[..., None] * col).astype(np.float32)
    return _leak_cache[key]


def transition_fx(rgb: np.ndarray, kind: str, k: float) -> np.ndarray:
    """k: 0 at the cut -> 1 when the transition is over."""
    e = 1 - (1 - min(1.0, max(0.0, k))) ** 3
    h, w = rgb.shape[:2]
    if kind == "whip":   # the new shot lands from a fast pan: shifted and smeared sideways, settling
        sh = int(round((1 - e) * 0.18 * w))
        L = int((1 - e) * 0.10 * w)
        out = np.roll(rgb, -sh, axis=1) if sh else rgb
        if L >= 3:
            small = cv2.resize(out, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            ker = np.ones((1, max(3, L // 4)), np.float32) / max(3, L // 4)
            out = cv2.resize(cv2.filter2D(small, -1, ker), (w, h), interpolation=cv2.INTER_LINEAR)
        return out
    if kind == "flash":  # light leak: a warm burst at the cut that drifts out (screen blend)
        kk = min(1.0, max(0.0, k))
        a = 0.95 * (1 - kk) ** 1.4
        lk = _leak(w, h) * a + (0.22 * (1 - kk) ** 4)       # + a short overall exposure flash on the first frames
        f = rgb.astype(np.float32) / 255.0
        return np.clip(np.rint((1 - (1 - f) * (1 - np.clip(lk, 0, 1))) * 255.0), 0, 255).astype(np.uint8)
    return rgb


CAMFX_EASE = 0.5


def camfx_amount(fx: dict, t: float) -> float:
    """0..1: how far a camera move is in (eased in and out over CAMFX_EASE seconds)."""
    if not (fx["t0"] <= t < fx["t1"]):
        return 0.0
    k = min(1.0, (t - fx["t0"]) / CAMFX_EASE, (fx["t1"] - t) / CAMFX_EASE)
    return k * k * k * (k * (6 * k - 15) + 10)   # smootherstep


def apply_camfx(rgb: np.ndarray, fx: dict, t: float) -> np.ndarray:
    """slide: the picture slides away from fx["side"] by fx["amt"] of the width; the freed side shows a blurred,
    darkened copy of the frame (the text panel). punch: eased zoom to fx["zoom"] around (cx, cy)."""
    e = camfx_amount(fx, t)
    if e <= 0.001:
        return rgb
    h, w = rgb.shape[:2]
    if fx["kind"] == "punch":
        return zoom_frame(rgb, 1 + (fx.get("zoom", 1.22) - 1) * e, fx.get("cx", 0.5), fx.get("cy", 0.42))
    dx = int(round(e * fx.get("amt", 0.36) * w))
    if dx < 2:
        return rgb
    small = cv2.resize(rgb, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (w, h), interpolation=cv2.INTER_LINEAR)
    out = (bg.astype(np.float32) * (1 - 0.62 * e)).astype(np.uint8)
    if fx.get("side", "left") == "left":     # panel on the left: picture moves right
        out[:, dx:] = rgb[:, :w - dx]
        edge = slice(max(0, dx - 24), dx)
    else:
        out[:, :w - dx] = rgb[:, dx:]
        edge = slice(w - dx, min(w, w - dx + 24))
    # a soft shadow where the picture meets the panel (depth)
    sh = out[:, edge].astype(np.float32)
    ramp = np.linspace(0.55, 1.0, sh.shape[1], dtype=np.float32)
    if fx.get("side", "left") != "left":
        ramp = ramp[::-1]
    out[:, edge] = (sh * ramp[None, :, None]).astype(np.uint8)
    return out


def zoom_frame(rgb: np.ndarray, z: float, cx: float, cy: float) -> np.ndarray:
    if z <= 1.001:
        return rgb
    h, w = rgb.shape[:2]
    cw, ch = w / z, h / z
    x0 = min(max(cx * w - cw / 2, 0), w - cw)
    y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[z, 0, -x0 * z], [0, z, -y0 * z]], np.float32)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  # sharper punch-ins


def fit_canvas(rgb: np.ndarray, W: int, H: int) -> np.ndarray:
    """Put a frame of another shape (e.g. a vertical phone/DJI video) into a W x H frame the way editors do:
    the whole picture in the middle, a blurred, darkened, zoomed copy of itself filling the sides."""
    h, w = rgb.shape[:2]
    if abs(w / h - W / H) < 0.12:
        return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA) if (w, h) != (W, H) else rgb
    s = min(W / w, H / h)
    fw, fh = max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)
    fg = cv2.resize(rgb, (fw, fh), interpolation=cv2.INTER_AREA)
    sc = max(W / w, H / h)
    small = cv2.resize(rgb, (max(4, int(w * sc / 12)), max(4, int(h * sc / 12))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    big = cv2.resize(small, (int(w * sc) + 2, int(h * sc) + 2), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
    out = (big[y0:y0 + H, x0:x0 + W].astype(np.float32) * 0.5).astype(np.uint8)
    ox, oy = (W - fw) // 2, (H - fh) // 2
    out[oy:oy + fh, ox:ox + fw] = fg
    return out


def canvas_geometry(src_w: int, src_h: int, W: int, H: int) -> dict:
    """Where the source picture sits inside the output frame (normalised), for placing text and keyframes."""
    if abs(src_w / src_h - W / H) < 0.12:
        return {"ox": 0.0, "oy": 0.0, "fw": 1.0, "fh": 1.0, "fit": False}
    s = min(W / src_w, H / src_h)
    fw, fh = src_w * s / W, src_h * s / H
    return {"ox": (1 - fw) / 2, "oy": (1 - fh) / 2, "fw": fw, "fh": fh, "fit": True}


def crop_vertical(rgb: np.ndarray, cx: float, out_w: int, out_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    cw = min(w, int(round(h * out_w / out_h)))
    x0 = int(round(min(max(cx * w - cw / 2, 0), w - cw)))
    return cv2.resize(rgb[:, x0:x0 + cw], (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def ease_out_back(a: float, k: float = 1.70158) -> float:
    a = min(1.0, max(0.0, a)) - 1
    return 1 + (k + 1) * a ** 3 + k * a ** 2


def anim_params(ov: "Overlay", t: float, h: int) -> tuple[float, float, float]:
    """(alpha 0..1, scale, dy pixels) of an overlay at time t."""
    a = 1.0
    if ov.fade > 0:
        a = min(1.0, (t - ov.t0) / ov.fade, (ov.t1 - t) / ov.fade)
    a = max(0.0, a)
    a = a * a * (3 - 2 * a)  # smoothstep
    sc, dy = 1.0, 0.0
    k = (t - ov.t0) / 0.35
    if ov.anim == "pop" and k < 1:
        sc = 0.55 + 0.45 * ease_out_back(k, 2.2)
        a = max(a, min(1.0, k * 3))
    elif ov.anim == "rise" and k < 1:
        e = 1 - (1 - min(1.0, k)) ** 3
        dy = (1 - e) * 0.05 * h
    elif ov.anim == "flicker" and k < 1.3:   # neon tube lighting up
        a *= (1.0, 0.15, 0.9, 0.3, 1.0, 0.6, 1.0)[min(6, int(k * 5.4))]
    elif ov.anim == "push":
        sc = 1.0 + 0.04 * min(1.0, (t - ov.t0) / max(0.1, ov.t1 - ov.t0))
    ke = (ov.t1 - t) / 0.28            # exit: pops shrink away, the others drop a little
    if ke < 1 and ov.anim in ("pop", "rise", "flicker"):
        ke = max(0.0, ke)
        if ov.anim == "pop":
            sc *= 0.86 + 0.14 * ke
        else:
            dy += (1 - ke) ** 2 * 0.025 * h
    elif ov.anim == "pop" and k >= 1:  # alive while on screen: a slow breath
        sc *= 1.0 + 0.012 * math.sin(2 * math.pi * (t - ov.t0 - 0.35) / 2.4)
    return a, sc, dy


SHINE_AT, SHINE_LEN = 0.55, 0.6


def _shine(out: np.ndarray, ov: Overlay, t: float) -> np.ndarray:
    """A diagonal glint passes once across the title after it landed (only on the title's own pixels)."""
    tt = t - ov.t0 - ov.meta.get("shine_at", SHINE_AT)
    if not ov.meta.get("shine") or not (0.0 <= tt <= SHINE_LEN):
        return out
    x0, y0, x1, y1 = ov.box
    H, W = out.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    key = "_shine_grid"
    if key not in ov.meta:
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        u = (xx + 0.45 * yy) / max(1.0, (x1 - x0) + 0.45 * (y1 - y0))
        ov.meta[key] = (u, ov.rgba[y0:y1, x0:x1, 3].astype(np.float32) / 255.0)
    u, a = ov.meta[key]
    c = -0.2 + 1.4 * (tt / SHINE_LEN)
    band = np.exp(-((u - c) / 0.07) ** 2) * 0.6 * a
    if band.max() < 0.02:
        return out
    out = out.copy() if out is not None else out
    reg = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (reg + (255.0 - reg) * band[..., None]).astype(np.uint8)
    return out


WIPE_IN, WIPE_OUT = 0.45, 0.30


def _reveal(rgb: np.ndarray, ov: Overlay, vis: int, alpha: float = 1.0) -> np.ndarray:
    """Draw the left `vis` pixels of the overlay's box."""
    x0, y0, x1, y1 = ov.box
    vis = min(vis, x1 - x0)
    if vis <= 0 or alpha <= 0.001:
        return rgb
    src = ov.rgba[y0:y1, x0:x0 + vis]
    a = src[..., 3:4].astype(np.float32) / 255.0 * alpha
    out = rgb.copy()
    dst = out[y0:y1, x0:x0 + vis].astype(np.float32)
    out[y0:y1, x0:x0 + vis] = (dst * (1 - a) + src[..., :3].astype(np.float32) * a).astype(np.uint8)
    return out


def _wipe(rgb: np.ndarray, ov: Overlay, t: float, bar: bool = True) -> np.ndarray:
    x0, y0, x1, y1 = ov.box
    bw = x1 - x0
    k_in = min(1.0, (t - ov.t0) / (WIPE_IN if bar else 0.35))
    k_out = min(1.0, max(0.0, (ov.t1 - t) / WIPE_OUT))
    e = (1 - (1 - k_in) ** 3) * (k_out ** 2 * (3 - 2 * k_out))   # ease-out in, smooth out
    vis = int(round(bw * e))
    out = _reveal(rgb, ov, vis)
    if bar and e < 0.999:   # the light bar at the moving edge
        bw_bar = max(3, int(rgb.shape[0] / 180))
        bx = min(rgb.shape[1] - bw_bar, x0 + vis)
        ys = ov.rgba[y0:y1, x0:x1, 3].max(axis=1) > 0
        if ys.any():
            r0, r1 = y0 + int(np.argmax(ys)), y0 + len(ys) - int(np.argmax(ys[::-1]))
            out = out.copy() if out is rgb else out
            out[r0:r1, bx:bx + bw_bar] = ov.meta.get("accent", (255, 212, 0))
    return out


def _type(rgb: np.ndarray, ov: Overlay, t: float) -> np.ndarray:
    cuts = ov.meta.get("cuts") or [ov.box[2] - ov.box[0]]
    n = min(len(cuts), int((t - ov.t0) / ov.meta.get("cps", 0.05)) + 1)
    k_out = min(1.0, max(0.0, (ov.t1 - t) / 0.25))
    return _reveal(rgb, ov, int(cuts[n - 1]), k_out * k_out * (3 - 2 * k_out))


def _words(rgb: np.ndarray, ov: Overlay, t: float) -> np.ndarray:
    """Kinetic sentence: meta["boxes"] (x0, y0, x1, y1) appear at meta["times"] (seconds after ov.t0); the word
    being spoken (until meta["ends"]) is emphasised; everything fades out at t1. meta["fx"]:
      karaoke - quick fade in, the spoken word turns the accent colour
      kutu    - the spoken word sits on an accent-coloured box (text turns dark/white for contrast)
      zipla   - every word pops in (scale 0.6 -> 1.12 -> 1), the spoken word in the accent colour"""
    tt = t - ov.t0
    k_out = min(1.0, max(0.0, (ov.t1 - t) / 0.3))
    k_out = k_out * k_out * (3 - 2 * k_out)
    out = rgb
    boxes, times, ends = ov.meta["boxes"], ov.meta["times"], ov.meta.get("ends") or []
    acc = np.array(ov.meta.get("accent", (255, 212, 0)), np.float32)
    fx = ov.meta.get("fx", "karaoke")
    H, W = rgb.shape[:2]
    for i, (bx0, by0, bx1, by1) in enumerate(boxes):
        if times[i] > tt:
            continue
        age = tt - times[i]
        a = min(1.0, age / (0.06 if fx == "zipla" else 0.12)) * k_out
        if a <= 0.001:
            continue
        src = ov.rgba[by0:by1, bx0:bx1]
        speaking = i < len(ends) and ends[i] is not None and tt < ends[i]
        x0, y0 = bx0, by0
        if fx != "zipla" and age < 0.18:      # each word rises into place as it is said
            y0 = min(H - (by1 - by0), by0 + int(round((1 - age / 0.18) ** 2 * 0.32 * (by1 - by0))))
        if fx == "zipla" and ends and i < len(ends) and ends[i] is not None and age < 0.2:
            sc = 0.6 + 0.52 * min(1.0, age / 0.11) if age < 0.11 else 1.12 - 0.12 * (age - 0.11) / 0.09
            nw, nh = max(2, int((bx1 - bx0) * sc)), max(2, int((by1 - by0) * sc))
            src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x0, y0 = (bx0 + bx1 - nw) // 2, (by0 + by1 - nh) // 2
            sx0, sy0 = max(0, -x0), max(0, -y0)
            x0, y0 = max(0, x0), max(0, y0)
            src = src[sy0:sy0 + min(nh, H - y0), sx0:sx0 + min(nw, W - x0)]
        hh, ww = src.shape[:2]
        if hh <= 0 or ww <= 0:
            continue
        al = src[..., 3:4].astype(np.float32) / 255.0 * a
        col = src[..., :3].astype(np.float32)
        if out is rgb:
            out = rgb.copy()
        if speaking and fx == "kutu":
            pad = max(2, (by1 - by0) // 10)
            r = out[max(0, y0 - pad // 2):y0 + hh + pad // 2, max(0, x0 - pad):x0 + ww + pad].astype(np.float32)
            out[max(0, y0 - pad // 2):y0 + hh + pad // 2, max(0, x0 - pad):x0 + ww + pad] = \
                (r * (1 - 0.92 * a) + acc * 0.92 * a).astype(np.uint8)
            dark = (0.299 * acc[0] + 0.587 * acc[1] + 0.114 * acc[2]) > 147
            bright = (col.mean(axis=2, keepdims=True) > 180).astype(np.float32)
            col = col * (1 - bright) + (np.array([12, 12, 14], np.float32) if dark else np.array([255, 255, 255], np.float32)) * bright
            al = bright * a if dark else al   # on a light box drop the dark outline, keep the letters
        elif speaking:
            bright = (col.mean(axis=2, keepdims=True) > 180).astype(np.float32)
            col = col * (1 - bright) + acc * bright
        dst = out[y0:y0 + hh, x0:x0 + ww].astype(np.float32)
        out[y0:y0 + hh, x0:x0 + ww] = (dst * (1 - al) + col * al).astype(np.uint8)
    return out


def blend(rgb: np.ndarray, ov: Overlay, t: float) -> np.ndarray:
    if not (ov.t0 <= t < ov.t1):
        return rgb
    if ov.anim == "draw":          # drawn frame by frame (animated widgets): meta["draw"](rgb, seconds since t0, ov)
        return ov.meta["draw"](rgb, t - ov.t0, ov)
    if ov.anim in ("wipe", "sweep"):
        return _shine(_wipe(rgb, ov, t, bar=ov.anim == "wipe"), ov, t)
    if ov.anim == "type":
        return _type(rgb, ov, t)
    if ov.anim == "words":
        return _words(rgb, ov, t)
    h, w = rgb.shape[:2]
    a, sc, dy = anim_params(ov, t, h)
    if a <= 0.001:
        return rgb
    x0, y0, x1, y1 = ov.box
    src = ov.rgba[y0:y1, x0:x1]
    if abs(sc - 1.0) > 1e-3 or dy:
        if abs(sc - 1.0) > 1e-3:
            nw, nh = max(2, int(round((x1 - x0) * sc))), max(2, int(round((y1 - y0) * sc)))
            src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + dy
        x0, y0 = int(round(cx - src.shape[1] / 2)), int(round(cy - src.shape[0] / 2))
        x1, y1 = x0 + src.shape[1], y0 + src.shape[0]
        # clip to the frame
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return rgb
        src = src[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = (src[..., 3:4].astype(np.float32) / 255.0) * a
    dst = rgb[y0:y1, x0:x1].astype(np.float32)
    out = rgb.copy()
    out[y0:y1, x0:x1] = (dst * (1 - alpha) + src[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    if abs(sc - 1.0) < 0.02 and not dy:
        out = _shine(out, ov, t)
    return out


# ------------------------------------------------------------------ audio --------------------------------------
def _atempo(k: float) -> str:
    """atempo chain (old ffmpeg builds take 0.5..2.0 per stage)."""
    parts = []
    while k > 2.0:
        parts.append("atempo=2.0")
        k /= 2.0
    while k < 0.5:
        parts.append("atempo=0.5")
        k /= 0.5
    parts.append(f"atempo={k:.5f}")
    return ",".join(parts)


def speed_filter(speed: float, pitch: float) -> str:
    """Faster by `speed`, voice higher by `pitch` (resample up = higher and faster, atempo makes up the rest)."""
    return f"aresample={SR},asetrate={int(SR * pitch)},aresample={SR}," + _atempo(speed / pitch)


def _pcm(path: str, t0: float, dur: float, speed: float = 1.0, pitch: float = 1.0) -> np.ndarray:
    """dur: output seconds."""
    af = ["-af", speed_filter(speed, pitch)] if abs(speed - 1.0) > 1e-3 else []
    p = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{t0:.4f}", "-i", path] + af +
                       ["-t", f"{dur:.4f}", "-vn", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                       capture_output=True, creationflags=NOWIN)
    a = np.frombuffer(p.stdout, np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
    n = int(round(dur * SR))
    if len(a) < n:
        a = np.vstack([a, np.zeros((n - len(a), 2), np.float32)])
    return a[:n]


def build_audio(pieces: list[Piece], out_wav: str, xfade_ms: float = 10.0, sfx: list[dict] | None = None) -> float:
    from ..audio.separate import voice_file   # music-free voice when separation is on
    k = int(SR * xfade_ms / 1000)
    ramp = np.linspace(0, 1, k, dtype=np.float32)[:, None]
    chunks = []
    for pc in pieces:
        if pc.kind in ("hold", "montage"):
            chunks.append(np.zeros((int(round(pc.dur * SR)), 2), np.float32))
            continue
        a = _pcm(voice_file(pc.a_file), pc.a_in, pc.dur, pc.speed, pc.pitch) if pc.a_file else \
            _pcm(voice_file(pc.file), pc.t_in, pc.dur, pc.speed, pc.pitch)
        if len(a) > 2 * k:
            a[:k] *= ramp
            a[-k:] *= ramp[::-1]
        chunks.append(a)
    audio = np.vstack(chunks) if chunks else np.zeros((1, 2), np.float32)
    for c in sfx or []:  # scene-change sounds mixed under the voice
        try:
            a = _pcm(c["file"], 0.0, min(2.5, c.get("max_s", 2.5)))
        except Exception:  # noqa: BLE001
            continue
        nz = np.nonzero(np.abs(a).max(axis=1) > 1e-4)[0]
        a = a[: nz[-1] + 1] if len(nz) else a[:0]
        i0 = int(c["t"] * SR)
        n = min(len(a), len(audio) - i0)
        if n > 0:
            peak = float(np.abs(a).max()) or 1.0
            audio[i0:i0 + n] += a[:n] / peak * (10 ** (c.get("gain_db", -17.0) / 20))
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return len(audio) / SR


def loudnorm_wav(src: str, dst: str, target: str = "I=-14:TP=-1.5:LRA=11") -> dict:
    """Two-pass EBU R128 normalisation (measure, then apply linear gain where possible)."""
    import json
    import re

    p = subprocess.run([ff.ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", src, "-af", f"loudnorm={target}:print_format=json",
                        "-f", "null", "-"], capture_output=True, creationflags=NOWIN)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr.decode("utf-8", "replace"), re.S)
    if not m:
        raise RuntimeError("Ses seviyesi ölçülemedi")
    me = json.loads(m.group(0))
    af = (f"loudnorm={target}:measured_I={me['input_i']}:measured_TP={me['input_tp']}:measured_LRA={me['input_lra']}"
          f":measured_thresh={me['input_thresh']}:offset={me['target_offset']}:linear=true")
    q = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", src, "-af", af, "-ar", str(SR), dst],
                       capture_output=True, creationflags=NOWIN)
    if q.returncode != 0:
        raise RuntimeError(q.stderr.decode("utf-8", "replace")[-400:])
    return {"before_lufs": float(me["input_i"]), "before_tp": float(me["input_tp"])}


# ------------------------------------------------------------------ video --------------------------------------
def render(pieces: list[Piece], out_path: str, out_w: int, out_h: int, fps: float,
           frame_fn: Callable[[np.ndarray, Piece, int, float], np.ndarray] | None = None,
           overlays: list[Overlay] | None = None, ass_path: str | None = None, vertical: bool = False,
           loudness: str = "I=-14:TP=-1.5:LRA=11", progress=None, cq: int = 18,
           compose: Callable | None = None, sfx: list[dict] | None = None, music: str | None = None,
           audio_master: bool = True, speech: list | None = None, image: dict | None = None, keep_audio: str | None = None,
           workers: int | None = None, upscale: tuple[int, int] | None = None, camfx: list[dict] | None = None,
           stage=None) -> dict:
    """Render pieces into one file. frame_fn(rgb, piece, global_frame_index, t_out) is applied first (retouch).
    upscale=(W, H): the finished frames are scaled (Lanczos) to W x H in the encoder - e.g. a 1080p phone video
    delivered as 1440p, which YouTube streams with its sharper, higher-bitrate codec."""
    from .vertical import _enc, pick_encoder

    out_path = str(out_path)
    work = Path(out_path).parent
    wav = str(work / (Path(out_path).stem + "_ses.wav"))
    wav_n = str(work / (Path(out_path).stem + "_ses_norm.wav"))
    if progress:
        progress(0.01, "Ses birleştiriliyor")
    build_audio(pieces, wav)
    from ..audio.master import master
    if audio_master:
        loud = master(wav, wav_n, loudness, music=music, sfx=sfx, speech=speech)
    else:  # raw voice (e.g. the camera file for Premiere; mastering happens after Premiere's export)
        import shutil
        shutil.copyfile(wav, wav_n)
        loud = {"mastering": "yok (ham ses)"}
    enc = pick_encoder()
    chain = []
    if ass_path:
        esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
        fonts = "C\\:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts"
        chain.append(f"subtitles='{esc}':fontsdir='{fonts}'")
    # RGB -> YUV with the HD (bt709) matrix and video range: without it ffmpeg uses the SD matrix and players show
    # shifted colours and slightly different brightness
    size = f"{upscale[0]}:{upscale[1]}:flags=lanczos:" if upscale else ""
    chain.append(f"scale={size}out_color_matrix=bt709:out_range=tv,format=yuv420p")
    vf = ["-vf", ",".join(chain)]
    tmp = out_path + ".yaziliyor.mp4"
    cmd = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-", "-i", wav_n, "-map", "0:v", "-map", "1:a"] + vf + \
        _enc(enc, cq) + ["-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-shortest", "-movflags", "+faststart", tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NOWIN)
    total = sum(int(round(p.dur * fps)) for p in pieces) or 1
    gi = 0
    last_out = None
    pool = None
    ahead = _Ahead(pieces, fps, out_h, vertical, image)
    if frame_fn is not None and hasattr(frame_fn, "prepare"):
        from concurrent.futures import ThreadPoolExecutor
        n_w = workers or max(2, min(8, (os.cpu_count() or 4) - 2))
        pool = ThreadPoolExecutor(max_workers=n_w)
    try:
        for k_piece, pc in enumerate(pieces):
            if pc.kind == "hold":
                n = int(round(pc.dur * fps))
                base = last_out if last_out is not None else np.zeros((out_h, out_w, 3), np.uint8)
                k = max(3, int(out_w / 40) | 1)
                frozen = (cv2.GaussianBlur(base, (k, k), 0).astype(np.float32) * 0.45).astype(np.uint8)
                for i in range(n):
                    t_out = gi / fps
                    rgb = frozen
                    for ov in overlays or []:
                        rgb = blend(rgb, ov, t_out)
                    proc.stdin.write(np.ascontiguousarray(rgb).data)
                    gi += 1
                continue
            montage = pc.kind == "montage"
            want = int(round(pc.dur * fps))
            got = 0
            last = None
            src_frames = ahead.frames(k_piece)
            frames = src_frames
            if frame_fn is not None and not montage and pool is not None and hasattr(frame_fn, "prepare"):
                frames = _parallel(frames, frame_fn, pc, gi, fps, pool, want)
                fn_done = True
            else:
                fn_done = False
            for rgb in frames:
                if got >= want:
                    break
                t_out = gi / fps
                if montage:   # slow push-in or pull-out (1.00 <-> 1.07), plus the zoom transition into the shot
                    f = got / max(1, want)
                    z = 1.0 + 0.07 * (f if pc.move >= 0 else 1 - f)
                    if pc.trans_in > 0 and got < 0.32 * fps:
                        e = 1 - (1 - got / (0.32 * fps)) ** 3
                        z *= 1 + pc.trans_in * (1 - e)
                    rgb = zoom_frame(rgb, z, 0.5, 0.5)
                elif frame_fn and not fn_done:
                    rgb = frame_fn(rgb, pc, gi, t_out)
                cam = rgb
                if vertical:
                    if pc.trans_in > 0 and got < 0.32 * fps:
                        e = 1 - (1 - got / (0.32 * fps)) ** 3
                        rgb = zoom_frame(rgb, 1 + pc.trans_in * (1 - e), pc.cx, pc.cy)
                    rgb = crop_vertical(rgb, pc.cx, out_w, out_h)
                else:
                    z = pc.zoom
                    if pc.trans_in > 0 and pc.trans == "zoom" and got < 0.32 * fps:  # zoom-in transition
                        e = 1 - (1 - got / (0.32 * fps)) ** 3
                        z = pc.zoom * (1 + pc.trans_in * (1 - e))
                    if not montage and prezoomed(pc, vertical):   # the decoder already cut the steady punch-in
                        if z > pc.zoom + 1e-4:                    # only the transition's extra zoom is left
                            fh, fw = rgb.shape[:2]
                            x0, y0, cw, ch = _crop_box(fw, fh, pc.zoom, pc.cx, pc.cy)
                            rgb = zoom_frame(rgb, z / pc.zoom, (pc.cx * fw - x0) / cw, (pc.cy * fh - y0) / ch)
                    elif not montage:
                        rgb = zoom_frame(rgb, z, pc.cx, pc.cy)
                    if rgb.shape[1] != out_w or rgb.shape[0] != out_h:
                        rgb = fit_canvas(rgb, out_w, out_h)
                    tn = TRANS_S.get(pc.trans, 0.32) * fps
                    if pc.trans_in > 0 and pc.trans != "zoom" and got < tn:
                        rgb = transition_fx(rgb, pc.trans, got / tn)
                    if camfx and not montage:   # sentence moments: slide to a text panel / punch-in
                        for fx in camfx:
                            if fx["t0"] <= t_out < fx["t1"]:
                                rgb = apply_camfx(rgb, fx, t_out)
                if stage is not None and not montage and not vertical:   # talking-head set: camera as a card
                    rgb = stage.apply(rgb, t_out)
                if compose is not None:
                    rgb = compose(rgb, cam, pc, pc.t_in + got / fps * pc.speed, t_out)
                for ov in overlays or []:
                    rgb = blend(rgb, ov, t_out)
                proc.stdin.write(np.ascontiguousarray(rgb).data)
                last = rgb
                last_out = rgb
                got += 1
                gi += 1
                if progress and gi % 90 == 0:
                    progress(0.05 + 0.93 * gi / total, f"Video işleniyor: {gi}/{total} kare")
            for _rest in src_frames:   # frames past `want` (or not taken by the pool): drain so the next piece lines up
                pass
            while got < want and last is not None:  # decoder returned a frame short: hold the last one
                proc.stdin.write(np.ascontiguousarray(last).data)
                got += 1
                gi += 1
    finally:
        ahead.close()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        try:
            proc.stdin.close()
        except OSError:
            pass
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Video kodlanamadı: {err[-800:]}")
    replace_file(tmp, out_path)
    kept = None
    if keep_audio:   # the finished video must never be lost because Premiere still holds the old sound file open
        import shutil
        for k in range(0, 20):
            dst = keep_audio if k == 0 else str(Path(keep_audio).with_name(f"{Path(keep_audio).stem}_{k + 1}{Path(keep_audio).suffix}"))
            try:
                shutil.copyfile(wav_n, dst)
                kept = dst
                break
            except OSError:
                continue
    for f in (wav, wav_n):
        try:
            os.remove(f)
        except OSError:
            pass
    return {"file": out_path, "frames": gi, "duration": gi / fps, "encoder": enc, "audio": loud, "kept_audio": kept}
