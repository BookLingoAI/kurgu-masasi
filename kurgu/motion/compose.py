"""Frame compositor for the main edit: screen recording + speaker box, b-roll cutaways, full-screen cards,
subscribe reminder. Works on top of the render engine (render.render(..., compose=...))."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..editing.director import CursorFollow, event_at
from ..integrations import ffmpeg as ff
from .captions import tr_upper
from .render import NOWIN, Overlay, zoom_frame
from .thumbnails import _font

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _smooth(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


class SeqReader:
    """Sequential frame reader at a fixed fps, scaled to fit (letterboxed) into w x h. Restarts on seeks."""

    def __init__(self, path: str, fps: float, w: int, h: int, fit: str = "contain"):
        self.path, self.fps, self.w, self.h, self.fit = path, fps, w, h, fit
        self.p = None
        self.t = None
        self.frame = None

    def _start(self, t: float) -> None:
        self.close()
        if self.fit == "cover":
            vf = f"fps={self.fps},scale={self.w}:{self.h}:force_original_aspect_ratio=increase,crop={self.w}:{self.h}"
        else:
            vf = (f"fps={self.fps},scale={self.w}:{self.h}:force_original_aspect_ratio=decrease:flags=bicubic,"
                  f"pad={self.w}:{self.h}:(ow-iw)/2:(oh-ih)/2:color=0x101010")
        self.p = subprocess.Popen([ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{max(0.0, t):.4f}", "-i", self.path,
                                   "-vf", vf, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NOWIN)
        self.t = max(0.0, t) - 1.0 / self.fps
        self.frame = None

    def _read(self) -> bool:
        n = self.w * self.h * 3
        buf = self.p.stdout.read(n) if self.p else b""
        if len(buf) < n:
            return False
        self.frame = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.t += 1.0 / self.fps
        return True

    def get(self, t: float) -> np.ndarray | None:
        if self.p is None or self.t is None or t < self.t - 0.5 / self.fps or t > self.t + 2.0:
            self._start(t)
        while self.t < t - 0.5 / self.fps:
            if not self._read():
                break
        return self.frame

    def close(self) -> None:
        if self.p:
            try:
                self.p.stdout.close()
                self.p.kill()
                self.p.wait()
            except OSError:
                pass
        self.p = None


def load_image(path: str, w: int, h: int) -> np.ndarray:
    """Photo fitted for a slow push-in: cover crop at 110% so the zoom never shows borders."""
    im = Image.open(path).convert("RGB")
    W, H = int(w * 1.1), int(h * 1.1)
    s = max(W / im.width, H / im.height)
    im = im.resize((int(im.width * s + 0.5), int(im.height * s + 0.5)), Image.LANCZOS)
    x0, y0 = (im.width - W) // 2, (im.height - H) // 2
    return np.asarray(im.crop((x0, y0, x0 + W, y0 + H)))


def ken_burns(img: np.ndarray, w: int, h: int, f: float) -> np.ndarray:
    z = 1.0 + 0.08 * _smooth(f)
    H, W = img.shape[:2]
    cw, ch = int(W / z / 1.1 * 1.0), int(H / z / 1.1 * 1.0)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    return np.asarray(m).astype(np.float32)[..., None] / 255.0


class SpeakerBox:
    """Camera in a rounded 4:3 box at the bottom-left of the frame (design_config 'demo')."""

    def __init__(self, out_w: int, out_h: int, cfg: dict):
        s = out_w / 1920.0
        ar = cfg.get("speaker_box_aspect", [4, 3])
        self.bw = int(out_w * cfg.get("speaker_box_width_fraction", 0.21)) // 2 * 2
        self.bh = int(self.bw * ar[1] / ar[0]) // 2 * 2
        m = int(out_w * cfg.get("speaker_box_left_fraction", 0.022))
        self.x0, self.y0 = m, out_h - m - self.bh
        self.border = max(2, int(cfg.get("speaker_box_border_px", 4) * s))
        r = int(cfg.get("speaker_box_corner_px", 18) * s)
        col = cfg.get("speaker_box_border_color", "#2A2A2A").lstrip("#")
        self.col = np.array([int(col[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
        self.outer = _rounded_mask(self.bw + 2 * self.border, self.bh + 2 * self.border, r + self.border)
        self.inner = _rounded_mask(self.bw, self.bh, r)

    def draw(self, frame: np.ndarray, cam: np.ndarray, cx: float, cy: float) -> np.ndarray:
        h, w = cam.shape[:2]
        ch = int(h * 0.62)
        cw = int(ch * self.bw / self.bh)
        x0 = int(min(max(cx * w - cw / 2, 0), w - cw))
        y0 = int(min(max(cy * h - ch * 0.45, 0), h - ch))
        crop = cv2.resize(cam[y0:y0 + ch, x0:x0 + cw], (self.bw, self.bh), interpolation=cv2.INTER_AREA)
        out = frame.copy()
        b = self.border
        X0, Y0 = self.x0 - b, self.y0 - b
        reg = out[Y0:Y0 + self.bh + 2 * b, X0:X0 + self.bw + 2 * b].astype(np.float32)
        reg = reg * (1 - self.outer) + self.col * self.outer
        inner = reg[b:b + self.bh, b:b + self.bw]
        reg[b:b + self.bh, b:b + self.bw] = inner * (1 - self.inner) + crop.astype(np.float32) * self.inner
        out[Y0:Y0 + self.bh + 2 * b, X0:X0 + self.bw + 2 * b] = reg.astype(np.uint8)
        return out


def draw_card(text: str, w: int, h: int) -> np.ndarray:
    """Full-screen title card (opaque RGBA)."""
    s = w / 1920.0
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = np.array([18, 20, 26], np.float32) * (1 - yy) + np.array([8, 9, 12], np.float32) * yy
    im = Image.fromarray(np.ascontiguousarray(np.broadcast_to(bg, (h, w, 3))).astype(np.uint8))
    d = ImageDraw.Draw(im)
    t = tr_upper(text)
    size = int(118 * s)
    f = _font(size)
    while f.getlength(t) > w * 0.82 and size > 40 * s:
        size -= int(6 * s) or 1
        f = _font(size)
    tw = f.getlength(t)
    x, y = (w - tw) / 2, h / 2 - size * 0.62
    d.text((x, y), t, font=f, fill=(255, 255, 255))
    bar_w = int(min(tw, w * 0.5) * 0.35)
    d.rounded_rectangle(((w - bar_w) / 2, y + size * 1.45, (w + bar_w) / 2, y + size * 1.45 + int(12 * s)), int(6 * s),
                        fill=(255, 212, 0))
    rgba = np.dstack([np.asarray(im), np.full((h, w), 255, np.uint8)])
    return rgba


def draw_subscribe(w: int, h: int) -> np.ndarray:
    """Generic subscribe reminder (no channel name or logo): red pill with a bell, lower third, centred."""
    s = w / 1920.0
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    f = _font(int(56 * s))
    t = "ABONE OL"
    tw = f.getlength(t)
    bell = int(54 * s)
    pw, ph = int(tw + bell + 110 * s), int(104 * s)
    x0, y0 = (w - pw) // 2, int(h * 0.66)
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x0 + 6, y0 + 10, x0 + pw + 6, y0 + ph + 10), ph // 2, fill=(0, 0, 0, 120))
    layer = Image.alpha_composite(layer, sh.filter(ImageFilter.GaussianBlur(int(12 * s) or 1)))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0, y0, x0 + pw, y0 + ph), ph // 2, fill=(229, 9, 20, 255))
    tx = x0 + int(40 * s)
    d.text((tx, y0 + (ph - f.size * 1.2) / 2), t, font=f, fill=(255, 255, 255, 255))
    # bell
    bx, by = tx + tw + int(26 * s), y0 + (ph - bell) // 2
    d.pieslice((bx, by, bx + bell, by + int(bell * 1.5)), 180, 360, fill=(255, 255, 255, 255))
    d.rectangle((bx, by + int(bell * 0.72), bx + bell, by + int(bell * 0.8)), fill=(255, 255, 255, 255))
    d.ellipse((bx + bell * 0.38, by + bell * 0.8, bx + bell * 0.62, by + bell * 1.0), fill=(255, 255, 255, 255))
    return np.asarray(layer)


class Compositor:
    """compose(default_frame, cam_rgb, piece, t_src, t_out) -> frame. Pieces carry .screen (file, in) mapping."""

    def __init__(self, events: list[dict], activity: dict, out_w: int, out_h: int, fps: float, demo_cfg: dict):
        self.events = [e for e in events if e["type"] in ("screen", "broll")]
        self.w, self.h, self.fps = out_w, out_h, fps
        self.box = SpeakerBox(out_w, out_h, demo_cfg)
        self.readers: dict[str, SeqReader] = {}
        self.images: dict[str, np.ndarray] = {}
        self.follow = {f: CursorFollow(a.get("cursor", []), demo_cfg.get("cursor_zoom", 1.4),
                                       demo_cfg.get("cursor_dead_zone", 0.35)) for f, a in activity.items()}
        self.used = {"screen_frames": 0, "broll_frames": 0, "missing_screen": 0}

    def _reader(self, path: str, fit: str) -> SeqReader:
        k = path + "|" + fit
        if k not in self.readers:
            self.readers[k] = SeqReader(path, self.fps, self.w, self.h, fit)
        return self.readers[k]

    def _screen(self, pc, cam, t_src):
        sc = getattr(pc, "screen", None)
        if not sc:
            return None
        ts = sc["in"] + (t_src - pc.t_in)
        fr = self._reader(sc["file"], "contain").get(ts)
        if fr is None:
            return None
        fol = self.follow.get(sc["file"])
        if fol:
            z, x, y = fol.step(ts)
            fr = zoom_frame(fr, z, x, y)
        return self.box.draw(fr, cam, pc.cx, pc.cy)

    def _broll(self, e, t_out):
        f = (t_out - e["t0"]) / max(0.1, e["t1"] - e["t0"])
        if e.get("image"):
            if e["file"] not in self.images:
                self.images[e["file"]] = load_image(e["file"], self.w, self.h)
            return ken_burns(self.images[e["file"]], self.w, self.h, f)
        return self._reader(e["file"], "cover").get(t_out - e["t0"])

    def __call__(self, frame, cam, pc, t_src, t_out):
        e = event_at(self.events, t_out)
        if e is None:
            return frame
        alt = self._screen(pc, cam, t_src) if e["type"] == "screen" else self._broll(e, t_out)
        if alt is None:
            self.used["missing_screen"] += 1
            return frame
        self.used["screen_frames" if e["type"] == "screen" else "broll_frames"] += 1
        # montage: back-to-back b-roll cuts hard between clips, fades only into/out of the camera
        joined_in = any(abs(o["t1"] - e["t0"]) < 0.02 and o is not e for o in self.events)
        joined_out = any(abs(o["t0"] - e["t1"]) < 0.02 and o is not e for o in self.events)
        a = _smooth(min(1.0 if joined_in else (t_out - e["t0"]) / 0.25, 1.0 if joined_out else (e["t1"] - t_out) / 0.25, 1.0))
        if a >= 0.999:
            return alt
        return (frame.astype(np.float32) * (1 - a) + alt.astype(np.float32) * a).astype(np.uint8)

    def close(self):
        for r in self.readers.values():
            r.close()


def overlays_for(events: list[dict], w: int, h: int) -> list[Overlay]:
    out = []
    for e in events:
        if e["type"] == "card":
            out.append(Overlay(e["t0"], e["t1"], draw_card(e["text"], w, h), fade=0.2, box=(0, 0, w, h), anim="push"))
        elif e["type"] == "subscribe":
            out.append(Overlay(e["t0"], e["t1"], draw_subscribe(w, h), fade=0.3, anim="pop"))
    return out


def media_duration(path: str) -> float | None:
    if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return None
    try:
        return ff.probe(path).duration
    except Exception:  # noqa: BLE001
        return None


def draw_headline(text: str, w: int, h: int) -> np.ndarray:
    """Shorts top title (first seconds): white bold text on a yellow-accent dark pill, upper area."""
    s = w / 1080.0
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    t = tr_upper(text)
    size = int(78 * s)
    f = _font(size)
    while f.getlength(t) > w * 0.84 and size > 40 * s:
        size -= int(4 * s) or 1
        f = _font(size)
    tw = f.getlength(t)
    pw, ph = int(tw + 70 * s), int(size * 1.55)
    x0, y0 = (w - pw) // 2, int(h * 0.12)
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0, y0, x0 + pw, y0 + ph), int(22 * s), fill=(12, 12, 14, 225))
    from .titles import _ac
    d.rounded_rectangle((x0, y0 + ph - int(10 * s), x0 + pw, y0 + ph), int(5 * s), fill=_ac() + (255,))
    d.text(((w - tw) / 2, y0 + (ph - size * 1.2) / 2), t, font=f, fill=(255, 255, 255, 255))
    return np.asarray(layer)


def draw_outro(w: int, h: int, lines: tuple[str, str] = ("Videonun uzun versiyonu", "YouTube kanalımda")) -> np.ndarray:
    """End card text for Shorts: two centred lines + a red play button."""
    s = w / 1080.0
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f1, f2 = _font(int(64 * s)), _font(int(92 * s))
    t1, t2 = tr_upper(lines[0]), tr_upper(lines[1])
    while f2.getlength(t2) > w * 0.86:
        f2 = _font(int(f2.size * 0.92))
    while f1.getlength(t1) > w * 0.82:
        f1 = _font(int(f1.size * 0.92))
    cy = h * 0.40
    d.text(((w - f1.getlength(t1)) / 2, cy), t1, font=f1, fill=(255, 255, 255, 255))
    d.text(((w - f2.getlength(t2)) / 2, cy + f1.size * 1.35), t2, font=f2, fill=(255, 212, 0, 255))
    bw, bh = int(170 * s), int(120 * s)
    bx, by = (w - bw) // 2, int(cy + f1.size * 1.35 + f2.size * 1.6)
    d.rounded_rectangle((bx, by, bx + bw, by + bh), int(30 * s), fill=(229, 9, 20, 255))
    d.polygon([(bx + bw * 0.40, by + bh * 0.28), (bx + bw * 0.40, by + bh * 0.72), (bx + bw * 0.68, by + bh * 0.5)],
              fill=(255, 255, 255, 255))
    return np.asarray(layer)
