"""Animated subscribe / like / comment reminder, drawn frame by frame (no logo, no channel branding).

A dark glass bar slides in from the side with a springy overshoot, its buttons pop in one after another, a mouse
cursor glides to each button and clicks it at the very word that asks for it:
  subscribe - red "ABONE OL" pill: press, ripple, turns grey "ABONE OLUNDU" with a tick; the bell next to it rings
              (damped swing, sound waves) and gets a notification dot;
  like      - thumbs-up button: press, fills with the accent colour, bounces, a burst of sparks and a "+1" floats up;
  comment   - speech-bubble button: press, a white bubble "Yorumlara yaz!" unrolls with typing dots.
At the end everything slides back out. build_cta() returns an Overlay (anim "draw") and the sounds to place.
Drawn at 2x and downsampled (smooth edges); each frame is drawn once and cached.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .fonts import font
from .titles import _ac, _dark_text

RED = (229, 23, 32)
GREY = (62, 62, 68)
GLASS = (16, 16, 20)
WHITE = (255, 255, 255)
SS = 2           # supersampling


def _clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if x < a else b if x > b else x


def ease_out_back(k: float, o: float = 1.9) -> float:
    k = _clamp(k) - 1
    return 1 + (o + 1) * k ** 3 + o * k ** 2


def ease_io(k: float) -> float:
    k = _clamp(k)
    return 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2


def ease_out(k: float) -> float:
    return 1 - (1 - _clamp(k)) ** 3


# ------------------------------------------------------------------ icons (local units: d = button size) --------
def _rot(pts, ang, px, py):
    c, s = math.cos(ang), math.sin(ang)
    return [(px + (x - px) * c - (y - py) * s, py + (x - px) * s + (y - py) * c) for x, y in pts]


def _bell(d: ImageDraw.ImageDraw, cx: float, cy: float, size: float, ang: float, col) -> None:
    u = size
    body = []
    for k in range(0, 19):          # dome
        a = math.pi + math.pi * k / 18
        body.append((cx + 0.25 * u * math.cos(a), cy - 0.04 * u + 0.27 * u * math.sin(a)))
    body += [(cx + 0.27 * u, cy + 0.20 * u), (cx + 0.36 * u, cy + 0.29 * u), (cx - 0.36 * u, cy + 0.29 * u),
             (cx - 0.27 * u, cy + 0.20 * u)]
    piv = (cx, cy - 0.36 * u)
    d.polygon(_rot(body, ang, *piv), fill=col)
    kx, ky = _rot([(cx, cy - 0.33 * u)], ang, *piv)[0]
    d.ellipse((kx - 0.06 * u, ky - 0.06 * u, kx + 0.06 * u, ky + 0.06 * u), fill=col)
    clx, cly = _rot([(cx + 0.9 * math.sin(-ang * 1.6) * 0.05 * u, cy + 0.37 * u)], ang, *piv)[0]
    d.ellipse((clx - 0.075 * u, cly - 0.075 * u, clx + 0.075 * u, cly + 0.075 * u), fill=col)


def _thumb(d: ImageDraw.ImageDraw, cx: float, cy: float, u: float, col) -> None:
    d.rounded_rectangle((cx - 0.33 * u, cy - 0.04 * u, cx - 0.18 * u, cy + 0.32 * u), radius=0.03 * u, fill=col)
    d.rounded_rectangle((cx - 0.14 * u, cy - 0.06 * u, cx + 0.30 * u, cy + 0.32 * u), radius=0.08 * u, fill=col)
    d.polygon([(cx - 0.14 * u, cy - 0.02 * u), (cx - 0.02 * u, cy - 0.16 * u), (cx + 0.02 * u, cy - 0.36 * u),
               (cx + 0.10 * u, cy - 0.38 * u), (cx + 0.15 * u, cy - 0.30 * u), (cx + 0.11 * u, cy - 0.08 * u),
               (cx + 0.20 * u, cy - 0.06 * u)], fill=col)


def _bubble(d: ImageDraw.ImageDraw, cx: float, cy: float, u: float, col, dots_col, dots: int = 3) -> None:
    d.rounded_rectangle((cx - 0.31 * u, cy - 0.25 * u, cx + 0.31 * u, cy + 0.15 * u), radius=0.12 * u, fill=col)
    d.polygon([(cx - 0.16 * u, cy + 0.12 * u), (cx - 0.22 * u, cy + 0.32 * u), (cx - 0.02 * u, cy + 0.13 * u)], fill=col)
    for k in range(dots):
        x = cx + (k - 1) * 0.15 * u
        d.ellipse((x - 0.045 * u, cy - 0.095 * u, x + 0.045 * u, cy - 0.005 * u), fill=dots_col)


def _tick(d: ImageDraw.ImageDraw, x: float, y: float, u: float, col, k: float = 1.0) -> None:
    pts = [(x, y), (x + 0.32 * u, y + 0.30 * u), (x + 0.86 * u, y - 0.30 * u)]
    k = _clamp(k)
    if k < 0.4:
        e = k / 0.4
        pts = [pts[0], (pts[0][0] + (pts[1][0] - pts[0][0]) * e, pts[0][1] + (pts[1][1] - pts[0][1]) * e)]
    else:
        e = (k - 0.4) / 0.6
        pts = [pts[0], pts[1], (pts[1][0] + (pts[2][0] - pts[1][0]) * e, pts[1][1] + (pts[2][1] - pts[1][1]) * e)]
    d.line(pts, fill=col, width=max(2, int(0.13 * u)), joint="curve")


def _cursor(d: ImageDraw.ImageDraw, x: float, y: float, u: float) -> None:
    shape = [(0, 0), (0, 1.0), (0.26, 0.76), (0.44, 1.12), (0.58, 1.05), (0.40, 0.70), (0.74, 0.70)]
    pts = [(x + px * u, y + py * u) for px, py in shape]
    d.polygon([(px + 0.05 * u, py + 0.06 * u) for px, py in pts], fill=(0, 0, 0, 90))
    d.polygon(pts, fill=WHITE, outline=(10, 10, 12, 255), width=max(2, int(0.07 * u)))


# ------------------------------------------------------------------ layout + timing ------------------------------
class CTA:
    def __init__(self, items: list[tuple[str, float]], t0: float, t1: float, w: int, h: int, side: str = "left",
                 captions: bool = True):
        self.w, self.h = w, h
        self.t0, self.t1 = t0, t1
        s = h / 1080.0
        self.s = s
        self.u = 100 * s                     # button height
        self.pad = 18 * s
        self.gap = 14 * s
        self.fb = font("bold", int(38 * s * SS))
        self.fl = font("bold", int(33 * s * SS))
        self.accent = _ac()
        # buttons in the order they were asked for; subscribe always brings its bell
        kinds = []
        for k, _ in items:
            if k == "bell" and "subscribe" in kinds:
                continue
            kinds.append(k)
            if k == "subscribe":
                kinds.append("bell")
        self.kinds = kinds
        sub_w = max(self.fb.getlength("ABONE OL"), self.fb.getlength("ABONE OLUNDU") + 0.9 * self.u * SS) / SS + 2 * 30 * s
        self.btn = []
        x = self.pad
        for k in kinds:
            bw = sub_w if k == "subscribe" else self.u
            self.btn.append({"kind": k, "x": x, "w": bw})
            x += bw + self.gap
        self.cw = x - self.gap + self.pad
        self.ch = self.u + 2 * self.pad
        self.label_w = (self.fl.getlength("Yorumlara yaz!") / SS + 70 * s) if "comment" in kinds else 0.0
        # sprite: container + label + room for the cursor/sparks around it
        self.mx, self.my = 70 * s, 90 * s
        self.sw = int(self.cw + self.label_w + 2 * self.mx)
        self.sh = int(self.ch + 2 * self.my)
        margin = 0.05 * w
        y = h * (0.62 if captions else 0.74)
        x0 = margin if side == "left" else w - margin - self.cw - self.label_w
        self.ox, self.oy = int(x0 - self.mx), int(y - self.my)
        self.side = side
        # click times: the asking word, but never before the bar is in, and >= 0.6 s apart
        when = {}
        for k, t in items:
            kk = "subscribe" if k == "bell" and "subscribe" in kinds else k
            when.setdefault(kk, t - t0)
        clicks = []
        last = 0.85
        for b in self.btn:
            if b["kind"] == "bell":
                continue
            ct = max(last, when.get(b["kind"], last) + 0.1)
            b["click"] = ct
            clicks.append(ct)
            last = ct + 0.62
        for b in self.btn:
            if b["kind"] == "bell":
                sub = next(x for x in self.btn if x["kind"] == "subscribe")
                b["click"] = sub["click"] + 0.3      # the bell starts ringing after the subscribe click
        self.dur = max(t1 - t0, (max(clicks) if clicks else 1.0) + 1.9)
        self.sounds = [(0.0, "whoosh", -4)]
        for b in self.btn:
            if b["kind"] == "bell":
                self.sounds.append((b["click"], "bell", -2))
            elif b["kind"] == "like":
                self.sounds += [(b["click"], "click", 0), (b["click"] + 0.04, "pop", -2), (b["click"] + 0.08, "sparkle", -6)]
            else:
                self.sounds.append((b["click"], "click", 0))
        self.sounds.append((self.dur - 0.45, "swish", -8))
        self._cache: dict = {}

    # centre of a button in sprite coordinates (1x)
    def _centre(self, b) -> tuple[float, float]:
        return self.mx + b["x"] + b["w"] / 2, self.my + self.pad + self.u / 2

    def _cursor_pos(self, tt: float) -> tuple[float, float] | None:
        targets = [(b["click"], self._centre(b)) for b in self.btn if b["kind"] != "bell"]
        if not targets or tt < 0.3 or tt > self.dur - 0.35:
            return None
        start = (self.mx + self.cw * 0.5 + 140 * self.s, self.my + self.ch + 110 * self.s)
        prev_t, prev_p = 0.3, start
        for ct, p in targets:
            arrive = ct - 0.06
            if tt <= arrive:
                k = ease_io((tt - prev_t) / max(0.05, arrive - prev_t))
                return prev_p[0] + (p[0] - prev_p[0]) * k, prev_p[1] + (p[1] - prev_p[1]) * k
            if tt <= ct + 0.5:
                return p
            prev_t, prev_p = ct + 0.5, p
        k = ease_io((tt - prev_t) / 0.5)      # glide away a little after the last click
        return prev_p[0] + 60 * self.s * k, prev_p[1] + 50 * self.s * k

    def frame(self, tt: float) -> np.ndarray:
        key = int(round(tt * 60))
        if key in self._cache:
            return self._cache[key]
        if len(self._cache) > 600:
            self._cache.clear()
        S = SS
        img = Image.new("RGBA", (self.sw * S, self.sh * S), (0, 0, 0, 0))
        k_in = _clamp(tt / 0.55)
        k_out = _clamp((self.dur - tt) / 0.4)
        slide = (1 - ease_out_back(k_in, 1.4)) + (1 - ease_out(k_out)) * 0.8
        dx = -slide * (self.cw + self.mx + 0.05 * self.w) if self.side == "left" else slide * (self.cw + self.label_w + self.mx + 0.05 * self.w)
        alpha = min(1.0, k_in * 2.5) * min(1.0, k_out * 1.6)
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        X = lambda v: (v + dx) * S   # noqa: E731
        Y = lambda v: v * S          # noqa: E731
        # glass bar + soft shadow
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((X(self.mx + 6 * self.s), Y(self.my + 10 * self.s), X(self.mx + self.cw + 6 * self.s),
                                              Y(self.my + self.ch + 10 * self.s)), radius=self.ch / 2 * S, fill=(0, 0, 0, 120))
        lay = Image.alpha_composite(lay, sh.filter(ImageFilter.GaussianBlur(12 * self.s * S)))
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle((X(self.mx), Y(self.my), X(self.mx + self.cw), Y(self.my + self.ch)), radius=self.ch / 2 * S,
                            fill=GLASS + (215,), outline=(255, 255, 255, 40), width=max(1, int(2 * self.s * S)))
        u = self.u
        for j, b in enumerate(self.btn):
            k_pop = ease_out_back((tt - 0.18 - 0.08 * j) / 0.32, 2.4)
            if k_pop <= 0.01:
                continue
            cx, cy = self._centre(b)
            ct = b["click"]
            done = tt >= ct + 0.06
            press = 1.0 - 0.12 * math.sin(math.pi * _clamp((tt - ct) / 0.16)) if ct <= tt <= ct + 0.16 else 1.0
            bounce = 1.0
            if b["kind"] == "like" and done:
                bounce = 1.0 + 0.22 * math.sin(math.pi * _clamp((tt - ct - 0.06) / 0.28))
            sc = k_pop * press * bounce
            bw, bh = b["w"] * sc, u * sc
            x0, y0, x1, y1 = X(cx - bw / 2), Y(cy - bh / 2), X(cx + bw / 2), Y(cy + bh / 2)
            if b["kind"] == "subscribe":
                d.rounded_rectangle((x0, y0, x1, y1), radius=bh / 2 * S, fill=(GREY if done else RED) + (255,))
                txt = "ABONE OLUNDU" if done else "ABONE OL"
                f = self.fb
                tw = f.getlength(txt)
                tick_w = 0.9 * u * S if done else 0
                tx = (x0 + x1) / 2 - (tw + tick_w) / 2
                ty = (y0 + y1) / 2 - f.size * 0.62
                if sc > 0.6:
                    d.text((tx, ty), txt, font=f, fill=WHITE + (255,))
                    if done:
                        _tick(d, tx + tw + 0.18 * u * S, (y0 + y1) / 2 - 0.02 * u * S, 0.55 * u * S, WHITE + (255,),
                              (tt - ct - 0.06) / 0.3)
            else:
                filled = done and b["kind"] == "like"
                col = self.accent if filled else (44, 44, 50)
                d.ellipse((x0, y0, x1, y1), fill=col + (255,))
                icon = (12, 12, 14, 255) if filled and _dark_text(self.accent) else WHITE + (255,)
                if b["kind"] == "bell":
                    ring = tt - ct
                    ang = math.radians(26) * math.sin(2 * math.pi * 4.2 * ring) * math.exp(-2.6 * ring) if ring > 0 else 0.0
                    _bell(d, X(cx), Y(cy), bh * S * 0.92, ang, icon)
                    if 0 < ring < 1.2:      # sound waves
                        a = int(200 * (1 - ring / 1.2))
                        for side in (-1, 1):
                            for r in (0.62, 0.8):
                                rr = r * bh * S
                                box = (X(cx) - rr, Y(cy) - rr, X(cx) + rr, Y(cy) + rr)
                                st = -30 if side > 0 else 150
                                d.arc(box, st, st + 60, fill=WHITE + (a,), width=max(2, int(4 * self.s * S)))
                    if ring > 0.9:          # notification dot
                        r = 0.14 * u * S * ease_out_back((ring - 0.9) / 0.25, 3)
                        d.ellipse((x1 - 0.32 * bw * S - r, y0 + 0.12 * bh * S - r, x1 - 0.32 * bw * S + r,
                                   y0 + 0.12 * bh * S + r), fill=RED + (255,))
                elif b["kind"] == "like":
                    _thumb(d, X(cx), Y(cy), bh * S, icon)
                else:
                    _bubble(d, X(cx), Y(cy), bh * S, icon, (44, 44, 50, 255))
            # ripple ring at the click
            if ct <= tt <= ct + 0.5:
                e = ease_out((tt - ct) / 0.5)
                rr = (0.5 + 0.9 * e) * u * S
                d.ellipse((X(cx) - rr, Y(cy) - rr, X(cx) + rr, Y(cy) + rr), outline=WHITE + (int(220 * (1 - e)),),
                          width=max(2, int(5 * self.s * S)))
            # like: sparks + "+1"
            if b["kind"] == "like" and ct <= tt <= ct + 0.9:
                e = ease_out((tt - ct) / 0.6)
                a = int(255 * (1 - _clamp((tt - ct - 0.3) / 0.6)))
                for q in range(10):
                    th = 2 * math.pi * q / 10 + 0.3
                    r0 = (0.55 + 0.75 * e) * u * S
                    px, py = X(cx) + math.cos(th) * r0, Y(cy) + math.sin(th) * r0
                    rs = (0.07 - 0.04 * e) * u * S
                    d.ellipse((px - rs, py - rs, px + rs, py + rs), fill=(self.accent if q % 2 else WHITE) + (a,))
                f = self.fl
                d.text((X(cx) + 0.2 * u * S, Y(cy) - (0.9 + 0.6 * e) * u * S), "+1", font=f, fill=self.accent + (a,))
            # comment: the bubble label unrolls to the right of the bar
            if b["kind"] == "comment" and tt >= ct + 0.05:
                e = ease_out_back((tt - ct - 0.05) / 0.35, 1.6)
                lw = self.label_w * _clamp(e, 0, 1.08)
                lx0 = X(self.mx + self.cw + 12 * self.s)
                ly0, ly1 = Y(cy - 0.36 * u), Y(cy + 0.36 * u)
                if lw > 8 * self.s:
                    d.rounded_rectangle((lx0, ly0, lx0 + lw * S, ly1), radius=0.36 * u * S, fill=WHITE + (255,))
                    d.polygon([(lx0 + 2, Y(cy) - 0.12 * u * S), (lx0 - 0.16 * u * S, Y(cy)), (lx0 + 2, Y(cy) + 0.12 * u * S)],
                              fill=WHITE + (255,))
                    if e > 0.85:
                        f = self.fl
                        txt = "Yorumlara yaz!"
                        n = int(len(txt) * _clamp((tt - ct - 0.35) / 0.45))
                        d.text((lx0 + 26 * self.s * S, Y(cy) - f.size * 0.6), txt[:n], font=f, fill=(18, 18, 20, 255))
        # cursor on top
        cp = self._cursor_pos(tt)
        if cp is not None:
            _cursor(d, (cp[0] - 6 * self.s + dx) * S, (cp[1] - 4 * self.s) * S, 56 * self.s * S)
        img = Image.alpha_composite(img, lay)
        small = img.resize((self.sw, self.sh), Image.LANCZOS)
        arr = np.asarray(small).copy()
        if alpha < 0.999:
            arr[..., 3] = (arr[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        self._cache[key] = arr
        return arr

    def draw(self, rgb: np.ndarray, tt: float, ov=None) -> np.ndarray:
        spr = self.frame(tt)
        H, W = rgb.shape[:2]
        x0, y0 = self.ox, self.oy
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x1, y1 = min(W, x0 + spr.shape[1]), min(H, y0 + spr.shape[0])
        x0c, y0c = max(0, x0), max(0, y0)
        if x1 <= x0c or y1 <= y0c:
            return rgb
        src = spr[sy0:sy0 + (y1 - y0c), sx0:sx0 + (x1 - x0c)]
        a = src[..., 3:4].astype(np.float32) / 255.0
        if not a.any():
            return rgb
        out = rgb.copy()
        dst = out[y0c:y1, x0c:x1].astype(np.float32)
        out[y0c:y1, x0c:x1] = (dst * (1 - a) + src[..., :3].astype(np.float32) * a).astype(np.uint8)
        return out


    def static_rgba(self) -> np.ndarray:
        """The finished state (everything clicked) as a full-frame RGBA, for the Premiere review project."""
        spr = self.frame(min(self.dur - 0.5, max(b["click"] for b in self.btn) + 1.0))
        out = np.zeros((self.h, self.w, 4), np.uint8)
        x0, y0 = max(0, self.ox), max(0, self.oy)
        x1, y1 = min(self.w, self.ox + spr.shape[1]), min(self.h, self.oy + spr.shape[0])
        out[y0:y1, x0:x1] = spr[y0 - self.oy:y1 - self.oy, x0 - self.ox:x1 - self.ox]
        return out


def build_cta(ev: dict, w: int, h: int, side: str = "left", captions: bool = True):
    """Overlay + [(seconds after t0, sound name, gain dB)] for one reminder event {t0, t1, items}."""
    from .render import Overlay
    c = CTA([tuple(x) for x in ev["items"]], ev["t0"], ev["t1"], w, h, side, captions)
    ov = Overlay(ev["t0"], ev["t0"] + c.dur, np.zeros((1, 1, 4), np.uint8), fade=0.0,
                 box=(c.ox, c.oy, c.ox + c.sw, c.oy + c.sh), anim="draw", meta={"draw": c.draw, "static": c.static_rgba})
    return ov, c.sounds
