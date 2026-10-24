"""Editorial motion typography (the clean look of modern explainer channels):

  topic  - a section title on the side panel the camera slides away to: a small letter-spaced kicker
           ("SIRADAKİ"), one huge grotesk word ("Soslar") and an italic serif line in the accent colour
           ("ve eklentiler"). The kicker types in, the big word rises out of a mask with a slight overshoot,
           the italic line is unveiled from the left; at the end everything lifts up and fades.
  bullets - white pills with a dark round icon, one per item of something the speaker lists
           ("Chili sos / Hoisin sos / Soya sosu"), each sliding in at the moment it is said; an optional kicker
           above. The last item can carry the accent colour.

Each is drawn from a few pre-rendered pieces per frame (moves, masks, fades in NumPy), returned as an Overlay with
anim "draw" plus the sounds to place.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .fonts import font
from .titles import _ac


def _clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if x < a else b if x > b else x


def _out(k: float) -> float:
    return 1 - (1 - _clamp(k)) ** 3


def _back(k: float, o: float = 1.6) -> float:
    k = _clamp(k) - 1
    return 1 + (o + 1) * k ** 3 + o * k ** 2


def _text_img(text: str, f, fill, shadow: bool = True, spacing: float = 0.0) -> np.ndarray:
    """RGBA of one line (optional letter spacing in em), with a soft drop shadow for legibility on video."""
    if spacing:
        widths = [f.getlength(ch) for ch in text]
        w = int(sum(widths) + spacing * f.size * max(0, len(text) - 1)) + 8
    else:
        w = int(f.getlength(text)) + 8
    asc, desc = f.getmetrics()
    h = asc + desc + 8
    pad = int(f.size * 0.25) if shadow else 0
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
        _draw_line(ds, text, f, pad + 2, pad + 4, (0, 0, 0, 150), spacing)
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(max(2, f.size // 14))))
        d = ImageDraw.Draw(img)
    _draw_line(d, text, f, pad, pad, fill, spacing)
    return np.asarray(img)


def _draw_line(d, text, f, x, y, fill, spacing):
    if not spacing:
        d.text((x, y), text, font=f, fill=fill)
        return
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + spacing * f.size


def _blit(dst: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0, clip: tuple | None = None) -> None:
    """Alpha-blend src onto dst at (x, y), optionally only inside clip=(x0, y0, x1, y1) (dst coordinates)."""
    if alpha <= 0.003:
        return
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = x, y, x + w, y + h
    if clip:
        x0, y0, x1, y1 = max(x0, clip[0]), max(y0, clip[1]), min(x1, clip[2]), min(y1, clip[3])
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255.0 * alpha
    reg = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (reg * (1 - a) + s[..., :3].astype(np.float32) * a).astype(np.uint8)


def split_title(text: str) -> tuple[str, str]:
    """"SOS VE EKLENTİLER" -> ("Sos", "ve eklentiler"): one big word + the rest as the italic line."""
    from .captions import tr_lower
    ws = [w for w in text.split() if w]
    if not ws:
        return "", ""
    low = [tr_lower(w) for w in ws]
    k = 2 if len(ws) >= 3 and len(ws[0]) <= 3 and low[1] not in {"ve", "ile", "de", "da", "ya", "veya", "mi", "mı"} else 1
    big = " ".join(low[:k])
    big = big[:1].upper().replace("i", "İ") + big[1:] if big[:1] == "i" else big[:1].upper() + big[1:]
    return big, " ".join(low[k:])


class Topic:
    def __init__(self, kicker: str, big: str, small: str, w: int, h: int, side: str = "left", dur: float = 3.6,
                 panel_frac: float = 0.36, cy: float = 0.47, t_small: float | None = None,
                 steps: tuple[int, int] | None = None, icon: str | None = None):
        from .captions import tr_upper
        from .icons import icon_rgba
        s = h / 1080.0
        self.w, self.h, self.dur, self.side = w, h, dur, side
        acc = _ac()
        self.kick = _text_img(tr_upper(kicker), font("mono", int(32 * s)), (235, 235, 238, 240), spacing=0.32) if kicker else None
        # topic icon (Lucide) in the accent colour, left of the kicker line (or on its own line without a kicker)
        self.icon = icon_rgba(icon, int(58 * s), acc) if icon else None
        self.dx_kick = 0
        if self.icon is not None:
            self.dx_kick = self.icon.shape[1] + int(18 * s)
            if self.kick is None:
                self.kick = np.zeros((self.icon.shape[0] + int(10 * s), 1, 4), np.uint8)
        size = int(176 * s)
        f = font("display", size)
        max_w = panel_frac * w * 0.86
        while f.getlength(big) > max_w and size > 60 * s:
            size = int(size * 0.92)
            f = font("display", size)
        self.big = _text_img(big, f, (255, 255, 255, 255))
        isz = int(104 * s)
        fi = font("italic", isz)
        while small and fi.getlength(small) > max_w * 1.05 and isz > 40 * s:
            isz = int(isz * 0.92)
            fi = font("italic", isz)
        self.small = _text_img(small, fi, acc + (255,)) if small else None
        panel_w = panel_frac * w
        self.x = int((w - panel_w) + panel_w * 0.07) if side == "right" else int(max(panel_w * 0.07, 0.055 * w))
        hk = self.kick.shape[0] if self.kick is not None else 0
        hb, hs = self.big.shape[0], (self.small.shape[0] if self.small is not None else 0)
        self.steps = _steps_img(*steps, s, acc) if steps else None
        hst = (self.steps[0].shape[0] + int(30 * s)) if self.steps else 0
        total = hk + hb + hs - int(0.18 * hb) + hst
        self.y = int(h * cy - total / 2)
        self.yk, self.yb = self.y, self.y + hk - int(0.05 * hb)
        self.ys = self.yb + hb - int(0.24 * hb)
        self.yst = self.ys + hs + int(18 * s)
        self.t_small = 0.55 if t_small is None else max(0.4, t_small)
        self.sounds = [(0.05, "whoosh", -6), (0.32, "pop", -12), (0.62, "swish", -10)]

    def draw(self, rgb: np.ndarray, tt: float, ov=None) -> np.ndarray:
        out = rgb.copy()
        k_out = _clamp((self.dur - tt) / 0.4)
        lift = int((1 - _out(k_out)) * 0.04 * self.h)
        fade = k_out ** 1.5
        if self.icon is not None:   # pops in just before the kicker types
            ki = _clamp((tt - 0.02) / 0.3)
            if ki > 0:
                ih = self.icon.shape[0]
                px = max(2, int(ih * max(0.05, _back(ki, 1.8))))
                ic = np.asarray(Image.fromarray(self.icon).resize((px, px), Image.BILINEAR))
                yc = self.yk - lift + self.kick.shape[0] // 2
                _blit(out, ic, self.x + ih // 2 - px // 2, yc - px // 2, fade * min(1.0, ki * 2))
        if self.kick is not None:   # typed in, letter by letter (mask from the left)
            k = _clamp((tt - 0.12) / 0.35)
            xk = self.x + self.dx_kick
            wk = int(self.kick.shape[1] * k)
            _blit(out, self.kick, xk, self.yk - lift, fade, clip=(xk, 0, xk + wk, self.h))
        kb = _clamp((tt - 0.22) / 0.45)
        if kb > 0:                  # rises out of a mask under its baseline, slight overshoot
            hb = self.big.shape[0]
            dy = int((1 - _back(kb, 1.3)) * hb * 0.9)
            _blit(out, self.big, self.x, self.yb + dy - lift, fade * min(1.0, kb * 3),
                  clip=(0, self.yb - lift - int(0.2 * hb), self.w, self.yb - lift + hb))
        if self.small is not None:  # unveiled from the left (as it is said, when the time is known)
            ks = _out((tt - self.t_small) / 0.55)
            ws = int(self.small.shape[1] * ks)
            _blit(out, self.small, self.x, self.ys - lift, fade, clip=(self.x, 0, self.x + ws, self.h))
        if self.steps:              # progress row: the reached steps light up one after another
            off, on = self.steps
            kk = _clamp((tt - self.t_small - 0.3) / 0.5)
            _blit(out, off, self.x, self.yst - lift, fade * kk)
            _blit(out, on, self.x, self.yst - lift, fade * kk, clip=(self.x, 0, self.x + int(on.shape[1] * _out(kk)), self.h))
        return out


class Bullets:
    def __init__(self, kicker: str, items: list[str], times: list[float], w: int, h: int, side: str = "left",
                 dur: float = 5.0, accent_last: bool = True, theme: str = "light", icon: str = "arrow",
                 cy: float = 0.44):
        from .captions import tr_upper
        s = h / 1080.0
        self.w, self.h, self.dur, self.side = w, h, dur, side
        self.times = times
        y_mid = cy
        acc = _ac()
        self.kick = _text_img(tr_upper(kicker), font("mono", int(30 * s)), (250, 250, 252, 245), spacing=0.32) if kicker else None
        f = font("text", int(54 * s))
        self.pills = []
        ph = int(102 * s)
        for i, it in enumerate(items):
            tw = int(f.getlength(it))
            pw = int(ph * 1.05 + tw + 46 * s)
            img = Image.new("RGBA", (pw + 30, ph + 30), (0, 0, 0, 0))
            sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).rounded_rectangle((15, 19, 15 + pw, 19 + ph), radius=ph // 2, fill=(0, 0, 0, 90))
            img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(8)))
            d = ImageDraw.Draw(img)
            dark = theme == "dark"
            if dark:
                d.rounded_rectangle((15, 15, 15 + pw, 15 + ph), radius=ph // 2, fill=(26, 26, 29, 240),
                                    outline=(255, 255, 255, 34), width=max(1, int(2 * s)))
            else:
                d.rounded_rectangle((15, 15, 15 + pw, 15 + ph), radius=ph // 2, fill=(255, 255, 255, 238))
            r = int(ph * 0.34)
            cx, cy = 15 + int(ph * 0.52), 15 + ph // 2
            disc = {"x": (232, 62, 52), "check": (48, 190, 92)}.get(icon, acc if dark else (14, 14, 16))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=disc + (255,))
            u = r * 0.55
            lw = max(2, int(r * 0.16))
            if icon == "x":
                d.line([(cx - u * 0.62, cy - u * 0.62), (cx + u * 0.62, cy + u * 0.62)], fill=(255, 255, 255), width=lw)
                d.line([(cx - u * 0.62, cy + u * 0.62), (cx + u * 0.62, cy - u * 0.62)], fill=(255, 255, 255), width=lw)
            elif icon == "check":
                d.line([(cx - u * 0.7, cy + u * 0.02), (cx - u * 0.2, cy + u * 0.55), (cx + u * 0.75, cy - u * 0.55)],
                       fill=(255, 255, 255), width=lw, joint="curve")
            # icon: an arrow (the last one points up-right, like a result)
            elif i == len(items) - 1 and len(items) > 1:
                d.line([(cx - u * 0.7, cy + u * 0.7), (cx + u * 0.7, cy - u * 0.7)], fill=(255, 255, 255), width=max(2, int(r * 0.16)))
                d.line([(cx - u * 0.1, cy - u * 0.7), (cx + u * 0.7, cy - u * 0.7), (cx + u * 0.7, cy + u * 0.1)],
                       fill=(255, 255, 255), width=max(2, int(r * 0.16)))
            else:
                d.line([(cx - u, cy), (cx + u, cy)], fill=(255, 255, 255), width=max(2, int(r * 0.16)))
                d.line([(cx + u * 0.3, cy - u * 0.65), (cx + u, cy), (cx + u * 0.3, cy + u * 0.65)],
                       fill=(255, 255, 255), width=max(2, int(r * 0.16)))
            col = acc if (accent_last and icon == "arrow" and i == len(items) - 1 and len(items) > 1) else \
                ((246, 246, 248) if dark else (18, 18, 20))
            asc, desc = f.getmetrics()
            d.text((15 + int(ph * 1.0) + int(12 * s), 15 + (ph - asc - desc) // 2 + 1), it, font=f, fill=col + (255,))
            self.pills.append(np.asarray(img))
        gap = int(18 * s)
        hk = (self.kick.shape[0] + int(10 * s)) if self.kick is not None else 0
        total = hk + sum(p.shape[0] for p in self.pills) + gap * (len(self.pills) - 1)
        maxw = max(p.shape[1] for p in self.pills)
        self.x = int(0.055 * w) if side == "left" else int(w - 0.055 * w - maxw)
        self.y = int(h * y_mid - total / 2)
        self.ys = []
        y = self.y + hk
        for p in self.pills:
            self.ys.append(y)
            y += p.shape[0] - 30 + gap      # the image has a 30 px shadow margin
        self.sounds = [(max(0.0, t - 0.04), "pop", -10) for t in times]

    def draw(self, rgb: np.ndarray, tt: float, ov=None) -> np.ndarray:
        out = rgb.copy()
        k_out = _clamp((self.dur - tt) / 0.35)
        fade = k_out ** 1.5
        if self.kick is not None:
            kk = _clamp((tt - 0.0) / 0.3)
            _blit(out, self.kick, self.x + 8, self.y, fade * kk, clip=(0, 0, self.x + 8 + int(self.kick.shape[1] * kk), self.h))
        for p, y, t in zip(self.pills, self.ys, self.times):
            k = (tt - t) / 0.38
            if k <= 0:
                continue
            e = _back(k, 1.4)
            dx = int((1 - e) * (-0.06 if self.side == "left" else 0.06) * self.w)
            sc = 0.94 + 0.06 * _clamp(e)
            src = p
            if sc < 0.999:
                img = Image.fromarray(p)
                src = np.asarray(img.resize((max(2, int(p.shape[1] * sc)), max(2, int(p.shape[0] * sc))), Image.BILINEAR))
            _blit(out, src, self.x + dx, y + (p.shape[0] - src.shape[0]) // 2, fade * min(1.0, k * 2.5))
        return out


def overlay(obj, t0: float):
    from .render import Overlay
    return Overlay(t0, t0 + obj.dur, np.zeros((1, 1, 4), np.uint8), fade=0.0, box=(0, 0, obj.w, obj.h), anim="draw",
                   meta={"draw": obj.draw, "static": lambda: _static(obj)})


def _static(obj) -> np.ndarray:
    """Finished state as a full-frame RGBA (Premiere review project)."""
    base = np.zeros((obj.h, obj.w, 3), np.uint8)
    marker = np.full((obj.h, obj.w, 3), 255, np.uint8)
    tt = min(obj.dur - 0.6, max(getattr(obj, "times", [1.2]) + [1.2]) + 0.6)
    a = obj.draw(base, tt)
    b = obj.draw(marker, tt)
    alpha = 255 - np.clip(b.astype(np.int16) - a.astype(np.int16), 0, 255).max(axis=2)
    rgb = np.where(alpha[..., None] > 0, a.astype(np.float32) * 255 / np.maximum(alpha[..., None], 1), 0)
    return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha.astype(np.uint8)])



class Counter:
    """Counting sped up: a round badge with the number being said (pops on every new number), an accent ring that
    fills as the count goes on, and a small "HIZLI ×3" label that follows the speed ramp.
    marks: [(seconds since start, value)], speeds: [(seconds since start, speed)]."""

    def __init__(self, marks: list[tuple[float, int]], speeds: list[tuple[float, float]], w: int, h: int,
                 side: str = "right", dur: float = 6.0):
        s = h / 1080.0
        self.w, self.h, self.dur, self.side = w, h, dur, side
        self.marks, self.speeds = marks, speeds
        self.times = [t for t, _ in marks]
        self.top = max(v for _, v in marks)
        acc = _ac()
        self.acc = acc
        self.r = int(118 * s)
        f = font("display", int(150 * s))
        self.nums = {}
        for _, v in marks:
            fs = int(150 * s)
            fv = f
            while fv.getlength(str(v)) > 1.35 * self.r and fs > 40:
                fs = int(fs * 0.92)
                fv = font("display", fs)
            self.nums[v] = _text_img(str(v), fv, (255, 255, 255, 255))
        fm = font("mono", int(30 * s))
        self.labels = {sp: _text_img(f"HIZLI ×{sp:g}", fm, acc + (255,), spacing=0.25) for _, sp in speeds}
        self.cx = int(w * (0.84 if side == "right" else 0.16))
        self.cy = int(h * 0.27)
        self.sounds = [(max(0.0, t - 0.02), "click", -16) for t, _ in marks]
        self._rings: dict[int, np.ndarray] = {}

    def _ring(self, frac: float) -> np.ndarray:
        key = int(frac * 60)
        if key not in self._rings:
            r, pad = self.r, 24
            img = Image.new("RGBA", (2 * (r + pad), 2 * (r + pad)), (0, 0, 0, 0))
            sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse((pad, pad + 6, pad + 2 * r, pad + 2 * r + 6), fill=(0, 0, 0, 120))
            img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(10)))
            d = ImageDraw.Draw(img)
            d.ellipse((pad, pad, pad + 2 * r, pad + 2 * r), fill=(16, 16, 18, 225))
            wdt = max(4, r // 9)
            d.ellipse((pad, pad, pad + 2 * r, pad + 2 * r), outline=(255, 255, 255, 50), width=wdt)
            if key > 0:
                d.arc((pad, pad, pad + 2 * r, pad + 2 * r), -90, -90 + 360 * key / 60, fill=self.acc + (255,), width=wdt)
            self._rings[key] = np.asarray(img)
        return self._rings[key]

    def draw(self, rgb: np.ndarray, tt: float, ov=None) -> np.ndarray:
        out = rgb.copy()
        k_in = _back(tt / 0.35, 1.5)
        k_out = _clamp((self.dur - tt) / 0.3)
        fade = min(1.0, tt / 0.15) * k_out ** 1.5
        cur = None
        for t, v in self.marks:
            if t <= tt + 0.02:
                cur = (t, v)
        frac = (cur[1] / self.top) if cur else 0.0
        ring = self._ring(frac)
        sc = max(0.05, k_in)
        if sc < 0.999:
            ring = np.asarray(Image.fromarray(ring).resize((max(2, int(ring.shape[1] * sc)), max(2, int(ring.shape[0] * sc))), Image.BILINEAR))
        _blit(out, ring, self.cx - ring.shape[1] // 2, self.cy - ring.shape[0] // 2, fade)
        if cur is not None:
            img = self.nums[cur[1]]
            pop = 1.0 + 0.28 * (1 - _out((tt - cur[0]) / 0.18))
            pop *= sc
            if abs(pop - 1) > 1e-3:
                img = np.asarray(Image.fromarray(img).resize((max(2, int(img.shape[1] * pop)), max(2, int(img.shape[0] * pop))), Image.BILINEAR))
            _blit(out, img, self.cx - img.shape[1] // 2, self.cy - int(img.shape[0] * 0.56), fade)
        sp = None
        for t, v in self.speeds:
            if t <= tt:
                sp = v
        if sp is not None:
            lab = self.labels[sp]
            _blit(out, lab, self.cx - lab.shape[1] // 2, self.cy + self.r + int(0.02 * self.h), fade)
        return out


def _steps_img(k: int, n: int, s: float, acc) -> tuple[np.ndarray, np.ndarray]:
    """A row of n numbered circles on a line (0..n-1); returns (all dim, the first k+1 lit) as RGBA."""
    r = int(26 * s)
    gap = int(150 * s) if n <= 6 else int(110 * s)
    wd = 2 * r + gap * (n - 1) + 20
    ht = 2 * r + 30
    fm = font("mono", int(24 * s))

    def row(lit: bool) -> np.ndarray:
        img = Image.new("RGBA", (wd, ht), (0, 0, 0, 0))
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d, dg = ImageDraw.Draw(img), ImageDraw.Draw(glow)
        y = 15 + r
        last = k if lit else -1
        for i in range(n - 1):
            x0, x1 = 10 + r + i * gap, 10 + r + (i + 1) * gap
            on = lit and i < k
            d.line([(x0 + r, y), (x1 - r, y)], fill=(acc + (255,)) if on else (255, 255, 255, 60), width=max(2, int(3 * s)))
        for i in range(n):
            cx = 10 + r + i * gap
            on = lit and i <= last
            if on:
                dg.ellipse((cx - r - 6, y - r - 6, cx + r + 6, y + r + 6), fill=acc + (150,))
                d.ellipse((cx - r, y - r, cx + r, y + r), fill=acc + (255,))
            elif not lit:
                d.ellipse((cx - r, y - r, cx + r, y + r), outline=(255, 255, 255, 110), width=max(2, int(2 * s)))
            if on or not lit:
                t = str(i + 1)
                asc, desc = fm.getmetrics()
                d.text((cx - fm.getlength(t) / 2, y - (asc + desc) / 2), t, font=fm,
                       fill=(255, 255, 255, 255) if on else (210, 210, 214, 230))
        return np.asarray(Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(8)), img))

    return row(False), row(True)


class Chapter:
    """Small label that stays in the top-left corner during a section: "● BÖLÜM 2 · KONU ADI"."""

    def __init__(self, label: str, sub: str, w: int, h: int, dur: float):
        from .captions import tr_upper
        s = h / 1080.0
        self.w, self.h, self.dur = w, h, dur
        acc = _ac()
        fm = font("mono", int(26 * s))
        a = _text_img(tr_upper(label), fm, (250, 250, 252, 250), spacing=0.3)
        b = _text_img("  ·  " + tr_upper(sub), fm, (170, 170, 176, 240), spacing=0.3) if sub else None
        r = int(6 * s)
        wd = 4 * r + 6 + a.shape[1] + (b.shape[1] if b is not None else 0) + 10
        ht = max(a.shape[0], 2 * r) + 4
        img = np.zeros((ht, wd, 4), np.uint8)
        dot = Image.new("RGBA", (2 * r + 2, 2 * r + 2), (0, 0, 0, 0))
        ImageDraw.Draw(dot).ellipse((1, 1, 2 * r, 2 * r), fill=acc + (255,))
        _blit_rgba(img, np.asarray(dot), 0, ht // 2 - r)
        _blit_rgba(img, a, 4 * r + 6, (ht - a.shape[0]) // 2)
        if b is not None:
            _blit_rgba(img, b, 4 * r + 6 + a.shape[1] - 8, (ht - b.shape[0]) // 2)
        self.img = img
        self.x, self.y = int(0.065 * w), int(0.062 * h)
        self.sounds: list = []

    def draw(self, rgb: np.ndarray, tt: float, ov=None) -> np.ndarray:
        out = rgb.copy()
        k = _out(tt / 0.6)
        fade = min(1.0, tt / 0.3) * _clamp((self.dur - tt) / 0.4)
        _blit(out, self.img, self.x, self.y, fade, clip=(self.x, 0, self.x + int(self.img.shape[1] * k), self.h))
        return out


def _blit_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """src over dst, both RGBA (for building a label from pieces)."""
    h, w = src.shape[:2]
    h, w = min(h, dst.shape[0] - y), min(w, dst.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    s, d = src[:h, :w].astype(np.float32) / 255.0, dst[y:y + h, x:x + w].astype(np.float32) / 255.0
    a = s[..., 3:4] + d[..., 3:4] * (1 - s[..., 3:4])
    rgb = (s[..., :3] * s[..., 3:4] + d[..., :3] * d[..., 3:4] * (1 - s[..., 3:4])) / np.maximum(a, 1e-6)
    dst[y:y + h, x:x + w] = (np.dstack([rgb, a]) * 255).astype(np.uint8)
