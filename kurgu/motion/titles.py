"""On-screen titles in the style of professional title packs, chosen by what the sentence says.

  bant     lower third: small coloured label above a bold headline on a dark bar; an accent light bar sweeps in
           from the left and uncovers it ("wipe"). Facts, places, dishes.
  fiyat    price: the amount huge in the accent colour (condensed display font, dark outline) next to the rest of
           the phrase on a dark bar; pops in with a cash-register "ding". Prices.
  kalem    highlighter: black bold text on a yellow marker stroke that is drawn from left to right ("sweep").
           Verdicts / opinions.
  daktilo  typewriter: dark panel, the text is typed letter by letter with soft key clicks. Durations, amounts.
  neon     glowing sign: white letters with an accent glow, flickers on.
  etiket   sticker: tilted white card, dark text, hard accent shadow, pops in.
  kutu     outlined frame on dark glass with a label tab, drawn in from the left.
Colours come from the theme (set_theme: a palette or one picked per video).

Placement: bottom-left by default, on the side away from the face; top when burned-in captions use the bottom.
build_title() returns the layers (full-frame RGBA + animation) and the sound effects to place.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .captions import tr_upper
from .fonts import font

ACCENT = (255, 212, 0)
PANEL = (14, 14, 18, 228)
# colour themes (accent used for bars, labels, highlighted words, light bars); "otomatik" picks one per video
PALETTES = {"sari": (255, 212, 0), "turkuaz": (0, 214, 201), "mercan": (255, 94, 87), "limon": (190, 242, 40),
            "mor": (168, 112, 255), "turuncu": (255, 145, 20), "pembe": (255, 78, 168)}
_THEME = {"accent": ACCENT}


def set_theme(name: str, seed: str = "") -> tuple[int, int, int]:
    """Choose the accent colour for this video: a palette name, or "otomatik" = a palette picked from `seed`
    (the project name), so each video gets its own look but stays consistent inside itself."""
    if name in PALETTES:
        _THEME["accent"] = PALETTES[name]
    else:
        import hashlib
        keys = list(PALETTES)
        _THEME["accent"] = PALETTES[keys[int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(keys)]]
    return _THEME["accent"]


def _ac() -> tuple[int, int, int]:
    return _THEME["accent"]


def _dark_text(c) -> bool:
    """Black text on this accent (light colours) or white (dark colours)?"""
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) > 147


def _on_ac() -> tuple[int, int, int, int]:
    return (12, 12, 14, 255) if _dark_text(_ac()) else (255, 255, 255, 255)
STYLE_FOR = {"FİYAT": "fiyat", "YORUM": "kalem", "SÜRE": "daktilo", "MİKTAR": "daktilo", "KİŞİ": "daktilo",
             "MESAFE": "daktilo", "ORAN": "daktilo"}
STYLES = ("bant", "fiyat", "kalem", "daktilo", "neon", "etiket", "kutu")
GENERAL_ROTATION = ("bant", "etiket", "neon", "kutu")   # facts/places/dishes: a different look each time


def _overlap(a, b) -> float:
    """Share of box a covered by box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def _place(w: int, h: int, bw: int, bh: int, face, captions: bool):
    """Top-left corner for a bw x bh block: first spot clear of the (padded) face; in a close-up the one covering
    least of it (chin/shoulder area); None when every spot would sit on the face."""
    mx = int(w * 0.055)
    spots = [("left", int(h * 0.86) - bh), ("right", int(h * 0.86) - bh), ("left", int(h * 0.10)), ("right", int(h * 0.10))]
    if captions:   # captions live at the bottom: titles go up
        spots = spots[2:] + spots[:2]
    cands = []
    for side, y0 in spots:
        x0 = mx if side == "left" else w - mx - bw
        box = (x0, y0, x0 + bw, y0 + bh)
        cands.append((_overlap(box, face) if face is not None else 0.0, len(cands), side, x0, y0, box))
    clear = [c for c in cands if c[0] == 0]
    best = clear[0] if clear else min(cands)
    return None if best[0] > 0.4 else best[2:]


def _fit(role: str, text: str, size: int, max_w: int, min_size: int):
    f = font(role, size)
    while f.getlength(text) > max_w and size > min_size:
        size -= 2
        f = font(role, size)
    return f, size


def _shadow(img: Image.Image, box, s: float, alpha: int = 120) -> Image.Image:
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    off = int(6 * s)
    ImageDraw.Draw(sh).rectangle((box[0] + off, box[1] + off, box[2] + off, box[3] + off), fill=(0, 0, 0, alpha))
    return Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(int(10 * s) or 1)))


def _label(k: str, s: float):
    kf = font("label", int(34 * s))
    return kf, (int(kf.getlength(k)) + int(30 * s), int(34 * s * 1.2) + int(10 * s)) if k else (0, 0)


def _bant(t, k, w, h, face, captions):
    s = h / 1080.0
    f, size = _fit("bold", t, int(60 * s), int(w * 0.44), int(34 * s))
    pad_x, pad_y, edge = int(30 * s), int(18 * s), int(9 * s)
    tw = int(f.getlength(t))
    bw, bh = tw + 2 * pad_x + edge, int(size * 1.15) + 2 * pad_y
    kf, (kw, kh) = _label(k, s)
    gap = int(8 * s) if k else 0
    spot = _place(w, h, max(bw, kw), bh + kh + gap, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    by0 = y0 + kh + gap
    img = _shadow(Image.new("RGBA", (w, h), (0, 0, 0, 0)), (x0, by0, x0 + bw, by0 + bh), s)
    d = ImageDraw.Draw(img)
    if k:
        d.rectangle((x0, y0, x0 + kw, y0 + kh), fill=_ac() + (255,))
        d.text((x0 + int(14 * s), y0 + kh / 2), k, font=kf, fill=_on_ac(), anchor="lm")
    d.rectangle((x0, by0, x0 + bw, by0 + bh), fill=PANEL)
    d.rectangle((x0, by0, x0 + edge, by0 + bh), fill=_ac() + (255,))
    d.text((x0 + edge + pad_x, by0 + bh / 2), t, font=f, fill=(255, 255, 255, 255), anchor="lm")
    return [{"rgba": np.asarray(img), "anim": "wipe", "fade": 0.0, "meta": {"accent": _ac()}}], side, box, \
        [(-0.05, "swish", -5)]


def _fiyat(t, k, w, h, face, captions):
    s = h / 1080.0
    m = re.match(r"^\s*([\d.,]+)\s*(.*)$", t)
    if not m:
        return _bant(t, k, w, h, face, captions)
    num, rest = m.group(1), m.group(2).strip()
    nf = font("heavy", int(150 * s))
    stroke = max(2, int(6 * s))
    nb = nf.getbbox(num, stroke_width=stroke)
    nw, nh = nb[2] - nb[0], nb[3] - nb[1]
    rf, rsize = _fit("bold", rest, int(50 * s), int(w * 0.30), int(30 * s)) if rest else (None, 0)
    pad_x, pad_y = int(24 * s), int(14 * s)
    rw = int(rf.getlength(rest)) + 2 * pad_x if rest else 0
    rh = int(rsize * 1.15) + 2 * pad_y if rest else 0
    gap = int(16 * s)
    bw, bh = nw + (gap + rw if rest else 0), max(nh, rh)
    spot = _place(w, h, bw, bh, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # soft dark glow behind the number so it reads on any background
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((x0 - nb[0], y0 - nb[1]), num, font=nf, fill=(0, 0, 0, 160), stroke_width=stroke * 3,
                              stroke_fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(int(12 * s) or 1)))
    d = ImageDraw.Draw(img)
    d.text((x0 - nb[0], y0 - nb[1]), num, font=nf, fill=_ac() + (255,), stroke_width=stroke, stroke_fill=(10, 10, 12, 255))
    if rest:
        rx0, ry0 = x0 + nw + gap, y0 + (bh - rh) // 2
        d.rectangle((rx0, ry0, rx0 + rw, ry0 + rh), fill=PANEL)
        d.rectangle((rx0, ry0 + rh - int(7 * s), rx0 + rw, ry0 + rh), fill=_ac() + (255,))
        d.text((rx0 + pad_x, ry0 + (rh - int(7 * s)) / 2), rest, font=rf, fill=(255, 255, 255, 255), anchor="lm")
    return [{"rgba": np.asarray(img), "anim": "pop", "fade": 0.25}], side, box, [(0.05, "ding", -7), (-0.04, "pop", -8)]


def _kalem(t, k, w, h, face, captions):
    s = h / 1080.0
    f, size = _fit("bold", t, int(62 * s), int(w * 0.46), int(34 * s))
    pad_x, pad_y = int(26 * s), int(14 * s)
    tw = int(f.getlength(t))
    bw, bh = tw + 2 * pad_x, int(size * 1.15) + 2 * pad_y
    spot = _place(w, h, bw, bh, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    img = _shadow(Image.new("RGBA", (w, h), (0, 0, 0, 0)), (x0, y0, x0 + bw, y0 + bh), s, alpha=90)
    d = ImageDraw.Draw(img)
    # a marker stroke: slightly slanted ends and a darker lower edge, not a perfect rectangle
    sk = int(10 * s)
    d.polygon([(x0 + sk, y0), (x0 + bw, y0 + int(3 * s)), (x0 + bw - sk, y0 + bh), (x0, y0 + bh - int(3 * s))],
              fill=_ac() + (245,))
    d.polygon([(x0, y0 + bh - int(10 * s)), (x0 + bw - sk // 2, y0 + bh - int(7 * s)), (x0 + bw - sk, y0 + bh),
               (x0, y0 + bh - int(3 * s))], fill=tuple(int(v * 0.85) for v in _ac()) + (245,))
    d.text((x0 + pad_x, y0 + bh / 2), t, font=f, fill=_on_ac(), anchor="lm")
    return [{"rgba": np.asarray(img), "anim": "sweep", "fade": 0.0}], side, box, [(-0.03, "swish", -6)]


def _daktilo(t, k, w, h, face, captions):
    s = h / 1080.0
    f, size = _fit("bold", t, int(58 * s), int(w * 0.44), int(34 * s))
    pad_x, pad_y, edge = int(28 * s), int(18 * s), int(9 * s)
    tw = int(f.getlength(t))
    bw, bh = tw + 2 * pad_x + edge + int(20 * s), int(size * 1.15) + 2 * pad_y
    kf, (kw, kh) = _label(k, s)
    gap = int(8 * s) if k else 0
    spot = _place(w, h, max(bw, kw), bh + kh + gap, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    by0 = y0 + kh + gap
    panel = _shadow(Image.new("RGBA", (w, h), (0, 0, 0, 0)), (x0, by0, x0 + bw, by0 + bh), s)
    d = ImageDraw.Draw(panel)
    if k:
        d.rectangle((x0, y0, x0 + kw, y0 + kh), fill=_ac() + (255,))
        d.text((x0 + int(14 * s), y0 + kh / 2), k, font=kf, fill=_on_ac(), anchor="lm")
    d.rectangle((x0, by0, x0 + bw, by0 + bh), fill=PANEL)
    d.rectangle((x0, by0, x0 + edge, by0 + bh), fill=_ac() + (255,))
    text = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tx = x0 + edge + pad_x
    ImageDraw.Draw(text).text((tx, by0 + bh / 2), t, font=f, fill=(255, 255, 255, 255), anchor="lm")
    rgba = np.asarray(text)
    ys, xs = np.nonzero(rgba[..., 3])
    tb0 = int(xs.min()) if len(xs) else tx
    # x offset (from the text layer's left edge) after each character: what is visible as it is typed
    cuts = [max(1, int(round(tx + f.getlength(t[:i + 1]) - tb0)) + 2) for i in range(len(t)) if not t[i].isspace()]
    keys = [(0.18 + i * 0.05, "key", -16) for i in range(0, len(cuts), 2)][:12]
    return [{"rgba": np.asarray(panel), "anim": "fade", "fade": 0.15},
            {"rgba": rgba, "anim": "type", "fade": 0.0, "delay": 0.15, "meta": {"cuts": cuts, "cps": 0.05}}], \
        side, box, keys


def _neon(t, k, w, h, face, captions):
    """Glowing neon sign: white core letters, accent-coloured glow, flickers on like a tube lighting up."""
    s = h / 1080.0
    f, size = _fit("heavy", t, int(84 * s), int(w * 0.46), int(44 * s))
    tb = f.getbbox(t)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = int(30 * s)
    spot = _place(w, h, tw + 2 * pad, th + 2 * pad, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    tx, ty = x0 + pad - tb[0], y0 + pad - tb[1]
    ac = _ac()
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 0))   # a dark haze so the glow reads on bright footage
    ImageDraw.Draw(dark).rounded_rectangle(box, radius=int(24 * s), fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, dark.filter(ImageFilter.GaussianBlur(int(18 * s) or 1)))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((tx, ty), t, font=f, fill=ac + (255,), stroke_width=int(6 * s), stroke_fill=ac + (255,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(int(16 * s) or 1)))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(int(5 * s) or 1)))
    ImageDraw.Draw(img).text((tx, ty), t, font=f, fill=(255, 255, 255, 255), stroke_width=max(1, int(2 * s)),
                             stroke_fill=ac + (255,))
    return [{"rgba": np.asarray(img), "anim": "flicker", "fade": 0.25}], side, box, [(0.0, "swish", -7), (0.08, "click", -10)]


def _etiket(t, k, w, h, face, captions):
    """Sticker: white rounded card, tilted a few degrees, dark text with a hard accent shadow; pops in."""
    s = h / 1080.0
    f, size = _fit("heavy", t, int(76 * s), int(w * 0.40), int(40 * s))
    tb = f.getbbox(t)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad_x, pad_y = int(34 * s), int(22 * s)
    cw, ch = tw + 2 * pad_x, th + 2 * pad_y
    card = Image.new("RGBA", (cw + int(40 * s), ch + int(40 * s)), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    o = int(10 * s)
    d.rounded_rectangle((o + int(9 * s), o + int(9 * s), o + cw + int(9 * s), o + ch + int(9 * s)), radius=int(18 * s),
                        fill=_ac() + (255,))   # hard offset shadow in the accent colour
    d.rounded_rectangle((o, o, o + cw, o + ch), radius=int(18 * s), fill=(250, 250, 248, 255))
    d.text((o + pad_x - tb[0], o + pad_y - tb[1]), t, font=f, fill=(16, 16, 18, 255))
    card = card.rotate(3.5, resample=Image.BICUBIC, expand=True)
    spot = _place(w, h, card.width, card.height, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.alpha_composite(card, (x0, y0))
    return [{"rgba": np.asarray(img), "anim": "pop", "fade": 0.2}], side, box, [(-0.02, "pop", -6)]


def _kutu(t, k, w, h, face, captions):
    """Outlined frame: white headline in a thin accent frame on dark glass, label tab on the frame; drawn in from
    the left."""
    s = h / 1080.0
    f, size = _fit("bold", t, int(58 * s), int(w * 0.44), int(34 * s))
    tw = int(f.getlength(t))
    pad_x, pad_y, lw = int(34 * s), int(22 * s), max(2, int(5 * s))
    bw, bh = tw + 2 * pad_x, int(size * 1.15) + 2 * pad_y
    kf, (kw, kh) = _label(k, s)
    top = kh // 2 if k else 0
    spot = _place(w, h, max(bw, kw + int(30 * s)), bh + top, face, captions)
    if spot is None:
        return None
    side, x0, y0, box = spot
    by0 = y0 + top
    img = _shadow(Image.new("RGBA", (w, h), (0, 0, 0, 0)), (x0, by0, x0 + bw, by0 + bh), s, alpha=90)
    d = ImageDraw.Draw(img)
    d.rectangle((x0, by0, x0 + bw, by0 + bh), fill=(10, 10, 12, 165), outline=_ac() + (255,), width=lw)
    d.text((x0 + pad_x, by0 + bh / 2 + top * 0.15), t, font=f, fill=(255, 255, 255, 255), anchor="lm")
    if k:
        kx = x0 + int(26 * s)
        d.rectangle((kx, y0, kx + kw, y0 + kh), fill=_ac() + (255,))
        d.text((kx + int(14 * s), y0 + kh / 2), k, font=kf, fill=_on_ac(), anchor="lm")
    return [{"rgba": np.asarray(img), "anim": "sweep", "fade": 0.0, "meta": {"accent": _ac()}}], side, box, \
        [(-0.04, "swish", -6)]


def build_title(text: str, kicker: str, w: int, h: int, face: tuple | None, captions: bool = False,
                style: str = "karisik", idx: int = 0) -> tuple[list[dict] | None, dict]:
    """style: "karisik" (chosen by the label: FİYAT -> fiyat, YORUM -> kalem, SÜRE/MİKTAR... -> daktilo, anything
    else rotates bant / etiket / neon / kutu by idx) or one of STYLES for all titles. Returns (layers or None, info with "style", "side", "box", "sfx")."""
    t, k = tr_upper(text.strip()), tr_upper(kicker.strip())
    st = style if style in STYLES else STYLE_FOR.get(k) or GENERAL_ROTATION[idx % len(GENERAL_ROTATION)]
    fn = {"bant": _bant, "fiyat": _fiyat, "kalem": _kalem, "daktilo": _daktilo, "neon": _neon, "etiket": _etiket,
          "kutu": _kutu}[st]
    if st == "fiyat" and not re.match(r"^\s*[\d.,]+", t):
        st, fn = "bant", _bant
    r = fn(t, k, w, h, face, captions)
    if r is None:
        return None, {"skipped": "yüzden kaçınılamadı"}
    layers, side, box, sfx = r
    for L in layers:      # a glint passes over the finished title (not over the typewriter / its panel)
        if L.get("anim") in ("wipe", "pop", "sweep", "flicker"):
            L["meta"] = dict(L.get("meta") or {}, shine=True, shine_at=0.62 if L["anim"] in ("wipe", "sweep") else 0.5)
    sfx = list(sfx) + [(0.55, "sparkle", -14)]
    return layers, {"style": st, "side": side, "box": box, "sfx": sfx}


def draw_title(text: str, w: int, h: int, face: tuple | None, kicker: str = "",
               captions: bool = False) -> tuple[np.ndarray | None, dict]:
    """The plain lower third as one RGBA (Premiere, tests)."""
    layers, info = build_title(text, kicker, w, h, face, captions, style="bant")
    return (layers[0]["rgba"] if layers else None), info


# ------------------------------------------------------------------ sentence moments (camera + kinetic text) ------
def _layout(words: list[str], f, max_w: int, gap: int) -> list[list[tuple[str, int]]]:
    """Greedy line wrap: [[(word, x offset)], ...]."""
    lines, cur, x = [], [], 0
    for wd in words:
        wl = int(f.getlength(wd))
        if cur and x + wl > max_w:
            lines.append(cur)
            cur, x = [], 0
        cur.append((wd, x))
        x += wl + gap
    if cur:
        lines.append(cur)
    return lines


def build_sentence(fx: dict, w: int, h: int) -> tuple[list[dict] | None, dict]:
    """Text for a camera moment (see editing.graphics.plan_camfx). fx: t0, t1, kind, side, amt, words [(w, t0, t1)].
    slide: the sentence word by word in the freed panel, vertically centred, accent line above it.
    punch: big words at the bottom on a dark translucent band.
    The current word lights up in the accent colour (karaoke style). Returns (layers, info with sfx)."""
    s = h / 1080.0
    items = [(wd, a, b) for wd, a, b in fx["words"] if wd]
    words = [wd for wd, _, _ in items]
    if not words:
        return None, {}
    slide = fx["kind"] == "slide"
    if slide:
        pw = int(fx.get("amt", 0.36) * w)
        margin = int(0.04 * w)
        max_w = pw - 2 * margin
        for size in (int(66 * s), int(58 * s), int(50 * s)):
            f = font("bold", size)
            lines = _layout(words, f, max_w, int(size * 0.28))
            if len(lines) <= 5:
                break
        else:
            return None, {}
        role_stroke = 0
    else:
        for size in (int(72 * s), int(62 * s), int(54 * s)):
            f = font("bold", size)
            lines = _layout(words, f, int(w * 0.78), int(size * 0.28))
            if len(lines) <= 2:
                break
        else:
            return None, {}
        role_stroke = max(2, int(3 * s))
    lh = int(size * 1.28)
    block_h = lh * len(lines)
    widths = [ln[-1][1] + int(f.getlength(ln[-1][0])) for ln in lines]
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    boxes, times, ends = [], [], []
    delay = 0.35 if slide else 0.12
    ov_t0 = fx["t0"] + delay
    if slide:
        x_left = margin if fx.get("side", "left") == "left" else w - pw + margin
        y_top = (h - block_h) // 2
        # accent line above the text block
        ax0, ay0 = x_left, y_top - int(26 * s)
        d.rectangle((ax0, ay0, ax0 + int(70 * s), ay0 + int(7 * s)), fill=_ac() + (255,))
        boxes.append((ax0, ay0, ax0 + int(70 * s) + 1, ay0 + int(7 * s) + 1))
        times.append(0.0)
        ends.append(None)
    else:
        y_top = int(h * 0.90) - block_h
        bw = max(widths) + int(60 * s)
        bx0 = (w - bw) // 2
        band = (bx0, y_top - int(18 * s), bx0 + bw, y_top + block_h + int(10 * s))
        band_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(band_img).rounded_rectangle(band, radius=int(16 * s), fill=(8, 8, 10, 150))
    k = 0
    for li, ln in enumerate(lines):
        y = y_top + li * lh
        x_line = x_left if slide else (w - widths[li]) // 2
        for wd, xo in ln:
            x = x_line + xo
            d.text((x, y), wd, font=f, fill=(255, 255, 255, 255), stroke_width=role_stroke,
                   stroke_fill=(10, 10, 12, 255))
            bb = d.textbbox((x, y), wd, font=f, stroke_width=role_stroke)
            pad = 3
            boxes.append((max(0, bb[0] - pad), max(0, bb[1] - pad), min(w, bb[2] + pad), min(h, bb[3] + pad)))
            _, wt0, wt1 = items[k]
            times.append(max(0.0, wt0 - ov_t0))
            ends.append(max(0.0, wt1 - ov_t0) + 0.05)
            k += 1
    layer = {"rgba": np.asarray(img), "anim": "words", "fade": 0.0, "delay": delay,
             "meta": {"boxes": boxes, "times": times, "ends": ends, "accent": _ac(), "fx": fx.get("fx", "karaoke")}}
    if not slide:   # the dark band is its own layer (its box would otherwise uncover every word at once)
        return [{"rgba": np.asarray(band_img), "anim": "fade", "fade": 0.2, "delay": delay}, layer], \
            {"style": fx["kind"], "sfx": [(-0.02, "impact", -6), (0.0, "swish", -8)]}
    sfx = [(-0.1, "whoosh", -4)] if slide else [(-0.02, "impact", -6), (0.0, "swish", -8)]
    return [layer], {"style": fx["kind"], "sfx": sfx}
