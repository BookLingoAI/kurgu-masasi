"""Topic icons for the talking-head titles (Lucide, ISC licence - assets/icons/LICENSE-lucide.txt).

The SVG sources sit in assets/icons/svg; `build_pngs()` rasterises them once into white-on-transparent PNGs
(assets/icons/<name>.png) with a tiny stroke renderer (Lucide icons are 24x24 outlines, stroke 2, round caps and
joins), so no SVG library is needed at run time. `pick(text)` chooses an icon only when a word of the title clearly
names its subject ("Yapay zekâ" -> bot, "Para" -> banknote); otherwise there is no icon.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..analysis.text import norm_word

ROOT = Path(__file__).resolve().parents[2] / "assets" / "icons"
SIZE = 192

# word stem -> icon; the longest matching stem wins ("düşün" before "düş")
STEMS = {
    "yapay": "bot", "robot": "bot", "beyin": "brain", "beyn": "brain", "düşün": "brain", "zihin": "brain",
    "işlemci": "cpu", "bilgisayar": "laptop", "laptop": "laptop", "telefon": "smartphone",
    "para": "banknote", "lira": "banknote", "dolar": "banknote", "euro": "banknote", "maaş": "wallet",
    "cüzdan": "wallet", "bütçe": "wallet", "yatırım": "coins", "borsa": "trending-up",
    "zaman": "clock", "saat": "clock", "dakika": "clock", "takvim": "calendar", "hafta": "calendar",
    "kalp": "heart", "sevgi": "heart", "aşk": "heart", "insan": "users", "arkadaş": "users", "aile": "users",
    "mesaj": "message-circle", "yorum": "message-circle", "sohbet": "message-circle", "fikir": "lightbulb",
    "öneri": "lightbulb", "hedef": "target", "amaç": "target", "büyüme": "trending-up", "artış": "trending-up",
    "kazan": "trending-up", "kayıp": "trending-down", "düşüş": "trending-down", "güven": "shield",
    "gizlilik": "lock", "şifre": "lock", "kod": "code", "yazılım": "code", "program": "code",
    "kamera": "camera", "fotoğraf": "camera", "video": "video", "youtube": "video", "mikrofon": "mic",
    "kitap": "book-open", "okul": "graduation-cap", "üniversite": "graduation-cap", "eğitim": "graduation-cap",
    "öğren": "graduation-cap", "kariyer": "briefcase", "şirket": "briefcase", "patron": "briefcase",
    "araba": "car", "uçak": "plane", "seyahat": "plane", "yemek": "utensils", "kahve": "coffee",
    "enerji": "zap", "girişim": "rocket", "dünya": "globe", "internet": "globe", "araştır": "search",
    "ayar": "settings", "tamir": "wrench", "istatistik": "chart-column", "grafik": "chart-column",
    "sihir": "sparkles", "dikkat": "triangle-alert", "tehlike": "triangle-alert", "uyarı": "triangle-alert",
    "soru": "circle-help", "oyun": "gamepad-2", "müzik": "music", "şarkı": "music", "spor": "dumbbell",
    "doğa": "leaf", "ateş": "flame", "hediye": "gift", "başarı": "trophy", "ödül": "trophy",
    "şampiyon": "trophy", "üzgün": "frown", "mutsuz": "frown", "mutlu": "smile",
}
_STEMS = sorted(((norm_word(k), v) for k, v in STEMS.items()), key=lambda kv: -len(kv[0]))


def pick(*texts: str) -> str | None:
    """Icon name for the first title word that names a subject, or None."""
    for text in texts:
        for w in (text or "").split():
            nw = norm_word(w)
            for stem, name in _STEMS:
                if nw.startswith(stem) and (len(nw) - len(stem) <= 6):
                    return name if (ROOT / f"{name}.png").exists() or (ROOT / "svg" / f"{name}.svg").exists() else None
    return None


@lru_cache(maxsize=32)
def _mask(name: str) -> np.ndarray | None:
    p = ROOT / f"{name}.png"
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            return img[..., 3]
    svg = ROOT / "svg" / f"{name}.svg"
    return rasterize(svg.read_text(encoding="utf-8"), SIZE) if svg.exists() else None


def icon_rgba(name: str, px: int, color: tuple[int, int, int]) -> np.ndarray | None:
    """The icon at px x px in this colour (RGBA, uint8)."""
    m = _mask(name)
    if m is None:
        return None
    a = cv2.resize(m, (px, px), interpolation=cv2.INTER_AREA)
    out = np.zeros((px, px, 4), np.uint8)
    out[..., :3] = np.array(color, np.uint8)
    out[..., 3] = a
    return out


def build_pngs(size: int = SIZE) -> list[str]:
    """Rasterise every SVG in assets/icons/svg into assets/icons/<name>.png (white, alpha = ink)."""
    done = []
    for svg in sorted((ROOT / "svg").glob("*.svg")):
        a = rasterize(svg.read_text(encoding="utf-8"), size)
        img = np.zeros((size, size, 4), np.uint8)
        img[..., :3] = 255
        img[..., 3] = a
        cv2.imwrite(str(ROOT / f"{svg.stem}.png"), img)
        done.append(svg.stem)
    _mask.cache_clear()
    return done


# --- a minimal stroke renderer for Lucide's subset of SVG -------------------------------------------------------

def rasterize(svg_text: str, size: int, ss: int = 4) -> np.ndarray:
    """Alpha mask (uint8, size x size) of a 24x24 outline icon."""
    root = ET.fromstring(re.sub(r"<!--.*?-->", "", svg_text, flags=re.S))
    vb = [float(v) for v in (root.get("viewBox") or "0 0 24 24").split()]
    k = size * ss / vb[2]
    sw = float(root.get("stroke-width") or 2)
    canvas = np.zeros((size * ss, size * ss), np.uint8)
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        polys = _shape(tag, el)
        if not polys:
            continue
        pts = [np.round((np.array(p) - [vb[0], vb[1]]) * k * 16).astype(np.int32) for p in polys if len(p) >= 1]
        if (el.get("fill") or "none") not in ("none", "transparent"):
            cv2.fillPoly(canvas, [p for p in pts if len(p) >= 3], 255, lineType=cv2.LINE_AA, shift=4)
        th = max(1, int(round(float(el.get("stroke-width") or sw) * k)))
        for p in pts:
            closed = len(p) > 2 and np.array_equal(p[0], p[-1])
            cv2.polylines(canvas, [p], closed, 255, th, lineType=cv2.LINE_AA, shift=4)
            for q in p:          # round caps and joins
                cv2.circle(canvas, (int(q[0]), int(q[1])), th * 8, 255, -1, lineType=cv2.LINE_AA, shift=4)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _f(el, key: str, default: float = 0.0) -> float:
    v = el.get(key)
    return float(v) if v not in (None, "", "auto") else default


def _ellipse(cx, cy, rx, ry, n: int = 48) -> list[tuple[float, float]]:
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]


def _shape(tag: str, el) -> list[list[tuple[float, float]]]:
    if tag == "path":
        return _path(el.get("d") or "")
    if tag == "line":
        return [[(_f(el, "x1"), _f(el, "y1")), (_f(el, "x2"), _f(el, "y2"))]]
    if tag in ("polyline", "polygon"):
        v = [float(x) for x in re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", el.get("points") or "")]
        p = list(zip(v[0::2], v[1::2]))
        return [p + p[:1]] if tag == "polygon" and p else [p]
    if tag == "circle":
        return [_ellipse(_f(el, "cx"), _f(el, "cy"), _f(el, "r"), _f(el, "r"))]
    if tag == "ellipse":
        return [_ellipse(_f(el, "cx"), _f(el, "cy"), _f(el, "rx"), _f(el, "ry"))]
    if tag == "rect":
        x, y, w, h = _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height")
        rx = _f(el, "rx", _f(el, "ry"))
        ry = _f(el, "ry", rx)
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        if rx <= 0:
            return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]
        p = []
        for cx, cy, a0 in ((x + w - rx, y + ry, -90), (x + w - rx, y + h - ry, 0), (x + rx, y + h - ry, 90), (x + rx, y + ry, 180)):
            p += [(cx + rx * math.cos(math.radians(a0 + 90 * i / 8)), cy + ry * math.sin(math.radians(a0 + 90 * i / 8))) for i in range(9)]
        return [p + p[:1]]
    return []


_TOK = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _path(d: str) -> list[list[tuple[float, float]]]:
    toks = _TOK.findall(d)
    polys: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    x = y = sx = sy = 0.0
    last_c = None          # last control point (for S / T reflection)
    cmd, i = "", 0

    def num() -> float:
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def flags() -> tuple[float, float]:
        # arc flags may be written together without separators ("0 01.5" / "011")
        nonlocal i
        out = []
        while len(out) < 2:
            t = toks[i]
            if t in ("0", "1"):
                out.append(float(t))
                i += 1
            elif re.fullmatch(r"[01]+(\.\d*)?", t) or (t[:1] in "01" and len(t) > 1):
                out.append(float(t[0]))
                toks[i] = t[1:]
            else:
                out.append(float(t))
                i += 1
        return out[0], out[1]

    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if cur:
                    cur.append((sx, sy))
                    polys.append(cur)
                cur, x, y, last_c = [], sx, sy, None
                continue
        rel = cmd.islower()
        c = cmd.upper()
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if c == "M":
            if cur:
                polys.append(cur)
            x, y = ox + num(), oy + num()
            sx, sy, cur, last_c = x, y, [(x, y)], None
            cmd = "l" if rel else "L"        # further pairs are line-tos
        elif c == "L":
            x, y = ox + num(), oy + num()
            cur.append((x, y))
            last_c = None
        elif c == "H":
            x = ox + num()
            cur.append((x, y))
            last_c = None
        elif c == "V":
            y = oy + num()
            cur.append((x, y))
            last_c = None
        elif c in "CS":
            if c == "C":
                x1, y1 = ox + num(), oy + num()
            else:
                x1, y1 = (2 * x - last_c[0], 2 * y - last_c[1]) if last_c and last_c[2] == "c" else (x, y)
            x2, y2 = ox + num(), oy + num()
            ex, ey = ox + num(), oy + num()
            cur += _bezier([(x, y), (x1, y1), (x2, y2), (ex, ey)])
            x, y, last_c = ex, ey, (x2, y2, "c")
        elif c in "QT":
            if c == "Q":
                x1, y1 = ox + num(), oy + num()
            else:
                x1, y1 = (2 * x - last_c[0], 2 * y - last_c[1]) if last_c and last_c[2] == "q" else (x, y)
            ex, ey = ox + num(), oy + num()
            cur += _bezier([(x, y), (x1, y1), (ex, ey)])
            x, y, last_c = ex, ey, (x1, y1, "q")
        elif c == "A":
            rx, ry, rot = num(), num(), num()
            fa, fs = flags()
            ex, ey = ox + num(), oy + num()
            cur += _arc(x, y, rx, ry, rot, fa, fs, ex, ey)
            x, y, last_c = ex, ey, None
        else:
            i += 1
        if not cur:
            cur = [(x, y)]
    if cur:
        polys.append(cur)
    return polys


def _bezier(p: list[tuple[float, float]], n: int = 16) -> list[tuple[float, float]]:
    out = []
    for j in range(1, n + 1):
        t = j / n
        if len(p) == 4:
            a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
            out.append((a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0], a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1]))
        else:
            a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
            out.append((a * p[0][0] + b * p[1][0] + c * p[2][0], a * p[0][1] + b * p[1][1] + c * p[2][1]))
    return out


def _arc(x1, y1, rx, ry, rot, fa, fs, x2, y2) -> list[tuple[float, float]]:
    """SVG endpoint arc -> points (SVG 1.1 implementation notes, F.6.5)."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [(x2, y2)]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rot)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    xp, yp = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (xp / rx) ** 2 + (yp / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num_ = rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp
    den = rx * rx * yp * yp + ry * ry * xp * xp
    co = math.sqrt(max(0.0, num_ / den)) if den else 0.0
    if fa == fs:
        co = -co
    cxp, cyp = co * rx * yp / ry, -co * ry * xp / rx
    cx, cy = cp * cxp - sp * cyp + (x1 + x2) / 2, sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (xp - cxp) / rx, (yp - cyp) / ry)
    dt = ang((xp - cxp) / rx, (yp - cyp) / ry, (-xp - cxp) / rx, (-yp - cyp) / ry)
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    n = max(4, int(abs(dt) / (math.pi / 24)))
    out = []
    for j in range(1, n + 1):
        t = t1 + dt * j / n
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        out.append((cp * ex - sp * ey + cx, sp * ex + cp * ey + cy))
    return out
