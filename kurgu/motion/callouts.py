"""Draw a callout text panel as a full-frame RGBA layer, on the side opposite the face (never over it)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .captions import tr_upper
from .thumbnails import _font

ACCENT = (255, 212, 0, 255)


def _box_overlap(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def draw_callout(text: str, w: int, h: int, face: tuple | None) -> tuple[np.ndarray | None, dict]:
    """face: (x0, y0, x1, y1) in pixels or None. Returns (rgba or None if it cannot avoid the face, info)."""
    s = w / 1920.0
    size = int(64 * s)
    f = _font(size)
    t = tr_upper(text)
    pad_x, pad_y = int(34 * s), int(22 * s)
    tw = int(f.getlength(t))
    max_w = int(w * 0.34)
    while tw + 2 * pad_x > max_w and size > 34 * s:
        size -= int(4 * s) or 1
        f = _font(size)
        tw = int(f.getlength(t))
    bw, bh = tw + 2 * pad_x + int(10 * s), int(size * 1.15) + 2 * pad_y
    margin = int(70 * s)
    y0 = int(h * 0.12)
    face_cx = (face[0] + face[2]) / 2 if face else w / 2
    sides = ["left", "right"] if face_cx >= w / 2 else ["right", "left"]
    for side in sides:
        x0 = margin if side == "left" else w - margin - bw
        box = (x0, y0, x0 + bw, y0 + bh)
        if face is None or _box_overlap(box, face) == 0:
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(box, radius=int(18 * s), fill=(12, 12, 14, 215))
            d.rectangle((box[0], box[1], box[0] + int(10 * s), box[3]), fill=ACCENT)
            d.text((x0 + int(10 * s) + pad_x, y0 + pad_y - int(size * 0.08)), t, font=f, fill=(255, 255, 255, 255))
            return np.asarray(img), {"side": side, "box": box}
    return None, {"skipped": "yüzden kaçınılamadı"}
