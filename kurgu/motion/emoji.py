"""Animated 3D emoji (Noto Emoji 3D, OFL) placed next to callouts or as reactions to what is said."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

DIR = Path(__file__).resolve().parents[2] / "assets" / "emoji"

# name the language model may use -> Noto file code
EMOJI = {
    "ateş": "1f525", "fikir": "1f4a1", "kahkaha": "1f602", "şaşkın": "1f62e", "düşünceli": "1f914", "kalp": "2764",
    "beğeni": "1f44d", "beğenmeme": "1f44e", "doğru": "2705", "yanlış": "274c", "uyarı": "26a0", "para": "1f4b0",
    "dolar": "1f4b5", "yükseliş": "1f4c8", "düşüş": "1f4c9", "saat": "23f0", "roket": "1f680", "hedef": "1f3af",
    "ev": "1f3e0", "uçak": "2708", "araba": "1f697", "abd": "1f1fa_1f1f8", "türkiye": "1f1f9_1f1f7", "güç": "1f4aa",
    "korku": "1f631", "havalı": "1f60e", "rica": "1f64f", "gözler": "1f440", "kutlama": "1f389", "not": "1f4cc",
    "soru": "2753", "ünlem": "2757", "iş": "1f4bc", "yemek": "1f354", "kahve": "2615", "telefon": "1f4f1",
    "bilgisayar": "1f4bb", "beyin": "1f9e0", "yıldız": "2b50", "kupa": "1f3c6", "üzgün": "1f622", "kızgın": "1f621",
    "şok": "1f92f", "para_uçuyor": "1f4b8", "takvim": "1f4c5", "alışveriş": "1f6d2", "hastane": "1f3e5",
    "okul": "1f393", "makas": "2702", "berber": "1f488", "dünya": "1f30e", "anahtar": "1f511", "duyuru": "1f4e2",
    "anlaşma": "1f91d", "uyku": "1f634", "yüz_puan": "1f4af", "işaret": "1f449", "zil": "1f514", "kamera": "1f4f8",
    "müzik": "1f3b5", "film": "1f3ac", "konuşma": "1f4ac", "kart": "1f4b3", "banka": "1f3e6", "aile": "1f46a",
    "restoran": "1f37d", "seyahat": "1f6eb", "amerika": "1f5fd", "şehir": "1f3d9",
}


def names() -> list[str]:
    return [n for n, c in EMOJI.items() if (DIR / f"emoji_u{c}.png").exists()]


@lru_cache(maxsize=64)
def _img(name: str) -> Image.Image | None:
    code = EMOJI.get((name or "").strip().lower())
    p = DIR / f"emoji_u{code}.png" if code else None
    return Image.open(p).convert("RGBA") if p and p.exists() else None


def emoji_layer(name: str, w: int, h: int, cx: float, cy: float, size: int) -> np.ndarray | None:
    """Full-frame RGBA with the emoji (soft shadow) centred at (cx, cy) pixels."""
    im = _img(name)
    if im is None:
        return None
    from PIL import ImageFilter
    e = im.resize((size, size), Image.LANCZOS)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    a = e.getchannel("A").point(lambda v: int(v * 0.45))
    sh.paste(Image.new("RGBA", e.size, (0, 0, 0, 255)), (x0 + size // 18, y0 + size // 12), a)
    layer = Image.alpha_composite(layer, sh.filter(ImageFilter.GaussianBlur(max(2, size // 20))))
    layer.alpha_composite(e, (max(0, x0), max(0, y0)))
    return np.asarray(layer)
