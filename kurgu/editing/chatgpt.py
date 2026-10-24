"""Thumbnails + titles + description through ChatGPT (the user's own free account, used by the user).

There is no API for the free ChatGPT, so the app prepares everything ChatGPT needs (3 frames of the finished video,
the time-stamped transcript and a ready prompt), the user pastes it into ChatGPT, and brings back the images and the
answer text. The answer is parsed into the A/B/C pairs, description, tags and Shorts text.
"""
from __future__ import annotations

import re
from pathlib import Path

from .shorts import sentences

PROMPT = """Ekte YouTube videomdan 3 kare var; videonun zaman damgalı transkripti de aşağıda. Videonun içeriğini oku ve şunları hazırla:

1) YouTube SEO'ya uygun, birbirinden farklı 3 YATAY kapak görseli (16:9, 1280x720). Görsellerdeki kişi benim: yüzümü ve görünüşümü değiştirme, ekteki karelerimi kullan. Kapak yazısı kısa (2-4 kelime), büyük ve okunaklı olsun, yüzümün üstüne gelmesin. Her kapak farklı bir merak açısı denesin. Kapakları tek tek A, B, C diye üret.
2) Her kapakla eşleşen birer başlık (YouTube A/B testi için): 45-70 karakter, ana anahtar kelime başa yakın, tıklama tuzağı değil.
3) 1 SEO uyumlu açıklama: ilk iki satır videonun özeti ve ana anahtar kelimeler; sonra kısa ayrıntı; sonra transkriptteki zamanlara göre bölümler (ilk bölüm 00:00); en sonda 3-5 hashtag.
4) 10-15 etiket.
5) Bu videodan çıkan Shorts için bir başlık ve kısa açıklama.

Videoda söylenmeyen bilgi ekleme. Metinleri TAM OLARAK şu biçimde yaz (uygulamam bu biçimi okuyor):

A BAŞLIK: ...
B BAŞLIK: ...
C BAŞLIK: ...
AÇIKLAMA:
...
ETİKETLER: etiket1, etiket2, ...
SHORTS BAŞLIK: ...
SHORTS AÇIKLAMA: ...

TRANSKRİPT:
"""


def _fmt(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def transcript_text(words_cut: list[dict]) -> str:
    return "\n".join(f"[{_fmt(s['start'])}] {s['text']}" for s in sentences(words_cut))


def build_prompt(words_cut: list[dict]) -> str:
    return PROMPT + transcript_text(words_cut) + "\n"


_LABEL = r"[\*\#_ ]*"


def _field(text: str, name: str) -> str | None:
    m = re.search(rf"^{_LABEL}{name}{_LABEL}\s*[:：]{_LABEL}\s*(.+?)\s*$", text, re.M | re.I)
    return m.group(1).strip().strip("*").strip() if m else None


def parse_answer(text: str) -> dict:
    """ChatGPT's answer (copied from the chat) -> meta dict. Raises ValueError with a readable reason."""
    t = text.replace("\r\n", "\n")
    titles = [_field(t, f"{k} BAŞLIK") or _field(t, f"{k} BASLIK") for k in "ABC"]
    if not all(titles):
        missing = ", ".join(k for k, v in zip("ABC", titles) if not v)
        raise ValueError(f"Cevapta şu başlıklar bulunamadı: {missing}. ChatGPT'den metni istenen biçimde yazmasını isteyin.")
    m = re.search(rf"^{_LABEL}A[ÇC]IKLAMA{_LABEL}\s*[:：]{_LABEL}\s*\n?(.*?)(?=^{_LABEL}(ET[İI]KETLER|SHORTS BA[ŞS]LIK)\b)",
                  t + "\nETİKETLER:", re.M | re.S | re.I)
    desc = (m.group(1) if m else "").strip()
    if len(desc) < 40:
        raise ValueError("Cevapta açıklama bulunamadı (\"AÇIKLAMA:\" satırının altında olmalı).")
    tags_raw = _field(t, "ET[İI]KETLER") or ""
    tags = [x.strip().lstrip("#") for x in re.split(r"[,;\n]", tags_raw) if x.strip()]
    titles = [re.sub(r"^[\"“”']+|[\"“”']+$", "", x) for x in titles]
    return {"method": "ChatGPT", "titles": titles, "description": desc + "\n", "tags": tags[:30],
            "shorts_title": _field(t, "SHORTS BA[ŞS]LIK"), "shorts_description": _field(t, "SHORTS A[ÇC]IKLAMA") or "",
            "variants": [{"title": x} for x in titles]}


def fit_thumbnail(src: Path, dst: Path) -> dict:
    """Any image ChatGPT made -> 1280x720 JPEG under YouTube's 2 MB limit (cover crop, no stretching)."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    W, H = 1280, 720
    s = max(W / im.width, H / im.height)
    im = im.resize((max(W, round(im.width * s)), max(H, round(im.height * s))), Image.LANCZOS)
    x0, y0 = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((x0, y0, x0 + W, y0 + H))
    q = 92
    while True:
        im.save(dst, "JPEG", quality=q, optimize=True)
        if dst.stat().st_size < 1_900_000 or q <= 60:
            break
        q -= 8
    return {"file": str(dst), "width": W, "height": H, "kb": dst.stat().st_size // 1024}
