"""Moments where the speaker SHOWS something without talking: holding the meat up to the lens ("açın ağzınızı"),
turning the phone to the sushi plate, panning over the sauces. The silence rules cut them as pauses, but they carry
the scene (the viewer is being shared the food). They are found here, kept in the edit (protected spans) and offered
to the music intro.

A wordless gap qualifies when the picture during it is a clean shot of food/place (broll_mine.clean: no big face,
no legs/feet, sharp, colourful), or when the speaker just announced it ("bakın", "açın ağzınızı", "göstereyim") and
the picture is close, sharp and colourful even with part of him in it.
"""
from __future__ import annotations

from ..analysis.text import norm_word

CUES = {"bakın", "bakin", "açın", "acin", "ağzınızı", "göstereyim", "gösteriyorum", "göstereceğim", "buyrun",
        "buyurun", "şuna", "şunlara", "görüyor", "görüyorsunuz", "geldi", "geldiler", "yakından", "yaklaştırayım",
        "kameraya", "ısırık"}
# offering food to the viewer ("açın ağzınızı", "buyurun", "alın bir ısırık"): the hand comes up to the lens, so the
# picture is close, blurred and dark - the camera check failed and the moment was cut as a pause (23.10.2026,
# Japon hot pot, twice). After these words the pause is kept as it is, without looking at the picture.
OFFER = {"açın", "acin", "aç", "ağzınızı", "ağzını", "agzinizi", "ağzınız", "buyurun", "buyrun", "buyur", "alın",
         "ısırık", "tadın", "tadına", "yiyin", "deneyin", "uzatıyorum", "kameraya"}


def _offer_before(words: list[dict], i: int, t: float, reach: float = 3.0) -> bool:
    j = i
    while j >= 0 and t - words[j]["end"] <= reach:
        if norm_word(words[j]["word"]) in OFFER:
            return True
        j -= 1
    return False


def _cue_before(words: list[dict], i: int, t: float, reach: float = 4.0) -> bool:
    j = i
    while j >= 0 and t - words[j]["end"] <= reach:
        if norm_word(words[j]["word"]) in CUES:
            return True
        j -= 1
    return False


def find_showing(words: list[dict], cam, min_gap: float = 1.2, max_keep: float = 5.0, max_scan: int = 220,
                 progress=None, offer_keep: float = 6.0, offer_min: float = 0.5) -> list[tuple[float, float]]:
    """Protected (camera-time) spans. cam: CameraTimeline (locate(t) -> part, local time)."""
    from . import broll_mine as bm
    gaps, out = [], []
    for i in range(len(words) - 1):
        a, b = words[i]["end"], words[i + 1]["start"]
        if b - a >= offer_min and _offer_before(words, i, a):
            # offered to the viewer: the gesture stays whatever the picture looks like
            out.append((round(a, 3), round(min(b, a + offer_keep), 3)))
            continue
        if b - a < min_gap:
            continue
        cue = _cue_before(words, i, a)
        if not cue and b - a < 2.5:
            continue
        gaps.append((a, b, cue))
    gaps = sorted(gaps, key=lambda g: (not g[2], -(g[1] - g[0])))[:max_scan]
    gaps.sort()
    vis = bm._Vision()
    if not vis.faces.ok:
        return sorted(out)
    for k, (a, b, cue) in enumerate(gaps):
        if progress and k % 10 == 0:
            progress(k / max(1, len(gaps)), f"Gösterilen anlar aranıyor {k}/{len(gaps)}")
        try:
            part, la = cam.locate(a + 0.05)
        except Exception:  # noqa: BLE001
            continue
        span = min(b - a, max_keep + 1.0)
        if la + span > part.duration:
            span = part.duration - la
        if span < min_gap:
            continue
        frames = bm._rgb_frames(part.path, la, span, fps=3.0)
        if len(frames) < 3:
            continue
        ms = bm.measure(frames, vis)
        good = []
        for m in ms:
            ok = bm.clean(m)
            if not ok and cue:   # announced: he holds it up to the lens (the segmenter calls a close-up plate "person")
                ok = m["sharp"] >= 40 and m["colour"] >= 25 and 30 <= m["bright"] <= 230
            good.append(ok)
        if sum(good) >= 0.5 * len(good) and (cue or bm.motion_kind(ms) != "shaky"):
            # keep from the start of the gap up to the last good sample (never more than max_keep)
            last = max(i for i, g in enumerate(good) if g)
            out.append((round(a, 3), round(min(b, a + min(max_keep, (last + 1) / 3.0 + 0.3)), 3)))
    return sorted(out)
