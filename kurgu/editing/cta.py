"""Subscribe / like / comment reminders exactly where the speaker asks for them.

The old reminder sat at 25 % and 65 % of the video, whatever was being said (it came up while the sauces were shown).
Now every request in the talk is found ("abone olun", "kanalıma abone olduysanız", "like atmayı unutmayın",
"videoyu beğenin", "yorumlara yazın", "bildirimleri açın") and the animated reminder clicks each button at the very
word that asks for it. Whisper invents "abone ol" over music (a classic hallucination from YouTube subtitles):
such lone, low-confidence words are ignored.
"""
from __future__ import annotations

import re

from ..analysis.text import norm_word

SUB_OL = re.compile(r"^ol(un|mayı|mak|du|dun|dunuz|duysanız|madıysanız|mayan|mayanlar|muyorsunuz|mazsanız|anlar|ursanız)?$")
LIKE_EN = re.compile(r"^lay?k|^like")
LIKE_TR = re.compile(r"^beğen(in|meyi|mek|di(y|ğ)sen|diyseniz|diysen|meyi|ip|irseniz|enler)$")
LIKE_VERB = re.compile(r"^(at|atın|atmayı|atmak|attıysanız|atarsanız|atabilirsiniz|atar|bırakın|basın)$")
COMMENT = re.compile(r"^yorum(lara|a|u|unuzu|larınızı|ları|da|larda)?$|^yorum$")
COMMENT_VERB = re.compile(r"^(yazın|yazmayı|yazabilirsiniz|bırakın|bıraktıysanız|bırakmayı|yaz|yazarsanız|yazıp|belirtin)$")
BELL = re.compile(r"^(zil|zili|zile|bildirim|bildirimleri|bildirimi)$")
WRITE_DOWN = re.compile(r"^(aşağıya|aşağı|fikrinizi|fikirlerinizi|tahmininizi|tahminlerinizi|düşüncelerinizi|cevabınızı)$")


def _ok(w: dict, floor: float = 0.6) -> bool:
    p = w.get("prob")
    return p is None or p >= floor


def find_requests(sents: list[dict]) -> list[tuple[str, float]]:
    """[(kind, time of the asking word on the timeline)]; kind: subscribe / like / comment / bell."""
    hits: list[tuple[str, float]] = []
    for s in sents:
        ws = s["words"]
        toks = [norm_word(w["word"]) for w in ws]
        probs = [w.get("prob") for w in ws if w.get("prob") is not None]
        conf = sum(probs) / len(probs) if probs else 1.0
        if len(ws) < 2 or conf < 0.55:
            continue
        for k, t in enumerate(toks):
            nxt = toks[k + 1:k + 4]
            prev = toks[max(0, k - 3):k]
            w = ws[k]
            if t.startswith("abone") and _ok(w) and (any(SUB_OL.match(x) for x in nxt) or t in ("aboneyiz", "abonelerim")
                                                      or "kanal" in " ".join(prev + nxt)):
                if len(ws) <= 3 and conf < 0.75:
                    continue              # "abone ol" alone over music: Whisper hallucination
                hits.append(("subscribe", w["t_start"]))
            elif (LIKE_EN.match(t) and _ok(w, 0.55) and any(LIKE_VERB.match(x) for x in nxt + prev)) or \
                    (LIKE_TR.match(t) and _ok(w)):
                hits.append(("like", w["t_start"]))
            elif COMMENT.match(t) and _ok(w) and any(COMMENT_VERB.match(x) for x in nxt + prev):
                hits.append(("comment", w["t_start"]))
            elif WRITE_DOWN.match(t) and _ok(w) and any(COMMENT_VERB.match(x) for x in nxt[:2]):
                hits.append(("comment", w["t_start"]))
            elif BELL.match(t) and _ok(w) and any(x.startswith(("aç", "bas", "dokun", "çal")) for x in nxt):
                hits.append(("bell", w["t_start"]))
    return hits


def plan_requests(sents: list[dict], duration: float, merge_gap: float = 6.0, min_gap: float = 40.0) -> list[dict]:
    """Group requests said together into one reminder: {t0, t1, items: [(kind, t)]} with one click per kind, in
    the order they were said. A kind repeated in the same breath ("like atın ... like'ınızı atın") is clicked once.
    Reminders closer than `min_gap` seconds are merged away (the second one would look like a glitch)."""
    groups: list[list[tuple[str, float]]] = []
    for kind, t in sorted(find_requests(sents), key=lambda h: h[1]):
        if groups and t - groups[-1][-1][1] <= merge_gap:
            if kind not in {k for k, _ in groups[-1]}:
                groups[-1].append((kind, t))
            continue
        groups.append([(kind, t)])
    out: list[dict] = []
    for g in groups:
        if out and g[0][1] - out[-1]["items"][-1][1] < min_gap:
            continue
        kinds = [k for k, _ in g]
        if "subscribe" in kinds and "bell" not in kinds:
            pass   # the bell rings after the subscribe click anyway
        t0 = max(0.0, g[0][1] - 0.9)
        t1 = min(duration, max(g[-1][1] + 2.4, t0 + 4.2))
        out.append({"t0": round(t0, 3), "t1": round(t1, 3), "items": [(k, round(t, 3)) for k, t in g]})
    return out
