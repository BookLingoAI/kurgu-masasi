"""Proofreading the transcript before anything is written on screen.

Whisper mishears food words and English ("yalan ederken" for "dana yanağı", "Çinli sos" for "chili sauce", "suçu"
for "suşi"), and every caption and title built on such a word is wrong twice. The local language model reads the
talk in windows and returns only the sentences it is sure were misheard, fixed word for word (no rewriting), plus
a short Turkish translation of the English sentences (the speaker's American wife, the waiters), which are shown
as translated subtitles. The result is cached per transcript and applied to the words in place.
"""
from __future__ import annotations

import difflib
import re

from ..analysis.text import norm_word
from .shorts import sentences

SCHEMA = {"type": "object", "properties": {"d": {"type": "array", "items": {"type": "array"}},
                                           "t": {"type": "array", "items": {"type": "array"}}},
          "required": ["d", "t"]}
EN_WORDS = {"the", "a", "is", "are", "you", "i", "it", "this", "that", "do", "what", "yes", "no", "okay", "ok", "good",
            "like", "want", "have", "can", "please", "thank", "thanks", "eat", "eating", "more", "one", "go", "together",
            "and", "for", "we", "your", "my", "it's", "i'm", "don't", "how", "much", "so", "very", "nice", "here"}


def is_english(s: dict) -> bool:
    ws = s["words"]
    if not ws:
        return False
    tagged = sum(1 for w in ws if w.get("lang") == "en")
    toks = [norm_word(w["word"]) for w in ws]
    return tagged >= 0.6 * len(ws) or (len(ws) >= 3 and sum(1 for t in toks if t in EN_WORDS) >= 0.6 * len(ws))


def _windows(sents: list[dict], max_chars: int = 6500, max_n: int = 110) -> list[list[dict]]:
    out, cur, size = [], [], 0
    for s in sents:
        if cur and (size + len(s["text"]) > max_chars or len(cur) >= max_n):
            out.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s["text"]) + 12
    if cur:
        out.append(cur)
    return out


SYSTEM = ("Sen Türkçe bir YouTube kanalının altyazı editörüsün. Transkript Whisper ile otomatik yazıldı ve yanlış duyulmuş "
          "kelimeler içeriyor. Konuşmacı Amerika'da (Teksas) yaşayan Türk bir vlogcu; restoranlarda yemek yiyor, yemekleri "
          "ve fiyatları anlatıyor; Amerikalı eşiyle ve garsonlarla bazen İngilizce konuşuyor.")


def _prompt(win: list[dict], vocab: list[str]) -> str:
    lines = "\n".join(f"[{s['i']}]{' (EN)' if is_english(s) else ''} {s['text']}" for s in win)
    return (f"Transkript:\n{lines}\n\n"
            "Bu kanalda sık geçen kelimeler: " + ", ".join(vocab[:60]) + ".\n"
            "Önceki videolarda görülen hatalar: 'yalan ederken'→'dana yanağı', 'vatopotlar'→'ahtapotlar', "
            "'açık alırım'→'aç kalırım', 'Çinli sos'→'chili sos', 'şerefino'→'jalapeño', 'suçu'→'suşi', "
            "'sinirsiz'→'sınırsız', 'müşroni'→'mushroom', 'krepe'→'crab'.\n\n"
            "1) \"d\": Yanlış duyulmuş kelime içeren cümleleri düzelt: [cümle_no, \"düzeltilmiş cümle\"]. YALNIZCA yanlış "
            "kelimeleri değiştir; cümleyi yeniden yazma, kelime ekleyip çıkarma, noktalamayı koru. Anlamca zaten doğru "
            "olan cümleleri yazma. Emin değilsen o cümleye dokunma.\n"
            "2) \"t\": (EN) işaretli ya da İngilizce konuşulan cümleler için kısa, doğal Türkçe çeviri: "
            "[cümle_no, \"Türkçe çeviri\"]. Yanlış duyulmuş İngilizceyi bağlamdan anla (ör. 'Together go' → 'Hadi birlikte gidelim').\n"
            "Sadece JSON yaz: {\"d\": [[...]], \"t\": [[...]]}")


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


def proofread(words: list[dict], llm, vocab: list[str], progress=None, budget_s: float = 900.0) -> dict:
    """{"fixes": {str(i): {"from", "to"}}, "translations": {str(i): {"from", "to"}}, "windows", "failed"}."""
    import time
    sents = sentences(words)
    wins = _windows(sents)
    fixes, trans, failed = {}, {}, 0
    t0 = time.time()
    by_i = {s["i"]: s for s in sents}
    for k, win in enumerate(wins):
        if progress:
            progress(k / max(1, len(wins)), f"Yazım denetimi {k + 1}/{len(wins)}")
        if time.time() - t0 > budget_s:
            failed += len(wins) - k
            break
        try:
            raw = llm.chat_json(SYSTEM, _prompt(win, vocab), SCHEMA, max_tokens=900, temperature=0.1)
        except Exception:  # noqa: BLE001 - the transcript stays as heard
            failed += 1
            continue
        for row in raw.get("d") or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2 or not str(row[0]).lstrip("-").isdigit():
                continue
            s = by_i.get(int(row[0]))
            new = re.sub(r"\s+", " ", str(row[1])).strip()
            if not s or not new or new == s["text"] or is_english(s):
                continue
            # a correction, not a rewrite: most of the sentence must stay the same
            if _similar(new, s["text"]) < 0.55 or abs(len(new.split()) - len(s["words"])) > max(2, len(s["words"]) // 3):
                continue
            fixes[str(s["i"])] = {"from": s["text"], "to": new}
        for row in raw.get("t") or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2 or not str(row[0]).lstrip("-").isdigit():
                continue
            s = by_i.get(int(row[0]))
            tr = re.sub(r"\s+", " ", str(row[1])).strip()
            if s and tr and is_english(s):
                trans[str(s["i"])] = {"from": s["text"], "to": tr}
    if progress:
        progress(1.0, "Yazım denetimi bitti")
    return {"fixes": fixes, "translations": trans, "windows": len(wins), "failed": failed}


def _respell(ws: list[dict], new_text: str) -> list[dict]:
    """Put the corrected words onto the timed words: unchanged words keep their times, changed stretches share the
    time of the words they replace."""
    old = [w["word"] for w in ws]
    new = new_text.split()
    out: list[dict] = []
    sm = difflib.SequenceMatcher(a=[norm_word(x) for x in old], b=[norm_word(x) for x in new], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for a, b in zip(range(i1, i2), range(j1, j2)):
                out.append(dict(ws[a], word=new[b]))
            continue
        if j2 == j1:          # the model dropped words: keep them as heard (never lose a timed word)
            out.extend(ws[i1:i2])
            continue
        if i2 == i1:          # inserted words: attach to the previous word's time
            base = out[-1] if out else ws[0]
            for b in range(j1, j2):
                out.append(dict(base, word=new[b], fixed=True))
            continue
        span = ws[i1:i2]
        t0, t1 = span[0]["t_start"], span[-1]["t_end"]
        s0, s1 = span[0]["start"], span[-1]["end"]
        n = j2 - j1
        for q, b in enumerate(range(j1, j2)):
            f0, f1 = q / n, (q + 1) / n
            out.append(dict(span[min(len(span) - 1, q * len(span) // n)], word=new[b], fixed=True,
                            t_start=round(t0 + (t1 - t0) * f0, 3), t_end=round(t0 + (t1 - t0) * f1, 3),
                            start=round(s0 + (s1 - s0) * f0, 3), end=round(s0 + (s1 - s0) * f1, 3)))
    return out


def apply(words: list[dict], result: dict | None) -> list[dict]:
    """New word list with the fixes and translations applied (sentences are matched by their original text, so a
    stale result never changes the wrong sentence)."""
    if not result:
        return words
    sents = sentences(words)
    fixes, trans = result.get("fixes") or {}, result.get("translations") or {}
    out: list[dict] = []
    for s in sents:
        ws = s["words"]
        fx = fixes.get(str(s["i"]))
        if fx and fx["from"] == s["text"]:
            ws = _respell(ws, fx["to"])
        tr = trans.get(str(s["i"]))
        if tr and tr["from"] == s["text"]:
            ws = [dict(w, lang="en", tr_line=tr["to"]) for w in ws]
        elif is_english(s):
            ws = [dict(w, lang="en") for w in ws]
        out.extend(ws)
    return out
