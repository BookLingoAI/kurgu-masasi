"""Talking-head (Anlatım / masa başı) graphics plan: what to set in big type next to the camera card.

The local language model reads the talk in windows and picks, ONLY from the speaker's own words:
  k - key statements: a small kicker, one to three words set huge, and the words right after them in the italic
      accent line ("Her çıkan aracı / kullanabilme yeteneği.") - about one every 45 s
  l - lists he counts off, with the icon they deserve: x (things that are missing / wrong), v (good / yes), o (plain)
Section titles come from the director (cards). Everything is checked against the transcript: a word that was not
said is not written.
"""
from __future__ import annotations

import re

from ..analysis.text import norm_word
from .graphics import grounded
from .proofread import _windows
from .shorts import sentences

SCHEMA = {"type": "object", "properties": {"k": {"type": "array", "items": {"type": "array"}},
                                           "l": {"type": "array", "items": {"type": "array"}}},
          "required": ["k", "l"]}
SYSTEM = ("Sen masa başında kameraya konuşulan YouTube videoları (izleyiciyle sohbet, dertleşme, anlatım) için çalışan "
          "bir hareketli-tipografi editörüsün. Konuşmacının KENDİ cümlelerinden ekranda kocaman yazılacak vurucu "
          "ifadeleri ve saydığı listeleri seçersin. Hiçbir kelime uydurmaz, cümleyi yeniden yazmazsın.")
NEG = {"yok", "değil", "olmaz", "yasak", "hayır", "olmuyor", "yapma", "yapmayın", "etme", "kötü", "yanlış", "hata"}


def _prompt(win: list[dict], n: int) -> str:
    lines = "\n".join(f"[{s['i']}] {s['text']}" for s in win)
    return (f"Transkript (cümle numarası, cümle):\n{lines}\n\n"
            f"1) \"k\": en fazla {n} vurucu ifade (izleyicinin aklında kalacak, duygusu ya da fikri güçlü cümleler; "
            "sıradan geçiş cümleleri OLMAZ): [cümle_no, \"ÜST SATIR (0-3 kelime bağlam, boş olabilir)\", "
            "\"iri kısım: cümleden AYNEN 1-3 kelime\", \"italik devam: cümlede iri kısımdan HEMEN SONRA gelen 1-5 "
            "kelime, AYNEN\"]. Örnek: cümle 'Asıl mesele her çıkan aracı kullanabilme yeteneği.' → "
            "[12, \"ASIL MESELE\", \"Her çıkan aracı\", \"kullanabilme yeteneği.\"].\n"
            "2) \"l\": konuşmacı art arda birkaç şey sayıyorsa: [ilk_cümle_no, \"1-3 kelime üst başlık\", "
            "[\"madde 1\", \"madde 2\", ...], \"simge\"]. Maddeler 1-4 kelime, konuşmada AYNEN geçen kelimelerle ve "
            "söylendiği sırayla; 2-5 madde. simge: \"x\" (olmayan/yanlış şeyler: 'El yok, Kol yok'), \"v\" (olumlu, "
            "yapılması gerekenler), \"o\" (düz liste). Yoksa boş liste.\n"
            "Sadece JSON yaz: {\"k\": [[...]], \"l\": [[...]]}")


def plan_talk(llm, words: list[dict], progress=None, budget_s: float = 900.0) -> dict:
    """{"phrases": [{sentence, kicker, big, small}], "lists": [{sentence, kicker, items, icon}], "failed"}."""
    import time
    sents = sentences(words)
    by_i = {s["i"]: s for s in sents}
    wins = _windows(sents, max_chars=6000)
    phrases, lists, failed = [], [], 0
    t0 = time.time()
    for k, win in enumerate(wins):
        if progress:
            progress(k / max(1, len(wins)), f"Ekran yazıları seçiliyor {k + 1}/{len(wins)}")
        if time.time() - t0 > budget_s:
            failed += len(wins) - k
            break
        dur = win[-1]["end"] - win[0]["start"]
        try:
            raw = llm.chat_json(SYSTEM, _prompt(win, max(1, round(dur / 45))), SCHEMA, max_tokens=800, temperature=0.3)
        except Exception:  # noqa: BLE001 - that part simply gets no big type
            failed += 1
            continue
        for row in raw.get("k") or []:
            p = _phrase(row, by_i)
            if p:
                phrases.append(p)
        for row in raw.get("l") or []:
            L = _list(row, by_i)
            if L:
                lists.append(L)
    return {"phrases": phrases, "lists": lists, "failed": failed, "windows": len(wins)}


def _clean(x) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().strip('"')


def _phrase(row, by_i: dict) -> dict | None:
    if not isinstance(row, (list, tuple)) or len(row) < 4 or not str(row[0]).lstrip("-").isdigit():
        return None
    s = by_i.get(int(row[0]))
    kicker, big, small = _clean(row[1]), _clean(row[2]).rstrip(",;:"), _clean(row[3])
    if not s or not big or not (1 <= len(big.split()) <= 3) or len(small.split()) > 6:
        return None
    if not grounded(big, s) or (small and not grounded(small, s)):
        return None
    if kicker and (len(kicker.split()) > 4 or not grounded(kicker, s)):
        kicker = ""
    return {"sentence": s["i"], "kicker": kicker, "big": big, "small": small}


def _list(row, by_i: dict) -> dict | None:
    if not isinstance(row, (list, tuple)) or len(row) < 3 or not str(row[0]).lstrip("-").isdigit():
        return None
    s = by_i.get(int(row[0]))
    items = [_clean(x) for x in (row[2] if isinstance(row[2], list) else []) if _clean(x)]
    if not s or not (2 <= len(items) <= 5) or not all(1 <= len(x.split()) <= 5 for x in items):
        return None
    # the items may run on into the next sentences (as item_times looks for them)
    said = {"words": [w for k in range(s["i"], s["i"] + 4) for w in (by_i.get(k) or {"words": []})["words"]]}
    if not all(grounded(it, said) for it in items):
        return None
    icon = str(row[3]).lower().strip() if len(row) > 3 else "o"
    if icon not in ("x", "v", "o"):
        icon = "o"
    if icon != "x" and all(norm_word(it.split()[-1]) in NEG for it in items):
        icon = "x"
    kicker = _clean(row[1])[:28]
    return {"sentence": s["i"], "kicker": kicker if grounded(kicker, said) else "", "items": items, "icon": icon}


def word_time(ws: list[dict], text: str, start: int = 0) -> tuple[float, int] | None:
    """(time, index) of the first word of `text` among ws from `start` (Turkish suffixes allowed)."""
    first = text.split()[0] if text.split() else ""
    key = norm_word(first)[:4]
    if not key:
        return None
    for j in range(start, len(ws)):
        if norm_word(ws[j]["word"]).startswith(key):
            return ws[j]["t_start"], j
    return None


def item_times(sent_by_i: dict, i: int, items: list[str], reach: int = 4) -> list[float]:
    """When each item is said (its first word, in order, in this sentence and the next few); [] if one is missing."""
    ws = [w for k in range(i, i + reach) for w in (sent_by_i.get(k) or {"words": []})["words"]]
    out, pos = [], 0
    for it in items:
        hit = word_time(ws, it, pos)
        if hit is None:
            return []
        out.append(hit[0])
        pos = hit[1] + 1
    return out
