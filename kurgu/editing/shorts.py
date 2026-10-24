"""Pick a curiosity-driven 1-3 minute Shorts section from the edited video.

Selection works on the CUT transcript (edited timeline). The local LLM picks a contiguous range of sentences plus an
optional 'hook' sentence from inside the range that is shown first (cold open). Without an LLM a rule-based scorer is
used and the result is marked as weaker.
"""
from __future__ import annotations

import re

from ..analysis.text import norm_word

HOOK_WORDS = {
    "ama", "çünkü", "aslında", "gerçekten", "asla", "hiç", "hiçbir", "kimse", "herkes", "en", "inanmayacaksınız",
    "sır", "gerçek", "zor", "kolay", "pişman", "hata", "korku", "korkunç", "inanılmaz", "şok", "neden", "nasıl",
    "niye", "kaybettim", "öldü", "öğrendim", "fark", "para", "dolar", "yalan", "doğru", "yanlış", "sakın", "mutlaka",
}


def sentences(words: list[dict], max_gap: float = 0.8, max_len_s: float = 20.0) -> list[dict]:
    """Group cut-timeline words (t_start/t_end) into sentences."""
    out: list[dict] = []
    cur: list[dict] = []

    def flush():
        if cur:
            out.append({"i": len(out), "start": round(cur[0]["t_start"], 3), "end": round(cur[-1]["t_end"], 3),
                        "text": " ".join(w["word"] for w in cur).strip(), "words": list(cur)})
            cur.clear()

    for w in words:
        if cur and (w["t_start"] - cur[-1]["t_end"] > max_gap or w["t_end"] - cur[0]["t_start"] > max_len_s):
            flush()
        cur.append(w)
        ww = w["word"].strip()
        # sentence end, but not an ellipsis or a Turkish ordinal like "3." ("3. cümle")
        if re.search(r"[.!?]$", ww) and not ww.endswith("...") and not re.fullmatch(r"\d+\.", ww):
            flush()
    flush()
    return out


def score_sentence(s: dict) -> float:
    toks = [norm_word(w["word"]) for w in s["words"]]
    sc = sum(1.0 for t in toks if t in HOOK_WORDS)
    sc += 1.5 if s["text"].rstrip().endswith("?") else 0.0
    sc += 1.0 if re.search(r"\d", s["text"]) else 0.0
    sc += 0.5 if any(t in ("ben", "benim", "bana") for t in toks) else 0.0
    n = len(toks)
    sc += 0.5 if 5 <= n <= 18 else 0.0
    return sc / (1 + 0.02 * max(0, n - 18))


def heuristic_pick(sents: list[dict], min_s: float, max_s: float, target_s: float = 90.0) -> dict:
    best = None
    for a in range(len(sents)):
        total = 0.0
        for b in range(a, len(sents)):
            dur = sents[b]["end"] - sents[a]["start"]
            if dur > max_s:
                break
            total += score_sentence(sents[b])
            if dur < min_s:
                continue
            opening = score_sentence(sents[a]) * 2.0
            ending = 1.0 if sents[b]["text"].rstrip().endswith((".", "!")) else 0.0
            sc = (total / max(1.0, dur / 30.0)) + opening + ending - abs(dur - target_s) / 60.0
            if best is None or sc > best[0]:
                best = (sc, a, b)
    if best is None:
        return {"start": 0, "end": len(sents) - 1, "hook": None, "reason": "Video kısa, tamamı kullanıldı", "method": "kural"}
    _, a, b = best
    inner = [s for s in sents[a + 2:b + 1] if 2.5 <= s["end"] - s["start"] <= 8.0]
    hook = max(inner, key=score_sentence)["i"] if inner else None
    return {"start": a, "end": b, "hook": hook, "method": "kural",
            "reason": "Kural tabanlı seçim (dil modeli yoktu): soru, sayı ve vurgu kelimeleri yoğun bölüm"}


PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": "integer"},
        "end": {"type": "integer"},
        "hook": {"type": ["integer", "null"]},
        "reason": {"type": "string"},
        "title": {"type": "string"},
    },
    "required": ["start", "end", "hook", "reason", "title"],
    "additionalProperties": False,
}


def llm_pick(llm, sents: list[dict], min_s: float, max_s: float) -> dict:
    lines = "\n".join(f"[{s['i']}] ({s['start']:.0f}-{s['end']:.0f} sn) {s['text']}" for s in sents)
    system = ("Sen deneyimli bir YouTube Shorts editörüsün. Türkçe konuşan bir içerik üreticisinin videosundan, "
              "tek başına anlaşılır ve izleyicide merak uyandıran dikey bir kısa video bölümü seçiyorsun. "
              "Sadece verilen transkripti kullan; söylenmemiş hiçbir şey ekleme.")
    user = (f"Aşağıda kurgulanmış videonun cümleleri numaralı ve zamanlı olarak var.\n\n{lines}\n\n"
            f"Görev:\n- Ardışık cümlelerden oluşan TEK bir bölüm seç (start ve end cümle numarası). "
            f"Süre {int(min_s)} ile {int(max_s)} saniye arasında olsun; 60-120 sn idealdir.\n"
            "- Bölüm güçlü bir cümleyle başlasın, bağlamsız kalan bir göndermeyle başlamasın ve tatmin edici bitsin.\n"
            "- 'hook': bölümün İÇİNDEN, en merak uyandıran kısa bir cümlenin numarası (açılışta önce gösterilecek). "
            "Uygun yoksa null.\n- 'title': bu Shorts için 40-70 karakterlik merak uyandıran Türkçe başlık.\n"
            "- 'reason': neden bu bölümü seçtiğini bir cümleyle açıkla.\nSadece JSON döndür.")
    out = llm.chat_json(system, user, PICK_SCHEMA, max_tokens=600, temperature=0.4)
    out["method"] = "dil modeli"
    return out


def validate_pick(pick: dict, sents: list[dict], min_s: float, max_s: float) -> dict:
    n = len(sents)
    a = max(0, min(n - 1, int(pick.get("start", 0))))
    b = max(a, min(n - 1, int(pick.get("end", a))))
    # grow / shrink to the allowed duration on sentence boundaries
    while sents[b]["end"] - sents[a]["start"] < min_s and (b < n - 1 or a > 0):
        if b < n - 1:
            b += 1
        else:
            a -= 1
    while sents[b]["end"] - sents[a]["start"] > max_s and b > a:
        b -= 1
    hook = pick.get("hook")
    if hook is not None:
        hook = int(hook)
        if not (a <= hook <= b) or hook == a or not (1.5 <= sents[hook]["end"] - sents[hook]["start"] <= 10.0):
            hook = None
    fixed = dict(pick, start=a, end=b, hook=hook)
    fixed["duration"] = round(sents[b]["end"] - sents[a]["start"], 2)
    fixed["adjusted"] = (a, b) != (pick.get("start"), pick.get("end"))
    return fixed


def timeline_to_source(t0: float, t1: float, segments: list[dict]) -> list[dict]:
    """Edited-timeline range -> list of source pieces {file, in, out}."""
    pieces = []
    for s in segments:
        a, b = max(t0, s["timeline_start"]), min(t1, s["timeline_end"])
        if b - a > 0.02:
            off = s["in"] - s["timeline_start"]
            pieces.append({"file": s["file"], "in": round(a + off, 4), "out": round(b + off, 4)})
    return pieces


def plan_short(sents: list[dict], pick: dict, segments: list[dict], pad: float = 0.08) -> dict:
    a, b = sents[pick["start"]], sents[pick["end"]]
    parts = []
    if pick.get("hook") is not None:
        h = sents[pick["hook"]]
        parts.append({"kind": "hook", "t0": max(0.0, h["start"] - pad), "t1": h["end"] + pad, "text": h["text"]})
    parts.append({"kind": "main", "t0": max(0.0, a["start"] - pad), "t1": b["end"] + pad,
                  "text": " ".join(s["text"] for s in sents[pick["start"]:pick["end"] + 1])})
    pieces = []
    for p in parts:
        for pc in timeline_to_source(p["t0"], p["t1"], segments):
            pieces.append(dict(pc, kind=p["kind"]))
    words = []
    t = 0.0
    for p in parts:
        for s in sents[pick["start"]:pick["end"] + 1] if p["kind"] == "main" else [sents[pick["hook"]]]:
            for w in s["words"]:
                if p["t0"] <= w["t_start"] and w["t_end"] <= p["t1"]:
                    words.append({"word": w["word"], "start": round(t + w["t_start"] - p["t0"], 3),
                                  "end": round(t + w["t_end"] - p["t0"], 3), "kind": p["kind"]})
        t += p["t1"] - p["t0"]
    return {"parts": parts, "pieces": pieces, "words": words, "duration": round(t, 2)}


# ------------------------------------------------------------------ story Shorts (summary of the whole video) ---
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "integer"},
        "headline": {"type": "string"},
        "clips": {"type": "array", "items": {"type": "object", "properties": {
            "start": {"type": "integer"}, "end": {"type": "integer"}, "role": {"type": "string"}},
            "required": ["start", "end", "role"], "additionalProperties": False}},
        "title": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["hook", "headline", "clips", "title", "reason"],
    "additionalProperties": False,
}

DANGLING = {"o", "bu", "şu", "onun", "bunun", "yani", "ve", "ama", "çünkü", "zaten", "işte", "dolayısıyla", "sonra"}


CTA_PREFIX = ("abone", "like", "yorum", "bildirim", "zil", "kanal", "paylaş", "takip")
CTA_WORDS = {"beğenin", "beğenmeyi", "unutmayın", "unutmayin", "beğenip", "atmayı"}
VENUE = {"restoran", "restoranı", "restorana", "mekan", "mekân", "mekana", "yer", "yere", "burası", "buraya", "bugün",
         "geldik", "geldim", "buradayız", "şehir", "şehrinde", "dükkan", "dükkân", "lokanta", "kafe", "otel", "japon",
         "japonya", "kore", "amerika", "türk", "brezilya", "sınırsız", "menü", "menüsü"}
PRICE = {"dolar", "dolarlık", "lira", "cent", "sent", "hesap", "hesabı", "toplam", "ödedik", "ödeyeceğiz", "ödedim",
         "fiyat", "fiyatı", "fiyatlar", "bahşiş", "tip", "para", "parası", "ucuz", "pahalı", "kişi", "kişilik"}
VERDICT = {"tavsiye", "öneririm", "değer", "değdi", "değmez", "beğendim", "bayıldım", "kesinlikle", "puan", "sonuç",
           "genel", "bence", "gelin", "mutlaka", "pişman", "memnun", "harika", "muhteşem", "tekrar"}
CURIOUS = {"kaç", "acaba", "bakalım", "merak", "sizce", "tahmin", "ne", "nasıl", "neden", "niye", "hangi", "kim"}


def _toks(s: dict) -> list[str]:
    return [norm_word(w["word"]) for w in s["words"]]


def is_cta(s: dict) -> bool:
    t = _toks(s)
    return any(x.startswith(CTA_PREFIX) for x in t) or bool(set(t) & CTA_WORDS)


def is_junk(s: dict) -> bool:
    t = _toks(s)
    if not t or s["end"] - s["start"] < 1.2:
        return True
    if any(t.count(x) >= 3 for x in set(t)) and len(set(t)) <= 3:   # "aç aç aç aç" (Whisper loop on music)
        return True
    if set(t) <= {"müzik", "altyazı", "music"}:
        return True
    probs = [w.get("prob") for w in s["words"] if w.get("prob") is not None]
    return bool(probs) and sum(probs) / len(probs) < 0.45


def proper_names(sents: list[dict]) -> set[str]:
    """Words Whisper always writes capitalised and that recur (Japonya, Brezilya, Hatun): names, not noise."""
    from collections import Counter
    lower = {norm_word(w["word"]) for s in sents for w in s["words"] if w["word"][:1].islower()}
    caps = Counter(norm_word(w["word"].split("'")[0]) for s in sents for w in s["words"] if w["word"][:1].isupper())
    return {w for w, c in caps.items() if c >= 2 and w not in lower and len(w) > 2}


def story_scores(s: dict, T: float, names: set[str] | None = None) -> dict:
    """Role scores for a sentence; T = video length (s). All from words actually said."""
    t = _toks(s)
    st = set(t)
    pos = s["start"] / max(1.0, T)
    q = s["text"].rstrip().endswith("?")
    num = bool(re.search(r"\d", s["text"]))
    from .graphics import PRAISE
    praise = sum(1 for x in t if x in PRAISE)
    cur = 2.0 * q + 1.5 * bool(st & CURIOUS) + min(2.0, sum(1 for x in t if x in HOOK_WORDS)) + 1.0 * num
    cur += 1.5 * bool(st & PRICE) * q
    ctx = (2.0 if pos < 0.3 else 0.0) + 1.5 * min(2, len(st & VENUE)) + 0.5 * num - (1.0 if q else 0.0)
    high = 2.0 * min(2, praise) + 1.0 * s["text"].rstrip().endswith("!") + 0.5 * num + 0.5 * bool(st & PRICE)
    pay = (3.0 * pos if pos > 0.6 else 0.0) + (2.0 if pos > 0.85 else 0.0) + 2.0 * (num and bool(st & PRICE)) + 1.5 * min(2, len(st & VERDICT))
    n = len(t)
    fit = 1.0 if 4 <= n <= 22 else 0.6
    # Whisper on noisy audio: random mid-sentence capitals and "O6"-like tokens = garbled, unclear line
    names = names or set()
    caps = sum(1 for w in s["words"][1:] if w["word"][:1].isupper() and norm_word(w["word"].split("'")[0]) not in names)
    garbled = sum(1 for w in s["words"] if re.search(r"[A-Za-zçğıöşüÇĞİÖŞÜ]\d|\d[A-Za-zçğıöşüÇĞİÖŞÜ]", w["word"]))
    if caps > max(2, n // 4) or garbled:
        fit *= 0.3
    if re.search(r"[.!?]$", s["text"].rstrip()):
        fit *= 1.15
    return {"hook": cur * fit, "giriş": ctx * fit, "gelişme": high * fit, "sonuç": pay * fit, "pos": pos}


def story_summary(sents: list[dict], min_s: float, max_s: float, target_s: float = 60.0) -> dict:
    """Without (or instead of a failing) language model: a real mini story, like an editor would cut it.
    hook   - the most curious question/claim (cold open, its answer comes at the end)
    giriş  - where we are / what this is about (first third)
    gelişme- the strongest reactions, numbers, dishes (spread over the middle)
    sonuç  - the payoff: the bill / the verdict (last 40 %), matching the hook's question when possible
    Calls to action ("abone olun", "like atın"), music-hallucinated lines and unclear fragments are never used."""
    n = len(sents)
    if n < 6:
        return heuristic_summary(sents, min_s, max_s, target_s)
    T = sents[-1]["end"]
    ok = [s for s in sents if not is_cta(s) and not is_junk(s) and norm_word(s["words"][0]["word"]) not in DANGLING]
    if len(ok) < 5:
        return heuristic_summary(sents, min_s, max_s, target_s)
    names = proper_names(sents)
    sc = {s["i"]: story_scores(s, T, names) for s in ok}
    dur = lambda s: s["end"] - s["start"]  # noqa: E731
    hook_c = [s for s in ok if 2.0 <= dur(s) <= 9.0]
    hook = max(hook_c or ok, key=lambda s: sc[s["i"]]["hook"] + (0.5 if sc[s["i"]]["pos"] < 0.6 else 0.0))
    price_q = bool(set(_toks(hook)) & (PRICE | {"kaç"})) and hook["text"].rstrip().endswith("?")
    used = {hook["i"]}

    def best(cands, role, bonus=None):
        cands = [s for s in cands if s["i"] not in used and dur(s) <= 12.0]
        if not cands:
            return None
        return max(cands, key=lambda s: sc[s["i"]][role] + (bonus(s) if bonus else 0.0))

    def clip(s, role):
        used.add(s["i"])
        end = s["i"]
        nxt = s["i"] + 1
        # continue into the next sentence when it completes the thought and the clip stays short
        if nxt < n and nxt in sc and nxt not in used and sents[nxt]["end"] - s["start"] <= 11.0 \
                and not s["text"].rstrip().endswith(("?", "!")) \
                and (dur(s) < 4.0 or sc[nxt][role] >= 0.6 * sc[s["i"]][role]):
            end = nxt
            used.add(nxt)
        return {"start": s["i"], "end": end, "role": role}
    clips = []
    g = best([s for s in ok if sc[s["i"]]["pos"] < 0.3], "giriş")
    if g:
        clips.append(clip(g, "giriş"))
    son = best([s for s in ok if sc[s["i"]]["pos"] > 0.6], "sonuç",
               (lambda s: 2.5 if price_q and bool(set(_toks(s)) & PRICE) and re.search(r"\d", s["text"]) else 0.0))
    mids, budget = [], target_s - dur(hook) - sum(sents[c["end"]]["end"] - sents[c["start"]]["start"] for c in clips) \
        - (dur(son) if son else 0)
    pool = sorted((s for s in ok if 0.2 <= sc[s["i"]]["pos"] <= 0.88 and s["i"] not in used),
                  key=lambda s: sc[s["i"]]["gelişme"], reverse=True)
    for s in pool:
        if len(mids) >= 4 or budget < 4:
            break
        if sc[s["i"]]["gelişme"] <= 0.5 or any(abs(s["start"] - m["start"]) < 0.07 * T for m in mids) or s["i"] in used:
            continue
        if son and s["i"] >= son["i"]:
            continue
        mids.append(s)
        budget -= dur(s)
    for s in sorted(mids, key=lambda s: s["i"]):
        clips.append(clip(s, "gelişme"))
    if son and son["i"] not in used:
        c = clip(son, "sonuç")
        p = son["i"] - 1   # a short payoff ("80, diğeri 90 dolar.") needs the sentence before it
        if dur(son) < 4.0 and p >= 0 and p in sc and p not in used and son["end"] - sents[p]["start"] <= 12.0:
            c["start"] = p
            used.add(p)
        clips.append(c)
    clips.sort(key=lambda c: c["start"])
    head = ""
    if hook["text"].rstrip().endswith("?"):
        part = re.split(r"[,;]\s*", hook["text"].strip())[-1]
        words = part.split()
        head = " ".join(words[-7:]) if len(words) > 7 else part
        from .graphics import tr_upper
        head = tr_upper(head)
    return {"hook": hook["i"], "headline": head, "clips": clips, "title": "", "method": "kural (hikâye)",
            "reason": "Kanca (merak sorusu) + giriş (nerede/ne) + en güçlü anlar + sonuç (hesap/karar); "
                      "abone/like çağrıları ve anlaşılmayan cümleler hariç"}


def llm_candidates(sents: list[dict], limit: int = 120) -> list[dict]:
    """A long video does not fit the local model's context: send only the sentences that could matter."""
    n = len(sents)
    if n <= limit:
        return [s for s in sents if not is_cta(s) and not is_junk(s)] or sents
    T = sents[-1]["end"]
    ok = [s for s in sents if not is_cta(s) and not is_junk(s)]
    names = proper_names(sents)
    sc = {s["i"]: story_scores(s, T, names) for s in ok}
    keep = set()
    for role, k in (("hook", 25), ("giriş", 20), ("gelişme", 45), ("sonuç", 25)):
        keep |= {s["i"] for s in sorted(ok, key=lambda s: sc[s["i"]][role], reverse=True)[:k]}
    step = max(1, len(ok) // 20)   # plus an even spread so the story has its context
    keep |= {s["i"] for s in ok[::step]}
    return [s for s in sents if s["i"] in keep][:limit + 20]


def llm_summary(llm, sents: list[dict], min_s: float, max_s: float) -> dict:
    T = sents[-1]["end"] if sents else 1.0
    cands = llm_candidates(sents)
    lines = "\n".join(f"[{s['i']}] (%{int(100 * s['start'] / max(1.0, T))}, {s['end'] - s['start']:.0f} sn) "
                      f"{s['text'][:95]}" for s in cands)
    system = ("Sen profesyonel bir YouTube Shorts kurgucususun. Uzun bir vlog videosundan, videonun ÖZETİ olan, "
              "merak uyandıran dikey bir kısa video kurguluyorsun. Sadece transkriptteki cümleleri kullanırsın.")
    user = (f"Kurgulanmış videonun cümleleri:\n\n{lines}\n\n"
            "Shorts'u şöyle kur:\n"
            "1) 'hook': videonun en merak uyandıran, şaşırtıcı ya da soru soran kısa cümlesi (2-8 sn). Shorts bununla "
            "açılacak; cevabını hemen vermeyen bir cümle seç.\n"
            "2) 'clips': kronolojik sırada 3-6 parça; her parça ardışık 1-3 cümle (start-end cümle numarası). "
            "İlk parça GİRİŞ (konu/bağlam), ortadakiler GELİŞME (en ilginç anlar, sayılar, deneyimler), son parça "
            "SONUÇ (ders, tavsiye ya da vurucu cümle). 'role' = giriş / gelişme / sonuç.\n"
            "3) Her parça kendi başına anlaşılsın: 'o yüzden', 'bu', 've' gibi öncekine bağlı sözlerle başlamasın.\n"
            f"4) Toplam süre (kanca dahil) {int(min_s)}-{int(max_s)} saniye; hook cümlesini parçalarda tekrar kullanma.\n"
            "5) 'headline': ilk saniyelerde ekranın üstünde duracak 3-6 kelimelik merak başlığı.\n"
            "6) 'title': 40-70 karakterlik Shorts başlığı. 'reason': kurguyu bir cümleyle açıkla.\n"
            "7) Abone ol / like / yorum çağrısı yapan cümleleri ASLA seçme. Satır başındaki %, cümlenin videodaki "
            "yeridir: giriş baştan, sonuç sondan olsun.\nSadece JSON döndür.")
    out = llm.chat_json(system, user, SUMMARY_SCHEMA, max_tokens=900, temperature=0.4)
    out["method"] = "dil modeli"
    return out


def heuristic_summary(sents: list[dict], min_s: float, max_s: float, target_s: float = 65.0) -> dict:
    """Without an LLM: hook = most 'curious' sentence; then best moments from the beginning, middle and end."""
    n = len(sents)
    if n < 4:
        return {"hook": None, "headline": "", "clips": [{"start": 0, "end": n - 1, "role": "giriş"}], "title": "",
                "method": "kural", "reason": "Video çok kısa, tamamı kullanıldı"}
    ok = [s for s in sents if 2.0 <= s["end"] - s["start"] <= 9.0]
    hook = max(ok or sents, key=score_sentence)["i"]
    thirds = [sents[: n // 3], sents[n // 3: 2 * n // 3], sents[2 * n // 3:]]
    clips = []
    for part, role, k in ((thirds[0], "giriş", 1), (thirds[1], "gelişme", 2), (thirds[2], "sonuç", 1)):
        cands = sorted((s for s in part if s["i"] != hook and norm_word(s["words"][0]["word"]) not in DANGLING),
                       key=score_sentence, reverse=True)
        for s in cands[:k]:
            end = s["i"] + 1 if s["i"] + 1 < n and s["i"] + 1 != hook else s["i"]
            clips.append({"start": s["i"], "end": end, "role": role})
    return {"hook": hook, "headline": "", "clips": clips, "title": "", "method": "kural",
            "reason": "Kural tabanlı özet (dil modeli yoktu): baştan, ortadan ve sondan en vurgulu anlar"}


def validate_summary(pick: dict, sents: list[dict], min_s: float, max_s: float) -> dict:
    n = len(sents)
    dur = lambda a, b: sents[b]["end"] - sents[a]["start"]  # noqa: E731
    hook = pick.get("hook")
    if hook is not None and not (0 <= int(hook) < n and 1.2 <= dur(int(hook), int(hook)) <= 12):
        hook = None
    hook = int(hook) if hook is not None else None
    clips = []
    for c in pick.get("clips", []):
        a, b = int(c.get("start", -1)), int(c.get("end", -1))
        if not (0 <= a < n):
            continue
        b = max(a, min(n - 1, b, a + 4))
        if hook is not None and a <= hook <= b:   # never repeat the hook
            if hook == a:
                a += 1
            else:
                b = hook - 1
        if a <= b:
            clips.append({"start": a, "end": b, "role": c.get("role", "gelişme")})
    clips.sort(key=lambda c: c["start"])
    merged = []
    for c in clips:
        if merged and c["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
        else:
            merged.append(dict(c))
    clips = merged or [{"start": 0, "end": min(n - 1, 3), "role": "giriş"}]

    def total():
        return (dur(hook, hook) if hook is not None else 0) + sum(dur(c["start"], c["end"]) for c in clips)
    # too long: drop sentences from the end of middle clips first
    while total() > max_s and len(clips) > 1:
        mids = [c for c in clips[1:-1] if c["end"] > c["start"]] or [c for c in clips if c["end"] > c["start"]]
        if mids:
            mids[0]["end"] -= 1
        else:
            clips.pop(len(clips) // 2)
    # too short: extend clips by the following sentence (not into the next clip or the hook)
    grew = True
    while total() < min_s and grew:
        grew = False
        for i, c in enumerate(clips):
            nxt = clips[i + 1]["start"] if i + 1 < len(clips) else n
            if c["end"] + 1 < nxt and c["end"] + 1 != hook:
                c["end"] += 1
                grew = True
                if total() >= min_s:
                    break
    return dict(pick, hook=hook, clips=clips, duration=round(total(), 2))


def plan_summary(sents: list[dict], pick: dict, segments: list[dict], pad: float = 0.1) -> dict:
    """Hook + clips -> source pieces, words on the Shorts timeline and the cut points (for transitions)."""
    parts = []
    if pick.get("hook") is not None:
        h = sents[pick["hook"]]
        parts.append({"kind": "hook", "t0": max(0.0, h["start"] - pad), "t1": h["end"] + pad, "sents": [h]})
    for c in pick["clips"]:
        a, b = sents[c["start"]], sents[c["end"]]
        parts.append({"kind": c.get("role", "gelişme"), "t0": max(0.0, a["start"] - pad), "t1": b["end"] + pad,
                      "sents": sents[c["start"]:c["end"] + 1]})
    pieces, words, cuts = [], [], []
    t = 0.0
    for k, p in enumerate(parts):
        if k:
            cuts.append(round(t, 3))
        for j, pc in enumerate(timeline_to_source(p["t0"], p["t1"], segments)):
            pieces.append(dict(pc, kind=p["kind"], part=k, first=(j == 0)))
        for s in p["sents"]:
            for w in s["words"]:
                if p["t0"] <= w["t_start"] and w["t_end"] <= p["t1"]:
                    words.append({"word": w["word"], "start": round(t + w["t_start"] - p["t0"], 3),
                                  "end": round(t + w["t_end"] - p["t0"], 3)})
        t += p["t1"] - p["t0"]
    return {"parts": [{k: v for k, v in p.items() if k != "sents"} | {"text": " ".join(s["text"] for s in p["sents"])}
                      for p in parts], "pieces": pieces, "words": words, "cuts": cuts, "duration": round(t, 2)}
