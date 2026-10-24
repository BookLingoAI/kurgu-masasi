"""Step 2: content-aware on-screen titles + punch-in zooms for the main edit.

- Titles (sentence level, not word level): the local LLM picks a few key moments (one per ~45-60 s) and writes a
  short headline that sums up what the SENTENCE says ("20 SANİYE HAŞLANIYOR", "167 DOLAR ÖDEDİK") plus a small label
  above it ("FİYAT", "SÜRE", "YORUM"). Without the LLM only facts that were actually said become titles: amounts with
  their unit and what follows ("17 DOLARLIK YER"), and "<thing> çok <adjective>" verdicts. Lone adjectives
  ("GÜZEL", "BÜYÜK") are never shown - they looked amateurish. Emoji are off unless chosen in the options.
- Zooms: jump cuts inside continuous speech alternate 100% / 108% framing around the face (hides the jump);
  emphasised moments get 115%.
"""
from __future__ import annotations

import re

from ..analysis.text import norm_word
from .shorts import sentences

CALLOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "callouts": {"type": "array", "items": {"type": "object", "properties": {
            "sentence": {"type": "integer"}, "trigger": {"type": "string"}, "text": {"type": "string"},
            "kicker": {"type": "string"}},
            "required": ["sentence", "trigger", "text", "kicker"], "additionalProperties": False}},
        "emphasis": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["callouts", "emphasis"],
    "additionalProperties": False,
}


EDITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "b": {"type": "array", "items": {"type": "array", "items": {"type": ["integer", "string"]}}},
        "l": {"type": "array", "items": {"type": "array"}},
        "v": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["b", "l", "v"],
}
KICKERS = ("FİYAT", "SÜRE", "MİKTAR", "MEKAN", "YEMEK", "YORUM", "İPUCU", "BİLGİ", "SORU")


def llm_callouts(llm, sents: list[dict], duration: float, per_s: float = 1 / 50, gap: float = 25.0) -> dict:
    """One editor's pass over a stretch of the talk (compact answer: the model writes ~8 tokens/s):
      b - titles: [sentence, "2-5 WORD HEADLINE", "KICKER", "trigger word"] - only real information
      l - lists the speaker counts off: [sentence, "KICKER", ["item 1", "item 2", "item 3"]]
      v - most emphasised sentences (punch-in zooms)."""
    n = max(1, min(4, int(duration / 150)))
    lines = "\n".join(f"[{s['i']}] ({int(s['start'] // 60)}:{int(s['start'] % 60):02d}) {s['text']}" for s in sents)
    system = ("Sen deneyimli bir YouTube vlog editörüsün. Amerika'da yaşayan Türk bir vlogcunun (restoranlar, yemekler, "
              "fiyatlar, günlük hayat) konuşmasının transkriptini okuyup ekrana konacak yazıları seçiyorsun. Yalnızca "
              "söylenenlere dayanırsın, bilgi uydurmazsın. Az ama değerli yazı koyarsın.")
    user = (f"Transkript (cümle numarası, dakika, cümle):\n{lines}\n\n"
            f"1) \"b\": en fazla {n} başlık, yalnızca izleyiciye YENİ ve somut bilgi veren anlar için: fiyat, süre, miktar, "
            "yemeğin ya da mekânın adı, önemli bir kural. [cümle_no, \"2-5 kelimelik BÜYÜK HARF başlık\", \"ETİKET\", "
            "\"cümlede AYNEN geçen tetik kelime\"]. ETİKET: FİYAT, SÜRE, MİKTAR, MEKAN, YEMEK, İPUCU ya da BİLGİ. "
            "Başlıktaki her bilgi cümlede açıkça söylenmiş olmalı; görüntüde zaten görülen ya da boş şeyler ('ETLER TAM "
            "KIVAMINDA', 'İLK DEFA GÖRÜYORUM') ve anlamı belirsiz/yanlış duyulmuş olabilecek cümleler başlık OLMAZ. "
            "Uygun an yoksa boş bırak. Örnek: [41, \"KİŞİ BAŞI 35 DOLAR\", \"FİYAT\", \"35\"].\n"
            "2) \"l\": konuşmacı birkaç şeyi art arda sayıyorsa (soslar, yemekler, kurallar) ekranda madde madde gösterilecek "
            "liste: [ilk_cümle_no, \"1-3 kelimelik üst başlık\", [\"madde 1\", \"madde 2\", ...]]. Maddeler 1-4 kelime, "
            "konuşmada AYNEN geçen kelimelerle ve söylendiği sırayla; 2-4 madde. Yoksa boş liste.\n"
            "3) \"v\": konuşmacının en çok vurgu yaptığı 3-6 cümlenin numarası.\n"
            "Sadece şu biçimde JSON yaz: {\"b\": [[...]], \"l\": [[...]], \"v\": [..]}")
    raw = llm.chat_json(system, user, EDITOR_SCHEMA, max_tokens=700, temperature=0.3)
    valid = {s["i"] for s in sents}
    callouts = []
    for row in raw.get("b") or []:
        if isinstance(row, dict):
            row = [row.get("sentence", row.get("i")), row.get("text", ""), row.get("kicker", ""), row.get("trigger", "")]
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            i = int(row[0])
        except (TypeError, ValueError):
            continue
        if i not in valid:
            continue
        kicker = str(row[2]).upper() if len(row) > 2 else "BİLGİ"
        callouts.append({"sentence": i, "text": str(row[1]), "kicker": kicker if kicker in KICKERS else "BİLGİ",
                         "trigger": str(row[3]) if len(row) > 3 else ""})
    lists = []
    for row in raw.get("l") or []:
        if not isinstance(row, (list, tuple)) or len(row) < 3 or not str(row[0]).lstrip("-").isdigit():
            continue
        items = [re.sub(r"\s+", " ", str(x)).strip() for x in (row[2] if isinstance(row[2], list) else []) if str(x).strip()]
        if int(row[0]) in valid and 2 <= len(items) <= 4 and all(1 <= len(x.split()) <= 5 for x in items):
            lists.append({"sentence": int(row[0]), "kicker": str(row[1])[:28], "items": items})
    ints = lambda xs: [int(x) for x in xs or [] if str(x).lstrip("-").isdigit() and int(x) in valid]   # noqa: E731
    return {"callouts": callouts, "emphasis": ints(raw.get("v")), "lists": lists}


def grounded(title: str, s: dict) -> bool:
    """The title says only what the sentence says: every content word of the title (numbers included) must appear
    in the sentence, allowing Turkish suffixes (first 4 letters). Stops "AMERİKA'DA SAHTE YENGEÇ" when nobody said
    "sahte", or a "GÖRÜYORUM" that was "yiyorum"."""
    sent = [norm_word(w["word"]) for w in s["words"]]
    probs = [float(w.get("prob", 1.0)) for w in s["words"]]
    nums = {re.sub(r"\D", "", w) for w in sent if re.search(r"\d", w)}
    words = [norm_word(t) for t in title.split()]
    if len(words) <= 2 and words and words[-1] in NARRATION:
        return False   # "DİNER YİYORUM": narrating what he does is not a title
    for n in words:
        if not n or n in GENERIC or n in TITLE_STOP:
            continue
        if re.search(r"\d", n):
            if re.sub(r"\D", "", n) not in nums:
                return False
            continue
        p = n[:max(3, min(4, len(n)))]
        hits = [k for k, w in enumerate(sent) if w and (w.startswith(p) or (len(w) >= 3 and n.startswith(w[:4])))]
        if not hits:
            return False
        if max(probs[k] for k in hits) < MIN_TITLE_PROB:
            return False   # Whisper was unsure of that word ("dil" heard as "diner"): never put it on screen
    return True


def resolve_callouts(raw: dict, sents: list[dict], min_gap: float = 30.0, dur: float = 3.2) -> list[dict]:
    out = []
    for c in sorted(raw.get("callouts", []), key=lambda c: c.get("sentence", 0)):
        i = int(c.get("sentence", -1))
        if not (0 <= i < len(sents)):
            continue
        text = re.sub(r"\s+", " ", c.get("text", "")).strip().strip('"').strip(".!")
        words = text.split()
        if not (1 <= len(words) <= 6) or not good_title(text):
            continue
        s = sents[i]
        if not grounded(text, s):
            continue      # says something the speaker did not say (or a misheard word): never on screen
        trig = norm_word(c.get("trigger", ""))
        t0 = next((w["t_start"] for w in s["words"] if norm_word(w["word"]) == trig), s["start"])
        if out and t0 - out[-1]["t0"] < min_gap:
            continue
        if any(o["text"].lower() == text.lower() and t0 - o["t0"] < 180 for o in out):
            continue   # the same text again so soon looks like a mistake
        kicker = tr_upper(re.sub(r"[^\wçğıöşüÇĞİÖŞÜ ]", "", str(c.get("kicker", ""))).strip())[:12]
        out.append({"t0": round(t0, 3), "t1": round(min(t0 + dur, s["end"] + 1.5), 3), "text": tr_upper(text),
                    "kicker": kicker, "sentence": i})
    return out


def plan_zooms(segments: list[dict], sents: list[dict], emphasis: list[int], jump_gap: float = 3.0,
               min_len: float = 1.2, z_alt: float = 1.08, z_emph: float = 1.15) -> list[float]:
    """Zoom factor per segment."""
    emph_ranges = [(sents[i]["start"], sents[i]["end"]) for i in emphasis if 0 <= i < len(sents)]
    zooms = []
    alt = False
    prev = None
    for s in segments:
        dur = s["timeline_end"] - s["timeline_start"]
        jump = prev is not None and s["file"] == prev["file"] and 0 <= s["in"] - prev["out"] < jump_gap
        z = 1.0
        if jump and dur >= min_len:
            alt = not alt
            z = z_alt if alt else 1.0
        else:
            alt = False
        mid = (s["timeline_start"] + s["timeline_end"]) / 2
        if any(a <= mid <= b for a, b in emph_ranges) and dur >= min_len:
            z = z_emph
        zooms.append(z)
        prev = s
    return zooms


PRAISE = {"güzel", "harika", "muhteşem", "inanılmaz", "efsane", "mükemmel", "bayıldım", "lezzetli", "nefis", "süper",
          "enfes", "şahane", "olağanüstü", "leziz", "müthiş", "kusursuz", "acayip", "çılgın", "pahalı", "ucuz", "sıcak",
          "acı", "tatlı", "taze", "yumuşak", "kocaman", "büyük", "doyurucu"}
NOT_NAMES = {"evet", "hayır", "allah", "tamam", "bence", "ama", "ve", "bu", "şu", "o", "ben", "biz", "siz", "sen",
             "hadi", "dostlar", "dostlarım", "arkadaşlar", "işte", "yani", "çünkü", "şimdi", "peki", "abi", "abla", "oh",
             "vay", "ya", "aa", "bak", "bakın", "merhaba", "selam", "teşekkürler", "teşekkür", "ok", "okay", "yes", "no",
             "himm", "hmm", "hım", "hımm", "eee", "ııı", "aaa", "off", "oha", "vallahi", "valla", "müzik"}


def tr_upper(t: str) -> str:
    return t.replace("i", "İ").replace("ı", "I").upper()


UNIT_KICKER = {"dolar": "FİYAT", "dolarlık": "FİYAT", "dolara": "FİYAT", "euro": "FİYAT", "lira": "FİYAT",
               "liralık": "FİYAT", "tl": "FİYAT", "cent": "FİYAT", "bin": "FİYAT", "milyon": "FİYAT", "saat": "SÜRE",
               "saatlik": "SÜRE", "dakika": "SÜRE", "dakikada": "SÜRE", "dakikalık": "SÜRE", "saniye": "SÜRE",
               "gün": "SÜRE", "hafta": "SÜRE", "ay": "SÜRE", "yıl": "SÜRE", "sene": "SÜRE", "gram": "MİKTAR",
               "gramlık": "MİKTAR", "kilo": "MİKTAR", "kg": "MİKTAR", "porsiyon": "MİKTAR", "tabak": "MİKTAR",
               "tane": "MİKTAR", "kişi": "KİŞİ", "kişilik": "KİŞİ", "mil": "MESAFE", "km": "MESAFE",
               "kilometre": "MESAFE", "yüzde": "ORAN", "derece": "BİLGİ"}
INTENSIFIERS = {"çok", "aşırı", "gerçekten", "cidden", "harbiden", "inanılmaz", "acayip", "baya", "bayağı", "fazla",
                "en", "efsane", "süper"}
VERDICT_ADJ = PRAISE | {"iyi", "kötü", "pahalı", "ucuz", "lezzetsiz", "tuzlu", "yağlı", "ince", "kalın", "sert",
                        "yumuşak", "hızlı", "yavaş", "kalabalık", "sakin", "temiz", "kirli", "farklı", "doyurucu"}
TITLE_STOP = {"bir", "bu", "şu", "o", "ya", "yani", "ama", "ve", "de", "da", "ki", "mi", "mı", "mu", "mü", "ben", "biz",
              "sen", "siz", "onlar", "bunlar", "şunlar", "bence", "valla", "vallahi", "işte", "şey", "şeyi", "böyle",
              "şöyle", "öyle", "hani", "gibi", "kadar", "daha", "hem", "ise", "ile", "için", "dostlar", "dostlarım",
              "arkadaşlar", "lan", "oğlum", "abi", "evet", "hayır", "tamam", "yok", "var", "olarak", "olan", "her",
              "hiç", "sonra", "önce", "civarı", "yaklaşık", "falan", "filan", "ne", "neyse", "şimdi", "veya", "yahut",
              "ila", "yada", "lazım"}
GENERIC = PRAISE | INTENSIFIERS | TITLE_STOP | {"iyi", "kötü", "oldu", "olmuş", "geldi", "var"}
# a title must not rest on a word Whisper was unsure of, nor merely narrate an action ("X YİYORUM")
MIN_TITLE_PROB = 0.6
NARRATION = {"yiyorum", "yiyoruz", "yedim", "yedik", "içiyorum", "içiyoruz", "geldik", "geldim", "gidiyoruz",
             "gidiyorum", "bakıyorum", "bakıyoruz", "deniyorum", "deniyoruz", "alıyorum", "alıyoruz"}


def good_title(text: str) -> bool:
    """A headline must say something: a number, or at least two words of which one is not a filler / lone adjective."""
    ws = [norm_word(w) for w in text.split() if norm_word(w)]
    if not ws:
        return False
    if any(re.search(r"\d", w) for w in ws):
        return True
    return len(ws) >= 2 and any(w not in GENERIC for w in ws)


def _clean(w: str) -> str:
    return w.strip('.,!?;:"\'()…')


def number_title(ws: list[dict]) -> tuple[str, str, str] | None:
    """'... 167 dolar verdik ...' -> ("167 DOLAR VERDİK", "FİYAT", trigger word)."""
    for k in range(len(ws) - 1):
        num = re.sub(r"[^\d.,]", "", ws[k]["word"]).strip(".,")
        if not num or not re.search(r"\d", num) or len(num) > 7:
            continue
        unit = norm_word(ws[k + 1]["word"])
        if unit not in UNIT_KICKER:
            continue
        parts = [num, _clean(ws[k + 1]["word"])]
        j = k + 2
        # "8 dolar 24 cent"
        if j + 1 < len(ws) and re.fullmatch(r"\d{1,2}", _clean(ws[j]["word"])) and norm_word(ws[j + 1]["word"]).startswith("cent"):
            parts += [_clean(ws[j]["word"]), "CENT"]
            j += 2
        # what the amount is about: the next real word ("17 dolarlık [bir] yer", "20 saniye haşlanıyor")
        while j < len(ws) and norm_word(ws[j]["word"]) in {"bir", "de", "da", "kadar", "civarı", "falan"}:
            j += 1
        if j < len(ws) and not ws[j - 1]["word"].rstrip().endswith((".", "?", "!")):
            nxt = _clean(ws[j]["word"])
            if len(nxt) >= 3 and nxt.isalpha() and norm_word(nxt) not in TITLE_STOP and norm_word(nxt) not in UNIT_KICKER:
                parts.append(nxt)
        return tr_upper(" ".join(parts)), UNIT_KICKER[unit], ws[k]["word"]
    return None


POSTPOSITIONS = ("konusunda", "hakkında", "açısından", "yüzünden", "sayesinde", "tarafından", "itibariyle", "olarak",
                 "rağmen", "beri", "göre", "kadar", "doğru", "karşı")


def not_a_subject(w: str) -> bool:
    """Interjections ("ooo", "vaaay", "hmm"), postpositions ("konusunda") and verb forms are not what a verdict is
    about."""
    if re.fullmatch(r"(.)\1+", w) or re.search(r"(.)\1\1", w):     # ooo, aaa, vaaay
        return True
    if w in POSTPOSITIONS or w.endswith(POSTPOSITIONS):
        return True
    return bool(re.search(r"(yor|yorum|dım|dim|dum|düm|tım|tim|mış|miş|muş|müş|acak|ecek|malı|meli)$", w))


def verdict_title(ws: list[dict]) -> tuple[str, str, str] | None:
    """'Bu sos gerçekten çok acı' -> ("SOS ÇOK ACI", "YORUM", trigger). Needs the thing it is about right before the
    intensifier - a bare 'ÇOK GÜZEL' says nothing."""
    for k in range(1, len(ws) - 1):
        if norm_word(ws[k]["word"]) not in INTENSIFIERS:
            continue
        j = k + 1
        while j < len(ws) and norm_word(ws[j]["word"]) in INTENSIFIERS:
            j += 1
        if j >= len(ws) or norm_word(ws[j]["word"]) not in VERDICT_ADJ:
            continue
        i = k - 1
        while i >= 0 and norm_word(ws[i]["word"]) in INTENSIFIERS:
            i -= 1
        if i < 0 or ws[i]["word"].rstrip().endswith((".", "?", "!", ",")):
            continue
        subj = _clean(ws[i]["word"])
        ns = norm_word(subj)
        if len(subj) < 2 or not subj.isalpha() or ns in GENERIC or not_a_subject(ns):
            continue
        return tr_upper(f"{subj} {_clean(ws[k]['word'])} {_clean(ws[j]['word'])}"), "YORUM", ws[k]["word"]
    return None


def heuristic_graphics(sents: list[dict]) -> dict:
    """Without an LLM: amounts with their unit and verdicts about a named thing become titles; rule-scored sentences
    get punch-ins. Nothing is invented."""
    from .shorts import is_junk, score_sentence

    ranked = sorted(sents, key=score_sentence, reverse=True)[:8]
    callouts = []
    for s in sents:
        ws = s["words"]
        if is_junk(s) or len(ws) > 40:
            continue
        hit = number_title(ws) or verdict_title(ws)
        if hit:
            text, kicker, trig = hit
            callouts.append({"sentence": s["i"], "trigger": trig, "text": text, "kicker": kicker})
    return {"callouts": callouts, "emphasis": [s["i"] for s in ranked]}


def _merge(parts: list[dict]) -> dict:
    out = {"callouts": [], "emphasis": [], "camfx": [], "lists": []}
    for p in parts:
        for k in out:
            out[k] += p.get(k, []) or []
    return out


def build_graphics(words_cut: list[dict], segments: list[dict], duration: float, llm=None, P: dict | None = None,
                   progress=None, budget_s: float = 900.0, call_timeout_s: float = 300.0) -> dict:
    """budget_s: total time the language model may take; a slow or stuck model never blocks the video - the
    remaining parts use the rules. progress(f, msg) is called per part (also lets the user pause/stop)."""
    import time as _time
    P = P or {"callout_gap": 25.0, "callout_per_s": 1 / 50, "z_alt": 1.08, "z_emph": 1.15}
    sents = sentences(words_cut)
    method, warning = "dil modeli", None
    rules = heuristic_graphics(sents)
    if llm is not None:
        # long videos in windows of ~80 sentences: short prompts, short answers, no timeouts
        parts, errors = [], []
        # windows of at most ~7000 characters / 110 sentences: the model is loaded with an 8k context
        starts, size = [0], 0
        for k, s_ in enumerate(sents):
            size += len(s_["text"]) + 14
            if k > starts[-1] and (size > 7000 or k - starts[-1] >= 110):
                starts.append(k)
                size = len(s_["text"]) + 14
        bounds = list(zip(starts, starts[1:] + [len(sents)]))
        llm_ranges = []
        t0 = _time.time()
        fails_in_row = 0
        old_timeout = getattr(llm, "cfg", {}).get("timeout_s") if hasattr(llm, "cfg") else None
        try:
            for j, (k, k_end) in enumerate(bounds):
                chunk = sents[k:k_end]
                span = chunk[-1]["end"] - chunk[0]["start"]
                left = budget_s - (_time.time() - t0)
                if progress:
                    progress(j / len(starts), f"Efekt ve yazı planı: bölüm {j + 1}/{len(starts)} (dil modeli)"
                             if left > 20 else f"Efekt ve yazı planı: bölüm {j + 1}/{len(starts)} (kurallarla)")
                if left <= 20:
                    errors.append(f"süre doldu ({budget_s / 60:.0f} dk)")
                    parts.append(heuristic_graphics(chunk))
                    continue
                if hasattr(llm, "cfg"):
                    llm.cfg["timeout_s"] = max(30.0, min(call_timeout_s, left))
                if fails_in_row >= 2:   # two answers in a row failed: the model is too slow / stuck today
                    errors.append("atlandı (dil modeli art arda cevap vermedi)")
                    parts.append(heuristic_graphics(chunk))
                    continue
                try:
                    got = llm_callouts(llm, chunk, span, P["callout_per_s"], P["callout_gap"])
                    # prices said aloud are the channel's signature: keep the rule-found ones too
                    got["callouts"] += [c for c in heuristic_graphics(chunk)["callouts"] if c["kicker"] == "FİYAT"
                                        and not any(g["sentence"] == c["sentence"] for g in got["callouts"])]
                    parts.append(got)
                    llm_ranges.append((chunk[0]["start"] - 0.5, chunk[-1]["end"] + 0.5))
                    fails_in_row = 0
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
                    fails_in_row += 1
                    parts.append(heuristic_graphics(chunk))
        finally:
            if hasattr(llm, "cfg"):
                if old_timeout is None:
                    llm.cfg.pop("timeout_s", None)
                else:
                    llm.cfg["timeout_s"] = old_timeout
        raw = _merge(parts)
        if errors:
            method = "dil modeli + kural" if len(errors) < len(parts) else "kural"
            warning = f"Dil modeli bazı bölümlerde cevap veremedi ({len(errors)}/{len(parts)}), oralarda kurallar kullanıldı: {errors[0][:120]}"
    else:
        llm_ranges = []
        raw, method, warning = rules, "kural", "Dil modeli yoktu: başlıklar söylenen sayı ve yorumlardan"
    callouts = resolve_callouts(raw, sents, min_gap=P["callout_gap"] * 0.8)
    zooms = plan_zooms(segments, sents, raw.get("emphasis", []), z_alt=P["z_alt"], z_emph=P["z_emph"])
    return {"callouts": callouts, "zooms": zooms, "emphasis": raw.get("emphasis", []), "method": method,
            "warning": warning, "camfx_pick": raw.get("camfx") or [], "llm_ranges": llm_ranges if llm is not None else [],
            "lists": raw.get("lists") or []}


# ------------------------------------------------------------------ camera + sentence text moves ----------------
STRONG = {"inanılmaz", "efsane", "muhteşem", "harika", "bayıldım", "mükemmel", "şok", "vay", "oha", "aşırı", "kesinlikle",
          "asla", "hayatımda", "ilk", "en", "gerçekten", "cidden", "tavsiye", "pişman", "sürpriz", "sonunda"}


ENGLISH = {"the", "a", "an", "is", "are", "you", "have", "has", "can", "why", "not", "what", "do", "does", "more",
           "on", "of", "to", "and", "it", "this", "that", "with", "for", "your", "we", "i", "yes", "no", "please",
           "thank", "thanks", "how", "much", "food", "chicken", "want", "would", "like", "okay", "ok"}


def _clean_sentence(s: dict) -> str:
    t = re.sub(r"\s+", " ", s["text"]).strip()
    return t[:1].upper() + t[1:] if t else t


VIEWER_Q = {"siz", "sizce", "sizin", "size", "sizler", "dostlar", "dostlarım", "arkadaşlar", "musunuz", "misiniz",
            "mısınız", "müsünüz", "biliyor", "tahmin", "düşünüyorsunuz", "ödedik", "öderdik"}


def meaningful(s: dict) -> bool:
    """Without the language model only sentences that clearly say something go on screen: a question to the
    viewers, an amount with its unit, a verdict about a named thing, or an exclamation with a strong word."""
    toks = [norm_word(w["word"]) for w in s["words"]]
    text = s["text"].rstrip()
    if text.endswith("?") and any(t in VIEWER_Q for t in toks):
        return True
    if number_title(s["words"]) or verdict_title(s["words"]):
        return True
    return text.endswith("!") and any(t in STRONG for t in toks)


def plan_camfx(sents: list[dict], segments: list[dict], emphasis: list[int], taken: list[tuple[float, float]],
               gap: float = 45.0, captions: bool = False, picks: list[int] | None = None,
               llm_ranges: list[tuple[float, float]] | None = None) -> list[dict]:
    """Sentence-level "camera + text" moments, the way vlog editors cut them:
      slide - the picture slides sideways, the freed side turns into a dark, blurred panel and the whole sentence
              is written there word by word as it is spoken (complete thoughts, 6-20 words);
      punch - a punch-in zoom on the speaker with the sentence as big kinetic words at the bottom on a dark band
              (short, strong sentences; skipped when burned-in captions are on, they would say the same).
    Only clean, complete sentences (ends with . ! ?, confident transcription), never over screen/b-roll/cards/titles
    (taken), at least `gap` seconds apart, in camera-only stretches. Where the language model read the talk
    (llm_ranges) only the sentences it picked are used; elsewhere only clearly meaningful ones (meaningful()).
    Returns [{t0, t1, kind, sentence, text, words}]."""
    picked = set(picks or [])
    ranges = llm_ranges or []
    from .shorts import is_junk, score_sentence

    def in_camera(a: float, b: float) -> bool:
        return any(sg.get("kind") != "montage" and sg["timeline_start"] <= a and b <= sg["timeline_end"] + 3.0
                   for sg in segments) and not any(a < y and b > x for x, y in taken)

    cands = []
    for s in sents:
        ws = s["words"]
        n = len(ws)
        dur = s["end"] - s["start"]
        chosen = s["i"] in picked      # the language model read it and found it worth showing: looser limits
        if not ((3 if chosen else 4) <= n <= (24 if chosen else 20)) or not (1.2 <= dur <= (10.0 if chosen else 8.0)) \
                or is_junk(s):
            continue
        if not chosen and not s["text"].rstrip().endswith((".", "!", "?")):
            continue
        probs = [w.get("prob") for w in ws if w.get("prob") is not None]
        if probs and sum(probs) / len(probs) < (0.68 if chosen else 0.8):
            continue
        toks = [norm_word(w["word"]) for w in ws]
        if sum(1 for t in toks if t in ENGLISH) >= 0.3 * n:
            continue   # the waiter / other people speaking English
        if sum(1 for t in toks if t in TITLE_STOP) > 0.6 * n:
            continue   # "yani şey işte böyle" - says nothing
        read_by_llm = any(a <= s["start"] <= b for a, b in ranges)
        if read_by_llm and s["i"] not in picked:
            continue
        if not read_by_llm and not meaningful(s):
            continue
        strong = sum(1 for t in toks if t in STRONG) + (1.5 if s["text"].rstrip().endswith("!") else 0)
        sc = score_sentence(s) + strong + (1.5 if s["i"] in emphasis else 0) + (0.5 if 7 <= n <= 14 else 0) \
            + (3.0 if s["i"] in picked else 0)
        kind = "punch" if (n <= 9 and (strong or s["text"].rstrip().endswith(("!", "?")))) else "slide"
        if kind == "punch" and captions:
            kind = "slide"
        t0, t1 = s["start"] - 0.25, min(s["end"] + 0.9, s["start"] + 9.0)
        if not in_camera(t0, t1):
            continue
        cands.append((sc, s, kind, t0, t1))
    out: list[dict] = []
    for sc, s, kind, t0, t1 in sorted(cands, key=lambda c: -c[0]):
        if sc < 1.5:
            break
        if any(abs(t0 - o["t0"]) < gap for o in out):
            continue
        out.append({"t0": round(t0, 3), "t1": round(t1, 3), "kind": kind, "sentence": s["i"],
                    "text": _clean_sentence(s),
                    "words": [(w["word"].strip(), round(w["t_start"], 3), round(w["t_end"], 3)) for w in s["words"]]})
    return sorted(out, key=lambda o: o["t0"])
