"""YouTube text that does not read like a chatbot: 3 titles, 1 description, a Shorts title + description.

The rules are the Humanizer skill (github.com/blader/humanizer, MIT, v3.1.0 - "Signs of AI writing" from
Wikipedia) adapted to Turkish. The local LLM writes a draft from the transcript, then rewrites it against these
rules; code finds the tells that survive (dashes, "X değil Y", stock words, emoji, caps) and the facts that were
not said (numbers, names), and asks for one more rewrite. Whatever still fails is written as a note, not hidden.
"""
from __future__ import annotations

import re

from .metadata import _fmt, clean_meta
from .shorts import sentences

RULES = """İNSAN GİBİ YAZMA KURALLARI (Humanizer, Türkçeye uyarlandı):
1. "X değil, Y" / "sadece X değil, aynı zamanda Y" karşıtlığı kurma; söyleyeceğini düz söyle.
2. Tek satırlık dramatik kapanış yok ("İşte bu kadar.", "Ve evet, gerçekten öyle."). Paragraf bilgiyle biter.
3. Derin görünen boş laf yok ("asıl mesele", "işin özü", "X, Y'nin dilidir").
4. Sahneleme yok: "Peki ya ...? Cevap videoda", "Şunu söyleyeyim:", "Hazır olun" gibi girişler.
5. Her şeyi üçlü sıralama; iki ya da dört şey varsa öyle yaz.
6. Aynı kelimeyle başlayan art arda cümle kurma.
7. Uzun tire (— ya da –) kullanma; virgül, nokta ya da "ve" yeter.
8. Yapay zekâ kelimeleri yok: keşfedin, yolculuk, eşsiz, benzersiz, büyüleyici, unutulmaz, muhteşem deneyim,
   dünyasına dalın, kapsamlı rehber, derinlemesine, olmazsa olmaz, vazgeçilmez, sizi bekliyor, göz atın.
9. Abartma ve satış dili yok: sıradan bir şeyi "hayatınızı değiştirecek", "inanamayacaksınız", "şok" diye anlatma.
10. Emoji, büyük harf yığını, kalın yazı, gereksiz ünlem yok (en fazla bir).
11. Videoyu anlatan cümleyi bir kez kur ("Bu videoda ..."); sonra doğrudan konuyu anlat.
12. Konuşmacının kendi ağzından, birinci tekil şahıs, sade konuşma dili. Cümle uzunlukları farklı olsun.
13. Konuşmada geçmeyen sayı, fiyat, yer, isim, iddia EKLEME. Emin değilsen daha genel yaz. Bir fiyatı yazarken
    kaç kişi ve hangi yer için olduğunu transkriptte AYNI cümleden al; başka yerde geçen sayıları birleştirme.
14. Türkçe dil bilgisi ve yazım kusursuz olsun (ör. "tabaklar", "denedik"); yarım kalmış ifade bırakma."""

SYSTEM = ("Türkçe YouTube kanalı için başlık ve açıklama yazan bir editörsün. Yalnız transkriptte söyleneni "
          "kullanırsın; bilgi uydurmazsın. Metni yapay zekâ yazmış gibi değil, videonun sahibi yazmış gibi yazarsın.")

SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {"type": "array", "items": {"type": "object", "properties": {
            "angle": {"type": "string"}, "title": {"type": "string"}},
            "required": ["angle", "title"], "additionalProperties": False}},
        "description_intro": {"type": "string"},
        "description_body": {"type": "string"},
        "chapters": {"type": "array", "items": {"type": "object", "properties": {
            "sentence": {"type": "integer"}, "title": {"type": "string"}},
            "required": ["sentence", "title"], "additionalProperties": False}},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "shorts_title": {"type": "string"},
        "shorts_description": {"type": "string"},
    },
    "required": ["titles", "description_intro", "description_body", "chapters", "hashtags", "shorts_title",
                 "shorts_description"],
    "additionalProperties": False,
}

STOCK = ["keşfed", "keşfet", "yolculu", "eşsiz", "benzersiz", "büyüleyici", "unutulmaz", "muhteşem bir deneyim",
         "dünyasına", "kapsamlı rehber", "derinlemesine", "olmazsa olmaz", "vazgeçilmez", "sizi bekliyor", "göz atın",
         "hazır olun", "inanamayacaksınız", "hayatınızı değiştir", "şok edici", "asıl mesele", "işin özü"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
CONTRAST = re.compile(r"\b(sadece|yalnızca|yalnız)\b[^.!?]{1,60}\bdeğil\b|\bdeğil\s*,\s*(aynı zamanda|tam tersine)\b"
                      r"|\b\w+\s+değil\s*,\s*\w+(dir|dır|dur|dür|tir|tır|tur|tür)\b", re.I)
MAX_CHARS = 6500   # the transcript part of the prompt must fit the local model's 8k context


def tells(text: str) -> list[str]:
    """The AI-writing tells code can see for sure (the model judges the rest)."""
    out = []
    low = text.lower()
    if re.search(r"[—–]", text):
        out.append("uzun tire")
    if CONTRAST.search(text):
        out.append("'X değil, Y' karşıtlığı")
    for w in STOCK:
        if w in low:
            out.append(f"kalıp kelime: {w}")
    if EMOJI.search(text):
        out.append("emoji")
    if re.search(r"\*\*|__", text):
        out.append("kalın yazı")
    if len(re.findall(r"\b[A-ZÇĞİÖŞÜ]{4,}\b", text)) >= 2:
        out.append("büyük harf yığını")
    if text.count("!") > 1:
        out.append("birden çok ünlem")
    return out


def _stem(w: str) -> str:
    return w.lower().replace("ı", "i")[:4]


def unsaid(text: str, spoken: str) -> list[str]:
    """Numbers and proper names in the text that the speaker never said."""
    said_nums = set(re.findall(r"\d+(?:[.,]\d+)?", spoken))
    said = {_stem(w) for w in re.findall(r"\w+", spoken)}
    out = [f"sayı {n}" for n in re.findall(r"\d+(?:[.,]\d+)?", text) if n not in said_nums]
    # numbers used together ("4 kişi 90 dolar") must be said together: "4 kişi" was the other restaurant's bill
    toks = re.findall(r"\d+|\w+", spoken)
    where: dict[str, list[int]] = {}
    for k, tok in enumerate(toks):
        if tok.isdigit():
            where.setdefault(tok, []).append(k)
    for part in re.split(r"[.!?\n]+", text):
        nums = sorted(set(re.findall(r"\d+", part)))
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                if a in where and b in where and not any(abs(x - y) <= 25 for x in where[a] for y in where[b]):
                    out.append(f"{a} ile {b} konuşmada birlikte geçmiyor")
    for part in re.split(r"[.!?:\n]+", text):   # a capital after the first word of a sentence = a name
        for w in re.findall(r"\w+", part)[1:]:
            if re.fullmatch(r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}", w) and _stem(w) not in said:
                out.append(f"ad {w}")
    return sorted(set(out))


KEY = re.compile(r"\d|dolar|lira|euro|para|hesap|fiyat|kişi|ücret", re.I)


def _transcript(sents: list[dict]) -> str:
    """Fits the prompt: sentences with numbers/prices always go (a long video's price line was left out and the
    model took another restaurant's "4 kişi"), the rest spread evenly over the video."""
    lines = [f"[{s['i']}] ({_fmt(s['start'])}) {s['text']}" for s in sents]
    if sum(len(x) + 1 for x in lines) <= MAX_CHARS:
        return "\n".join(lines)
    keep = {k for k, s in enumerate(sents) if KEY.search(s["text"])}
    while sum(len(lines[k]) + 1 for k in keep) > MAX_CHARS * 0.5:   # very number-heavy talk: thin the key lines too
        keep = set(sorted(keep)[::2])
    budget = MAX_CHARS - sum(len(lines[k]) + 1 for k in keep)
    rest = [k for k in range(len(lines)) if k not in keep]
    step = max(1.0, sum(len(lines[k]) + 1 for k in rest) / max(budget, 1))
    keep |= {rest[int(j * step)] for j in range(int(len(rest) / step))}
    return "\n".join(lines[k] for k in sorted(keep))


def evidence(text: str, spoken: str) -> list[str]:
    """Where each number of the draft was said, word for word (for the rewrite pass)."""
    toks = spoken.split()
    out = []
    for n in sorted(set(re.findall(r"\d+", text)), key=int):
        hits = [k for k, t in enumerate(toks) if re.search(rf"(?<!\d){n}(?!\d)", t)][:3]
        if hits:
            out.append(f"{n}: " + " / ".join("'" + " ".join(toks[max(0, k - 12):k + 8]) + "'" for k in hits))
    return out


def _draft_prompt(sents, duration, shorts_hint):
    return (f"Videonun kurgulanmış transkripti (süre {_fmt(duration)}), cümle numarası ve zamanıyla:\n\n"
            f"{_transcript(sents)}\n\n{RULES}\n\nYaz (hepsi Türkçe):\n"
            "- titles: birbirinden farklı açıda 3 başlık (angle: açının kısa adı). Her biri 45-70 karakter, ana konu başa "
            "yakın, merak uyandırır ama dürüst, en fazla bir soru işareti.\n"
            "- description_intro: aramada görünen ilk 2 cümle, en fazla 200 karakter.\n"
            "- description_body: 2-3 kısa paragraf, videoda anlatılanlar, konuşmacının ağzından; sonda izleyiciye "
            "videoyla ilgili tek bir soru.\n"
            "- chapters: 4-8 bölüm; sentence = bölümün başladığı cümle numarası (ilki 0), title 2-5 kelime.\n"
            "- hashtags: 3-5 Türkçe hashtag.\n"
            f"- shorts_title: Shorts için en fazla 60 karakter başlık{(' (Shorts şu soruyla açılıyor: ' + shorts_hint + ')') if shorts_hint else ''}.\n"
            "- shorts_description: 1-2 cümle + 3 hashtag (#shorts dahil).\nSadece JSON döndür.")


def form(d: dict) -> list[str]:
    """Shape rules code can check: title lengths, the closing question, the Shorts title."""
    out = [f"{k + 1}. başlık {len(t['title'])} karakter, 45-70 olmalı (uzat ya da kısalt, bilgi uydurmadan)"
           for k, t in enumerate(d.get("titles", [])) if not 45 <= len(t["title"]) <= 70]
    if len(d.get("titles", [])) < 3:
        out.append("3 başlık olmalı")
    if not (d.get("description_body") or "").rstrip().endswith("?"):
        out.append("açıklama izleyiciye videoyla ilgili tek bir soruyla bitmeli")
    if len(d.get("shorts_title") or "") > 60:
        out.append("Shorts başlığı en fazla 60 karakter")
    return out


def _flat(d: dict) -> str:
    return "\n".join([*(t.get("title", "") for t in d.get("titles", [])), d.get("description_intro", ""),
                      d.get("description_body", ""), d.get("shorts_title", ""), d.get("shorts_description", "")])


def _rewrite_prompt(draft: dict, problems: list[str], spoken_sample: str, proof: list[str] = ()):
    import json
    return (f"Aşağıdaki YouTube metinleri taslak. Kuralları uygulayarak baştan yaz; bilgiyi koru, bilgi ekleme.\n\n"
            f"{RULES}\n\n" + (f"Kodun bulduğu sorunlar (mutlaka düzelt): {'; '.join(problems)}\n\n" if problems else "")
            + ("Taslaktaki sayıların konuşmada geçtiği yerler (sayıyı yalnız bu anlamda kullan; uymuyorsa sayıyı "
               "çıkar):\n" + "\n".join(proof) + "\n\n" if proof else "")
            + "Önce taslakta yapay zekâ gibi duran yerleri kendin bul, sonra düzelt. Başlıklar 45-70 karakter kalsın.\n\n"
            f"Transkriptten örnek (üslup için): {spoken_sample}\n\nTASLAK:\n{json.dumps(draft, ensure_ascii=False)}\n\n"
            "Aynı JSON biçiminde döndür.")


def _clean(d: dict) -> dict:
    def one(s):
        s = re.sub(r"[—–]", ",", s or "")
        s = re.sub(r"\*\*|__", "", s)
        return re.sub(r"\s+,", ",", re.sub(r"[ \t]+", " ", s)).strip().strip('"“”')
    d = dict(d)
    d["titles"] = [dict(t, title=one(t.get("title"))) for t in d.get("titles", []) if one(t.get("title"))][:3]
    for k in ("description_intro", "shorts_title", "shorts_description"):
        d[k] = one(d.get(k))
    d["description_body"] = "\n\n".join(one(p) for p in re.split(r"\n\s*\n", d.get("description_body") or "") if one(p))
    return d


def write(llm, words_cut: list[dict], duration: float, shorts_hint: str | None = None, rounds: int = 2) -> dict:
    sents = sentences(words_cut)
    spoken = " ".join(w["word"] for w in words_cut)
    sample = " ".join(s["text"] for s in sents[: min(6, len(sents))])[:600]
    # answers are ~600 tokens; a tight cap ends a looping model sooner (one call ran 15 min at 3 tok/s)
    d = _clean(llm.chat_json(SYSTEM, _draft_prompt(sents, duration, shorts_hint), SCHEMA, max_tokens=1400,
                             temperature=0.7))
    history, failed = [], None
    for _ in range(rounds):   # the Humanizer pass: always once, again only if code still sees tells
        problems = tells(_flat(d)) + form(d) + [f"söylenmemiş {x}" for x in unsaid(_flat(d), spoken)]
        history.append(problems)
        if not problems and len(history) > 1:
            break
        try:
            new = _clean(llm.chat_json(SYSTEM, _rewrite_prompt(d, problems, sample, evidence(_flat(d), spoken)),
                                       SCHEMA, max_tokens=1400, temperature=0.5))
        except Exception as e:  # noqa: BLE001 - keep the text we already have instead of losing it all
            failed = f"düzeltme turu yapılamadı ({type(e).__name__}); önceki metin kullanıldı"
            break
        if len(new.get("titles", [])) == 3 and new.get("description_body"):
            d = new
    notes = tells(_flat(d)) + form(d) + [f"söylenmemiş olabilir: {x}" for x in unsaid(_flat(d), spoken)]
    if failed:
        notes.append(failed)
    meta = clean_meta({"chapters": d.get("chapters", []), "hashtags": d.get("hashtags", [])}, sents)
    return {"titles": d["titles"], "description_intro": d["description_intro"], "description_body": d["description_body"],
            "chapters": meta["chapters"], "hashtags": meta["hashtags"], "shorts_title": d["shorts_title"],
            "shorts_description": d["shorts_description"], "notes": notes, "rounds": history,
            "method": "dil modeli + Humanizer kuralları"}


def text_file(r: dict) -> str:
    out = [f"BAŞLIK {k + 1} ({t.get('angle', '')}): {t['title']}" for k, t in enumerate(r["titles"])]
    desc = [r["description_intro"], "", r["description_body"]]
    if r.get("chapters"):
        desc += ["", "Bölümler:"] + [f"{_fmt(c['t'])} {c['title']}" for c in r["chapters"]]
    if r.get("hashtags"):
        desc += ["", " ".join(r["hashtags"])]
    out += ["", "AÇIKLAMA:", *desc, "", f"SHORTS BAŞLIK: {r['shorts_title']}", f"SHORTS AÇIKLAMA: {r['shorts_description']}"]
    if r.get("notes"):
        out += ["", "KONTROL ET (kod şunları gördü, yayınlamadan önce bak):", *(f"- {n}" for n in r["notes"])]
    return "\n".join(out).strip() + "\n"
