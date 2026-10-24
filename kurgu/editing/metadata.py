"""YouTube titles, SEO description, tags, chapters and thumbnail texts from the cut transcript (local LLM)."""
from __future__ import annotations

import re

from .shorts import sentences

META_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "description_intro": {"type": "string"},
        "description_body": {"type": "string"},
        "chapters": {"type": "array", "items": {"type": "object", "properties": {
            "sentence": {"type": "integer"}, "title": {"type": "string"}},
            "required": ["sentence", "title"], "additionalProperties": False}},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "variants": {"type": "array", "items": {"type": "object", "properties": {
            "angle": {"type": "string"}, "title": {"type": "string"},
            "thumbnail_text": {"type": "string"}, "highlight": {"type": "integer"}},
            "required": ["angle", "title", "thumbnail_text", "highlight"], "additionalProperties": False}},
        "shorts_description": {"type": "string"},
    },
    "required": ["topic", "keywords", "variants", "description_intro", "description_body", "chapters", "hashtags",
                 "tags", "shorts_description"],
    "additionalProperties": False,
}

SYSTEM = ("Sen Türkçe YouTube içerikleri için çalışan deneyimli bir SEO ve paketleme uzmanısın. "
          "Yalnızca verilen transkriptte söylenenleri kullanırsın; konuşmacının söylemediği bilgi, sayı, yer, isim "
          "veya iddia UYDURMAZSIN. Tıklama tuzağı yalan içermez; merak uyandırır ama dürüsttür.")


def _fmt(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def build_prompt(sents: list[dict], duration: float, shorts_title: str | None) -> str:
    lines = "\n".join(f"[{s['i']}] ({_fmt(s['start'])}) {s['text']}" for s in sents)
    return (
        f"Videonun kurgulanmış hâlinin transkripti (süre {_fmt(duration)}), cümle numaraları ve zamanlarıyla:\n\n{lines}\n\n"
        "Üret (hepsi Türkçe):\n"
        "- topic: videonun ana konusu, 3-6 kelime.\n"
        "- keywords: insanların YouTube'da arayacağı 8-12 anahtar kelime/öbek (videoda gerçekten geçen konulardan).\n"
        "- variants: YouTube A/B testi için birbirinden FARKLI açılarda 3 paket (A, B, C). Her pakette:\n"
        "  'angle' açının kısa adı (ör. 'merak', 'kişisel hikâye', 'uyarı/fayda');\n"
        "  'title' 45-70 karakter başlık, ana anahtar kelime başa yakın, merak uyandırıcı ama dürüst, emoji ve büyük harf "
        "yığını yok, en fazla bir soru işareti;\n"
        "  'thumbnail_text' bu başlıkla AYNI fikri taşıyan ama onu tekrar etmeyen 2-4 kelimelik kapak yazısı;\n"
        "  'highlight' kapak yazısında sarı vurgulanacak kelimenin sırası (0'dan).\n"
        "- description_intro: açıklamanın ilk 2 cümlesi (aramada görünen kısım), ana anahtar kelimeleri doğal kullan, "
        "en fazla 200 karakter.\n"
        "- description_body: 2-3 kısa paragraf, videoda anlatılanları özetle, birinci tekil şahıs (konuşmacının ağzından), "
        "izleyiciyi yoruma davet eden bir soru ile bitir.\n"
        "- chapters: videonun bölümleri için 4-8 madde; 'sentence' bölümün başladığı cümle numarası, ilk madde 0 numaralı "
        "cümle olmalı; 'title' 2-5 kelimelik bölüm adı.\n"
        "- hashtags: 3-5 hashtag (# ile, boşluksuz, Türkçe).\n"
        "- tags: 10-15 etiket (toplam 450 karakteri geçmesin).\n"
        f"- shorts_description: {'Shorts başlığı: ' + shorts_title + '. ' if shorts_title else ''}"
        "Shorts için 1-2 cümlelik açıklama + 3 hashtag (#shorts dahil).\n"
        "Sadece JSON döndür."
    )


def clean_meta(meta: dict, sents: list[dict]) -> dict:
    variants = []
    for k, v in enumerate(meta.get("variants", [])[:3]):
        title = re.sub(r"\s+", " ", v.get("title", "")).strip().strip('"')
        words = v.get("thumbnail_text", "").split()[:5]
        if title and words:
            variants.append({"label": "ABC"[k], "angle": v.get("angle", "").strip(), "title": title,
                             "thumbnail_text": " ".join(words),
                             "highlight": max(0, min(len(words) - 1, int(v.get("highlight", 0))))})
    meta = dict(meta, variants=variants)
    if variants:
        meta["titles"] = [v["title"] for v in variants]
        meta["thumbnail_texts"] = [{"text": v["thumbnail_text"], "highlight": v["highlight"]} for v in variants]
    titles = [re.sub(r"\s+", " ", t).strip().strip('"') for t in meta.get("titles", []) if t.strip()][:3]
    tags, total = [], 0
    for t in meta.get("tags", []):
        t = t.strip().strip("#")
        if t and total + len(t) + 1 <= 480 and t.lower() not in (x.lower() for x in tags):
            tags.append(t)
            total += len(t) + 1
    hashtags = []
    for h in meta.get("hashtags", [])[:5]:
        h = "#" + re.sub(r"[\s#]+", "", h)
        if len(h) > 1:
            hashtags.append(h)
    # chapters: real times from sentence numbers; YouTube needs 0:00 first, >= 3 chapters, >= 10 s apart
    ch = []
    for c in sorted(meta.get("chapters", []), key=lambda c: c.get("sentence", 0)):
        i = max(0, min(len(sents) - 1, int(c.get("sentence", 0))))
        t = 0.0 if not ch else sents[i]["start"]
        if ch and t - ch[-1]["t"] < 10:
            continue
        ch.append({"t": round(t, 1), "title": c.get("title", "").strip()})
    if ch:
        ch[0]["t"] = 0.0
    if len(ch) < 3:
        ch = []
    thumbs = []
    for x in meta.get("thumbnail_texts", [])[:3]:
        words = x.get("text", "").split()
        if words:
            thumbs.append({"text": " ".join(words[:5]), "highlight": max(0, min(len(words) - 1, int(x.get("highlight", 0))))})
    return dict(meta, titles=titles, tags=tags, hashtags=hashtags, chapters=ch, thumbnail_texts=thumbs)


def description_text(meta: dict) -> str:
    parts = [meta.get("description_intro", "").strip(), "", meta.get("description_body", "").strip()]
    if meta.get("chapters"):
        parts += ["", "Bölümler:"] + [f"{_fmt(c['t'])} {c['title']}" for c in meta["chapters"]]
    if meta.get("hashtags"):
        parts += ["", " ".join(meta["hashtags"])]
    return "\n".join(parts).strip() + "\n"


def generate(llm, words_cut: list[dict], duration: float, shorts_title: str | None = None) -> dict:
    sents = sentences(words_cut)
    raw = llm.chat_json(SYSTEM, build_prompt(sents, duration, shorts_title), META_SCHEMA, max_tokens=2500, temperature=0.7)
    meta = clean_meta(raw, sents)
    meta["description"] = description_text(meta)
    meta["method"] = "dil modeli"
    return meta


def fallback(words_cut: list[dict], duration: float) -> dict:
    """No LLM: honest, plain output built from the most frequent content words. Marked as weak in the UI."""
    from collections import Counter

    from ..analysis.text import norm_word

    stop = set("ve bir bu da de ki ne ben sen o biz siz onlar çok için ile ama gibi diye yani işte şey daha en mi "
               "değil var yok olan olarak kadar sonra şimdi zaman burada orada benim bizim sizin onun her hiç bile".split())
    c = Counter(w for w in (norm_word(x["word"]) for x in words_cut) if len(w) > 3 and w not in stop)
    top = [w for w, _ in c.most_common(10)]
    kw = " ".join(top[:3]).title()
    return {"method": "kural (dil modeli yoktu - zayıf)", "topic": kw, "keywords": top,
            "titles": [f"{kw} Hakkında Bilmeniz Gerekenler", f"{kw}: Kendi Deneyimim", f"{kw} Gerçekten Nasıl?"],
            "description": f"Bu videoda {', '.join(top[:5])} hakkında konuşuyorum.\n", "tags": top, "hashtags": [],
            "chapters": [], "thumbnail_texts": [{"text": " ".join(top[:2]), "highlight": 0}] * 3,
            "variants": [{"label": l, "angle": "kural", "title": t, "thumbnail_text": " ".join(top[:2]), "highlight": 0}
                         for l, t in zip("ABC", [f"{kw} Hakkında Bilmeniz Gerekenler", f"{kw}: Kendi Deneyimim",
                                                 f"{kw} Gerçekten Nasıl?"])],
            "shorts_description": "#shorts"}
