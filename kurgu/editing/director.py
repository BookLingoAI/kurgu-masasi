"""Edit director: decides WHAT is on screen along the edited timeline.

Events (timeline seconds of the edited video):
- screen:    the speaker looks at the screen while something meaningful happens there -> screen recording full frame,
             speaker in a small box bottom-left; back to full camera when they look at the camera again.
- broll:     an extra video/photo the user uploaded, placed where the speech talks about it (LLM or embeddings;
             never at a random spot).
- card:      full-screen graphic with a short title for list items / key points (LLM only, never invented facts).
- subscribe: generic "subscribe" reminder (no channel branding).
- sfx:       a sound from the user's library on every scene change.
Intensity (low / medium / high) scales how many of these are used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..analysis.text import norm_word
from .shorts import sentences

INTENSITY = {
    "low": {"callout_gap": 45.0, "callout_per_s": 1 / 90, "broll_gap": 40.0, "cards": 2, "subscribe": 1,
            "sfx_db": -22.0, "z_alt": 1.0, "z_emph": 1.10},
    "medium": {"callout_gap": 25.0, "callout_per_s": 1 / 50, "broll_gap": 20.0, "cards": 4, "subscribe": 1,
               "sfx_db": -17.0, "z_alt": 1.08, "z_emph": 1.15},
    "high": {"callout_gap": 12.0, "callout_per_s": 1 / 25, "broll_gap": 10.0, "cards": 7, "subscribe": 2,
             "sfx_db": -13.0, "z_alt": 1.12, "z_emph": 1.20},
}
DEFAULT_OPTIONS = {"screen_side": "right", "intensity": "medium", "captions": True, "subscribe": True,
                   "engine": "premiere", "music": True, "resolution": "1440", "montage": True,
                   "format": "yatay", "tur": "vlog",
                   "muzik_temizle": True, "yazi_stili": "karisik",
                   "capcut": True, "kamera_oyunlari": True,
                   "renk": "otomatik", "yazi_efekti": "karisik",
                   "intro_otomatik": True, "sayma_hizli": True,
                   "motor": "kurgu"}   # kurgu = our engine, karma = our cut + HyperFrames graphics, hyperframes = all in HyperFrames
# words that point at the screen ("look here", "click") - lower the bar for switching to the screen recording
DEICTIC = {"burada", "burası", "buraya", "burda", "şurada", "şuraya", "şurası", "bakın", "bakalım", "görüyorsunuz",
           "tıklıyorsunuz", "tıkladım", "tıklayın", "ekranda", "ekrana", "sayfada", "şu", "aşağıda", "yukarıda"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def options_of(settings: dict) -> dict:
    o = dict(DEFAULT_OPTIONS)
    o.update({k: v for k, v in (settings.get("options") or {}).items() if k in DEFAULT_OPTIONS})
    if o["intensity"] not in INTENSITY:
        o["intensity"] = "medium"
    return o


# ------------------------------------------------------------------ gaze -> screen spans ------------------------
def yaw_ratio(pts: np.ndarray) -> float:
    """Horizontal head turn from face landmarks: nose tip offset from the eye midpoint / eye distance.
    0 = facing the camera; negative = nose towards image left."""
    nose, le, re_ = pts[1], pts[33], pts[263]
    d = abs(re_[0] - le[0]) + 1e-6
    return float((nose[0] - (le[0] + re_[0]) / 2) / d)


def screen_spans(gaze: list[dict], segments: list[dict], words: list[dict], activity: dict, side: str,
                 thr: float = 0.16, min_len: float = 2.0, merge_gap: float = 1.2) -> tuple[list[dict], list[str]]:
    """gaze: [{t (timeline s), yaw}] samples. activity: screen file -> analyze_frames() result.
    Returns screen events on the timeline + notes."""
    notes = []
    if not gaze:
        return [], notes
    yaws = np.array([g["yaw"] for g in gaze if g["yaw"] is not None])
    if len(yaws) < 4:
        return [], ["Yüz yönü ölçülemedi: ekran kaydına geçiş yapılmadı"]
    base = float(np.median(yaws))
    sign = -1.0 if side == "right" else 1.0   # unmirrored camera: turning to your right moves the nose to image left
    dt = float(np.median(np.diff([g["t"] for g in gaze]))) if len(gaze) > 1 else 0.5
    away_pos = sum(1 for y in yaws if (y - base) > thr) * dt
    away_neg = sum(1 for y in yaws if (y - base) < -thr) * dt
    mine, other = (away_neg, away_pos) if sign < 0 else (away_pos, away_neg)
    if other > 2 * mine and other > 10:
        sign = -sign
        notes.append("Ekrana bakış yönü ayardakinin tersi çıktı (kamera aynalı olabilir); ölçülen yön kullanıldı")

    seg_of = lambda t: next((s for s in segments if s["timeline_start"] <= t < s["timeline_end"]), None)  # noqa: E731
    deictic_t = [w["t_start"] for w in words if norm_word(w["word"]) in DEICTIC]

    def active(seg, t) -> bool:
        sc = seg.get("screen")
        act = activity.get(sc["file"]) if sc else None
        if not act:
            return False
        ts = sc["in"] + (t - seg["timeline_start"])
        near = [s for s in act.get("samples", []) if abs(s["t"] - ts) <= 1.5]
        return any(s.get("changed", 0) > 0.002 or s.get("typing") or s.get("scroll") for s in near)

    flags = []
    for g in gaze:
        seg = seg_of(g["t"])
        ok = (seg is not None and seg.get("screen") is not None and g["yaw"] is not None
              and sign * (g["yaw"] - base) > thr)
        flags.append((g["t"], ok, active(seg, g["t"]) if ok else False))
    spans, cur = [], None
    for t, ok, act in flags:
        if ok:
            if cur and t - cur["t1"] <= merge_gap:
                cur["t1"] = t + dt
                cur["act"] += act
                cur["n"] += 1
            else:
                cur = {"t0": t, "t1": t + dt, "act": int(act), "n": 1}
                spans.append(cur)
    out = []
    for s in spans:
        length = s["t1"] - s["t0"]
        point = any(s["t0"] - 1.0 <= t <= s["t1"] for t in deictic_t)
        meaningful = s["act"] / s["n"] >= 0.25 or point
        if length >= min_len and meaningful:
            # never across a segment without a synced screen recording
            a, b = s["t0"], s["t1"]
            for seg in segments:
                if seg.get("screen") is None and seg["timeline_end"] > a and seg["timeline_start"] < b:
                    if seg["timeline_start"] <= a:
                        a = seg["timeline_end"]
                    else:
                        b = min(b, seg["timeline_start"])
            if b - a >= min_len:
                out.append({"type": "screen", "t0": round(max(0.0, a - 0.2), 3), "t1": round(b, 3),
                            "why": "ekrana bakıyor" + (", ekranı gösteriyor" if point else "")})
        elif length >= min_len:
            notes.append(f"{s['t0']:.0f}. sn: ekrana bakış var ama ekranda anlamlı hareket yok, kamerada kaldı")
    return out, notes


# ------------------------------------------------------------------ cursor zoom ---------------------------------
class CursorFollow:
    """Smooth zoom/pan that follows the mouse pointer of a screen recording (time constants in seconds)."""

    def __init__(self, cursor: list[dict], zoom: float = 1.4, dead_zone: float = 0.35, tau: float = 0.45):
        self.c = [c for c in cursor if c.get("reliable") and c.get("x") is not None]
        self.ts = np.array([c["t"] for c in self.c]) if self.c else np.zeros(0)
        self.zoom, self.dead, self.tau = zoom, dead_zone, tau
        self.z, self.x, self.y, self.last_t = 1.0, 0.5, 0.5, None

    def target(self, t: float) -> tuple[float, float, float]:
        if not len(self.ts):
            return 1.0, 0.5, 0.5
        i0, i1 = np.searchsorted(self.ts, t - 1.0), np.searchsorted(self.ts, t + 1.0)
        win = self.c[i0:i1]
        if len(win) < 3:
            return 1.0, self.x, self.y
        xs = np.array([c["x"] for c in win])
        ys = np.array([c["y"] for c in win])
        moving = sum(1 for c in win if c.get("moving")) / len(win)
        if moving > 0.15 and xs.std() < 0.12 and ys.std() < 0.12:
            return self.zoom, float(xs.mean()), float(ys.mean())
        return 1.0, float(xs.mean()), float(ys.mean())

    def step(self, t: float) -> tuple[float, float, float]:
        tz, tx, ty = self.target(t)
        if self.last_t is None or t < self.last_t or t - self.last_t > 1.0:
            self.z, self.x, self.y = tz, tx, ty
        else:
            k = 1 - np.exp(-(t - self.last_t) / self.tau)
            self.z += (tz - self.z) * k
            # pan only when the pointer leaves the dead zone of the current view
            half = 0.5 / max(self.z, 1.0)
            if abs(tx - self.x) > half * self.dead * 2 or tz <= 1.0:
                self.x += (tx - self.x) * k
            if abs(ty - self.y) > half * self.dead * 2 or tz <= 1.0:
                self.y += (ty - self.y) * k
        self.last_t = t
        return self.z, self.x, self.y


# ------------------------------------------------------------------ b-roll / cards / subscribe ------------------
def media_label(item: dict) -> str:
    note = (item.get("note") or "").strip()
    auto = (item.get("auto") or "").strip()
    if note and auto:
        return f"{note} ({auto})"
    if note or auto:
        return note or auto
    stem = Path(item["file"]).stem
    stem = re.sub(r"[_\-.]+", " ", stem)
    stem = re.sub(r"\b(img|vid|dsc|pxl|mov|video|stok|stock|broll|b roll)\b", " ", stem, flags=re.I)
    stem = re.sub(r"\d{3,}", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


BROLL_SCHEMA = {"type": "object", "properties": {"matches": {"type": "array", "items": {"type": "object", "properties": {
    "item": {"type": "integer"}, "sentence": {"type": "integer"}}, "required": ["item", "sentence"],
    "additionalProperties": False}}}, "required": ["matches"], "additionalProperties": False}
CARD_SCHEMA = {"type": "object", "properties": {"cards": {"type": "array", "items": {"type": "object", "properties": {
    "sentence": {"type": "integer"}, "title": {"type": "string"}, "ust": {"type": "string"}, "buyuk": {"type": "string"},
    "italik": {"type": "string"}}, "required": ["sentence", "title"]}}}, "required": ["cards"]}


def match_broll(items: list[dict], sents: list[dict], llm=None, sim=None) -> tuple[dict[int, int], str]:
    """item index -> sentence index. LLM first, embeddings second; unmatched items are left out."""
    labels = [media_label(it) for it in items]
    if llm is not None:
        try:
            lines = "\n".join(f"[{s['i']}] {s['text']}" for s in sents)
            its = "\n".join(f"<{i}> {lab or '(adı yok)'}" for i, lab in enumerate(labels))
            raw = llm.chat_json(
                "Sen bir YouTube video editörüsün. Ek görüntüleri (stok video/fotoğraf) konuşmada tam o şeyden "
                "bahsedilen cümleye yerleştirirsin. Emin değilsen o görüntüyü kullanmazsın.",
                f"Transkript cümleleri:\n{lines}\n\nEk görüntüler ÇEKİLİŞ SIRASIYLA (açıklama / içerik):\n{its}\n\n"
                "Her ek görüntü için içeriğine en uygun cümlenin numarasını ver; uygun cümle yoksa sentence=-1. "
                "Konuşmacı bir mekândan ya da olaydan bahsediyorsa o mekânın görüntülerini o anlatıma koy; mümkün "
                "olduğunca çekiliş sırasını koru. Uzun bir anlatım cümlesine en fazla 3 görüntü verebilirsin (art arda "
                "montaj olur). Sadece JSON döndür.", BROLL_SCHEMA, max_tokens=1200, temperature=0.2)
            out = {}
            for m in raw.get("matches", []):
                i, s = int(m.get("item", -1)), int(m.get("sentence", -1))
                if 0 <= i < len(items) and 0 <= s < len(sents) and list(out.values()).count(s) < 3:
                    out[i] = s
            return out, "dil modeli"
        except Exception:  # noqa: BLE001
            pass
    if sim is not None:
        out = {}
        for i, lab in enumerate(labels):
            if len(lab) < 3:
                continue
            best = max(((sim(lab, s["text"]), s["i"]) for s in sents), default=(0, -1))
            if best[0] >= 0.45 and list(out.values()).count(best[1]) < 3:
                out[i] = best[1]
        return out, "anlam benzerliği"
    return {}, "eşleştirici yok"


def condensed_lines(sents: list[dict], max_chars: int = 11000, cut: int = 70) -> str:
    """The whole talk in a prompt that fits the model's 8k context: short sentences dropped, long ones trimmed,
    and if still too long, evenly thinned out (section changes stay visible)."""
    keep = [s for s in sents if len(s["words"]) >= 4] or sents
    lines = [f"[{s['i']}] {s['text'][:cut]}" for s in keep]
    total = sum(len(x) + 1 for x in lines)
    if total > max_chars:
        step = total / max_chars
        lines = [lines[int(k * step)] for k in range(int(len(lines) / step))]
    return "\n".join(lines)


def llm_cards(llm, sents: list[dict], n: int) -> list[dict]:
    lines = condensed_lines(sents)
    raw = llm.chat_json(
        "Sen bir YouTube video editörüsün. Videonun yapısını gösteren bölüm başlıkları hazırlarsın. "
        "Yalnızca konuşmada söylenenleri kullanırsın.",
        f"Transkript:\n{lines}\n\nEn fazla {n} bölüm başlığı seç: konuşmacının yeni bir ana bölüme geçtiği cümleler "
        "(mekâna giriş, yemekler geliyor, sosları hazırlama, hesap zamanı, veda...). Her biri için: 'sentence' cümle "
        "numarası; 'title' 2-5 kelimelik Türkçe başlık; 'ust' 1-2 kelimelik küçük üst etiket (ör. 'SIRADAKİ', 'MERAK "
        "ETTİĞİMİZ', 'HESAP ZAMANI'); 'buyuk' tek kelimelik büyük kelime (ör. 'Soslar'); 'italik' 2-4 kelimelik "
        "tamamlayan ifade (ör. 'nasıl hazırlanıyor?'). Aralarında en az 3 dakika olsun. Sadece JSON döndür.",
        CARD_SCHEMA, max_tokens=700, temperature=0.3)
    return raw.get("cards", [])


def _free(t0: float, t1: float, events: list[dict], pad: float = 0.5) -> bool:
    return all(t1 + pad <= e["t0"] or t0 >= e["t1"] + pad for e in events)


def plan_events(segments: list[dict], words: list[dict], duration: float, options: dict, screen_ev: list[dict],
                broll_items: list[dict], broll_info: dict, llm=None, sim=None) -> dict:
    """broll_info: file -> {"duration": s or None (image)}. Returns events + notes (all on the edited timeline)."""
    P = INTENSITY[options["intensity"]]
    sents = sentences(words)
    events = list(screen_ev)
    notes = []
    # b-roll
    if broll_items:
        m, how = match_broll(broll_items, sents, llm, sim)
        last = -1e9
        by_sent: dict[int, list[int]] = {}
        for i, si in m.items():
            by_sent.setdefault(si, []).append(i)
        for si in sorted(by_sent, key=lambda k: sents[k]["start"]):
            group = sorted(by_sent[si])          # items are in shooting order
            s = sents[si]
            span = max(2.5, s["end"] - s["start"] + 1.0)
            each = max(1.6, min(4.0, span / len(group))) if len(group) > 1 else max(2.5, min(5.0, s["end"] - s["start"]))
            t0 = s["start"] + 0.15
            if t0 < 3.0 or t0 - last < P["broll_gap"]:
                for i in group:
                    notes.append(f"Ek görüntü '{Path(broll_items[i]['file']).name}' uygun boş yer bulamadı")
                continue
            placed = False
            for i in group:
                it = broll_items[i]
                info = broll_info.get(it["file"], {})
                is_img = Path(it["file"]).suffix.lower() in IMAGE_EXT
                d = each if is_img else min(each, max(1.2, (info.get("duration") or each) - 0.1))
                if not _free(t0, t0 + d, events, pad=0.0 if placed else 0.5):
                    notes.append(f"Ek görüntü '{Path(it['file']).name}' uygun boş yer bulamadı")
                    continue
                events.append({"type": "broll", "t0": round(t0, 3), "t1": round(t0 + d, 3), "file": it["file"],
                               "image": is_img, "sentence": s["text"][:120], "how": how,
                               "montage": len(group) > 1})
                t0 += d
                placed = True
            if placed:
                last = s["start"]
        used = {e["file"] for e in events if e["type"] == "broll"}
        for it in broll_items:
            if it["file"] not in used and not any(Path(it["file"]).name in n for n in notes):
                notes.append(f"Ek görüntü '{Path(it['file']).name}' konuşmayla eşleşmedi (açıklama yazarsanız eşleşir)")
    # full-screen cards
    if llm is not None and P["cards"]:
        try:
            for c in llm_cards(llm, sents, P["cards"]):
                i = int(c.get("sentence", -1))
                title = re.sub(r"\s+", " ", str(c.get("title", ""))).strip().strip('"')
                if not (0 <= i < len(sents)) or not (1 <= len(title.split()) <= 6):
                    continue
                # the card says only what was said (here and in the next 3 sentences), with words Whisper was sure of
                from .graphics import grounded
                said = {"words": [w for s_ in sents[i:i + 4] for w in s_["words"]]}
                if not grounded(title, said):
                    notes.append(f"Bölüm kartı atlandı (söylenenle uyuşmuyor ya da emin olunmayan kelime): '{title}'")
                    continue
                for k_ in ("buyuk", "italik", "ust"):
                    if c.get(k_) and not grounded(str(c[k_]), said):
                        c[k_] = ""
                t0 = sents[i]["start"]
                if t0 < 2.0 or not _free(t0, t0 + 3.6, events) or any(
                        e["type"] == "card" and abs(e["t0"] - t0) < 90 for e in events):
                    continue
                events.append({"type": "card", "t0": round(t0, 3), "t1": round(t0 + 3.6, 3), "text": title,
                               "ust": str(c.get("ust") or "")[:24], "buyuk": str(c.get("buyuk") or "")[:20],
                               "italik": str(c.get("italik") or "")[:40]})
        except Exception as e:  # noqa: BLE001
            notes.append(f"Tam ekran kartlar hazırlanamadı: {e}")
    elif P["cards"]:
        notes.append("Dil modeli yoktu: tam ekran başlık kartı eklenmedi")
    # subscribe / like / comment reminders exactly where the speaker asks for them (never at random places)
    if options.get("subscribe"):
        from .cta import plan_requests
        for r in plan_requests(sents, duration):
            events.append({"type": "subscribe", "t0": r["t0"], "t1": r["t1"], "items": r["items"]})
    events.sort(key=lambda e: e["t0"])
    return {"events": events, "notes": notes}


def sfx_cues(events: list[dict], library: list[str] | dict, gain_db: float) -> list[dict]:
    """One sound per scene change. library: the user's files (cycled) or the built-in set {name: file} by type."""
    if not library:
        return []
    typed = isinstance(library, dict)
    cues, k = [], 0
    for e in events:
        if e["type"] == "subscribe" and not typed:
            continue
        if typed:
            prev_joined = any(abs(o["t1"] - e["t0"]) < 0.02 for o in events if o is not e)
            next_joined = any(abs(o["t0"] - e["t1"]) < 0.02 for o in events if o is not e)
            kinds = {"screen": [("whoosh", e["t0"]), ("swish", e["t1"])],
                     "broll": ([("swish", e["t0"])] if prev_joined else [("whoosh", e["t0"])]) +
                              ([] if next_joined else [("swish", e["t1"])]),
                     }.get(e["type"], [])   # reminders and section titles bring their own sounds
            for name, t in kinds:
                cues.append({"t": round(max(0.0, t - 0.12), 3), "file": library[name],
                             "gain_db": gain_db - (6 if name == "click" else 0)})
            continue
        for t in ((e["t0"], e["t1"]) if e["type"] in ("screen", "broll") else (e["t0"],)):
            cues.append({"t": round(max(0.0, t - 0.12), 3), "file": library[k % len(library)], "gain_db": gain_db})
            k += 1
    return cues


def event_at(events: list[dict], t: float, kinds=("broll", "card", "screen")) -> dict | None:
    for e in events:
        if e["type"] in kinds and e["t0"] <= t < e["t1"]:
            return e
    return None
