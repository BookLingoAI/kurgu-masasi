"""Content profiles: what kind of video this is, and what that changes for each board member.

The agents keep their fields; a profile adds its own checklist items, evidence emphasis and allowed/avoided material
per field, kept apart from the general rules (bilgi.py). Five profiles: yemek, gezi, araba_fiyat, telefon_fiyat, vlog.
The user picks the profile; otherwise it is guessed from the title and the transcript, and when the guess is weak the
chair asks the user before the views. Stored per project in kurul/profil.json:
  {"profil", "kaynak": "kullanici"|"tahmin", "guven", "hedef_izleyici", "amac", "platform", "kanal_tercihleri",
   "kacinilacak", "iddialar": [...]}
Prices, specifications, places and other facts may only come from the transcript or sources the user gave; every
such claim is listed with where it came from and how old it is ("iddialar"), and is shown in the minutes.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..analysis.text import norm_word

PROFILES = {
    "yemek": {
        "ad": "Yemek", "hedef_izleyici": "yemek ve restoran merak edenler, gezerken nerede yenir diye bakanlar",
        "amac": "yemeği iştah açıcı göstermek, tadını ve fiyatını dürüstçe anlatmak",
        "kelimeler": ["yemek", "lezzet", "lezzetli", "tadı", "sos", "restoran", "tabak", "tatlı", "menü", "acı", "et", "tavuk",
                      "pilav", "çorba", "garson", "sipariş", "hesap", "porsiyon", "mutfak", "sushi", "ramen", "burger", "pizza"],
        "ajan": {
            "hikaye": ["Yemeğin geldiği an ve ilk tadım tepkisi kesilmez; açılışta en iştah açıcı tabak gösterilebilir."],
            "grafik": ["Fiyat yazısı yalnız konuşmacının söylediği tutar ve para birimiyle; tarih tutanakta görünür."],
            "animasyon": ["Tabak gösterilirken yazı ve hareket yemeğin üstüne binmez."],
            "gecis": ["Gösterme anlarında ('bakın', 'geldi') yakın plan / punch-in tercih edilir; sert geçiş yemeği böler."],
            "ses": ["Mekân sesi (çatal, cızırtı) tamamen silinmez; efektler konuşmayı bastırmaz."],
            "renk": ["Yemekte doygunluk/beyaz dengesi bozulmaz; aşırı sıcak ya da soluk görüntü işaretlenir."],
        },
        "kullanilir": {"muzik": "hafif, neşeli, sözsüz", "efekt": "pop, swish, ding (fiyat)", "grafik": "fiyat bandı, yemek adı",
                       "animasyon": "yumuşak giriş, kısa vurgu"},
        "kacinilir": ["yemeğin üstüne binen büyük yazı", "agresif glitch geçişleri"],
    },
    "gezi": {
        "ad": "Gezi", "hedef_izleyici": "o yeri gezmeyi düşünenler, seyahat videosu sevenler",
        "amac": "yeri hissettirmek, rota/ulaşım/fiyat gibi işe yarar bilgiyi doğru vermek",
        "kelimeler": ["gezi", "otel", "şehir", "havalimanı", "uçak", "plaj", "müze", "tatil", "rota", "manzara", "sokak", "vize",
                      "bilet", "tren", "otobüs", "harita", "ülke", "kilometre", "deniz", "dağ"],
        "ajan": {
            "hikaye": ["Yeni bir yere geçişte kısa bir 'neredeyiz' kurulumu (mekân çekimi, yer adı) olmalı."],
            "grafik": ["Yer adı yalnız konuşmada geçtiği gibi yazılır; harita/konum bilgisi kaynaksız eklenmez."],
            "animasyon": ["Yer adı başlıkları sakin girer; manzarayı kapatmaz."],
            "gecis": ["Mekân değişimlerinde geçiş kullanılabilir; aynı mekânda sert geçiş dağıtır."],
            "ses": ["Ortam sesi ve müzik dengesi: konuşma her zaman anlaşılır kalır."],
            "renk": ["İç/dış mekân geçişlerinde parlaklık ve renk sıçraması işaretlenir."],
        },
        "kullanilir": {"muzik": "akıcı, yolculuk hissi", "efekt": "whoosh (mekân geçişi)", "grafik": "yer adı, rota bilgisi",
                       "animasyon": "yavaş kayma, harita çizgisi (kaynak varsa)"},
        "kacinilir": ["kaynaksız mesafe/fiyat/saat bilgisi"],
    },
    "araba_fiyat": {
        "ad": "Araba fiyatları", "hedef_izleyici": "araba almayı düşünenler, fiyat ve donanım karşılaştıranlar",
        "amac": "fiyatı ve özellikleri doğru, kaynaklı ve tarihli vermek; karşılaştırmayı kolaylaştırmak",
        "kelimeler": ["araba", "araç", "otomobil", "model", "motor", "beygir", "galeri", "sıfır", "ikinci", "kasko", "yakıt",
                      "benzin", "dizel", "elektrikli", "hibrit", "şanzıman", "donanım", "paket", "kredi", "taksit", "km",
                      "toyota", "honda", "ford", "bmw", "mercedes", "tesla", "hyundai", "kia", "renault", "fiat", "volkswagen"],
        "ajan": {
            "hikaye": ["Fiyatın söylendiği an kesilmez; karşılaştırma sırası izleyiciye net olmalı."],
            "grafik": ["Fiyat, model adı ve teknik değer YALNIZ transkriptten ya da kullanıcının kaynağından; para birimi, "
                       "kaynak ve tarih tutanakta görünür. Güncel fiyatı tahmin etme."],
            "animasyon": ["Fiyat/özellik tablosu okunacak kadar ekranda kalır; sayılar hızla değişmez."],
            "gecis": ["Model değişiminde net ayrım (geçiş + başlık); aynı model içinde sade kesim."],
            "ses": ["Galeri/yol sesi konuşmayı bastırmaz."],
            "renk": ["Aracın rengi bozulmaz: beyaz dengesi kayması işaretlenir."],
        },
        "kullanilir": {"muzik": "tempolu, sözsüz", "efekt": "ding (fiyat), click (özellik)", "grafik": "fiyat kartı, özellik listesi",
                       "animasyon": "sayı sayma yok, net giriş"},
        "kacinilir": ["kaynaksız fiyat", "tarihi belli olmayan kampanya bilgisi"],
        "iddia_kurali": True,
    },
    "telefon_fiyat": {
        "ad": "Telefon fiyatları", "hedef_izleyici": "telefon almayı düşünenler, fiyat/özellik karşılaştıranlar",
        "amac": "fiyatı ve özellikleri doğru, kaynaklı ve tarihli vermek",
        "kelimeler": ["telefon", "iphone", "samsung", "galaxy", "xiaomi", "pixel", "pro", "max", "ultra", "gb", "kamera",
                      "batarya", "şarj", "ekran", "işlemci", "hafıza", "depolama", "garanti", "kılıf", "model", "fiyatı"],
        "ajan": {
            "hikaye": ["Fiyatın ve en önemli özelliğin söylendiği an kesilmez."],
            "grafik": ["Fiyat ve özellik (GB, mAh, MP) YALNIZ transkriptten ya da kullanıcının kaynağından; kaynak ve tarih "
                       "tutanakta görünür. Güncel fiyatı tahmin etme."],
            "animasyon": ["Özellik karşılaştırması okunacak hızda; ürün görüntüsünü kapatmaz."],
            "gecis": ["Ürün değişiminde net ayrım; ürün yakın planında punch-in."],
            "ses": ["Kutu açılışı ve tıklama sesleri konuşmayı bastırmaz."],
            "renk": ["Ürün rengi ve ekran parlaması (patlak beyaz) işaretlenir."],
        },
        "kullanilir": {"muzik": "modern, sözsüz", "efekt": "click, swish", "grafik": "özellik kartı, fiyat bandı",
                       "animasyon": "sade, hızlı giriş"},
        "kacinilir": ["kaynaksız özellik/fiyat", "teknik terimi yanlış yazmak"],
        "iddia_kurali": True,
    },
    "vlog": {
        "ad": "Genel vlog", "hedef_izleyici": "kanalı takip edenler, günlük hayat ve sohbet sevenler",
        "amac": "samimiyeti korurken temposu iyi, sıkmayan bir video",
        "kelimeler": [],
        "ajan": {
            "hikaye": ["Samimi anlar ve espriler sırf hız için kesilmez."],
            "grafik": ["Yazı az ve anlamlı: vurgu, bölüm başlığı."],
            "animasyon": ["Donuk yazı yerine hafif hareket; abartılı efekt yok."],
            "gecis": ["Jump cut vlog üslubudur; yalnız art arda aynı kadrajda göze batarsa gizlenir."],
            "ses": ["Konuşma net; müzik konuşmanın altında."],
            "renk": ["Çekimler arası tutarlılık."],
        },
        "kullanilir": {"muzik": "hafif, sözsüz", "efekt": "whoosh, pop (seyrek)", "grafik": "bölüm başlığı, vurgu",
                       "animasyon": "yumuşak giriş/çıkış"},
        "kacinilir": [],
    },
}
ASK_BELOW = 0.55          # guesses below this confidence are put to the user


def guess(title: str, words: list[dict]) -> dict:
    """{"profil", "guven", "puanlar"} from keyword hits in the title (counted 3x) and the transcript."""
    toks = [norm_word(w["word"]) for w in words] + [norm_word(t) for t in (title or "").split()] * 3
    scores = {}
    for name, p in PROFILES.items():
        keys = [norm_word(k) for k in p["kelimeler"]]
        scores[name] = sum(1 for t in toks if t and any(t.startswith(k) for k in keys if len(k) >= 3) or t in keys)
    total = sum(scores.values())
    best = max(scores, key=scores.get)
    if total < 5:
        return {"profil": "vlog", "guven": 0.3, "puanlar": scores}
    share = scores[best] / total
    second = sorted(scores.values())[-2] if len(scores) > 1 else 0
    margin = (scores[best] - second) / max(1, scores[best])
    return {"profil": best, "guven": round(min(1.0, 0.5 * share + 0.5 * margin + 0.1), 2), "puanlar": scores}


PRICE = re.compile(r"(\d[\d.,]*)\s*(bin|milyon)?\s*(lira|tl|dolar|euro|avro|sterlin|\$|€|£)", re.I)
SPEC = re.compile(r"(\d[\d.,]*)\s*(gb|tb|mah|mp|hz|beygir|hp|km|cc|inç|watt|w)\b", re.I)


def claims(words: list[dict], recorded: str) -> list[dict]:
    """Prices and specs said in the video, with where (transcript time) and when (the recording date) they come from."""
    out = []
    for k in range(len(words)):
        chunk = " ".join(w["word"] for w in words[k:k + 4])
        for rx, kind in ((PRICE, "fiyat"), (SPEC, "ozellik")):
            m = rx.match(chunk.lower().replace("’", "'"))
            if m and not any(abs(c["t"] - words[k]["t_start"]) < 1.0 and c["tur"] == kind for c in out):
                out.append({"tur": kind, "metin": m.group(0), "t": round(words[k]["t_start"], 2),
                            "kaynak": f"transkript @{int(words[k]['t_start'] // 60)}:{words[k]['t_start'] % 60:04.1f}",
                            "tarih": recorded, "dogrulama": "konuşmacının sözü; güncelliği doğrulanmadı"})
    return out


def load(proj) -> dict:
    p = Path(proj.dir) / "kurul" / "profil.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(proj, data: dict) -> dict:
    p = Path(proj.dir) / "kurul" / "profil.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def resolve(proj) -> dict:
    """The project's profile: the user's choice if any, else a fresh guess; claims are always refreshed."""
    cur = load(proj)
    words = (proj.load("edit_plan.json") or {}).get("transcript_cut") or []
    title = (proj.settings or {}).get("title") or proj.name
    if cur.get("kaynak") != "kullanici":
        g = guess(title, words)
        cur.update({"profil": g["profil"], "kaynak": "tahmin", "guven": g["guven"], "puanlar": g["puanlar"]})
    p = PROFILES[cur["profil"]]
    created = (proj.settings or {}).get("created") or time.strftime("%Y-%m-%d")
    cur.setdefault("hedef_izleyici", p["hedef_izleyici"])
    cur.setdefault("amac", p["amac"])
    cur.setdefault("platform", "YouTube")
    cur.setdefault("kanal_tercihleri", ["Abone/beğeni/yorum sesleri sabit", "Emoji yok", "Cilt rötuşu yok (CapCut'ta)",
                                        "Söylenmeyen bilgi yazılmaz"])
    cur["kacinilacak"] = sorted(set(cur.get("kacinilacak", [])) | set(p["kacinilir"]))
    cur["iddialar"] = claims(words, f"kayıt/proje tarihi {created[:10]}") if p.get("iddia_kurali") else []
    cur["sor"] = cur["kaynak"] == "tahmin" and cur.get("guven", 0) < ASK_BELOW
    return save(proj, cur)


def set_by_user(proj, name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"bilinmeyen profil '{name}'")
    cur = load(proj)
    cur.update({"profil": name, "kaynak": "kullanici", "guven": 1.0, "sor": False})
    save(proj, cur)
    return resolve(proj)


def prompt_block(prof: dict, agent: str) -> str:
    """The profile part of an agent's prompt, kept apart from the general rules."""
    p = PROFILES[prof["profil"]]
    src = "kullanıcı seçti" if prof.get("kaynak") == "kullanici" else f"tahmin, güven {prof.get('guven', 0):.0%}"
    lines = [f"İÇERİK PROFİLİ: {p['ad']} ({src}). Hedef izleyici: {prof['hedef_izleyici']}. Amaç: {prof['amac']}."]
    lines += [f"- [PROFİL KURALI] {r}" for r in p["ajan"].get(agent, [])]
    lines.append("- Kullanılabilir: " + "; ".join(f"{k}: {v}" for k, v in p["kullanilir"].items()))
    if prof.get("kacinilacak"):
        lines.append("- Kaçınılacak: " + "; ".join(prof["kacinilacak"]))
    if p.get("iddia_kurali"):
        cl = prof.get("iddialar") or []
        lines.append("- İDDİA KURALI: fiyat/özellik yalnız aşağıdaki listeden (transkript) ya da kullanıcının kaynağından; "
                     "güncel fiyat uydurma. Listede yoksa yazma.")
        lines += [f"  · {c['metin']} ({c['kaynak']}, {c['tarih']})" for c in cl[:12]] or ["  · (videoda fiyat/özellik geçmiyor)"]
    return "\n".join(lines)
