"""The six board members: who they are, what they may propose, and what they have learned.

Each agent argues only from its own measurements (measure.py) and its evidence sheet, and may only propose actions
from the catalogue below (validated - nothing free-form reaches the edit). Memory lives in <root>/ajanlar/<id>.json:
lessons (seeded from the user's real feedback, then added from approvals and rejections) and per-action counts of
what was proposed, agreed by the board and approved or rejected by the user.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROLES = {
    "hikaye": {
        "ad": "Hikâye ve İzleyici",
        "uzmanlik": "YouTube izlenme süresi: ilk 30 saniyede izleyiciyi yakalamak, tempo, sıkıcı ve boş yerler, bölüm yapısı.",
    },
    "grafik": {
        "ad": "Baş Editör ve Grafik",
        "uzmanlik": "Kurulu yönetir ve kararları açıklar; videonun bütününe, öncelik sırasına ve ekran yazılarına (ne yazıldığı, "
                    "okunabilirlik, sıklık, ekranda kalma süresi, yüze ve kenara taşma) bakar.",
        "ek": "Kararı diğerlerinin sesini bastırarak değil, kanıtları karşılaştırarak verirsin.",
    },
    "animasyon": {
        "ad": "Animasyon",
        "uzmanlik": "Hareket: yazıların giriş/bekleme/çıkış zamanlaması, donuk kalan anlar, kamera hareketleri, canlılık.",
    },
    "gecis": {
        "ad": "Geçiş, Ritim ve Premiere Teknik",
        "uzmanlik": "Kesimler ve geçişler: aynı kadrajda zıplayan kesimler, geçiş çeşitliliği, ritim, kesimdeki ses efekti; "
                    "önerilerin Premiere Pro zaman çizelgesinde gerçekten uygulanabilir olup olmadığı.",
        "ek": "Programda olmayan bir Premiere özelliğini varmış gibi önermezsin.",
    },
    "ses": {
        "ad": "Ses Mix-Master",
        "uzmanlik": "Ses: YouTube seviyesi (-14 LUFS, tepe ≤ -1 dBTP), konuşma/müzik/efekt dengesi, patlama ve kısık yerler.",
    },
    "renk": {
        "ad": "Eleştirel Denetçi ve Görüntü",
        "uzmanlik": "Diğerlerinin önerilerindeki zayıf kanıtı, riski ve geri dönüş maliyetini denetler; görüntüde pozlama, "
                    "patlamış ışık, çekimler arası renk/parlaklık farkı ve netliğe bakar.",
        "ek": "Kanıtı zayıf ya da riskli önerilere somut gerekçeyle itiraz edersin; gerçek bir sorun yoksa itiraz uydurmazsın.",
    },
}

# the chair: opens the meeting and announces the decisions (the vote count itself is plain code, meeting.rule)
CHAIR = "grafik"
# rules every member follows (from the user's council design)
COMMON_RULES = ("Gerçek ölçümü, çıkarımı ve tahmini birbirinden ayır; tahmini kesin sonuç gibi sunma. Bilmediğin zamanı, "
                "dosyayı ya da yeteneği uydurma. Hiçbir şeyin uygulandığını söyleme: kurul yalnızca önerir, kullanıcı onaylar. "
                "Diğerlerinin önerilerini gerektiğinde eleştir; sırf uyum için onaylama. Konuşmacının anlatımını ve anlamını koru.")

# action -> (owner agent, what it means, parameters). The owner may veto it when another agent proposes it.
ACTIONS = {
    "yazi_ekle": ("grafik", "konuşmacının o anda söylediği sözden ekrana yazı ekle", {"t": "saniye", "metin": "sözden aynen", "tur": "baslik|liste|vurgu"}),
    "yazi_kaldir": ("grafik", "o andaki yazıyı kaldır", {"t": "saniye"}),
    "yazi_suresi": ("grafik", "yazının ekranda kalma süresini değiştir", {"t": "saniye|null(hepsi)", "sure_sn": "sayı"}),
    "yazi_animasyonu": ("animasyon", "yazıların hareket biçimi", {"t": "saniye|null(hepsi)", "stil": "kelime_kelime|nefes|sirali_cikis|uzun_giris"}),
    "kamera_hareketi": ("animasyon", "kameraya hareket ver", {"t": "saniye", "tur": "punch_in|kart|kaydir"}),
    "gecis_ekle": ("gecis", "kesime geçiş ekle", {"t": "saniye", "tur": "zoom|whip|flash"}),
    "atlamayi_gizle": ("gecis", "aynı kadraj kesimini gizle", {"t": "saniye", "yontem": "punch_in|gecis"}),
    "kesim_ayari": ("hikaye", "boşluğu kes ya da sahneyi kısalt", {"t": "saniye", "islem": "boslugu_kes|kisalt", "sn": "sayı"}),
    "acilis_kancasi": ("hikaye", "videonun başına merak uyandıran bir cümleyi öne al", {"t": "öne alınacak cümlenin saniyesi"}),
    "bolum_basligi": ("hikaye", "bölüm başlığı ekle ya da değiştir", {"t": "saniye", "baslik": "sözden"}),
    "efekt_seviyesi": ("ses", "ses efektlerinin seviyesini değiştir", {"t": "saniye|null(hepsi)", "db": "sayı (+/-)"}),
    "ses_seviyesi": ("ses", "toplam ses seviyesi", {"hedef_lufs": "sayı"}),
    "muzik": ("ses", "müzik yatağı", {"islem": "ekle|kis|kaldir"}),
    "renk_esitle": ("renk", "çekimi diğerlerine eşitle", {"t": "saniye"}),
    "pozlama": ("renk", "çekimin ışığını değiştir", {"t": "saniye", "ev": "sayı"}),
    "secenek": ("*", "proje seçeneğini değiştir", {"ad": "intensity|captions|tur|yazi_stili|yazi_efekti|kamera_oyunlari|music|sayma_hizli|renk", "deger": "yeni değer"}),
    "gelistirici": ("*", "programda olmayan bir yetenek iste (kod değişikliği)", {"istek": "kısa açıklama"}),
    "kaynak_iste": ("*", "öneri için malzeme iste (müzik, efekt, grafik, animasyon, yazı tipi, simge); ajan indirmez, "
                         "paket yöneticisi önce yerelde arar", {"tur": "muzik|efekt|grafik|animasyon|font|simge",
                                                                 "amac": "neden gerekli", "anahtarlar": ["kelime"]}),
}
OPTION_NAMES = {"intensity", "captions", "tur", "yazi_stili", "yazi_efekti", "kamera_oyunlari", "music", "sayma_hizli",
                "renk", "subscribe", "montage", "intro_otomatik"}

# lessons the user taught in earlier rounds (CLAUDE.md) - the starting memory
SEED_LESSONS = {
    "hikaye": ["Söylenmeyen bilgi yazılmaz; her öneri konuşmacının kendi sözüne dayanır.",
               "Kötü kaba kurgunun üstüne efekt yığılmaz: önce kesim ve tempo."],
    "grafik": ["Emoji önerme: kullanıcı kaldırttı.", "Ekrana yalnız söylenen kelimeler yazılır (grounded).",
               "Altyazı ile aynı cümleyi ikinci kez yazma (çift yazı)."],
    "animasyon": ["Kullanıcı yazıların donuk durmasını sevmiyor: 'yazılar animasyonlu değil' dedi.",
                  "Düz masa başı çekimlerde kameranın küçülüp kenara kayması ve yazının animasyonla gelmesi isteniyor."],
    "gecis": ["Hep aynı geçiş olmasın: zoom / whip / flash sırayla."],
    "ses": ["Nefes tanıyıcı önerme: denendi, kullanıcı beğenmedi (15.10.2026).",
            "Abone/beğeni/yorum sesleri SABİT kalır, değiştirme önerme.",
            "RNNoise'u güçlendirme: robot gibi ses yapıyordu."],
    "renk": ["Cilt rötuşu önerme: kullanıcı rötuşu CapCut'ta yapıyor.",
             "S-eğrisi ve sert kontrast önerme: gölgeleri eziyordu ('zarar verme' düzeltmesi)."],
}


def memory_path(root: Path, agent: str) -> Path:
    return Path(root) / "ajanlar" / f"{agent}.json"


# memory kinds - a user's decision, a candidate heard by speech recognition, a project note and the agent's own guess
# never mix: only "kullanici" lessons are standing rules; candidates wait for an explicit button
KINDS = {"kullanici": "KULLANICI KARARI", "kullanici_sesli": "SESLİ, ONAY BEKLİYOR", "proje": "PROJE NOTU",
         "ajan": "AJAN ÇIKARIMI"}
MAX_SOFT = 24             # candidate / project / agent lessons kept per agent (oldest go first); user rules are never cut


def _norm_lesson(d: dict) -> dict:
    """Old memories ({"ders", "kaynak", "t"}) in the structured form."""
    src = str(d.get("kaynak", ""))
    kind = d.get("tur") or ("kullanici" if src.startswith("kullanıcı (CLAUDE") or src.startswith("kullanici") and
                            "sesli" not in src else "kullanici_sesli" if "sesli" in src else "ajan")
    return {"ders": d["ders"], "tur": kind, "kaynak": src or kind, "tarih": d.get("tarih") or d.get("t") or "",
            "kapsam": d.get("kapsam", "genel"), "guven": float(d.get("guven", 1.0 if kind == "kullanici" else 0.5)),
            "durum": d.get("durum", "aktif" if kind == "kullanici" else "aday" if kind == "kullanici_sesli" else "aktif")}


def load_memory(root: Path, agent: str) -> dict:
    p = memory_path(root, agent)
    mem = None
    if p.exists():
        try:
            mem = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            mem = None
    if mem is None:
        mem = {"ajan": agent, "dersler": [{"ders": d, "tur": "kullanici", "kaynak": "kullanıcı (CLAUDE.md)", "tarih": "başlangıç"}
                                         for d in SEED_LESSONS.get(agent, [])], "istatistik": {}, "toplanti": 0}
    mem["dersler"] = [_norm_lesson(d) for d in mem.get("dersler", []) if d.get("ders")]
    return mem


def save_memory(root: Path, agent: str, mem: dict) -> None:
    _prune(mem)
    p = memory_path(root, agent)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mem, ensure_ascii=False, indent=1), encoding="utf-8")


def _prune(mem: dict) -> None:
    """Keep memory small: user rules stay; the rest (candidates, notes, guesses) is capped, oldest dropped."""
    ls = mem.get("dersler", [])
    hard = [d for d in ls if d["tur"] == "kullanici"]
    soft = [d for d in ls if d["tur"] != "kullanici" and d.get("durum") != "eski"]
    mem["dersler"] = hard + soft[-MAX_SOFT:]


def confirm_lesson(mem: dict, text: str, approve: bool) -> None:
    """The user pressed a button: a decision becomes a standing rule (and a matching candidate is retired)."""
    for d in mem.get("dersler", []):
        if d["tur"] == "kullanici_sesli" and d.get("durum") == "aday" and text[:40] in d["ders"]:
            d["durum"] = "eski"
    add_lesson(mem, text, "kullanici", confidence=1.0, status="aktif")


def lessons_text(mem: dict, limit: int = 14) -> str:
    """Lessons for a prompt: the user's rules first, then the rest, each labelled with its kind."""
    ls = [d for d in mem.get("dersler", []) if d.get("durum") != "eski"]
    ls = sorted(ls, key=lambda d: (d["tur"] != "kullanici", -d.get("guven", 0)))[:limit]
    return "\n".join(f"- [{KINDS.get(d['tur'], d['tur'])}] {d['ders']}" for d in ls)


def record(mem: dict, action: str, field: str) -> None:
    """field: oneri (proposed) | kurul_kabul (board agreed) | kullanici_onay | kullanici_ret."""
    st = mem.setdefault("istatistik", {}).setdefault(action, {"oneri": 0, "kurul_kabul": 0, "kullanici_onay": 0, "kullanici_ret": 0})
    st[field] = st.get(field, 0) + 1


def add_lesson(mem: dict, lesson: str, source: str, scope: str = "genel", confidence: float | None = None,
               status: str | None = None) -> None:
    """source: kullanici | kullanici_sesli | proje | ajan (older free-text sources are mapped)."""
    if not lesson:
        return
    kind = source if source in KINDS else _norm_lesson({"ders": lesson, "kaynak": source})["tur"]
    for d in mem.setdefault("dersler", []):
        if d["ders"] == lesson:          # the same lesson again: refresh, do not duplicate
            d["tarih"] = time.strftime("%Y-%m-%d %H:%M")
            return
    mem["dersler"].append({"ders": lesson, "tur": kind, "kaynak": source, "tarih": time.strftime("%Y-%m-%d %H:%M"),
                           "kapsam": scope, "guven": confidence if confidence is not None else (1.0 if kind == "kullanici" else 0.5),
                           "durum": status or ("aday" if kind == "kullanici_sesli" else "aktif")})


def track_record(mem: dict) -> str:
    """One line per action the agent has history with - how the board and the user received it."""
    out = []
    for a, st in sorted(mem.get("istatistik", {}).items()):
        if st.get("oneri"):
            out.append(f"{a}: {st['oneri']} öneri, kurul {st.get('kurul_kabul', 0)} kabul, kullanıcı "
                       f"{st.get('kullanici_onay', 0)} onay / {st.get('kullanici_ret', 0)} ret")
    return "\n".join(out)


def validate(action: str, params: dict) -> str | None:
    """None if the action can be carried out later as given, else why not."""
    if action not in ACTIONS:
        return f"bilinmeyen eylem '{action}'"
    if not isinstance(params, dict):
        return "parametre yok"
    if action == "secenek" and params.get("ad") not in OPTION_NAMES:
        return f"bilinmeyen seçenek '{params.get('ad')}'"
    if action == "gelistirici" and not str(params.get("istek") or "").strip():
        return "istek boş"
    if action == "kaynak_iste":
        from .packages import TYPES
        if params.get("tur") not in TYPES:
            return f"bilinmeyen kaynak türü '{params.get('tur')}'"
        if not str(params.get("amac") or "").strip():
            return "kaynak isteğinin amacı yok"
    t = params.get("t")
    if t is not None and not isinstance(t, (int, float)):
        try:
            params["t"] = float(str(t).replace(",", "."))
        except ValueError:
            return f"zaman anlaşılmadı '{t}'"
    return None
