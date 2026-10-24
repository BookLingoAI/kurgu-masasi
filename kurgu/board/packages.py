"""The local package catalog and the single package manager of the board.

Agents never download. When a proposal needs material (music, a sound effect, a graphic, an animation, a font) the
agent files a resource request (Kaynakİsteği: action "kaynak_iste" with tur / amac / anahtarlar); the manager looks in
the local library first. Downloading (stage 3) only ever happens from sources the user approved in
paketler/kaynaklar.json, after a license check - this module never downloads anything.

Catalog: <root>/paketler/katalog.json, one entry per file:
  id, tur, ad, gerekce, kaynak, surum, lisans, lisans_dogrulandi, ticari_kullanim, premiere_uyumlu, boyut, sha256,
  konum, onizleme, etiketler, durum (kullanilabilir | lisans_bekliyor | yasak)
A package is used in a project only when durum == "kullanilabilir" (license known and verified, not banned).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

TYPES = ("muzik", "efekt", "grafik", "animasyon", "font", "simge")
AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
PREMIERE = {".wav": "evet (ses, doğrudan içe aktarılır)", ".mp3": "evet (ses)", ".flac": "hayır (Premiere FLAC'ı desteklemiyor; "
            "önce WAV'a çevrilmeli)", ".m4a": "evet (ses)", ".aac": "evet (ses)", ".ogg": "hayır (önce WAV'a çevrilmeli)",
            ".ttf": "evet (Windows'a kurulu yazı tipi olarak; uygulama kendi motorunda doğrudan kullanır)",
            ".otf": "evet (yazı tipi)", ".png": "evet (resim)", ".svg": "hayır (Premiere SVG'yi doğrudan almaz; PNG'si var)",
            ".mov": "evet (video, saydam kanallı olabilir)", ".mp4": "evet (video)"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tags(name: str) -> list[str]:
    stem = re.sub(r"[_\-.()&,]+", " ", Path(name).stem.lower())
    return [t for t in re.findall(r"[a-zçğıöşü0-9]+", stem) if len(t) > 1][:12]


# what is already on disk and what is known about each folder (checked 22.10.2026)
LOCAL = [
    {"klasor": "assets/sfx_hazir", "tur": "efekt", "lisans": "kendi üretimimiz (kodla sentezlendi)", "dogrulandi": True,
     "ticari": True, "kaynak": "yerel: kurgu/audio sentezi", "gerekce": "uygulamanın hazır efektleri"},
    {"klasor": "assets/fonts", "tur": "font", "lisans": "SIL Open Font License 1.1", "dogrulandi": "assets/fonts/lisanslar",
     "ticari": True, "kaynak": "https://github.com/google/fonts", "gerekce": "yazı stilleri (Google Fonts)", "uzanti": {".ttf", ".otf"}},
    {"klasor": "assets/icons", "tur": "simge", "lisans": "ISC (Lucide v1.52.0)", "dogrulandi": "assets/icons/LICENSE-lucide.txt",
     "ticari": True, "kaynak": "https://unpkg.com/lucide-static@1.52.0/", "gerekce": "anlatım başlıkları simgeleri",
     "uzanti": {".png"}},
    {"klasor": "muzik", "tur": "muzik", "lisans": "bilinmiyor (kullanıcı ekledi; dosya adı 'Royalty Free' diyor ama koşulları "
     "doğrulanmadı)", "dogrulandi": False, "ticari": None, "kaynak": "kullanıcının klasörü", "gerekce": "müzik yatağı / intro"},
    {"klasor": "ses_efektleri", "tur": "efekt", "lisans": "bilinmiyor (kullanıcının kütüphanesi)", "dogrulandi": False,
     "ticari": None, "kaynak": "kullanıcının klasörü", "gerekce": "sahne geçiş efektleri"},
    {"klasor": "assets/emoji", "tur": "grafik", "lisans": "OFL (Noto Emoji)", "dogrulandi": True, "ticari": True,
     "kaynak": "https://github.com/googlefonts/noto-emoji", "gerekce": "eski emoji katmanı", "yasak": "kullanıcı emojiyi kaldırttı"},
]


def library(root: Path) -> Path:
    return Path(root) / "paketler"


def load(root: Path) -> dict:
    p = library(root) / "katalog.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"surum": 1, "paketler": []}


def save(root: Path, cat: dict) -> None:
    p = library(root) / "katalog.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cat["guncellendi"] = time.strftime("%Y-%m-%d %H:%M")
    p.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")


def index_local(root: Path) -> dict:
    """Bring every local file into the catalog once (same SHA-256 = same package, never twice)."""
    root = Path(root)
    cat = load(root)
    known = {p["sha256"]: p for p in cat["paketler"]}
    for src in LOCAL:
        folder = root / src["klasor"]
        if not folder.exists():
            continue
        lic_ok = src["dogrulandi"] if isinstance(src["dogrulandi"], bool) else (root / src["dogrulandi"]).exists()
        exts = src.get("uzanti") or (AUDIO if src["tur"] in ("muzik", "efekt") else None)
        for f in sorted(folder.rglob("*")):
            if not f.is_file() or (exts and f.suffix.lower() not in exts):
                continue
            sha = _sha(f)
            entry = known.get(sha) or {}
            entry.update({"id": entry.get("id") or f"{src['tur']}-{sha[:10]}", "tur": src["tur"], "ad": f.stem,
                          "gerekce": src["gerekce"], "kaynak": src["kaynak"], "surum": entry.get("surum") or "yerel",
                          "lisans": entry.get("lisans") if entry.get("lisans_kullanici") else src["lisans"],
                          "lisans_dogrulandi": bool(entry.get("lisans_kullanici") or lic_ok),
                          "ticari_kullanim": entry.get("ticari_kullanim", src["ticari"]),
                          "premiere_uyumlu": PREMIERE.get(f.suffix.lower(), "bilinmiyor"), "boyut": f.stat().st_size,
                          "sha256": sha, "konum": str(f.relative_to(root)), "onizleme": str(f.relative_to(root)),
                          "etiketler": _tags(f.name)})
            if src.get("yasak"):
                entry["durum"], entry["neden"] = "yasak", src["yasak"]
            elif entry["lisans_dogrulandi"]:
                entry["durum"], entry["neden"] = "kullanilabilir", None
            else:
                entry["durum"], entry["neden"] = "lisans_bekliyor", "lisansı doğrulanmadan projede kullanılmaz"
            known[sha] = entry
    cat["paketler"] = sorted(known.values(), key=lambda p: (p["tur"], p["ad"]))
    save(root, cat)
    return cat


def confirm_license(root: Path, pkg_id: str, license_text: str, commercial: bool) -> dict:
    """The user states a package's license (e.g. the music's terms): it becomes usable and stays so on re-indexing."""
    cat = load(root)
    for p in cat["paketler"]:
        if p["id"] == pkg_id and p.get("durum") != "yasak":
            p.update({"lisans": license_text, "lisans_kullanici": True, "lisans_dogrulandi": True,
                      "ticari_kullanim": commercial, "durum": "kullanilabilir", "neden": None,
                      "lisans_tarihi": time.strftime("%Y-%m-%d %H:%M")})
            save(root, cat)
            return p
    raise KeyError(pkg_id)


def request(root: Path, req: dict) -> dict:
    """A resource request (Kaynakİsteği) {ajan, tur, amac, anahtarlar, ticari_sart} -> local answer. No downloads here."""
    tur = req.get("tur")
    if tur not in TYPES:
        return {"durum": "gecersiz", "neden": f"bilinmeyen tür '{tur}'"}
    cat = load(root)
    if not cat["paketler"]:
        cat = index_local(root)
    keys = [k.lower() for k in req.get("anahtarlar") or []]
    pool = [p for p in cat["paketler"] if p["tur"] == tur]
    usable = [p for p in pool if p["durum"] == "kullanilabilir" and (p.get("ticari_kullanim") or not req.get("ticari_sart"))]
    score = lambda p: sum(any(t.startswith(k[:4]) for t in p["etiketler"]) for k in keys)  # noqa: E731
    hits = sorted(usable, key=score, reverse=True)
    if hits and (not keys or score(hits[0]) > 0 or tur in ("font", "simge")):
        return {"durum": "yerelde_var", "paketler": [{k: p[k] for k in ("id", "ad", "lisans", "konum")} for p in hits[:3]]}
    waiting = [p for p in pool if p["durum"] == "lisans_bekliyor"]
    sources = json.loads((library(root) / "kaynaklar.json").read_text(encoding="utf-8")).get("kaynaklar", []) \
        if (library(root) / "kaynaklar.json").exists() else []
    return {"durum": "eksik",
            "lisans_bekleyen": [{k: p[k] for k in ("id", "ad", "lisans")} for p in waiting[:3]],
            "sonraki_adim": ("lisansı belli olmayan yerel dosya var; lisansını onaylarsan kullanılabilir" if waiting else
                             "izinli kaynak yok: indirme için kullanıcının bir kaynağı onaylaması gerekiyor" if not sources else
                             "onaylı kaynaklarda aranabilir (aşama 3)")}
