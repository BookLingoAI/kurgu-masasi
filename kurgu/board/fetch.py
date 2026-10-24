"""Verified downloads for the package manager - only from sources the user approved (paketler/kaynaklar.json).

Approved by the user on 22.10.2026, downloads automatic for free + commercially usable items:
  mixkit        sound effects under the Mixkit *Free* License only (commercial use, no attribution); an item marked
                Restricted (non-commercial only) is never taken. Full-quality WAV from assets.mixkit.co.
  kenney        CC0 packs (sound / graphics), the zip behind "continue without donating".
  google_fonts  OFL fonts from github.com/google/fonts, with the license text.
  lucide        ISC icons from unpkg (pinned version), rendered to PNG by icons.py.
Listed only, never downloaded or installed here: Mixkit Premiere templates (each zip must be checked for its
Premiere version first), Adobe Stock templates (they need the user's Adobe sign-in inside Premiere) and Creative Cloud
Marketplace plugins (installing one always needs the user's yes).
Every download lands in paketler/_gecici first, is checked (file signature, size, SHA-256, zip integrity), then moved
to paketler/<tur>/<kaynak>-<id>/ with a kaynak.json (link, license, license link, date, hash).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from . import packages as PK

UA = {"User-Agent": "Mozilla/5.0 (KurguMasasi paket yoneticisi)"}
APPROVED = {
    "mixkit": {"onay": True, "tur": ["efekt"], "lisans": "Mixkit Sound Effects Free License",
               "lisans_url": "https://mixkit.co/license/#sfxFree", "ticari": True,
               "not": "yalnız Free License; Restricted (ticari olmayan) içerik alınmaz"},
    "kenney": {"onay": True, "tur": ["efekt", "grafik"], "lisans": "Creative Commons CC0", "ticari": True,
               "lisans_url": "https://creativecommons.org/publicdomain/zero/1.0/"},
    "google_fonts": {"onay": True, "tur": ["font"], "lisans": "SIL Open Font License 1.1", "ticari": True,
                     "lisans_url": "https://openfontlicense.org"},
    "lucide": {"onay": True, "tur": ["simge"], "lisans": "ISC", "ticari": True, "lisans_url": "https://lucide.dev/license",
               "surum": "1.52.0"},
}
LIST_ONLY = {
    "mixkit_sablon": "Mixkit Premiere şablonları: her zip'in Premiere sürümü (mogrt) önce kontrol edilmeli; şimdilik yalnız listelenir.",
    "adobe_stock": "Adobe Stock ücretsiz hareketli grafik şablonları Premiere'in Temel Grafikler panelinden, senin Adobe "
                   "oturumunla eklenir; uygulama oturum açamaz, yalnız diske gelen .mogrt'leri kataloğa alır.",
    "cc_marketplace": "Creative Cloud Marketplace eklentileri yalnız listelenir; kurulum her zaman senin onayınla ve Adobe'nin "
                      "kendi aracıyla yapılır.",
}
MAGIC = {".wav": (b"RIFF", 8, b"WAVE"), ".zip": (b"PK\x03\x04", None, None), ".ttf": (b"\x00\x01\x00\x00", None, None),
         ".svg": (b"<", None, None)}


def policy(root: Path) -> dict:
    """paketler/kaynaklar.json - written once with the user's decision, then only the user changes it."""
    p = PK.library(root) / "kaynaklar.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"karar_tarihi": time.strftime("%Y-%m-%d %H:%M"), "otomatik": True,
                                 "kaynaklar": APPROVED, "yalniz_listele": LIST_ONLY}, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def _get(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def verify(data: bytes, ext: str, min_size: int = 64) -> str | None:
    """None if the bytes really are that kind of file, else why not."""
    if len(data) < min_size:
        return f"çok küçük ({len(data)} bayt)"
    head, off, tag = MAGIC.get(ext, (None, None, None))
    if head and not data.startswith(head):
        return f"dosya imzası {ext} değil"
    if tag and data[off:off + len(tag)] != tag:
        return f"dosya imzası {ext} değil"
    if ext == ".zip":
        try:
            bad = zipfile.ZipFile(io.BytesIO(data)).testzip()
        except zipfile.BadZipFile as e:
            return f"bozuk zip: {e}"
        if bad:
            return f"zip içinde bozuk dosya: {bad}"
    return None


def _store(root: Path, tur: str, src: str, item_id: str, files: dict[str, bytes], meta: dict) -> list[dict]:
    """Temp first, verified, then into the library and the catalog. Same SHA-256 already there: not stored twice."""
    lib = PK.library(root)
    tmp = lib / "_gecici" / f"{src}-{item_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    cat = PK.load(root)
    have = {p["sha256"] for p in cat["paketler"]}
    dest = lib / tur / f"{src}-{item_id}"
    added = []
    for name, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        if sha in have:
            continue
        (tmp / name).write_bytes(data)
        if hashlib.sha256((tmp / name).read_bytes()).hexdigest() != sha:      # what landed on disk is what we checked
            raise IOError(f"{name}: diske yazılan dosya doğrulanamadı")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / name), str(dest / name))
        rel = str((dest / name).relative_to(root))
        entry = {"id": f"{tur}-{sha[:10]}", "tur": tur, "ad": Path(name).stem, "gerekce": meta.get("gerekce", ""),
                 "kaynak": meta["url"], "surum": meta.get("surum", "indirildi " + time.strftime("%Y-%m-%d")),
                 "lisans": meta["lisans"], "lisans_url": meta.get("lisans_url"), "lisans_dogrulandi": True,
                 "ticari_kullanim": meta["ticari"], "premiere_uyumlu": PK.PREMIERE.get(Path(name).suffix.lower(), "bilinmiyor"),
                 "boyut": len(data), "sha256": sha, "konum": rel, "onizleme": rel, "etiketler": PK._tags(meta.get("baslik", name)),
                 "durum": "kullanilabilir", "neden": None, "indirme_tarihi": time.strftime("%Y-%m-%d %H:%M")}
        cat["paketler"].append(entry)
        have.add(sha)
        added.append(entry)
    if added:
        (dest / "kaynak.json").write_text(json.dumps(dict(meta, dosyalar=[a["konum"] for a in added],
                                                          tarih=time.strftime("%Y-%m-%d %H:%M")), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        PK.save(root, cat)
    shutil.rmtree(tmp, ignore_errors=True)
    return added


# ---- sources ----------------------------------------------------------------------------------------------------
def mixkit_search(keyword: str, limit: int = 3, get=_get) -> list[dict]:
    """Free sound effects on Mixkit's category page for `keyword`; the page must carry the Free-license mark."""
    slug = re.sub(r"[^a-z0-9]+", "-", keyword.lower()).strip("-")
    html = get(f"https://mixkit.co/free-sound-effects/{slug}/").decode("utf-8", "replace")
    if "sfxFree" not in html or re.search(r"Restricted", html, re.I):
        return []                                    # no Free-license mark, or something restricted on the page: skip
    out = []
    for m in re.finditer(r'data-audio-player-preview-url-value="https://assets\.mixkit\.co/active_storage/sfx/(\d+)/\1-preview\.mp3"', html):
        sid = m.group(1)
        if any(o["id"] == sid for o in out):
            continue
        near = html[max(0, m.start() - 1500): m.start()]
        title = re.findall(r'<h2[^>]*>\s*([^<]{2,80})\s*</h2>', near)
        out.append({"id": sid, "baslik": (title[-1].strip() if title else keyword),
                    "wav": f"https://assets.mixkit.co/active_storage/sfx/{sid}/{sid}.wav",
                    "sayfa": f"https://mixkit.co/free-sound-effects/{slug}/"})
        if len(out) >= limit:
            break
    return out


def fetch_mixkit(root: Path, keyword: str, why: str, limit: int = 2, get=_get) -> list[dict]:
    src = policy(root)["kaynaklar"]["mixkit"]
    added = []
    for it in mixkit_search(keyword, limit, get):
        data = get(it["wav"])
        bad = verify(data, ".wav", 1024)
        if bad:
            continue
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", it["baslik"]).strip("_")[:40] or it["id"]
        added += _store(root, "efekt", "mixkit", it["id"], {f"{name}_{it['id']}.wav": data},
                        {"url": it["wav"], "sayfa": it["sayfa"], "baslik": f"{it['baslik']} {keyword}", "gerekce": why,
                         "lisans": src["lisans"], "lisans_url": src["lisans_url"], "ticari": True})
    return added


def fetch_kenney(root: Path, pack: str, why: str, get=_get) -> list[dict]:
    """A Kenney asset pack: the page must say CC0; audio and images from the zip go to the library."""
    page = f"https://kenney.nl/assets/{pack}"
    html = get(page).decode("utf-8", "replace")
    if "CC0" not in html:
        raise ValueError(f"{pack}: sayfada CC0 lisansı görünmüyor; indirilmedi")
    m = re.search(r'https://kenney\.nl/media/pages/assets/[^"\']+\.zip', html)
    if not m:
        raise ValueError(f"{pack}: indirme bağlantısı bulunamadı")
    data = get(m.group(0), timeout=180)
    bad = verify(data, ".zip")
    if bad:
        raise ValueError(f"{pack}: {bad}")
    z = zipfile.ZipFile(io.BytesIO(data))
    files = {}
    for n in z.namelist():
        ext = Path(n).suffix.lower()
        if ext in (".wav", ".ogg", ".png") and not n.endswith("/"):
            files[Path(n).name] = z.read(n)
    tur = "efekt" if any(Path(n).suffix in (".wav", ".ogg") for n in files) else "grafik"
    lic = z.read(next(n for n in z.namelist() if "license" in n.lower())) if any("license" in n.lower() for n in z.namelist()) else b""
    added = _store(root, tur, "kenney", pack, files, {"url": m.group(0), "sayfa": page, "baslik": pack.replace("-", " "),
                                                      "gerekce": why, "lisans": "Creative Commons CC0", "ticari": True,
                                                      "lisans_url": APPROVED["kenney"]["lisans_url"],
                                                      "lisans_metni": lic.decode("utf-8", "replace")[:400]})
    return added


def fetch_google_font(root: Path, family: str, why: str, get=_get) -> list[dict]:
    """An OFL family from github.com/google/fonts (the folder's OFL.txt must exist)."""
    slug = re.sub(r"[^a-z0-9]", "", family.lower())
    api = f"https://api.github.com/repos/google/fonts/contents/ofl/{slug}"
    items = json.loads(get(api).decode("utf-8"))
    names = {i["name"]: i["download_url"] for i in items if i.get("download_url")}
    if "OFL.txt" not in names:
        raise ValueError(f"{family}: OFL.txt yok; indirilmedi")
    files = {}
    for n, u in names.items():
        if n.endswith(".ttf"):
            data = get(u)
            if not verify(data, ".ttf"):
                files[n] = data
    files["OFL.txt"] = get(names["OFL.txt"])
    return _store(root, "font", "googlefonts", slug, files, {"url": api, "baslik": family, "gerekce": why,
                                                             "lisans": APPROVED["google_fonts"]["lisans"], "ticari": True,
                                                             "lisans_url": APPROVED["google_fonts"]["lisans_url"]})


def fetch_lucide(root: Path, names: list[str], why: str, get=_get) -> list[dict]:
    v = APPROVED["lucide"]["surum"]
    files = {}
    for n in names:
        data = get(f"https://unpkg.com/lucide-static@{v}/icons/{n}.svg")
        if not verify(data, ".svg", 40):
            files[f"{n}.svg"] = data
    return _store(root, "simge", "lucide", v, files, {"url": f"https://unpkg.com/lucide-static@{v}/", "baslik": " ".join(names),
                                                      "gerekce": why, "lisans": "ISC", "ticari": True, "surum": v,
                                                      "lisans_url": APPROVED["lucide"]["lisans_url"]})


def satisfy(root: Path, req: dict, get=_get) -> dict:
    """A resource request the local library could not answer: try the approved sources (automatic when allowed)."""
    pol = policy(root)
    if not pol.get("otomatik"):
        return {"durum": "onay_gerekli", "neden": "otomatik indirme kapalı"}
    tur, why = req.get("tur"), str(req.get("amac") or "")
    keys = [k for k in (req.get("anahtarlar") or []) if k] or [why.split()[0] if why else "whoosh"]
    added = []
    try:
        if tur == "efekt" and pol["kaynaklar"].get("mixkit", {}).get("onay"):
            for k in keys[:2]:
                added += fetch_mixkit(root, k, why, get=get)
        elif tur == "font" and pol["kaynaklar"].get("google_fonts", {}).get("onay"):
            added += fetch_google_font(root, keys[0], why, get=get)
        elif tur == "simge" and pol["kaynaklar"].get("lucide", {}).get("onay"):
            added += fetch_lucide(root, keys[:5], why, get=get)
        elif tur == "muzik":
            return {"durum": "eksik", "neden": "müzik için onaylı kaynak yok (Mixkit müziği ayrı lisanslı, onay verilmedi)"}
        else:
            return {"durum": "eksik", "neden": f"'{tur}' için otomatik kaynak yok; " + LIST_ONLY["mixkit_sablon"]}
    except Exception as e:  # noqa: BLE001 - the meeting must not fail because of the network
        return {"durum": "hata", "neden": f"{type(e).__name__}: {str(e)[:160]}"}
    return {"durum": "indirildi" if added else "bulunamadi",
            "paketler": [{k: a[k] for k in ("id", "ad", "lisans", "konum", "kaynak", "sha256")} for a in added]}
