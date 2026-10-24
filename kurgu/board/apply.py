"""Approved board proposals -> preview, apply, verify, undo. Nothing happens without the user's button.

Only actions with a safe, checkable path are applied; everything else is reported as "not applicable yet":
  secenek - one project option in proje.json (validated name and value). The current video is not touched; the
            change shows in the next "Video" job. Premiere is not written to.
Every apply keeps a copy of the file it changes (kurul/geri_al/<time>_<id>/), reads the result back to verify it,
and is written to kurul/islemler.jsonl and the decision log. Undo puts the copy back and verifies again.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import agents as A

BOOL = {"captions", "kamera_oyunlari", "music", "sayma_hizli", "subscribe", "montage", "intro_otomatik"}
CHOICES = {"intensity": {"low", "medium", "high"}, "tur": {"vlog", "hikaye", "ekran", "anlatim"},
           "yazi_stili": {"karisik", "bant", "fiyat", "kalem", "daktilo"},
           "yazi_efekti": {"karisik", "karaoke", "kutu", "zipla"},
           "renk": {"otomatik", "sari", "turkuaz", "mercan", "limon", "mor", "turuncu", "pembe"}}


def _value(name: str, v):
    if name in BOOL:
        s = str(v).strip().lower()
        if s in ("true", "1", "evet", "açık", "acik", "on"):
            return True
        if s in ("false", "0", "hayır", "hayir", "kapalı", "kapali", "off"):
            return False
        raise ValueError(f"'{name}' için açık/kapalı bekleniyordu, '{v}' geldi")
    if name in CHOICES:
        s = str(v).strip().lower()
        if s not in CHOICES[name]:
            raise ValueError(f"'{name}' için geçerli değerler: {', '.join(sorted(CHOICES[name]))}")
        return s
    raise ValueError(f"'{name}' seçeneği uygulanabilir listede değil")


def preview(proj, finding: dict) -> dict:
    """What would change, without changing anything."""
    act, par = finding.get("eylem"), finding.get("parametre") or {}
    if act != "secenek":
        return {"uygulanabilir": False, "neden": f"'{act}' için güvenli uygulama yolu henüz yok; öneri olarak kalıyor."}
    name = par.get("ad")
    if name not in A.OPTION_NAMES:
        return {"uygulanabilir": False, "neden": f"bilinmeyen seçenek '{name}'"}
    try:
        new = _value(name, par.get("deger"))
    except ValueError as e:
        return {"uygulanabilir": False, "neden": str(e)}
    cur = (proj.settings.get("options") or {}).get(name)
    return {"uygulanabilir": cur != new, "neden": None if cur != new else "zaten bu değerde",
            "degisiklik": [{"dosya": "proje.json", "alan": f"options.{name}", "once": cur, "sonra": new}],
            "not": "Şimdiki videoya dokunulmaz; değişiklik bir sonraki 'Video' işinde görünür."}


def _log(proj, rec: dict) -> None:
    rec = dict(rec, t=time.strftime("%Y-%m-%d %H:%M:%S"))
    kdir = Path(proj.dir) / "kurul"
    kdir.mkdir(parents=True, exist_ok=True)
    with (kdir / "islemler.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if getattr(proj, "log", None):
        proj.log.write("kurul", rec["islem"], rec.get("ozet", ""), bulgu=rec.get("bulgu"))


def apply(proj, finding: dict) -> dict:
    pv = preview(proj, finding)
    if not pv["uygulanabilir"]:
        return {"tamam": False, "neden": pv["neden"]}
    src = Path(proj.dir) / "proje.json"
    token = f"{time.strftime('%Y%m%d-%H%M%S')}_{finding['id']}"
    backup = Path(proj.dir) / "kurul" / "geri_al" / token
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup / "proje.json")
    ch = pv["degisiklik"][0]
    name = ch["alan"].split(".", 1)[1]
    st = json.loads(src.read_text(encoding="utf-8"))
    st.setdefault("options", {})[name] = ch["sonra"]
    src.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    back = json.loads(src.read_text(encoding="utf-8")).get("options", {}).get(name)       # verify by reading it back
    ok = back == ch["sonra"]
    if not ok:
        shutil.copy2(backup / "proje.json", src)
    _log(proj, {"islem": "uygula" if ok else "uygulanamadi", "bulgu": finding["id"], "geri_al": token,
                "ozet": f"{ch['alan']}: {ch['once']} -> {ch['sonra']}" + ("" if ok else " (doğrulanamadı, geri alındı)")})
    return {"tamam": ok, "geri_al": token if ok else None, "degisiklik": pv["degisiklik"], "not": pv["not"]}


def undo(proj, token: str) -> dict:
    backup = Path(proj.dir) / "kurul" / "geri_al" / Path(token).name / "proje.json"
    if not backup.exists():
        return {"tamam": False, "neden": "yedek bulunamadı"}
    src = Path(proj.dir) / "proje.json"
    shutil.copy2(backup, src)
    ok = src.read_bytes() == backup.read_bytes()
    _log(proj, {"islem": "geri_al", "geri_al": token, "ozet": "proje.json yedekten geri yüklendi" + ("" if ok else " (DOĞRULANAMADI)")})
    return {"tamam": ok}


def decide(proj, finding_id: str, approve: bool) -> dict:
    """The user's explicit verdict on a finding (a button, not speech): stored in the minutes and the agent's memory."""
    kdir = Path(proj.dir) / "kurul"
    p = kdir / "tutanak.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    f = next((f for f in m["bulgular"] if f["id"] == finding_id), None)
    if not f:
        return {"tamam": False, "neden": "bulgu yok"}
    f["kullanici"] = "onay" if approve else "ret"
    p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    root = Path(proj.cfg.root)
    mem = A.load_memory(root, f["ajan"])
    A.record(mem, f["eylem"] or "?", "kullanici_onay" if approve else "kullanici_ret")
    A.confirm_lesson(mem, f"Kullanıcı '{f['baslik']}' önerisini {'ONAYLADI' if approve else 'REDDETTİ'} "
                          f"(eylem {f['eylem']}, {m.get('proje')}, {m.get('tarih')}).", approve)
    A.save_memory(root, f["ajan"], mem)
    _log(proj, {"islem": "onay" if approve else "ret", "bulgu": finding_id, "ozet": f["baslik"]})
    return {"tamam": True}
