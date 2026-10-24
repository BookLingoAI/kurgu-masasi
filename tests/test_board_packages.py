"""Local package catalog and the single package manager (kurgu/board/packages.py) - no downloads anywhere."""
import json

from kurgu.board import agents as A
from kurgu.board import packages as PK


def _root(tmp_path):
    (tmp_path / "assets" / "sfx_hazir").mkdir(parents=True)
    (tmp_path / "assets" / "sfx_hazir" / "whoosh.wav").write_bytes(b"RIFF....WAVEfmt whoosh")
    (tmp_path / "assets" / "sfx_hazir" / "ding.wav").write_bytes(b"RIFF....WAVEfmt ding")
    (tmp_path / "muzik").mkdir()
    (tmp_path / "muzik" / "Samurai Sake (Royalty Free).mp3").write_bytes(b"ID3 music")
    (tmp_path / "assets" / "emoji").mkdir()
    (tmp_path / "assets" / "emoji" / "fire.png").write_bytes(b"\x89PNG emoji")
    (tmp_path / "assets" / "fonts" / "lisanslar").mkdir(parents=True)
    (tmp_path / "assets" / "fonts" / "Inter.ttf").write_bytes(b"font")
    (tmp_path / "assets" / "fonts" / "lisanslar" / "inter_OFL.txt").write_text("OFL", encoding="utf-8")
    return tmp_path


def test_index_records_license_hash_premiere_and_status(tmp_path):
    cat = PK.index_local(_root(tmp_path))
    by = {p["ad"]: p for p in cat["paketler"]}
    w = by["whoosh"]
    assert w["durum"] == "kullanilabilir" and len(w["sha256"]) == 64 and w["premiere_uyumlu"].startswith("evet")
    assert w["konum"].replace("\\", "/") == "assets/sfx_hazir/whoosh.wav" and w["boyut"] > 0 and "whoosh" in w["etiketler"]
    m = by["Samurai Sake (Royalty Free)"]
    assert m["durum"] == "lisans_bekliyor" and not m["lisans_dogrulandi"]          # not used until the license is known
    assert by["fire"]["durum"] == "yasak"                                          # the user removed emoji
    assert by["Inter"]["lisans"].startswith("SIL Open Font") and by["Inter"]["durum"] == "kullanilabilir"
    again = PK.index_local(tmp_path)
    assert len(again["paketler"]) == len(cat["paketler"])                          # same SHA-256: never twice


def test_requests_are_answered_locally_or_explained(tmp_path):
    root = _root(tmp_path)
    PK.index_local(root)
    r = PK.request(root, {"ajan": "ses", "tur": "efekt", "amac": "mekân geçişi", "anahtarlar": ["whoosh"]})
    assert r["durum"] == "yerelde_var" and r["paketler"][0]["ad"] == "whoosh"
    r = PK.request(root, {"ajan": "hikaye", "tur": "muzik", "amac": "intro", "anahtarlar": ["rock"]})
    assert r["durum"] == "eksik" and r["lisans_bekleyen"] and "lisansını onaylarsan" in r["sonraki_adim"]
    r = PK.request(root, {"ajan": "grafik", "tur": "grafik", "amac": "harita", "anahtarlar": ["harita"]})
    assert r["durum"] == "eksik" and "izinli kaynak yok" in r["sonraki_adim"]
    assert PK.request(root, {"tur": "video"})["durum"] == "gecersiz"


def test_user_confirms_a_license_and_it_sticks(tmp_path):
    root = _root(tmp_path)
    cat = PK.index_local(root)
    music = next(p for p in cat["paketler"] if p["tur"] == "muzik")
    PK.confirm_license(root, music["id"], "CC BY 4.0 - Darren Curtis, atıf gerekli", commercial=True)
    again = {p["id"]: p for p in PK.index_local(root)["paketler"]}
    assert again[music["id"]]["durum"] == "kullanilabilir" and "CC BY" in again[music["id"]]["lisans"]
    assert PK.request(root, {"tur": "muzik", "amac": "intro", "anahtarlar": ["samurai"]})["durum"] == "yerelde_var"


def test_resource_request_action_is_validated():
    assert A.validate("kaynak_iste", {"tur": "efekt", "amac": "geçiş", "anahtarlar": ["whoosh"]}) is None
    assert "bilinmeyen kaynak türü" in A.validate("kaynak_iste", {"tur": "video", "amac": "x"})
    assert "amacı yok" in A.validate("kaynak_iste", {"tur": "efekt"})
    assert json.dumps(A.ACTIONS["kaynak_iste"], ensure_ascii=False)
