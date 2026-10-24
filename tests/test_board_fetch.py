"""Approved-source downloads (kurgu/board/fetch.py) with a scripted web - nothing leaves the computer in these tests."""
import io
import json
import zipfile

import pytest

from kurgu.board import fetch as F
from kurgu.board import packages as PK

WAV = b"RIFF" + (2000).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 2000
PAGE = ('<html><a href="/license/#sfxFree">Mixkit License</a>'
        '<h2 class="item">Air woosh</h2><div data-audio-player-preview-url-value="https://assets.mixkit.co/active_storage/sfx/1489/1489-preview.mp3"></div>'
        '<h2 class="item">Arrow whoosh</h2><div data-audio-player-preview-url-value="https://assets.mixkit.co/active_storage/sfx/1490/1490-preview.mp3"></div></html>')


def web(pages: dict):
    def get(url, timeout=60.0):
        if url not in pages:
            raise AssertionError(f"unexpected download: {url}")
        return pages[url]
    return get


def test_verify_checks_signatures_and_zips():
    assert F.verify(WAV, ".wav") is None
    assert "imzası" in F.verify(b"<html>not audio</html>" * 10, ".wav")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.wav", WAV)
    assert F.verify(buf.getvalue(), ".zip") is None
    assert "bozuk zip" in F.verify(b"PK\x03\x04" + b"x" * 100, ".zip")


def test_mixkit_free_items_are_downloaded_verified_and_cataloged(tmp_path):
    get = web({"https://mixkit.co/free-sound-effects/whoosh/": PAGE.encode(),
               "https://assets.mixkit.co/active_storage/sfx/1489/1489.wav": WAV,
               "https://assets.mixkit.co/active_storage/sfx/1490/1490.wav": WAV + b"\x01"})
    added = F.fetch_mixkit(tmp_path, "whoosh", "mekân geçişi", get=get)
    assert len(added) == 2 and all(a["lisans"] == "Mixkit Sound Effects Free License" and a["ticari_kullanim"] for a in added)
    a = added[0]
    assert (tmp_path / a["konum"]).read_bytes() == WAV and len(a["sha256"]) == 64 and a["lisans_url"].endswith("#sfxFree")
    meta = json.loads((tmp_path / "paketler" / "efekt" / "mixkit-1489" / "kaynak.json").read_text(encoding="utf-8"))
    assert meta["url"].endswith("1489.wav") and meta["sayfa"].startswith("https://mixkit.co/")
    assert not (tmp_path / "paketler" / "_gecici" / "mixkit-1489").exists()           # temp area cleaned
    assert F.fetch_mixkit(tmp_path, "whoosh", "tekrar", get=get) == []                # same SHA-256: never twice
    assert PK.request(tmp_path, {"tur": "efekt", "amac": "geçiş", "anahtarlar": ["whoosh"]})["durum"] == "yerelde_var"


def test_restricted_or_unmarked_pages_are_not_taken(tmp_path):
    restricted = PAGE.replace("</html>", "<span>Restricted License</span></html>")
    assert F.mixkit_search("whoosh", get=web({"https://mixkit.co/free-sound-effects/whoosh/": restricted.encode()})) == []
    unmarked = PAGE.replace("sfxFree", "x")
    assert F.mixkit_search("whoosh", get=web({"https://mixkit.co/free-sound-effects/whoosh/": unmarked.encode()})) == []


def test_bad_files_never_reach_the_library(tmp_path):
    get = web({"https://mixkit.co/free-sound-effects/pop/": PAGE.replace("1489", "77").replace("1490", "78").encode(),
               "https://assets.mixkit.co/active_storage/sfx/77/77.wav": b"<html>error page</html>" * 100,
               "https://assets.mixkit.co/active_storage/sfx/78/78.wav": b"short"})
    assert F.fetch_mixkit(tmp_path, "pop", "x", get=get) == []
    assert not (tmp_path / "paketler" / "efekt").exists()


def test_kenney_needs_cc0_on_the_page(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Audio/click_001.ogg", b"OggS" + b"\x00" * 300)
        z.writestr("License.txt", "Creative Commons Zero, CC0")
    zurl = "https://kenney.nl/media/pages/assets/interface-sounds/abc/kenney_interface-sounds.zip"
    page = f'<p>License: Creative Commons CC0</p><a href="{zurl}">Continue without donating</a>'.encode()
    added = F.fetch_kenney(tmp_path, "interface-sounds", "arayüz sesi", get=web({"https://kenney.nl/assets/interface-sounds": page, zurl: buf.getvalue()}))
    assert len(added) == 1 and added[0]["lisans"] == "Creative Commons CC0" and added[0]["tur"] == "efekt"
    with pytest.raises(ValueError, match="CC0"):
        F.fetch_kenney(tmp_path, "x", "y", get=web({"https://kenney.nl/assets/x": b"<p>License: proprietary</p>"}))


def test_policy_and_satisfy(tmp_path):
    pol = F.policy(tmp_path)
    assert pol["otomatik"] and set(pol["kaynaklar"]) == {"mixkit", "kenney", "google_fonts", "lucide"}
    assert "adobe_stock" in pol["yalniz_listele"] and "cc_marketplace" in pol["yalniz_listele"]
    assert F.satisfy(tmp_path, {"tur": "muzik", "amac": "intro"})["durum"] == "eksik"
    assert F.satisfy(tmp_path, {"tur": "animasyon", "amac": "x"})["durum"] == "eksik"

    def offline(url, timeout=60.0):
        raise OSError("ağ yok")
    assert F.satisfy(tmp_path, {"tur": "efekt", "amac": "geçiş", "anahtarlar": ["whoosh"]}, get=offline)["durum"] == "hata"
