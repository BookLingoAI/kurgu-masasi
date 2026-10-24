"""Content profiles (kurgu/board/profiles.py): guessing, the user's choice, claims with source and date, prompts."""
import json
from pathlib import Path

from kurgu.board import live as L
from kurgu.board import profiles as PR


def _ws(text, t0=0.0):
    return [{"word": w, "t_start": t0 + i * 0.4, "t_end": t0 + i * 0.4 + 0.3} for i, w in enumerate(text.split())]


FOOD = _ws("bugün bu restoranda sushi ve ramen yiyoruz sos çok lezzetli tabak geldi tadı harika hesap 45 dolar tuttu porsiyon büyük")
CAR = _ws("bu toyota corolla hibrit modelin fiyatı 1 milyon 450 bin lira motor 140 beygir yakıt tüketimi düşük galeri sıfır km")


def test_guess_by_keywords_and_weak_guess():
    g = PR.guess("Japon restoranı", FOOD)
    assert g["profil"] == "yemek" and g["guven"] >= PR.ASK_BELOW
    assert PR.guess("Corolla inceleme", CAR)["profil"] == "araba_fiyat"
    weak = PR.guess("Günaydın", _ws("bugün biraz sohbet edelim nasılsınız"))
    assert weak["profil"] == "vlog" and weak["guven"] < PR.ASK_BELOW


def test_claims_keep_source_time_and_date():
    cl = PR.claims(CAR, "kayıt/proje tarihi 2026-10-22")
    kinds = {c["tur"] for c in cl}
    assert {"fiyat", "ozellik"} <= kinds
    price = next(c for c in cl if c["tur"] == "fiyat")
    assert "lira" in price["metin"] and price["kaynak"].startswith("transkript @0:") and price["tarih"].endswith("2026-10-22")
    assert "doğrulanmadı" in price["dogrulama"]


class P:
    def __init__(self, d, words, title="Deneme"):
        self.dir, self.name, self._w = d, "Deneme", words
        self.settings = {"title": title, "created": "2026-10-22 09:00"}

    def load(self, fn, default=None):
        return {"transcript_cut": self._w} if fn == "edit_plan.json" else default


def test_users_choice_beats_the_guess_and_claims_follow_the_profile(tmp_path):
    proj = P(tmp_path, CAR, "Corolla")
    prof = PR.resolve(proj)
    assert prof["profil"] == "araba_fiyat" and prof["kaynak"] == "tahmin" and prof["iddialar"]
    prof = PR.set_by_user(proj, "yemek")
    assert prof["kaynak"] == "kullanici" and prof["profil"] == "yemek" and prof["iddialar"] == []
    assert PR.resolve(proj)["profil"] == "yemek"                     # a later look keeps the user's choice
    assert json.loads((tmp_path / "kurul" / "profil.json").read_text(encoding="utf-8"))["kaynak"] == "kullanici"


def test_prompt_block_is_per_agent_and_lists_claims(tmp_path):
    prof = PR.resolve(P(tmp_path, CAR, "Corolla"))
    g = PR.prompt_block(prof, "grafik")
    assert "İÇERİK PROFİLİ: Araba fiyatları" in g and "[PROFİL KURALI]" in g and "İDDİA KURALI" in g and "lira" in g
    s = PR.prompt_block(prof, "ses")
    assert "Galeri/yol sesi" in s and "Toyota" not in s
    from kurgu.board import meeting as MT
    rep = {"duration_s": 60, "missing": [], "grafik": {"metrics": {}, "flags": []}}
    text = MT._view_prompt("grafik", rep, {"dersler": []}, {}, vision=False, profile=prof)
    assert text.index("GENEL UZMAN KONTROL") < text.index("İÇERİK PROFİLİ")       # general and profile rules apart


def test_chair_asks_when_unsure_and_takes_a_spoken_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "say", lambda who, text, d: Path(f"{who}.wav"))
    proj = P(tmp_path, _ws("bugün biraz sohbet edelim nasılsınız"))

    class Cfg:
        root = tmp_path
    proj.cfg = Cfg
    m = L.LiveMeeting(proj, llm=None)
    m.inbox.put("bu bir gezi videosu")
    prof = m._settle_profile(wait=2.0)
    assert prof["profil"] == "gezi" and prof["kaynak"] == "kullanici"
    m.wait_voices()
    assert any(e.get("soru") == "profil" for e in m.since(-1)["events"])
