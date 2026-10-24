"""Humanizer-style YouTube text: code finds the AI tells and unsaid facts, the rewrite pass runs, the file is written."""
from kurgu.editing import humanize as hz


def _words(text, t0=0.0):
    out = []
    for k, w in enumerate(text.split()):
        out.append({"word": w, "t_start": t0 + k * 0.4, "t_end": t0 + k * 0.4 + 0.3})
    return out


WORDS = _words("Bugün Tokyo'da bir Japon restoranına geldik. Burada hot pot yiyeceğiz. Hesap 167 dolar tuttu. "
               "Sos çok acıydı ama et çok güzeldi. Siz olsanız hangisini seçerdiniz?")


def test_tells_find_the_patterns():
    t = hz.tells("Bu sadece bir yemek değil, bir yolculuk — eşsiz bir deneyim!!")
    assert "uzun tire" in t and "'X değil, Y' karşıtlığı" in t and "birden çok ünlem" in t
    assert any("yolculu" in x for x in t) and any("eşsiz" in x for x in t)
    assert hz.tells("Tokyo'da hot pot yedik, hesap 167 dolar tuttu.") == []


def test_unsaid_numbers_and_names():
    spoken = " ".join(w["word"] for w in WORDS)
    assert hz.unsaid("Hesap 167 dolar, Tokyo'da yedik.", spoken) == []
    bad = hz.unsaid("Hesap 250 dolar, Osaka'da yedik.", spoken)
    assert "sayı 250" in bad and "ad Osaka" in bad


def test_numbers_said_apart_are_not_joined():
    filler = " ".join(["ve"] * 40)
    spoken = f"Çin restoranında 4 kişi ödedik. {filler} Burada 90 dolar verdik, iki kişi."
    assert "4 ile 90 konuşmada birlikte geçmiyor" in hz.unsaid("4 kişi 90 dolar ödedik.", spoken)
    assert hz.unsaid("İki kişi 90 dolar verdik.", spoken) == []


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat_json(self, system, user, schema, **kw):
        self.calls.append(user)
        robotic = len(self.calls) == 1
        title = "Tokyo'da Japon restoranı: sadece yemek değil, bir yolculuk" if robotic else \
            "Tokyo'da Japon restoranında hot pot yedik, hesap 167 dolar"
        return {"titles": [{"angle": a, "title": title} for a in ("merak", "fiyat", "deneyim")],
                "description_intro": "Tokyo'da bir Japon restoranında hot pot yedik.",
                "description_body": "Sos çok acıydı ama et güzeldi.\n\nSiz hangisini seçerdiniz?",
                "chapters": [{"sentence": 0, "title": "Giriş"}],
                "hashtags": ["#tokyo", "japon yemeği"], "shorts_title": "167 dolarlık hot pot",
                "shorts_description": "Tokyo'da hot pot. #shorts #tokyo #yemek"}


def test_write_runs_the_rewrite_pass_and_reports(tmp_path):
    llm = FakeLLM()
    r = hz.write(llm, WORDS, 30.0, "Kaç para ödeyeceğiz?")
    assert len(llm.calls) >= 2 and "Kodun bulduğu sorunlar" in llm.calls[1]
    assert "Kaç para ödeyeceğiz?" in llm.calls[0]
    assert all("değil" not in t["title"] for t in r["titles"]) and len(r["titles"]) == 3
    assert r["hashtags"] == ["#tokyo", "#japonyemeği"]
    txt = hz.text_file(r)
    for key in ("BAŞLIK 1", "BAŞLIK 2", "BAŞLIK 3", "AÇIKLAMA:", "SHORTS BAŞLIK:", "SHORTS AÇIKLAMA:"):
        assert key in txt
    assert "—" not in txt


class SlowLLM(FakeLLM):
    def chat_json(self, system, user, schema, **kw):
        if self.calls:
            raise TimeoutError("timed out")
        return super().chat_json(system, user, schema, **kw)


def test_a_failed_rewrite_keeps_the_draft():
    r = hz.write(SlowLLM(), WORDS, 30.0)
    assert len(r["titles"]) == 3 and any("düzeltme turu yapılamadı" in n for n in r["notes"])


def test_cleanup_removes_dashes_and_bold():
    d = hz._clean({"titles": [{"angle": "a", "title": "**Hot pot** — Tokyo"}], "description_intro": "a – b",
                   "description_body": "x", "shorts_title": "", "shorts_description": ""})
    assert d["titles"][0]["title"] == "Hot pot, Tokyo" and "–" not in d["description_intro"]
