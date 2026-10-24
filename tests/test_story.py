from kurgu.editing.shorts import sentences, story_summary, validate_summary


def _words(lines):
    out, t = [], 0.0
    for line in lines:
        for w in line.split():
            out.append({"word": w, "t_start": t, "t_end": t + 0.35, "prob": 0.9})
            t += 0.4
        t += 1.0
    return out


def test_story_has_hook_intro_middle_end_and_no_call_to_action():
    lines = ["Bugün Japon restoranındayız, sınırsız et yiyeceğiz."] + \
            [f"Burada masa {i} var ve garson geliyor." for i in range(30)] + \
            ["Bakalım sınırsız et yersek kaç para ödeyeceğiz?", "Videoyu beğenin abone olmayı unutmayın."] + \
            [f"Et geldi tabak {i} burada." for i in range(30)] + ["Bu et inanılmaz lezzetli, çok güzel!"] + \
            [f"Sos {i} ile deniyoruz şimdi." for i in range(30)] + ["Toplam hesap iki kişi 90 dolar geldi."]
    S = sentences(_words(lines))
    p = validate_summary(story_summary(S, 20, 60), S, 20, 60)
    texts = [S[i]["text"] for c in p["clips"] for i in range(c["start"], c["end"] + 1)]
    assert "kaç para" in S[p["hook"]]["text"]
    assert not any("abone" in t for t in texts)
    assert any("90 dolar" in t for t in texts)
    assert p["clips"][0]["role"] == "giriş" and p["clips"][-1]["role"] == "sonuç"
    assert "KAÇ PARA" in p["headline"]
