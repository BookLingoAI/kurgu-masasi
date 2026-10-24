"""Sentence-level titles, the do-no-harm grade, transitions and the YouTube 1440p delivery."""
import numpy as np

from kurgu.editing.graphics import good_title, heuristic_graphics, number_title, resolve_callouts, verdict_title


def _ws(text, t0=0.0):
    return [{"word": w, "t_start": t0 + i * 0.3, "t_end": t0 + i * 0.3 + 0.25, "prob": 0.9} for i, w in enumerate(text.split())]


def test_good_title_rejects_lone_adjectives():
    assert not good_title("GÜZEL")
    assert not good_title("ÇOK GÜZEL")
    assert good_title("20 SANİYE")
    assert good_title("SOSU ÇOK ACI")


def test_number_title_keeps_meaning():
    assert number_title(_ws("bir önceki de 167 dolar verdik, Brezilya'da"))[:2] == ("167 DOLAR VERDİK", "FİYAT")
    assert number_title(_ws("tutarın 8 dolar 24 centi."))[0] == "8 DOLAR 24 CENT"
    assert number_title(_ws("yaklaşık 3 dakika veya 4 dakika"))[0] == "3 DAKİKA"
    assert number_title(_ws("bugün hava güzel")) is None


def test_verdict_needs_a_subject():
    assert verdict_title(_ws("bu sos çok acı ya"))[0] == "SOS ÇOK ACI"
    assert verdict_title(_ws("ama çok güzel")) is None


def test_heuristic_titles_spaced_and_labelled():
    sents = []
    for i in range(6):
        ws = _ws(f"burada şimdi {10 + i} dolar verdik", t0=i * 20.0)
        sents.append({"i": i, "start": ws[0]["t_start"], "end": ws[-1]["t_end"], "words": ws,
                      "text": " ".join(w["word"] for w in ws)})
    out = resolve_callouts(heuristic_graphics(sents), sents, min_gap=30.0)
    assert out and all(c["kicker"] == "FİYAT" for c in out)
    assert all(b["t0"] - a["t0"] >= 30.0 for a, b in zip(out, out[1:]))


def test_grade_never_darkens():
    from kurgu.vision.grade import apply_grade, compute_grade
    rng = np.random.default_rng(1)
    bright = rng.integers(0, 245, (180, 320, 3), dtype=np.uint8)
    dim = (bright * 0.4).astype(np.uint8)
    for f in (bright, dim):
        g = compute_grade([f])
        o = apply_grade(f, g)
        Y = lambda a: a.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)  # noqa: E731
        assert Y(o).mean() >= Y(f).mean() - 1.0
        assert np.percentile(Y(o), 5) >= np.percentile(Y(f), 5) - 2.0   # shadows are not crushed


def test_transitions_and_upscale():
    from kurgu.integrations.premiere_edit import youtube_upscale
    from kurgu.motion.render import transition_fx
    f = np.full((108, 192, 3), 90, np.uint8)
    assert transition_fx(f, "flash", 0.0).mean() > f.mean() + 20
    assert abs(float(transition_fx(f, "flash", 1.0).mean()) - 90) < 1
    assert transition_fx(f, "whip", 0.0).shape == f.shape
    assert youtube_upscale({"w": 1920, "h": 1080, "opts": {"resolution": "1440", "capcut": False}}) == (2560, 1440)
    assert youtube_upscale({"w": 1920, "h": 1080, "opts": {"resolution": "1440", "capcut": True}}) is None
    assert youtube_upscale({"w": 1920, "h": 1080, "opts": {"resolution": "1080"}}) is None
    assert youtube_upscale({"w": 2560, "h": 1440, "opts": {"resolution": "1440"}}) is None


def test_wipe_reveals_from_left():
    from kurgu.motion.render import Overlay, blend
    rgba = np.zeros((100, 200, 4), np.uint8)
    rgba[40:60, 20:180] = (255, 255, 255, 255)
    ov = Overlay(0.0, 3.0, rgba, fade=0.0, anim="wipe")
    base = np.zeros((100, 200, 3), np.uint8)
    early, full = blend(base, ov, 0.1), blend(base, ov, 1.0)
    assert early[50, 25].max() > 0 and early[50, 170].max() == 0
    assert full[50, 170].max() == 255


def test_title_styles_by_meaning():
    from kurgu.motion.render import Overlay, blend
    from kurgu.motion.titles import build_title
    seen = {}
    for text, k in [("167 dolar verdik", "Fiyat"), ("Sosu çok acı", "Yorum"), ("27 dakika bekledik", "Süre"),
                    ("Sınırsız et büfesi", "Bilgi")]:
        layers, info = build_title(text, k, 1920, 1080, None)
        seen[k] = info["style"]
        assert layers and info["sfx"]
        ovs = [Overlay(L.get("delay", 0.0), 3.0, L["rgba"], fade=L.get("fade", 0.25), anim=L["anim"], meta=L.get("meta") or {})
               for L in layers]
        base = np.zeros((1080, 1920, 3), np.uint8)
        f = base
        for o in ovs:
            f = blend(f, o, 1.5)
        assert f.any()
    assert seen == {"Fiyat": "fiyat", "Yorum": "kalem", "Süre": "daktilo", "Bilgi": "bant"}
    assert build_title("Et çok yumuşak", "Fiyat", 1920, 1080, None)[1]["style"] == "bant"   # no number: no price style


def test_typewriter_reveals_progressively():
    from kurgu.motion.render import Overlay, blend
    from kurgu.motion.titles import build_title
    layers, _ = build_title("27 dakika bekledik", "Süre", 1920, 1080, None)
    L = layers[1]
    ov = Overlay(0.0, 3.0, L["rgba"], fade=0.0, anim="type", meta=L["meta"])
    base = np.zeros((1080, 1920, 3), np.uint8)
    a, b = blend(base, ov, 0.06).astype(bool).sum(), blend(base, ov, 1.5).astype(bool).sum()
    assert 0 < a < b


def test_sentence_moment_words_follow_speech():
    from kurgu.motion.render import Overlay, apply_camfx, blend
    from kurgu.motion.titles import build_sentence
    words = [(wd, round(6.2 + i * 0.35, 3), round(6.48 + i * 0.35, 3)) for i, wd in enumerate("Bu kızarmış şeye bayıldım ya!".split())]
    for kind in ("punch", "slide"):
        fx = {"t0": 6.0, "t1": 9.3, "kind": kind, "side": "left", "amt": 0.36, "cx": 0.5, "cy": 0.4, "zoom": 1.2,
              "words": words}
        layers, info = build_sentence(fx, 1920, 1080)
        ovs = [Overlay(6.0 + L.get("delay", 0.0), 9.0, L["rgba"], fade=L.get("fade", 0.0), anim=L["anim"],
                       meta=L.get("meta") or {}) for L in layers]
        base = np.zeros((1080, 1920, 3), np.uint8)

        def lit(t):
            f = base
            for o in ovs:
                f = blend(f, o, t)
            return int((f.max(axis=2) > 200).sum())
        assert 0 < lit(6.55) < lit(7.0) < lit(8.2)          # words appear as they are said
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    slid = apply_camfx(frame, {"t0": 0.0, "t1": 4.0, "kind": "slide", "side": "left", "amt": 0.36}, 2.0)
    assert slid[:, :500].mean() < 120 and slid[:, 1500:].mean() > 190   # dark panel on the left, picture on the right


def test_plan_camfx_spacing_and_quality():
    from kurgu.editing.graphics import plan_camfx

    def sent(i, text, t0, prob=0.95):
        ws = [{"word": w, "t_start": t0 + k * 0.3, "t_end": t0 + k * 0.3 + 0.25, "prob": prob} for k, w in enumerate(text.split())]
        return {"i": i, "start": ws[0]["t_start"], "end": ws[-1]["t_end"], "text": text, "words": ws}
    sents = [sent(0, "Bu restorana bayıldım, kesinlikle tavsiye ederim!", 10.0),
             sent(1, "Bu restorana bayıldım, kesinlikle tavsiye ederim!", 20.0),
             sent(2, "Can have more food on please?", 80.0),
             sent(3, "Burada sınırsız et yiyoruz ve fiyatı çok uygun.", 140.0, prob=0.4)]
    segs = [{"timeline_start": 0.0, "timeline_end": 300.0, "file": "a", "in": 0.0, "out": 300.0}]
    out = plan_camfx(sents, segs, [], [], gap=45.0)
    assert [o["sentence"] for o in out] == [0]      # spaced, no English, no unsure transcription


def test_all_title_styles_and_themes():
    from kurgu.motion.render import Overlay, blend
    from kurgu.motion.titles import PALETTES, STYLES, build_title, set_theme
    for i, st in enumerate(STYLES):
        set_theme(list(PALETTES)[i % len(PALETTES)])
        layers, info = build_title("167 dolar verdik" if st == "fiyat" else "Sınırsız et büfesi", "Bilgi", 1920, 1080,
                                   None, style=st)
        assert layers and info["style"] == st
        f = np.zeros((1080, 1920, 3), np.uint8)
        for L in layers:
            f = blend(f, Overlay(L.get("delay", 0.0), 4.0, L["rgba"], fade=L.get("fade", 0.25), anim=L["anim"],
                                 meta=L.get("meta") or {}), 2.0)
        assert f.any()
    a = set_theme("otomatik", "proje-1")
    assert a in PALETTES.values() and set_theme("otomatik", "proje-1") == a   # stable per video
    set_theme("sari")


def test_caption_box_style():
    from kurgu.motion.captions import build_ass
    ws = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.35} for i, w in enumerate("bu çok güzel".split())]
    ass = build_ass(ws, 1080, 1920, style="kutu", active_color="&H00C9D600")
    assert "\\bord" in ass and "&H00141414&" in ass


def test_per_recording_grade_and_subjects():
    from kurgu.editing.graphics import not_a_subject
    from kurgu.vision.grade import apply_grade, compute_grade, grade_for
    rng = np.random.default_rng(3)
    dim = (rng.integers(0, 245, (90, 160, 3)) * 0.35).astype(np.uint8)
    bright = rng.integers(0, 245, (90, 160, 3)).astype(np.uint8)
    g = {"version": 3, "default": compute_grade([bright]), "files": {"ic.mp4": compute_grade([dim]), "dis.mp4": compute_grade([bright])}}
    assert grade_for(g, "ic.mp4")["gamma"] < grade_for(g, "dis.mp4")["gamma"] + 1e-9   # the dim clip is lifted more
    assert grade_for(g, "yok.mp4") is g["default"]
    assert apply_grade(dim, g, "ic.mp4").mean() > dim.mean()
    assert all(not_a_subject(w) for w in ("ooo", "vaaay", "konusunda", "geldim", "yiyorum"))
    assert not any(not_a_subject(w) for w in ("sos", "et", "suşi", "tatlı"))


def test_orphan_jobs_closed(tmp_path):
    import json

    from kurgu.config import Config
    from kurgu.worker.jobs import JobManager
    (tmp_path / "isler" / "sonuc").mkdir(parents=True)
    (tmp_path / "isler" / "sonuc" / "abc.calisiyor.json").write_text(json.dumps({"id": "abc", "state": "running"}), encoding="utf-8")
    import shutil
    from pathlib import Path
    shutil.copy(Path(__file__).resolve().parents[1] / "design_config.json", tmp_path / "design_config.json")
    JobManager(Config.load(tmp_path))
    assert not (tmp_path / "isler" / "sonuc" / "abc.calisiyor.json").exists()
    assert json.loads((tmp_path / "isler" / "sonuc" / "abc.json").read_text(encoding="utf-8"))["state"] == "stopped"
