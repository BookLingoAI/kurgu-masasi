"""This round: counting speed-up, proofreading, grounded titles, editorial graphics, protected spans, SFX variety."""
import numpy as np

from kurgu.editing import counting, proofread
from kurgu.editing.graphics import grounded


def _w(text, t, d=0.35, gap=0.25):
    out = []
    for tok in text.split():
        out.append({"word": tok, "start": t, "end": t + d, "t_start": t, "t_end": t + d})
        t += d + gap
    return out, t


def test_find_counts_turkish_words_and_compounds():
    w1, t = _w("şimdi sayıyorum", 0.0)
    w2, t = _w("bir iki üç dört beş altı yedi sekiz dokuz on on bir on iki", t + 0.5, gap=0.6)
    w3, _ = _w("tamam oldu", t + 1.0)
    runs = counting.find_counts(w1 + w2 + w3)
    assert len(runs) == 1
    vals = [v for _, v in runs[0]["marks"]]
    assert vals[0] == 1 and vals[-1] == 12 and 11 in vals


def test_find_counts_ignores_single_numbers():
    w, _ = _w("bir tane aldım iki dolar üç kişi yedik", 0.0)
    assert counting.find_counts(w) == []


def test_apply_speed_retimes_plan_and_words():
    words, t = _w("bir iki üç dört beş altı yedi sekiz dokuz on", 1.0, gap=0.65)
    seg = {"id": 0, "file": "a.mp4", "in": 0.0, "out": 20.0, "cam_start": 0.0, "cam_end": 20.0,
           "timeline_start": 0.0, "timeline_end": 20.0}
    pl = {"segments": [seg], "transcript_cut": words, "stats": {"timeline_s": 20.0}}
    run = counting.find_counts(words)[0]
    new, remap = counting.apply_speed(pl, counting.zones_of(run), 30.0)
    segs = new["segments"]
    assert len(segs) == 5 and [s.get("speed", 1.0) for s in segs] == [1.0, 2.0, 3.0, 4.0, 1.0]
    # source stays contiguous, timeline is shorter, last part ends where the source did
    for a, b in zip(segs, segs[1:]):
        assert abs(a["out"] - b["in"]) < 1e-6 and abs(a["timeline_end"] - b["timeline_start"]) < 1e-6
    assert abs(segs[-1]["out"] - 20.0) < 1e-6
    saved = (run["t1"] - run["t0"]) - (remap(run["t1"]) - remap(run["t0"]))
    assert saved > 0.6 * (run["t1"] - run["t0"])
    assert abs(new["stats"]["timeline_s"] - (20.0 - saved)) < 0.1
    assert all(w.get("fast") for w in new["transcript_cut"])
    assert abs(remap(0.5) - 0.5) < 0.05


def test_speed_filter_chain():
    from kurgu.motion.render import Piece, speed_filter
    f = speed_filter(4.0, 1.85)
    assert "asetrate" in f and f.count("atempo") >= 2
    p = Piece("a.mp4", 10.0, 16.0, speed=3.0)
    assert abs(p.dur - 2.0) < 1e-9 and abs(p.src_dur - 6.0) < 1e-9


def test_grounded_titles():
    s = {"words": _w("Çinli sos çok acı ama güzel", 0.0)[0]}
    assert grounded("ÇİNLİ SOS ÇOK ACI", s)
    assert not grounded("AMERİKA'DA SAHTE YENGEÇ", {"words": _w("burada yengeç var", 0.0)[0]})


def test_proofread_apply_and_guards():
    class LLM:
        def chat_json(self, system, prompt, schema, **k):
            return {"d": [[0, "Bugün dana yanağı yiyoruz."], [1, "tamamen başka bir cümle yazdım buraya"]],
                    "t": [[2, "Hadi birlikte gidelim."]]}
    w = []
    t = 0.0
    for sent in ("Bugün yalan ederken yiyoruz.", "Çok güzel olmuş.", "Together go."):
        ws, t = _w(sent, t)
        w += ws
        t += 1.5
    for x in w[-2:]:
        x["lang"] = "en"
    res = proofread.proofread(w, LLM(), ["dana yanağı"])
    assert "0" in res["fixes"] and "1" not in res["fixes"] and "2" in res["translations"]
    out = proofread.apply(w, res)
    text = " ".join(x["word"] for x in out)
    assert "dana yanağı" in text and "yalan" not in text
    assert out[-1]["tr_line"] == "Hadi birlikte gidelim."
    assert out[-1]["t_end"] == w[-1]["t_end"]


def test_split_title_and_editorial_draw():
    from kurgu.motion.editorial import Bullets, Counter, Topic, split_title
    assert split_title("SOS VE EKLENTİLER") == ("Sos", "ve eklentiler")
    bg = np.zeros((360, 640, 3), np.uint8)
    for obj in (Topic("MERAK ETTİĞİMİZ", "Nasıl", "zengin oluyor?", 640, 360),
                Bullets("LİSTE", ["Bir", "İki", "Üç"], [0.2, 0.6, 1.0], 640, 360),
                Counter([(0.1, 1), (0.4, 2), (0.6, 3)], [(0.0, 2.0)], 640, 360, dur=2.0)):
        assert obj.draw(bg, 1.2).sum() > 0


def test_protected_spans_survive_silence_cuts():
    from kurgu.editing.cuts import Removal, _outside
    r = Removal(10.0, 20.0, "silence", "sessizlik")
    out = _outside([r], [(12.0, 15.0)])
    spans = sorted((x.start, x.end) for x in out)
    assert spans == [(10.0, 12.0), (15.0, 20.0)]


def test_sfx_bank_rotates_variants(tmp_path):
    from kurgu.audio.master import builtin_sfx
    b = builtin_sfx(tmp_path)
    seq = [b["whoosh"] for _ in range(5)]
    assert len(set(seq)) == 5
    # fixed() is always the first synthesised take (CTA sounds); the rotation starts with downloaded sounds
    assert b.fixed("whoosh") == b.fixed("whoosh") and b.fixed("whoosh").endswith("whoosh.wav")
    assert "pop" in b and len({b["pop"] for _ in range(4)}) == 4
