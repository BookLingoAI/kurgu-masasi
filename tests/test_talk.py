"""Masa başı anlatım: the language-model type plan (talk.py), the camera-card stage (stage.py) and topic icons."""
import numpy as np
import pytest

from kurgu.editing.shorts import sentences
from kurgu.editing.talk import item_times, plan_talk, word_time
from kurgu.motion.stage import Stage, plan_layout


def _ws(text, t0=0.0, step=0.4):
    return [{"word": w, "t_start": round(t0 + i * step, 3), "t_end": round(t0 + i * step + 0.3, 3), "prob": 0.9}
            for i, w in enumerate(text.split())]


TALK = (_ws("Bugün size yapay zekâ hakkında konuşacağım.", 0.0)
        + _ws("Asıl mesele her çıkan aracı kullanabilme yeteneği.", 5.0)
        + _ws("Bu işte el yok, kol yok, göz yok.", 10.0)
        + _ws("Sonra kod yazmayı, test etmeyi ve yayınlamayı öğreniyoruz.", 15.0))


class FakeLLM:
    def __init__(self, answer=None, fail=False):
        self.answer, self.fail, self.calls = answer, fail, 0

    def chat_json(self, system, prompt, schema, max_tokens=800, temperature=0.3):
        self.calls += 1
        if self.fail:
            raise TimeoutError("no answer")
        return self.answer


def test_plan_talk_keeps_only_what_was_said():
    llm = FakeLLM({"k": [[1, "ASIL MESELE", "Her çıkan aracı", "kullanabilme yeteneği."],
                         [1, "", "Sihirli değnek", "yok."],                       # never said
                         [0, "UYDURMA BAĞLAM", "yapay zekâ", "hakkında"]],         # kicker not said -> dropped
                   "l": [[2, "El kol", ["el yok", "kol yok", "göz yok"], "o"],  # all negative -> x
                         [3, "Adımlar", ["kod yazmayı", "test etmeyi", "yayınlamayı"], "v"],
                         [3, "Uydurma", ["deploy", "monitor"], "v"]]})              # items not said
    res = plan_talk(llm, TALK)
    assert llm.calls == 1 and res["failed"] == 0
    assert [(p["sentence"], p["big"]) for p in res["phrases"]] == [(1, "Her çıkan aracı"), (0, "yapay zekâ")]
    assert res["phrases"][0]["kicker"] == "ASIL MESELE" and res["phrases"][1]["kicker"] == ""
    assert [(L["sentence"], L["icon"]) for L in res["lists"]] == [(2, "x"), (3, "v")]
    assert [L["kicker"] for L in res["lists"]] == ["El kol", ""]                # "Adımlar" was not said


def test_plan_talk_big_type_drops_a_trailing_comma():
    ws = _ws("Berberlik yaparken, 12 saat çalışıyordum.")
    res = plan_talk(FakeLLM({"k": [[0, "", "Berberlik yaparken,", "12 saat çalışıyordum."]], "l": []}), ws)
    assert res["phrases"][0]["big"] == "Berberlik yaparken"


def test_plan_talk_survives_a_silent_model():
    res = plan_talk(FakeLLM(fail=True), TALK)
    assert res["phrases"] == [] and res["lists"] == [] and res["failed"] == res["windows"] == 1


def test_plan_talk_ignores_malformed_rows():
    res = plan_talk(FakeLLM({"k": [["x", "", "a", "b"], [1, "", "", ""], [99, "", "Her", ""], "junk"],
                             "l": [[2, "", "el yok", "x"], [2, "", ["tek"], "x"]]}), TALK)
    assert res["phrases"] == [] and res["lists"] == []


def test_word_time_allows_suffixes_and_order():
    ws = _ws("kod yazmayı sonra kodları test etmeyi")
    assert word_time(ws, "kod") == (0.0, 0)
    assert word_time(ws, "kodlar", 1) == (ws[3]["t_start"], 3)
    assert word_time(ws, "python") is None
    assert word_time(ws, "") is None


def test_item_times_in_order_or_nothing():
    by_i = {s["i"]: s for s in sentences(TALK)}
    t = item_times(by_i, 3, ["kod yazmayı", "test etmeyi", "yayınlamayı"])
    assert len(t) == 3 and t == sorted(t) and t[0] >= 15.0
    assert item_times(by_i, 3, ["yayınlamayı", "kod yazmayı"]) == []      # said in the other order
    assert item_times(by_i, 2, ["el", "kol", "göz"]) == pytest.approx([10.8, 11.6, 12.4])


def test_plan_layout_moves_camera_away_from_text_and_back():
    lay = plan_layout([{"t0": 10, "t1": 14, "side": "left"}, {"t0": 16, "t1": 19, "side": "left"},
                       {"t0": 40, "t1": 41, "side": "right"}], 60.0)
    assert lay[0] == {"t0": 0.0, "t1": 9.65, "state": "full"}
    states = [s["state"] for s in lay]
    assert states == ["full", "right", "full", "left", "full"]     # 10-19 joined (gap 2 s < 4 s): no ping-pong
    assert lay[3]["t1"] - lay[3]["t0"] >= 2.99                      # a 1 s title still holds the card 3 s
    assert all(abs(a["t1"] - b["t0"]) < 1e-6 for a, b in zip(lay, lay[1:]))
    assert lay[-1]["t1"] == 60.0


def test_plan_layout_gives_screen_recordings_to_the_compositor():
    lay = plan_layout([{"t0": 10, "t1": 20, "side": "left"}], 40.0, screen=[(15.0, 25.0)])
    assert [s["state"] for s in lay] == ["full", "right", "none", "full"]
    right = lay[1]
    assert right["t1"] == 15.0 and lay[2] == {"t0": 15.0, "t1": 25.0, "state": "none"}


def _frame(w=640, h=360):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 200       # flat red camera picture, easy to find
    return img


def test_stage_full_and_none_pass_the_frame_through():
    st = Stage([{"t0": 0, "t1": 5, "state": "full"}, {"t0": 5, "t1": 10, "state": "none"}], 640, 360, (255, 120, 40))
    f = _frame()
    assert st.apply(f, 1.0) is f
    assert st.apply(f, 7.0) is f
    assert st.apply(f, 99.0) is f


def test_stage_card_on_the_right_with_rounded_corners():
    w, h = 640, 360
    st = Stage([{"t0": 0, "t1": 2, "state": "full"}, {"t0": 2, "t1": 8, "state": "right"}], w, h, (255, 120, 40))
    out = st.apply(_frame(w, h), 5.0)                                   # well after the 0.6 s move
    x, y, cw, ch, rad = st.rect("right")
    assert out.shape == (h, w, 3) and out.dtype == np.uint8
    assert out[int(y + ch / 2), int(x + cw / 2), 0] == 200            # camera inside the card
    assert out[h // 2, int(0.2 * w)].max() < 60                         # dark set on the left (the text side)
    assert out[int(y) + 1, int(x) + 1, 0] < 150                         # corner is cut round
    mid = st.apply(_frame(w, h), 2.3)                                   # mid-move: card bigger than at rest
    assert (mid[..., 0] == 200).sum() > (out[..., 0] == 200).sum()


def test_stage_eases_back_to_full_screen():
    w, h = 320, 180
    st = Stage([{"t0": 0, "t1": 3, "state": "left"}, {"t0": 3, "t1": 9, "state": "full"}], w, h, (60, 200, 255))
    f = _frame(w, h)
    assert st.apply(f, 3.1) is not f                                    # still moving
    assert st.apply(f, 3.0 + 0.61) is f                                 # arrived


def test_icon_pick_names_only_clear_subjects():
    from kurgu.motion.icons import pick
    assert pick("Yapay zekâ") == "bot"
    assert pick("Paranın", "değeri") == "banknote"
    assert pick("düşünce", "biçimi") == "brain"
    assert pick("işte", "böyle") is None
    assert pick("", "Her çıkan aracı") is None


def test_icon_rasterizer_draws_lucide_outlines():
    from kurgu.motion.icons import icon_rgba, rasterize
    svg = ('<svg viewBox="0 0 24 24" stroke-width="2"><path d="M4 12h16" /><circle cx="12" cy="12" r="10" />'
           '<path d="M8 8a4 4 0 0 1 8 0" /></svg>')
    a = rasterize(svg, 96)
    assert a.shape == (96, 96) and a[48, 48] == 255 and a[48, 8] > 200          # line, circle
    assert a[16, 48] == 255 and a[30, 48] < 20                                   # arc bulges upwards (sweep 1)
    ic = icon_rgba("bot", 64, (255, 120, 40))
    if ic is None:
        pytest.skip("assets/icons not installed")
    assert ic.shape == (64, 64, 4) and ic[..., 3].max() == 255 and tuple(ic[0, 0, :3]) == (255, 120, 40)


def test_topic_with_icon_draws_it_left_of_the_kicker():
    from kurgu.motion.editorial import Topic
    from kurgu.motion.icons import icon_rgba
    if icon_rgba("bot", 8, (0, 0, 0)) is None:
        pytest.skip("assets/icons not installed")
    w, h = 1280, 720
    plain = Topic("Bölüm 1", "Yapay", "zekâ", w, h, dur=4.0)
    tp = Topic("Bölüm 1", "Yapay", "zekâ", w, h, dur=4.0, icon="bot")
    assert tp.dx_kick > 0 and plain.dx_kick == 0
    bg = np.full((h, w, 3), 20, np.uint8)
    a, b = plain.draw(bg, 1.5), tp.draw(bg, 1.5)
    ih = tp.icon.shape[0]
    zone = (slice(tp.yk, tp.yk + tp.kick.shape[0]), slice(tp.x, tp.x + ih))
    assert np.abs(b[zone].astype(int) - a[zone].astype(int)).max() > 100   # the icon is there
