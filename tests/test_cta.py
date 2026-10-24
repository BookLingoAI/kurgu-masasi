"""Subscribe/like/comment reminders follow the speaker's words; invented "abone ol" over music is dropped."""
import numpy as np

from kurgu.editing import cta
from kurgu.editing.cuts import find_hallucinations
from kurgu.editing.shorts import sentences
from kurgu.motion.captions import displayable


def _words(text, t0, prob=0.95, step=0.4):
    return [{"word": w, "t_start": t0 + k * step, "t_end": t0 + k * step + 0.3, "start": t0 + k * step,
             "end": t0 + k * step + 0.3, "prob": prob} for k, w in enumerate(text.split())]


def test_requests_found_where_said():
    w = (_words("Bu sos çok acı ama seviyorum.", 0) + _words("Yemeğe geçmeden önce videoyu beğenin.", 10)
         + _words("Siz de yorumlara yazın.", 13) + _words("Etler geldi dostlar.", 60)
         + _words("İzliyorsunuz ama abone olmuyorsunuz, kanalı büyütelim.", 120)
         + _words("abone ol", 200, prob=0.52))                       # Whisper over the music
    plan = cta.plan_requests(sentences(w), 260.0)
    assert [[k for k, _ in p["items"]] for p in plan] == [["like", "comment"], ["subscribe"]]
    assert 10.0 <= plan[0]["t0"] <= 11.6 and plan[1]["items"][0][1] > 120


def test_cta_widget_draws_and_clicks():
    from kurgu.motion.cta import CTA
    c = CTA([("subscribe", 101.0), ("like", 103.0)], 100.0, 105.0, 1280, 720, "left")
    assert [b["kind"] for b in c.btn] == ["subscribe", "bell", "like"]
    assert c.btn[0]["click"] < c.btn[1]["click"] < c.btn[2]["click"]
    bg = np.zeros((720, 1280, 3), np.uint8)
    mid = c.draw(bg, 1.5)
    assert mid.any()
    names = [n for _, n, _ in c.sounds]
    assert "bell" in names and "pop" in names and names[0] == "whoosh"


def test_invented_subscribe_text_removed():
    w = _words("Etler geldi.", 0) + [{"word": "abone", "start": 5.0, "end": 5.3, "prob": 0.52},
                                     {"word": "ol", "start": 5.3, "end": 5.4, "prob": 0.99}] + _words("Şimdi yiyoruz.", 7.0)
    rem, idx = find_hallucinations(w, set())
    assert len(rem) == 1 and idx == {2, 3}
    real = _words("Lütfen abone olun.", 0, prob=0.8)
    assert find_hallucinations(real, set())[1] == set()
    assert not displayable({"word": "suççççççç", "prob": 0.9}) and not displayable({"word": ",7", "prob": 0.4})
    assert displayable({"word": "167", "prob": 0.9}) and displayable({"word": "Abone", "prob": 0.8})
