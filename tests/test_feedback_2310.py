"""User feedback 23.10.2026 (Japon hot pot, karma engine): offered food kept, misheard words never on screen,
narration is not a title, library sound effects lead the rotation, HyperFrames graphics have a Premiere still."""
from pathlib import Path

from kurgu.audio import master
from kurgu.editing import showing
from kurgu.editing.graphics import grounded


def _w(word, start, end, prob=0.95):
    return {"word": word, "start": start, "end": end, "t_start": start, "t_end": end, "prob": prob}


class _Part:
    path, duration = "yok.mp4", 600.0


class _Cam:
    def locate(self, t):
        return _Part(), t


class _NoVision:
    class faces:
        ok = False


def test_offered_food_is_kept_without_looking_at_the_picture(monkeypatch):
    from kurgu.editing import broll_mine
    monkeypatch.setattr(broll_mine, "_Vision", _NoVision)   # the camera check is not needed for an offer
    words = [_w("Açın", 10.0, 10.3), _w("ağzınızı.", 10.3, 10.8), _w("Nasıl?", 14.9, 15.3),
             _w("Bakın", 30.0, 30.3), _w("burası", 30.3, 30.7), _w("güzel", 31.4, 31.8)]
    spans = showing.find_showing(words, _Cam())
    assert spans == [(10.8, 14.9)]          # the whole hold to the lens, not only "a sharp picture"


def test_short_offer_pause_is_kept_too(monkeypatch):
    from kurgu.editing import broll_mine
    monkeypatch.setattr(broll_mine, "_Vision", _NoVision)
    words = [_w("buyurun", 5.0, 5.4), _w("ye", 6.2, 6.4)]
    assert showing.find_showing(words, _Cam()) == [(5.4, 6.2)]


def test_misheard_word_never_becomes_a_title():
    s = {"words": [_w("Diner", 1.0, 1.3, prob=0.31), _w("yiyorum", 1.3, 1.8)]}
    assert not grounded("DİNER YİYORUM", s)
    s2 = {"words": [_w("Gerçek", 0, 0.4), _w("yengeç", 0.4, 0.8, prob=0.4), _w("suşi", 0.8, 1.2)]}
    assert not grounded("GERÇEK YENGEÇ SUŞİ", s2)     # "yengeç" was a guess
    s3 = {"words": [_w("Gerçek", 0, 0.4), _w("yengeç", 0.4, 0.8), _w("suşi", 0.8, 1.2)]}
    assert grounded("GERÇEK YENGEÇ SUŞİ", s3)


def test_narration_is_not_a_title():
    s = {"words": [_w("dil", 0, 0.3), _w("yiyorum", 0.3, 0.8)]}
    assert not grounded("DİL YİYORUM", s)


def test_library_sounds_lead_and_cta_stays_fixed(tmp_path):
    bank = master.builtin_sfx()
    lib = Path(master.ASSETS.parent / "paketler" / "efekt")
    if not any(lib.glob("mixkit-*")):
        return   # no downloaded library on this machine: the synth set alone is fine
    assert "__" in Path(bank["whoosh"]).name                 # first whoosh is a downloaded one
    assert Path(bank.fixed("click")).name == "click.wav"      # subscribe/like sounds keep their fixed sound


def test_hyperframes_overlay_has_a_still_for_premiere(tmp_path):
    from kurgu.motion import hf_graphics as hg
    if not hg.available():
        return
    ov, _info = hg.overlay({"kind": "title", "t0": 1.0, "text": "SOS ÇOK ACI", "kicker": "YORUM", "idx": 2},
                           (255, 196, 0), tmp_path, 1280, 720, 30)
    img = ov.meta["static"]()
    assert img.shape == (720, 1280, 4) and img[..., 3].max() > 200
    ov.meta["draw"].close()


def test_same_sound_chunk_is_transcribed_once():
    """Verification rounds 2-3 re-use the chunks whose sound did not change (speed, 23.10.2026)."""
    import numpy as np
    from kurgu.analysis import transcript as T
    eng = T.WhisperEngine({"model": "test-model", "beam_size": 1})
    calls = []
    eng._transcribe_retry = lambda a, sr, p=None: calls.append(1) or [{"start": 0, "end": 1, "text": "x", "words": [{"word": "x", "start": 0, "end": 1, "prob": 1}]}]
    a = np.random.default_rng(1).standard_normal(16000).astype(np.float32)
    r1 = eng.transcribe(a, 16000, "p")
    r1[0]["words"][0]["start"] = 99          # a caller changing the result must not change the memo
    r2 = eng.transcribe(a.copy(), 16000, "p")
    assert len(calls) == 1 and r2[0]["words"][0]["start"] == 0
    eng.transcribe(a, 16000, "başka")         # another prompt is another job
    assert len(calls) == 2
