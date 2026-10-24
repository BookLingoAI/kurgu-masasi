"""Automatic intro from the talk recordings: shot filters, camera-move kinds, repeat detection."""
import numpy as np

from kurgu.editing import broll_mine as bm


def _m(**kw):
    m = {"face": 0.0, "body": 0.0, "body_top": 0.0, "body_rows": 0.0, "sharp": 500.0, "bright": 120.0,
         "colour": 40.0, "structure": 0.1, "dx": 0.0, "dy": 0.0}
    m.update(kw)
    return m


def test_clean_rules():
    assert bm.clean(_m())                                        # a colourful, sharp plate
    assert not bm.clean(_m(face=0.3))                            # the speaker talking
    assert not bm.clean(_m(body=0.5, body_rows=0.9))             # legs / body fill the frame
    assert not bm.clean(_m(body=0.7, body_rows=0.8))             # a person fills most of the frame
    assert not bm.clean(_m(body=0.45, body_rows=0.7, colour=26)) # big dull body part
    assert bm.clean(_m(body=0.45, body_rows=0.72, colour=41))    # a dessert held up to the lens (segmented as "person")
    assert not bm.clean(_m(body=0.3, body_rows=0.8, body_top_c=0.2))   # head / torso in the middle
    assert not bm.clean(_m(body=0.2, colour=15))                 # dull body part: shoe, trousers, pocket
    assert bm.clean(_m(body=0.3, body_rows=0.7, colour=40))      # a hand holding a dessert
    assert not bm.clean(_m(body=0.22, body_rows=0.95))           # a person in profile (face not detected)
    assert not bm.clean(_m(bright=20)) and not bm.clean(_m(sharp=10))
    assert not bm.clean(_m(colour=5, structure=0.01))            # empty floor / ceiling


def test_motion_kinds():
    steady = [_m(dx=0.1, dy=0.05) for _ in range(7)]
    tilt = [_m(dx=0.2, dy=3.0 + 0.1 * k) for k in range(7)]                 # slow deliberate tilt up
    shaky = [_m(dx=(-1) ** k * 6.0, dy=(-1) ** (k // 2) * 5.0) for k in range(7)]
    spin = [_m(dx=40.0, dy=40.0) for _ in range(7)]
    assert bm.motion_kind(steady) == "steady"
    assert bm.motion_kind(tilt) == "smooth"
    assert bm.motion_kind(shaky) == "shaky"
    assert bm.motion_kind(spin) == "shaky"


def _plate(colour):
    img = np.full((180, 320, 3), 235, np.uint8)            # white rice everywhere
    img[60:120, 110:210] = colour                           # the filling in the middle
    img[:, :40] = (90, 60, 40)                              # same restaurant table at the side
    return img


def test_fillings_differ_rice_ignored():
    import cv2
    salmon, salmon2, crab = _plate((250, 120, 60)), _plate((245, 125, 70)), _plate((200, 30, 40))
    same = cv2.compareHist(bm.food_hist(salmon), bm.food_hist(salmon2), cv2.HISTCMP_CORREL)
    diff = cv2.compareHist(bm.food_hist(salmon), bm.food_hist(crab), cv2.HISTCMP_CORREL)
    assert same > 0.9 > diff


def test_spoken_names_only_when_showing():
    words = [{"word": w, "start": 10 + i * 0.4} for i, w in enumerate("yeni suşiler geldi gerçek yengeç".split())]
    assert "yengeç" in bm.spoken_names(words, 10, 12)
    talk = [{"word": w, "start": 10 + i * 0.4} for i, w in enumerate("kuzu eti candır abi".split())]
    assert bm.spoken_names(talk, 10, 12) == set()           # talking while eating names nothing


def test_candidates_from_face_timeline():
    pl = {"camera": {"parts": [{"path": "a.mp4", "start": 0.0, "duration": 60.0}]}}
    tl = [(float(t), 0.3) for t in range(0, 20)] + [(float(t), 0.0) for t in range(20, 31)] + \
        [(float(t), 0.3) for t in range(31, 60)]
    c = bm.candidate_windows(pl, {"a.mp4": tl}, min_len=2.4)
    assert c and all(19.5 <= x["in"] < x["out"] <= 31.7 for x in c)
    assert sum(x["out"] - x["in"] for x in c) >= 10


def test_pick_drops_same_scene_and_orders():
    base = {"motion": "steady", "why": "gösterme", "_mid": _plate((250, 120, 60))}
    shots = [dict(base, file="a.mp4", **{"in": 10.0, "out": 12.0, "score": 0.9, "vt0": 10.0, "_hist": bm.food_hist(_plate((250, 120, 60)))}),
             dict(base, file="a.mp4", **{"in": 15.0, "out": 17.0, "score": 0.8, "vt0": 15.0, "_hist": bm.food_hist(_plate((40, 160, 60)))}),
             dict(base, file="b.mp4", **{"in": 5.0, "out": 7.0, "score": 0.7, "vt0": 100.0, "_mid": _plate((200, 30, 40)),
                                          "_hist": bm.food_hist(_plate((200, 30, 40)))})]
    chosen, notes = bm.pick(shots, [], 10)
    assert [c["file"] for c in chosen] == ["a.mp4", "b.mp4"] and notes and "aynı sahne" in notes[0]


def test_faces_run_without_mediapipe_runtime(monkeypatch):
    """Smart App Control blocked MediaPipe's DLL: the same models now run in OpenCV, MediaPipe is optional."""
    import numpy as np
    import pytest

    from kurgu.vision import faces
    if not (faces.model_path("face") and faces.model_path("segment")):
        pytest.skip("modeller klasörü yok")
    monkeypatch.setattr(faces, "_mp", lambda: None)
    faces._cv_ok.cache_clear()
    assert faces.backend() == "opencv"
    det = faces.FaceDetector()
    assert det.ok and det.detect(np.zeros((360, 640, 3), np.uint8)) == []
    m = faces.person_mask(np.zeros((360, 640, 3), np.uint8))
    assert m is not None and m.shape == (360, 640) and float(m.max()) < 0.5
    assert faces.yaw_from_face({"kp": [(0.4, 0.4), (0.6, 0.4), (0.5, 0.5)]}) == 0.0
