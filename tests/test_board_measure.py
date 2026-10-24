"""Review board measurements (kurgu/board/measure.py) on a small synthetic video with known events:
a jump cut with the same framing, a punch-in cut, a title that arrives and then stands still, a tone at -14 LUFS."""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from kurgu.board import measure as M
from kurgu.decisions import DecisionLog
from kurgu.integrations import ffmpeg as ff

FPS, W, H = 30, 640, 360


def _scene(seed=7):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 6)
    img = cv2.normalize(img, None, 40, 200, cv2.NORM_MINMAX)
    cv2.circle(img, (W // 2, H // 2 - 20), 70, (180, 140, 120), -1)          # a "face"
    return img


def _zoom(img, z):
    ch, cw = int(H / z), int(W / z)
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    """6 s: 0-2 s shot A, 2-4 s shot A again (jump cut), 4-6 s shot A punched in 1.25x (zoom cut).
    A title box on a dark panel arrives at 0.5 s (0.5 s wipe), then stands still until 3.5 s."""
    d = tmp_path_factory.mktemp("board")
    (d / "cikti").mkdir()
    (d / "logs").mkdir()
    base = _scene()
    silent = d / "v.avi"
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(6 * FPS):
        t = i / FPS
        f = base.copy() if t < 4 else _zoom(base, 1.25)
        f = np.roll(f, 4 + int(2 * np.sin(t * 7)), axis=1) if t >= 2 else f  # the speaker moved a little
        if 0.5 <= t < 3.5:
            cv2.rectangle(f, (20, 100), (300, 260), (20, 20, 20), -1)
            k = min(1.0, (t - 0.5) / 0.5)
            cv2.putText(f, "BASLIK", (40, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
            f[100:260, 20 + int(280 * k):300] = 20                            # wiped in from the left
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    video = d / "cikti" / "kurgu.mp4"
    subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-i", str(silent), "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=6", "-af", "volume=-12dB",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)], check=True)
    pieces = [{"t0": 0.0, "t1": 2.0, "file": "a", "zoom": 1.0, "trans": None, "trans_in": 0.0, "speed": 1.0, "kind": "main"},
              {"t0": 2.0, "t1": 4.0, "file": "a", "zoom": 1.0, "trans": None, "trans_in": 0.0, "speed": 1.0, "kind": "main"},
              {"t0": 4.0, "t1": 6.0, "file": "a", "zoom": 1.25, "trans": None, "trans_in": 0.0, "speed": 1.0, "kind": "main"}]
    man = {"version": 1, "w": W, "h": H, "fps": FPS, "pieces": pieces,
           "overlays": [{"t0": 0.5, "t1": 3.5, "kind": "Topic", "box": [20, 100, 300, 260]}],
           "sfx": [{"t": 1.0, "name": "whoosh", "gain_db": -12}]}
    words = [{"word": w, "t_start": 0.2 + k * 0.5, "t_end": 0.6 + k * 0.5} for k, w in enumerate("bugün size bir şey anlatacağım".split())]
    for fn, data in (("kurgu.json", {"file": str(video), "duration": 6.0}), ("kurgu_manifest.json", man),
                     ("edit_plan.json", {"segments": [], "transcript_cut": words})):
        (d / fn).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    class P:
        dir = d
        log = DecisionLog(d)

        def load(self, fn, default=None):
            p = d / fn
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    return P()


@pytest.fixture(scope="module")
def rep(proj):
    return M.measure(proj)


def test_report_has_every_agent_and_is_saved(proj, rep):
    assert all(a in rep and "metrics" in rep[a] and "flags" in rep[a] for a in M.AGENTS)
    saved = json.loads((proj.dir / "kurul" / "olcum.json").read_text(encoding="utf-8"))
    assert saved["duration_s"] == 6.0 and saved["manifest"] is True and saved["missing"] == []
    assert (proj.dir / "kurul" / "kanit_ses.jpg").exists()


def test_title_entry_and_still_hold_are_measured(rep):
    a = rep["animasyon"]["metrics"]["yazilar"][0]
    assert 0.3 <= a["giris_sn"] <= 0.8                      # the 0.5 s wipe
    assert 2.0 <= a["en_uzun_donuk_sn"] <= 3.0              # then still until it goes
    assert any(f["tur"] == "donuk" for f in rep["animasyon"]["flags"])


def test_jump_cut_found_but_punch_in_is_not(rep):
    g = rep["gecis"]["metrics"]
    assert g["kesim"] == 2 and g["ayni_kadraj_kesim"] == 1
    assert [round(f["t"]) for f in rep["gecis"]["flags"]] == [2]


def test_loudness_of_the_tone(rep):
    s = rep["ses"]["metrics"]
    assert s["I_LUFS"] is not None and -45 < s["I_LUFS"] < -5 and s["TP_dBTP"] is not None
    assert s["efekt_sayisi"] == 1


def test_framing_tells_zoom_from_jump():
    g = cv2.cvtColor(cv2.resize(_scene(3), (160, 90)), cv2.COLOR_RGB2GRAY)
    sim, scale = M._framing(g, np.roll(g, 1, axis=1))
    assert sim > 0.9 and scale == 1.0
    sim, scale = M._framing(g, _zoom_gray(g, 1.2))
    assert scale != 1.0


def _zoom_gray(g, z):
    h, w = g.shape
    ch, cw = int(h / z), int(w / z)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(g[y0:y0 + ch, x0:x0 + cw], (w, h))


def test_runs():
    m = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1], bool)
    assert M._runs(m, 2) == [(1, 4), (7, 9)]


def test_manifest_of_a_render_context():
    from kurgu.motion.render import Overlay, Piece
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[50:80, 100:200, 3] = 255
    ctx = {"pieces": [Piece("a.mp4", 0, 2), Piece("a.mp4", 5, 6, zoom=1.2, trans_in=0.2, trans="whip")],
           "overlays": [Overlay(1.0, 2.5, rgba, anim="pop")], "w": W, "h": H, "fps": 30.0,
           "d": {"plan": {"sfx": [{"t": 2.0, "file": "C:/x/whoosh_2.wav", "gain_db": -10}]}}}
    man = M.manifest_of(ctx)
    assert [(p["t0"], p["t1"], p["trans"]) for p in man["pieces"]] == [(0.0, 2.0, None), (2.0, 3.0, "whip")]
    assert man["overlays"][0]["box"] == [100, 50, 200, 80] and man["overlays"][0]["kind"] == "pop"
    assert man["sfx"] == [{"t": 2.0, "name": "whoosh_2", "gain_db": -10}]
    json.dumps(man)
