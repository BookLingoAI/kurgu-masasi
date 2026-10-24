"""Shorts summary, mastering, animation keys, grading."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kurgu.editing.shorts import heuristic_summary, plan_summary, sentences, validate_summary


def make_words(n=40):
    words, t = [], 0.0
    for i in range(n):
        txt = f"neden {i} dolar kazandım?" if i % 7 == 0 else f"bugün {i}. işe gittim ve çalıştım."
        for w in txt.split():
            words.append({"word": w, "t_start": t, "t_end": t + 0.4})
            t += 0.45
        t += 1.0
    return words, t


class SummaryTests(unittest.TestCase):
    def test_summary_story(self):
        w, T = make_words()
        s = sentences(w)
        v = validate_summary(heuristic_summary(s, 45, 90), s, 45, 90)
        self.assertLessEqual(v["duration"], 90.5)
        self.assertGreaterEqual(v["duration"], 20)
        starts = [c["start"] for c in v["clips"]]
        self.assertEqual(starts, sorted(starts))
        self.assertFalse(any(c["start"] <= v["hook"] <= c["end"] for c in v["clips"]))
        sp = plan_summary(s, v, [{"file": "a", "in": 0, "out": T, "timeline_start": 0, "timeline_end": T}])
        self.assertEqual(len(sp["cuts"]), len(v["clips"]))  # hook + clips -> one cut before every clip

    def test_llm_pick_repeating_hook_fixed(self):
        w, _ = make_words()
        s = sentences(w)
        v = validate_summary({"hook": 5, "clips": [{"start": 4, "end": 6, "role": "giriş"}, {"start": 30, "end": 31}]}, s, 10, 90)
        self.assertFalse(any(c["start"] <= 5 <= c["end"] for c in v["clips"]))


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg yok")
class MasterTests(unittest.TestCase):
    def test_master_levels_and_sfx(self):
        from kurgu.audio.master import builtin_sfx, master
        d = Path(tempfile.mkdtemp())
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=6", "-af", "volume=0.1",
                        "-ar", "48000", "-ac", "2", str(d / "v.wav")], check=True)
        fx = builtin_sfx(d / "fx")
        r = master(str(d / "v.wav"), str(d / "o.wav"), sfx=[{"t": 1.0, "file": fx["whoosh"], "gain_db": -10}])
        self.assertIn("ses efekti", r["mastering"])
        self.assertTrue((d / "o.wav").exists())


class KeysTests(unittest.TestCase):
    def test_pop_keys_keep_centre(self):
        from kurgu.integrations.premiere_edit import overlay_keys
        from kurgu.motion.render import Overlay
        rgba = np.zeros((1080, 1920, 4), np.uint8)
        rgba[100:200, 1500:1800, 3] = 255
        k = overlay_keys(Overlay(1.0, 3.0, rgba, anim="pop"), 1920, 1080)
        bx, by = 1650 / 1920, 150 / 1080
        for t, sc, x, y in k:  # centre of the graphic stays put while scaling
            self.assertAlmostEqual(x + sc / 100 * (bx - 0.5), bx, places=3)
            self.assertAlmostEqual(y + sc / 100 * (by - 0.5), by, places=3)


class GradeTests(unittest.TestCase):
    def test_grade_lifts_dim_footage(self):
        from kurgu.vision.grade import apply_grade, compute_grade
        rng = np.random.default_rng(1)
        f = np.clip(rng.normal(70, 25, (360, 640, 3)), 0, 255).astype(np.uint8)
        g = compute_grade([f])
        out = apply_grade(f, g)
        self.assertGreater(out.mean(), f.mean() + 5)
        self.assertLess(out.mean(), 160)


if __name__ == "__main__":
    unittest.main()
