"""Premiere full-edit spec (pure Python parts) and ES3 syntax."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kurgu.integrations import premiere_edit as pe
from kurgu.motion.render import Overlay, Piece


class SpecTests(unittest.TestCase):
    def test_zoom_pos_matches_render_crop(self):
        # render.zoom_frame keeps source point c (clamped) in the centre; Premiere position must do the same
        z, cx, cy = 1.15, 0.55, 0.45
        x, y = pe._zoom_pos(z, cx, cy)
        self.assertAlmostEqual(x + z * (cx - 0.5), 0.5, places=6)
        self.assertAlmostEqual(y + z * (cy - 0.5), 0.5, places=6)
        x2, _ = pe._zoom_pos(1.1, 0.99, 0.5)          # face near the edge: frame edge must stay covered
        self.assertLessEqual(abs(x2 - 0.5) , 0.05 + 1e-9)
        self.assertEqual(pe._zoom_pos(1.0, 0.9, 0.9), (0.5, 0.5))

    def test_build_spec_tracks(self):
        d = Path(tempfile.mkdtemp())
        segs = [{"file": "cam.mp4", "in": 0.0, "out": 5.0, "timeline_start": 0.0, "timeline_end": 5.0,
                 "screen": {"file": "scr.mp4", "in": 2.0, "out": 7.0}},
                {"file": "cam.mp4", "in": 6.0, "out": 10.0, "timeline_start": 5.0, "timeline_end": 9.0,
                 "screen": {"file": "scr.mp4", "in": 8.0, "out": 12.0}}]
        events = [{"type": "screen", "t0": 3.0, "t1": 7.0}, {"type": "broll", "t0": 7.5, "t1": 8.5, "file": "b.jpg", "image": True}]
        ov = Overlay(1.0, 2.0, np.zeros((108, 192, 4), np.uint8) + 255)
        ctx = {"w": 192, "h": 108, "fps": 30, "segs": segs, "events": events, "overlays": [ov],
               "pieces": [Piece("cam.mp4", 0, 5, zoom=1.08, cx=0.6, cy=0.4), Piece("cam.mp4", 6, 10)],
               "d": {"plan": {"sfx": [{"t": 2.9, "file": "w.wav", "gain_db": -17}]}}}
        cursor = [{"t": 2 + k / 5, "x": 0.8, "y": 0.2, "moving": True, "reliable": True} for k in range(60)]
        spec = pe.build_spec(ctx, {"__timeline__": "kamera_kurgu.mp4"}, {"scr.mp4": (384, 216), "b.jpg": (400, 300)},
                             {"scr.mp4": {"cursor": cursor}}, d, {}, {"w.wav": ("sfx_00.wav", 0.4)})
        self.assertEqual(len(spec["v1"]), 2)
        self.assertEqual(spec["v1"][0]["f"], "kamera_kurgu.mp4")   # one retouched camera-timeline file
        self.assertAlmostEqual(spec["v1"][1]["i"], segs[1]["timeline_start"])
        self.assertAlmostEqual(spec["v1"][0]["s"], 108.0)
        self.assertEqual(len(spec["v2"]), 3)                    # screen split at the cut + one photo
        a, b = spec["v2"][0], spec["v2"][1]
        self.assertAlmostEqual(a["i"], 5.0)                     # screen time = 2 + (3 - 0)
        self.assertAlmostEqual(b["i"], 8.0)
        self.assertEqual((a["fi"], a["fo"], b["fi"], b["fo"]), (0.25, 0, 0, 0.25))
        self.assertAlmostEqual(a.get("s", 0), 70.0)             # fit 0.5 x cursor zoom 1.4 (steady pointer)
        self.assertAlmostEqual(a["x"], 0.3, places=3)           # pointer at the right -> image shifted left,
        self.assertAlmostEqual(a["x"] * 192 + 0.7 * 384 / 2, 192, places=1)  # right edge exactly at frame edge
        self.assertEqual(len(spec["v3"]), 2)
        self.assertAlmostEqual(spec["v3"][0]["i"], 3.0)
        self.assertEqual(len(spec["v4"]), 1)
        self.assertEqual(len(spec["v5"]), 1)
        self.assertTrue(Path(spec["v5"][0]["png"]).exists())
        self.assertEqual(spec["a4"], [{"f": "sfx_00.wav", "i": 0.0, "o": 0.4, "t": 2.9, "db": 0.0}])
        shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(shutil.which("node"), "node yok")
class JsSyntaxTest(unittest.TestCase):
    def test_lib_and_place_parse(self):
        lib = (Path(pe.__file__).parent / "es3" / "kurgu_lib.jsx").read_text(encoding="utf-8")
        body = pe.PLACE_JS.replace("NAME", '"x"').replace("DATA", "[]").replace("TRACK", "seq.videoTracks[0]")
        src = "function f(){" + lib + "\n" + body + "}"
        p = subprocess.run(["node", "-e", "new Function(require('fs').readFileSync(0,'utf8'))"], input=src.encode(),
                           capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr.decode()[-500:])


if __name__ == "__main__":
    unittest.main()
