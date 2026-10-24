import unittest

import numpy as np

from kurgu.analysis.screen import analyze_frames
from kurgu.config import Config

CFG = Config.load().design["screen_activity"]


def frames(n=60, fps=5, cursor_path=None, video_box=None, video_from=0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((360, 640), 30, np.uint8)
    base[40:60, 50:400] = 200  # static text bar
    for i in range(n):
        g = base.copy()
        if video_box and i >= video_from:
            y0, y1, x0, x1 = video_box
            g[y0:y1, x0:x1] = rng.integers(0, 255, (y1 - y0, x1 - x0), dtype=np.uint8)
        if cursor_path:
            x, y = cursor_path(i)
            g[int(y):int(y) + 12, int(x):int(x) + 8] = 255
        yield i / fps, g


class TestScreen(unittest.TestCase):
    def test_cursor_from_motion_and_hold(self):
        path = lambda i: (100 + 6 * min(i, 30), 200)  # moves for 30 frames then stops
        r = analyze_frames(frames(cursor_path=path), CFG)
        moving = [c for c in r["cursor"] if c["moving"] and c["reliable"]]
        self.assertGreater(len(moving), 20)
        # tracked x close to truth (box centre is +4 px)
        c = moving[10]
        i = int(round(c["t"] * 5)) - 1  # three-frame diff locates prev1
        self.assertAlmostEqual(c["x"] * 640, 100 + 6 * i + 4, delta=8)
        held = [c for c in r["cursor"] if not c["moving"]]
        self.assertTrue(held and held[-1]["x"] is not None)

    def test_video_playing_detected(self):
        r = analyze_frames(frames(n=50, video_box=(100, 300, 200, 500), video_from=10), CFG)
        self.assertTrue(r["video_playing"])
        a, b = r["video_playing"][0]
        self.assertLess(a, 3.0)
        self.assertGreater(b, 8.0)

    def test_static_screen_no_video(self):
        r = analyze_frames(frames(n=30), CFG)
        self.assertEqual(r["video_playing"], [])

    def test_unreliable_when_video_and_no_cursor(self):
        r = analyze_frames(frames(n=30, video_box=(0, 360, 0, 640)), CFG)
        self.assertTrue(all(not c["reliable"] or c["x"] is None for c in r["cursor"]))


if __name__ == "__main__":
    unittest.main()
