"""Director (screen switch, b-roll, cards, sfx) + compositor render on synthetic media."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kurgu.editing import director as dr
from kurgu.integrations import ffmpeg as ff

HAS_FF = shutil.which("ffmpeg") is not None or Path(ff.ffmpeg_bin()).exists()


def seg(i, t0, t1, screen=True):
    return {"id": i, "file": "cam.mp4", "in": t0, "out": t1, "timeline_start": t0, "timeline_end": t1,
            "screen": {"file": "scr.mp4", "in": t0 + 2.0, "out": t1 + 2.0} if screen else None}


def words(text, t0=0.0, step=0.4):
    out = []
    for k, w in enumerate(text.split()):
        out.append({"word": w, "t_start": t0 + k * step, "t_end": t0 + k * step + 0.3})
    return out


class DirectorTests(unittest.TestCase):
    def test_screen_span_needs_gaze_and_activity(self):
        segs = [seg(0, 0, 30)]
        # looks to image-left (user's right) between 10 and 16 s
        gaze = [{"t": t / 2, "yaw": (-0.35 if 20 <= t < 32 else 0.0)} for t in range(60)]
        act = {"scr.mp4": {"samples": [{"t": 2 + t / 5, "changed": 0.01 if 50 <= t <= 95 else 0.0} for t in range(150)]}}
        ev, notes = dr.screen_spans(gaze, segs, [], act, "right")
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0]["t0"], 9.8, delta=0.3)
        self.assertAlmostEqual(ev[0]["t1"], 16.0, delta=0.6)
        # static screen and no pointing words -> stays on camera
        act0 = {"scr.mp4": {"samples": [{"t": 2 + t / 5, "changed": 0.0} for t in range(150)]}}
        ev0, notes0 = dr.screen_spans(gaze, segs, [], act0, "right")
        self.assertEqual(ev0, [])
        self.assertTrue(notes0)
        # ... unless the speaker points at it ("bakın burada")
        ev1, _ = dr.screen_spans(gaze, segs, words("bakın burada şunu yapıyoruz", 11.0), act0, "right")
        self.assertEqual(len(ev1), 1)

    def test_no_switch_without_screen_recording(self):
        segs = [seg(0, 0, 30, screen=False)]
        gaze = [{"t": t / 2, "yaw": (-0.35 if 20 <= t < 32 else 0.0)} for t in range(60)]
        ev, _ = dr.screen_spans(gaze, segs, [], {}, "right")
        self.assertEqual(ev, [])

    def test_mirrored_camera_detected(self):
        segs = [seg(0, 0, 60)]
        gaze = [{"t": t / 2, "yaw": (0.4 if t % 10 < 6 and t > 20 else 0.0)} for t in range(120)]
        act = {"scr.mp4": {"samples": [{"t": 2 + t / 5, "changed": 0.01} for t in range(320)]}}
        ev, notes = dr.screen_spans(gaze, segs, [], act, "right")
        self.assertTrue(any("ters" in n for n in notes))

    def test_broll_matching_and_spacing(self):
        w = words("bugün size kahve makinesi nasıl temizlenir anlatacağım . sonra bahçede çiçek sulama konusuna geçelim .", 0, 1.0)
        w[7]["word"] = "anlatacağım."
        w[-1]["word"] = "geçelim."
        items = [{"file": "/x/kahve_makinesi.mp4", "note": ""}, {"file": "/x/IMG_1234.jpg", "note": ""},
                 {"file": "/x/c.jpg", "note": "çiçek sulama"}]
        sim = lambda a, b: 0.8 if (("kahve" in a and "kahve" in b) or ("çiçek" in a and "çiçek" in b)) else 0.1  # noqa: E731
        opts = dict(dr.DEFAULT_OPTIONS, intensity="high", subscribe=False)
        out = dr.plan_events([seg(0, 0, 20, False)], w, 20.0, opts, [], items, {"/x/kahve_makinesi.mp4": {"duration": 3.0}}, None, sim)
        files = [e["file"] for e in out["events"] if e["type"] == "broll"]
        self.assertIn("/x/c.jpg", files)
        self.assertTrue(any("IMG_1234" in n for n in out["notes"]))
        self.assertIn("Dil modeli yoktu: tam ekran başlık kartı eklenmedi", out["notes"])

    def test_sfx_and_subscribe(self):
        w = []
        for k in range(40):
            text = "lütfen kanalıma abone olun dostlar." if k == 20 else f"cümle numara {k} burada bitiyor."
            w += words(text, k * 5.0, 0.8)
        opts = dict(dr.DEFAULT_OPTIONS)
        out = dr.plan_events([seg(0, 0, 200, False)], w, 200.0, opts, [{"type": "screen", "t0": 10, "t1": 20}], [], {}, None, None)
        subs = [e for e in out["events"] if e["type"] == "subscribe"]
        self.assertEqual(len(subs), 1)                       # only where it is asked for
        self.assertTrue(99 <= subs[0]["t0"] <= 101.5 and subs[0]["items"][0][0] == "subscribe")
        cues = dr.sfx_cues(out["events"], ["a.wav", "b.wav"], -17)
        self.assertEqual([c["file"] for c in cues], ["a.wav", "b.wav"])
        self.assertEqual(dr.sfx_cues(out["events"], [], -17), [])
        w2 = [x for k in range(40) for x in words(f"cümle numara {k} burada bitiyor.", k * 5.0, 0.8)]
        out2 = dr.plan_events([seg(0, 0, 200, False)], w2, 200.0, opts, [], [], {}, None, None)
        self.assertFalse([e for e in out2["events"] if e["type"] == "subscribe"])   # never at random places

    def test_cursor_follow_smooth(self):
        cur = [{"t": t / 5, "x": 0.8, "y": 0.2, "moving": True, "reliable": True} for t in range(50)]
        f = dr.CursorFollow(cur, 1.4)
        zs = [f.step(t / 30)[0] for t in range(1, 240)]
        self.assertGreater(zs[-1], 1.35)
        self.assertTrue(all(b - a < 0.05 for a, b in zip(zs, zs[1:])))  # no jumps


@unittest.skipUnless(HAS_FF, "ffmpeg yok")
class ComposeRenderTest(unittest.TestCase):
    def test_render_screen_broll_card_sfx(self):
        from PIL import Image

        from kurgu.motion.compose import Compositor, overlays_for
        from kurgu.motion.render import Piece, render
        d = Path(tempfile.mkdtemp())
        ffb = ff.ffmpeg_bin()
        run = lambda *a: subprocess.run([ffb, "-v", "error", "-y", *a], check=True)  # noqa: E731
        run("-f", "lavfi", "-i", "color=c=0x3060a0:s=640x360:r=30:d=8", "-f", "lavfi", "-i", "sine=f=300:d=8",
            "-shortest", "-pix_fmt", "yuv420p", str(d / "cam.mp4"))
        run("-f", "lavfi", "-i", "testsrc2=s=800x500:r=30:d=12", "-pix_fmt", "yuv420p", str(d / "scr.mp4"))
        run("-f", "lavfi", "-i", "color=c=green:s=320x240:r=30:d=3", "-pix_fmt", "yuv420p", str(d / "broll.mp4"))
        run("-f", "lavfi", "-i", "sine=f=1200:d=0.4", str(d / "whoosh.wav"))
        Image.new("RGB", (400, 300), (200, 30, 30)).save(d / "foto.jpg")
        events = [{"type": "screen", "t0": 0.5, "t1": 2.5}, {"type": "broll", "t0": 3.0, "t1": 4.5, "file": str(d / "broll.mp4"), "image": False},
                  {"type": "broll", "t0": 5.0, "t1": 6.0, "file": str(d / "foto.jpg"), "image": True},
                  {"type": "card", "t0": 6.2, "t1": 7.0, "text": "1. kurulum"}, {"type": "subscribe", "t0": 7.0, "t1": 7.9}]
        pieces = [Piece(str(d / "cam.mp4"), 0.0, 4.0, screen={"file": str(d / "scr.mp4"), "in": 2.0}),
                  Piece(str(d / "cam.mp4"), 4.0, 8.0)]
        comp = Compositor(events, {}, 640, 360, 30, {})
        out = d / "out.mp4"
        res = render(pieces, str(out), 640, 360, 30, overlays=overlays_for(events, 640, 360), compose=comp,
                     sfx=[{"t": 0.4, "file": str(d / "whoosh.wav"), "gain_db": -10}])
        comp.close()
        self.assertEqual(res["frames"], 240)
        self.assertGreater(comp.used["screen_frames"], 50)
        self.assertGreater(comp.used["broll_frames"], 50)

        def frame(t):
            p = subprocess.run([ffb, "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-"], capture_output=True)
            return np.frombuffer(p.stdout, np.uint8).reshape(360, 640, 3).astype(int)
        cam = frame(7.95 - 0.5)  # plain camera (blue) area before... subscribe overlay is centred, corner is camera
        self.assertTrue(abs(cam[5, 5] - [48, 96, 160]).max() < 25)
        scr = frame(1.5)
        box = scr[360 - 20, 30]          # speaker box (camera blue) bottom-left
        self.assertTrue(abs(box - [48, 96, 160]).max() < 30, box)
        self.assertGreater(np.abs(scr[50:150, 300:500] - [48, 96, 160]).mean(), 30)  # screen content, not camera
        g = frame(3.8)
        self.assertGreater(g[180, 320, 1], 100)  # green b-roll
        self.assertLess(g[180, 320, 0], 60)
        r = frame(5.6)
        self.assertGreater(r[180, 320, 0], 150)  # red photo
        c = frame(6.6)
        self.assertLess(c[10, 10].max(), 40)     # dark card
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
