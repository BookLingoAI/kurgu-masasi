import json
import shutil
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from kurgu.config import Config
from kurgu.editing import metadata as md
from kurgu.editing.shorts import heuristic_pick, plan_short, sentences, timeline_to_source, validate_pick
from kurgu.motion.captions import build_ass, group_words, tr_upper

ROOT = Path(__file__).resolve().parent.parent


def cut_words(text_blocks):
    """[(start, 'sentence ...'), ...] -> timeline words."""
    out = []
    for t0, text in text_blocks:
        t = t0
        for w in text.split():
            out.append({"word": w, "t_start": round(t, 3), "t_end": round(t + 0.35, 3), "start": t, "end": t + 0.35})
            t += 0.4
    return out


def demo_words(n=40):
    blocks, t = [], 0.0
    for i in range(n):
        s = f"Bu {i}. cümle ve burada önemli bir şey anlatıyorum." if i % 5 else "Peki neden bunu hiç kimse söylemiyor?"
        blocks.append((t, s))
        t += len(s.split()) * 0.4 + 0.5
    return cut_words(blocks)


class TestShorts(unittest.TestCase):
    def test_sentences_split(self):
        s = sentences(demo_words(5))
        self.assertEqual(len(s), 5)
        self.assertTrue(s[0]["text"].endswith("?"))

    def test_heuristic_within_limits(self):
        s = sentences(demo_words(60))
        p = validate_pick(heuristic_pick(s, 60, 180), s, 60, 180)
        self.assertTrue(60 <= p["duration"] <= 180, p)
        if p["hook"] is not None:
            self.assertTrue(p["start"] < p["hook"] <= p["end"])

    def test_validate_fixes_bad_llm_answer(self):
        s = sentences(demo_words(60))
        p = validate_pick({"start": 50, "end": 51, "hook": 3}, s, 60, 180)
        self.assertGreaterEqual(p["duration"], 60)
        self.assertIsNone(p["hook"])  # hook outside the range is dropped
        p = validate_pick({"start": 0, "end": 59, "hook": None}, s, 60, 180)
        self.assertLessEqual(p["duration"], 180)

    def test_timeline_to_source_and_plan(self):
        segs = [{"file": "a.mp4", "in": 10.0, "out": 20.0, "timeline_start": 0.0, "timeline_end": 10.0},
                {"file": "a.mp4", "in": 30.0, "out": 40.0, "timeline_start": 10.0, "timeline_end": 20.0}]
        pcs = timeline_to_source(5.0, 15.0, segs)
        self.assertEqual(pcs, [{"file": "a.mp4", "in": 15.0, "out": 20.0}, {"file": "a.mp4", "in": 30.0, "out": 35.0}])
        s = sentences(cut_words([(1.0, "Merak ettiğin şey şu."), (3.0, "Asıl mesele burada başlıyor."), (6.0, "Sonunda anladım.")]))
        sp = plan_short(s, {"start": 0, "end": 2, "hook": 1}, segs)
        self.assertEqual(sp["parts"][0]["kind"], "hook")
        self.assertAlmostEqual(sum(p["out"] - p["in"] for p in sp["pieces"]), sp["duration"], places=2)
        self.assertEqual(sp["words"][0]["word"], "Asıl")  # captions start with the hook


class TestCaptions(unittest.TestCase):
    def test_upper_tr(self):
        self.assertEqual(tr_upper("iyi ışık"), "İYİ IŞIK")

    def test_groups_and_ass(self):
        w = [{"word": x, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, x in enumerate("bir iki üç dört beş altı.".split())]
        g = group_words(w)
        self.assertTrue(all(len(x) <= 3 for x in g))
        ass = build_ass(w, 1080, 1920)
        self.assertIn("PlayResY: 1920", ass)
        self.assertEqual(ass.count("Dialogue:"), len(w))


class TestMetadata(unittest.TestCase):
    def test_clean_and_chapters(self):
        s = sentences(demo_words(40))
        raw = {"variants": [{"angle": "merak", "title": " Başlık bir ", "thumbnail_text": "Kimse bunu söylemiyor", "highlight": 9},
                            {"angle": "hikâye", "title": "İki", "thumbnail_text": "Gurbet zor", "highlight": 0},
                            {"angle": "uyarı", "title": "Üç", "thumbnail_text": "Gelmeden izle", "highlight": 1},
                            {"angle": "fazla", "title": "Dört", "thumbnail_text": "x", "highlight": 0}], "tags": ["a"] * 3 + ["uzun etiket " * 10] * 10,
               "hashtags": ["amerika hayatı", "#gurbet"], "chapters": [{"sentence": 3, "title": "Giriş"},
               {"sentence": 4, "title": "Çok yakın"}, {"sentence": 15, "title": "Orta"}, {"sentence": 30, "title": "Son"}],
               "description_intro": "Giriş.",
               "description_body": "Gövde."}
        m = md.clean_meta(raw, s)
        self.assertEqual(len(m["titles"]), 3)
        self.assertEqual([v["label"] for v in m["variants"]], ["A", "B", "C"])
        self.assertEqual(m["titles"][0], "Başlık bir")
        self.assertEqual(m["chapters"][0]["t"], 0.0)
        self.assertTrue(all(b["t"] - a["t"] >= 10 for a, b in zip(m["chapters"], m["chapters"][1:])))
        self.assertLessEqual(sum(len(t) + 1 for t in m["tags"]), 480)
        self.assertEqual(m["hashtags"], ["#amerikahayatı", "#gurbet"])
        self.assertEqual(m["thumbnail_texts"][0]["highlight"], 2)
        d = md.description_text(m)
        self.assertIn("0:00 Giriş", d)

    def test_fallback_is_marked_weak(self):
        m = md.fallback(demo_words(10), 60)
        self.assertIn("zayıf", m["method"])
        self.assertEqual(len(m["titles"]), 3)


class TestThumbnails(unittest.TestCase):
    def test_designs_keep_text_off_face(self):
        from kurgu.motion import thumbnails as T
        rgb = np.full((1080, 1920, 3), 120, np.uint8)
        rgb[200:600, 800:1100] = 200  # "face" patch
        face = {"x": 800 / 1920, "y": 200 / 1080, "w": 300 / 1920, "h": 400 / 1080, "score": 0.9}
        for name, fn in T.DESIGNS:
            img, boxes = fn({"face": face}, rgb, "Gurbetin bilinmeyen yüzü", 1)
            self.assertEqual(img.size, (1280, 720))
            self.assertLess(T.overlap(boxes["text"], boxes["face"]), 0.05, name)


class TestUpload(unittest.TestCase):
    def test_upload_creates_project(self):
        from http.server import ThreadingHTTPServer

        from kurgu.web.server import App, make_handler
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(ROOT / "design_config.json", tmp / "design_config.json")
            shutil.copy(ROOT / "dictionary.json", tmp / "dictionary.json")
            app = App(Config.load(tmp))
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            port = httpd.server_address[1]
            data = b"\x00" * 1_000_000
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/upload?name=" + urllib.request.quote("Çekim 1.mp4"),
                                         data=data, method="POST")
            r = json.loads(urllib.request.urlopen(req).read())
            self.assertTrue(Path(r["video"]).exists())
            self.assertEqual(Path(r["video"]).stat().st_size, len(data))
            self.assertEqual(r["project"]["name"], "Cekim-1")
            lst = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/projects").read())
            self.assertEqual(lst["projects"][0]["name"], "Cekim-1")
            bad = urllib.request.Request(f"http://127.0.0.1:{port}/api/state?p=../../etc")
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(bad)
            httpd.shutdown()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
