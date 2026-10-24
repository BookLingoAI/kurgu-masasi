"""End-to-end plumbing test on generated media (Whisper replaced by a fake engine that knows the script)."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kurgu.config import Config
from kurgu.integrations import ffmpeg as ff
from kurgu.pipeline import Project, analyze, plan, render_cut_audio, verify_cut
from tests.test_signal import bursts

SR = 16000
ROOT = Path(__file__).resolve().parent.parent

# script: (start, end, word) in camera virtual time
SCRIPT = [(0.5, 0.9, "merhaba"), (0.95, 1.5, "arkadaşlar"),
          (2.4, 2.8, "bugün"), (2.85, 3.2, "size"),          # abandoned start
          (4.0, 4.4, "bugün"), (4.45, 4.8, "size"), (4.85, 5.3, "yeni"), (5.35, 5.9, "model"),
          (6.2, 6.5, "ııı"),
          (7.0, 7.5, "göstereceğim"),
          (14.0, 14.5, "tamam"), (14.55, 15.1, "başlıyoruz")]


class FakeEngine:
    """Returns the script words that fall inside the given audio window (located by matching audio)."""
    device_used = "fake"

    def __init__(self, full_audio, script):
        self.full = full_audio
        self.script = script

    def transcribe(self, audio, sr, prompt=None):
        # locate chunk inside full audio by exact match of a few samples
        n = min(len(audio), 400)
        key = audio[1000:1000 + n] if len(audio) > 1400 else audio[:n]
        off = 1000 if len(audio) > 1400 else 0
        idx = None
        step = 1
        for i in range(0, len(self.full) - n, step):
            if self.full[i] == key[0] and np.array_equal(self.full[i:i + n], key):
                idx = i - off
                break
        if idx is None:
            return []
        a = idx / sr
        b = a + len(audio) / sr
        words = [{"word": w, "start": s - a, "end": e - a, "prob": 0.95} for s, e, w in self.script if s >= a - 0.01 and e <= b + 0.01]
        if not words:
            return []
        return [{"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w["word"] for w in words),
                 "avg_logprob": -0.2, "no_speech_prob": 0.01, "words": words}]


def make_media(tmp: Path):
    spec = [(s, e) for s, e, _ in SCRIPT] + [(9.0, 12.0)]  # 9-12 = loud render noise, no words
    cam_audio = bursts(spec, seed=5)[: int(16.0 * SR)]
    # screen recording = 30 s, camera audio sits at 7.25 s inside it, plus extra speech elsewhere
    scr = bursts([(0.5, 3.0), (3.5, 6.0), (24.0, 27.0)], seed=6)[: int(30 * SR)]
    scr[int(7.25 * SR):int(7.25 * SR) + len(cam_audio)] = 0.7 * cam_audio + 0.003 * np.random.default_rng(1).normal(size=len(cam_audio))
    files = {}

    def mk(name, audio, start, dur, size="640x360", rate=30):
        wav = tmp / f"{name}.wav"
        ff.write_wav(wav, audio[int(start * SR):int((start + dur) * SR)], SR)
        out = tmp / f"{name}.mp4"
        subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
                        "-i", str(wav), "-t", f"{dur}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
        return str(out)

    files["cam1"] = mk("C0001", cam_audio, 0.0, 8.0, rate=60)
    files["cam2"] = mk("C0002", cam_audio, 8.0, 8.0, rate=60)
    files["screen"] = mk("bandicam", scr, 0.0, 30.0, size="960x540", rate=30)
    return files, cam_audio


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        shutil.copy(ROOT / "design_config.json", cls.tmp / "design_config.json")
        shutil.copy(ROOT / "dictionary.json", cls.tmp / "dictionary.json")
        cls.files, cls.cam_audio = make_media(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_end_to_end(self):
        cfg = Config.load(self.tmp)
        cfg.save_user({"project_name": "deneme", "camera_files": [self.files["cam2"], self.files["cam1"]],
                       "screen_files": [self.files["screen"]]})
        proj = Project(cfg)
        # the fake engine needs the decoded virtual audio
        a1 = ff.load_audio(self.files["cam1"], SR)
        a2 = ff.load_audio(self.files["cam2"], SR)
        cam1_d = ff.probe(self.files["cam1"]).duration
        virtual = np.concatenate([a1[:int(round(cam1_d * SR))], a2])
        eng = FakeEngine(virtual, SCRIPT)
        an = analyze(proj, engine=eng)
        # parts ordered by name despite reversed input
        self.assertTrue(an["camera"]["parts"][0]["path"].endswith("C0001.mp4"))
        for part, sy in an["sync"].items():
            self.assertIsNotNone(sy["screen"], sy)
            self.assertAlmostEqual(sy["offset_s"] - (an["camera"]["parts"][0]["duration"] if part.endswith("C0002.mp4") else 0),
                                   7.25, delta=0.03, msg=sy)
        pl = plan(proj, use_semantic=False)
        text = " ".join(w["word"] for w in pl["transcript_cut"])
        self.assertEqual(text, "merhaba arkadaşlar bugün size yeni model göstereceğim tamam başlıyoruz")
        rules = pl["stats"]["by_rule"]
        self.assertTrue("restart" in rules or "repetition" in rules, rules)
        self.assertIn("filler", rules)
        self.assertIn("long_wait", rules)
        # 4K-preserving logic: sequence follows camera format
        self.assertEqual((pl["sequence"]["width"], pl["sequence"]["height"]), (640, 360))
        self.assertAlmostEqual(pl["sequence"]["fps"], 60.0, places=2)
        # segments contiguous on the timeline, frame accurate
        for a, b in zip(pl["segments"], pl["segments"][1:]):
            self.assertEqual(a["timeline_end_frame"], b["timeline_start_frame"])
        # every segment has a synced screen range
        self.assertTrue(all(s["screen"] for s in pl["segments"]))
        self.assertLess(pl["stats"]["timeline_s"], 9.0)
        # decisions.log written with reasons
        log = (proj.dir / "logs" / "decisions.log").read_text(encoding="utf-8")
        self.assertIn("son deneme tutuldu", log)
        # cut audio re-transcription verification (fake engine can only find words still present in order)
        cut = render_cut_audio(proj, pl)
        self.assertGreater(len(cut), SR * 3)


if __name__ == "__main__":
    unittest.main()
