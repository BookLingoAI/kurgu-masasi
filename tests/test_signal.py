import unittest

import numpy as np

from kurgu.analysis.audio import active_regions, energy_profile, invert_regions, pauses
from kurgu.config import Config
from kurgu.media.sync import sync_camera_to_screen

SR = 16000
CFG = Config.load().design


def bursts(spec, sr=SR, seed=0, noise_db=-60):
    """spec: list of (start, end) seconds of 'speech' (modulated noise) inside total length spec[-1][1]+1."""
    rng = np.random.default_rng(seed)
    total = max(b for _, b in spec) + 1.0
    x = rng.normal(0, 10 ** (noise_db / 20), int(total * sr)).astype(np.float32)
    for a, b in spec:
        n = int((b - a) * sr)
        t = np.arange(n) / sr
        env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.28)))  # syllable-like modulation
        x[int(a * sr):int(a * sr) + n] += (rng.normal(0, 0.2, n) * (0.3 + 0.7 * env)).astype(np.float32)
    return x


class TestEnergy(unittest.TestCase):
    def test_pause_detection(self):
        spec = [(0.5, 2.0), (2.1, 3.0), (3.6, 5.0), (9.0, 10.0)]
        x = bursts(spec)
        prof = energy_profile(x, SR, CFG["audio_analysis"])
        regs = active_regions(prof, CFG["audio_analysis"])
        p = pauses(prof, CFG["audio_analysis"], 0.25)
        # 0.1 s gap must not be a cut-worthy pause; 0.6 s and 4 s gaps must be.
        mids = [(a + b) / 2 for a, b in p]
        self.assertTrue(any(3.0 < m < 3.6 for m in mids), p)
        self.assertTrue(any(5.0 < m < 9.0 for m in mids), p)
        self.assertFalse(any(2.0 < m < 2.1 for m in mids), p)
        # speech boundaries within 60 ms
        self.assertAlmostEqual(regs[0][0], 0.5, delta=0.06)

    def test_ambient_noise_not_speech(self):
        # loud room tone (-35 dBFS) must still separate from speech
        x = bursts([(1.0, 3.0), (6.0, 8.0)], noise_db=-35)
        prof = energy_profile(x, SR, CFG["audio_analysis"])
        p = pauses(prof, CFG["audio_analysis"], 0.25)
        self.assertTrue(any(a < 3.5 and b > 5.5 for a, b in p), p)

    def test_invert(self):
        self.assertEqual(invert_regions([(1, 2), (3, 4)], 5), [(0, 1), (2, 3), (4, 5)])


class TestSync(unittest.TestCase):
    def _make(self, offset, drift_ppm=0.0, cam_len=200.0, scr_len=400.0):
        rng = np.random.default_rng(3)
        spec, t = [], 0.5
        while t < scr_len - 2:
            d = rng.uniform(0.3, 2.5)
            spec.append((t, t + d))
            t += d + rng.uniform(0.1, 1.2)
        scr = bursts(spec, seed=1)[: int(scr_len * SR)]
        # camera = screen audio starting at `offset`, with clock drift, plus different noise/filter
        n = int(cam_len * SR)
        cam_t = np.arange(n) / SR
        scr_t = offset + cam_t * (1 + drift_ppm * 1e-6)
        cam = np.interp(scr_t, np.arange(len(scr)) / SR, scr).astype(np.float32)
        cam = 0.6 * cam + rng.normal(0, 0.01, n).astype(np.float32)
        return cam, scr

    def test_offset(self):
        cam, scr = self._make(offset=123.456)
        r = sync_camera_to_screen(cam, scr, SR, CFG["sync"])
        self.assertIsNotNone(r.offset_s, r.note)
        self.assertAlmostEqual(r.offset_s, 123.456, delta=0.003)

    def test_drift(self):
        cam, scr = self._make(offset=50.0, drift_ppm=100.0, cam_len=300.0)
        r = sync_camera_to_screen(cam, scr, SR, CFG["sync"])
        self.assertAlmostEqual(r.screen_time(280.0), 50.0 + 280.0 * (1 + 100e-6), delta=0.004)

    def test_no_match(self):
        cam = bursts([(0.5, 3), (4, 9), (10, 20)], seed=9)
        _, scr = self._make(offset=10.0)
        r = sync_camera_to_screen(cam, scr, SR, CFG["sync"])
        self.assertIsNone(r.screen, r.confidence)


if __name__ == "__main__":
    unittest.main()
