import unittest

from kurgu.analysis.text import apply_dictionary, correct_words, is_hallucination
from kurgu.config import Config
from kurgu.editing.cuts import compute_cuts, find_repetitions

CFG = Config.load().design


def W(text: str, t0: float = 0.0, wd: float = 0.3, gap: float = 0.05, pauses: dict | None = None):
    """Build timed words. `pauses` maps word index -> extra gap before that word."""
    out, t = [], t0
    for i, w in enumerate(text.split()):
        t += (pauses or {}).get(i, 0.0)
        out.append({"word": w, "start": round(t, 3), "end": round(t + wd, 3), "prob": 0.9})
        t += wd + gap
    return out


def kept_text(words, res):
    dead = set(res["removed_word_idx"])
    return " ".join(w["word"] for i, w in enumerate(words) if i not in dead)


def run(words):
    return find_repetitions(words, CFG["cuts"], set())


def text_after(words):
    rem, dead = run(words)
    return " ".join(w["word"] for i, w in enumerate(words) if i not in dead)


class TestRepetitions(unittest.TestCase):
    def test_restart_same_opening(self):
        w = W("bugün size yeni bugün size yeni bir model göstereceğim", pauses={3: 0.6})
        self.assertEqual(text_after(w), "bugün size yeni bir model göstereceğim")

    def test_abandoned_phrase(self):
        w = W("bu model bu model çok hızlı")
        self.assertEqual(text_after(w), "bu model çok hızlı")

    def test_evet_restart(self):
        w = W("şimdi bunu evet şimdi bunu açıyoruz")
        self.assertEqual(text_after(w), "şimdi bunu açıyoruz")

    def test_suffix_correction(self):
        w = W("modelin modeli çok iyi")
        self.assertEqual(text_after(w), "modeli çok iyi")

    def test_reduplication_kept(self):
        for s in ("teker teker anlatacağım", "yavaş yavaş gidiyoruz", "yan yana koyalım", "baş başa kaldık"):
            self.assertEqual(text_after(W(s)), s, s)

    def test_list_kept(self):
        s = "hangi model gelmiş hangi model gitmiş"
        self.assertEqual(text_after(W(s, pauses={3: 0.4})), s)

    def test_complete_then_shorter_kept(self):
        # B shorter than A is not a restart
        s = "bu çok güzel bir özellik bu çok güzel"
        self.assertEqual(text_after(W(s, pauses={5: 0.5})), s)

    def test_same_word_across_long_pause_kept(self):
        s = "işte bu bu model çok iyi"
        w = W(s, pauses={2: 3.0})
        self.assertEqual(text_after(w), s)

    def test_emphasis_sentence_kept(self):
        s = "İş hayatı zor. İş hayatı zor."
        self.assertEqual(text_after(W(s)), s)

    def test_elongated_fillers(self):
        from kurgu.editing.cuts import is_filler
        fill = set(CFG["cuts"]["fillers"])
        for f in ("Eeeh...", "ÖÖÖÖÖÖÖÖÖÖÖÖ", "hıııı", "mmmh", "ııı", "Hmm."):
            self.assertTrue(is_filler(f, fill), f)
        for f in ("evet", "bu", "ama", "o", "ve", "eş", "a", "aa", "e", "hı"):
            self.assertFalse(is_filler(f, fill), f)

    def test_complete_sentences_not_retakes(self):
        for s, gaps in (("Ben kendimden geçtim. Kendimden geçtiğim kısmını anlatayım", {3: 0.3}),
                        ("Bu amaçla gelmedim. Bu amaçla gelmek istiyorsanız mutsuz olursunuz", {3: 0.3}),
                        ("Çocuklarım için ben buradayım. Burada çok para kazanacağım", {}),
                        ("Diyorlar ki eskisi gibi değil. Abi eskisi gibi değil.", {}),
                        ("Evini al. Yani evini al ama", {})):
            self.assertEqual(text_after(W(s, pauses=gaps)), s, s)

    def test_trailing_off_is_retake(self):
        s = "fiyatı yükseltemeyiz diyorlar. Toplum... toplum bilinci var"
        self.assertEqual(text_after(W(s, pauses={4: 0.9})), "fiyatı yükseltemeyiz diyorlar. toplum bilinci var")

    def test_exact_complete_retake_cut(self):
        s = "Bugün size yeni bir model göstereceğim. Bugün size yeni bir model göstereceğim ve başlıyoruz"
        self.assertEqual(text_after(W(s, pauses={6: 0.8})), "Bugün size yeni bir model göstereceğim ve başlıyoruz")

    def test_stutter(self):
        self.assertEqual(text_after(W("ve ve sonra")), "ve sonra")


class TestCompute(unittest.TestCase):
    def test_fillers_pauses_and_waits(self):
        words = W("merhaba arkadaşlar") + W("ııı", t0=1.5) + W("bugün başlıyoruz", t0=2.5) + W("tamam", t0=12.0)
        active = [(0.0, 0.75), (1.5, 1.8), (2.5, 3.15), (5.0, 9.0), (12.0, 12.3)]  # 5-9 = render noise, no words
        res = compute_cuts(words, active, 13.0, CFG["cuts"])
        self.assertNotIn("ııı", kept_text(words, res))
        rules = {r["rule"] for r in res["removals"]}
        self.assertIn("filler", rules)
        self.assertIn("long_wait", rules)
        self.assertIn("pause", rules)
        # no kept word clipped
        for w in words:
            if w["word"] == "ııı":
                continue
            self.assertTrue(any(a <= w["start"] and w["end"] <= b for a, b in res["kept"]), w)
        # the noisy wait (3.15-12.0) is gone despite energy
        self.assertFalse(any(a < 8.0 < b for a, b in res["kept"]))

    def test_video_playing_protected(self):
        words = W("bakın") + W("işte bu", t0=10.0)
        active = [(0.0, 0.3), (1.0, 9.5), (10.0, 10.65)]
        res = compute_cuts(words, active, 11.0, CFG["cuts"], protected=[(1.0, 9.5)])
        self.assertTrue(any(a <= 5.0 <= b for a, b in res["kept"]), res["kept"])

    def test_cough_removed(self):
        words = W("bir") + W("iki", t0=1.0)
        active = [(0.0, 0.3), (0.45, 0.8), (1.0, 1.3)]  # 0.45-0.8 cough, no words
        res = compute_cuts(words, active, 1.5, CFG["cuts"])
        self.assertTrue(any(r["rule"] == "nonspeech" for r in res["removals"]))

    def test_misaligned_word_protects_island(self):
        # Whisper puts "bunları" 0.5 s early, over silence; its real sound is the next island
        words = W("paylaşmak istedim") + [{"word": "bunları.", "start": 1.0, "end": 1.8, "prob": 0.7}]
        active = [(0.0, 0.9), (1.75, 2.3)]
        res = compute_cuts(words, active, 3.0, CFG["cuts"])
        self.assertFalse(any(r["rule"] == "nonspeech" for r in res["removals"]), res["removals"])
        self.assertTrue(any(a <= 2.0 <= b for a, b in res["kept"]), res["kept"])

    def test_short_pause_kept(self):
        words = W("bir iki", gap=0.2)
        active = [(0.0, 0.3), (0.5, 0.8)]
        res = compute_cuts(words, active, 0.8, CFG["cuts"])
        self.assertEqual(len(res["kept"]), 1)


class TestText(unittest.TestCase):
    def test_dictionary(self):
        self.assertEqual(apply_dictionary("premier pro açıyorum", {"premier pro": "Premiere Pro"}), "Premiere Pro açıyorum")
        ws = correct_words(W("klod ile çalışıyoruz"), {"klod": "Claude"})
        self.assertEqual(ws[0]["word"], "Claude")

    def test_hallucination(self):
        pats = CFG["transcript"]["hallucination_patterns"]
        self.assertTrue(is_hallucination("İzlediğiniz için teşekkür ederim.", pats))
        self.assertFalse(is_hallucination("Bugün yeni bir model var", pats))


if __name__ == "__main__":
    unittest.main()
