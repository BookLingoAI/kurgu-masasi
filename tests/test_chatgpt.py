import tempfile
import unittest
from pathlib import Path

from kurgu.editing import chatgpt as cg

ANSWER = """Harika! İşte istediklerin:

**A BAŞLIK:** Amerika'da Berber Olmak: 12 Saat Çalışmanın Gerçek Yüzü
**B BAŞLIK:** "Türkiye mi Amerika mı? Berberlikte Kazanç Farkı"
**C BAŞLIK:** Amerika'da Hayat Gerçekten Kolay mı? Berber Anlatıyor

**AÇIKLAMA:**
Amerika'da berber olarak çalışmak nasıl? Bu videoda Türkiye ile Amerika arasındaki farkları anlatıyorum.
Çalışma saatleri, kazanç ve günlük hayat.

00:00 Giriş
02:15 Türkiye'de berberlik

#amerika #berber #göç

**ETİKETLER:** amerika'da hayat, berber, amerikada iş, göç
**SHORTS BAŞLIK:** 12 Saat Çalışıp Ne Kazanıyordum?
**SHORTS AÇIKLAMA:** Türkiye'deki berberlik günlerim #shorts
"""


class ChatGPTTests(unittest.TestCase):
    def test_parse(self):
        m = cg.parse_answer(ANSWER)
        self.assertEqual(m["titles"][0], "Amerika'da Berber Olmak: 12 Saat Çalışmanın Gerçek Yüzü")
        self.assertEqual(m["titles"][1], "Türkiye mi Amerika mı? Berberlikte Kazanç Farkı")
        self.assertIn("00:00 Giriş", m["description"])
        self.assertNotIn("ETİKETLER", m["description"])
        self.assertEqual(m["tags"][1], "berber")
        self.assertEqual(m["shorts_title"], "12 Saat Çalışıp Ne Kazanıyordum?")

    def test_parse_missing(self):
        with self.assertRaises(ValueError):
            cg.parse_answer("A BAŞLIK: x\nAÇIKLAMA: kısa")

    def test_prompt_and_fit(self):
        words = [{"word": w, "t_start": i * 0.5, "t_end": i * 0.5 + 0.4} for i, w in enumerate("Merhaba arkadaşlar bugün konuşacağız.".split())]
        p = cg.build_prompt(words)
        self.assertIn("[00:00] Merhaba", p)
        from PIL import Image
        d = Path(tempfile.mkdtemp())
        Image.new("RGB", (1024, 1536), (200, 0, 0)).save(d / "a.png")
        info = cg.fit_thumbnail(d / "a.png", d / "k.jpg")
        self.assertEqual(Image.open(d / "k.jpg").size, (1280, 720))
        self.assertLess(info["kb"], 2000)


if __name__ == "__main__":
    unittest.main()
