from kurgu.editing.cuts import breathe_segments


def W(word, a, b):
    return {"word": word, "start": a, "end": b}


def test_sentence_end_gets_tail_and_next_word_is_respected():
    words = [W("Merhaba", 0.5, 1.0), W("dostlar.", 1.0, 1.6), W("Bugün", 3.0, 3.4), W("ııı", 3.5, 3.9), W("geldik", 4.2, 4.8)]
    kept = [w for w in words if w["word"] != "ııı"]
    segs = [(0.49, 1.6), (2.99, 3.4), (4.19, 4.8)]
    out = breathe_segments(segs, words, kept, total=10.0)
    assert out[0][1] >= 1.6 + 0.3 - 1e-6          # sentence end: ~0.32 s of decay kept
    assert out[0][0] <= 0.5 - 0.1 + 1e-6          # phrase start: pre-roll
    b = [s for s in out if s[0] < 3.0 < s[1] or abs(s[0] - 2.88) < 0.05][0]
    assert b[1] <= 3.5 - 0.04 + 1e-6               # never runs into the removed filler
    assert out[-1][1] >= 4.8 + 0.18 - 1e-6


def test_wordless_crumbs_dropped():
    words = [W("a", 0.0, 0.5)]
    out = breathe_segments([(0.0, 0.5), (2.0, 2.2), (5.0, 7.0)], words, words, total=10.0)
    assert (2.0, 2.2) not in out and (5.0, 7.0) in out
