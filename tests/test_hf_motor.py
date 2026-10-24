"""Engines "karma" and "hyperframes": catalog blocks get our text/fonts/colour and stay offline; the HyperFrames
timeline holds the cut parts, the transitions and the graphics; a short render really comes out."""
import re
import subprocess

import pytest

from kurgu.integrations import ffmpeg as ff
from kurgu.motion import hf_engine as he
from kurgu.motion import hf_graphics as hg
from kurgu.motion.render import Piece, build_audio

needs_hf = pytest.mark.skipif(not hg.available(), reason="HyperFrames / katalog kurulu değil")


@needs_hf
@pytest.mark.parametrize("idx", range(len(hg.LOWER_THIRDS)))
def test_every_lower_third_takes_our_text_offline(idx):
    html, dur, block = hg.build({"kind": "title", "text": "SOS ÇOK ACI", "kicker": "YORUM", "idx": idx}, (255, 0, 0))
    assert block == hg.LOWER_THIRDS[idx][0] and dur > 2
    assert "SOS ÇOK ACI" in html and "Maya Chen" not in html
    assert not re.search(r"https?://(cdn\.jsdelivr|fonts\.googleapis|fonts\.gstatic)", html)
    assert 'src="gsap.min.js"' in html


@needs_hf
def test_price_and_list_blocks():
    html, _d, block = hg.build({"kind": "price", "text": "167 DOLAR VERDİK"}, (255, 196, 0))
    assert block == "apple-money-count" and "167" in html and "10000" not in html and "<audio" not in html
    html, _d, block = hg.build({"kind": "list", "items": ["Kuzu eti", "Dana dili"]}, (255, 196, 0))
    assert block == "mk-specs-list" and "Kuzu eti" in html and 'scheme: "dark"' in html
    # a price without a currency falls back to a lower third
    assert hg.build({"kind": "price", "text": "20 SANİYE"}, (255, 196, 0))[2] in [b for b, _, _ in hg.LOWER_THIRDS]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("hf") / "kamera.mp4"
    subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=8",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


@needs_hf
def test_hyperframes_engine_renders_the_cut(clip, tmp_path):
    pieces = [Piece(str(clip), 0.0, 2.0, zoom=1.2), Piece(str(clip), 4.0, 6.0, trans_in=0.18, trans="flash"),
              Piece(str(clip), 6.0, 8.0, speed=2.0, trans_in=0.18, trans="whip")]
    ctx = {"pieces": pieces, "fps": 30, "accent": (255, 196, 0),
           "hf_specs": [{"kind": "title", "t0": 0.5, "text": "DENEME BAŞLIK", "kicker": "TEST", "idx": 0}]}
    wav = str(tmp_path / "ses.wav")
    build_audio(pieces, wav)
    meta = he.build_project(ctx, tmp_path / "proje", wav)
    html = (tmp_path / "proje" / "index.html").read_text(encoding="utf-8")
    assert meta["clips"] == 3 and meta["graphics"] == 1 and abs(meta["duration"] - 5.0) < 0.01
    assert 'data-media-start="4.000"' in html and 'data-playback-rate="2.0000"' in html and "#flash" in html
    out = tmp_path / "son.mp4"
    he.render_project(tmp_path / "proje", out, 30, meta["duration"], lambda f, m: None)
    info = ff.probe(out)
    assert abs(info.duration - 5.0) < 0.2 and info.width == 1920
