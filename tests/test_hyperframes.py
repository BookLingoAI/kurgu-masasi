"""HyperFrames overlay: a transparent, animated title over a test pattern (needs araclar/hyperframes + Chrome)."""
import subprocess

import numpy as np
import pytest

from kurgu.integrations import ffmpeg as ff
from kurgu.integrations import hyperframes as hf

pytestmark = pytest.mark.skipif(not hf.available(), reason="HyperFrames kurulu değil")

CSS = """.t{position:absolute;left:140px;top:700px;font:150px/1 Anton;color:#fff;clip-path:inset(0 100% 0 0);
animation:w .6s ease-out .3s forwards}@keyframes w{to{clip-path:inset(0 0 0 0)}}"""


def _frame(video, t):
    r = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-c:v", "libvpx-vp9", "-ss", str(t), "-i", str(video),
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(1080, 1920, 4)


def test_overlay_is_transparent_and_animates(tmp_path):
    html = hf.page('<div class="t">ÖDEDİK 167</div>', CSS, 2.0, fonts={"Anton": "Anton-Regular.ttf"})
    out = hf.render(html, tmp_path / "o.webm", tmp_path / "work")
    info = ff.probe(out)
    assert abs(info.duration - 2.0) < 0.1
    start, end = _frame(out, 0.1), _frame(out, 1.5)
    assert start[..., 3].max() == 0                 # nothing visible before the wipe
    assert end[..., 3].mean() > 1 and end[0, 0, 3] == 0   # text drawn, corners stay transparent
