"""Frame grabs at a recording's very end (motion/thumbnails.grab) - a 2-frame piece there stopped a whole job."""
import subprocess

import cv2
import numpy as np
import pytest

from kurgu.integrations import ffmpeg as ff
from kurgu.motion.thumbnails import grab


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("grab") / "kisa.mp4"
    subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_grab_in_the_last_frames_falls_back_a_little_earlier(clip):
    dur = ff.probe(clip).duration
    for t in (dur - 0.01, dur - 0.034, dur):
        img = grab(str(clip), t)
        assert isinstance(img, np.ndarray) and img.shape[:2] == (180, 320)


def test_grab_far_beyond_the_end_says_why(clip):
    with pytest.raises(RuntimeError, match="kare alınamadı"):
        grab(str(clip), 60.0)


def test_grab_in_the_middle_is_unchanged(clip):
    img = grab(str(clip), 1.5)
    assert img.dtype == np.uint8 and cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).std() > 5
