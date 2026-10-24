"""Render a vertical 9:16 short from source pieces: face-following crop, word captions, loudness normalisation."""
from __future__ import annotations

from functools import lru_cache

import os
import subprocess
from pathlib import Path

import numpy as np

from ..integrations import ffmpeg as ff
from ..vision.faces import FaceDetector


def rgb_frames(path: str, t0: float, t1: float, fps: float, width: int):
    info = ff.probe(path)
    height = int(round(info.height * width / max(1, info.width) / 2) * 2)
    args = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{t0:.3f}", "-i", path, "-t", f"{max(0.05, t1 - t0):.3f}",
            "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(args, capture_output=True, creationflags=0x08000000 if os.name == "nt" else 0)
    n = width * height * 3
    buf = p.stdout
    for i in range(len(buf) // n):
        yield np.frombuffer(buf[i * n:(i + 1) * n], dtype=np.uint8).reshape(height, width, 3)


def face_centers(pieces: list[dict], fps: float = 2.0, width: int = 640) -> list[float | None]:
    """Median face-centre x (0..1) for every piece; None when no face was found."""
    det = FaceDetector()
    out = []
    for pc in pieces:
        xs = []
        for fr in rgb_frames(pc["file"], pc["in"], pc["out"], fps, width):
            faces = det.detect(fr)
            if faces:
                f = faces[0]
                xs.append(f["x"] + f["w"] / 2)
        out.append(float(np.median(xs)) if xs else None)
    return out


def smooth_centers(xs: list[float | None], default: float = 0.5, hold: float = 0.04) -> list[float]:
    """Fill gaps and avoid twitchy re-framing: keep the previous crop unless the face moved more than `hold`."""
    filled = []
    last = next((x for x in xs if x is not None), default)
    for x in xs:
        x = last if x is None else x
        if abs(x - last) < hold:
            x = last
        filled.append(x)
        last = x
    return filled


def _run(args):
    return subprocess.run(args, capture_output=True, creationflags=0x08000000 if os.name == "nt" else 0)


# bt709 tags: players then show the colours/brightness exactly as the phone recorded them
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
ENCODERS = {
    # high quality NVENC (RTX): slower preset, adaptive quantisation, look-ahead, B-frames as reference
    "h264_nvenc_hq": ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "{cq}", "-b:v", "0",
                      "-maxrate", "80M", "-bufsize", "160M", "-spatial-aq", "1", "-temporal-aq", "1",
                      "-rc-lookahead", "32", "-bf", "3", "-b_ref_mode", "middle", "-profile:v", "high",
                      "-pix_fmt", "yuv420p"] + TAGS,
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "{cq}", "-b:v", "0", "-pix_fmt",
                   "yuv420p"] + TAGS,
    "libx264": ["-c:v", "libx264", "-preset", "medium", "-crf", "{cq}", "-pix_fmt", "yuv420p"] + TAGS,
}


def _enc(name: str, cq: int) -> list[str]:
    return [x.replace("{cq}", str(cq)) for x in ENCODERS[name]]


@lru_cache(maxsize=1)
def pick_encoder() -> str:
    """The best encoder that actually works on this machine: HQ NVENC, plain NVENC (older GPU/driver), libx264."""
    for name in ("h264_nvenc_hq", "h264_nvenc"):
        p = _run([ff.ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.3"]
                 + _enc(name, 18) + ["-f", "null", "-"])
        if p.returncode == 0:
            return name
    return "libx264"


def render_vertical(pieces: list[dict], centers: list[float], ass_path: str | None, out_path: str,
                    src_w: int, src_h: int, out_w: int = 1080, out_h: int = 1920, fps: int = 30,
                    loudnorm: str = "I=-14:TP=-1.5:LRA=11", progress=None) -> dict:
    """Two passes: (1) every piece cut + cropped + scaled on its own (PCM audio, no clicks at joins),
    (2) concat, burn captions, normalise loudness, final encode."""
    crop_w = min(int(round(src_h * out_w / out_h / 2) * 2), src_w)
    work = Path(out_path).with_suffix("")
    work = work.parent / (work.name + "_parcalar")
    work.mkdir(parents=True, exist_ok=True)
    enc = pick_encoder()
    listing = []
    for k, (pc, cx) in enumerate(zip(pieces, centers)):
        if progress:
            progress(0.05 + 0.65 * k / max(1, len(pieces)), f"Parça {k + 1}/{len(pieces)}")
        x = int(round(min(max(cx * src_w - crop_w / 2, 0), src_w - crop_w)))
        part = work / f"p{k:03d}.mkv"
        dur = pc["out"] - pc["in"]
        args = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-ss", f"{pc['in']:.4f}", "-i", pc["file"],
                "-t", f"{dur:.4f}", "-vf", f"crop={crop_w}:{src_h}:{x}:0,scale={out_w}:{out_h}:flags=lanczos,fps={fps},setsar=1",
                "-af", "aresample=48000", "-ac", "2"] + _enc(enc, 16) + ["-c:a", "pcm_s16le", str(part)]
        p = _run(args)
        if p.returncode != 0:
            raise RuntimeError(f"Parça {k} kodlanamadı: {p.stderr.decode('utf-8', 'replace')[-600:]}")
        listing.append(f"file '{part.as_posix()}'")
    lst = work / "liste.txt"
    lst.write_text("\n".join(listing) + "\n", encoding="utf-8")
    vf = []
    if ass_path:
        esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
        fonts = "C\\:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts"
        vf = ["-vf", f"subtitles='{esc}':fontsdir='{fonts}'"]
    if progress:
        progress(0.75, "Altyazı ve ses ayarı ile son kodlama")
    args = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)] + vf + \
        ["-af", f"loudnorm={loudnorm}", "-r", str(fps)] + _enc(enc, 20) + \
        ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out_path)]
    p = _run(args)
    if p.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError(f"Son kodlama başarısız: {p.stderr.decode('utf-8', 'replace')[-800:]}")
    for f in work.glob("p*.mkv"):
        try:
            f.unlink()
        except OSError:
            pass
    return {"encoder": enc, "crop_w": crop_w, "pieces": len(pieces)}
