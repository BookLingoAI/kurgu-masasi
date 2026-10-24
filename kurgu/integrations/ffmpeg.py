"""FFmpeg / FFprobe wrapper: probing, audio decoding to numpy, frame extraction, loudness."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _find(tool: str) -> str:
    env = os.environ.get(f"KURGU_{tool.upper()}")
    if env and Path(env).exists():
        return env
    found = shutil.which(tool)
    if found:
        return found
    app_root = Path(__file__).resolve().parents[2]   # the installer puts its own copy here
    for cand in (app_root / "ffmpeg" / "bin" / f"{tool}.exe", Path("C:/ffmpeg/bin") / f"{tool}.exe",
                 Path("C:/ffmpeg") / f"{tool}.exe"):
        if cand.exists():
            return str(cand)
    # processes started from other apps can have a reduced environment: look in the usual places directly
    homes = {Path(os.environ["LOCALAPPDATA"])} if os.environ.get("LOCALAPPDATA") else set()
    homes.add(Path.home() / "AppData" / "Local")
    for h in homes:
        local = h / "Microsoft" / "WinGet" / "Links" / f"{tool}.exe"
        if local.exists():
            return str(local)
        for cand in sorted(h.glob(f"Microsoft/WinGet/Packages/*FFmpeg*/**/bin/{tool}.exe"), reverse=True):
            return str(cand)
    raise FileNotFoundError(f"{tool} not found (PATH, C:\\ffmpeg, WinGet Links)")


def ffmpeg_bin() -> str:
    return _find("ffmpeg")


def ffprobe_bin() -> str:
    return _find("ffprobe")


def run(args: list[str], capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(args, capture_output=capture, creationflags=_CREATE_NO_WINDOW)
    if check and p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace")[-2000:] if p.stderr else ""
        raise RuntimeError(f"{Path(args[0]).name} failed ({p.returncode}): {err}")
    return p


@dataclass
class MediaInfo:
    path: str
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_fraction: str = ""
    audio_rate: int = 0
    audio_channels: int = 0
    creation_time: str = ""
    video_codec: str = ""
    rotation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str | Path) -> MediaInfo:
    p = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(p.stdout.decode("utf-8", "replace"))
    fmt = data.get("format", {})
    info = MediaInfo(path=str(path), duration=float(fmt.get("duration") or 0.0), has_video=False, has_audio=False)
    info.creation_time = (fmt.get("tags") or {}).get("creation_time", "")
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not info.has_video and (s.get("disposition") or {}).get("attached_pic", 0) == 0:
            info.has_video = True
            info.width = int(s.get("width") or 0)
            info.height = int(s.get("height") or 0)
            # phones / DJI store some videos rotated + a "rotate 90" flag; ffmpeg decodes them upright,
            # so the DISPLAY size must be used everywhere (else the picture is squashed)
            rot = 0
            for sd in s.get("side_data_list") or []:
                if "rotation" in sd:
                    rot = int(float(sd["rotation"]))
            if not rot and (s.get("tags") or {}).get("rotate"):
                rot = int(float(s["tags"]["rotate"]))
            if abs(rot) % 180 == 90:
                info.width, info.height = info.height, info.width
            info.rotation = rot
            rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
            try:
                fr = Fraction(rate)
            except (ValueError, ZeroDivisionError):
                fr = Fraction(0)
            info.fps = float(fr) if fr else 0.0
            info.fps_fraction = rate
            info.video_codec = s.get("codec_name", "")
            if not info.creation_time:
                info.creation_time = (s.get("tags") or {}).get("creation_time", "")
        elif s.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_rate = int(s.get("sample_rate") or 0)
            info.audio_channels = int(s.get("channels") or 0)
    return info


def load_audio(path: str | Path, sr: int = 16000, start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Decode the first audio stream to mono float32 at `sr`."""
    args = [ffmpeg_bin(), "-v", "error", "-nostdin"]
    if start is not None:
        args += ["-ss", f"{max(0.0, start):.6f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    args += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = run(args)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    import wave

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def extract_frame(path: str | Path, t: float, out_png: str | Path, width: int | None = None) -> Path:
    args = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1"]
    if width:
        args += ["-vf", f"scale={width}:-2"]
    args += [str(out_png)]
    run(args)
    return Path(out_png)


def gray_frames(path: str | Path, fps: float, width: int, start: float = 0.0, duration: float | None = None):
    """Yield (time, uint8 HxW gray frame) at the given analysis fps."""
    info = probe(path)
    height = int(round(info.height * width / max(1, info.width) / 2) * 2) or width * 9 // 16
    args = [ffmpeg_bin(), "-v", "error", "-nostdin"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vf", f"fps={fps},scale={width}:{height},format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)
    n = width * height
    i = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield start + i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


_LOUD_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def loudness(path: str | Path) -> dict:
    """EBU R128 integrated loudness / true peak via loudnorm analysis pass."""
    p = run([ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"], check=True)
    m = _LOUD_RE.search(p.stderr.decode("utf-8", "replace"))
    if not m:
        raise RuntimeError("loudnorm output not found")
    d = json.loads(m.group(0))
    return {"integrated_lufs": float(d["input_i"]), "true_peak_dbtp": float(d["input_tp"]), "lra": float(d["input_lra"])}


def free_space_gb(path: str | Path) -> float:
    return shutil.disk_usage(str(Path(path).anchor or path)).free / 1e9
