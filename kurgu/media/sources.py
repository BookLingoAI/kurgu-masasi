"""Source handling: camera parts form one virtual continuous timeline; screen recordings are matched by audio.

The camera splits one recording into several files. They are ordered (by creation time, then name) and placed
back-to-back; `CameraTimeline` maps virtual time <-> (file, local time).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..integrations.ffmpeg import MediaInfo, probe

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".mts", ".m2ts", ".avi"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


@dataclass
class Part:
    path: str
    start: float      # virtual time where this file starts
    duration: float
    info: dict

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class CameraTimeline:
    parts: list[Part] = field(default_factory=list)

    @classmethod
    def from_files(cls, files: list[str]) -> "CameraTimeline":
        infos: list[MediaInfo] = [probe(f) for f in files]
        infos.sort(key=lambda i: (i.creation_time or "", natural_key(Path(i.path).name)))
        t = 0.0
        parts = []
        for i in infos:
            parts.append(Part(path=i.path, start=t, duration=i.duration, info=i.to_dict()))
            t += i.duration
        return cls(parts)

    @property
    def duration(self) -> float:
        return self.parts[-1].end if self.parts else 0.0

    @property
    def fps(self) -> float:
        return self.parts[0].info["fps"] if self.parts else 0.0

    @property
    def size(self) -> tuple[int, int]:
        return (self.parts[0].info["width"], self.parts[0].info["height"]) if self.parts else (0, 0)

    def locate(self, t: float) -> tuple[Part, float]:
        for p in self.parts:
            if t < p.end or p is self.parts[-1]:
                return p, max(0.0, t - p.start)
        raise ValueError(t)

    def split_segment(self, a: float, b: float) -> list[tuple[Part, float, float]]:
        """Split virtual [a,b) into per-file local ranges."""
        out = []
        for p in self.parts:
            lo, hi = max(a, p.start), min(b, p.end)
            if hi - lo > 1e-4:
                out.append((p, lo - p.start, hi - p.start))
        return out

    def to_dict(self) -> dict:
        return {"parts": [asdict(p) for p in self.parts], "duration": self.duration}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraTimeline":
        return cls([Part(**p) for p in d["parts"]])


def list_videos(folder: str | Path) -> list[str]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted((str(p) for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT), key=lambda s: natural_key(Path(s).name))
