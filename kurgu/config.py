"""Configuration: design_config.json (tunables), user_settings.json (UI values), paths, scaling."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("KURGU_ROOT", Path(__file__).resolve().parent.parent))

DEFAULT_USER_SETTINGS: dict[str, Any] = {
    "camera_files": [],
    "screen_files": [],
    "project_name": "",
    "subtitles_enabled": True,
}

def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return copy.deepcopy(default)


def replace_file(src: str | Path, dst: str | Path, tries: int = 40) -> None:
    """os.replace that survives Windows sharing locks: while another thread/process (the web page reading the same
    file, antivirus, OneDrive, search indexer) has the destination open, Windows answers "Access denied" (WinError 5/32).
    Retry for up to ~10 s, then fall back to copying the content over."""
    import time
    delay = 0.05
    for _ in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
            delay = min(0.5, delay * 1.6)
    import shutil
    shutil.copyfile(src, dst)          # last resort: overwrite in place (works when only delete/rename is blocked)
    try:
        os.remove(src)
    except OSError:
        pass


def write_json(path: Path, data: Any) -> None:
    import threading
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # unique temp name: two threads writing the same file must not share one temp file
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    replace_file(tmp, path)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    root: Path
    design: dict
    user: dict

    @classmethod
    def load(cls, root: Path | None = None) -> "Config":
        root = Path(root or ROOT)
        design = read_json(root / "design_config.json", {})
        if not design:
            raise FileNotFoundError(f"design_config.json not found in {root}")
        user_path = root / design["paths"]["user_settings"]
        user = _deep_merge(DEFAULT_USER_SETTINGS, read_json(user_path, {}) or {})
        return cls(root=root, design=design, user=user)

    def save_user(self, updates: dict) -> dict:
        self.user = _deep_merge(self.user, updates)
        write_json(self.root / self.design["paths"]["user_settings"], self.user)
        return self.user

    def path(self, key: str) -> Path:
        return self.root / self.design["paths"][key]

    def project_dir(self, name: str | None = None) -> Path:
        d = self.path("projects_dir") / (name or self.user.get("project_name") or "proje")
        d.mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(exist_ok=True)
        (d / "cache").mkdir(exist_ok=True)
        return d

    def __getitem__(self, section: str) -> dict:
        return self.design[section]


def scale_frames(frames_at_ref: float, fps: float, ref_fps: float = 30.0) -> int:
    """Convert a frame count specified at reference fps (30) to the source fps."""
    return max(1, int(round(frames_at_ref * fps / ref_fps)))


def scale_px(px_at_ref: float, width: int, ref_width: int = 1920) -> float:
    """Convert a pixel value specified at 1920 width to the source width."""
    return px_at_ref * width / ref_width
