"""Per-file cache for heavy analysis steps.

Key = (step name, source file path + size + mtime, params hash). Results are JSON or .npy files
in the project's cache directory, so re-running analysis only recomputes what changed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from .config import replace_file


def file_fingerprint(path: str | Path) -> str:
    st = os.stat(path)
    return f"{Path(path).resolve()}|{st.st_size}|{int(st.st_mtime)}"


def _key(step: str, sources: list[str | Path], params: Any) -> str:
    h = hashlib.sha1()
    h.update(step.encode())
    for s in sources:
        h.update(file_fingerprint(s).encode("utf-8"))
    h.update(json.dumps(params, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return f"{step}_{h.hexdigest()[:16]}"


class Cache:
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def json(self, step: str, sources: list, params: Any, compute: Callable[[], Any]) -> Any:
        p = self.dir / (_key(step, sources, params) + ".json")
        if self.enabled and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        value = compute()
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        replace_file(tmp, p)
        return value

    def array(self, step: str, sources: list, params: Any, compute: Callable[[], np.ndarray]) -> np.ndarray:
        p = self.dir / (_key(step, sources, params) + ".npy")
        if self.enabled and p.exists():
            return np.load(p)
        value = compute()
        tmp = self.dir / (p.stem + ".tmp.npy")
        np.save(tmp, value)
        replace_file(tmp, p)
        return value
