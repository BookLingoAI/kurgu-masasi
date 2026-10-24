"""Decision log: every edit decision is written with its reason to logs/decisions.log."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()


class DecisionLog:
    def __init__(self, project_dir: Path):
        self.path = Path(project_dir) / "logs" / "decisions.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, action: str, reason: str, **data) -> None:
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "action": action, "reason": reason}
        rec.update(data)
        line = json.dumps(rec, ensure_ascii=False)
        with _lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def section(self, title: str) -> None:
        with _lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(f"# ---- {title} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ----\n")

    def read(self, limit: int = 2000) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out


class NullLog(DecisionLog):
    def __init__(self):  # noqa: D401 - no file
        self.records: list[dict] = []

    def write(self, stage, action, reason, **data):
        self.records.append({"stage": stage, "action": action, "reason": reason, **data})

    def section(self, title):
        pass
