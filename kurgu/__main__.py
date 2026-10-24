"""python -m kurgu            -> web UI on http://localhost:8770 (+ job queue watcher)
python -m kurgu run <action> -> run one pipeline action in the foreground (analyze/plan/apply/verify/rough/ping)
"""
from __future__ import annotations

import json
import sys

from .config import Config


def _own_log(cfg: Config) -> None:
    """pythonw has no console: write stdout/stderr to logs/sunucu-<time>.log (a new file each start)."""
    import os
    import time

    if sys.stdout is not None and sys.stderr is not None and not sys.executable.lower().endswith("pythonw.exe"):
        return
    d = cfg.root / "logs"
    d.mkdir(exist_ok=True)
    f = open(d / f"sunucu-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f


def main(argv: list[str]) -> int:
    from .integrations.cuda_env import suppress_error_dialogs

    suppress_error_dialogs()
    cfg = Config.load()
    _own_log(cfg)
    if not argv or argv[0] == "serve":
        from .web.server import serve
        port = int(argv[1]) if len(argv) > 1 else 8770
        serve(cfg, port=port)
        return 0
    if argv[0] == "run" and len(argv) > 1:
        import time

        from .worker.jobs import JobManager
        jm = JobManager(cfg)
        job = jm.start(argv[1], argv[2] if len(argv) > 2 else None, source="cli")
        last = ""
        while job["state"] == "running":
            if job["message"] != last:
                last = job["message"]
                print(f"[{job['progress'] * 100:5.1f}%] {last}", flush=True)
            time.sleep(0.5)
        print(json.dumps({k: job.get(k) for k in ("state", "error", "result")}, ensure_ascii=False, indent=2, default=str))
        if job.get("trace"):
            print(job["trace"])
        return 0 if job["state"] == "done" else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
