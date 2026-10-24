"""Grab frames from the Premiere sequence (QE exportFramePNG) for visual inspection."""
from __future__ import annotations

import time
from pathlib import Path

from ..integrations.premiere import js_str


def grab_frames(pr, proj, progress, count: int = 8) -> dict:
    pl = proj.load("edit_plan.json")
    if not pl:
        raise RuntimeError("Plan yok")
    name = pl["sequence"]["name"]
    dur = pl["stats"]["timeline_s"]
    out_dir = proj.dir / "kareler"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    times = [dur * (i + 0.5) / count for i in range(count)]
    files = []
    for i, t in enumerate(times):
        progress(i / count, f"Kare {i + 1}/{count}")
        stem = out_dir / f"{stamp}_{i:02d}_{int(round(t * 100)):07d}"
        path = Path(str(stem) + ".png")  # exportFramePNG appends ".png" itself
        tc = pr.run(f"""
            var seq = KL.findSequence({js_str(name)});
            app.project.activeSequence = seq;
            return KL.ok(KL.exportFrame(seq, {t}, {js_str(str(stem))}));""", f"frame{i}", 120)
        # exportFramePNG is asynchronous on some versions: wait for the file
        for _ in range(50):
            if path.exists() and path.stat().st_size > 0:
                break
            time.sleep(0.2)
        files.append({"t": round(t, 2), "timecode": tc, "file": str(path), "exists": path.exists()})
    progress(1.0, "Kareler alındı")
    return {"frames": files}
