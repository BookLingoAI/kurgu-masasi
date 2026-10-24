"""HyperFrames (heygen-com/hyperframes, Apache-2.0): render an HTML composition to a transparent WebM overlay.

Installed locally under araclar/hyperframes (pinned 0.8.140, npm --ignore-scripts). Uses the system Chrome and our
FFmpeg; telemetry is switched off. Compositions animate with CSS/WAAPI (no GSAP: its licence is not MIT and the
scaffold loads it from a CDN) and load fonts from local files, so rendering needs no internet.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "araclar" / "hyperframes"
FONT_DIR = ROOT / "assets" / "fonts"


def cli() -> Path | None:
    p = TOOL_DIR / "node_modules" / ".bin" / ("hyperframes.cmd" if os.name == "nt" else "hyperframes")
    return p if p.exists() else None


def available() -> bool:
    return cli() is not None and shutil.which("node") is not None


def _env() -> dict:
    env = dict(os.environ)
    env.update(HYPERFRAMES_NO_TELEMETRY="1", DO_NOT_TRACK="1", HYPERFRAMES_SKIP_SKILLS="1")
    return env


def page(body: str, css: str, dur: float, width: int = 1920, height: int = 1080, fonts: dict[str, str] | None = None) -> str:
    """A transparent single-clip composition. fonts maps a CSS family name to a file in assets/fonts."""
    faces = "".join(f"@font-face{{font-family:'{fam}';src:url('assets/{fn}')}}" for fam, fn in (fonts or {}).items())
    return f"""<!doctype html><html lang="tr"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width={width}, height={height}"/>
<style>{faces}html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden;background:transparent}}
#root{{position:relative;width:100%;height:100%}}{css}</style></head><body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{dur:.3f}" data-width="{width}"
 data-height="{height}" data-no-timeline><div class="clip" data-start="0" data-duration="{dur:.3f}" data-track-index="0"
 style="position:absolute;inset:0">{body}</div></div></body></html>"""


def render(html: str, out: str | Path, work: str | Path, fps: int = 30, fonts: dict[str, str] | None = None,
           workers: int = 2, timeout: float = 600) -> Path:
    """Write the composition into work/, render it and return the WebM (VP9 + alpha)."""
    exe = cli()
    if exe is None:
        raise RuntimeError("HyperFrames kurulu değil (araclar\\hyperframes).")
    work = Path(work)
    (work / "assets").mkdir(parents=True, exist_ok=True)
    for fn in (fonts or {}).values():
        src = FONT_DIR / fn
        if not src.exists():
            raise RuntimeError(f"Yazı tipi yok: {src}")
        shutil.copy2(src, work / "assets" / fn)
    (work / "index.html").write_text(html, encoding="utf-8")
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(exe), "render", ".", "--format", "webm", "--fps", str(fps), "-o", str(out),
                        "--workers", str(workers), "--quiet"], cwd=work, env=_env(), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("HyperFrames çizemedi: " + (r.stderr or r.stdout)[-600:])
    return out
