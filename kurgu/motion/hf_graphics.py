"""On-screen graphics drawn by HyperFrames catalog blocks (engines "karma" and "hyperframes").

The blocks come from heygen-com/hyperframes/registry (Apache-2.0), saved once in araclar/hyperframes/katalog;
GSAP (gsap@3.14.2, "no charge" licence, user approved) is served from the local node_modules, fonts from
assets/fonts (Turkish letters), so a render never touches the internet. Each block keeps its own animation; we only
put our text, our accent colour and our fonts into it.

A graphic spec: {"kind": "title" | "price" | "list", "t0", "text", "kicker", "items", "idx"}.
  karma       -> every spec is rendered to its own transparent WebM and blended by our renderer (HFClip)
  hyperframes -> the specs become sub-compositions of one HyperFrames timeline (motion/hf_engine.py)
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..integrations import hyperframes as hf

KATALOG = hf.TOOL_DIR / "katalog"
GSAP = hf.TOOL_DIR / "node_modules" / "gsap" / "dist" / "gsap.min.js"
VERSION = 1

# lower thirds take turns, never the same look twice in a row: (block, title element, small-text element)
LOWER_THIRDS = [("lt-kicker-name", "lt-name", "lt-kicker"), ("lt-mask-reveal", "lt-name", "lt-role"),
                ("lt-bold-block", "lt-name", "lt-tag"), ("lt-neon-border", "nb-name", "nb-role"),
                ("lt-accent-underline", "lt-name", "lt-role"), ("lt-soft-pill", "lt-name", "lt-role"),
                ("lt-color-block", "lt-name", "lt-role"), ("lt-dark-card", "lt-name", "lt-role")]
ACCENTS = ("#46e5b7", "#ffd23f", "#2756ff", "#f5b942", "#cc9149", "#ff5a36", "#30d158", "#7d5cff")
# the blocks' web fonts -> our local files with Turkish letters
FONT_FILES = {"Archivo Black": "Anton-Regular.ttf", "League Gothic": "Oswald[wght].ttf", "Oswald": "Oswald[wght].ttf",
              "Montserrat": "Montserrat[wght].ttf", "Inter": "Inter[opsz,wght].ttf", "Space Mono": "IBMPlexMono-Medium.ttf",
              "Bebas Neue": "Anton-Regular.ttf", "JetBrains Mono": "IBMPlexMono-Medium.ttf", "Lato": "Inter[opsz,wght].ttf",
              "Anton": "Anton-Regular.ttf", "Poppins": "Poppins-ExtraBold.ttf"}
CURRENCY = {"dolar": "$", "lira": "₺", "tl": "₺", "euro": "€", "avro": "€", "yen": "¥", "sterlin": "£", "pound": "£"}


def available() -> bool:
    return hf.available() and GSAP.exists() and (KATALOG / "blocks" / "lt-kicker-name").exists()


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb[:3]])


def _block(name: str) -> tuple[Path, str, float]:
    d = next(p for p in (KATALOG / "blocks" / name, KATALOG / "components" / name) if p.exists())
    meta = json.loads((d / "registry-item.json").read_text(encoding="utf-8"))
    html = (d / f"{name}.html").read_text(encoding="utf-8")
    dur = float(meta.get("duration") or re.search(r'data-duration="([\d.]+)"', html).group(1))
    return d, html, dur


def _set_text(html: str, el_id: str, text: str) -> str:
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    m = re.search(rf'<(\w+)[^>]*\bid="{re.escape(el_id)}"[^>]*>([^<]*)', html)
    if not m:
        return html
    old = m.group(2).strip()
    html = html[:m.end() - len(m.group(2))] + esc + html[m.end():]
    if len(old) >= 4:   # some blocks also carry it as a variable default and write it back at run time
        html = html.replace(json.dumps(old, ensure_ascii=False), json.dumps(text, ensure_ascii=False))
    return html


def localize(html: str, accent) -> str:
    """CDN GSAP -> local file, web fonts -> local fonts, the block's accent -> ours."""
    html = re.sub(r'<script[^>]+src="https://cdn\.jsdelivr\.net/npm/gsap@[^"]+"[^>]*></script>', '<script src="gsap.min.js"></script>', html)
    html = re.sub(r'<link[^>]+fonts\.(googleapis|gstatic)\.com[^>]*>', "", html)
    faces = "".join(f"@font-face{{font-family:'{fam}';src:url('fonts/{i}.ttf')}}" for i, fam in enumerate(FONT_FILES))
    html = html.replace("<style>", "<style>" + faces, 1)
    for a in ACCENTS:
        html = re.sub(re.escape(a), _hex(accent), html, flags=re.I)
    return html


def _fit(html: str, el_id: str, text: str, room_px: int = 1500) -> str:
    """Lower thirds are single-line (nowrap) at ~70 px: long titles get a smaller size instead of leaving the frame."""
    if len(text) <= 18:
        return html
    px = max(38, int(room_px / (len(text) * 0.62)))
    return html.replace("</style>", f"#{el_id}{{font-size:{px}px !important}}</style>", 1)


def _price(text: str) -> tuple[int, str] | None:
    m = re.search(r"(\d[\d.,]*)\s*([A-Za-zÇĞİÖŞÜçğıöşü]+)?", text)
    if not m:
        return None
    sym = CURRENCY.get((m.group(2) or "").lower().replace("İ", "i"))
    if not sym:
        return None
    try:
        return int(float(m.group(1).replace(".", "").replace(",", "."))), sym
    except ValueError:
        return None


def build(spec: dict, accent) -> tuple[str, float, str]:
    """-> (html, duration, block name) for one graphic."""
    kind = spec.get("kind", "title")
    if kind == "price" and _price(spec.get("text", "")):
        value, sym = _price(spec["text"])
        _d, html, dur = _block("apple-money-count")
        html = html.replace("10000", str(max(1, value))).replace('"$" + rounded', f'"{sym}" + rounded') \
                   .replace('"en-US"', '"tr-TR"').replace(">$0<", f">{sym}0<")
        # over footage: no white page, a soft dark veil so the number reads
        html = html.replace("</style>", "html,body{background:transparent !important}"
                            "#root{--bg:rgba(8,9,12,0.5) !important;--fg:#ffffff !important}"
                            ".amount{text-shadow:0 6px 28px rgba(0,0,0,.6) !important}"   # white glow -> dark shadow
                            ".stage-content{align-items:flex-end !important;padding-bottom:90px !important}"  # below the face
                            "</style>", 1)
        html = re.sub(r"<audio\b.*?</audio>", "", html, flags=re.S)   # our own sound cue plays instead
        return localize(html, accent), dur, "apple-money-count"
    if kind == "list" and spec.get("items"):
        _d, html, dur = _block("mk-specs-list")
        rows = ",".join(json.dumps({"label": it, "value": ""}, ensure_ascii=False) for it in spec["items"][:6])
        html = re.sub(r"rows:\s*\[.*?\],", f"rows: [{rows}],", html, count=1, flags=re.S)
        html = html.replace('scheme: "light"', 'scheme: "dark"').replace("scrim: 0,", "scrim: 0.6,")
        return localize(html, accent), dur, "mk-specs-list"
    name, big_id, small_id = LOWER_THIRDS[int(spec.get("idx", 0)) % len(LOWER_THIRDS)]
    _d, html, dur = _block(name)
    text = spec.get("text", "").strip()
    html = _set_text(html, big_id, text)
    html = _set_text(html, small_id, (spec.get("kicker") or "").strip() or " ")
    html = _fit(html, big_id, text)
    return localize(html, accent), dur, name


def stage_files(work: Path, block: str) -> None:
    """gsap + fonts + the block's own assets next to its html."""
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GSAP, work / "gsap.min.js")
    (work / "fonts").mkdir(exist_ok=True)
    for i, fn in enumerate(FONT_FILES.values()):
        dst = work / "fonts" / f"{i}.ttf"
        if not dst.exists():
            shutil.copy2(hf.FONT_DIR / fn, dst)
    d = next(p for p in (KATALOG / "blocks" / block, KATALOG / "components" / block) if p.exists())
    if (d / "assets").exists():
        shutil.copytree(d / "assets", work / "assets", dirs_exist_ok=True)


def render_clip(spec: dict, accent, cache: Path, fps: int = 30) -> dict:
    """One graphic -> transparent WebM (cached by content) + the box it ever covers (for fast blending)."""
    html, dur, block = build(spec, accent)
    key = hashlib.sha1(f"{VERSION}|{fps}|{html}".encode("utf-8")).hexdigest()[:16]
    work = cache / key
    out = work / "katman.webm"
    info_f = work / "katman.json"
    if out.exists() and info_f.exists():
        return json.loads(info_f.read_text(encoding="utf-8"))
    stage_files(work, block)
    (work / "index.html").write_text(html, encoding="utf-8")
    r = subprocess.run([str(hf.cli()), "render", ".", "--format", "webm", "--fps", str(fps), "-o", str(out),
                        "--workers", "2", "--quiet"], cwd=work, env=hf._env(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"HyperFrames '{block}' çizemedi: " + (r.stderr or r.stdout)[-500:])
    info = {"file": str(out), "dur": dur, "block": block, "box": alpha_box(out)}
    info_f.write_text(json.dumps(info), encoding="utf-8")
    return info


def alpha_box(webm: Path, sw: int = 480, sh: int = 270) -> list[float]:
    """Union of the visible area over the whole clip, as fractions of the frame (x0, y0, x1, y1)."""
    from ..integrations import ffmpeg as ff
    r = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-c:v", "libvpx-vp9", "-i", str(webm), "-vf",
                        f"format=rgba,alphaextract,scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8)
    if not a.size:
        return [0.0, 0.0, 1.0, 1.0]
    m = a.reshape(-1, sh, sw).max(axis=0)
    ys, xs = np.nonzero(m > 4)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    pad = 3
    return [max(0, xs.min() - pad) / sw, max(0, ys.min() - pad) / sh, min(sw, xs.max() + 1 + pad) / sw,
            min(sh, ys.max() + 1 + pad) / sh]


class HFClip:
    """Plays a HyperFrames WebM over our frames: decodes only the box the graphic covers, in order, on demand."""

    def __init__(self, info: dict, w: int, h: int, fps: float):
        bx = info["box"]
        self.file, self.fps = info["file"], fps
        x0, y0 = int(bx[0] * w) // 2 * 2, int(bx[1] * h) // 2 * 2
        x1, y1 = min(w, -(-int(bx[2] * w) // 2) * 2), min(h, -(-int(bx[3] * h) // 2) * 2)
        self.box, self.w, self.h = (x0, y0, x1, y1), w, h
        self.proc, self.k, self.cur = None, -1, None

    def _open(self):
        from ..integrations import ffmpeg as ff
        x0, y0, x1, y1 = self.box
        self.proc = subprocess.Popen([ff.ffmpeg_bin(), "-v", "error", "-c:v", "libvpx-vp9", "-i", self.file, "-vf",
                                      f"fps={self.fps},scale={self.w}:{self.h}:flags=lanczos,format=rgba,"
                                      f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}",
                                      "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def __call__(self, rgb: np.ndarray, dt: float, ov) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        if x1 <= x0 or y1 <= y0:
            return rgb
        want = int(dt * self.fps + 1e-6)
        if self.proc is None:
            self._open()
        n = (x1 - x0) * (y1 - y0) * 4
        while self.k < want:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                self.close()
                self.cur = None
                self.k = 10 ** 9
                break
            self.cur = np.frombuffer(buf, np.uint8).reshape(y1 - y0, x1 - x0, 4)
            self.k += 1
        if self.cur is None:
            return rgb
        a = self.cur[..., 3:4].astype(np.float32) / 255.0
        out = rgb.copy()
        dst = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (dst * (1 - a) + self.cur[..., :3].astype(np.float32) * a).astype(np.uint8)
        return out


def still(info: dict, w: int, h: int) -> np.ndarray:
    """One full-frame RGBA picture of the graphic (fully in) - for the Premiere review project's V5 track."""
    from ..integrations import ffmpeg as ff
    t = min(2.0, max(0.0, info["dur"] * 0.5))
    r = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-c:v", "libvpx-vp9", "-ss", f"{t:.2f}", "-i", info["file"],
                        "-frames:v", "1", "-vf", f"scale={w}:{h},format=rgba", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8)[: w * h * 4].reshape(h, w, 4).copy()


def overlay(spec: dict, accent, cache: Path, w: int, h: int, fps: float):
    """karma: spec -> our Overlay (anim "draw") that plays the HyperFrames clip."""
    from .render import Overlay
    info = render_clip(spec, accent, cache)
    clip = HFClip(info, w, h, fps)
    return Overlay(spec["t0"], spec["t0"] + info["dur"], np.zeros((1, 1, 4), np.uint8), fade=0.0,
                   box=clip.box if clip.box[2] > clip.box[0] else (0, 0, 1, 1), anim="draw",
                   meta={"draw": clip, "static": lambda: still(info, w, h), "hyperframes": info["block"],
                         "text": spec.get("text", "")}), info
