"""Thumbnail frames from the video + three 1280x720 YouTube thumbnail designs (PIL)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from ..integrations import ffmpeg as ff
from ..motion.captions import tr_upper
from ..vision.faces import FaceDetector, person_mask, sharpness

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

W, H = 1280, 720
YELLOW = (255, 212, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ---------------------------------------------------------------- frames ------------------------------------------
def grab(path: str, t: float) -> np.ndarray:
    """The frame at t. In the last frames of a recording ffmpeg can return nothing (a 2-frame piece ending at the file's
    very end did, 22.10.2026, and the whole job stopped): then a little earlier, then 1 s earlier."""
    from io import BytesIO
    err = b""
    for tt in (t, t - 0.25, t - 1.0):
        args = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{max(0.0, tt):.3f}", "-i", path, "-frames:v", "1",
                "-f", "image2pipe", "-vcodec", "png", "-"]
        p = subprocess.run(args, capture_output=True, creationflags=0x08000000 if os.name == "nt" else 0)
        if p.stdout:
            return np.asarray(Image.open(BytesIO(p.stdout)).convert("RGB"))
        err = p.stderr or err
    raise RuntimeError(f"{Path(path).name} dosyasından {t:.2f}. saniyede kare alınamadı: "
                       f"{err.decode('utf-8', 'replace')[-160:] or 'ffmpeg boş döndü'}")


def score_frame(rgb: np.ndarray, det: FaceDetector) -> dict:
    faces = det.detect(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = {"face": None, "score": 0.0, "sharp": sharpness(gray), "bright": float(gray.mean())}
    if not faces:
        return out
    f = faces[0]
    out["face"] = f
    h, w = rgb.shape[:2]
    x0, y0 = int(f["x"] * w), int(f["y"] * h)
    roi = gray[max(0, y0):max(0, y0 + int(f["h"] * h)), max(0, x0):max(0, x0 + int(f["w"] * w))]
    face_sharp = sharpness(roi) if roi.size else 0.0
    area = f["w"] * f["h"]
    size_sc = 1.0 - min(1.0, abs(area - 0.08) / 0.08)
    ex = det.expression(rgb)
    eyes = ex.get("eyes_open", 0.7)
    lively = min(1.0, ex.get("smile", 0) + 0.6 * ex.get("mouth_open", 0) + 0.6 * ex.get("brow_up", 0)) if ex else 0.3
    bright_sc = 1.0 - min(1.0, abs(out["bright"] - 120) / 120)
    out["face_sharp"] = face_sharp
    out["eyes"] = round(eyes, 3)
    out["expression"] = {k: round(v, 3) for k, v in ex.items()}
    # eyes closed / mid-blink frames are unusable for a thumbnail
    blink_penalty = 2.0 if eyes < 0.55 else 0.0
    out["score"] = round(2.0 * min(1.0, face_sharp / 300.0) + 1.5 * eyes + 1.2 * lively + 1.0 * size_sc
                         + 0.5 * bright_sc + f["score"] - blink_penalty, 3)
    return out


def pick_frames(segments: list[dict], n_samples: int = 48, keep: int = 6, min_gap_s: float = 25.0,
                progress=None) -> list[dict]:
    """Sample frames across the kept segments, score them, keep the best `keep` that are far apart."""
    total = sum(s["out"] - s["in"] for s in segments)
    if total <= 0:
        return []
    det = FaceDetector(expression=True)
    cands = []
    for k in range(n_samples):
        target = total * (k + 0.5) / n_samples
        acc = 0.0
        for s in segments:
            d = s["out"] - s["in"]
            if acc + d >= target:
                t_src = s["in"] + (target - acc)
                t_tl = s["timeline_start"] + (target - acc)
                break
            acc += d
        rgb = grab(s["file"], t_src)
        sc = score_frame(rgb, det)
        cands.append({"file": s["file"], "t": round(t_src, 3), "t_timeline": round(t_tl, 3), **sc})
        if progress:
            progress(k / n_samples, f"Kare adayları {k + 1}/{n_samples}")
    cands = [c for c in cands if c["face"]]
    cands.sort(key=lambda c: -c["score"])
    chosen: list[dict] = []
    for c in cands:
        if all(abs(c["t_timeline"] - o["t_timeline"]) >= min_gap_s for o in chosen):
            chosen.append(c)
        if len(chosen) >= keep:
            break
    return chosen


# ---------------------------------------------------------------- text ------------------------------------------
def _font(size: int) -> ImageFont.FreeTypeFont:
    cands = []
    if os.name == "nt":
        fd = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        cands = [(fd / "bahnschrift.ttf", "Bold"), (fd / "seguibl.ttf", None), (fd / "arialbd.ttf", None)]
    cands += [(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), None)]
    for path, var in cands:
        if path.exists():
            f = ImageFont.truetype(str(path), size)
            if var:
                try:
                    names = [n.decode() if isinstance(n, bytes) else n for n in f.get_variation_names()]
                    pick = next((n for n in names if n.replace(" ", "").lower() in ("boldcondensed", "boldsemicondensed")), None) \
                        or next((n for n in names if n.lower() == var.lower()), None)
                    if pick:
                        f.set_variation_by_name(pick)
                except Exception:  # noqa: BLE001
                    pass
            return f
    return ImageFont.load_default()


def _wrap(words: list[str], font, max_w: int) -> list[list[str]]:
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def fit_text(text: str, box_w: int, box_h: int, max_lines: int = 3, start: int = 150, min_size: int = 48):
    words = tr_upper(text).split()
    size = start
    while size >= min_size:
        f = _font(size)
        lines = _wrap(words, f, box_w)
        lh = int(size * 1.02)
        if len(lines) <= max_lines and lh * len(lines) <= box_h and all(f.getlength(" ".join(l)) <= box_w for l in lines):
            return f, lines, lh
        size -= 6
    f = _font(min_size)
    return f, _wrap(words, f, box_w)[:max_lines], int(min_size * 1.02)


def draw_text(img: Image.Image, text: str, box: tuple[int, int, int, int], highlight: int | None = None,
              align: str = "left", stroke: int = 8, color=WHITE, hi_color=YELLOW, valign: str = "center") -> tuple:
    x0, y0, x1, y1 = box
    f, lines, lh = fit_text(text, x1 - x0, y1 - y0)
    d = ImageDraw.Draw(img)
    total_h = lh * len(lines)
    y = y0 + ((y1 - y0 - total_h) // 2 if valign == "center" else 0)
    widx = 0
    drawn = [x1, y, x0, y]
    for line in lines:
        lw = f.getlength(" ".join(line))
        x = x0 if align == "left" else (x0 + (x1 - x0 - lw) / 2 if align == "center" else x1 - lw)
        for w in line:
            col = hi_color if highlight is not None and widx == highlight else color
            d.text((x, y), w, font=f, fill=col, stroke_width=stroke, stroke_fill=BLACK)
            x += f.getlength(w + " ")
            widx += 1
        drawn = [min(drawn[0], x0), drawn[1], max(drawn[2], int(x)), y + lh]
        y += lh
    return tuple(drawn)


# ---------------------------------------------------------------- layout helpers ---------------------------------
def place(rgb: np.ndarray, face: dict, cx: float, cy: float, face_h: float) -> tuple[Image.Image, tuple]:
    """Scale the frame so the face is `face_h` of the thumbnail height and its centre lands at (cx, cy).
    Areas the frame does not cover are filled with a blurred, darkened enlargement of the same frame."""
    src = Image.fromarray(rgb)
    sw, sh = src.size
    scale = max(face_h * H / max(1e-6, face["h"] * sh), W / sw * 0.6)
    nw, nh = int(sw * scale), int(sh * scale)
    fg = src.resize((nw, nh), Image.LANCZOS)
    fcx = (face["x"] + face["w"] / 2) * nw
    fcy = (face["y"] + face["h"] / 2) * nh
    left = int(round(cx * W - fcx))
    top = int(round(cy * H - fcy))
    # background: cover-fit, blurred, darker
    bs = max(W / sw, H / sh) * 1.15
    bg = src.resize((int(sw * bs), int(sh * bs)), Image.LANCZOS)
    bx, by = (bg.size[0] - W) // 2, (bg.size[1] - H) // 2
    bg = bg.crop((bx, by, bx + W, by + H)).filter(ImageFilter.GaussianBlur(28))
    bg = ImageEnhance.Brightness(bg).enhance(0.5)
    canvas = bg.copy()
    # feather the foreground edges so the seam to the fill is invisible
    m = Image.new("L", (nw, nh), 255)
    feather = max(12, int(min(nw, nh) * 0.04))
    md = ImageDraw.Draw(m)
    for i in range(feather):
        v = int(255 * (i + 1) / feather)
        md.rectangle((i, i, nw - 1 - i, nh - 1 - i), outline=v)
    canvas.paste(fg, (left, top), m)
    fb = (int(face["x"] * nw + left), int(face["y"] * nh + top), int((face["x"] + face["w"]) * nw + left),
          int((face["y"] + face["h"]) * nh + top))
    return canvas, fb


def _grade(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Color(img).enhance(1.15)
    return ImageEnhance.Sharpness(img).enhance(1.3)


def _gradient(w: int, h: int, side: str, strength: float = 0.88) -> Image.Image:
    xs = np.linspace(1, 0, w) if side == "left" else np.linspace(0, 1, w)
    a = np.clip((xs - 0.25) / 0.6, 0, 1) * strength
    alpha = (np.tile(a, (h, 1)) * 255).astype(np.uint8)
    g = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    g.putalpha(Image.fromarray(alpha))
    return g


def overlap(a: tuple, b: tuple) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    fa = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / fa


# ---------------------------------------------------------------- designs ------------------------------------------
def design_a(frame: dict, rgb: np.ndarray, text: str, hi: int | None) -> tuple[Image.Image, dict]:
    """Face on the right third, dark gradient left, big stacked text left."""
    img, fb = place(rgb, frame["face"], 0.72, 0.42, 0.50)
    img = _grade(img).convert("RGBA")
    img.alpha_composite(_gradient(W, H, "left"))
    tb = draw_text(img, text, (56, 70, int(W * 0.52), H - 70), highlight=hi)
    return img.convert("RGB"), {"face": fb, "text": tb}


def design_b(frame: dict, rgb: np.ndarray, text: str, hi: int | None) -> tuple[Image.Image, dict]:
    """Person cut out on a bold colour field; without a person mask: face card with white border on the right."""
    grad = np.zeros((H, W, 3), np.float32)
    top, bot = np.array([196, 28, 44], np.float32), np.array([38, 6, 12], np.float32)
    t = np.linspace(0, 1, H)[:, None, None]
    grad[:] = top * (1 - t) + bot * t
    bg = Image.fromarray(grad.astype(np.uint8))
    base, fb = place(rgb, frame["face"], 0.72, 0.40, 0.52)
    mask = person_mask(np.asarray(base))
    if mask is not None:
        m = Image.fromarray((mask * 255).astype(np.uint8))
        glow = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        glow.putalpha(m.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(3)))
        img = bg.convert("RGBA")
        img.alpha_composite(glow)
        person = _grade(base).convert("RGBA")
        person.putalpha(m)
        img.alpha_composite(person)
        img = img.convert("RGB")
        method = "kişi kesildi"
    else:
        img = bg
        cw, ch = int(W * 0.40), int(H * 0.84)
        x0, y0 = W - cw - 44, (H - ch) // 2
        card, cfb = place(rgb, frame["face"], 0.5, 0.42, 0.50)
        card = _grade(card).crop(((W - cw) // 2, (H - ch) // 2, (W + cw) // 2, (H + ch) // 2))
        rm = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(rm).rounded_rectangle((0, 0, cw, ch), 30, fill=255)
        border = Image.new("RGB", (cw + 14, ch + 14), WHITE)
        bm = Image.new("L", border.size, 0)
        ImageDraw.Draw(bm).rounded_rectangle((0, 0, border.size[0], border.size[1]), 34, fill=255)
        img.paste(border, (x0 - 7, y0 - 7), bm)
        img.paste(card, (x0, y0), rm)
        dx, dy = x0 - (W - cw) // 2, y0 - (H - ch) // 2
        fb = (cfb[0] + dx, cfb[1] + dy, cfb[2] + dx, cfb[3] + dy)
        method = "renkli zemin + yüz kartı (kişi kesme modeli yok)"
    tb = draw_text(img, text, (52, 80, int(W * 0.50), H - 80), highlight=hi)
    return img, {"face": fb, "text": tb, "method": method}


def design_c(frame: dict, rgb: np.ndarray, text: str, hi: int | None) -> tuple[Image.Image, dict]:
    """Tight close-up (face centred-left) with a bottom text band."""
    img, fb = place(rgb, frame["face"], 0.50, 0.36, 0.62)
    img = _grade(img)
    band_h = int(H * 0.30)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, H - band_h, W, H), fill=(8, 8, 8, 235))
    d.rectangle((0, H - band_h, W, H - band_h + 10), fill=YELLOW + (255,))
    tb = draw_text(img, text, (48, H - band_h + 22, W - 48, H - 18), highlight=hi, align="center", stroke=6)
    return img, {"face": fb, "text": tb}


DESIGNS = [("Yüz + büyük yazı", design_a), ("Renkli zemin", design_b), ("Yakın plan + şerit", design_c)]


def make_thumbnails(frames: list[dict], texts: list[dict], out_dir: Path) -> list[dict]:
    """frames: picked candidates (best first); texts: [{text, highlight}] x3. One design per text/frame pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for k, (name, fn) in enumerate(DESIGNS):
        fr = frames[min(k, len(frames) - 1)]
        rgb = grab(fr["file"], fr["t"])
        tx = texts[min(k, len(texts) - 1)]
        img, boxes = fn(fr, rgb, tx["text"], tx.get("highlight"))
        ov = overlap(boxes["text"], boxes["face"])
        path = out_dir / f"kapak_{k + 1}.jpg"
        img.save(path, quality=92)
        results.append({"design": name, "file": str(path), "text": tx["text"], "frame_t": fr["t"],
                        "face_overlap": round(ov, 3), "ok": ov < 0.05, "method": boxes.get("method", "")})
    return results
