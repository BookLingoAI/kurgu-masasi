"""Where a punch-in zoom should look: the face while talking, the thing being shown while showing something.

object_focus(): spectral-residual saliency (OpenCV contrib) weighted by colour saturation and local sharpness - food,
products and hands holding things stand out against a table / room - with the speaker's face (and hair/shoulders
around it) masked out. Returns a centre and a confidence; a weak, scattered map means "nothing in particular is
shown" and the caller does not zoom at all rather than zooming into a random spot.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# words that announce "look at this" (Turkish, normalised)
SHOW_WORDS = {"burada", "burası", "buraya", "burda", "şurada", "şuraya", "şurası", "bakın", "bakalım", "bakar",
              "görüyorsunuz", "görüyor", "gördüğünüz", "göreceğiz", "göstereyim", "gösteriyorum", "gösterelim",
              "işte", "şuna", "şunu", "şu", "yakından", "tabak", "tabağı", "tabağımız", "porsiyon", "sunum",
              "geldi", "geliyor", "yemeğimiz", "yemeği", "tatlı", "tatlımız", "içeceğimiz", "menü", "menüsü"}


def object_focus(rgb: np.ndarray, face_box: tuple | None = None) -> tuple[float, float, float] | None:
    """rgb: frame. face_box: (x0, y0, x1, y1, w, h) in pixels of this frame or None.
    Returns (cx, cy, confidence) with cx, cy in 0..1, or None."""
    if cv2 is None or rgb is None:
        return None
    h0, w0 = rgb.shape[:2]
    W = 320
    H = max(2, int(round(h0 * W / w0)))
    small = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, m = sal.computeSaliency(small)
        m = m.astype(np.float32) if ok else None
    except Exception:  # noqa: BLE001
        m = None
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sharp = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 3)
    sharp /= sharp.max() + 1e-6
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    if m is None:
        m = sharp.copy()
    m = cv2.GaussianBlur(m, (0, 0), 4)
    m /= m.max() + 1e-6
    score = m * (0.4 + sat) * (0.5 + sharp)
    # the speaker: face, hair above it, neck/shoulders below it
    if face_box:
        x0, y0, x1, y1, fw, fh = face_box
        sx, sy = W / fw, H / fh
        bw, bh = (x1 - x0) * sx, (y1 - y0) * sy
        a0, a1 = int(max(0, x0 * sx - 0.6 * bw)), int(min(W, x1 * sx + 0.6 * bw))
        b0, b1 = int(max(0, y0 * sy - 0.7 * bh)), int(min(H, y1 * sy + 0.9 * bh))
        score[b0:b1, a0:a1] = 0
    # frame borders are rarely the subject
    bx, by = int(W * 0.06), int(H * 0.06)
    score[:by] *= 0.3
    score[-by:] *= 0.3
    score[:, :bx] *= 0.3
    score[:, -bx:] *= 0.3
    flat = score.ravel()
    if flat.max() <= 1e-6:
        return None
    thr = np.percentile(flat, 97)
    top = (score >= thr).astype(np.uint8)
    # the single strongest blob (not the average of a plate and a glass on opposite sides)
    top = cv2.dilate(top, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(top)
    if n < 2:
        return None
    sums = np.bincount(lab.ravel(), weights=score.ravel(), minlength=n)
    sums[0] = 0
    best = int(np.argmax(sums))
    blob = (lab == best) & (score >= thr)
    ys, xs = np.nonzero(blob)
    if len(xs) < 5:
        return None
    wts = score[blob]
    cx, cy = float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())
    # compact and much stronger than the rest = a real subject
    spread = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean())) / W
    conf = float(wts.mean() / (flat.mean() + 1e-6)) * (1.0 if spread < 0.18 else 0.6)
    return cx / W, cy / H, conf


def segment_words(words: list[dict], seg: dict) -> list[str]:
    from ..analysis.text import norm_word
    return [norm_word(w["word"]) for w in words if seg["timeline_start"] - 0.05 <= w["t_start"] < seg["timeline_end"]]
