"""Faces and person masks (models in <root>/modeller).

- face detection:   blaze_face_short_range.tflite  (boxes + 6 key points: eyes, nose, mouth, ears)
- person mask:      selfie_segmenter.tflite
- expression:       face_landmarker.task  (optional: eyes open / smile for thumbnail picking)

The two main models are MediaPipe's own files, but they are run by OpenCV's DNN module (TFLite importer), not by
the MediaPipe runtime: Windows Smart App Control kept blocking MediaPipe's DLL (WinError 4551) and a blocked DLL
meant no faces, no intro, no thumbnails. OpenCV is already loaded by the app and is never blocked, and the results
match MediaPipe's to ~1 % of the frame (same model, same anchors, same letterboxing). MediaPipe itself is only
tried for the optional expression scores and silently skipped when it cannot load.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

MODEL_DIR = Path(os.environ.get("KURGU_ROOT", Path(__file__).resolve().parents[2])) / "modeller"
MODEL_FILES = {
    "face": "blaze_face_short_range.tflite",
    "segment": "selfie_segmenter.tflite",
    "landmarks": "face_landmarker.task",
}
MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "segment": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "landmarks": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

# kept for callers that report a blocked runtime; with the OpenCV path it stays empty
BLOCKED: dict = {"reason": None}


def model_path(kind: str) -> Path | None:
    p = MODEL_DIR / MODEL_FILES[kind]
    return p if p.exists() else None


# ------------------------------------------------------------------ OpenCV runtime -------------------------------
def _read_tflite(path: Path):
    """TFLite model in OpenCV. The classic engine exposes every named output (the new graph engine of OpenCV 5
    only returned the first one of BlazeFace)."""
    eng = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
    if eng is not None:
        try:
            return cv2.dnn.readNetFromTFLite(str(path), eng)
        except Exception:  # noqa: BLE001
            pass
    return cv2.dnn.readNetFromTFLite(str(path))


@lru_cache(maxsize=1)
def _cv_ok() -> bool:
    """OpenCV can run both models (checked once on a blank image)."""
    if cv2 is None or not hasattr(cv2.dnn, "readNetFromTFLite") or not (model_path("face") and model_path("segment")):
        return False
    try:
        _Blaze().detect(np.zeros((90, 160, 3), np.uint8))
        _Selfie().mask(np.zeros((90, 160, 3), np.uint8))
        return True
    except Exception:  # noqa: BLE001
        return False


def _anchors() -> np.ndarray:
    """SSD anchors of BlazeFace short range: 128 px input, strides 8 (2 per cell) and 16 (6 per cell) = 896."""
    out = []
    for stride, n in ((8, 2), (16, 6)):
        g = 128 // stride
        for y in range(g):
            for x in range(g):
                out += [((x + 0.5) / g, (y + 0.5) / g)] * n
    return np.array(out, np.float32)


_ANCHORS = _anchors() if cv2 is not None else None


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


class _Blaze:
    """BlazeFace exactly as MediaPipe runs it: keep-aspect letterbox to 128x128, [-1, 1] input, sigmoid scores,
    weighted non-max suppression (IoU 0.3)."""

    def __init__(self):
        self.net = _read_tflite(model_path("face"))
        self.lock = threading.Lock()

    def _forward(self, blob):
        with self.lock:
            self.net.setInput(blob)
            try:
                reg, cls = self.net.forward(["regressors", "classificators"])
            except Exception:  # noqa: BLE001 - engines that expose one output at a time
                reg = self.net.forward("regressors")
                self.net.setInput(blob)
                cls = self.net.forward("classificators")
        return reg.reshape(-1, 16), cls.reshape(-1)

    def detect(self, rgb: np.ndarray, thr: float = 0.5) -> list[dict]:
        h, w = rgb.shape[:2]
        s = 128.0 / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        ox, oy = (128 - nw) // 2, (128 - nh) // 2
        canvas = np.zeros((128, 128, 3), np.float32)
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1.0
        reg, raw = self._forward(cv2.dnn.blobFromImage(canvas))
        sc = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
        cands = []
        for i in np.nonzero(sc >= thr)[0]:
            ax, ay = _ANCHORS[i]
            cx, cy = reg[i, 0] / 128 + ax, reg[i, 1] / 128 + ay
            bw, bh = reg[i, 2] / 128, reg[i, 3] / 128
            box = [((cx - bw / 2) * 128 - ox) / nw, ((cy - bh / 2) * 128 - oy) / nh, bw * 128 / nw, bh * 128 / nh]
            kp = [(((reg[i, 4 + 2 * k] / 128 + ax) * 128 - ox) / nw, ((reg[i, 5 + 2 * k] / 128 + ay) * 128 - oy) / nh)
                  for k in range(6)]
            cands.append((box, kp, float(sc[i])))
        cands.sort(key=lambda c: -c[2])
        out = []
        while cands:
            best = cands[0]
            grp = [c for c in cands if _iou(best[0], c[0]) > 0.3]
            cands = [c for c in cands if _iou(best[0], c[0]) <= 0.3]
            wt = np.array([c[2] for c in grp], np.float32)
            b = (np.array([c[0] for c in grp], np.float32) * wt[:, None]).sum(0) / wt.sum()
            k = (np.array([c[1] for c in grp], np.float32) * wt[:, None, None]).sum(0) / wt.sum()
            out.append({"x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3]), "score": best[2],
                        "kp": [(float(p[0]), float(p[1])) for p in k]})
        out.sort(key=lambda f: -f["w"] * f["h"])
        return out


class _Selfie:
    """MediaPipe selfie segmenter: 256x256 RGB in [0, 1] -> person probability."""

    def __init__(self):
        self.net = _read_tflite(model_path("segment"))
        self.lock = threading.Lock()

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        with self.lock:
            self.net.setInput(cv2.dnn.blobFromImage(x))
            o = self.net.forward()
        return np.squeeze(o).astype(np.float32)


_shared: dict = {}


def _shared_selfie() -> "_Selfie":
    if "selfie" not in _shared:
        _shared["selfie"] = _Selfie()
    return _shared["selfie"]


# ------------------------------------------------------------------ optional MediaPipe ----------------------------
@lru_cache(maxsize=1)
def _mp():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401

        return mp
    except Exception:  # noqa: BLE001
        return None


def _is_blocked(e: BaseException) -> bool:
    t = str(e)
    return "4551" in t or "Uygulama Denetimi" in t or "Application Control" in t or "0xc0e90002" in t


@lru_cache(maxsize=1)
def mediapipe_ok() -> bool:
    """MediaPipe's native library loads (one quiet try; Smart App Control may block it). Only optional features
    use it, so a block is never reported as a problem."""
    if (os.environ.get("KURGU_MEDIAPIPE") != "1" and _cv_ok()) or not _mp():
        return False      # never touch the blocked DLL when OpenCV already does the job (no Windows warnings)
    try:
        import importlib
        cb = importlib.import_module("mediapipe.tasks.python.core.mediapipe_c_bindings")
        if hasattr(cb, "load_raw_library"):
            cb.load_raw_library()
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_runtime(tries: int = 1) -> bool:   # backwards compatible name
    return mediapipe_ok()


def backend() -> str:
    if _cv_ok():
        return "opencv"
    if _mp() and model_path("face") and mediapipe_ok():
        return "mediapipe"
    return "yok"


def have_faces() -> bool:
    return backend() != "yok"


def _image(rgb: np.ndarray):
    mp = _mp()
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class FaceDetector:
    def __init__(self, expression: bool = False):
        self.kind = backend()
        self.ok = self.kind != "yok"
        self._det = None
        self._lm = None
        if self.kind == "opencv":
            self._det = _Blaze()
        elif self.kind == "mediapipe":
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision

            self._det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model_path("face"))), min_detection_confidence=0.5))
        if expression and model_path("landmarks") and mediapipe_ok():
            try:
                from mediapipe.tasks import python as mpt
                from mediapipe.tasks.python import vision
                self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=mpt.BaseOptions(model_asset_path=str(model_path("landmarks"))),
                    output_face_blendshapes=True, num_faces=1))
            except Exception:  # noqa: BLE001
                self._lm = None

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Faces as {x, y, w, h, score, kp} normalized (0..1), largest first. kp: right eye, left eye, nose tip,
        mouth, right ear, left ear (subject's sides). Empty if no model."""
        if not self.ok:
            return []
        if self.kind == "opencv":
            return self._det.detect(rgb)
        h, w = rgb.shape[:2]
        res = self._det.detect(_image(rgb))
        faces = []
        for d in res.detections or []:
            bb = d.bounding_box
            faces.append({"x": bb.origin_x / w, "y": bb.origin_y / h, "w": bb.width / w, "h": bb.height / h,
                          "score": float(d.categories[0].score) if d.categories else 0.5,
                          "kp": [(k.x, k.y) for k in (d.keypoints or [])]})
        faces.sort(key=lambda f: -f["w"] * f["h"])
        return faces

    def expression(self, rgb: np.ndarray) -> dict:
        """Blendshape summary: eyes_open, smile, mouth_open, brow_up (0..1). Empty dict when unavailable."""
        if self._lm is None:
            return {}
        try:
            res = self._lm.detect(_image(rgb))
        except Exception:  # noqa: BLE001
            return {}
        if not res.face_blendshapes:
            return {}
        b = {c.category_name: c.score for c in res.face_blendshapes[0]}
        return {
            "eyes_open": 1.0 - (b.get("eyeBlinkLeft", 0) + b.get("eyeBlinkRight", 0)) / 2,
            "smile": (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2,
            "mouth_open": b.get("jawOpen", 0),
            "brow_up": b.get("browInnerUp", 0),
        }


def yaw_from_face(face: dict) -> float | None:
    """Horizontal head turn from the detector's key points: nose offset from the eye midpoint / eye distance.
    0 = facing the camera; negative = nose towards image left (same scale as the old landmark yaw)."""
    kp = face.get("kp") or []
    if len(kp) < 3:
        return None
    (rx, _), (lx, _), (nx, _) = kp[0], kp[1], kp[2]
    d = abs(lx - rx) + 1e-6
    return float((nx - (lx + rx) / 2) / d)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def person_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Soft person mask (0..1), or None without the model."""
    m = None
    if _cv_ok():
        m = _shared_selfie().mask(rgb)
    elif _mp() and model_path("segment") and mediapipe_ok():
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path("segment"))), output_confidence_masks=True))
        masks = seg.segment(_image(rgb)).confidence_masks or []
        if masks:
            m = masks[-1].numpy_view().astype(np.float32)
            if m.ndim == 3:
                m = m[..., 0]
    if m is None:
        return None
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]))
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return np.clip((m - 0.35) / 0.3, 0, 1)
