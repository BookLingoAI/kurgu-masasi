"""Sentence similarity with fastembed (multilingual MiniLM, ONNX). Optional: returns None if unavailable."""
from __future__ import annotations

from functools import lru_cache

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL)


def paraphrase_scorer():
    """Return f(a, b) -> cosine similarity, or None (with reason) if fastembed is not usable."""
    try:
        import numpy as np

        m = _model()
    except Exception as e:  # noqa: BLE001
        return None, f"fastembed kullanılamıyor: {e}"
    cache: dict[str, "np.ndarray"] = {}

    def vec(t: str):
        if t not in cache:
            v = next(iter(m.embed([t])))
            cache[t] = v / (np.linalg.norm(v) + 1e-9)
        return cache[t]

    def sim(a: str, b: str) -> float:
        return float(vec(a) @ vec(b))

    return sim, "fastembed"
