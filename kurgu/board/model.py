"""Which local model the board uses and what it can really do - discovered and tested, never assumed.

LM Studio's own model list (/api/v0/models: id, type llm/vlm, state, context) picks the model: the one set in
design_config "llm.model", else the loaded one (a vision model first), else the first downloaded one. Before the first
meeting with a model two small real tests run on it and the result is kept in ajanlar/model.json:
  json   - asked for a fixed JSON object, must parse
  vision - shown a generated picture with "KURUL 47" written on it, must read 47
An agent gets evidence pictures only when vision passed; otherwise it works from the measurements alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..integrations import llm as L

RETEST_DAYS = 30


def discover() -> list[dict]:
    """Models LM Studio knows about (downloaded), with type and state; [] if the server is down."""
    out = []
    for m in L.models_info():
        kind = m.get("type") or ("embeddings" if "embed" in m.get("id", "") else "llm")
        out.append({"id": m.get("id"), "type": kind, "state": m.get("state"),
                    "context": m.get("loaded_context_length") or m.get("max_context_length"),
                    "quant": m.get("quantization"), "arch": m.get("arch")})
    return out


def choose(cfg: dict, models: list[dict] | None = None) -> dict | None:
    models = [m for m in (models if models is not None else discover()) if m["type"] in ("llm", "vlm")]
    if not models:
        return None
    if cfg.get("model"):
        return next((m for m in models if m["id"] == cfg["model"]), {"id": cfg["model"], "type": "?", "state": None})
    loaded = [m for m in models if m.get("state") == "loaded"]
    pool = loaded or models
    prefer = str(cfg.get("prefer") or "").lower()
    pool = sorted(pool, key=lambda m: (m["type"] != "vlm", prefer not in m["id"].lower()))
    return pool[0]


def _probe_image() -> bytes:
    import cv2
    import numpy as np
    img = np.full((140, 420, 3), 255, np.uint8)
    cv2.putText(img, "KURUL 47", (24, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def capabilities(llm, root: Path, force: bool = False) -> dict:
    """{"model", "json", "vision", "tested"} for the model llm uses - from ajanlar/model.json when fresh."""
    p = Path(root) / "ajanlar" / "model.json"
    key = getattr(llm, "key", None)
    if not force and p.exists():
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            if c.get("model") == key and time.time() - c.get("t", 0) < RETEST_DAYS * 86400:
                return c
        except ValueError:
            pass
    caps = {"model": key, "t": time.time(), "tested": time.strftime("%Y-%m-%d %H:%M"), "json": False, "vision": False}
    try:
        r = llm.chat_json("Sadece JSON yaz.", 'Şunu aynen yaz: {"tamam": true, "sayi": 3}', {"type": "object"},
                          max_tokens=40, temperature=0.0)
        caps["json"] = r.get("tamam") is True and int(r.get("sayi", 0)) == 3
    except Exception as e:  # noqa: BLE001
        caps["json_hata"] = str(e)[:120]
    try:
        r = llm.see_json("Sadece JSON yaz.", 'Resimdeki yazıyı oku. {"metin": "..."} biçiminde yaz.', [_probe_image()],
                         max_tokens=40, temperature=0.0)
        caps["vision"] = "47" in str(r.get("metin", ""))
        caps["vision_cevap"] = str(r.get("metin", ""))[:40]
    except Exception as e:  # noqa: BLE001 - a text-only model: no pictures for the agents
        caps["vision_hata"] = str(e)[:120]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(caps, ensure_ascii=False, indent=1), encoding="utf-8")
    return caps
