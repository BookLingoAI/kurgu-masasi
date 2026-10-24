"""Local LLM through LM Studio's OpenAI-compatible server (http://127.0.0.1:1234).

Only HTTP is used: the bundled `lms` CLI on this machine does not match the LM Studio service version
("Invalid passkey"), so the server is started in the LM Studio app and models load just-in-time. Nothing is downloaded.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("KURGU_LLM_URL", "http://127.0.0.1:1234")
LOG = Path(os.environ.get("KURGU_ROOT", Path(__file__).resolve().parents[2])) / "logs" / "llm.log"
STATS: dict = {"calls": 0, "ok": 0, "seconds": 0.0, "tok_s": None, "model": None, "load_s": None}


def _log(rec: dict) -> None:
    """One JSON line per model call in logs/llm.log (duration, tokens, speed, errors): the only way to see later
    why the language model was slow or silent."""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(rec, t=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False) + "\n")
    except OSError:
        pass


def models_info() -> list[dict]:
    """LM Studio's model list with load state, quantization and context length (REST v0), or the plain list."""
    try:
        return _get("/api/v0/models", 10).get("data", [])
    except Exception:  # noqa: BLE001
        try:
            return _get("/v1/models", 10).get("data", [])
        except Exception:  # noqa: BLE001
            return []


class LLMUnavailable(RuntimeError):
    pass


def _get(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def loaded_instances() -> list[str]:
    """Instance ids of the language models LM Studio holds in memory right now (REST v1)."""
    try:
        return [i["id"] for m in _get("/api/v1/models", 10).get("models", []) if m.get("type") == "llm"
                for i in m.get("loaded_instances") or []]
    except Exception:  # noqa: BLE001
        return []


def unload(instance_id: str) -> bool:
    try:
        _post("/api/v1/models/unload", {"instance_id": instance_id}, 120)
        return True
    except Exception:  # noqa: BLE001
        return False


def free_gpu(progress=None) -> list[str]:
    """Take every language model out of the GPU before Whisper / Demucs / the render: a model left loaded (17.7 GB
    on a 12 GB card) slowed them down - measured 23.10.2026. LM Studio loads it again on the next request."""
    gone = [i for i in loaded_instances() if unload(i)]
    if gone and progress:
        progress(None, "Ekran kartı boşaltıldı (dil modeli bekliyor)")
    return gone


def server_up() -> bool:
    try:
        _get("/v1/models", 3)
        return True
    except Exception:  # noqa: BLE001
        return False


def model_key(cfg: dict) -> str:
    """Model id from the server's own list (LM Studio lists downloaded models and loads them on first request)."""
    info = models_info()
    ids = [m.get("id", "") for m in info if "embed" not in (m.get("id", "") + str(m.get("type", ""))).lower()]
    if cfg.get("model") and (cfg["model"] in ids or not info):
        return cfg["model"]     # the configured model - on another computer it may not exist: then choose below
    loaded = [m.get("id", "") for m in info if m.get("state") == "loaded" and m.get("id", "") in ids]
    if loaded:      # a model the user already loaded in LM Studio (with settings that fit the GPU) comes first
        return loaded[0]
    pref = cfg.get("prefer", "qwen")
    for i in ids:
        if pref and pref in i.lower():
            return i
    if ids:
        return ids[0]
    raise LLMUnavailable("LM Studio sunucusunda dil modeli görünmüyor")


def _lms() -> str | None:
    import shutil
    p = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms")
    return str(p) if p.exists() else shutil.which("lms")


def _run_lms(args: list[str], timeout: float) -> tuple[int, str]:
    exe = _lms()
    if not exe:
        return 1, "lms yok"
    try:
        p = subprocess.run([exe] + args, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 1, "zaman aşımı"


class LocalLLM:
    """LM Studio over HTTP. If the server is down it is started with the `lms` CLI; the model is loaded with
    settings that fit a 12 GB GPU and unloaded again after the job (frees VRAM for video work)."""

    def __init__(self, cfg: dict, progress=None):
        self.cfg = cfg
        self.key = None
        self.loaded_by_us = False
        self.progress = progress or (lambda f, m: None)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self) -> None:
        if not server_up():
            self.progress(0.0, "LM Studio sunucusu başlatılıyor")
            _run_lms(["server", "start", "--port", "1234"], 120)
            for _ in range(45):
                if server_up():
                    break
                time.sleep(2)
            else:
                raise LLMUnavailable("LM Studio sunucusu başlatılamadı")
        self.key = model_key(self.cfg)
        loaded = loaded_instances()
        if self.key not in loaded:
            # one model at a time on the 12 GB card, loaded over REST (the lms CLI does not work on this machine,
            # so models were never unloaded before 23.10.2026)
            for other in loaded:
                unload(other)
            self.progress(0.02, "Dil modeli yükleniyor")
            body = {"model": self.key, "flash_attention": True}
            if self.cfg.get("context_length"):
                body["context_length"] = int(self.cfg["context_length"])
            try:
                _post("/api/v1/models/load", body, 600)
                self.loaded_by_us = True
            except Exception:  # noqa: BLE001 - LM Studio still loads it just-in-time on the first request
                self.loaded_by_us = False

    def warm(self, timeout: float = 600.0) -> float:
        """One tiny request so LM Studio loads the model now (just-in-time loading of a big model can take
        minutes - that made the first real request time out). Returns the seconds it took."""
        old = self.cfg.get("timeout_s")
        self.cfg["timeout_s"] = timeout
        t0 = time.time()
        try:
            self.cfg["tag"] = "warm"
            self.chat("Kısa cevap ver.", "Sadece 'tamam' yaz.", max_tokens=6, temperature=0.0, _retry=True)
        except RuntimeError:
            pass     # an empty answer is fine: the model is loaded
        finally:
            self.cfg["tag"] = ""
            if old is None:
                self.cfg.pop("timeout_s", None)
            else:
                self.cfg["timeout_s"] = old
        STATS["load_s"] = round(time.time() - t0, 1)
        return STATS["load_s"]

    def stop(self) -> None:
        if self.loaded_by_us and self.cfg.get("unload_after", True) and self.key:
            unload(self.key)

    def raw(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.6,
            schema: dict | None = None) -> str:
        """Answer WITHOUT the model's hidden "thinking". Qwen3-style models in LM Studio ignored /no_think and
        enable_thinking=false: they thought for hundreds of tokens at ~8 tokens/s, ran out of budget and returned
        nothing (every title/Shorts request timed out). Through the plain completions endpoint the prompt is
        written in the model's own chat format with an empty think block - exactly what the template does when
        thinking is off - so the answer starts at once."""
        if schema:
            user = user + "\n\nCevap bu JSON şemasına uymalı: " + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        body = {"model": self.key, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]}
        t0 = time.time()
        rec = {"model": self.key, "mode": "raw", "prompt_chars": len(prompt), "max_tokens": max_tokens,
               "timeout_s": self.cfg.get("timeout_s", 900), "tag": self.cfg.get("tag", "")}
        STATS["calls"] += 1
        try:
            r = _post("/v1/completions", body, timeout=self.cfg.get("timeout_s", 900))
        except Exception as e:  # noqa: BLE001
            _log(dict(rec, seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}"[:300]))
            raise
        dt = time.time() - t0
        u = r.get("usage") or {}
        ct = u.get("completion_tokens")
        STATS["ok"] += 1
        STATS["seconds"] += dt
        STATS["model"] = self.key
        if ct and dt > 0:
            STATS["tok_s"] = round(ct / dt, 2)
        text = strip_think((r.get("choices") or [{}])[0].get("text") or "")
        _log(dict(rec, seconds=round(dt, 1), prompt_tokens=u.get("prompt_tokens"), completion_tokens=ct,
                  finish=(r.get("choices") or [{}])[0].get("finish_reason"), answer=text[:120]))
        return text

    def chat(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.6,
             schema: dict | None = None, _retry: bool = False) -> str:
        if self.cfg.get("raw", True) and not self.cfg.get("_raw_broken"):
            try:
                text = self.raw(system, user, max_tokens, temperature, schema)
                if text.strip():
                    return text
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 422, 501):     # no completions endpoint here: use chat from now on
                    self.cfg["_raw_broken"] = True
                else:
                    raise
        body = {
            "model": self.key,
            # Qwen3 reads the /no_think switch from the USER turn; in the system turn it was ignored and the model
            # spent the whole token budget thinking (empty answer)
            "messages": [{"role": "system", "content": system + (" /no_think" if self.cfg.get("no_think", True) else "")},
                         {"role": "user", "content": user + (" /no_think" if self.cfg.get("no_think", True) else "")}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.cfg.get("no_think", True) and not _retry:
            # thinking models spent the whole budget "thinking": ask the server to switch it off (ignored if unknown)
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["reasoning_effort"] = "low"
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}}
        t0 = time.time()
        rec = {"model": self.key, "prompt_chars": len(system) + len(user), "max_tokens": max_tokens,
               "timeout_s": self.cfg.get("timeout_s", 900), "tag": self.cfg.get("tag", "")}
        STATS["calls"] += 1
        try:
            try:
                r = _post("/v1/chat/completions", body, timeout=self.cfg.get("timeout_s", 900))
            except urllib.error.HTTPError as e:
                if e.code in (400, 422):  # server without structured output / extra fields: retry plain
                    rec["retry_plain"] = e.read().decode("utf-8", "replace")[:300]
                    body.pop("response_format", None)
                    body.pop("chat_template_kwargs", None)
                    body.pop("reasoning_effort", None)
                    r = _post("/v1/chat/completions", body, timeout=self.cfg.get("timeout_s", 900))
                else:
                    raise
        except Exception as e:  # noqa: BLE001
            _log(dict(rec, seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}"[:300]))
            raise
        dt = time.time() - t0
        u = r.get("usage") or {}
        ct = u.get("completion_tokens")
        STATS["ok"] += 1
        STATS["seconds"] += dt
        STATS["model"] = self.key
        if ct and dt > 0:
            STATS["tok_s"] = round(ct / dt, 2)
        _log(dict(rec, seconds=round(dt, 1), prompt_tokens=u.get("prompt_tokens"), completion_tokens=ct,
                  finish=(r.get("choices") or [{}])[0].get("finish_reason"), stats=r.get("stats")))
        ch = r["choices"][0]
        msg = ch.get("message") or {}
        text = strip_think(msg.get("content") or "")
        if not text.strip():
            # thinking models: the answer may sit in reasoning_content, or the budget ran out while thinking
            if ch.get("finish_reason") == "length" and not _retry:
                return self.chat(system, user, max(6000, max_tokens * 4), temperature, schema, _retry=True)
            raise RuntimeError(f"Model boş cevap verdi (bitiş nedeni: {ch.get('finish_reason')})")
        return text

    def describe_images(self, jpegs: list[bytes], prompt: str, max_tokens: int = 120) -> str:
        """Ask a vision-capable model about a few frames. Raises if the loaded model cannot see images."""
        import base64
        content = [{"type": "text", "text": prompt + (" /no_think" if self.cfg.get("no_think", True) else "")}]
        for j in jpegs:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}})
        model = self.cfg.get("vision_model") or self.key
        r = _post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}],
                                           "max_tokens": max_tokens, "temperature": 0.2}, timeout=300)
        text = strip_think((r["choices"][0].get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError("Model görüntüyü tarif etmedi")
        return text

    def see_json(self, system: str, user: str, jpegs: list[bytes], schema: dict | None = None,
                 max_tokens: int = 900, temperature: float = 0.3) -> dict:
        """Ask about images and get JSON back. `"reasoning_effort": "none"` is what really switches Qwen's hidden
        thinking off on the chat endpoint (measured: 10 s instead of 50-98 s; enable_thinking=false is ignored)."""
        import base64
        if schema:
            user = user + "\n\nCevap bu JSON şemasına uymalı: " + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        content = [{"type": "text", "text": user}]
        for j in jpegs:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}})
        body = {"model": self.cfg.get("vision_model") or self.key, "temperature": temperature, "max_tokens": max_tokens,
                "reasoning_effort": "none",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        rec = {"model": body["model"], "mode": "see", "prompt_chars": len(system) + len(user), "images": len(jpegs),
               "max_tokens": max_tokens, "tag": self.cfg.get("tag", "")}
        t0 = time.time()
        STATS["calls"] += 1
        try:
            r = _post("/v1/chat/completions", body, timeout=self.cfg.get("timeout_s", 900))
        except Exception as e:  # noqa: BLE001
            _log(dict(rec, seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}"[:300]))
            raise
        ch = (r.get("choices") or [{}])[0]
        text = strip_think((ch.get("message") or {}).get("content") or "")
        u = r.get("usage") or {}
        _log(dict(rec, seconds=round(time.time() - t0, 1), prompt_tokens=u.get("prompt_tokens"),
                  completion_tokens=u.get("completion_tokens"), finish=ch.get("finish_reason"), answer=text[:120]))
        STATS["ok"] += 1
        STATS["seconds"] += time.time() - t0
        return parse_json(text)

    def chat_json(self, system: str, user: str, schema: dict, **kw) -> dict:
        text = self.chat(system, user, schema=schema, **kw)
        try:
            return parse_json(text)
        except ValueError:   # json.JSONDecodeError is a ValueError: ask once more, strictly
            kw = dict(kw, temperature=0.1)
            text = self.chat(system, user + "\n\nÖNEMLİ: Yalnızca geçerli JSON yaz; açıklama, yorum ya da boş alan bırakma.",
                             schema=schema, **kw)
            return parse_json(text)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict:
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError(f"Model JSON döndürmedi: {text[:300]}")
    raw = text[a:b + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # common small-model slips: trailing commas, empty values, Python literals, comments
        fixed = re.sub(r"//[^\n]*", "", raw)
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        fixed = re.sub(r":\s*(?=[,}\]])", ": null", fixed)
        fixed = re.sub(r"\bNone\b", "null", re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", fixed)))
        return json.loads(fixed)
