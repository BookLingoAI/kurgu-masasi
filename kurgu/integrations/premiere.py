"""Premiere Pro client over the MCP Bridge file protocol.

The MCP Bridge CEP panel (running inside Premiere) polls a folder for cmd_<id>.jsx files, runs them in
ExtendScript and writes res_<id>.json. This is the same transport the premiere-pro-mcp server uses for its
`execute_extendscript` tool; we talk to it directly so the app does not depend on an MCP client.

All scripts get the ES3 helper library (es3/kurgu_lib.jsx) prepended.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def _forget_stale_engine() -> None:
    """The server drops edited modules from sys.modules so the next job imports new code, but
    `from kurgu import pipeline` still found the OLD module as an attribute of the `kurgu` package (so new
    pipeline code was never used until a restart). This module is imported fresh by every job after a reload:
    remove those leftovers so the next job really imports the new code."""
    import sys
    pkg = sys.modules.get("kurgu")
    if pkg is None:
        return
    for name in ("pipeline", "cache", "decisions"):
        if f"kurgu.{name}" not in sys.modules and name in vars(pkg):
            delattr(pkg, name)


_forget_stale_engine()

LIB_PATH = Path(__file__).parent / "es3" / "kurgu_lib.jsx"
TICKS_PER_SECOND = 254016000000


class PremiereError(RuntimeError):
    pass


class BridgeNotRunning(PremiereError):
    pass


def js_str(s: str) -> str:
    """Encode a Python string as an ES3 string literal."""
    return json.dumps(str(s), ensure_ascii=True)


def js_val(v) -> str:
    return json.dumps(v, ensure_ascii=True)


def seconds_to_ticks(s: float) -> int:
    return int(round(s * TICKS_PER_SECOND))


def snap_to_frame(s: float, fps: float) -> float:
    return round(s * fps) / fps


_WORKING_DIR: dict = {}  # bridge folder the panel was last seen polling


def _alt_dirs(primary: Path) -> list[Path]:
    """The panel may poll our folder or (if it fell back to its saved setting) its default temp folder."""
    out = [primary]
    tmp = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        out.append(Path(tmp) / "premiere-mcp-bridge")
    return out


class Premiere:
    def __init__(self, bridge_dir: str | Path, timeout: float = 120.0):
        self.primary = Path(bridge_dir)
        self.dir = _WORKING_DIR.get("dir", self.primary)
        self.timeout = timeout
        self._lib = LIB_PATH.read_text(encoding="utf-8")

    def _write_cmd(self, body: str, tag: str) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        cid = f"{int(time.time() * 1000):013d}_{tag}_{uuid.uuid4().hex[:6]}"
        script = "(function(){\n" + self._lib + "\n" + body + "\n})();"
        tmp = self.dir / f"tmp_{cid}.part"
        tmp.write_text(script, encoding="utf-8")
        os.replace(tmp, self.dir / f"cmd_{cid}.jsx")
        return cid

    def run(self, body: str, tag: str = "x", timeout: float | None = None):
        """Run ExtendScript `body` (must `return KL.ok(...)`/`KL.fail(...)`); return parsed data or raise."""
        try:
            return self._run(body, tag, timeout)
        except BridgeNotRunning:
            for d in _alt_dirs(self.primary):
                if d == self.dir:
                    continue
                old, self.dir = self.dir, d
                try:
                    r = self._run(body, tag, timeout)
                    _WORKING_DIR["dir"] = d
                    return r
                except BridgeNotRunning:
                    self.dir = old
            raise

    def _run(self, body: str, tag: str = "x", timeout: float | None = None):
        cid = self._write_cmd(body, tag)
        cmd = self.dir / f"cmd_{cid}.jsx"
        res = self.dir / f"res_{cid}.json"
        t0 = time.time()
        deadline = t0 + (timeout or self.timeout)
        pickup_deadline = t0 + 8.0
        picked = False
        while time.time() < deadline:
            if res.exists():
                time.sleep(0.05)
                try:
                    raw = res.read_text(encoding="utf-8")
                    data = json.loads(raw)
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.1)
                    continue
                try:
                    res.unlink()
                except OSError:
                    pass
                if not data.get("success", False):
                    raise PremiereError(data.get("error") or raw)
                return data.get("data")
            if not picked and not cmd.exists():
                picked = True
            if not picked and time.time() > pickup_deadline:
                # Not picked up within 8 s: panel not running.
                try:
                    cmd.unlink()
                except OSError:
                    pass
                raise BridgeNotRunning(
                    "Premiere'deki MCP Bridge paneli komutu almadı. Panel açık ve 'Running' durumda mı? "
                    f"(klasör: {self.dir})")
            time.sleep(0.1)
        raise PremiereError(f"Premiere yanıt vermedi ({tag}, {timeout or self.timeout:.0f} sn)")

    # ---- convenience -------------------------------------------------------------------------
    def ping(self) -> dict:
        return self.run("return KL.ok({version: app.version, project: app.project ? app.project.name : ''});", "ping", 15)

    def project_info(self) -> dict:
        return self.run("return KL.ok(KL.projectInfo());", "info")

    def sequence_state(self, name: str | None = None) -> dict:
        arg = js_str(name) if name else "null"
        return self.run(f"return KL.ok(KL.sequenceState({arg}));", "seqstate", 300)
