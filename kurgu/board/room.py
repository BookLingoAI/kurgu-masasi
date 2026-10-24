"""The board room server: `python -m kurgu.board.room` -> http://127.0.0.1:8771 (only this computer).

Separate from the main app server on purpose: the main server only reloads editing modules, new routes there would
need a restart in the middle of the user's jobs. This one serves the room page, the voices, the live meeting's events
and turns the microphone into text with Whisper on the CPU (the GPU belongs to the language model during a meeting).
A meeting does not start while a video job is running - both need the GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..config import Config
from ..integrations import ffmpeg as ff

WEB = Path(__file__).resolve().parents[1] / "web"
PORT = int(os.environ.get("KURGU_KURUL_PORT", "8771"))


class Room:
    def __init__(self, root: Path | None = None):
        self.cfg = Config.load(root)
        self.root = Path(self.cfg.root)
        self.pdir = self.cfg.path("projects_dir")   # Videolarım: one folder per video
        self.live = None
        self.project = None
        self._stt = None
        self._stt_lock = threading.Lock()

    # ---- projects and jobs ------------------------------------------------------------------------------------
    def projects(self) -> list[dict]:
        out = []
        for d in sorted(self.pdir.iterdir() if self.pdir.exists() else []):
            k = d / "kurgu.json"
            if d.is_dir() and k.exists():
                info = json.loads(k.read_text(encoding="utf-8"))
                st = json.loads((d / "proje.json").read_text(encoding="utf-8")) if (d / "proje.json").exists() else {}
                out.append({"ad": d.name, "baslik": st.get("title") or d.name, "sure_dk": round(info.get("duration", 0) / 60, 1),
                            "tutanak": (d / "kurul" / "tutanak.json").exists()})
        return out

    def job_running(self) -> str | None:
        busy = list((self.root / "isler" / "sonuc").glob("*.calisiyor.json")) + list((self.root / "isler" / "gelen").glob("*.json"))
        return busy[0].name if busy else None

    def start(self, name: str) -> dict:
        from .. import pipeline
        from ..integrations.llm import LocalLLM
        from .live import LiveMeeting
        if self.live and self.live.state not in ("bitti", "kapandı", "hata"):
            return {"hata": "Zaten bir toplantı sürüyor."}
        job = self.job_running()
        if job:
            return {"hata": f"Şu an bir video işi çalışıyor ({job}). Ekran kartı ikisine yetmez; iş bitince başlat."}
        proj = pipeline.Project(Config.load(self.root), name)
        if not (proj.dir / "kurgu.json").exists():
            return {"hata": "Bu projede henüz video yok."}
        # the model may need minutes to load (LM Studio unloads it when idle): the meeting starts at once and
        # says so in the room instead of keeping the page waiting
        llm = LocalLLM(dict(self.cfg.design.get("llm", {})))
        self.live = LiveMeeting(proj, llm, warm=True, busy=self.job_running)    # pauses while a video job runs
        self.project = name
        self.live.start()
        return {"tamam": True}

    # ---- the user's decisions: explicit buttons, then preview / apply / undo (kurgu/board/apply.py) ------------
    def _project(self, name: str):
        from .. import pipeline
        return pipeline.Project(Config.load(self.root), name)

    def decisions(self, name: str) -> dict:
        from .apply import preview
        p = self.pdir / name / "kurul" / "tutanak.json"
        if not name or not p.exists():
            return {"kararlar": []}
        m = json.loads(p.read_text(encoding="utf-8"))
        proj = self._project(name)
        done = {}
        log = p.parent / "islemler.jsonl"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("bulgu") and r["islem"] == "uygula":
                    done[r["bulgu"]] = r["geri_al"]
                elif r["islem"] == "geri_al":
                    done = {k: v for k, v in done.items() if v != r["geri_al"]}
        out = []
        for f in m["bulgular"]:
            if f["karar"] not in ("kabul", "gelistirici_isi"):
                continue
            pv = preview(proj, f)
            out.append({"id": f["id"], "ajan": f["ajan"], "baslik": f["baslik"], "kanit": f["kanit"], "eylem": f["eylem"],
                        "destek": f["destek"], "karar": f["karar"], "kullanici": f.get("kullanici"),
                        "uygulanabilir": pv["uygulanabilir"], "neden": pv.get("neden"), "degisiklik": pv.get("degisiklik"),
                        "uygulandi": done.get(f["id"])})
        return {"proje": name, "kararlar": out}

    def act(self, what: str, j: dict) -> dict:
        from . import apply as AP
        proj = self._project(j.get("proje") or self.project or "")
        if what == "geri_al":
            return AP.undo(proj, j["geri_al"])
        m = json.loads((proj.dir / "kurul" / "tutanak.json").read_text(encoding="utf-8"))
        f = next((f for f in m["bulgular"] if f["id"] == j.get("id")), None)
        if not f:
            return {"tamam": False, "neden": "bulgu yok"}
        if what == "karar":
            return AP.decide(proj, f["id"], j.get("karar") == "onay")
        if what == "onizle":
            return AP.preview(proj, f)
        if f.get("kullanici") != "onay":
            return {"tamam": False, "neden": "Önce bu öneriyi onaylaman gerekiyor."}
        return AP.apply(proj, f)

    # ---- starting by itself: a new project's video is ready and Premiere is open -> one meeting, once ------------
    def premiere_failed(self, name: str) -> bool:
        """The video finished but its Premiere review project could not be built (kurgu.json says so)."""
        try:
            info = json.loads((self.pdir / name / "kurgu.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return "Premiere inceleme projesi kurulamadı" in str(info.get("engine_note") or "")

    def premiere_running(self) -> bool:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Adobe Premiere Pro.exe", "/NH"], capture_output=True,
                           text=True, creationflags=ff._CREATE_NO_WINDOW)
        return "Adobe Premiere Pro.exe" in (r.stdout or "")

    def due(self) -> str | None:
        """The project that should get its meeting now, if any. Only videos finished after the board was installed
        (ajanlar/oto.json) and only once per project (kurul/oto.json) - old projects and re-renders never start one."""
        mark = self.root / "ajanlar" / "oto.json"
        if not mark.exists():
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text(json.dumps({"kurulum": time.time()}), encoding="utf-8")
        since = json.loads(mark.read_text(encoding="utf-8")).get("kurulum", time.time())
        for d in sorted(self.pdir.glob("*/kurgu.json"), key=lambda p: -p.stat().st_mtime):
            pdir = d.parent
            if (pdir / "kurul" / "oto.json").exists() or d.stat().st_mtime < since:
                continue
            info = json.loads(d.read_text(encoding="utf-8"))
            if info.get("file") and Path(info["file"]).exists():
                return pdir.name
        return None

    def tick(self) -> str | None:
        """One look: start the due project's meeting if nothing else needs the GPU and Premiere is open - or Premiere
        could not be opened for this video (blocked by Windows on 23.10.2026): then the board meets without it."""
        busy = self.live and self.live.state not in ("bitti", "kapandı", "hata")
        name = None if busy or self.job_running() else self.due()
        if not name or not (self.premiere_running() or self.premiere_failed(name)):
            return None
        kd = self.pdir / name / "kurul"
        kd.mkdir(parents=True, exist_ok=True)
        (kd / "oto.json").write_text(json.dumps({"basladi": time.time()}), encoding="utf-8")   # once, even if it fails
        r = self.start(name)
        if r.get("hata"):
            print(f"bekçi: {name}: {r['hata']}", flush=True)
            return None
        open_window(auto=True)
        return name

    def watch(self, every: float = 15.0) -> None:
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - the watcher must not die
                print(f"bekçi: {e}", flush=True)
            time.sleep(every)

    # ---- speech to text ---------------------------------------------------------------------------------------
    def transcribe(self, blob: bytes) -> str:
        """The browser's recording (webm/opus) -> 16 kHz mono -> Whisper large-v3-turbo on the CPU, Turkish."""
        p = subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                           input=blob, capture_output=True, creationflags=ff._CREATE_NO_WINDOW, timeout=60)
        audio = np.frombuffer(p.stdout, np.float32)
        if len(audio) < 1600:
            return ""
        with self._stt_lock:
            if self._stt is None:
                from faster_whisper import WhisperModel
                self._stt = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
            segs, _ = self._stt.transcribe(audio, language="tr", beam_size=1, vad_filter=True,
                                           initial_prompt="Derya, Zeynep, Selin, Can, Emre, Mert, kurul, kurgu, yazı, kesim.")
            return " ".join(s.text.strip() for s in segs).strip()


ROOM: Room | None = None


def open_window(auto: bool = False) -> None:
    """The room in its own window (Edge/Chrome app mode, own profile). Sound may start without a click, so a
    meeting that starts by itself is heard at once."""
    from ..desktop import ROOT, _browser_app
    url = f"http://127.0.0.1:{PORT}/" + ("?oto=1" if auto else "")
    exe = _browser_app()
    if not exe:
        import webbrowser
        webbrowser.open(url)
        return
    profile = ROOT / "logs" / "pencere_kurul"
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1400,900", "--no-first-run",
                      "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
                      "--disable-features=Translate"], close_fds=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/kurul", "/kurul.html"):
            return self._send(200, (WEB / "kurul.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/kurul/projeler":
            return self._json({"projeler": ROOM.projects(), "is": ROOM.job_running(), "proje": ROOM.project})
        if u.path == "/api/kurul/olaylar":
            if not ROOM.live:
                return self._json({"events": [], "durum": "hazır", "dusunen": None})
            return self._json(ROOM.live.since(int(q.get("after", ["-1"])[0])))
        if u.path == "/api/kurul/profil":
            from . import profiles as PR
            name = q.get("proje", [ROOM.project or ""])[0]
            if not name:
                return self._json({"profiller": {k: v["ad"] for k, v in PR.PROFILES.items()}, "profil": None})
            prof = PR.load(ROOM._project(name)) or PR.resolve(ROOM._project(name))
            return self._json({"profiller": {k: v["ad"] for k, v in PR.PROFILES.items()}, "profil": prof})
        if u.path == "/api/kurul/paketler":
            from .packages import index_local
            cat = index_local(ROOM.root)
            return self._json({"paketler": [{k: p.get(k) for k in ("id", "tur", "ad", "lisans", "durum", "neden", "konum")}
                                            for p in cat["paketler"]]})
        if u.path == "/api/kurul/kararlar":
            return self._json(ROOM.decisions(q.get("proje", [ROOM.project or ""])[0]))
        if u.path.startswith("/ses/") and ROOM.live:
            f = (ROOM.live.voice_dir / Path(u.path).name)
            if f.exists() and f.suffix == ".wav":
                return self._send(200, f.read_bytes(), "audio/wav")
        return self._send(404, b"yok", "text/plain")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if u.path == "/api/kurul/baslat":
            name = json.loads(body or b"{}").get("proje", "")
            try:
                return self._json(ROOM.start(name))
            except Exception as e:  # noqa: BLE001
                return self._json({"hata": f"Toplantı başlatılamadı: {e}"[:300]})
        if u.path == "/api/kurul/soz":
            try:
                text = ROOM.transcribe(body)
            except Exception as e:  # noqa: BLE001
                return self._json({"hata": f"Ses yazıya çevrilemedi: {e}"[:200]})
            if text and ROOM.live:
                ROOM.live.user_says(text)
            return self._json({"metin": text or "(anlaşılamadı)"})
        if u.path == "/api/kurul/yaz":
            text = json.loads(body or b"{}").get("metin", "").strip()
            if text and ROOM.live:
                ROOM.live.user_says(text)
            return self._json({"metin": text})
        if u.path == "/api/kurul/profil":
            from . import profiles as PR
            j = json.loads(body or b"{}")
            try:
                return self._json({"profil": PR.set_by_user(ROOM._project(j.get("proje") or ROOM.project or ""), j.get("profil", ""))})
            except Exception as e:  # noqa: BLE001
                return self._json({"hata": str(e)[:200]})
        if u.path in ("/api/kurul/karar", "/api/kurul/onizle", "/api/kurul/uygula", "/api/kurul/geri_al"):
            j = json.loads(body or b"{}")
            try:
                return self._json(ROOM.act(u.path.rsplit("/", 1)[1], j))
            except Exception as e:  # noqa: BLE001
                return self._json({"tamam": False, "neden": f"{type(e).__name__}: {e}"[:200]})
        return self._send(404, b"yok", "text/plain")


def server_up() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/kurul/projeler", timeout=1.5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    global ROOM
    if sys.stdout is None or not sys.stdout.isatty():      # pythonw: no console - keep a log like the main server
        log = Path(__file__).resolve().parents[2] / "logs" / f"kurul-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
        log.parent.mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(log, "a", encoding="utf-8", buffering=1)
    ROOM = Room()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Kurgu Kurulu odası: http://127.0.0.1:{PORT}", flush=True)
    if os.environ.get("KURGU_KURUL_OTO", "1") == "1":
        threading.Thread(target=ROOM.watch, name="kurul-bekci", daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
