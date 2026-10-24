"""Local web UI at http://localhost:8770 (stdlib only): projects, video upload, jobs, results."""
from __future__ import annotations

import json
import mimetypes
import re
import shutil
import time
import unicodedata
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import Config, read_json, write_json
from ..decisions import DecisionLog
from ..media.sources import VIDEO_EXT, list_videos
from ..worker.jobs import JobManager

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
# upload kind -> (folder inside the project or None for the shared library, proje.json key, allowed extensions)
KINDS = {"kamera": ("medya/kamera", "camera_files", VIDEO_EXT), "ekran": ("medya/ekran", "screen_files", VIDEO_EXT),
         "ek": ("medya/ek", "broll", VIDEO_EXT | IMAGE_EXT), "ses": ("@ses_efektleri", None, AUDIO_EXT), "muzik": ("@muzik", None, AUDIO_EXT),
         "kapak": ("cikti/chatgpt/gelen", None, IMAGE_EXT)}

HTML = Path(__file__).parent / "index.html"
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def slug(name: str) -> str:
    s = unicodedata.normalize("NFC", Path(name).stem).translate(_TR)
    s = re.sub(r"[^\w\- ]+", "", s).strip().replace(" ", "-")
    return (s or "video")[:60]


_health_cache = {"t": 0.0, "v": None}


def health() -> dict:
    if time.time() - _health_cache["t"] < 15 and _health_cache["v"]:
        return _health_cache["v"]
    from ..integrations.llm import server_up
    from ..vision.faces import backend
    from ..audio.separate import installed
    from ..motion.fonts import installed as fonts_installed
    v = {"llm": server_up(), "faces": backend() != "yok", "separator": installed(), "fonts": len(fonts_installed()) > 0}
    _health_cache.update(t=time.time(), v=v)
    return v


def _dir_size(d: Path) -> int:
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


_size_cache: dict = {}


def dir_size_cached(d: Path, ttl: float = 120.0) -> int:
    hit = _size_cache.get(str(d))
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    v = _dir_size(d)
    _size_cache[str(d)] = (time.time(), v)
    return v


INTERMEDIATE = ("premiere/rotus_*.mp4", "premiere/rotus2_*.mp4", "premiere/rotus3_*.mp4", "premiere/kamera_kurgu_*.mp4",
                "premiere/montaj_*.mp4", "premiere/premiere_cikti.mp4", "premiere/konusma_temiz.wav", "premiere/pr_ses.wav", "premiere/pr_ses_norm.wav",
                "cikti/*_ses.wav", "cikti/*_ses_norm.wav", "cut_audio.wav")


KEEP_NEWEST = ("premiere/kamera_kurgu_*.mp4", "premiere/montaj_*.mp4")


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        (cfg.root / "ham").mkdir(exist_ok=True)
        cfg.path("projects_dir").mkdir(exist_ok=True)
        self.jobs = JobManager(cfg)

    # ---------------------------------------------------------------- projects
    def pdir(self, name: str) -> Path:
        if not re.fullmatch(r"[\w\-. ]{1,80}", name or "") or ".." in name:
            raise ValueError("Geçersiz proje adı")
        return self.cfg.path("projects_dir") / name

    def projects(self) -> list[dict]:
        out = []
        for d in sorted(self.cfg.path("projects_dir").iterdir(), key=lambda p: -p.stat().st_mtime):
            if not d.is_dir() or d.name.startswith("_") or not (d / "proje.json").exists():
                continue   # "_ortak" = shared caches, not a video
            st = read_json(d / "proje.json", {}) or {}
            plan = read_json(d / "edit_plan.json", None)
            thumbs = sorted((d / "cikti").glob("kapak_*.jpg")) if (d / "cikti").exists() else []
            out.append({
                "name": d.name, "title": st.get("title") or d.name, "created": st.get("created"),
                "source": [Path(f).name for f in st.get("camera_files", [])],
                "source_s": plan["stats"]["source_s"] if plan else None,
                "timeline_s": plan["stats"]["timeline_s"] if plan else None,
                "has_plan": bool(plan), "has_shorts": (d / "cikti" / "shorts.mp4").exists(),
                "has_package": (d / "paket.json").exists(),
                "thumb": str(thumbs[0]) if thumbs else None,
                "size_gb": round(dir_size_cached(d) / 1e9, 1),
            })
        return out

    def create_project(self, video: Path, title: str | None = None, more: list[Path] | None = None,
                       broll: list[Path] | None = None) -> dict:
        """A project REFERENCES the videos where they are (no copies)."""
        base = slug(title or video.name)
        name, k = base, 2
        while (self.cfg.path("projects_dir") / name).exists():
            name, k = f"{base}-{k}", k + 1
        d = self.cfg.path("projects_dir") / name
        d.mkdir(parents=True)
        st = {"title": title or Path(video.name).stem, "camera_files": [str(video)] + [str(m) for m in more or []],
              "screen_files": [], "broll": [{"file": str(b), "note": ""} for b in broll or []],
              "created": time.strftime("%Y-%m-%d %H:%M")}
        write_json(d / "proje.json", st)
        return {"name": name, **st}

    # ---------------------------------------------------------------- disk
    def disk(self) -> dict:
        u = shutil.disk_usage(str(self.cfg.root))
        return {"free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)}

    def find_existing(self, name: str, size: int) -> str | None:
        """A file with the same name and size already in the ham folder or any project: reuse it instead of uploading."""
        cands = [self.cfg.root / "ham" / name] + list(self.cfg.path("projects_dir").glob(f"*/medya/*/{name}"))
        for c in cands:
            try:
                if c.is_file() and c.stat().st_size == size:
                    return str(c)
            except OSError:
                pass
        return None

    # ---------------------------------------------------------------- native file picker (no copying)
    _pick_lock = __import__("threading").Lock()

    def pick(self, kind: str) -> dict:
        """Open the Windows file dialog on this PC; the chosen files are used where they are (no upload, no copy)."""
        import os
        import subprocess
        import sys
        import tempfile
        if not self._pick_lock.acquire(blocking=False):
            raise RuntimeError("Dosya seçme penceresi zaten açık (görev çubuğuna bakın)")
        try:
            last = self.cfg.root / "logs" / "son_klasor.txt"
            init = last.read_text(encoding="utf-8").strip() if last.exists() else str(self.cfg.root / "ham")
            out = Path(tempfile.gettempdir()) / f"kurgu_secim_{time.time_ns()}.json"
            p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("picker.py")), str(out), kind, init],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=0x08000000 if os.name == "nt" else 0)
            try:
                p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                p.kill()
                raise RuntimeError("Dosya seçimi zaman aşımına uğradı")
            res = read_json(out, None) or {"error": "Dosya seçme penceresi açılamadı"}
            out.unlink(missing_ok=True)
            if res.get("error"):
                raise RuntimeError(f"Dosya seçme penceresi açılamadı: {res['error']}")
            files = res.get("files") or []
            if files:
                last.parent.mkdir(exist_ok=True)
                last.write_text(str(Path(files[0]).parent), encoding="utf-8")
            return {"files": [self.file_info(f) for f in files]}
        finally:
            self._pick_lock.release()

    @staticmethod
    def file_info(f: str) -> dict:
        p = Path(f)
        d = {"path": str(p), "name": p.name, "gb": round(p.stat().st_size / 1e9, 2) if p.exists() else 0,
             "duration": None, "image": p.suffix.lower() in IMAGE_EXT}
        if p.suffix.lower() in VIDEO_EXT:
            try:
                from ..integrations import ffmpeg as ff
                d["duration"] = round(ff.probe(str(p)).duration, 1)
            except Exception:  # noqa: BLE001
                pass
        return d

    @staticmethod
    def media_path(x: str, exts) -> Path:
        """A file the user picked on this PC (any folder): must exist and have an allowed extension."""
        v = Path(str(x)).expanduser()
        if not v.is_absolute() or v.suffix.lower() not in exts or not v.is_file():
            raise ValueError(f"Dosya bulunamadı ya da türü desteklenmiyor: {v.name}")
        return v

    def cleanup(self, name: str) -> dict:
        """Delete re-creatable intermediate files of a project (user clicked the button). Sources, the plan,
        the finished video, Shorts and the Premiere project stay."""
        d = self.pdir(name)
        freed, n = 0, 0
        for pat in INTERMEDIATE:
            files = sorted(d.glob(pat), key=lambda f: f.stat().st_mtime)
            if pat in KEEP_NEWEST:   # the current Premiere project uses the newest one
                files = files[:-1]
            for f in files:
                try:
                    freed += f.stat().st_size
                    f.unlink()
                    n += 1
                except OSError:
                    pass
        _size_cache.pop(str(d), None)
        return {"files": n, "freed_gb": round(freed / 1e9, 1)}

    def delete_project(self, name: str) -> dict:
        """Remove a project folder (user confirmed). Videos in the ham folder are never touched; copies inside the project
        are removed only if no other project uses them."""
        d = self.pdir(name)
        if not d.exists():
            raise FileNotFoundError(name)
        used_elsewhere = set()
        for o in self.cfg.path("projects_dir").iterdir():
            if o.is_dir() and o != d:
                st = read_json(o / "proje.json", {}) or {}
                used_elsewhere |= set(st.get("camera_files", [])) | set(st.get("screen_files", [])) | \
                    {b["file"] for b in st.get("broll", [])}
        keep_dir = None
        if any(str(Path(f)).startswith(str(d)) for f in used_elsewhere):
            keep_dir = d / "medya"   # another project points at these copies: keep them
        size = dir_size_cached(d, ttl=0)
        for f in sorted(d.rglob("*"), key=lambda p: -len(p.parts)):
            if keep_dir and (f == keep_dir or keep_dir in f.parents):
                continue
            try:
                f.unlink() if f.is_file() else f.rmdir()
            except OSError:
                pass
        if not keep_dir:
            try:
                d.rmdir()
            except OSError:
                pass
        _size_cache.pop(str(d), None)
        return {"deleted": name, "freed_gb": round(size / 1e9, 1), "kept_media": bool(keep_dir)}

    def media(self, name: str) -> dict:
        from ..editing.director import options_of
        st = read_json(self.pdir(name) / "proje.json", {}) or {}
        info = lambda f: {"path": f, "name": Path(f).name, "ok": Path(f).exists(),  # noqa: E731
                          "mb": round(Path(f).stat().st_size / 1e6, 1) if Path(f).exists() else 0}
        lib = self.cfg.root / "ses_efektleri"
        mus = self.cfg.root / "muzik"
        return {"kamera": [info(f) for f in st.get("camera_files", [])],
                "ekran": [info(f) for f in st.get("screen_files", [])],
                "ek": [dict(info(b["file"]), note=b.get("note", ""), auto=b.get("auto", "")) for b in st.get("broll", [])],
                "ses": [info(str(f)) for f in sorted(lib.iterdir()) if f.suffix.lower() in AUDIO_EXT] if lib.exists() else [],
                "muzik": [info(str(f)) for f in sorted(mus.iterdir()) if f.suffix.lower() in AUDIO_EXT] if mus.exists() else [],
                "options": options_of(st)}

    def add_media(self, name: str, kind: str, path: Path, note: str = "") -> None:
        folder, key, _ = KINDS[kind]
        if key is None:
            return
        p = self.pdir(name) / "proje.json"
        st = read_json(p, {}) or {}
        lst = st.get(key, [])
        if kind == "ek":
            if not any(b["file"] == str(path) for b in lst):
                lst.append({"file": str(path), "note": (note or "")[:200]})
        elif str(path) not in lst:
            lst.append(str(path))
        st[key] = lst
        write_json(p, st)

    def relink(self, name: str, picked: list[str]) -> dict:
        """Recordings were moved: the user picks one or more of them in their new place; every missing file whose
        name is found next to the picked ones (same folder) gets its new path. Nothing is copied or moved."""
        p = self.pdir(name) / "proje.json"
        st = read_json(p, {}) or {}
        folders = {Path(x).parent for x in picked}
        found, still = 0, []

        def fix(path: str) -> str:
            nonlocal found
            if Path(path).exists():
                return path
            for d in folders:
                c = d / Path(path).name
                if c.exists():
                    found += 1
                    return str(c)
            still.append(Path(path).name)
            return path
        st["camera_files"] = [fix(f) for f in st.get("camera_files", [])]
        st["screen_files"] = [fix(f) for f in st.get("screen_files", [])]
        st["broll"] = [dict(b, file=fix(b["file"])) for b in st.get("broll", [])]
        write_json(p, st)
        return {"found": found, "missing": still, "media": self.media(name)}

    def edit_media(self, data: dict) -> dict:
        name, kind = data["project"], data["kind"]
        p = self.pdir(name) / "proje.json"
        st = read_json(p, {}) or {}
        act = data["action"]
        if act == "options":
            from ..editing.director import options_of
            st["options"] = options_of({"options": dict(st.get("options") or {}, **data.get("options", {}))})
        elif act == "add_existing":
            # files picked on this PC are used where they are; only small audio goes into the shared library
            if kind not in ("kamera", "ekran", "ek", "ses", "muzik"):
                raise ValueError("Bu tür buradan eklenemez")
            write_json(p, st)
            for x in data.get("paths") or [data["path"]]:
                src = self.media_path(x, KINDS[kind][2])
                if KINDS[kind][1] is None:
                    lib = self.cfg.root / KINDS[kind][0][1:]
                    lib.mkdir(exist_ok=True)
                    if not (lib / src.name).exists():
                        shutil.copy2(src, lib / src.name)
                    continue
                self.add_media(name, kind, src, data.get("note", ""))
            return self.media(name)
        elif act == "note_all" and kind == "ek":
            for b in st.get("broll", []):
                if not b.get("note"):
                    b["note"] = str(data.get("note", ""))[:200]
        elif kind in ("ses", "muzik") and act == "remove":
            f = (self.cfg.root / KINDS[kind][0][1:] / Path(data["path"]).name)
            if f.exists():
                f.unlink()
        else:
            folder, key, _ = KINDS[kind]
            lst = st.get(key, [])
            i = int(data["index"])
            if act == "remove":
                item = lst.pop(i)
                f = Path(item["file"] if isinstance(item, dict) else item)
                # only files uploaded into this project are deleted; originals elsewhere are just unlisted
                if (self.pdir(name) / "medya").resolve() in f.resolve().parents and f.exists():
                    f.unlink()
            elif act == "note" and kind == "ek":
                lst[i]["note"] = str(data.get("note", ""))[:200]
            st[key] = lst
        write_json(p, st)
        return self.media(name)

    def refresh_archive(self, name: str) -> str:
        """Update the YouTube folder after ChatGPT results arrive (in the background, if nothing else runs)."""
        if not (self.pdir(name) / "kurgu.json").exists():
            return "video yok"
        try:
            self.jobs.start("archive", name)
            return "güncelleniyor"
        except RuntimeError:
            return "başka iş çalışıyor; bitince “Klasörü güncelle”ye basın"

    def state(self, name: str) -> dict:
        d = self.pdir(name)
        if not d.exists():
            raise FileNotFoundError(name)
        plan = read_json(d / "edit_plan.json", None)
        analysis = read_json(d / "analysis.json", None)
        frames = sorted((d / "kareler").glob("*.png"))[-8:] if (d / "kareler").exists() else []
        return {
            "name": name,
            "settings": read_json(d / "proje.json", {}),
            "jobs": self.jobs.status(),
            "analysis": None if not analysis else {
                "created": analysis.get("created"), "sync": analysis.get("sync"), "energy": analysis.get("energy"),
                "device": analysis.get("transcript_device")},
            "plan": None if not plan else {"created": plan["created"], "stats": plan["stats"], "sequence": plan["sequence"],
                                           "notes": plan.get("notes", []), "n_segments": len(plan["segments"]),
                                           "review": [r for r in plan.get("review", []) if r["rule"] not in ("pause",)],
                                           "auto_counts": {k: sum(1 for r in plan.get("review", []) if r["rule"] == k and r["applied"])
                                                           for k in ("pause", "filler")}},
            "preview": read_json(d / "onizleme.json", None),
            "qa_cut": read_json(d / "qa_cut.json", None),
            "qa_premiere": read_json(d / "qa_premiere.json", None),
            "shorts": read_json(d / "shorts.json", None),
            "kurgu": read_json(d / "kurgu.json", None),
            "intro": read_json(d / "intro.json", None),
            "archive": read_json(d / "arsiv.json", None),
            "package": read_json(d / "paket.json", None),
            "director": read_json(d / "yonetmen.json", None),
            "media": self.media(name),
            "frames": [str(f) for f in frames],
            "feedback": read_json(d / "feedback.json", []),
        }


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "Kurgu/2"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def _file(self, path: Path, download: bool = False):
            root = app.cfg.root.resolve()
            p = path.resolve()
            if root not in p.parents:
                return self._json({"error": "izin yok"}, 403)
            if not p.exists():
                return self._json({"error": "yok"}, 404)
            size = p.stat().st_size
            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                    if m.group(2):
                        end = min(size - 1, int(m.group(2)))
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-cache")
            if download:
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{urllib.parse.quote(p.name)}")
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            try:
                if u.path == "/":
                    body = HTML.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/projects":
                    self._json({"projects": app.projects(), "ham": [{"path": v, "name": Path(v).name,
                                "mb": round(Path(v).stat().st_size / 1e6)} for v in list_videos(app.cfg.root / "ham")],
                                "jobs": app.jobs.status(), "health": health(), "disk": app.disk()})
                elif u.path == "/api/precheck":
                    name = Path(urllib.parse.unquote(q.get("name", [""])[0])).name
                    size = int(q.get("size", ["0"])[0] or 0)
                    disk = app.disk()
                    ex = app.find_existing(name, size)
                    need = size / 1e9 + 5
                    self._json({"existing": ex, "free_gb": disk["free_gb"], "ok": bool(ex) or disk["free_gb"] >= need,
                                "error": None if (ex or disk["free_gb"] >= need) else
                                f"Diskte yer yok: {disk['free_gb']:.0f} GB boş, bu dosya için {need:.0f} GB gerekiyor. "
                                "Ana sayfadaki eski projeleri silin ya da ara dosyalarını temizleyin."})
                elif u.path == "/api/state":
                    self._json(app.state(q.get("p", [""])[0]))
                elif u.path == "/api/transcript":
                    d = app.pdir(q.get("p", [""])[0])
                    pl = read_json(d / "edit_plan.json", {}) or {}
                    t = read_json(d / "transcript.json", {}) or {}
                    self._json({"source": t.get("words", []), "cut": pl.get("transcript_cut", []),
                                "removals": pl.get("removals", [])})
                elif u.path == "/api/decisions":
                    self._json(DecisionLog(app.pdir(q.get("p", [""])[0])).read())
                elif u.path == "/file":
                    self._file(Path(q.get("p", [""])[0]), download=q.get("dl", ["0"])[0] == "1")
                elif u.path == "/favicon.ico":
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self._json({"error": "yok"}, 404)
            except FileNotFoundError:
                self._json({"error": "bulunamadı"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            try:
                if u.path == "/api/upload":
                    self._upload(q)
                    return
                data = self._body()
                if u.path == "/api/project":
                    vids = [app.media_path(x, VIDEO_EXT) for x in (data.get("videos") or [data["video"]])]
                    ek = [app.media_path(x, VIDEO_EXT | IMAGE_EXT) for x in data.get("broll", [])]
                    if not vids:
                        raise ValueError("Konuşma videosu seçilmedi")
                    pr = app.create_project(vids[0], data.get("title"), vids[1:], ek)
                    intro = [str(app.media_path(x, VIDEO_EXT)) for x in data.get("intro", [])]
                    if intro:   # "İkisi de": talk recordings that are also scanned whole for intro shots
                        pj = app.pdir(pr["name"]) / "proje.json"
                        st = read_json(pj, {}) or {}
                        st["intro_files"] = intro
                        write_json(pj, st)
                    if data.get("note") and ek:
                        app.edit_media({"project": pr["name"], "kind": "ek", "action": "note_all", "note": data["note"]})
                    self._json(pr)
                elif u.path == "/api/intro_exclude":
                    # the user's "do not use this shot in the intro" list (empty key = clear the list)
                    pj = app.pdir(data["project"]) / "proje.json"
                    st = read_json(pj, {}) or {}
                    lst = list(st.get("intro_haric") or [])
                    k = str(data.get("key") or "")
                    if not k:
                        lst = []
                    elif k not in lst:
                        lst.append(k)
                    st["intro_haric"] = lst
                    write_json(pj, st)
                    self._json({"intro_haric": lst})
                elif u.path == "/api/fileinfo":
                    self._json({"files": [app.file_info(str(app.media_path(x, VIDEO_EXT | IMAGE_EXT)))
                                          for x in data.get("paths", [])]})
                elif u.path == "/api/relink":
                    picked = app.pick("video")["files"]
                    self._json(app.relink(data["project"], [f["path"] for f in picked]))
                elif u.path == "/api/pick":
                    self._json(app.pick(data.get("kind", "video")))
                elif u.path == "/api/cleanup":
                    self._json(app.cleanup(data["project"]))
                elif u.path == "/api/delete_project":
                    cur = app.jobs.status()["current"]
                    if cur and cur.get("project") == data["project"] and cur.get("state") in ("running", "paused"):
                        raise RuntimeError("Bu projede çalışan bir iş var; önce durdurun")
                    self._json(app.delete_project(data["project"]))
                elif u.path == "/api/open":
                    import os
                    arc = read_json(app.pdir(data["project"]) / "arsiv.json", {}) or {}
                    folder = Path(arc.get("folder", ""))
                    if not folder.exists() or app.cfg.root.resolve() not in folder.resolve().parents:
                        raise FileNotFoundError("Arşiv klasörü yok")
                    os.startfile(str(folder)) if os.name == "nt" else None
                    self._json({"opened": str(folder)})
                elif u.path == "/api/chatgpt":
                    from .. import pipeline
                    proj = pipeline.Project(Config.load(app.cfg.root), data["project"])
                    pk = pipeline.apply_chatgpt_answer(proj, data.get("text", ""))
                    self._json({"package": pk, "archive": app.refresh_archive(data["project"])})
                elif u.path == "/api/override":
                    p = app.pdir(data["project"]) / "proje.json"
                    st = read_json(p, {}) or {}
                    ov = st.get("cut_overrides") or {}
                    if data.get("value") in ("keep", "cut"):
                        ov[data["key"]] = data["value"]
                    else:
                        ov.pop(data["key"], None)
                    st["cut_overrides"] = ov
                    write_json(p, st)
                    self._json({"overrides": ov})
                elif u.path == "/api/media":
                    self._json(app.edit_media(data))
                elif u.path == "/api/job":
                    self._json(app.jobs.control(data["cmd"]))
                elif u.path == "/api/run":
                    self._json(app.jobs.start(data["action"], data.get("project")))
                elif u.path == "/api/feedback":
                    d = app.pdir(data["project"])
                    fb = read_json(d / "feedback.json", []) or []
                    fb.append({"t": time.strftime("%Y-%m-%d %H:%M"), "timecode": data.get("timecode", ""),
                               "text": data.get("text", ""), "state": "yeni"})
                    write_json(d / "feedback.json", fb)
                    self._json(fb)
                else:
                    self._json({"error": "yok"}, 404)
            except Exception as e:  # noqa: BLE001
                try:
                    self._json({"error": str(e) if isinstance(e, (ValueError, RuntimeError)) else
                                f"{type(e).__name__}: {e}"}, 400)
                except OSError:
                    pass   # the browser already went away

        def _upload(self, q):
            name = Path(urllib.parse.unquote(q.get("name", ["video.mp4"])[0])).name
            kind = q.get("kind", [None])[0]
            project = q.get("project", [None])[0]
            if kind:
                if kind not in KINDS:
                    return self._json({"error": "Bilinmeyen tür"}, 400)
                folder, _, exts = KINDS[kind]
                ham = (app.cfg.root / folder[1:]) if folder.startswith("@") else (app.pdir(project) / folder)
                ham.mkdir(parents=True, exist_ok=True)
            else:
                exts = VIDEO_EXT
                ham = app.cfg.root / "ham"
            if Path(name).suffix.lower() not in exts:
                return self._json({"error": "Desteklenmeyen dosya türü"}, 400)
            n = int(self.headers.get("Content-Length") or 0)
            free = shutil.disk_usage(str(ham)).free
            if n <= 0 or n > free - 5e9:
                left = n   # read the body away, otherwise the browser only sees a broken connection
                while left > 0:
                    chunk = self.rfile.read(min(4 << 20, left))
                    if not chunk:
                        break
                    left -= len(chunk)
                return self._json({"error": f"Diskte yer yok ({free / 1e9:.0f} GB boş). Eski projeleri silin ya da ara "
                                            "dosyaları temizleyin."}, 400)
            target = ham / name
            k = 2
            while target.exists():
                target = ham / f"{Path(name).stem}-{k}{Path(name).suffix}"
                k += 1
            part = target.with_suffix(target.suffix + ".yukleniyor")
            left, err = n, None
            try:
                with open(part, "wb") as f:
                    while left > 0:
                        chunk = self.rfile.read(min(4 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
            except OSError as e:   # disk filled up while writing: drop the partial file, read the rest away
                err = f"Dosya yazılamadı ({e.strerror or e}). Diskte yer açıp tekrar deneyin."
                part.unlink(missing_ok=True)
                try:
                    while left > 0:
                        chunk = self.rfile.read(min(4 << 20, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                except OSError:
                    return
            if err or left:
                part.unlink(missing_ok=True)
                try:
                    return self._json({"error": err or "Yükleme yarıda kesildi"}, 400)
                except OSError:
                    return
            part.rename(target)
            if kind == "kapak":
                from .. import pipeline
                proj = pipeline.Project(Config.load(app.cfg.root), project)
                pk = pipeline.add_chatgpt_thumbnail(proj, (q.get("label", [""])[0] or "").upper(), target)
                return self._json({"package": pk, "archive": app.refresh_archive(project)})
            if kind:
                app.add_media(project, kind, target, urllib.parse.unquote(q.get("note", [""])[0]))
                return self._json({"file": str(target), "media": app.media(project)})
            if q.get("noproject", ["0"])[0] == "1":   # the page asks which file is what, then creates the project
                return self._json({"video": str(target)})
            self._json({"video": str(target), "project": app.create_project(target, q.get("title", [None])[0])})

    return H


def _code_stamp(root: Path) -> tuple:
    files = list((root / "kurgu").rglob("*.py")) + [root / "design_config.json"]
    return tuple(sorted((str(f), f.stat().st_mtime) for f in files if f.exists()))


def _spawn_new_server(root: Path) -> None:
    """Start a fresh server that outlives this one. On Windows the venv launcher keeps its children in a job that
    dies with it, so start through Explorer (the autostart shortcut) or break away from the job."""
    import os
    import subprocess
    import sys
    if os.name == "nt":
        lnk = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/Kurgu Masasi Sunucu.lnk"
        if lnk.exists():
            subprocess.Popen(["explorer.exe", str(lnk)], close_fds=True)
            return
        try:
            subprocess.Popen([sys.executable, "-m", "kurgu"], cwd=str(root), close_fds=True,
                             creationflags=0x00000008 | 0x00000200 | 0x01000000)  # detached, new group, breakaway
            return
        except OSError:
            pass
    subprocess.Popen([sys.executable, "-m", "kurgu"], cwd=str(root), close_fds=True,
                     creationflags=(0x00000008 | 0x00000200) if os.name == "nt" else 0)


def _auto_reload(httpd, app: App) -> None:
    """New code was copied in: load it without restarting the server (restarting from inside a Windows venv
    launcher proved unreliable). The editing modules are dropped from the import cache when no job runs, so the
    next job imports the new code; the page itself is read from disk on every request anyway."""
    import sys
    import threading

    reloadable = ("kurgu.pipeline", "kurgu.editing", "kurgu.motion", "kurgu.audio", "kurgu.vision",
                  "kurgu.analysis", "kurgu.media", "kurgu.integrations", "kurgu.qa", "kurgu.cache", "kurgu.decisions")

    def loop():
        base = _code_stamp(app.cfg.root)
        while True:
            time.sleep(20)
            try:
                now = _code_stamp(app.cfg.root)
                if now == base:
                    continue
                cur = app.jobs.status()["current"]
                if cur and cur.get("state") in ("running", "paused"):
                    continue
                time.sleep(5)  # let a multi-file copy finish
                for name in [m for m in sys.modules if m.startswith(reloadable)]:
                    del sys.modules[name]
                import kurgu as _pkg   # `from kurgu import pipeline` would otherwise still find the old module
                for attr in ("pipeline", "cache", "decisions"):
                    if attr in vars(_pkg):
                        delattr(_pkg, attr)
                base = _code_stamp(app.cfg.root)
                print("Kod değişti: yeni kod bir sonraki işte kullanılacak", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"otomatik yenileme hatası: {e}", flush=True)
    threading.Thread(target=loop, daemon=True).start()


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8770) -> None:
    app = App(cfg)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    _auto_reload(httpd, app)
    print(f"Kurgu arayüzü: http://localhost:{port}", flush=True)
    httpd.serve_forever()
