"""Background job runner shared by the web UI and the file-drop queue (isler/gelen/*.json).

Only whitelisted actions run. One job at a time; progress and results are persisted so the UI (and anyone reading
the files) can follow along.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from ..config import Config, write_json

ACTIONS = ("install_separator", "install_assets", "analyze", "plan", "apply", "verify", "rough", "video", "shorts", "package", "youtube_text", "archive", "premiere", "all", "tests", "ping", "frames", "premiere_test", "preview", "draft", "replan")


HEAVY = ("analyze", "rough", "video", "shorts", "all", "premiere", "preview", "draft", "replan", "package")
MIN_FREE_GB = 15


class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.current: dict | None = None
        self.history: list[dict] = []
        self.queue_dir = cfg.root / "isler"
        (self.queue_dir / "gelen").mkdir(parents=True, exist_ok=True)
        (self.queue_dir / "sonuc").mkdir(parents=True, exist_ok=True)
        self._stop = False
        self._close_orphans()
        threading.Thread(target=self._watch_queue, daemon=True).start()

    def _close_orphans(self) -> None:
        """Jobs that were running when the app was closed (or crashed) leave a *.calisiyor.json behind: turn them
        into finished "stopped" records so nothing looks like it is still running."""
        for f in (self.queue_dir / "sonuc").glob("*.calisiyor.json"):
            try:
                import json as _json
                job = _json.loads(f.read_text(encoding="utf-8"))
                job.update(state="stopped", message="Uygulama kapandığı için yarıda kaldı; düğmeyle yeniden başlatabilirsin")
                write_json(f.with_name(f.name.replace(".calisiyor.json", ".json")), job)
                f.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    # ---- public --------------------------------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            return {"current": dict(self.current) if self.current else None, "history": self.history[-20:]}

    def start(self, action: str, project: str | None = None, source: str = "web", job_id: str | None = None) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"Bilinmeyen iş: {action}")
        with self.lock:
            if self.current and self.current["state"] in ("running", "paused"):
                raise RuntimeError(f"Şu an çalışan bir iş var: {self.current['action']}")
            job = {"id": job_id or uuid.uuid4().hex[:8], "action": action, "project": project, "source": source,
                   "state": "running", "progress": 0.0, "message": "Başlıyor", "started": time.time(), "log": []}
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    # ---- internals -------------------------------------------------------------------------------
    def control(self, cmd: str) -> dict:
        """pause / resume / stop the running job (checked at every progress step)."""
        with self.lock:
            job = self.current
            if not job or job["state"] not in ("running", "paused"):
                raise RuntimeError("Çalışan bir iş yok")
            job["control"] = {"pause": "pause", "resume": "run", "stop": "stop"}[cmd]
            if cmd == "resume":
                job["state"] = "running"
            return dict(job)

    def _checkpoint(self, job: dict) -> None:
        while job.get("control") == "pause":
            if job["state"] != "paused":
                job["state"] = "paused"
            time.sleep(0.5)
        if job.get("control") == "stop":
            raise JobCancelled("Kullanıcı durdurdu")
        if job["state"] == "paused":
            job["state"] = "running"

    def _progress(self, job: dict, f: float, msg: str) -> None:
        self._checkpoint(job)
        now = time.time()
        if now - job.get("_saved", 0) > 2.0:
            job["_saved"] = now
            try:
                write_json(self.queue_dir / "sonuc" / f"{job['id']}.calisiyor.json",
                           {k: v for k, v in job.items() if not k.startswith("_")})
            except Exception:  # noqa: BLE001
                pass
        with self.lock:
            job["progress"] = round(max(job["progress"], min(1.0, max(0.0, f))), 4)
            if msg != job["message"]:
                job["log"].append({"t": round(time.time() - job["started"], 1), "m": msg})
                job["log"] = job["log"][-200:]
            job["message"] = msg

    def _run(self, job: dict) -> None:
        result = None
        from ..vision.faces import BLOCKED
        BLOCKED["reason"] = None     # Smart App Control blocks are temporary: every job tries again
        try:
            pipeline, Premiere, apply_rough_plan = _import_engine()

            cfg = Config.load(self.cfg.root)  # fresh settings for every job
            proj = pipeline.Project(cfg, job.get("project"))
            pr = Premiere(cfg.path("bridge_dir"))
            prog = lambda f, m: self._progress(job, f, m)  # noqa: E731
            a = job["action"]
            from ..audio import separate as sep
            if job.get("project"):
                from ..editing.director import options_of
                sep.configure(options_of(proj.settings).get("muzik_temizle", True))
            else:
                sep.configure(False)
            if a == "install_separator":
                result = _install_separator(prog)
            if a == "install_assets":
                result = _install_assets(prog)
            if a in HEAVY and job.get("project"):
                st = proj.settings or {}
                srcs = list(st.get("camera_files") or []) + list(st.get("screen_files") or []) + \
                    [b["file"] for b in st.get("broll") or []]
                missing = [Path(f).name for f in srcs if not Path(f).exists()]
                if missing:
                    raise RuntimeError(f"{len(missing)} kaynak dosya bulunamadı (taşınmış ya da silinmiş): "
                                       f"{', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}. "
                                       "Malzemeler ve ayarlar bölümündeki “Yeni yerini göster” ile yerini seçin.")
            if a in HEAVY:
                import shutil
                free = shutil.disk_usage(str(cfg.root)).free / 1e9
                if free < MIN_FREE_GB:
                    raise RuntimeError(f"Diskte yalnızca {free:.0f} GB boş yer var; bu iş için en az {MIN_FREE_GB} GB "
                                       "gerekiyor. Ana sayfadan eski projeleri silin ya da ara dosyalarını temizleyin.")
            if a in HEAVY and a != "package" and job.get("project"):
                from ..editing.director import options_of
                if options_of(proj.settings).get("muzik_temizle", True):
                    if not sep.installed():
                        job["uyari"] = ["Müzik temizleyici kurulu değil: mekândaki müzik konuşmanın altında duyulacak. "
                                        "İş bitince ya da durdurunca buradaki “Müzik temizleyiciyi şimdi kur” düğmesiyle bir kez kurman yeterli."]
                    else:   # also for projects whose analysis was made before the separator was installed
                        cams = [f for f in (proj.settings or {}).get("camera_files") or [] if Path(f).exists()]
                        todo = [f for f in cams if not sep.cached_voice(f)]
                        if todo:
                            try:
                                from ..integrations.llm import free_gpu
                                free_gpu()     # Demucs gets the whole GPU (a loaded language model slowed it down)
                                sep.separate(todo, lambda f, m: prog(0.0, m))
                            except Exception as e:  # noqa: BLE001 - never lose the edit because of it
                                sep.configure(False)
                                job["uyari"] = [f"Müzik ayrılamadı, orijinal ses kullanıldı: {str(e)[:200]}"]
            if a in ("all", "video") and job.get("project"):
                from ..editing.director import options_of
                o = options_of(proj.settings)
                if o.get("montage", True) and o.get("music", True) and not pipeline.music_library(cfg):
                    job.setdefault("uyari", []).append("Müzik klasörün (muzik) boş: müzikli intro yapılamaz. "
                                                      "Telifsiz bir müziği “muzik” klasörüne koyman yeterli.")
            if a in ("video", "preview", "shorts", "package", "premiere") and not proj.load("edit_plan.json"):
                # no cut yet (step 1 was skipped): make it first instead of failing
                self._rough(cfg, proj, pr, pipeline, None, lambda f, m: prog(0.0, f"Önce kesim yapılıyor: {m}"))
            if a == "ping":
                result = pr.ping()
            elif a == "analyze":
                result = _summ(pipeline.analyze(proj, prog))
            elif a == "plan":
                result = pipeline.plan(proj, prog)["stats"]
            elif a == "apply":
                result = apply_rough_plan(pr, proj.load("edit_plan.json"), prog,
                                          on_plan=lambda pl: proj.save("edit_plan.json", pl))
                proj.save("qa_premiere.json", result)
            elif a == "verify":
                result = pipeline.verify_cut(proj, prog)
            elif a == "rough":
                result = self._rough(cfg, proj, pr, pipeline, apply_rough_plan, prog)
            elif a == "shorts":
                result = pipeline.make_shorts(proj, prog)
            elif a == "package":
                result = pipeline.make_package(proj, prog)
            elif a == "youtube_text":
                result = pipeline.make_youtube_text(proj, prog)
            elif a == "all":
                # 1 cut -> 2-4 effects/retouch/render -> 5 thumbnails+titles+description -> 6 shorts -> 7 archive
                result = {}
                steps = [("Kesim", 0.0, 0.25), ("Video", 0.25, 0.65), ("Kapak kareleri", 0.65, 0.75),
                         ("Shorts", 0.75, 0.97), ("Arşiv", 0.97, 1.0)]

                def sub(i):
                    name, a0, a1 = steps[i]
                    return lambda f, m: prog(a0 + (a1 - a0) * f, f"{name}: {m}")
                result["Kesim"] = self._rough(cfg, proj, pr, pipeline, None, sub(0))
                llm, why = pipeline._llm_or_none(proj, sub(1))
                problems = []
                try:
                    result["Video"] = pipeline.make_video(proj, sub(1), llm=llm)

                    def soft(label, fn):  # a side product failing must not lose the finished video
                        try:
                            result[label] = fn()
                        except JobCancelled:
                            raise
                        except Exception as e:  # noqa: BLE001
                            problems.append(f"{label}: {type(e).__name__}: {e}")
                            job["log"].append({"t": round(time.time() - job["started"], 1), "m": f"{label} atlandı: {e}"})
                    soft("Kapak kareleri", lambda: pipeline.make_package(proj, sub(2), llm=llm))
                    soft("Shorts", lambda: pipeline.make_shorts(proj, sub(3), llm=llm))
                    if llm:
                        soft("Başlık ve açıklama", lambda: pipeline.make_youtube_text(proj, sub(3), llm=llm))
                    soft("Arşiv", lambda: pipeline.make_archive(proj, sub(4)))
                finally:
                    if llm:
                        llm.stop()
                if why:
                    result["uyarı"] = why
                if problems:
                    result["sorunlar"] = problems
            elif a == "replan":   # after the user kept/cut items in the review list
                result = {"Plan": pipeline.plan(proj, lambda f, m: prog(0.3 * f, m))["stats"]}
                result["Önizleme"] = pipeline.make_preview(proj, lambda f, m: prog(0.3 + 0.7 * f, m))
            elif a == "preview":
                result = pipeline.make_preview(proj, prog)
            elif a == "draft":
                # step 1 of the reviewed flow: cut plan + check + quick preview (nothing heavy, nothing in Premiere)
                result = {"Kesim": self._rough(cfg, proj, pr, pipeline, None, lambda f, m: prog(0.75 * f, m))}
                result["Önizleme"] = pipeline.make_preview(proj, lambda f, m: prog(0.75 + 0.25 * f, m))
            elif a == "video":
                llm, why = pipeline._llm_or_none(proj, prog)
                try:
                    result = pipeline.make_video(proj, prog, llm=llm)
                finally:
                    if llm:
                        llm.stop()
            elif a == "archive":
                result = pipeline.make_archive(proj, prog)
            elif a == "premiere":
                from ..integrations.premiere_launch import ensure_premiere, quit_premiere
                opened = ensure_premiere(pr, prog)
                try:
                    result = apply_rough_plan(pr, proj.load("edit_plan.json"), prog,
                                              on_plan=lambda pl: proj.save("edit_plan.json", pl))
                finally:
                    if opened:
                        quit_premiere(pr)
            elif a == "premiere_test":
                from ..integrations.premiere_edit import self_test
                result = self_test(cfg, prog)
            elif a == "tests":
                result = _run_tests(self.cfg.root, prog)
            elif a == "frames":
                from ..qa.frames import grab_frames
                result = grab_frames(pr, proj, prog)
            job["state"] = "done"
            job["result"] = result
            from ..vision.faces import BLOCKED
            if BLOCKED["reason"]:
                job.setdefault("uyari", []).append(BLOCKED["reason"] + ". Bir dahaki çalıştırmada genelde geçer; "
                                                   "olmazsa bana yaz.")
                BLOCKED["reason"] = None     # try again on the next job
        except JobCancelled:
            job["state"] = "stopped"
            job["message"] = "Durduruldu"
        except Exception as e:  # noqa: BLE001
            job["state"] = "error"
            msg = str(e)
            if "No such file or directory" in msg:   # a recording was moved/deleted while the job ran
                import re as _re
                f = _re.search(r"([^\\/:]+\.\w{2,4}): No such file", msg)
                msg = (f"Kaynak dosya iş sırasında kayboldu{': ' + f.group(1) if f else ''} (taşınmış ya da silinmiş). "
                       "İş bitene kadar videoları yerinden oynatmayın. Taşıdıysanız “Yeni yerini göster” ile seçin; "
                       "yapılan adımlar önbellekte, tekrar başlatınca hızlı geçer.")
                job["error"] = msg
            else:
                job["error"] = msg if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
            job["trace"] = traceback.format_exc()[-4000:]
        finally:
            job["finished"] = time.time()
            job["progress"] = 1.0 if job["state"] == "done" else job["progress"]
            out = self.queue_dir / "sonuc" / f"{job['id']}.json"
            write_json(out, {k: v for k, v in job.items() if not k.startswith("_")})
            (self.queue_dir / "sonuc" / f"{job['id']}.calisiyor.json").unlink(missing_ok=True)
            with self.lock:
                self.history.append({k: job.get(k) for k in ("id", "action", "state", "error", "started", "finished", "message")})

    def _rough(self, cfg, proj, pr, pipeline, apply_rough_plan, prog) -> dict:
        result = {"Analiz": _summ(pipeline.analyze(proj, lambda f, m: prog(0.3 * f, f"Analiz: {m}")))}
        proj.save("extra_cuts.json", [])
        rounds = cfg.design.get("qa", {}).get("max_rounds", 3)
        for rnd in range(1, rounds + 1):
            base = 0.3 + 0.5 * (rnd - 1) / rounds
            sub = lambda f, m, b=base, r=rnd: prog(b + 0.5 / rounds * f, f"Tur {r}: {m}")  # noqa: E731
            result[f"Plan {rnd}"] = pipeline.plan(proj, lambda f, m: sub(0.2 * f, m))["stats"]
            qa = pipeline.verify_cut(proj, lambda f, m: sub(0.2 + 0.8 * f, m))
            result[f"Doğrulama {rnd}"] = {k: qa[k] for k in ("ok", "match_ratio", "real_loss_words",
                                                            "remaining_repetitions", "extra_fillers")}
            if not qa["extra_fillers"]:
                break
            prev = proj.load("extra_cuts.json", []) or []
            proj.save("extra_cuts.json", prev + qa["extra_fillers"])
        if apply_rough_plan is None:
            return result
        try:
            result["Premiere"] = apply_rough_plan(pr, proj.load("edit_plan.json"),
                                                  lambda f, m: prog(0.8 + 0.2 * f, f"Premiere: {m}"),
                                                  on_plan=lambda pl: proj.save("edit_plan.json", pl))
            proj.save("qa_premiere.json", result["Premiere"])
        except Exception as e:  # noqa: BLE001
            # Premiere closed or panel off: the edit plan is still valid; report instead of failing everything
            result["Premiere"] = {"ok": False, "problems": [f"Premiere'e kurulamadı: {e}"]}
            proj.save("qa_premiere.json", result["Premiere"])
        return result

    def _watch_queue(self) -> None:
        inbox = self.queue_dir / "gelen"
        while not self._stop:
            try:
                for f in sorted(inbox.glob("*.json")):
                    if self.current and self.current["state"] in ("running", "paused"):
                        break
                    try:
                        req = json.loads(f.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    f.unlink(missing_ok=True)
                    try:
                        self.start(req.get("action", ""), req.get("project"), source="dosya", job_id=f.stem)
                    except Exception as e:  # noqa: BLE001
                        write_json(self.queue_dir / "sonuc" / f"{f.stem}.json", {"id": f.stem, "state": "error", "error": str(e)})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)


def _install_assets(prog) -> dict:
    """Free professional title fonts (Google Fonts, SIL Open Font License) into <root>/assets/fonts, with their
    licence texts. Each file is checked to be a real font with every Turkish letter; failures are listed."""
    import requests

    from ..motion.fonts import FONT_DIR, LICENSES, PACKAGE, has_turkish_bytes
    base = "https://raw.githubusercontent.com/google/fonts/main/"
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    (FONT_DIR / "lisanslar").mkdir(exist_ok=True)
    ok, bad = [], []
    items = list(PACKAGE.items())
    for i, (name, rel) in enumerate(items):
        prog(0.05 + 0.8 * i / len(items), f"Yazı tipi indiriliyor: {name.split('[')[0].split('-')[0]}")
        dst = FONT_DIR / name
        try:
            r = requests.get(base + rel, timeout=60)
            r.raise_for_status()
            if r.content[:4] not in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
                raise ValueError("yazı tipi dosyası değil")
            if not has_turkish_bytes(r.content):
                raise ValueError("Türkçe harfleri yok")
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(r.content)
            from ..config import replace_file
            replace_file(tmp, dst)
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: {e}")
    for fam, rel in LICENSES.items():
        try:
            r = requests.get(base + rel, timeout=60)
            if r.ok:
                (FONT_DIR / "lisanslar" / f"{fam}_OFL.txt").write_bytes(r.content)
        except Exception:  # noqa: BLE001
            pass
    from ..motion import fonts as _f
    _f.font.cache_clear()
    _f.has_turkish.cache_clear()
    for leftover in FONT_DIR.glob("*.part"):   # from the failed first attempt
        try:
            leftover.unlink()
        except OSError:
            pass
    if not ok:
        raise RuntimeError("Yazı tipleri indirilemedi: " + " | ".join(bad[:3]))
    prog(1.0, f"{len(ok)} yazı tipi kuruldu")
    return {"kuruldu": ok, "sorunlar": bad}


def _install_separator(prog) -> dict:
    """User clicked "Müzik temizleyiciyi kur": PyTorch (CUDA build if there is an NVIDIA GPU) + Demucs into this
    .venv, then the model weights (one time, ~80 MB). Output lines go to the job log."""
    import shutil
    exe = Path(sys.executable)
    py = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
    nowin = 0x08000000 if os.name == "nt" else 0

    def run(args: list[str], f0: float, f1: float, label: str) -> int:
        prog(f0, label)
        p = subprocess.Popen([str(py), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             text=True, encoding="utf-8", errors="replace", creationflags=nowin)
        n = 0
        for line in p.stdout:
            line = line.strip()
            if line:
                n += 1
                prog(min(f1, f0 + (f1 - f0) * (1 - 1 / (1 + n / 15))), f"{label}: {line[:110]}")
        return p.wait()

    gpu = bool(shutil.which("nvidia-smi")) or (os.name == "nt" and Path(os.environ.get("SystemRoot", r"C:\Windows"),
                                                                             "System32", "nvidia-smi.exe").exists())
    index = "https://download.pytorch.org/whl/cu128" if gpu else "https://download.pytorch.org/whl/cpu"
    pip = ["-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--prefer-binary"]
    if run(pip + ["torch", "torchaudio", "--index-url", index], 0.02, 0.7,
           "PyTorch indiriliyor (~2,5 GB)" if gpu else "PyTorch indiriliyor") != 0:
        raise RuntimeError("PyTorch kurulamadı (internet bağlantısını kontrol edip tekrar deneyin)")
    if run(pip + ["demucs"], 0.7, 0.85, "Demucs kuruluyor") != 0:
        # lameenc (mp3 export, not needed here) sometimes has no wheel: install the rest without it
        if run(pip + ["--no-deps", "demucs"], 0.85, 0.87, "Demucs (bağımlılıksız)") != 0 or \
                run(pip + ["dora-search", "einops", "julius", "openunmix", "pyyaml", "tqdm"], 0.87, 0.9,
                    "Demucs bağımlılıkları") != 0:
            raise RuntimeError("Demucs kurulamadı")
    code = ("import torch, demucs.pretrained as p; m = p.get_model('htdemucs'); "
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    if run(["-c", code], 0.9, 0.99, "Model indiriliyor ve deneniyor") != 0:
        raise RuntimeError("Demucs modeli indirilemedi ya da çalışmadı")
    import importlib
    importlib.invalidate_caches()
    return {"kuruldu": True, "gpu": gpu}


def _import_engine():
    """Import the editing engine. Windows Smart App Control sometimes blocks a DLL for a moment
    ("DLL load failed ... Uygulama Denetimi"); the half-imported modules are dropped and the import is retried."""
    last = None
    for attempt in range(4):
        try:
            from .. import pipeline
            from ..integrations.premiere import Premiere
            from ..integrations.premiere_apply import apply_rough_plan
            return pipeline, Premiere, apply_rough_plan
        except ImportError as e:
            if "DLL" not in str(e):
                raise
            last = e
            for m in [m for m in sys.modules if m.startswith(("scipy", "numpy.random", "kurgu.pipeline", "kurgu.media",
                                                              "kurgu.analysis", "kurgu.editing", "kurgu.audio"))]:
                sys.modules.pop(m, None)
            time.sleep(2 + 2 * attempt)
    raise RuntimeError("Windows Akıllı Uygulama Denetimi bir kütüphane dosyasını (DLL) engelledi; 4 deneme de "
                       f"başarısız oldu. İşi yeniden başlatın; tekrar ederse bilgisayarı yeniden başlatın. ({last})")


def _summ(analysis: dict) -> dict:
    return {"camera_s": analysis["camera"]["duration"], "parts": len(analysis["camera"]["parts"]),
            "screens": len(analysis["screens"]), "sync": {Path(k).name: (v.get("offset_s"), v.get("note")) for k, v in analysis["sync"].items()},
            "transcript_device": analysis.get("transcript_device"), "seconds": analysis.get("seconds")}


def _run_tests(root: Path, prog) -> dict:
    prog(0.1, "Testler çalışıyor")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KURGU_GPU_TEST="1")
    # the server may run under pythonw.exe (no console); child tools (PowerShell/SAPI) hang there -> use python.exe
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    try:
        import pytest  # noqa: F401
        cmd = [exe, "-X", "faulthandler", "-m", "pytest", "-v", "-s", "-p", "no:cacheprovider",
               "-o", "faulthandler_timeout=120", "tests"]
    except ImportError:
        cmd = [exe, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    p = subprocess.run(cmd, cwd=root, capture_output=True, env=env, stdin=subprocess.DEVNULL,
                       creationflags=0x08000000 if os.name == "nt" else 0, timeout=1800)
    out = (p.stdout + p.stderr).decode("utf-8", "replace")
    prog(1.0, "Testler bitti")
    return {"returncode": p.returncode, "ok": p.returncode == 0, "output": out[-8000:], "runner": " ".join(cmd[1:4])}
