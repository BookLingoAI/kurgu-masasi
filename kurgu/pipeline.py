"""Pipeline: analyze -> plan -> apply -> verify. Each step reads/writes files in projeler/<name>/."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis.audio import active_regions, energy_profile, peak_dbfs
from .analysis.screen import analyze_frames
from .analysis.transcript import WhisperEngine, transcribe_audio
from .cache import Cache
from .config import Config, read_json, write_json
from .decisions import DecisionLog
from .editing.cuts import compute_cuts, find_repetitions
from .editing.plan import build_rough_plan, cut_words
from .integrations import ffmpeg as ff
from .media.sources import CameraTimeline
from .media.sync import sync_camera_to_screen

Progress = Callable[[float, str], None]


def _noop(f, m):
    pass


class Project:
    def __init__(self, cfg: Config, name: str | None = None):
        self.cfg = cfg
        self.name = name or cfg.user.get("project_name") or "proje"
        if not re.fullmatch(r"[\w\-. ]{1,80}", self.name) or ".." in self.name:
            raise ValueError(f"Geçersiz proje adı: {self.name}")
        self.dir = cfg.project_dir(self.name)
        self.cache = Cache(self.dir / "cache")
        self.log = DecisionLog(self.dir)

    @property
    def settings(self) -> dict:
        """projeler/<ad>/proje.json (camera/screen files); falls back to user_settings for old projects."""
        st = read_json(self.dir / "proje.json", None)
        if st:
            return st
        return {"camera_files": self.cfg.user.get("camera_files") or [], "screen_files": self.cfg.user.get("screen_files") or []}

    def save_settings(self, data: dict) -> dict:
        st = dict(self.settings, **data)
        write_json(self.dir / "proje.json", st)
        return st

    def p(self, fn: str) -> Path:
        return self.dir / fn

    def load(self, fn: str, default=None):
        return read_json(self.p(fn), default)

    def save(self, fn: str, data) -> None:
        write_json(self.p(fn), data)


SR = 16000


def _audio(proj: Project, path: str, voice: bool = True) -> np.ndarray:
    """16 kHz mono of a recording - the music-free voice when separation is on (see audio/separate.py)."""
    from .audio.separate import voice_file
    src = voice_file(path) if voice else path
    return proj.cache.array("audio16k", [src], {"sr": SR}, lambda: ff.load_audio(src, SR))


def transcript_settings(cfg: Config) -> tuple[dict, dict]:
    """(spelling corrections, Whisper settings incl. the channel vocabulary from dictionary.json)."""
    d = read_json(cfg.path("dictionary"), {}) or {}
    tcfg = dict(cfg["transcript"])
    tcfg["vocabulary"] = list(dict.fromkeys([*(tcfg.get("vocabulary") or []), *(d.get("vocabulary") or [])]))
    return d.get("replacements", {}), tcfg


def analyze(proj: Project, progress: Progress = _noop, engine=None) -> dict:
    cfg = proj.cfg
    cams = proj.settings.get("camera_files") or []
    scrs = proj.settings.get("screen_files") or []
    if not cams:
        raise ValueError("Kamera dosyası seçilmedi")
    for f in cams + scrs:
        if not Path(f).exists():
            raise FileNotFoundError(f)
    from .integrations.llm import free_gpu
    free_gpu()   # Whisper (and Demucs) get the whole GPU; the language model is loaded again when needed
    t0 = time.time()
    progress(0.01, "Dosyalar inceleniyor")
    cam = CameraTimeline.from_files(cams)
    screens = [ff.probe(s).to_dict() for s in scrs]
    proj.log.section("Analiz")
    proj.log.write("media", "camera", f"{len(cam.parts)} kamera parçası, {cam.duration:.1f} sn, "
                   f"{cam.size[0]}x{cam.size[1]} @ {cam.fps:.3f}", parts=[p.path for p in cam.parts])

    from .audio import separate as sep
    if sep.enabled():
        progress(0.02, "Müzik ve arka plan sesi ayrılıyor")
        try:
            r = sep.separate([p.path for p in cam.parts], lambda f, m: progress(0.02 + 0.03 * f, m))
            proj.log.write("audio", "separate", f"Ses müzikten ayrıldı ({r['device'] or 'önbellek'})", **r)
        except Exception as e:  # noqa: BLE001 - never lose the edit because of the optional separator
            sep.configure(False)
            proj.log.write("audio", "separate", f"Müzik ayrılamadı, orijinal ses kullanılıyor: {e}")
    progress(0.05, "Kamera sesi okunuyor")
    cam_audio_parts = {p.path: _audio(proj, p.path) for p in cam.parts}
    virtual = np.concatenate([cam_audio_parts[p.path][: int(round(p.duration * SR))] for p in cam.parts]) \
        if cam.parts else np.zeros(1, np.float32)

    syncs: dict[str, dict] = {}
    if screens:
        progress(0.12, "Ekran kaydı sesi okunuyor")
        scr_audio = {s["path"]: _audio(proj, s["path"], voice=False) for s in screens}
        cam_raw = {p.path: _audio(proj, p.path, voice=False) for p in cam.parts}   # sync needs the room sound too
        for i, p in enumerate(cam.parts):
            progress(0.15 + 0.1 * i / len(cam.parts), f"Senkron: {Path(p.path).name}")
            best = None
            for s in screens:
                r = proj.cache.json("sync", [p.path, s["path"]], cfg["sync"],
                                    lambda: sync_camera_to_screen(cam_raw[p.path], scr_audio[s["path"]], SR,
                                                                  cfg["sync"], p.path, s["path"]).to_dict())
                if r.get("screen") and (best is None or r["confidence"] > best["confidence"]):
                    best = r
            if best is None:
                best = {"camera": p.path, "screen": None, "offset_s": None, "note": "Hiçbir ekran kaydında bulunamadı"}
            syncs[p.path] = best
            proj.log.write("sync", "camera_part", best.get("note", ""), camera=p.path, screen=best.get("screen"),
                           offset=best.get("offset_s"), drift_s_per_h=(best.get("drift") or 0) * 3600,
                           confidence=best.get("confidence"), correlation=best.get("correlation"))

    progress(0.27, "Ses enerjisi ölçülüyor")
    prof = energy_profile(virtual, SR, cfg["audio_analysis"])
    active = active_regions(prof, cfg["audio_analysis"])

    progress(0.30, "Transkript (Whisper)")
    dictionary, tcfg = transcript_settings(cfg)
    engine = engine or WhisperEngine(tcfg)

    def tr():
        return transcribe_audio(virtual, SR, tcfg, cfg["audio_analysis"], dictionary, engine,
                                lambda f, m: progress(0.30 + 0.5 * f, m))
    transcript = proj.cache.json("transcript", [sep.voice_file(p.path) for p in cam.parts],
                                 {"t": tcfg, "a": cfg["audio_analysis"], "d": dictionary}, tr)

    # screen activity -> 'video playing' spans in camera time (protects them from long-wait cuts)
    protected = []
    screen_act = {}
    for i, s in enumerate(screens):
        progress(0.82 + 0.12 * i / max(1, len(screens)), f"Ekran etkinliği: {Path(s['path']).name}")
        sa = cfg["screen_activity"]
        act = proj.cache.json("screen_activity", [s["path"]], sa, lambda: analyze_frames(
            ff.gray_frames(s["path"], sa["analysis_fps"], sa["analysis_width"]), sa))
        screen_act[s["path"]] = {"video_playing": act["video_playing"], "n": len(act["samples"])}
        for p in cam.parts:
            sy = syncs.get(p.path)
            if not sy or sy.get("screen") != s["path"]:
                continue
            for a, b in act["video_playing"]:
                ca = (a - sy["offset_s"]) / (1 + sy.get("drift", 0)) + p.start
                cb = (b - sy["offset_s"]) / (1 + sy.get("drift", 0)) + p.start
                if cb > p.start and ca < p.end:
                    protected.append((max(ca, p.start), min(cb, p.end)))
    analysis = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": cam.to_dict(), "screens": screens, "sync": syncs,
        "energy": {"floor_db": prof.floor_db, "speech_db": prof.speech_db, "threshold_db": prof.threshold_db,
                   "peak_dbfs": peak_dbfs(virtual)},
        "active": active, "protected": protected, "screen_activity": screen_act,
        "transcript_device": transcript.get("device"),
        "seconds": round(time.time() - t0, 1),
    }
    proj.save("analysis.json", analysis)
    proj.save("transcript.json", transcript)
    progress(1.0, "Analiz tamam")
    return analysis


def plan(proj: Project, progress: Progress = _noop, use_semantic: bool = True) -> dict:
    a = proj.load("analysis.json")
    t = proj.load("transcript.json")
    if not a or not t:
        raise RuntimeError("Önce analiz çalıştırılmalı")
    cam = CameraTimeline.from_dict(a["camera"])
    progress(0.1, "Anlam modeli yükleniyor")
    from .analysis.semantic import paraphrase_scorer
    sim, how = paraphrase_scorer() if use_semantic else (None, "kapalı")
    if sim is None:
        proj.log.write("rough_cut", "warning", f"Başka sözlerle tekrar kuralı çalışmadı ({how})")
    progress(0.2, "Kesim kuralları uygulanıyor")
    from .editing.cuts import Removal
    extra = []
    for f in (proj.load("extra_cuts.json", []) or []):
        extra.append(Removal(f["start"], f["end"], "filler", f"Dolgu sesi (doğrulama turunda duyuldu): '{f['word']}'",
                             [f["word"]]))
    # wordless moments that SHOW the food (meat held up to the lens, a pan over the sushi): kept, not cut as pauses
    show = []
    try:
        from .editing.showing import find_showing
        show = proj.cache.json("showing", [p.path for p in cam.parts], {"v": 2, "n": len(t["words"])},
                               lambda: [list(x) for x in find_showing(
                                   t["words"], cam, progress=lambda f, m: progress(0.2 + 0.3 * f, m))])
    except Exception as e:  # noqa: BLE001 - a vision problem must never stop the cut
        proj.log.write("rough_cut", "warning", f"Gösterilen anlar bulunamadı: {e}")
    res = compute_cuts(t["words"], [tuple(x) for x in a["active"]], cam.duration, proj.cfg["cuts"],
                       protected=[tuple(x) for x in a.get("protected", [])] + [tuple(x) for x in show],
                       paraphrase_sim=sim, extra=extra, overrides=proj.settings.get("cut_overrides") or {})
    for sa, sb in show:
        proj.log.write("rough_cut", "showing", f"Gösterme anı korundu ({sb - sa:.1f} sn, konuşmasız)", start=sa, end=sb)
    progress(0.6, "Plan yazılıyor")
    pl = build_rough_plan(proj.name, cam, res["kept"], res["removals"], a.get("sync", {}), a.get("screens", []), proj.log)
    # one note per spot: overlapping notes (same sentence flagged by two rules) are merged
    merged = []
    for n in sorted(res.get("notes", []), key=lambda n: n["start"]):
        if merged and n["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], n["end"])
            continue
        merged.append(dict(n))
    res["notes"] = merged
    for n in res.get("notes", []):
        proj.log.write("rough_cut", "kept", n["reason"], rule=n["rule"], start=round(n["start"], 3), end=round(n["end"], 3))
    pl["notes"] = res.get("notes", [])
    pl["showing"] = show
    # every decision for the review list, with the source file/time to listen to it
    review = []
    for r in res.get("review", []):
        part, local = cam.locate(r["start"])
        review.append(dict(r, file=part.path, file_t=round(local, 3)))
    pl["review"] = review
    pl["transcript_cut"] = cut_words([w for i, w in enumerate(t["words"]) if i not in set(res["removed_word_idx"])],
                                     pl["segments"])
    proj.save("edit_plan.json", pl)
    progress(1.0, "Plan hazır")
    return pl


def render_cut_audio(proj: Project, pl: dict) -> np.ndarray:
    cam = CameraTimeline.from_dict(pl["camera"])
    parts = {p.path: _audio(proj, p.path) for p in cam.parts}
    chunks = [parts[s["file"]][int(round(s["in"] * SR)):int(round(s["out"] * SR))] for s in pl["segments"]]
    return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)


def _to_source(t: float, segments: list[dict]) -> float | None:
    for sg in segments:
        if sg["timeline_start"] <= t < sg["timeline_end"]:
            return sg["cam_start"] + (t - sg["timeline_start"])
    return None


def verify_cut(proj: Project, progress: Progress = _noop, engine=None) -> dict:
    """Section 6: re-transcribe the CUT audio; there must be no repetition left and no word lost.

    Spelling variants between two Whisper passes ("5 30da" / "beş buçukta") are not losses; repetitions that the
    plan deliberately kept (notes) are listed separately; fillers that only the second pass hears are mapped back
    to source time so the next round can cut them."""
    import difflib

    from .analysis.text import norm_word
    from .editing.cuts import is_filler

    pl = proj.load("edit_plan.json")
    from .integrations.llm import free_gpu
    free_gpu()
    audio = render_cut_audio(proj, pl)
    ff.write_wav(proj.p("cut_audio.wav"), audio, SR)
    cfg = proj.cfg
    qcfg = cfg.design.get("qa", {"variant_similarity": 0.6, "max_real_loss_fraction": 0.01})
    dictionary, tcfg = transcript_settings(cfg)
    # the check only looks for leftover fillers: the fast model is enough (the main transcript uses the big one)
    tcfg = dict(tcfg, model=tcfg.get("verify_model") or tcfg["model"])
    engine = engine or WhisperEngine(tcfg)
    progress(0.1, "Kesilmiş ses yeniden yazıya dökülüyor")
    re_t = transcribe_audio(audio, SR, tcfg, cfg["audio_analysis"], dictionary, engine,
                            lambda f, m: progress(0.1 + 0.8 * f, m))
    exp_w = pl["transcript_cut"]
    expected = [norm_word(w["word"]) for w in exp_w]
    got = [norm_word(w["word"]) for w in re_t["words"]]
    sm = difflib.SequenceMatcher(a=expected, b=got, autojunk=False)
    lost, variants, added = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        a_txt, b_txt = " ".join(expected[i1:i2]), " ".join(got[j1:j2])
        t = exp_w[i1]["t_start"] if i1 < len(exp_w) else (re_t["words"][j1]["start"] if j1 < len(got) else 0)
        if op == "replace" and difflib.SequenceMatcher(a=a_txt, b=b_txt).ratio() >= qcfg["variant_similarity"]:
            variants.append({"t": t, "a": a_txt, "b": b_txt})
            continue
        if op in ("delete", "replace"):
            lost.append({"words": a_txt, "t": t, "heard": b_txt,
                         "source_t": exp_w[i1]["start"] if i1 < len(exp_w) else None})
        if op == "insert":
            added.append({"words": b_txt, "t": t})
    # repetitions still audible; split into deliberate (plan notes) and unexpected
    rem, _ = find_repetitions(re_t["words"], cfg["cuts"], set())
    note_src = [(n["start"], n["end"]) for n in pl.get("notes", [])]
    reps, known = [], []
    for r in rem:
        src = _to_source(r.start, pl["segments"])
        item = {"t": round(r.start, 2), "source_t": src, "text": " ".join(r.words), "reason": r.reason}
        if src is not None and any(a - 2.0 <= src <= b + 2.0 for a, b in note_src):
            known.append(item)
        else:
            reps.append(item)
    # fillers only the second pass heard -> source time ranges for the next round
    extra = []
    for w in re_t["words"]:
        if is_filler(w["word"], set(cfg["cuts"]["fillers"])):
            a = _to_source(w["start"], pl["segments"])
            b = _to_source(max(w["start"], w["end"] - 1e-3), pl["segments"])
            if a is not None and b is not None and b > a:
                extra.append({"start": round(a, 3), "end": round(b, 3), "word": w["word"]})
    # a "lost" word can only be caused by cutting if it sits at a cut edge AND there is sound on the removed side
    an = proj.load("analysis.json") or {}
    active = [tuple(x) for x in an.get("active", [])]

    def sound_in(a: float, b: float) -> bool:
        return any(c < b and d > a for c, d in active)

    clipped, asr_only = [], []
    for l in lost:
        st = l.get("source_t")
        seg = next((sg for sg in pl["segments"] if st is not None and sg["cam_start"] - 0.05 <= st <= sg["cam_end"] + 0.05), None)
        if seg is None:
            clipped.append(l)
            continue
        w_end = st + 0.6
        near_in = st - seg["cam_start"] < 0.15
        near_out = seg["cam_end"] - st < 0.6
        risk = (near_in and sound_in(seg["cam_start"] - 0.12, seg["cam_start"])) or \
               (near_out and sound_in(seg["cam_end"], min(w_end, seg["cam_end"] + 0.12)))
        (clipped if risk else asr_only).append(l)
    real_loss_words = sum(len(l["words"].split()) for l in clipped)
    report = {
        "match_ratio": round(sm.ratio(), 4),
        "lost": clipped[:200], "asr_omissions": asr_only[:300], "variants": variants[:300], "added": added[:200],
        "remaining_repetitions": reps, "kept_repetitions": known, "extra_fillers": extra,
        "ok": not reps and not extra and real_loss_words <= max(2, int(qcfg["max_real_loss_fraction"] * len(expected))),
        "expected_words": len(expected), "got_words": len(got), "real_loss_words": real_loss_words,
    }
    proj.save("qa_cut.json", report)
    proj.log.write("qa", "retranscribe", f"Eşleşme {report['match_ratio']:.3f}, yazım farkı {len(variants)}, "
                   f"kırpılmış olabilecek {real_loss_words} kelime, yalnızca transkript farkı {len(asr_only)}, beklenmeyen tekrar {len(reps)}, yeni dolgu {len(extra)}")
    progress(1.0, "Kesim doğrulaması tamam")
    return report


# ------------------------------------------------------------------ steps 2-7 (no Premiere needed) -------------
def _llm_or_none(proj: Project, progress: Progress):
    from .integrations.llm import LLMUnavailable, LocalLLM

    llm = LocalLLM(dict(proj.cfg.design.get("llm", {})), progress)
    try:
        llm.start()
        progress(0.0, "Dil modeli hazırlanıyor")
        secs = llm.warm(600)
        proj.log.write("llm", "ready", f"Dil modeli hazır: {llm.key} ({secs:.0f} sn)")
        return llm, None
    except LLMUnavailable as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001
        return None, f"Dil modeline ulaşılamadı: {e}"


def _out_fps(pl: dict) -> float:
    fps = pl["sequence"].get("source_fps") or pl["sequence"]["fps"]
    return min((23.976, 24, 25, 29.97, 30, 50, 59.94, 60), key=lambda r: abs(r - fps))


def project_grade(proj: Project) -> dict | None:
    """Colour correction per recording, the way an editor matches shots in Lumetri: indoor and outdoor clips, a
    window and a dim corner each get their own gentle correction (measured on up to 4 frames of the kept parts of
    that recording), plus a project-wide default. Saved to renk.json."""
    from .motion.thumbnails import grab
    from .vision.grade import GRADE_VERSION, compute_grade
    if not proj.cfg.design.get("grade", {}).get("enabled", True):
        return None
    g = proj.load("renk.json")
    if g and g.get("version") == GRADE_VERSION:   # older grades darkened shadows / ignored lighting changes
        return g
    pl = proj.load("edit_plan.json") or {}
    segs = [sg for sg in pl.get("segments") or [] if sg.get("kind") != "montage"]
    if not segs:
        return None
    strength = proj.cfg.design.get("grade", {}).get("strength", 1.0)
    by_file: dict[str, list] = {}
    for sg in segs:
        by_file.setdefault(sg["file"], []).append(sg)
    files, all_frames = {}, []
    for f, ss in by_file.items():
        picks = [ss[int(i)] for i in np.linspace(0, len(ss) - 1, min(4, len(ss)))]
        try:
            frames = [grab(f, (x["in"] + x["out"]) / 2) for x in picks]
        except Exception:  # noqa: BLE001 - a broken file keeps the default grade
            continue
        files[f] = compute_grade(frames, strength)
        all_frames += frames[:2]
    if not all_frames:
        return None
    g = {"version": GRADE_VERSION, "default": compute_grade(all_frames, strength), "files": files}
    proj.save("renk.json", g)
    return g


OUT_HEIGHT = {"1080": 1080, "1440": 1440}


def retouch_params(proj: Project) -> None:
    """Skin retouch was removed (the user retouches in CapCut): kept so old callers get "nothing to bake"."""
    return None


def _retoucher(proj: Project):
    """Per-frame colour correction (per recording), or (None, reason) when there is nothing to apply."""
    from .vision.grade import apply_grade
    grade = project_grade(proj)
    if not grade:
        return None, "Renk düzeltme gerekmedi"

    def only_grade(rgb, piece, gi, t_out):
        return apply_grade(rgb, grade, getattr(piece, "file", None))
    only_grade.state = {"faces": 0}
    # the renderer runs prepare in order and apply on a thread pool (OpenCV frees the GIL): colour correction was a
    # third of the render time on one core (23.10.2026)
    only_grade.prepare = lambda rgb, piece, gi, t_out: getattr(piece, "file", None)
    only_grade.apply = lambda rgb, file: apply_grade(rgb, grade, file)
    return only_grade, "Renk düzeltme (cilt rötuşu CapCut'ta)"


def _segment_faces(segments: list[dict], keep_small: dict | None = None) -> list[tuple | None]:
    """Face box (pixels, source frame) at the middle of every segment; None if not found / no model.
    keep_small: filled with a 320 px copy of each grabbed frame (the zoom-target step reuses it: no second seek)."""
    from .motion.thumbnails import grab
    from .vision.faces import FaceDetector, backend
    if backend() == "yok":
        return [None] * len(segments)
    det = FaceDetector()
    out = []
    import cv2
    for k, s in enumerate(segments):
        try:
            rgb = grab(s["file"], (s["in"] + s["out"]) / 2)
        except RuntimeError:     # one unreadable frame (e.g. a sliver at a file's end) must not stop the video
            out.append(None)
            continue
        if keep_small is not None and rgb is not None:
            hh, ww = rgb.shape[:2]
            keep_small[k] = cv2.resize(rgb, (320, max(2, int(round(hh * 320 / ww)))), interpolation=cv2.INTER_AREA)
        f = det.detect(rgb)
        if f:
            h, w = rgb.shape[:2]
            g = f[0]
            out.append((g["x"] * w, g["y"] * h, (g["x"] + g["w"]) * w, (g["y"] + g["h"]) * h, w, h))
        else:
            out.append(None)
    return out


def _segment_focus(segments: list[dict], faces: list, words: list[dict], zooms: list[float],
                   small: dict | None = None) -> tuple[list, dict]:
    """Per segment: (cx, cy, zoom) when the zoom must NOT go to the face - the speaker is showing something
    ("bakın", "işte", "şu tabak"...) or no face is in the picture - else None (face-centred as before).
    Showing something -> zoom onto the most salient object (food, product, hand) or no zoom at all."""
    from .motion.thumbnails import grab
    from .vision.focus import SHOW_WORDS, object_focus, segment_words
    out, stats = [], {"nesne": 0, "zoom_yok": 0, "yuz": 0}
    for k, (s, fb, z) in enumerate(zip(segments, faces, zooms)):
        if s.get("kind") == "montage" or s.get("screen"):
            out.append(None)
            continue
        show = bool(set(segment_words(words, s)) & SHOW_WORDS)
        if fb is not None and not show:
            out.append(None)
            stats["yuz"] += 1
            continue
        dur = s["timeline_end"] - s["timeline_start"]
        try:
            frame = (small or {}).get(k)
            r = object_focus(frame if frame is not None else grab(s["file"], (s["in"] + s["out"]) / 2), fb)
        except Exception:  # noqa: BLE001
            r = None
        if r and r[2] >= 3.5:
            out.append((r[0], r[1], max(z, 1.22) if (show and dur >= 1.0) else z))
            stats["nesne"] += 1
        else:   # nothing clear to look at: keep the full picture instead of zooming into the face or a wall
            cx = ((fb[0] + fb[2]) / 2 / fb[4]) if fb else 0.5
            cy = ((fb[1] + fb[3]) / 2 / fb[5]) if fb else 0.45
            out.append((cx, cy, 1.0))
            stats["zoom_yok"] += 1
    return out, stats


AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}


def _audio_len(path: str) -> float:
    try:
        return ff.probe(path).duration or 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def music_library(cfg: Config) -> list[str]:
    """Music folder + long tracks someone put among the sound effects (a song is not a whoosh)."""
    out = []
    for name in ("muzik", "ses_efektleri"):
        d = cfg.root / name
        if d.exists():
            out += [str(f) for f in sorted(d.iterdir()) if f.suffix.lower() in AUDIO_EXT
                    and (name == "muzik" or _audio_len(str(f)) > 8.0)]
    return out


def music_for(proj: "Project") -> str | None:
    """One track from the user's music library per video (stable choice per project)."""
    import zlib
    lib = music_library(proj.cfg)
    return lib[zlib.crc32(proj.name.encode("utf-8")) % len(lib)] if lib else None


def sfx_library(cfg: Config) -> list[str]:
    d = cfg.root / "ses_efektleri"
    return [str(f) for f in sorted(d.iterdir()) if f.suffix.lower() in AUDIO_EXT and 0 < _audio_len(str(f)) <= 8.0] \
        if d.exists() else []


def _gaze(proj: Project, segs: list[dict], progress: Progress) -> list[dict]:
    """Head-turn samples (2 per second) on the edited timeline; only for segments with a synced screen recording.
    Yaw from the face detector's eye/nose key points (no landmark model needed)."""
    from .vision.faces import FaceDetector, backend, yaw_from_face
    want = [s for s in segs if s.get("screen")]
    if not want or backend() == "yok":
        return []
    from .motion.render import _decode

    def compute():
        det = FaceDetector()
        out = []
        for k, s in enumerate(want):
            if k % 20 == 0:
                progress(k / len(want), f"Bakış yönü {k}/{len(want)}")
            info = ff.probe(s["file"])
            sw = 480
            sh = int(round(info.height * sw / info.width / 2) * 2)
            for j, rgb in enumerate(_decode(s["file"], s["in"], s["out"] - s["in"], 2, sw, sh)):
                f = det.detect(rgb)
                out.append({"t": round(s["timeline_start"] + j * 0.5, 3), "yaw": yaw_from_face(f[0]) if f else None})
        return out
    key = [(s["file"], round(s["in"], 3), round(s["out"], 3), round(s["timeline_start"], 3)) for s in want]
    return proj.cache.json("gaze", sorted({s["file"] for s in want}), {"v": 2, "segs": key}, compute)


def _screen_activity(proj: Project, files: set[str]) -> dict:
    sa = proj.cfg.design["screen_activity"]
    return {f: proj.cache.json("screen_activity", [f], sa, lambda f=f: analyze_frames(
        ff.gray_frames(f, sa["analysis_fps"], sa["analysis_width"]), sa)) for f in files if Path(f).exists()}


def _shot_time(path: str) -> str:
    """When the clip/photo was shot (container metadata, else file time) - used to keep shooting order."""
    import datetime as _dt
    if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        try:
            ct = ff.probe(path).creation_time
            if ct:
                return ct
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            from PIL import Image
            with Image.open(path) as im:
                dt = (im.getexif() or {}).get(306)  # DateTime
            if dt:
                return str(dt).replace(":", "-", 2).replace(" ", "T")
        except Exception:  # noqa: BLE001
            pass
    return _dt.datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat()


def describe_broll(proj: Project, items: list[dict], llm, progress: Progress = _noop) -> tuple[list[dict], list[str]]:
    """Sort extra footage by shooting time and let a vision model say what each clip shows (cached in proje.json)."""
    from io import BytesIO

    from PIL import Image

    from .motion.thumbnails import grab
    notes = []
    st = proj.settings
    stored = {b["file"]: b for b in st.get("broll", [])}
    for it in items:
        if not it.get("shot"):
            it["shot"] = stored[it["file"]]["shot"] = _shot_time(it["file"])
    items = sorted(items, key=lambda b: (b["shot"], b["file"]))
    todo = [b for b in items if not b.get("auto")]
    if todo and llm is not None:
        for k, it in enumerate(todo):
            progress(k / len(todo), f"Ek görüntüler izleniyor {k + 1}/{len(todo)}")
            try:
                if Path(it["file"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    frames = [np.asarray(Image.open(it["file"]).convert("RGB"))]
                else:
                    dur = ff.probe(it["file"]).duration or 1.0
                    frames = [grab(it["file"], dur * f) for f in (0.15, 0.5, 0.85)]
                jpgs = []
                for fr in frames:
                    im = Image.fromarray(fr)
                    im.thumbnail((512, 512))
                    buf = BytesIO()
                    im.save(buf, "JPEG", quality=85)
                    jpgs.append(buf.getvalue())
                text = llm.describe_images(jpgs, "Bu kareler bir vlog için çekilmiş ek görüntüden. Ne gösteriyor? "
                                                 "Mekânı, nesneleri ve ortamı Türkçe, en fazla 15 kelimeyle yaz.")
                it["auto"] = stored[it["file"]]["auto"] = text.strip().strip('"')[:200]
            except Exception as e:  # noqa: BLE001
                notes.append("Ek görüntülerin içeriği otomatik görülemedi (LM Studio'daki model görsel desteklemiyor "
                             f"olabilir: {str(e)[:80]}). Dosya adı ve yazdığınız açıklama kullanıldı.")
                break
    elif todo:
        notes.append("Dil modeli yoktu: ek görüntülerin içeriği otomatik görülemedi")
    proj.save_settings({"broll": list(stored.values())})
    return items, notes


def direct(proj: Project, pl: dict, progress: Progress = _noop, llm=None) -> dict:
    """Decide screen switches, b-roll, cards, subscribe and sound effects (saved to yonetmen.json)."""
    from .analysis.semantic import paraphrase_scorer
    from .editing import director as dr
    from .motion.compose import media_duration

    opts = dr.options_of(proj.settings)
    segs, words = pl["segments"], pl["transcript_cut"]
    screens = {s["screen"]["file"] for s in segs if s.get("screen")}
    notes = []
    activity = {}
    screen_ev = []
    if screens and opts.get("tur", "vlog") != "ekran":
        notes.append("Ekran kaydı var ama kurgu türü 'Ekran anlatımı' değil: ekran kaydına geçilmedi")
        screens = set()
    if screens:
        progress(0.05, "Ekran kaydı inceleniyor")
        activity = _screen_activity(proj, screens)
        gaze = _gaze(proj, segs, lambda f, m: progress(0.1 + 0.5 * f, m))
        screen_ev, n2 = dr.screen_spans(gaze, segs, words, activity, opts["screen_side"])
        notes += n2
        if not gaze:
            notes.append("Yüz haritası modeli yok: ekran kaydına geçiş yapılamadı")
    elif proj.settings.get("screen_files"):
        notes.append("Ekran kaydı kamerayla eşleşmedi (ekran kaydında sesiniz de olmalı): geçiş yapılmadı")
    in_montage = {sh["file"] for sh in (pl.get("montage") or {}).get("shots", [])}
    items = [b for b in proj.settings.get("broll", []) if Path(b["file"]).exists() and b["file"] not in in_montage]
    if items:
        items, n3 = describe_broll(proj, items, llm, lambda f, m: progress(0.6 + 0.05 * f, m))
        notes += n3
    info = {b["file"]: {"duration": media_duration(b["file"])} for b in items}
    sim = None
    if items and llm is None:
        sim, _ = paraphrase_scorer()
    progress(0.65, "Ek görüntüler ve kartlar yerleştiriliyor")
    ev = dr.plan_events(segs, words, pl["stats"]["timeline_s"], opts, screen_ev, items, info, llm, sim)
    from .audio.master import builtin_sfx
    lib = sfx_library(proj.cfg) or builtin_sfx()
    cues = dr.sfx_cues(ev["events"], lib, dr.INTENSITY[opts["intensity"]]["sfx_db"])
    if isinstance(lib, dict):
        notes.append("Ses efekti kütüphanesi boş: uygulamanın hazır efektleri (whoosh, pop...) kullanıldı")
    out = {"options": opts, "events": ev["events"], "sfx": cues, "notes": notes + ev["notes"],
           "counts": {t: sum(1 for e in ev["events"] if e["type"] == t) for t in ("screen", "broll", "card", "subscribe")}}
    proj.save("yonetmen.json", out)
    for n in out["notes"]:
        proj.log.write("director", "note", n)
    return {"plan": out, "activity": activity}


def _intro_from_footage(proj: Project, pl: dict, music: str, items: list[dict], progress: Progress) -> dict | None:
    """Music intro cut from the moments that SHOW the food / the place: silent stretches the cut removed and
    "bakın / geldi" moments of the talk recordings, plus whole files marked as extra footage / "İkisi de".
    Repeats, feet, faces and shaky moves are filtered out (editing/broll_mine.py)."""
    from .editing import broll_mine as bm
    from .editing.montage import beat_grid
    from .motion.compose import media_duration
    grid = proj.cache.json("beats", [music], {"v": 1}, lambda: beat_grid(music))
    period = grid["period"]
    per_cut = max(1, int(round(2.0 / period)))
    slot = per_cut * period
    extra = [b["file"] for b in items] + [f for f in proj.settings.get("intro_files") or [] if Path(f).exists()]
    durs = {f: media_duration(f) or 0.0 for f in extra}
    vis = bm._Vision()
    if not vis.faces.ok or not vis.seg:   # without faces and bodies the filters cannot work: no intro
        proj.log.write("director", "intro", "İntro yapılmadı: yüz/beden modeli yok (modeller klasörü)")
        return None
    faces = {}
    parts = pl["camera"]["parts"]
    for k, prt in enumerate(parts):
        progress(0.01, f"İntro için kayıtlara bakılıyor {k + 1}/{len(parts)}")
        if Path(prt["path"]).exists():
            faces[prt["path"]] = proj.cache.json("face_keyframes", [prt["path"]], {"v": 2},
                                                 lambda pth=prt["path"]: bm.face_timeline(pth, vis))
    cands = bm.candidate_windows(pl, faces, min_len=slot + 0.4, extra_files=extra, durations=durs)
    # the moments he showed the food to the camera (kept in the edit) are the best intro material
    from .media.sources import CameraTimeline
    cam_tl = CameraTimeline.from_dict(pl["camera"])
    for sa, sb in pl.get("showing") or []:
        if sb - sa < slot + 0.3:
            continue
        part, la = cam_tl.locate(sa)
        cands.insert(0, {"file": part.path, "in": round(la, 3), "out": round(min(part.duration, la + (sb - sa)), 3),
                         "vt0": sa, "vt1": sb, "why": "gösterme anı"})
    if not cands:
        proj.log.write("director", "intro", "İntro yapılmadı: kayıtlarda yüzün görünmediği (yemek/mekân gösterilen) an yok")
        return None

    def _scan():
        shots = bm.scan(cands, slot, progress=lambda f, m: progress(0.02 + 0.5 * f, m), vis=vis)
        for sh in shots:   # numpy arrays are not JSON: keep only what pick() needs
            sh["_hist"] = sh["_hist"].tolist()
            sh["_mid"] = None
        return shots
    shots = proj.cache.json("intro_shots", sorted({c["file"] for c in cands}),
                            {"v": 3, "slot": round(slot, 3), "n": len(cands), "c": [[c["file"][-24:], c["in"]] for c in cands]},
                            _scan)
    skip = set(proj.settings.get("intro_haric") or [])
    shots = [dict(sh) for sh in shots if bm.key_of(sh) not in skip]
    import numpy as _np
    for sh in shots:
        sh["_hist"] = _np.asarray(sh["_hist"], _np.float32)
        if sh.get("_mid") is None:   # middle frame again (not cached) for the same-view check
            fr = bm._rgb_frames(sh["file"], (sh["in"] + sh["out"]) / 2, 0.4, fps=2.0)
            sh["_mid"] = fr[0] if fr else _np.zeros((180, 320, 3), _np.uint8)
    max_shots = int(_np.clip(round(36.0 / slot), 6, 18))
    chosen, notes = bm.pick(shots, pl.get("transcript_cut") or [], max_shots)
    if len(chosen) < 4:
        proj.log.write("director", "intro", f"İntro için yeterli temiz görüntü bulunamadı ({len(chosen)} çekim)")
        return None
    for i, sh in enumerate(chosen):
        sh["move"] = 1 if i % 2 == 0 else -1
    thumbs = bm.preview(chosen, proj.dir / "cikti" / "intro")
    proj.save("intro.json", {"shots": [dict(sh, key=bm.key_of(sh), thumb=t) for sh, t in zip(chosen, thumbs)],
                             "dropped": notes[:60], "excluded": sorted(skip)})
    for n in notes[:40]:
        proj.log.write("director", "intro_repeat", f"İntroya alınmadı (tekrar): {n}")
    return {"shots": [{k: sh[k] for k in ("file", "in", "out", "move")} for sh in chosen], "slot": slot,
            "beats_per_cut": per_cut, "bpm": grid["bpm"], "music_offset": round(grid["start"], 3),
            "duration": round(len(chosen) * slot, 3), "music_file": music, "source": "otomatik"}


def _insert_montage(proj: Project, pl: dict, music: str, progress: Progress) -> tuple[dict, str]:
    from .editing.montage import insert_into_plan, insert_point, plan_montage
    from .editing.shorts import sentences
    from .motion.compose import media_duration
    items = [b for b in proj.settings.get("broll", []) if Path(b["file"]).exists()
             and Path(b["file"]).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    if dr_options(proj).get("intro_otomatik", True):
        m = _intro_from_footage(proj, pl, music, items, progress)
        if m:
            at = insert_point(sentences(pl["transcript_cut"]), [b.get("note", "") for b in items], pl["segments"])
            out = insert_into_plan(pl, m, at)
            return out, (f"Müzikli intro (videodaki yemek/mekân anlarından): {len(m['shots'])} çekim, "
                         f"{m['duration']:.1f} sn, {m['bpm']:.0f} BPM; {at:.0f}. saniyede giriş konuşmasından sonra")
    if len(items) < 2:
        return pl, None
    items = sorted(items, key=lambda b: (b.get("shot") or _shot_time(b["file"]), b["file"]))
    progress(0.02, "Müzik ritmi ve ek görüntülerin en iyi anları bulunuyor")
    durs = {b["file"]: media_duration(b["file"]) or 0.0 for b in items}
    m = proj.cache.json("montage", [music] + [b["file"] for b in items], {"v": 3}, lambda: plan_montage(items, music, durs))
    at = insert_point(sentences(pl["transcript_cut"]), [b.get("note", "") for b in items], pl["segments"])
    notes = {(b.get("note") or "").strip().lower() for b in items}
    m = dict(m, title=(notes.pop() if len(notes) == 1 else ""))
    out = insert_into_plan(pl, m, at)
    return out, (f"Müzikli tanıtım: {len(m['shots'])} çekim, {m['duration']:.1f} sn, {m['bpm']:.0f} BPM, "
                 f"her kesim {m['beats_per_cut']} vuruşta; {at:.0f}. saniyede giriş konuşmasından sonra")


def polish_transcript(proj: Project, pl: dict, llm=None, progress: Progress = _noop) -> dict:
    """Misheard words fixed and English sentences translated before any caption or title is made
    (editing/proofread.py). The language model's answer is saved in yazim.json for this exact transcript and reused
    by the Shorts and later renders; without the model the saved answer (if any) is still applied."""
    import hashlib

    from .editing import proofread as prf
    words = pl.get("transcript_cut") or []
    if not words:
        return pl
    key = hashlib.sha1(" ".join(w["word"] for w in words).encode("utf-8")).hexdigest()[:16]
    saved = proj.load("yazim.json") or {}
    if saved.get("key") != key and llm is not None:
        _, tcfg = transcript_settings(proj.cfg)
        res = prf.proofread(words, llm, tcfg.get("vocabulary") or [], progress)
        res["key"] = key
        if res["failed"] < res["windows"]:
            proj.save("yazim.json", res)
            saved = res
            proj.log.write("text", "proofread", f"Yazım denetimi: {len(res['fixes'])} cümle düzeltildi, "
                           f"{len(res['translations'])} İngilizce cümle çevrildi ({res['failed']} bölüm atlandı)")
            for i, fx in list(res["fixes"].items())[:400]:
                proj.log.write("text", "fix", f"[{i}] '{fx['from']}' → '{fx['to']}'")
    if saved.get("key") == key:
        pl = dict(pl, transcript_cut=prf.apply(words, saved))
    return pl


def _speed_counting(proj: Project, pl: dict, fps: float) -> tuple[dict, list[dict]]:
    """Counting out loud ("bir, iki ... yirmi") plays faster and faster with a helium voice (editing/counting.py).
    Returns the re-timed plan and the counter badges [{t0, t1, marks, speeds}] in the new timeline."""
    from .editing.counting import apply_speed, find_counts, zones_of
    runs = find_counts(pl.get("transcript_cut") or [])
    if not runs:
        return pl, []
    zones = [z for r in runs for z in zones_of(r)]
    pl, remap = apply_speed(pl, zones, fps)
    out = []
    for r in runs:
        t0, t1 = remap(r["t0"]), remap(r["t1"])
        marks = [(round(remap(t) - t0, 3), v) for t, v in r["marks"]]
        speeds = [(round(max(0.0, remap(z[0]) - t0), 3), z[2]) for z in zones_of(r)]
        out.append({"t0": round(t0, 3), "t1": round(t1, 3), "marks": marks, "speeds": speeds})
        proj.log.write("director", "count", f"Sayma hızlandırıldı ({marks[0][1]}→{marks[-1][1]}): "
                       f"{r['t1'] - r['t0']:.1f} sn → {t1 - t0:.1f} sn, incecik ses + sayaç", start=t0, end=t1)
    return pl, out


def _talk_graphics(proj: Project, pl: dict, events: list[dict], w: int, h: int, accent, llm, progress) -> dict:
    """Talking-head mode: section titles with a step row and a corner label, key statements in big type, lists with
    icons - each on the left while the camera card moves to the right (motion/stage.py)."""
    from .editing.shorts import sentences as _sents
    from .editing.talk import item_times, plan_talk, word_time
    from .motion.editorial import Bullets, Chapter, Topic, overlay as ed_overlay, split_title
    from .motion.icons import pick as pick_icon
    from .motion.stage import Stage, plan_layout
    words, dur = pl["transcript_cut"], pl["stats"]["timeline_s"]
    by_i = {s["i"]: s for s in _sents(words)}
    overlays, sfx, wins = [], [], []
    busy = [(e["t0"], e["t1"]) for e in events if e["type"] in ("screen", "broll", "subscribe")]
    taken: list[tuple[float, float]] = []

    def free(a: float, b: float) -> bool:
        return not any(a < y and b > x for x, y in taken + busy)

    def place(obj, t0: float, t1: float, what: str, db: float = 0.0):
        overlays.append(ed_overlay(obj, t0))
        wins.append((t0, t1))
        taken.append((t0 - 1.0, t1 + 1.0))
        sfx.extend({"t": round(max(0.0, t0 + dt), 3), "name": name, "db": d_ + db} for dt, name, d_ in obj.sounds)
        proj.log.write("graphics", "talk", what, start=t0, end=t1)

    cards = sorted((e for e in events if e["type"] == "card"), key=lambda e: e["t0"])
    for k, e in enumerate(cards):
        big, small = (e.get("buyuk") or "").strip(), (e.get("italik") or "").strip()
        if not big:
            big, small = split_title(e["text"])
        label = f"Bölüm {k + 1}"
        t0, t1 = e["t0"], e["t0"] + 4.4
        icon = pick_icon(big, small, e.get("text") or "")
        tp = Topic(label, big, small, w, h, side="left", dur=t1 - t0, panel_frac=0.48, cy=0.41,
                   steps=(k, len(cards)) if len(cards) >= 3 else None, icon=icon)
        place(tp, t0, t1, f"Bölüm başlığı: {label} · {big} / {small}" + (f" [simge {icon}]" if icon else ""))
        end = cards[k + 1]["t0"] - 0.2 if k + 1 < len(cards) else dur
        if end - t1 > 2.0:
            overlays.append(ed_overlay(Chapter(label, big, w, h, end - t1), t1))
    res = {"phrases": [], "lists": []}
    if llm is not None:
        try:
            res = plan_talk(llm, words, progress)
        except Exception as ex:  # noqa: BLE001
            proj.log.write("graphics", "note", f"Anlatım yazı planı yapılamadı: {ex}")
    for L in res["lists"]:
        times = item_times(by_i, L["sentence"], L["items"])
        if len(times) < 2 or times[-1] - times[0] > 16:
            continue
        t0, t1 = times[0] - 0.5, times[-1] + 2.8
        if not free(t0, t1):
            continue
        bl = Bullets(L["kicker"], L["items"], [t - t0 for t in times], w, h, side="left", dur=t1 - t0, theme="dark",
                     icon={"x": "x", "v": "check"}.get(L["icon"], "arrow"), cy=0.43)
        place(bl, t0, t1, f"Liste ({L['icon']}): [{L['kicker']}] " + " / ".join(L["items"]), db=-2)
    last = -1e9
    for p in sorted(res["phrases"], key=lambda p: p["sentence"]):
        s = by_i.get(p["sentence"])
        hb = word_time(s["words"], p["big"]) if s else None
        if not hb:
            continue
        hs = word_time(s["words"], p["small"], hb[1] + 1) if p["small"] else None
        t0 = hb[0] - 0.35
        t1 = min(t0 + 7.0, max(t0 + 3.4, s["end"] + 1.2))
        if t0 - last < 12 or not free(t0, t1):
            continue
        big = p["big"][:1].upper() + p["big"][1:]
        icon = pick_icon(p["kicker"], big)
        tp = Topic(p["kicker"], big, p["small"], w, h, side="left", dur=t1 - t0, panel_frac=0.48, cy=0.41,
                   t_small=(hs[0] - t0) if hs else None, icon=icon)
        tp.sounds = [(0.0, "swish", -12)]      # calmer than a section title
        place(tp, t0, t1, f"Vurgu: [{p['kicker']}] {big} / {p['small']}" + (f" [simge {icon}]" if icon else ""))
        last = t0
    layout = plan_layout([{"t0": a, "t1": b, "side": "left"} for a, b in wins], dur,
                         [(a, b) for a, b in busy])
    n_card = sum(1 for s in layout if s["state"] not in ("full", "none"))
    proj.log.write("director", "stage", f"Anlatım sahnesi: {len(wins)} yazı, kamera {n_card} kez karta geçti")
    return {"overlays": overlays, "sfx": sfx, "stage": Stage(layout, w, h, tuple(accent)), "windows": wins}


def _prepare_main(proj: Project, progress: Progress, llm) -> dict:
    """Shared by both engines: director plan, zooms, face centres, text overlays, captions file."""
    from .editing import director as dr
    from .editing.graphics import build_graphics
    from .editing.plan import resnap
    from .motion.titles import build_title
    from .motion.captions import build_ass
    from .motion.compose import overlays_for
    from .motion.render import Overlay, Piece

    pl = proj.load("edit_plan.json")
    if not pl:
        raise RuntimeError("Önce kesim yapılmalı")
    fps = _out_fps(pl)
    res = str(dr_options(proj).get("resolution", "1440"))
    if fps > 31 and res in ("1080", "1440"):
        fps = round(fps / 2, 3)   # fast mode: 50/60 fps recordings are edited at 25/30 fps (half the frames)
    pl = resnap(pl, fps)
    pl = polish_transcript(proj, pl, llm, lambda f, m: progress(0.0, m))
    counts = []
    if dr.options_of(proj.settings).get("sayma_hizli", True):
        try:
            pl, counts = _speed_counting(proj, pl, fps)
        except Exception as e:  # noqa: BLE001 - the count simply stays at normal speed
            proj.log.write("director", "note", f"Sayma hızlandırılamadı: {e}")
    segs = pl["segments"]
    # screen recordings removed/moved after the cut was planned: those parts simply stay on the camera
    current = {str(Path(f)) for f in proj.settings.get("screen_files") or []}
    dropped = 0
    for sg in segs:
        sc = sg.get("screen")
        if sc and (str(Path(sc["file"])) not in current or not Path(sc["file"]).exists()):
            sg["screen"] = None
            dropped += 1
    if dropped:
        proj.log.write("director", "note", f"{dropped} parçanın ekran kaydı artık projede yok: kamera kullanıldı")
    opts = dr.options_of(proj.settings)
    # music-cut promo of the extra footage right after the intro talk
    music = music_for(proj) if opts.get("music", True) else None
    mont_note = None
    if opts.get("montage", True) and music:
        try:
            pl, mont_note = _insert_montage(proj, pl, music, progress)
            segs = pl["segments"]
        except Exception as e:  # noqa: BLE001
            mont_note = f"Müzikli tanıtım yapılamadı: {e}"
    if mont_note:
        proj.log.write("director", "montage", mont_note)
    P = dr.INTENSITY[opts["intensity"]]
    d = direct(proj, pl, lambda f, m: progress(0.6 * f, m), llm)
    events = d["plan"]["events"]
    progress(0.6, "Yüz konumları ölçülüyor")
    small_frames: dict = {}
    faces = _segment_faces(segs, small_frames)
    progress(0.8, "Efekt ve yazı planı")
    talk = opts.get("tur") == "anlatim"     # talking-head mode: camera card on a dark set + editorial type
    gfx = build_graphics(pl["transcript_cut"], segs, pl["stats"]["timeline_s"], None if talk else llm, P,
                         progress=lambda f, m: progress(0.8 + 0.18 * f, m))
    if talk:   # the vlog titles and lists are replaced by the talk plan (_talk_graphics)
        gfx["callouts"], gfx["lists"] = [], []
    from .motion.render import canvas_geometry
    src_w, src_h = pl["sequence"]["width"], pl["sequence"]["height"]
    cam0 = next((sg["file"] for sg in segs if sg.get("kind") != "montage"), None)
    if cam0:   # display size (older plans stored the raw, un-rotated size)
        info0 = ff.probe(cam0)
        if info0.width and info0.height:
            src_w, src_h = info0.width, info0.height
    w, h = src_w, src_h
    if opts.get("format", "yatay") == "yatay" and src_h > src_w:     # vertical recording -> 16:9 YouTube frame
        w, h = {"1080": (1920, 1080), "1440": (2560, 1440)}.get(res, (3840, 2160))
        if src_w < h:   # never upscale the source
            w, h = (2560, 1440) if src_w >= 1440 else (1920, 1080)
    elif res in OUT_HEIGHT and h > OUT_HEIGHT[res]:   # 4K source -> 1440p (YouTube's sharper codec) or fast 1080p
        w, h = int(round(w * OUT_HEIGHT[res] / h / 2) * 2), OUT_HEIGHT[res]
    geo = canvas_geometry(src_w, src_h, w, h)

    def to_out(x: float, y: float, fw_px: float, fh_px: float) -> tuple[float, float]:
        """Source pixel -> output pixel (picture may sit in the middle of a wider frame)."""
        return (geo["ox"] + x / fw_px * geo["fw"]) * w, (geo["oy"] + y / fh_px * geo["fh"]) * h
    progress(0.99, "Zoom hedefleri (yüz mü, gösterilen şey mi)")
    focus, fstats = _segment_focus(segs, faces, pl["transcript_cut"], gfx["zooms"], small_frames)
    small_frames.clear()
    proj.log.write("director", "focus", f"Zoom hedefi: {fstats['nesne']} parçada gösterilen nesne, "
                   f"{fstats['zoom_yok']} parçada zoom yok, {fstats['yuz']} parçada yüz", **fstats)
    pieces = []
    for s, z, fb, fo in zip(segs, gfx["zooms"], faces, focus):
        cx = ((fb[0] + fb[2]) / 2 / fb[4]) if fb else 0.5
        cy = ((fb[1] + fb[3]) / 2 / fb[5]) if fb else 0.42
        if fo:
            cx, cy, z = fo
        if s.get("kind") == "montage":
            first = not pieces or pieces[-1].kind != "montage"
            pieces.append(Piece(s["file"], s["in"], s["out"], kind="montage", move=s.get("move", 1),
                                trans_in=0.25 if first else 0.0))
            continue
        pieces.append(Piece(s["file"], s["in"], s["out"], zoom=z, cx=cx, cy=cy, screen=s.get("screen"),
                            speed=float(s.get("speed", 1.0)), pitch=float(s.get("pitch", 1.0))))
    from .audio.master import builtin_sfx
    pops = builtin_sfx()
    sfx_extra = []
    overlays, placed = [], []
    style = str(opts.get("yazi_stili", "karisik"))
    from .motion.titles import set_theme
    accent = set_theme(str(opts.get("renk", "otomatik")), proj.name)
    n_title = 0
    motor = str(opts.get("motor", "kurgu"))
    hfx = motor in ("karma", "hyperframes")     # on-screen graphics come from HyperFrames catalog blocks
    hf_specs: list[dict] = []
    hf_sound = {"title": "swish", "price": "ding", "list": "pop"}
    for c in gfx["callouts"]:
        if any(c["t0"] < e["t1"] + 0.3 and c["t1"] > e["t0"] - 0.3 for e in events):
            continue  # the camera is not full-frame there (screen / b-roll / card)
        if hfx:
            kind = "price" if c.get("kicker") == "FİYAT" else "title"
            hf_specs.append({"kind": kind, "t0": c["t0"], "text": c["text"], "kicker": c.get("kicker", ""), "idx": n_title})
            n_title += 1
            placed.append(dict(c, style="hyperframes"))
            proj.log.write("graphics", "callout", f"HyperFrames başlık: [{c.get('kicker', '')}] '{c['text']}'",
                           start=c["t0"], end=c["t1"])
            continue
        seg_i = next((k for k, s in enumerate(segs) if s["timeline_start"] <= c["t0"] < s["timeline_end"]), None)
        fb = faces[seg_i] if seg_i is not None else None
        z = pieces[seg_i].zoom if seg_i is not None else 1.0
        face_px = None
        if fb:
            x0, y0 = to_out(fb[0], fb[1], fb[4], fb[5])
            x1, y1 = to_out(fb[2], fb[3], fb[4], fb[5])
            if z > 1.001:  # face grows with the punch-in: widen the keep-out box
                cxp, cyp = (x0 + x1) / 2, (y0 + y1) / 2
                x0, x1 = cxp - (cxp - x0) * z, cxp + (x1 - cxp) * z
                y0, y1 = cyp - (cyp - y0) * z, cyp + (y1 - cyp) * z
            pad = 0.25 * (x1 - x0)
            face_px = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        layers, info = build_title(c["text"], c.get("kicker", ""), w, h, face_px, captions=bool(opts["captions"]),
                                   style=style, idx=n_title)
        n_title += 1
        if not layers:
            proj.log.write("graphics", "skip", f"Yazı yüze binerdi, eklenmedi: '{c['text']}'", start=c["t0"])
            continue
        for L in layers:
            overlays.append(Overlay(c["t0"] + L.get("delay", 0.0), c["t1"], L["rgba"], fade=L.get("fade", 0.25),
                                    anim=L["anim"], meta=L.get("meta") or {}))
        placed.append(dict(c, **{k: v for k, v in info.items() if k != "sfx"}))
        for dt, name, db in info["sfx"]:
            if name in pops:
                sfx_extra.append({"t": round(max(0.0, c["t0"] + dt), 3), "file": pops[name], "gain_db": P["sfx_db"] + db + 4})
        proj.log.write("graphics", "callout", f"Başlık ({info['style']}): [{c.get('kicker', '')}] '{c['text']}' ({info['side']})",
                       start=c["t0"], end=c["t1"])
    # section titles (editorial style): the camera slides away and the title is set on the free side; lists the
    # speaker counts off appear as pills, item by item as they are said. (Sentences already in the captions are
    # never repeated on screen.)
    camfx = []
    from .motion.editorial import Bullets, Topic, overlay as ed_overlay, split_title

    def _face_x(t):
        k = next((k for k, sg in enumerate(segs) if sg["timeline_start"] <= t < sg["timeline_end"]), None)
        if k is None:
            return None, 0.5, 0.42
        pc = pieces[k]
        return k, geo["ox"] + pc.cx * geo["fw"], geo["oy"] + pc.cy * geo["fh"]
    if opts.get("kamera_oyunlari", True) and not talk:
        for e in events:
            if e["type"] != "card":
                continue
            k, fcx, fcy = _face_x(e["t0"] + 0.3)
            if k is None or segs[k].get("kind") == "montage":
                continue
            side = "right" if fcx < 0.5 else "left"
            big, small = (e.get("buyuk") or "").strip(), (e.get("italik") or "").strip()
            if not big:
                big, small = split_title(e["text"])
            if hfx:   # a HyperFrames lower third; the camera stays put
                hf_specs.append({"kind": "title", "t0": e["t0"], "text": big, "kicker": e.get("ust") or small, "idx": n_title})
                n_title += 1
                proj.log.write("graphics", "topic", f"HyperFrames bölüm başlığı: {big}", start=e["t0"], end=e["t1"])
                continue
            tp = Topic(e.get("ust") or "", big, small, w, h, side=side, dur=e["t1"] - e["t0"])
            overlays.append(ed_overlay(tp, e["t0"]))
            camfx.append({"t0": e["t0"], "t1": e["t1"], "kind": "punch" if geo["fit"] else "slide", "side": side,
                          "amt": 0.36, "cx": fcx, "cy": fcy, "zoom": 1.15, "text": e["text"]})
            for dt, name, db in tp.sounds:
                if name in pops:
                    sfx_extra.append({"t": round(e["t0"] + dt, 3), "file": pops[name], "gain_db": P["sfx_db"] + db + 4})
            proj.log.write("graphics", "topic", f"Bölüm başlığı: [{e.get('ust', '')}] {big} / {small}", start=e["t0"], end=e["t1"])
    from .editing.shorts import sentences as _sents
    from .analysis.text import norm_word as _nw
    sent_by_i = {s_["i"]: s_ for s_ in _sents(pl["transcript_cut"])}
    taken = [(c["t0"] - 0.5, c["t1"] + 0.5) for c in placed] + [(e["t0"] - 0.5, e["t1"] + 0.5) for e in events]
    last_list = -1e9
    for L in gfx.get("lists") or []:
        s_ = sent_by_i.get(L["sentence"])
        if not s_:
            continue
        # item times: where the first word of each item is said (in this or the next sentences)
        ws = [w_ for i_ in range(L["sentence"], L["sentence"] + 4) for w_ in (sent_by_i.get(i_) or {"words": []})["words"]]
        times, pos = [], 0
        for it in L["items"]:
            key = _nw(it.split()[0])[:4]
            hit = next((j for j in range(pos, len(ws)) if _nw(ws[j]["word"]).startswith(key)), None)
            if hit is None:
                times = []
                break
            times.append(ws[hit]["t_start"])
            pos = hit + 1
        if len(times) < 2 or times[-1] - times[0] > 14 or times[0] - last_list < 60:
            continue
        t0, t1 = times[0] - 0.5, times[-1] + 2.6
        if any(t0 < y and t1 > x for x, y in taken):
            continue
        k, fcx, _ = _face_x(times[0])
        if k is None or segs[k].get("kind") == "montage":
            continue
        if hfx:
            hf_specs.append({"kind": "list", "t0": t0, "items": L["items"], "kicker": L.get("kicker") or ""})
            taken.append((t0, t0 + 8.0))
            last_list = times[0]
            proj.log.write("graphics", "list", "HyperFrames liste: " + " / ".join(L["items"]), start=t0)
            continue
        bl = Bullets(L.get("kicker") or "", L["items"], [t_ - t0 for t_ in times], w, h,
                     side="right" if fcx < 0.5 else "left", dur=t1 - t0)
        overlays.append(ed_overlay(bl, t0))
        for dt, name, db in bl.sounds:
            if name in pops:
                sfx_extra.append({"t": round(t0 + dt, 3), "file": pops[name], "gain_db": P["sfx_db"] + db + 4})
        taken.append((t0, t1))
        last_list = times[0]
        proj.log.write("graphics", "list", f"Liste: [{L.get('kicker', '')}] " + " / ".join(L["items"]), start=t0, end=t1)
    # topic changes without an LLM: a long removed pause between two kept parts = new thought -> transition
    last_t = -1e9
    n_topic = 0
    for k in range(1, len(segs)):
        a, b = segs[k - 1], segs[k]
        if b.get("kind") == "montage" or a.get("kind") == "montage" or pieces[k].trans_in:
            continue
        gap = (b["in"] - a["out"]) if a["file"] == b["file"] else 5.0
        if gap >= 4.0 and b["timeline_start"] - last_t >= 45.0 and b["timeline_end"] - b["timeline_start"] >= 2.0:
            # professional variety: punch-in zoom, whip pan, warm light leak - in turn, never the same twice in a row
            kind = ("zoom", "whip", "flash")[n_topic % 3]
            n_topic += 1
            pieces[k].trans_in, pieces[k].trans = 0.18, kind
            last_t = b["timeline_start"]
            sfx_extra.append({"t": round(max(0.0, b["timeline_start"] - (0.12 if kind == "whip" else 0.15)), 3),
                              "file": pops["swish" if kind == "flash" else "whoosh"],
                              "gain_db": P["sfx_db"] - (5 if kind == "flash" else 2)})
    overlays += overlays_for([e for e in events if e["type"] not in ("subscribe", "card")], w, h)
    stage, text_wins = None, []
    if talk:
        tg = _talk_graphics(proj, pl, events, w, h, accent, llm, lambda f, m: progress(0.99, m))
        overlays += tg["overlays"]
        for c in tg["sfx"]:
            if c["name"] in pops:
                sfx_extra.append({"t": c["t"], "file": pops[c["name"]], "gain_db": P["sfx_db"] + c["db"]})
        stage, text_wins = tg["stage"], tg["windows"]
    # subscribe / like / comment: animated, clicked at the very words that ask for it, away from the face
    from .motion.cta import build_cta
    for e in events:
        if e["type"] != "subscribe" or not e.get("items"):
            continue
        k = next((k for k, sg in enumerate(segs) if sg["timeline_start"] <= e["t0"] + 0.5 < sg["timeline_end"]), None)
        fcx = geo["ox"] + pieces[k].cx * geo["fw"] if k is not None else 0.5
        ov, sounds = build_cta(e, w, h, side="right" if fcx < 0.5 else "left", captions=bool(opts["captions"]))
        overlays.append(ov)
        for dt, name, db in sounds:
            if name in pops:
                sfx_extra.append({"t": round(max(0.0, e["t0"] + dt), 3), "file": pops.fixed(name) if hasattr(pops, "fixed") else pops[name], "gain_db": P["sfx_db"] + db + 3})
        proj.log.write("graphics", "cta", "Abone/beğeni hatırlatması (söylendiği an): " +
                       ", ".join(f"{kd} @{t:.1f}s" for kd, t in e["items"]), start=e["t0"], end=e["t1"])
    # "look at this": a swish as the camera pushes in on the thing being shown
    last_sw = -99.0
    for k, (s_, fo) in enumerate(zip(segs, focus)):
        if fo and fo[2] > 1.15 and s_["timeline_start"] - last_sw > 6.0:
            sfx_extra.append({"t": round(max(0.0, s_["timeline_start"] - 0.05), 3), "file": pops["swish"],
                              "gain_db": P["sfx_db"] - 3})
            last_sw = s_["timeline_start"]
    out_dir = proj.dir / "cikti"
    out_dir.mkdir(exist_ok=True)
    # counting sped up: only the voice speeds up. The counter badge ("19 / HIZLI x4") was removed (23.10.2026): the
    # user found it distracting and it showed before the speed-up was audible.
    for c in counts:
        proj.log.write("director", "counting", f"Sayma hızlandı ({c['t1'] - c['t0']:.1f} sn), rozet yok", start=c["t0"], end=c["t1"])
    # every cue is in by now (the counter's ticks above were added after an earlier merge and never played)
    d["plan"]["sfx"] = sorted(d["plan"]["sfx"] + sfx_extra, key=lambda c: c["t"])
    ass = None
    if opts["captions"]:
        cards = [e for e in events if e["type"] == "card"]
        hide = [(e["t0"] - 0.1, e["t1"]) for e in cards] + [(a - 0.1, b) for a, b in text_wins]
        cw = [x for x in pl["transcript_cut"] if not x.get("fast") and
              not any(a <= x["t_start"] < b for a, b in hide)]
        cw = [dict(x, start=x["t_start"], end=x["t_end"]) for x in cw]
        ass = out_dir / "kurgu_altyazi.ass"
        brk = [pl["montage"]["at"]] if pl.get("montage") else []
        ass.write_text(build_ass(cw, w, h, proj.cfg.design.get("font", "Bahnschrift"), size=int(66 * h / 1080),
                                 y_frac=0.885, breaks=brk, active_color=ass_color(accent),
                                 style=caption_style(opts, vertical=False)), encoding="utf-8")
    if pl.get("montage"):
        from .audio.master import builtin_sfx as _fx
        from .motion.compose import draw_headline
        fx = _fx()
        m = pl["montage"]
        a0, a1 = m["at"], m["at"] + m["duration"]
        pre = min(1.8, m["music_offset"])          # music swells in under the last words (pre-lap)
        cues = [{"t": round(max(0.0, a0 - pre), 3), "file": m["music_file"], "offset": round(m["music_offset"] - pre, 3),
                 "dur": round(m["duration"] + pre + 2.0, 3), "gain_db": -9.0, "fade_in": pre, "fade_out": 2.0},
                {"t": round(max(0.0, a0 - 0.35), 3), "file": fx["whoosh"], "gain_db": -14.0},
                {"t": round(a0, 3), "file": fx["impact"], "gain_db": -18.0},
                {"t": round(max(0.0, a1 - 0.3), 3), "file": fx["whoosh"], "gain_db": -15.0}]
        d["plan"]["sfx"] = sorted(d["plan"]["sfx"] + cues, key=lambda c: c["t"])
        # back to the talk with a zoom transition
        k = next((k for k, sg in enumerate(segs) if sg.get("kind") != "montage" and sg["timeline_start"] >= a1 - 0.05), None)
        if k is not None:
            pieces[k].trans_in = 0.2
        if m.get("title") and hfx:
            hf_specs.append({"kind": "title", "t0": a0 + 0.35, "text": m["title"].upper(), "kicker": "", "idx": n_title})
            n_title += 1
        elif m.get("title"):
            overlays.append(Overlay(a0 + 0.35, min(a1, a0 + 3.2), draw_headline(m["title"], w, h), fade=0.3, anim="pop"))
        d["plan"]["notes"].append(mont_note or "")
    if hfx and hf_specs:
        pops_ = builtin_sfx()
        cues = [{"t": round(max(0.0, s_["t0"] + 0.1), 3), "file": pops_[hf_sound[s_["kind"]]],
                 "gain_db": P["sfx_db"] + (0 if s_["kind"] == "price" else -2)} for s_ in hf_specs]
        d["plan"]["sfx"] = sorted(d["plan"]["sfx"] + cues, key=lambda c: c["t"])
    if motor == "karma" and hf_specs:
        from .motion import hf_graphics as hg
        if not hg.available():
            raise RuntimeError("Karma motoru için HyperFrames kurulu değil (araclar\\hyperframes)")
        for k, s_ in enumerate(sorted(hf_specs, key=lambda s: s["t0"])):
            progress(0.99, f"HyperFrames grafik çiziyor: {k + 1}/{len(hf_specs)}")
            ov, _info = hg.overlay(s_, accent, proj.dir / "cache" / "hyperframes", w, h, fps)
            overlays.append(ov)
    speech = [(x["t_start"], x["t_end"]) for x in pl["transcript_cut"]]
    return {"pl": pl, "fps": fps, "segs": segs, "opts": opts, "P": P, "d": d, "events": events, "gfx": gfx, "geo": geo,
            "speech": speech,
            "music": None if pl.get("montage") else music,
            "pieces": pieces, "overlays": overlays, "placed": placed, "camfx": camfx, "w": w, "h": h, "ass": ass,
            "stage": stage, "motor": motor, "hf_specs": hf_specs, "accent": accent,
            "out_dir": out_dir}


def ass_color(rgb) -> str:
    """(r, g, b) -> ASS colour &H00BBGGRR."""
    return f"&H00{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}"


def caption_style(opts: dict, vertical: bool) -> str:
    """Burned-in caption effect from the option "yazi_efekti": karaoke -> coloured word, kutu -> word on a coloured
    box, zipla -> words pop; "karisik": Shorts get the box (the TikTok look), the long video the coloured word."""
    e = str(opts.get("yazi_efekti", "karisik"))
    return {"karaoke": "vurgu", "kutu": "kutu", "zipla": "zipla"}.get(e, "kutu" if vertical else "vurgu")


def _main_info(proj: Project, ctx: dict, out: Path, extra: dict) -> dict:
    probe = ff.probe(out)
    fps = ctx["fps"]
    expected = sum(round(p.dur * fps) for p in ctx["pieces"]) / fps
    gfx, d = ctx["gfx"], ctx["d"]
    info = {"file": str(out), "duration": round(probe.duration, 2), "expected": round(expected, 2),
            "width": probe.width, "height": probe.height, "fps": fps,
            "graphics": {"callouts": ctx["placed"], "camfx": len(ctx.get("camfx") or []),
                         "zoom_segments": sum(1 for z in gfx["zooms"] if z > 1.001),
                         "method": gfx["method"], "warning": gfx["warning"]},
            "director": {"counts": d["plan"]["counts"], "sfx": len(d["plan"]["sfx"]), "notes": d["plan"]["notes"],
                         "options": ctx["opts"]},
            "captions": bool(ctx["ass"]), "ok": abs(probe.duration - expected) < 0.5}
    info.update(extra)
    proj.save("kurgu.json", info)
    return info


def make_main(proj: Project, progress: Progress = _noop, llm=None, ctx: dict | None = None, note: str | None = None,
              keep_audio: str | None = None) -> dict:
    """Steps 2-4 with our own render engine (no Premiere) -> cikti/kurgu.mp4."""
    from .motion.compose import Compositor
    from .motion.render import render

    ctx = ctx or _prepare_main(proj, lambda f, m: progress(0.1 * f, m), llm)
    events, w, h, fps = ctx["events"], ctx["w"], ctx["h"], ctx["fps"]
    rt, rt_note = _retoucher(proj)
    out = ctx["out_dir"] / {"karma": "kurgu_karma.mp4"}.get(ctx.get("motor"), "kurgu.mp4")
    comp = Compositor(events, ctx["d"]["activity"], w, h, fps, proj.cfg.design.get("demo", {})) \
        if any(e["type"] in ("screen", "broll") for e in events) else None
    from .integrations.premiere_edit import final_cq, youtube_upscale
    up = youtube_upscale(ctx)
    try:
        res = render(ctx["pieces"], str(out), w, h, fps, frame_fn=rt, overlays=ctx["overlays"], image=proj.cfg.design.get("image"),
                     cq=final_cq(ctx, up), upscale=up, camfx=ctx.get("camfx"), keep_audio=keep_audio,
                     stage=ctx.get("stage"),
                     ass_path=str(ctx["ass"]) if ctx["ass"] else None,
                     loudness=proj.cfg.design.get("shorts", {}).get("loudness", "I=-14:TP=-1.5:LRA=11"),
                     progress=lambda f, m: progress(0.1 + 0.88 * f, m), compose=comp, sfx=ctx["d"]["plan"]["sfx"],
                     music=ctx.get("music"), speech=ctx.get("speech"))
    finally:
        if comp:
            comp.close()
    info = _main_info(proj, ctx, out, {
        "engine": "kendi", "engine_note": note, "render": res,
        "retouch": {"applied": rt is not None, "note": rt_note, "frames_with_face": rt.state["faces"] if rt else 0}})
    progress(1.0, "Kurgulanmış video hazır")
    return info


def make_preview(proj: Project, progress: Progress = _noop) -> dict:
    """Quick look at the cut before anything heavy: small frame, no retouch/effects, the voice as recorded.
    Takes minutes, not hours; meant to decide whether the plan is right."""
    from .editing.plan import resnap
    from .motion.render import Piece, render
    pl = proj.load("edit_plan.json")
    if not pl:
        raise RuntimeError("Önce taslak kesim yapılmalı")
    fps = 25.0
    pl = resnap(pl, fps)
    info0 = ff.probe(pl["segments"][0]["file"])
    sw, sh = info0.width or 1920, info0.height or 1080
    h = 540
    w = int(round(sw * h / sh / 2) * 2)
    if sh > sw and (proj.settings.get("options") or {}).get("format", "yatay") == "yatay":
        w = 960
    pieces = [Piece(s["file"], s["in"], s["out"]) for s in pl["segments"]]
    out = proj.dir / "cikti" / "onizleme.mp4"
    out.parent.mkdir(exist_ok=True)
    res = render(pieces, str(out), w, h, fps, audio_master=False, cq=30,
                 progress=lambda f, m: progress(f, "Önizleme: " + m))
    info = {"file": str(out), "duration": round(res["duration"], 2), "plan_created": pl.get("created"),
            "segments": len(pieces)}
    proj.save("onizleme.json", info)
    progress(1.0, "Önizleme hazır")
    return info


def make_video(proj: Project, progress: Progress = _noop, llm=None) -> dict:
    """The finished video always comes from our engine in ONE pass straight from the recordings (all animations,
    camera moves and the intro need it; Premiere's export flattened them and took hours). With the "premiere"
    option a Premiere review project is built afterwards on the original recordings."""
    if dr_options(proj).get("cikis", "yatay") != "yatay":
        return make_vertical(proj, progress, llm)
    ctx = _prepare_main(proj, lambda f, m: progress(0.08 * f, m), llm)
    review = ctx["opts"].get("engine", "premiere") == "premiere"   # Premiere review project (and the board meeting)
    span = 0.80 if review else 0.92
    work = proj.dir / "premiere"
    if ctx.get("motor") == "hyperframes":   # the whole video is assembled and drawn by HyperFrames
        from .motion.hf_engine import make_hyperframes
        info = make_hyperframes(proj, ctx, lambda f, m: progress(0.08 + span * f, m))
    elif ctx.get("motor") == "karma":       # our cut and camera + HyperFrames graphics
        work.mkdir(exist_ok=True)
        info = make_main(proj, lambda f, m: progress(0.08 + span * f, m), llm, ctx,
                         note="Karma motor: kesim ve kamera Kurgu Masası'ndan, yazılar HyperFrames'ten",
                         keep_audio=str(work / "kurgu_ses_son.wav") if review else None)
    elif not review:
        return make_main(proj, lambda f, m: progress(0.08 + 0.92 * f, m), llm, ctx)
    else:
        work.mkdir(exist_ok=True)
        info = make_main(proj, lambda f, m: progress(0.08 + 0.80 * f, m), llm, ctx,
                         keep_audio=str(work / "kurgu_ses_son.wav"))
    if not review:
        return info
    from .integrations.premiere_edit import make_main_premiere
    try:
        return make_main_premiere(proj, ctx, lambda f, m: progress(0.88 + 0.12 * f, f"Premiere inceleme projesi: {m}"),
                                  final=info)
    except Exception as e:  # noqa: BLE001 - the video is finished; the review project is a bonus
        if type(e).__name__ == "JobCancelled":
            raise
        why = str(e)[:200]
        if "4551" in why:   # 23.10.2026: Smart App Control blocked Adobe Premiere Pro.exe (its signature did not match)
            why = ("Windows'un Akıllı Uygulama Denetimi Premiere'i engelledi (programın Adobe imzası uyuşmuyor). "
                   "Premiere'i Creative Cloud'dan onarın ya da yeniden kurun; ajanlar Premiere'siz toplanır.")
        info["engine_note"] = f"Video hazır; Premiere inceleme projesi kurulamadı: {why}"
        proj.log.write("render", "premiere_review", info["engine_note"])
        proj.save("kurgu.json", info)
        return info


def make_package(proj: Project, progress: Progress = _noop, llm=None) -> dict:
    """Step 5: material for the thumbnail/title work the user does in ChatGPT: 6 clean frames (no text, colour
    corrected and retouched, taken from the camera, never from the finished video with captions) + transcript."""
    from PIL import Image

    from .editing import chatgpt as cg
    from .motion.thumbnails import grab, pick_frames
    from .vision.faces import backend

    pl = proj.load("edit_plan.json")
    if not pl:
        raise RuntimeError("Önce kesim yapılmalı")
    from .editing.plan import resnap
    pl = resnap(pl, _out_fps(pl))
    kit = proj.dir / "cikti" / "kareler"
    kit.mkdir(parents=True, exist_ok=True)
    for old in kit.glob("kare_*.jpg"):
        old.unlink()   # frames this step wrote last time
    frames = []
    if backend() == "yok":
        raise RuntimeError("Kareleri seçmek için yüz modeli gerekli (modeller klasörü)")
    progress(0.05, "En iyi kareler seçiliyor")
    cands = pick_frames(pl["segments"], n_samples=72, keep=6, min_gap_s=20.0,
                        progress=lambda f, m: progress(0.05 + 0.7 * f, m))
    rt, _ = _retoucher(proj)
    for k, f in enumerate(cands[:6]):
        rgb = grab(f["file"], f["t"])
        if rt:
            from types import SimpleNamespace
            rgb = rt(rgb, SimpleNamespace(file=f["file"]), k * 40, 0.0)
        dst = kit / f"kare_{k + 1}.jpg"
        Image.fromarray(rgb).save(dst, "JPEG", quality=95)
        frames.append({"file": str(dst), "t": f["t_timeline"], "score": round(f.get("score", 0), 3)})
    progress(0.9, "Transkript yazılıyor")
    words = pl["transcript_cut"]
    (kit / "transkript_zamanli.txt").write_text(cg.transcript_text(words) + "\n", encoding="utf-8")
    plain = " ".join(w["word"] for w in words)
    (kit / "transkript.txt").write_text(plain + "\n", encoding="utf-8")
    res = {"frames": frames, "transcript_file": str(kit / "transkript_zamanli.txt"),
           "transcript_plain": str(kit / "transkript.txt"), "meta": {}, "thumbnails": []}
    proj.save("paket.json", res)
    progress(1.0, "Kareler ve transkript hazır")
    return res


def _pair_titles(pk: dict) -> None:
    titles = (pk.get("meta") or {}).get("titles") or []
    for t in pk.get("thumbnails", []):
        i = "ABC".find(t.get("label", "")[:1])
        if 0 <= i < len(titles):
            t["title"] = titles[i]


def apply_chatgpt_answer(proj: Project, text: str) -> dict:
    from .editing import chatgpt as cg
    pk = proj.load("paket.json")
    if not pk:
        raise RuntimeError("Önce ChatGPT paketi hazırlanmalı (Kapak ve metin)")
    meta = cg.parse_answer(text)
    meta["raw"] = text[:20000]
    pk["meta"] = meta
    _pair_titles(pk)
    proj.save("paket.json", pk)
    proj.save("youtube.json", meta)
    proj.log.write("package", "chatgpt", "ChatGPT cevabı alındı: " + " | ".join(meta["titles"]))
    return pk


def add_chatgpt_thumbnail(proj: Project, label: str, src: Path) -> dict:
    from .editing import chatgpt as cg
    if label not in ("A", "B", "C"):
        raise ValueError("Kapak harfi A, B ya da C olmalı")
    pk = proj.load("paket.json")
    if not pk:
        raise RuntimeError("Önce ChatGPT paketi hazırlanmalı (Kapak ve metin)")
    out = proj.dir / "cikti" / f"kapak_{label}.jpg"
    info = cg.fit_thumbnail(src, out)
    th = [t for t in pk.get("thumbnails", []) if t.get("label") != label]
    th.append({"label": label, "file": info["file"], "design": "ChatGPT", "text": "", "ok": True, "kb": info["kb"]})
    pk["thumbnails"] = sorted(th, key=lambda t: t["label"])
    _pair_titles(pk)
    proj.save("paket.json", pk)
    return pk


VERTICAL_MAX_S = 178.0   # YouTube counts a vertical video as a Short only up to 3 minutes


def make_vertical(proj: Project, progress: Progress = _noop, llm=None) -> dict:
    """The vertical modes ("Yataydan dikey video çıkar", "Dikey video yükle"): the product IS a vertical video of at
    most 3 minutes. A cut that fits is turned vertical whole (face-tracked crop); a longer one is cut down to its
    strongest parts. Written as the project's main video (kurgu.json), so archive, package and the board use it."""
    res = make_shorts(proj, progress, llm, product=True)
    info = {"file": res["file"], "duration": res["duration"], "width": res["width"], "height": res["height"],
            "fps": proj.cfg.design.get("shorts", {}).get("fps", 30), "engine": "dikey",
            "engine_note": res.get("note"), "vertical": res,
            "director": {"options": dr_options(proj), "notes": [], "counts": {}, "sfx": 0},
            "graphics": {"callouts": [], "camfx": 0, "zoom_segments": 0, "method": res["pick"].get("method"),
                         "warning": res["pick"].get("warning")},
            "captions": True, "ok": res["duration"] <= VERTICAL_MAX_S + 2}
    proj.save("kurgu.json", info)
    return info


def make_shorts(proj: Project, progress: Progress = _noop, llm=None, product: bool = False) -> dict:
    """Step 6: a story Shorts that SUMMARISES the video: cold-open hook -> intro -> best moments -> conclusion,
    zoom transitions + whoosh between parts, top headline, animated captions, end card
    "Videonun uzun versiyonu YouTube kanalımda", colour + retouch, mastered audio.
    product=True (vertical modes): the whole cut when it fits in 3 minutes, else a <= 3 minute cut, no end card."""
    from .audio.master import builtin_sfx
    from .editing.plan import resnap
    from .editing.shorts import is_cta, is_junk, llm_summary, plan_summary, sentences, story_summary, validate_summary
    from .motion.captions import build_ass
    from .motion.compose import draw_headline, draw_outro
    from .motion.render import Overlay, Piece, render
    from .motion.vertical import face_centers, smooth_centers
    from .vision.faces import backend

    pl = proj.load("edit_plan.json")
    if not pl:
        raise RuntimeError("Önce kesim yapılmalı")
    pl = resnap(pl, _out_fps(pl))
    pl = polish_transcript(proj, pl, None)   # the same spelling fixes as the main video
    scfg = proj.cfg.design.get("shorts", {})
    lo, hi = scfg.get("summary_min_s", 50), scfg.get("summary_max_s", 90)
    sents = sentences(pl["transcript_cut"])
    whole = product and bool(sents) and sents[-1]["end"] - sents[0]["start"] <= VERTICAL_MAX_S - 1
    if product and not whole:
        lo, hi = 110, VERTICAL_MAX_S - 6
    progress(0.02, "Dikey video kurgulanıyor" if product else "Shorts özeti kurgulanıyor")
    why = None
    if whole:
        llm = llm or False     # nothing to choose: the whole cut is used
    if llm is None:
        llm, why = _llm_or_none(proj, lambda f, m: progress(0.03, m))
    pick = None
    if llm:
        old_t = llm.cfg.get("timeout_s") if hasattr(llm, "cfg") else None
        try:
            if hasattr(llm, "cfg"):
                llm.cfg["timeout_s"] = 150   # a slow model must not hold the Shorts for many minutes
            pick = llm_summary(llm, sents, lo, hi)
            # never a call to action or a garbled line, and a real story needs at least 3 parts
            n = len(sents)
            pick["clips"] = [c for c in pick.get("clips", []) if 0 <= int(c.get("start", -1)) < n and
                             not any(is_cta(sents[i]) or is_junk(sents[i])
                                     for i in range(int(c["start"]), min(n, int(c.get("end", c["start"])) + 1)))]
            h = pick.get("hook")
            if h is not None and (not (0 <= int(h) < n) or is_cta(sents[int(h)])):
                pick["hook"] = None
            if len(pick["clips"]) < 3:
                why = "Dil modelinin özeti zayıftı (3'ten az kullanılabilir parça); hikâye kurallarıyla kuruldu"
                pick = None
        except Exception as e:  # noqa: BLE001
            why = f"Dil modeli özet çıkaramadı ({str(e)[:120]}); hikâye kurallarıyla kuruldu"
            pick = None
        finally:
            if hasattr(llm, "cfg") and old_t is not None:
                llm.cfg["timeout_s"] = old_t
    if whole:
        pick = {"hook": None, "clips": [{"start": 0, "end": len(sents) - 1, "role": "tam"}], "headline": None,
                "reason": "Kesilmiş video 3 dakikaya sığıyor: tamamı dikey", "method": "tamamı"}
    else:
        pick = validate_summary(pick or story_summary(sents, lo, hi, target_s=150 if product else 60), sents, lo, hi)
    if why:
        pick["warning"] = why
    sp = plan_summary(sents, pick, pl["segments"])
    while product and sp["duration"] > VERTICAL_MAX_S and len(pick["clips"]) > 1:
        # never over 3 minutes: drop the longest middle part (the opening and the payoff stay)
        mid = pick["clips"][1:-1] or pick["clips"][:-1]
        longest = max(mid, key=lambda c: sents[c["end"]]["end"] - sents[c["start"]]["start"])
        pick["clips"] = [c for c in pick["clips"] if c is not longest]
        sp = plan_summary(sents, pick, pl["segments"])
    proj.log.write("shorts", "summary", pick.get("reason", ""), hook=pick.get("hook"), clips=pick["clips"],
                   method=pick.get("method"))
    progress(0.1, "Yüz takibi")
    centers = face_centers(sp["pieces"]) if backend() != "yok" else [None] * len(sp["pieces"])
    centers = smooth_centers(centers)
    out_dir = proj.dir / "cikti"
    out_dir.mkdir(exist_ok=True)
    W, H, fps = scfg.get("width", 1080), scfg.get("height", 1920), scfg.get("fps", 30)
    pieces = [Piece(pc["file"], pc["in"], pc["out"], cx=cx, kind=pc["kind"],
                    trans_in=0.18 if pc["first"] and pc["part"] > 0 else 0.0)
              for pc, cx in zip(sp["pieces"], centers)]
    outro_s = 0.0 if product else 3.0
    if outro_s:
        pieces.append(Piece(pieces[-1].file, 0.0, outro_s, kind="hold"))
    body = sp["duration"]
    fx = builtin_sfx()
    sfx = [{"t": max(0.0, t - 0.15), "file": fx["whoosh"], "gain_db": -16} for t in sp["cuts"]]
    if outro_s:
        sfx.append({"t": max(0.0, body - 0.9), "file": fx["riser"], "gain_db": -20})
        sfx.append({"t": body, "file": fx["impact"], "gain_db": -15})
    overlays = []
    headline = (pick.get("headline") or "").strip()
    if headline and len(headline.split()) <= 8:
        overlays.append(Overlay(0.15, min(3.8, body), draw_headline(headline, W, H), fade=0.25, anim="pop"))
        sfx.append({"t": 0.1, "file": fx["pop"], "gain_db": -18})
    if outro_s:
        overlays.append(Overlay(body + 0.05, body + outro_s, draw_outro(W, H), fade=0.25, anim="pop"))
    ass = out_dir / ("dikey_altyazi.ass" if product else "shorts_altyazi.ass")
    from .motion.titles import set_theme
    s_opts = dr_options(proj)
    s_acc = set_theme(str(s_opts.get("renk", "otomatik")), proj.name)
    ass.write_text(build_ass(sp["words"], W, H, font=proj.cfg.design.get("font", "Bahnschrift"), size=96, y_frac=0.68,
                             active_color=ass_color(s_acc), style=caption_style(s_opts, vertical=True)),
                   encoding="utf-8")
    rt, rt_note = _retoucher(proj)
    out = out_dir / ("dikey.mp4" if product else "shorts.mp4")
    info = render(pieces, str(out), W, H, fps, frame_fn=rt, ass_path=str(ass), vertical=True, overlays=overlays,
                  image=proj.cfg.design.get("image"),
                  loudness=scfg.get("loudness", "I=-14:TP=-1.5:LRA=11"), sfx=sfx,
                  speech=[(x["start"], x["end"]) for x in sp["words"]],
                  music=music_for(proj) if dr_options(proj).get("music", True) else None,
                  progress=lambda f, m: progress(0.15 + 0.83 * f, m))
    probe = ff.probe(out)
    res = {"file": str(out), "duration": round(probe.duration, 2), "width": probe.width, "height": probe.height,
           "pick": {k: pick.get(k) for k in ("hook", "clips", "headline", "reason", "title", "method", "warning", "duration")},
           "parts": sp["parts"], "hook_text": sents[pick["hook"]]["text"] if pick.get("hook") is not None else None,
           "start_text": sp["parts"][0]["text"] if sp["parts"] else "",
           "render": info, "face_note": None if backend() != "yok" else "Yüz modeli yok: kadraj ortalandı",
           "retouch_note": rt_note,
           "ok": probe.duration <= VERTICAL_MAX_S + 2 if product else 30 <= probe.duration <= 130,
           "note": ("Kesilmiş videonun tamamı dikey" if whole else
                    f"Kesilmiş video {fmt_min(sents[-1]['end'])} sürüyordu: en güçlü anlarından 3 dakikayı aşmayan bir "
                    "dikey kurgu yapıldı") if product else None}
    if product:
        proj.save("dikey.json", res)
        progress(1.0, "Dikey video hazır")
        return res
    proj.save("shorts.json", res)
    progress(1.0, "Shorts hazır")
    return res


def fmt_min(s: float) -> str:
    return f"{int(s) // 60} dk {int(s) % 60} sn"


def dr_options(proj: Project) -> dict:
    from .editing.director import options_of
    return options_of(proj.settings)


def safe_name(title: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip().rstrip(".")
    s = re.sub(r"\s+", " ", s)
    return (s or "Video")[:90]


def make_youtube_text(proj: Project, progress: Progress = _noop, llm=None) -> dict:
    """3 titles + 1 description + Shorts title/description written by the local LLM with the Humanizer rules,
    into cikti/youtube_metin.txt. Without a language model nothing is written (no fake text)."""
    from .editing import humanize as hz

    pl = proj.load("edit_plan.json")
    if not pl:
        raise RuntimeError("Önce kesim yapılmalı")
    own = llm is None
    if own:
        llm, why = _llm_or_none(proj, progress)
        if llm is None:
            raise RuntimeError(f"Başlık ve açıklama için dil modeli gerekli: {why}")
    try:
        progress(0.1, "Başlık ve açıklama yazılıyor (taslak + insan gibi yazma geçişi)")
        words = pl["transcript_cut"]
        dur = max((w["t_end"] for w in words), default=0.0)
        sh = proj.load("shorts.json") or {}
        hint = (sh.get("pick") or {}).get("headline") or (sh.get("pick") or {}).get("title")
        res = hz.write(llm, words, dur, hint)
    finally:
        if own:
            llm.stop()
    out = proj.dir / "cikti" / "youtube_metin.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(hz.text_file(res), encoding="utf-8")
    res["file"] = str(out)
    proj.save("youtube_metin.json", res)
    proj.log.write("package", "humanizer", " | ".join(t["title"] for t in res["titles"])
                   + (f" | notlar: {'; '.join(res['notes'])}" if res["notes"] else ""))
    progress(1.0, "Başlık ve açıklama hazır")
    return res


def make_archive(proj: Project, progress: Progress = _noop) -> dict:
    """Step 7: one folder per video holding everything for the YouTube upload."""
    import shutil

    km, sh, pk = proj.load("kurgu.json"), proj.load("shorts.json"), proj.load("paket.json") or {}
    if not km:
        raise RuntimeError("Önce video hazırlanmalı")
    title = proj.settings.get("title") or proj.name
    motor = str(((km.get("director") or {}).get("options") or {}).get("motor", "kurgu"))
    akey = "archive" if motor == "kurgu" else f"archive_{motor}"   # each engine gets its own folder
    # everything of one video lives in ONE folder (Videolarım\<project>): the upload files go into its
    # "Yüklenecekler" subfolder, next to the work files - deleting the project folder removes it all (23.10.2026)
    d = proj.dir / ("Yüklenecekler" + {"karma": " (Karma)", "hyperframes": " (HyperFrames)"}.get(motor, ""
                                       if motor == "kurgu" else f" ({motor})"))
    d.mkdir(parents=True, exist_ok=True)
    progress(0.1, "Dosyalar arşive kopyalanıyor")
    files = []

    def put(src, name):
        dst = d / name
        st = Path(src).stat()
        if not (dst.exists() and dst.stat().st_size == st.st_size and dst.stat().st_mtime >= st.st_mtime):
            if dst.exists():
                dst.unlink()   # an older render of the same file in our own upload folder
            try:   # same folder tree: a hard link (the 10 GB video is not stored twice); else a copy
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        files.append(str(dst))

    put(km["file"], "1 - Dikey video (Shorts).mp4" if km.get("engine") == "dikey" else "1 - Video (YouTube).mp4")
    if sh:
        put(sh["file"], "2 - Shorts.mp4")
    for i, f in enumerate(pk.get("frames", []), 1):
        if Path(f["file"]).exists():
            put(f["file"], f"3 - Kapak için kare {i}.jpg")
    if pk.get("transcript_file") and Path(pk["transcript_file"]).exists():
        put(pk["transcript_file"], "4 - Transkript (zamanlı).txt")
    if pk.get("transcript_plain") and Path(pk["transcript_plain"]).exists():
        put(pk["transcript_plain"], "4 - Transkript.txt")
    if sh and (sh.get("pick") or {}).get("title"):
        (d / "5 - Shorts başlığı.txt").write_text(sh["pick"]["title"] + "\n", encoding="utf-8")
        files.append(str(d / "5 - Shorts başlığı.txt"))
    yt = proj.dir / "cikti" / "youtube_metin.txt"
    if yt.exists():
        put(yt, "5 - Başlık ve açıklama.txt")
    pr_proj = (km.get("premiere") or {}).get("project")
    if pr_proj and Path(pr_proj).exists():
        # a pointer, not a copy: the Premiere project references media in the project folder
        (d / "6 - Premiere projesi.txt").write_text(f"Premiere projesi:\n{pr_proj}\n", encoding="utf-8")
        files.append(str(d / "6 - Premiere projesi.txt"))
    proj.save_settings({akey: str(d)})
    res = {"folder": str(d), "files": files, "title": title}
    proj.save("arsiv.json", res)
    progress(1.0, "Arşiv klasörü hazır")
    return res
