"""Measurements of a finished edit for the review board - one section per agent, each with numbers, flagged
moments and an evidence sheet (a few labelled frames) the agent can look at.

  hikaye    - pace and audience: words per minute, the first 30 s, long stretches where nothing on screen changes
  grafik    - on-screen text: how much, how long, reading speed, safe margins, text over the face
  animasyon - movement: how long text stands still after it arrives, low-motion stretches, punch-ins
  gecis     - cuts: shot lengths, jump cuts that keep the same framing, transition variety, sound on the cut
  ses       - loudness (EBU R128): integrated / true peak / range, loud spikes, quiet speech, effects over the voice
  renk      - picture: exposure, clipping, colour and brightness differences between shots, sharpness

The video is decoded ONCE at 15 fps / 320 px (motion, cuts, text stillness, colour) and its audio once through
ffmpeg's ebur128 filter. What the render did (pieces, overlays, effects) comes from cikti/kurgu_manifest.json; for
older renders it is rebuilt from edit_plan.json and the decision log, and the report says what is missing.
Output: projeler/<ad>/kurul/olcum.json + kurul/kanit_<agent>.jpg.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from ..integrations import ffmpeg as ff

AGENTS = ("hikaye", "grafik", "animasyon", "gecis", "ses", "renk")
FPS = 15.0
W, H = 320, 180
EDGE_STILL = 0.25            # changed text pixels / text pixels below which the text counts as standing still
CPS_MAX = 17              # on-screen text reading speed, characters per second (Netflix timed text guide)
SAFE = 0.05               # graphics-safe margin per edge (EBU R 95)
JUMP_SIM = 0.90             # likeness of 80x45 frames across a cut above which the framing is "the same"


def _noop(f, m):
    pass


# --- what the render did -----------------------------------------------------------------------------------------

def manifest_of(ctx: dict) -> dict:
    """Written by the pipeline after a render: pieces on the output timeline, overlays with their real boxes,
    sound effects. Small and JSON-safe."""
    t, pieces = 0.0, []
    for p in ctx["pieces"]:
        pieces.append({"t0": round(t, 3), "t1": round(t + p.dur, 3), "file": p.file, "zoom": round(p.zoom, 3),
                       "trans": p.trans if p.trans_in else None, "trans_in": round(p.trans_in, 3),
                       "speed": round(p.speed, 3), "kind": p.kind})
        t += p.dur
    overlays = []
    for ov in ctx["overlays"]:
        box, label = list(ov.box), ov.anim
        draw = ov.meta.get("draw") if ov.meta else None
        if draw is not None:
            label = type(getattr(draw, "__self__", None)).__name__
            try:      # editorial objects cover the whole frame: their real box is the drawn part
                img = ov.meta["static"]()
                ys, xs = np.nonzero(img[..., 3]) if img.shape[-1] == 4 else (np.array([]), np.array([]))
                if len(xs):
                    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            except Exception:  # noqa: BLE001 - keep the full-frame box
                pass
        overlays.append({"t0": round(ov.t0, 3), "t1": round(ov.t1, 3), "kind": label, "box": box})
    sfx = [{"t": round(c["t"], 3), "name": Path(str(c.get("file", ""))).stem, "gain_db": c.get("gain_db")}
           for c in ctx["d"]["plan"]["sfx"]]
    return {"version": 1, "w": ctx["w"], "h": ctx["h"], "fps": ctx["fps"], "pieces": pieces,
            "overlays": overlays, "sfx": sfx}


def _rebuilt(proj, pl: dict, w: int, h: int) -> tuple[dict, list[str]]:
    """Older renders without a manifest: pieces from the plan, text windows from the decision log of the last run."""
    missing = ["geçiş türleri ve zoom'lar", "ses efektlerinin zamanları", "yazıların gerçek kutuları"]
    pieces = [{"t0": s["timeline_start"], "t1": s["timeline_end"], "file": s["file"], "zoom": None, "trans": None,
               "trans_in": 0.0, "speed": 1.0, "kind": "main"} for s in pl["segments"]]
    recs = []
    log = proj.dir / "logs" / "decisions.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    gfx = [r for r in recs if r.get("stage") == "graphics" and "start" in r and "end" in r]
    last = max((r["t"] for r in gfx), default=None)
    overlays = []
    if last:
        t_last = time.mktime(time.strptime(last, "%Y-%m-%d %H:%M:%S"))
        for r in gfx:   # the last render's entries are written within a few seconds of each other
            if t_last - time.mktime(time.strptime(r["t"], "%Y-%m-%d %H:%M:%S")) <= 120:
                # the side is unknown (the camera slides away from the face): look at the whole frame
                overlays.append({"t0": r["start"], "t1": r["end"], "kind": r["action"], "box": [0, 0, w, h],
                                 "text": r.get("reason", ""), "box_guess": True})
    return {"version": 0, "w": w, "h": h, "pieces": pieces, "overlays": overlays, "sfx": []}, missing


# --- one pass over the picture -------------------------------------------------------------------------------------

def _scan_video(path: str, dur: float, overlays: list[dict], src_w: int, src_h: int, progress,
                keep: set[int] | None = None) -> dict:
    """motion per frame, text stillness inside each overlay, and 160x90 grey frames kept at the indices asked for."""
    n_exp = int(dur * FPS) + 2
    motion = np.zeros(n_exp, np.float32)
    hist: list[np.ndarray | None] = [None] * n_exp
    sx, sy = W / src_w, H / src_h
    wins = []
    for ov in overlays:
        x0, y0, x1, y1 = ov["box"]
        wins.append({"a": int(ov["t0"] * FPS), "b": int(ov["t1"] * FPS),
                     "r": (max(0, int(x0 * sx)), max(0, int(y0 * sy)), min(W, int(np.ceil(x1 * sx))), min(H, int(np.ceil(y1 * sy)))),
                     "prev": None, "change": []})
    cmd = [ff.ffmpeg_bin(), "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=ff._CREATE_NO_WINDOW)
    size, prev, i = W * H * 3, None, 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            rgb = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if i >= n_exp:
                motion = np.append(motion, np.zeros(64, np.float32))
                hist += [None] * 64
                n_exp += 64
            if prev is not None:
                motion[i] = float(cv2.absdiff(gray, prev).mean())
            if keep and i in keep:
                hist[i] = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            for wn in wins:
                if wn["a"] <= i <= wn["b"]:
                    x0, y0, x1, y1 = wn["r"]
                    # the text itself: bright strokes on a dark ground (the darkened panel / dark set / band). A bright
                    # wall or a waving hand is bright on bright, so the moving video does not count as text movement
                    if x1 > x0 and y1 > y0:
                        g = gray[y0:y1, x0:x1]
                        e = (g > 200) & (cv2.blur(g, (15, 15)) < 120)
                    else:
                        e = None
                    if e is not None and wn["prev"] is not None:
                        ink = max(1, int(e.sum()), int(wn["prev"].sum()))
                        wn["change"].append(float((e != wn["prev"]).sum()) / ink if ink > 30 else 0.0)
                    wn["prev"] = e
            prev = gray
            i += 1
            if i % 900 == 0:
                progress(min(0.6, 0.6 * i / max(1, n_exp)), f"Video okunuyor {i / FPS / 60:.1f}/{dur / 60:.1f} dk")
    finally:
        proc.stdout.close()
        proc.wait()
    return {"n": i, "motion": motion[:i], "hist": hist[:i], "wins": wins}


def _grab(path: str, t: float, width: int = 640) -> np.ndarray | None:
    cmd = [ff.ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", f"scale={width}:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True, creationflags=ff._CREATE_NO_WINDOW)
    hh = len(r.stdout) // (width * 3)
    return np.frombuffer(r.stdout[:width * 3 * hh], np.uint8).reshape(hh, width, 3) if hh else None


def _framing(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(likeness, scale) of two grey frames across a cut. b is also tried punched in (1.1-1.3x, centre crop) and
    a punched in towards b: a plain jump cut matches best at 1.0, a zoom cut at another scale."""
    def ncc(x, y):
        return float(cv2.matchTemplate(x.astype(np.float32), y.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0])

    def zoom(img, z):
        h, w = img.shape
        ch, cw = int(h / z), int(w / z)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
    best = (ncc(a, b), 1.0)
    for z in (1.1, 1.2, 1.3):
        for s in (ncc(zoom(a, z), b), ncc(a, zoom(b, z))):
            if s > best[0] + 0.02:
                best = (s, z)
    return best


# --- the audio -------------------------------------------------------------------------------------------------

_EBU_LINE = re.compile(r"t:\s*([\d.]+)\s+TARGET.*?M:\s*(-?[\d.]+|-inf)\s+S:\s*(-?[\d.]+|-inf)")


def _scan_audio(path: str) -> dict:
    cmd = [ff.ffmpeg_bin(), "-nostats", "-hide_banner", "-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=ff._CREATE_NO_WINDOW)
    err = r.stderr
    ts, mom, short = [], [], []
    for m in _EBU_LINE.finditer(err):
        ts.append(float(m.group(1)))
        mom.append(-120.0 if m.group(2) == "-inf" else float(m.group(2)))
        short.append(-120.0 if m.group(3) == "-inf" else float(m.group(3)))
    summ = err[err.rfind("Summary:"):] if "Summary:" in err else ""

    def grab(pat):
        m = re.search(pat, summ)
        return float(m.group(1)) if m else None
    return {"I": grab(r"I:\s*(-?[\d.]+) LUFS"), "LRA": grab(r"LRA:\s*([\d.]+) LU"),
            "TP": grab(r"Peak:\s*(-?[\d.]+) dBFS"), "t": np.array(ts), "M": np.array(mom), "S": np.array(short)}


# --- helpers ---------------------------------------------------------------------------------------------------

def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True, at least min_len long."""
    out, a = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and a is None:
            a = i
        elif not v and a is not None:
            if i - a >= min_len:
                out.append((a, i))
            a = None
    return out


def _speech_mask(words: list[dict], t: np.ndarray) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for w in words:
        m |= (t >= w["t_start"]) & (t <= w["t_end"])
    return m


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _sheet(frames: list[tuple[str, np.ndarray]], cols: int = 3, tile: tuple[int, int] = (512, 288)) -> np.ndarray | None:
    frames = [(lab, f) for lab, f in frames if f is not None][:6]
    if not frames:
        return None
    tw, th = tile if len(frames) > 1 else (frames[0][1].shape[1], frames[0][1].shape[0] + 30)
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * th, min(cols, len(frames)) * tw, 3), 16, np.uint8)
    for k, (lab, f) in enumerate(frames):
        t = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tw, 30), (0, 0, 0), -1)
        cv2.putText(t, lab[:44], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 80), 1, cv2.LINE_AA)
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return sheet


def _flag(t: float, kind: str, detail: str, **data) -> dict:
    return dict({"t": round(t, 2), "zaman": _fmt(t), "tur": kind, "aciklama": detail}, **data)


# --- the report ----------------------------------------------------------------------------------------------

def measure(proj, progress=_noop) -> dict:
    info = proj.load("kurgu.json")
    if not info or not Path(info.get("file", "")).exists():
        raise RuntimeError("Önce video oluşturulmalı (kurgu.json / kurgu.mp4 yok)")
    pl = proj.load("edit_plan.json") or {}
    video = info["file"]
    dur = float(info["duration"])
    probe = ff.probe(video)
    vw, vh = probe.width or 1920, probe.height or 1080
    man = proj.load("kurgu_manifest.json")
    missing: list[str] = []
    if not man:
        man, missing = _rebuilt(proj, pl, vw, vh)
    words = [w for w in pl.get("transcript_cut", []) if w.get("t_end", 0) <= dur + 1]
    out_dir = proj.dir / "kurul"
    out_dir.mkdir(exist_ok=True)

    progress(0.0, "Video okunuyor")
    cut_t = [p["t0"] for p in man["pieces"][1:] if 0.2 < p["t0"] < dur - 0.2]
    keep = {int(round(c * FPS)) + d for c in cut_t for d in range(-6, 7)}
    scan = _scan_video(video, dur, man["overlays"], vw, vh, progress, keep)
    progress(0.62, "Ses ölçülüyor (EBU R128)")
    au = _scan_audio(video)
    progress(0.75, "Ölçümler değerlendiriliyor")
    n = scan["n"]
    tt = np.arange(n) / FPS
    motion = scan["motion"]
    cuts = [p["t0"] for p in man["pieces"][1:] if 0.2 < p["t0"] < dur - 0.2]
    shot = np.diff([0.0] + [p["t1"] for p in man["pieces"]])
    ovs = man["overlays"]
    rep: dict = {"video": video, "duration_s": round(dur, 2), "size": [vw, vh], "manifest": bool(man.get("version")),
                 "missing": missing, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    grabs: dict[str, list[tuple[str, float]]] = {a: [] for a in AGENTS}

    # hikaye ----------------------------------------------------------------------------------------------
    flags = []
    # words per minute; a last part shorter than 20 s is left out and a shorter last minute is scaled to its length
    # (the first meeting read a 28 s ending as "50 words a minute" and asked to cut a non-existent slow ending)
    mins = max(1, int(np.ceil(dur / 60)))
    wpm = []
    for k in range(mins):
        secs = min(60.0, dur - k * 60)
        if secs >= 20 or k == 0:
            wpm.append(round(sum(1 for w in words if k * 60 <= w["t_start"] < (k + 1) * 60) * 60 / max(1.0, secs)))
    first = [w for w in words if w["t_start"] < 30]
    first_text = " ".join(w["word"] for w in first)
    events = sorted(set([0.0, dur] + cuts + [o["t0"] for o in ovs] + [o["t1"] for o in ovs] +
                        [p["t0"] for p in man["pieces"] if (p.get("zoom") or 1) > 1.01 or p.get("trans")]))
    gaps = sorted(((b - a, a) for a, b in zip(events, events[1:])), reverse=True)
    for g, a in gaps[:5]:
        if g >= 25:
            flags.append(_flag(a, "durgun", f"{g:.0f} sn boyunca görüntüde hiçbir değişiklik yok (kesim, yazı, zoom)", sure=round(g, 1)))
            grabs["hikaye"].append((f"{_fmt(a + g / 2)} {g:.0f} sn degisiklik yok", a + g / 2))
    pauses = [(b["t_start"] - a["t_end"], a["t_end"]) for a, b in zip(words, words[1:]) if b["t_start"] - a["t_end"] > 1.2]
    for p, a in sorted(pauses, reverse=True)[:3]:
        flags.append(_flag(a, "bosluk", f"konuşmada {p:.1f} sn sessizlik kalmış", sure=round(p, 2)))
    slow = [k for k, v in enumerate(wpm) if v < 0.6 * np.median(wpm)]
    for k in slow[:3]:
        flags.append(_flag(k * 60.0, "tempo", f"{k + 1}. dakikada konuşma yavaş ({wpm[k]} kelime/dk, ortanca {np.median(wpm):.0f})"))
    grabs["hikaye"] = [(f"{_fmt(t)} acilis", t) for t in (0.5, 6.0, 15.0, 28.0)] + grabs["hikaye"]
    rep["hikaye"] = {"metrics": {"sure_dk": round(dur / 60, 1), "kelime_dk_ortanca": float(np.median(wpm)),
                                 "kelime_dk": wpm, "ilk_30sn_kelime": len(first), "ilk_30sn_soru_var": "?" in first_text,
                                 "ilk_30sn_metin": first_text[:400],
                                 "en_uzun_degisimsiz_sn": round(gaps[0][0], 1) if gaps else 0.0,
                                 "bolum_basligi": sum(1 for o in ovs if o["kind"] in ("topic", "Topic", "card"))},
                     "flags": flags}

    # grafik -----------------------------------------------------------------------------------------------
    flags = []
    for o in ovs:
        d_ = o["t1"] - o["t0"]
        x0, y0, x1, y1 = o["box"]
        txt = re.sub(r"^[^:]*:\s*", "", o.get("text", ""))
        txt = re.sub(r"\[[^\]]*\]|\(simge [^)]*\)|\[simge [^\]]*\]", "", txt).replace(" / ", " ").strip()
        nc = len(txt)
        # reading speed in characters per second (Netflix style guide: ~17 for adults; bilgi.py)
        if nc and d_ > 0 and nc / d_ > CPS_MAX:
            flags.append(_flag(o["t0"], "okuma", f"{nc} karakter {d_:.1f} sn'de okunamaz ({nc / d_:.0f} karakter/sn, sınır {CPS_MAX})"))
        # graphics-safe area: 5 % from every edge (EBU R 95)
        if not o.get("box_guess") and (x0 < SAFE * vw or y0 < SAFE * vh or x1 > (1 - SAFE) * vw or y1 > (1 - SAFE) * vh):
            flags.append(_flag(o["t0"], "kenar", "yazı grafik güvenli alanının dışında (kenarlardan %5, EBU R 95)"))
        grabs["grafik"].append((f"{_fmt((o['t0'] + o['t1']) / 2)} {o['kind']}", min(o["t1"] - 0.3, o["t0"] + 1.8)))
    per_min = len(ovs) / max(1e-6, dur / 60)
    if per_min < 0.5:
        flags.append(_flag(0.0, "seyrek", f"dakikada {per_min:.2f} ekran yazısı - {dur / 60:.0f} dk'da yalnız {len(ovs)} yazı"))
    rep["grafik"] = {"metrics": {"yazi_sayisi": len(ovs), "dakikada": round(per_min, 2),
                                 "yazilar": [{"t0": o["t0"], "t1": o["t1"], "tur": o["kind"],
                                              "metin": re.sub(r"^[^:]*:\s*", "", o.get("text", ""))[:80]} for o in ovs]},
                     "flags": flags}

    # animasyon -------------------------------------------------------------------------------------------
    flags, anim = [], []
    for o, wn in zip(ovs, scan["wins"]):
        ch = np.array(wn["change"], np.float32) if wn["change"] else np.zeros(1, np.float32)
        if len(ch) >= 3:      # one-frame flickers (compression, a hand passing) do not break a still stretch
            ch = cv2.medianBlur(ch.reshape(1, -1), 3).ravel() if len(ch) < 512 else ch
        moving = ch > EDGE_STILL
        still = _runs(~moving, int(0.5 * FPS))
        hold = max((b - a for a, b in still), default=0) / FPS
        first_still = next((a for a, b in still), len(ch)) / FPS
        d_ = o["t1"] - o["t0"]
        anim.append({"t0": o["t0"], "sure": round(d_, 2), "giris_sn": round(first_still, 2),
                     "en_uzun_donuk_sn": round(hold, 2), "donuk_orani": round(float((~moving).mean()), 2)})
        if 1.0 < first_still < d_ - 0.5:      # entrances beyond ~0.4 s feel slow (Material motion; bilgi.py)
            flags.append(_flag(o["t0"], "agir_giris", f"yazının girişi {first_still:.1f} sn sürüyor (hareket kılavuzlarına göre ağır)"))
        if hold >= 2.0:
            a_ = next(a for a, b in still if (b - a) / FPS == hold)
            flags.append(_flag(o["t0"], "donuk", f"yazı geldikten sonra {hold:.1f} sn hiç kıpırdamıyor "
                               f"(giriş {first_still:.1f} sn'de bitiyor, {d_:.1f} sn ekranda)", sure=round(hold, 2)))
            grabs["animasyon"] += [(f"{_fmt(o['t0'] + 0.2)} giris", o["t0"] + 0.2),
                                   (f"{_fmt(o['t0'] + first_still)} giris bitti", o["t0"] + first_still),
                                   (f"{_fmt(o['t0'] + (a_ + hold * FPS / 2) / FPS)} donuk", o["t0"] + a_ / FPS + hold / 2)]
    low = motion < max(0.6, float(np.percentile(motion, 20)))
    dead = [(a / FPS, (b - a) / FPS) for a, b in _runs(low, int(8 * FPS))]
    for a, ln in sorted(dead, key=lambda x: -x[1])[:3]:
        flags.append(_flag(a, "hareketsiz", f"{ln:.0f} sn boyunca görüntü neredeyse hiç hareket etmiyor"))
    zooms = sum(1 for p in man["pieces"] if (p.get("zoom") or 1) > 1.01)
    rep["animasyon"] = {"metrics": {"yazilar": anim, "hareket_ortanca": round(float(np.median(motion)), 2) if n else 0,
                                    "zoom_parca": zooms if man.get("version") else None},
                        "flags": flags}

    # gecis -----------------------------------------------------------------------------------------------
    flags, jumps = [], 0
    hist = scan["hist"]
    for c in cuts:
        i0 = int(round(c * FPS))
        # the real cut: the biggest picture change within +-0.3 s of the planned time (timelines drift a little)
        cand = [k for k in range(i0 - 4, i0 + 5) if 1 <= k < n and hist[k - 1] is not None and hist[k] is not None]
        if not cand:
            continue
        i = max(cand, key=lambda k: motion[k])
        if motion[i] < 1.5:          # no visible change at all: a seamless cut, nothing to see
            continue
        if True:
            sim, scale = _framing(hist[i - 1], hist[i])
            piece = next((p for p in man["pieces"] if abs(p["t0"] - c) < 0.05), {})
            if sim >= JUMP_SIM and scale == 1.0 and not piece.get("trans"):
                jumps += 1
                if jumps <= 40:
                    flags.append(_flag(i / FPS, "ziplama", f"aynı kadrajda kesim (benzerlik {sim:.2f}), zoom ya da geçiş görünmüyor",
                                       benzerlik=round(sim, 3)))
    worst = sorted((f for f in flags if f["tur"] == "ziplama"), key=lambda f: -f["benzerlik"])[:3]
    for f in worst:
        grabs["gecis"] += [(f"{f['zaman']} kesim oncesi", f["t"] - 0.1), (f"{f['zaman']} kesim sonrasi", f["t"] + 0.05)]
    short_shots = int((shot < 0.6).sum())
    trans = {}
    for p in man["pieces"]:
        if p.get("trans"):
            trans[p["trans"]] = trans.get(p["trans"], 0) + 1
    sfx_t = [s["t"] for s in man["sfx"]]
    off = [min((abs(s - c) for s in sfx_t), default=None) for c in cuts if any(abs(s - c) < 0.6 for s in sfx_t)]
    rep["gecis"] = {"metrics": {"kesim": len(cuts), "dakikada_kesim": round(len(cuts) / max(1e-6, dur / 60), 1),
                                "cekim_ortanca_sn": round(float(np.median(shot)), 2) if len(shot) else 0,
                                "cekim_en_uzun_sn": round(float(shot.max()), 1) if len(shot) else 0,
                                "kisa_cekim_0_6sn": short_shots, "ayni_kadraj_kesim": jumps,
                                "ayni_kadraj_orani": round(jumps / max(1, len(cuts)), 2),
                                "gecis_turleri": trans if man.get("version") else None,
                                "efekt_kesim_farki_ms": round(1000 * float(np.median(off)), 0) if off else None},
                    "flags": flags[:12]}

    # ses -------------------------------------------------------------------------------------------------
    flags = []
    I, TP = au["I"], au["TP"]
    if I is not None and abs(I + 14) > 1.0:
        flags.append(_flag(0.0, "seviye", f"ortalama ses {I:.1f} LUFS (YouTube hedefi -14)"))
    if TP is not None and TP > -1.0:
        flags.append(_flag(0.0, "tepe", f"gerçek tepe {TP:.1f} dBTP (en çok -1)"))
    if len(au["t"]) and I is not None:
        S, Mo, at = au["S"], au["M"], au["t"]
        sp = _speech_mask(words, at)
        for a, b in _runs(S > I + 7, 8)[:5]:
            pk = float(S[a:b].max())
            flags.append(_flag(at[a], "patlama", f"{(b - a) / 10:.1f} sn boyunca ortalamadan {pk - I:.0f} LU yüksek"))
        for a, b in _runs(sp & (S < I - 10) & (S > -70), 20)[:5]:
            flags.append(_flag(at[a], "kisik", f"konuşma {(b - a) / 10:.1f} sn ortalamadan {I - float(S[a:b].mean()):.0f} LU kısık"))
        for s in man["sfx"]:
            k0, k1 = np.searchsorted(at, s["t"]), np.searchsorted(at, s["t"] + 0.5)
            kb = np.searchsorted(at, s["t"] - 3.0)
            ref = Mo[kb:k0][sp[kb:k0]] if k0 > kb else np.array([])
            if k1 > k0 and len(ref) > 3:
                over = float(Mo[k0:k1].max() - np.median(ref))
                if over > 3.0:
                    flags.append(_flag(s["t"], "efekt", f"'{s['name']}' efekti konuşmadan {over:.1f} LU yüksek", lu=round(over, 1)))
    rep["ses"] = {"metrics": {"I_LUFS": I, "LRA_LU": au["LRA"], "TP_dBTP": TP, "efekt_sayisi": len(man["sfx"]) if man.get("version") else None},
                  "flags": flags[:15]}

    # renk ------------------------------------------------------------------------------------------------
    flags, shots = [], []
    mids = [((p["t0"] + p["t1"]) / 2, p) for p in man["pieces"] if p["t1"] - p["t0"] >= 1.0]
    step = max(1, len(mids) // 40)
    for tm, p in mids[::step]:
        f = _grab(video, tm, 320)
        if f is None:
            continue
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        shots.append({"t": round(tm, 2), "L": float(lab[:, 0].mean() * 100 / 255), "a": float(lab[:, 1].mean() - 128),
                      "b": float(lab[:, 2].mean() - 128), "patlak": float((g > 250).mean()), "karanlik": float((g < 6).mean()),
                      "netlik": float(cv2.Laplacian(g, cv2.CV_64F).var())})
    if shots:
        mL = float(np.median([s["L"] for s in shots]))
        ma, mb = float(np.median([s["a"] for s in shots])), float(np.median([s["b"] for s in shots]))
        mS = float(np.median([s["netlik"] for s in shots]))
        for s in shots:
            dE = float(np.hypot(s["a"] - ma, s["b"] - mb))
            if abs(s["L"] - mL) > 8 or dE > 5:
                flags.append(_flag(s["t"], "renk_farki", f"çekim diğerlerinden farklı: parlaklık {s['L'] - mL:+.0f}, renk farkı {dE:.1f}"))
            if s["patlak"] > 0.03:
                flags.append(_flag(s["t"], "patlak", f"görüntünün %{100 * s['patlak']:.0f}'i bembeyaz (patlamış ışık)"))
            if s["netlik"] < 0.35 * mS:
                flags.append(_flag(s["t"], "bulanik", "çekim diğerlerinden belirgin bulanık"))
        grabs["renk"] = [(f"{_fmt(s['t'])} ortanca", s["t"]) for s in sorted(shots, key=lambda s: abs(s["L"] - mL))[:1]]
        grabs["renk"] += [(f"{f['zaman']} {f['tur']}", f["t"]) for f in flags[:5]]
    rep["renk"] = {"metrics": {"cekim_olculen": len(shots),
                               "parlaklik_ortanca": round(float(np.median([s["L"] for s in shots])), 1) if shots else None,
                               "parlaklik_aralik": [round(min(s["L"] for s in shots), 1), round(max(s["L"] for s in shots), 1)] if shots else None},
                   "flags": flags[:10]}

    # evidence sheets ---------------------------------------------------------------------------------------
    progress(0.85, "Kanıt kareleri hazırlanıyor")
    for a in AGENTS:
        frames = [(lab, _grab(video, t)) for lab, t in grabs[a][:6]]
        if a == "ses" and len(au["t"]):
            frames = [("ses seviyesi (S) - kirmizi: -14 LUFS", _loud_chart(au["t"], au["S"], I))]
        sheet = _sheet(frames)
        if sheet is not None:
            p = out_dir / f"kanit_{a}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            rep[a]["kanit"] = str(p)
        rep[a]["n_flags"] = len(rep[a]["flags"])
    (out_dir / "olcum.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    progress(1.0, "Ölçüm bitti")
    return rep


def _loud_chart(t: np.ndarray, S: np.ndarray, I: float | None, w: int = 1536, h: int = 432) -> np.ndarray:
    img = np.full((h, w, 3), 18, np.uint8)

    def y(v):
        return int(h - 20 - (np.clip(v, -40, 0) + 40) / 40 * (h - 40))
    for db in (-10, -20, -30):
        cv2.line(img, (0, y(db)), (w, y(db)), (50, 50, 50), 1)
        cv2.putText(img, str(db), (4, y(db) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    cv2.line(img, (0, y(-14)), (w, y(-14)), (220, 60, 60), 1)
    if len(t):
        xs = (t / max(1e-6, t[-1]) * (w - 1)).astype(int)
        pts = np.stack([xs, [y(v) for v in S]], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (90, 200, 255), 1, cv2.LINE_AA)
    return img
