"""Full edit inside Premiere Pro, fully automatic: open Premiere (if closed), open/create the video's own project,
build the timeline from the director plan, export, close Premiere again.

Track layout
  V1 camera (retouched copy, punch-in zooms as Motion keyframes)   A1 voice
  V2 screen recording (cursor-follow keyframes) / b-roll            A2, A3 cleared (linked audio of V2/V3)
  V3 camera in the speaker box during screen parts                  A4 sound effects
  V4 speaker box border
  V5 text callouts, full-screen cards, subscribe reminder
Captions (word highlight) and loudness (-14 LUFS) are applied to the exported file by ffmpeg, so the look matches
the preview engine exactly.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from . import ffmpeg as ff
from .premiere import BridgeNotRunning, Premiere, PremiereError, js_str, js_val
from ..config import replace_file

BATCH = 30
NV, NA = 6, 6


class PremiereUnavailable(PremiereError):
    pass


# ------------------------------------------------------------------ launch / project ---------------------------
KNOWN_EXE = Path(__file__).resolve().parents[2] / "premiere_yolu.txt"   # remembered Premiere.exe path


def premiere_exe() -> str | None:
    cands = []
    if KNOWN_EXE.exists():
        cands.append(KNOWN_EXE.read_text(encoding="utf-8").strip())
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Adobe Premiere Pro.exe") as k:
            cands.append(winreg.QueryValue(k, None))
    except Exception:  # noqa: BLE001
        pass
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files") + r"\Adobe", r"C:\Premiere Pro",
                 r"C:\Adobe", r"D:\Adobe"):
        cands += sorted(glob.glob(os.path.join(base, "Adobe Premiere Pro*", "Adobe Premiere Pro.exe")), reverse=True)
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def remember_install(pr: Premiere) -> None:
    """Store Premiere.exe's location (asked from the running app) so it can be started later."""
    try:
        d = pr.run("return KL.ok(String(app.path));", "path", 15)
        exe = Path(d) / "Adobe Premiere Pro.exe"
        if exe.exists():
            KNOWN_EXE.write_text(str(exe), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def find_preset() -> str | None:
    """H.264 (.mp4) 'Match Source' export preset that ships with Premiere."""
    exe = premiere_exe()
    if not exe:
        return None
    root = Path(exe).parent
    h264 = sorted(glob.glob(str(root / "MediaIO" / "systempresets" / "4E49434B_48323634" / "*.epr")))
    extra = sorted(glob.glob(str(root / "CEP" / "extensions" / "*" / "**" / "*.epr"), recursive=True))
    for pat in (r"^match source - adaptive high", r"^match source - high", r"high quality 2160p", r"^match source"):
        for f in h264 + [x for x in extra if re.search(r"match source - adaptive high", Path(x).name, re.I)]:
            if re.search(pat, Path(f).name, re.I):
                return f
    return None


def _ping(pr: Premiere) -> bool:
    try:
        pr.ping()
        return True
    except (BridgeNotRunning, PremiereError):
        return False


TEMPLATE_NAME = "bos_proje.prproj"   # empty project kept to start Premiere straight into a project


def _premiere_running() -> bool:
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Adobe Premiere Pro.exe", "/NH"], capture_output=True,
                             text=True, creationflags=0x08000000, timeout=20).stdout
        return "Adobe Premiere Pro.exe" in out
    except Exception:  # noqa: BLE001
        return False


def _open_file(path: Path, exe: str) -> str:
    """Start Premiere with a project the way a double-click does (ShellExecute); fall back to the exe."""
    if os.name == "nt":
        try:
            os.startfile(str(path))  # noqa: S606
            return "dosya ilişkilendirmesi"
        except OSError:
            pass
    subprocess.Popen([exe, str(path)], creationflags=0x00000008 | 0x00000200 if os.name == "nt" else 0,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    return "exe"


def launch(pr: Premiere, prproj: Path, last: Path | None, progress, wait_s: float = 300) -> bool:
    """Make sure Premiere runs WITH A PROJECT OPEN (the bridge panel only loads inside a project). True if we started it."""
    import shutil
    if _ping(pr):
        remember_install(pr)
        return False
    exe = premiere_exe()
    if not exe:
        raise PremiereUnavailable("Premiere Pro bulunamadı")
    tpl = Path(__file__).resolve().parents[2] / "premiere" / TEMPLATE_NAME
    if not prproj.exists() and tpl and tpl.exists():
        prproj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tpl, prproj)          # a copy of the empty project opens directly as this video's project
    start = prproj if prproj.exists() else (last if last and last.exists() else None)
    if start is None:
        raise PremiereUnavailable("Premiere'i bir projeyle açmam gerekiyor ama henüz hiç proje yok: Premiere'i bir kez "
                                  "açıp herhangi bir proje açın, sonrası kendiliğinden olur")
    progress(0.0, "Premiere açılıyor")
    was_running = _premiere_running()
    how = _open_file(start, exe)
    t0 = time.time()
    retried = False
    while time.time() - t0 < wait_s:
        time.sleep(5)
        el = int(time.time() - t0)
        if _ping(pr):
            time.sleep(3)  # let the project finish loading
            remember_install(pr)
            return True
        running = _premiere_running()
        progress(min(0.9, el / wait_s), f"Premiere açılıyor ({el} sn{'' if running else ', henüz başlamadı'})")
        if not running and el >= 45 and not retried:
            retried = True                      # the shell route did not start it: try the exe directly
            subprocess.Popen([exe, str(start)], creationflags=0x00000008 | 0x00000200 if os.name == "nt" else 0,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
            how += " + exe"
    state = "çalışıyor" if _premiere_running() else "çalışmıyor"
    raise PremiereUnavailable(f"Premiere {wait_s:.0f} sn içinde köprüye bağlanmadı (Premiere {state}, açma yolu: {how}"
                              f"{', önceden açıktı' if was_running else ''}). Proje açık mı ve MCP Bridge paneli "
                              "çalışma alanında mı kontrol edin.")


def open_project(pr: Premiere, prproj: Path) -> str:
    prproj.parent.mkdir(parents=True, exist_ok=True)
    return pr.run(f"""
        var p = {js_str(str(prproj))};
        var cur = '';
        try {{ cur = String(app.project.path); }} catch (e) {{}}
        if (cur.replace(/\\//g, '\\\\').toLowerCase() === p.toLowerCase()) return KL.ok('acik');
        if (cur) {{ try {{ app.project.save(); }} catch (e) {{}} }}
        var ok = new File(p).exists ? app.openDocument(p, true, true, true) : app.newProject(p);
        if (!ok) throw new Error('Premiere projesi açılamadı: ' + p);
        return KL.ok(String(app.project.path));""", "open", 300)


def ensure_template(pr: Premiere) -> None:
    """Keep an empty project file so Premiere can later be started straight into a project."""
    import shutil
    tpl = Path(__file__).resolve().parents[2] / "premiere" / TEMPLATE_NAME
    if tpl.exists():
        return
    try:
        info = pr.run("""app.project.save();
            return KL.ok({path: String(app.project.path), n: app.project.rootItem.children.numItems});""", "tpl", 60)
        src = Path(info["path"])
        if info["n"] == 0 and src.exists():
            tpl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tpl)
    except Exception:  # noqa: BLE001
        pass


def quit_premiere(pr: Premiere) -> None:
    try:
        pr.run("app.project.save(); app.quit(); return KL.ok(true);", "quit", 30)
    except Exception:  # noqa: BLE001  (Premiere exits before it can answer)
        pass


# ------------------------------------------------------------------ assets -------------------------------------
def save_png(rgba: np.ndarray, path: Path) -> str:
    from PIL import Image
    Image.fromarray(rgba).save(path, optimize=False, compress_level=3)
    return str(path)


def border_png(w: int, h: int, box: tuple, path: Path) -> str:
    """Thin dark frame + soft shadow around the speaker box (transparent elsewhere)."""
    from PIL import Image, ImageDraw, ImageFilter
    x0, y0, x1, y1 = box
    s = w / 1920
    b = max(2, int(4 * s))
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle((x0 - b + 4, y0 - b + 8, x1 + b + 4, y1 + b + 8), fill=(0, 0, 0, 110))
    sh = sh.filter(ImageFilter.GaussianBlur(int(10 * s) or 1))
    hole = Image.new("L", (w, h), 255)
    ImageDraw.Draw(hole).rectangle((x0, y0, x1 - 1, y1 - 1), fill=0)
    sh.putalpha(Image.fromarray(np.minimum(np.asarray(sh.getchannel("A")), np.asarray(hole))))
    im = Image.alpha_composite(im, sh)
    ImageDraw.Draw(im).rectangle((x0 - b, y0 - b, x1 + b - 1, y1 + b - 1), outline=(42, 42, 42, 255), width=b)
    im.save(path)
    return str(path)


def sfx_copy(src: str, dst: Path, gain_db: float) -> tuple[str, float]:
    """Peak-normalised copy (silence trimmed) so Premiere can play it at 0 dB."""
    from ..motion.render import SR, _pcm
    a = _pcm(src, 0.0, 2.5)
    nz = np.nonzero(np.abs(a).max(axis=1) > 1e-4)[0]
    a = a[: nz[-1] + 1] if len(nz) else a[:1]
    a = a / (float(np.abs(a).max()) or 1.0) * (10 ** (gain_db / 20))
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    return str(dst), len(a) / SR


# ------------------------------------------------------------------ spec ---------------------------------------
def _zoom_pos(z: float, cx: float, cy: float) -> tuple[float, float]:
    """Motion position (normalised) that keeps source point c centred under scale z, edges never exposed."""
    if z <= 1.001:
        return 0.5, 0.5
    cx = min(max(cx, 0.5 / z), 1 - 0.5 / z)
    cy = min(max(cy, 0.5 / z), 1 - 0.5 / z)
    return 0.5 - z * (cx - 0.5), 0.5 - z * (cy - 0.5)


def build_spec(ctx: dict, cam_map: dict, sizes: dict, activity: dict, work: Path, demo: dict, sfx_files: dict) -> dict:
    from ..editing.director import CursorFollow
    W, H, fps = ctx["w"], ctx["h"], ctx["fps"]
    segs, events = ctx["segs"], ctx["events"]
    spec = {"W": W, "H": H, "v1": [], "v2": [], "v3": [], "v4": [], "v5": [], "v6": [], "a4": [], "a5": []}
    tl_file = cam_map.get("__timeline__")

    def cam_src(s, a):  # source (file, in) of camera time `a` (timeline seconds) inside segment s
        if tl_file and s.get("cam_tl") is not None:
            return tl_file, s["cam_tl"] + (a - s["timeline_start"])
        if tl_file:
            return tl_file, s["timeline_start"] + (a - s["timeline_start"])
        return s["file"], s["in"] + (a - s["timeline_start"])
    geo = ctx.get("geo") or {"ox": 0.0, "oy": 0.0, "fw": 1.0, "fh": 1.0}
    gx = lambda cx: geo["ox"] + cx * geo["fw"]  # noqa: E731  (face position inside the output frame)
    gy = lambda cy: geo["oy"] + cy * geo["fh"]  # noqa: E731
    mont_done = False
    for s, pc in zip(segs, ctx["pieces"]):
        if s.get("kind") == "montage":
            if cam_map.get("__montage__"):
                if not mont_done:   # one baked clip for the whole promo
                    ms = [x for x in segs if x.get("kind") == "montage"]
                    M = ms[-1]["timeline_end"] - ms[0]["timeline_start"]
                    spec["v1"].append({"f": cam_map["__montage__"], "i": 0.0, "o": M, "t": ms[0]["timeline_start"], "mute": True})
                    mont_done = True
                continue
            bw0, bh0 = sizes.get(s["file"], (W, H))
            cov = 100 * min(W / bw0, H / bh0)
            t0, t1 = s["timeline_start"], s["timeline_end"]
            spec["v1"].append({"f": s["file"], "i": s["in"], "o": s["out"], "t": t0, "mute": True,
                               "keys": [[round(t0, 3), round(cov, 3), 0.5, 0.5], [round(t1, 3), round(cov * 1.06, 3), 0.5, 0.5]]})
            continue
        x, y = _zoom_pos(pc.zoom, gx(pc.cx), gy(pc.cy))
        f0, i0 = cam_src(s, s["timeline_start"])
        item = {"f": f0, "i": i0, "o": i0 + (s["timeline_end"] - s["timeline_start"]), "t": s["timeline_start"],
                "s": round(100 * pc.zoom, 3), "x": round(x, 5), "y": round(y, 5)}
        if getattr(pc, "trans_in", 0) > 0:  # zoom-in transition at a chapter change
            z0 = pc.zoom * (1 + pc.trans_in)
            x0, y0 = _zoom_pos(z0, gx(pc.cx), gy(pc.cy))
            t0 = s["timeline_start"]
            item["keys"] = [[round(t0, 3), round(100 * z0, 3), round(x0, 5), round(y0, 5)],
                            [round(t0 + 0.32, 3), item["s"], item["x"], item["y"]]]
        spec["v1"].append(item)
    bw = int(W * demo.get("speaker_box_width_fraction", 0.21))
    bh = int(bw * H / W)
    m = int(W * demo.get("speaker_box_left_fraction", 0.022))
    box = (m, H - m - bh, m + bw, H - m)
    border = None
    for e in events:
        if e["type"] == "screen":
            parts = [s for s in segs if s.get("screen") and s["timeline_end"] > e["t0"] and s["timeline_start"] < e["t1"]]
            for k, s in enumerate(parts):
                a, b = max(e["t0"], s["timeline_start"]), min(e["t1"], s["timeline_end"])
                if b - a < 1 / fps:
                    continue
                sc = s["screen"]
                si = sc["in"] + (a - s["timeline_start"])
                sw, sh = sizes.get(sc["file"], (W, H))
                fit = min(W / sw, H / sh)
                keys = []
                fol = CursorFollow(activity.get(sc["file"], {}).get("cursor", []), demo.get("cursor_zoom", 1.4),
                                   demo.get("cursor_dead_zone", 0.35))
                last = None
                t = a
                while t <= b + 1e-6:
                    z, cx, cy = fol.step(si + (t - a))
                    S = fit * z
                    cx = min(max(cx, 0.5 / z), 1 - 0.5 / z)
                    cy = min(max(cy, 0.5 / z), 1 - 0.5 / z)
                    X = 0.5 - S * (cx - 0.5) * sw / W
                    Y = 0.5 - S * (cy - 0.5) * sh / H
                    key = [round(t, 3), round(100 * S, 3), round(X, 5), round(Y, 5)]
                    if last is None or abs(key[1] - last[1]) > 0.3 or abs(key[2] - last[2]) > 0.002 or abs(key[3] - last[3]) > 0.002:
                        keys.append(key)
                        last = key
                    t += 0.25
                fin = 0.25 if k == 0 else 0
                fout = 0.25 if k == len(parts) - 1 else 0
                item = {"f": sc["file"], "i": si, "o": si + (b - a), "t": a, "fi": fin, "fo": fout}
                if len(keys) > 1:
                    item["keys"] = keys
                else:
                    item.update(s=keys[0][1], x=keys[0][2], y=keys[0][3])
                spec["v2"].append(item)
                cf, ci = cam_src(s, a)
                spec["v3"].append({"f": cf, "i": ci, "o": ci + (b - a), "t": a,
                                   "s": round(100 * bw / W, 3), "x": round((box[0] + bw / 2) / W, 5),
                                   "y": round((box[1] + bh / 2) / H, 5), "fi": fin, "fo": fout})
            if border is None:
                border = border_png(W, H, box, work / "kutu_cerceve.png")
            spec["v4"].append({"png": border, "t": e["t0"], "d": e["t1"] - e["t0"], "fi": 0.25, "fo": 0.25})
        elif e["type"] == "broll":
            j_in = any(abs(o["t1"] - e["t0"]) < 0.02 for o in events if o is not e and o["type"] == "broll")
            j_out = any(abs(o["t0"] - e["t1"]) < 0.02 for o in events if o is not e and o["type"] == "broll")
            fi, fo = (0.0 if j_in else 0.2), (0.0 if j_out else 0.2)
            bw2, bh2 = sizes.get(e["file"], (W, H))
            cover = 100 * max(W / bw2, H / bh2)
            d = e["t1"] - e["t0"]
            if e.get("image"):
                spec["v2"].append({"png": e["file"], "t": e["t0"], "d": d, "fi": fi, "fo": fo,
                                   "keys": [[e["t0"], round(cover, 3), 0.5, 0.5], [e["t1"], round(cover * 1.08, 3), 0.5, 0.5]]})
            else:
                spec["v2"].append({"f": e["file"], "i": 0.0, "o": d, "t": e["t0"], "fi": fi, "fo": fo,
                                   "s": round(cover, 3), "x": 0.5, "y": 0.5})
    ends = {"v5": -1.0, "v6": -1.0}
    for k, ov in enumerate(ctx["overlays"]):
        p = save_png(ov.meta["static"]() if ov.anim == "draw" else ov.rgba, work / f"grafik_{k:03d}.png")
        fd = ov.fade if ov.anim not in ("wipe", "sweep", "type", "words") else 0.2   # reveal animations: short fade there
        item = {"png": p, "t": ov.t0, "d": ov.t1 - ov.t0, "fi": fd, "fo": fd}
        keys = overlay_keys(ov, W, H)
        if keys:
            item["keys"] = keys
        tr = "v5" if ends["v5"] <= ov.t0 + 1e-3 else "v6"   # overlapping layers (title panel + text) on two tracks
        ends[tr] = max(ends[tr], ov.t1)
        spec[tr].append(item)
    # sequence markers, like an editor leaves for review: every title, camera move, transition and the montage
    marks = [{"t": c["t0"], "n": f"Başlık: {c.get('text', '')}", "c": c.get("style", ""), "col": 1} for c in ctx.get("placed") or []]
    marks += [{"t": f["t0"], "n": ("Kaydırma: " if f["kind"] == "slide" else "Yakınlaştırma: ") + f.get("text", "")[:60],
               "c": "hızlı motorda kamera hareketi", "col": 3} for f in ctx.get("camfx") or []]
    marks += [{"t": s["timeline_start"], "n": f"Geçiş ({getattr(pc, 'trans', 'zoom')})", "c": "", "col": 5}
              for s, pc in zip(segs, ctx["pieces"]) if getattr(pc, "trans_in", 0) > 0]
    spec["markers"] = sorted(({**m, "t": round(m["t"], 3)} for m in marks), key=lambda m: m["t"])
    aend = {"a4": -1.0, "a5": -1.0}
    for c in ctx["d"]["plan"]["sfx"]:
        if c.get("dur"):   # music cue (montage): the original file, from its offset
            f, i0, dur = c["file"], c.get("offset", 0.0), c["dur"]
        else:
            (f, dur), i0 = sfx_files[c["file"]], 0.0
        tr = "a4" if aend["a4"] <= c["t"] + 1e-3 else "a5"
        aend[tr] = max(aend[tr], c["t"] + dur)
        spec[tr].append({"f": f, "i": i0, "o": i0 + dur, "t": c["t"],
                         "db": round(c.get("gain_db", -9.0) - 5.0, 1) if c.get("dur") else 0.0})
    return spec


def overlay_keys(ov, W: int, H: int) -> list | None:
    """Premiere Motion keyframes that reproduce the overlay animation (full-frame PNG, so the position is
    compensated to scale around the graphic's own centre)."""
    x0, y0, x1, y1 = ov.box
    bx, by = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    pos = lambda sc: (round(bx - sc / 100 * (bx - 0.5), 5), round(by - sc / 100 * (by - 0.5), 5))  # noqa: E731
    if ov.anim == "pop":
        return [[round(ov.t0 + dt, 3), sc, *pos(sc)] for dt, sc in ((0.0, 55.0), (0.2, 108.0), (0.35, 100.0))]
    if ov.anim == "push":
        return [[round(ov.t0, 3), 100.0, 0.5, 0.5], [round(ov.t1, 3), 104.0, 0.5, 0.5]]
    if ov.anim in ("rise", "wipe", "sweep", "type", "words"):   # Premiere: reveal titles slide up instead
        return [[round(ov.t0, 3), 100.0, 0.5, 0.55], [round(ov.t0 + 0.35, 3), 100.0, 0.5, 0.5]]
    return None


# ------------------------------------------------------------------ build in Premiere --------------------------
PLACE_JS = """
var seq = KL.findSequence(NAME);
KL.setSeqSize(seq.frameSizeHorizontal, seq.frameSizeVertical);
var b = DATA, cache = {}, n = 0, warn = [];
function item(p) { if (!cache[p]) cache[p] = KL.findItemByPath(p); return cache[p]; }
var tr = TRACK;
for (var i = 0; i < b.length; i++) {
  var x = b[i], c, it = item(x.png || x.f);
  if (!it) { if (warn.length < 5) warn.push("projede yok: " + (x.png || x.f)); continue; }   // never abort the batch
  try {
    if (x.png) c = KL.placeStill(tr, it, x.t, x.d);
    else c = KL.placeOn(tr, it, x.i, x.o, x.t);
  } catch (eP) { if (warn.length < 5) warn.push("yerleştirilemedi: " + eP); continue; }
  if (x.mute) { try { var ac = KL.clipAt(seq.audioTracks[0], x.t); if (ac) ac.remove(false, false); } catch (eM) {} }
  if (x.db !== undefined) {
    try {
      for (var ci = 0; ci < c.components.numItems; ci++) {
        var cp = c.components[ci];
        if (cp.displayName === "Volume" || cp.matchName === "Internal Volume Stereo") {
          for (var pi = 0; pi < cp.properties.numItems; pi++) {
            var pp = cp.properties[pi];
            if (pp.displayName === "Level") pp.setValue(Math.pow(10, (x.db - 15) / 20), true);
          }
        }
      }
    } catch (eV) { if (warn.length < 5) warn.push("ses seviyesi: " + eV); }
  }
  try {
    if (x.keys) KL.motionKeys(c, x.keys);
    else if (x.s !== undefined && (Math.abs(x.s - 100) > 0.01 || Math.abs(x.x - 0.5) > 1e-4 || Math.abs(x.y - 0.5) > 1e-4)) KL.setMotion(c, x.s, x.x, x.y);
    if (x.fi || x.fo) KL.fade(c, x.fi || 0, x.fo || 0);
  } catch (e) { if (warn.length < 5) warn.push(String(e)); }
  n++;
}
for (var p in cache) if (cache.hasOwnProperty(p)) KL.resetInOut(cache[p]);
return KL.ok({n: n, warn: warn});
"""


def build_edit(pr: Premiere, name: str, spec: dict, imports: dict, progress) -> dict:
    progress(0.0, "Dosyalar Premiere'e alınıyor")
    for binname, files in imports.items():
        if files:
            pr.run(f"KL.importFiles({js_val(files)}, {js_str(binname)}); return KL.ok(true);", "import", 900)
    progress(0.1, "Sekans oluşturuluyor")
    info = pr.run(f"""
        var name = {js_str(name)};
        var old = KL.findSequence(name);
        if (old) app.project.deleteSequence(old);
        var seq = KL.sequenceFromClip(name, KL.findItemByPath({js_str(spec['v1'][0]['f'])}), "Sekanslar");
        app.project.activeSequence = seq;
        KL.ensureTracks(seq, {NV}, {NA});
        return KL.ok({{w: seq.frameSizeHorizontal, h: seq.frameSizeVertical, tb: seq.timebase,
                      v: seq.videoTracks.numTracks, a: seq.audioTracks.numTracks}});""", "seq", 180)
    warns = []
    steps = [("v1", "seq.videoTracks[0]", "Kesimler"), ("v2", "seq.videoTracks[1]", "Ekran kaydı ve ek görüntüler"),
             ("v3", "seq.videoTracks[2]", "Konuşmacı kutusu")]
    total = sum(len(spec.get(k, [])) for k in ("v1", "v2", "v3", "v4", "v5", "v6", "a4", "a5")) or 1
    done = 0

    def place(key, track, label):
        nonlocal done
        rows = spec.get(key, [])
        for k in range(0, len(rows), BATCH):
            progress(0.15 + 0.8 * done / total, f"{label} {k}/{len(rows)}")
            js = PLACE_JS.replace("NAME", js_str(name)).replace("DATA", js_val(rows[k:k + BATCH])).replace("TRACK", track)
            try:
                r = pr.run(js, f"{key}_{k}", 900)
            except Exception as e:  # noqa: BLE001  (one bad batch must not lose the whole edit)
                warns.append(f"{label}: {e}"[:300])
                r = None
            if isinstance(r, dict):
                warns.extend(r.get("warn", []))
            elif r is not None:
                warns.append(f"{label}: beklenmeyen cevap {str(r)[:200]}")
            done += len(rows[k:k + BATCH])
    for key, track, label in steps:
        place(key, track, label)
    # linked audio of screen / b-roll / camera copies would double the voice
    pr.run(f"""var seq = KL.findSequence({js_str(name)});
        KL.clearTrack(seq.audioTracks[1]); KL.clearTrack(seq.audioTracks[2]); return KL.ok(true);""", "clear", 300)
    place("v4", "seq.videoTracks[3]", "Kutu çerçevesi")
    place("v5", "seq.videoTracks[4]", "Yazılar ve kartlar")
    place("v6", "seq.videoTracks[5]", "Yazılar (2. katman)")
    place("a4", "seq.audioTracks[3]", "Ses efektleri")
    place("a5", "seq.audioTracks[4]", "Ses efektleri 2")
    if spec.get("markers"):
        try:
            r = pr.run(f"""var seq = KL.findSequence({js_str(name)}); var M = {js_val(spec['markers'])}, n = 0;
                for (var i = 0; i < M.length; i++) {{
                    try {{ var m = seq.markers.createMarker(M[i].t); m.name = M[i].n; m.comments = M[i].c;
                           try {{ m.setColorByIndex(M[i].col, 0); }} catch (e) {{}} n++; }} catch (e) {{}}
                }}
                return KL.ok(n);""", "markers", 300)
            if isinstance(r, (int, float)) and r < len(spec["markers"]):
                warns.append(f"İşaretler: {int(r)}/{len(spec['markers'])} eklendi")
        except Exception as e:  # noqa: BLE001
            warns.append(f"İşaretler eklenemedi: {e}"[:200])
    pr.run("app.project.save(); return KL.ok(true);", "save", 300)
    progress(0.97, "Premiere'den geri okunuyor")
    state = pr.sequence_state(name)
    return {"sequence": info, "warnings": warns[:10], "check": check_state(spec, state, ctx_fps=254016000000 / float(info["tb"]))}


def check_state(spec: dict, state: dict, ctx_fps: float) -> dict:
    problems = []
    vt = state.get("video", [])
    at = state.get("audio", [])
    want = {0: len(spec["v1"]), 1: len(spec["v2"]), 2: len(spec["v3"]), 3: len(spec["v4"]), 4: len(spec["v5"]),
            5: len(spec.get("v6", []))}
    for i, n in want.items():
        got = len(vt[i]["clips"]) if i < len(vt) else 0
        if got != n:
            problems.append(f"V{i + 1}: Premiere'de {got} klip, planda {n}")
    for i in (1, 2):
        if i < len(at) and at[i]["clips"]:
            problems.append(f"A{i + 1}'de {len(at[i]['clips'])} fazladan ses klibi (çift ses)")
    if len(at) > 3 and len(at[3]["clips"]) != len(spec["a4"]):
        problems.append(f"A4: {len(at[3]['clips'])} ses efekti, planda {len(spec['a4'])}")
    tol = 1.01 / ctx_fps
    for s, c in zip(spec["v1"], vt[0]["clips"] if vt else []):
        if abs(c["start"] - s["t"]) > tol:
            problems.append(f"V1 kesim {s['t']:.2f} sn'de olmalıydı, {c['start']:.2f} sn'de")
            break
    return {"ok": not problems, "problems": problems}


def export(pr: Premiere, name: str, out: Path, preset: str, progress) -> str:
    progress(0.0, "Premiere dışa aktarıyor (uzun sürebilir)")
    if out.exists():
        out.unlink()
    r = pr.run(f"""
        var seq = KL.findSequence({js_str(name)});
        app.project.activeSequence = seq;
        // sound effects are mixed after the voice is cleaned (a denoiser would eat them): mute them for export
        var muted = [];
        for (var i = 3; i < seq.audioTracks.numTracks; i++) {{ try {{ seq.audioTracks[i].setMute(1); muted.push(i); }} catch (e) {{}} }}
        var r = seq.exportAsMediaDirect({js_str(str(out))}, {js_str(preset)}, app.encoder.ENCODE_ENTIRE);
        for (var j = 0; j < muted.length; j++) {{ try {{ seq.audioTracks[muted[j]].setMute(0); }} catch (e) {{}} }}
        return KL.ok(String(r));""", "export", 4 * 3600)
    if not out.exists() or out.stat().st_size < 1000:
        raise PremiereError(f"Premiere dışa aktaramadı: {r}")
    return r


def finish(src: Path, dst: Path, ass: Path | None, loudness: str, progress, music: str | None = None,
           sfx: list[dict] | None = None, speech: list | None = None, speech_wav: Path | None = None) -> dict:
    """Captions + -14 LUFS on Premiere's export. speech_wav: the voice track built by us (music removed,
    soft joins) - used instead of the sound in Premiere's export."""
    from ..motion.render import NOWIN, SR
    from ..motion.vertical import _enc, pick_encoder
    work = dst.parent
    wav, wav_n = work / "pr_ses.wav", work / "pr_ses_norm.wav"
    if speech_wav and Path(speech_wav).exists():
        import shutil
        shutil.copyfile(speech_wav, wav)
    else:
        subprocess.run([ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR),
                        str(wav)], check=True, creationflags=NOWIN)
    progress(0.1, "Ses seviyesi ayarlanıyor")
    from ..audio.master import master
    loud = master(str(wav), str(wav_n), loudness, music=music, sfx=sfx, speech=speech)
    tmp = dst.with_suffix(".yaziliyor.mp4")
    cmd = [ff.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-i", str(wav_n), "-map", "0:v", "-map", "1:a"]
    if ass:
        esc = str(ass).replace("\\", "/").replace(":", "\\:")
        fonts = "C\\:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts"
        cmd += ["-vf", f"subtitles='{esc}':fontsdir='{fonts}'"] + _enc(pick_encoder(), 16)
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(tmp)]
    progress(0.3, "Altyazı ekleniyor" if ass else "Birleştiriliyor")
    p = subprocess.run(cmd, capture_output=True, creationflags=NOWIN)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-600:])
    replace_file(tmp, dst)
    final = work / "kurgu_ses_son.wav"
    try:
        replace_file(wav_n, final)
    except OSError:
        final = None
    wav.unlink(missing_ok=True)
    return {"audio": loud, "final_audio": str(final) if final else None}


# ------------------------------------------------------------------ entry ---------------------------------------
def make_main_premiere(proj, ctx: dict, progress, final: dict | None = None) -> dict:
    """Premiere review project (+ the finished video unless `final` is given).
    final: info of a video already rendered by our engine in one pass (no skin retouch to bake): the Premiere
    timeline then uses the original recordings directly, no intermediate camera file is rendered."""

    cfg = proj.cfg
    if not premiere_exe() and not _ping(Premiere(cfg.path("bridge_dir"))):
        raise PremiereUnavailable("Premiere Pro'nun yeri bilinmiyor: bir kez Premiere'i açın, uygulama yerini öğrenir")
    work = proj.dir / "premiere"
    work.mkdir(exist_ok=True)
    from ..pipeline import project_grade, retouch_params
    rc = [retouch_params(proj), cfg.design.get("image", {})]
    grade = project_grade(proj)
    # 1. ONE retouched + colour-corrected camera file holding only the kept parts (camera-only timeline);
    #    much faster than retouching whole recordings, and every cut stays a separate clip in Premiere
    import hashlib
    import json as _json

    from ..motion.render import Piece, render
    from ..pipeline import _retoucher
    cam_segs = [s for s in ctx["segs"] if s.get("kind") != "montage"]
    key = hashlib.sha1(_json.dumps([[s["file"], round(s["in"], 4), round(s["out"], 4)] for s in cam_segs]
                                   + [ctx["w"], ctx["h"], ctx["fps"], rc, _json.dumps(grade, sort_keys=True)[:4000] if grade else None, 6]).encode()).hexdigest()[:12]
    cam_file = work / f"kamera_kurgu_{key}.mp4"
    rt_note = (final or {}).get("retouch", {}).get("note") if final else None
    if final is None and not cam_file.exists():
        rt, rt_note = _retoucher(proj)
        pieces = [Piece(s["file"], s["in"], s["out"], cx=pc.cx, cy=pc.cy)
                  for s, pc in zip(ctx["segs"], ctx["pieces"]) if s.get("kind") != "montage"]
        render(pieces, str(cam_file), ctx["w"], ctx["h"], ctx["fps"], frame_fn=rt, audio_master=False, cq=14,
               image=cfg.design.get("image"),
               progress=lambda x, m: progress(0.3 * x, "Rötuş ve renk: " + m))
    cam_map = {"__timeline__": str(cam_file)} if final is None else {}
    mont_segs = [s for s in ctx["segs"] if s.get("kind") == "montage"] if final is None else []
    if mont_segs:   # the music promo as one finished clip (cuts on the beat, push-ins, same frame shape)
        mkey = hashlib.sha1(_json.dumps([[s["file"], s["in"], s["out"]] for s in mont_segs]
                                        + [ctx["w"], ctx["h"], ctx["fps"], cfg.design.get("image"), 2]).encode()).hexdigest()[:12]
        mont_file = work / f"montaj_{mkey}.mp4"
        if not mont_file.exists():
            render([Piece(s["file"], s["in"], s["out"], kind="montage") for s in mont_segs], str(mont_file),
                   ctx["w"], ctx["h"], ctx["fps"], audio_master=False, cq=14, image=cfg.design.get("image"),
                   progress=lambda x, m: progress(0.3, "Müzikli tanıtım: " + m))
        cam_map["__montage__"] = str(mont_file)
    progress(0.3, "Premiere için malzemeler hazırlanıyor")
    sizes = {}
    for f in ({s["screen"]["file"] for s in ctx["segs"] if s.get("screen")} | {e["file"] for e in ctx["events"] if e["type"] == "broll"}
              | {s["file"] for s in ctx["segs"] if s.get("kind") == "montage"}):
        if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            from PIL import Image
            with Image.open(f) as im:
                sizes[f] = im.size
        else:
            i = ff.probe(f)
            sizes[f] = (i.width, i.height)
    sfx_files = {}
    for k, f in enumerate(sorted({c["file"] for c in ctx["d"]["plan"]["sfx"] if not c.get("dur")})):
        sfx_files[f] = sfx_copy(f, work / f"sfx_{k:02d}.wav", ctx["P"]["sfx_db"])
    spec = build_spec(ctx, cam_map, sizes, ctx["d"]["activity"], work, cfg.design.get("demo", {}), sfx_files)
    # 2. Premiere
    pr = Premiere(cfg.path("bridge_dir"))
    prproj = work / f"{proj.name}.prproj"
    last_file = cfg.root / "premiere_son_proje.txt"
    last = Path(last_file.read_text(encoding="utf-8").strip()) if last_file.exists() else None
    opened = launch(pr, prproj, last, lambda f, m: progress(0.32 + 0.05 * f, m))
    preset = find_preset()
    try:
        if not preset and final is None and ctx["opts"].get("premiere_export", False):
            raise PremiereUnavailable("Premiere'in H.264 dışa aktarma ayarı (.epr) bulunamadı")
        open_project(pr, prproj)
        ensure_template(pr)
        last_file.write_text(str(prproj), encoding="utf-8")
        name = f"{proj.settings.get('title') or proj.name} - kurgu"
        imports = {"Kamera": sorted(set(x["f"] for x in spec["v1"])),
                   "Ekran ve ek görüntüler": sorted(set(x.get("f") or x.get("png") for x in spec["v2"])),
                   "Grafikler": sorted(set(x["png"] for x in spec["v4"] + spec["v5"] + spec.get("v6", []) if x.get("png"))),
                   "Ses efektleri": sorted(set(x["f"] for x in spec["a4"] + spec.get("a5", [])))}
        rep = build_edit(pr, name, spec, imports, lambda f, m: progress(0.38 + 0.3 * f, m))
        raw = work / "premiere_cikti.mp4"
        premiere_export = final is None and bool(ctx["opts"].get("premiere_export", False))
        if premiere_export:   # slow (hours on long videos): only when the user asked for it
            export(pr, name, raw, preset, lambda f, m: progress(0.7, m))
    finally:
        # Premiere stays open on the finished sequence for preview; the user closes it
        try:
            pr.run(f"""var s = KL.findSequence({js_str(name)}); if (s) app.project.activeSequence = s;
                app.project.save(); return KL.ok(true);""", "show", 120)
        except Exception:  # noqa: BLE001
            pass
    out = Path(final["file"]) if final is not None and final.get("file") else ctx["out_dir"] / "kurgu.mp4"
    if final is not None:
        fin = dict(final.get("render") or {}, final_audio=(final.get("render") or {}).get("kept_audio") or str(work / "kurgu_ses_son.wav"))
        if not Path(fin["final_audio"]).exists():
            fin["final_audio"] = None
        return _after_final(pr, proj, ctx, out, fin, name, prproj, preset, rep, opened, rt_note, progress)
    if not premiere_export:
        fin = fast_final(ctx, cam_file, out, work, cfg, lambda f, m: progress(0.7 + 0.29 * f, m))
        return _after_final(pr, proj, ctx, out, fin, name, prproj, preset, rep, opened, rt_note, progress)
    speech_wav = None
    from ..audio.separate import enabled as sep_enabled
    if sep_enabled():   # the voice without the venue music, cut exactly like the timeline, soft joins
        from ..motion.render import build_audio
        progress(0.85, "Temiz konuşma sesi hazırlanıyor")
        speech_wav = work / "konusma_temiz.wav"
        build_audio(ctx["pieces"], str(speech_wav))
    fin = finish(raw, out, ctx["ass"], cfg.design.get("shorts", {}).get("loudness", "I=-14:TP=-1.5:LRA=11"), music=ctx.get("music"),
                 sfx=ctx["d"]["plan"]["sfx"], speech=ctx.get("speech"), speech_wav=speech_wav, progress=
                 lambda f, m: progress(0.85 + 0.14 * f, m))
    return _after_final(pr, proj, ctx, out, fin, name, prproj, preset, rep, opened, rt_note, progress)


def fast_final(ctx: dict, cam_file: Path, out: Path, work: Path, cfg, progress) -> dict:
    """The finished video WITHOUT Premiere's export (which took ~3 h for 34 min): the already retouched/colour
    corrected camera timeline file + the same zooms, titles, captions, montage, sound effects and mastered
    voice, encoded on the GPU. The Premiere project is still built and left open for checking or editing."""
    from ..motion.compose import Compositor
    from ..motion.render import Piece, render
    geo = ctx.get("geo") or {"ox": 0.0, "oy": 0.0, "fw": 1.0, "fh": 1.0}
    fast, orig = [], {}
    for s, pc in zip(ctx["segs"], ctx["pieces"]):
        if s.get("kind") == "montage" or pc.kind in ("montage", "hold"):
            fast.append(pc)
            continue
        t0 = float(s.get("cam_tl", s["timeline_start"]))
        q = Piece(str(cam_file), t0, t0 + pc.dur, zoom=pc.zoom, cx=geo["ox"] + pc.cx * geo["fw"],
                  cy=geo["oy"] + pc.cy * geo["fh"], screen=pc.screen, trans_in=pc.trans_in, trans=pc.trans, a_file=pc.file, a_in=pc.t_in)
        orig[id(q)] = pc
        fast.append(q)
    for pc in ctx["pieces"][len(ctx["segs"]):]:   # anything after the segments (e.g. an end hold)
        fast.append(pc)
    events = ctx["events"]
    comp = Compositor(events, ctx["d"]["activity"], ctx["w"], ctx["h"], ctx["fps"], cfg.design.get("demo", {})) \
        if any(e["type"] in ("screen", "broll") for e in events) else None

    def compose(rgb, cam, pc, src_t, t_out):   # the compositor thinks in original recording time
        o = orig.get(id(pc), pc)
        return comp(rgb, cam, o, o.t_in + (src_t - pc.t_in), t_out)
    final_audio = work / "kurgu_ses_son.wav"
    up = youtube_upscale(ctx)
    try:
        res = render(fast, str(out), ctx["w"], ctx["h"], ctx["fps"], frame_fn=None, overlays=ctx["overlays"],
                     cq=final_cq(ctx, up), upscale=up, camfx=ctx.get("camfx"),
                     ass_path=str(ctx["ass"]) if ctx["ass"] else None,
                     loudness=cfg.design.get("shorts", {}).get("loudness", "I=-14:TP=-1.5:LRA=11"),
                     progress=progress, compose=compose if comp else None, sfx=ctx["d"]["plan"]["sfx"],
                     music=ctx.get("music"), speech=ctx.get("speech"), keep_audio=str(final_audio))
    finally:
        if comp:
            comp.close()
    return {"audio": res.get("audio"), "final_audio": str(final_audio), "encoder": res.get("encoder"),
            "export": "hızlı motor (GPU), Premiere dışa aktarması atlandı"
                      + (f", YouTube için {up[0]}x{up[1]} (1440p) olarak verildi" if up else "")}


def final_cq(ctx: dict, up) -> int:
    """Encoder quality of the finished video: near-lossless master when it goes on to CapCut (one more encode
    there), otherwise YouTube upload quality."""
    if ctx["opts"].get("capcut", True):
        return 14
    return 17 if up else 16


def youtube_upscale(ctx: dict) -> tuple[int, int] | None:
    """A 1080p horizontal edit with "1440" chosen is delivered as 2560x1440: YouTube serves 1440p uploads with its
    sharper VP9/AV1 streams (also to 1080p viewers), so the video looks clearly better there than a 1080p upload."""
    w, h = int(ctx["w"]), int(ctx["h"])
    if ctx["opts"].get("capcut", True):   # CapCut re-encodes anyway: give it the native size at top quality
        return None
    if str(ctx["opts"].get("resolution", "1440")) != "1440" or w <= h or not (1000 <= h < 1440):
        return None
    return int(round(w * 1440 / h / 2) * 2), 1440


def _after_final(pr, proj, ctx, out, fin, name, prproj, preset, rep, opened, rt_note, progress) -> dict:
    from ..pipeline import _main_info
    if fin.get("final_audio"):   # Premiere preview = the finished sound (cleaned, levelled, music, effects)
        try:
            pr.run(f"""
                KL.importFiles([{js_str(fin['final_audio'])}], "Ses (son hali)");
                var seq = KL.findSequence({js_str(name)});
                KL.ensureTracks(seq, {NV}, {NA});
                var it = KL.findItemByPath({js_str(fin['final_audio'])});
                KL.clearTrack(seq.audioTracks[5]);
                seq.audioTracks[5].overwriteClip(it, KL.timeFromSeconds(0));
                for (var i = 0; i < 5; i++) {{ try {{ seq.audioTracks[i].setMute(1); }} catch (e) {{}} }}
                app.project.save();
                return KL.ok(true);""", "final_audio", 600)
        except Exception as e:  # noqa: BLE001
            fin["final_audio_note"] = f"Premiere'e son ses eklenemedi: {e}"
    info = _main_info(proj, ctx, out, {
        "engine": "premiere", "premiere": {"project": str(prproj), "sequence": name, "preset": Path(preset).name if preset else None,
                                           "check": rep["check"], "warnings": rep["warnings"], "opened_by_app": opened},
        "render": fin, "retouch": {"applied": rt_note is None, "note": rt_note}})
    progress(1.0, "Premiere kurgusu hazır")
    return info


# ------------------------------------------------------------------ self test -----------------------------------
def self_test(cfg, progress) -> dict:
    """End-to-end check on this machine: open Premiere by itself, build a small edit with every track type,
    export, close, and compare pixels of the export with what the plan says."""
    from ..motion.compose import draw_card, draw_subscribe
    from ..motion.render import NOWIN

    d = cfg.root / "premiere_test"
    d.mkdir(exist_ok=True)
    fb = ff.ffmpeg_bin()

    def mk(*a):
        subprocess.run([fb, "-v", "error", "-nostdin", "-y", *a], check=True, creationflags=NOWIN)
    progress(0.02, "Deneme dosyaları hazırlanıyor")
    cam, scr, foto, wh = d / "kamera.mp4", d / "ekran.mp4", d / "foto.png", d / "whoosh.wav"
    mk("-f", "lavfi", "-i", "color=c=0x3060a0:s=1920x1080:r=30:d=10", "-f", "lavfi", "-i", "sine=f=300:d=10",
       "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cam))
    mk("-f", "lavfi", "-i", "color=c=0xd02020:s=1280x720:r=30:d=14", "-f", "lavfi", "-i", "sine=f=900:d=14",
       "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(scr))
    from PIL import Image
    Image.new("RGB", (800, 600), (30, 190, 60)).save(foto)
    mk("-f", "lavfi", "-i", "sine=f=1500:d=0.3", str(wh))
    W, H = 1920, 1080
    work = d / "is"
    work.mkdir(exist_ok=True)
    spec = {"W": W, "H": H,
            "v1": [{"f": str(cam), "i": 0.0, "o": 4.0, "t": 0.0, "s": 100.0, "x": 0.5, "y": 0.5},
                   {"f": str(cam), "i": 5.0, "o": 10.0, "t": 4.0, "s": 115.0, "x": 0.5, "y": 0.5}],
            "v2": [{"f": str(scr), "i": 2.0, "o": 4.0, "t": 1.0, "fi": 0.25, "fo": 0.25, "s": 150.0, "x": 0.5, "y": 0.5},
                   {"png": str(foto), "t": 5.0, "d": 1.5, "fi": 0.2, "fo": 0.2,
                    "keys": [[5.0, 240.0, 0.5, 0.5], [6.5, 260.0, 0.5, 0.5]]}],
            "v3": [{"f": str(cam), "i": 1.0, "o": 3.0, "t": 1.0, "s": 21.0, "x": 0.127, "y": 0.86, "fi": 0.25, "fo": 0.25}],
            "v4": [{"png": border_png(W, H, (42, 1080 - 42 - 226, 42 + 403, 1080 - 42), work / "kutu.png"), "t": 1.0, "d": 2.0,
                    "fi": 0.25, "fo": 0.25}],
            "v5": [{"png": save_png(draw_card("Premiere testi", W, H), work / "kart.png"), "t": 7.0, "d": 1.0, "fi": 0.2, "fo": 0.2},
                   {"png": save_png(draw_subscribe(W, H), work / "abone.png"), "t": 8.0, "d": 1.0, "fi": 0.3, "fo": 0.3}],
            "a4": [{"f": str(wh), "i": 0.0, "o": 0.3, "t": 0.9}]}
    pr = Premiere(cfg.path("bridge_dir"))
    prproj = d / "deneme.prproj"
    last_file = cfg.root / "premiere_son_proje.txt"
    last = Path(last_file.read_text(encoding="utf-8").strip()) if last_file.exists() else None
    t0 = time.time()
    opened = launch(pr, prproj, last, lambda f, m: progress(0.05 + 0.2 * f, m))
    t_launch = round(time.time() - t0, 1)
    raw = d / "premiere_cikti.mp4"
    preset = find_preset()
    try:
        if not preset:   # the test always exports
            raise PremiereUnavailable("Premiere'in H.264 dışa aktarma ayarı (.epr) bulunamadı")
        open_project(pr, prproj)
        last_file.write_text(str(prproj), encoding="utf-8")
        imports = {"Kamera": [str(cam)], "Ekran ve ek görüntüler": [str(scr), str(foto)],
                   "Grafikler": [x["png"] for x in spec["v4"] + spec["v5"]], "Ses efektleri": [str(wh)]}
        rep = build_edit(pr, "Premiere testi", spec, imports, lambda f, m: progress(0.3 + 0.4 * f, m))
        export(pr, "Premiere testi", raw, preset, lambda f, m: progress(0.75, m))
    finally:
        if opened:
            quit_premiere(pr)

    def px(t, x, y):
        p = subprocess.run([fb, "-v", "error", "-nostdin", "-ss", str(t), "-i", str(raw), "-frames:v", "1",
                            "-vf", "scale=1920:1080", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True,
                           creationflags=NOWIN)
        a = np.frombuffer(p.stdout, np.uint8)
        return a.reshape(1080, 1920, 3)[y, x].tolist() if a.size == 1920 * 1080 * 3 else None
    near = lambda a, b, tol=40: a is not None and max(abs(int(x) - y) for x, y in zip(a, b)) <= tol  # noqa: E731
    probes = {
        "kamera (V1)": (0.5, 960, 300, (48, 96, 160)),
        "ekran kaydı (V2)": (2.0, 960, 300, (208, 32, 32)),
        "konuşmacı kutusu (V3)": (2.0, 240, 930, (48, 96, 160)),
        "fotoğraf (V2)": (5.75, 960, 540, (30, 190, 60)),
        "kart (V5)": (7.5, 30, 30, (15, 17, 22)),
    }
    checks = {}
    for k, (t, x, y, want) in probes.items():
        got = px(t, x, y)
        checks[k] = {"ok": near(got, want), "got": got, "want": list(want)}
    dur = ff.probe(raw).duration
    problems = list(rep["check"]["problems"]) + [f"{k}: renk {v['got']} (beklenen {v['want']})" for k, v in checks.items() if not v["ok"]]
    if abs(dur - 9.0) > 0.2:
        problems.append(f"Süre {dur:.2f} sn, beklenen 9.00")
    return {"ok": not problems, "problems": problems, "checks": checks, "warnings": rep["warnings"], "duration": dur,
            "launch_s": t_launch, "opened_by_app": opened, "preset": preset, "sequence": rep["sequence"]}
