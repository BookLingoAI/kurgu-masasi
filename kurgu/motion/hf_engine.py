"""Engine "hyperframes": the whole edit is one HyperFrames composition, rendered by HyperFrames (headless Chrome).

What stays ours: the cut list (which parts of which recordings), face-centred zoom targets, the transition points,
the graphic texts and the mastered sound (voice chain, effects, music) - HyperFrames has no speech editing.
What HyperFrames does: plays every kept part as a <video> clip (trim + speed), the punch-ins, the zoom / whip /
flash transitions (GSAP), all titles, prices and lists (catalog blocks as sub-compositions), mixing in the audio and
encoding. Not available here: our colour correction, the subscribe/like animation, the counting badge and burned-in
captions (they are drawn by our Python renderer) - the result says so.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..integrations import ffmpeg as ff
from ..integrations import hyperframes as hf
from . import hf_graphics as hg

W, H = 1920, 1080   # catalog blocks are laid out for 1080p


def _link(src: str, media: Path, cache: dict, progress=None) -> str:
    """Each recording once as a light HyperFrames-friendly copy: 1080p, constant 30 fps, a keyframe every second,
    no sound (the mastered sound is a separate track). On 23.10.2026 HyperFrames failed on the phone originals
    (variable frame rate, keyframes 2 s apart): 294 of 568 clip extractions broke and memory ran out."""
    if src in cache:
        return cache[src]
    media.mkdir(parents=True, exist_ok=True)
    name = f"p{len(cache):02d}.mp4"
    dst, ok = media / name, media / f"p{len(cache):02d}.ok"
    if not (dst.exists() and ok.exists() and ok.read_text(encoding="utf-8") == str(src)):
        if progress:
            progress(None, f"HyperFrames için kayıt hazırlanıyor: {Path(src).name}")
        from .vertical import pick_encoder
        enc = pick_encoder()
        venc = (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"] if "nvenc" in str(enc)
                else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"])
        tmp = media / f"{name}.yaziliyor.mp4"
        ff.run([ff.ffmpeg_bin(), "-v", "error", "-y", "-i", str(src), "-an", "-vf", "scale=-2:'min(1080,ih)':flags=lanczos,fps=30,format=yuv420p",
                *venc, "-g", "30", "-keyint_min", "30", "-bf", "0", "-movflags", "+faststart", str(tmp)])
        os.replace(tmp, dst)
        ok.write_text(str(src), encoding="utf-8")
    cache[src] = f"media/{name}"
    return cache[src]


def _sub_composition(spec: dict, accent, comp_dir: Path, gid: str) -> tuple[str, float]:
    html, dur, block = hg.build(spec, accent)
    m = re.search(r'data-composition-id="([^"]+)"', html)
    if m:   # every copy needs its own id (the timeline registry is keyed by it)
        html = html.replace(f'"{m.group(1)}"', f'"{gid}"')
    hg.stage_files(comp_dir, block)
    (comp_dir / f"{gid}.html").write_text(html, encoding="utf-8")
    return f"compositions/{gid}.html", dur


def build_project(ctx: dict, work: Path, audio_wav: str, progress=None) -> dict:
    pieces, fps, accent = ctx["pieces"], ctx["fps"], ctx.get("accent") or (255, 196, 0)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hg.GSAP, work / "gsap.min.js")
    shutil.copy2(audio_wav, work / "ses.wav")
    links: dict = {}
    clips, tweens, t = [], [], 0.0
    for k, pc in enumerate(pieces):
        dur = pc.dur
        if pc.kind == "hold" or dur <= 0.01:
            t += max(0.0, dur)
            continue
        src = _link(pc.file, work / "media", links, progress)
        z = 1.0 if pc.kind == "montage" else max(1.0, pc.zoom)
        rate = f' data-playback-rate="{pc.speed:.4f}"' if abs(pc.speed - 1) > 1e-3 else ""
        clips.append(f'<video id="v{k}" class="clip cam" src="{src}" muted playsinline data-start="{t:.3f}" '
                     f'data-duration="{dur:.3f}" data-media-start="{pc.t_in:.3f}"{rate} '
                     f'style="transform:scale({z:.4f});transform-origin:{pc.cx * 100:.1f}% {pc.cy * 100:.1f}%"></video>')
        if pc.kind == "montage":   # slow push in / pull out
            a, b = (1.0, 1.07) if pc.move >= 0 else (1.07, 1.0)
            tweens.append(f'tl.fromTo("#v{k}",{{scale:{a}}},{{scale:{b},duration:{dur:.3f},ease:"none"}},{t:.3f});')
        if pc.trans_in > 0:
            if pc.trans == "whip":
                tweens.append(f'tl.fromTo("#v{k}",{{xPercent:-7,filter:"blur(18px)"}},{{xPercent:0,filter:"blur(0px)",'
                              f'duration:0.28,ease:"power3.out"}},{t:.3f});')
            elif pc.trans == "flash":
                tweens.append(f'tl.fromTo("#flash",{{opacity:0.9}},{{opacity:0,duration:0.42,ease:"power2.out"}},{t:.3f});')
            else:   # zoom: arrives slightly bigger and settles on the punch-in
                tweens.append(f'tl.fromTo("#v{k}",{{scale:{z * (1 + pc.trans_in):.4f}}},{{scale:{z:.4f},duration:0.32,'
                              f'ease:"power3.out"}},{t:.3f});')
        t += dur
    total = t
    hosts = []
    comp_dir = work / "compositions"
    for i, spec in enumerate(sorted(ctx.get("hf_specs") or [], key=lambda s: s["t0"])):
        if spec["t0"] >= total:
            continue
        gid = f"g{i:03d}"
        src, dur = _sub_composition(spec, accent, comp_dir, gid)
        dur = min(dur, total - spec["t0"])
        hosts.append(f'<div id="{gid}" class="gfx" data-composition-id="{gid}" data-composition-src="{src}" '
                     f'data-start="{spec["t0"]:.3f}" data-duration="{dur:.3f}"></div>')
    html = f"""<!doctype html>
<html lang="tr"><head><meta charset="UTF-8"/><meta name="viewport" content="width={W}, height={H}"/>
<script src="gsap.min.js"></script>
<style>
html,body{{margin:0;width:{W}px;height:{H}px;overflow:hidden;background:#000}}
#root{{position:relative;width:{W}px;height:{H}px;overflow:hidden;background:#000}}
.cam{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;z-index:1}}
#flash{{position:absolute;inset:0;z-index:5;opacity:0;pointer-events:none;
  background:radial-gradient(circle at 70% 30%,#fff7e0 0%,rgba(255,200,120,.85) 45%,rgba(255,140,60,.4) 100%)}}
.gfx{{position:absolute;inset:0;z-index:10}}
</style></head><body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{total:.3f}" data-width="{W}" data-height="{H}">
{chr(10).join(clips)}
<div id="flash"></div>
{chr(10).join(hosts)}
<audio id="ses" src="ses.wav" data-start="0" data-duration="{total:.3f}" data-volume="1"></audio>
</div>
<script>
window.__timelines = window.__timelines || {{}};
const tl = gsap.timeline({{paused: true}});
{chr(10).join(tweens)}
tl.set({{}}, {{}}, {total:.3f});
window.__timelines["main"] = tl;
</script></body></html>"""
    (work / "index.html").write_text(html, encoding="utf-8")
    meta = {"duration": round(total, 3), "clips": len(clips), "graphics": len(hosts), "files": len(links), "fps": fps}
    (work / "kurgu_hf.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def render_project(work: Path, out: Path, fps: float, total: float, progress) -> dict:
    """Runs `hyperframes render` and turns its log into progress; the job's cancel stops Chrome too."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_yaziliyor.mp4")
    cmd = [str(hf.cli()), "render", ".", "--format", "mp4", "--fps", str(int(round(fps))), "-o", str(tmp),
           "--quality", "looks", "--gpu", "--workers", "6",   # 28 workers ran out of memory
           "--frames-cache-dir", str(work / "kare_onbellegi")]   # extracted frames stay in the project folder, not %TEMP%
    log = (work / "render.log").open("w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=work, env=hf._env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    total_frames = max(1, int(total * fps))
    tail = []
    try:
        for line in proc.stdout:
            log.write(line)
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            if clean:
                tail = (tail + [clean])[-30:]
            m = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:frames|kare)?", clean)
            pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", clean)
            if m and int(m.group(2)) >= total_frames * 0.5:
                f = int(m.group(1)) / int(m.group(2))
            elif pct:
                f = float(pct.group(1)) / 100
            else:
                f = None
            if f is not None:
                el = time.time() - t0
                eta = f" · kalan ~{int(el / f * (1 - f) / 60)} dk" if f > 0.02 else ""
                progress(min(0.99, f), f"HyperFrames çiziyor: %{f * 100:.0f}{eta}")
            elif clean:
                progress(None, f"HyperFrames: {clean[:90]}")
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError("HyperFrames videoyu çizemedi: " + " | ".join(tail[-6:])[-700:])
    os.replace(tmp, out)
    return {"seconds": round(time.time() - t0), "log": str(work / "render.log")}


def make_hyperframes(proj, ctx: dict, progress) -> dict:
    from ..audio.master import master
    from ..pipeline import _main_info
    from .render import build_audio

    if not hg.available():
        raise RuntimeError("HyperFrames kurulu değil (araclar\\hyperframes)")
    work = proj.dir / "hyperframes"
    last = {"f": 0.0}

    def prog(f, m):
        if f is not None:
            last["f"] = f
        progress(last["f"], m)

    prog(0.0, "Ses hazırlanıyor (konuşma, efektler, müzik)")
    work.mkdir(parents=True, exist_ok=True)
    raw, mastered = str(work / "ses_ham.wav"), str(work / "ses_son.wav")
    build_audio(ctx["pieces"], raw)
    loud = master(raw, mastered, proj.cfg.design.get("shorts", {}).get("loudness", "I=-14:TP=-1.5:LRA=11"),
                  music=ctx.get("music"), sfx=ctx["d"]["plan"]["sfx"], speech=ctx.get("speech"))
    prog(0.03, "HyperFrames sahnesi kuruluyor")
    meta = build_project(ctx, work, mastered, lambda f, m: prog(None, m))
    out = ctx["out_dir"] / "kurgu_hyperframes.mp4"
    res = render_project(work, out, ctx["fps"], meta["duration"],
                         lambda f, m: prog(None if f is None else 0.04 + 0.95 * f, m))
    missing = ["renk düzeltme", "abone/beğeni animasyonu", "sayaç rozeti"]
    if ctx.get("ass"):
        missing.append("yakılmış altyazı")
    note = ("HyperFrames motoru: kesim listesi, zoom hedefleri ve ses Kurgu Masası'ndan; videoyu HyperFrames kurdu ve "
            f"çizdi ({meta['clips']} parça, {meta['graphics']} grafik, {res['seconds'] // 60} dk). Bu motorda yok: "
            + ", ".join(missing) + ".")
    info = _main_info(proj, ctx, out, {"engine": "hyperframes", "engine_note": note, "render": dict(res, audio=loud, kept_audio=mastered),
                                       "hyperframes": meta})
    proj.log.write("render", "hyperframes", note)
    return info
