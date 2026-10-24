"""Apply edit_plan.json to Premiere and read the result back for verification.

Known MCP tool problems are avoided by doing everything through execute_extendscript-style scripts:
- audio level tool writes dB into the raw scale (mutes the clip)  -> we never use it; levels set via raw gain math
- sequence size tool has no effect                              -> sequence is created FROM the camera clip
- work-area export is ignored                                    -> exports use explicit in/out on the sequence
"""
from __future__ import annotations

from typing import Callable

from .premiere import Premiere, js_str, js_val

BATCH = 40


def apply_rough_plan(pr: Premiere, plan: dict, progress: Callable[[float, str], None] | None = None,
                     replace: bool = True, on_plan: Callable[[dict], None] | None = None) -> dict:
    seq = plan["sequence"]
    cam_files = [p["path"] for p in plan["camera"]["parts"]]
    scr_files = sorted({s["screen"]["file"] for s in plan["segments"] if s.get("screen")})
    name = seq["name"]

    def say(f, m):
        if progress:
            progress(f, m)

    say(0.02, "Dosyalar Premiere'e alınıyor")
    pr.run(f"""
        KL.importFiles({js_val(cam_files)}, "Kamera");
        var s = {js_val(scr_files)};
        if (s.length) KL.importFiles(s, "Ekran");
        return KL.ok(true);""", "import", 600)

    say(0.08, "Sekans oluşturuluyor (kamera biçimiyle)")
    info = pr.run(f"""
        var name = {js_str(name)};
        var old = KL.findSequence(name);
        if (old) {{
            if (!{str(replace).lower()}) throw new Error('Sekans zaten var: ' + name);
            app.project.deleteSequence(old);
        }}
        var item = KL.findItemByPath({js_str(cam_files[0])});
        var seq = KL.sequenceFromClip(name, item, "Sekanslar");
        app.project.activeSequence = seq;
        return KL.ok({{w: seq.frameSizeHorizontal, h: seq.frameSizeVertical, tb: seq.timebase}});""", "seq", 120)

    # use the frame rate Premiere actually gave the sequence (VFR sources get conformed)
    from ..editing.plan import resnap
    seq_fps = 254016000000 / float(info["tb"])
    plan = resnap(plan, seq_fps)
    if on_plan:
        on_plan(plan)
    segs = plan["segments"]
    fps = plan["sequence"]["fps"]
    for k in range(0, len(segs), BATCH):
        batch = [{"f": s["file"], "i": s["in_frame"] / fps, "o": s["out_frame"] / fps, "t": s["timeline_start_frame"] / fps,
                  "sf": s["screen"]["file"] if s.get("screen") else None,
                  "si": s["screen"]["in"] if s.get("screen") else None,
                  "so": s["screen"]["out"] if s.get("screen") else None} for s in segs[k:k + BATCH]]
        say(0.1 + 0.8 * k / max(1, len(segs)), f"Kesimler yerleştiriliyor {k}/{len(segs)}")
        pr.run(f"""
            var seq = KL.findSequence({js_str(name)});
            var b = {js_val(batch)};
            var cache = {{}};
            function item(p) {{ if (!cache[p]) cache[p] = KL.findItemByPath(p); return cache[p]; }}
            var n = 0;
            for (var i = 0; i < b.length; i++) {{
                var it = item(b[i].f);
                KL.placeSegment(seq, it, b[i].i, b[i].o, b[i].t, 0);
                if (b[i].sf) {{
                    var si = item(b[i].sf);
                    KL.placeSegment(seq, si, b[i].si, b[i].so, b[i].t, 1);
                }}
                n++;
            }}
            for (var p in cache) if (cache.hasOwnProperty(p)) KL.resetInOut(cache[p]);
            return KL.ok(n);""", f"place{k}", 600)

    say(0.92, "Ekran kaydı sesi kaldırılıyor, ekran izi gizleniyor")
    pr.run(f"""
        var seq = KL.findSequence({js_str(name)});
        // screen recording audio lands on A2: remove it (camera mic is the voice track)
        if (seq.audioTracks.numTracks > 1) {{
            var a2 = seq.audioTracks[1];
            for (var j = a2.clips.numItems - 1; j >= 0; j--) a2.clips[j].remove(false, false);
        }}
        // rough cut: screen on V2 is placed in sync but disabled until demo decisions are made
        var dis = 0;
        if (seq.videoTracks.numTracks > 1) {{
            var v2 = seq.videoTracks[1];
            for (var k = 0; k < v2.clips.numItems; k++) {{ try {{ v2.clips[k].disabled = true; dis++; }} catch (e) {{}} }}
        }}
        return KL.ok(dis);""", "post", 300)
    say(0.96, "Premiere'den geri okunuyor")
    state = pr.sequence_state(name)
    report = verify_sequence(plan, state)
    say(1.0, "Kaba kurgu Premiere'de")
    return report


def verify_sequence(plan: dict, state: dict) -> dict:
    """Compare what Premiere actually has with the plan (1 frame tolerance)."""
    fps = plan["sequence"]["fps"]
    tol = 1.01 / fps
    problems = []
    v1 = state["video"][0]["clips"] if state["video"] else []
    a1 = state["audio"][0]["clips"] if state["audio"] else []
    segs = plan["segments"]
    if state.get("width") != plan["sequence"]["width"] or state.get("height") != plan["sequence"]["height"]:
        problems.append(f"Sekans boyutu {state.get('width')}x{state.get('height')}, beklenen "
                        f"{plan['sequence']['width']}x{plan['sequence']['height']}")
    if state.get("fps") and abs(state["fps"] - fps) > 0.01:
        problems.append(f"Sekans kare hızı {state['fps']:.3f}, beklenen {fps:.3f}")
    if len(v1) != len(segs):
        problems.append(f"V1'de {len(v1)} klip var, planda {len(segs)} parça")
    if len(a1) != len(segs):
        problems.append(f"A1'de {len(a1)} klip var, planda {len(segs)} parça")
    for s, c in zip(segs, v1):
        if abs(c["start"] - s["timeline_start"]) > tol or abs(c["end"] - s["timeline_end"]) > tol:
            problems.append(f"Parça {s['id']}: zaman çizelgesinde {c['start']:.3f}-{c['end']:.3f}, plan "
                            f"{s['timeline_start']:.3f}-{s['timeline_end']:.3f}")
        if abs(c["inPoint"] - s["in"]) > tol:
            problems.append(f"Parça {s['id']}: kaynak giriş {c['inPoint']:.3f}, plan {s['in']:.3f}")
        if len(problems) > 30:
            break
    n_screen = sum(1 for s in segs if s.get("screen"))
    v2 = state["video"][1]["clips"] if len(state["video"]) > 1 else []
    if n_screen and len(v2) != n_screen:
        problems.append(f"V2'de {len(v2)} ekran klibi, planda {n_screen}")
    a2 = state["audio"][1]["clips"] if len(state["audio"]) > 1 else []
    if a2:
        problems.append(f"A2'de {len(a2)} ekran sesi klibi kaldı (çift ses)")
    return {"ok": not problems, "problems": problems, "v1": len(v1), "a1": len(a1), "v2": len(v2),
            "sequence_end": state.get("end"), "plan_end": plan["stats"]["timeline_s"]}
