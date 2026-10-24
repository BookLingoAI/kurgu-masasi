"""edit_plan.json builder: the single source of truth. Premiere only ever applies this plan."""
from __future__ import annotations

import time

from ..media.sources import CameraTimeline


def frames(t: float, fps: float) -> int:
    return int(round(t * fps))


def build_rough_plan(project: str, cam: CameraTimeline, kept: list[tuple[float, float]], removals: list[dict],
                     syncs: dict[str, dict], screens: list[dict], log) -> dict:
    fps = cam.fps or 30.0
    w, h = cam.size
    segments = []
    tl = 0  # timeline position in frames
    sid = 0
    for a, b in kept:
        for part, lin, lout in cam.split_segment(a, b):
            fin, fout = frames(lin, fps), frames(lout, fps)
            if fout - fin < 1:
                continue
            seg = {
                "id": sid,
                "cam_start": round(part.start + fin / fps, 6), "cam_end": round(part.start + fout / fps, 6),
                "file": part.path, "in": fin / fps, "out": fout / fps,
                "in_frame": fin, "out_frame": fout,
                "timeline_start_frame": tl, "timeline_end_frame": tl + (fout - fin),
                "timeline_start": tl / fps, "timeline_end": (tl + fout - fin) / fps,
                "screen": None,
            }
            sy = syncs.get(part.path)
            if sy and sy.get("screen") and sy.get("offset_s") is not None:
                s_in = lin + sy["offset_s"] + sy.get("drift", 0.0) * lin
                s_out = lout + sy["offset_s"] + sy.get("drift", 0.0) * lout
                sdur = next((s["duration"] for s in screens if s["path"] == sy["screen"]), None)
                if sdur is not None and s_in >= 0 and s_out <= sdur:
                    seg["screen"] = {"file": sy["screen"], "in": round(s_in, 6), "out": round(s_in + (fout - fin) / fps, 6)}
            segments.append(seg)
            tl += fout - fin
            sid += 1
    log.section("Kaba kurgu planı")
    for r in removals:
        log.write("rough_cut", "cut", r["reason"], rule=r["rule"], start=round(r["start"], 3), end=round(r["end"], 3),
                  words=r.get("words", []))
    log.write("rough_cut", "summary", f"{len(segments)} parça, {tl / fps:.1f} sn (kaynak {cam.duration:.1f} sn)")
    return {
        "version": 1,
        "stage": "rough",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project": project,
        "sequence": {"name": f"{project} - kaba kurgu", "width": w, "height": h, "fps": fps},
        "camera": cam.to_dict(),
        "screens": screens,
        "sync": syncs,
        "segments": segments,
        "removals": removals,
        "stats": {"source_s": cam.duration, "timeline_s": tl / fps, "segments": len(segments),
                  "removed_s": cam.duration - tl / fps,
                  "by_rule": _by_rule(removals)},
    }


def _by_rule(removals: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in removals:
        d = out.setdefault(r["rule"], {"count": 0, "seconds": 0.0})
        d["count"] += 1
        d["seconds"] = round(d["seconds"] + (r["end"] - r["start"]), 3)
    return out


def cut_words(words: list[dict], segments: list[dict]) -> list[dict]:
    """Map kept source words onto the cut timeline (word fully inside a kept segment)."""
    out = []
    for w in words:
        for s in segments:
            if s["cam_start"] - 0.02 <= w["start"] and w["end"] <= s["cam_end"] + 0.02:
                off = s["timeline_start"] - s["cam_start"]
                out.append({**w, "t_start": round(w["start"] + off, 3), "t_end": round(w["end"] + off, 3), "seg": s["id"]})
                break
    return out


def resnap(plan: dict, fps: float) -> dict:
    """Re-snap every segment to the sequence's real frame rate (VFR phone footage is conformed by Premiere)."""
    if abs(plan["sequence"]["fps"] - fps) < 1e-6:
        return plan
    tl = 0
    out = []
    for s in plan["segments"]:
        fin, fout = frames(s["in"], fps), frames(s["out"], fps)
        if fout - fin < 1:
            continue
        part_start = s["cam_start"] - s["in"]
        s = dict(s, in_frame=fin, out_frame=fout, **{"in": fin / fps, "out": fout / fps},
                 cam_start=round(part_start + fin / fps, 6), cam_end=round(part_start + fout / fps, 6),
                 timeline_start_frame=tl, timeline_end_frame=tl + fout - fin,
                 timeline_start=tl / fps, timeline_end=(tl + fout - fin) / fps)
        if s.get("screen"):
            s["screen"] = dict(s["screen"], out=round(s["screen"]["in"] + (fout - fin) / fps, 6))
        out.append(s)
        tl += fout - fin
    for i, s in enumerate(out):
        s["id"] = i
    plan = dict(plan, segments=out)
    plan["sequence"] = dict(plan["sequence"], fps=fps, source_fps=plan["sequence"]["fps"])
    plan["stats"] = dict(plan["stats"], timeline_s=tl / fps, segments=len(out),
                         removed_s=plan["stats"]["source_s"] - tl / fps)
    if plan.get("transcript_cut") is not None:
        words = [{k: v for k, v in w.items() if k not in ("t_start", "t_end", "seg")} for w in plan["transcript_cut"]]
        plan["transcript_cut"] = cut_words(words, out)
    return plan
