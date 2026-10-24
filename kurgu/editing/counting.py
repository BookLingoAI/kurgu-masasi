"""Counting out loud ("bir, iki, üç ... yirmi" while the meat cooks): a long, slow stretch that is funny when it is
sped up. The run of numbers is found in the cut transcript, the timeline is re-timed so that stretch plays faster
and faster (2x -> 3x -> 4x) with a pitched-up "helium" voice (render: Piece.speed), and a big counter is shown.

Runs before everything that places things on the timeline (montage, director, graphics, captions), so they all see
the new times.
"""
from __future__ import annotations

from ..analysis.text import norm_word

UNITS = {"bir": 1, "iki": 2, "üç": 3, "uc": 3, "dört": 4, "dort": 4, "beş": 5, "bes": 5, "altı": 6, "alti": 6,
         "yedi": 7, "sekiz": 8, "dokuz": 9}
TENS = {"on": 10, "yirmi": 20, "otuz": 30}
COMPOUND = {f"{t}{u}": tv + uv for t, tv in TENS.items() for u, uv in UNITS.items()}   # "onbir", "yirmibeş"
RAMP = ((2.0, 1.45), (3.0, 1.65), (4.0, 1.85))    # (speed, pitch factor) for the three thirds of the run


def _value(tok: str) -> int | None:
    t = tok.strip(".,!?;:…'\"-").lower()
    if t.isdigit():
        v = int(t)
        return v if 0 < v <= 40 else None
    t = norm_word(t)
    return UNITS.get(t) or TENS.get(t) or COMPOUND.get(t)


def numbers(words: list[dict]) -> list[tuple[int, int, int]]:
    """[(first_word_index, last_word_index, value)]: "on" + "bir" is one number (11)."""
    out = []
    i = 0
    while i < len(words):
        v = _value(words[i]["word"])
        if v is None:
            i += 1
            continue
        if v in (10, 20, 30) and i + 1 < len(words):
            u = _value(words[i + 1]["word"])
            if u is not None and u < 10 and words[i + 1]["t_start"] - words[i]["t_end"] < 0.9:
                out.append((i, i + 1, v + u))
                i += 2
                continue
        out.append((i, i, v))
        i += 1
    return out


def find_counts(words: list[dict], min_len: int = 6, min_last: int = 8, max_gap_s: float = 3.0) -> list[dict]:
    """Runs of counting: values going up one by one (one skipped or repeated number is forgiven), at most one other
    word between two numbers. [{"t0", "t1", "marks": [(t, value)], "i0", "i1"}] in timeline seconds."""
    nums = numbers(words)
    runs, cur = [], []
    for n in nums:
        if cur:
            pi0, pi1, pv = cur[-1]
            gap_words = n[0] - pi1 - 1
            gap_t = words[n[0]]["t_start"] - words[pi1]["t_end"]
            if n[2] in (pv + 1, pv + 2) and gap_words <= 1 and gap_t <= max_gap_s:
                cur.append(n)
                continue
            if n[2] == pv and gap_words <= 1 and gap_t <= max_gap_s:
                continue
            if len(cur) >= min_len and cur[-1][2] >= min_last:
                runs.append(cur)
            cur = []
        cur = [n] if n[2] <= 3 else []    # a count starts at the beginning
    if len(cur) >= min_len and cur[-1][2] >= min_last:
        runs.append(cur)
    out = []
    for r in runs:
        t0, t1 = words[r[0][0]]["t_start"] - 0.08, words[r[-1][1]]["t_end"] + 0.12
        if t1 - t0 < 3.0:     # already quick: nothing to speed up
            continue
        out.append({"t0": round(t0, 3), "t1": round(t1, 3), "i0": r[0][0], "i1": r[-1][1],
                    "marks": [(round(words[a]["t_start"], 3), v) for a, _, v in r]})
    return out


def zones_of(run: dict) -> list[tuple[float, float, float, float]]:
    """(t0, t1, speed, pitch) - the run in thirds, faster and faster."""
    t0, t1 = run["t0"], run["t1"]
    step = (t1 - t0) / len(RAMP)
    return [(t0 + k * step, t0 + (k + 1) * step, sp, pt) for k, (sp, pt) in enumerate(RAMP)]


def apply_speed(pl: dict, zones: list[tuple[float, float, float, float]], fps: float) -> tuple[dict, callable]:
    """Split segments at the zone edges, give the parts inside a zone that speed, re-time the whole timeline and
    the cut transcript. Returns (new plan, old-time -> new-time function). Zones must not overlap."""
    zones = sorted(zones)
    new_segs, mapping = [], []      # mapping: (old_a, old_b, new_a, speed)
    t_new = 0.0
    for s in pl["segments"]:
        a0, a1 = s["timeline_start"], s["timeline_end"]
        fixed = s.get("kind") == "montage" or float(s.get("speed", 1.0)) != 1.0
        cuts = [] if fixed else sorted({c for z in zones for c in (z[0], z[1]) if a0 + 1e-3 < c < a1 - 1e-3})
        edges = [a0] + cuts + [a1]
        src = s["in"]
        for a, b in zip(edges[:-1], edges[1:]):
            mid = (a + b) / 2
            z = None if fixed else next((z for z in zones if z[0] <= mid < z[1]), None)
            sp = z[2] if z else 1.0
            n_out = max(1, round((b - a) * fps / sp))
            src_len = n_out * sp / fps        # whole output frames (the render counts frames per piece)
            if b == a1:      # the last part ends exactly where the segment did in the source
                src_len = s["out"] - src
                n_out = max(1, round(src_len * fps / sp))
            part = dict(s, **{"in": round(src, 6), "out": round(src + src_len, 6)},
                        timeline_start=round(t_new, 6), timeline_end=round(t_new + n_out / fps, 6),
                        timeline_start_frame=round(t_new * fps), timeline_end_frame=round(t_new * fps) + n_out)
            part["cam_start"] = round(s.get("cam_start", s["in"]) + (src - s["in"]), 6)
            part["cam_end"] = round(part["cam_start"] + src_len, 6)
            if z:
                part["speed"], part["pitch"] = sp, z[3]
            if s.get("screen"):
                sin = s["screen"]["in"] + (src - s["in"])
                part["screen"] = dict(s["screen"], **{"in": round(sin, 6), "out": round(sin + src_len, 6)})
            mapping.append((a, b, t_new, (b - a) / (n_out / fps)))
            new_segs.append(part)
            src += src_len
            t_new += n_out / fps
    for i, s in enumerate(new_segs):
        s["id"] = i

    def remap(t: float) -> float:
        for a, b, na, k in mapping:
            if a - 1e-6 <= t <= b + 1e-6:
                return na + (t - a) / k
        return mapping[-1][2] + (mapping[-1][1] - mapping[-1][0]) / mapping[-1][3] + (t - mapping[-1][1]) if mapping else t

    words = []
    for w in pl.get("transcript_cut") or []:
        w2 = dict(w, t_start=round(remap(w["t_start"]), 3), t_end=round(remap(w["t_end"]), 3))
        if any(z[0] <= (w["t_start"] + w["t_end"]) / 2 < z[1] for z in zones):
            w2["fast"] = True
        words.append(w2)
    out = dict(pl, segments=new_segs, transcript_cut=words)
    out["stats"] = dict(pl["stats"], timeline_s=round(t_new, 6), segments=len(new_segs))
    return out, remap
