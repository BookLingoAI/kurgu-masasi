"""A read-only look at the Premiere timeline for the board (Can's evidence) - nothing in Premiere is changed.

Through the existing bridge (integrations/premiere.py: project_info + sequence_state) it reads the project, the
review sequence ("<title> - kurgu", else the active one), its frame rate and every clip on every track. Sequence time
(start/end, where the clip sits on the timeline) and source time (in/out, which part of the recording) are kept in
separate fields and snapped to whole frames. Gaps and overlaps on V1 and a comparison with edit_plan.json
(premiere_apply.verify_sequence) become Can's flags. If the bridge panel is closed the board is told so.
"""
from __future__ import annotations

import json
from pathlib import Path


def snapshot(info: dict, state: dict) -> dict:
    fps = float(state.get("fps") or 0) or 30.0

    def fr(s) -> int:
        return int(round(float(s or 0) * fps))

    def track(t: dict) -> dict:
        clips = [{"ad": c.get("name"), "kaynak": Path(c.get("path") or "").name,
                  "sekans_kare": [fr(c["start"]), fr(c["end"])], "kaynak_kare": [fr(c["inPoint"]), fr(c["outPoint"])],
                  "sekans_sn": [round(fr(c["start"]) / fps, 3), round(fr(c["end"]) / fps, 3)],
                  "kaynak_sn": [round(fr(c["inPoint"]) / fps, 3), round(fr(c["outPoint"]) / fps, 3)],
                  "kapali": bool(c.get("disabled"))} for c in t.get("clips", [])]
        return {"kanal": t.get("index"), "ad": t.get("name"), "klip": clips}

    video = [track(t) for t in state.get("video", [])]
    audio = [track(t) for t in state.get("audio", [])]
    gaps, overlaps = [], []
    v1 = sorted(video[0]["klip"], key=lambda c: c["sekans_kare"][0]) if video else []
    for a, b in zip(v1, v1[1:]):
        d = b["sekans_kare"][0] - a["sekans_kare"][1]
        if d > 1:
            gaps.append({"sn": round(a["sekans_kare"][1] / fps, 3), "kare": d})
        elif d < 0:
            overlaps.append({"sn": round(b["sekans_kare"][0] / fps, 3), "kare": -d})
    return {"proje": info.get("name"), "surum": info.get("version"), "sekans": state.get("name"), "fps": round(fps, 3),
            "boyut": [state.get("width"), state.get("height")], "sure_sn": round(float(state.get("end") or 0), 3),
            "kanallar": {"video": len(video), "ses": len(audio)},
            "klip_sayisi": {"video": sum(len(t["klip"]) for t in video), "ses": sum(len(t["klip"]) for t in audio)},
            "kapali_klip": sum(c["kapali"] for t in video + audio for c in t["klip"]),
            "v1_bosluk": gaps[:20], "v1_cakisma": overlaps[:20], "video": video, "ses": audio}


def read(proj, timeout: float = 20.0) -> dict:
    from ..integrations.premiere import Premiere
    pr = Premiere(proj.cfg.path("bridge_dir"), timeout=timeout)
    pr.ping()
    info = pr.project_info()
    names = [s["name"] for s in info.get("sequences", [])]
    want = f"{proj.settings.get('title') or proj.name} - kurgu"
    state = pr.sequence_state(want if want in names else None)
    return snapshot(info, state)


def attach(proj, rep: dict, reader=None) -> dict:
    """Read Premiere (or say why not) into Can's evidence pack and flags; saved as kurul/premiere.json."""
    sec = rep.setdefault("gecis", {"metrics": {}, "flags": []})
    try:
        snap = (reader or read)(proj)
    except Exception as e:  # noqa: BLE001 - the meeting goes on without it
        sec["premiere"] = {"okunamadi": f"{type(e).__name__}: {str(e)[:160]}"}
        sec.setdefault("paket", {})["premiere"] = "Premiere okunamadı (köprü paneli kapalı ya da Premiere açık değil)."
        rep.setdefault("missing", []).append("Premiere zaman çizelgesi")
        return sec["premiere"]
    summary = {k: snap[k] for k in ("proje", "sekans", "fps", "boyut", "sure_sn", "kanallar", "klip_sayisi", "kapali_klip",
                                     "v1_bosluk", "v1_cakisma")}
    plan = proj.load("edit_plan.json") if hasattr(proj, "load") else None
    if plan and plan.get("segments") and plan.get("sequence"):
        try:
            from ..integrations.premiere_apply import verify_sequence
            state = {"width": snap["boyut"][0], "height": snap["boyut"][1], "fps": snap["fps"],
                     "video": [{"clips": [{"start": c["sekans_sn"][0], "end": c["sekans_sn"][1]} for c in t["klip"]]} for t in snap["video"]],
                     "audio": [{"clips": [{"start": c["sekans_sn"][0], "end": c["sekans_sn"][1]} for c in t["klip"]]} for t in snap["ses"]]}
            summary["plan_karsilastirma"] = verify_sequence(plan, state)
        except Exception as e:  # noqa: BLE001
            summary["plan_karsilastirma"] = {"hata": str(e)[:120]}
    for g in snap["v1_bosluk"][:5]:
        sec["flags"].append({"t": g["sn"], "zaman": f"{int(g['sn'] // 60)}:{g['sn'] % 60:04.1f}", "tur": "premiere_bosluk",
                             "aciklama": f"Premiere V1'de {g['kare']} karelik boşluk"})
    for o in snap["v1_cakisma"][:5]:
        sec["flags"].append({"t": o["sn"], "zaman": f"{int(o['sn'] // 60)}:{o['sn'] % 60:04.1f}", "tur": "premiere_cakisma",
                             "aciklama": f"Premiere V1'de {o['kare']} karelik üst üste binme"})
    sec["premiere"] = summary
    sec.setdefault("paket", {})["premiere"] = summary
    kdir = Path(proj.dir) / "kurul"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "premiere.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary
