"""Replay of a held meeting in the board room (kurgu/web/kurul.html?demo=...): every line is what the agents
actually wrote in the minutes (findings, evidence, vote reasons), spoken in each agent's own voice."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import agents as A
from .measure import AGENTS
from .voice import VOICES, say


def speakable(text: str) -> str:
    """Times, decimals and code-ish bits read aloud naturally in Turkish."""
    t = re.sub(r"`[^`]*`", "", text)
    t = re.sub(r"\b(\d+):(\d{2})(?:\.\d+)?\b",
               lambda m: (f"{int(m.group(1))} dakika " if int(m.group(1)) else "") + f"{int(m.group(2))}. saniye", t)
    t = re.sub(r"(\d+)\.(\d+)", r"\1 virgül \2", t)
    t = t.replace("_", " ").replace("LUFS", "lufs")
    t = re.sub(r"(?<!\w)sn(?!\w)", "saniye", re.sub(r"(?<!\w)dk(?!\w)", "dakika", t))
    t = re.sub(r"[{}\[\]\"]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _short(text: str, n: int = 170) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    return cut[:cut.rfind(" ")] + "…"


def build(proj_dir: Path, out_dir: Path, max_debates: int = 3) -> Path:
    m = json.loads((proj_dir / "kurul" / "tutanak.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    flow = []

    def line(who, text, to=None, phase="", **extra):
        shown = _short(text)
        wav = say(who, speakable(shown), out_dir)
        flow.append(dict({"who": who, "to": to, "text": shown, "audio": wav.name, "phase": phase}, **extra))

    line("hakem", f"Kurgu Kurulu toplantısı başlıyor. Konu: {m['proje']}. Her üye önce kendi bulgusunu anlatacak.",
         to="sen", phase="açılış")
    findings = [f for f in m["bulgular"] if f["karar"] != "gecersiz"]
    for a in AGENTS:
        own = [f for f in findings if f["ajan"] == a]
        if not own:
            continue
        f = sorted(own, key=lambda f: f["oncelik"])[0]
        action = A.ACTIONS.get(f["eylem"], ("", f["eylem"]))[1]
        line(a, f"{f['baslik']}. {f['kanit']} Önerim: {action}.", to="sen", phase=f"görüşler · {VOICES[a]['isim']}")
    # the debates: the findings with the most disagreement first, voters answer the proposer face to face
    debated = sorted(findings, key=lambda f: (-(f["itiraz"] > 0), -f["itiraz"], -f["destek"]))[:max_debates]
    for f in debated:
        name = VOICES[f["ajan"]]["isim"]
        line(f["ajan"], f"{f['baslik']} konusunda ne düşünüyorsunuz?", to=None, phase=f"tartışma · {name}", clearVotes=True)
        order = sorted((v for v in f["oylar"] if v != f["ajan"]), key=lambda v: {"itiraz": 0, "destek": 1}.get(f["oylar"][v]["oy"], 2))
        for voter in order:
            v = f["oylar"][voter]
            if v["oy"] == "cekimser" or not v["neden"]:
                flow.append({"who": voter, "to": f["ajan"], "text": "(çekimser)", "audio": None, "phase": f"tartışma · {name}",
                             "votes": {voter: "cekimser"}, "pause": 150})
                continue
            line(voter, f"{name}, {v['neden']}", to=f["ajan"], phase=f"tartışma · {name}", votes={voter: v["oy"]})
        verdict = {"kabul": "kabul edildi", "red": "reddedildi", "veto": "alan sahibinin itirazıyla durduruldu",
                   "cakisma": "başka bir öneriyle çakıştı", "gelistirici_isi": "geliştiriciye iletilecek"}[f["karar"]]
        line("hakem", f"{f['destek']} destek, {f['itiraz']} itiraz. Öneri {verdict}.", to="sen", phase="karar")
    passed = [f for f in findings if f["karar"] in ("kabul", "gelistirici_isi")]
    titles = "; ".join(f["baslik"] for f in passed[:4])
    line("hakem", f"Toplam {len(passed)} öneri kabul edildi: {titles}. Bunlar senin onayını bekliyor; hiçbiri uygulanmadı.",
         to="sen", phase="kararlar", clearVotes=True)
    script = {"proje": m["proje"], "tarih": m["tarih"], "akis": flow}
    p = out_dir / "akis.json"
    p.write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
