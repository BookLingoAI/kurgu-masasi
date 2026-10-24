"""Board fixes after the first real meeting, the live meeting with a scripted model, and the room server's rules."""
import json
import time
from pathlib import Path

import pytest

from kurgu.board import agents as A
from kurgu.board import live as L
from kurgu.board import meeting as MT
from kurgu.board.measure import AGENTS

REP = {"duration_s": 748.0, "missing": [],
       "hikaye": {"metrics": {}, "flags": [{"t": 566.5, "tur": "bosluk", "zaman": "9:26.5", "aciklama": "1.7 sn sessizlik"}]},
       "ses": {"metrics": {"I_LUFS": -14.1}, "flags": []},
       **{a: {"metrics": {}, "flags": []} for a in ("grafik", "animasyon", "gecis", "renk")}}
WORDS = [{"word": w, "t_start": 58.0 + k * 0.4, "t_end": 58.3 + k * 0.4} for k, w in enumerate("bu işte zaman çok önemli".split())]


def _one(agent, action, params):
    return MT._findings(agent, {"bulgular": [{"baslik": "x", "kanit": "k", "eylem": action, "parametre": params}]}, REP, WORDS)[0]


def test_placeholder_text_is_refused():
    f = _one("grafik", "yazi_ekle", {"t": 60, "metin": "söylediği anahtar kelime (ör. 'zaman')", "tur": "vurgu"})
    assert "yer tutucu" in f["gecersiz"]


def test_text_must_have_been_said_there():
    assert _one("grafik", "yazi_ekle", {"t": 60, "metin": "zaman çok önemli", "tur": "vurgu"})["gecersiz"] is None
    assert "söylenmemiş" in _one("grafik", "yazi_ekle", {"t": 60, "metin": "para kazanmak", "tur": "vurgu"})["gecersiz"]
    assert "söylenmemiş" in _one("grafik", "yazi_ekle", {"t": 400, "metin": "zaman çok önemli", "tur": "vurgu"})["gecersiz"]


def test_absurd_numbers_are_refused():
    assert "ölçüsüz" in _one("renk", "pozlama", {"t": 720.2, "ev": -9})["gecersiz"]
    assert _one("renk", "pozlama", {"t": 720.2, "ev": -0.5})["gecersiz"] is None
    assert "ölçüsüz" in _one("grafik", "yazi_suresi", {"t": None, "sure_sn": 20})["gecersiz"]


def test_no_change_when_the_sound_is_on_target():
    assert "hedefte" in _one("ses", "ses_seviyesi", {"hedef_lufs": -14})["gecersiz"]


def test_cuts_need_measured_evidence():
    assert _one("hikaye", "kesim_ayari", {"t": 566.54, "islem": "boslugu_kes", "sn": 1.5})["gecersiz"] is None
    assert "kanıt yok" in _one("hikaye", "kesim_ayari", {"t": 300.0, "islem": "boslugu_kes", "sn": 1.5})["gecersiz"]


def test_does_no_harm_is_not_support():
    f = {"id": "GRA1", "ajan": "grafik", "baslik": "b", "kanit": "k", "eylem": "yazi_suresi", "parametre": {"t": None, "sure_sn": 5},
         "oncelik": 1, "gecersiz": None}
    votes = {"hikaye": {"GRA1": {"oy": "destek", "neden": "Okunması için yeterli süre tanır."}},
             "animasyon": {"GRA1": {"oy": "destek", "neden": "Okunabilirlik artar."}},
             "ses": {"GRA1": {"oy": "destek", "neden": "Ses mix'ine hiçbir etkisi yoktur."}},
             "renk": {"GRA1": {"oy": "destek", "neden": "Renk tutarlılığını bozmaz, zarar vermez."}}}
    r = MT.rule([f], votes)[0]
    assert r["oylar"]["ses"]["oy"] == "cekimser" and r["oylar"]["renk"]["oy"] == "cekimser"
    assert r["destek"] == 3 and r["karar"] == "kabul"          # 3 real supports pass (user rule 22.10.2026)


def test_short_last_minute_is_not_a_slow_ending():
    import numpy as np  # noqa: F401 - measure's helpers need it
    from kurgu.board import measure as M
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert "secs >= 20" in src and "60 / max(1.0, secs)" in src


def test_addressee_by_name_or_field():
    assert L.addressee("Selin, neden böyle düşünüyorsun?") == "animasyon"
    assert L.addressee("Emre sesi biraz açalım mı") == "ses"
    assert L.addressee("bu yazılar çok küçük") == "grafik"
    assert L.addressee("günaydın") is None


class LiveLLM:
    key = "sahte"

    def __init__(self):
        self.asked = []

    def see_json(self, system, user, jpegs, schema=None, **kw):
        if "Animasyon" in system:
            return {"bulgular": [{"baslik": "Yazılar donuk", "kanit": "2,5 sn donuk", "eylem": "yazi_animasyonu",
                                  "parametre": {"t": None, "stil": "nefes"}, "oncelik": 1}]}
        if "Geçiş" in system:
            return {"bulgular": [{"baslik": "Zıplayan kesim", "kanit": "1:04.8", "eylem": "atlamayi_gizle",
                                  "parametre": {"t": "1:04.8", "yontem": "punch_in"}, "oncelik": 1}]}
        return {"bulgular": []}

    def chat_json(self, system, user, schema, **kw):
        if "bulgular" in (schema.get("properties") or {}):      # a model that cannot see gets the views as text
            return self.see_json(system, user, [])
        out = []
        for fid in ("ANI1", "GEC1"):
            if fid in user:
                obj = "Hikâye" in system and fid == "GEC1"
                out.append({"id": fid, "oy": "itiraz" if obj else "destek", "neden": "vlogda normal" if obj else "izleyiciyi tutar"})
        return {"oylar": out}

    def chat(self, system, user, **kw):
        self.asked.append(user)
        return "Haklısın, notumu aldım; bundan sonra buna göre önereceğim." if "Yönetmen sana" in user else "Kanıtım ölçümde, savunuyorum."


def _project(tmp_path):
    d = tmp_path / "Videolarım" / "Deneme"
    (d / "kurul").mkdir(parents=True)
    (d / "kurul" / "olcum.json").write_text(json.dumps(dict(REP, **{a: dict(REP[a], kanit=None) for a in AGENTS})), encoding="utf-8")

    class Cfg:
        root = tmp_path

    class P:
        name, dir, cfg = "Deneme", d, Cfg
        settings = {"title": "Deneme", "options": {"tur": "vlog"}}
        log = None

        def load(self, fn, default=None):
            return {"transcript_cut": WORDS} if fn == "edit_plan.json" else default
    return P()


def test_live_meeting_speaks_debates_and_listens(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "say", lambda who, text, d: Path(f"{who}.wav"))
    proj = _project(tmp_path)
    llm = LiveLLM()
    m = L.LiveMeeting(proj, llm)
    m.user_says("Selin, bu kelime kelime animasyonu istemiyorum.")          # said before her turn: answered at the next break
    m._meeting()
    evs = m.since(-1)["events"]
    who = [e["who"] for e in evs]
    assert who[0] == "sen" and who[1] == "grafik"                              # Zeynep chairs the meeting
    selin = [e for e in evs if e["who"] == "animasyon" and e["to"] == "sen"]
    assert any("notumu aldım" in e["text"] for e in selin)                     # she answered the user
    assert all(e["audio"] for e in evs if e["who"] in AGENTS and e["text"] != "(çekimser)")
    debate = [e for e in evs if e["phase"].startswith("tartışma")]
    assert any(e["who"] == "hikaye" and e["to"] == "gecis" for e in debate)     # Derya objects to Can, looking at him
    assert any(e["who"] == "gecis" and e["to"] == "hikaye" for e in debate)     # and Can answers her
    assert (proj.dir / "kurul" / "tutanak.json").exists()
    mem = A.load_memory(tmp_path, "animasyon")
    heard = [d for d in mem["dersler"] if "istemiyorum" in d["ders"]]
    assert heard and heard[0]["tur"] == "kullanici_sesli" and heard[0]["durum"] == "aday"   # speech is only a candidate
    caps = json.loads((tmp_path / "ajanlar" / "model.json").read_text(encoding="utf-8"))
    assert caps["vision"] is False                                            # the scripted model cannot read the probe


def test_meeting_waits_while_a_video_job_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "say", lambda who, text, d: Path(f"{who}.wav"))
    calls = {"n": 0}

    def busy():
        calls["n"] += 1
        return "video.calisiyor.json" if calls["n"] <= 3 else None
    m = L.LiveMeeting(_project(tmp_path), LiveLLM(), busy=busy)
    m.poll_s = 0.01
    seen = []
    m._ask("grafik", lambda: seen.append(m.state) or "ok")
    assert calls["n"] >= 4 and seen == ["hazır"]                    # waited, then ran with the old state back


def test_room_lists_projects_and_waits_for_jobs(tmp_path):
    from kurgu.board.room import Room
    (tmp_path / "design_config.json").write_text((Path(__file__).parents[1] / "design_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    p = tmp_path / "Videolarım" / "Yemek"
    p.mkdir(parents=True)
    (p / "kurgu.json").write_text(json.dumps({"duration": 600}), encoding="utf-8")
    (p / "proje.json").write_text(json.dumps({"title": "Yemek videosu"}), encoding="utf-8")
    (tmp_path / "isler" / "sonuc").mkdir(parents=True)
    (tmp_path / "isler" / "gelen").mkdir(parents=True)
    room = Room(tmp_path)
    assert room.projects() == [{"ad": "Yemek", "baslik": "Yemek videosu", "sure_dk": 10.0, "tutanak": False}]
    assert room.job_running() is None
    (tmp_path / "isler" / "sonuc" / "abc.calisiyor.json").write_text("{}", encoding="utf-8")
    assert "video işi çalışıyor" in room.start("Yemek")["hata"]


def _root(tmp_path):
    (tmp_path / "design_config.json").write_text((Path(__file__).parents[1] / "design_config.json").read_text(encoding="utf-8"),
                                                encoding="utf-8")
    for d in ("isler/sonuc", "isler/gelen", "Videolarım"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _video(root, name, t):
    d = root / "Videolarım" / name
    (d / "cikti").mkdir(parents=True, exist_ok=True)
    (d / "cikti" / "kurgu.mp4").write_bytes(b"x")
    k = d / "kurgu.json"
    k.write_text(json.dumps({"file": str(d / "cikti" / "kurgu.mp4"), "duration": 60}), encoding="utf-8")
    import os
    os.utime(k, (t, t))


def test_watcher_starts_once_for_a_new_project_while_premiere_is_open(tmp_path, monkeypatch):
    from kurgu.board import room as R
    root = _root(tmp_path)
    _video(root, "Eski", time.time() - 3600)
    (root / "ajanlar").mkdir()
    (root / "ajanlar" / "oto.json").write_text(json.dumps({"kurulum": time.time() - 60}), encoding="utf-8")
    room = R.Room(root)
    started, windows = [], []
    monkeypatch.setattr(room, "start", lambda name: started.append(name) or {"tamam": True})
    monkeypatch.setattr(R, "open_window", lambda auto=False: windows.append(auto))
    premiere = {"open": False}
    monkeypatch.setattr(room, "premiere_running", lambda: premiere["open"])
    assert room.tick() is None                                   # the old project never starts a meeting
    _video(root, "Yemek", time.time())
    assert room.tick() is None and started == []                 # new project, but Premiere is closed: wait
    premiere["open"] = True
    (root / "isler" / "sonuc" / "x.calisiyor.json").write_text("{}", encoding="utf-8")
    assert room.tick() is None                                   # a video job is running: wait
    (root / "isler" / "sonuc" / "x.calisiyor.json").unlink()
    assert room.tick() == "Yemek" and started == ["Yemek"] and windows == [True]
    assert room.tick() is None and started == ["Yemek"]          # once per project, re-renders do not restart it
    _video(root, "Yemek", time.time() + 5)
    assert room.tick() is None


def test_install_mark_is_written_on_first_look(tmp_path):
    from kurgu.board.room import Room
    root = _root(tmp_path)
    _video(root, "Onceki", time.time() - 10)
    room = Room(root)
    assert room.due() is None and (root / "ajanlar" / "oto.json").exists()


def test_watcher_meets_without_premiere_when_premiere_could_not_open(tmp_path, monkeypatch):
    """23.10.2026: Windows blocked Premiere, so the board never met. A video whose review project failed still gets
    its one meeting."""
    from kurgu.board import room as R
    root = _root(tmp_path)
    (root / "ajanlar").mkdir()
    (root / "ajanlar" / "oto.json").write_text(json.dumps({"kurulum": time.time() - 60}), encoding="utf-8")
    _video(root, "Hotpot", time.time())
    k = root / "Videolarım" / "Hotpot" / "kurgu.json"
    info = json.loads(k.read_text(encoding="utf-8"))
    info["engine_note"] = "Video hazır; Premiere inceleme projesi kurulamadı: [WinError 4551] ..."
    k.write_text(json.dumps(info), encoding="utf-8")
    room = R.Room(root)
    started = []
    monkeypatch.setattr(room, "start", lambda name: started.append(name) or {"tamam": True})
    monkeypatch.setattr(R, "open_window", lambda auto=False: None)
    monkeypatch.setattr(room, "premiere_running", lambda: False)
    assert room.tick() == "Hotpot" and started == ["Hotpot"]
