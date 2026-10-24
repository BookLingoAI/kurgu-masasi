"""Review board meeting (kurgu/board/meeting.py, agents.py) with a scripted model, plus the counter-sound fix."""
import inspect
import json
from pathlib import Path

from kurgu.board import agents as A
from kurgu.board import meeting as MT
from kurgu.board.measure import AGENTS


def _f(fid, agent, action, params, prio=1):
    return {"id": fid, "ajan": agent, "baslik": fid, "kanit": "-", "eylem": action, "parametre": params,
            "oncelik": prio, "gecersiz": None}


def _votes(fid, **who):
    return {voter: {fid: {"oy": oy, "neden": "."}} for voter, oy in who.items()}


def _merge(*vs):
    out = {}
    for v in vs:
        for voter, d in v.items():
            out.setdefault(voter, {}).update(d)
    return out


def test_referee_passes_four_supports_and_respects_the_owner_veto():
    a = _f("ANI1", "animasyon", "yazi_animasyonu", {"t": None, "stil": "kelime_kelime"})
    b = _f("HIK1", "hikaye", "efekt_seviyesi", {"t": None, "db": 6})          # an audio action from outside audio
    c = _f("GEC1", "gecis", "gecis_ekle", {"t": 30.0, "tur": "whip"})
    votes = _merge(_votes("ANI1", grafik="destek", gecis="destek", hikaye="destek", ses="cekimser", renk="itiraz"),
                   _votes("HIK1", grafik="destek", gecis="destek", animasyon="destek", renk="destek", ses="itiraz"),
                   _votes("GEC1", grafik="destek", animasyon="destek", hikaye="itiraz", ses="itiraz", renk="destek"))
    r = {f["id"]: f for f in MT.rule([a, b, c], votes)}
    assert r["ANI1"]["karar"] == "kabul" and r["ANI1"]["destek"] == 4 and r["ANI1"]["itiraz"] == 1
    assert r["HIK1"]["karar"] == "veto"                   # 5 supports, but the mix-master objects to its own field
    assert r["GEC1"]["karar"] == "red"                    # two objections


def test_referee_keeps_the_better_supported_of_two_clashing_proposals():
    a = _f("GEC1", "gecis", "gecis_ekle", {"t": 30.0, "tur": "whip"})
    b = _f("GEC2", "gecis", "gecis_ekle", {"t": 30.4, "tur": "zoom"})
    all5 = {v: "destek" for v in AGENTS if v != "gecis"}
    four = dict(all5, renk="cekimser")
    r = {f["id"]: f for f in MT.rule([a, b], _merge(_votes("GEC1", **four), _votes("GEC2", **all5)))}
    assert r["GEC2"]["karar"] == "kabul" and r["GEC1"]["karar"] == "cakisma"


def test_developer_requests_are_set_apart():
    f = _f("ANI2", "animasyon", "gelistirici", {"istek": "yazı beklerken nefes alsın"})
    r = MT.rule([f], _votes("ANI2", **{v: "destek" for v in AGENTS if v != "animasyon"}))
    assert r[0]["karar"] == "gelistirici_isi"


def test_findings_are_validated():
    raw = {"bulgular": [
        {"baslik": "a", "kanit": "k", "eylem": "gecis_ekle", "parametre": {"t": "1:04.8", "tur": "zoom"}, "oncelik": 1},
        {"baslik": "b", "kanit": "k", "eylem": "renk_esitle", "parametre": {"t": 5}},          # not its field
        {"baslik": "c", "kanit": "k", "eylem": "sihir_yap", "parametre": {}},                    # not an action
        {"baslik": "d", "kanit": "k", "eylem": "secenek", "parametre": {"ad": "ruh_hali", "deger": 1}},
        {"baslik": "e", "kanit": "k", "eylem": "secenek", "parametre": {"ad": "intensity", "deger": "high"}}]}
    got = MT._findings("gecis", raw)
    assert len(got) == 4 and got[0]["parametre"]["t"] == 64.8 and got[0]["gecersiz"] is None
    assert "renk" in got[1]["gecersiz"] and "bilinmeyen eylem" in got[2]["gecersiz"] and "seçenek" in got[3]["gecersiz"]


class ScriptedLLM:
    key = "sahte"

    def __init__(self):
        self.seen = []

    def see_json(self, system, user, jpegs, schema=None, **kw):
        self.seen.append(("see", system[:60], len(jpegs)))
        if "Animasyon" in system:
            return {"bulgular": [{"baslik": "Yazılar 2,5 sn donuk", "kanit": "0:03.2 donuk", "eylem": "yazi_animasyonu",
                                  "parametre": {"t": None, "stil": "nefes"}, "oncelik": 1}]}
        if "Ses" in system:
            raise TimeoutError("sessiz")
        return {"bulgular": []}

    def chat_json(self, system, user, schema, **kw):
        self.seen.append(("vote", system[:60], 0))
        return {"oylar": [{"id": "ANI1", "oy": "destek", "neden": "donukluk izleyiciyi kaçırır"}]}


def _project(tmp_path):
    d = tmp_path / "projeler" / "Deneme"
    (d / "kurul").mkdir(parents=True)
    rep = {"duration_s": 120.0, "missing": [], **{a: {"metrics": {"x": 1}, "flags": [], "n_flags": 0} for a in AGENTS}}
    (d / "kurul" / "olcum.json").write_text(json.dumps(rep), encoding="utf-8")

    class Cfg:
        root = tmp_path

    class P:
        name, dir, cfg = "Deneme", d, Cfg
        settings = {"options": {"tur": "hikaye", "intensity": "high", "captions": False}}

        def load(self, fn, default=None):
            return default
    return P()


def test_meeting_end_to_end_with_a_scripted_model(tmp_path):
    proj = _project(tmp_path)
    llm = ScriptedLLM()
    m = MT.hold_meeting(proj, llm)
    assert [s[0] for s in llm.seen].count("see") == 6
    assert m["turlar"]["ses"]["durum"].startswith("cevap yok")          # a silent agent only loses its turn
    f = m["bulgular"][0]
    assert f["id"] == "ANI1" and f["karar"] == "kabul" and f["destek"] == 6   # the silent one still votes
    md = (proj.dir / "kurul" / "tutanak.md").read_text(encoding="utf-8")
    assert "KABUL" in md and "Yazılar 2,5 sn donuk" in md and "sadece öner" in md
    mem = A.load_memory(tmp_path, "animasyon")
    assert mem["toplanti"] == 1 and mem["istatistik"]["yazi_animasyonu"] == {"oneri": 1, "kurul_kabul": 1,
                                                                             "kullanici_onay": 0, "kullanici_ret": 0}
    assert any("donuk" in d["ders"] for d in mem["dersler"])            # seeded from the user's feedback


def test_memory_lessons_and_track_record(tmp_path):
    mem = A.load_memory(tmp_path, "ses")
    assert any("Nefes" in d["ders"] for d in mem["dersler"])
    A.add_lesson(mem, "Efektleri konuşmanın 6 dB altında tut.", "kullanıcı")
    A.add_lesson(mem, "Efektleri konuşmanın 6 dB altında tut.", "kullanıcı")
    A.record(mem, "efekt_seviyesi", "oneri")
    A.record(mem, "efekt_seviyesi", "kullanici_ret")
    A.save_memory(tmp_path, "ses", mem)
    back = A.load_memory(tmp_path, "ses")
    assert sum("6 dB" in d["ders"] for d in back["dersler"]) == 1
    assert "efekt_seviyesi: 1 öneri" in A.track_record(back) and "1 ret" in A.track_record(back)


def test_counter_sounds_are_merged_after_they_are_added():
    """The counter's ticks were appended to sfx_extra after the merge into the plan and never played."""
    from kurgu import pipeline
    src = inspect.getsource(pipeline._prepare_main)
    merge = src.index('d["plan"]["sfx"] = sorted(d["plan"]["sfx"] + sfx_extra')
    assert src.rindex("sfx_extra.append(") < merge
    assert Path(inspect.getsourcefile(pipeline)).exists()


def test_three_supports_and_one_objection_pass_but_the_owner_still_vetoes():
    """User rule (22.10.2026): at least 3 supports (proposer included), at most 1 objection, owner veto kept."""
    a = _f("ANI1", "animasyon", "yazi_animasyonu", {"t": None, "stil": "nefes"})
    b = _f("GEC1", "gecis", "gecis_ekle", {"t": 30.0, "tur": "zoom"})
    c = _f("HIK1", "hikaye", "efekt_seviyesi", {"t": None, "db": -3})          # an audio action from outside audio
    votes = _merge(_votes("ANI1", grafik="destek", gecis="destek", hikaye="itiraz", ses="cekimser", renk="cekimser"),
                   _votes("GEC1", grafik="destek", animasyon="cekimser", hikaye="cekimser", ses="cekimser", renk="cekimser"),
                   _votes("HIK1", grafik="destek", gecis="destek", animasyon="cekimser", renk="cekimser", ses="itiraz"))
    r = {f["id"]: f for f in MT.rule([a, b, c], votes)}
    assert MT.PASS_SUPPORT == 3
    assert r["ANI1"]["karar"] == "kabul" and (r["ANI1"]["destek"], r["ANI1"]["itiraz"]) == (3, 1)
    assert r["GEC1"]["karar"] == "red" and r["GEC1"]["destek"] == 2
    assert r["HIK1"]["karar"] == "veto"                                          # 3 supports, but Emre owns audio
