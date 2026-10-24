"""Expert board pieces: knowledge base, model discovery/capability test, structured memory, self-check, read-only
Premiere view, preview/apply/undo and voice choice. No local model, Premiere or GPU needed (scripted stand-ins)."""
import json
import os
from pathlib import Path

import pytest

from kurgu.board import agents as A
from kurgu.board import apply as AP
from kurgu.board import bilgi, model as MD
from kurgu.board import meeting as MT
from kurgu.board import premiere_view as PV


def test_checklists_put_user_rules_first_and_name_their_sources():
    txt = bilgi.checklist("ses")
    lines = txt.splitlines()
    assert lines[0].startswith("- [KULLANICININ KURALI]")
    assert "-23 LUFS" in txt and "[STANDART]" in txt and "YAYGIN UYGULAMA (doğrulanmamış)" in txt
    assert "https://tech.ebu.ch/docs/r/r128-2014.pdf" in bilgi.sources("ses")
    assert all(bilgi.checklist(a) for a in ("hikaye", "grafik", "animasyon", "gecis", "ses", "renk"))


def test_model_choice_prefers_a_loaded_vision_model():
    models = [{"id": "embed-x", "type": "embeddings", "state": "loaded"},
              {"id": "qwen-text", "type": "llm", "state": "loaded"},
              {"id": "qwen-vl", "type": "vlm", "state": "loaded"},
              {"id": "other-vl", "type": "vlm", "state": "not-loaded"}]
    assert MD.choose({}, models)["id"] == "qwen-vl"
    assert MD.choose({"model": "qwen-text"}, models)["id"] == "qwen-text"
    assert MD.choose({}, [models[0]]) is None


class ProbeLLM:
    key = "vl"

    def __init__(self, sees=True):
        self.sees, self.calls = sees, 0

    def chat_json(self, system, user, schema, **kw):
        self.calls += 1
        return {"tamam": True, "sayi": 3}

    def see_json(self, system, user, jpegs, schema=None, **kw):
        self.calls += 1
        assert jpegs and jpegs[0][:2] == b"\xff\xd8"             # a real JPEG probe picture
        if not self.sees:
            raise RuntimeError("model görüntü almıyor")
        return {"metin": "KURUL 47"}


def test_capabilities_are_tested_once_and_cached(tmp_path):
    llm = ProbeLLM()
    c = MD.capabilities(llm, tmp_path)
    assert c["json"] and c["vision"] and llm.calls == 2
    assert MD.capabilities(llm, tmp_path)["vision"] and llm.calls == 2      # from ajanlar/model.json
    blind = ProbeLLM(sees=False)
    blind.key = "text-only"
    c2 = MD.capabilities(blind, tmp_path)
    assert c2["json"] and not c2["vision"] and "vision_hata" in c2


def test_structured_memory_kinds_candidates_and_pruning(tmp_path):
    mem = A.load_memory(tmp_path, "ses")
    assert all(d["tur"] == "kullanici" and d["durum"] == "aktif" for d in mem["dersler"])       # seeded user rules
    A.add_lesson(mem, "Kullanıcı karşı çıktı: \"efekt istemiyorum\"", "kullanici_sesli", scope="proje:Yemek")
    cand = mem["dersler"][-1]
    assert cand["durum"] == "aday" and cand["guven"] == 0.5 and cand["kapsam"] == "proje:Yemek"
    assert "SESLİ, ONAY BEKLİYOR" in A.lessons_text(mem)
    A.confirm_lesson(mem, "Kullanıcı karşı çıktı: \"efekt istemiyorum\" - düğmeyle", approve=False)
    assert cand["durum"] == "eski" and mem["dersler"][-1]["tur"] == "kullanici"
    for i in range(40):
        A.add_lesson(mem, f"ajan notu {i}", "ajan")
    A.save_memory(tmp_path, "ses", mem)
    back = A.load_memory(tmp_path, "ses")
    soft = [d for d in back["dersler"] if d["tur"] != "kullanici"]
    assert len(soft) <= A.MAX_SOFT and soft[-1]["ders"] == "ajan notu 39"
    assert sum(d["tur"] == "kullanici" for d in back["dersler"]) == len(A.SEED_LESSONS["ses"]) + 1     # never cut


def test_old_memory_format_is_read():
    d = A._norm_lesson({"ders": "x", "kaynak": "kullanıcı (sesli)", "t": "2026-10-22 13:00"})
    assert d["tur"] == "kullanici_sesli" and d["durum"] == "aday" and d["tarih"] == "2026-10-22 13:00"
    assert A._norm_lesson({"ders": "y", "kaynak": "kullanıcı (CLAUDE.md)", "t": "başlangıç"})["tur"] == "kullanici"


def _finding(agent, action, params, **extra):
    raw = {"bulgular": [dict({"baslik": "b", "kanit": "k", "eylem": action, "parametre": params}, **extra)]}
    return MT._findings(agent, raw, {"duration_s": 600, "ses": {"metrics": {}}, "hikaye": {"flags": []}}, [])[0]


def test_weak_evidence_abstains_and_user_rules_hold():
    assert "düşük güven" in _finding("animasyon", "yazi_animasyonu", {"t": None, "stil": "nefes"}, guven=0.3)["gecersiz"]
    assert _finding("animasyon", "yazi_animasyonu", {"t": None, "stil": "nefes"}, guven="80%")["gecersiz"] is None
    f = _finding("renk", "pozlama", {"t": 30, "ev": 0.5}, guven=0.9, yorum="S-eğrisi ile kontrastı artıralım")
    assert "kalıcı tercihine aykırı" in f["gecersiz"]
    g = _finding("gecis", "gecis_ekle", {"t": 30, "tur": "zoom"}, guven=0.8, olcum="benzerlik 0.96", gozlem="aynı kadraj")
    assert g["gecersiz"] is None and g["olcum"] == "benzerlik 0.96" and g["guven"] == 0.8


def _state():
    return {"name": "Yemek - kurgu", "fps": 29.97, "width": 1920, "height": 1080, "end": 10.0,
            "video": [{"index": 0, "name": "V1", "clips": [
                {"name": "a", "start": 0.0, "end": 3.0, "inPoint": 10.0, "outPoint": 13.0, "path": "C:/k/a.mp4"},
                {"name": "b", "start": 3.5, "end": 6.0, "inPoint": 20.0, "outPoint": 22.5, "path": "C:/k/a.mp4"},
                {"name": "c", "start": 5.9, "end": 9.0, "inPoint": 30.0, "outPoint": 33.1, "path": "C:/k/a.mp4", "disabled": True}]}],
            "audio": [{"index": 0, "name": "A1", "clips": []}]}


def test_premiere_snapshot_keeps_sequence_and_source_time_apart_on_frames():
    s = PV.snapshot({"name": "Yemek.prproj", "version": "26.5.2"}, _state())
    c = s["video"][0]["klip"][1]
    assert c["sekans_kare"] == [round(3.5 * 29.97), round(6.0 * 29.97)] and c["kaynak_kare"] == [round(20 * 29.97), round(22.5 * 29.97)]
    assert c["sekans_sn"][0] == round(round(3.5 * 29.97) / 29.97, 3)
    assert s["v1_bosluk"] and s["v1_bosluk"][0]["sn"] == pytest.approx(3.0, abs=0.04)
    assert s["v1_cakisma"] and s["kapali_klip"] == 1 and s["klip_sayisi"] == {"video": 3, "ses": 0}


class _P:
    name = "Yemek"
    settings = {"title": "Yemek", "options": {"intensity": "medium", "captions": True}}
    log = None

    def __init__(self, d, root=None):
        self.dir = d

        class Cfg:
            pass
        Cfg.root = root or d
        self.cfg = Cfg

    def load(self, fn, default=None):
        return default


def test_premiere_attach_reads_or_says_why_not(tmp_path):
    rep = {"gecis": {"metrics": {}, "flags": []}}
    PV.attach(_P(tmp_path), rep, reader=lambda p: PV.snapshot({"name": "x"}, _state()))
    assert rep["gecis"]["paket"]["premiere"]["sekans"] == "Yemek - kurgu"
    assert any(f["tur"] == "premiere_bosluk" for f in rep["gecis"]["flags"]) and (tmp_path / "kurul" / "premiere.json").exists()
    rep2 = {"gecis": {"metrics": {}, "flags": []}}

    def closed(p):
        raise RuntimeError("panel komutu almadı")
    PV.attach(_P(tmp_path), rep2, reader=closed)
    assert "okunamadi" in rep2["gecis"]["premiere"] and "Premiere zaman çizelgesi" in rep2["missing"]


def _project(tmp_path):
    d = tmp_path / "projeler" / "Yemek"
    (d / "kurul").mkdir(parents=True)
    (d / "proje.json").write_text(json.dumps({"title": "Yemek", "options": {"intensity": "medium", "captions": True}}), encoding="utf-8")
    m = {"proje": "Yemek", "tarih": "2026-10-22 18:00", "bulgular": [
        {"id": "HIK1", "ajan": "hikaye", "baslik": "Daha yoğun kurgu", "kanit": "k", "eylem": "secenek",
         "parametre": {"ad": "intensity", "deger": "high"}, "karar": "kabul", "destek": 5},
        {"id": "ANI1", "ajan": "animasyon", "baslik": "Yazılar donuk", "kanit": "k", "eylem": "yazi_animasyonu",
         "parametre": {"t": None, "stil": "nefes"}, "karar": "kabul", "destek": 6}]}
    (d / "kurul" / "tutanak.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")

    class P(_P):
        @property
        def settings(self):
            return json.loads((d / "proje.json").read_text(encoding="utf-8"))
    return P(d, tmp_path), m


def test_preview_apply_verify_undo_and_explicit_decision(tmp_path):
    proj, m = _project(tmp_path)
    hik, ani = m["bulgular"]
    assert AP.preview(proj, ani)["uygulanabilir"] is False                                     # no safe path yet
    pv = AP.preview(proj, hik)
    assert pv["uygulanabilir"] and pv["degisiklik"][0] == {"dosya": "proje.json", "alan": "options.intensity", "once": "medium", "sonra": "high"}
    assert AP.decide(proj, "HIK1", True)["tamam"]
    saved = json.loads((proj.dir / "kurul" / "tutanak.json").read_text(encoding="utf-8"))
    assert saved["bulgular"][0]["kullanici"] == "onay"
    mem = A.load_memory(tmp_path, "hikaye")
    assert any(d["tur"] == "kullanici" and "ONAYLADI" in d["ders"] for d in mem["dersler"])
    r = AP.apply(proj, hik)
    assert r["tamam"] and proj.settings["options"]["intensity"] == "high"
    assert (proj.dir / "kurul" / "geri_al" / r["geri_al"] / "proje.json").exists()
    assert AP.undo(proj, r["geri_al"])["tamam"] and proj.settings["options"]["intensity"] == "medium"
    log = [json.loads(x) for x in (proj.dir / "kurul" / "islemler.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["islem"] for x in log] == ["onay", "uygula", "geri_al"]


def test_option_values_are_validated():
    with pytest.raises(ValueError):
        AP._value("intensity", "extreme")
    assert AP._value("captions", "kapalı") is False and AP._value("tur", "ANLATIM") == "anlatim"


def test_voice_choice_comes_from_the_settings_file(tmp_path, monkeypatch):
    from kurgu.board import voice as V
    monkeypatch.setattr(V, "SETTINGS", tmp_path / "sesler.json")
    assert V.speaker_of("hikaye") == V.VOICES["hikaye"]["xtts"]
    (tmp_path / "sesler.json").write_text(json.dumps({"sesler": {"hikaye": "Sofia Hellen"}}), encoding="utf-8")
    assert V.speaker_of("hikaye") == "Sofia Hellen" and V.speaker_of("ses") == V.VOICES["ses"]["xtts"]


def test_spoken_lines_are_cleaned():
    from kurgu.board.live import _one_para
    assert _one_para('"**Bence** `kesim` erken [olmuş]."') == "Bence kesim erken olmuş."


@pytest.mark.skipif(os.environ.get("KURGU_ENTEGRASYON") != "1" or
                    not (Path(__file__).parents[1] / "modeller" / "xtts_v2" / "model.pth").exists(),
                    reason="integration test: set KURGU_ENTEGRASYON=1 (loads the 1.9 GB XTTS model on the CPU)")
def test_xtts_speaks_turkish(tmp_path):
    """Integration: the real local XTTS model (CPU, ~20 s with loading)."""
    from kurgu.board import voice as V
    out = V.say("grafik", "Merhaba, toplantıyı açıyorum.", tmp_path)
    import soundfile as sf
    a, sr = sf.read(str(out))
    assert sr == 24000 and len(a) > sr * 0.8 and V.engine() == "xtts"
