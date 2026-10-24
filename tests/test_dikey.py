"""Upload-screen choice (24.10.2026): "Yatay video edit" keeps the usual 16:9 edit untouched; the two vertical
choices make one vertical video of at most 3 minutes instead."""
from kurgu import pipeline
from kurgu.editing.director import options_of


class _Proj:
    def __init__(self, cikis=None):
        self.settings = {"options": {"cikis": cikis}} if cikis else {}


def test_default_is_the_usual_horizontal_edit():
    assert options_of({})["cikis"] == "yatay"
    assert options_of({"options": {"cikis": "dikey"}})["cikis"] == "dikey"


def test_only_vertical_projects_take_the_vertical_path(monkeypatch):
    calls = []
    from kurgu import desktop
    monkeypatch.setattr(desktop, "start_room", lambda: None)   # no board-room process from a test
    monkeypatch.setattr(pipeline, "make_vertical", lambda proj, progress, llm: calls.append("dikey") or {"engine": "dikey"})
    monkeypatch.setattr(pipeline, "_prepare_main", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yatay yol")))
    for mode in ("dikey", "yataydan_dikey"):
        assert pipeline.make_video(_Proj(mode))["engine"] == "dikey"
    assert calls == ["dikey", "dikey"]
    try:   # the horizontal project goes on to the normal 16:9 edit (here stopped right at its first step)
        pipeline.make_video(_Proj())
        raise AssertionError("yatay proje dikey yola girdi")
    except RuntimeError as e:
        assert "yatay yol" in str(e)


def test_three_minute_limit():
    assert pipeline.VERTICAL_MAX_S <= 180
