"""Counting sped up: the counter's ticks must be added before the effects are merged into the plan (22.10.2026)."""
import inspect
from pathlib import Path


def test_counter_sounds_are_merged_after_they_are_added():
    """The counter's ticks were appended to sfx_extra after the merge into the plan and never played."""
    from kurgu import pipeline
    src = inspect.getsource(pipeline._prepare_main)
    merge = src.index('d["plan"]["sfx"] = sorted(d["plan"]["sfx"] + sfx_extra')
    assert src.rindex("sfx_extra.append(") < merge
    assert Path(inspect.getsourcefile(pipeline)).exists()
