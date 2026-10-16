"""The producer's CLI surface, offline: ``opts=`` reaches the method as typed fields, so another
model's lens artifacts are named from the shell. No model is loaded."""

import pytest

from wsbench.cli import main
from wsbench.produce import OLens, Producer, method


def test_numeric_fields_take_their_type_and_strings_stay_strings():
    m = method("olens", alpha="60", repo_type="model", lora="org/repo:sub")
    assert m.alpha == 60.0 and isinstance(m.alpha, float)
    assert (m.repo_type, m.lora) == ("model", "org/repo:sub")
    assert method("logit_lens", k="5").k == 5
    with pytest.raises(TypeError):
        method("jlens", no_such_field="x")


def test_olens_keeps_its_dataset_repo_by_default():
    assert OLens().repo_type == "dataset"


def test_cli_opts_reach_producer_load(monkeypatch):
    seen = {}

    def fake_load(model, method_spec, *, device, **kw):
        seen.update(model=model, method=method_spec, device=device, kw=kw)
        raise SystemExit(0)  # stop before any read

    monkeypatch.setattr(Producer, "load", staticmethod(fake_load))
    argv = ["produce", "text=hi", "model=org/tiny", "method=jlens", "device=cpu"]
    with pytest.raises(SystemExit):
        main([*argv, "opts=filename=a/b.pt,repo=org/lenses"])
    assert seen == {
        "model": "org/tiny",
        "method": "jlens",
        "device": "cpu",
        "kw": {"filename": "a/b.pt", "repo": "org/lenses"},
    }
