"""The producer's CLI surface, offline: ``opts=`` reaches the method as typed fields, so another
model's lens artifacts are named from the shell. No model is loaded."""

import pytest

from wsbench.cli import main
from wsbench.produce import OLens, Producer, Readout, Rendered, method


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


class _Backend:
    """Activations are ``(layer, position)`` tuples; no model, no torch."""

    def capture(self, ids, layers, positions):
        return {L: [(L, p) for p in positions] for L in layers}


class _Lens:
    def __init__(self):
        self.calls = []

    def read(self, h, layer):
        self.calls.append("read")
        return Readout(tokens=[f"{h}"])


class _Verbalizer(_Lens):
    def read_many(self, hs, layers):
        self.calls.append(("read_many", len(hs)))
        return [Readout(samples=[f"{h}@{L}"]) for h, L in zip(hs, layers, strict=True)]


def _rows(method):
    p = Producer(backend=_Backend(), method=method)
    r = Rendered(ids=[1, 2, 3], tokens=["a", "b", "c"], decoded=["a", "b", "c"], render="plain")
    return list(p._read_rendered("item", r, [0, 2], [8, 10])), method.calls


def test_a_verbalizer_reads_the_whole_item_in_one_call_in_cell_order():
    rows, calls = _rows(_Verbalizer())
    assert calls == [("read_many", 4)]
    assert [(r.layer, r.pos, r.token) for r in rows] == [
        (8, 0, "a"),
        (8, 2, "c"),
        (10, 0, "a"),
        (10, 2, "c"),
    ]
    assert [r.readout.samples for r in rows] == [
        ["(8, 0)@8"],
        ["(8, 2)@8"],
        ["(10, 0)@10"],
        ["(10, 2)@10"],
    ]


def test_a_vector_lens_still_reads_cell_by_cell():
    rows, calls = _rows(_Lens())
    assert calls == ["read"] * 4 and rows[3].readout.tokens == ["(10, 2)"]
