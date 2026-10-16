"""The causal-ablation residual edits on a toy model (docs/causal.md): the ablation zeroes its
direction on the edited rows only, hooks always come off, the replay control moves each row as far
as the ablation did, the J-lens vector matches the readout, and the scorer matches a naive pass.
Skipped where torch is not installed (CI installs only the dev extra)."""

import pytest

torch = pytest.importorskip("torch")


class Block(torch.nn.Module):
    def __init__(self, d: int, tuple_out: bool) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.tuple_out = tuple_out

    def forward(self, x):
        y = x + self.lin(x)
        return (y, None) if self.tuple_out else y


def _blocks(d: int = 6):
    torch.manual_seed(0)
    return torch.nn.ModuleList([Block(d, False), Block(d, True), Block(d, False)])


def _run(blocks, x):
    outs = []
    for blk in blocks:
        y = blk(x)
        x = y[0] if isinstance(y, tuple) else y
        outs.append(x)
    return outs


def test_ablation_zeroes_the_direction_on_the_edited_rows_only():
    from wsbench.produce.intervene import ablate, edit, unit

    blocks, x = _blocks(), torch.randn(1, 5, 6)
    u = unit(torch.randn(6))
    clean = _run(blocks, x)
    rec: dict = {}
    with edit(blocks, [1], ablate({1: u}, rec), 1, 4):
        edited = _run(blocks, x)
    assert (edited[1][0, 1:4] @ u).abs().max() < 1e-5
    assert torch.equal(edited[1][0, 0], clean[1][0, 0]) and torch.equal(
        edited[1][0, 4], clean[1][0, 4]
    )
    assert torch.allclose(rec[1], clean[1][0, 1:4] @ u, atol=1e-6)


def test_hooks_are_removed_on_exit_and_on_an_exception():
    from wsbench.produce.intervene import ablate, edit, unit

    blocks, x = _blocks(), torch.randn(1, 4, 6)
    clean = _run(blocks, x)[-1]
    with edit(blocks, [0, 2], ablate({0: unit(torch.ones(6)), 2: unit(torch.ones(6))}, {}), 1, 4):
        pass
    with pytest.raises(RuntimeError), edit(blocks, [1], lambda _layer, h: h * 0, 1, 4):
        raise RuntimeError("boom")
    assert torch.equal(_run(blocks, x)[-1], clean)


def test_the_replay_moves_each_row_as_far_as_the_ablation_along_the_control():
    from wsbench.produce.intervene import ablate, edit, replay, unit

    blocks, x = _blocks(), torch.randn(1, 5, 6)
    u, w = unit(torch.randn(6)), unit(torch.randn(6))
    clean = _run(blocks, x)[0]
    rec: dict = {}
    with edit(blocks, [0], ablate({0: u}, rec), 1, 5):
        ab = _run(blocks, x)[0]
    with edit(blocks, [0], replay(rec, {0: w}), 1, 5):
        rp = _run(blocks, x)[0]
    step_a, step_r = (ab - clean)[0, 1:], (rp - clean)[0, 1:]
    assert torch.allclose(step_a.norm(dim=-1), step_r.norm(dim=-1), atol=1e-5)
    assert torch.allclose((step_r @ w).abs(), step_r.norm(dim=-1), atol=1e-5)  # parallel to w


def test_the_jlens_vector_is_the_readouts_own_orientation():
    from wsbench.produce.methods import JLens

    torch.manual_seed(1)
    jac, w_u, h = torch.randn(1, 6, 6), torch.randn(9, 6), torch.randn(6)
    jl = JLens()
    jl._b, jl._jac, jl._w_u = type("B", (), {"device": "cpu"})(), jac, w_u
    v = w_u[4] @ jac[0]  # what causal_check.jdirs ablates
    assert torch.allclose(jl.scores(h, 0)[4], (h @ v) / v.norm(), atol=1e-5)


def test_the_final_norm_of_ones_is_its_gain():
    w = torch.randn(6)
    norm = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w)  # noqa: E731
    assert torch.allclose(norm(torch.ones(6)), 1.0 + w, atol=1e-5)


class TinyLM(torch.nn.Module):
    """``model(x, use_cache=, logits_to_keep=).logits`` like a transformers causal LM."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(2)
        self.emb, self.head = torch.nn.Embedding(11, 6), torch.nn.Linear(6, 11)
        self.device = torch.device("cpu")

    def forward(self, x, use_cache=False, logits_to_keep=0):
        logits = self.head(torch.cumsum(self.emb(x), dim=1))
        return type(
            "Out", (), {"logits": logits[:, -logits_to_keep:] if logits_to_keep else logits}
        )


def test_score_matches_a_naive_full_logits_computation():
    from wsbench.produce.intervene import kl, score

    model, ids, p = TinyLM(), [1, 4, 2, 7, 3], 3
    got = score(model, ids, p)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0], dim=-1)
    want = sum(lp[i - 1, ids[i]] for i in range(p, len(ids)))
    assert abs(got["logp"] - float(want)) < 1e-5
    assert got["correct"] == all(int(lp[i - 1].argmax()) == ids[i] for i in range(p, len(ids)))
    assert torch.allclose(got["next"], lp[p - 1])
    assert kl(got["next"], got["next"]) == pytest.approx(0.0, abs=1e-6)
    assert kl(got["next"], got["next"].roll(1), exclude=[0, 1]) > 0


def _basis():
    from wsbench.produce.intervene import clamp_basis, unit

    torch.manual_seed(3)
    u = unit(torch.randn(6))
    w = unit(torch.randn(6) - (torch.randn(6) @ u) * u)
    w = unit(w - (w @ u) * u)
    return u, unit(0.8 * u + 0.6 * w), clamp_basis(u, unit(0.8 * u + 0.6 * w))


def test_coords_record_without_editing():
    from wsbench.produce.intervene import coords, edit

    blocks, x = _blocks(), torch.randn(1, 5, 6)
    _, _, basis = _basis()
    clean = _run(blocks, x)
    rec: dict = {}
    with edit(blocks, [1], coords({1: basis}, rec), 1, 4):
        seen = _run(blocks, x)
    assert all(torch.equal(a, b) for a, b in zip(seen, clean, strict=True))
    assert torch.allclose(rec[1], clean[1][0, 1:4] @ basis[1].T, atol=1e-5)


def test_the_ported_clamp_hits_its_targets_inside_the_span_only():
    from wsbench.produce.intervene import clamp, edit

    blocks, x = _blocks(), torch.randn(1, 5, 6)
    _, _, (atoms, pinv) = _basis()
    clean = _run(blocks, x)
    target = torch.tensor([[1.0, -2.0], [0.5, 0.5], [3.0, 0.0]])
    with edit(blocks, [1], clamp({1: (atoms, pinv)}, {1: target}), 1, 4):
        ed = _run(blocks, x)
    assert torch.allclose(ed[1][0, 1:4] @ pinv.T, target, atol=1e-4)
    delta = (ed[1] - clean[1])[0, 1:4]
    assert torch.allclose(delta - delta @ (atoms @ pinv).T, torch.zeros_like(delta), atol=1e-5)
    assert torch.equal(ed[1][0, 0], clean[1][0, 0]) and torch.equal(ed[1][0, 4], clean[1][0, 4])


def test_the_scale_one_swap_is_a_reflection_and_ablate_is_a_one_atom_clamp():
    from wsbench.produce.intervene import ablate, clamp, clamp_basis, swapped_targets, unit

    u_s, u_t, (atoms, pinv) = _basis()
    x = torch.randn(4, 6)
    swapped = clamp({0: (atoms, pinv)}, {0: swapped_targets(x @ pinv.T, 1.0)})(0, x)
    n = unit(u_s - u_t)
    assert torch.allclose(swapped, x - 2 * (x @ n)[:, None] * n, atol=1e-5)
    assert torch.allclose(swapped.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    # unclamped twice is the identity: the reason every layer is pinned to clean coordinates
    again = clamp({0: (atoms, pinv)}, {0: swapped_targets(swapped @ pinv.T, 1.0)})(0, swapped)
    assert torch.allclose(again, x, atol=1e-5)
    one = clamp_basis(u_s)
    zero = clamp({0: one}, {0: torch.zeros(4, 1)})(0, x)
    assert torch.allclose(zero, ablate({0: u_s}, {})(0, x), atol=1e-5)
