"""multihop_mt_v2: the bank is the shortcut-gated successor (plan 0013) — its option sets match the
golden, every item's gate evidence is present and passing, and the family is registered like v1."""

import json
from pathlib import Path

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.multitoken.options import judged_roles, option_sets
from wsbench.multitoken.prompts import CANNOT

REPO = Path(__file__).resolve().parents[1]


def _bank():
    return load_bank(REPO / "evals/multihop_mt_v2/items.json")


def test_options_match_the_golden_and_the_bank():
    header, items = _bank()
    assert header["family"] == "multihop-hard" and header["version"] == "v2"
    opts = option_sets(items)
    golden = json.loads(
        (REPO / "tests/golden/multihop_mt_v2_options.json").read_text(encoding="utf-8")
    )
    want = {
        i: {r: (v["options"], v["gold_index"]) for r, v in per.items()}
        for i, per in golden["options"].items()
    }
    assert want == opts
    for it in items:
        assert set(judged_roles(it)) == set(opts[it["id"]])
        for role, (o, g) in opts[it["id"]].items():
            assert len(o) == 6 and o[-1] == CANNOT and 0 <= g < 5 and len(set(o)) == 6
            assert o[g] == it["mc"][role]["gold"]


def test_every_item_carries_passing_gate_evidence():
    _header, items = _bank()
    names = [it["name"] for it in items]
    assert len(names) == len(set(names))
    for it in items:
        n_bridges = 2 if it["subfamily"] == "3hop" else 1
        assert [u["role"] for u in it["units"]] == [f"bridge{k + 1}" for k in range(n_bridges)]
        assert len(it["intermediates"]) == len(it["bridges"]) == n_bridges, it["name"]
        assert it["consistency"] >= 0.8 and it["consistency_greedy"], it["name"]
        assert all(v >= 0.5 for v in it["bridge_consistency"].values()), it["name"]
        assert it["shortcut_probes"], it["name"]
        assert len(it["probe_token_lens"]["target_alts"]) == len(it["target_alts"]), it["name"]
        if it.get("relabeled_from"):
            assert it["subfamily"] == "2hop" and it["relabel_reason"], it["name"]


def test_the_strict_gate_holds_for_every_item():
    """No stored shortcut probe reaches the target (greedy, or > 4/20), except the drop_b1 probe
    that shows a relabeled item's dropped bridge is decorative (plan 0013 rev 3)."""
    _header, items = _bank()
    for it in items:
        for sp in it["shortcut_probes"]:
            hit = sp["greedy_pass"] or sp["n_pass"] > 4
            excused = (
                it.get("relabeled_from")
                and sp["id"] == "S1_drop_b1"
                and sp["hides"] == ["dropped_bridge"]
            )
            assert not hit or excused, (it["name"], sp["id"], sp["n_pass"])


def test_registered():
    from wsbench.evals.multihop_mt_v2 import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    assert SPEC.group == "basic_mt" and SPEC.scorer == "regex"
    assert SPEC.judge.prompt_version == "mt-regex-2026-09-23"
