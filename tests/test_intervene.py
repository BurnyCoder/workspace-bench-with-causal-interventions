"""The causal-ablation pieces that need no torch (docs/causal.md): token spans and bridge tokens
on a fake Qwen-like tokenizer, control pairing, and the summary maths. The residual edits are
tested in test_intervene_torch.py."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from wsbench.produce.intervene import answer_span, bridge_token, overlaps, variant_ids

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "causal_check.py"


def _script():
    spec = importlib.util.spec_from_file_location("causal_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["causal_check"] = mod
    spec.loader.exec_module(mod)
    return mod


class Tok:
    """Qwen-like splits: a space joins the following word, digits are one token each (a space
    before a digit is its own token), punctuation stands alone. ``merge_quote`` glues an opening
    quote to the next word, so a prompt ending in a quote stops being a token prefix."""

    def __init__(self, merge_quote: bool = False) -> None:
        self.vocab: dict[str, int] = {}
        word = r"\"?[A-Za-zÀ-ÿ]+" if merge_quote else r"[A-Za-zÀ-ÿ]+"
        self.rx = re.compile(rf" ?{word}| |\d|[^\sA-Za-zÀ-ÿ\d]")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {
            "input_ids": [self.vocab.setdefault(t, len(self.vocab)) for t in self.rx.findall(text)]
        }

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_a_word_after_a_trailing_space_carries_the_space():
    tok = Tok()
    ids, p, stripped = answer_span(tok, "Fact: X is ", "Fe")
    assert stripped and tok.decode(ids[p:]) == " Fe" and tok.decode(ids[:p]) == "Fact: X is"


def test_digits_keep_their_space_token_in_the_prompt():
    tok = Tok()
    ids, p, _ = answer_span(tok, "month number ", "12")
    assert tok.decode([ids[p - 1]]) == " " and tok.decode(ids[p:]) == "12"


def test_an_opening_quote_is_followed_directly():
    tok = Tok()
    ids, p, stripped = answer_span(tok, 'ist "', "fern")
    assert not stripped and tok.decode(ids[p:]) == "fern"


def test_a_prompt_that_is_not_a_token_prefix_raises():
    with pytest.raises(ValueError, match="token prefix"):
        answer_span(Tok(merge_quote=True), 'ist "', "fern")


def test_bridge_token_is_one_visible_token_or_nothing():
    tok = Tok()
    t, form = bridge_token(tok, ["6"])  # " 6" is [' ', '6']: the bare digit is taken instead
    assert form == "6" and tok.decode([t]) == "6"
    assert bridge_token(tok, ["Los Angeles"])[0] is None
    assert bridge_token(tok, ["Los Angeles", "LA"])[1] == " LA"  # falls through the intermediates
    assert len(variant_ids(tok, "iron")) == 4  # " iron", "iron", " Iron", "Iron"


def test_overlap_folds_case_and_accents():
    assert overlaps("Paris", "París") and overlaps("Australia", "australia")
    assert not overlaps("Brazil", "Portuguese")


def test_controls_come_from_the_same_template_group_and_never_overlap_the_bridge():
    cc = _script()
    assert cc.group_key("multihop", "holiday-christmas-monthnum") == "holiday-monthnum"
    assert cc.group_key("multilingual", "de-place-near-far") == "de"
    rows = [
        {"id": "a", "group": "g", "bridge": ["iron"]},
        {"id": "b", "group": "h", "bridge": ["gold"]},
        {"id": "c", "group": "g", "bridge": ["Iron"]},  # a variant of a's bridge: never a control
        {"id": "d", "group": "g", "bridge": ["copper"]},
    ]
    assert cc.control_words(rows, 0) == ["copper", "gold"]


def _trial(set_, id_, bridge, answer, j, same, logit_kl=0.5, j_kl=1.0, rank=3, ok=True):
    cond = {"dlogp": 0.0, "correct": True, "kl": 0.1, "kl_excl": 0.1}
    t = {c: dict(cond) for c in _script().CONDITIONS}
    t["ablate_j"] |= {"dlogp": j, "kl": j_kl, "kl_excl": j_kl, "correct": False}
    t["samecat"] |= {"dlogp": same}
    t["ablate_logit"] |= {"kl": logit_kl, "kl_excl": logit_kl}
    return t | {
        "set": set_,
        "id": id_,
        "bridge": bridge,
        "answer": answer,
        "clean_correct": ok,
        "rank_by_layer": {"9": rank},
        "removed_norm_j": 2.0,
        "removed_norm_logit": 1.0,
    }


def test_pooling_drops_a_probe_swap_item_that_repeats_a_bank_item():
    cc = _script()
    trials = [
        _trial("multihop", "atomic-26", " iron", "Fe", -3, -1),
        _trial("probe_swap", "iron-symbol", "iron", "Fe", -3, -1),
        _trial("probe_swap", "amazon", " Brazil", "Portuguese", -2, -1),
    ]
    assert [t["id"] for t in cc.pooled(trials)] == ["atomic-26", "amazon"]


def test_ratio_of_means_and_the_summary_tables():
    cc = _script()
    r, lo, hi = cc.ratio_ci([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert r == 2.0 and lo == 2.0 and hi == 2.0
    trials = [
        _trial("multihop", "a", " iron", "Fe", -3.0, -1.0),
        _trial("multilingual", "b", " far", "fern", -2.0, -1.0, rank=50),
        _trial("probe_swap", "c", " Brazil", "Portuguese", -1.0, -1.0, ok=False),
        {"set": "multihop", "id": "x", "excluded": "no single-token form"},
    ]
    text = cc.summarize(trials, {"a": True, "b": False})
    assert "| pooled | -1.50" in text  # (-2 + -1) / 2 over the two clean-correct trials
    assert "| multihop | 2 | 1 | 1 | 1 |" in text
    assert "KL ratio, J / logit: 2.00" in text
    assert "ablate_j | -2.50" in text and "0/2" in text


def test_swap_groups_normalise_language_and_category():
    cc = _script()
    assert cc.swap_group("multilingual", "b3-ml-es-opposite-hot") == "es-opposite"
    assert cc.swap_group("multilingual", "french-number-seven") == "fr-number"
    assert cc.swap_group("multilingual", "frn-month-after-jun") == "fr-month"
    assert cc.swap_group("multihop", "holiday-christmas-monthnum") == "holiday-monthnum"


def test_swap_partner_skips_overlapping_bridges_and_answers():
    cc = _script()
    rows = [
        {"id": "a", "swap_group": "g", "bridge": ["night"], "answer": "day"},
        {"id": "b", "swap_group": "g", "bridge": ["day"], "answer": "noon"},  # bridge = a's answer
        {"id": "c", "swap_group": "g", "bridge": ["Mexico"], "answer": "Day"},  # same answer
        {"id": "d", "swap_group": "g", "bridge": ["Night"], "answer": "dusk"},  # same bridge
        {"id": "e", "swap_group": "h", "bridge": ["Brazil"], "answer": "Portuguese"},
        {"id": "f", "swap_group": "g", "bridge": ["cold"], "answer": "hot"},
    ]
    assert cc.swap_partner(rows, 0)["id"] == "f"
    assert cc.swap_partner(rows, 4) is None  # alone in its group


def test_wilson_and_mcnemar():
    cc = _script()
    lo, hi = cc.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert cc.mcnemar(0, 0) == 1.0
    assert cc.mcnemar(6, 0) == pytest.approx(2 / 64)  # two-sided exact binomial
    assert cc.mcnemar(3, 3) == 1.0


def _swap_row(id_, set_="probe_swap", j=True, logit=False, null=False, zero=False, **kw):
    cell = {
        "success": False,
        "success_top5": False,
        "success_word_top1": False,
        "greedy_swap": False,
        "orig_correct": True,
        "dlogp_swap": 0.0,
        "dlogp_orig": 0.0,
        "leak": False,
    }
    by = {"j_swap": j, "logit_swap": logit, "null_swap": null}
    row = {c: {"1.0": cell | {"success": ok}} for c, ok in by.items()}
    return (
        row
        | {
            "set": set_,
            "id": id_,
            "bridge": " " + id_,
            "answer": "ans-" + id_,
            "zero": cell | {"success": zero},
            "clean_correct": True,
            "clean_success_rank": 7,
            "rank_by_layer": {"9": 3},
            "partner_known": None,
        }
        | kw
    )


def test_the_swap_summary_counts_primary_trials_and_pairs():
    cc = _script()
    rows = [
        _swap_row("a"),
        _swap_row("b", logit=True),
        _swap_row("c", j=False),
        _swap_row("d", clean_success_rank=1),  # already the top-1: not a primary trial
        {"set": "multihop", "id": "x", "excluded": "no swap partner"},
    ]
    text = cc.summarize_swaps(rows)
    assert "| pooled | 2/3 = 67%" in text
    assert "| probe_swap | 4 | 0 | 4 | 4 | 3 |" in text
    assert "| logit_swap | 1 | 0 | 1 |" in text and "| null_swap | 2 | 0 | 0.5 |" in text
