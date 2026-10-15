"""Bank judge v2 (plan 0012 §2): the family rules, the forbidden answer, per-family PROMPTS."""

import importlib
import json
from pathlib import Path

from wsbench.banks import load_bank
from wsbench.basic import family as bfam
from wsbench.basic import judge as bjudge
from wsbench.basic import prompts as P  # noqa: N812
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.readouts import Cell
from wsbench.registry import JudgeArgs

REPO = Path(__file__).resolve().parents[1]
CELL = Cell(id="x", layer=20, pos=3, samples=("a sample",), tokens=None, scores=None)


def test_user_renders_rule_and_forbidden_only_when_given():
    plain = P.render_user(["cat"], "prose", [CELL], family_rule="", forbidden="")
    assert plain == f'targets: ["cat"]\n{P.KIND_NOTE_PROSE}\n\n[position 3]\n  a sample'
    both = P.render_user(["cat"], "prose", [CELL], family_rule="RULE", forbidden="FORB")
    assert both.startswith(f'targets: ["cat"]\n{P.KIND_NOTE_PROSE}\nRULE\nFORB\n\n[position 3]')


def test_forbidden_line_is_multihop_only_and_json_quoted():
    assert P.forbidden_line("multihop", {"target": "Fe"}).startswith(
        'The input\'s final answer "Fe"'
    )
    assert '"Fe"' in P.forbidden_line("multihop", {"target": "Fe"}).split("only ")[1]
    assert P.forbidden_line("multilingual", {"target": "drei"}) == ""
    assert P.forbidden_line("multihop", {}) == ""
    _h, items = load_bank(REPO / "evals/multihop/items.json")
    assert all(P.forbidden_line("multihop", it) for it in items)


def test_family_rules_and_per_family_prompts():
    assert set(P.FAMILY_RULES) == {"poetry", "typo"} and {"multihop"} == P.FORBIDDEN_FAMILIES
    assert "misspelled" in P.FAMILY_RULE_TYPO and "Synonyms" in P.FAMILY_RULE_POETRY
    own = {
        "poetry": "FAMILY_RULE_POETRY",
        "typo": "FAMILY_RULE_TYPO",
        "multihop": "FORBIDDEN_TEMPLATE",
    }
    for f in ["association", "basic_readout", "multihop", "multilingual", "poetry", "typo"]:
        mod = importlib.import_module(f"wsbench.evals.{f}.prompts")
        extra = set(mod.PROMPTS) - set(P.PROMPTS)
        assert extra == ({own[f]} if f in own else set()), f
        assert set(P.PROMPTS) <= set(mod.PROMPTS)
    assert "ranked first among the candidates of its type" in P.KIND_NOTE_TOKENS
    assert "overrides the rules" in P.SYSTEM and "NEVER combine" in P.SYSTEM


def test_family_run_sends_the_items_own_forbidden_answer(tmp_path, monkeypatch):
    sent: list[str] = []

    def fake_run_calls(calls, **kw):
        sent.extend(c.user for c in calls)
        return {c.key: {"expressed": False, "target": "", "quote": ""} for c in calls}

    monkeypatch.setattr(bjudge, "run_calls", fake_run_calls)
    _h, items = load_bank(REPO / "evals/multihop/items.json")
    it = items[0]
    rows = [{"id": it["id"], "layer": 20, "pos": 5, "samples": ["something"]}]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = JudgeArgs(
        readouts=path,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=P.PROMPT_VERSION)),
        layers=[20],
        items=[it["id"]],
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
        extra={},
    )
    bfam.run_family(args, name="multihop", bank=REPO / "evals/multihop/items.json")
    line = P.forbidden_line("multihop", it)
    assert len(sent) == 1 and line in sent[0]
    assert sent[0].startswith(f"targets: {json.dumps(bfam.targets_of(it))}")
