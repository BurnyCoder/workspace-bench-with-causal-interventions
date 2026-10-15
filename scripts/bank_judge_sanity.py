"""Run the bank judge v2 sanity set live (plan 0012 §2.3): each case 3x, each repetition on a
FRESH cache so all three are live calls; exit 1 unless every case gives its `expect` 3/3.

A dev script, not part of the package; needs OPENROUTER_API_KEY.
"""

import json
import sys
import tempfile
from pathlib import Path

from wsbench.basic.judge import judge_cells
from wsbench.basic.prompts import FAMILY_RULES, PROMPT_VERSION, forbidden_line
from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.llm import Spend
from wsbench.mcjudge import Preflighter
from wsbench.readouts import Cell

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/bank_judge_v2_sanity.json"
REPS = 3


def main() -> int:
    cases = json.loads(FIXTURE.read_text())["cases"]
    judge = resolve(JudgeConfig(prompt_version=PROMPT_VERSION))
    got: dict[str, list[bool | None]] = {c["name"]: [] for c in cases}
    spend = Spend()
    for _rep in range(REPS):
        with tempfile.TemporaryDirectory() as d, Cache(Path(d) / "cells.jsonl") as cache:
            # one family per batch: the family rule is batch-wide
            for fam in sorted({c["family"] for c in cases}):
                fc = [c for c in cases if c["family"] == fam]
                groups = {}
                for c in fc:
                    s = tuple(c["samples"])
                    cell = Cell(
                        id=c["name"],
                        layer=0,
                        pos=0,
                        samples=s if c["kind"] == "prose" else None,
                        tokens=s if c["kind"] == "tokens" else None,
                        scores=None,
                    )
                    groups[(c["name"], 0)] = [cell]
                for kind in ("prose", "tokens"):
                    sub = {k: v for k, v in groups.items() if v[0].kind == kind}
                    if not sub:
                        continue
                    out = judge_cells(
                        sub,
                        {c["name"]: c["targets"] for c in fc},
                        family_rule=FAMILY_RULES.get(fam, ""),
                        forbidden={
                            c["name"]: forbidden_line(fam, {"target": c.get("surface_target")})
                            for c in fc
                        },
                        kind=kind,
                        judge=judge,
                        cache=cache,
                        spend=spend,
                        preflight=Preflighter(False),
                        concurrency=16,
                        rpm=240.0,
                        dry_run=False,
                    )
                    for (name, _layer), v in out.verdicts.items():
                        got[name].append(v.expressed)
                    for name, _layer in sub:
                        if (name, 0) not in out.verdicts:
                            got[name].append(None)
    ok = True
    for c in cases:
        vs = got[c["name"]]
        good = sum(v is c["expect"] for v in vs)
        flag = "ok " if good == REPS else "BAD"
        ok &= good == REPS
        print(f"{flag} {c['name']:<34} expect={c['expect']!s:<5} got={vs}")
    print(f"spend ${spend.usd:.4f}; {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
