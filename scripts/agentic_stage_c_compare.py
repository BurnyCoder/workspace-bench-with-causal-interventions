"""Plan 0012 §3.1 validation: Stage C v1 (bank descriptor) vs v2 (targets.json action block) on
the SAME cached Stage B accounts, one model for both, so only the Stage C prompt differs.

A dev script, not part of the package. The family refuses non-``claude-*`` judges (its free-text
stages need the Anthropic route); this script calls Stage C only, through OpenRouter
(``OPENROUTER_API_KEY``), with reasoning on as the judge runs it.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

from wsbench.evals.agentic_misalignment import judge, prompts
from wsbench.mcjudge import load_bank

REPO = Path(__file__).resolve().parents[1]
V1 = (
    json.loads((REPO / "tests/golden/agentic_misalignment_prompts.json").read_text())["templates"][
        "DESIGN_SCORE_PROMPT"
    ]
    .replace("{{", "{")
    .replace("}}", "}")
)


def render_v1(account: str, descriptor: str, scenario: str) -> str:
    return (
        V1.replace("{families}", ", ".join(prompts.FAMILIES))
        .replace("{descriptor}", descriptor)
        .replace("{scenario}", scenario)
        .replace("{account}", account)
    )


def accounts(cells: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in cells.read_text().splitlines():
        r = json.loads(line)
        k = r["key"]
        if k.startswith("B:") and ":chunk" not in k and r.get("result"):
            out[k[2:]] = r["result"]  # the last landed row wins, as the cache does
    return out


def call(client: Any, model: str, user: str) -> str | None:
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
                max_tokens=judge.STAGE_C_MAX_TOKENS,
                messages=[{"role": "user", "content": user}],
                extra_body={"reasoning": {"enabled": True}},
            )
            text = r.choices[0].message.content
            if text:
                return text
        except Exception:  # a dev script: retry anything
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    bank = {it["id"]: it for it in load_bank("agentic_misalignment")["items"]}
    targets = judge.load_targets()
    acc = {k: v for k, v in accounts(a.cells).items() if k in bank}
    client = (
        None
        if a.model.startswith("claude-")
        else OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
        )
    )
    jobs: list[tuple[str, str, str]] = []
    for lb, text in sorted(acc.items()):
        it = bank[lb]
        sc = judge.scenario_of(it)
        jobs.append((lb, "v1", render_v1(text, it["descriptor"], sc)))
        jobs.append((lb, "v2", prompts.render_design_score(text, targets[lb], sc)))
    if a.model.startswith("claude-"):  # the judge's own Anthropic path (thinking on), e.g. via
        # an `ant auth login` profile (ANTHROPIC_PROFILE); identical for v1 and v2
        from wsbench import llm

        raw: list[str | None] = [None] * len(jobs)
        llm.stream_text(
            [j[2] for j in jobs],
            model=a.model,
            on_result=lambda i, t: raw.__setitem__(i, t),
            thinking=True,
            max_tokens=judge.STAGE_C_MAX_TOKENS,
            concurrency=32,
        )
    else:
        with ThreadPoolExecutor(16) as ex:
            raw = list(ex.map(lambda j: call(client, a.model, j[2]), jobs))
    rows: dict[str, dict[str, Any]] = {}
    for (lb, ver, _u), text in zip(jobs, raw, strict=True):
        parsed = prompts.parse_design_score(text) if text else None
        rows.setdefault(lb, {"arm": bank[lb]["arm"], "category": bank[lb]["category"]})[ver] = (
            parsed
        )
    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    print(f"{'item':<44} arm         v1 fid  v2 fid  v1 rank v2 rank")
    for lb, r in rows.items():
        f = [(r.get(v) or {}).get("design_fidelity") for v in ("v1", "v2")]
        k = [(r.get(v) or {}).get("design_rank") for v in ("v1", "v2")]
        print(f"{lb:<44} {r['arm']:<11} {f[0]!s:>6}  {f[1]!s:>6}  {k[0]!s:>7} {k[1]!s:>7}")
    for v in ("v1", "v2"):
        fs = [
            (r.get(v) or {}).get("design_fidelity")
            for r in rows.values()
            if r["arm"] == "misaligned"
        ]
        fs = [x for x in fs if isinstance(x, int)]
        print(f"{v}: design_score (misaligned) = {sum(fs) / 3 / len(fs):.3f} over {len(fs)}")


if __name__ == "__main__":
    main()
