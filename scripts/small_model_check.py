"""The tables of docs/small_model.md from a finished slice run (``summary``), plus the one input
``wsbench`` has no command for: the bank's plain render continued greedily and graded by the
``capable`` grader (``raw``, GPU and API).

A dev script, not part of the package:
    python scripts/small_model_check.py raw --out outputs/small_model_n30 --limit 30
    python scripts/small_model_check.py null --out outputs/small_model_n30 --arm olens
    python scripts/small_model_check.py summary --out outputs/small_model_n30
"""

import argparse
import json
import re
from pathlib import Path

from wsbench.banks import item_targets, load_bank
from wsbench.mc import fold
from wsbench.registry import REPO_ROOT
from wsbench.results import bootstrap_ci

FAMILIES = ("multihop", "multilingual", "typo")
RAW_FAMILIES = ("multihop", "multilingual")  # a plain continuation never corrects a typo
ARMS = ("logit_lens", "jlens", "olens")
JUDGED = ("14", "22")


def raw(out: Path, model: str, limit: int) -> None:
    """Greedy 8-token continuation of each plan render, graded against the gate's golds."""
    import torch

    from wsbench import readplan
    from wsbench.cache import Cache
    from wsbench.capable.questions import build
    from wsbench.capable.run import GRADE_SCHEMA, GRADE_SYSTEM, PROMPT_VERSION, render_grade
    from wsbench.judge_config import JudgeConfig, resolve
    from wsbench.llm import Spend
    from wsbench.mcjudge import Call, run_calls
    from wsbench.produce import Backend, render

    b = Backend.load(model)
    judge = resolve(JudgeConfig(prompt_version=PROMPT_VERSION))
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for fam in RAW_FAMILIES:
        qs = {q.id: q for q in build(fam)}
        rows = []
        for spec in readplan.plan(fam)[:limit]:
            x = torch.tensor([render(spec, b.tokenizer).ids], device=b.device)
            with torch.no_grad():
                g = b.model.generate(
                    x,
                    attention_mask=torch.ones_like(x),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=b.tokenizer.eos_token_id,
                )
            text = b.tokenizer.decode(g[0, x.shape[1] :], skip_special_tokens=True)
            rows.append({"id": spec.id, "golds": qs[spec.id].golds, "raw": text})
        calls = [
            Call(
                key=f"raw|{fam}|{r['id']}",
                system=GRADE_SYSTEM,
                user=render_grade(qs[r["id"]], r["raw"].strip()),
                meta={"item": r["id"]},
            )
            for r in rows
        ]
        with Cache(out / "raw" / f"{fam}.cells.jsonl") as cache:
            got = run_calls(
                calls,
                schema=GRADE_SCHEMA,
                judge=judge,
                prompt_version=PROMPT_VERSION,
                cache=cache,
                spend=Spend(),
                concurrency=16,
                rpm=240.0,
                dry_run=False,
                temperature=0.0,
            )
        for r, c in zip(rows, calls, strict=True):
            v = got.get(c.key)
            r["raw_correct"] = v["correct"] if isinstance(v, dict) else None
        (out / "raw" / f"{fam}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"{fam}: {sum(r['raw_correct'] is True for r in rows)}/{len(rows)} raw correct")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fmt(flags: list[bool]) -> str:
    if not flags:
        return "—"
    lo, hi = bootstrap_ci([float(f) for f in flags])
    return f"{sum(flags)}/{len(flags)} [{lo:.2f}, {hi:.2f}]"


def _targets() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for fam in FAMILIES:
        _, items = load_bank(REPO_ROOT / "evals" / fam / "items.json")
        out |= {it["id"]: [fold(t) for t in item_targets(it)] for it in items}
    return out


def null(out: Path, arm: str) -> None:
    """A derangement null: within each family every item's readouts are relabelled as the item
    ``d`` places later (cyclic, bank order), with the smallest ``d`` at which no item meets one
    sharing a target. Judging ``<arm>_null`` scores each readout against another item's targets."""
    targets = _targets()
    for fam in FAMILIES:
        rows = _jsonl(out / "readouts" / arm / f"{fam}.jsonl")
        ids = list(dict.fromkeys(r["id"] for r in rows))

        def clashes(d: int, ids: list[str] = ids) -> int:
            n = len(ids)
            return sum(
                bool(set(targets[i]) & set(targets[ids[(k + d) % n]])) for k, i in enumerate(ids)
            )

        d = min(range(1, len(ids)), key=lambda d: (clashes(d), d))
        to = {i: ids[(k + d) % len(ids)] for k, i in enumerate(ids)}
        dst = out / "readouts" / f"{arm}_null" / f"{fam}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(
            "".join(json.dumps(r | {"id": to[r["id"]]}, ensure_ascii=False) + "\n" for r in rows)
        )
        print(f"{fam}: shift {d}, {clashes(d)} shared-target pairs -> {dst}")


def _hit(row: dict, targets: list[str]) -> bool:
    """An English target verbatim: a whole token of a bag, or a whole word of prose."""
    if "tokens" in row:
        return any(fold(t.strip()) in targets for t in row["tokens"])
    text = fold(" ".join(row["samples"]))
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in targets)


def summary(out: Path) -> None:
    gate: dict[str, dict] = {}  # item -> {"chat", "raw", "family"}
    for fam in FAMILIES:
        for r in json.loads((out / "capable" / fam / "capable.json").read_text())["rows"]:
            gate[r["id"]] = {"family": fam, "chat": r["greedy_correct"] is True, "raw": None}
        raw_path = out / "raw" / f"{fam}.json"
        if raw_path.exists():
            for r in json.loads(raw_path.read_text()):
                gate[r["id"]]["raw"] = r["raw_correct"] is True
    can = {i: g["chat"] or bool(g["raw"]) for i, g in gate.items()}

    print(
        "## Gate (greedy)\n\n| family | chat turn | raw continuation | either |\n|---|---|---|---|"
    )
    for fam in FAMILIES:
        ids = [i for i, g in gate.items() if g["family"] == fam]
        raws = [gate[i]["raw"] for i in ids if gate[i]["raw"] is not None]
        raw_cell = f"{sum(raws)}/{len(raws)}" if raws else "—"
        chat = sum(gate[i]["chat"] for i in ids)
        print(f"| {fam} | {chat}/{len(ids)} | {raw_cell} | {sum(can[i] for i in ids)}/{len(ids)} |")

    arms = [*ARMS, *(f"{a}_null" for a in ARMS if (out / "judged" / f"{a}_null").exists())]
    judged = {
        (arm, fam): json.loads((out / "judged" / arm / fam / "results.json").read_text())["rows"]
        for arm in arms
        for fam in FAMILIES
    }
    print("\n## Judged pass rate (any judged layer), [95% bootstrap CI]\n")
    print("| arm | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * len(FAMILIES))
    for arm in arms:
        cells = [_fmt([r["pass"] is True for r in judged[(arm, f)]]) for f in FAMILIES]
        print(f"| {arm} | " + " | ".join(cells) + " |")

    print("\n## Pass rate by whether the model does the task (either format)\n")
    print("| arm | family | model does it | model does not |\n|---|---|---|---|")
    for arm in ARMS:
        for fam in FAMILIES:
            rows = judged[(arm, fam)]
            yes = [r["pass"] is True for r in rows if can[r["id"]]]
            no = [r["pass"] is True for r in rows if not can[r["id"]]]
            print(f"| {arm} | {fam} | {_fmt(yes)} | {_fmt(no)} |")

    print("\n## Judged expressed rate per layer, all families\n")
    print("| arm | " + " | ".join(f"L{L}" for L in JUDGED) + " |\n|---|" + "---|" * len(JUDGED))
    for arm in ARMS:
        rows = [r for f in FAMILIES for r in judged[(arm, f)]]
        print(
            f"| {arm} | "
            + " | ".join(_fmt([r["layers"][L]["expressed"] is True for r in rows]) for L in JUDGED)
            + " |"
        )

    targets = _targets()
    layers = sorted(
        {r["layer"] for r in _jsonl(out / "readouts" / ARMS[0] / f"{FAMILIES[0]}.jsonl")}
    )
    print("\n## Exact English target per layer (free, not the judge): items with a hit\n")
    print("| arm | " + " | ".join(f"L{L}" for L in layers) + " |\n|---|" + "---|" * len(layers))
    for arm in ARMS:
        hits: dict[int, set[str]] = {L: set() for L in layers}
        ids: set[str] = set()
        for fam in FAMILIES:
            for r in _jsonl(out / "readouts" / arm / f"{fam}.jsonl"):
                ids.add(r["id"])
                if _hit(r, targets[r["id"]]):
                    hits[r["layer"]].add(r["id"])
        print(f"| {arm} (n={len(ids)}) | " + " | ".join(str(len(hits[L])) for L in layers) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("raw", "null", "summary"))
    ap.add_argument("--arm", default="olens")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--limit", type=int, default=30)
    ns = ap.parse_args()
    if ns.command == "raw":
        raw(ns.out, ns.model, ns.limit)
    elif ns.command == "null":
        null(ns.out, ns.arm)
    else:
        summary(ns.out)
