"""Targeted ablation of the J-lens bridge direction (docs/causal.md): does the answer depend on
the concept the lens reads, more than on an equal edit along another concept?

A dev script, not part of the package (GPU for ``ablate``, nothing for ``summary``):
    python scripts/causal_check.py ablate --set probe_swap --out outputs/causal
    python scripts/causal_check.py ablate --set multihop --out outputs/causal
    python scripts/causal_check.py ablate --set multilingual --out outputs/causal
    python scripts/causal_check.py summary --out outputs/causal
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import urllib.request
import zlib
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from wsbench.banks import item_targets, load_bank
from wsbench.mc import fold
from wsbench.produce.intervene import answer_span, bridge_token, overlaps, variant_ids
from wsbench.registry import REPO_ROOT
from wsbench.results import bootstrap_ci

PROBE_SWAP_URL = (
    "https://raw.githubusercontent.com/anthropics/jacobian-lens/"
    "581d398613e5602a5af361e1c34d3a92ea82ba8e/data/experiments/probe-swap.json"
)
SETS = ("probe_swap", "multihop", "multilingual")
JLENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
# every edit but the first is a control or a comparison for ablate_j
CONDITIONS = (
    "ablate_j",  # project the bridge's J-lens vector out (the paper's ablation)
    "samecat",  # replay the same per-cell magnitudes along a same-category concept's vector
    "samecat_orth",  # ... along that vector made orthogonal to the bridge's
    "randvocab",  # ... along the mean J-lens vector of 8 random words (the shared component)
    "isotropic",  # ... along a random unit vector
    "ablate_logit",  # project the bridge's logit-lens direction out
    "ablate_j_final",  # the J-lens ablation at the final prompt position only
)
_LOG: list[Path] = []


def log(msg: str) -> None:
    line = f"[{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {msg}"
    print(line, flush=True)
    for path in _LOG:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------- trial specs (no torch)


def group_key(family: str, item_id: str) -> str:
    """Items sharing a template: multihop by first and last id part (``holiday-*-monthnum``),
    multilingual by language prefix."""
    parts = item_id.split("-")
    return f"{parts[0]}-{parts[-1]}" if family == "multihop" else parts[0]


def control_words(rows: list[dict], k: int) -> list[str]:
    """Control candidates for row ``k``, best first: the latents of the same template group in
    bank order after ``k`` (cyclic), then every other row's; never a word overlapping the bridge."""
    n = len(rows)
    order = [rows[(k + d) % n] for d in range(1, n)]
    same = [r for r in order if r["group"] == rows[k]["group"]]
    rest = [r for r in order if r["group"] != rows[k]["group"]]
    own = rows[k]["bridge"]
    return [
        r["bridge"][0] for r in same + rest if not any(overlaps(r["bridge"][0], b) for b in own)
    ]


def load_specs(name: str, out: Path) -> list[dict]:
    """``{id, prompt, bridge[], answer, control[], group, swap_answer?}`` per item."""
    if name == "probe_swap":
        path = out / "probe-swap.json"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(PROBE_SWAP_URL, timeout=60).read())
        data = path.read_bytes()
        (out / "probe-swap.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
        return [
            {
                "id": it["name"],
                "prompt": it["prompt"],
                "bridge": [it["intermediate"]],
                "answer": it["answer"],
                "control": [it["swap_to"]],
                "group": it["category"],
                "swap_answer": it["swap_answer"],
            }
            for it in json.loads(data)["items"]
        ]
    _, items = load_bank(REPO_ROOT / "evals" / name / "items.json")
    rows = [
        {
            "id": it["id"],
            "prompt": it["prompt"],
            "bridge": item_targets(it),
            "answer": it["target"],
            "group": group_key(name, it["id"]),
        }
        for it in items
    ]
    for k, r in enumerate(rows):
        r["control"] = control_words(rows, k)
    return rows


# ---------------------------------------------------------------- the GPU run


def run_set(name: str, out: Path, model: str, band: Sequence[int], limit: int) -> None:
    import torch

    from wsbench.produce import Backend, JLens
    from wsbench.produce.intervene import ablate, edit, kl, replay, score, unit

    torch.set_grad_enabled(False)  # inference only; also keeps W_U's fp32 copy out of autograd
    b = Backend.load(model)
    tok = b.tokenizer
    jl = JLens(filename=JLENS_FILE)
    jl.bind(b)
    layers = list(range(band[0], band[1] + 1))
    gain = b.final_norm(torch.ones(b.unembed.shape[1], device=b.device, dtype=b.unembed.dtype))
    gain = gain.float()  # the final RMSNorm's per-dimension gain: norm(ones) = (1 + w)
    names = tok.convert_ids_to_tokens(list(range(min(len(tok), b.unembed.shape[0]))))
    words = [i for i, t in enumerate(names) if re.fullmatch(r"Ġ?[a-z]{4,}", t or "")]

    def jdirs(t: int) -> dict[int, torch.Tensor]:
        """Unit J-lens vectors ``W_U[t] @ J[layer]`` (the readout's own orientation)."""
        return {L: unit(jl._w_u[t] @ jl._jac[L]) for L in layers}

    def mean_dirs(ts: list[int]) -> dict[int, torch.Tensor]:
        return {L: unit(sum(jdirs(t)[L] for t in ts)) for L in layers}

    def isotropic(seed: int) -> dict[int, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        return dict.fromkeys(layers, unit(torch.randn(gain.numel(), generator=g).to(b.device)))

    def rank(h: torch.Tensor, layer: int, t: int) -> int:
        s = jl.scores(h, layer)
        return int((s > s[t]).sum())

    specs = load_specs(name, out)[: limit or None]
    dst = out / name / "trials.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    with dst.open("w", encoding="utf-8") as fh:
        for spec in specs:
            row: dict = {"set": name, **{k: spec[k] for k in ("id", "group", "answer")}}
            t_b, form_b = bridge_token(tok, spec["bridge"])
            ctrl = [(w, *bridge_token(tok, [w])) for w in spec["control"]]
            ctrl = [(w, t, f) for w, t, f in ctrl if t is not None and t != t_b]
            reason = (
                form_b
                if t_b is None
                else "bridge overlaps the answer"
                if overlaps(form_b, spec["answer"])
                else "no single-token control"
                if not ctrl
                else None
            )
            try:
                ids, p, stripped = answer_span(tok, spec["prompt"], spec["answer"])
            except ValueError as e:
                reason = reason or str(e)
            if reason:
                skipped[reason.split(" of ")[0]] += 1
                fh.write(json.dumps({**row, "excluded": reason}, ensure_ascii=False) + "\n")
                continue
            _, t_c, form_c = ctrl[0]
            seed = zlib.crc32(spec["id"].encode())
            clean = score(b.model, ids, p)
            q = max(i for i in range(p) if tok.decode([ids[i]]).strip())  # last visible token
            acts = b.capture(ids[:p], layers, [q])
            ranks = {L: rank(acts[L][0], L, t_b) for L in layers}
            dj, dc = jdirs(t_b), jdirs(t_c)
            dirs = {
                "samecat": dc,
                "samecat_orth": {L: unit(dc[L] - (dc[L] @ dj[L]) * dj[L]) for L in layers},
                "randvocab": mean_dirs(random.Random(seed).sample(words, 8)),
                "isotropic": isotropic(seed),
            }
            res: dict[str, dict] = {}
            rec: dict = {}
            with edit(b.blocks, layers, ablate(dj, rec), 1, p):
                res["ablate_j"] = score(b.model, ids, p)
            removed_j = float(sum((rec[L] ** 2).sum() for L in layers) ** 0.5)
            for cond, d in dirs.items():
                with edit(b.blocks, layers, replay(rec, d), 1, p):
                    res[cond] = score(b.model, ids, p)
            rec_l: dict = {}
            dl = dict.fromkeys(layers, unit(gain * b.unembed[t_b].float()))
            with edit(b.blocks, layers, ablate(dl, rec_l), 1, p):
                res["ablate_logit"] = score(b.model, ids, p)
            with edit(b.blocks, layers, ablate(dj, {}), p - 1, p):
                res["ablate_j_final"] = score(b.model, ids, p)
            again = score(b.model, ids, p)
            if abs(again["logp"] - clean["logp"]) > 1e-3:
                raise RuntimeError(f"{spec['id']}: an edit leaked into the clean pass")
            excl = variant_ids(tok, form_b.strip())
            row |= {
                "bridge": form_b,
                "control": form_c,
                "stripped": stripped,
                "n_answer_tokens": len(ids) - p,
                "clean_logp": clean["logp"],
                "clean_correct": clean["correct"],
                "bridge_in_next_top10": t_b in torch.topk(clean["next"], 10).indices.tolist(),
                "rank_by_layer": ranks,
                "cos_bridge_control": statistics.fmean(float(dj[L] @ dc[L]) for L in layers),
                "removed_norm_j": removed_j,
                "removed_norm_logit": float(sum((rec_l[L] ** 2).sum() for L in layers) ** 0.5),
            }
            for cond in CONDITIONS:
                r = res[cond]
                row[cond] = {
                    "dlogp": r["logp"] - clean["logp"],
                    "correct": r["correct"],
                    "kl": kl(clean["next"], r["next"]),
                    "kl_excl": kl(clean["next"], r["next"], excl),
                }
            if "swap_answer" in spec:
                ids_s, p_s, _ = answer_span(tok, spec["prompt"], spec["swap_answer"])
                if p_s == p:
                    t_s = ids_s[p_s]
                    row["swap_first_dlogp_samecat"] = float(
                        res["samecat"]["next"][t_s] - clean["next"][t_s]
                    )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            log(
                f"{name}/{spec['id']}: clean {'ok' if clean['correct'] else 'wrong'} "
                f"dlogp ablate_j {row['ablate_j']['dlogp']:+.2f} "
                f"samecat {row['samecat']['dlogp']:+.2f} logit {row['ablate_logit']['dlogp']:+.2f} "
                f"min rank {min(ranks.values())}"
            )
    log(f"{name}: {len(specs) - sum(skipped.values())} trials, excluded {dict(skipped)} -> {dst}")


# ---------------------------------------------------------------- summary (no torch)


def excess(t: dict) -> float:
    """How much more the J-lens ablation lowers log p(answer) than the same-category replay
    (negative = the bridge's own direction matters more)."""
    return t["ablate_j"]["dlogp"] - t["samecat"]["dlogp"]


def _ci(xs: list[float]) -> str:
    if not xs:
        return "—"
    lo, hi = bootstrap_ci(xs)
    return f"{statistics.fmean(xs):+.2f} [{lo:+.2f}, {hi:+.2f}] (n={len(xs)})"


def ratio_ci(
    a: list[float], b: list[float], n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """``mean(a) / mean(b)`` with a paired percentile bootstrap (a ratio of means, not a mean of
    ratios, which a near-zero denominator blows up)."""
    rng = random.Random(seed)
    k = len(a)
    stats = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        den = sum(b[i] for i in idx)
        if den > 0:
            stats.append(sum(a[i] for i in idx) / den)
    stats.sort()
    return (
        sum(a) / sum(b),
        stats[round(0.025 * (len(stats) - 1))],
        stats[round(0.975 * (len(stats) - 1))],
    )


def pooled(trials: list[dict]) -> list[dict]:
    """Every scored trial, with probe-swap items that repeat a multihop bank item (same bridge and
    answer) dropped, so one fact does not count twice."""
    seen = {
        (fold(t["bridge"].strip()), fold(t["answer"])) for t in trials if t["set"] == "multihop"
    }
    return [
        t
        for t in trials
        if t["set"] != "probe_swap" or (fold(t["bridge"].strip()), fold(t["answer"])) not in seen
    ]


def summarize(trials: list[dict], judged: dict[str, bool] | None = None) -> str:
    scored = [t for t in trials if "excluded" not in t]
    ok = pooled([t for t in scored if t["clean_correct"]])
    lines = [
        "## Trials",
        "",
        "| set | items | excluded | scored | clean-correct |",
        "|---|---|---|---|---|",
    ]
    for s in SETS:
        all_s = [t for t in trials if t["set"] == s]
        sc = [t for t in scored if t["set"] == s]
        n_ok = sum(t["clean_correct"] for t in sc)
        lines.append(f"| {s} | {len(all_s)} | {len(all_s) - len(sc)} | {len(sc)} | {n_ok} |")
    lines += [
        "",
        "## Primary: Δlog p(answer), J-lens ablation minus the same-category replay "
        "(clean-correct, pooled; mean [95% CI])",
        "",
        "| scope | ablate_j - samecat |",
        "|---|---|",
        f"| pooled | {_ci([excess(t) for t in ok])} |",
    ]
    for s in SETS:
        lines.append(f"| {s} | {_ci([excess(t) for t in ok if t['set'] == s])} |")
    lines += ["", "## Every condition (clean-correct, pooled)", ""]
    lines += [
        "| condition | Δlog p(answer) | still correct | KL at the answer |",
        "|---|---|---|---|",
    ]
    for c in CONDITIONS:
        acc = sum(t[c]["correct"] for t in ok)
        lines.append(
            f"| {c} | {_ci([t[c]['dlogp'] for t in ok])} | {acc}/{len(ok)} | "
            f"{statistics.fmean(t[c]['kl'] for t in ok):.2f} |"
            if ok
            else f"| {c} | — | — | — |"
        )
    if ok:
        r, lo, hi = ratio_ci(
            [t["ablate_j"]["kl"] for t in ok], [t["ablate_logit"]["kl"] for t in ok]
        )
        rx, lox, hix = ratio_ci(
            [t["ablate_j"]["kl_excl"] for t in ok], [t["ablate_logit"]["kl_excl"] for t in ok]
        )
        nj = statistics.fmean(t["removed_norm_j"] for t in ok)
        nl = statistics.fmean(t["removed_norm_logit"] for t in ok)
        lines += [
            "",
            "## J-lens vs logit-lens ablation (clean-correct, pooled)",
            "",
            f"- KL ratio, J / logit: {r:.2f} [{lo:.2f}, {hi:.2f}]; "
            f"excluding the bridge's own tokens: {rx:.2f} [{lox:.2f}, {hix:.2f}]",
            f"- mean removed norm: J {nj:.2f}, logit {nl:.2f}",
        ]
        top = [t for t in ok if min(t["rank_by_layer"].values()) < 10]
        rest = [t for t in ok if min(t["rank_by_layer"].values()) >= 10]
        lines += [
            "",
            "## Does the effect follow the lens reading? (clean-correct, pooled)",
            "",
            "| bridge in the J-lens top 10 at some band layer | ablate_j - samecat |",
            "|---|---|",
            f"| yes | {_ci([excess(t) for t in top])} |",
            f"| no | {_ci([excess(t) for t in rest])} |",
        ]
        if len(ok) > 2:
            rho = statistics.correlation(
                [-min(t["rank_by_layer"].values()) for t in ok],
                [t["samecat"]["dlogp"] - t["ablate_j"]["dlogp"] for t in ok],
                method="ranked",
            )
            lines.append(f"\nSpearman(bridge loading, ablation excess): {rho:+.2f}")
    allt = pooled(scored)
    lines += [
        "",
        "## Secondary: every scored trial, clean-correct or not (pooled)",
        "",
        f"- ablate_j - samecat: {_ci([excess(t) for t in allt])}",
    ]
    swaps = [t["swap_first_dlogp_samecat"] for t in ok if "swap_first_dlogp_samecat" in t]
    if swaps:
        lines.append(
            f"- positive control (probe_swap): the same-category replay lowers log p of the swap "
            f"answer's first token in {sum(x < 0 for x in swaps)}/{len(swaps)} trials"
        )
    if judged:
        bank = [t for t in scored if t["set"] != "probe_swap" and t["id"] in judged]
        for verdict in (True, False):
            sel = [excess(t) for t in bank if judged[t["id"]] is verdict]
            lines.append(
                f"- bank items the 30-item run's J-lens judge passed={verdict}: {_ci(sel)}"
            )
    return "\n".join(lines)


def _judged(root: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for fam in ("multihop", "multilingual"):
        path = root / "judged" / "jlens" / fam / "results.json"
        if path.exists():
            out |= {r["id"]: r["pass"] is True for r in json.loads(path.read_text())["rows"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("ablate", "summary"))
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--band", default="9-19", help="first-last block, inclusive")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judged", type=Path, default=Path("outputs/small_model_n30"))
    ns = ap.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    if ns.command == "ablate":
        (ns.out / "logs").mkdir(exist_ok=True)
        _LOG.append(ns.out / "logs" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{ns.set}.log")
        lo, hi = (int(x) for x in ns.band.split("-"))
        run_set(ns.set, ns.out, ns.model, (lo, hi), ns.limit)
    else:
        rows = [
            json.loads(line)
            for s in SETS
            if (ns.out / s / "trials.jsonl").exists()
            for line in (ns.out / s / "trials.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        text = summarize(rows, _judged(ns.judged))
        (ns.out / "summary.md").write_text(text + "\n", encoding="utf-8")
        print(text)
