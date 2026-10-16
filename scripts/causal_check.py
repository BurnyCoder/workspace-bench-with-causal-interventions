"""The J-lens paper's causal tests on the bridge concept (docs/causal.md): ``ablate`` (does the
answer depend on the direction the lens reads?) and ``swap`` (does clamping the bridge's lens
coordinate to another concept's give that concept's answer?).

A dev script, not part of the package (GPU for ``ablate`` and ``swap``, nothing for ``summary``):
    python scripts/causal_check.py ablate --set probe_swap --out outputs/causal
    python scripts/causal_check.py swap --set probe_swap --out outputs/causal
    python scripts/causal_check.py summary --out outputs/causal
(sets: probe_swap, multihop, multilingual)
"""

import argparse
import hashlib
import json
import math
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
from wsbench.produce.intervene import (
    NEAR_PARALLEL_ABS_COSINE,
    RANDOM_NULL_POOL,
    ablate,
    abs_cosine,
    answer_span,
    bridge_token,
    clamp,
    clamp_basis,
    coords,
    edit,
    kl,
    overlaps,
    replay,
    score,
    swapped_targets,
    unit,
    variant_ids,
)
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
SCALES = (0.5, 1.0, 2.0, 4.0)  # J++ PROBE_SWAP_SCALES; 1.0 is the pre-registered one
SWAPS = ("j_swap", "logit_swap", "null_swap", "answer_swap")  # each at every scale, plus "zero"
REFERENCE = (
    "Gurnee et al. (2026): probe-swap J-lens top-1 60% (Sonnet 4.5); two-hop swap 54% (Haiku "
    "4.5), 70% (Sonnet 4.5). J++ on Qwen3.6-27B: J-lens 49.3%, J++ lens 59.7%."
)
LANG = {"french": "fr", "frn": "fr", "german": "de", "spanish": "es", "portuguese": "pt"}
LANG |= {"italian": "it", "dutch": "nl"}
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


def swap_group(family: str, item_id: str) -> str:
    """Items a bridge can be swapped between, so the partner's answer is the counterfactual:
    multihop by template (``group_key``), multilingual by language and category
    (``b3-ml-es-opposite-hot`` -> ``es-opposite``, ``french-number-seven`` -> ``fr-number``)."""
    if family != "multilingual":
        return group_key(family, item_id)
    parts = item_id.split("-")
    lang, cat = (parts[2], parts[3]) if item_id.startswith("b3-ml-") else (parts[0], parts[1])
    return f"{LANG.get(lang, lang)}-{cat}"


def swap_partner(rows: list[dict], k: int) -> dict | None:
    """The next row of row ``k``'s swap group in bank order (cyclic) whose bridge and answer
    overlap neither row ``k``'s bridge nor its answer; ``None`` when the group has none."""
    me, n = rows[k], len(rows)
    for d in range(1, n):
        r = rows[(k + d) % n]
        if r["swap_group"] != me["swap_group"]:
            continue
        clash = [(r["bridge"][0], b) for b in me["bridge"]]
        clash += [(r["bridge"][0], me["answer"]), (r["answer"], me["answer"])]
        if not any(overlaps(a, b) for a, b in clash):
            return r
    return None


def swap_specs(name: str, out: Path) -> list[dict]:
    """``load_specs`` plus ``swap_to`` / ``swap_answer`` / ``partner``: probe-swap's own, or the
    bank partner's bridge and answer (a row without a partner has none and is excluded)."""
    rows = load_specs(name, out)
    if name == "probe_swap":
        return [r | {"swap_to": r["control"][0]} for r in rows]
    for r in rows:
        r["swap_group"] = swap_group(name, r["id"])
    for k, r in enumerate(rows):
        partner = swap_partner(rows, k)
        if partner:
            r |= {
                "swap_to": partner["bridge"][0],
                "swap_answer": partner["answer"],
                "partner": partner["id"],
            }
    return rows


# ---------------------------------------------------------------- the GPU runs


class Lab:
    """The model, its J-lens and the per-token directions both experiments edit along."""

    def __init__(self, model: str, band: Sequence[int]) -> None:
        import torch

        from wsbench.produce import Backend, JLens

        torch.set_grad_enabled(False)  # inference only; keeps W_U's fp32 copy out of autograd
        self.torch = torch
        self.b = b = Backend.load(model)
        self.tok = b.tokenizer
        self.jl = JLens(filename=JLENS_FILE)
        self.jl.bind(b)
        self.layers = list(range(band[0], band[1] + 1))
        ones = torch.ones(b.unembed.shape[1], device=b.device, dtype=b.unembed.dtype)
        self.gain = b.final_norm(ones).float()  # the final RMSNorm's gain: norm(ones) = (1 + w)
        vocab = b.unembed.shape[0]
        names = [
            t or "" for t in self.tok.convert_ids_to_tokens(list(range(min(len(self.tok), vocab))))
        ]
        self.words = [i for i, t in enumerate(names) if re.fullmatch(r"Ġ?[a-z]{4,}", t)]
        shown = [t.replace("Ġ", " ").replace("Ċ", " ").replace("ĉ", " ") for t in names]
        fmt = [not any(ch.isalnum() for ch in t) for t in shown] + [True] * (vocab - len(shown))
        self.formatting = torch.tensor(fmt, device=b.device)  # whitespace, punctuation, specials

    def jdirs(self, t: int) -> dict:
        """Unit J-lens vectors ``W_U[t] @ J[layer]`` (the readout's own orientation)."""
        return {L: unit(self.jl._w_u[t] @ self.jl._jac[L]) for L in self.layers}

    def logit_dirs(self, t: int) -> dict:
        """The logit-lens direction ``g * W_U[t]`` (``g`` the final norm's gain), every layer."""
        return dict.fromkeys(self.layers, unit(self.gain * self.b.unembed[t].float()))

    def mean_dirs(self, ts: list[int]) -> dict:
        return {L: unit(sum(self.jdirs(t)[L] for t in ts)) for L in self.layers}

    def isotropic(self, seed: int) -> dict:
        g = self.torch.Generator().manual_seed(seed)
        r = self.torch.randn(self.gain.numel(), generator=g).to(self.b.device)
        return dict.fromkeys(self.layers, unit(r))

    def rank(self, h, layer: int, t: int) -> int:
        """0-based J-lens readout rank of token ``t`` for residual ``h`` at ``layer``."""
        s = self.jl.scores(h, layer)
        return int((s > s[t]).sum())

    def loading(self, ids: list[int], p: int, t: int) -> dict[int, int]:
        """The bridge's J-lens rank per band layer at the last visible prompt token."""
        q = max(i for i in range(p) if self.tok.decode([ids[i]]).strip())
        acts = self.b.capture(ids[:p], self.layers, [q])
        return {L: self.rank(acts[L][0], L, t) for L in self.layers}

    def min_rank(self, lp, ids: list[int], words_only: bool = False) -> int | None:
        """1-based rank of the best of ``ids`` in log-probs ``lp`` (J++
        ``min_rank_over_candidates``); with ``words_only`` formatting tokens are removed first
        (J++'s word rank). ``None`` when ``ids`` is empty."""
        if not ids:
            return None
        if words_only:
            lp = lp.masked_fill(self.formatting[: lp.numel()], float("-inf"))
        best = lp[ids].max()
        return int((lp > best).sum()) + 1


def run_set(name: str, out: Path, model: str, band: Sequence[int], limit: int) -> None:
    lab = Lab(model, band)
    b, tok, layers, torch = lab.b, lab.tok, lab.layers, lab.torch
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
            ranks = lab.loading(ids, p, t_b)
            dj, dc = lab.jdirs(t_b), lab.jdirs(t_c)
            dirs = {
                "samecat": dc,
                "samecat_orth": {L: unit(dc[L] - (dc[L] @ dj[L]) * dj[L]) for L in layers},
                "randvocab": lab.mean_dirs(random.Random(seed).sample(lab.words, 8)),
                "isotropic": lab.isotropic(seed),
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
            dl = lab.logit_dirs(t_b)
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


def _swap_exclusion(tok, spec: dict, t_b, form_b, t_t, form_t) -> str | None:
    if "swap_to" not in spec:
        return "no swap partner"
    if t_b is None or t_t is None:
        return form_b if t_b is None else form_t
    if t_b == t_t:
        return "bridge and swap target are one token"
    if overlaps(form_b, spec["answer"]) or overlaps(form_t, spec["answer"]):
        return "bridge overlaps the answer"
    if overlaps(spec["swap_answer"], spec["answer"]):
        return "answers overlap"
    return None


def _measure(lab: Lab, ctx: dict, fn, swap_layers: list[int]) -> dict:
    """One clamp condition: the original-answer pass gives the next-token ranks (J++'s success =
    the swap answer's best single-token surface at rank 1, top 5 and word rank) and the original's
    log p; the swap-answer pass gives whether greedy decoding produces the swap answer."""
    ids, ids_w, p = ctx["ids"], ctx["ids_w"], ctx["p"]
    with edit(lab.b.blocks, swap_layers, fn, 1, p):
        ro, rw = score(lab.b.model, ids, p), score(lab.b.model, ids_w, p)
    rank = lab.min_rank(ro["next"], ctx["success"])
    word = lab.min_rank(ro["next"], ctx["success"], words_only=True)
    return {
        "success": None if rank is None else rank == 1,
        "success_top5": None if rank is None else rank <= 5,
        "success_word_top1": None if word is None else word == 1,
        "success_rank": rank,
        "baseline_rank": lab.min_rank(ro["next"], ctx["baseline"]),
        "greedy_swap": rw["correct"],
        "orig_correct": ro["correct"],
        "dlogp_swap": rw["logp"] - ctx["clean_w"]["logp"],
        "dlogp_orig": ro["logp"] - ctx["clean_o"]["logp"],
        "kl": kl(ctx["clean_o"]["next"], ro["next"]),
        "leak": int(ro["next"].argmax()) in ctx["leak"],
        "n_layers": len(swap_layers),
    }


def run_swaps(name: str, out: Path, model: str, band: Sequence[int], limit: int) -> None:
    """The clamp swap (J++ runner conventions, ported in ``wsbench.produce.intervene``): per
    trial and condition, a clean recording pass for the clean coordinates, then at each scale one
    pass with the original answer appended (next-token ranks and the original's log p) and one
    with the swap answer (whether greedy decoding gives it). Rows go to ``<set>/swaps.jsonl``."""
    lab = Lab(model, band)
    b, tok, layers = lab.b, lab.tok, lab.layers
    specs = swap_specs(name, out)[: limit or None]
    by_id = {s["id"]: s for s in load_specs(name, out)}
    known: dict[str, bool] = {}  # does the clean model answer the partner item itself?
    dst = out / name / "swaps.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    with dst.open("w", encoding="utf-8") as fh:
        for spec in specs:
            row: dict = {"set": name, **{k: spec.get(k) for k in ("id", "group", "answer")}}
            row |= {k: spec.get(k) for k in ("swap_to", "swap_answer", "partner")}
            t_b, form_b = bridge_token(tok, spec["bridge"])
            t_t, form_t = bridge_token(tok, [spec["swap_to"]]) if "swap_to" in spec else (None, "")
            reason = _swap_exclusion(tok, spec, t_b, form_b, t_t, form_t)
            if not reason:
                try:
                    ids, p, stripped = answer_span(tok, spec["prompt"], spec["answer"])
                    ids_w, p_w, _ = answer_span(tok, spec["prompt"], spec["swap_answer"])
                    if p_w != p or ids_w[:p] != ids[:p]:
                        reason = "prompt rows differ"
                except ValueError as e:
                    reason = str(e)
            if reason:
                skipped[reason.split(" of ")[0]] += 1
                fh.write(json.dumps({**row, "excluded": reason}, ensure_ascii=False) + "\n")
                continue
            seed = zlib.crc32(spec["id"].encode())
            noun = random.Random(seed).choice(RANDOM_NULL_POOL)
            success_ids = variant_ids(tok, spec["swap_answer"].strip())
            baseline_ids = variant_ids(tok, spec["answer"].strip())
            leak_ids = set(variant_ids(tok, form_b.strip()) + variant_ids(tok, form_t.strip()))
            clean_o, clean_w = score(b.model, ids, p), score(b.model, ids_w, p)
            if spec.get("partner") and spec["partner"] not in known:
                pa = by_id[spec["partner"]]
                try:
                    ids_p, p_p, _ = answer_span(tok, pa["prompt"], pa["answer"])
                    known[spec["partner"]] = score(b.model, ids_p, p_p)["correct"]
                except ValueError:
                    known[spec["partner"]] = False
            t_n = bridge_token(tok, [noun])[0]
            t_a, t_sa = (
                bridge_token(tok, [spec["answer"]])[0],
                bridge_token(tok, [spec["swap_answer"]])[0],
            )
            pairs = {
                "j_swap": (lab.jdirs(t_b), lab.jdirs(t_t)),
                "logit_swap": (lab.logit_dirs(t_b), lab.logit_dirs(t_t)),
                "null_swap": (lab.jdirs(t_b), lab.jdirs(t_n)),
            }
            if t_a is not None and t_sa is not None and t_a != t_sa:
                pairs["answer_swap"] = (lab.jdirs(t_a), lab.jdirs(t_sa))
            dj, dt = pairs["j_swap"]
            row |= {
                "bridge": form_b,
                "swap_form": form_t,
                "null_noun": noun,
                "stripped": stripped,
                "n_swap_answer_tokens": len(ids_w) - p,
                "clean_correct": clean_o["correct"],
                "clean_swap_greedy": clean_w["correct"],
                "clean_success_rank": lab.min_rank(clean_o["next"], success_ids),
                "partner_known": known.get(spec.get("partner")),
                "rank_by_layer": lab.loading(ids, p, t_b),
                "cos_bridge_swap": statistics.fmean(float(dj[L] @ dt[L]) for L in layers),
            }

            ctx = {"ids": ids, "ids_w": ids_w, "p": p, "clean_o": clean_o, "clean_w": clean_w}
            ctx |= {"success": success_ids, "baseline": baseline_ids, "leak": leak_ids}
            for cond, (ds, dt_) in pairs.items():
                ok_layers = [
                    L for L in layers if abs_cosine(ds[L], dt_[L]) <= NEAR_PARALLEL_ABS_COSINE
                ]
                bases = {L: clamp_basis(ds[L], dt_[L]) for L in ok_layers}
                rec: dict = {}
                with edit(b.blocks, ok_layers, coords(bases, rec), 1, p):
                    check = score(b.model, ids, p)
                shape_ok = rec[ok_layers[0]].shape == (p - 1, 2)
                if abs(check["logp"] - clean_o["logp"]) > 1e-3 or not shape_ok:
                    raise RuntimeError(f"{spec['id']}: the recording pass changed the output")
                row[cond] = {}
                for sc in SCALES:
                    targets = {L: swapped_targets(rec[L], sc) for L in ok_layers}
                    row[cond][str(sc)] = _measure(lab, ctx, clamp(bases, targets), ok_layers)
                if cond == "j_swap":
                    zeros = {L: rec[L] * 0 for L in ok_layers}
                    row["zero"] = _measure(lab, ctx, clamp(bases, zeros), ok_layers)
            again = score(b.model, ids, p)
            if abs(again["logp"] - clean_o["logp"]) > 1e-3:
                raise RuntimeError(f"{spec['id']}: an edit leaked into the clean pass")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            j1 = row["j_swap"]["1.0"]
            log(
                f"{name}/{spec['id']}: {form_b.strip()}->{form_t.strip()} "
                f"({spec['answer']}->{spec['swap_answer']}) "
                f"clean {'ok' if clean_o['correct'] else 'wrong'} "
                f"| j@1 rank {j1['success_rank']} greedy {j1['greedy_swap']} "
                f"| logit@1 rank {row['logit_swap']['1.0']['success_rank']} "
                f"| null@1 rank {row['null_swap']['1.0']['success_rank']}"
            )
    n = len(specs) - sum(skipped.values())
    log(f"{name}: {n} swap trials, excluded {dict(skipped)} -> {dst}")


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for ``k`` successes in ``n`` trials (Wilson 1927; the interval
    the paper's probe-swap figure reports)."""
    if n == 0:
        return (math.nan, math.nan)
    ph, d = k / n, 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p for paired binary outcomes: ``b`` and ``c`` are the discordant
    counts, tested against Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _rate(flags: list[bool]) -> str:
    if not flags:
        return "—"
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]"


def eligible(t: dict) -> bool:
    """A primary swap trial: scored, the clean model answers it, the swap answer has a
    single-token surface and is not already the clean top-1."""
    return (
        "excluded" not in t
        and t["clean_correct"]
        and t["clean_success_rank"] is not None
        and t["clean_success_rank"] > 1
    )


def summarize_swaps(rows: list[dict]) -> str:
    scored = [t for t in rows if "excluded" not in t]
    prim = pooled([t for t in scored if eligible(t)])
    lines = [
        "## Swap trials",
        "",
        "| set | items | excluded | scored | clean-correct | primary |",
        "|---|---|---|---|---|---|",
    ]
    for s in SETS:
        all_s = [t for t in rows if t["set"] == s]
        sc = [t for t in scored if t["set"] == s]
        n_ok = sum(t["clean_correct"] for t in sc)
        n_p = sum(eligible(t) for t in sc)
        lines.append(
            f"| {s} | {len(all_s)} | {len(all_s) - len(sc)} | {len(sc)} | {n_ok} | {n_p} |"
        )
    excluded = Counter(t["excluded"].split(" of ")[0] for t in rows if "excluded" in t)
    lines += ["", f"Excluded: {dict(excluded)}", ""]

    def flag(t: dict, cond: str, key: str = "success", scale: str = "1.0"):
        cell = t.get(cond) if cond == "zero" else (t.get(cond) or {}).get(scale)
        return None if cell is None else cell[key]

    lines += [
        "## Primary: J-lens clamp swap at scale 1, the swap answer becomes top-1 (Wilson 95% CI)",
        "",
        "| scope | success |",
        "|---|---|",
        f"| pooled | {_rate([flag(t, 'j_swap') for t in prim])} |",
    ]
    for s in SETS:
        lines.append(f"| {s} | {_rate([flag(t, 'j_swap') for t in prim if t['set'] == s])} |")
    lines += ["", REFERENCE, "", "## Paired against the J-lens swap at scale 1 (McNemar exact)", ""]
    lines += ["| control | J only | control only | p |", "|---|---|---|---|"]
    for other in ("logit_swap", "null_swap", "zero"):
        pairs = [(flag(t, "j_swap"), flag(t, other)) for t in prim]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        b_ = sum(a and not o for a, o in pairs)
        c_ = sum(o and not a for a, o in pairs)
        lines.append(f"| {other} | {b_} | {c_} | {mcnemar(b_, c_):.2g} |")
    lines += ["", "## Every condition and scale (primary trials)", ""]
    lines += [
        "| condition | scale | top-1 | top-5 | word top-1 | greedy swap | original kept | "
        "Δlog p(swap) | Δlog p(original) | leak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cond in (*SWAPS, "zero"):
        for sc in ("—",) if cond == "zero" else tuple(str(x) for x in SCALES):
            cells = [t[cond] if cond == "zero" else t[cond].get(sc) for t in prim if cond in t]
            cells = [c for c in cells if c]
            if not cells:
                continue
            n = len(cells)

            def frac(key: str, cells: list[dict] = cells, n: int = n) -> str:
                return f"{sum(bool(c[key]) for c in cells)}/{n}"

            lines.append(
                f"| {cond} | {sc} | {frac('success')} | {frac('success_top5')} | "
                f"{frac('success_word_top1')} | {frac('greedy_swap')} | {frac('orig_correct')} | "
                f"{statistics.fmean(c['dlogp_swap'] for c in cells):+.2f} | "
                f"{statistics.fmean(c['dlogp_orig'] for c in cells):+.2f} | {frac('leak')} |"
            )
    top = [t for t in prim if min(t["rank_by_layer"].values()) < 10]
    rest = [t for t in prim if min(t["rank_by_layer"].values()) >= 10]
    lines += [
        "",
        "## Does success follow the lens reading and the model's knowledge? (J swap, scale 1)",
        "",
        "- bridge in the J-lens top 10 at some band layer: "
        f"{_rate([flag(t, 'j_swap') for t in top])}",
        f"- bridge not in the top 10: {_rate([flag(t, 'j_swap') for t in rest])}",
    ]
    for known in (True, False):
        sel = [flag(t, "j_swap") for t in prim if t.get("partner_known") is known]
        lines.append(f"- banks, the model answers the partner item itself={known}: {_rate(sel)}")
    every = pooled([t for t in scored if t["clean_success_rank"] is not None])
    lines += [
        "",
        "## Secondary: every scored trial with a single-token swap answer (pooled)",
        "",
        f"- J swap top-1 at scale 1: {_rate([flag(t, 'j_swap') for t in every])}",
        f"- greedy swap answer, J at scale 1: "
        f"{_rate([flag(t, 'j_swap', 'greedy_swap') for t in pooled(scored)])}",
    ]
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
    ap.add_argument("command", choices=("ablate", "swap", "summary"))
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--band", default="9-19", help="first-last block, inclusive")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judged", type=Path, default=Path("outputs/small_model_n30"))
    ns = ap.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    if ns.command in ("ablate", "swap"):
        (ns.out / "logs").mkdir(exist_ok=True)
        stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        _LOG.append(ns.out / "logs" / f"{stamp}_{ns.command}_{ns.set}.log")
        lo, hi = (int(x) for x in ns.band.split("-"))
        run = run_set if ns.command == "ablate" else run_swaps
        run(ns.set, ns.out, ns.model, (lo, hi), ns.limit)
    else:

        def read(kind: str) -> list[dict]:
            files = [ns.out / s / f"{kind}.jsonl" for s in SETS]
            return [
                json.loads(line)
                for f in files
                if f.exists()
                for line in f.read_text(encoding="utf-8").splitlines()
                if line
            ]

        text = summarize(read("trials"), _judged(ns.judged)) if read("trials") else ""
        if read("swaps"):
            text += ("\n\n" if text else "") + summarize_swaps(read("swaps"))
        (ns.out / "summary.md").write_text(text + "\n", encoding="utf-8")
        print(text)
