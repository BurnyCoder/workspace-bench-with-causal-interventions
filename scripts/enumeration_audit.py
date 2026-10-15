"""Plan 0012 §4.2: how many PASSING cells rest on a readout sample that enumerates 5+ distinct
same-type candidates (months, weekdays, seasons, colours, number words or numerals).

A dev script, not part of the package; reads finished ``wsbench`` result dirs and the readouts
they were judged from. The count is a screen, not a verdict: a listing sample may still single
the target out. Prose arms only: a token bag's "sample" is one token, so it never lists 5.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from wsbench.mc import fold

LEXICONS = {
    "month": "january february march april may june july august september october november "
    "december",
    "weekday": "monday tuesday wednesday thursday friday saturday sunday",
    "season": "spring summer autumn fall winter",
    "colour": "red orange yellow green blue purple violet pink brown black white grey gray",
    "number_word": "zero one two three four five six seven eight nine ten eleven twelve",
}
LEX = {k: set(v.split()) for k, v in LEXICONS.items()}
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
BANK = ["association", "basic_readout", "multihop", "multilingual", "poetry", "typo"]


def enumerates(sample: str, n: int = 5) -> str | None:
    words = set(re.findall(r"[a-z]+", fold(sample)))
    for kind, lex in LEX.items():
        if len(words & lex) >= n:
            return kind
    if len(set(_NUM.findall(sample))) >= n:
        return "numerals"
    return None


def readouts(path: Path) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["id"], r["layer"])].extend(r.get("samples") or r.get("tokens") or [])
    return out


def bank_family(res: Path, rd: Path) -> dict[str, int]:
    d = json.loads((res / "results.json").read_text())
    ro = readouts(rd)
    n_pass_cells = n_enum = 0
    kinds: dict[str, int] = defaultdict(int)
    for row in d["rows"]:
        for layer, v in row["layers"].items():
            if not v.get("expressed"):
                continue
            n_pass_cells += 1
            q = fold(v.get("quote", ""))
            sample = next((s for s in ro[(row["id"], int(layer))] if q and q in fold(s)), "")
            k = enumerates(sample)
            if k:
                n_enum += 1
                kinds[k] += 1
    return {"passing_cells": n_pass_cells, "enumerating": n_enum, **kinds}


def readouts_by_pos(path: Path) -> dict[tuple[str, int, int], list[str]]:
    out: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["id"], r["layer"], r["pos"])].extend(r.get("samples") or r.get("tokens") or [])
    return out


def numeric_family(res: Path, rd: Path) -> dict[str, int]:
    """Passing cells whose OWN samples hold 5+ distinct numerals: arithmetic rows list their
    hit cells as ``hits_at`` (layer, pos); chain's last-token rows list ``hitting_layers``."""
    d = json.loads((res / "results.json").read_text())
    by_pos = readouts_by_pos(rd)
    by_layer = readouts(rd)
    n_pass = n_enum = 0
    for row in d["rows"]:
        if row.get("hits_at"):
            cells = [by_pos[(row["id"], int(L), int(p))] for L, p in row["hits_at"]]
        else:
            cells = [by_layer[(row["id"], int(L))] for L in row.get("hitting_layers") or []]
        for samples in cells:
            n_pass += 1
            n_enum += any(len(set(_NUM.findall(s))) >= 5 for s in samples)
    return {"passing_cells": n_pass, "enumerating": n_enum}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-results", type=Path, required=True)
    ap.add_argument("--bank-readouts", type=Path, required=True)
    ap.add_argument("--numeric", nargs="*", default=[], help="family=results_dir=readouts_file")
    a = ap.parse_args()
    out = {}
    for fam in BANK:
        out[fam] = bank_family(a.bank_results / fam, a.bank_readouts / f"{fam}.jsonl")
    for spec in a.numeric:
        fam, res, rd = spec.split("=")
        out[fam] = numeric_family(Path(res), Path(rd))
    for fam, r in out.items():
        share = r["enumerating"] / r["passing_cells"] if r["passing_cells"] else 0.0
        kinds = {k: v for k, v in r.items() if k not in ("passing_cells", "enumerating")}
        print(
            f"{fam:<26} passing cells {r['passing_cells']:>5}  enumerating {r['enumerating']:>4}"
            f"  ({100 * share:.1f}%)  {kinds}"
        )


if __name__ == "__main__":
    main()
