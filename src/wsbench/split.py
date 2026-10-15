"""Report-only layer protocols (plan 0012 §5, Neel B8): a hashed dev/test split, the middle
band, a layer fixed on dev and scored on test, and any-layer restricted to the band."""

from collections.abc import Callable, Mapping

from wsbench.mc import seed_int
from wsbench.results import bootstrap_ci

DEV_SHARE = 0.30
MODEL_LAYERS = 64  # Qwen3.6-27B
SKIP_FIRST, SKIP_LAST = 0.10, 0.15  # of model depth; the band can't just be the next token


def is_dev(family: str, key: str) -> bool:
    """30% dev, hashed on ``family:key``; pass a pair id as ``key`` so twins share a split."""
    return seed_int(f"{family}:{key}") % 1000 < DEV_SHARE * 1000


def mid_band(layers: list[int], n_layers: int = MODEL_LAYERS) -> list[int]:
    """The judged layers inside [10%, 85%] of model depth (layers 7..54 of 64: 20..52 on GRID)."""
    lo, hi = SKIP_FIRST * n_layers, (1 - SKIP_LAST) * n_layers
    return [L for L in sorted(layers) if lo <= L <= hi]


def _rate(xs: list[bool]) -> dict:
    vals = [1.0 if x else 0.0 for x in xs]
    return {
        "rate": sum(vals) / len(vals) if vals else None,
        "ci95": list(bootstrap_ci(vals)) if vals else None,
        "n": len(vals),
    }


def layer_protocols(
    cell_pass: Mapping[str, Mapping[int, bool]],
    *,
    family: str,
    layers: list[int],
    group_of: Callable[[str], str] | None = None,
) -> dict:
    """``cell_pass`` = item -> {layer: judged pass}; layers absent from an item's map were not
    decided there. ``fixed_layer`` is the band layer with the best dev rate (ties -> earliest),
    reported on test items that have a verdict at that layer; ``mid_any_layer`` is the any-layer
    rule over the band, over items with at least one decided band layer. Never the headline."""
    band = mid_band(layers)
    group = group_of or (lambda i: i)
    dev = {i: m for i, m in cell_pass.items() if is_dev(family, group(i))}
    test = {i: m for i, m in cell_pass.items() if not is_dev(family, group(i))}
    dev_rates = {L: _rate([m[L] for m in dev.values() if L in m]) for L in band}
    scored = [L for L in band if dev_rates[L]["rate"] is not None]
    best = max(scored, key=lambda x: (dev_rates[x]["rate"], -x)) if scored else None
    mid = [
        any(m[L] for L in band if L in m) for m in cell_pass.values() if any(L in m for L in band)
    ]
    return {
        "band": band,
        "fixed_layer": {
            "layer": best,
            "dev": dev_rates[best] if best is not None else None,
            "test": _rate([m[best] for m in test.values() if best in m])
            if best is not None
            else None,
        },
        "mid_any_layer": _rate(mid),
        "mid_any_layer_test": _rate(
            [any(m[L] for L in band if L in m) for m in test.values() if any(L in m for L in band)]
        ),
        "n_dev_items": len(dev),
        "n_test_items": len(test),
    }
