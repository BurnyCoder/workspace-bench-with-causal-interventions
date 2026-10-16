"""Causal edits on the residual stream (docs/causal.md): project a lens direction out of chosen
blocks and positions, replay the same per-cell magnitudes along a control direction, and score
the answer teacher-forced. The tokenizer helpers need no torch; the rest imports it lazily."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from wsbench.mc import fold

OPEN_QUOTES = ('"', "'", "«", "“", "„", "\u2018")  # an answer follows these with no space


def _ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def answer_span(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], int, bool]:
    """``(ids, P, stripped)``: prompt + answer as one token sequence and ``P``, the index of the
    first answer token with visible content (a lone ``' '`` before digits stays in the prompt).
    A prompt ending in whitespace is stripped and joined with one space; after an opening quote
    the answer follows directly. Raises when the prompt is not a token prefix of the whole."""
    base = prompt.rstrip()
    sep = "" if base.endswith(OPEN_QUOTES) else " "
    head, full = _ids(tokenizer, base), _ids(tokenizer, base + sep + answer)
    if full[: len(head)] != head or len(full) == len(head):
        raise ValueError(f"prompt {base!r} is not a token prefix of its answer {answer!r}")
    p = len(head)
    while p < len(full) and not tokenizer.decode([full[p]]).strip():
        p += 1  # the joining space as its own token belongs to the prompt
    if p == len(full):
        raise ValueError(f"answer {answer!r} has no visible token")
    return full, p, base != prompt


def word_forms(word: str) -> list[str]:
    return list(dict.fromkeys([" " + word, word, " " + word.capitalize(), word.capitalize()]))


def bridge_token(tokenizer: Any, words: Sequence[str]) -> tuple[int | None, str]:
    """The first of each word's forms (``" w"``, ``"w"``, ``" W"``, ``"W"``) that is exactly one
    token with visible content: ``(token_id, form)``, or ``(None, reason)``. Never the first token
    of a longer word, and never a whitespace token (``" 6"`` is ``[' ', '6']``)."""
    for w in words:
        for form in word_forms(w):
            ids = _ids(tokenizer, form)
            if len(ids) == 1 and tokenizer.decode(ids).strip():
                return ids[0], form
    return None, f"no single-token form of {list(words)}"


def variant_ids(tokenizer: Any, word: str) -> list[int]:
    """Token ids of every single-token form of ``word`` (to exclude it from a KL)."""
    out = [_ids(tokenizer, f) for f in word_forms(word)]
    return sorted({ids[0] for ids in out if len(ids) == 1})


def overlaps(a: str, b: str) -> bool:
    """One word equals or begins the other after folding case, accents and quotes."""
    fa, fb = fold(a.strip()), fold(b.strip())
    return bool(fa and fb) and (fa.startswith(fb) or fb.startswith(fa))


# ---------------------------------------------------------------- torch


def unit(v: Any) -> Any:
    return v / v.norm().clamp_min(1e-9)


@contextmanager
def edit(
    blocks: Any, layers: Sequence[int], fn: Callable[[int, Any], Any], start: int, stop: int
) -> Iterator[None]:
    """While open, the output of each block in ``layers`` has rows ``start:stop`` of its first
    sequence replaced by ``fn(layer, rows)`` (fp32 in and out, cast back). Hooks are removed on
    exit, also on an exception."""

    def make(layer: int) -> Callable[..., Any]:
        def hook(_mod: Any, _inp: Any, out: Any) -> Any:
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[0, start:stop] = fn(layer, h[0, start:stop].float()).to(h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h

        return hook

    handles = [blocks[layer].register_forward_hook(make(layer)) for layer in layers]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def ablate(dirs: dict[int, Any], record: dict[int, Any]) -> Callable[[int, Any], Any]:
    """Project ``dirs[layer]`` out of every edited row, recording the signed coefficient removed
    per row in ``record[layer]`` (the magnitudes ``replay`` reuses)."""

    def fn(layer: int, x: Any) -> Any:
        u = dirs[layer]
        c = x @ u
        record[layer] = c
        return x - c[:, None] * u

    return fn


def replay(record: dict[int, Any], dirs: dict[int, Any]) -> Callable[[int, Any], Any]:
    """The control: subtract the ablation's recorded coefficient along another direction, so each
    (layer, row) moves by the same norm and sign as the ablation did and the pushes cannot add up
    across layers the way an independent ``|h·u|`` push would."""

    def fn(layer: int, x: Any) -> Any:
        return x - record[layer][:, None] * dirs[layer]

    return fn


def score(model: Any, ids: list[int], p: int) -> dict[str, Any]:
    """Teacher-forced ``logp`` of ``ids[p:]`` given ``ids[:p]``, whether the argmax reproduces
    every answer token (``correct``, i.e. greedy decoding would), and the next-token log-probs at
    ``p - 1`` (``next``). One pass, no cache: the linear-attention state is rebuilt each time."""
    import torch

    n_ans = len(ids) - p
    if n_ans < 1:
        raise ValueError("no answer tokens to score")
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(x, use_cache=False, logits_to_keep=n_ans + 1).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    ans = torch.tensor(ids[p:], device=lp.device)
    picked = lp[torch.arange(n_ans, device=lp.device), ans]
    return {
        "logp": picked.sum().item(),
        "correct": bool((lp.argmax(dim=-1) == ans).all()),
        "next": lp[0],
    }


def kl(lp_clean: Any, lp_edit: Any, exclude: Sequence[int] = ()) -> float:
    """KL(clean ‖ edited) of two next-token log-prob vectors, optionally over the vocabulary
    without ``exclude`` (both renormalised), so a bridge that is the output is not counted."""
    import torch

    if exclude:
        keep = torch.ones_like(lp_clean, dtype=torch.bool)
        keep[list(exclude)] = False
        lp_clean = torch.log_softmax(lp_clean[keep], dim=-1)
        lp_edit = torch.log_softmax(lp_edit[keep], dim=-1)
    return float((lp_clean.exp() * (lp_clean - lp_edit)).sum())
