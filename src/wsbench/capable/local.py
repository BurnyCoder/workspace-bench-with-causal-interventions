"""The capable answerer for a model on this machine instead of OpenRouter (``local=True``): the
same system prompt and question as one chat turn, thinking off, generated with transformers."""

from functools import cache
from typing import Any

from .questions import Question

MAX_NEW_TOKENS = 48  # every question asks for a word, a value or a short phrase


@cache
def _backend(model: str, device: str) -> Any:
    from wsbench.produce import Backend

    return Backend.load(model, device=device)  # once per process, not once per family


def answer_locally(
    family: str,
    questions: list[Question],
    draw_ids: list[int],
    *,
    model: str,
    system: str,
    temperature: float,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, dict]:
    """Replies keyed as ``run_calls`` keys them (``family|id|d<draw>`` -> ``{"answer": text}``):
    a negative draw is the greedy answer, the others are sampled at ``temperature`` in one
    ``generate`` call per item, seeded so a re-run asks the grader about the same answers."""
    import torch

    from wsbench.produce import render_text

    b = _backend(model, device)
    tok = b.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    greedy = [d for d in draw_ids if d < 0]
    sampled = [d for d in draw_ids if d >= 0]
    modes = [
        (greedy, {"do_sample": False}),
        (sampled, {"do_sample": True, "temperature": temperature}),
    ]
    torch.manual_seed(seed)
    out: dict[str, dict] = {}
    for q in questions:
        ids = render_text(q.ask, tok, chat=True, system=system).ids
        x = torch.tensor([ids], device=b.device)
        for draws, kw in modes:
            if not draws:
                continue
            with torch.no_grad():
                gen = b.model.generate(
                    x,
                    attention_mask=torch.ones_like(x),
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_return_sequences=len(draws),
                    pad_token_id=pad,
                    **kw,
                )
            texts = tok.batch_decode(gen[:, x.shape[1] :], skip_special_tokens=True)
            for d, text in zip(draws, texts, strict=True):
                out[f"{family}|{q.id}|d{d}"] = {"answer": text.strip()}
    return out
