# Causal test: does Qwen3.5-0.8B use the bridge the J-lens reads?

[small_model.md](small_model.md) showed the lenses reading bridge and English-pivot concepts on
some items. Reading a concept is not the same as the model using it. This page runs the paper's
targeted ablation: remove the bridge concept's J-lens direction from the residual stream and see
whether the answer breaks more than it does under an equal edit along another concept.

Setup:
- Model: Qwen/Qwen3.5-0.8B, 24 blocks, with neuronpedia's J-lens.
- Items: 290 prompts. 66 of them are answered correctly by the clean model and enter the primary
  test.
- Cost: no API calls; about 7 minutes on an 8 GB laptop GPU. Run 2026-10-16.

**Result.** Ablating the bridge's J-lens direction lowers log p(answer) by 2.78 nats and leaves
11 of 66 answers correct. The same-magnitude edit along a same-category concept costs 0.72 nats
(37/66 correct), and a random direction costs 0.12 (56/66).
- The primary contrast is −2.06 [−2.41, −1.72] nats, negative in all three prompt sets.
- J-lens ablation moves the output distribution 6.2× more than logit-lens ablation; the paper
  reports about 2×.
- The effect is larger where the lens reads the bridge.

## Method

**What the paper does.** From Gurnee et al. (2026), Methods:
- J-lens vectors: "We refer to the rows of W_U J_ℓ as the Jacobian lens (J-lens) vectors".
- Ablation: steering is h ← h + α v_t, and "with negative α, or by projecting out the
  component of h along v_t entirely, this becomes an ablation".
- Control: "a perturbation of equal magnitude, inhibiting the J-lens vector of some other
  concept of the same category".
- Lens comparison: J-lens ablation gives "roughly twice the output KL" of logit-lens ablation.

**What this run does** (`scripts/causal_check.py`, built on `wsbench.produce.intervene`):
- **Directions.** The bridge's J-lens vector is u_ℓ = unit(W_U[t] · J_ℓ). This is the readout's
  own orientation: `JLens.scores` is h·v_t/‖v_t‖.
  - Its token t is the first single-token visible form of the bridge word, trying " w", "w",
    " W", "W" in that order. Items with no such form are skipped.
  - The logit-lens direction is unit(g ⊙ W_U[t]), with g the gain of the model's final RMSNorm.
- **Where.**
  - Layers: blocks 9–19 of 24, the paper's L38–83 band rescaled.
  - Positions: every prompt position except 0. The tokenizer adds no BOS, and position 0 acts as
    an attention sink. Positions from the answer onwards are never edited.
- **The ablation.** At each band layer and position, h ← h − (h·u)u. The signed coefficient
  removed in each cell is recorded.
- **The controls (replayed).** Each control edits every cell by the same recorded coefficient,
  h ← h − c·w, along another direction w.
  - So every cell moves by the same norm and sign as in the ablation.
  - This is a deviation from the naive reading `h −= |h·u|·w`. That form pushes the full loading
    again at all 11 layers, because the bridge is never removed, and in a pilot it hurt *more*
    than the ablation itself.
  - The controls, by direction w:

| control | w |
|---|---|
| samecat (the primary control) | the J-lens vector of a same-category concept |
| samecat_orth | that vector made orthogonal to the bridge's |
| randvocab | the mean J-lens vector of 8 random words, which matches the shared component of J-lens vectors |
| isotropic | a random unit vector |

- **Same-category concepts.**
  - probe_swap: the paper's own `swap_to` (Brazil → Mexico).
  - multihop bank: the latent of the next item in the same template, e.g. `holiday-*-monthnum`.
  - multilingual bank: the next item in the same language. For banks this is often not truly the
    same category; see Caveats.
- **Metrics.**
  - Δ log p(answer): teacher-forced over every answer token.
  - Still correct: the argmax reproduces every answer token.
  - KL(clean ‖ edited) at the answer position, also with the bridge's own tokens excluded.
  - The bridge's J-lens rank at the last visible prompt token per band layer ("loading").
- **Trials and the primary test.**
  - The primary analysis counts only trials the clean model answers correctly, as the paper and
    J++ do.
  - A probe-swap prompt that repeats a multihop bank fact is counted once.
  - Pre-registered contrast: Δlog p(ablate_j) − Δlog p(samecat), paired, bootstrap CI.
- **Self-checks.**
  - A clean pass after all edits must reproduce the first clean pass, or the run aborts. This
    catches leaked hooks.
  - Positive control: under the same-category replay, the probe-swap counterfactual answer's first
    token loses log-probability in 16 of 17 trials, so the control direction is live.
- **Prompt rule.** A prompt ending in whitespace is stripped and the answer joined with one
  space. After an opening quote the answer follows directly. Digit answers keep their space
  token in the prompt (`" 12"` is `[' ', '1', '2']`).

## Running it

```bash
for s in probe_swap multihop multilingual; do
  uv run python scripts/causal_check.py ablate --set $s --out outputs/causal   # GPU, ~2 s per item
done
uv run python scripts/causal_check.py summary --out outputs/causal              # writes summary.md
```

Options are `--band 9-19`, `--limit N` and `--model`. The script records `probe-swap.json` (from
pinned commit `581d398` of `anthropics/jacobian-lens`) and its sha256 in `--out`. It writes one
JSONL row per trial to `<out>/<set>/trials.jsonl` and a timestamped log to `<out>/logs/`.

## Results

| set | items | excluded | scored | clean-correct |
|---|---|---|---|---|
| probe_swap | 90 | 2 | 88 | 20 |
| multihop | 100 | 14 | 86 | 28 |
| multilingual | 100 | 3 | 97 | 21 |

Exclusions:
- 9 multihop bridges have no single-token form (12, 11, octagon, polar bear, Sherlock Holmes, San
  Francisco, Los Angeles, New Zealand, New York).
- 10 bridges equal or begin their answer once case and accents are folded, so ablating one
  would ablate the other. Examples: Russia → Russian, Paris → París, May → mayo,
  February → Februar, and the `firstletter-*` items (February → F). All are listed in
  `trials.jsonl`.
- Of the 20 clean-correct probe-swap items, 3 repeat a multihop bank fact, which leaves 66 pooled
  trials.

**Primary: Δlog p(answer), ablate_j − samecat** (clean-correct; mean and 95% bootstrap CI)

| scope | ablate_j − samecat |
|---|---|
| pooled | −2.06 [−2.41, −1.72] (n=66) |
| probe_swap | −2.04 [−2.53, −1.54] (n=17) |
| multihop | −1.76 [−2.20, −1.25] (n=28) |
| multilingual | −2.48 [−3.35, −1.64] (n=21) |

**Every condition** (clean-correct, pooled, n=66)

| condition | Δlog p(answer) | still correct | KL at the answer |
|---|---|---|---|
| ablate_j | −2.78 [−3.15, −2.40] | 11/66 | 1.01 |
| samecat | −0.72 [−1.01, −0.48] | 37/66 | 0.40 |
| samecat_orth | −0.20 [−0.39, −0.02] | 47/66 | 0.27 |
| randvocab | −0.67 [−0.85, −0.49] | 38/66 | 0.45 |
| isotropic | −0.12 [−0.19, −0.05] | 56/66 | 0.04 |
| ablate_logit | −0.60 [−0.78, −0.43] | 48/66 | 0.16 |
| ablate_j_final (final position only) | −1.50 [−1.81, −1.20] | 20/66 | 0.55 |

**J-lens versus logit-lens.** The KL ratio is 6.20 [4.73, 8.89]. Excluding the bridge's own
tokens it is 6.59 [4.86, 9.87]. The J-lens ablation also removes more norm (mean 1.24 against
0.87).

**Does the effect follow the lens reading?**

| bridge in the J-lens top 10 at some band layer | ablate_j − samecat |
|---|---|
| yes | −2.66 [−3.37, −2.01] (n=26) |
| no | −1.67 [−2.03, −1.32] (n=40) |

- Spearman correlation between bridge loading (best rank across the band) and the ablation
  excess: +0.43.
- The extremes fit:

| item | bridge | best rank | Δlog p, ablate_j | Δlog p, samecat |
|---|---|---|---|---|
| `b3-ml-es-opposite-day` | " night" | 2 | −5.75 | +0.56 |
| `pt-size-big-small` | " small" | 2 | −5.06 | +1.11 |
| `month-7-namedafter` | " July" | 1857 | 0.00 | +0.01 |
| `fr-place-up-down` | " down" | 251 | 0.00 | −0.06 |

**Not the output in disguise.** Where the bridge is itself in the clean model's top 10 next
tokens, the excess is −2.14 [−2.87, −1.29] (n=8). Elsewhere it is −2.05 [−2.48, −1.64] (n=58).

**Secondary results:**
- Over every scored trial, clean-correct or not, the excess is −2.15 [−2.39, −1.92] (n=249).
- Against the 30-item run's judged J-lens verdict (L14 / L22), bank items the judge passed give
  −1.73 [−2.78, −0.90] (n=15), and those it failed give −1.48 [−1.96, −1.02] (n=43). The judged
  verdict at two layers barely separates them; the lens's own rank across the band does.

## What it shows

- **The bridge is used, not just readable.** On the 0.8B, removing the direction the J-lens names
  breaks the answer about four times as much as an equal edit along a same-category concept.
  It breaks it more than twenty times as much as a random direction. The result holds in the
  paper's own prompts and in both bench banks.
- **The J-lens direction is the causally relevant one.** The logit-lens direction for the same
  token barely matters (−0.60, 48/66 still correct), and the orthogonalised same-category control
  barely matters either (−0.20). So the J-lens isolates a direction the model uses, which the
  logit lens does not, consistent with the paper's lens comparison.
- **Lens reading predicts causal use.** Items whose bridge the J-lens ranks highly lose more.
  Items it never surfaces (July, at rank 1857) lose nothing. That is the link the benchmark
  assumes for its J-lens arm.
- **The edit works across positions.** Editing only the final prompt position recovers about half
  the effect (−1.50 against −2.78). The bridge is also carried at earlier positions.
- **Scope.** This validates the J-lens and its readings on this model. It says nothing about the
  O-lens, whose prose [small_model.md](small_model.md) found partly unanchored. It is also not
  the paper's swap (sufficiency) test, which was out of scope here.

## Caveats

- **Bank controls are template-matched, not category-matched.** Pairs like night → four or
  cold → London are not the same category: the mean cosine between bridge and control vectors is
  0.41 pooled. The probe-swap set's controls are curated same-category concepts (Brazil → Mexico),
  and its excess (−2.04) matches the banks'.
- **The raw same-category control also removes part of the bridge.** It shares about cos × c
  with the bridge. That makes it conservative, and the orthogonalised control is weaker still.
- **One band.** Blocks 9–19, rescaled from the paper and not re-derived for this model. The
  readouts in small_model.md peak at L16–L20.
- **Small n.** There are 66 clean-correct trials, and a 0.8B model answers few of these items.
  The paper's prompts were written for much larger models.
- **The KL ratio is not like for like.** Ours is 6.2×, the paper's about 2×. The ablations remove
  different norms, and the paper's number is over its own models and prompts.

## Sources

- Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in Language
  Models*, Transformer Circuits:
  https://transformer-circuits.pub/2026/workspace/index.html. Methods, "Technical details"
  (steering, ablation, the same-category control and the lens comparison).
- Code and prompts: https://github.com/anthropics/jacobian-lens (Apache-2.0), including
  `data/experiments/probe-swap.json` at commit `581d398`.
- J++ Lens (Ayonrinde and Lindsey), whose causal-eval runner informed the trial rules (clean-correct
  only, position 0 skipped, unit-normalised directions):
  https://github.com/safety-research/jpp_lens and https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr.
- Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*,
  causal basis extraction: https://arxiv.org/abs/2303.08112.
- Lenses: https://huggingface.co/neuronpedia/jacobian-lens (`qwen3.5-0.8b/jlens/`, Apache-2.0).
