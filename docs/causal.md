# Causal tests: does Qwen3.5-0.8B use the bridge the J-lens reads?

[small_model.md](small_model.md) showed the lenses reading bridge and English-pivot concepts on
some items. Reading a concept is not the same as the model using it. This page runs the J-lens
paper's two causal tests on the bridge concept:
- **[Ablation](#ablation) (necessity).** Remove the bridge's J-lens direction from the residual
  stream. Does the answer break more than under an equal edit along another concept?
- **[Swap](#swap) (sufficiency).** Clamp the bridge's J-lens coordinate to another concept's. Does
  the model give *that* concept's answer? For example, Brazil→Mexico in "the language of the
  country where the Amazon ends" should give Spanish.

Setup:
- Model: Qwen/Qwen3.5-0.8B, 24 blocks, with neuronpedia's J-lens.
- Items: 290 prompts. 66 of them are answered correctly by the clean model and enter the primary
  test.
- Cost: no API calls. The ablation took about 7 minutes on an 8 GB laptop GPU (2026-10-16) and
  the swap about 18 minutes (2026-10-17).

**Ablation result.** Ablating the bridge's J-lens direction lowers log p(answer) by 2.78 nats and leaves
11 of 66 answers correct. The same-magnitude edit along a same-category concept costs 0.72 nats
(37/66 correct), and a random direction costs 0.12 (56/66).
- The primary contrast is −2.06 [−2.41, −1.72] nats, negative in all three prompt sets.
- J-lens ablation moves the output distribution 6.2× more than logit-lens ablation; the paper
  reports about 2×.
- The effect is larger where the lens reads the bridge.

**Swap result.** The J-lens clamp swap makes the counterfactual answer the top-1 next token in
13 of 42 primary trials (31% [19%, 46%]).
- It is 41% on the paper's prompts and 43% on the multihop bank. For reference, J++ gets 49%
  with the J-lens on Qwen3.6-27B, and the paper reports 60% with the J-lens on Sonnet 4.5.
- The same swap along logit-lens directions succeeds 4 times, and a random-noun null and a zero
  clamp once each. Paired, the J-lens beats each (McNemar p = 0.012, 0.002, 0.002).
- On the multilingual bank it raises the counterfactual's rank about 19-fold but never to the
  top (0/11), because the prompt itself pins the answer.

## Ablation

### Method

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

### Running it

```bash
for s in probe_swap multihop multilingual; do
  uv run python scripts/causal_check.py ablate --set $s --out outputs/causal   # GPU, ~2 s per item
done
uv run python scripts/causal_check.py summary --out outputs/causal              # writes summary.md
```

Options are `--band 9-19`, `--limit N` and `--model`. The script records `probe-swap.json` (from
pinned commit `581d398` of `anthropics/jacobian-lens`) and its sha256 in `--out`. It writes one
JSONL row per trial to `<out>/<set>/trials.jsonl` and a timestamped log to `<out>/logs/`.

### Results

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

### What it shows

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
  O-lens, whose prose [small_model.md](small_model.md) found partly unanchored. The swap below is
  the sufficiency side.

### Caveats

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

## Swap

### Method

**What the paper does.** From Gurnee et al. (2026), Methods, "patching in lens coordinates":
- "we form V = [v_s v_t], read the lens coordinates c = V†h (where V† is the pseudoinverse of V),
  and set h_patched = h + V(σ(c) − c), where σ swaps the two entries of c (optionally scaled by a
  factor α). The component of h orthogonal to span{v_s, v_t} is unchanged."
- Most experiments clamp the swap. The coordinates are pinned to the clean pass's swapped values
  "at every band layer at the specified positions" (jacobian-lens README).
- Success means the counterfactual answer becomes top-1.

**What this run does: J++'s implementation, ported.**
- `clamp`, `swapped_targets`, the near-parallel guard and the null nouns in
  `wsbench.produce.intervene` are ported from J++ (`safety-research/jpp_lens`, Apache-2.0; see
  NOTICE.md). `scripts/causal_check.py swap` follows J++'s causal runner:
  - **Directions.** Unit J-lens vectors for the bridge and the swap target, per band layer.
  - **The swap.** One two-atom clamp per band layer (blocks 9–19), at every prompt position but 0.
    The targets are the clean pass's coordinates, exchanged and multiplied by the scale. A layer
    is skipped if |cos(u_s, u_t)| > 0.999; no layer reached that here.
  - **Scales.** J++'s probe-swap grid: 0.5, 1, 2 and 4. Scale 1 is the pre-registered one.
  - **Success.** The swap answer's best single-token surface form (" w", "w", " W", "W") is the
    top-1 next token, as in J++. Top-5, J++'s word rank (formatting tokens removed) and whether
    greedy decoding produces the whole swap answer are reported too.
  - **Trials.** Only prompts the clean model answers correctly, whose swap answer is not already
    its top-1. A probe-swap fact repeated in the multihop bank is counted once.
- **Controls**, each the same clamp swap at the same scales:

| control | what it swaps | expected |
|---|---|---|
| `logit_swap` | the same pair along logit-lens directions | the paper's lens comparison |
| `null_swap` | the bridge with an unrelated noun (J++'s pool: table, window, pencil, …; seeded per item) | ~0 success |
| `zero` | both J-lens coordinates clamped to 0 | separates installing the new concept from removing the old |
| `answer_swap` | the answer with the swap answer directly | J++'s ceiling: if it fails, the main result is uninterpretable |

- **Swap pairs.**
  - probe_swap: the paper's own `swap_to` and `swap_answer`.
  - multihop bank: the next item of the same template (`holiday-*-monthnum`, `b3-*-capital`, …);
    its bridge and answer become the swap target and swap answer.
  - multilingual bank: the next item of the same language and category, so the swap answer is
    in the prompt's language.
  - Pairs whose bridges, answers or bridge and answer overlap are skipped.
- **Deviations from J++.**
  - The logit-lens direction includes the final norm's gain, g ⊙ W_U[t], as in the ablation;
    J++ uses raw W_U[t].
  - Trailing-space prompts are stripped, as in the ablation; J++ keeps them as written.
  - Choices motivated by a 25-trial pilot:
    - At scale 4 the top-1 becomes the injected word itself (" Japan", " Easter").
    - The unclamped swap is a reflection that later layers undo, scoring 0/24.

### Running it

```bash
for s in probe_swap multihop multilingual; do
  uv run python scripts/causal_check.py swap --set $s --out outputs/causal   # GPU, ~5 s per item
done
uv run python scripts/causal_check.py summary --out outputs/causal            # adds the swap tables
```

Rows go to `<out>/<set>/swaps.jsonl`. Each holds every condition and scale with the swap answer's
rank, the original answer's rank, greedy success, Δlog p of both answers, KL and `leak`.

### Results

| set | items | excluded | scored | clean-correct | primary |
|---|---|---|---|---|---|
| probe_swap | 90 | 2 | 88 | 20 | 18 |
| multihop | 100 | 51 | 49 | 16 | 14 |
| multilingual | 100 | 51 | 49 | 11 | 11 |

Exclusions:
- 85 bank items have no partner in their template or language-and-category group.
- 12 bridges have no single-token form.
- 7 bridges overlap their answer.

There are 42 primary trials pooled; one probe-swap fact repeats a multihop item.

**Primary: the J-lens swap at scale 1 makes the swap answer top-1** (Wilson 95% CI)

| scope | success |
|---|---|
| pooled | 13/42 = 31% [19%, 46%] |
| probe_swap | 7/17 = 41% [22%, 64%] |
| multihop | 6/14 = 43% [21%, 67%] |
| multilingual | 0/11 = 0% [0%, 26%] |

References: the paper's probe-swap J-lens 60% (Sonnet 4.5); its two-hop swap 54% (Haiku 4.5) and
70% (Sonnet 4.5); J++ on Qwen3.6-27B, J-lens 49%.

**Paired against the J-lens swap at scale 1** (McNemar exact)

| control | J only | control only | p |
|---|---|---|---|
| logit_swap | 10 | 1 | 0.012 |
| null_swap | 13 | 1 | 0.002 |
| zero | 13 | 1 | 0.002 |

**Every condition by scale** (42 primary trials; 41 for answer_swap, whose answers must differ in
their first token)

| condition | scale | top-1 | greedy swap | original kept | Δlog p(swap) | Δlog p(original) | top-1 is the injected word |
|---|---|---|---|---|---|---|---|
| j_swap | 0.5 | 5 | 5 | 7 | +1.60 | −2.33 | 0 |
| **j_swap** | **1** | **13** | **13** | 9 | +2.93 | −2.55 | 6 |
| j_swap | 2 | 7 | 8 | 5 | +3.20 | −3.65 | 22 |
| j_swap | 4 | 0 | 0 | 1 | +0.51 | −6.99 | 36 |
| logit_swap | 1 | 4 | 5 | 20 | +1.59 | −0.84 | 3 |
| logit_swap | 4 | 4 | 4 | 8 | +1.50 | −4.57 | 19 |
| null_swap | 1 | 1 | 1 | 3 | −0.69 | −4.51 | 0 |
| zero | — | 1 | 0 | 5 | −1.13 | −3.04 | 0 |
| answer_swap (ceiling) | 1 | 17 | 17 | 11 | +3.78 | −3.05 | 1 |
| answer_swap (ceiling) | 2 | 26 | 22 | 11 | +5.18 | −3.51 | 0 |
| answer_swap (ceiling) | 4 | 32 | 24 | 8 | +3.92 | −6.37 | 0 |

The full grid (scales 0.5–4 for every control, top-5 and word rank) is in `summary.md`.

**Multilingual: moved, but not to the top.**
- The counterfactual answer starts far down: median clean rank 12,969. The prompt names the word
  whose opposite is wanted ('Lo contrario de "caliente" es "'), so swapping cold→night asks for
  "noche" against the visible "caliente".
- The J-lens swap raises the counterfactual's rank by a median ×19.3 at scale 1, against ×10.5
  for the logit lens and ×0.7 for the null and the zero clamp. On that item, "noche" goes from
  rank 21,585 to 729.
- Even the answer-swap ceiling needs scale 2 there (0/11 at 1, 5/11 at 2), against 9/13 multihop
  and 9/18 probe-swap at scale 1.
- On probe-swap and multihop, the median rank gains at scale 1 are ×3.5 and ×12.3 (J-lens),
  ×1.7 and ×3.6 (logit lens), and ×0.4 and ×0.5 (null).

**Other results:**
- **Lens loading.** Unlike the ablation, swap success does not follow it. It is 5/16 when the
  bridge is in the J-lens top 10 at some band layer, and 8/26 when it is not (31% each).
- **Whether the model knows the partner fact** (banks): 3/10 when it answers the partner item
  itself, 3/15 when it doesn't.
- **Every scored trial with a single-token swap answer,** clean-correct or not: 17/162 (10%).

### What it shows

- **The bridge is sufficient as well as necessary, at the J-lens coordinate.** Clamping the
  bridge's J-lens coordinate to another concept's makes the model give that concept's answer in
  about a third of the prompts it answers. That is about two in five on the two-hop sets. The
  matched controls rarely succeed:
  - the same swap along logit-lens directions: 4 of 42;
  - a swap toward an unrelated noun: 1;
  - clamping both coordinates to zero: 1.
  The zero control matters: removing the bridge alone does not produce the counterfactual. The
  swapped-in concept is what does.
- **Close to the larger models.** 41–43% on the two-hop sets, against J++'s 49% for the J-lens on
  Qwen3.6-27B and the paper's 54–70% on Claude models. The intervals are wide at this n.
- **The strength window is narrow.** Scale 1 is best. At scale 2 the injected word leaks into the
  output on half the prompts, and at scale 4 it is the output on 36 of 42. The paper and J++
  found larger scales helped larger models; this 0.8B tolerates less.
- **Multilingual is a harder target, not a null.** The swap moves the counterfactual far toward
  the top (×19), more than the logit lens and unlike the controls. But the prompt names the
  antonym, which the model also reads, and even the direct answer swap needs double strength.

### Caveats

- **Small n.** There are 42 primary trials, about ±14 points at the pooled rate. Only 25 of them
  are bank trials, 11 of those multilingual.
- **Bank pairs are mechanical.** The next item of the same template or language and category.
  The model does not know about 60% of the partner facts. The paper's probe-swap set has curated
  pairs, and its rate (41%) is in line with the banks'.
- **One band and four scales,** from the paper (rescaled) and J++, not tuned for this model.
- **Success is J++'s single-token criterion.** Answers with no single-token form are left out of
  the top-1 count. Greedy decoding, reported alongside, agrees (13/42 at scale 1).

## Sources

- Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in Language
  Models*, Transformer Circuits:
  https://transformer-circuits.pub/2026/workspace/index.html. Methods, "Technical details"
  (steering, ablation, the same-category control and the lens comparison).
- Code and prompts: https://github.com/anthropics/jacobian-lens (Apache-2.0), including
  `data/experiments/probe-swap.json` at commit `581d398`.
- J++ Lens (Ayonrinde and Lindsey): https://github.com/safety-research/jpp_lens (Apache-2.0) and
  https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr.
  - The swap's `Clamp`, swapped targets, near-parallel guard and null nouns are ported from it at
    commit `93f02cd` (see NOTICE.md).
  - Its causal runner set the trial rules for both tests: clean-correct trials only, position 0
    skipped, unit-normalised directions, and success as rank 1 over the answer's single-token
    surface forms.
- Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*,
  causal basis extraction: https://arxiv.org/abs/2303.08112.
- Lenses: https://huggingface.co/neuronpedia/jacobian-lens (`qwen3.5-0.8b/jlens/`, Apache-2.0).
