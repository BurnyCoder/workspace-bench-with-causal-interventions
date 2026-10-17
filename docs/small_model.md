# A <1B model on a slice of the bench: Qwen3.5-0.8B

Does the bench still give sensible numbers on a model under 1B parameters? This page runs one
slice end to end on **Qwen/Qwen3.5-0.8B**: the gate, three lens arms, the judge, and a
derangement null. It also reads the verdicts by eye.

There were two runs, both on 2026-10-16:
- a first look at 3 items per family, about 60 judge calls;
- a 10× rerun at 30 items per family, about 870 calls and $0.55.

The numbers below are from the 30-item run. "What the 3-item run got wrong" lists what changed.

## Why this model

- **Same family as the bank model.** Both configs on the Hub give model type `qwen3_5`
  (gated-delta-net linear attention, with full attention every fourth block) and a 248,320-token
  vocabulary, the same as the Qwen3.6-27B every bank was gated on.
- **Same renders.** The two chat templates differ only in the default thinking mode and in
  tool-call formatting. With `enable_thinking=False`, which every render passes, both emit the
  same empty `<think>` block, so the plan's renders and read rules carry over unchanged.
- **Size.** It has 24 decoder blocks of width 1024, against the 27B's 64 of width 5120.
  `AutoModelForCausalLM` loads it as `Qwen3_5ForCausalLM`, with `model.layers` and `model.norm`,
  which `Backend` already finds.
- **Lenses.** neuronpedia publishes two for it in the model repo
  [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens), so the same
  three arm types as on the 27B can run:
  - a J-lens, `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`: Jacobians
    for blocks 0–22, fit on 233 wikitext prompts with the
    [anthropics/jlens](https://github.com/anthropics/jlens) code;
  - an oracle LoRA (O-lens), `qwen3.5-0.8b/oracle`, checkpoint `olens-qwen3.5-0.8b-rl-mb4-s220`.
    Its `run.json` gives read layers 8–22 step 2, the `unit` transform, alpha 60 and the
    `concepts_raw` prompt. Its `CREDITS.md` says that read contract is the same as the 27B
    lens's, so the run uses `OLens`'s default prompt with `alpha=60`. The method is Appendix
    A.9.2 of [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html).

## What was run

- **Families.** multihop, multilingual and typo: single-token families that AGENTS.md lists as
  usable on a new model once it passes the gate.
- **Items.** `limit=30`, the first 30 bank items of each family. Bank order, plan order and gate
  order are the same.
- **Layers.** Readouts at layers 8, 10, …, 22. That is the oracle's trained grid, and roughly
  the 27B's 20..60 scaled by 24/64. Judging used layers 14 and 22 only, chosen before the first
  run as "mid" and "late" and kept for comparability.
- **Judge.** The repo default, `google/gemini-3.8-flash`, under its pinned prompt
  (`bank-2026-10-10`).
- **Gate.** Two greedy answers per item:
  - `wsbench capable local=True draws=0`: the bank's question as a chat turn, thinking off;
  - the raw continuation: the plan's plain render continued greedily for 8 tokens.
  The repo grader grades both. An item counts as "the model does it" when either is right.
  Typo has only the chat turn, since a plain continuation never corrects a typo.
- **Null.** Within each family, item *k*'s O-lens readouts are relabelled as item *k*+1, then
  judged the same way. No pair shares a target. Neighbouring bank items usually share a template
  (atomic number 26 beside atomic number 79), so this null is topic-matched.

```bash
set -a; source .env; set +a
M=Qwen/Qwen3.5-0.8B; L=8,10,12,14,16,18,20,22; O=outputs/small_model_n30; N=30
JL=filename=qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt
OL=lora=neuronpedia/jacobian-lens:qwen3.5-0.8b/oracle,repo_type=model,alpha=60
uv run wsbench capable model=$M local=True families=multihop,multilingual,typo limit=$N draws=0 out=$O/capable
uv run python scripts/small_model_check.py raw --out $O --limit $N
for f in multihop multilingual typo; do
  uv run wsbench produce family=$f model=$M method=logit_lens layers=$L limit=$N out=$O/readouts/logit_lens/$f.jsonl
  uv run wsbench produce family=$f model=$M method=jlens layers=$L limit=$N out=$O/readouts/jlens/$f.jsonl opts=$JL
  uv run wsbench produce family=$f model=$M method=olens layers=$L limit=$N out=$O/readouts/olens/$f.jsonl opts=$OL
done
uv run python scripts/small_model_check.py null --out $O --arm olens
for a in logit_lens jlens olens olens_null; do
  uv run wsbench run families=multihop,multilingual,typo readouts_root=$O/readouts/$a out=$O/judged/$a limit=$N layers=14,22
done
uv run python scripts/small_model_check.py summary --out $O
```

Cost and time on an 8 GB RTX 5070 laptop GPU:
- **Calls.** 870 Gemini Flash calls: 720 judge cells, 90 gate grades and 60 raw-continuation
  grades. About $0.55.
- **Vector lenses.** Seconds per family.
- **O-lens.** 1 h 36 min for the 90 items.
  - Generation is bound by per-token overhead, not batch size. 256 tokens with the LoRA take
    33 s alone and 34 s as a batch of 8.
  - So `OLens.read_many` generates an item's cells together, one batch per carrier length.
    Unbatched, it was about 37 s per cell.
  - Installing `flash-linear-attention`, which transformers' fallback warning suggests, did not
    change the time.

## Results (30 items per family)

### The gate (greedy)

| family | chat turn | raw continuation | either |
|---|---|---|---|
| multihop | 6/30 | 6/30 | 11/30 |
| multilingual | 6/30 | 7/30 | 10/30 |
| typo | 19/30 | — | 19/30 |

The bank recorded 1.0 for Qwen3.6-27B on every one of these items. The two formats rarely agree.
On multihop only 1 item is right in both formats, 5 only in the chat turn and 5 only in the raw
continuation. Raw multihop continuations start after the prompt's trailing space token, which is
an unnatural place to continue. Some answers by format:
- emerald's month, chat `October`, raw `5` (right);
- Christmas's month, chat `1`, raw `12` (right);
- atomic number 26, chat `Fe` (right), raw `60`.

### Judged pass rate (an item passes on either judged layer; 95% bootstrap CI)

| arm | multihop | multilingual | typo |
|---|---|---|---|
| logit lens | 4/30 [0.03, 0.27] | 13/30 [0.23, 0.60] | 3/30 [0.00, 0.20] |
| J-lens | 4/30 [0.03, 0.27] | 11/30 [0.20, 0.53] | 15/30 [0.33, 0.67] |
| O-lens | 19/30 [0.47, 0.80] | 23/30 [0.60, 0.90] | 27/30 [0.77, 1.00] |
| O-lens, derangement null | 2/30 [0.00, 0.17] | 2/30 [0.00, 0.17] | 2/30 [0.00, 0.17] |
| *Qwen3.6-27B O-lens, 100 items, 11 layers* ([plan0012_results.md](plan0012_results.md)) | *0.99* | *1.00* | *1.00* |

All 720 judged cells came back: none missing, unjudged or empty.

### Pass rate by whether the model does the task (either format)

| arm | family | model does it | model does not |
|---|---|---|---|
| logit lens | multihop | 2/11 | 2/19 |
| logit lens | multilingual | 7/10 | 6/20 |
| logit lens | typo | 2/19 | 1/11 |
| J-lens | multihop | 2/11 | 2/19 |
| J-lens | multilingual | 6/10 | 5/20 |
| J-lens | typo | 11/19 | 4/11 |
| O-lens | multihop | 7/11 | 12/19 |
| O-lens | multilingual | 8/10 | 15/20 |
| O-lens | typo | 19/19 | 8/11 |

### Depth

Judged expressed rate per layer, over all 90 items:

| arm | L14 | L22 |
|---|---|---|
| logit lens | 2/90 | 19/90 |
| J-lens | 16/90 | 16/90 |
| O-lens | 53/90 | 63/90 |

The table below needs no judge: it counts the items whose English target appears verbatim (a
whole token of a bag, or a whole word of prose), at each of the 8 produced layers. Translations
do not count.

| arm (90 items) | L8 | L10 | L12 | L14 | L16 | L18 | L20 | L22 |
|---|---|---|---|---|---|---|---|---|
| logit lens | 1 | 0 | 1 | 0 | 4 | 12 | 19 | 12 |
| J-lens | 18 | 19 | 19 | 18 | 26 | 37 | 33 | 25 |
| O-lens | 36 | 37 | 43 | 43 | 51 | 50 | 56 | 53 |

Readout excerpts (top tokens, best first):
- **Logit lens, atomic number 26.**
  - L16 ` metals`, `otope`
  - L18 ` atoms`, ` elemental`
  - L20 ` chromium`, ` iron`, ` titanium`
  - L22 ` iron` ranked first
- **Logit lens, opposite of "nah".** L16–L18 Chinese 距离/远 (distance/far) and ` distant`; L20 ` far`;
  L22 ` nah`, ` fern`, ` entfernt`. This is the English-pivot-then-output-language pattern of
  [Wendler et al. (2024)](https://arxiv.org/abs/2402.10588).
- **J-lens, "…last day of Febuary".** ` February` ranked first at L14 and L16. By L22 the bag is
  `,` `.` ` and`, the next-token output.

## Does it make sense?

**The pipeline: yes.** It ports cleanly:
- the O-lens LoRA loads onto `Qwen3_5ForCausalLM` with no PEFT warning;
- the J-lens Jacobians match the residual width;
- all 720 judged cells came back.

**The vector lenses: yes, as far as a weak model allows.**
- *Depth.* The logit lens finds targets only late (peak L20). The J-lens finds them earlier and
  more often at every layer; on typo at L14 it scores 15 items against the logit lens's 2. Both lenses
  peak at L18–L20. Block 22 is the second-to-last block and mostly carries the next token, which
  is why the judged J-lens rate does not rise from L14 to L22.
- *Following what the model computes.* Hits concentrate on items the model does: multilingual
  7/10 against 6/20 (logit lens) and 6/10 against 5/20 (J-lens), and J-lens typo 11/19 against
  4/11. But all 13 logit-lens multilingual passes come at L22, and 11 of them quote the
  foreign-language answer (`unten`, `klein`, `lejos`, `loin`, `petit`, `rechts`, …), not the
  English concept. That part of the split shows what the model is about to say, not a hidden
  latent.
- *Multihop.* Both lenses barely see a bridge (4/30 each). Only 11 of 30 multihop items are done
  at all.

**The O-lens: item-specific, but not anchored to what the model computes.**
- *It beats the null.* Real 0.63–0.90 against 0.07 in every family. The prose names this item's
  target, not targets in general.
- *It does not follow the gate.* It passes 12 of the 19 multihop items and 15 of the 20
  multilingual items the model fails in both formats. It also names the exact target at L8 on
  36 of 90 items, against 1 for the logit lens.
- *Why, from its passes on failed items:*
  - The judge credits enumerations its rule should reject: "a) East, b) West, c) North, or d)
    South?" for south, "January, February, and" for February, "A. Aluminum B. Oxygen C." for
    oxygen.
  - The oracle is this same 0.8B with a LoRA. It reconstructs the prompt's topic and can name the
    bridge from its own knowledge, e.g. "1. Carbon" for the element in organic compounds, where
    the model itself answered `H`.
- *How to read it.* On a model this small, the O-lens number says what the oracle can say about
  the prompt as much as what the model computed. Read it beside the null and the gate split,
  never alone.

**Against the 27B: consistent.** The O-lens gives 0.63, 0.77 and 0.90 against the 27B's 0.99,
1.00 and 1.00, with the same family ordering.

## Follow-up: is the read bridge used?

[causal.md](causal.md) runs the J-lens paper's two causal tests on these banks and the paper's
probe-swap prompts:
- **Ablation.** Removing the bridge's J-lens direction breaks the answer about four times as
  much as an equal edit along a same-category concept, and more where the J-lens reads the
  bridge.
- **Swap.** Clamping the bridge's J-lens coordinate to another concept's gives that concept's
  answer in 31% of prompts, against 10% for the logit lens and 2% for a random-noun null.

The J-lens readings on this model track a direction the model uses. The O-lens was not tested.

## What the 3-item run got wrong

- **Competence.** The first three items of each bank are among the easiest. The gate passed 6 of
  9 there; over 30 items the model does about a third of multihop and multilingual.
- **"Lens hits follow what the model computes."** This still holds for the vector lenses, but
  not for the O-lens, whose 8/9 looked aligned only by chance at n = 3.
- **Non-Latin tokens.** The 3-item write-up said the judge cannot read byte-level BPE. It
  sometimes can:
  - In this run it credited ` Ð·Ð¾Ð»Ð¾ÑĤÐ°` ("золота", of gold) at L22 of atomic number 79, and
    `ÙĪØ¨ÙĬÙĨ` ("وبين", and between) on a typo item.
  - The same atomic-79 L22 bag was judged not expressed in the 3-item run.
  - So such tokens are read inconsistently, not never.
  - The spelling itself is deliberate: `multitoken/regex.py` keeps what the source scorer saw.
- **Environment.** The 30-item run had `flash-linear-attention` installed, so its activations can
  differ from the 3-item run's in the last bf16 digits.

## What to change before a real run on a small model

- **Gate first, and score only what passes.** Two thirds of multihop and multilingual items are
  not done by this model in either format. Their readouts cannot be interpreted (AGENTS.md).
  Pass the passing ids as `items=`.
- **Gate in the bank's format too.** `capable` asks a chat turn. For a 0.8B that measures
  instruction following as much as the task, and it disagrees with the raw continuation on most
  items. `scripts/small_model_check.py raw` is a stop-gap, not a `wsbench` command.
- **Judge the latent band.** On 24 blocks that is L16–L20. L22 is the next token, and a grid
  scaled down from the 27B misses the band.
- **Carry a null for the O-lens.** Measured by itself, the O-lens credits the oracle's own
  inference and enumerations. The derangement here costs one more judge pass.
- **Re-measure the floors.** The prompt-only floors in `evals/baselines/` are the 27B's (AGENTS.md).

## Caveats

- **Sample size and depth.** 30 items per family and two judged layers: the CIs above are wide.
- **Judge consistency.** Two identical token bags flipped verdict between the runs:
  - the opposite of "nah" at L22 (` nah`, ` fern`, …) was credited at 3 items, not at 30;
  - atomic number 79 at L22 went the other way.
  The judge runs at temperature 0, and repeated prompts still move about 0.01 (plan 0012).
- **Lenient verdicts.**
  - Beyond the enumerations, the null's 6 passes are generous readings: "address this crisis"
    credited for the noun "address", and "two grammatical forms" for "two".
  - Some J-lens passes rest on a target ranked 10th behind same-type candidates (` remote` after
    ` cerca`, `close`, `near`).
- **Echo.** On typo the O-lens largely reconstructs the prompt's scene ("bank account details",
  "ran directly across the busy street"), so the corrected word appears either way. The 27B
  prompt-only floor for typo is 1.00, and the report marks it saturated.
- **Multilingual credits the output.** The judge accepts a translation in any language, so the
  surface answer at L22 (`klein`, `petit`, `rechts`) counts. There it is the output, not a hidden
  latent.
- **Not the instrument of record.** The `layer_protocols` extra in `results.json` assumes 64
  layers (`split.py` `MODEL_LAYERS`) and is meaningless here. A `layers=` subset is never
  `complete`.

## Sources

- Lens artifacts: [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
  under `qwen3.5-0.8b/jlens/` (its `config.yaml` records the fit) and `qwen3.5-0.8b/oracle/`
  (its `run.json` and `CREDITS.md`).
- J-lens code: [anthropics/jlens](https://github.com/anthropics/jlens).
- O-lens method: Gurnee et al. (2026),
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html),
  Appendix A.9.2.
- English pivot: Wendler, Veselovsky, Monea and West (2024),
  [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588),
  ACL 2024.
- Model: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B); bank model
  [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B).
