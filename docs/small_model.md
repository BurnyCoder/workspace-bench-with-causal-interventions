# A <1B model on a slice of the bench: Qwen3.5-0.8B

Does the bench still give sensible numbers on a model under 1B parameters? This page runs one
small slice end to end (gate, three lens arms, judge) on **Qwen/Qwen3.5-0.8B** and reads every
verdict by eye. It was run on 2026-10-16. The slice is deliberately tiny, about 60 judge calls,
so it is a sanity check, not a measurement: n = 3 items per family.

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
    A.9.2 of
    [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html).

## What was run

- **Families.** multihop, multilingual and typo: single-token families that AGENTS.md lists as
  usable on a new model once it passes the gate.
- **Items.** `limit=3`, the first three bank items of each family. Bank order, plan order and
  gate order are the same.
- **Layers.** Readouts at layers 8, 10, …, 22. That is the oracle's trained grid, and roughly
  the 27B's 20..60 scaled by 24/64. Judging used layers 14 and 22 only, chosen before the run as
  "mid" and "late".
- **Judge.** The repo default, `google/gemini-3.8-flash`, under its pinned prompt
  (`bank-2026-10-10`).
- **Gate.** Local: `wsbench capable local=True`, greedy answer only (`draws=0`). It asks the
  bank's own question as a chat turn, thinking off, and the repo grader grades the answer.

```bash
set -a; source .env; set +a
M=Qwen/Qwen3.5-0.8B; L=8,10,12,14,16,18,20,22; O=outputs/small_model
JL=filename=qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt
OL=lora=neuronpedia/jacobian-lens:qwen3.5-0.8b/oracle,repo_type=model,alpha=60
uv run wsbench capable model=$M local=True families=multihop,multilingual,typo limit=3 draws=0 out=$O/capable
for f in multihop multilingual typo; do
  uv run wsbench produce family=$f model=$M method=logit_lens layers=$L limit=3 out=$O/readouts/logit_lens/$f.jsonl
  uv run wsbench produce family=$f model=$M method=jlens layers=$L limit=3 out=$O/readouts/jlens/$f.jsonl opts=$JL
  uv run wsbench produce family=$f model=$M method=olens layers=$L limit=3 out=$O/readouts/olens/$f.jsonl opts=$OL
done
for a in logit_lens jlens olens; do
  uv run wsbench run families=multihop,multilingual,typo readouts_root=$O/readouts/$a out=$O/judged/$a limit=3 layers=14,22
  uv run wsbench report dir=$O/judged/$a
done
```

Cost and time on an 8 GB RTX 5070 laptop GPU:
- 63 Gemini Flash calls (9 gate grades and 54 judge cells), about $0.04 in total.
- The vector lenses take seconds per family.
- The O-lens takes about 37 s per cell, because generation runs 256 tokens through the
  linear-attention layers. Without the optional kernels, transformers warns that
  `chunk_gated_delta_rule` and `causal_conv1d_fn` are "falling back to its reference PyTorch
  implementation" (install `flash-linear-attention` and `causal-conv1d` to avoid it).

## Results

### The gate

The `capable` gate asks the question as a chat turn. The raw continuation is the bank's plain
render continued greedily for 8 tokens, without the chat template. It is not a `wsbench` command;
it is shown here because the banks were built on continuations.

| item | gold | `capable` greedy (chat turn) | raw continuation |
|---|---|---|---|
| multihop/`atomic-26-symbol` | Fe | Fe ✓ | `60.` |
| multihop/`atomic-79-symbol` | Au | Au ✓ | `79.` |
| multihop/`birthstone-emerald-month` | 5 | October ✗ | `5.` |
| multilingual/`de-half-six` | drei | 6 geteilt durch 2 ist 3 ✓ | `6"` |
| multilingual/`de-place-near-far` | fern | fern ✓ | `fern".` |
| multilingual/`de-place-up-down` | unten | oben ✗ | `unter".` |
| typo/`typo-February` | February | February ✓ | `.` |
| typo/`typo-account` | account | password ✗ | `.` |
| typo/`typo-across` | across | across ✓ | ` the street.` |

The bank recorded 1.0 for Qwen3.6-27B on all nine items. The 0.8B passes 6 of 9 in the chat
turn. On the six non-typo items the two formats disagree four times. One caveat: the multihop
prompts end in a trailing space, so their raw continuation starts after a lone space token.

### Judged pass rate (layers 14 and 22; n = 3 per family)

| arm | multihop | multilingual | typo |
|---|---|---|---|
| logit lens | 1/3 | 2/3 | 1/3 |
| J-lens | 0/3 | 1/3 | 3/3 |
| O-lens | 2/3 | 3/3 | 3/3 |
| *Qwen3.6-27B O-lens, all 100 items, 11 layers* ([plan0012_results.md](plan0012_results.md)) | *0.99* | *1.00* | *1.00* |

At n = 3 the bootstrap CI is [0, 1] for every rate strictly between 0 and 1. The run had 0
missing, 0 unjudged and 0 empty cells.

### Per item

The scored latent follows the arrow, e.g. Fe → iron. "✓ / ·" is the verdict at L14 / L22.

| item | gold → latent | greedy (chat) | logit lens | J-lens | O-lens |
|---|---|---|---|---|---|
| multihop/`atomic-26-symbol` | Fe → iron | ✓ | · / ✓ | · / · | · / ✓ |
| multihop/`atomic-79-symbol` | Au → gold | ✓ | · / · | · / · | · / ✓ |
| multihop/`birthstone-emerald-month` | 5 → May | ✗ | · / · | · / · | · / · |
| multilingual/`de-half-six` | drei → three | ✓ | · / · | · / · | · / ✓ |
| multilingual/`de-place-near-far` | fern → far | ✓ | · / ✓ | · / · | ✓ / ✓ |
| multilingual/`de-place-up-down` | unten → down | ✗ | · / ✓ | · / ✓ | ✓ / ✓ |
| typo/`typo-February` | February | ✓ | · / · | ✓ / · | ✓ / ✓ |
| typo/`typo-account` | account | ✗ | · / ✓ | ✓ / · | ✓ / ✓ |
| typo/`typo-across` | across | ✓ | · / · | ✓ / · | ✓ / ✓ |

### Depth: where the latent is, layer by layer (free, not the judge)

This counts, per layer, the items whose English target appears verbatim, as a whole token of a bag
or a whole word of a prose sample. Translations do not count and no judge is involved. It is a
read of all 8 produced layers, not just the two that were judged.

| arm (9 items) | L8 | L10 | L12 | L14 | L16 | L18 | L20 | L22 |
|---|---|---|---|---|---|---|---|---|
| logit lens | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 2 |
| J-lens | 2 | 3 | 2 | 3 | 3 | 5 | 4 | 3 |
| O-lens | 4 | 6 | 5 | 5 | 3 | 7 | 5 | 6 |

Readout excerpts, top tokens best first:
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
- **O-lens, atomic number 79.** L22: "金 (Au). The name of this element is **Platinum**. Gold and
  platinum are both heavy transition metals…" (one sample per cell, temperature 1).

## Does it make sense?

Each criterion below was written down before the run.

1. **Clean run: yes.** No missing, unjudged or empty cell. The O-lens LoRA loaded onto
   `Qwen3_5ForCausalLM` with no PEFT warning, and the J-lens Jacobians matched the residual width.
2. **A plausible gate: yes.** The 0.8B passes the easy items. Its chat-turn misses are one
   knowledge gap (emerald → "October") and two format failures: it echoes "oben" for the opposite
   of "oben", and answers "password" when asked to correct "acount".
3. **A plausible depth profile: mostly.**
   - Logit-lens hits sit late (L18–L22, peak L20).
   - The J-lens decodes the target earlier and more often: typo 3/3 at L14, against the logit
     lens's 0/3.
   - The O-lens prose is on topic at every layer.
   - The expectation "judged L22 ≥ L14" holds for the logit lens (0/9 at L14, 4/9 at L22) and the
     O-lens (5/9, 8/9). It fails for the J-lens (3/9, 1/9). In a 24-block model, block 22 is the
     second-to-last block and mostly carries the next token, so the J-lens's latent band is
     L16–L20.
4. **Lens hits follow what the model computes: yes.**
   - The one knowledge gap (emerald → May) is negative on every arm at both judged layers.
   - The two format failures are positive on the lenses: the continuation computes `unter` and
     the readouts name down/unten and account. A lens is not wrong to find what the chat gate
     missed.
   - Where the continuation itself is wrong (`6"` for six divided by two), the logit lens shows
     `6` and ` zwei`, not "three", and fails.
5. **At or below the 27B: yes.** The O-lens gives 2/3, 3/3, 3/3 against the 27B's 0.99, 1.00,
   1.00, with the same ordering. At n = 3 this says only that nothing is upside down.

## What to change before a real run on a small model

- **Judge the latent band, not the 27B's mapped grid.** On 24 blocks the band is L16–L20, and L22
  is the next token. A fixed `layers=` choice that is right for 64 blocks is wrong here. Judge all
  produced layers, or pick the band from a dev split.
- **Gate in the bank's format too.** `capable` asks a chat turn. For a 0.8B that measures
  instruction following as much as the task, and here it disagrees with the raw continuation on 4
  of 6 continuation items. A raw-continuation gate, closer to how the banks were gated, is not
  implemented.
- **Re-measure the floors.** The prompt-only floors in `evals/baselines/` are the 27B's (AGENTS.md).

## Caveats

- **n = 3 per family.** These are not measurements.
- **Lenient verdicts.**
  - On the antonym items, O-lens prose tends to name both poles. One sample says "上面 (top) и
    "down" (bottom)" and also, in Russian, "the correct answer … is **up**", and the judge
    credited "down".
  - `de-half-six` passes only on L22's "…die zwei und drei kombiniert". Read strictly, O-lens
    multilingual is 2/3.
  - The J-lens `typo-across` L14 pass rests on ` across` ranked 10th, behind street and hill
    tokens.
- **Echo.**
  - The O-lens on typo largely reconstructs the prompt's scene ("bank account details",
    "ran directly across the busy street"), so the corrected word appears whether or not it was computed.
    The 27B prompt-only floor for typo is 1.00, and the report marks it saturated.
  - The multilingual judge credits a translation in any language, so the German surface answer
    (`fern`, `unten`) at L22 counts. There it is the output, not a hidden latent.
- **Non-Latin tokens.** Token bags spell non-Latin tokens as byte-level BPE, so the judge cannot
  read them. The logit lens on atomic number 79 has `éĩĳçļĦ` ("金的", gold's) at L18
  and ` Ð·Ð¾Ð»Ð¾ÑĤÐ°` ("золота", of gold) at L22, and L22 was judged not expressed. This is deliberate:
  `multitoken/regex.py` keeps the spelling the source scorer saw. It applies to the token arms on
  any model.
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
