# association

A scene implies a concept the text never names; does the lens name it at the final
prompt token. 100 items, plain render (no chat template), one read position, layers 20..60
step 4. The bank header's `eval_render_note` ("directed-modulation is the only chat-templated
family") predates the chat-rendered families and is superseded by `wsbench plan`.

Example: "Os tambores começaram na avenida ao anoitecer, e as fantasias cobertas de plumas dançaram até o amanhecer." -> target `carnaval`.

## Scoring

- **Headline: judged pass rate.** One judge call per (item, layer) sees every position's
  samples at that layer and decides whether any target concept is EXPRESSED: its meaning is
  stated (the word, a synonym, a translation, a paraphrase, a narrower instance, or a
  description that depicts the target itself; several phrases inside one sample may combine).
  Clues or evidence it could be inferred from, a restated question, hypernyms, the general
  topic, derived words, fragments, and enumerations of same-type candidates do not count, and
  pieces in different samples never combine. Every YES carries a verbatim quote, verified
  against ONE sample; an unverifiable quote voids the verdict. An item passes if any judged
  layer is expressed.
- Targets shown to the judge (`wsbench.banks.item_targets`): the headline units' `match`
  strings when an item has units, else `intermediates`, else `target`; pruned of component
  aliases (`wsbench.banks.exact_targets`).
- An item whose every expected layer is judged negative fails; one with no positive and at
  least one unjudged or missing cell is undecided and left out of the denominator
  (`extras.n_items_undecided`).
- A (item, layer) cell whose samples are all blank is a negative without a judge call.
- No analytic chance floor (free text). The measured floor is the prompt-only baseline: stock
  Qwen3.6-27B given the same prompt text and no activation, asked what a model would be
  thinking there, judged by this same instrument.
- Missing (item, layer) cells are fatal (exit 2) unless `allow_missing=True`; a dry run only
  reports them. `layers=` restricts the grid and counts as a subset.

## Readouts

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer,
position), where the label is the item `name` with characters outside `[A-Za-z0-9._-]`
replaced by `_` (`wsbench.banks.label_of`); `tokens` rows for a token lens.
`examples/readouts/association.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=association readouts=examples/readouts/association.jsonl out=outputs/toy/association dry_run=True
wsbench judge family=association readouts=<arm>.jsonl out=outputs/<arm>/association limit=3
```

## Judge prompts

`prompt_version`: `bank-2026-10-10`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. The prompts below are sent verbatim (`wsbench.basic.prompts`, shared by every
single-token basic family, plus this family's own strings). Output schema:
`{"expressed": bool, "target": str, "quote": str}`.

System prompt:

```text
You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. Grade meaning, not wording:
- Expressed means a sample states the target's MEANING: the word itself (any case or
  inflection), an established synonym or alias, a faithful translation into any language, a
  paraphrase, or a description that states or depicts the target itself (target "let the
  executive die" is expressed by "ignored the freezing man, leading to his demise").
- A narrower instance that necessarily entails the target counts ("a puppy" expresses "dog";
  "the toddler years" expresses "child").
- Several phrases inside ONE sample may together express the target when between them they
  state its specific content ("off-by-one | fencepost | hotel nights" expresses "fencepost
  error in the nights count"). Pieces in DIFFERENT samples or different tokens NEVER combine.
- NOT expressed by clues, circumstances or evidence from which the target could be inferred,
  including a restated question, index, definition or calculation: "the element with atomic
  number 26" does NOT express "iron"; "six divided by two" does NOT express "three"; "crayon
  drawings on the fridge" does NOT express "child". The sample must state the target itself.
- NOT expressed by a broader category (hypernym), a merely related or co-occurring idea, or the
  general topic.
- NOT expressed by a different or derived word that merely contains the target string
  ("blackout" does NOT express "black"), or by word fragments or partial tokens ("paragus"
  does NOT express "asparagus").
- Match the direction of the claim: if the target asserts a stance, a sample asserting the
  opposite, or naming the topic without the stance, does NOT express it.
- A sample that lists many interchangeable candidates of the same type (most of the digits,
  a run of month or sport names) expresses none of them unless it singles one out (states it
  as the answer, or sets it apart).
- A family rule or a forbidden-answer line, when given below the targets, overrides the rules
  above.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from ONE sample: the shortest
  contiguous span that expresses the target. No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}.
```

User message (`{targets}` = the JSON list of target strings, `{kind_note}` = one of the two
notes below; `{family_rule}` is empty for this family; `{forbidden}` is empty for this family; each of the last two is preceded by a newline when present;
`{blocks}` = one block per read position, joined by blank lines):

```text
targets: {targets}
{kind_note}{family_rule}{forbidden}

{blocks}
```

```text
[position {pos}]
{samples}
```

```text
The samples below are free-text readouts.
```

```text
The samples below are the lens's top VOCABULARY TOKENS, best first (one per line). Each position's tokens are one ranked list (rank 1 first). A target token ranked first among the candidates of its type is singled out, even if unrelated tokens rank above it; a target token ranked below another candidate of its type, in a list made mostly of such candidates, is not.
```
