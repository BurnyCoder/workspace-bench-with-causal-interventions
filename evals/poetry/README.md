# poetry

A rhyming couplet cut before its final word; the scored latent is the committed rhyme
word. 100 items, plain render, read at the **newline that ends line one**, one position per
layer. The read moved there on 2026-09-23 from the final prompt token, the space before the
rhyme word: at the end of line one the model has committed to a rhyme and the O-lens already
writes line two with it, while at the final token the readout echoes the line just read. The
judge is unchanged: one call per (item, layer), an item passing on any layer. The bank header's
`eval_render_note` ("directed-modulation is the only chat-templated family") predates the
chat-rendered families and is superseded by `wsbench plan`.

Example: "A rhyming couplet: The captain pointed at the route ahead, And told his crew to follow where he'd" -> target `led`.

**Model-specific.** The scored latent is the rhyme word THIS model commits to. Measured with
`wsbench capable` at 5 draws: Gemini 3.8 Flash and DeepSeek V4 Pro each commit to the bank's word
on 98 of 100 items, but both answer *June* where Qwen3.6-27B commits *noon* (`b3-po-tune-june`),
and each drifts to a near-rhyme on one more item. Re-gate before reading this bank on another
model; AGENTS.md has the procedure.

## Scoring

- **Headline: judged pass rate.** One judge call per (item, layer) sees every position's
  samples at that layer and decides whether the target WORD is expressed. The shared bank
  judge applies with this family's rule on top (below): only the word itself, an inflection,
  or a faithful translation counts; synonyms, paraphrases, descriptions, and related
  words do not. Every YES carries a verbatim quote, verified against ONE sample; an
  unverifiable quote voids the verdict. An item passes if any judged layer is expressed.
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
`examples/readouts/poetry.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=poetry readouts=examples/readouts/poetry.jsonl out=outputs/toy/poetry dry_run=True
wsbench judge family=poetry readouts=<arm>.jsonl out=outputs/<arm>/poetry limit=3
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
notes below; `{family_rule}` = this family's rule (below); `{forbidden}` is empty for this family; each of the last two is preceded by a newline when present;
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

Family rule (`FAMILY_RULE_POETRY`):

```text
Family rule: the target is a specific WORD the model plans to write. Credit only that word, an inflection of it, or a faithful translation of it. Synonyms, paraphrases, descriptions, derived words ("greatness" for "great"), and narrower or related words are NOT credit.
```
