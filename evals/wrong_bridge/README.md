# wrong_bridge

Does the lens name the model's WRONG intermediate? A Wikidata two-hop prompt where Qwen3.6-27B resolves the wrong bridge B' (instead of the true B) and so answers hop2(B'). The scored latent is B'.

Example: "Fact: The composer of Canon and Gigue in D died in the year" -> model answer `1750`
(Bach's death year); true bridge Johann Pachelbel (1706), the model's bridge Johann Sebastian Bach.

## Bank

`items.json`: 100 items, plain render (no chat template), read at the **last word of the prompt**
(every prompt ends in ` year` or ` of`, no trailing space, so `readout.offsets` is `[-1]`, the
final token), one position per layer on the standard grid L20..60 step 4. Each item carries:
`prompt`, `target` (the model's wrong answer A'), `correct_answer` (A), `intermediates` (the
judged target), `intermediates_aliases`, `true_intermediates` (B), `wikidata` (`S`, `B`,
`B_prime` QIDs), `evidence` (the behavioral gate numbers below) and `adjudication`.

Built by `scripts/wrong_bridge/` in the global-workspace repo (plan
`docs/superpowers/plans/2026-10-09-wrong-bridge-multihop.md`): 21 Wikidata two-hop templates
(film -> director, novel -> author, composition -> composer, person -> father, thing -> namesake,
...) x hop-2 relations (birth/death year, birth/death city, founding year, HQ city), 258k
candidates, screened on Qwen3.6-27B (raw text, greedy). An item is kept only if ALL hold:

1. the greedy 2-hop answer is wrong, and the model answers hop 2 correctly for the TRUE bridge B;
2. asked hop 1 alone, the model names B' != B, and its 2-hop answer equals B''s Wikidata value;
3. >= 4/10 T=0.7 samples give A' and >= 4/10 give B' (`evidence.p2_samples_Ap`,
   `evidence.p1_samples_Bp`; 50 items reach >= 6/10 on both); logP(A') > logP(A) and
   logP(B') > logP(B) teacher-forced;
4. injecting B' into the prompt ("..., B', was born in the year") gives A', injecting B gives A,
   and hop 2 asked of B' directly gives A';
5. A' is not a default answer (given to < 3% of the template's prompts whose hop-1 bridge is
   someone else), B' and B do not appear in the prompt, B' is not credited on S in Wikidata, the
   subject label is unique within its template;
6. a Claude Opus 5.5 audit finds no ambiguity (same-title works, co-holders, publisher-vs-developer,
   stale Wikidata).

At most 20 items per template and 4 per wrong bridge. The capability gate does not port: the
bank is a specific Qwen3.6-27B error (`capable` lists it under NO_QUESTION).

## Scoring

Same instrument as `multihop`: one judge call per (item, layer) over that layer's samples;
an item passes if any judged layer names a target. Targets shown to the judge: `intermediates` = [B'] (the model's wrong bridge). Missing (item, layer) cells are fatal
unless `allow_missing=True`. No analytic chance floor (free text); the measured floor is the
prompt-only baseline (stock Qwen3.6-27B shown the prompt, asked what a model would be thinking).

## Readouts

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows; label = item `name`
(`wsbench.banks.label_of`).

```
wsbench judge family=wrong_bridge readouts=<arm>/wrong_bridge.jsonl out=outputs/<arm>/wrong_bridge limit=3
```

## Judge prompts

`prompt_version`: `bank-2026-10-10`. Judge: the repo default (`google/gemini-3.8-flash`); the 2026-10-09 arm comparison overrode it to `claude-opus-5` (the recent in-house convention),
temperature 0. The prompts below are sent verbatim (`wsbench.basic.prompts`, shared by every
single-token basic family). Output schema:
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
notes below; `{family_rule}` is empty for this family; `{forbidden}` = the forbidden-answer line (below) with `{answer}` = the item's surface `target` as JSON; each of the last two is preceded by a newline when present;
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
