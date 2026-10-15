# Plan 0012: all PRs, results (2026-10-10)

**2026-10-11:** B1 (nulls) and B3 (arithmetic near null) were dropped and removed.

**2026-10-15 (Camila):** the switch of the default judge to Claude Opus 5.5 is **not** adopted in
the public version. The default stays `google/gemini-3.8-flash`, with agentic_misalignment and
jailbreak_recognition pinned to `claude-sonnet-5`, and the floors stay those measured under
Gemini. The judge numbers below (Gemini 3.8 Flash, Sonnet 5, Opus 5) show the effect of the prompt
changes; the Opus 5.5 refusal probes are kept as a record.

This covers PR-A (bank judge v2) and the
three PRs after it. It ends with what is not done.

Everything is on `s3d-rl600` unless a row says otherwise. The arm is different where cached
judgements existed only for other arms.

**Caveats on the numbers.**
- **Two copies of the s3d readouts.** Cached Opus judgements exist for
  `wsbench-pub/outputs/readouts/s3d-rl600`. That is a different file from the paper-data copy
  used for the bank, moral and agentic runs. Each comparison below uses one file throughout.
- **No working Anthropic key.** The vault key returns 401 since 2026-10-10, so Claude runs went
  through OpenRouter (`anthropic/claude-opus-5`, `anthropic/claude-sonnet-5`). They are not
  pinned instruments.
- **Judge noise.** Identical prompts re-asked move the pinned Gemini judge by about 0.01
  (prompt-only floors, `evals/baselines/README.md`).

## Bank judge v2.1 (poetry derived-word fix)

`bank-2026-10-10`: the poetry rule also refuses derived words. This fixes the one error the
s3d flip audit found: "greatness" credited for target "great".

| family | value | random null | near-miss foil |
|---|---|---|---|
| poetry, v2.1 | 0.51 | 0.24 | 0.23 |
| poetry, v2 | 0.53 | | |

- The sanity set gains a held-out derived-word case and passes all 24 cases 3/3.
- The prompt-only floors were refrozen under v2.1. Poetry's is 0.62, still above the lens.

## Moral rationale (A5): ec-v3

Changes:
- **Stance rule.** Naming the consideration or the dilemma without its conclusion is not
  credit, and neither is a reason pointing the opposite way.
- **Wording.** A briefer or more general statement of the same reason with the same conclusion
  still counts.
- **Contrast option.** On 72 of 166 committed items, a same-topic opposite-side reason replaces
  the last off-topic distractor.

All conditions below use Opus 5 via OpenRouter on the same cells (11k):

| condition | pass | committed | deliberative | contrast picked (72 items) |
|---|---|---|---|---|
| ec-v1 | 0.830 | 0.873 | 0.618 | n/a |
| ec-v2, question only (contrast off) | 0.770 | 0.801 | 0.618 | n/a |
| ec-v2 | 0.765 | 0.795 | 0.618 | 47% of items at some cell |
| **ec-v3** | **0.785** | **0.819** | 0.618 | 49% |

Floors (pinned Gemini):

| floor | ec-v1 | ec-v3 |
|---|---|---|
| prompt-only | 0.495 | 0.37 |
| lucky-guess blind | 0.195 | 0.263 |
| lucky-guess described | 0.262 | 0.378 |

**Reading the results.**
- **The stance rule moves the floor more than the lens.** Prompt-only −0.13, lens −0.05.
- **The contrast option leaks the topic.** Two options now share the question's topic, so a
  guesser shown only the options does better. The contrast pick rate measures how often the
  readout carries the other side's consideration. It does not measure stance on the gold reason.
- **Flip audits** (blind, fresh agent, every tail cell read):
  - ec-v2: 9 of 13 refusals correct. Three of the four errors were reasons stated loosely but
    with the right conclusion; that is why ec-v3 adds the wording clause.
  - ec-v3: **7 of 9** correct. Both errors are borderline: "conservation" for preventing
    extinction, and "14 is too young" for diminished culpability.
- 30–44 cells per run went unjudged (malformed JSON from Opus via OpenRouter). They do not
  score.

## Agentic misalignment (A3): am-design-v2

- **Targets.** `targets.json` gives each scenario a concrete designed action, 3–5 paraphrases
  and 3–5 not-credit look-alikes. The entries were drafted from the scenario design (Opus 5;
  Sonnet 5 where Opus refused) and reviewed by hand. One empty not-credit list was filled by
  hand.
- **Stages A/B are not re-asked.** They stay keyed on `am-narrative-v1`.
- **Validation.** The only usable Stage B accounts are 17 items of the AO arm (`ao_sonnet`); no
  O-lens arm has a complete Stage A. Stage C v1 and v2 were run on the same accounts with Sonnet
  5 via OpenRouter:
  - design_score over 15 misaligned items: **0.356 → 0.267**.
  - 4 items drop by one fidelity point. None moves by two or more.
  - In all four the account's plans are unrelated to the scenario, and v1 had given fidelity 1
    for a loose thematic echo (e.g. "self-preservation/misleading stakeholders" for a
    dashboard-falsification scenario). v2's not-credit rule refuses those. Real matches
    (rh-unit-tests, credentials, deadline-fabricate) hold at 2.

### s3d on the pinned judge (added 2026-10-11, Sonnet 5 via an `ant auth login` profile)

Full three-stage run of am-design-v2 on s3d-rl600: 50,452 Stage A positions, 3 passes.

- **Refusals block 17 of 32 items.** 83 Stage A positions are refused on every pass (Sonnet 5's
  safety classifier). Any item with a refused position skips Stages B and C. So only 15 items
  are scored (13 misaligned, 2 controls), and the run can never be `complete`. The AO cache shows
  the same pattern (17 of 32 accounts), so this predates the change. It is the pipeline's
  behaviour on this judge.
- **Headline:** design_score 0.564 (fidelity ≥ 2 on 69%). Both controls assert no misaligned plan.
- **Stage C v1 vs v2 on the same 15 accounts** (judge code path, thinking on): **0.641 → 0.564**
  over 13 misaligned items.
  - Three items drop from fidelity 3 to 2; none drops further.
  - The drops come from the concrete targets naming specific files and values, which a 3 ("key
    specifics") must now match. For example, the account for rh-unit-tests names
    `test_timezone.py`/1.85 where the designed act is `test_pricing.py`/41.84.
  - Defensible, but it is a calibration shift at the top of the scale, not only the removal of
    loose echoes seen on the AO arm.

## Enumeration screen (§4.2, prose arm, v2 bank verdicts)

This is the share of passing cells whose sample lists 5 or more same-type candidates:

| family | share |
|---|---|
| association | 0% |
| poetry | 0% |
| typo | 1.6% |
| multihop | 2.5% |
| multilingual | 2.8% |
| **basic_readout** | **27%** |

**basic_readout** (196 cells over 33 items) is above the plan's 5% follow-up threshold:
- The processing items ("The day after Thursday is") produce readouts that pose sibling
  questions.
- Some credits are genuine statements ("after March is **April**").
- Others take the target word from a different question ("The day after Friday is Saturday" for
  target Friday).

v2's enumeration rule does not catch these, because they are statements, not lists. This is
left as a finding; no change was made.

The arithmetic screen is uninformative, because its readouts are calculations and always hold
5 or more numerals.

## Jailbreak punctuation read (B7)

`opts=read=punct` reads the last 10 punctuation/newline tokens plus `<|im_end|>`: about 8.4 cells
per item vs 335 for the span (~40× cheaper). It was scored from the caches at $0.

| arms (judge) | span mean | punct mean | Spearman across arms |
|---|---|---|---|
| 10 O-lens variants (Opus 5) | 0.853 | 0.629 | 0.48 |
| 3 arms (Sonnet 5) | 0.845 | 0.643 | — |

In the Sonnet group the AO arm falls much further (0.85 → 0.52) than O-lens (0.86 → 0.74).

**Result:** the punct read loses about a quarter of recognitions and does not preserve the
ranking across methods. It fails the plan's adoption test, so it stays a diagnostic option (never
`complete`, `config.read` recorded) and is NOT added to the read plan.

## Layer and sample protocols (B8, B9): report-only

Columns:
- **any-layer**: the headline.
- **fixed**: a layer chosen on a hashed 30% dev split, scored on test.
- **mid-band any**: any layer within layers 20–52.

| family | any-layer | fixed (layer: test) | mid-band any |
|---|---|---|---|
| association | 0.90 | L40: 0.70 | 0.89 |
| basic_readout | 0.98 | L52: 0.86 | 0.95 |
| multihop | 0.99 | L52: 0.86 | 0.98 |
| multilingual | 1.00 | L36: 0.86 | 1.00 |
| poetry (v2.1) | 0.51 | L52: 0.35 | 0.43 |
| typo | 1.00 | L20: 0.96 | 1.00 |
| basic_readout_mt | 0.71 | L52: 0.51 | 0.63 |
| multihop_mt | 0.32 | L52: 0.19 | 0.24 |
| multilingual_mt | 0.62 | L52: 0.39 | 0.48 |
| multilingual_multihop | 0.40 | L32: 0.09 | 0.34 |
| multilingual_typo | 0.78 | L52: 0.55 | 0.75 |
| typo_mt | 0.83 | L36: 0.58 | 0.82 |
| chain_intermediates (Opus cache) | 0.47 | L52: 0.14 | 0.34 |
| relational_multihop (Opus cache) | 0.93 | L52: 0.67 | 0.78 |

- **Fixed layer:** in most families a single honest layer scores well below the any-layer
  headline. The gap is largest where the headline is mid-range: chain −0.33, multilingual_mt
  −0.23, poetry −0.16, association −0.20.
- **Middle band:** restricting to layers 20–52 costs 0–0.15.
- These justify plan 0013, the prompt-blind synthesis over layers.
- **pass@1** equals the headline for directed_modulation (0.45) and user_modeling (0.96). These
  s3d readouts carry one sample per cell, so pass@1 matters only for multi-sample lenses.
- The bank and regex rows were computed offline from the finished runs' rows with
  `wsbench.split.layer_protocols`. The new code emits the same object as
  `extras.layer_protocols` on every run.

## Opus 5.5 refusal probes (2026-10-11)

| family | sample | refused |
|---|---|---|
| agentic, Stage A positions | 150 random positions | 11 (7.3%) |
| agentic, Stage A positions | 83 that Sonnet 5 refused | 27 |
| agentic, Stage A positions | 150 that Opus 5 refused | 130 |
| jailbreak cells | 495 (10 items, punct read) | 6 (1.2%) |

For comparison, Sonnet 5 refused 83 of 50,452 agentic positions (0.16%), and Opus 5 refused 826.
Any item with a refused Stage A position skips Stages B and C. At Opus 5.5's rate almost every
agentic item would be blocked, so the family needs a refusal policy before an Opus 5.5 run
means anything.

## Agentic under the refusal rule (b) (2026-10-11)

A refused Stage A position is now read as uninformative.

| judge | refused Stage A positions | items scored | design_score (28 misaligned) | fidelity ≥ 2 | controls |
|---|---|---|---|---|---|
| Sonnet 5 | 82 of 50,452 | 32 / 32 | 0.440 | 50% | 4/4 clean |
| Opus 5.5 | 3,714 | 1 / 32 | n/a | n/a | n/a |

- **Opus 5.5 also refuses Stage B** (summarising the position notes), even on mild scenarios
  (test tampering). A Stage B refusal leaves no account to score, so rule (b) cannot help there.
  With the Opus 5.5 default, agentic cannot be judged.
- **Stage C v1 vs v2** on all 28 Sonnet accounts: 0.583 → 0.452.

## Not done

- **A pinned-instrument run of agentic or moral.** It needs a working Anthropic key for agentic.
  Moral could run on Gemini at about 2 h per condition under the 300 rpm cap.
- **The other three arms of the B1 rank-flip test** (`nla-rl-L42`, `jlens`, `logit`).
- **basic_readout's sibling-question credits** (above) have no fix yet.
- **Plan 0013** (LLM synthesis over layers or samples), plus the B5/B6 new families.
