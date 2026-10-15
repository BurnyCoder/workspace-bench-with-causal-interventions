# 0013: multihop_mt v2, a shortcut-gated hard multihop bank

Status: implemented (rev 3, folds in the decisions made while building it; each one below is
marked **[rev 3]** with its evidence). Owner: Camila. Date: 2026-10-15.

## Why

An audit of the frozen `multihop_mt` bank (100 items, frozen 2026-09-10) ran ablation probes on
Qwen3.6-27B with the bank's own gate recipe. It found that **93/100 items, including all 32
"3hop" items, are reachable while skipping a claimed bridge**: removing the clue that pins a
bridge, or keeping only the final clause with a generic subject, still gives the target at
≥5/10. Neel flagged three of these (the Sogdian-commander, Xuanzang-monk and
Victorian-noblewoman items).

The failures come in two kinds:
- **Specific final clause.** The answer clause names T by itself: "the ninth planet found in
  1930 by" → Tombaugh 10/10.
- **Category prior.** T is the default answer of its class: "the famous American racehorse
  spent his breeding career at" → Claiborne 10/10.

The root cause is the v1 generator prompt, which asked for "long-tail, obscure-but-real" bridges.
Qwen knows only an obscure entity's single most famous fact, so the final hop had to be that
fact, and that fact is what makes the final clause a shortcut. The v1 gate checked that Qwen
*can* answer, never that it *must* go through the bridges.

Three repair rounds verified 15 clean items. They are carried in below as seeds, subject to the
v2 gate.

## Goal

User's statement (Camila, 2026-10-15): "significantly harder than the standard single token
multihop eval, pushing the limits of what qwen can faithfully do, but we must make sure qwen can
actually do these without cheating", using "more commonly known questions where the bridges are
nonobvious".

**Target: 100 items, at least 50 three-hop.**
- A shortfall is reported, never padded.
- The bank is NOT VIABLE below 60 items in total (v1's floor). Separately, it is reported as
  **not meeting the harder goal** below 30 three-hop items.
- The single-token `multihop` family is one silent hop through a one-token bridge ("…home to
  the Eiffel Tower" → France → euro). v2's difficulty comes from mostly-3-hop chains,
  multi-token bridges, and composition through clues that each pin only their own entity.

## Item contract

**Retained from v1, restated so the implementation keeps it:**
- Plain render. The prompt starts `Fact: ` and ends immediately before the answer word (no
  trailing space or punctuation).
- Read = `final_prompt_token`, `offsets: [-1]`.
- `units[]` = one required unit `bridgeK` per bridge with `en` forms.
- Bank `contract` = `{multi_token: true, include_target: true, conjunctive_units: true}`. Every
  required bridge must hit in one layer, and only forms stamped >1 Qwen token are creditable.
- The optional `target` unit feeds `any_hit`, never the pass.
- `bridges[]` keep their gate-only `answers`.
- **`probe_token_lens` stamps `units`, `target` AND `target_alts`**, positionally aligned. The
  wsbench scorer requires `target_alts` under `include_target` (`regex.py:325-338`). v1's gw
  finalizer never stamped it, so v2's finalizer must.
- Header `family: "multihop-hard"`, in both repos, exactly as v1's frozen wsbench copy
  (`tests/test_multihop_mt.py:17` asserts it for v1). Then `lint_hard_item`, `hard_mt_lint`
  and `hard_mt_mc_options` treat it like v1, and the header `contract` governs scoring. It is
  stored in its own directory (below), so it never collides with the frozen v1 file.
- `consistency` is stored as a fraction (n/20), because wsbench `capable` reads it as a rate.
  `bridges[].question` holds the G2 question, because wsbench `capable` asks that field.
  `spans` are kept on the final items, so the v2 lint can rerun on the frozen bank.

**New, enforced by the generator prompt AND lint:**
1. **Bridges are strictly multi-token** (Qwen count >1, the scorer's rule; single-word names
   such as "Boethius" or "ivermectin" qualify).
   - No single-token bridges, because the scorer cannot credit them.
   - **No compound bridges** ("X and Y").
   - T may be single-token: its unit is optional.
2. **The leak rule also covers target_alts.** v1's lint (`lint.py:189-196`) checks every unit
   form and the target, but not target_alts; v2 closes that gap.
3. **Generic final relation.** T is not the modal answer of its class.
4. **Isolated clues** (lint-checkable via spans, below). No capitalized token of D1 (other than
   its first word and the lint's `STOPWORDS`, compared case-sensitively) and no digit run of D1
   appears in the downstream relations.
5. **No overlap across creditable forms.** No pair of forms across units may match each other
   in either direction under `unicode_word_matcher`: bridge1 against bridge2, and each bridge
   against the target and every target_alt. This is the lint's `_collisions` logic applied
   across roles. Otherwise a readout naming only T, or only one bridge, would satisfy a second
   unit.
6. **Well-known hops.** Each hop is among the entity's top encyclopedic facts. Difficulty comes
   from composition, not obscurity.

**Spans.** The generator returns the prompt as spans, and the lint checks they concatenate to
the prompt exactly:
- 2-hop: `"Fact: " + D1 + R1`.
- 3-hop: `"Fact: " + D1 + R1a + N2 + R2`.

Here `D1` is the clue for B1; `R1a` is the relation from B1 up to the B2 slot (e.g. " had
earlier served in the senate of"); `N2` is the noun phrase standing for B2 (" a state"); `R2` is
the final relation (" whose capital is the city of"). The generator also returns `G1`: a generic noun phrase of B1's class with no distinguishing
detail ("A nineteenth-century American printer").

## Gate

The gate runs Qwen3.6-27B on Modal through the HF `ModelBackend`, with the same rendering as
`mt_check.py`: chat template, `enable_thinking=False`, then `MT_ANSWER_STEM`,
`max_new_tokens=24`, `qwen_check.contains_target`. Fill-in probes are wrapped in
`FILL_IN_MT_QUESTION`; question probes are the user turn.

| stage | probe | pass rule |
|---|---|---|
| G1 correctness | the item prompt | greedy hit AND ≥16/20 @0.7 (v1 was ≥8/10, the same 80%; 20 samples because the audit saw a keep item drift 8→7→6/10) |
| G2 bridge recall | B1: the descriptor question **built from the prompt's own clue text** ("Name " + D1) → B1 | greedy AND ≥8/10 |
| G3 hop recall | each remaining hop with the source NAMED: 2-hop "B1 → T"; 3-hop "B2 → T" | greedy AND ≥8/10 |
| G4 in-context B2 (3-hop) **[rev 3]** | the full statement with the answer blanked: "In this statement, who or what is `N2`?" → B2 | greedy AND ≥5/10 |
| S1 mechanical ablation | `drop_b1` = `"Fact: " + G1 + R1` (2-hop) / `"Fact: " + G1 + R1a + N2 + R2` (3-hop) | ≤4/20 AND greedy miss |
| S1 authored ablations | `local_cue` (the final relation with a fully generic subject, answer type kept; it tests the category prior; lint: ends with the prompt's last 4 words, no D1 capitalized token/number, no bridge form); 3-hop `direct_b1_to_t` (B1 named + **the item's final relation**, with B2's relation removed, e.g. "Fact: Andrew Wyeth … the Brandywine village of"; lint: contains a B1 form, no B2 form, ends with the prompt's last 4 words) | each ≤4/20 AND greedy miss |
| S2 adversary | ≤4 probes from an independent call (fresh context) that sees the prompt, the chain and T but NOT the S1 probes. It is given the audit's full shortcut taxonomy (local cue, skip a bridge by date/place, B1→T direct, bridge named or near-named, descriptor-is-T-fact, category prior, bridge not on the path) and returns probes plus `factual_issue`. **[rev 3]** Every probe must be an ablation of the prompt; a probe with a capitalised word or number not in the prompt is discarded | each ≤4/20 AND greedy miss; any `factual_issue` rejects |
| ~~ID check~~ **[rev 3, withdrawn]** | (see below) | no shortcut probe is ever excused |

Rules:
- **A missing adversary row, zero S2 probes, or an API failure rejects the item (or retries it).
  It never passes.** For 3-hop items the adversary must return at least one B1-named probe;
  without one the row counts as missing. One live preflight call runs before the batch.
- **Shortcut probes are scored against a superset** of answer forms: target, every
  generator-proposed alt (including alts trimmed for the leak rule), and a person-name target's
  surname. Trimming a form from the creditable lists must never make a shortcut probe easier
  to fail.
- **No `drop_b2` probe.** The smoke pilot (16 candidates, 2026-10-15) showed that "D1 + a vague
  relation + R2" still routes through B2 whenever B2 is B1's most famous associate: "Kepler had
  some connection to an astronomer whose observatory was called" → Uraniborg 5/10, via Tycho.
  That probe tests whether the *relation* is needed, not whether the *bridge* is. B2's necessity
  is tested by `direct_b1_to_t` (B1 → T without B2), by `local_cue` (the category prior of the final relation) and
  by S2. `N2v` is dropped from the spans.
- Smoke yield (rough, n=16): 2-hop 4/8 pass every stage; 3-hop 0/8. The 3-hop losses split
  into **composition failures** (every hop known at ≥9/10 but the full prompt 0–1/10, e.g. Davy
  → Faraday → *The Chemical History of a Candle*, Oppenheimer → IAS → Flexner) and real
  shortcuts (Irène Joliot-Curie → Panthéon direct 10/10; "the dimensionless quantity" →
  fine-structure constant 7/10). 3-hop yield is the bottleneck, so the pilot sizes the run.
- Stages run in order (G1 → G2/G3/G4 → S1/S2), each a Modal fan-out with one H200 per shard,
  `k` = shards. Each stage writes its rows, so a rerun resumes.

### Gate decisions made during implementation [rev 3]

Each change below was made after reading real failures, and is logged here so a reviewer can
check the code against it:
- **The prefix check was replaced** by the in-context B2 check (G4 above).
  - The rev-2 G4 cut the prompt before B2 (`"Fact: " + D1 + R1a` → B2), and G2 for B2 asked
    `"Fact: <B1> " + R1a` → B2. Both rejected valid chains in bulk, because B1 → B2 is often
    one-to-many and `N2 + R2` pick B2 out. Example: "The founder of the Mongol Empire was the
    grandfather of" → Kublai only with "…a ruler who twice attempted to invade".
  - About 22 of the 50 three-hop items that passed G1 in main run 1 failed only these two
    checks. Their rows are kept but superseded (`gate_modal.SUPERSEDED`).
  - The new check's bar is ≥5/10 because the question is format-sensitive: Maupassant →
    Flaubert scores greedy-correct while some samples name B1. The causal evidence for both
    bridges comes from the ablations.
- **The ID check (new; SUPERSEDED, see "ID check withdrawn" below).** An adversary probe can re-describe a bridge by another unique clue
  ("the dome engineered in the 1420s without centering"), and reaching T from such a probe is
  not a shortcut. The excuse is limited to adversary probes that hide B1 without naming it.
  - The excuse was first also applied to `direct_b1_to_t` and B1-named S2 probes, on the theory
    that the model reconstructs B2 from B1. The fresh-eye review of the 22 three-hop survivors
    showed that let direct links through: Indira Gandhi → Maruti 20/20, Yoshitsune → Kamakura
    20/20, Kiichiro Toyoda → loom 16/20.
  - Naming B1 and reaching T is never *excused*. It is evidence that B2 is skippable, so the item
    can only become `relabel_b1` (a 2-hop through B1), and only when B2 is NOT nameable from those
    B1-named hits. Otherwise B1 → [B2] → T still runs through an unscored B2 (Uther → [Arthur] →
    Guinevere), and the item fails. Under the corrected rule, true 3-hop survivors fell from 27
    to 8.
- **Code-review fixes (2026-10-15, before merge; SUPERSEDED by "ID check withdrawn").**
  - The `relabel_b1` guard first asked "which entity is described?" of B1-named probes, and the
    model answered B1, so the guard never fired. It now asks "apart from B1, which entity does it
    refer to?"
  - ID replies are scored on the answer line only. Castro → Che → Argentina had been excused by a
    bridge name in the model's trailing reasoning.
  - Hidden bridges come from the probe text, never from the adversary's own `skips` label.
  - ID rows are versioned `IDv3`.
  - If the generic `drop_b1` / `local_cue` reach T, the clue is not load-bearing, however the
    model got there. Example: "A sixteenth-century astronomer built his observatory on the
    island of" → Hven 20/20.
  - The ID question asks the model to *infer* the described entity. An earlier "list the named
    entities" wording made the model extract surface words ("British suffragette activist,
    1850s") rather than name Pankhurst. The rows are versioned (`IDv2`).
- **Adversary v3: ablations only.** A spot-check of 8 adversary-only kills found half invalid:
  - the probe added facts not in the prompt ("Austrian", "suffragette", "antimalarial");
  - or it asked a different question about T (Baratieri via Coatit).

  Probe keys are versioned (`S2adv3_j`).
- **Relabel (Camila, 2026-10-15: keep as 2-hop).** A 3-hop item whose every real shortcut hides
  only one bridge, while the other bridge's own checks pass, is frozen as an honest 2-hop
  through the load-bearing bridge. Since the second code review only `relabel_b2` remains (see
  "ID check withdrawn"). The decorative clue stays in
  the prompt; only the load-bearing bridge is scored.
  - `relabel_b2` example: Kant → Hume → Edinburgh, where "a skeptic who was born in" works
    without Kant.
  - `relabel_b1` example (SUPERSEDED, `relabel_b1` was withdrawn): Edward I → Edward II →
    Bannockburn, where Edward I links to Bannockburn directly.
- **Measured yield, and the push (Camila, 2026-10-15).** Of the first 628 candidates, 49
  survived (7.8%), but only **3 true 3-hop items of 409 (0.7%)**. The 3-hop losses are mostly
  composition: about 85% of 3-hop prompts score below 16/20, even when every hop is known at
  9–10/20 in isolation. Asked how to proceed, Camila chose a large 3-hop-only push over
  relaxing the gate. The push uses:
  - the "3-hop design rule" generator prompt: `N2 + R2` generic so B2 is reachable only through
    B1; R1 a near-unique famous link; T = B2's best-known fact;
  - 24 domains and 4 parallel generators;
  - the gate run as G1 + recall first, with the adversary only on survivors.
  The bank reports its true-3-hop count beside the target. It is not padded.

- **ID check withdrawn (second code review, 2026-10-15).** Every version of the ID excuse
  leaked, and each patch exposed another hole:
  - it could not tell a unique paraphrase from a class default (Tu Youyou, Piazzolla: both
    confirmed shortcuts at 10/10);
  - it matched bare surnames shared across bridges ("Minamoto");
  - its "apart from B1" question was answered with T itself in about a third of replies.

  So the gate is now strict: every shortcut probe that reaches T fails the item, and
  `relabel_b1` (which needed the ID check to certify that B2 is off the path) is withdrawn. Under
  the strict rule the first-round pools gave 47 items, below the floor, so a further 2-hop round
  was generated.

### Outcome [rev 3]

The frozen bank has **70 items**:
- **6 true 3-hop**, against the plan's target of ≥30;
- 59 two-hop;
- 5 two-hop items relabeled from 3-hop (`relabel_b2` only).

It drew on about 4,500 candidates, about 2,800 of them 3-hop. The gate is strict: no shortcut
probe is ever excused (see "ID check withdrawn"). Fresh-eye reviews of every item recorded 11
overrides, each with its probe evidence, in global-workspace
`scripts/oracle_lens_evals/multihop_v2/review_overrides.json`. Two 2-hop-only top-ups (483 and
654 candidates) brought the bank over the 60-item floor.

The 3-hop shortfall is structural for this model. Qwen3.6-27B answers about 20% of
well-designed 3-hop prompts reliably without chain-of-thought, and in most of those the first
bridge has its own direct link to the answer.

## Seeds

A conversion step turns the audit's 15 items into full v2 items with spans, units, forms
(adding e.g. "Boethius" beside the full name), hops and probes. Each seed is linted and then
run through the **full v2 gate**.

Known outcomes before the gate:
- **Drop the 3-hop seed** (Sholes → Wisconsin → Madison). "Wisconsin" alone cannot be a
  creditable bridge if it is single-token, and inventing "Wisconsin State Senate" leaks prompt
  words and names a different concept.
- **Drop seed 94** (51 Pegasi b → Geneva). Its listed bridge "Michel Mayor and Didier Queloz" is
  compound, and 51 Pegasi b → Geneva may be the real path.
- **Re-word seeds whose target_alts leak** (4, 60, 63, 69) by trimming the leaking alt or
  rewording the prompt, then gate them like any candidate.
- **Keep one of 4/63.** Both are about Adwa 1896.

Seeds get new `mh2-` names, so their ids never collide with v1 readouts.

## Pipeline (global-workspace, `scripts/oracle_lens_evals/multihop_v2/`)

0. **Pilot.** About 100 candidates through the full gate, to measure per-stage yield. The audit
   suggests 3-hop yield is the bottleneck. Feed the failure reasons back into the generator
   prompt, then size the main run. Expected: the main run is several rounds of generation, each
   prompted with the previous round's rejection taxonomy.
1. **`gen.py`** calls `claude-opus-5-5` through `llm_client.async_json` with a strict JSON
   schema.
   - **4 items per call** (v1 overflowed 16k tokens at 12 items per call, and v2 items carry
     about twice the fields). `JUDGE_MAX_TOKENS=32000`, and `None` rows are counted.
   - Cells are 10 domains × {2hop, 3hop}, weighted toward 3-hop.
   - A taken-entities list keeps items diverse.
   - The mechanical lint runs before any GPU time: `lint_hard_item(item, "multihop-hard", ntok)`, plus
     the span, leak-over-alts, isolation, overlap and probe lints above.
2. **`adversary.py`** makes one fresh-context call per lint-clean candidate. It uses the same
   provider, because OpenRouter (the only non-Anthropic route here) is out of budget. This is a
   recorded limitation.
3. **`gate_modal.py`** runs the staged fan-out and writes `outputs/multihop_v2/gate.json`.
4. **`finalize.py`**:
   - keeps every-stage passers;
   - de-duplicates on bridges and targets, true 3-hop items first (descriptor proper nouns: not
  implemented, because no duplicates were found by review);
   - fills the quotas, 3-hop first, ranked by G1 margin minus the worst shortcut score;
   - stamps `probe_token_lens` (`units`, `target`, `target_alts`) with the strict count from
     `make_ntok`;
   - stamps per-item provenance: `consistency` (G1 /20), `bridge_consistency`,
     `hop_consistency`, `prefix_consistency`, and `shortcut_probes` (every S1/S2 probe text with
     its score), so each decision is auditable;
   - writes the header (`family: "multihop-hard"`, `parent_family`, `contract`, `gate`,
     `frozen`);
   - MC options come from `hard_mt_mc_options.py` run on the staged file.

   The bank is written to `evals/oracle_lens/staging_multihop_v2/lens-eval-multihop-hard.json`.
   It is not added to the default hillclimbing set; capture tooling can point at it later.

Cost: about 1,500–2,500 candidates. With 20-sample probes the gate is ≈10k probes, ≈40 min on
32 H200s including the per-stage model loads.

## Bank (workspace-bench)

- **New family `multihop_mt_v2`:**
  - `evals/multihop_mt_v2/{items.json,README.md}`: the frozen copy, with header
    `family: "multihop-hard"` unchanged, as for v1;
  - `src/wsbench/evals/multihop_mt_v2/__init__.py` (`mt_family`);
  - `examples/readouts/multihop_mt_v2.jsonl`;
  - `tests/test_multihop_mt_v2.py` and `tests/golden/multihop_mt_v2_options.json`;
  - `NOTICE.md` lists it among the in-house banks.
- **Registration points** (each must list v2 beside v1):
  - `readplan.py:219` (the final-token family list; otherwise `plan`/`produce` raise KeyError);
  - `capable/questions.py:261` (`_with_bridges`);
  - `baselines/lucky_guessing.py:68` (the description) and `:257` (the builder);
  - the `MT` lists in `tests/test_mt_regex.py` and `tests/test_multitoken_family.py`;
  - the README family entry (three-line invariant, "Basic (multi-token)" group) and the Judges
    row (`README.md:306`).
- **Sanctioned changes to pre-existing tests and goldens.** All are additive, so v1
  expectations are unchanged:
  - `tests/test_mt_regex.py:87` becomes
    `contract.include_target == (name in ("multihop_mt", "multihop_mt_v2"))`;
  - `tests/golden/mt_regex_units.json` gains v2 entries, generated with gw
    `contract.scored_units` and the real `ntok` (the source code, as for v1);
  - `tests/golden/capable_questions.json` gains the v2 builder's key and nothing else;
  - `tests/test_lucky_guessing.py` gets a v2 copy of its v1 `mc` check (`:122-126`), written
    as one loop over both families;
  - **[rev 3]** `tests/test_readplan.py:37`, the registry family count, goes from 27 to 28.
  Any other assertion change to a pre-existing test is out of scope.
- The finalizer refuses a bank where any required bridge lacks an `mc` block.
- **`multihop_mt` is not edited** (frozen banks are never edited in place). Its README gets an
  "Audit (2026-10-15)" section with the 93/100 finding and a pointer to v2. The audit table is
  committed as `evals/multihop_mt/AUDIT_2026-10-15.md`.
- **Macro double-counting.** `results.macro` averages every complete pass-rate family
  (`results.py:127`), so once v2 has readouts, v1 and v2 would both count. The follow-up that
  lands v2 readouts must exclude v1 from the macro, with an exclusion reason "superseded".
  Until then, v2 has no readouts and so is never `complete`, and the macro is unchanged.

## Not in scope

Producing lens readouts for v2 (capture plus every arm's generation). v2 has no numbers until
then.

## Checks

- Each repo's CI line: workspace-bench runs ruff + pytest; global-workspace runs ruff src tests
  workspace-bench, mypy src, pytest, and the lint budget.
- The lint passes with 0 problems on the final bank.
- A fresh-eye review of 20 random final items tries by hand to find a shortcut the probes
  missed. Any hit tightens the adversary prompt and reruns S2.
