# Code-Domain Next Development Cycle Design

## Status

This document designs a new development cycle after the completed Qwen3-4B
code-domain confirmatory experiment. It does not revise, rescue, or reinterpret
the completed v1 confirmatory result.

Current locked v1 result:

- Confirmatory status: `confirmatory_decision_reject_primary_margin_failure`.
- Interpretation: negative primary-margin result with a positive directional
  curation signal.
- Frozen margin: `0.005`.
- Observed curated vs Stage-A-random NLL reduction:
  `0.0037666601293226964`.
- Gap to margin: `0.0012333398706773037`.

The next cycle is a separate v2 development cycle. It may use the v1
postmortem as diagnosis, but it must have fresh development decisions and a
newly frozen untouched confirmatory protocol before any v2 confirmatory model
outcomes are read.

## Claim

The practical framework target is not to prove intrinsic data quality. The
target is an operational curation decision:

```text
For a pre-registered raw-code continued-pretraining setting, the framework uses
frozen pre-outcome selection proxies to produce either a curated subset that
improves a small language-model update over an equal-budget random usable-data
baseline while satisfying guardrails, or an explicit abstention/rejection when
the evidence does not support release.
```

The completed v1 evidence does not yet support that claim. It supports a
weaker statement:

```text
The current code-domain recipe produced a directional NLL improvement over
Stage-A-random and raw-random on all confirmatory seeds, but failed the frozen
practical margin.
```

## Non-Negotiable Boundaries

- Core-Metric-Policy structure stays intact.
- Stage A remains a chunk-level hard gate.
- Stage B remains chunk-level selection among usable chunks.
- Stage C remains subset/model validation.
- Utility, benchmark outcomes, retention outcomes, human review labels, and
  LLM review labels must never enter the Stage-B selector objective.
- Confirmatory outcomes must never be used to tune the selector, margin, seed
  set, heldout slice, token budget, or guardrail thresholds for the same cycle.
- Every Utility sensitivity arm must share one common disjoint Stage-A baseline
  pool.

## Diagnosis To Design Translation

The v1 postmortem found three design problems that the next cycle must address:

1. Development-to-confirmatory effect shrinkage:
   development reduction was `0.011155656973521166`, confirmatory reduction was
   `0.0037666601293226964`, and the retained ratio was about `0.3376`.
2. Heldout shift:
   development used 175 records from 7 repositories with test ratio `0.2857`;
   confirmatory used 110 records from 5 repositories with test ratio `0.4182`.
3. Margin calibration:
   an absolute `0.005` NLL margin was too brittle across a lower-base-NLL
   confirmatory split. The next margin must be calibrated before confirmatory
   outcomes using only the new development cycle.

## v2 Design

### Candidate Pool And Splits

Use raw-like Python code data with provenance, license, contamination, and
content-type metadata. Splits must be both time-disjoint and
repository-disjoint.

Minimum design constraints:

- Train split: at least 30 repositories after Stage A.
- Development heldout: at least 10 repositories after Stage A.
- Confirmatory heldout: at least 10 repositories after Stage A.
- No repository may contribute more than 25% of selected training tokens or
  heldout tokens.
- Development and confirmatory heldouts must each target at least 65k
  token-proxy units, with 128k preferred when available.
- Development and confirmatory code/test ratios must be stratified. The
  maximum allowed absolute test-ratio difference before freeze is `0.05`.
- Heldout profiles must report repository counts, top-repository shares,
  content-type ratios, chunk-kind ratios, token-proxy distribution, and base
  NLL scale before any promotion decision.

If these constraints cannot be met, the correct result is
`insufficient_usable_data` or a scoped smaller claim, not forced selection.

### Stage A

Stage A remains a hard usability gate:

- independent parseability for Python chunks
- exact duplicate rejection
- canonical exact-duplicate rejection; fuzzy near-duplicates are Stage-B signals
- pathological repetition rejection
- structural and policy quarantine inherited from Stage 0

Stage A must not score semantic usefulness or training Utility.

### Stage B

Stage B should be strengthened with code-local Core proxies, still without
Utility leakage:

- AST granularity and independently parseable code units
- test/code balance support
- import/API meaningfulness
- concise example, bug-fix, and regression-test support
- docstring/comment density bounds
- generated, template, vendored, and boilerplate risk
- harmful soft redundancy and near-template saturation
- useful recurrence and cluster coverage support
- repository/path diversity caps
- length and packing support

The selector must not treat AST richness, length, or apparent complexity as
quality by themselves. Operational readiness requires feature reports showing
that concise but useful examples, tests, bug fixes, and API-usage chunks are
not systematically removed merely because they are short or structurally simple.

Required ablations:

- full selector
- quality-only
- redundancy-only
- no-coverage-support
- no-test-code-balance
- no-repository-diversity-cap
- Stage-A-random equal-budget
- raw-random equal-budget

Ablations explain mechanism. They cannot be selected post hoc from
confirmatory outcomes.

### Stage C

Primary v2 Stage-C evidence remains equal-token, equal-compute QLoRA continued
pretraining:

- base model
- raw-random equal-budget
- Stage-A-random equal-budget
- curated-v2 equal-budget
- known-high-quality equal-budget reference

Optional supporting arms:

- all-raw compute-efficiency reference
- all-Stage-A compute-efficiency reference
- curated-v2 plus frozen replay, if retention needs it

The primary comparison is curated-v2 versus Stage-A-random. Raw-random,
base-no-update, known-high-quality, and all-data arms are supporting context
unless separately frozen as primary.

### Margin And Power Calibration

The next cycle must separate development calibration from confirmatory testing.

Development-only calibration must report:

- paired seed deltas for curated-v2 minus Stage-A-random
- seed-level variance
- detectable-effect floor
- base-NLL scale
- heldout stratification profile
- whether the observed development effect is large enough to justify a
  confirmatory run

Before v2 confirmatory outcomes are read, freeze exactly one primary success
rule. The rule may use an absolute margin, a relative NLL margin, or a
stratified margin, but the choice must be documented before confirmatory
training/evaluation outcomes.

The v1 confirmatory result may justify why calibration is needed. It must not
be used to retrofit the v1 margin or to tune v2 after v2 confirmatory outcomes.

### Promotion Rule

Promote a v2 recipe from development to confirmatory only if all are true:

- curated-v2 beats Stage-A-random on the development primary metric by the
  frozen development margin or calibrated detectable-effect rule
- curated-v2 is directionally no worse than raw-random
- Stage-C code guardrails do not regress beyond frozen limits
- general retention evidence is present and passes
- heldout stratification constraints pass
- Stage-A-random hardness diagnostics are recorded
- all training arms completed the required seed set or the report remains a
  feasibility-only partial result

### Confirmatory Rule

Before v2 confirmatory training outcomes are read, freeze:

- model and tokenizer identity
- arms
- token budget and optimizer steps
- seed set
- heldout slice and hash
- primary metric and margin
- code guardrail thresholds
- retention guardrail thresholds
- contamination audit
- common-disjoint Stage-A baseline design for sensitivity arms

Confirmatory failure is a valid outcome. It must be reported as a result, not
repaired by changing the same protocol.

## Execution Order

1. Freeze this design as a draft contract, not as an executable confirmatory
   protocol.
2. Expand or rebuild the raw-like Python candidate pool until split and
   stratification requirements can be checked.
3. Run Stage 0 and Stage A audits.
4. Generate Stage-B v2 arms and required ablations.
5. Build development heldouts and baseline-hardness diagnostics.
6. Run development QLoRA arms.
7. Calibrate the v2 margin using development-only evidence.
8. If promotion passes, freeze a new untouched confirmatory protocol.
9. Run confirmatory training/evaluation.
10. Build the final decision report and paper evidence table, including
    negative or abstention outcomes.

## Paper Interpretation

The paper should present v1 as useful negative evidence: the system produced a
stable directional signal, but the predeclared practical margin did not hold
under distribution shift. The v2 cycle tests whether better split
stratification, development-only margin calibration, and stronger non-Utility
code proxies produce a confirmatory-grade result.
