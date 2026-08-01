# Target-Aware Core Completion Plan

Status: design-only contract, 2026-08-02
Frozen implementation baseline: `ed87fe8`
Frozen tag: `conservative-structural-baseline-v1`

This document defines the work that must be agreed before the frozen runtime is
changed. It does not activate a policy, alter a curation output, or authorize a
new confirmatory experiment.

## 1. Achievable Goal

The framework cannot guarantee a universally perfect dataset. A document has no
model-independent, task-independent scalar Quality: its value changes with the
base model, tokenizer, context window, current corpus, training objective, and
target capability distribution.

The implementable objective is:

> Given a declared continued-pretraining model and a raw corpus, materialize a
> smaller corpus that removes unusable or repeated payload, prioritizes expected
> marginal learning value per training token, and preserves auditable semantic
> support, without reading downstream benchmark outcomes at runtime.

The scientific success condition is not a fixed compression ratio. It is that a
frozen policy uses fewer natural training tokens than Raw while preserving or
improving preregistered external capabilities. A clean input may change little;
a corrupted, repetitive, or low-value input should change substantially more.

## 2. Evidence From Prior Work

The redesign follows results that separate filtering, model-relative selection,
deduplication, and distribution control rather than combining them into an
unvalidated document score.

| Evidence | Design implication |
| --- | --- |
| [DataComp-LM](https://arxiv.org/abs/2406.11794) reports that controlled model-based filtering is central to its strongest baseline and evaluates data recipes through downstream training. | Quality candidates need model-based evidence and external policy-level validation. |
| [Rho-1](https://arxiv.org/abs/2404.07965) uses reference-model excess loss to identify tokens that are difficult for the training model but aligned with a desired distribution. | Base loss alone is not Quality; base-versus-reference excess loss is a candidate learnability signal. |
| [D4](https://arxiv.org/abs/2308.12284) combines deduplication with embedding-based diversification and reports efficiency and downstream gains. | Redundancy and Coverage must interact at subset level; lexical deduplication alone is insufficient. |
| [SemDeDup](https://arxiv.org/abs/2303.09540) identifies semantic duplicates through pretrained embeddings. | A semantic relation is a separate Redundancy candidate, with independent-solution false-positive tests. |
| [DoReMi](https://arxiv.org/abs/2305.10429) shows that data-mixture weights affect model performance and can be learned using proxy models. | Corpus composition matters, but inferred groups are audit/optimization strata rather than fixed human quotas. |
| [DSIR](https://arxiv.org/abs/2302.03169) selects data to match a declared target distribution and validates a data-space metric against downstream accuracy. | A target distribution must be declared when target alignment is claimed; no universal target can be inferred from Raw alone. |
| [FineWeb](https://arxiv.org/abs/2406.17557) and [Dolma](https://arxiv.org/abs/2402.00159) publish filtering ablations and intermediate corpus analyses. | Every policy must be evaluated independently and corpus changes must be fully reported. |
| [RefinedWeb](https://arxiv.org/abs/2306.01116) shows that carefully filtered and deduplicated web data can support strong models. | Source identity is not a Quality verdict; observable payload and measured effects are the relevant evidence. |

These works support the direction of the redesign. They do not validate this
project's thresholds or prove that one method transfers to every domain.

## 3. Dataset Characteristics That Must Be Measured

The profiler is audit-only. It describes the opportunity and risk in an input
corpus before any deletion decision.

### 3.1 Acquisition and unit integrity

- input field ambiguity, missing payload, decoding and encoding failures;
- binary or control-character contamination;
- extraction artifacts, truncation declarations, and malformed containers;
- document, file, page, conversation, table, and snippet granularity;
- context completeness declarations when available.

### 3.2 Repetition structure

- exact normalized duplicates;
- lexical near duplicates;
- subset/superset containment;
- repeated templates or spans across otherwise distinct records;
- semantic duplicate candidates;
- family size and token mass, not only pair counts.

### 3.3 Content support

- content morphology and route: prose, code, mathematics, technical
  documentation, conversation, instruction, structured/table data, mixed, and
  unknown;
- language and script;
- format and structural family;
- semantic or skill clusters from a frozen encoder;
- cluster density, tail mass, and boundary uncertainty.

These labels describe support. They never imply that Code, Math, English, or a
particular source is intrinsically better.

### 3.4 Training-interface fit

- exact tokens under the declared target tokenizer;
- tokenizer fertility by language, format, and cluster;
- sequence-length and context-window fit;
- payload lost to packing boundaries or overlong truncation;
- base-model and reference-model loss distributions when model-relative
  Quality is enabled.

### 3.5 Corpus provenance and risk

Collection time, source identity, rights, PII, secrets, poisoning indicators,
and benchmark contamination remain auditable sidecar dimensions. They must not
be converted into Quality scores. A production safety layer may quarantine
them under a separate declared risk contract, but source reputation never
authorizes LM-usefulness deletion.

## 4. Language-Model Characteristics That Must Be Frozen

Quality cannot be implemented until the following target declaration is
immutable for one experiment:

1. training interface: continued pretraining only;
2. base checkpoint and publication/training cutoff;
3. tokenizer revision and context length;
4. packing, optimizer, learning-rate schedule, and number of epochs/passes;
5. natural-token accounting rule;
6. target capability distributions and benchmark exclusion hashes;
7. reference model, if used, with compatible likelihood normalization;
8. seed set and compute environment.

The framework may remain domain-general at its interface while a Quality policy
is model-relative. This distinction must be explicit:

- **Framework generality:** accepts the same canonical contract and emits the
  same evidence schema across corpora.
- **Policy scope:** the model family, training objective, and routes on which a
  policy was calibrated.
- **Evidence scope:** the domains and external suites on which downstream
  behavior was confirmed.

## 5. Final Core Definitions and Authority

| Core | Question | Unit | Authority | Forbidden shortcut |
| --- | --- | --- | --- | --- |
| Validity | Can this unit be interpreted under the declared input and training contract without inventing or destroying payload? | record/chunk | normalize reversibly, quarantine, or hard reject | generic parser failure, shortness, unfamiliar format |
| Redundancy | Is this payload already represented by another unit, and what is the duplicate relation? | pair/family/span | form families and compact only proven repeated payload | one global similarity threshold |
| Quality | What is the expected marginal learning contribution per target-model training token? | token/chunk, conditional on corpus/model | provide calibrated keep/reject/uncertain evidence | intrinsic score, source reputation, handwritten weighted sum |
| Coverage | Does the selected subset preserve the valid semantic and structural support needed to avoid capability collapse? | subset | constrain/veto selection and choose representatives | fixed domain percentages or rarity-is-quality |

Validity is non-compensatory. Redundancy defines equivalence or containment.
Quality estimates learning value. Coverage evaluates interactions among retained
units. No strong Quality score can repair invalid text, and no common content
may be deleted merely because it is common.

## 6. Validity v2 Design

### 6.1 Closed decisions

Validity emits one of `valid`, `valid_after_reversible_repair`, `quarantine`, or
`invalid`. Every non-valid decision carries original bytes/text hash, observed
evidence, transformation trace, and reason code.

### 6.2 Hard-invalid conditions

- no recoverable textual payload;
- ambiguous populated input fields under the canonical contract;
- unrecoverable decoding failure or binary payload presented as text;
- forbidden control-character domination or corruption after bounded repair;
- a declared complete container that cannot be materialized without data loss;
- a chunking result with no usable residual.

Partial files, snippets, equations, tables, JSON, multilingual text, uncommon
formats, and parser errors are retained unless their contract explicitly
declares a complete artifact and a compatible parser/version.

### 6.3 Validation

- synthetic corruption fixtures with known ground truth;
- reversible metamorphic pairs;
- clean controls from Code, Math, General, multilingual, tables, and snippets;
- adversarial cases that resemble corruption but carry payload;
- per-reason false-positive and false-negative confidence intervals;
- exact round-trip and token-delta accounting.

Hard deletion is not promoted while the preregistered clean-control false-
positive bound is exceeded.

## 7. Redundancy v2 Design

### 7.1 Relation taxonomy

Redundancy is a typed graph, not a scalar:

1. `exact_equivalent`;
2. `formatting_equivalent`;
3. `near_substitute`;
4. `contained_payload` or `superset_payload`;
5. `repeated_span`;
6. `semantic_duplicate_candidate`;
7. `related_complementary`, which must be retained as nonredundant.

### 7.2 Length-sensitive near duplication

A single global overlap threshold cannot distinguish one changed word in a
short statement from one changed word in a long template. Candidate relations
therefore combine:

- absolute changed-token count;
- changed-token ratio;
- bidirectional containment;
- normalized edit operations;
- semantic similarity;
- route-aware difference payload checks.

Short units default to exact-only compaction unless the changed content is
proven formatting. Long units can be compacted when the residual difference is
both small and non-substantive under a validated relation policy. Code
identifiers, numeric constants, mathematical operators, negation, named
entities, API signatures, and answer labels are substantive differences.

### 7.3 Family and representative contract

- candidate pairs are retrieved with MinHash/LSH and frozen embedding ANN;
- relation classification forms a versioned graph with no silent transitive
  assumption across incompatible relation types;
- each removal links transitively to a final surviving representative;
- the final representative is chosen after Quality and Coverage evidence, not
  merely because it appeared first;
- repeated spans are removed at span level when a valid residual remains;
- independent implementations, test matrices, API documentation, and
  complementary explanations are explicit false-positive families.

## 8. Quality v2 Design

### 8.1 Latent target

For target evaluation distribution `T`, current corpus `D`, model initialization
`m`, seed `s`, and exact target-token count `tau(x)`:

```text
Q_T(x | D) = E_m,s[
  (Risk_T(theta(D; m,s)) - Risk_T(theta(D plus x; m,s))) / tau(x)
]
```

This is the definition, not a directly observable runtime value. Leave-one-out
training for every chunk is infeasible, so the runtime uses a calibrated
estimator and the external experiment validates only the frozen policy.

### 8.2 Evidence layers

Quality evidence is deliberately non-compensatory by layer:

1. **Explicit negative payload:** generated-control artifacts, empty shells,
   control-only chrome, and other closed structural proofs.
2. **Model-relative learnability:** target-base loss, stronger-reference loss,
   and excess loss normalized to a common unit. High base loss alone is not
   good; high loss for both models may indicate irreducible noise. Positive
   base-minus-reference excess is a candidate signal that the reference can
   model content the target base has not mastered.
3. **Training-interface fit:** usable tokens after packing, context completeness,
   and nontruncated payload.
4. **Policy-level causal evidence:** rule-on/off proxy training and disjoint
   natural-budget external evaluation.

No lexical-diversity formula, source score, or manual weighted sum may replace
these layers.

### 8.3 Calibrated decision

The estimator predicts marginal gain with uncertainty for route `r`:

```text
(gain_hat_r(x), LCB_r(x), UCB_r(x))
```

Route is a conditioning variable, not Quality evidence. Mixed, unknown, and
out-of-distribution routes are explicit states.

- frozen Conservative Structural keeps its current closed negative proofs;
- Normal rejects only when the estimated upper
  confidence bound is nonpositive and Coverage permits removal;
- Hard may compare expected gain with a declared per-token
  training cost, but only after an independently confirmed Pareto study;
- uncertain units are reported separately and retained until evidence supports
  a stricter policy.

There is no target retention fraction. A per-token cost, if used, is derived
from the declared training compute and preregistered risk tolerance, not chosen
to obtain an attractive compression number.

### 8.4 Calibration without subjective labels

- create perturbation pairs with known removed information;
- use explicit structural artifacts as negative anchors;
- use source-disjoint clean controls only as retention controls, not as labels
  that every member is high Quality;
- partition examples into frozen evidence bins and run small proxy-training
  ablations;
- fit a monotonic calibrated estimator to measured bin-level marginal effects;
- reserve human review for error analysis, never sole promotion authority;
- test whether score bins monotonically predict held-out marginal gain and
  report calibration error and uncertainty.

## 9. Coverage v2 Design

### 9.1 Definition

Coverage is preservation of the support of valid, distinct, potentially
learnable content under subset selection. It is not a preferred domain mix and
not a synonym for diversity.

### 9.2 Views

- redundancy-family support;
- deterministic route, language/script, format, and morphology views;
- semantic/skill clusters from a frozen encoder;
- unknown, mixed, and low-confidence strata;
- intersections that expose hidden collapse, such as language by content route.

Source is audit-only. No route receives a fixed percentage.

### 9.3 Cluster contract

- freeze encoder, revision, normalization, distance metric, ANN index, and
  clustering algorithm;
- estimate cluster stability through bootstrap/resampling;
- call only stable clusters selection strata;
- treat unstable points as an explicit uncertain stratum rather than noise;
- audit encoder bias with multilingual, code, mathematics, tables, and short
  text perturbations.

### 9.4 Selection role

Coverage supplies a marginal support gain, for example a facility-location
gain over the frozen corpus representation:

```text
G_cov(x | S) = F(S union {x}) - F(S)
```

It also imposes hard constraints:

- each compacted redundancy family has a final representative;
- no stable supported cluster reaches zero survivors unless every member has an
  independent Validity or confirmed negative-Quality reason;
- transformed chunks retain a valid residual;
- unknown/mixed/OOD support is not silently erased.

Coverage may select the best representative or veto a deletion. It does not
declare rare content useful, restore invalid content, or remove common content.

### 9.5 Report

Report vectors rather than one score: support recall, token-mass shift, Jensen-
Shannon divergence, tail survival, cluster extinction, nearest-representative
radius, effective sample size, and representative linkage. Compression and
coverage change are always separate columns.

## 10. Joint Selection Contract

The final selector operates in this order:

```text
Stage A: canonical adaptation and Validity hard gate
Stage B: chunking and typed Redundancy family construction
Stage C: Quality evidence + Coverage-constrained representative/materialization
```

The conceptual constrained problem is:

```text
maximize   expected model-relative learning gain of S
minimize   training-token cost and redundant payload
subject to Validity, representative, residual, and Coverage constraints
```

Implementation must not hide this in a handwritten weighted sum. Candidate
policies are compared on a Pareto frontier of natural tokens, estimated gain,
coverage loss, and external performance. One operating point is preregistered
before confirmatory evaluation. Benchmarks never enter a corpus-selection run.

## 11. Implementation Blocks

### Block 0 - Baseline freeze - COMPLETE

- commit: `ed87fe8`;
- tag: `conservative-structural-baseline-v1`;
- all 120 directly runnable validations passed;
- no runtime redesign is included in this block.

### Block 1 - Research and target contract - COMPLETE

- approve this Core definition;
- freeze base model, tokenizer, cutoff, context, training objective, reference
  model, target capability distributions, seed set, and natural-token rule;
- preregister candidate-development and confirmatory separation;
- decide whether production risk screening is a separate pre-curation layer.

Exit: one immutable target-model experiment contract and zero ambiguous terms.

Frozen artifact: `protocols/target_aware_core_completion_v1.json`. The target
is Qwen3-4B-Base continued pretraining, the stronger-reference candidate is
Qwen3-8B-Base, the Coverage encoder candidate is Qwen3-Embedding-0.6B, and the
seed set is `101/202/303`. The model cutoff remains explicitly unknown, so
temporal-new-knowledge and post-cutoff LiveCodeBench claims remain blocked.

### Block 2 - Audit-only corpus profiler

- implement all Dataset measurements in Section 3;
- produce exact tokenizer counts, opportunity rates, route/cluster uncertainty,
  and no selection decisions;
- validate deterministic hashes and bounded memory use.

Exit: the same profiler runs unchanged on Code, Math, General, multilingual,
and mixed corpora.

### Block 3 - Validity v2

- implement the closed four-way decision and reversible repairs;
- build cross-domain positive, negative, metamorphic, and adversarial fixtures;
- publish per-reason error bounds and quarantine traces.

Exit: all hard gates meet preregistered clean-control false-positive bounds.

### Block 4 - Redundancy v2

- implement typed relation retrieval, classification, family graph, and span
  compaction;
- replace the single-threshold assumption with length- and difference-aware
  policies;
- defer final representative choice to Stage C.

Exit: each relation passes independent precision, false-positive, family-link,
and representative-survival gates.

### Block 5 - Quality evidence engine

- implement frozen base/reference scoring and common-unit normalization;
- build evidence bins and low-cost development ablations;
- calibrate gain and uncertainty without a global handwritten formula;
- retain mixed, unknown, and OOD cases by explicit state.

Exit: held-out score bins predict measured marginal effect monotonically within
their confidence bounds; failed routes remain unsupported rather than borrowed.

### Block 6 - Coverage engine

- implement frozen semantic/skill support views and stability tests;
- add facility-location gain, extinction guards, representative choice, and
  vector reports;
- stress test encoder and cluster bias.

Exit: no unexplained stable-support extinction, orphan family, or residual loss.

### Block 7 - Joint selector and immutable profiles

- preserve the frozen Conservative Structural profile unchanged;
- implement development-only Normal and Hard target-aware profiles over the
  same Base input;
- emit complete policy/evidence/model hashes and per-removal traces;
- prohibit benchmark, source reputation, target fraction, and post-run override.

Exit: deterministic replay and policy-leakage tests pass.

### Block 8 - Development selection

- use record-, hash-, source-, and time-disjoint Code, Math, and General
  development corpora;
- include clean, duplicate-heavy, malformed, boilerplate-heavy, and mixed
  raw-like scenarios;
- run Core and rule ablations and select one Pareto operating point;
- freeze all thresholds before confirmatory materialization.

Exit: one frozen Normal profile, one frozen Hard profile, their immutable
hashes, and no result-dependent tuning path remain.

### Block 9 - Confirmatory external evaluation

- materialize Base, Normal, and Hard dataset arms;
- train each on its own natural token count with identical non-data settings and
  seeds `101/202/303`;
- evaluate preregistered domain suites plus general-capability retention;
- estimate uncertainty from seeds and use a non-inferiority margin derived and
  frozen from development measurement noise.

Exit: Normal and Hard token deltas are reported against Base and each profile's
preregistered downstream criterion passes. A failed domain limits the policy
claim rather than being hidden by an average.

### Block 10 - Release and paper

- promote only confirmed policies;
- publish configs, hashes, traces, ablations, failures, token deltas, coverage
  vectors, and seed-level outcomes;
- separate Framework, Policy, and Evidence claims;
- retain Conservative Structural as the fail-safe profile.

Exit: release artifacts reproduce the reported corpus and every paper claim is
linked to a frozen evidence bundle.

## 12. Non-Negotiable Acceptance Gates

1. No fixed retention percentage or hidden token cap.
2. No benchmark, NLL result, Utility, or source reputation in a runtime
   selection decision.
3. Every removal has one owning Core, policy/version, evidence, original hash,
   exact token delta, and final representative or non-payload reason.
4. Every model-based score identifies model, revision, tokenizer, normalization,
   and calibration scope.
5. Clean-control false positives, not only compression, are reported with
   confidence intervals.
6. Dataset shift is visible: clean inputs may be retained almost entirely,
   while degraded inputs must expose higher rule opportunity.
7. External confirmation uses natural budgets and at least three seeds.
8. Code evidence supports a Code claim only; domain-general downstream claims
   require independent Math and General confirmation.
9. A failed Core or route abstains or falls back to Conservative Structural.
10. The paper never claims universal intrinsic Quality or guaranteed
    improvement on every corpus.

## 13. Block 1 Resolution

Block 1 froze the target SLM and tokenizer, the stronger-reference candidate,
the Coverage encoder candidate, and the capability panels in
`protocols/target_aware_core_completion_v1.json`. The model's auditable
pretraining cutoff remains unknown by explicit contract, so temporal-new-data
claims and post-cutoff LiveCodeBench use remain disabled.

Block 2 is now authorized to implement an audit-only corpus profiler. It may
measure the characteristics in Section 3, but it may not rank, select, delete,
or change the frozen Conservative Structural runtime.
