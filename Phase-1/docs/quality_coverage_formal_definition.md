# Formal Definition of Quality and Coverage

Status: implementation contract v4, 2026-08-08

This document defines the two Cores that must not be inferred from their names.
It separates the scientific target, runtime-observable evidence, and decision
authority. A mathematical definition does not by itself make a quantity
observable or justify a runtime threshold.

## 1. Shared Boundary

Let `V` be the valid, chunked corpus after Stage A and the Stage-B hard gate.
Let `C` be the final materialized corpus. Curation may quarantine an invalid
unit, remove a nonrepresentative redundancy-family member, decline to select a
chunk that lacks positive Quality support, or retain a Coverage representative.
For Hard, a Coverage representative must also belong to the final Normal
retained set; a Hard-only restoration is forbidden.

The runtime may read text, stable identifiers, normalized structural evidence,
and explicitly declared metadata authorized by a policy card. It may not read
Utility, NLL, benchmark outcomes, a target retention fraction, source
reputation, source tier, or a domain quota. External training and benchmarks
validate a frozen policy after curation; they never reselect the same corpus.

## 2. Quality

### 2.1 Operational definition

Quality is an **evidence-bounded decision vector** over four independent
questions about one training unit. It is not a scalar, a source reputation, a
human preference score, or a claim that usefulness is intrinsic to a document.

```text
QualityEvidence(x) = (Q1(x), Q2(x), Q3(x), Q4(x))
Qi(x) in {pass, fail, abstain}
```

The four coordinates are fixed:

1. `Q1 Correctness Evidence`: locally observable or attached verifier evidence
   supports the payload under its declared context.
2. `Q2 Semantic Coherence`: the parts form a consistent, recoverable semantic
   unit.
3. `Q3 Substantive Payload`: substantive content remains after navigation,
   metadata, boilerplate, or an empty template is excluded.
4. `Q4 Learnable Relations`: at least one recoverable relation exists among
   entities, operations, claims, conditions, or outcomes.

A Policy fails only on its named closed boundary. Missing external knowledge,
undeclared execution assumptions, missing context, specialized notation,
low-confidence output, or out-of-distribution input produces `abstain` and
therefore supplies no positive retention evidence. The runtime never averages
or weights the four decisions. External natural-token training and benchmarks
evaluate a frozen Policy after curation and are not part of
`QualityEvidence(x)`.

### 2.2 Runtime meaning

Q1 first consumes typed declared-verifier evidence when available. A versioned
verifier identity, binary result, and evidence SHA-256 are required; its result
is authoritative and bypasses model judgment. Otherwise, GPT-5.6 Luna supplies
offline Q1-Q4 calibration labels through the Batch API. Those labels train one
frozen local four-head ranker; the external teacher never reads the full corpus
and has no runtime membership authority.

For chunk `x`, let `P(x)` be the set of confident in-distribution heads whose
decision is `pass`, and let `F_m(x)` be the set of qualified fails under mode
`m`. The unweighted selection rule is:

```text
Select_Normal(x) = 1[|P(x)| >= 1 and |F_Normal(x)| = 0]
Select_Hard(x)   = 1[|P(x)| >= 2 and |F_Hard(x)| = 0]
```

Normal and Hard use the same Q1-Q4 Policy family and differ only in the frozen
operating point. A qualified fail always blocks Stage-B selection. `abstain`,
OOD, low confidence, and missing evidence do not count as passes. Stage C may
restore a non-selected chunk only through an explicit Coverage veto followed by
complete rematerialization and recheck.

Shortness, perplexity, source reputation, lexical diversity, weighted formulas,
Utility, NLL, benchmarks, domain quotas, and token budgets are not Policy inputs.

### 2.3 Measurement output

The framework reports all four class-probability vectors, discrete Policy
decisions, confidence and OOD state, passed and failed Policy IDs, required pass
count, frozen ranker identity, reason code, and Coverage outcome. This auditable
vector is the Quality measurement. No `overall_quality_score` is produced.

### 2.4 Quality validation gate

A Quality rule becomes enabled only after all of the following pass:

1. versioned trigger and non-trigger conditions;
2. positive, false-positive, adversarial, and clean-corpus fixtures;
3. reason code, policy hash, original-text hash, and token-delta trace;
4. exact expected behavior on the deterministic 512-task matrix;
5. disjoint protected observations for each head and explicit error analysis of
   both false retention and false non-selection;
6. Coverage invariants and distribution-impact report;
7. benchmark-disjoint external evaluation of the frozen curation result.

The current implementation provides the measurement, positive-selection,
Coverage-veto, and natural-token materialization paths. GPT-5.6 Luna Batch
observations trained the frozen Code-7M ranker candidate. Its behavior fixtures
pass, but scientific promotion remains blocked until the new positive operating
points receive disjoint multidomain and external validation.

## 3. Coverage

### 3.1 Definition

Coverage is **the preservation and auditable accounting of the support of valid
learnable content while local deletion policies transform `V` into `C`**.

Coverage answers: "Did curation erase an entire valid family, route, format, or
tail stratum without an authorized explanation?" It does not answer: "Is this
chunk good?" It also does not prescribe a desired Code/Math/General percentage.

Coverage is a corpus-level constraint Core. Redundancy proposes family
compaction; Quality proposes positive membership decisions; Coverage verifies
that their combined materialization has not produced an orphaned or unexplained
loss. It has veto-only authority: it may abort materialization or return typed
`required_retain_uids` for explicit rematerialization and a complete second
Coverage check. It may not rank, delete, quota-select, or silently restore
records.

### 3.2 Coverage universe

Coverage is evaluated over multiple nonexclusive views of `V`:

- redundancy families defined by the active exact or symmetric policy;
- content-route labels with `unknown` retained as a valid label;
- language/script, format, and structural-family labels;
- semantic support strata built from reciprocal-neighbor evidence shared by a
  frozen primary embedding provider and an independently frozen audit provider;
- transformed chunks and their residual payload links.

Source identity is an optional audit dimension, not a selection axis. Rare does
not imply useful, and common does not imply removable.

The v3 candidate uses mutual-kNN graphs from both providers. Shared reciprocal
edges form stable local support groups; provider-specific neighborhoods form
overlapping uncertainty groups. Stage C restores a deterministic representative
only when a group would otherwise be extinct. Singletons, unknown routing, and
unsupported tags remain explicitly represented. Embedding similarity alone
never authorizes deletion. Normal and Hard use exactly the same Coverage
invariants; their difference remains limited to Stage-B non-selection proposals and
span transformations.

### 3.3 Mandatory runtime invariants

For each removal family `f`, let `S_f` be its members and `C_f = S_f intersect C`.
The active invariant is:

```text
I_rep(f) = 1 if |C_f| >= 1
           1 if every member has an explicit authorized non-payload reason
           0 otherwise
```

For each span transformation `t`, let `residual(t)` be the materialized text:

```text
I_res(t) = 1 if the residual preserves the chunk identity and satisfies the
           declared minimum residual contract; 0 otherwise
```

Runtime materialization succeeds only when:

```text
min_f I_rep(f) = 1 and min_t I_res(t) = 1
```

These are non-compensatory constraints. Strong performance in another stratum
cannot cancel a missing representative.

When Redundancy supplies a directional representative, Coverage preserves that
representative if it is eligible. Facility-location selection is only a
deterministic fallback. A veto must identify every required retained unit,
rematerialize explicitly, and rerun the complete invariant set. The second
decision must pass; hidden restoration is forbidden and audited as false.

### 3.4 Distribution audit

For each declared audit axis `a`, compare token-incidence distributions
`P_a^V` and `P_a^C`. Report, but do not optimize:

```text
JSD_a = JSD(P_a^V || P_a^C) / ln(2)
Support_a = |{z: P_a^C(z) > 0}| / |{z: P_a^V(z) > 0}|
Tail_a = sum_{z in tail_a} w_z * 1[P_a^C(z) > 0] / sum_{z in tail_a} w_z
```

Also report orphan-family rate, unexplained zero-survivor rate, and residual
transformation failure rate. The authoritative output is a vector, not a
weighted average:

```text
CoverageVector = (
  1 - orphan_family_rate,
  1 - unexplained_zero_survivor_rate,
  1 - residual_failure_rate,
  1 - JSD_a for every axis,
  Support_a for every axis,
  Tail_a for every axis
)
```

If a single display number is unavoidable, use the bottleneck value, not a
compensating weighted sum:

```text
Coverage_min = min(normalized components of CoverageVector)
```

The runtime also emits `composition_audit.json`, `composition_by_route.csv`,
`composition_by_language.csv`, and `eligible_curated_composition_delta.csv`.
The comparable delta uses Stage-B eligible chunks and Stage-C curated chunks,
so both sides share the same chunk unit and immutable chunk IDs. Raw and Stage-A
record-level composition remains descriptive only. Record-to-chunk deltas and
divergences are not emitted; those stages are named under
`excluded_cross_unit_deltas` instead. Primary route share is
exclusive; route and language/script incidence are multi-label and may sum
above 100%. These artifacts are never target distributions or selector inputs.

No distributional threshold has runtime veto authority until its taxonomy,
unknown handling, confidence interval, false-positive fixtures, and development
ablation are frozen. The implemented semantic v3 candidate is limited to
representative linkage, provider-agreed stable strata, zero-survivor
explanation, and residual-payload integrity. Its model providers remain
unpromoted until protected false-veto and independent multilingual and
multidomain confirmatory gates pass. The current primary provider is a
development/confirmatory runtime experiment and the second provider remains an
audit view.

### 3.5 Boundary with the other Cores

| Core | Primary question | May exclude? | May veto output? |
|---|---|---:|---:|
| Validity | Is the unit structurally usable under the declared input contract? | Yes, by hard reason | No |
| Redundancy | Is the information already represented by a linked family member? | Yes, while retaining a representative | No |
| Quality | Does the unit meet the positive Q1-Q4 membership gate? | Yes, by reason-coded non-selection | No |
| Coverage | Did the combined removals create an unexplained support failure? | No | Yes |

## 4. Stage Placement

1. Stage A applies Validity and quarantines closed observable failures.
2. Stage B applies Redundancy and promoted Quality Policies to create
   reason-coded non-selection proposals.
3. Stage C applies Coverage invariants and may veto unexplained support loss
   before writing the curated corpus.

Coverage is therefore part of curation even though it does not delete data. It
is the final validity condition for the corpus-level result.

## 5. Current Claim

The implementation may claim an auditable, domain-agnostic curation interface,
an executable Q1-Q4 Quality measurement protocol, and an implemented Semantic
Coverage qualification candidate. It may not claim that either model-driven
Core is scientifically promoted, production-ready, or effective in every
domain. Those claims require completed protected-fixture, multilingual
provider-stability, corpus-scale ANN, and independent external evidence under
the gates above.
