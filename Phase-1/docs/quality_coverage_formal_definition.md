# Formal Definition of Quality and Coverage

Status: design contract v2, 2026-08-04

This document defines the two Cores that must not be inferred from their names.
It separates the scientific target, runtime-observable evidence, and decision
authority. A mathematical definition does not by itself make a quantity
observable or justify a runtime threshold.

## 1. Shared Boundary

Let `V` be the valid, chunked corpus after Stage A and the Stage-B hard gate.
Let `C` be the final materialized corpus. Curation may remove a whole chunk,
retain one representative of a redundancy family, or remove a separable span.

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
undeclared execution assumptions, missing context, or specialized notation
produce `abstain`, not deletion. The runtime never averages the four decisions.
External natural-budget training and benchmarks evaluate a frozen Policy after
curation and are not part of `QualityEvidence(x)`.

### 2.2 Runtime meaning

Q1 first consumes typed declared-verifier evidence when available. A versioned
verifier identity, binary result, and evidence SHA-256 are required; its result
is authoritative and bypasses teacher generation. Without that evidence, three
frozen teacher organizations evaluate Q1. Q2-Q4 always use those teachers. A
first-pass 3-of-3 decision is accepted. A 2-of-3 result is accepted only when a
blinded second pass preserves the decision and at least two of the same
teachers. Invalid output and teacher unavailability abstain.

Normal and Hard use identical Policy semantics:

| Mode | Stage-B removal proposal |
|---|---|
| Normal | At least one Policy has a first-pass unanimous FAIL |
| Hard | At least one Policy has a unanimous FAIL or stable repeated 2-of-3 FAIL |

All other outcomes retain. Stage C may veto a removal to preserve Coverage.
Teacher output has no authority before the fixture and false-removal gates pass.
Shortness, perplexity, source reputation, lexical diversity, weighted formulas,
Utility, NLL, benchmarks, domain quotas, and token budgets are not Policy inputs.

### 2.3 Measurement output

The framework reports the four Policy decisions, teacher votes, closed reason
codes, blinded-pass stability, model identities, response hashes, transport
status, and Coverage outcome. This auditable vector is the Quality measurement.
No `overall_quality_score` is produced.

### 2.4 Quality validation gate

A Quality rule becomes enabled only after all of the following pass:

1. versioned trigger and non-trigger conditions;
2. positive, false-positive, adversarial, and clean-corpus fixtures;
3. reason code, policy hash, original-text hash, and token-delta trace;
4. exact expected behavior on the deterministic 512-task matrix;
5. at least 800 protected fixtures with a one-sided 95% false-removal upper
   bound at most 0.5% for Normal or 2.0% for Hard;
6. Coverage invariants and distribution-impact report;
7. benchmark-disjoint external evaluation of the frozen curation result.

The current implementation provides the complete measurement and qualification
path, including observation-schema isolation and Q1 verifier precedence, but
runtime activation remains blocked until the complete observations pass.

## 3. Coverage

### 3.1 Definition

Coverage is **the preservation and auditable accounting of the support of valid
learnable content while local deletion policies transform `V` into `C`**.

Coverage answers: "Did curation erase an entire valid family, route, format, or
tail stratum without an authorized explanation?" It does not answer: "Is this
chunk good?" It also does not prescribe a desired Code/Math/General percentage.

Coverage is a corpus-level constraint Core. Redundancy proposes family
compaction; Quality proposes explicit non-payload removal; Coverage verifies
that their combined materialization has not produced an orphaned or unexplained
loss. It has veto-only authority: it may abort materialization, but it may not
rank, delete, quota-select, or silently restore records.

### 3.2 Coverage universe

Coverage is evaluated over multiple nonexclusive views of `V`:

- redundancy families defined by the active exact or symmetric policy;
- content-route labels with `unknown` retained as a valid label;
- language/script, format, and structural-family labels;
- semantic clusters only when their model, snapshot, and clustering contract
  are frozen;
- transformed chunks and their residual payload links.

Source identity is an optional audit dimension, not a selection axis. Rare does
not imply useful, and common does not imply removable.

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

No distributional threshold has runtime veto authority until its taxonomy,
unknown handling, confidence interval, false-positive fixtures, and development
ablation are frozen. The current runtime veto is limited to representative
linkage, zero-survivor explanation, and residual-payload integrity.

### 3.5 Boundary with the other Cores

| Core | Primary question | May delete? | May veto output? |
|---|---|---:|---:|
| Validity | Is the unit structurally usable under the declared input contract? | Yes, by hard reason | No |
| Redundancy | Is the information already represented by a linked family member? | Yes, while retaining a representative | No |
| Quality | Is there promoted evidence of explicit non-payload, or promoted positive retention evidence? | Yes, by named policy | No |
| Coverage | Did the combined removals create an unexplained support failure? | No | Yes |

## 4. Stage Placement

1. Stage A applies Validity and quarantines closed observable failures.
2. Stage B applies Redundancy and promoted Quality Policies to create
   reason-coded removal proposals.
3. Stage C applies Coverage invariants and may veto unexplained support loss
   before writing the curated corpus.

Coverage is therefore part of curation even though it does not delete data. It
is the final validity condition for the corpus-level result.

## 5. Current Claim

The implementation may claim an auditable, domain-agnostic curation interface
and an executable Q1-Q4 Quality measurement protocol. It may not yet claim that
the Quality Policy is runtime-qualified or that it improves every domain.
Those claims require completed protected-fixture, Coverage, corpus-scale, and
independent external evidence under the gates above.
