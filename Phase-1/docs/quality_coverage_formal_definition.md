# Formal Definition of Quality and Coverage

Status: design contract v1, 2026-08-01

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

### 2.1 Scientific definition

Quality is **expected learning contribution per training token under a declared
LM training objective and evaluation distribution**. It is not an intrinsic
property of a document.

For model initialization `m`, training seed `s`, current training corpus `D`,
evaluation distribution `T`, and chunk `x`, define marginal learning gain:

```text
Delta_T(x | D, m, s)
  = [L_T(theta(D; m, s)) - L_T(theta(D plus x; m, s))] / tau(x)
```

`L_T` is a declared external evaluation loss or preregistered task risk and
`tau(x)` is the exact tokenizer token count. Positive `Delta_T` means that
adding `x` reduced evaluation risk per token. The population target is:

```text
Q_T(x | D) = E_{m,s}[Delta_T(x | D, m, s)]
```

This definition is conditional on `D`, `T`, model family, tokenizer, and
training procedure. Interactions between chunks mean that document Quality is
not additive and no universal context-free Quality score exists.

At corpus level, the externally measured learning efficiency is:

```text
Eff_T(C) = E_{m,s}[L_T(theta_0) - L_T(theta(C; m, s))] / Tau(C)
```

This quantity is evidence for a frozen curation policy. It is not a selector
input.

### 2.2 Runtime meaning

The active runtime does **not** claim to observe `Q_T`. Quality has deletion
authority only when a named, versioned policy provides reproducible evidence
that the unit is non-payload for the declared training interface. Examples are
an empty HTML shell or a separable control/license span with residual payload.

Every Quality decision is one of:

| Decision | Meaning | Runtime action |
|---|---|---|
| `reject` | A promoted policy proves an explicit non-payload condition | Remove the chunk or declared span and emit the full trace |
| `keep` | A separately promoted positive retention policy proves its condition | Retain; currently no positive provider is active |
| `abstain_retain` | Neither proof exists | Retain without calling the chunk high quality |

Shortness, lexical diversity, model perplexity, source reputation, and a
handwritten weighted sum are not deletion proofs.

### 2.3 Permissible future mathematical estimator

A future scalar must estimate the latent target rather than redefine it. Given
an evidence vector `E(x)` and declared content route `r`, the candidate is:

```text
q_hat_r(x) = P(Q_T(x | D) > delta | E(x), r)
```

or an estimated gain with a calibrated interval `[LCB_r(x), UCB_r(x)]`.
Weights must be learned and calibrated on a development corpus disjoint by
stable record ID and normalized-text hash from confirmatory corpora. Human
labels are optional diagnostics, never the sole ground truth. Perturbation
pairs, explicit artifact fixtures, frozen proxy-model evidence, and external
natural-budget training provide the development evidence.

Promotion may use the following non-compensatory decision rule:

```text
reject candidate: UCB_r(x) <= 0 and an explicit negative policy trigger exists
keep candidate:   LCB_r(x) > 0 and a positive policy has been validated
otherwise:        abstain_retain
```

Normal and Hard may differ only through separately frozen confidence levels or
additional promoted policies. They may not share arbitrary weights and merely
change a score threshold.

### 2.4 Quality validation gate

A Quality rule becomes enabled only after all of the following pass:

1. versioned trigger and non-trigger conditions;
2. positive, false-positive, adversarial, and clean-corpus fixtures;
3. reason code, policy hash, original-text hash, and token-delta trace;
4. rule-off versus rule-on development ablation;
5. Coverage invariants and distribution-impact report;
6. benchmark-disjoint, three-seed, natural-budget external evaluation;
7. a preregistered non-inferiority or improvement criterion.

The current active system therefore supports **evidence-bounded non-payload
removal**, not universal document Quality scoring.

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

1. Stage A adapts input and performs source-agnostic text integrity handling.
2. Stage B chunks the released text and applies hard invalid/exact-duplicate
   gates under the immutable profile.
3. Stage C applies enabled Redundancy and Quality policies, then runs Coverage
   invariants before writing the curated corpus.

Coverage is therefore part of curation even though it does not delete data. It
is the final validity condition for the corpus-level result.

## 5. Current Claim

The implementation may claim an auditable, domain-agnostic curation interface
with immutable profiles, reason-coded structural policies, and Coverage
materialization invariants. It may not yet claim a universal mathematical
Quality estimator or domain-general downstream improvement. Those claims
require a calibrated estimator and independent Code, Math, and General
confirmatory evidence under the gates above.
