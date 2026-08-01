# Paper Method: Core-Metric-Policy Curation Framework

This section defines the method claim supported by the current artifacts. The
framework is a curation-stage layer for language-model training data. It does
not claim to measure intrinsic data quality, certify legal clearance, or act as
a production-ready universal filter.

## Problem Setup

The input is a candidate corpus produced by an upstream collection process and a
declared training or deployment contract. The contract may include an optional
domain or capability mix target, but the framework must treat it as a declared
objective rather than a universal ratio. The output is one of four scoped
actions:

| Action | Meaning |
| --- | --- |
| `retain_all` | Every eligible record can be kept because no binding budget forces competition. |
| `selected_for_training_budget` | The record receives budget in a constrained training subset. |
| `budget_not_selected` | The record remains in the curated pool but is outside the constrained subset. |
| `reject`, `quarantine`, or `abstain` | The evidence does not support direct use, either because a hard rule failed, a risk is unresolved, or validation is incomplete. |

The method is intentionally not a forced downsampler. If a raw candidate corpus
is already usable and the declared budget can hold it, the correct behavior is
to preserve it.

## Core-Metric-Policy Contract

The framework separates measurement roles from policy decisions. Core signals
produce auditable evidence; policy decides what action is allowed at each stage.

| Core surface | Operational role | Policy boundary |
| --- | --- | --- |
| Validity | Detect structurally unusable chunks. | Can support Stage-B rejection only for explicit hard failures. |
| Selection Value Evidence | Rank observable pre-outcome usefulness signals. | Can allocate budget, but cannot certify intrinsic quality or hard-reject alone. |
| Redundancy | Control exact duplicates, near duplicates, and harmful saturation. | Must preserve useful recurrence and representative lineage. |
| Coverage | Track retention and drift across observable source, style, path, content, cluster, and declared domain/capability axes. | Supports collapse and composition diagnostics; does not prove Utility, intrinsic quality, or target-mix satisfaction unless a target mix is declared and validated. |

`Quality` is retained only as a legacy artifact alias for Selection Value
Evidence. In the paper text, the canonical construct is Selection Value
Evidence, not intrinsic Quality.

`Utility` is not a Core. It is measured only by the External Evaluation Protocol
after the Stage-C output has been frozen.

## Stage A: Candidate Boundary

Stage A normalizes candidate records, records provenance, and quarantines
unresolved hazards before chunk-level scoring. The current implementation covers
project-defined PII, secret, benchmark-contamination, poisoning, and rights-risk
fixtures. These checks support a scoped quarantine boundary, not production
detector certification.

## Stage B: Chunk-Level Hard Gate

Stage B answers whether a chunk may enter the curated pool at all. It is a hard
gate for structural invalidity, unrecoverable parsing failures, pathological
repetition, and explicit quarantine outcomes. Stage A does not judge downstream
training Utility or semantic preference.

Every Stage-B survivor belongs to the full curated pool. This preserves the
case where the input corpus is already strong and should not be reduced.

## Stage C: Optional Budget Allocation

Stage C runs only when the declared token or compute budget is smaller than the
full curated pool. It ranks Stage-B survivors using frozen pre-outcome evidence:
Selection Value Evidence, redundancy controls, useful recurrence, length
support, and observable coverage support.

Stage C produces a training subset and marks the remaining retained records as
`budget_not_selected`. That label is not a rejection and is not a low-quality
claim. Stage C must not consume Utility, benchmark outcomes, validation NLL, or
any downstream model-result field.

The Stage-C policy contract is audited separately from external-evaluation success. The
current legacy-named gate is `221_build_stage_b_policy_contract_audit.py`, which verifies
that Stage C is optional budget allocation over retained Stage-B survivors, not
mandatory shrinking or quality rejection.

## External Evaluation Protocol

The External Evaluation Protocol evaluates candidate training releases after Stage-C selection or
`retain_all`. It compares frozen equal-token arms against matched baselines from
the same candidate corpus and checks retention or task guardrails required by
the deployment contract.

Utility belongs here. A positive external-evaluation result supports the claim that the
curated release is useful for the tested training setting. A missing or failed
guardrail produces abstention, rejection, or a scoped caveat; it does not license
post-hoc tuning of Stage C on the same frozen evidence.

## Decision Layer

The decision layer combines Stage A, Stage B, Stage C, and external-evaluation evidence into
a scoped paper claim. The current supported claim tier is:

```text
curation_stage_research_framework
```

The current unsupported claim tier is:

```text
production_deployment_claim
```

The production blocker is Core metric validity at production scale. Current
Core behavior checks, redundancy calibration, Stage-0 fixtures, coverage
diagnostics, and Utility-leakage audits are sufficient for a bounded research
framework claim, but they are not an external production certification.

Historical artifact and script names retain the old notation for reproducibility:
`legacy Stage 0 -> Stage A`, `legacy Stage A -> Stage B`, `legacy Stage B ->
Stage C`, and `legacy Stage C -> External Evaluation Protocol`.

## Reproducibility Surface

The method is reproducible through frozen configs, source hashes, report hashes,
and machine-readable validation ledgers. The paper package joins the hard paper
gate, Core claim defense, Stage-C training validation, and confirmatory decision
boundary. The package must report both supported claims and forbidden claims so
that the method cannot silently drift into intrinsic quality or production-ready
claims.
