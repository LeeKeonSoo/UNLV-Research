# LM Curation Operational Framework

## Status

This document defines the practical target of the project. The goal is not to
defend a claim that the framework measures intrinsic data quality. The goal is
to build an operational curation layer that receives candidate data from an
upstream collection process and decides what, if anything, should be used for
language-model training.

```text
candidate corpus -> full curated pool -> optional budgeted training subset
                 -> supported LM-training release or explicit abstention
```

The framework must be allowed to say `insufficient_usable_data`, `reject`, or
`abstain`. A system that always emits a curated dataset is not a trustworthy
training-data framework.

## Practical Claim

The allowed operational claim is:

```text
The framework uses frozen pre-outcome selection-value, redundancy, and coverage
proxies to construct candidate training subsets and validates them with the
External Evaluation Protocol against matched baselines.
```

The forbidden stronger claim is:

```text
The framework measures intrinsic data quality.
```

Quality is retained as a legacy runtime label inside the Core taxonomy, but in
implementation it aliases pre-outcome selection-value evidence for observable signals
such as information density, structural usefulness, and boilerplate risk. It
does not certify that a chunk is universally good training data and cannot
authorize hard rejection.

The framework is not required to shrink every corpus. If all records pass the
hard and policy boundaries and the declared training budget can hold them, the
correct result is `retain_all`.

## Operational Core Roles

| Core | Operational role | Stage | Boundary |
| --- | --- | --- | --- |
| Validity | Structural usability gate | Stage B | Not semantic usefulness |
| Selection Value Evidence | Observable pre-outcome evidence for budget allocation | Stage C | Not intrinsic quality; no hard-reject authority |
| Quality | Legacy field/artifact alias only | Compatibility | Not a separate Core construct |
| Redundancy | Duplicate, saturation, and recurrence control | Stage B/C | Must separate harmful duplication from useful recurrence |
| Coverage | Observable distribution-retention and domain/capability-mix drift diagnostic | Stage C/External Evaluation | Domain or mix coverage only when metadata and a declared contract support it |

## What Must Improve For A Real Framework

`Utility` is an External Evaluation Protocol measurement, not a Core and never
a selector objective. The current v2 code-domain result gives a positive target-domain NLL signal,
but that is not the same as a production-ready framework. Operational readiness
requires the selector to handle raw collected corpora without collapsing into a
single narrow definition of quality.

Required Stage-C improvements:

- split harmful duplication from useful recurrence
- preserve concise but useful examples, tests, bug fixes, and API usage chunks
- separate AST richness from learnable code usefulness
- cap repository, path, and template concentration
- report selected-vs-budget-not-selected feature shifts before External Evaluation
- emit failure diagnoses when curated does not beat Stage-B-random

Required External Evaluation evidence:

- target-domain heldout NLL
- raw-random and Stage-B-random equal-token baselines
- known-high-quality reference arm when available
- external domain benchmark guardrail
- general-text retention guardrail
- general-task retention guardrail
- contamination audit

Missing evidence must produce `abstain`, not a forced release.

## Code-Domain Next Target

For raw-like Python corpora, the next selector should explicitly represent:

- parseable unit type
- code/test/doc content type
- import and API meaningfulness
- concise example support
- bug-fix or regression-test signal
- generated, template, vendored, and boilerplate risk
- useful recurrence bucket
- harmful duplication bucket
- repository/path diversity bucket
- length and packing bucket

These features are still pre-outcome signals. They can support Stage-C
selection only because their downstream usefulness is later tested externally.

## Full Curated Pool And Optional Budgeting

Stage A and Stage B define whether a record may enter the full curated pool.
Stage C does not redefine records as good or bad. It allocates a constrained
training budget when one exists.
`221_build_stage_b_policy_contract_audit.py` is the current machine-readable
gate for this boundary.

```text
Stage A quarantine
  -> Stage B hard gate
  -> full curated pool
  -> optional Stage C budget allocation
     -> selected_for_training_budget
     -> budget_not_selected
```

Canonical dispositions:

| Dimension | Values | Meaning |
| --- | --- | --- |
| Curation | `retained`, `rejected`, `quarantined` | Eligibility for the curated pool |
| Training budget | `not_requested`, `selected_for_training_budget`, `budget_not_selected` | Allocation under a declared training budget |

`budget_not_selected` is never equivalent to rejection. No fixed rejection
quota, target reduction ratio, or forced downsampling is allowed.

## Operational Use Cases

| Input condition | Expected behavior |
| --- | --- |
| All records are usable and high-value | Preserve the full pool; use `retain_all` when budget permits |
| All but a few records are broken | Reject only the explicit hard failures |
| Usable records have mixed selection evidence | Preserve all in the curated pool; rank only under a binding budget |
| Exact or fuzzy-near-duplicate dump | Remove raw/canonical exact copies with lineage; route fuzzy similarity through reversible Stage-B redundancy controls |
| Narrow but valid high-value domain | Preserve and route or scope the release; do not force artificial broadness |
| Rare concise tests, fixes, or API examples | Preserve unless an independent hard rule applies |
| Safety or licensing status is uncertain | Quarantine rather than silently reject or release |
| Budget smaller than the curated pool | Produce a budgeted subset and mark the rest `budget_not_selected` |
| Budget large enough for the curated pool | Skip competitive selection and emit `retain_all` |

The current legacy-named Stage-C policy contract status is
`stage_b_policy_contract_audit_passed`. It verifies that `budget_not_selected`
is neither rejection nor a low-quality label, that `retain_all` is valid when
the budget does not bind, and that selector Utility leakage remains blocked.

## Decision Standard

A curated release is supported only when:

- Stage B has removed structurally unusable or unsafe chunks
- Stage C has either emitted `retain_all` or selected under frozen non-Utility rules
- External Evaluation beats matched equal-token baselines under the frozen protocol
- required retention and contamination guardrails pass
- the decision layer emits a scoped action with caveats

If any mandatory evidence is missing, the correct framework action is
`abstain`. If the primary comparison fails, the correct action is reject or
redesign a later cycle, not tune the current frozen cycle until it passes.
