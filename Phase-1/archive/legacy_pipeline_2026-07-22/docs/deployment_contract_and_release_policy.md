# Deployment Contract and Release Policy

## Definition

A curated dataset is not universally good in isolation.

```text
curated dataset =
data that improves the declared outcome for a target model and training budget,
while satisfying declared retention, safety, and contamination constraints
```

The framework input is:

```text
candidate corpus
+ target model
+ training budget
+ deployment objective
+ evaluation distribution
+ retention/safety constraints
-> supported training release or explicit abstention
```

Collection remains upstream. Utility and target-model outcomes remain Stage C
evidence and must never become Stage-B selector objectives.

## Layer Ownership

- Stage 0 normalizes, routes, and quarantines candidate records.
- Stage A creates the broad usable pool.
- Stage B creates a selected core using frozen non-Utility selection-value,
  redundancy, and coverage proxies.
- Stage C evaluates candidate release arms under declared outcomes.
- The release layer applies the predeclared Deployment Contract and emits a
  release action or abstention.

The release layer may choose among:

```text
selected_only
coverage_backfilled
stageA_broad
reject
insufficient_usable_data
```

It does not modify Stage-B scores or selection.

## Deployment Objectives

### Broad Refresh

- primary outcome: broad Stage-A-like heldout
- comparator: equal-budget Stage-A broad/random
- allowed fallback: `stageA_broad` when curated releases do not improve it but
  Stage-A broad improves over base

### Targeted Update

- primary outcome: declared target-distribution evaluation
- required guardrail: broad/general performance regression limit
- a targeted release is not a broad-default claim

### Capability-Preserving Update

- primary outcome: target evaluation
- required guardrails: external general capability and forgetting/regression
- missing required guardrail evidence forces rejection

## Contract Files

- `configs/deployment_contract_broad_refresh.json`
- `configs/deployment_contract_targeted_coverage_refresh.json`
- `configs/deployment_contract_capability_preserving_update.json`

Changing the objective, primary evaluation, comparator, minimum improvement,
guardrail, or preference order creates a new contract and requires new
confirmation evidence.

## Current FineWeb Decisions

Using the same current frozen evidence:

```text
broad_refresh -> stageA_broad
targeted_coverage_refresh -> coverage_backfilled
capability_preserving_update -> reject
```

This records a distribution-dependent tradeoff:

- coverage backfill improves the coverage-stratified target
- coverage backfill does not improve the broad Stage-A primary

Neither release is certified universally.

The provisional capability-preserving check uses the frozen WikiText103
validation/test subset as an external-corpus NLL retention holdout. All update
arms regress against the base model:

| Arm | External WikiText mean NLL | Delta vs base |
| --- | ---: | ---: |
| base no update | 2.741912568 | 0.000000000 |
| selected only | 2.861365947 | +0.119453379 |
| coverage backfilled | 2.861948720 | +0.120036152 |
| Stage-A broad | 2.860333636 | +0.118421069 |

Exact normalized-text overlap between the external holdout and the three
training arms is zero. This supports a provisional forgetting diagnosis, not a
task-benchmark, safety, near-duplicate-contamination, or deployment claim.

## Commands

```powershell
conda run --no-capture-output -n research python 49_build_fineweb_deployment_evidence.py
conda run --no-capture-output -n research python 50_prepare_external_guardrail_holdout.py
conda run --no-capture-output -n research python 51_build_capability_guardrail_evidence.py
conda run --no-capture-output -n research python 48_build_release_decision_report.py --contract configs\deployment_contract_broad_refresh.json --evidence outputs\validation\fineweb_deployment_evidence.json --output outputs\validation\release_decision_broad_refresh.json
conda run --no-capture-output -n research python 48_build_release_decision_report.py --contract configs\deployment_contract_targeted_coverage_refresh.json --evidence outputs\validation\fineweb_deployment_evidence.json --output outputs\validation\release_decision_targeted_coverage_refresh.json
conda run --no-capture-output -n research python 48_build_release_decision_report.py --contract configs\deployment_contract_capability_preserving_update.json --evidence outputs\validation\fineweb_capability_guardrail_evidence.json --output outputs\validation\release_decision_capability_preserving_update.json
python validation\test_release_policy_contract.py
```

## Claim Boundary

The release decision is scoped to its contract and evidence. It does not prove
an optimal mixture, dataset-independent improvement, target-model-independent
improvement, raw-corpus success, or deployment safety without required external
guardrails.
