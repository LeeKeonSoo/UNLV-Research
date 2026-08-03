# LM Training Data Curation

This directory contains an auditable curation framework for language-model
training corpora. Its runtime boundary ends when it emits a curated dataset and
its decision trace. Continued pretraining and downstream benchmarks are an
external evaluation protocol, not selector inputs.

## Current Authority

Read these files in order:

1. `docs/framework_consistency_baseline.md` records observed behavior, known
   inconsistencies, and their implementation-resolution status.
2. `docs/quality_coverage_formal_definition.md` defines the scientific target
   and runtime authority of Quality and Coverage.
3. `HANDOFF.md` gives the current repository state and verification commands.
4. `docs/current_curation_framework.md` preserves detailed design context.

When status documents disagree, the consistency baseline wins. Quality and
Coverage semantics are governed by their formal definition together with the
executable curation contract. Candidate designs, development reports, and
historical experiments do not become active runtime policy merely because
their files exist.

## Research Boundary

The framework keeps the Core-Metric-Policy separation and the four public
Cores: Validity, Quality, Redundancy, and Coverage. The current implementation
does not support a claim that it measures intrinsic document Quality or
universally improves downstream performance.

Runtime policy must not consume benchmark scores, NLL, Utility, source
reputation, domain quotas, or a forced token budget. External experiments may
use those measurements only to validate a frozen policy after curation.

## Repository Map

- `run_curation.py`: public curation entry point.
- `ingestion/`: input adaptation and candidate contracts.
- `configs/`: runtime contracts plus candidate evidence configurations. See
  `configs/README.md` before treating a file as active.
- `protocols/`: development and external-evaluation protocols. They are not
  selector policy. See `protocols/README.md`.
- `validation/`: behavior and contract checks.
- `external_evaluation/`: post-curation training and benchmark tooling.
- `docs/`: current authority, design notes, development records, and historical
  analyses. See `docs/README.md`.
- `archive/`: legacy pipeline and pre-reduction snapshots. It is not imported by
  the current runtime.

The top-level Python surface still contains active runtime modules and
unpromoted candidate modules together. This is a known organization debt, not
evidence that every module is active. Physical relocation is deferred because
it changes imports and the active-surface test contract.

## Current Policy State

Normal and Hard now expose the same immutable Policy families. Their intended
difference is a separately calibrated versioned operating point: Normal is the
more conservative removal mode and Hard is stronger while preserving
`Hard subset-or-equal Normal`. Neither accepts a run-local threshold, retention
fraction, or token budget. Both redesigned operating points remain uncalibrated
and release-disabled; the legacy-compatible selector behavior is unchanged.

Block 10B adds a fail-closed Contrastive Quality operating-point gate. The
current two-model audit cannot emit Normal or Hard thresholds: the qualified
background role, one arm-disjoint Stage-A baseline, a third development source
group per route, three ordered effect bins per route, and natural-budget
external evidence are still missing. These are evidence requirements, not
runtime selector inputs.

## Verification

From `Phase-1` in the `research` environment:

```powershell
conda run -n research python validation\test_active_surface.py
conda run -n research python validation\test_curation_contract.py
conda run -n research python validation\test_candidate_processing.py
conda run -n research python validation\test_curation_runtime.py
conda run -n research python validation\test_policy_profile_contract.py
conda run -n research python validation\test_core_policy_runtime_linkage.py
conda run -n research python validation\test_core_behavior_audit_v3.py
conda run -n research python validation\test_contrastive_operating_point_gate_v1.py
conda run -n research python validation\test_source_contract.py
```

Passing these checks confirms executable contracts and fixtures. It does not
establish universal Quality measurement or downstream effectiveness.
