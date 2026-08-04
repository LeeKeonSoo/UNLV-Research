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

The sole current model-driven Quality candidate is the three-teacher Quality
Ranker in `configs/quality_teacher_panel_v2.json`. Its active runtime-experiment
panel is GLM-5.2, Nemotron 3 Ultra, and MiniMax M3. It evaluates Q1 Correctness
Evidence, Q2 Semantic Coherence, Q3 Substantive Payload, and Q4 Learnable
Relations independently as `pass`, `fail`, or `abstain`. It remains blocked
from scientific promotion until fresh behavior fixtures, protected-fixture
false-removal bounds, consensus stability, and Normal/Hard operating points
pass. Runtime execution is experimental, and teacher output alone cannot
delete data.

The shared hosted/local adapter, closed Policy-specific reason-code schema,
single schema retry, blinded consensus runner, deterministic 512-item behavior
matrix, 800-item protected set, resumable qualification executor, exact
false-removal report, and Normal/Hard staged-policy bridge are implemented.
The bridge remains inactive until the full observations pass and Stage C
Coverage evidence can veto support loss.

Q1 uses typed declared-verifier evidence before model judgment. A declared
verifier result bypasses teacher generation; the panel is used only when that
evidence is absent. Qualification observations use the incompatible-with-v1
`quality-teacher-observation-v2` contract so pre-fix diagnostics cannot be
resumed into promotion evidence.

The first v2 behavior qualification is negative evidence: Q1 passed all 128
controlled tasks, but Q2 produced two PASS-to-ABSTAIN mismatches by task 143.
Because exact behavior was preregistered, the protected run was not started and
both operating points remain blocked. See
`validation/frozen_contracts/quality_teacher_behavior_gate_v2.json`.

Those results belong to a replaced provider panel and cannot qualify the active
GLM-5.2/Nemotron/MiniMax panel. GLM-5.2 has passed endpoint and schema smoke
only; its observed NVIDIA Build latency remains a release blocker.

The previous loss-gap Quality experiment is retired and preserved only under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.

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
conda run -n research python validation\test_quality_candidate_authority_v1.py
conda run -n research python validation\test_quality_teacher_panel_v1.py
conda run -n research python validation\test_quality_teacher_adapters_v1.py
conda run -n research python validation\test_quality_teacher_runtime_v1.py
conda run -n research python validation\test_quality_teacher_fixture_matrix_v1.py
conda run -n research python validation\test_quality_teacher_qualification_runner_v1.py
conda run -n research python validation\test_quality_qualification_report_v1.py
conda run -n research python validation\test_quality_operating_points_v1.py
conda run -n research python validation\test_quality_stage_bridge_v1.py
conda run -n research python validation\test_source_contract.py
```

Passing these checks confirms executable contracts and fixtures. It does not
establish universal Quality measurement or downstream effectiveness.
