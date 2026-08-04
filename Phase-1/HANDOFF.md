# Handoff

## Start Here

The sole current-status authority is
`docs/framework_consistency_baseline.md`. The machine-readable root is
`configs/curation_framework_v1.json`. Read those files before changing policy
or running experiments.

## Framework Boundary

The public boundary is corpus input to curated dataset plus an auditable
decision trace. Continued pretraining, NLL, and downstream benchmarks are
external validation and are never selector inputs.

The public Cores and Stages are fixed:

| Stage | Core | Authority |
|---|---|---|
| Stage A | Validity | Quarantine or remove closed observable failures |
| Stage B | Redundancy | Remove nonrepresentatives from reproducibly linked families |
| Stage B | Quality | Apply only promoted independent Quality Policies |
| Stage C | Coverage | Veto unexplained representative or support loss |

Runtime must not consume Utility, NLL, benchmark outcomes, source reputation,
domain quotas, a target retention fraction, or a maximum token budget.

## Current Runtime State

- The central manifest, typed object registry, profile registry, runtime bridge,
  and Stage permissions are hash-verified before corpus input is read.
- Normal and Hard contain the same Policy families. Hard must retain a subset
  of or the same units as Normal.
- Exact-text family removal is `development_passed`.
- Symmetric near-duplicate removal is `blocked`; no safe threshold was found.
- Closed deterministic non-payload rules remain the only current Quality
  behavior in the legacy-compatible selector.
- Coverage is a non-deleting materialization veto.
- The redesigned Normal and Hard profiles remain release-disabled.
- Implementation integrity passes, while scientific release remains blocked
  by unpromoted Policies and missing operating-point calibration.

## Quality Ranker Decision

`quality.teacher_panel_candidate` is the sole current model-driven Quality
candidate. It evaluates four independent Policies and emits no scalar Quality
score:

1. Q1 Correctness Evidence
2. Q2 Semantic Coherence
3. Q3 Substantive Payload
4. Q4 Learnable Relations

The frozen three-teacher panel is:

| Teacher | Location | Frozen identity |
|---|---|---|
| Google Gemma 4 31B IT | NVIDIA Build | Endpoint observed 2026-08-04 |
| Meta Llama 3.1 8B Instruct | NVIDIA Build | Endpoint observed 2026-08-04 |
| Qwen3.5-9B | Local RTX 4060 Ti | Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, int8 |

The common hosted/local adapter, strict response parser, one schema-only retry,
blinded second pass, and Q1-Q4 independent fail-gate aggregation are now
implemented. Reason codes come from a closed Policy-specific vocabulary.
Hosted transport retries are disabled, unavailable teachers produce an audited
`abstain`, and the local teacher loaded in about 11.2 GiB observed VRAM. The
local backend is now lazy-loaded, so verifier-resolved Q1 tasks do not allocate
the 9B model.

Q1 has a deterministic evidence precedence rule. When a typed declared
verifier supplies a versioned `pass` or `fail` result and evidence hash, that
result is authoritative and no teacher is called. The panel evaluates Q1 only
when no declared verifier is available. This prevents probabilistic teachers
from overruling locally checkable evidence while preserving `abstain` for
unsupported correctness claims.

Historical endpoint probes showed that the earlier 70B/753B hosted pair could
exceed 120 seconds per small fixture. The current Gemma/Llama hosted pair was
selected for observable endpoint availability; model adequacy is not assumed
and must be established by the same qualification matrix.

The current panel completed a public Q3 smoke on 2026-08-04 with first-pass
unanimous `pass`. Observed generation times were 5.67 seconds for Gemma, 0.48
seconds for Llama, and 20.16 seconds for local Qwen. This verifies the current
endpoint, prompt, closed reason-code, and local inference path only.

Teacher output alone has no deletion authority. The candidate remains
`blocked` until fixture, consensus-stability, false-removal, and operating-point
gates pass.

The deterministic 512-item behavior matrix and 800-item protected set are
implemented. The resumable executor, qualification report, Normal/Hard
consensus operating points, and Stage-B proposal/Stage-C Coverage-veto bridge
are also implemented. Full teacher observations are not complete, so runtime
activation remains false.

The first pre-precedence Q1 diagnostic exposed teachers overruling a trivial
declared verifier result. Those `quality-teacher-observation-v1` records are
excluded from qualification. The `v2` runner rejects legacy observation files.
Its first eight controlled Q1 tasks completed 8/8 `pass` from
`declared_verifier` with zero model-generation traces. This validates evidence
precedence and resume isolation, not full Quality promotion.

## Retired Contrastive Research

The target/reference NLL-gap Quality design did not identify a stable
route-general deletion boundary and is no longer a current candidate. Its
code, protocols, tests, source-pool work, and frozen negative evidence are
preserved under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.
Current runtime modules, configs, profiles, and status documents must not
import or authorize that archive.

## Next Authorized Work

1. Execute the generated 512-item behavior matrix through the frozen panel.
2. Execute all four Policies on the 800 protected fixtures and measure exact
   one-sided false-removal bounds.
3. Freeze the resulting report and promote only passing Normal/Hard modes.
4. Activate the existing Stage-B proposal/Stage-C Coverage-veto bridge only
   for a promoted mode.
5. Run an admitted corpus-scale reason-code and compression audit.
6. Redesign near-duplicate authority around route-appropriate equivalence
   witnesses rather than a similarity threshold alone.
7. Validate Coverage representation invariants on Code, Math, General prose,
   and structured data.
8. Run Base, Normal, and Hard curation, followed by external three-seed
   natural-budget evaluation.

## Verification

Run from `Phase-1` with the repository on `PYTHONPATH`:

```powershell
python validation\test_quality_candidate_authority_v1.py
python validation\test_quality_teacher_panel_v1.py
python validation\test_quality_teacher_response_v1.py
python validation\test_quality_teacher_adapters_v1.py
python validation\test_quality_teacher_runtime_v1.py
python validation\test_quality_teacher_qualification_v1.py
python validation\test_quality_teacher_fixture_matrix_v1.py
python validation\test_quality_teacher_qualification_runner_v1.py
python validation\test_quality_qualification_report_v1.py
python validation\test_quality_operating_points_v1.py
python validation\test_quality_stage_bridge_v1.py
python validation\test_framework_manifest_v1.py
python validation\test_framework_objects_v1.py
python validation\test_framework_profiles_v1.py
python validation\test_framework_runtime_bridge_v1.py
python validation\test_framework_policy_ablation_v1.py
python validation\test_framework_release_validation_v1.py
python validation\test_active_surface.py
```

Generated datasets, model caches, raw API responses containing corpus text,
benchmark outputs, and local work directories must remain untracked.

The current direct validation surface contains 153 scripts and passed 153/153
after the Quality qualification and staged-policy bridge Blocks.
