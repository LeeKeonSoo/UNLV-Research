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
| Nemotron 3 Ultra 550B | NVIDIA Build | Endpoint observed 2026-08-04 |
| GLM-5.2 | NVIDIA Build | Endpoint observed 2026-08-04 |
| Qwen3.5-9B | Local RTX 4060 Ti | Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, int8 |

The local teacher loaded and generated successfully with 10.76 GiB observed
maximum allocated VRAM. Both NVIDIA endpoints responded to a public smoke
request. Their raw output formats were not uniformly schema-compliant, so the
next implementation must use one strict schema-only retry and otherwise
convert the vote to `abstain`.

Teacher output alone has no deletion authority. The candidate remains
`blocked` until fixture, consensus-stability, false-removal, and operating-point
gates pass.

## Retired Contrastive Research

The target/reference NLL-gap Quality design did not identify a stable
route-general deletion boundary and is no longer a current candidate. Its
code, protocols, tests, source-pool work, and frozen negative evidence are
preserved under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.
Current runtime modules, configs, profiles, and status documents must not
import or authorize that archive.

## Next Authorized Work

1. Implement one hosted/local teacher adapter and strict Q1-Q4 response schema.
2. Build and execute the 512-item controlled smoke fixture matrix.
3. Build at least 800 protected fixtures and measure exact one-sided
   false-removal bounds.
4. Validate first-pass and blinded second-pass consensus stability.
5. Freeze separate Normal and Hard operating points without a token budget.
6. Integrate only a promoted Quality provider into Stage B.
7. Redesign near-duplicate authority around route-appropriate equivalence
   witnesses rather than a similarity threshold alone.
8. Validate Coverage representation invariants on Code, Math, General prose,
   and structured data.
9. Run Base, Normal, and Hard curation, followed by external three-seed
   natural-budget evaluation.

## Verification

Run from `Phase-1` with the repository on `PYTHONPATH`:

```powershell
python validation\test_quality_candidate_authority_v1.py
python validation\test_quality_teacher_panel_v1.py
python validation\test_quality_teacher_response_v1.py
python validation\test_quality_teacher_qualification_v1.py
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
