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
- Witness-based symmetric Near-duplicate removal is active in the final
  runtime experiment. It remains a release-disabled candidate until external
  validation closes.
- Retrieval similarity alone has no deletion authority. Normal and Hard both
  execute exact, formatting, bounded near-substitute, containment, and
  token-preserving reflow witnesses; Hard uses broader frozen bounds. Every
  proposal has a stable representative trace and Stage-C Coverage veto check.
- Closed deterministic non-payload rules and the GLM-5.2/Nemotron/MiniMax
  Q1-Q4 panel both execute in Stage B. Provider output is evidence; frozen
  Normal/Hard Policy consensus owns the removal proposal.
- Coverage is a non-deleting materialization veto. A veto emits typed
  `required_retain_uids`, performs explicit rematerialization, and reruns the
  complete Coverage contract; silent restore is forbidden.
- Semantic Coverage v3 uses reciprocal-neighbor support shared by independent
  Qwen3-Embedding-0.6B and BGE-M3 graphs plus route, script, format, and
  Redundancy-family strata. Qwen is a development/confirmatory runtime
  experiment and BGE-M3 is the independent audit view. Neither is
  scientifically promoted.
- Normal and Hard use identical Coverage invariants. Stage-B Redundancy or
  Quality proposals are the only source of mode strength.
- Raw-to-Curated route and language/script composition files are explanation
  artifacts only. They do not impose target percentages or affect selection.
- The redesigned Normal and Hard profiles remain release-disabled.
- Implementation integrity passes, while scientific release remains blocked
  by unpromoted Policies and missing operating-point calibration.

## Previous Structural Snapshot

The final candidate paths are under
`D:/UNLV-Research/final_framework_test_v1/`. Normal and Hard each receive the
same 8,024 Stage-B chunks and 40 removal proposals. Stage C restores 15 support
representatives and emits 7,999 chunks. Normal has a 2,571,572 whitespace-token
proxy; the then-development Hard candidate had 2,461,632 after span compaction.
Those span candidates are disabled in the final runtime. Exact Qwen3-4B stream-token
counts are 6,984,438 Raw, 6,961,249 Normal, and 6,747,888 Hard. Packed totals
are 6,979,584, 6,946,816, and 6,733,824. The authoritative report is
`training_inputs_v2/training_inputs_report.json`.

The mixed 768-record Code/Math/General semantic audit passes its implementation
gate but not scientific promotion. Mean provider-neighbor Jaccard is 0.2178;
this supports fail-closed local consensus, not a domain-general semantic
taxonomy claim.

These measurements predate all-policy Quality and witness Redundancy execution.
The replacement outputs are under `D:/UNLV-Research/final_all_policy_v1/` and
must not reuse the old token totals as final results.

`final_experiment_preflight_v1.json` reports framework materialization ready
and external confirmatory ready. Paper-claim and production-release readiness
remain false because Policy and Coverage scientific promotion gates are open.

## Quality Ranker Decision

`quality.teacher_panel_candidate` is the sole current model-driven Quality
candidate. It evaluates four independent Policies and emits no scalar Quality
score:

1. Q1 Correctness Evidence
2. Q2 Semantic Coherence
3. Q3 Substantive Payload
4. Q4 Learnable Relations

The blocked v1 three-teacher panel was:

| Teacher | Location | Frozen identity |
|---|---|---|
| Google Gemma 4 31B IT | NVIDIA Build | Endpoint observed 2026-08-04 |
| Meta Llama 3.1 8B Instruct | NVIDIA Build | Endpoint observed 2026-08-04 |
| Qwen3.5-9B | Local RTX 4060 Ti | Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, int8 |

That panel failed its behavior gate after 143/512 tasks. Q1 completed all 128
tasks exactly, but two Q2 Code protected PASS constructions resolved to
`abstain`. The frozen failure evidence is
`validation/frozen_contracts/quality_teacher_behavior_gate_v2.json`. Those
fixtures are spent and must not be used to tune a replacement panel.

The historical all-hosted provider-selection candidate was:

| Teacher | Location | Frozen identity |
|---|---|---|
| Mistral Medium 3.5 128B | NVIDIA Build | Endpoint observed 2026-08-04, reasoning disabled |
| NVIDIA Nemotron 3 Ultra 550B A55B | NVIDIA Build | Endpoint observed 2026-08-04, thinking disabled |
| DeepSeek V4 Pro | NVIDIA Build | Endpoint observed 2026-08-04, thinking disabled |

Its candidate contract is outside the active config surface at
`validation/candidate_contracts/quality_teacher_panel_v2.json`. The 64-cell
provider-selection matrix matched 64/64 expected panel decisions. Mistral,
Nemotron, and DeepSeek first-pass expected-decision matches were 52/52, 46/52,
and 51/52 respectively. Seven Nemotron generation traces were unavailable;
Mistral p50/p95 latency was 61.668/150.064 seconds. Therefore the behavior
development check passed, but provider operational readiness did not. The
frozen boundary is
`validation/frozen_contracts/quality_teacher_development_gate_v2.json`.

The current runtime-experiment panel in
`configs/quality_teacher_panel_v2.json` is:

| Teacher | Location | Frozen identity |
|---|---|---|
| Z.ai GLM-5.2 | NVIDIA Build | Endpoint observed 2026-08-05; 600-second timeout, one retry |
| NVIDIA Nemotron 3 Ultra 550B A55B | NVIDIA Build | Endpoint observed 2026-08-04, thinking disabled |
| MiniMax M3 | NVIDIA Build | Endpoint observed 2026-08-04 |

GLM-5.2 accepted the production batched Q1-Q4 response schema. Synthetic
endpoint probes observed successes at 237.971 and 409.265 seconds and one
timeout at 300.508 seconds. This establishes adapter compatibility, not
latency readiness or Quality validity. The provider change invalidates all
inherited calibration and requires fresh disjoint behavior and protected
false-removal evidence.

The common hosted/local adapter, strict response parser, one schema-only retry,
blinded second pass, and Q1-Q4 independent fail-gate aggregation are now
implemented. Reason codes come from a closed Policy-specific vocabulary.
Hosted transport retries are normally disabled; GLM-5.2 alone receives one
transport retry because the free endpoint exceeded 300 seconds in observed
health probes. Unavailable teachers produce an audited `abstain`. The retired
local panel remains lazy-loaded in its historical implementation.

Q1 has a deterministic evidence precedence rule. When a typed declared
verifier supplies a versioned `pass` or `fail` result and evidence hash, that
result is authoritative and no teacher is called. The panel evaluates Q1 only
when no declared verifier is available. This prevents probabilistic teachers
from overruling locally checkable evidence while preserving `abstain` for
unsupported correctness claims.

Historical endpoint probes remain development diagnostics only. Hosted model
availability is not inferred from an NVIDIA catalog entry; every replacement
must pass the same endpoint, schema, behavior, latency, and protected-fixture
gates.

Teacher output alone has no deletion authority. The frozen Stage-B Policy now
uses panel consensus in the runtime experiment; scientific promotion remains
blocked until fixture, consensus-stability, false-removal, and operating-point
gates pass.

The deterministic 512-item behavior matrix and 800-item protected set are
implemented. The resumable executor, qualification report, Normal/Hard
consensus operating points, and Stage-B proposal/Stage-C Coverage-veto bridge
are also implemented. Runtime materialization is enabled only as an experiment;
full teacher observations are incomplete and scientific promotion remains
blocked.

The first pre-precedence Q1 diagnostic exposed teachers overruling a trivial
declared verifier result. Those `quality-teacher-observation-v1` records are
excluded from qualification. The `v2` runner rejects legacy observation files.
Its first eight controlled Q1 tasks completed 8/8 `pass` from
`declared_verifier` with zero model-generation traces. This validates evidence
precedence and resume isolation, not full Quality promotion.

Normal and Hard remain unqualified for release. The active v2 panel has not run
a fresh disjoint 512-item behavior gate or the
3,200-policy protected run. Do not weaken the gate or reuse spent confirmatory
fixtures; provider changes require development evidence followed by a fresh
disjoint confirmatory set.

## Retired Contrastive Research

The target/reference NLL-gap Quality design did not identify a stable
route-general deletion boundary and is no longer a current candidate. Its
code, protocols, tests, source-pool work, and frozen negative evidence are
preserved under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.
Current runtime modules, configs, profiles, and status documents must not
import or authorize that archive.

## Next Authorized Work

1. Stabilize or replace the unavailable v2 provider and freeze a practical
   provider operational gate without changing Q1-Q4 semantics.
2. Generate a fresh, hash-disjoint 512-item behavior matrix and execute it once
   through the frozen replacement panel.
3. Execute all four Policies on 800 fresh protected fixtures and measure exact
   one-sided false-removal bounds.
4. Freeze the resulting report and promote only passing Normal/Hard modes.
5. Complete the current all-policy 7M runtime and freeze its reason-code,
   removal, restoration, and exact-token report.
6. Run an admitted corpus-scale reason-code and compression audit.
7. Validate the active witness-based Redundancy modes on disjoint behavior
   and protected false-removal fixtures; promote only a passing operating point.
8. Validate Semantic Coverage on multilingual Code, Math, General prose, and
   structured data: provider agreement and bias, extinction recall, protected
   false-veto bounds, and corpus-scale ANN behavior.
9. Run Base, Normal, and Hard curation, followed by external three-seed
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

Do not copy a historical test-count claim into a paper or handoff. Run the
current active and targeted validation commands and record their dated output.
