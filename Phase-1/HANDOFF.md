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
| Stage B | Quality | Retain only chunks that meet the positive Q1-Q4 gate |
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
- Closed deterministic non-payload rules and the Luna-distilled local Q1-Q4
  ranker execute in Stage B. Normal requires one independent pass, Hard two,
  and any qualified fail blocks selection. Abstain, OOD, and low confidence do
  not support retention.
- Coverage is a non-deleting materialization veto. A veto emits typed
  `required_retain_uids`, performs explicit rematerialization, and reruns the
  complete Coverage contract; silent restore is forbidden.
- Semantic Coverage v3 uses reciprocal-neighbor support shared by independent
  Qwen3-Embedding-0.6B and BGE-M3 graphs plus route, script, format, and
  Redundancy-family strata. Qwen is a development/confirmatory runtime
  experiment and BGE-M3 is the independent audit view. Neither is
  scientifically promoted.
- Normal and Hard use identical Coverage invariants. Stage-B Redundancy bounds
  and Quality pass-count operating points are the only source of mode strength.
- Raw record-level composition is descriptive only. The comparable composition
  delta is Stage-B eligible chunk to Stage-C curated chunk, using the same unit
  and immutable chunk IDs. Neither view imposes target percentages or affects
  selection.
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

`quality.distilled_ranker_v1` is the current model-driven Quality candidate. It
evaluates four independent Policies and emits no scalar Quality score:

1. Q1 Correctness Evidence
2. Q2 Semantic Coherence
3. Q3 Substantive Payload
4. Q4 Learnable Relations

GPT-5.6 Luna labels a source-blind calibration sample through the Batch API;
one frozen local ranker then scores the full corpus. Luna is offline evidence
only and cannot select or delete runtime data. The membership rule is fixed:

| Mode | Positive Quality requirement |
|---|---|
| Normal | At least one confident in-distribution Q1-Q4 pass and no qualified fail |
| Hard | At least two confident in-distribution Q1-Q4 passes and no qualified fail |

Abstain, OOD, low confidence, and missing support do not count as passes. A
non-selected chunk can return only through an explicit Stage-C Coverage veto.
Hard Coverage restoration is limited to the final Normal retained set; this
enforces `Hard subset-or-equal Normal` without weakening either Quality gate.
The final Code-7M run configs are `configs/code_7m_luna_final_normal_v1.json`
and `configs/code_7m_luna_final_hard_v1.json`.

### Historical Provider Attempts

The following panels are preserved as negative development history. They are
not the current calibration oracle and must not be cited as the final ranker
lineage.

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

The later NVIDIA runtime-experiment candidate in
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

That panel was intended as an offline calibration oracle and is now superseded.
It never received final runtime membership authority.

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

Teacher output alone has no membership authority. The Stage-B Policy uses only
a hash-bound Ranker artifact and frozen operating points. Missing support creates
no positive pass and therefore cannot select a chunk.

The deterministic sampling, exhaustive frozen embedding artifact, JSON/NPZ
Ranker artifact, OOD detector, positive pass-count gate, and Stage-B
non-selection/Stage-C Coverage-veto bridge are implemented. The final Luna
ranker manifest is
`D:/UNLV-Research/final_all_policy_v1/quality_ranker_luna_v1/ranker_v2/quality_ranker_manifest.json`.
Scientific promotion remains blocked until multidomain and external evidence
qualifies the positive operating points.

The first pre-precedence Q1 diagnostic exposed teachers overruling a trivial
declared verifier result. Those `quality-teacher-observation-v1` records are
excluded from qualification. The `v2` runner rejects legacy observation files.
Its first eight controlled Q1 tasks completed 8/8 `pass` from
`declared_verifier` with zero model-generation traces. This validates evidence
precedence and resume isolation, not full Quality promotion.

Normal and Hard remain unqualified for production release. The Luna-derived
positive-selection operating points now require disjoint multidomain error
analysis and natural-token external evaluation; historical provider fixtures
cannot be reused as confirmatory evidence.

## Retired Contrastive Research

The target/reference NLL-gap Quality design did not identify a stable
route-general deletion boundary and is no longer a current candidate. Its
code, protocols, tests, source-pool work, and frozen negative evidence are
preserved under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.
Current runtime modules, configs, profiles, and status documents must not
import or authorize that archive.

## Current Code 7M Materialization

The 2026-08-08 run completed with 7,147 Normal chunks and 5,859 Hard chunks;
Hard-only chunks are zero. Exact Qwen3-4B stream tokens are 6,984,438 Raw,
6,125,213 Normal, and 5,032,400 Hard. Packed natural-training tokens are
6,979,584, 6,111,232, and 5,029,888 respectively. No equal-token resampling,
target fraction, or maximum token budget was used.

The frozen Raw provenance is documented in
`docs/code_7m_corpus_provenance.md`: 4,723,925 final stream tokens come from
`bigcode/the-stack-dedup`, and 2,260,513 come from version-pinned snapshots of
eight public GitHub repositories. The historical `known_high_quality_reference`
name is provenance metadata, not a Quality judgment or selector input.

### Dataset Integrity Audit

The 2026-08-10 integrity audit found no duplicate record IDs, orphan curated
chunks, duplicate curated chunk IDs, non-parent-substring transformations, or
Hard-only chunks. The final Raw contains one normalized exact duplicate family;
Stage B removes its nonrepresentative. Normal retains 85.64% of The Stack tokens
and 92.00% of reference-pool tokens; Hard retains 69.40% and 77.59%, respectively.
This source-correlated retention is an observed outcome, not a source-aware rule.

Three limitations are frozen explicitly. First, the old composition files
compare Raw whole records with curated chunks and are not like-for-like; future
runs emit `eligible_curated_composition_delta.csv`. Second, all 662 GitHub
reference records carry `rights.license=unknown` and
`partition.source_content_sha256=unknown` in the JSONL even though repository
licenses and snapshot commits are declared externally. This blocks a
self-contained dataset release until a hash-bound metadata repair is
materialized. Third, the 2026-08-08 outputs predate a Stage-A false-positive
fix: one valid 112-token Django error-handler record was mistaken for an
acquisition failure because it mentioned `Page Not Found` and
`Internal Server Error`. The runtime now requires explicit acquisition status,
an error HTTP status plus marker, or an exact short failure body. Existing
training results must be identified as the pre-fix frozen experiment.

## External Evaluation Hierarchy Amendment

The six-suite execution matrix remains mandatory, but its analysis hierarchy
was amended on 2026-08-10 after observing EvalPlus results and before observing
any non-Base reasoning-suite result. BigCodeBench Complete, CRUXEval-I,
CRUXEval-O, and DS-1000 are the primary reasoning suite. HumanEval+ and MBPP+
are mandatory secondary short-function diagnostics. Every arm, seed, and
benchmark remains reportable; an unfavorable EvalPlus result cannot be
discarded. The primary summary is the unweighted macro mean of the four
reasoning benchmark percentages. A positive result supports only a bounded
reasoning-intensive Code claim for the corresponding frozen profile, while
any short-function regression must be disclosed as a capability trade-off.
The timestamped authority is
`protocols/code_reasoning_primary_amendment_v1.json`.

## Completed Three-Seed External Evaluation

The 2026-08-14 matrix is complete: all 60 model-arm-by-benchmark cells exist
and pass the task-level provenance audit at
`D:/UNLV-Research/final_all_policy_v1/external_training_v1/benchmarks_v1/confirmatory_benchmark_provenance_audit.json`.
The audit recomputed 42,820 task judgments and verified exact generated/scored
task-set equality. EvalPlus uses the official `evaluate()` base-plus contract;
BigCodeBench uses the public `bigcode/bigcodebench-evaluator`; CRUXEval uses
upstream `evaluate_generations()`; DS-1000 preserves the official test programs
under Windows subprocess isolation.

The Base/Raw/Normal/Hard primary reasoning macros are 18.38/20.86/20.59/20.80.
Hard retains 72.05% of Raw stream tokens and trails Raw by 0.06 percentage
points on the primary macro; Normal retains 87.70% and trails Raw by 0.27
points. This supports a bounded compression-with-near-retention result, not a
claim that curated arms consistently outperform Raw. The complete seed table
and across-seed statistics are frozen in
`confirmatory_benchmark_results.{json,md}` under the same benchmark root.

## Next Authorized Work

1. Audit Q1-Q4 false retention and false non-selection on disjoint Code, Math,
   General, and multilingual observations.
2. Validate the active witness-based Redundancy modes on disjoint behavior
   and protected false-removal fixtures; promote only a passing operating point.
3. Validate Semantic Coverage on multilingual Code, Math, General prose, and
   structured data: provider agreement and bias, extinction recall, protected
   false-veto bounds, and corpus-scale ANN behavior.
4. Use the completed Base, Raw, Normal, and Hard three-seed natural-token
   evaluation without dropping unfavorable secondary outcomes.

## Verification

Run from `Phase-1` with the repository on `PYTHONPATH`:

```powershell
python validation\test_quality_candidate_authority_v1.py
python validation\test_quality_ranker_sampling_v1.py
python validation\test_quality_ranker_policy_v1.py
python validation\test_quality_ranker_runtime_v1.py
python validation\test_quality_runtime_dispatch_v1.py
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
python validation\test_bigcodebench_remote_runner.py
python validation\test_bigcodebench_chunked_runner.py
python validation\test_benchmark_provenance_audit.py
python validation\test_confirmatory_benchmark_collector.py
```

Generated datasets, model caches, raw API responses containing corpus text,
benchmark outputs, and local work directories must remain untracked.

Do not copy a historical test-count claim into a paper or handoff. Run the
current active and targeted validation commands and record their dated output.
