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

Normal and Hard expose the same immutable Policy families. Normal requires one
independent positive Q1-Q4 pass; Hard requires two, while preserving
`Hard subset-or-equal Normal`. Neither accepts a run-local threshold, retention
fraction, or token budget. Both redesigned operating points are now wired into
the final runtime experiment but remain uncalibrated and release-disabled.

The current model-driven Quality candidate is a distilled Q1-Q4 ranker rooted
in `configs/quality_ranker_v1.json`. GPT-5.6 Luna, declared in
`configs/quality_teacher_luna_single_v1.json`, supplies offline Batch calibration
labels only. The frozen local ranker evaluates Q1 Correctness Evidence, Q2
Semantic Coherence, Q3 Substantive Payload, and Q4 Learnable Relations as
independent `pass`, `fail`, or `abstain` decisions. A qualified fail blocks
selection. Abstain, OOD, and low confidence provide no positive retention
evidence. Stage C receives every typed non-selection proposal and may restore
only the support required by its Coverage invariants. Hard restoration is
bounded by the final Normal retained set, so Coverage cannot create a
Hard-only survivor while satisfying its support checks.

Q1 uses typed declared-verifier evidence before model judgment. A declared
verifier result bypasses teacher generation; the panel is used only when that
evidence is absent. Qualification observations use the incompatible-with-v1
`quality-teacher-observation-v2` contract so pre-fix diagnostics cannot be
resumed into promotion evidence.

Historical multi-provider failures remain negative development evidence and do
not qualify the Luna-derived ranker. The current positive-selection behavior
fixtures pass, but the Policy remains release-blocked pending disjoint
multidomain and natural-token external validation.

The previous loss-gap Quality experiment is retired and preserved only under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/`.

Redundancy uses the witness-based Normal/Hard runtime experiment.
Similarity and embedding retrieval only produce candidate pairs. Normal may
propose removal for exact, formatting, strict bounded near-substitute, exact
containment, and token-preserving reflow witnesses. Hard executes the same
families with broader frozen changed-token bounds and may also consume a
versioned declared equivalence verifier. Near edges are pairwise rather than
silently transitively closed. All uncertainty retains, substantive changes are
protected, and Stage C must veto or accept every family proposal before final
materialization. The Policy remains release-disabled pending external evidence.

Semantic Coverage v3 is implemented as the Stage-C qualification candidate.
It builds local support groups from reciprocal-neighbor evidence shared by
independently frozen Qwen3-Embedding-0.6B and BGE-M3 graphs, then combines them
with deterministic route, script, format, and Redundancy-family strata. A veto
returns explicit required retentions, rematerializes, and reruns the complete
Coverage contract. Embedding similarity never deletes. Qwen is enabled only as
a development/confirmatory runtime experiment; BGE-M3 remains the independent
audit view. Neither is scientifically promoted.

Every curation run now emits explanatory Raw-to-Curated route and
language/script composition JSON/CSV artifacts. They are audit-only, may be
multi-label, and never enforce a target domain distribution or enter the
selector.

## Current Positive-Quality Snapshot

The 2026-08-08 Code 7M run applied the Luna-distilled Q1-Q4 gate and Semantic
Coverage to the same audited Raw input. Exact Qwen3-4B tokenizer stream counts
are 6,984,438 Raw, 6,125,213 Normal, and 5,032,400 Hard. Normal removes 12.30%
of Raw tokens and Hard removes 27.95%, without a target retention fraction or
maximum token budget. Final retained chunk counts are 7,147 Normal and 5,859
Hard, with zero Hard-only chunks. The complete natural-training groups contain
6,979,584 Raw, 6,111,232 Normal, and 5,029,888 Hard tokens; only each arm's
incomplete final optimizer group is dropped.

The Raw corpus is not a single-source 7M The Stack sample. Its frozen
provenance is 4,723,925 tokens from `bigcode/the-stack-dedup` and 2,260,513
tokens from eight version-pinned public GitHub repositories. See
`docs/code_7m_corpus_provenance.md` for exact records, commits, hashes, source
claim boundaries, and the 12-record benchmark-exclusion audit.

## Previous Structural Snapshot

The previous Code 7M candidate run materialized both Normal and Hard through
Stages A-B-C. Stage A released 4,889 records, Stage B produced 8,024 chunks and
40 removal proposals, and Stage C restored 15 support representatives. The
result is 7,999 retained chunks in each mode. Normal reports a 2,571,572
whitespace-token proxy; Hard reports 2,461,632 because Hard also performs
span-level compaction. Frozen Qwen3-4B stream-token counts are 6,984,438 Raw,
6,961,249 Normal, and 6,747,888 Hard. Relative to Raw, Normal removes 0.33% and
Hard removes 3.39%. The packed natural-budget totals are 6,979,584, 6,946,816,
and 6,733,824 respectively. These numbers predate active Quality-panel and
witness-based Near-duplicate deletion and are not the final all-policy result.

The final run contracts are
`configs/code_7m_all_policy_final_normal_v1.json` and
`configs/code_7m_all_policy_final_hard_v1.json`. They execute every declared
Validity, Redundancy, Quality, and Coverage Policy. Hosted Quality observations
are resumable and shared by exact `(panel, chunk UID, text)` identity, while
the full corpus uses a hash-bound embedding and Ranker artifact.

The mixed Code/Math/General Coverage audit contains 768 records. Its mean
cross-provider mutual-neighbor Jaccard is 0.2178, with 499 records in stable
local support groups and 638 in overlapping uncertainty groups. The
implementation gate passes, but scientific promotion remains false pending
protected false-veto and independent multidomain confirmatory evidence.
The final preflight marks framework materialization and external confirmatory
execution ready; paper-claim and production-release readiness remain false.

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
conda run -n research python validation\test_quality_ranker_sampling_v1.py
conda run -n research python validation\test_quality_ranker_policy_v1.py
conda run -n research python validation\test_quality_ranker_runtime_v1.py
conda run -n research python validation\test_quality_runtime_dispatch_v1.py
conda run -n research python validation\test_quality_teacher_panel_v1.py
conda run -n research python validation\test_quality_teacher_adapters_v1.py
conda run -n research python validation\test_quality_teacher_runtime_v1.py
conda run -n research python validation\test_quality_teacher_fixture_matrix_v1.py
conda run -n research python validation\test_quality_teacher_qualification_runner_v1.py
conda run -n research python validation\test_quality_qualification_report_v1.py
conda run -n research python validation\test_quality_operating_points_v1.py
conda run -n research python validation\test_quality_stage_bridge_v1.py
conda run -n research python validation\test_source_contract.py
conda run -n research python validation\test_coverage_engine_v2.py
conda run -n research python validation\test_semantic_coverage_materializer_v1.py
conda run -n research python validation\test_final_experiment_preflight_v1.py
```

Passing these checks confirms executable contracts and fixtures. It does not
establish universal Quality measurement or downstream effectiveness.
