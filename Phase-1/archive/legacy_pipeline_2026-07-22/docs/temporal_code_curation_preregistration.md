# Temporal Code Curation Preregistration

## Decision

The first raw-corpus operational validation will use recently created Python
software changes and an approximately 4B-parameter base model.

Primary model:

```text
Qwen/Qwen3-4B-Base
```

Later model-transfer replication:

```text
bigcode/starcoder2-3b
```

Qwen3-4B-Base is the primary model because it exactly satisfies the 4B target,
is a base rather than instruction-only model, and was publicly released on
April 28, 2025. Only data created after May 1, 2025 is eligible, creating a
defensible temporal non-exposure boundary even though its complete pretraining
corpus is not public.

StarCoder2-3B is reserved for later transfer replication because it is a
code-specific base model trained from the documented The Stack v2 corpus with
opt-out handling. A positive result on both models would strengthen the claim
that the curation effect is not model-specific.

## Research Question

Under matched update tokens and compute, does framework-curated raw-like
Python code data improve code-model capability over raw-random and Stage-A
random data while preserving prior code and general capabilities?

The active protocol decision is recorded in
`docs/code_domain_training_validation_protocol.md`. That decision makes the
raw-vs-curated equal-budget training comparison the primary validation path.
Strict E2 repository-patch tasks remain secondary executable evidence and are
not the main blocker for starting target-model training validation.

## Claim Boundary

The local experiment uses identical QLoRA continued-pretraining recipes across
all update arms. Its primary claim is therefore limited to parameter-efficient
continued pretraining. A full-parameter replication is required before
extending the claim to full-model continued pretraining.

Utility and benchmark outcomes remain Stage C evidence only. They must never
enter the Stage-B selector objective.

Human or LLM review is optional diagnostic evidence only. It is not required
to approve Stage B, cannot tune the selector, and cannot block Stage-C entry.
The primary validation is the pre-registered equal-budget downstream
comparison against the disjoint Stage-A-random arm and frozen ablations.

## Collection Unit

The collection unit is a merged pull-request change bundle, not an isolated
source file:

```text
repository and license metadata
+ issue or pull-request description
+ parent and merge commits
+ pre-change files
+ patch
+ post-change files
+ changed tests
+ merge timestamp
+ provenance URLs
```

This preserves enough context for curation, contamination auditing, and
executable evaluation.

Repository code, tests, and documentation may enter the training payload only
when covered by an allowlisted repository license. Issue and pull-request
prose is retained for provenance and executable evaluation, but is excluded
from training unless a separate documented use basis is recorded. The
collector must use approved public interfaces, honor rate limits and opt-out
or deletion signals, and record the collector version and acquisition time.

Verified executable test commands are a separate executable-evaluation
eligibility gate. They are not a licensed training-content gate. A bundle that
passes content, provenance, split, contamination, and substantive-change gates
may enter Stage 0/A/B even when repository-specific execution is unavailable;
it cannot enter an executable Stage-C holdout until execution verification
passes.

## Frozen Time Splits

| Split | Date range | Primary use |
| --- | --- | --- |
| Training candidates | 2025-05-01 through 2025-12-31 | Raw corpus and curated training releases |
| Development holdout | 2026-01-01 through 2026-02-28 | Candidate and protocol development |
| Frozen confirmation | 2026-03-01 through 2026-05-31 | Untouched final confirmation |

Primary splits are both time-disjoint and repository-disjoint. A repository
identity may appear in only one primary split. Same-repository temporal
evaluation may be reported only as a secondary diagnostic.

Repository assignment is frozen before Core scoring using:

```text
sha256(normalized_repository_identity) modulo 100
```

- buckets `0-79`: training
- buckets `80-89`: development
- buckets `90-99`: frozen confirmation

Only changes whose timestamp window matches the repository's assigned split
may enter that split. This prevents an active repository from leaking across
primary train and evaluation splits.

## Source Eligibility

Initially include only Python repositories that:

- are public, non-fork, non-mirror, and non-archived
- have an allowlisted permissive license
- have a reproducibly fetchable merge commit and parent
- provide a test suite or executable validation command
- are not benchmark source repositories

Allowlisted licenses:

- Apache-2.0
- MIT
- BSD-2-Clause
- BSD-3-Clause
- ISC

Unknown licenses, copyleft licenses, generated code, vendored code, binary
files, lock files, pure formatting changes, secrets, high-risk PII, and
non-reproducible merge states are excluded or quarantined.

## Contamination Contract

No benchmark task, solution, repository, patch, or near-duplicate may enter
training. The quarantine includes LiveCodeBench, BigCodeBench, HumanEval and
EvalPlus, MBPP and EvalPlus, SWE-bench, Multi-SWE-bench, and the project's own
temporal executable holdouts.

Benchmark repositories and benchmark target repositories are not equivalent.
Dedicated benchmark repositories such as HumanEval, LiveCodeBench,
BigCodeBench, and EvalPlus are excluded in full. SWE-bench and Multi-SWE-bench
target real software repositories, so their target repositories are not
excluded in full; only the benchmark's specific issue, commit, patch, test, and
near-duplicate artifacts are quarantined.

For the initial Python-only protocol, SWE-bench task artifacts are mandatory.
The currently published Multi-SWE-bench dataset is centered on non-Python
languages and exposes an empty Python dataset file, so Multi-SWE-bench
task-level artifacts are deferred until the multilingual expansion. Its
dedicated benchmark source repositories remain fully quarantined now.

Checks must include repository identity, exact normalized hashes, token and AST
near-duplicate search, problem-statement and test-signature search, and
provenance auditing.

## Comparison Arms

All update arms use matched tokens, optimizer steps, sequence packing,
learning-rate schedule, QLoRA configuration, seeds, and evaluation settings.

1. Base model without update
2. Raw-random equal-token update
3. Stage-A-random equal-token update
4. Curated equal-token update
5. Curated plus frozen general-replay equal-token update
6. All-raw update, reported separately as a compute-efficiency comparison

The equal-token comparisons test selection quality. The all-raw comparison
tests whether curation provides a useful compute-efficiency tradeoff.

The Stage-B contribution ablations are frozen separately in
`configs/temporal_code_stage_b_ablation_protocol_v1.json`: full selector,
Quality-only, Redundancy-only, no-Coverage-support, Stage-A random, and raw
random. Human/LLM review, Utility, benchmarks, development outcomes, and
confirmatory outcomes are forbidden selector signals for every arm.

## Local Training Contract

- Method: QLoRA continued pretraining
- Quantization: 4-bit NF4 with double quantization
- Compute dtype: bf16
- Adapter: rank 32, alpha 64, dropout 0.05, all linear projections
- Sequence length: 2048
- Micro batch: 1
- Gradient accumulation: 8
- Gradient checkpointing: enabled
- Smoke budget: 1M tokens
- Development budget: 10M tokens
- Confirmatory budget: 20M tokens
- Minimum development seeds: 5
- Minimum fresh confirmatory seeds: 5

The smoke run may calibrate feasibility and a practical effect margin. It must
not select a final candidate or inspect the frozen confirmatory holdout.

## Evaluation

Primary Utility is the predeclared target-model code-evaluation aggregate under
equal-token continued pretraining. It combines external code benchmarks,
internal disjoint heldout NLL, contamination checks, and retention checks.

External code evaluation:

- LiveCodeBench slice created after the training window
- EvalPlus HumanEval+
- EvalPlus MBPP+
- BigCodeBench
- DS-1000 for data-science Python

Secondary executable evaluation may use the strict E2 repository-patch pool,
SWE-bench-compatible tasks, and Multi-SWE-bench-compatible heldout tasks. These
are secondary because strict E2 acquisition is expensive and a 4B base model may
exhibit a floor effect on full repository repair.

General retention includes external general-text NLL, HellaSwag,
ARC-Challenge, PIQA, and WinoGrande.

## Release Rule

A future release requires all of the following:

- curated release beats recipe-matched Stage-A random on the primary
  executable aggregate
- one-sided 95% paired training-seed confidence bound exceeds zero
- the pre-frozen practical effect margin is met
- both development and untouched confirmatory holdouts pass
- all frozen code and general retention guardrails pass
- task-based evidence passes; NLL-only evidence is insufficient

The framework may reject or abstain. A Stage-C failure must not be repaired by
tuning Stage B against benchmark outcomes.

## Immediate Next Work

Implemented pre-collection contracts:

- `ingestion/code_change.py`: change-bundle validation and training-payload
  authorization
- `ingestion/temporal_code_manifests.py`: frozen repository split and
  benchmark-quarantine logic
- `63_build_temporal_code_collection_manifests.py`: manifest/report builder
- `64_discover_temporal_code_repositories.py`: authenticated metadata-only
  discovery
- `65_enrich_temporal_code_repositories.py`: path-only tree and prose-free
  merged-PR metadata enrichment
- `66_probe_temporal_code_commit_reproducibility.py`: code-free sampled merge
  and parent commit identity probe
- `67_build_temporal_code_collection_readiness.py`: conservative freeze
  readiness report
- `68_generate_benchmark_task_artifact_manifest.py`: derived benchmark
  repository, commit, and normalized-hash quarantine rules without retaining
  raw task content
- `69_freeze_temporal_code_smoke_plan.py`: bounded one-repository-per-split
  smoke plan frozen before content fetch
- `70_fetch_temporal_code_smoke_bundles.py`: prose-free bounded code/test/doc
  content fetch with secret and PII quarantine
- `71_audit_temporal_code_smoke_bundles.py`: bundle contract, split, normalized
  hash quarantine, and pre-Stage-0 blocker audit
- `72_verify_temporal_code_test_commands.py`: Docker-only parent/merge smoke
  command verification with frozen isolation limits
- `73_prepare_temporal_code_stage0_candidates.py`: split-preserving adaptation
  of collection-approved payloads into generic Stage-0 records
- `85_freeze_temporal_code_broad_manifest.py`: freezes every broad repository
  that passed discovery, enrichment, commit reproducibility, license, split,
  and benchmark-collision gates before broad content fetch
- `ingestion/code_chunks.py`: syntax-aware Python/documentation chunking and
  code-domain Stage-A hard gates
- `74_run_temporal_code_stage_a_smoke.py`: split-isolated bounded Stage-A smoke
- `ingestion/code_selection.py`: train-only code-domain Stage-B Core proxies,
  coverage constraints, and disjoint Stage-A-random construction
- `75_run_temporal_code_stage_b_smoke.py`: bounded frozen-contract Stage-B
  engineering smoke
- `validation/test_temporal_code_ingestion.py`: regression fixtures
- `validation/test_temporal_code_smoke_audit.py`: normalized signature and
  prose-exclusion regression check

Next:

1. Fetch a bounded broad-corpus tranche under the frozen repository manifest
   and apply the smoke-proven automated content and execution gates.
2. Build broad Stage-A and equal-token Stage-B selected/common-disjoint-random
   arms without Utility or review-label leakage.
3. Build raw-random, Stage-A-random, curated, and known-high-quality
   equal-token code training arms under
   `docs/code_domain_training_validation_protocol.md`.
4. Confirm Qwen3-4B-Base QLoRA feasibility and freeze the practical benchmark
   effect margin before full development runs.

The corrected SWE-bench derived-artifact quarantine invalidated the earlier
bounded-smoke Stage-A/B result. The bounded smoke now contains 4 Stage-0
records and produces 34 syntax-aware chunks: 23 pass Stage A, 11 are rejected,
and zero train chunks remain. Stage B therefore returns
`insufficient_usable_data`. This is the required abstention behavior, not a
failed selector result. The earlier 341-pass/181-selected result must not be
used as current evidence.

The active operational evidence is the frozen broad tranche. Its 19
training-content-eligible bundles produce 23 Stage-0 records, 283 chunks, and
254 Stage-A-pass chunks. The train split contains 175 Stage-A-pass chunks;
Stage B selects 94 and constructs a selected-disjoint 49-chunk Stage-A-random
arm at 99.9744% of the selected token-proxy budget. This demonstrates Stage
A/B engineering operation only. It does not establish Utility or target-model
benefit.

The initial frozen preservation/destruction benchmark passes `7 / 7`,
including pass-through-chain, verbose-filler, identifier-rename, and template
saturation checks. The indexed exact redundancy implementation matches
all-pairs on all 175 broad-tranche train chunks with zero risk, objective,
selected-set, or baseline-set difference. These are engineering and initial
metric-direction checks, not broad scientific validation.

The train-only review expansion added `bytedance/deer-flow` and
`scikit-learn/scikit-learn`, contributing ten and three review-only
Stage-A-pass chunks respectively. A frozen `mem0ai/mem0` attempt yielded no
allowed text files and is retained as a negative fetch result. They are
combined with the Scrapy smoke into a 331-chunk, three-repository blind review
corpus. The 72-record packet hides scores, selection arms, repository identity,
paths, and sampling strata. It is retained as an optional diagnostic and does
not block automated validation, corpus expansion, or Stage C. Expansion
content has unverified executable tests and is not a Stage-0 release candidate
or training-approved input.
