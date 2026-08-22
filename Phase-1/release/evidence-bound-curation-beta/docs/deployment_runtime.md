# Deployment Runtime Contract

## Boundary

The framework accepts declared JSONL records and emits a curated JSONL corpus
plus a replayable decision bundle. Model training and downstream evaluation are
outside the curation runtime. Their results cannot alter a frozen run.

The runtime forbids Utility, NLL, benchmark outcomes, source reputation,
source tier, target retention fractions, maximum token budgets, domain quotas,
and target domain distributions from all membership decisions.

## Core, Metric, Policy, Method

| Core | Metric or observable evidence | Active Policy | Method |
|---|---|---|---|
| Validity | Closed input-contract failures | normalized text integrity | `ingestion/candidate_processing.py` |
| Validity | BOM without payload change | BOM normalization | `ingestion/candidate_processing.py` |
| Validity | Empty/control-only chunk | chunk integrity | `run_curation.py` Stage-A chunk gate |
| Redundancy | Exact normalized digest | exact duplicate family | `run_curation.py` Stage-B exact family |
| Redundancy | Exact sentence occurrence family | intra-chunk sentence compaction | `repeated_sentence_compaction.py` |
| Redundancy | Exact, token-MinHash, and character-24-gram MinHash retrieval followed by bounded equivalence or containment witnesses | verified duplicate family | `redundancy_v2_retrieval.py`, `redundancy_equivalence.py`, `redundancy_mode_policy.py` |
| Redundancy | Identical structural scaffold signature | scaffold representative | `stage_b_policy.py` |
| Quality | Explicit generated-and-do-not-edit evidence | generated artifact rejection | `quality_retention.py` |
| Quality | Comment-only license payload | license-only rejection | `quality_retention.py` |
| Quality | Empty complete HTML shell | empty-shell rejection | `quality_retention.py` |
| Quality | Closed cookie-control text only | web-chrome rejection | `quality_retention.py` |
| Quality | Independent Q1 correctness, Q2 coherence, Q3 substantive payload, Q4 learnable-relation evidence | positive-support gate with Luna Batch fallback | `quality_ranker_runtime.py`, `quality_operating_points.py`, `quality_fallback_evidence.py` |
| Coverage | Representative survival and shared reciprocal semantic support | Coverage veto and explicit restoration | `coverage_engine.py`, `semantic_coverage_materializer.py` |

`configs/runtime_policy_registry_v1.json` is the executable inventory. A Policy
is not deployable unless its implementation and positive/false-positive
fixtures resolve inside the release surface.

## Stage Semantics

### Stage A: Validity

Input adaptation and deterministic normalization run first. Stage A then
quarantines closed record failures and rejects empty or control-only chunks.
Short but nonempty chunks are not rejected merely for being short.

### Stage B: Redundancy and Quality

Redundancy decisions require an exact identity or an explicit witness with a
stable representative link. Retrieval similarity alone has no deletion
authority. Exact repeated sentences may be removed span-by-span only when one
occurrence and the residual chunk survive.

Candidate retrieval uses token MinHash plus a densified one-permutation
character 24-gram MinHash. LSH buckets emit stable-anchor star candidates, so
bucket expansion is linear rather than all-pairs. Every emitted pair is then
rechecked by the typed equivalence/containment verifier; a retrieval collision
that lacks a witness is retained.

Structural Quality rules reject only closed non-payload cases. The frozen
ranker emits independent pass, fail, or abstain evidence for Q1-Q4. A qualified
local failure removes the chunk. Q2, Q3, and Q4 must all pass to retain it;
Q1 acts as a qualified-failure veto. All other local outcomes require
hash-bound GPT-5.6 Luna Batch evidence.
The fallback uses the same positive-support rule: confirmed failure or a
completed observation without positive support removes the chunk. Missing,
invalid, or transport-failed evidence stops the run without changing corpus
membership. Coverage can veto and restore a proposed Quality removal.

### Stage C: Coverage

Coverage cannot create a new deletion, rank chunks, or impose a quota. It
checks representative families and semantic support groups after all Stage-B
proposals. When a supported group would disappear without an authorized
explanation, Stage C restores an explicit chunk, records the evidence hash and
selection method, rematerializes the corpus, and reruns the complete check.

Route, script, morphology, and domain composition are reported after curation
for transparency. Those labels are never optimization targets.

## Model Artifacts

Automatic mode requires:

- a frozen local Quality-ranker manifest;
- `Qwen/Qwen3-Embedding-0.6B` at its declared revision;
- `BAAI/bge-m3` at its declared revision.

The deployment runtime makes no online teacher/API call. Unsupported local
cases must have completed Luna Batch observations before the final replay.
Batch submission is a support workflow outside membership materialization;
the runtime accepts only observations bound to the exact chunk text hash.

## Verification

The release manifest fixes the exact runtime, config, and validation files.
Run the following from the release root:

```powershell
python validation\test_active_surface.py
python validation\test_deployment_surface_v1.py
python validation\test_curation_runtime.py
python validation\test_policy_registry_contract.py
python validation\test_policy_fixture_contract.py
python validation\test_redundancy_v2.py
python validation\test_quality_ranker_runtime_v1.py
python validation\test_semantic_coverage_materializer_v1.py
```

Passing these checks proves declared behavior, trace completeness, and package
closure. It does not prove universal downstream improvement or production
detector coverage for PII, licensing, benchmark contamination, or poisoning.
