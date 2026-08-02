# Block 1 Target Contract Decision

Status: frozen design decision, 2026-08-02
Machine-readable authority: `protocols/target_aware_core_completion_v1.json`

## Decision

The first target-aware policy study will retain the existing
`Qwen/Qwen3-4B-Base` checkpoint and tokenizer. This avoids changing the curation
policy, target model, and evaluation protocol simultaneously. It also preserves
comparability with the project's existing natural-budget QLoRA runs.

The target is a general-purpose 4B base SLM under continued pretraining. The
framework does not claim to curate SFT, DPO, or reinforcement-learning data in
this study.

| Role | Frozen artifact | Revision |
| --- | --- | --- |
| Target model and tokenizer | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Optional Quality probe | `Qwen/Qwen3-8B-Base` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Optional semantic provider | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |

These are optional audit-only candidates, not default Core definitions or
active runtime authorities. Their snapshots, compatibility, precision error,
calibration, development validation, and confirmatory validation must pass
before either can contribute to a promoted policy.

Qwen is the frozen backend for this experiment, not a framework dependency.
Quality, semantic support, diagnostic Validity, and routing are versioned
provider slots. A user may replace a provider, but any change to its artifact,
revision, tokenizer, normalization, or output semantics resets it to
`audit_only`; calibration and confirmatory evidence cannot be inherited.

## Why These Models

The Qwen3 technical report describes a shared Qwen tokenizer across the dense
family and reports the 8B base model as a larger member of the same architecture
family. That makes 8B a controlled stronger-reference candidate for measuring
what the 4B target does not yet model well. The reference is not a truth oracle:
base-minus-reference excess loss remains an optional hypothesis. It has no
deletion authority unless its provider and a separately reason-coded policy
complete the full promotion lifecycle.

Qwen3-Embedding-0.6B is small enough for corpus-scale use and its official
release targets multilingual, code, retrieval, classification, and clustering
work. It is still a learned representation with possible domain and length
bias. Coverage may use it only after stability and perturbation probes; route,
language, format, mixed, and unknown views remain independent checks.

Primary sources:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3-8B-Base model card](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [Qwen3 Embedding technical report](https://arxiv.org/abs/2506.05176)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Quality Target

Quality is expected marginal external-risk reduction per target-model training
token. It is not a universal document label. Provider-independent structural
evidence may be studied directly. Model evidence may be registered only as an
optional candidate, including:

1. closed structural non-payload evidence;
2. Qwen3-4B base loss per target token;
3. Qwen3-8B reference loss over the same token IDs;
4. base-minus-reference excess loss;
5. explicit uncertainty and unsupported-scope state.

High target-model loss by itself is ambiguous. It may mean useful novelty or
irreducible corruption. Positive excess loss is also only a candidate signal:
it says the larger same-family model predicts the content better, not that the
content will improve every downstream capability.

If that optional loss provider is executed, scoring is fixed to the target
tokenizer, 2,048-token context, EOS per record, no special-token injection, and
nats per non-padding target token. Primary likelihood evidence is bfloat16.
Quantized scores may be used only if a frozen probe bounds their error against
bfloat16. None of these settings makes the provider mandatory.

## Coverage Target

Coverage preserves support over redundancy families, route, language/script,
format/morphology, stable semantic/skill clusters, and unknown or mixed
intersections. It does not enforce domain percentages and never treats source
identity or rarity as value.

The encoder may propose semantic support, but a stable cluster becomes a
selection constraint only after bootstrap stability and cross-format bias tests.
Every compacted family must retain a final representative and no stable cluster
may disappear without independent Validity or confirmed negative-Quality
evidence.

## Training Contract

External validation compares three dataset arms:

1. **Base:** risk-eligible, Validity-passing data before Redundancy, Quality,
   and Coverage subset selection;
2. **Normal:** validated high-confidence removal policies, with uncertain units
   retained and Coverage constraints enforced;
3. **Hard:** validated stricter policies or a calibrated per-token-cost boundary,
   with the same Coverage constraints enforced.

The original Qwen3-4B-Base checkpoint is still benchmarked as an untrained model
reference, but it is not a fourth dataset arm. Primary comparisons are Normal
versus Base and Hard versus Base; Hard versus Normal is supporting evidence.

Every dataset arm receives one seed-conditioned deterministic pass over all of
its own materialized 2,048-token blocks. Seeds are `101`, `202`, and `303`.
Equal-token resampling, a target retained fraction, and benchmark-dependent
corpus changes remain forbidden.

The hardware-feasible method remains QLoRA continued pretraining: NF4 4-bit
base weights, bfloat16 compute, LoRA rank 32/alpha 64/dropout 0.05 on all linear
modules, AdamW at `5e-5`, constant learning rate, and gradient accumulation 8.
The protocol records current package versions as an environment reference; the
final execution image or lockfile must be hashed before confirmatory training.

## Capability Panels

Results remain vectors; one domain cannot compensate for another.

| Panel | Frozen suites |
| --- | --- |
| Code primary | HumanEval+, MBPP+, BigCodeBench Complete, CRUXEval-I, CRUXEval-O, DS-1000 |
| Math primary | GSM8K test, Hendrycks MATH test |
| Science reasoning guard | GPQA Diamond |
| General retention | MMLU-Pro, BBH, ARC-Challenge, HellaSwag |
| Multilingual retention | MGSM, MMMLU |

The Code snapshots already have hashes in the existing record-disjoint protocol.
Math, General, and Multilingual suite identities are frozen here, but their
dataset snapshots, evaluator revisions, task counts, and benchmark-exclusion
hashes must be materialized before a corresponding corpus can enter development
or confirmatory evaluation.

## Temporal Boundary

The official Qwen3 material reviewed for this decision describes its training
scale and composition but does not provide an auditable pretraining cutoff.
The May 2025 release date cannot be substituted for that cutoff. Therefore:

- the current study does not claim that input data is temporally novel to the
  base model;
- LiveCodeBench remains blocked as a post-cutoff test;
- the earlier idea of a 2024 model versus 2025/2026 data requires a separate
  protocol using a model with an authoritative cutoff.

This limitation does not block testing curation efficiency. It blocks only the
stronger temporal-new-knowledge claim.

## Risk Boundary

Rights and license eligibility, PII and secrets, known benchmark contamination,
and declared poisoning or malware risk belong to a separate input-eligibility
layer before A-B-C. That layer may quarantine data under its own versioned
contract, but none of those fields is Quality or Coverage evidence. The Valid
Raw training arm therefore means risk-eligible, Validity-passing raw payload;
it does not mean the uninspected bytes collected from a source.

## Block 1A And Block 2 Exit

The target model and tokenizer, optional candidate identities, training
interface, Base/Normal/Hard dataset arms, natural-budget rule, seeds, capability
panels, provider replacement lifecycle, and temporal claim boundary are fixed.
No target-aware selection policy has been activated.

Block 2 implements an audit-only corpus profiler. It emits input hashes,
streaming size statistics, exact-duplicate opportunities, deterministic routing
incidence, optional exact tokenizer counts, and provider lifecycle state. It
cannot execute provider scores, select, rank, delete, or write a curated
dataset. The machine contracts are `configs/model_provider_registry_v1.json`
and `configs/corpus_profiler_contract_v1.json`.
