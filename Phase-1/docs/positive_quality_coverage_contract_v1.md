# Positive Quality And Coverage Contract V1

> Historical design contract. The responsibility boundary was superseded by
> `docs/content_routing_quality_coverage_contract_v2.md`, which separates the
> shared Content Router from Coverage and moves structural coherence to
> Validity. This file is preserved for experiment traceability.

## Status

This document is historical and has no runtime authority. The current
implementation is `configs/curation_contract.json` plus
`docs/quality_coverage_formal_definition.md`, where Quality is a positive
selection gate and an unmatched or abstaining chunk is not selected unless the
Stage-C Coverage veto restores it.

Activation requires the implementation, calibration, fixtures, development
ablations, and frozen confirmatory evaluation named in
`configs/positive_quality_coverage_contract_v1.json`. Until those gates close,
the current Normal output and historical results keep their existing meaning.

## Core Definitions

| Core | Frozen authority |
|---|---|
| Validity | Reject only records or chunks that cannot be interpreted as declared text payload under closed integrity rules. |
| Redundancy | Reduce repeated learning payload while retaining a deterministic representative for every removed family. |
| Quality | Decide whether a distinct valid unit has positive, reproducible evidence that makes it eligible for LM training. This is not an intrinsic-goodness claim. |
| Coverage | Classify and audit representation across declared axes. It cannot assign cross-domain importance, impose quotas, or rescue a Quality reject. |

## Quality Decisions

Quality uses route-specific positive evidence instead of one global weighted
score. The declared routes are General Prose, Code, Math, Technical
Documentation, Conversation/Instruction, and Unknown. Multiple routes may be
assigned to one chunk.

| Decision | Contract |
|---|---|
| `eligible_keep` | At least one known route and every mandatory evidence head pass their frozen route-specific thresholds. |
| `reject` | A named active negative policy establishes explicit non-payload evidence, or a calibrated in-scope route passes a separately validated low-evidence rejection boundary. |
| `abstain` | Evidence is incomplete, conflicting, out of distribution, or between the keep and reject boundaries. |

Every known route must provide route confidence, substantive-payload evidence,
coherence/completeness evidence, and route-specific evidence. These values are
combined conjunctively. A formula such as a weighted sum of length, lexical
diversity, or other proxies is not permitted.

Evidence values remain on the frozen provider's native finite scale. They are
not forced through sigmoid, min-max normalization, or another undocumented
conversion merely to resemble probabilities. Thresholds are calibrated on the
same provider version and scale, and scores from different providers are never
added or compared directly.

Normal retains `eligible_keep` and `abstain`. Hard retains only
`eligible_keep`. Both modes write every excluded or abstained chunk to a
reason-coded not-selected artifact; neither silently deletes source data.

## Provider Evidence Audit

The decision gate and calibrator are implemented, but a score is not accepted
merely because a paper or checkpoint calls it Quality. The provider registry at
`configs/positive_quality_provider_registry_v1.json` records the training
objective, supported evidence heads, limitations, and local calibration duty
for every candidate.

| Route | Public candidate evidence | Supported portion | Current action |
|---|---|---|---|
| General Prose | [QuRater](https://huggingface.co/princeton-nlp/QuRater-1.3B), [DCLM](https://github.com/mlfoundations/dclm), [FineWeb-Edu](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | Current routing precondition or two-head Quality evidence failed source-transfer or semantic-stress gates | Runtime abstain |
| Code | [Stack-Edu Python classifier](https://huggingface.co/HuggingFaceTB/stack-edu-classifier-python) | Python complete-source bundle scored; strict source-balanced calibration failed | Abstain |
| Math | [OpenWebMath MathScore](https://huggingface.co/open-web-math/filtering-models) + [FineMath classifier](https://huggingface.co/HuggingFaceTB/finemath-classifier) | Route cascade and educational-reasoning evidence; two structural heads missing | Abstain |
| Technical Documentation | No dedicated validated bundle identified | None | Abstain |
| Conversation/Instruction | [DEITA](https://github.com/hkust-nlp/deita) | Instruction-tuning pair quality; objective mismatch | Abstain |
| Unknown | No provider is permitted | None | Abstain |

No public checkpoint alone supplies both current mandatory Quality heads behind
the routing precondition under the continued-pretraining objective. A candidate score cannot be copied into
missing heads, and paper-level results cannot replace source- and dataset-
disjoint local calibration. Block 3 is therefore split honestly: 3A (gate,
calibrator, and provider audit) is complete; 3B has one calibrated General
informational-prose candidate and one scored but calibration-blocked Python Code
candidate and one scored but structurally incomplete Math candidate. All
runtime activation gates remain closed.

### General Informational Prose Development Result

QuRater revision `bd61c778...` is connected only as an inactive English
informational-prose candidate. Its raw dimensions were mapped before observing
the full development distribution: Facts/Trivia to substantive payload,
Writing Style to coherence only, and Educational Value to route-specific
evidence. Required Expertise is audit-only. This mapping explicitly does not
claim that Writing Style measures document completeness.

| Quantity | Normal | Hard |
|---|---:|---:|
| Selected clean-control profile | `clean_q0` | `clean_q0.01` |
| Route candidate tokens separated | 83,063 / 783,876 (10.60%) | 198,962 / 783,876 (25.38%) |
| 95% clean-control false-reject upper bound | 0.42% | 3.91% |

The clean pool contained 642 route-eligible chunks from 1,000 FineWeb-Edu
controls; the candidate pool contained 1,285 route-eligible chunks from 1,685
broad FineWeb chunks. Out-of-route chunks are reported to Coverage and are not
counted as General-route compression. The frozen machine artifact is
`configs/qurater_general_prose_development_bundle_v1.json`. Adversarial,
provider-bias, leave-route-out, Stage-C integration, and external evaluation
gates remain open, so runtime continues to abstain.

### Python Complete-Source Development Result

The Code candidate uses four non-interchangeable heads. An authoritative
row-level Python declaration plus complete-source shape establishes route
scope; an AST semantic statement establishes substantive payload; AST parse
success establishes coherence; and frozen Stack-Edu-Python supplies only the
educational-code head. The provider score is never copied into the other three
heads. Snippets, unknown languages, non-Python files, parser failures, and
unsupported formats abstain rather than become negative examples.

The ingestion regression fix is material: the original 10M GitHub-Code slice
stored all 5,134 records as `und`, even though the upstream dataset exposes a
row-level `language` field. The new frozen slice preserves that declaration:

| Quantity | Result |
|---|---:|
| Whole corpus | 4,847 files / 10,008,758 exact Qwen3-4B tokens |
| Complete declared Python | 323 files / 714,283 tokens |
| AST payload + coherence passed and provider-scored | 279 files / 501,400 tokens |
| Published threshold candidate 2 | 173 files / 321,948 tokens (45.07% of Python tokens) |
| Published threshold candidate 3 | 30 files / 72,077 tokens (10.09% of Python tokens) |

The threshold rows reproduce candidate settings supported by the Stack-Edu
paper and model card; they are not general-code Normal/Hard policies. The
primary evidence reports Python binary F1 of 0.9438 at minimum 2 and 0.8018 at
minimum 3, but the model was trained on The Stack v2 annotations, has a
1,024-token context, and warns of OOD and comment-density bias.

Calibration used four permissively licensed Python repositories created after
the Stack v2 source snapshot: 1,299 complete files / 3,709,176 tokens, with
zero stable-ID or normalized-text overlap against the candidate pool.
Published threshold 2 falsely rejected 338 controls and threshold 3 rejected
1,212. Lower source-balanced thresholds were also insufficient: the least
strict threshold rejected zero pooled controls and 1/279 candidate records,
but its worst leave-one-repository-out Wilson upper bound was 6.61%, above the
5% Hard tolerance. No Normal or Hard profile was selected. Stack-Edu therefore
has no general-code runtime authority; the negative result is frozen in
`configs/stack_edu_python_calibration_report_v1.json` and
`configs/stack_edu_python_development_bundle_v1.json`.

### Math Development Result

The Math candidate keeps the two public classifiers in their documented roles.
OpenWebMath first recalls closed explicit math notation; only pages without
that evidence rely on MathScore, matching the paper's prefilter cascade rather
than applying `0.8` globally. FineMath supplies a distinct 0-to-5 regression
score for educational mathematical reasoning and deduction. Neither score is
copied into substantive payload or coherence completeness.

| Quantity | Result |
|---|---:|
| Whole corpus | 3,936 pages / 5,000,169 Qwen3-4B tokens |
| Closed explicit-notation evidence | 2,763 pages / 3,922,835 tokens |
| No notation and MathScore at least 0.8 | 381 pages / 162,472 tokens |
| Combined route cascade | 3,144 pages / 4,085,307 tokens |
| Cascade plus published FineMath 3+ diagnostic | 1,521 pages / 1,399,842 tokens |
| Cascade plus published FineMath 4+ diagnostic | 557 pages / 501,107 tokens |

The `3+` and `4+` rows reproduce FineMath dataset settings, not locally
calibrated policies. The learned coherence candidate failed source transfer and
was replaced by explicit structural guard v2. On 2,574 previously observed
development controls, v2 flagged 3 records/3,157 tokens, passed every source's
5% Wilson false-reject gate, and detected all six registered corruption
families. It diagnosed 117 candidate records/470,003 tokens, but this is an
inactive opportunity audit rather than a curated result. Four-head v5 still
selected no profile: q0 leave-one-source-out failures were 7 payload plus 1
route failure on 59 Hefferon controls and 1 route-specific failure on 328 CLRS
controls; coherence contributed zero. Math therefore remains `ABSTAIN` pending
revised payload evidence, provider-bias fixtures, and external validation. The
coherence candidate itself subsequently passed a post-freeze Winitzki control
with 0/131 false rejects over 300,894 tokens and all six corruption fixtures.
An OpenStax Physics stress source contained 19 objective C1 control-character
defects, so it is retained as diagnostic corruption evidence rather than being
counted as clean false-reject controls. The decision is frozen in
`configs/math_quality_evidence_decision_v3.json`.

## Threshold Calibration

Thresholds are selected independently per route and evidence head. The
selection objective is the most compressive feasible candidate, not a retained
token fraction. At 95% confidence, the upper bound on clean-control false
rejection must be at most 1% for Normal and 5% for Hard. These are declared risk
tolerances, not claims about intrinsic Quality.

Calibration must hold out source and dataset identity, include a leave-route-out
stress test, and preserve an Unknown/OOD outcome. If no threshold satisfies the
constraints, that route abstains and receives no removal authority. A frozen
confirmatory result cannot be used to retune thresholds.

## Coverage Contract

Coverage classifies four independent multi-label axes with an explicit Unknown
value: semantic domain, language/script, format/genre, and content morphology.
It reports document and token shares, stratum retention, semantic-cluster
survival, tail loss, Jensen-Shannon divergence, eligible-representative
presence, and Unknown/OOD rates before and after curation.

Coverage does not decide that Code is more important than Math, that a rare
cluster is automatically valuable, or that a target percentage must be kept.
Within an already eligible pool, representative choice is lexicographic:
eligible status, completeness evidence, cluster-medoid distance, then stable
record ID. A Quality reject cannot be rescued to satisfy Coverage.

Block 2 implements the text-only deterministic taxonomy in
`coverage_taxonomy.py`, with its machine-readable registry in
`configs/coverage_taxonomy_v1.json`. The classifier is active only as audit
metadata and is never selector-visible. `unknown` means that no closed evidence
rule fired; it is not a low-Quality label and does not pretend to distinguish
novel content from every possible out-of-distribution format. The report uses
multi-label incidence shares, so shares may sum above one, and records
Raw-to-Curated Jensen-Shannon divergence separately for each axis.

## Stage Contract

```text
Stage A: record Validity
Stage B: chunk Validity and exact Redundancy
Stage C-1: Coverage pre-tagging
Stage C-2: positive-retention Quality gate
Stage C-3: near-duplicate and scaffold representative resolution
Stage C-4: Coverage post-audit and reason-coded materialization
```

Quality precedes non-exact family resolution so a low-evidence family member
cannot become the stable representative while an eligible member is discarded.
Exact duplicates remain in Stage B because identical normalized payload has the
same Quality evidence.

External natural-budget training and benchmarks remain outside A-B-C. They may
validate a frozen profile but cannot be read by runtime policies or feed back
into the same confirmatory cycle.
