# Code 7M Dataset Integrity Audit

Audit date: 2026-08-10

## Scope

This audit checks the frozen Code-7M Raw input and the Luna Normal/Hard outputs.
It does not change selection membership or read benchmark outcomes.

| Arm | Records or chunks | Exact Qwen stream tokens |
|---|---:|---:|
| Raw | 4,890 records | 6,984,438 |
| Normal | 7,147 chunks | 6,125,213 |
| Hard | 5,859 chunks | 5,032,400 |

## Passed Integrity Checks

- Raw record IDs are unique: 4,890/4,890.
- Every `provenance.normalized_sha256` matches the stored UTF-8 text.
- Curated chunk IDs are unique in both modes.
- Every curated `stage_a_record_id` resolves to a frozen Raw record.
- Every curated text is an exact substring of its parent Raw text; active span
  transformation count is zero.
- Normal and Hard contain no normalized exact duplicate text.
- Hard is an exact UID subset of Normal; Hard-only chunk count is zero.
- Raw contains one normalized exact duplicate family and no cross-source exact
  duplicate family. Stage B records the nonrepresentative removal.
- Benchmark exclusion removed 12 of 4,902 candidates against the seven frozen
  suites by exact normalized segment or shared 16-token shingle.
- Every source, arm, and packed-block SHA-256 in the training-input report
  matches the file on disk.
- Packed tensors have the declared shapes: Raw `3408 x 2048`, Normal
  `2984 x 2048`, and Hard `2456 x 2048`. Their tensor sizes exactly equal the
  reported materialized-token counts, and no arm contains a repeated complete
  2,048-token block.

## Source Composition

| Source | Raw tokens | Normal tokens | Normal retention | Hard tokens | Hard retention |
|---|---:|---:|---:|---:|---:|
| `bigcode/the-stack-dedup` | 4,723,925 | 4,045,511 | 85.64% | 3,278,435 | 69.40% |
| GitHub reference pool | 2,260,513 | 2,079,702 | 92.00% | 1,753,965 | 77.59% |
| **Total** | **6,984,438** | **6,125,213** | **87.70%** | **5,032,400** | **72.05%** |

The selector does not read source identity. The higher reference-pool retention
is nevertheless a source-correlated outcome and must be disclosed. It can arise
from document length, repository style, model familiarity, or genuine payload
differences; the current experiment does not identify a causal explanation.

The chunk-level audit shows the same shift using the non-training whitespace
token proxy. These values are internally comparable because every column uses
the same chunk unit and token proxy.

| Stage | Stack chunks | Reference chunks | Stack token share | Reference token share |
|---|---:|---:|---:|---:|
| Stage-B eligible | 5,969 | 2,055 | 63.98% | 36.02% |
| Normal curated | 5,252 | 1,895 | 62.60% | 37.40% |
| Hard curated | 4,294 | 1,565 | 61.94% | 38.06% |

Reference records are 13.54% of Raw records but are longer and become 25.61%
of Stage-B chunks. Record counts, chunk counts, and tokenizer token shares must
therefore never be presented as interchangeable composition denominators.

## Comparable Route Composition

The deterministic route classifier is an explanatory heuristic, not a ground
truth domain label or selector input. On the comparable Stage-B-to-Stage-C
chunk view, the largest exclusive primary routes are:

| Stage | Code artifact | Mixed | General prose | Other routes |
|---|---:|---:|---:|---:|
| Stage-B eligible | 49.90% | 43.42% | 5.61% | 1.06% |
| Normal curated | 49.84% | 44.33% | 4.98% | 0.85% |
| Hard curated | 49.56% | 44.46% | 5.23% | 0.76% |

The large `mixed` share is expected for repository files containing code plus
comments, docstrings, or markup. It is not evidence that 43-44% of the corpus
belongs to a separate semantic domain.

## Confirmed Defects

### 1. Cross-unit composition delta

The 2026-08-08 `raw_curated_composition_delta.csv` compares 4,890 whole Raw
records with selected chunks. Chunking changes the classifier's unit and can
make a label gain tokens even though no text was created. Those historical
deltas are not valid conservation or causal selection reports.

The runtime now keeps Raw and Stage-A record composition as descriptive views,
does not emit record-to-chunk deltas or divergences, and emits the comparable
delta only from Stage-B eligible chunks to Stage-C curated chunks. Curated IDs
must be a subset of the eligible baseline IDs.

### 2. Reference metadata is not self-contained

All 662 GitHub reference records have `rights.license=unknown` and
`partition.source_content_sha256=unknown` in the frozen JSONL. Repository-level
licenses and commits are recorded in `docs/code_7m_corpus_provenance.md`, but an
external document is not a substitute for record-level release metadata.
Selection did not consume these fields, so this does not explain benchmark
results. It does block a self-contained public dataset release until a
hash-bound metadata repair is materialized.

### 3. Stage-A acquisition-failure false positive

The pre-fix run quarantined one valid Django source file because its code
contained `Page Not Found` and `Internal Server Error`. The record contains 112
Qwen tokens including EOS, or 0.0016% of Raw. The corrected rule accepts normal
mentions and requires explicit acquisition-failure metadata, a matching error
HTTP status, or an exact short failure body.

The existing Normal/Hard training artifacts remain a frozen pre-fix experiment.
They must not be described as byte-identical outputs of the corrected runtime.

## Experimental Confounds

- Raw is a designed mixture, not a purely raw corpus: 67.64% The Stack dedup
  and 32.36% established-project snapshots by exact tokens.
- The corpus has no auditable end-to-end temporal boundary relative to the
  Qwen3-4B-Base pretraining cutoff. Collection timestamps and model release
  dates are not substitutes for source snapshot dates or a declared model
  cutoff. This experiment therefore cannot support a 2024-model versus
  2025/2026-new-data claim or a post-cutoff LiveCodeBench claim.
- The Stack source is already deduplicated, reducing the opportunity for the
  Redundancy Core to demonstrate large removal.
- Reference files are much longer: median 6,283 characters versus 2,039 for
  The Stack records. Chunking therefore gives the reference pool a larger share
  of chunk-level decisions than its record share suggests.
- Raw training uses whole-record EOS boundaries, while Normal/Hard use chunk
  boundaries. A Chunked-All control is required to isolate membership selection
  from segmentation effects.
- Quality calibration and protected samples are UID/text-disjoint from each
  other but sampled from the same Code-7M corpus later curated. This is a
  corpus-fitted experiment, not independent evidence of cross-corpus Quality
  generalization.
- The benchmark audit establishes only its declared exact/shingle scope. It
  does not prove absence of semantic, translated, or shorter overlap.

The Quality split check found 1,024 calibration and 800 protected chunk UIDs,
with zero overlap and complete resolution into the 8,024 Stage-B chunks. Their
Normal/Hard retention rates were 85.55%/70.21% for calibration,
89.25%/74.25% for protected, and 89.63%/73.32% for the other 6,200 chunks.
This does not show preferential retention of calibration examples. It also
does not turn the protected sample into cross-corpus evidence: both sets came
from the same corpus being curated.

## Required Before Paper Freeze

1. Materialize a hash-bound record-metadata repair for the GitHub reference
   pool without changing text or selector inputs.
2. Regenerate Normal/Hard with the corrected Stage-A rule, or explicitly freeze
   the current models as a pre-fix sensitivity run and quantify the 112-token
   difference.
3. Add a Chunked-All natural-budget control for causal claims about curation.
4. Use only Stage-B-eligible-to-Curated composition deltas in figures and tables.
5. Report source-stratified retention and the mixed-corpus construction.
