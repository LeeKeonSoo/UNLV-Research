# Repeated Label-Block Development Matrix v1

## Scope

This is a candidate-only Redundancy audit for
`stage_c_repeated_label_block_candidate`. It reads only frozen Stage-B chunk
text and the frozen Qwen3-4B tokenizer. It cannot modify Normal or Hard,
consume source identity, composition, Utility, NLL, benchmark outcomes, or a
target retention fraction.

The candidate removes only a later exact occurrence of a closed-set
navigation-marker block within the same chunk. It preserves the first
occurrence, the complete chunk, and a Stage-B-valid residual.

## Frozen Inputs And Results

| Corpus | Stage-B chunks | Input tokenizer tokens | Candidate spans | Token delta | Coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| Code | 8,058 | 6,873,133 | 0 | 0 | pass |
| Math | 3,632 | 2,915,236 | 0 | 0 | pass |
| General raw web | 707 | 940,958 | 0 | 0 | pass |

Reports are stored outside Git at:

- `D:/UNLV-Research/repeated_label_block_development_matrix_v1/code_report.json`
- `D:/UNLV-Research/repeated_label_block_development_matrix_v1/math_report.json`
- `D:/UNLV-Research/repeated_label_block_development_matrix_v1/general_report.json`

Each report records the frozen input hash, frozen tokenizer path, reason-code
impact audit, and the following Coverage invariants: whole-chunk preservation,
Stage-B residual preservation, and earlier-in-same-chunk representative
linkage.

## Decision

The closed trigger is safe on its explicit fixtures but has **zero observed
opportunity** across all three frozen development corpora. It therefore has no
evidence of useful compression and must not be promoted to Normal or Hard.
Q8 must either archive the candidate as an inactive research artifact or
discover a different, independently bounded redundancy pattern; it may not
widen this trigger based on token-reduction desire or downstream benchmarks.

## Supersession Notice

This v1 matrix used a Stage-B snapshot later found to flatten long code
paragraphs. It remains a provenance artifact only. The format-preserving Code
rerun is `D:/UNLV-Research/code_5m_corpus_v2/format_preserving_7m_v2/repeated_label_block_matrix_v2.json`:
7,309 chunks, 6,400,945 frozen-tokenizer tokens, zero candidate spans, zero
token delta, and all Coverage invariants passed. The policy conclusion did not
change: it has no observed compression opportunity and cannot be promoted.
