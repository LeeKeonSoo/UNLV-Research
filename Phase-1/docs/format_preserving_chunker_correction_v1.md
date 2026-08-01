# Format-Preserving Stage-B Chunker Correction v1

## Finding

The former Stage-B chunker split a paragraph longer than `max_chunk_chars` by
calling `split()` and joining its words with spaces. That preserved lexical
tokens but destroyed line breaks and indentation. A traced Python source record
had 33 newlines at Raw and Stage A, but a long Stage-B fragment had zero.

This is a representation-preservation defect, not evidence that compacted code
is low Quality. Any historical Curated/Hard artifact produced through the old
chunker is retained for provenance but excluded from final external-evaluation
claims.

## Correction

`run_curation.chunk_text()` now partitions the original text by line units.
For an overlong physical line it uses fixed character slices, never whitespace
reconstruction. For every nonempty input, concatenating the produced chunks
reconstructs the original text exactly. Regression fixtures cover both a long
multi-line code paragraph and an unbroken long line.

## Format-Preserving 7M Normal Replay

The new replay uses the same audited Raw input and Normal policies, but writes
to a non-overlapping v2 path:

`D:/UNLV-Research/code_5m_corpus_v2/format_preserving_7m_v2/normal_curation`

| Measure | Result |
| --- | ---: |
| Stage-A release records | 4,889 |
| Stage-B pass chunks | 7,309 |
| Stage-C retained chunks | 7,271 |
| Stage-C removed chunks | 38 |
| Explicit generated-artifact removals | 37 |
| License-comment-only removals | 1 |
| Stage-C proxy tokens retained | 2,568,426 |

The exact frozen-tokenizer total belongs to the later new v2 training-input
materialization. It must not reuse a token block made from the flattened
historical Curated/Hard output.

## Repository-Code Context Correction

The first format-preserving replay still omitted the declared
`pii_context=repository_code` default. Stage A consequently applied the
general-text whitespace normalizer, which changed Code tokenization without a
deletion decision. That v2 replay is also historical only.

The final corrected replay is v3:

`D:/UNLV-Research/code_5m_corpus_v2/format_preserving_repository_code_7m_v3/normal_curation`

| Stage | Rows or chunks | Qwen3-4B text tokens |
| --- | ---: | ---: |
| Audited Raw | 4,890 records | 6,979,548 |
| Stage-A release | 4,889 records | 6,978,049 |
| Stage-B pass | 8,024 chunks | 6,978,054 |
| Stage-C Curated | 7,984 chunks | 6,927,358 |

The framework therefore removes **52,190 text tokens (0.748%)** from Raw to
Curated. Stage C accounts for 50,696 of those tokens: 50,677 from 39 explicit
generated-artifact chunks and 19 from one license-comment-only chunk. Stage A
quarantines one 111-token acquisition-failure record; the small remaining
difference comes from the Stage-B exact-duplicate rejection and tokenizer
chunk-boundary accounting.

These counts use `Qwen/Qwen3-4B-Base` with `add_special_tokens=false`. The
later training materialization adds one EOS token per materialized row and
must be regenerated from v3 rather than reusing any v1/v2 block artifact.

## R1 Opportunity Re-Audit

On the corrected Stage-B snapshot, the diagnostic audit found:

| Signal | Chunks | Proxy tokens | Decision |
| --- | ---: | ---: | --- |
| One/two-line minified-like text | 26 | 13,129 | Candidate discovery only; blocked pending a closed false-positive boundary |
| Path-based vendor/generated pattern | 0 | 0 | No opportunity remains after format preservation |
| Explicit generated marker | present in already handled source records | n/a | Active generated-artifact policy already accounts for the 37 removed chunks |
| Repeated closed navigation-label block | 0 | 0 | Candidate remains non-runtime; no promotion evidence |

The former flattened snapshot reported 116 minified-like chunks and 31 path
candidates. That difference is evidence that formatting loss inflated the
apparent deletion opportunity. The format-preserving audit therefore does not
authorize a stronger deletion policy.
