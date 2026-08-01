# Block 8: Corpus Suitability Decision v1

## Decision

The format-preserving The Stack-based 7M Code corpus is retained as a
**clean-control corpus**, not as the primary corpus for demonstrating large
structural compression.

The decision is about the observed input, not a claim that The Stack is
intrinsically high quality. Its corrected v2 replay exposes little removal
opportunity under the current auditable policy boundary:

| Evidence | Corrected v2 result |
| --- | ---: |
| Input records | 4,890 |
| Stage-A release records | 4,889 |
| Stage-B pass chunks | 8,024 |
| Normal Curated chunks | 7,984 |
| Authorized Stage-C removals | 40 |
| Explicit generated-artifact removals | 39 |
| License-comment-only removals | 1 |
| Raw-to-Curated Qwen text-token removal | 52,190 / 0.748% |
| Remaining minified-like diagnostic chunks | pending v3 rerun; v2 diagnostic is historical only |
| Retired repeated-label candidate spans | 0 |

Coverage passed. The 25 remaining diagnostic chunks are static literal and
data-table forms, not a justified default deletion class. The previous larger
apparent minified surface was caused mainly by the corrected formatting-loss
bug and is not evidence of removable payload.

## Role Separation

| Corpus role | Question it answers | Expected outcome | Prohibited interpretation |
| --- | --- | --- | --- |
| Clean control: corrected The Stack Code | Does the framework abstain when no closed rule has authority? | Small removal volume; explicit reasons only | Small removal does not prove the corpus is universally high quality |
| Raw-like stress corpus: future Block 9 | Can the same Core-Policy interface compress real, observed structural artifacts while preserving valid payload? | Reason-coded removal opportunity and Coverage preservation | A target removal fraction is not a success criterion |

The two corpora must remain separate. Mixing them merely to reach a desired
deletion rate would hide whether a policy works on genuine raw structure or
only on a constructed mixture.

## Block 9 Input Contract

The raw-like stress corpus must satisfy all of the following before candidate
rule design begins:

1. **Acquisition fidelity:** retain the collected text snapshot and stable
   record identifier; do not pre-filter by model score, benchmark result,
   source reputation, target token count, or desired curation outcome.
2. **Observed opportunity:** record the actual frequency of explicit artifacts,
   duplicates, malformed acquisition results, and other candidate structures
   after Stage A/B. Do not inject synthetic noise into the confirmatory corpus.
3. **Audit separation:** source, path, license, and collection metadata may be
   retained for traceability and false-positive analysis, but cannot be passed
   to selector-visible runtime input unless a separate source-dependent policy
   is explicitly proposed and validated.
4. **Development-confirmatory split:** divide records by stable record ID and
   normalized-text hash before rule tuning. Development artifacts cannot enter
   confirmatory training or benchmark snapshots.
5. **Benchmark exclusion:** complete contamination audit before any external
   training materialization.
6. **No compression target:** retain the natural output of frozen policies. A
   candidate is evaluated by its reason trace, false-positive boundary,
   Coverage invariants, and external natural-budget outcome, never by reaching
   a prescribed token fraction.

## Consequence For Claims

The corrected The Stack result supports a bounded abstention claim: the
framework does not manufacture a deletion decision when explicit structural
evidence is absent. It cannot by itself support a claim of aggressive
compression or downstream improvement. Those require a separately frozen
raw-like corpus and the subsequent candidate-development and external
evaluation blocks.
