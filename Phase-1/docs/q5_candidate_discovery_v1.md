# Q5 Candidate Discovery v1

## Method

The Common Crawl WET development source was read as UTF-8 JSONL. Candidate
discovery used only record text and did not read source identity, domain labels,
benchmarks, NLL, Utility, or a retention budget.

## Findings

| Structural finding | Records | Core classification | Decision |
| --- | ---: | --- | --- |
| Explicit error marker (`404 page not found`, `page not found`, or `nothing found for`) plus at least three exact navigation lines | 1 / 514 | Quality candidate | Do not activate: one observed record is insufficient to establish a general boundary. |
| Exact normalized nontrivial line repeated at least three times in one record | 161 / 514 | Redundancy candidate | Extend the existing exact repeated-template research only after false-positive fixtures distinguish navigation duplication from repeated quotations, tables, code, and legitimate reference entries. |
| Generic footer/navigation words (`Home`, `About`, `Subscribe`, `Loading`) | Common | No deletion authority | Reject as a candidate basis: these words occur in substantive prose and multilingual documents. |
| Apparent mojibake in PowerShell display | Not present in UTF-8 source | Not a data condition | Reject. It was terminal code-page rendering, not text corruption. |

## Q5 Decision

No new Quality rule is promoted. The evidence supports two narrowly scoped
research paths only:

1. an explicit error-shell candidate, held until it has multiple raw-corpus
   positives and adversarial article-about-errors fixtures;
2. a line-level exact-repetition candidate under Redundancy, not Quality.

The second path is more promising for compression, but it must preserve one
stable occurrence and must not delete an entire record. It therefore remains
separate from the Quality Core.

## Q6 Narrow Candidate Package

`stage_c_repeated_label_block_candidate` now records the most conservative
version of the second path under **Redundancy**. It considers only a later
exact repeat of a contiguous block whose labels all belong to a closed
navigation-marker set. The first occurrence and the complete chunk remain;
materialization is allowed only when the residual meets the declared Stage-B
minimum length. Repeated headings, quotations, tables, code, reference lists,
and test matrices are executable non-trigger fixtures.

The policy is candidate-only and has no Normal or Hard runtime authority.
Its purpose is to make the uncertainty testable through a rule-on/off
development matrix, not to infer that arbitrary repeated prose is removable.
