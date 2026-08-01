# Historical Selector Forensics

## Status

Historical diagnostic only. This document does not reactivate the former
selector and does not authorize a runtime deletion rule.

## Question

What were the distinct historical mechanisms behind aggressive Code subset
selection, and what can the current framework learn from their downstream
signal?

## Reconstructed Mechanism

The archive contains two distinct mechanisms that must not be conflated.

1. The earlier `temporal_code_curation_protocol_v1` declared a binding
   token-proxy budget equal to 40% of the Stage-A-pass pool. It ranked chunks
   by the formula below and packed chunks until the budget was exhausted.
2. The later `abc_curation_operational_v1` replay used the same style of
   priority features but a minimum score threshold of `0.8`, with no binding
   token budget. The materialized JSONL retained approximately half of its
   token proxy, not 40%.

Both mechanisms used the following score family:

```text
code_quality_proxy =
  0.45 * length_support
+ 0.35 * structural_richness
+ 0.20 * lexical_or_identifier_diversity
- 0.25 * pass_through_assignment_ratio

stage_b_objective =
  0.80 * code_quality_proxy
+ 0.20 * (1 - soft_redundancy_risk)
```

Coverage constraints reserved at least one representative for each observed
bundle, content type, change type, path family, and difficulty bucket. A
second distribution constraint preserved a partial token share for selected
bundle, content-type, and difficulty buckets. The remaining budget was filled
by descending objective score. Chunks that did not fit the binding budget were
marked `budget_not_selected`; that disposition was not a defect, rejection, or
quality label.

In the earlier budget experiment, the exact 40% cap and whole-chunk packing
explain why the retained share could fall below 40%: an otherwise eligible
chunk was skipped if it exceeded the remaining capacity. The original
40%-budget JSONL has not been retained in the current D: output tree, so its
exact selected/rejected composition must be regenerated before it can be used
as numerical evidence.

## Observed Threshold-Replay Counts

Source: `D:/UNLV-Research/code_5m_corpus_v2/final_replay_v1/abc_curation_operational_v1/curation_report.json`.
These counts belong to the later threshold replay, not the earlier 40%-budget
experiment.

| Historical outcome | Chunks |
| --- | ---: |
| Stage-B hard-gate pass | 8,092 |
| Selected at the priority threshold | 5,350 |
| Priority-threshold rejected | 2,723 |
| Explicit near-duplicate removals | 5 |
| Explicit scaffold-family removals | 14 |
| Weighted-priority removals | 2,723 |
| Hard-gate rejections | 2 |

The threshold replay's compression was driven overwhelmingly by weighted
priority thresholding, not by explicit artifact, validity, or redundancy
removals. The same qualitative warning applies to the earlier 40%-budget
experiment, but its exact artifact counts are pending regeneration.

## What The Proxy-Rejected Group Actually Contains

Source: `D:/UNLV-Research/code_5m_corpus_v2/final_replay_v1/proxy_removal_forensics/historical_proxy_forensics.json`,
`historical_proxy_rejected_inventory.json`, and
`historical_selector_decomposition_v1.json`. These are threshold-replay
artifacts.

| Observable candidate in 2,723 proxy-rejected chunks | Chunks | Interpretation |
| --- | ---: | --- |
| One- or two-line minified candidate | 101 | Candidate only; long legitimate generated literals and chunk boundaries are false-positive risks. |
| License/comment-only candidate | 27 | Explicit non-payload candidate, subject to context and residual-payload checks. |
| Strong generated marker | 3 | Candidate only when the declaration is explicit and the payload contract is satisfied. |
| Generic generated marker | 89 | Not a deletion reason: markers also occur in useful authored source. |
| Python syntax error after chunking | 1,853 | Not a validity failure: partial chunks of valid files are expected to be unparsable. |

The current v3 policy retains 2,681 of the 2,723 historical proxy-rejected
chunks (98.46%). This establishes that the former score cannot be reinterpreted
as a disguised artifact detector.

### Token-Level Decomposition Of The Threshold Replay

The new decomposition reports exclusive families, so that a chunk is not
counted twice. It confirms that most of the historical reduction cannot be
recovered merely by turning obvious artifact checks into active rules.

| Family | Selected token proxy | Rejected token proxy | Interpretation |
| --- | ---: | ---: | --- |
| Explicit license/comment-only candidate | 0 | 11,951 | Small, explainable candidate family. |
| Explicit generated-and-do-not-edit candidate | 1,016 | 961 | Too small and present on both sides; requires strict fixtures. |
| Minified-shape candidate | 4,732 | 41,589 | Potentially material but requires a false-positive audit. |
| Partial or unparsable chunk diagnostic | 955,771 | 968,697 | Nearly symmetric; cannot be a deletion rule because chunking splits valid source. |
| Proxy-only, no explicit evidence | 343,917 | 264,865 | The remaining unresolved selection signal. |

The selected and rejected groups contain 1,305,436 and 1,288,076 token-proxy
tokens respectively. The selected group has a higher recorded priority mean
(`0.862909` versus `0.744464`), but the family decomposition shows that this
difference is mainly an uncalibrated structural ranking, not a discovered
high-volume artifact class.

## What We Can Learn

The historical downstream result remains a development signal: a more compact
subset may train as well as, or better than, the full corpus. It does not show
that every omitted chunk was low quality. Across the two historical mechanisms,
the selector bundled a fixed resource cap or an uncalibrated score threshold,
a weighted proxy, soft redundancy pressure, and coverage constraints. Earlier
runs also had a format-preservation defect, so their training result is not
confirmatory evidence for the current runtime.

The actionable hypothesis is narrower: the corpus likely contains a large
amount of low-marginal-value material, but its removable families have not yet
been identified with sufficient precision.

## Required Decomposition Before A New Hard Profile

1. Treat the historical selected/proxy-rejected split as a discovery label
   only, never as a runtime target.
2. Cluster the proxy-rejected material by reproducible text and structure
   signatures, then compare each cluster with selected material.
3. For every high-prevalence cluster, write a candidate policy card with its
   input contract, reason code, non-trigger examples, false-positive fixture,
   and coverage effect.
4. Run every candidate independently and cumulatively on a development split.
   Measure removed tokens and collateral retention; do not tune against a
   benchmark result.
5. Freeze only candidates with acceptable fixture and coverage behavior into a
   Hard profile, then evaluate Raw, Normal, and Hard at their natural token
   budgets with benchmark-disjoint three-seed training.

No target retention fraction belongs in this process. If the validated Hard
rules remove 5%, 40%, or 60%, that is an observed outcome rather than a target
that forces removal.
