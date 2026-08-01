# OpenWebText2 Raw-Like Failure Analysis

## Status

This is a diagnostic analysis of the current `paper_release_certification`
outputs. It does not change the Stage-B selector and does not use Utility as a
selector objective.

Reproduce the report with:

```bash
python 31_build_openwebtext2_slice_diagnostic.py
```

Generated evidence:

```text
outputs/validation/openwebtext2_slice_diagnostic.json
outputs/validation/openwebtext2_slice_diagnostic.md
```

## Observed Behavior

The full scored corpus separates into:

| Slice | Records | Quality | Redundancy risk | Repeat pressure | Technical-reference share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected | 163,331 | 0.928413 | 0.125237 | 0.467889 | 0.707355 |
| Stage-A usable, not selected | 994,186 | 0.630527 | 0.185119 | 0.575679 | 0.345060 |
| Stage-A rejected | 9,131 | 0.454828 | 0.812961 | 0.637998 | 0.371591 |

Stage A is behaving usefully: the rejected slice has much higher redundancy
risk, lower quality, and lower predictive proxy support.

Stage B also produces the intended Core-feature movement against the usable
unselected pool:

```text
quality:             +0.297886
redundancy risk:     -0.059882
structural validity: +0.010564
lexical diversity:   +0.017576
repeat pressure:     -0.107790
```

However, the selected subset also changes the corpus shape sharply:

```text
technical-reference share: 34.5% -> 70.7%
general-prose share:        35.1% -> 8.6%
conversational share:       19.8% -> 2.5%
```

## Current Diagnostic Hypotheses

### H-OWT2-01: Easy-NLL Strict-Baseline Confound

The canonical Stage-A and multi-matched baselines contain more repetition and
longer chunks, which can make them easier for a short-budget small-LM NLL probe
to learn even when the selected subset is stronger under the assigned Core
metrics.

Current status: supported by the certification-budget anti-memorization
diagnostic described below.

Required action: report the repeat-pressure/length matched control in Stage C
and revise strict-baseline interpretation before changing Stage B.

### H-OWT2-02: Reference-Quality Style Concentration

The reference-quality metric may strongly favor technical-reference text,
causing a selected subset that is cleaner but less representative of the usable
raw-web distribution.

Current status: supported as a diagnostic candidate by the 36.2 percentage
point technical-reference share increase.

Required automated test: evaluate Coverage and target-model outcomes by style
slice. Manual labels may be reported as an optional diagnostic only.

### H-OWT2-03: Residual Extraction Corruption

Preview-level mojibake rates are nearly identical for selected and usable
unselected records, so the current evidence does not support residual mojibake
as the primary transfer-gap explanation.

Required test: re-ingest a provenance-rich raw-web sample through Stage 0 and
compare before/after curation outcomes.

## Research Interpretation

OpenWebText2 is not evidence that the framework direction is invalid. It is the
first strong raw-like case showing that:

- Stage A can remove obvious unusable material.
- Stage B can improve its assigned Core features.
- Improving those features can still create a distribution shift that fails
  Stage-C Utility.
- The correct next action is a controlled Core/Policy ablation and target-model
  validation, not adding Utility to the selector objective.

## Style Taxonomy Alignment Result

The initial style-concentration finding exposed a contract bug: Stage B wrote a
style bucket derived from truncated preview text while Stage C recounted style
from full text using a separate taxonomy. The selector could therefore report
perfect style balance while the released subset was strongly concentrated.

The fix makes the Core scorer's full-text style taxonomy canonical for Stage B
and Stage C, and adds an exact selected-count alignment contract. A targeted
OpenWebText2 development run changed the selected technical-reference share
from `70.7%` to `42.3%` under the canonical taxonomy and reduced the selected
versus usable-pool gap from `+36.2` percentage points to `+3.1` percentage
points.

The same targeted run improved the Stage-A-random Utility delta from the prior
paper-release result of `-0.004280 NLL` to `-0.000783 NLL`, but the result is
still negative and is not certification evidence because the development probe
uses a smaller budget.

## Anti-Memorization Strict-Control Result

A certification-budget Stage-C diagnostic matched the Stage-A baseline on
quality band, fine-grained length, style, source/domain, and repeat-pressure
bucket. This isolates whether the canonical strict baseline wins because it is
longer or more repetition-heavy.

The profile-matched OpenWebText2 result strongly supports the selected subset:

```text
runs:                         16
mean delta NLL:               +0.002316
minimum cell delta NLL:       +0.000774
aggregate delta NLL CI low:   +0.000386
minimum effect/MDE ratio:     2.239582
detectable effect fraction:   1.0
positive run fraction:        1.0
```

Therefore the current evidence supports an easy-NLL strict-baseline confound,
not a Stage-B selector failure. The selector action is `hold`; the required
change is to include the anti-memorization baseline as a reported Stage-C
strict control. This diagnostic does not permit a certification claim by
itself, and Utility remains excluded from the selector objective.

## Semantic Coverage Backbone Audit Result

The original Stage-C semantic backbone audit compared within-cluster pairs
against unions of three documents from different clusters. Those union
representatives systematically inflated between-cluster lexical similarity.
The audit could also be bypassed by high source/domain-bucket purity even when
those buckets were only input filenames rather than explicit semantic labels.

The revised audit compares equally scoped document pairs:

- within-cluster document-pair Jaccard similarity
- cross-cluster document-pair Jaccard similarity
- the fraction of matched comparisons where within-cluster similarity is higher

Anchor purity is retained as a diagnostic but cannot bypass lexical separation.
On the same OpenWebText2 selection, the revised audit reports:

```text
within-cluster coherence:       0.043216
cross-cluster similarity:       0.029451
pairwise separation margin:     0.013766
within > between fraction:      0.666667
semantic backbone pass:         true
Stage-C Coverage pass:          true
```

All four current datasets showed positive pairwise separation in the diagnostic
comparison. OpenWebText2 remains failed at Stage C because Utility does not pass
the canonical strict protocol, not because Coverage lacks a semantic backbone.

## Nuisance-Matched Operational Candidate Result

An additional default Stage-C baseline now exactly matches length, style,
domain/source bucket, and repeat pressure while leaving Quality and
redundancy-risk selector targets unmatched. Hierarchical fallback is disabled
because the initial fallback implementation expanded to the entire Stage-A
pool and became equivalent to Stage-A random.

On the OpenWebText2 development profile:

```text
Stage-A random mean delta NLL:       -0.000783
canonical multi-matched mean:        -0.000783
exact nuisance-matched mean:         -0.000405
exact nuisance-matched worst cell:   -0.001009
anti-memorization certification run: +0.002316
```

The nuisance control reduces the negative mean gap but does not reverse it
under the development budget. Therefore matching Quality is not the sole cause
of the Utility failure. The remaining discrepancy is jointly sensitive to
control construction, probe budget, and replication. The nuisance control is
an operational-counterfactual candidate, not a promoted canonical baseline.

## Same-Condition Certification Baseline Comparison

The development-budget ambiguity was removed by comparing all three baseline
roles under the same certification-grade probe settings. OpenWebText2 reports:

```text
canonical multi-matched:   -0.002310
exact nuisance-matched:    -0.001620
anti-memorization matched: +0.002316
```

FineWeb-Edu was run as the clean positive comparison under the same
certification-grade settings:

```text
canonical multi-matched:   +0.013901
exact nuisance-matched:    -0.004971
anti-memorization matched: +0.015506
```

The sign changes while the selected subset and probe protocol stay fixed.
This establishes baseline/counterfactual identification as the immediate
Utility problem. It does not justify tuning Stage B toward FineWeb-Edu:
FineWeb-Edu itself fails the nuisance control, while both datasets pass the
anti-memorization control.

No baseline is promoted from this result. The next experiment must decompose
the matching variables one at a time and report matched-selected coverage and
feature balance. See `docs/utility_baseline_comparison.md`.

That decomposition is now complete. OpenWebText2 remains negative against
Stage-A random, exact length/style/domain, and exact nuisance controls. It
becomes positive only after Quality is conditioned on. This supports a
within-Quality mechanism effect but does not establish a positive total
operational curation effect. OpenWebText2 should therefore remain an abstention
or rejected training-use case under the current evidence.
