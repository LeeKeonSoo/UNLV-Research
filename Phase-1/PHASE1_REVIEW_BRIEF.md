# Phase-1 Reviewer Brief (One Page)

## Objective
Phase-1 is a descriptive/comparative characterization stage for educational pretraining corpora used in SLM data curation.  
This stage prioritizes measurement validity over interpretive breadth.

## What Phase-1 Claims
1. Cross-dataset descriptive differences in measurable corpus properties.
2. Reliability status of each metric family under defined validation gates.

## What Phase-1 Does Not Claim
1. No causal statements about training efficiency.
2. No causal statements about downstream model quality.
3. No intervention-effect claims (deferred to Phase-2 experiments).

## Frozen RQ Set
1. RQ1: Fine-grained domain coverage differences.
2. RQ2: Educational-structure quality differences.
3. RQ3: Metric stability/reliability under validation checks.

## Metric Policy (Tiered)
### Core
1. Domain coverage.
2. Quality markers.
3. Difficulty.

### Exploratory
1. Redundancy.
2. Perplexity.

## Claiming Rule
1. Headline conclusions may use Core metrics only.
2. Exploratory metrics are diagnostic only.
3. Any metric failing or missing gate validation is marked `non-claimable`.

## Reliability Gates
1. Domain: Top-1 accuracy >= 0.60, Top-3 recall >= 0.85 (200 labeled chunks).
2. Quality: macro precision >= 0.80 (200 labeled chunks).
3. Difficulty: out-of-range rate <= 1% (100 sanity chunks).
4. Redundancy: non-degenerate distribution required.
5. Perplexity: remains exploratory unless non-null coverage >= 90%.

## Current Phase-1 v2 Interface
1. Record-level: `schema_version`, `metric_tier`, `validity_flags`.
2. Run-level: `outputs/run_manifest.json` with dataset versions, commit hash, thresholds, gate outcomes.
3. Dashboard: Core/Exploratory badges and suppression of non-claimable metrics from headline comparison.

## Main Validity Risks
1. Cosmopedia proxy bias vs real Khan Academy source.
2. Taxonomy mapping ambiguity.
3. Redundancy artifact risk (duplicate-ID/logic issues).
4. Perplexity availability gaps.

## Review Ask
1. Confirm Phase-1 claim boundaries are appropriate for publication-quality reporting.
2. Confirm gate thresholds and manual validation set sizes.
3. Approve transition criteria from Phase-1 to Phase-2 causal experiments.
