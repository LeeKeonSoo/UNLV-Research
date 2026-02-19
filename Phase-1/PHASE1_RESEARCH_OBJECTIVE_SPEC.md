# PHASE-1 Research Objective Specification (Professor Review)

## 1. Purpose
Phase-1 defines a research-first characterization protocol for educational pretraining corpora used in SLM data curation.  
The objective is to measure reproducible corpus properties, compare datasets descriptively, and certify which metrics are valid for interpretation.

## 2. Locked Decisions
1. Specification format: Research-oriented.
2. Priority mode: Reliability-first.
3. Primary audience: Professor review.
4. Claim scope: Descriptive/comparative only in Phase-1.
5. Metric scope: Tiered (`Core` vs `Exploratory`).
6. Canonical language: English.

## 3. Problem Statement
Characterize educational pretraining corpora for SLM data curation using measurable, reproducible dataset properties, while separating robust evidence from provisional diagnostics.

## 4. Phase-1 Scope
### In Scope
1. Dataset characterization.
2. Metric reliability assessment.
3. Cross-dataset descriptive comparison.

### Out of Scope (Deferred to Phase-2)
1. Causal claims on training efficiency.
2. Causal claims on downstream performance uplift.
3. Training-intervention conclusions.

## 5. Research Questions (RQ)
1. RQ1: How does fine-grained domain coverage differ across datasets?
2. RQ2: How does educational-structure quality differ across datasets?
3. RQ3: How stable/reliable are characterization metrics under validation checks?

## 6. Hypotheses (Descriptive, Non-Causal)
1. H1: Khan-like corpus has higher educational marker prevalence than Tiny corpus.
2. H2: Tiny corpus has broader apparent topical spread but lower marker density.
3. H3: Validity-corrected redundancy metrics will materially differ from current degenerate outputs.

## 7. Metric Tiering
### Core Metrics
1. Domain coverage.
2. Quality markers.
3. Difficulty.

### Exploratory Metrics
1. Redundancy.
2. Perplexity.

## 8. Claiming Policy
1. Only Core metrics may support headline conclusions.
2. Exploratory metrics are diagnostic only.
3. Any metric failing reliability gate is labeled `non-claimable`.
4. Phase-1 reports comparative evidence only; no causal training-effect language.

## 9. Reliability-First Workstream
### Objective A: Taxonomy/Label Validity
1. Resolve course-to-subject mapping precedence conflicts.
2. Enforce unique and stable Khan-side document IDs.

### Objective B: Redundancy Validity
1. Redefine exact duplicate logic to avoid universal true positives.
2. Separate semantic duplicate score from near-duplicate proxy.
3. Replace placeholder n-gram overlap with real outputs.

### Objective C: Perplexity Availability
1. Keep exploratory status unless non-null coverage is sufficient.
2. Record fallback behavior in run manifest.

### Objective D: Threshold Calibration
1. Calibrate `MIN_SIMILARITY` on labeled samples.
2. Freeze calibrated threshold in spec and run manifest.

## 10. Validation Protocol and Gates
### Validation Sets
1. Domain: 200 manually labeled chunks.
2. Quality markers: 200 chunks with human binary labels for three markers.
3. Difficulty sanity: 100 chunks for range/pathology checks.

### Gates
1. Domain Top-1 accuracy >= 0.60 and Top-3 recall >= 0.85.
2. Quality marker macro precision >= 0.80.
3. Difficulty out-of-range rate <= 1%.
4. Redundancy must show non-degenerate distribution; exact-dup cannot be ~100% without independent justification.
5. Perplexity remains exploratory unless non-null coverage >= 90% on evaluation subset.

### Gate Failure Rule
If a gate fails (or is not yet validated), the metric is `non-claimable` for headline interpretation.

## 11. Threats to Validity
1. Synthetic source bias (Cosmopedia proxy vs real Khan Academy).
2. Taxonomy mapping ambiguity.
3. Duplicate-ID and redundancy artifacts.
4. Missing or sparse perplexity coverage.

## 12. Interface and Artifact Contract (Phase-1 v2)
1. Output records include `schema_version: "v2"`.
2. Output records include `metric_tier` with fixed keys:
   `domain`, `quality`, `difficulty`, `redundancy`, `perplexity`.
3. Output records include `validity_flags` with fixed boolean keys:
   `domain_valid`, `quality_valid`, `difficulty_valid`, `redundancy_valid`, `perplexity_valid`.
4. Run-level metadata required at `outputs/run_manifest.json`:
   dataset versions/counts, commit hash, threshold config, reliability gate outcomes.
5. Dashboard interpretation layer must:
   display Core/Exploratory badges and suppress headline comparisons for non-claimable metrics.

## 13. Test Scenarios
1. Document consistency test across core Phase-1 docs.
2. Schema compatibility test for dashboard + CSV export.
3. Redundancy degeneracy test (non-constant outputs).
4. Perplexity null-coverage visibility and comparative exclusion.
5. Reproducibility test under same manifest settings.

## 14. Two-Week Planning Horizon
1. Day 1-2: Freeze objective spec, RQs, hypotheses, claim policy.
2. Day 3-5: Reliability A/B (taxonomy + redundancy).
3. Day 6-7: Reliability C/D (perplexity status + threshold calibration).
4. Day 8-9: Validation sampling + gate evaluation + manifest finalization.
5. Day 10: Professor-ready package and decision memo.

## 15. Defaults and Assumptions
1. Phase-1 does not claim training causality.
2. English is canonical review language.
3. Core/Exploratory split is mandatory in reports.
4. Current dashboard metrics are provisional until gates are evaluated.
5. If a metric fails a gate, default action is downgrade to exploratory/non-claimable.
