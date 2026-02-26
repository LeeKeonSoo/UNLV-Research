# Phase-1 Closeout Criteria

## Purpose
This document defines the exact completion criteria for ending Phase-1.
It is designed to prevent ambiguous interpretation after long runtime jobs.

## Phase-1 Mission (Locked)
1. Produce descriptive/comparative characterization only.
2. Do not claim causal training effects.
3. Separate claimable evidence from diagnostic evidence by reliability gates.

## Required Artifacts
1. `outputs/run_manifest.json`
2. `outputs/khan_analysis.jsonl`
3. `outputs/tiny_textbooks_analysis.jsonl`
4. `outputs/dashboard.html`

## Gate-Based Closeout Rules
All items below must be true to mark Phase-1 as closed.

### A. Manifest/Schema Integrity
1. `schema_version == "v2"` in run manifest.
2. JSONL records contain `schema_version`, `metric_tier`, `validity_flags`.
3. Manifest chunk counts match JSONL line counts.

### B. Reliability Gates
1. Domain:
   - Top-1 accuracy gate pass is `true`.
   - Top-3 recall gate pass is `true`.
2. Quality:
   - Macro precision gate pass is `true`.
3. Difficulty:
   - Out-of-range rate gate pass is `true`.
4. Redundancy:
   - Non-degenerate distribution gate pass is `true`.
5. Perplexity:
   - Non-null coverage gate pass is `true`.

### C. Interpretation Policy
1. Headline statements use only claimable core metrics.
2. Exploratory/non-claimable metrics stay diagnostic only.
3. Any failed/pending gate must be explicitly reported.

## Current Known Blockers (as of latest run)
1. Difficulty gate currently fails in the latest observed run.
2. Domain and quality gates are pending manual validation sets.

## Closeout Procedure
1. Run the full active sequence on final config.
2. Fill manual validation results for domain and quality gates.
3. Review outputs and validation evidence against these criteria:
   - Manifest, analysis JSONL, and dashboard availability
   - Gate status and blockable/claimable rules
4. If all criteria pass, freeze artifacts and close Phase-1.
5. If any criteria fail, fix failed/pending gates and rerun.

## Non-Negotiable Anti-Regression Rule
Do not move to Phase-2 causal experiments until this document's gate criteria are fully satisfied.
