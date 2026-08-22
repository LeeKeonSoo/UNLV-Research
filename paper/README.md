# Paper Working Notes

Target: IEEE BigData 2026 full-length paper, at most 10 pages in IEEE
two-column format with references included in the limit.

## Current Snapshot

- Working title: **Evidence-Bound Curation: A Traceable Framework for
  Language-Model Training Data**.
- `draft.tex` uses the IEEE conference template and reports the final frozen
  policy, not the earlier Normal/Hard development profiles.
- The final manuscript arms are `Base`, `Raw`, `Ours`, `Data-Juicer`, and
  `NeMo Curator`.
- Every trained arm uses one complete natural-budget pass with seeds 101, 202,
  and 303. Base is the no-update reference.
- BigCodeBench values come only from one common frozen ground-truth cache and
  evaluator configuration recorded in
  `reproducibility/bigcodebench_common_cache_results.json`.

## Claim and Boundary

The framework separates curation evidence from corpus-membership authority.
Validity can isolate explicit input failures. Redundancy can remove a unit only
after a relation is verified and a representative is retained. Quality selects
units through positive Q2--Q4 support, while a qualified Q1 failure is a veto.
Coverage can restore support lost after earlier proposals but cannot create a
new removal.

The final confirmatory runs materialized Coverage decisions and passed the
complete post-Coverage recheck in both domains. The packaged policy remains a
`beta_release`: the paper does not claim production promotion, universal
quality measurement, or uniform downstream improvement across domains.
Benchmarks, training loss, utility, source reputation, target mixtures, and
token budgets are not selector inputs.

## Final Curation Facts

| Domain | Raw packed tokens | Ours packed tokens | Retention | Stage-A chunks | Final chunks | Quality not selected | Redundancy removals | Coverage restores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Code | 6,979,584 | 6,242,304 | 89.44% | 8,026 | 7,270 | 816 | 3 | 104 |
| Math | 6,979,584 | 5,767,168 | 82.63% | 6,619 | 5,528 | 1,223 | 6 | 143 |

The copied final runtime reports are:

- `reproducibility/final_code_curation_report.json`
- `reproducibility/final_code_coverage_audit.json`
- `reproducibility/final_math_curation_report.json`
- `reproducibility/final_math_coverage_audit.json`
- `reproducibility/final_code_training_inputs_report.json`
- `reproducibility/final_math_training_inputs_report.json`

Both Coverage audits report `complete_recheck_passed=true`,
`rematerialization_applied=true`, `may_create_new_removal=false`, and
`scientific_promotion_claimed=false`.

## Final Benchmark Means

All trained-arm entries are three-seed means in percent.

| Code benchmark | Base | Raw | Ours | Data-Juicer | NeMo Curator |
|---|---:|---:|---:|---:|---:|
| HumanEval+ | 31.10 | 28.86 | 31.50 | 28.25 | 30.08 |
| MBPP+ | 58.47 | 48.94 | 54.67 | 41.27 | 50.97 |
| BigCodeBench Complete | 39.65 | 40.61 | 39.12 | 40.35 | 40.41 |
| CRUXEval-I | 3.50 | 7.17 | 5.79 | 9.79 | 7.42 |
| CRUXEval-O | 2.13 | 2.08 | 2.83 | 1.83 | 2.25 |
| DS-1000 | 28.00 | 33.20 | 34.13 | 34.23 | 34.63 |

| Math benchmark | Base | Raw | Ours | Data-Juicer | NeMo Curator |
|---|---:|---:|---:|---:|---:|
| GSM8K strict | 75.13 | 77.79 | 77.53 | 78.72 | 78.29 |
| GSM8K flexible | 57.54 | 71.19 | 63.08 | 67.60 | 64.87 |
| GSM8K normalized | 76.42 | 80.11 | 79.05 | 79.78 | 80.34 |
| MATH-500 | 3.80 | 8.67 | 7.07 | 8.07 | 8.07 |

The paper therefore states the bounded result: Ours exceeds both executable
curation baselines on three of six code benchmarks, while trailing both on all
four reported math scores.

## Numeric Authority

- Final Code Ours: `reproducibility/framework_code_results.json`
- Final Math Ours: `reproducibility/framework_math_results.json`
- Common-cache BigCodeBench: `reproducibility/bigcodebench_common_cache_results.json`
- Data-Juicer aggregate: `reproducibility/same_corpus_baseline_results.json`
- NeMo Code: `reproducibility/nemo_curator_baseline_results.json`
- NeMo Math: `reproducibility/nemo_curator_math_baseline_results.json`
- Final variability summary: `reproducibility/seed_robustness_summary.json`

`benchmark_results.json`, `math_transfer_results.json`, and the Normal/Hard
reports are retained as historical development artifacts. They are not the
numeric authority for the final manuscript.

## Release Artifacts

- `Evidence_Bound_Curation_Draft.pdf`: compiled manuscript.
- `Evidence_Bound_Curation_Overleaf.zip`: self-contained TeX source and figures.
- `Evidence_Bound_Curation_Reproducibility.zip`: compact protocols, reports,
  result summaries, source entry points, and SHA-256 manifest. Corpora, model
  weights, adapters, and large task-level generations remain excluded.
