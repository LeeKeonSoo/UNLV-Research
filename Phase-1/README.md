# Training Data Evaluation Pipeline

This repository contains the canonical training-data evaluation pipeline used for the paper-release run. The active framework keeps five Core axes:

- `Validity`
- `Quality`
- `Redundancy`
- `Coverage`
- `Utility`

The current execution model is Stage-based:

- Stage A: chunk-level hard gate
- Stage B: chunk-level selection
- Stage C: subset-level validation

## Active Code Surface

Keep day-to-day execution on these scripts:

1. `01_validate_inputs.py`
2. `02_build_index.py`
3. `03_score_core_metrics.py`
4. `04_generate_subsets.py`
5. `05_build_dashboard.py`
6. `06_validate_outputs.py`
7. `07_run_property_benchmarks.py`
8. `08_build_metric_maturity_snapshot.py`
9. `13_run_paper_release.py`

Multi-step runners:

- `00_run_data_eval.py`: development/core pipeline runner
- `13_run_paper_release.py`: paper-release runner with release-mode guardrails

Utility and selector audit scripts:

- `14_run_utility_causal_diagnostics.py`: Utility sensitivity audit
- `15_run_selector_baseline_audit.py`: selector-vs-baseline audit
- `16_run_good_chunk_dropout_audit.py`: high-quality rejected chunk audit
- `17_run_policy_ablation_audit.py`: selector policy ablation audit
- `18_compare_candidate_profile.py`: candidate profile comparison
- `19_run_utility_probe_power_sweep.py`: Utility probe power sweep

## Paper Release Run

Preflight only:

```bash
python 13_run_paper_release.py
```

Full paper-release execution:

```bash
python 13_run_paper_release.py --execute
```

The runner prints each step's live progress to the terminal and also writes a persistent log. Latest log:

```bash
tail -f "$(cat outputs/logs/latest_paper_release.log)"
```

Paper-release guarantees checked before execution:

- no `synthetic_smoke`
- no `runtime_limits`
- `evaluation_mode=certification`
- `certification_scope=general_purpose`
- real canonical Utility probe: `sshleifer/tiny-gpt2`
- all-pairwise OOD Utility enforcement
- strict `min` Utility statistic with positive delta-NLL and positive CI requirement
- selector objective does not include Utility

## Development Run

Core development run:

```bash
python 00_run_data_eval.py
```

Full development run with property benchmarks and maturity snapshot:

```bash
python 00_run_data_eval.py --flow full
```

The development runner also prints live step progress and writes a persistent log. Latest log:

```bash
tail -f "$(cat outputs/logs/latest_data_eval.log)"
```

## Manual Step Run

Use this when you want to inspect one step at a time:

```bash
python 01_validate_inputs.py
python 02_build_index.py
python 03_score_core_metrics.py
python 04_generate_subsets.py --profiles configs/paper_release.json
python 05_build_dashboard.py
python 06_validate_outputs.py
python 08_build_metric_maturity_snapshot.py
```

## Main Configs

- `configs/paper_release.json`: paper-release certification profile
- `configs/curation_profiles.json`: development canonical profile
- `configs/learnability_rescue_probe.json`: Utility/learnability candidate profile
- `configs/utility_scope_smoke.json`: fast schema/regression smoke profile, not paper evidence
- `configs/metric_spec_with_citations.json`: canonical metric contract
- `configs/metric_spec_with_citations.md`: readable metric contract
- `datasets_config.json`: active dataset config

## Core Metrics

Chunk-level selection metrics:

- `structural_validity_gate`
- `reference_quality_score`
- `exact_duplicate_indicator`
- `shingle_near_duplicate_indicator`
- `shingle_near_duplicate_risk_score`

Subset-level validators:

- `subset_coverage_retention_score`
- `small_lm_probe_gain_score`

Diagnostics only:

- `structural_validity_score`
- `explanatory_quality_proxy`
- `tail_cluster_rarity_proxy`
- `predictive_utility_proxy`
- `fixed_token_probe_gain_score`

## Output Contract

Primary outputs:

- `outputs/index/index.sqlite`
- `outputs/index/index_manifest.json`
- `outputs/scored/scoring_manifest.json`
- `outputs/scored/<dataset>.jsonl`
- `outputs/subsets/<profile>/<dataset>.jsonl`
- `outputs/run_manifest.json`
- `outputs/run_summary.json`
- `outputs/dashboard.html`
- `outputs/validation/full_validation_report.json`
- `outputs/metric_maturity_snapshot.json`

Useful result checks:

```bash
jq '.summary' outputs/validation/full_validation_report.json
```

```bash
jq '.tracked_metrics[] | {metric, maturity, action, validation:(.evidence.validation.label // .evidence.validation.certification_label // null), certification_ready:(.evidence.validation.certification_ready // null)}' outputs/metric_maturity_snapshot.json
```

```bash
jq '.profiles.paper_release_certification | to_entries[] | select(.key|startswith("_")|not) | {dataset:.key, stage_c_pass:.value.stage_c_core_validation.passed, fail_reasons:.value.stage_c_fail_reasons, coverage:.value.subset_coverage_retention_score, utility:.value.small_lm_probe_gain_score, strict_min:.value.stage_c_core_validation.utility_strict_min_gain, utility_pass:.value.stage_c_core_validation.utility_pass, coverage_pass:.value.stage_c_core_validation.coverage_pass}' outputs/run_summary.json
```

## Repository Hygiene

Generated datasets, indexes, scored JSONL files, selected subsets, model caches, logs, and dashboards are intentionally excluded from Git. Recreate them by running the pipeline commands above.
