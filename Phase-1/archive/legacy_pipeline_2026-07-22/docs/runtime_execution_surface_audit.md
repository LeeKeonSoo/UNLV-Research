# Runtime Execution Surface Audit

Date: 2026-07-22

## Finding

There is no single current operational runner for the revised A-B-C curation
framework. The repository contains three materially different execution
surfaces. They must not be treated as equivalent.

## 1. Legacy Generic Runner

Command:

```powershell
python 00_run_data_eval.py
```

This path is not the revised minimal curation engine. Its `core` flow directly
executes these 20 scripts:

```text
validation/test_decision_contracts.py
validation/test_stage0_processing.py
validation/test_style_taxonomy_contract.py
30_process_stage0_candidates.py
29_validate_stage0_contract.py
01_validate_inputs.py
02_build_index.py
03_score_core_metrics.py
04_generate_subsets.py
05_build_dashboard.py
15_run_selector_baseline_audit.py
21_build_utility_transfer_gap_report.py
23_build_core_proxy_alignment_report.py
24_build_core_proxy_calibration_report.py
20_build_curation_readiness_report.py
25_build_stage_c_protocol_decision_report.py
26_build_strict_baseline_control_report.py
27_build_curation_decision_report.py
28_build_paper_evidence_table.py
06_validate_outputs.py
```

The `full` flow additionally runs:

```text
07_run_property_benchmarks.py
31_build_openwebtext2_slice_diagnostic.py
08_build_metric_maturity_snapshot.py
```

It can also prepare `prepare_openwebtext2_subset.py`,
`prepare_fineweb_edu_sample.py`, and `prepare_reference_quality_model.py`.
The latter is automatic and proves that this runner still depends on the
legacy reference-quality model.

Its transitive local modules include:

```text
data_eval_common.py
index/build.py
ingestion/normalize.py
ingestion/schema.py
policy/dispositions.py
policy/stage_b_budget.py
policy/subsets.py
property_benchmarks.py
quality/reference_quality.py
reports/dashboard.py
reports/metric_maturity.py
reports/summary.py
signals/core.py
utility/features.py
utility/lm_probe.py
validate_outputs.py
```

Conclusion: this is a legacy research/diagnostic pipeline, not a safe command
to label as the current Framework run. It resets shared `outputs/` unless
`--reuse-existing-outputs` is supplied.

## 2. Paper-Evidence Rebuild Runner

Command:

```powershell
python run_canonical_paper_evidence.py --execute
```

This runner executes nine report builders only:

```text
229_build_code_livecodebench_confirmation_summary.py
211_build_code_paper_evidence_report.py
227_build_code_livecodebench_pilot_summary.py
213_build_final_paper_evidence_table.py
190_run_paper_claim_release_gate.py
219_build_domain_composition_audit.py
220_build_coverage_domain_mix_audit.py
221_build_stage_b_policy_contract_audit.py
218_build_paper_claim_consistency_audit.py
```

Its helper modules are the `paper_evidence/` package plus
`data_eval_common.py`, `policy/dispositions.py`, and
`policy/stage_b_budget.py`. It does not collect data, curate a corpus, train,
or run benchmarks. It reconstructs reports from existing artifacts.

Conclusion: it is an evidence/report runner, not the Framework execution
command.

## 3. Current Raw-Mixed Corpus Paths

### Math A-B-C Materialization

Commands:

```powershell
python 243_collect_math_raw_mixed_5m.py
python 244_materialize_math_raw_mixed_abc_curation.py
```

Files used by the materializer:

```text
243_collect_math_raw_mixed_5m.py
244_materialize_math_raw_mixed_abc_curation.py
configs/math_raw_mixed_5m_collection_v1.json
configs/math_raw_mixed_5m_abc_curation_v1.json
data_eval_common.py
ingestion/normalize.py
ingestion/schema.py
```

This is the closest path to the revised Framework: Stage A risk/provenance,
Stage B exact/structural hard gate, and Stage C `retain_all` when no budget is
declared. It does not call a quality model, Utility code, or a selector.

### Historical Code 5M Path

Commands:

```powershell
python 236_prepare_code_5m_stage0_input.py
python 237_run_code_5m_stages.py
conda run --no-capture-output -n research python 238_prepare_code_5m_external_validation.py ...
```

Files used by the Code curation path:

```text
236_prepare_code_5m_stage0_input.py
237_run_code_5m_stages.py
configs/code_5m_corpus_acquisition_v2.json
configs/temporal_code_curation_protocol_v1.json
data_eval_common.py
ingestion/code_chunks.py
ingestion/code_fingerprints.py
ingestion/code_selection.py
policy/dispositions.py
```

The external materializer additionally uses:

```text
238_prepare_code_5m_external_validation.py
configs/code_5m_natural_budget_execution_qwen3_4b_v1.json
torch and transformers from conda environment research
```

`237_run_code_5m_stages.py` is historical: it calls
`ingestion/code_selection.py`, whose objective contains
`code_quality_proxy` and a configured fixed budget fraction. Its names also
place old hard gates in `stage_a` and allocation in `stage_b`. Therefore it
cannot be presented as the revised A-B-C implementation without migration.

## Smoke Results

`--help` completed successfully for `00_run_data_eval.py`,
`run_canonical_paper_evidence.py`, `236`, `237`, `243`, and `244` under the
default Python interpreter. `238` requires the `research` conda environment:
default Python lacks `torch`, while `research` completed its CLI help.

## Reconsideration Gates

1. Declare one operational Framework command that runs only the revised
   Stage A/B/C contract and writes one audit manifest.
2. Move `00_run_data_eval.py` to an explicitly legacy diagnostic namespace or
   remove it from active entry-point documentation; it still auto-prepares a
   reference-quality model and runs Utility-oriented reports.
3. Migrate or retire `237_run_code_5m_stages.py` from the active Framework
   path. Its fixed-fraction `code_quality_proxy` selection contradicts the
   revised retain-all-by-default decision.
4. Keep `run_canonical_paper_evidence.py` explicitly report-only, so a paper
   rebuild can never be mistaken for curation or downstream validation.
5. After the operational runner is fixed, add a single end-to-end smoke test
   that asserts its manifest contains no Utility/benchmark input and that
   Stage C is `retain_all` without a declared binding budget.
