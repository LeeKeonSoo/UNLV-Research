# Phase-1 (Ordered Execution Layout)

This folder is now organized around ordered entry scripts and a dataset config.
The same 5-feature pipeline applies to any dataset that matches `datasets_config.json`.

## Canonical run (one command)

```bash
python 00_run_phase1.py
```

Windows:

```bat
00_run_phase1.bat
```

## Ordered scripts

1. `01_collect_khan_academy.py`
2. `02_collect_tiny_textbooks.py`
3. `03_extract_khan_taxonomy.py`
4. `04_build_corpus_index.py`
5. `05_compute_metrics.py`
6. `06_build_dashboard.py`
7. `07_validate_phase1_outputs.py`
8. `08_generate_label_templates.py`
9. `09_calibrate_ood_thresholds.py`
10. `10_score_metric_gates.py`
11. `11_certify_phase1.py`

## Dataset-agnostic config

Edit `datasets_config.json` to run the same evaluation criteria on new datasets.

Supported formats:

- `json_list`: one JSON file containing a top-level list of records.
- `json_batch_dir`: directory of batch JSON files (top-level list per file).

Per-dataset keys:

- `name`
- `format`
- `source`
- `text_field`
- `id_fields`
- `min_text_chars`
- optional: `metadata_fields`, `batch_glob`, `output_file`

Robustness defaults now included in outputs:

- OOD labeling per chunk: `in_domain`, `borderline`, `ood_near`, `ood_far`
- OOD threshold config recorded in `run_manifest.json`
- Gate-first validation via `07_validate_phase1_outputs.py`

## Validation outputs

- `outputs/run_manifest.json`
- `outputs/run_summary.json`
- `outputs/validation/full_validation_report.json` (if `--write-report` used)
- `outputs/validation/gate_scores_main.json`
- `outputs/validation/gate_scores_transfer.json`
- `outputs/validation/ood_calibration_report.json`

Example final validation report command:

```bash
python 07_validate_phase1_outputs.py --profile full --require-gates --write-report outputs/validation/full_validation_report.json
```

`--require-gates` is strict:
- required gate missing => FAIL
- required gate `pass` is `null` => FAIL
- required gate `pass` is `false` => FAIL

## Certification flow (Strict Closeout)

1. Generate label templates:

```bash
python 08_generate_label_templates.py
```

2. Fill gold labels in `validation/labels/*.csv`.

3. Calibrate OOD thresholds on calibration split:

```bash
python 09_calibrate_ood_thresholds.py
```

4. Score gates:

```bash
python 10_score_metric_gates.py
```

5. Certify strict closeout:

```bash
python 11_certify_phase1.py
```

Current default in `configs/metric_identity_v1.json`:
- `transfer_policy.required = false` (two-dataset closeout mode)
- you can switch to strict transfer mode later by setting `required=true` and `min_datasets>=1`
