# Phase-1 (Execution-Only Layout)

This folder is intentionally trimmed for execution and validation.
Root contains only the scripts needed to run Phase-1 end-to-end.

## Canonical run

Use this command for professor-facing final validation:

```bash
python run_phase1_lowmem.py
```

Then verify outputs:

```bash
python validate_phase1_outputs.py
```

Optional smoke test (not final reporting):

```bash
python run_phase1_lowmem.py --quick
```

## What each script does

- `collect_khan_academy.py`: collect and normalize Khan-side corpus.
- `collect_tiny_textbooks.py`: collect Tiny-Textbooks raw batches.
- `extract_khan_taxonomy.py`: build concept prototypes/taxonomy artifacts.
- `build_corpus_index.py`: build redundancy index (`corpus_index.pkl`).
- `compute_metrics.py`: compute Phase-1 metrics and run manifest.
- `build_dashboard.py`: build dashboard HTML from output JSONL.
- `validate_phase1_outputs.py`: strict artifact validation checks.
- `run_phase1_lowmem.py`: orchestrator for all steps.
- `run_phase1_lowmem.bat`: Windows launcher wrapper.

## Expected validated outputs

- `outputs/khan_analysis.jsonl`
- `outputs/tiny_textbooks_analysis.jsonl`
- `outputs/corpus_index.pkl`
- `outputs/corpus_texts.sqlite`
- `outputs/run_manifest.json`
- `outputs/run_summary.json`
- `outputs/dashboard.html`

## Archived docs

Non-execution docs were moved to:

- `legacy/docs/`
