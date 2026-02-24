# Phase-1 Run Playbook (Correctness First)

## Objective
Focus on valid, reproducible results first. Use speed improvements only after output semantics are locked.

## 1) Reproducibility baseline (before any optimization)
- Keep current script set and defaults.
- Pin deterministic behavior where it exists and document non-deterministic parts.
- Use identical inputs and environments for reruns.

## 2) Baseline checks after every major run
- Verify file existence
  - `outputs/khan_analysis.jsonl`
  - `outputs/tiny_textbooks_analysis.jsonl`
  - `outputs/run_manifest.json`
  - `outputs/run_summary.json`
- Verify counts
  - `run_manifest.dataset_versions_and_counts.khan_academy.chunks_written == khan_analysis lines`
  - `run_manifest.dataset_versions_and_counts.tiny_textbooks.chunks_written == tiny_analysis lines`
- Verify schema version
  - `run_manifest.schema_version == v2`
  - each record includes `schema_version`, `metric_tier`, `validity_flags`
- Run closeout gate view:
  - No legacy helper command is provided in active setup.

## 3) Result sanity (before optimization)
Use these thresholds as expected ranges and investigate only when outside range:
- Domain labels: top labels should be non-empty for most chunks (`domain_valid` ratio should be high)
- Quality score range: `[0.0, 1.0]`
- Difficulty ratio validity: `difficulty_valid` should not degrade unexpectedly
- Redundancy distribution: `exact_duplicate_rate < 0.99` and variance of at least one redundancy metric should be > 0

## 4) Stage-wise runs
- Full production run sequence:
  1. `python collect_khan_academy.py`
  2. `python collect_tiny_textbooks.py`
  3. `python extract_khan_taxonomy.py`
  4. `python build_corpus_index.py`
  5. `python compute_metrics.py`
  6. `python build_dashboard.py`

- Quick validation subset:
  - Set quick limits inside `compute_metrics.py` or env (recommended `PHASE1_MAX_BATCHES=2`) and run from Step 5.

## 5) Strict regression check (if needed)
- Keep a fixed tiny validation subset (or one batch)
- Compare `outputs/run_summary.json` and manifest gate outcomes across reruns:
  - chunk counts
  - key core metric claimability flags
  - gate pass/fail states
- If difference exceeds tolerance unexpectedly, inspect:
  - text preprocessing
  - dependency versions
  - GPU path fallback (cpu/mps/cuda)

## 6) Optimization policy
- After baseline passes, optimize only in this order:
  1) chunk-level streaming and intermediate checkpointing in compute loop
  2) environment-level config tuning (batch sizes)
  3) index build parameter tuning

## 7) Commands
- Full run sequence:
  - `python collect_khan_academy.py`
  - `python collect_tiny_textbooks.py`
  - `python extract_khan_taxonomy.py`
  - `python build_corpus_index.py`
  - `python compute_metrics.py`
  - `python build_dashboard.py`
- Quick verification:
  - Set `PHASE1_MAX_BATCHES=2` in env for a limited run of `compute_metrics.py`.
