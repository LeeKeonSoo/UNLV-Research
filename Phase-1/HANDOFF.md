# Handoff: Training Data Evaluation and Curation Framework

This document transfers the project context for continuing work in a fresh Codex environment. Read this before changing code.

## 1. Project Goal

The project is a research-oriented framework for evaluating and curating training data. The goal is not to optimize one dataset's score, but to build a reliable and reproducible framework that can judge whether a dataset or selected subset is usable, high-quality, non-redundant, representative, and useful for learning.

The framework is organized around five Core axes:

```text
Core 5
├─ Validity
├─ Quality
├─ Redundancy
├─ Coverage
└─ Utility
```

The central design principle is that training-data curation is not just quality filtering. Clean text, high-quality text, diverse text, and useful training text are related but not identical. The framework therefore separates metric roles and execution stages.

## 2. Core - Metric - Policy Structure

Current intended structure:

```text
Core
├─ Validity
│  └─ structural_validity_gate
│     └─ Stage A hard gate
├─ Quality
│  └─ reference_quality_score
│     └─ Stage B selection signal
├─ Redundancy
│  ├─ exact_duplicate_indicator
│  │  └─ Stage A hard gate
│  ├─ shingle_near_duplicate_indicator
│  │  └─ Stage A hard gate
│  └─ shingle_near_duplicate_risk_score
│     └─ Stage B risk signal
├─ Coverage
│  ├─ coverage-preserving selector support
│  │  └─ Stage B support, not primary objective
│  └─ subset_coverage_retention_score
│     └─ Stage C subset-level validator
└─ Utility
   └─ small_lm_probe_gain_score / evidence-aware Utility protocol
      └─ Stage C outcome validator only
```

Important rule: `Utility` must not be added back into the selector objective. It is an outcome validator, not a selector signal.

## 3. Stage A/B/C Meaning

```text
Stage A = Can this chunk be used at all?
Stage B = Among usable chunks, which chunks should be selected?
Stage C = Is the selected subset good as a subset?
```

Detailed roles:

- Stage A: chunk-level hard gate
  - removes structurally invalid chunks
  - removes exact and hard near-duplicate chunks
  - should not judge semantic usefulness or downstream Utility

- Stage B: chunk-level selection
  - ranks surviving chunks using Quality and Redundancy risk
  - uses Coverage support to avoid collapsing rare/style/source buckets
  - should not directly optimize Utility

- Stage C: subset-level validation
  - validates whether the final selected subset preserves Coverage
  - validates whether the subset provides learning Utility under a fixed probe protocol

## 4. Core Definitions and Current Meaning

### Validity

Validity means structural usability only. It answers whether a chunk is usable as text for model training.

It should judge:

- empty or too-short text
- encoding/control-character corruption
- excessive symbol noise
- markup/extraction residue
- broken repetition patterns
- non-language fragments

It should not judge:

- semantic quality
- educational value
- duplication
- coverage
- Utility

Current canonical metric:

- `structural_validity_gate`

Diagnostic support:

- `structural_validity_score`

### Quality

Quality means whether a chunk is informative, coherent, and useful as readable text, after accounting for style and length bias.

Current canonical metric:

- `reference_quality_score`

Important caveat: high Quality alone does not imply Utility. A high-quality subset can still fail Utility if it is too narrow, too easy, too homogeneous, or not useful for transfer.

### Redundancy

Redundancy means duplicate or harmful repetition burden.

Current metrics:

- `exact_duplicate_indicator`
- `shingle_near_duplicate_indicator`
- `shingle_near_duplicate_risk_score`

Role split:

- exact and near-duplicate indicators are Stage A hard gates
- redundancy risk is Stage B selection penalty

Important caveat: not all recurrence is bad. Useful recurrence in definitions, examples, formulas, and technical references should not automatically be treated as harmful duplication.

### Coverage

Coverage means selected subset retention of important distributional structure.

Current canonical metric:

- `subset_coverage_retention_score`

Coverage includes source/style/semantic cluster retention and supports domain coverage only when explicit domain metadata exists. If explicit domain labels do not exist, the framework should not overclaim true domain coverage; it should report source-bucket fallback support.

Important caveat: Coverage can be stable while Utility still fails. This is expected and is one reason Utility exists as a separate Core.

### Utility

Utility means fixed-budget learning outcome measured after subset selection.

Current canonical instrument:

- `small_lm_probe_gain_score`

Default probe model:

- `sshleifer/tiny-gpt2`

Utility remains required, but it is the hardest and most sensitive Core. It should not be collapsed into a single naive pass/fail number without checking whether the probe itself is valid.

## 5. Utility Protocol: Current Intended Interpretation

Utility was redefined from a single selected-vs-baseline pass/fail number into an evidence-aware protocol.

The current Utility report should separate:

```text
Utility evidence
├─ probe sensitivity
│  └─ Can the small-LM probe distinguish positive/random/negative controls?
├─ curation benefit
│  └─ Does selected beat Stage-A random?
└─ strict counterfactual benefit
   └─ Does selected beat multi-matched Stage-A random?
```

Evidence tiers:

- `invalid_probe_evidence`
  - probe control ordering fails
  - do not use Utility result as selector evidence

- `random_baseline_gain`
  - selected beats Stage-A random
  - curation has some benefit, but strict evidence is not established

- `matched_baseline_inconclusive`
  - selected and multi-matched baseline are too close under CI/MDE
  - do not treat as strict gain or strict failure without stronger evidence

- `matched_baseline_gain`
  - selected beats multi-matched baseline under mean/CI/MDE criteria

- `strict_certification_ready`
  - in-domain and required OOD strict criteria pass

Baseline roles:

- `baseline_stageA_random`
  - curation benefit baseline
  - asks whether selection is better than random usable chunks

- `baseline_multi_matched_stageA_random`
  - strict counterfactual baseline
  - asks whether selected is better than a fair matched alternative

- quality/length/style/full baselines
  - diagnostic stress tests only

## 6. Most Important Recent Utility Bug Fix

A serious protocol issue was found in the Utility sensitivity audit.

Previous problem:

```text
Each sensitivity arm used a different Stage-A random baseline.
```

Why this was wrong:

```text
positive_control, stageA_random, corrupted_negative_control, selected
were being compared against different baseline pools.
```

That made the control ordering unreliable because arm differences could be caused by different baselines rather than real probe sensitivity.

Fix implemented:

```text
All sensitivity arms now share one common Stage-A baseline pool,
disjoint from the union of all sensitivity arms.
```

Files involved:

- `14_run_utility_causal_diagnostics.py`
- `19_run_utility_probe_power_sweep.py`

`19_run_utility_probe_power_sweep.py` now treats old per-arm-baseline sweep outputs as stale/incompatible. Do not trust older sweep outputs unless they report the common baseline policy.

Expected baseline policy string:

```text
common_stageA_baseline_disjoint_from_all_sensitivity_arms
```

## 7. Current State Before Transfer

The repository was cleaned for Git push. Large generated data and local artifacts were removed. Only source code, configs, metric specs, README, and small fixtures remain.

Removed intentionally:

- `outputs/` generated files
- scored JSONL files
- selected subset JSONL files
- index SQLite database
- raw dataset files
- model cache
- calibration samples
- teacher-label generated data
- local IDE/Codex files
- old legacy/archive scripts

This means full experiments cannot run immediately after clone unless datasets are restored or regenerated.

## 8. Scripts to Know

Main pipeline:

```text
01_validate_inputs.py
02_build_index.py
03_score_core_metrics.py
04_generate_subsets.py
05_build_dashboard.py
06_validate_outputs.py
07_run_property_benchmarks.py
08_build_metric_maturity_snapshot.py
```

Runners:

```text
00_run_data_eval.py
13_run_paper_release.py
```

Utility/selector diagnostics:

```text
14_run_utility_causal_diagnostics.py
15_run_selector_baseline_audit.py
16_run_good_chunk_dropout_audit.py
17_run_policy_ablation_audit.py
18_compare_candidate_profile.py
19_run_utility_probe_power_sweep.py
```

Dataset preparation:

```text
prepare_openwebtext2_subset.py
prepare_wikitext103_subset.py
prepare_reference_quality_model.py
```

## 9. Recommended First Checks After Clone

Run syntax/import check:

```bash
python -m py_compile 00_run_data_eval.py 01_validate_inputs.py 02_build_index.py 03_score_core_metrics.py 04_generate_subsets.py 05_build_dashboard.py 06_validate_outputs.py 07_run_property_benchmarks.py 08_build_metric_maturity_snapshot.py 13_run_paper_release.py 14_run_utility_causal_diagnostics.py 15_run_selector_baseline_audit.py 16_run_good_chunk_dropout_audit.py 17_run_policy_ablation_audit.py 18_compare_candidate_profile.py 19_run_utility_probe_power_sweep.py data_eval_common.py validate_outputs.py
```

Run input validation smoke fixture:

```bash
python 01_validate_inputs.py --datasets-config validation/fixtures/mini_datasets_config.json
```

Do not expect full `04_generate_subsets.py` Utility validation to pass on the mini fixture. The mini fixture is intentionally tiny and may not contain enough Stage-A disjoint baseline pool for the full Utility protocol.

## 10. Full Pipeline After Dataset Restoration

Once real datasets are restored or regenerated:

```bash
python 01_validate_inputs.py
python 02_build_index.py
python 03_score_core_metrics.py
python 04_generate_subsets.py
python 05_build_dashboard.py
python 06_validate_outputs.py
python 08_build_metric_maturity_snapshot.py
```

Or:

```bash
python 00_run_data_eval.py --flow full
```

Paper-release preflight:

```bash
python 13_run_paper_release.py
```

Paper-release execution:

```bash
python 13_run_paper_release.py --execute
```

## 11. Utility Debugging Continuation Plan

After real outputs exist again, continue Utility debugging in this order.

### Step 1: Confirm common-baseline sensitivity audit

Run one dataset/preset first:

```bash
python 19_run_utility_probe_power_sweep.py --profile learnability_rescue_no_anti_collapse --datasets tiny_textbooks --presets current_like_b0 --force
```

Check output:

- compatible run count should be positive
- baseline policy should be `common_stageA_baseline_disjoint_from_all_sensitivity_arms`
- old stale outputs should not be reused

### Step 2: Run full TinyTextbooks sweep only after Step 1 is clean

```bash
python 19_run_utility_probe_power_sweep.py --profile learnability_rescue_no_anti_collapse --datasets tiny_textbooks --force
```

### Step 3: Interpret Utility correctly

If probe control ordering fails:

```text
Do not call it selector failure.
Call it probe/protocol sensitivity failure.
```

If selected beats Stage-A random but not multi-matched baseline:

```text
Report curation benefit but not strict counterfactual benefit.
```

If selected and multi-matched baseline differ only by tiny deltas below MDE/CI:

```text
Report inconclusive strict counterfactual evidence.
```

Do not force thresholds just to make Utility pass. The point is to find a defensible protocol.

## 12. Important Research Decisions Already Made

Do not undo these without strong reason:

- Keep five Core axes: Validity, Quality, Redundancy, Coverage, Utility.
- Keep Utility as required Core.
- Do not use Utility in selector objective.
- Keep Stage A/B/C separation.
- Treat Coverage and Utility as subset-level validators.
- Treat Utility failure carefully: it may indicate probe invalidity, curation weakness, strict counterfactual weakness, or transfer limitation.
- Do not claim that curation universally improves Utility unless strict evidence supports it.
- Do not overclaim domain coverage when only source/style/cluster support exists.

## 13. Known Pitfalls

### Pitfall 1: Randomness can drop good chunks

Chunk selection is ratio/budget constrained. A good chunk can be excluded if:

- the target ratio is too small
- cluster quotas are already filled
- redundancy risk pushes it down
- coverage support favors another bucket
- tie-breaking/random order affects boundary cases

This is why good-chunk dropout audit exists:

```text
16_run_good_chunk_dropout_audit.py
```

### Pitfall 2: Quality can conflict with Coverage

Selecting only top-quality chunks can collapse distributional coverage. Stage B therefore includes coverage-preserving support and Stage C validates Coverage.

### Pitfall 3: Coverage can pass while Utility fails

This is expected. Coverage only says the subset preserves distributional structure. It does not prove the subset improves learning under a fixed probe.

### Pitfall 4: Utility deltas are small

Small LM probe deltas can be numerically tiny. A value like `-0.0003` does not automatically mean catastrophic failure. It means selected training produced slightly worse held-out NLL than the baseline under that protocol. Interpretation must consider CI, MDE, probe validity, and baseline fairness.

### Pitfall 5: Probe invalidity is not selector failure

If positive/random/negative control ordering fails, the Utility instrument itself is not reliable for selector judgment on that dataset/preset.

## 14. Paper/Research Positioning

The intended paper framing is:

```text
A Stage-Based Framework for Reliability-Aware Training Data Evaluation and Curation
```

Safe claim:

- The framework defines a reproducible Core-Metric-Policy contract.
- Validity, Quality, Redundancy, and Coverage are relatively stable.
- Utility is the hardest axis and exposes limitations that upstream metrics cannot reveal.
- The framework is valuable because it separates gates, selection signals, subset validators, and diagnostic evidence.

Avoid claiming:

- curation always improves Utility
- tiny-gpt2 is a universal Utility proxy
- quality/coverage imply learning benefit
- current datasets prove universal general-purpose transfer

## 15. Current Repository Hygiene Rule

Keep Git focused on experiment code and reproducibility config.

Do not commit:

- `outputs/`
- raw datasets
- model caches
- large scored/subset JSONL files
- generated dashboards/logs
- local IDE files
- temporary transfer zip files

The `.gitignore` is set up for this.

## 16. What To Tell A New Codex Session

Use this prompt after cloning:

```text
Read HANDOFF.md and README.md first. Continue the training-data evaluation framework from the current codebase. Preserve the Core-Metric-Policy and Stage A/B/C design. Utility is Stage C only and must not be added to selector objective. The next important work is to restore/regenerate datasets, rerun the common-baseline Utility sensitivity audit, and interpret Utility using probe sensitivity, selected > Stage-A random, and selected > multi-matched baseline evidence.
```
