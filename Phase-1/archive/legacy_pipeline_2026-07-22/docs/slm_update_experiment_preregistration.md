# Target-SLM Update Experiment Preregistration

## Purpose

This experiment directly tests the intended framework claim:

```text
Given a candidate corpus, the framework produces either a curated
LM-training subset or an explicit abstention.
```

The target-model experiment is the decisive validation because small probe
Utility can be sensitive to counterfactual definition, budget, and common
support. The target-SLM result answers whether the curated data actually helps
the intended model update.

## Frozen Before Training

Freeze all of the following before observing target-model results:

- candidate corpus and provenance snapshot
- curation profile and metric specification fingerprints
- Stage-A, Stage-B, and Stage-C outputs
- arm construction seed and token/word budget
- base small language model checkpoint
- training hyperparameters and compute budget
- primary and secondary evaluation sets
- contamination, safety, and forgetting thresholds

## Frozen Target Model Config

The first frozen target-SLM config is:

```text
configs/slm_update_qwen25_0p5b_experiment.json
```

It selects `Qwen/Qwen2.5-0.5B` as the base causal LM checkpoint and tokenizer
for the FineWeb-Edu G4 demonstration run. The model is small enough for local
continued-pretraining experiments and has a permissive public model card. This
choice is not a claim that Qwen is the only valid SLM target; it is the
pre-registered target for this run.

## Required Arms

| Arm | Role | Budget |
| --- | --- | --- |
| `base_no_update` | Reference for update benefit and forgetting | no training |
| `stageA_random_equal_budget` | Primary operational baseline | same tokens/compute as curated |
| `curated_equal_budget` | Primary treatment | same tokens/compute as Stage-A random |
| `raw_random_equal_budget` | Measures value of Stage-A cleaning | same tokens/compute as curated |
| `stageA_all_reference` | Optional larger-budget usable-data reference | all Stage-A data or capped larger budget |
| `raw_all_reference` | Optional raw-volume reference | all raw candidate data or capped larger budget |

The primary claim is `curated_equal_budget > stageA_random_equal_budget`.
The stronger efficiency claim requires curated to match or beat a larger
raw/Stage-A reference under reported cost.

## Primary Success Criterion

The curated arm succeeds only if all conditions hold:

1. `curated_equal_budget` improves the pre-registered primary target-SLM
   evaluation over `stageA_random_equal_budget`.
2. The improvement is replicated across at least three training seeds.
3. The curated arm stays within pre-registered forgetting/regression limits.
4. The contamination audit finds no known benchmark leakage in the released
   training subset.
5. Safety/quarantine constraints are not violated.

Failure or abstention remains valid evidence if the framework correctly avoids
an unsupported training-use claim.

## Interpretation Matrix

| Result | Interpretation |
| --- | --- |
| `curated > Stage-A random` | Curation has same-budget operational value. |
| `Stage-A random >= curated` | Current Core/Policy does not select target-SLM-helpful data for this corpus. |
| `raw_all > curated > Stage-A random` | Curation improves per-token efficiency but does not replace raw volume. |
| `curated ~= raw_all` or `curated > raw_all` | Strong cost-efficiency evidence. |
| `raw_random > Stage-A random` | Stage-A gate may be over-filtering useful signal. |
| `Stage-A random > raw_random` | Hard cleaning helps before Stage-B selection. |

## Arm Preparation

Use:

```bash
python 34_prepare_slm_update_experiment.py --dataset DATASET --profile PROFILE --profiles CONFIG --experiment-name NAME
```

The script writes `outputs/slm_update_experiments/NAME/manifest.json` and
equal-budget JSONL arms. The default equal budget is the curated subset's word
count, used as a tokenizer-independent proxy until the target model tokenizer
is frozen.

Then freeze the target-token budget with:

```bash
python 35_freeze_slm_update_plan.py --experiment-dir outputs/slm_update_experiments/NAME --training-config configs/slm_update_qwen25_0p5b_experiment.json
```

For `fineweb_edu_canonical_slm_update_v1`, the frozen Qwen-token primary
budget is `22,199,800` tokens. Long records are split into packed sequence
blocks rather than truncated at document boundaries.

These scripts do not run training. They freeze the arm, model, tokenizer, and
budget contracts before target-model results are observed.

Training and evaluation are run with:

```bash
python 36_prepare_slm_eval_holdout.py --target-word-budget 1000000 --output-name heldout_stageA_eval.jsonl
python 37_run_slm_update_training.py prepare-blocks --blocks-dir outputs/slm_update_experiments/NAME/token_blocks_full --eval-jsonl outputs/slm_update_experiments/NAME/heldout_stageA_eval.jsonl
python 37_run_slm_update_training.py train --arm curated_equal_budget --seed 20260608 --blocks-dir outputs/slm_update_experiments/NAME/token_blocks_full
python 37_run_slm_update_training.py eval --model-path outputs/slm_update_experiments/NAME/model_runs/...
```

Pilot runs may use `--max-sequences`, `--max-eval-sequences`, and `--max-steps`
to validate the runner. Pilot runs are not certification evidence.

The first pilot result used 256 training sequences per arm, 128 eval sequences,
and 32 optimizer steps. It showed `curated_equal_budget` slightly lower-NLL
than `stageA_random_equal_budget`, but all update arms were worse than
`base_no_update`. This supports running the larger experiment; it does not
support a paper claim by itself.

## Claim Boundary

Passing the existing framework reports is not enough for the final paper
claim. The paper claim should be frozen from the target-SLM result:

```text
For a pre-registered corpus/model/update setting, the framework improves
continued training over an equal-budget random usable-data baseline.
```

Do not claim universal raw-corpus improvement unless this is replicated across
multiple raw-corpus settings.
