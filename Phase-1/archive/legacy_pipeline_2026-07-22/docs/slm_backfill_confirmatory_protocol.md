# Coverage-Backfill Confirmatory Protocol

## Frozen Question

Does the frozen `coverage_backfilled_interleaved50_equal_budget` release arm
improve broad untouched heldout NLL over equal-budget Stage-A random for
`Qwen/Qwen2.5-0.5B` on the current FineWeb-Edu experiment?

This is a release/training-construction test. It does not change the Stage-B
selector and does not add Utility to the selector objective.

## Frozen Candidate

```text
50% selected non-Utility proxy-supported core
50% disjoint Stage-A coverage backfill
interleaved by token share
22,199,800 matched target-token budget
```

The candidate, comparators, holdouts, hashes, recipe, seeds, and success rules
are frozen in:

```text
configs/slm_backfill_confirmatory_plan_qwen25_0p5b_fineweb.json
```

## Untouched Internal Evaluations

Primary:

```text
confirmatory_broad_stageA_eval
```

- stable-hash random sample from remaining Stage-A records
- excludes all four training arms and the legacy diagnostic holdout
- determines primary success

Secondary:

```text
confirmatory_coverage_stratified_stageA_eval
```

- excludes all training arms, the legacy holdout, and the primary holdout
- balances style x length strata
- mechanism diagnostic only
- cannot rescue a failed primary result

Exact UID overlap is zero across training arms and both confirmatory holdouts.
Near-duplicate and public-benchmark contamination audits remain required before
final claims.

## Confirmatory Seeds

Fresh confirmatory seeds:

```text
20260609
20260610
```

Seed `20260608` is excluded from confirmatory success counting because the
50/50 candidate was created after observing that seed's selected-only
full-budget reversal.

## Primary Success Rule

All are required:

1. backfilled mean NLL is lower than Stage-A random mean NLL on the primary
   holdout across the two fresh seeds
2. backfilled is lower than Stage-A random on both fresh seeds
3. no missing or NaN primary outcomes
4. frozen files and training recipe match the plan

The two fresh seeds provide directional replication, not a final
high-statistical-power confidence-interval claim.

## Preparation And Validation

```powershell
conda run --no-capture-output -n research python 44_prepare_slm_confirmatory_holdouts.py
conda run --no-capture-output -n research python 45_freeze_slm_backfill_confirmatory_plan.py
conda run --no-capture-output -n research python 46_validate_slm_confirmatory_contract.py
```

Prepare separate eval blocks:

```powershell
conda run --no-capture-output -n research python 37_run_slm_update_training.py prepare-blocks --arms --blocks-dir outputs\slm_update_experiments\fineweb_edu_canonical_slm_update_v1\token_blocks_full --eval-jsonl outputs\slm_update_experiments\fineweb_edu_canonical_slm_update_v1\confirmatory_broad_stageA_eval.jsonl --eval-name confirmatory_broad_stageA_eval
conda run --no-capture-output -n research python 37_run_slm_update_training.py prepare-blocks --arms --blocks-dir outputs\slm_update_experiments\fineweb_edu_canonical_slm_update_v1\token_blocks_full --eval-jsonl outputs\slm_update_experiments\fineweb_edu_canonical_slm_update_v1\confirmatory_coverage_stratified_stageA_eval.jsonl --eval-name confirmatory_coverage_stratified_stageA_eval
```

## Claim Boundary

A successful result supports only a scoped FineWeb-Edu/Qwen internal-heldout
claim. It does not establish an optimal mixture ratio, raw-corpus success,
dataset-independent improvement, target-model-independent improvement, or
deployment readiness.

## First Fresh-Seed Outcome

Seed `20260609` completed after the plan and holdouts were frozen:

```text
primary broad Stage-A:
  backfilled:    2.766298764
  Stage-A random: 2.765921665
  delta:         +0.000377098

secondary coverage-stratified:
  backfilled:    2.782092087
  Stage-A random: 2.783072731
  delta:         -0.000980644
```

The candidate loses the frozen primary evaluation and wins the secondary
diagnostic. Because the frozen success rule requires a primary win on both
fresh seeds, confirmatory success is no longer possible. The remaining
expensive seed is stopped. Do not change the success rule or tune the mixture
ratio on these holdouts.
