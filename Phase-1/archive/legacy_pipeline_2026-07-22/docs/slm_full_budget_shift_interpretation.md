# SLM Full-Budget Shift Interpretation

## Status

The scaled pilot and the first certification-scale seed disagree.

Scaled pilot `pilot_1024_lr1e5`:

```text
curated better than Stage-A random on 3/3 seeds
curated_minus_stageA_random_mean_nll: -0.000772247
```

Certification-scale first seed `cert_lr1e5_full`, seed `20260608`:

```text
base_no_update NLL: 2.778654529
curated NLL: 2.780961865
Stage-A random NLL: 2.778531128
curated_minus_stageA_random_nll: +0.002430737
```

The certification report therefore marks:

```text
early_negative_signal_pause_recommended
```

## Main Diagnosis

The full-budget failure does not look like a simple Core metric collapse.
Curated remains higher quality and lower redundancy than Stage-A random in
both the pilot prefix and the full budget.

Full-budget curated versus Stage-A random:

```text
quality_token_weighted_mean:        +0.113141
redundancy_risk_token_weighted:     -0.117754
repeat_pressure_token_weighted:     -0.114536
useful_recurrence_token_weighted:   +0.586355
validity_soft_token_weighted:       +0.026373
mean_record_tokens_weighted:        -992.842357
```

The issue is more likely distributional and training-exposure related:

- curated records are much shorter than Stage-A random records
- curated is selected toward high-quality/useful-recurrence regions
- the internal heldout evaluation is broad Stage-A heldout, not curated-like
- full-budget exposure may amplify the selected subset's narrowness
- Stage-A random may win broad heldout NLL because it contains longer and more
  broadly distributed usable text

The heldout comparison supports this:

```text
eval_minus_stageA_random_full quality:  +0.002738
eval_minus_curated_full quality:        -0.110403
eval_minus_stageA_random mean tokens:   +32.168827
eval_minus_curated mean tokens:         +1025.011184
```

So the current primary eval is much closer to Stage-A random than to curated.

## Research Meaning

This is not evidence that Stage A/B/C is useless. It is evidence that the
current Stage-B selected-only training subset may be too narrow for broad
continued-pretraining under full exposure.

The framework behavior is now clearer:

- Stage A produces broad usable data.
- Stage B produces a high-quality selected subset.
- Stage C target-SLM validation shows that a high-quality selected subset may
  not be enough for broad heldout improvement at full budget.

That is a real finding. It argues against claiming deployment readiness from
the current selected-only policy.

## Next Framework Action

Do not add Utility or target-SLM outcomes to the selector objective.

The next candidate should be a policy-level training-set construction variant,
not a Utility-optimized selector:

```text
curated training set = high-quality selected core + coverage-preserving Stage-A support/backfill
```

This keeps the Core-Metric-Policy structure:

- Stage A still owns hard usability.
- Stage B still selects high-quality/useful chunks.
- Stage C exposes that selected-only may be too narrow.
- A release/training-construction layer can choose a supported mixture when
  the intended claim is broad continued-pretraining.

The next test should compare:

```text
selected_only_curated
coverage_backfilled_curated
Stage-A random
raw random
```

under equal target-token budget. This is a new predeclared candidate, not a
post-hoc certification success.

## Exploratory Backfill Follow-Up

The first full-budget `coverage_backfilled_interleaved50_equal_budget` run has
completed for seed `20260608`:

```text
base_no_update NLL:                         2.778654529
selected_only_curated NLL:                  2.780961865
Stage-A random NLL:                         2.778531128
coverage_backfilled_interleaved50 NLL:      2.777828628
backfilled_minus_Stage-A_random NLL:       -0.000702499
```

This is the first full-budget result supporting the release-layer mixture
direction. It does not certify the `50/50` ratio because the candidate was
created after the selected-only reversal and has only one completed
full-budget seed. Read `docs/slm_backfilled_full_result.md` before making a
deployment or paper claim.

## Claim Boundary

Do not claim:

```text
The current selected-only framework is deployment ready.
```

Current defensible claim:

```text
The framework can identify when a selected high-quality subset is not enough
for broad target-SLM improvement, and Stage C prevents unsupported deployment
claims.
```

The exploratory follow-up adds:

```text
A coverage-preserving Stage-A backfill can recover the selected-only
full-budget loss under the current internal heldout, but the mixture requires
predeclared replication and untouched evaluation before certification.
```
