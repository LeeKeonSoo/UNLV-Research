# Utility Baseline Comparison Decision

## Question

Stage-C Utility must answer a precise counterfactual question. A baseline is not
fair merely because it is difficult to beat, and different matching variables
change the estimand.

This audit compares three disjoint Stage-A controls under the same
certification-grade probe budget, holdout buckets, seeds, and selected subset:

| Baseline | Matching policy | Question |
| --- | --- | --- |
| `baseline_multi_matched_stageA_random` | Quality, length, style, and domain with hierarchical fallback | Does selected beat the current canonical strict control? |
| `baseline_nuisance_matched_stageA_random` | Exact length, style, domain, and repeat pressure; leaves Quality and redundancy risk unmatched | Does the complete Stage-B selection beat a control matched only on declared easy-NLL nuisance variables? |
| `baseline_anti_memorization_matched_stageA_random` | Exact Quality, length, style, domain, and repeat pressure | Does selected add benefit within comparable Quality and easy-NLL conditions? |

All three are Stage-C validators. None is a selector objective.

## Certification-Condition Results

| Dataset | Canonical multi-matched | Nuisance-matched | Anti-memorization matched |
| --- | ---: | ---: | ---: |
| `fineweb_edu_sample` | `+0.013901` | `-0.004971` | `+0.015506` |
| `openwebtext2_subset` | `-0.002310` | `-0.001620` | `+0.002316` |

The signs are not probe-budget artifacts: each row uses the same
certification-grade protocol for every baseline. The comparison also shows
that the result is not simply biased toward FineWeb-Edu:

- FineWeb-Edu changes from positive to negative when only the control
  construction changes.
- OpenWebText2 changes from negative to positive when only the control
  construction changes.
- Anti-memorization is positive in all 32 dataset/cell runs.
- Nuisance matching is negative in 30 of 32 FineWeb-Edu cells and all 16
  OpenWebText2 cells.

## Interpretation

The current Utility bottleneck is primarily **counterfactual identification**,
not evidence that Utility should be added to Stage B and not evidence that the
selector only works on FineWeb-Edu.

The three controls measure different effects:

- The nuisance control is closest to the intended total Stage-B effect, but
  current results do not support it as a certification baseline. Its negative
  FineWeb-Edu result means that excluding Quality from matching does not by
  itself recover the expected selection benefit.
- The anti-memorization control provides strong evidence that selected chunks
  train better than comparable same-Quality, same-shape alternatives. It does
  not prove the total benefit of Quality-based selection because Quality is
  conditioned away.
- The canonical control is not a stable strict estimand. Exact-bucket
  availability is only `0.031373` for FineWeb-Edu and `0.011921` for
  OpenWebText2, so hierarchical fallback dominates its construction.

Therefore no baseline is promoted or removed from this two-dataset result.
The canonical decision remains unchanged, and the other two remain reported
diagnostic controls with explicit estimands.

## Matching Decomposition Result

The one-factor-at-a-time decomposition was run under the same certification
protocol:

| Arm | FineWeb-Edu delta NLL | FineWeb matched selected | OpenWebText2 delta NLL | OpenWeb matched selected |
| --- | ---: | ---: | ---: | ---: |
| Stage-A random | `+0.013901` | `1.000000` | `-0.002310` | `1.000000` |
| Exact length/style/domain | `-0.016631` | `0.997086` | `-0.002824` | `0.999620` |
| Add repeat pressure | `-0.004971` | `0.728735` | `-0.001620` | `0.993749` |
| Add Quality | `+0.015506` | `0.522948` | `+0.002316` | `0.713502` |
| Add redundancy risk | `+0.025002` | `0.032579` | `+0.002433` | `0.585419` |

The result separates two effects:

- FineWeb-Edu has a positive **total operational curation effect** against
  Stage-A random, but not a positive within-length/style/domain effect. Its
  benefit is therefore partly caused by the distribution that Stage B chooses
  to construct.
- OpenWebText2 does not show a positive total operational effect. It becomes
  positive only after Quality is conditioned on, which changes the question
  from total curation benefit to within-Quality selection benefit.
- Adding Quality is the consistent sign-change point, but it also removes a
  Stage-B target from the estimand.
- Highly restrictive matching loses common support. The final FineWeb-Edu arm
  can match only `3.3%` of selected records and cannot support a general
  counterfactual claim.

## Decision

For the intended framework claim, the primary Stage-C Utility estimand should
be the total effect of curation against an equal-budget random usable-data
control. Exact matched controls remain mechanism and confounding diagnostics.
They must not all be required pass gates because Quality, redundancy, style,
domain, and coverage composition are partly outputs of the curation policy.

This does not automatically promote Stage-A random as sufficient paper
certification. The target-SLM curated versus Stage-A-random equal-budget
experiment remains the decisive primary validation, with raw-random and all-data
arms as supporting references plus forgetting, safety, and contamination
constraints.

Before changing the current canonical gate in code, pre-register this estimand
hierarchy and regenerate downstream decision reports:

1. Primary operational effect: selected versus equal-budget disjoint Stage-A
   random.
2. Mechanism diagnostics: exact nuisance and selector-target-matched controls,
   each with common-support warnings.
3. Practical certification: target SLM, curated versus Stage-A random usable
   data, equal tokens/compute and multiple seeds; raw-random and all-data arms
   are supporting references unless separately pre-registered as primary.

## Claim Boundary

Current evidence supports:

```text
Stage-C Utility conclusions are sensitive to the declared counterfactual.
FineWeb-Edu currently supports a total operational curation effect, while
OpenWebText2 does not. Matched controls explain mechanisms but do not replace
the total-effect comparison.
```

Current evidence does not yet support:

```text
One strict Utility baseline is universally fair, or the current selector
improves target-model training for arbitrary raw candidate corpora.
```
