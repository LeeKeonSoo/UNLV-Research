# Retention Replay Pareto Analysis

## Question

Can a small general-replay component preserve external language-model
retention while keeping the coverage-target benefit of the curated release?

This is a release/training-construction question. Stage B is unchanged, and
Utility or target-model outcomes are not selector objectives.

## Development Contract

- target component: FineWeb 50/50 selected-core plus Stage-A coverage backfill
- replay component: WikiText103 `train` split only
- external retention holdout: WikiText103 `validation` plus `test`
- exact normalized-text replay/holdout overlap: zero
- matched comparator: Stage-A random
- model: Qwen2.5-0.5B
- pilot: 1024 sequences, 128 optimizer steps, seed `20260611`
- GPU: physical CUDA device 1, RTX 3070 Ti only

Joint development success required:

```text
target NLL < matched Stage-A random target NLL
and
external WikiText NLL <= base no-update external NLL
```

## Key References

| Reference | Target NLL | External WikiText NLL |
| --- | ---: | ---: |
| base no update | 2.784997879 | 2.741912568 |
| matched Stage-A random | 2.765567785 | 2.748400536 |

## Boundary Result

| Target / replay | Target gain vs Stage-A | External regression vs base | Joint pass |
| --- | ---: | ---: | --- |
| 100% / 0% | +0.000336015 | +0.008814765 | no |
| 99.25% / 0.75% | -0.000051036 | +0.003347838 | no |
| 99% / 1% | -0.000316823 | -0.001199686 | no |
| 98.5% / 1.5% | -0.000060877 | -0.012498261 | no |
| 97.5% / 2.5% | -0.000143059 | -0.020352275 | no |
| 95% / 5% | -0.000287508 | -0.035542752 | no |
| 90% / 10% | -0.001131879 | -0.047396805 | no |
| 75% / 25% | -0.003092094 | -0.071956338 | no |
| 50% / 50% | -0.006307079 | -0.095294918 | no |

Positive target gain means lower NLL than matched Stage-A random. Negative
external regression means lower NLL than the base model.

## Interpretation

Replay strongly changes the retention outcome. At one percent replay, the
external WikiText regression is removed. However, every replay arm loses the
strict target comparison against matched Stage-A random in this pilot.

Therefore:

- forgetting is not an unavoidable property of any update
- replay is a viable retention-control mechanism
- the current simple ratio mixture does not jointly satisfy both objectives
- further fine-grained ratio tuning on this evidence should stop
- the next development axis is the training recipe or replay-source contract

The narrow target margin and non-monotonic boundary results also require fresh
seeds before interpreting tiny ratio differences.

## Next Work

The matched training-recipe follow-up found two development joint-pass
candidates:

| Recipe | Arm | Target gain vs matched Stage-A | External regression vs base |
| --- | --- | ---: | ---: |
| `lr5e6_s128` | 99% target + 1% replay | +0.000152922 | -0.000169115 |
| `lr1e5_s64` | 99% target + 1% replay | +0.000077145 | -0.001800845 |

The first recipe is selected for fresh confirmation because the Deployment
Contract treats target improvement as primary and retention as a required
guardrail. Development results are excluded from confirmatory success counts.

## Fresh Confirmation Result

The frozen candidate was run on two fresh training seeds and newly frozen
target/external holdouts:

| Seed | Target gain vs matched Stage-A | External regression vs base | Joint pass |
| ---: | ---: | ---: | --- |
| 20260612 | +0.000425387 | -0.010134662 | yes |
| 20260613 | -0.000011424 | -0.009567447 | no |

The frozen overall rule required both fresh seeds to pass. The confirmatory
result is therefore `confirmatory_joint_not_supported`.

The replay-aware recipe solves the observed external forgetting problem on
both fresh seeds. What remains unsupported is a seed-stable target advantage
over the recipe-matched Stage-A comparator.

Across the two fresh seeds, mean target gain is approximately `+0.000206981`
NLL with sample standard deviation `0.000308872`. The observed effect is small
relative to seed variation, so the next diagnostic must estimate target-effect
power and minimum detectable effect before another candidate is proposed.

## Target-Effect Stability Diagnosis

The paired-block power diagnostic shows that the evaluator is sensitive enough
to detect the observed effect when it exists. Seed `20260612` has a clear
positive paired interval, while seed `20260613` is genuinely indistinguishable
from zero. This is not primarily an aggregate evaluation-noise failure.

The development-only checkpoint curve then evaluated three additional seeds at
steps `32/64/96/128`. At step `128`, all three seeds were target-positive and
passed the external retention guardrail. Across all six available model pairs
evaluated on the previously used development target, all six are positive.

Cross-holdout evaluation rules out train-block construction as the cause: the
development and confirmatory train tensors are exactly identical. Two of three
stored model pairs remain positive on both target holdouts. The third has only
a very small development gain (`+0.000074868`) and crosses to a near-zero
fresh-holdout result (`-0.000011424`).

The current conclusion is therefore:

```text
The replay-aware recipe has a small positive development effect, but the
effect margin is too close to zero to support the frozen strict all-seed
confirmatory release rule across target holdout variation.
```

This is a Stage-C rejection of an insufficiently robust release claim, not a
selector-objective failure.

Next protocol work:

1. Do not tune another recipe on the frozen confirmatory holdout.
2. Predeclare a practical target-effect margin and training-seed replication
   count for future candidate protocols.
3. Require distributionally distinct target holdouts and task-based outcomes
   before a release claim.
4. Keep retention replay because it consistently addresses the observed
   external forgetting failure.
5. Compare continuous replay with scheduled replay and a broader frozen replay
   pool only as a new development protocol.

The future-candidate rule is frozen in
`configs/target_effect_release_protocol_v1.json`. It is explicitly
non-retroactive. It requires at least five fresh training seeds, two untouched
target holdouts, a target mean improvement of at least `0.00015` NLL with a
positive one-sided seed-level confidence bound, per-seed external retention,
and at least one predeclared task suite. The current candidate remains rejected
under its original frozen rule.

## Claim Boundary

This is development Pareto evidence, not certification. It does not establish
deployment readiness, task-capability preservation, universal replay ratios,
or raw-corpus success.
