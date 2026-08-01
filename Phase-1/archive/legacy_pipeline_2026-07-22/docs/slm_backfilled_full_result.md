# Exploratory Full-Budget Coverage-Backfill Result

## Status

The selected-only full-budget result showed that Stage-B proxy-supported
selection could become too narrow for the broad Stage-A heldout distribution.
An exploratory release/training-construction arm was therefore built:

```text
coverage_backfilled_interleaved50_equal_budget
= 50% selected non-Utility proxy-supported core
+ 50% disjoint Stage-A coverage backfill
```

The two components are interleaved so that training prefixes preserve the
mixture. The arm uses the same `22,199,800` Qwen-token budget and the same
seed/model/training settings as the first full-budget primary comparison.

## Full-Budget Result

Seed `20260608`:

```text
base_no_update NLL:                         2.778654529
selected_only_curated NLL:                  2.780961865
Stage-A random NLL:                         2.778531128
coverage_backfilled_interleaved50 NLL:      2.777828628

backfilled - Stage-A random NLL:            -0.000702499
backfilled - base NLL:                      -0.000825901
backfilled - selected-only curated NLL:     -0.003133237
```

The exploratory backfilled arm is the best result among these four conditions
on the current internal Stage-A heldout evaluation.

## Interpretation

This result supports the diagnosis that the selected-only arm was not failing
because its selected chunks had low selection-value proxy support. It was more
likely losing broad usable-data coverage under full-budget exposure. Combining
a selected proxy-supported core with coverage-preserving Stage-A support
improved the full-budget result and reversed the selected-only loss.

This does not change the framework's stage ownership:

- Stage A still defines usable data.
- Stage B still selects the proxy-supported core without Utility.
- Stage C still judges subset-level learning outcomes.
- The release/training-construction policy decides whether deployment uses a
  selected-only subset or a supported mixture.

Utility and target-SLM outcomes must not be added to the Stage-B selector
objective.

## Claim Boundary

This is exploratory evidence, not certification evidence:

- the backfill candidate was created after observing the selected-only
  full-budget reversal
- only one full-budget seed has completed
- the same internal heldout evaluation informed the diagnosis
- external benchmark, forgetting, and contamination checks remain incomplete

Do not claim that `50/50` is an optimal or universal mixture.

## Confirmatory Next Step

Freeze `coverage_backfilled_interleaved50_equal_budget` as a new
release/training-construction candidate before running additional outcomes.
Then:

1. replicate the frozen arm across the remaining predeclared seeds
2. evaluate on untouched external and/or newly constructed heldout data
3. compare selected-only, backfilled, Stage-A random, raw random, and base
4. report general capability, target-domain gain, and forgetting separately
5. treat any mixture-ratio sweep as exploratory and confirm the chosen ratio on
   a fresh evaluation split or corpus
