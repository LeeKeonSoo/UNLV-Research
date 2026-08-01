# Code-Domain Confirmatory Postmortem

## Status

- Confirmatory status: `confirmatory_decision_reject_primary_margin_failure`.
- Interpretation: negative primary-margin result with a positive directional curation signal.
- This is a completed confirmatory experiment, not an infrastructure failure.

## Primary NLL Finding

- Required frozen margin: `0.005`.
- Curated vs Stage-A-random reduction: `0.0037666601293226964`.
- Gap to margin: `0.0012333398706773037`.
- Curated vs raw-random reduction: `0.001210669734898584`.
- Known-HQ minus curated: `-0.00014812525580909508`.

## Development To Confirmatory Shift

- Development primary reduction: `0.011155656973521166`.
- Confirmatory primary reduction: `0.0037666601293226964`.
- Retention ratio: `0.33764574675101267`.
- Absolute shrink: `0.00738899684419847`.
- Development base NLL: `1.1782967034313414`.
- Confirmatory base NLL: `1.0118654002161587`.

## Heldout Shift

- Record count change: `-65`.
- Repository Jaccard overlap: `0.0`.
- Test ratio development: `0.2857142857142857`.
- Test ratio confirmatory: `0.41818181818181815`.
- Test ratio increase: `0.13246753246753246`.
- Mean token-proxy change: `185.72935064935064`.

## Locked Interpretation

- The completed frozen confirmatory protocol must remain negative on the primary margin.
- Margins, seeds, heldout slices, token budgets, and Stage-C thresholds must not be changed post hoc.
- Utility, EvalPlus, and retention outcomes remain Stage C only and must not enter Stage B selector objectives.

## Next Development Cycle

- Start a new development cycle if improving the recipe.
- Freeze larger stratified heldouts by repository and content type.
- Calibrate the practical margin before confirmatory outcomes using development-only power/effect-size analysis.
- Strengthen Stage B proxy selection without Utility or benchmark leakage.
- Treat this result as valid negative evidence in the paper trail.
