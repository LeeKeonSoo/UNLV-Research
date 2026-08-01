# Math Domain Failure Postmortem

## Status

Math selector v2 failed Stage-C natural-budget validation. This is negative
evidence for the current Math Core/Metric/Policy configuration, not permission
to tune Stage B against the failed holdout.

## Evidence

| Arm | Records | Packed training tokens | Steps | Mean heldout NLL |
| --- | ---: | ---: | ---: | ---: |
| Raw | 512 | 1,120,256 | 69 | 1.495650 |
| Curated v2 | 326 | 626,688 | 39 | 1.527065 |

Curated v2 uses 44.1% fewer packed training tokens than raw, but worsens mean
heldout NLL by `+0.031415`. Under the current protocol, lower NLL is better.

## Interpretation

The failure is not treated as seed noise: selector v2 still fails under the
current multi-seed Stage-C summary. The most likely conclusion is that the
current Math Core/Metric/Policy stack removes useful mathematical signal or
under-retains context needed for downstream training.

Plausible mechanisms:

- long reasoning context is under-retained by current selection proxies
- concise-looking examples are over-valued relative to worked derivations
- coverage over problem type, solution style, and difficulty is incomplete
- heldout NLL may not capture all math capability, but it is still a valid
  failed training-signal test under the frozen protocol
- the natural curated budget may be too small for this domain

This is not Utility leakage. Utility remains Stage C only and is not a Stage-B
selector objective.

## Required Next Actions

1. Build math-specific fixture cases for long reasoning, multi-step derivation,
   proof-like text, short-answer items, and noisy extraction artifacts.
2. Redesign Math Selection Value Evidence and Coverage proxies before another
   confirmatory run.
3. Add retain-all and broader-curated-pool arms to test whether Stage B is
   over-selecting under this domain.
4. If capability benchmarks are used, compare raw, Stage-A pass, curated, and
   base models on GSM8K/MATH-style evaluation with frozen prompts and seeds.
5. Keep the current failed result in the paper evidence ledger as a boundary
   condition, not as an implementation detail to hide.

The current fixture contract is frozen in
`validation/fixtures/math_failure_selector_cases.json`, the selector v3
redesign boundary is frozen in
`configs/math_domain_selector_v3_redesign_contract.json`, and the generated
audit is `outputs/validation/math_failure_fixture_contract_report.json`.
