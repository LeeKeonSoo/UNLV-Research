# Paper Limitations and Threats to Validity

This document defines the limitations that must accompany the paper claim. The
supported claim is a bounded curation-stage research framework for
language-model training data. The current artifacts do not support a production
deployment claim, a universal data-quality detector claim, or a legal-clearance
claim.

## Claim Boundary

The paper may claim that the framework separates collection from curation,
keeps the Core-Metric-Policy stage boundaries explicit, preserves the full
curated pool unless a declared budget is binding, and validates selected
training releases through frozen Stage-C comparisons.

The paper must not claim that the framework measures intrinsic data quality.
`Quality` remains a legacy artifact alias for Selection Value Evidence. The
canonical construct is observable pre-outcome selection evidence, not universal
or human-grounded quality.

## Internal Validity

Stage B is not allowed to consume Utility, benchmark outcomes, heldout NLL, or
downstream model results. This reduces leakage risk, but it also means Stage-B
signals are hypotheses about selection value until Stage C tests them.
Stage B is also not a rejection stage: when the budget binds,
`budget_not_selected` records remain retained in the full curated pool. When the
budget does not bind, `retain_all` is the expected policy outcome.

The current positive code-domain result is based on frozen equal-token
comparisons and required guardrails, but it remains tied to the tested model
family, corpus construction, budget, tokenizer, and target-code heldout. The
paper should report the raw-random, Stage-A-random, curated, reference, and
ablation tables without converting them into a universal ranking of data
quality.

## Construct Validity

The Core surfaces are operational responsibilities rather than fully validated
construct measurements.

| Surface | Limitation |
| --- | --- |
| Validity | Structural usability fixtures do not prove semantic usefulness, legal safety, or downstream training benefit. |
| Selection Value Evidence | Pre-outcome proxies can guide budget allocation but cannot justify hard rejection or intrinsic quality claims. |
| Redundancy | Current evidence favors conservative high-precision duplicate control; recall is not complete. |
| Coverage | Current coverage is observable source/style/path/content/cluster retention and observed domain-mix drift; true domain coverage needs explicit metadata and target-mix satisfaction needs a declared contract. |
| Utility | Utility is Stage C outcome evidence only and cannot become a selector objective. |

## External Validity

The current strongest result is a code-domain experiment. It supports the
curation-stage framework claim under the tested setting, but it does not prove
that the same thresholds, weights, or proxy formulas transfer unchanged to
medical data, legal data, multilingual web data, math data, or arbitrary raw
web crawls.

The current math-domain result is not a success claim. Selector v2 is preserved
as a negative over-filtering reference, and selector v3 repairs that failure
without beating raw natural-budget training. Until GSM8K and MATH benchmark
guardrails are complete, the math-domain paper decision remains `abstain`.

The framework is designed to allow `retain_all`, `reject`, `quarantine`, and
`abstain`. A future corpus may be mostly usable, mostly hazardous, too narrow
for the declared deployment contract, or unsuitable for Stage-C validation. In
those cases, abstention or scoped release is the correct behavior.

## Safety, Rights, and Contamination

Stage 0 includes project-defined fixtures and heldout checks for PII, secrets,
benchmark contamination, poisoning, and rights-risk boundaries. These are
development evidence, not production detector certification. The paper must not
claim legal compliance, license clearance, exhaustive contamination removal, or
adversarial poisoning robustness.

## Statistical and Compute Limits

The paper uses frozen seeds, equal-token arms, source hashes, and report hashes
to reduce post-hoc tuning risk. Remaining threats include finite seed count,
finite heldout size, hardware-specific execution conditions, model-family
dependence, and sensitivity to tokenization and packing choices.

## Production Boundary

Production deployment remains blocked by:

```text
production_core_validity_not_supported
```

Closing that blocker would require larger and more external metric-validity
evidence, production-grade Stage-0 detectors, stronger redundancy recall
calibration, explicit domain coverage metadata where domain claims are made,
and deployment-specific monitoring. Those requirements are outside the current
paper claim.
