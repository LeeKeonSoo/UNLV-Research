# Framework Definition And Paper Success Criteria

## Purpose

This project builds a language-model training-data curation decision framework.
It is not a universal data-quality scorer and does not claim that any single
metric can determine whether data is intrinsically good.

The active one-month execution plan is
`docs/30_day_paper_sprint_plan.md`. Production-readiness requirements are
scoped in `docs/production_readiness_gate_spec.md`; the current target is a
production-gate prototype, not production certification.

The framework receives a candidate corpus and a deployment contract, then
returns a scoped decision:

```text
candidate corpus
  -> Stage 0 risk quarantine
  -> Stage A hard usability gate
  -> full curated pool
  -> optional Stage B budgeted training subset
  -> Stage C downstream validation
  -> accept, reject, retain_all, or abstain
```

The central research claim is that LM training-data curation should be treated
as an auditable staged decision problem, not as a single quality score.

## What The Framework Is

| Component | Definition |
| --- | --- |
| Candidate corpus | Raw or raw-like data delivered by an upstream collection process |
| Deployment contract | Target model, domain or capability mix, budget, risks, guardrails, and evaluation protocol |
| Full curated pool | Data that survives risk quarantine and hard usability gates |
| Budgeted training subset | Optional subset chosen only when the training budget is smaller than the full curated pool |
| Utility | Downstream training effect measured only in Stage C |
| Abstention | Required output when evidence is missing, negative, or outside the declared scope |

The framework may shrink a dataset, but shrinkage is not the goal. If all data
is usable and the budget can hold it, the correct output is `retain_all`.

## Stage Boundaries

| Stage | Role | Allowed evidence | Forbidden evidence |
| --- | --- | --- | --- |
| Stage 0 | Ingest, normalize, quarantine risk | PII/secrets/license/contamination/poisoning detectors and provenance metadata | Downstream model performance |
| Stage A | Chunk-level hard gate | Structural validity and hard unusability rules | Semantic preference or Utility |
| Stage B | Optional budget allocation | Frozen pre-outcome selection-value, redundancy, and coverage evidence | NLL, benchmark score, EvalPlus, or any Stage-C outcome |
| Stage C | Subset-level validation | Heldout NLL, target benchmark, guardrails, retention tests | Tuning the already-frozen Stage-B policy |

## Core Definitions

| Core | Paper-safe definition | Claim boundary |
| --- | --- | --- |
| Validity | Whether a chunk is structurally usable for training | Not semantic correctness or legal clearance |
| Selection Value Evidence | Observable pre-outcome signal used for budget allocation | Not intrinsic quality and not hard-reject authority |
| Redundancy | Duplicate, saturation, and recurrence control | Must distinguish harmful duplicates from useful recurrence |
| Coverage | Observable retention and drift over source, style, path, content type, clusters, and declared domain/capability mix when metadata and contract support exist | Not proof of Utility, intrinsic quality, true domain coverage without metadata, or target-mix satisfaction without a declared contract |
| Utility | Downstream training effect under a frozen protocol | Stage C only; never a selector objective |

`Quality` should be treated as a legacy compatibility label. In the paper, use
`Selection Value Evidence` unless discussing earlier artifacts.

## Paper Claim Levels

| Level | Required evidence | Current status | Can be claimed now? |
| --- | --- | --- | --- |
| L0: Framework design | Clear Stage 0/A/B/C boundaries and Core-Metric-Policy separation | Present | Yes |
| L1: Leakage-safe implementation | Audit that Stage B does not consume Utility or downstream outcomes | Present | Yes |
| L2: Single-domain positive case | Raw-vs-curated natural-budget training improves downstream metrics | Current Code NLL/EvalPlus positive; independent LiveCodeBench seed-101 pilot is neutral | Yes, bounded to the measured protocol |
| L3: Multi-domain robustness | At least two distinct domains pass under frozen protocols | Math v2 fails; Math v3 repairs the regression but remains worse than raw and is an abstain | No |
| L4: Production-ready curation system | External safety/license/PII/contamination benchmarks and operational guardrails | Incomplete | No |
| L5: Universal data-quality framework | Works for arbitrary corpora and measures intrinsic quality | Unsupported by design | No |

## Current Evidence Boundary

The current evidence supports this claim:

```text
We propose an auditable staged framework for LM training-data curation.
Current code-domain evidence reports a smaller natural-budget subset with
better five-seed heldout NLL and EvalPlus than raw training. A frozen
48-task LiveCodeBench seed-101 pilot is neutral: base, raw, and curated all
score 18.75% pass@1, so independent benchmark transfer is not demonstrated.
In math, selector v2 failed; v3 repaired most of the regression but did not
beat raw training. The framework therefore preserves explicit abstention
rather than forcing a release.
```

The current evidence does not support this claim:

```text
The framework is a general-purpose data-quality detector that improves every
domain or arbitrary incoming corpus.
```

## Minimum Bar For Paper Inclusion

The paper can be written if all of the following are true:

1. The method section clearly defines the framework as a staged curation
   decision system, not a universal quality scorer.
2. Stage-B Utility leakage audits pass.
3. At least one domain has a frozen raw-vs-curated natural-budget success case.
4. Failed domains are reported as boundary evidence, not hidden.
5. The release gate distinguishes research-claim support from production
   deployment readiness.
6. Tables and figures make the difference between base, raw, curated, and
   failed-domain outcomes visible.
7. Composition tables distinguish observed raw-vs-curated domain arms from any
   future joint production mixture target.

## Target Gate For A Strong Conference Submission

The target submission gate is stricter than the minimum bar for writing a
bounded paper. All of the following are required:

1. The current-framework Code rerun completes with at least five frozen seeds,
   paired confidence intervals, heldout NLL, EvalPlus, and retention guardrails.
2. Natural-budget raw-versus-curated results demonstrate training-efficiency
   impact, while equal-token Stage-A-random-versus-curated results isolate the
   Stage-B selection effect.
3. The provenance-rich `clean_retain_all`, `raw_mixed`, and `risk_heavy`
   corpus matrix is materialized without source-tier leakage and reports
   pre/post records, chunks, tokens, composition, license, provenance, and
   contamination evidence.
4. At least two distinct domains pass frozen Stage-C protocols before a
   multi-domain claim is used. A failed domain remains visible as boundary
   evidence but does not substitute for a second positive domain.
5. Core behavior checks are supplemented by the heldout construct benchmarks
   frozen in `configs/core_external_validity_benchmark_contract_v1.json`.
6. Stage-B ablations identify the contribution of Selection Value Evidence,
   Redundancy, and Coverage without using Utility as an objective.
7. Every paper result is regenerated by the canonical runner with matching
   code, config, data, and artifact fingerprints.
8. A powered, pre-registered independent code benchmark shows that any
   claimed external capability gain is not limited to EvalPlus-aligned task
   format. The current 48-task LiveCodeBench pilot does not meet this bar.

Stage-0 production detector certification is required only for a production
claim. It is not a substitute for the training-effect evidence above.

## Required Wording

Use:

```text
stage-separated LM training-data curation decision framework
```

or:

```text
auditable deployment-conditioned curation framework
```

Avoid:

```text
universal curation framework
general data-quality framework
first LM data curation framework
framework that measures data quality
```

## Bounded Draft Success Criterion

For the current paper, success means:

```text
The framework is clearly defined, leakage-safe, reproducible, and supported by
one positive raw-vs-curated LM training result, while explicitly reporting a
negative domain result and limiting the claim accordingly.
```

That is a valid research paper claim. It is not a production release claim.

The paper should not be submitted with a multi-domain or generally applicable
training-use claim until the stronger target gate above passes.

## Execution Order From The Current Training Run

1. Keep the active scorer, selector, metric specification, dataset payloads,
   manifests, training scripts, and current rerun outputs frozen.
2. Complete all five Code raw and curated seeds, then run heldout NLL,
   EvalPlus, retention guardrails, paired confidence intervals, and effect-size
   reporting.
3. Regenerate the canonical evidence package and require the full validation
   suite to pass with matching fingerprints.
4. Execute the heldout Core construct benchmarks from
   `configs/core_external_validity_benchmark_contract_v1.json`. A behavior-only
   pass must remain labeled as development evidence.
5. Materialize the frozen raw corpus matrix and run the mixed raw-like/reference
   experiment with both equal-token and natural-budget comparisons. Apply the
   benchmark sensitivity protocol before reading new Stage-C outcomes.
6. Complete a frozen non-Code Stage-C benchmark. Keep Math as abstain unless
   task-level GSM8K/MATH evidence satisfies the preregistered gate.
7. Only after the evidence is frozen, change canonical metric names, separate
   diagnostic Utility fields from chunk scoring, and refactor oversized active
   modules. Those changes create a new implementation version and require
   fingerprint-sensitive evidence to be regenerated.
8. Rebuild the paper tables and figures from the final canonical artifacts,
   then run the hard paper release gate.
