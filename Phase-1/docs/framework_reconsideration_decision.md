# Framework Reconsideration Decision

## Decision

The project is not a universal classifier of whether every chunk is necessary
for language-model training. It is an auditable curation layer that must not
discard usable data merely to meet a target reduction ratio.

Two different tasks had become entangled and are now separated.

| Task | Output | May remove data? | Current claim status |
| --- | --- | --- | --- |
| Curation | Full curated pool | Only with an explicit, validated reason code | Core framework |
| Budget allocation | Optional training subset | No; non-selected data remains retained | Separate deployment option |

A fixed selection fraction, including the historical Code `0.4` fraction, is
a resource-allocation experiment. It is not evidence that the omitted data was
unnecessary, low value, or low quality.

## Minimal A-B-C Framework

```text
Candidate corpus
  -> Stage A: provenance, normalization, risk quarantine
  -> Stage B: integrity hard gate
  -> Stage C: declared reason-coded operational selection and materialization

Frozen output
  -> External Evaluation Protocol
```

This is a historical design-decision record. Its disabled near-duplicate state
was superseded on 2026-08-02. The active runtime contract is
`docs/current_curation_framework.md`; Normal and Hard now enable the frozen
symmetric 0.95 near-duplicate rule.

Stage A owns provenance, rights status, PII/secrets including embedded session
headers, contamination risk, and other unresolved hazards. Stage B owns minimum
chunk length and normalized exact duplicates. Stage C owns candidate-only
near-duplicate representative retention, explicit generated-and-do-not-edit
artifact removal, code-only license/comment chunk removal, and import/export-only structural scaffold compaction.
The candidate near-duplicate rule requires at least 40 lexical tokens and
symmetric 95% overlap over 5-token shingles; it keeps the stable first
representative and records every removal. Scaffold compaction retains one stable
representative so the observed scaffold structural bucket is not erased.
Pathological repetition remains a proposed extension. Both removal stages
require explicit reason codes, declared non-trigger conditions, and regression
fixtures that verify those boundaries.

The historical weighted operational-priority rule has been retired because its
coefficients and threshold were not independently calibrated. Stage C now
may remove only a symmetric near-duplicate copy, a source record with an explicit
generated-and-do-not-edit declaration, a code chunk consisting only of an
explicit license/copyright comment, or an import/export-only scaffold copy
beyond one stable representative. A future allocator, if introduced, must
not interpret non-allocation as rejection, necessity, or low quality.

The External Evaluation Protocol trains or evaluates a frozen output. Utility,
NLL, and benchmark outcomes are external measurements, not Core metrics and
not selector inputs.

Before a confirmatory run, the development and confirmatory audited candidate
corpora must have zero overlap in stable record IDs and normalized-text hashes,
must each have a complete benchmark-exclusion audit, and must be materialized
under an identical policy fingerprint. This integrity gate is external-only:
its split metadata and outcomes are unavailable to the A-B-C runtime.

## Core Authority

| Core | Supported role now | May hard-remove now? | Required evidence before stronger authority |
| --- | --- | --- | --- |
| Validity | Provenance/risk quarantine, malformed-content rules, and embedded credential detection | Yes, for declared reason codes | Reason-code fixture coverage and declared-boundary regression by content type |
| Redundancy | Normalized exact duplicates, exact scaffold-family representative retention, and candidate-only conservative near-duplicate evidence | Exact duplicates and identical normalized scaffold families; symmetric 95% shingle overlap is disabled in frozen protocols | Near-duplicate boundary fixtures and useful-retention regression |
| Quality | Explicit generated-and-non-editable artifact, license/comment-only chunk, or payload-preserving span compaction | Only the declared non-payload artifact or validated separable span | Structural-rule boundary fixtures and fresh downstream evidence |
| Coverage | Representative preservation plus retention/drift audit | Protects against erasing an observed scaffold bucket | Explicit metadata and deployment-contract validation |

Quality is a Core, but never an intrinsic universal score. `Utility` is not a
runtime Core and remains external evaluation evidence only.

## What Is Not Yet Supported

- A fixed fraction as a general curation policy.
- A global weighted threshold that says a Stage-C non-selected record is universally unnecessary.
- A domain-general semantic usefulness classifier.
- Universal downstream improvement.
- Production-grade rights, contamination, PII, or poisoning certification.

## Validation Design

### Curation Validity

Evaluate each removal-capable reason code directly, before model training:

1. Safety/provenance fixtures and corpus audits for Stage A.
2. Structural and exact-duplicate fixtures plus useful-retention audits for
   Stage B.
3. Clean, raw-mixed, duplicate-heavy, and risk-heavy corpus scenarios.
4. Reason-coded removal rate, non-trigger fixture pass rate, retained-token
   fraction, and provenance coverage.

The expected outcome for a clean corpus is near-`retain_all`, not a forced
reduction.

### Downstream Evaluation

The primary deployment comparison is natural budget:

```text
Policy-selected equal-token vs Stage-A random equal-token vs discarded-only
equal-token vs Raw-safe natural training vs policy-selected natural training
```

The equal-token comparison measures selection signal separately from exposure;
the natural arms measure the deployment proposition of training less data. The
discarded-only arm is required to test whether the selected-versus-random gap is
actually attributable to the rule. None of these results proves every omitted
chunk unnecessary.

## Historical Evidence

Prior Code selection and downstream-evaluation artifacts are retained only in
`archive/legacy_pipeline_2026-07-22/`. The prior 40% Code selection result is
evidence that the historical ranking policy may have selection signal, but it
does not prove the omitted records unnecessary and cannot be confirmatory
evidence for the active no-budget policy. The forensic replay found that 2,681
of 2,723 historical proxy-rejected chunks (98.46%) are retained by current v3;
the retired score therefore cannot be reinterpreted as an artifact detector.

## Immediate Gate

Do not launch more Math training until the new Math corpus has a benchmark
contamination audit and the paper reports which portions are curation evidence,
optional allocation evidence, and external evaluation evidence.
