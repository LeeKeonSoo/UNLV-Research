# Production Readiness Gate Specification

## Purpose

This document defines what would have to be true before the framework could be
called production-ready. It also defines the smaller prototype gate that can be
claimed in the current paper.

The current implementation is not production-certified. The paper may define
these gates and implement a subset as research evidence.

## Readiness Levels

| Level | Name | Meaning | Current target |
| --- | --- | --- | --- |
| R0 | Research claim ready | Method is defined, leakage-safe, reproducible, and supported by scoped evidence | Yes |
| R1 | Production-gate prototype | Required production gates are specified and partially exercised by fixtures/reports | One-month target |
| R2 | Internal pilot ready | Real incoming corpora can be processed with audit trails, quarantines, and rollback | Future work |
| R3 | Production certified | External detector validation, legal/safety review, monitoring, and rollback are operational | Not current scope |

## Mandatory Production Gates

| Gate | Required guarantee | Current claim boundary |
| --- | --- | --- |
| Input provenance | Source, timestamp, transformation lineage, license status, and split identity are recorded | Required for production; partially available |
| Risk quarantine | PII, secrets, license uncertainty, benchmark contamination, and poisoning risk cannot silently enter training | Prototype only |
| Stage A hard validity | Structurally unusable data is rejected with reason codes | Research-supported |
| Utility leakage prevention | Stage B cannot consume NLL, benchmark scores, EvalPlus, or downstream outcomes | Research-supported |
| Redundancy control | Exact, near, template, and saturation duplication are controlled without deleting useful recurrence | Partially supported |
| Coverage retention | Source/style/path/content/cluster collapse is detected before release | Observable retention only |
| Retain-all behavior | High-value usable corpora are not forced to shrink | Required behavior |
| Stage C validation | Raw, curated, and baseline arms are trained/evaluated under a frozen protocol | Historical Code positive requires current-framework rerun; Math abstains |
| Retention guardrails | Target-domain gains do not hide general capability regressions | Incomplete |
| Contamination guardrail | Benchmark overlap is detected and blocked or reported | Prototype only |
| Release decision | System emits `accept`, `reject`, `retain_all`, or `abstain` with evidence | Required behavior |
| Rollback/versioning | Bad releases can be traced and reverted | Future work |
| Monitoring | Distribution drift and detector drift are tracked over time | Future work |

## Paper-Allowed Production Statement

Allowed:

```text
We define production-readiness gates and implement a research prototype of the
decision boundary. The current system is not production certified.
```

Forbidden:

```text
The system is production-ready for arbitrary LM training-data curation.
```

## R1 Prototype Pass Criteria

For the one-month sprint, the production-gate prototype passes only if:

1. Stage 0 hazard fixtures cover PII, secrets, license uncertainty, benchmark
   contamination, and poisoning-like payloads.
2. Known fixture hazards are quarantined or blocked.
3. Uncertain rights or contamination status cannot silently pass into a
   training subset.
4. Stage B Utility leakage audit passes.
5. Record-level dispositions are traceable.
6. Release decisions distinguish paper readiness from production readiness.

If any required item is missing, the correct output is `abstain` or
`production_gate_blocked`, not `production_ready`.

## Production Certification Gap

R3 production certification would require all of the following beyond the
current paper:

- external labeled benchmarks for PII, secrets, licensing, contamination, and
  poisoning detectors;
- legal review or enforceable license policy for code and text sources;
- benchmark-contamination procedures tied to the exact evaluation suite;
- monitoring for incoming-corpus drift and detector drift;
- release versioning and rollback;
- documented incident response for bad releases;
- at least one real operational pilot on newly collected data.

These items should be discussed as future work unless completed.
