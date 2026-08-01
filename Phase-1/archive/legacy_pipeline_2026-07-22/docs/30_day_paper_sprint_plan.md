# 30-Day Paper Sprint Plan

## Goal

Deadline window: 2026-07-07 through 2026-08-06.

The goal is to reach a defensible paper-submission package for the LM
training-data curation decision framework. The goal is not to certify a
production-ready universal curation system in one month.

Target claim:

```text
We propose an auditable stage-separated LM training-data curation decision
framework. The framework separates risk quarantine, hard usability gating,
optional budget allocation, and downstream validation. Current evidence
records a current-framework five-seed code-domain raw-vs-curated improvement,
and exposes a math-domain
v2 failure as a non-release condition while freezing a redesigned v3 candidate
for a new Stage-C test.
```

Forbidden claim:

```text
The framework is a universal data-quality detector or production-certified
curation system that improves arbitrary corpora.
```

## Current Evidence Baseline

| Domain | Arm | Packed train tokens | Mean NLL | Benchmark | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Code | base_no_update | 0 | 1.234183 | EvalPlus 47.1798% | reference |
| Code | raw_full_natural | 980,992 | 1.209983 | EvalPlus 51.0796% | current raw reference |
| Code | curated_v2_natural | 385,024 | 1.200903 | EvalPlus 58.2183% | current five-seed pass |
| Math | base_no_update | 0 | 1.559449 | not run | reference |
| Math | raw_full_natural | 1,120,256 | 1.495650 | not run | reference raw |
| Math | curated_math_v2_natural | 626,688 | 1.527065 | not run | fail |
| Math | curated_math_v3_natural | 1,026,048 | pending | not run | frozen candidate |

Interpretation:

- Code is a current-framework positive case under the frozen natural-budget protocol.
- Math v2 is the current negative case.
- Math v3 is a pre-outcome redesigned candidate, not yet a success.
- Production certification remains unsupported.
- Utility remains Stage C only and must never become a Stage-B objective.

## Block Schedule

| Block | Dates | Objective | Exit condition |
| --- | --- | --- | --- |
| 1 | Jul 7-9 | Lock claim, sprint plan, and production-readiness gates | Complete: plan and gate spec exist; claim levels are explicit |
| 2 | Jul 10-14 | Minimum Stage 0 production-gate prototype | Complete: R1 prototype passes; production release remains blocked |
| 3 | Jul 15-18 | Record-level audit and decision schema hardening | Complete: disposition audit passes and feeds R1 production gate |
| 4 | Jul 19-23 | Math failure handling | In progress: v3 materialization, protocol, and token blocks are frozen before outcomes |
| 5 | Jul 24-27 | Stage-C evidence consolidation | Final tables use frozen Code/Math values and include token reductions |
| 6 | Jul 28-Aug 2 | Paper rewrite and figures | IEEE draft compiles; figures/tables support the bounded claim |
| 7 | Aug 3-6 | Final reproducibility and readiness audit | Paper-readiness report passes or lists blockers |

## Current Progress

Status after Block 4 pre-outcome freeze:

- Block 1 is complete.
- Block 2 is complete at R1 prototype level.
- Block 3 is complete at record-disposition audit level.
- Block 4 has a frozen Math v3 candidate, natural-budget protocol, and token
  blocks, but no downstream success claim yet.
- `outputs/validation/production_readiness_gate_report.md` reports
  `production_gate_prototype_passed`.
- `outputs/validation/record_disposition_audit_report.md` reports
  `record_disposition_audit_passed`.
- `outputs/validation/math_domain_selector_v3_materialization_report.json`
  reports `math_selector_v3_materialized`.
- `outputs/validation/math_domain_natural_budget_v3_freeze_report.json`
  reports `math_natural_budget_v3_protocol_frozen`.
- `outputs/validation/math_domain_natural_budget_v3_blocks_report.json`
  reports `math_natural_budget_v3_blocks_frozen`.
- Math v3 retains `624,540` token-proxy count versus raw `679,711`, and
  `1,026,048` packed training tokens versus raw `1,120,256`. It restores
  proof/theorem token retention relative to v2.
- `outputs/validation/stage0_release_blocker_report.md` still reports
  `stage0_release_blocked_production_guardrails`.
- This is the intended boundary: prototype gates can support a paper claim,
  but production certification remains blocked.

Next required action:

```text
Run Math v3 Stage-C training/evaluation under the frozen natural-budget protocol
```

## Block 1: Claim And Production Gate Lock

Deliverables:

- `docs/30_day_paper_sprint_plan.md`
- `docs/production_readiness_gate_spec.md`
- `.omo/plans/lm-curation-30-day-paper-sprint.md`

Done when:

- The paper claim is bounded.
- The production-ready claim is explicitly out of scope.
- A production-readiness gate prototype is defined.
- The same-protocol current Code EvalPlus values are `51.0796%` raw and
  `58.2183%` curated across five seeds.

## Block 2: Stage 0 Gate Prototype

Deliverables:

- Hazard fixture benchmark for PII, secrets, licensing uncertainty,
  benchmark contamination, and poisoning-like payloads.
- Report that says pass, block, or abstain per hazard family.

Done when:

- Known hazards cannot enter training subsets by default.
- Uncertain rights or contamination status produces quarantine or abstention.
- The paper can describe this as a prototype gate, not production detector
  certification.

## Block 3: Decision Audit Trail

Deliverables:

- Record-level disposition schema/report.
- Retain-all fixture.
- Budget-not-selected fixture.

Done when:

- `budget_not_selected` is never treated as rejection.
- `retain_all` is valid when a corpus is usable and the budget can hold it.
- Every release decision has a reason-coded evidence path.

## Block 4: Math Failure Cycle

Deliverables:

- Math selector v3 fixture contract or explicit retain-all/abstain arms.
- Report explaining whether Math passes under a new frozen protocol or remains
  unsupported.
- Natural-budget v3 protocol frozen before training outcomes.

Done when:

- The current Math failure is not hidden.
- No Stage-B policy is tuned using the failed Stage-C holdout.
- Long reasoning, worked derivations, short-answer items, and noisy extraction
  cases are represented in fixtures.
- The v3 arm is evaluated under Stage C and either passes or is reported as a
  second Math failure.

## Block 5: Evidence Consolidation

Deliverables:

- Final result table for paper and slides.
- Token/record reduction table.
- Stage-C decision summary.

Done when:

- Base, raw, and curated rows are visible for Code and Math.
- Code is presented as a positive case.
- Math is presented as a failure or abstention case.
- No stale EvalPlus values remain in paper-facing docs.

## Block 6: Paper And Figures

Deliverables:

- Revised IEEE paper draft.
- Figure briefs or generated figures for:
  - pipeline overview;
  - Core-Metric-Policy map;
  - Utility leakage boundary;
  - historical Code positive vs Math fail result summary;
  - production-readiness gate.

Done when:

- The draft reads like a conference paper, not a project log.
- Contributions are explicit.
- Discussion explains the historical Code positive result, why Math failed, and what the boundary
  means.

## Block 7: Final Readiness Audit

Deliverables:

- Paper release gate report.
- Reproducibility manifest.
- Stale-number scan.
- Utility leakage audit.

Done when:

- The final report says either `paper-ready bounded claim` or lists exact
  blockers.
- Production certification remains blocked unless external detector and
  deployment requirements are actually satisfied.

## Priority Rule

If time runs short, prioritize in this order:

1. Claim correctness.
2. Frozen evidence tables.
3. Production gate spec.
4. Paper readability and figures.
5. Math redesign.
6. Additional domains or larger experiments.

The paper can survive with a reported Math failure. It cannot survive with an
overclaimed framework definition.
