# Framework Consistency Baseline

Status: current authority after retirement of the Contrastive Quality
candidate and registration of the three-teacher Quality Ranker candidate.

## Scientific Claim Boundary

The framework curates LM training corpora through auditable, typed decisions.
It does not claim to measure an intrinsic universal document Quality value.
It does not use downstream outcomes while selecting data. External training
and benchmarks validate a frozen curation policy after the framework emits its
dataset.

## Fixed Core and Stage Ownership

| Core | Stage | Question | Decision authority |
|---|---|---|---|
| Validity | A | Is the unit trainable under the declared input contract? | Closed failure quarantine or removal |
| Redundancy | B | Is equivalent training information already represented? | Stable-family nonrepresentative removal |
| Quality | B | Does an independently validated Policy support rejection or retention? | Promoted Policy decision only |
| Coverage | C | Did combined removals erase support without authorization? | Materialization veto only |

Coverage cannot rank data, impose a domain mix, delete data, or restore by
quota. Quality cannot use Utility, NLL, benchmark outcomes, source reputation,
domain quotas, or a forced token budget.

## Current Object Authority

The authority chain is:

1. `configs/curation_framework_v1.json`
2. `configs/framework_objects_v1.json`
3. `configs/framework_profiles_v1.json`
4. `configs/framework_runtime_bridge_v1.json`
5. `configs/framework_release_validation_v1.json`

Every file is hash-linked. The runtime bridge preserves legacy-compatible
output and prevents blocked Policies from becoming active merely because a
candidate file exists.

## Current Policy Status

| Policy | Lifecycle | Current meaning |
|---|---|---|
| `validity.interpretable_text` | candidate | Closed observable input failures only |
| `redundancy.exact_text_family` | development_passed | Exact family representatives validated in development |
| `redundancy.symmetric_near_duplicate_candidate` | blocked | No safe non-exact equivalence boundary identified |
| `quality.explicit_nonpayload` | candidate | Closed deterministic non-payload cases |
| `quality.teacher_panel_candidate` | blocked | Q1-Q4 teacher-panel qualification not complete |
| `coverage.representative_guard` | candidate | Veto unexplained zero-survivor materialization |

Normal and Hard share these Policy families. Their operating points are not
calibrated, and both profiles are release-disabled. `Hard subset-or-equal
Normal` is a mandatory materialized-output invariant.

## Quality Ranker Candidate

The current candidate uses three independent model organizations and four
independent fail gates. It does not average dimensions into a weighted score.
Each teacher returns `pass`, `fail`, or `abstain` plus reason codes for Q1-Q4.
A 2-of-3 first pass requires the same majority and at least two stable teachers
on a blinded second pass. All other outcomes abstain.

The candidate cannot delete data by itself. Promotion requires:

- a 512-item smoke fixture matrix;
- at least 800 protected fixtures;
- one-sided 95% exact false-removal upper bounds no greater than 0.5% for
  Normal and 2.0% for Hard;
- stable consensus and schema behavior;
- a frozen provider identity and operating point;
- Coverage compatibility;
- independent external evaluation after curation.

The initial language scope is English. External NVIDIA teachers may receive
only public, license-compatible calibration samples. Private or undeclared
corpora remain local.

## Contrastive Retirement

The previous target/reference loss-gap design is retired as a current Quality
candidate. Its observed evidence did not support a stable route-general
deletion threshold. All implementation and evidence files are preserved under
`archive/historical_contracts/contrastive_quality_candidate_2026-08-04/` for
provenance, but no active manifest, profile, provider registry, runtime
fingerprint, or release gate references that candidate.

## Release State

`validation/frozen_contracts/framework_release_validation_v1.json` reports:

- implementation integrity: `passed`;
- framework release: `blocked`;
- blockers: unpromoted policies, uncalibrated profile operating points,
  blocked near-duplicate authority, and blocked Quality teacher panel.

This is a fail-closed scientific state, not evidence that the active
legacy-compatible curation output is broken.

## Next Exit Gates

1. Complete strict teacher adapters and retry behavior.
2. Complete smoke and protected fixture qualification.
3. Freeze Normal and Hard Quality operating points.
4. Complete witness-based near-duplicate redesign.
5. Complete route-spanning Coverage validation.
6. Run Base, Normal, and Hard external three-seed natural-budget evaluation.

Only after those gates pass may the release claim or paper claim be expanded.
