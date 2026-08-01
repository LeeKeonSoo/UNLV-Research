# Paper Claim Boundary and Release Gate

This framework must not claim intrinsic data quality or production-ready
curation from the current evidence.

The defensible claim is narrower:

- Stage A removes structurally unusable or hazardous records under conservative
  prefilters.
- Stage B ranks retained records using pre-outcome selection evidence,
  redundancy risk, and coverage controls.
- Stage B does not consume Utility, benchmark outcomes, or downstream model
  results.
- Stage B is optional budget allocation: `retain_all` is valid when no budget
  binds, and `budget_not_selected` remains retained data rather than rejection.
- Utility is judged only in Stage C through downstream validation and
  guardrails.
- Missing guardrails require abstention.

## Current Claim Levels

| Surface | Current Evidence | Allowed Claim |
| --- | --- | --- |
| Core claim defense | `core_claim_defense_scoped_not_release_ready` | Scoped behavior and boundary evidence only |
| Core behavior audit | Fixture behavior checks pass with evidence gaps | Behavior checks only, not full metric validity |
| Stage-0 risk boundary | `stage0_risk_boundary_scoped_not_production_ready` | Project-defined quarantine behavior evidence only |
| Selection Value Evidence | WikiText-like vs synthetic-corruption reference model plus heuristics | Pre-outcome proxy, not intrinsic quality |
| Redundancy | Labeled fixture precision `1.0`, recall `0.5`, plus silver/holdout/dropout audits | Conservative duplicate and saturation control |
| Stage-C training validation | `stage_c_training_validation_nll_supported_curation_claim_ready` | Equal-token target-code NLL effect supports curation-stage claim |
| Confirmatory decision boundary | Current `paper_curation_stage_claim_gate_passed` with matching implementation hashes | Code-domain curation-stage claim only |
| Utility / target NLL | Five-seed natural-budget NLL and EvalPlus evidence | Code-domain Stage-C evidence |
| Target-size Qwen3-4B | Narrow target-code NLL pass with required guardrails observed | Development support only |
| Curation-stage paper claim | Hard gate passed for the current code-domain evidence | Bounded curation-stage research-framework claim |
| Production deployment claim | Production gate blocked | Not currently supported |

## Required Release Gate

`190_run_paper_claim_release_gate.py` is the hard-fail gate for the paper's
bounded curation-stage claim. It is stricter than report generation:

- supported curation-stage paper claims exit `0`;
- abstain, reject, missing guardrails, incomplete evidence, or non-release
  decisions exit non-zero;
- the generated report is
  `outputs/validation/paper_claim_release_gate_report.json`.

The current observed status is:

```text
paper_curation_stage_claim_gate_passed
```

Current paper-claim blockers:

```text
none
```

Production deployment remains blocked:

```text
production_core_validity_not_supported
```

The current artifacts support the code-domain curation-stage training-effect
claim under the frozen natural-budget protocol. They do not certify a
production deployment, cross-domain success, or a universal data-quality
detector.

`196_build_curation_stage_paper_package.py` materializes this boundary as the
paper-writing package:

```text
outputs/validation/curation_stage_paper_package.json
outputs/validation/curation_stage_paper_package.md
```

The bounded Method section is frozen in:

```text
docs/paper_method_core_metric_policy.md
```

The limitations and threats-to-validity section is frozen in:

```text
docs/paper_limitations_and_threats.md
```

The frozen paper comparison tables are:

```text
outputs/validation/paper_comparison_tables.json
outputs/validation/paper_comparison_tables.md
outputs/validation/paper_comparison_tables.csv
```

The frozen paper reproducibility manifest is:

```text
outputs/validation/paper_reproducibility_manifest.json
outputs/validation/paper_reproducibility_manifest.md
```

## Core Metric Validity Release Gap

| Core surface | Current evidence tier | Remaining production gap |
| --- | --- | --- |
| Validity | Structural usability fixtures pass | Does not prove semantic correctness, licensing, or downstream Utility |
| Selection Value Evidence | Pre-outcome proxy fixtures and no-hard-reject contract pass | Does not measure intrinsic Quality or human preference ground truth |
| Redundancy | High-precision canonical threshold plus silver/holdout/dropout evidence | Not recall-complete; threshold still conservative by design |
| Coverage | Observable source/style/path/content/cluster retention and declared domain/capability-mix drift evidence | True domain coverage needs explicit metadata; target-mix satisfaction needs a declared contract |
| Stage 0 Risk Boundary | Project-defined heldout hazard fixtures pass with caveats | Production detector validity requires larger/external labeled benchmarks |
| Utility | Stage-C-only leakage and schema separation pass | Not a selector objective and not a Core scoring proof |

## Reproducibility Hardening

`03_score_core_metrics.py` and `191_score_core_metrics_parallel.py` record a
scoring reproducibility surface in the scoring manifest:

- scorer entrypoint hash;
- Windows full-corpus scorer hash;
- `signals/core.py` hash;
- `quality/reference_quality.py` hash;
- `data_eval_common.py` hash;
- reference-quality model hash;
- reference-quality metadata hash;
- index input hash.

`198_build_paper_reproducibility_manifest.py` freezes the paper package
reproducibility surface: commands, configs, source scripts, paper artifacts,
documentation artifacts, and the Windows GPU policy (`CUDA_VISIBLE_DEVICES=1`
for the RTX 3070 Ti by default).

The current Windows index and scoring artifacts have been rebuilt. Treat
`outputs/scored/scoring_manifest.json` as the scoring completion marker.
Per-dataset `outputs/scored/<dataset>.jsonl.tmp` files are incomplete and must
not be used as evidence.

## Core Claim Defense

`192_build_core_claim_defense_report.py` joins the Core behavior audit,
Redundancy validity benchmark, scoring schema separation audit, selector
Utility leakage audit, and hard release gate into a single machine-readable
claim boundary:

- intrinsic Quality measurement is not supported;
- Selection Value Evidence is a pre-outcome proxy with no hard-reject authority;
- Redundancy is currently conservative and not recall-complete;
- Coverage is observable retention unless explicit domain metadata exists;
- Utility remains Stage-C-only;
- production deployment claims remain blocked.

## Stage-0 Risk Boundary

`193_build_stage0_risk_boundary_report.py` joins the Stage-0 hazard fixture,
detector validation precheck, heldout detector benchmark, and real-corpus
Stage-0/Coverage metadata audit into a single safety-boundary record.

Current allowed claim:

- project-defined quarantine behavior passes development and heldout fixtures
  for PII, secrets, benchmark contamination, poisoning, and rights status;
- current real-corpus Stage-0 lineage and quarantine counts are reported;
- observable metadata supports retention auditing.

Current forbidden claim:

- production-grade PII, secret, license, benchmark-contamination, or poisoning
  detector;
- external public detector benchmark certification;
- legal clearance or license-compliance opinion;
- exhaustive benchmark-contamination removal;
- adversarial poisoning robustness;
- training-release safety certification.

## Stage-C Training Validation

`194_build_stage_c_training_validation_report.py` joins the v2 confirmatory
NLL decision, Stage-C guardrail gap report, canonical 0.5B guardrail decision,
and Qwen3-4B target-size development report into one training-effect boundary.

Current allowed claim:

- curated-v2 equal-budget improves target-code heldout NLL over Stage-A random
  equal-budget under the frozen v2 comparison;
- curated-v2 is directionally better than raw random on target-code NLL;
- canonical binary-current passed 0.5B development guardrails;
- Qwen3-4B target-size target-code NLL passed with required guardrails observed.

Current forbidden claim:

- production-ready framework;
- using Stage-C outcomes to tune Stage B;
- Utility in selector objective.

## Confirmatory Decision Boundary

`195_build_confirmatory_decision_boundary_report.py` joins the v2 confirmatory
decision, Stage-C guardrail gap report, Stage-C training validation report, and
hard paper release gate into one final decision-boundary ledger.

Current final decision:

```text
paper_curation_stage_claim_gate_passed
```

Current allowed claim:

- the frozen current-framework implementation hashes match the Stage-B artifact;
- five-seed curated natural-budget training uses 60.75% fewer packed tokens than raw;
- curated improves heldout code NLL and EvalPlus macro pass rate over raw;
- the bounded code-domain curation-stage paper claim passes its hard gate.

Current forbidden claim:

- production-ready framework;
- treating missing guardrails as pass or fail;
- using Stage-C outcomes to tune Stage B;
- using Utility in the selector objective.

## Utility Leakage Hardening

`164_build_selector_utility_leakage_audit.py` now audits both selector
surfaces:

- `policy/subsets.py`;
- `ingestion/code_selection.py`.

It also scans the full temporal-code Stage-B evidence artifact by default and
checks the Stage-B evidence keys against an explicit allowlist. The current
audit status is:

```text
selector_utility_leakage_audit_passed
```
