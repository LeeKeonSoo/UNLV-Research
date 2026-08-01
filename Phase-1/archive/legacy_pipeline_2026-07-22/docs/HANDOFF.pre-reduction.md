# Handoff: Training Data Evaluation and Curation Framework

## Current Direction: Practical LM-Training Curation Framework

The active goal is to build an operational curation framework for language
model training data, not a tool that claims to measure intrinsic data quality.
Use `docs/lm_curation_operational_framework.md` and
`configs/lm_curation_operational_framework_v1.json` as the practical target.

## Current Operational Boundary: 5M Code Artifact

`docs/operational_curation_boundary_v2.md` and
`configs/operational_curation_contract_v2.json` define the current execution
boundary:

- Curation ends after Stage 0-A-B; Stage C is external offline validation and
  never a curation transformation.
- The current corpus has completed Stage 0-A-B. External validation is running
  as a three-seed natural-budget QLoRA comparison; it cannot change Stage-B
  policy or reselect the frozen artifact.
- Its Stage-B output used the historical fixed `0.4` fraction in
  `temporal_code_curation_protocol_v1.json`. Treat it as an archived
  budgeted-subset instance, not as a universal operating rule or a claim that
  budget-not-selected data was low quality.
- Future operational runs require a declared binding budget; otherwise Stage B
  emits `retain_all` or `abstain`.

Verify the boundary with:

```powershell
conda run --no-capture-output -n research python validation\test_operational_curation_contract_v2.py
```

Validation scopes:

- `python 06_validate_outputs.py --scope canonical` validates the active
  framework surface and excludes historical temporal-code evidence.
- `python 06_validate_outputs.py --scope full` preserves the complete
  historical reproducibility audit.
- Historical implementations may live under `archive/`; a numbered root
  compatibility wrapper remains whenever retained tests or automation import it.
- Default code search excludes `archive/`; see `docs/active_code_surface.md`.

The correct top-level contract is:

```text
candidate corpus -> full curated pool -> optional budgeted training subset
                 -> supported LM-training release or explicit abstention
```

This means `reject`, `abstain`, and `insufficient_usable_data` are valid
framework outputs. Do not frame the project as guaranteeing that every raw
candidate corpus can be transformed into a useful training dataset.

Core interpretation for future work:

- Validity = structural usability hard gate.
- Selection Value Evidence = frozen observable pre-outcome evidence for
  optional budget allocation, not intrinsic or ground-truth quality and never
  a hard-reject authority.
- Quality = legacy runtime/artifact alias only.
- Redundancy = duplicate/saturation control; split harmful duplication from
  useful recurrence.
- Coverage = observable source/style/path/content/cluster retention; domain
  coverage and domain/capability mix claims only when metadata and a declared
  Deployment Contract support them. Without a declared target mix, Coverage can
  report observed composition drift only.
- Utility = Stage-C protocol-bound downstream effect only.

Critical disposition invariant:

- every non-quarantined Stage-A pass belongs to the full curated pool
- Stage B is optional and activates only for a binding training budget
- `retain_all` is a valid result
- `budget_not_selected` is retained data, not rejected or low-quality data
- no fixed rejection quota or target reduction ratio is allowed
- uncertain safety cases are quarantined; uncertain selection-value cases stay
  retained unless an independent hard gate applies

Current canonical evidence snapshot:

- Code natural-budget validation is a historical positive case: curated v2 reduces
  packed training tokens by 60.8%, improves five-seed heldout NLL from
  `1.209983` to `1.200903`, and improves the same-protocol natural-budget
  EvalPlus macro pass rate from `51.08%` to `58.22%`. The Stage-B implementation
  hashes match current code and the bounded curation-stage paper gate passes.
- The independent LiveCodeBench pilot is complete and neutral. The frozen
  48-task 2025 slice gives base, raw-natural, and curated-natural the same
  `9/48` (`18.75%`) pass@1, with zero paired correctness transitions. Arm
  generations differ, so this is not an adapter/no-op artifact, but it does
  not demonstrate benchmark transfer. The pilot remains Stage C only and must
  not be used to tune Stage B.
- Math natural-budget validation is the abstain case: selector v2 over-filtered
  and worsened heldout NLL from `1.495650` to `1.527065`; selector v3 repairs
  the v2 failure to `1.498987` while using 8.4% fewer packed training tokens
  than raw, but it still does not beat raw and lacks GSM8K/MATH benchmark
  guardrails.
- Treat this as evidence for a universal curation-control structure, not a
  universal data-quality score. The framework claim is
  deployment-conditioned curation with explicit accept/reject/abstain behavior.
- Domain/capability mix is now part of the Deployment Contract surface when a
  target mix is declared. If no target mix is declared, the framework reports
  observed composition only. Current Block-2 evidence reports paper-domain arms:
  raw Code/Math packed-token shares are `46.69%`/`53.31%`, and current
  curated-arm shares are `27.29%`/`72.71%`. This must not be described as a
  joint production corpus mix or a universal domain ratio.
- Do not claim all-domain improvement until another domain passes the same
  frozen Stage-C discipline. Use `docs/math_domain_failure_postmortem.md` and
  `docs/paper_claim_redefinition.md` as the current claim boundary. The current
  machine-checkable consistency gate is
  `218_build_paper_claim_consistency_audit.py` over
  `configs/paper_claim_consistency_contract_v1.json`.

Domain composition audit:

- script: `219_build_domain_composition_audit.py`
- config: `configs/domain_mix_contract_v1.json`
- test: `validation/test_domain_composition_audit.py`
- outputs:
  - `outputs/validation/domain_composition_audit_report.json`
  - `outputs/validation/domain_composition_audit_report.md`
- latest status: `domain_composition_audit_completed`
- current target mix status: `not_declared_for_current_paper_evidence`

Coverage/domain-mix audit:

- script: `220_build_coverage_domain_mix_audit.py`
- config: `configs/coverage_domain_mix_contract_v1.json`
- test: `validation/test_coverage_domain_mix_audit.py`
- outputs:
  - `outputs/validation/coverage_domain_mix_audit_report.json`
  - `outputs/validation/coverage_domain_mix_audit_report.md`
- latest status: `coverage_domain_mix_audit_passed_with_scope_boundary`
- current scope: `observed_paper_domain_arm_composition`
- current domain share drift: Code `-0.194002`, Math `+0.194002`
- claim boundary: observed-composition evidence only; no target-mix
  satisfaction, Utility, intrinsic-quality, or universal domain-ratio claim.

Stage-B policy contract audit:

- script: `221_build_stage_b_policy_contract_audit.py`
- config: `configs/stage_b_policy_contract_v1.json`
- test: `validation/test_stage_b_policy_contract_audit.py`
- outputs:
  - `outputs/validation/stage_b_policy_contract_audit_report.json`
  - `outputs/validation/stage_b_policy_contract_audit_report.md`
- latest status: `stage_b_policy_contract_audit_passed`
- policy boundary: Stage B is optional budget allocation over retained
  Stage-A survivors. It emits `retain_all` when no binding budget exists,
  emits `selected_for_training_budget` plus `budget_not_selected` when a budget
  binds, and never treats budget exclusion as rejection or low quality.

Canonical execution registry:

- script: `222_build_canonical_execution_registry.py`
- config: `configs/canonical_execution_path_v1.json`
- test: `validation/test_canonical_execution_registry.py`
- outputs:
  - `outputs/validation/canonical_execution_registry_report.json`
  - `outputs/validation/canonical_execution_registry_report.md`
- latest status: `canonical_execution_registry_passed`
- canonical path:
  - `211_build_code_paper_evidence_report.py`
  - `213_build_final_paper_evidence_table.py`
  - `190_run_paper_claim_release_gate.py`
  - `219_build_domain_composition_audit.py`
  - `220_build_coverage_domain_mix_audit.py`
  - `221_build_stage_b_policy_contract_audit.py`
  - `218_build_paper_claim_consistency_audit.py`
- boundary: this is the lightweight paper-evidence rebuild path. It does not
  claim to rerun raw-data acquisition, GPU training, benchmark generation, or a
  production release pipeline.
- one command: `python run_canonical_paper_evidence.py --execute`
- exit code `2` is an explicit blocked/abstain evidence decision, not a runner
  failure.
- active operator entry points: `00_run_data_eval.py` and
  `run_canonical_paper_evidence.py`; `13_run_paper_release.py` is compatibility
  only. Other numbered scripts are historical/experimental unless the registry
  explicitly marks them otherwise.

Qwen3-4B HF mixed-corpus retest protocol:

- script: `223_build_hf_mixed_corpus_retest_protocol.py`
- config: `configs/hf_mixed_corpus_retest_protocol_qwen3_4b_v1.json`
- test: `validation/test_hf_mixed_corpus_retest_protocol.py`
- outputs:
  - `outputs/validation/hf_mixed_corpus_retest_protocol_report.json`
  - `outputs/validation/hf_mixed_corpus_retest_protocol_report.md`
- latest status: `hf_mixed_corpus_retest_protocol_frozen`
- primary mix: `70%` raw-like Python code sources, `30%`
  known-high-quality reference sources
- stress mix: `90%` raw-like, `10%` reference
- HF source plan:
  - raw-like candidates: `bigcode/the-stack-v2`,
    `codeparrot/github-code`
  - reference candidates: `irds/codesearchnet`,
    `Nan-Do/code-search-net-python`
- critical leakage rule: source dataset, source tier, and
  known-high-quality labels are preserved for audit but forbidden as Stage-B
  selector inputs.

Next Stage-B improvement target: preserve concise but useful code examples,
tests, bug fixes, and API usage chunks; separate AST richness from learnable
code usefulness; report selected-vs-budget-not-selected feature shifts before
Stage C without treating the latter as rejection.

Core operational audit:

- script: `161_build_core_operational_audit.py`
- test: `validation/test_core_operational_audit.py`
- outputs:
  - `outputs/validation/core_operational_audit.json`
  - `outputs/validation/core_operational_audit.md`
- latest status: `core_operational_audit_passed`

This audit treats Core axes as operational curation responsibilities rather
than intrinsic data-quality truths. It checks that Selection Value Evidence
remains Stage-B-only with no hard-reject authority, Redundancy separates
harmful duplication from useful recurrence, and Utility remains Stage-C-only.

Important limitation: this audit is not metric behavior validity evidence. It
does not prove that Validity, Selection Value Evidence, Redundancy, or Coverage metrics measure
their intended constructs. It only verifies that the framework/spec/policy
claims remain stage-consistent.

Core construct-boundary review:

- script: `163_build_core_construct_validity_review.py`
- test: `validation/test_core_construct_validity_review.py`
- outputs:
  - `outputs/validation/core_construct_validity_review.json`
  - `outputs/validation/core_construct_validity_review.md`
- latest status: `core_construct_validity_review_passed`
- decision: intrinsic Quality as a Core is rejected; canonical Core is
  `Selection Value Evidence`, while the runtime label `Quality` is retained
  only as a legacy compatibility alias.

Selector Utility leakage audit:

- script: `164_build_selector_utility_leakage_audit.py`
- test: `validation/test_selector_utility_leakage_audit.py`
- outputs:
  - `outputs/validation/selector_utility_leakage_audit.json`
  - `outputs/validation/selector_utility_leakage_audit.md`
- latest status: `selector_utility_leakage_audit_passed`
- checked records: `1913`
- forbidden Stage-B Utility fields seen: none

Next Core-validity work is not another Utility experiment. Build a Core
behavior audit v2 using labeled, metamorphic, and adversarial fixtures for
Validity, Selection Value Evidence, Redundancy, and Coverage. Any metric that
cannot behaviorally represent its Core construct must be demoted to diagnostic.

Core behavior audit v2:

- script: `165_build_core_behavior_audit_v2.py`
- test: `validation/test_core_behavior_audit_v2.py`
- outputs:
  - `outputs/validation/core_behavior_audit_v2.json`
  - `outputs/validation/core_behavior_audit_v2.md`
- historical status (superseded by the current implementation-hash gate):
  `core_behavior_audit_behavior_checks_passed_with_evidence_gaps`
- blockers: none
- remaining evidence gaps:
  - `stage0_detector_heldout_benchmark_not_external_public_benchmark`
  - `explicit_domain_metadata_missing_for_true_domain_coverage_claim`
  - `real_corpus_stage0_hazard_counts_not_production_detector_validation`
  - `core_behavior_fixture_suite_expanded_but_not_exhaustive`
- current behavior-check counts:
  - Validity: `5`
  - Selection Value Evidence: expanded with retain-all and
    budget-not-selected behavior checks
  - Redundancy: `4`
  - Coverage: `5`
  - Utility stage boundary: `2`

Interpretation: the current expanded labeled/metamorphic checks pass for
Validity, Selection Value Evidence, Redundancy, Coverage, and Utility stage
boundary behavior. The audit now includes the frozen temporal-code Stage-B
proxy fixture contract, the heldout Stage-0 detector benchmark, and the current
real-corpus Stage-0/Coverage metadata audit. This is still not a
production-grade Core metric validity proof and does not support a release
claim by itself.

Block 3 Core claim defense:

- script: `192_build_core_claim_defense_report.py`
- test: `validation/test_core_claim_defense_report.py`
- outputs:
  - `outputs/validation/core_claim_defense_report.json`
  - `outputs/validation/core_claim_defense_report.md`
- latest status: `core_claim_defense_scoped_not_release_ready`
- release gate status: `paper_curation_stage_claim_gate_passed`

Current allowed Core claim surface:

- Validity: structural usability gate behavior only.
- Selection Value Evidence: pre-outcome selection-value proxy only; `Quality`
  remains a legacy alias and intrinsic Quality measurement is not supported.
- Redundancy: conservative duplicate and saturation control only. The current
  labeled fixture threshold is precision `1.0`, recall `0.5`, F1 `0.666667`,
  with known false-negative and binary-saturation gaps.
- Coverage: observable source/style/path/content/cluster retention only; true
  domain coverage requires explicit domain metadata.
- Utility: Stage-C downstream validation only; not a selector or Stage-B
  tuning signal.

The report intentionally keeps `release_claim_supported=false`,
`core_metric_validity_fully_proven=false`,
`intrinsic_quality_claim_supported=false`, and
`utility_in_selector_supported=false`. This is progress because the Core
claim boundary is now machine-readable, but it is not a paper-final release
claim.

Stage-0 hazard fixture benchmark:

- script: `166_build_stage0_hazard_benchmark.py`
- test: `validation/test_stage0_hazard_benchmark.py`
- fixture:
  `validation/fixtures/stage0_hazard_benchmark_cases.json`
- outputs:
  - `outputs/validation/stage0_hazard_benchmark_report.json`
  - `outputs/validation/stage0_hazard_benchmark_report.md`
- latest status: `stage0_hazard_fixture_benchmark_passed`
- cases: `10`
- passed: `10`
- blockers: none

Coverage: PII email/phone, secret, benchmark contamination, poisoning,
restricted/unknown rights, repository-code operator preservation, and one
numeric false-positive suppression case. This reduces the Stage-0 gap from
"missing fixture benchmark" to "fixture benchmark exists but not production
detector validation."

Stage-0 detector validation precheck:

- script: `170_build_stage0_detector_validation.py`
- test: `validation/test_stage0_detector_validation.py`
- fixture:
  `validation/fixtures/stage0_detector_validation_cases.json`
- outputs:
  - `outputs/validation/stage0_detector_validation_report.json`
  - `outputs/validation/stage0_detector_validation_report.md`
- latest status:
  `stage0_detector_validation_precheck_passed_with_scope_caveats`
- cases: `13`
- passed: `13`
- blockers: none
- axis metrics:
  - PII: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - secrets: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - benchmark contamination: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - poisoning: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - rights allowed: precision `1.0`, recall `1.0`, FP `0`, FN `0`

Interpretation: this is stronger than the original Stage-0 smoke fixture and
now includes token-pattern secrets, additional code benchmarks, hidden-trigger
poisoning, rights decisions, and numeric-code false-positive suppression. It
is still project-defined and small, so it is not an external production
detector benchmark.

Stage-0 detector heldout benchmark:

- script: `172_build_stage0_detector_heldout_benchmark.py`
- test: `validation/test_stage0_detector_heldout_benchmark.py`
- fixture:
  `validation/fixtures/stage0_detector_heldout_cases.json`
- outputs:
  - `outputs/validation/stage0_detector_heldout_benchmark_report.json`
  - `outputs/validation/stage0_detector_heldout_benchmark_report.md`
- latest status:
  `stage0_detector_heldout_benchmark_passed_with_scope_caveats`
- cases: `12`
- passed: `12`
- blockers: none
- axis metrics:
  - PII: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - secrets: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - benchmark contamination: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - poisoning: precision `1.0`, recall `1.0`, FP `0`, FN `0`
  - rights allowed: precision `1.0`, recall `1.0`, FP `0`, FN `0`

Interpretation: this closes the "only development fixture" weakness for the
current detector precheck. It is still project-defined, so do not claim
external public detector certification.

Block 4 Stage-0 risk boundary:

- script: `193_build_stage0_risk_boundary_report.py`
- test: `validation/test_stage0_risk_boundary_report.py`
- outputs:
  - `outputs/validation/stage0_risk_boundary_report.json`
  - `outputs/validation/stage0_risk_boundary_report.md`
- latest status: `stage0_risk_boundary_scoped_not_production_ready`
- supporting statuses:
  - `stage0_hazard_fixture_benchmark_passed`
  - `stage0_detector_validation_precheck_passed_with_scope_caveats`
  - `stage0_detector_heldout_benchmark_passed_with_scope_caveats`
  - `real_corpus_stage0_coverage_audit_passed_with_scope_caveats`

Current allowed Stage-0 claim surface:

- project-defined quarantine behavior passes development and heldout fixtures
  for PII, secrets, benchmark contamination, poisoning, and rights status;
- current real-corpus Stage-0 lineage and quarantine counts are reported;
- observable metadata supports retention auditing, not legal or production
  safety certification.

Current real-corpus Stage-0 counts:

- release candidates: `312`
- quarantined candidates: `6`
- quarantine reasons: `mojibake_detected=2`, `poisoning_suspected=4`
- rights status: `allowed=318`

Forbidden Stage-0 claims:

- production-grade PII, secret, license, benchmark-contamination, or poisoning
  detector;
- external public detector benchmark certification;
- legal clearance or license-compliance opinion;
- exhaustive benchmark-contamination removal;
- adversarial poisoning robustness;
- training-release safety certification.

Block 5 Stage-C training validation:

- script: `194_build_stage_c_training_validation_report.py`
- test: `validation/test_stage_c_training_validation_report.py`
- outputs:
  - `outputs/validation/stage_c_training_validation_report.json`
  - `outputs/validation/stage_c_training_validation_report.md`
- latest status: `stage_c_training_validation_nll_supported_curation_claim_ready`

Current Stage-C result:

- v2 confirmatory training completed `20 / 20` Qwen3-4B QLoRA runs and
  `21 / 21` heldout NLL evaluations.
- v2 NLL gate passed: curated-v2 equal-budget improves over Stage-A random by
  `0.007115888208537813` mean NLL and is directionally better than raw random
  by `0.002589671100888813`.
- 0.5B canonical proxy path passed development target-code NLL, general-text,
  general-task, and EvalPlus development guardrails, but remains
  `abstain_not_a_production_release`.
- Qwen3-4B target-size target-code NLL passed with mean reduction
  `0.005033614734808604`, with required guardrails observed.

Open production blockers:

- Production-grade Core validity is not supported by the current scoped
  evidence.

Allowed Stage-C claim: target-code NLL training-effect evidence supports the
curated arm under the frozen equal-token comparisons. Forbidden claims:
release supported, production-ready framework, completed confirmatory
guardrails, or using Stage-C outcomes to tune Stage B.

Block 6 Confirmatory decision boundary:

- script: `195_build_confirmatory_decision_boundary_report.py`
- test: `validation/test_confirmatory_decision_boundary_report.py`
- outputs:
  - `outputs/validation/confirmatory_decision_boundary_report.json`
  - `outputs/validation/confirmatory_decision_boundary_report.md`
- latest status:
  `confirmatory_decision_curation_stage_claim_passed`
- final decision: `curation_stage_claim_pass`

Current confirmatory interpretation:

- frozen v2 target-code NLL gate passed;
- required confirmatory guardrails passed;
- the current natural-budget Stage-C rerun completed with five seeds and
  matching Stage-B implementation hashes;
- current NLL and EvalPlus both favor curated over raw while using 60.75% fewer
  packed training tokens;
- the bounded curation-stage paper gate passes, while production remains blocked;
- Stage-C outcomes cannot tune Stage B or add Utility to the selector.

Remaining work beyond the bounded code-domain paper claim:

1. Resolve or explicitly bound the Math-domain abstain result.
2. Add independent domain evidence before making a cross-domain claim.
3. Expand external Core and Stage-0 detector validity evidence before any
   production claim.
4. Keep production deployment and universal data-quality claims blocked.

Block 7 Confirmatory guardrail execution progress:

- EvalPlus Docker daemon was restored through Docker Desktop and verified as
  Docker server `29.5.2 linux`.
- EvalPlus Docker execution completed all sample files that existed at the
  start of Block 7: 14 result files were produced under
  `outputs/code_domain_v2_confirmatory_qwen3_4b/evalplus_guardrail/results/`.
- Additional EvalPlus sample generation and Docker execution completed for
  `raw_random_equal_budget` seed `239` on both `HumanEval+` and `MBPP+`.
- `outputs/validation/code_domain_v2_evalplus_guardrail_report.json` was
  rebuilt and no longer reports missing raw-random seed `239`.
- General-task confirmatory report was rebuilt, but full lm-eval remains
  incomplete. A direct `batch-size 1` missing run was stopped after showing
  HellaSwag full evaluation would take hours per remaining run on the RTX
  3070 Ti.
- `outputs/validation/stage_c_guardrail_gap_report.json`,
  `outputs/validation/code_domain_v2_confirmatory_decision_report.json`, and
  `outputs/validation/confirmatory_decision_boundary_report.json` were rebuilt.
  At that point, the final decision remained `abstain_not_release_pass`.

Remaining Block 7 order:

1. Generate missing EvalPlus samples for:
   - `stageA_random_equal_budget` seeds `131`, `163`, `197`, `239`
     (`seed131` needs MBPP repair only);
   - `curated_v2_equal_budget` seeds `101`, `131`, `163`, `197`, `239`;
   - `known_high_quality_equal_budget` seeds `101`, `131`, `163`, `197`,
     `239`.
2. Re-run `144_run_code_domain_evalplus_guardrail.py` after each generated
   batch to fill Docker result files.
3. Rebuild `145_build_code_domain_evalplus_guardrail_report.py` and verify
   whether the EvalPlus confirmatory guardrail passes or fails.
4. Complete the remaining full general-task lm-eval results. Current blocker
   groups include `curated_v2_equal_budget` seeds `131`, `163`, `197`, `239`
   and `known_high_quality_equal_budget` seed `101`; rebuild the report for the
   exact full list before launching the next long batch.
5. Rebuild Stage-C gap, v2 confirmatory decision, confirmatory boundary, and
   hard paper release gate.

Coverage/domain fixture benchmark:

- script: `167_build_coverage_domain_fixture_benchmark.py`
- test: `validation/test_coverage_domain_fixture_benchmark.py`
- fixture:
  `validation/fixtures/coverage_domain_fixture_cases.json`
- outputs:
  - `outputs/validation/coverage_domain_fixture_benchmark_report.json`
  - `outputs/validation/coverage_domain_fixture_benchmark_report.md`
- latest status: `coverage_domain_fixture_benchmark_passed`
- cases: `5`
- passed: `5`
- blockers: none
- support scopes observed:
  - `explicit_domain_metadata`: `3`
  - `mixed_domain_and_source_bucket`: `1`
  - `source_bucket_fallback`: `1`

Interpretation: Coverage can claim explicit domain retention only when
metadata/URL labels support it. Source-bucket fallback and mixed buckets remain
observable retention evidence, not true domain coverage.

Scoring schema separation audit:

- script: `168_build_scoring_schema_separation_audit.py`
- test: `validation/test_scoring_schema_separation_audit.py`
- outputs:
  - `outputs/validation/scoring_schema_separation_audit.json`
  - `outputs/validation/scoring_schema_separation_audit.md`
- latest status: `scoring_schema_separation_audit_passed`
- blockers: none

Interpretation: raw scorer methods may compute diagnostic fields, including
`predictive_utility_proxy`, but canonical scoring artifacts split scorer output
through `split_metric_groups()` in `03_score_core_metrics.py`. The audit
verifies that Utility surrogate fields are not in `CORE_SELECTION_METRICS`,
that `predictive_utility_proxy` is diagnostic-only, and that unknown raw-scorer
extras are not promoted into either canonical group.

Real-corpus Stage-0/Coverage metadata audit:

- script: `169_build_real_corpus_stage0_coverage_audit.py`
- test: `validation/test_real_corpus_stage0_coverage_audit.py`
- outputs:
  - `outputs/validation/real_corpus_stage0_coverage_audit.json`
  - `outputs/validation/real_corpus_stage0_coverage_audit.md`
- latest status:
  `real_corpus_stage0_coverage_audit_passed_with_scope_caveats`
- blockers: none
- Stage-0 release candidates: `312`
- Stage-0 quarantined candidates: `6`
- Stage-A train chunks audited: `1913`
- selected chunks audited: `1424`
- repository buckets retained: `47 / 47`
- content-type buckets retained: `3 / 3`
- path-family buckets retained: `38 / 38`
- support scope: `source_or_repository_bucket_fallback`
- true domain coverage claim allowed: `false`
- caveats:
  - `true_domain_coverage_not_claimable_without_explicit_domain_metadata`
  - `stage0_hazard_counts_do_not_replace_production_detector_validation`
  - `real_corpus_audit_is_metadata_support_not_metric_validity_proof`

Interpretation: the current v2 corpus carries enough repository, content-type,
path, rights, and provenance metadata to support observable Coverage retention
auditing. It does not carry explicit domain labels, so true domain coverage
must not be claimed. The Stage-0 hazard counts are real-corpus lineage evidence,
not production detector validation.

Code-domain Stage-B feature-shift diagnostic:

- script: `162_build_code_domain_stage_b_feature_shift_report.py`
- test: `validation/test_code_domain_stage_b_feature_shift.py`
- outputs:
  - `outputs/validation/code_domain_stage_b_feature_shift_report.json`
  - `outputs/validation/code_domain_stage_b_feature_shift_report.md`
- latest status: `code_domain_stage_b_feature_shift_report_ready`
- risk flags: none

Current v2 feature-shift reading:

- selected concise-useful share: `0.561798`
- budget-not-selected concise-useful share: `0.408998`
- selected concise test/regression share: `0.268258`
- budget-not-selected concise test/regression share: `0.202454`
- selected template/boilerplate risk share: `0.070927`
- budget-not-selected template/boilerplate risk share: `0.296524`
- selected mean token proxy: `229.790730`
- budget-not-selected mean token proxy: `1003.758691`

This indicates the current v2 selector is not obviously discarding concise
useful code/test chunks or over-selecting long/AST-rich chunks. This is still a
Stage-B diagnostic only, not Utility evidence and not permission to tune from
Stage-C outcomes.

This document transfers the project context for continuing work in a fresh Codex environment. Read this before changing code.

## GPU Operating Rule

On this Windows host, check GPU state with `nvidia-smi` before launching any
Stage-C training or evaluation job.

Default local research GPU policy:

```text
physical CUDA device 1: NVIDIA GeForce RTX 3070 Ti
```

Use physical CUDA device `1` / `RTX 3070 Ti` for local QLoRA development and
confirmatory jobs, including when the `RTX 4060 Ti` is idle. Use the
`RTX 4060 Ti` only if the user explicitly approves it for a specific run, or if
the `RTX 3070 Ti` is infeasible and the user approves the fallback.

PowerShell default:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
```

## 0. Current Windows Status - 2026-06-19

Disk cleanup has removed legacy regenerable text-pipeline artifacts and old
FineWeb model/token-block outputs. The preserved active research path is now
the code-domain raw-vs-curated validation.

New code-domain training-freeze artifacts are materialized under:

```text
outputs/temporal_code_training_freeze_v1/
```

Reference pool:

- script: `136_fetch_known_high_quality_python_reference_pool.py`
- config: `configs/code_domain_known_high_quality_reference_pool_v1.json`
- output: `outputs/temporal_code_training_freeze_v1/known_high_quality_reference_pool/`
- status: `reference_pool_materialized`
- source repositories: Flask, Click, Requests, pytest, Pydantic, Poetry,
  scikit-learn, Transformers
- Stage-A-pass chunks: 1,911
- Stage-A-pass token proxy: 637,656

Equal-token training arms:

- script: `137_freeze_code_domain_equal_token_training_arms.py`
- config: `configs/code_domain_training_arm_freeze_v1.json`
- output: `outputs/temporal_code_training_freeze_v1/equal_token_arms/`
- status: `training_arms_frozen`
- common training token-proxy cap: 120,191
- arms:
  - `raw_random_equal_budget.jsonl`
  - `stageA_random_equal_budget.jsonl`
  - `curated_equal_budget.jsonl`
  - `known_high_quality_equal_budget.jsonl`

The arm JSONL files may contain slightly more than the common cap because rows
are chunk-level units. The training loader must pack or truncate every arm to
the same frozen cap. Utility, benchmark outcomes, development outcomes,
confirmatory outcomes, and human/LLM review labels are forbidden selector
signals.

2026-06-28 Block 2 update: the generic text-pipeline index and scored outputs
have been rebuilt on Windows. `outputs/index/index.sqlite` is valid again, and
`outputs/scored/scoring_manifest.json` was rebuilt with
`191_score_core_metrics_parallel.py --workers 4`.

Current generic scored corpus:

- `fineweb_edu_sample`: 245,119 scored chunks
- `openwebtext2_subset`: 1,166,648 scored chunks
- `tiny_textbooks`: 1,139,335 scored chunks
- `wikitext103_subset`: 565,236 scored chunks

Treat `outputs/scored/scoring_manifest.json` as the completion marker for
scoring. A per-dataset `outputs/scored/<dataset>.jsonl.tmp` file means the
parallel scorer is still running or failed before finalization; it is not valid
evidence.

Qwen3-4B QLoRA smoke:

- script: `138_run_code_domain_qlora_smoke.py`
- config: `configs/code_domain_qlora_smoke_qwen3_4b_v1.json`
- report builder: `139_build_code_domain_qlora_smoke_report.py`
- output: `outputs/code_domain_qlora_smoke_qwen3_4b_v1/`
- report: `outputs/validation/code_domain_qlora_smoke_qwen3_4b_report.json`
- status: `qlora_smoke_feasible`
- GPU used: physical CUDA device 1, NVIDIA GeForce RTX 3070 Ti
- arms completed: raw-random, Stage-A-random, curated, known-high-quality
- per-arm smoke budget: 1 optimizer step, 8 micro-steps
- common packed-token budget: 118,784
- frozen training token-proxy cap before packing: 120,191
- trainable LoRA parameters: 66,060,288

This establishes local 4B QLoRA feasibility only. It is not a Utility result,
not an effect-size estimate, and not a release claim. The next step is to
freeze the development training plan: seeds, optimizer steps or token budget,
heldout NLL slices, external code benchmark commands, retention guardrails,
and practical effect margin before running multi-step development arms.

Development plan freeze:

- script: `140_freeze_code_domain_development_plan.py`
- config: `configs/code_domain_development_plan_qwen3_4b_v1.json`
- report: `outputs/validation/code_domain_development_plan_qwen3_4b_report.json`
- heldout: `outputs/code_domain_development_qwen3_4b_v1/heldouts/development_code_nll_heldout.jsonl`
- status: `development_plan_frozen`
- training arms: base, raw-random, Stage-A-random, curated, known-high-quality
- primary comparison: curated vs Stage-A-random
- supporting comparisons: curated vs raw-random and base-no-update
- reference arm: known-high-quality
- development seeds: 11, 23, 37, 53, 71
- optimizer steps per run: 8
- gradient accumulation: 8
- common packed-token budget: 118,784
- training token-proxy cap: 120,191
- development heldout NLL slice: 175 chunks, 65,526 token proxy, 125 code and
  50 test chunks, 7 repositories
- practical effect margin: curated must reduce development heldout mean NLL
  versus Stage-A-random by at least 0.005; curated must be directionally no
  worse than raw-random unless below the detectable-effect floor

The freeze report explicitly records `confirmatory_outcomes_read: false`.
Do not change seeds, optimizer steps, token budgets, heldout slice, benchmark
split, guardrail margins, or practical effect margin after development
outcomes.

## 1. Project Goal

The project is a research-oriented framework for evaluating and curating training data. The goal is not to optimize one dataset's score, but to build a reliable and reproducible framework that can judge whether a dataset or selected subset is structurally usable, supported by frozen pre-outcome selection-value proxies, non-redundant, sufficiently representative for its declared scope, and useful for learning under Stage-C validation.

Read `docs/research_framing.md` before changing curation or Utility logic. Use `docs/framework_requirements_and_test_matrix.md` as the canonical requirements, pipeline-ownership, test-matrix, and milestone contract. The current research framing is:

```text
candidate corpus -> full curated pool -> optional budgeted training subset
                 -> supported LM-training release or explicit abstention
```

Data collection is assumed to happen upstream. This framework is the curation layer that decides which candidate data should actually be used for language-model training, or whether the correct decision is `reject`, `abstain`, or `insufficient_usable_data`. The same framing can support from-scratch pretraining, continued pretraining, domain adaptation, or periodic data refresh. A time-window setting is an application scenario, not the main claim.

The framework is organized around five canonical Core axes. Some older
ASCII-tree labels below still spell the compatibility alias `Quality`; read
those entries as `Selection Value Evidence`:

```text
Core 5
├─ Validity
├─ Quality
├─ Redundancy
├─ Coverage
└─ Utility
```

The central design principle is that training-data curation is not just quality filtering. Clean text, high-quality text, diverse text, and useful training text are related but not identical. The framework therefore separates metric roles and execution stages.

## 2. Core - Metric - Policy Structure

Current intended structure:

```text
Core
├─ Validity
│  └─ structural_validity_gate
│     └─ Stage A hard gate
├─ Quality
│  └─ reference_quality_score
│     └─ Stage B selection signal
├─ Redundancy
│  ├─ exact_duplicate_indicator
│  │  └─ Stage A hard gate
│  ├─ shingle_near_duplicate_indicator
│  │  └─ Stage A hard gate
│  └─ shingle_near_duplicate_risk_score
│     └─ Stage B risk signal
├─ Coverage
│  ├─ coverage-preserving selector support
│  │  └─ Stage B support, not primary objective
│  └─ subset_coverage_retention_score
│     └─ Stage C subset-level validator
└─ Utility
   └─ small_lm_probe_gain_score / evidence-aware Utility protocol
      └─ Stage C outcome validator only
```

Important rule: `Utility` must not be added back into the selector objective. It is an outcome validator, not a selector signal.

## 3. Stage A/B/C Meaning

```text
Stage A = Can this chunk be used at all?
Stage B = If a binding budget exists, which usable chunks receive it?
Stage C = Is the selected subset good as a subset?
```

Detailed roles:

- Stage 0: candidate-record provenance, normalization, and quarantine boundary
  - `ingestion/schema.py` defines the current versioned candidate contract
  - `ingestion/normalize.py` and `30_process_stage0_candidates.py` implement the initial auditable heuristic processor
  - `29_validate_stage0_contract.py` validates the contract fixture
  - `166_build_stage0_hazard_benchmark.py` validates a labeled hazard fixture benchmark
  - production-grade and real-corpus hazard detector validation remains follow-up work

- Stage A: chunk-level hard gate
  - removes structurally invalid chunks
  - removes raw and canonical-content exact duplicate chunks
  - reports fuzzy near-duplicate evidence without irreversible rejection
  - should not judge semantic usefulness or downstream Utility

- Stage B: optional budget-constrained chunk selection
  - ranks surviving chunks using Selection Value Evidence and Redundancy risk
  - uses Coverage support to avoid collapsing rare/style/source buckets
  - should not directly optimize Utility
  - emits `retain_all` when the budget is not binding
  - never treats `budget_not_selected` as rejection

- Stage C: subset-level validation
  - validates whether the final selected subset preserves Coverage
  - validates whether the subset provides learning Utility under a fixed probe protocol

## 4. Core Definitions and Current Meaning

### Validity

Validity means structural usability only. It answers whether a chunk is usable as text for model training.

It should judge:

- empty or too-short text
- encoding/control-character corruption
- excessive symbol noise
- markup/extraction residue
- broken repetition patterns
- non-language fragments

It should not judge:

- semantic quality
- educational value
- duplication
- coverage
- Utility

Current canonical metric:

- `structural_validity_gate`

Diagnostic support:

- `structural_validity_score`

### Quality

Quality is a legacy Core-axis label. Operationally it means a frozen pre-outcome selection-value proxy for observable information density, coherence-like structure, useful content support, and boilerplate risk after accounting for style and length bias. It does not mean intrinsic or ground-truth data quality.

Current canonical metric:

- `reference_quality_score`

Important caveat: high Quality/selection-value proxy alone does not imply Utility. A proxy-supported subset can still fail Utility if it is too narrow, too easy, too homogeneous, or not useful for transfer.

### Redundancy

Redundancy means duplicate or harmful repetition burden.

Current metrics:

- `exact_duplicate_indicator`
- `shingle_near_duplicate_indicator`
- `shingle_near_duplicate_risk_score`

Role split:

- raw/canonical exact-duplicate indicators are Stage A hard gates
- fuzzy near-duplicate and redundancy risk are reversible Stage B signals

Important caveat: not all recurrence is bad. Useful recurrence in definitions, examples, formulas, and technical references should not automatically be treated as harmful duplication.

### Coverage

Coverage means selected subset retention of important distributional structure.

Current canonical metric:

- `subset_coverage_retention_score`

Coverage includes source/style/semantic cluster retention and supports domain coverage only when explicit domain metadata exists. If explicit domain labels do not exist, the framework should not overclaim true domain coverage; it should report source-bucket fallback support.

Important caveat: Coverage can be stable while Utility still fails. This is expected and is one reason Utility exists as a separate Core.

Current semantic-backbone audit contract:

- compare sampled document pairs within clusters against equally scoped pairs across clusters
- require a positive pairwise separation margin and majority within-pair advantage
- retain source/domain/style anchor purity as diagnostic evidence only
- never let filename/source-bucket purity bypass semantic separation

The previous union-representative comparison inflated between-cluster similarity
and incorrectly failed OpenWebText2 Coverage. After the pairwise audit fix, the
same OpenWebText2 selected subset passes semantic Coverage (`margin=0.013766`,
`within_gt_between_fraction=0.666667`) while still failing Stage-C Utility.

### Utility

Utility means fixed-budget learning outcome measured after subset selection.

Current canonical instrument:

- `small_lm_probe_gain_score`

Default probe model:

- `sshleifer/tiny-gpt2`

Utility remains required, but it is the hardest and most sensitive Core. It should not be collapsed into a single naive pass/fail number without checking whether the probe itself is valid.

## 5. Utility Protocol: Current Intended Interpretation

Utility was redefined from a single selected-vs-baseline pass/fail number into an evidence-aware protocol.

Current Windows continuation interpretation:

```text
Utility evidence
1. destructive probe sensitivity
   - Can the small-LM probe distinguish positive/random/destructive-negative controls?
   - Control margins are interpreted against MDE so tiny differences are near-noise/inconclusive rather than decisive failures.
2. token-inventory stress
   - Does token_shuffle_negative_control reveal token-exposure confounding?
   - Below-MDE token-shuffle reversals are reported as inconclusive caveats, not confirmed confounding.
3. curation benefit
   - Does selected beat Stage-A random?
4. strict counterfactual benefit
   - Does selected beat multi-matched Stage-A random?
```

Current evidence tiers:

- `not_evaluable_utility_evidence`
  - destructive probe control ordering fails
  - do not use Utility result as selector evidence

- `probe_valid_token_exposure_caveat`
  - destructive probe ordering passes
  - token-shuffle stress suggests possible token-inventory exposure confounding
  - report as a caveated Utility signal rather than clean strict evidence

- `random_baseline_gain`
  - selected beats Stage-A random
  - curation has some benefit, but strict evidence is not established

- `random_baseline_gain_with_token_exposure_caveat`
  - selected beats Stage-A random
  - token-shuffle stress caveat is present

- `matched_baseline_inconclusive`
  - selected and multi-matched baseline are too close under CI/MDE
  - do not treat as strict gain or strict failure without stronger evidence

- `matched_baseline_gain`
  - selected beats multi-matched baseline under mean/CI/MDE criteria

- `strict_certification_ready`
  - in-domain and required OOD strict criteria pass

Baseline roles:

- `baseline_stageA_random`
  - curation benefit baseline
  - asks whether selection is better than random usable chunks

- `baseline_multi_matched_stageA_random`
  - strict counterfactual baseline
  - asks whether selected is better than a fair matched alternative

- `baseline_nuisance_matched_stageA_random`
  - operational-counterfactual candidate, not canonical
  - exactly matches length, style, domain/source bucket, and repeat pressure
  - deliberately does not match Quality or redundancy-risk selector targets
  - has no hierarchical fallback because fallback can collapse the pool into Stage-A random
  - requires certification-budget replicated validation before any promotion

Same-condition certification comparison:

```text
dataset                canonical     nuisance      anti-memorization
fineweb_edu_sample     +0.013901     -0.004971     +0.015506
openwebtext2_subset    -0.002310     -0.001620     +0.002316
```

This isolates the current Utility bottleneck as counterfactual identification:
the sign changes when only baseline construction changes. Do not promote any
baseline or tune Stage B from this result. Next run the one-factor-at-a-time
matched-control decomposition in `docs/utility_baseline_comparison.md`.

The decomposition is now complete:

```text
arm                                      FineWeb       OpenWeb
Stage-A random                           +0.013901     -0.002310
exact length/style/domain                -0.016631     -0.002824
+ repeat pressure                        -0.004971     -0.001620
+ Quality                                +0.015506     +0.002316
+ redundancy risk                        +0.025002     +0.002433
```

Quality conditioning is the consistent sign-change point. Restrictive matching
also loses common support; the final FineWeb arm matches only `3.3%` of
selected records. The current research recommendation is to pre-register
Stage-A random as the primary total operational curation estimand and retain
matched controls as conditional mechanism diagnostics. Do not change the
canonical gate until that estimand hierarchy is explicitly approved and
downstream decision reports are regenerated.

- quality/length/style/full baselines
  - diagnostic stress tests only

Sensitivity controls:

- `positive_control`
  - high learnability / high quality Stage-A records

- `stageA_random`
  - feasible random usable chunks

- `corrupted_negative_control`
  - destructive negative control; default corruption is `hash_noise`

- `token_shuffle_negative_control`
  - token-inventory stress control; always token-shuffled

- `low_quality_negative_control`
  - same-dataset low-quality diagnostic, not the canonical destructive negative

## 6. Most Important Recent Utility Bug Fix

A serious protocol issue was found in the Utility sensitivity audit.

Previous problem:

```text
Each sensitivity arm used a different Stage-A random baseline.
```

Why this was wrong:

```text
positive_control, stageA_random, corrupted_negative_control, selected
were being compared against different baseline pools.
```

That made the control ordering unreliable because arm differences could be caused by different baselines rather than real probe sensitivity.

Fix implemented:

```text
All sensitivity arms now share one common Stage-A baseline pool,
disjoint from the union of all sensitivity arms.
```

Files involved:

- `14_run_utility_causal_diagnostics.py`
- `19_run_utility_probe_power_sweep.py`

`19_run_utility_probe_power_sweep.py` now treats old per-arm-baseline sweep outputs as stale/incompatible. Do not trust older sweep outputs unless they report the common baseline policy.

Expected baseline policy string:

```text
common_stageA_baseline_disjoint_from_all_sensitivity_arms
```

## 7. Current State Before Transfer

The repository was originally cleaned for Git push. Large generated data and local artifacts were removed from version control. On the Windows continuation machine, datasets and outputs may exist locally again, but they remain generated/gitignored artifacts.

Removed intentionally:

- `outputs/` generated files
- scored JSONL files
- selected subset JSONL files
- index SQLite database
- raw dataset files
- model cache
- calibration samples
- teacher-label generated data
- local IDE/Codex files
- old legacy/archive scripts

This means full experiments cannot run immediately after clone unless datasets are restored or regenerated.

## 8. Scripts to Know

Main pipeline:

```text
01_validate_inputs.py
02_build_index.py
03_score_core_metrics.py
04_generate_subsets.py
05_build_dashboard.py
15_run_selector_baseline_audit.py
21_build_utility_transfer_gap_report.py
20_build_curation_readiness_report.py
06_validate_outputs.py
07_run_property_benchmarks.py
08_build_metric_maturity_snapshot.py
```

Runners:

```text
00_run_data_eval.py
run_canonical_paper_evidence.py
```

Utility/selector diagnostics:

```text
14_run_utility_causal_diagnostics.py
15_run_selector_baseline_audit.py
16_run_good_chunk_dropout_audit.py
17_run_policy_ablation_audit.py
18_compare_candidate_profile.py
19_run_utility_probe_power_sweep.py
20_build_curation_readiness_report.py
21_build_utility_transfer_gap_report.py
22_run_anti_memorization_probe.py
23_build_core_proxy_alignment_report.py
24_build_core_proxy_calibration_report.py
25_build_stage_c_protocol_decision_report.py
26_build_strict_baseline_control_report.py
27_build_curation_decision_report.py
28_build_paper_evidence_table.py
196_build_curation_stage_paper_package.py
197_build_paper_comparison_tables.py
198_build_paper_reproducibility_manifest.py
32_compare_utility_baselines.py
33_decompose_utility_matching.py
34_prepare_slm_update_experiment.py
35_freeze_slm_update_plan.py
36_prepare_slm_eval_holdout.py
37_run_slm_update_training.py
38_build_slm_update_pilot_report.py
```

The current Stage-C Utility diagnostics include a targeted anti-memorization matched Stage-A baseline:

```text
baseline_anti_memorization_matched_stageA_random
```

It matches quality/length/style/domain plus repeat-pressure buckets. Use `22_run_anti_memorization_probe.py` to test whether a multi-matched baseline is winning because it is longer or more repetition/template-heavy. This is diagnostic-only, is not part of the default full pipeline, and must not be added to the Stage-B selector objective. For multiple datasets, run `22_run_anti_memorization_probe.py --datasets ...`; dataset-specific files are written as `outputs/validation/anti_memorization_probe_report_<dataset>.json` and are collected by `21_build_utility_transfer_gap_report.py` alongside the legacy default report. Anti-memorization evidence must match both dataset and profile; older `canonical` anti-mem reports must not be counted as evidence for `core_proxy_length_recurrence_guard`.

For large selected subsets, `22_run_anti_memorization_probe.py` supports diagnostic-only budget/sample overrides such as `--train-token-budget`, `--eval-token-budget`, `--max-train-steps`, `--min-probe-bucket-count`, and `--max-selected-records`. These are useful for quick triage but are not certification evidence.

`21_build_utility_transfer_gap_report.py` and `20_build_curation_readiness_report.py` now expose `framework_implication` so dataset-level failures map back to framework actions: probe/control redesign, Core/Policy proxy inspection, or strict baseline control revision. `23_build_core_proxy_alignment_report.py` summarizes Core proxy versus easy-NLL tension as diagnostic-only evidence. This keeps dataset-specific diagnostics from becoming dataset-specific pass criteria.

`24_build_core_proxy_calibration_report.py` converts Core/Utility mismatch evidence into diagnostic Core proxy calibration targets. It is an audit/planning report only; do not add Utility to the Stage-B selector objective.

`25_build_stage_c_protocol_decision_report.py` records the Stage-C protocol decision per dataset. Its purpose is to keep the project aligned with the real goal: a curation framework for producing refined datasets. It can recommend different Stage-C follow-ups for different datasets, but it always keeps selector action on hold and keeps Utility out of Stage-B.

`26_build_strict_baseline_control_report.py` records strict-baseline control decisions. It separates the canonical `baseline_multi_matched_stageA_random` strict counterfactual from reported diagnostic controls such as `baseline_anti_memorization_matched_stageA_random`. Anti-memorization support can justify strict-baseline control revision/reporting, but it is not a selector objective and does not by itself permit certification claims.

`27_build_curation_decision_report.py` is the final LM-training curation decision layer. It maps the Stage A/B/C evidence matrix into explicit training-use decisions such as `needs_certification_utility`, `utility_probe_unstable`, and `strict_baseline_confounded`. This is the report that answers whether a selected subset should be used for training, held for certification Utility, or held for strict-baseline/probe revision.

`28_build_paper_evidence_table.py` converts the final Stage A/B/C reports into reproducible JSON, Markdown, and CSV evidence tables for the paper. It must report the claim boundary that Utility is Stage C validation only and never a selector objective.

`196_build_curation_stage_paper_package.py` is the current top-level paper
claim package. It joins the hard paper gate, Core claim defense, Stage-C
training validation, and confirmatory decision boundary into:

```text
outputs/validation/curation_stage_paper_package.json
outputs/validation/curation_stage_paper_package.md
```

Latest status: `curation_stage_paper_package_ready`. It supports the bounded
`curation_stage_research_framework` paper claim and keeps
`production_core_validity_not_supported` as a production-deployment blocker.
The Method section for this bounded claim is now frozen as:

```text
docs/paper_method_core_metric_policy.md
```

The limitations and threats-to-validity section is now frozen as:

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

Remaining paper-submission packaging tasks:

```text
none
```

`34_prepare_slm_update_experiment.py` prepares the pre-registered target-SLM
continued-training validation arms from frozen curation outputs. It writes
equal-budget `curated_equal_budget`, `stageA_random_equal_budget`, and
`raw_random_equal_budget` JSONL files plus a manifest under
`outputs/slm_update_experiments/<experiment>/`. The required reference arms
`stageA_all_reference` and `raw_all_reference` are recorded as summaries by
default, or materialized with `--include-reference-all`. This is a Stage-C/G4
experiment-prep step only; target-model outcomes must not feed back into
Stage-B selection.

`35_freeze_slm_update_plan.py` reads the SLM arm manifest plus
`configs/slm_update_qwen25_0p5b_experiment.json`, loads the target tokenizer,
and writes `frozen_training_plan.json` with target-token counts, matched
token budget, model/training config, required seeds, and claim boundary. The
current FineWeb-Edu frozen plan uses `Qwen/Qwen2.5-0.5B`, a primary matched
budget of `22,199,800` Qwen tokens, sequence length `1024`, and packed
sequence blocks that split long raw records. This is still Stage-C/G4
validation infrastructure only; do not use target-SLM outcomes as selector
features.

The active code-domain validation decision is now documented in
`docs/code_domain_training_validation_protocol.md`. The main next validation is
raw-vs-curated equal-budget continued pretraining on raw-like permissive Python
code. Strict retrospective E2 remains useful secondary executable evidence, but
it should no longer block the primary raw-corpus training validation.

`36_prepare_slm_eval_holdout.py` prepares Stage-A heldout eval records that are
disjoint from `curated_equal_budget`, `stageA_random_equal_budget`, and
`raw_random_equal_budget`. `37_run_slm_update_training.py` tokenizes arms into
packed blocks and runs target-SLM training/eval. `38_build_slm_update_pilot_report.py`
summarizes pilot results and explicitly marks them as non-certification
evidence.

Current target-SLM pilot status:

```text
model: Qwen/Qwen2.5-0.5B
GPUs visible: RTX 4060 Ti + RTX 3070 Ti
pilot train size: 256 sequences per arm
pilot eval size: 128 sequences
seed: 20260608
optimizer steps: 32
base_no_update NLL: 2.816260
curated_equal_budget NLL: 2.940322
stageA_random_equal_budget NLL: 2.945509
raw_random_equal_budget NLL: 2.949955
curated_minus_stageA_random_nll: -0.005186
```

Interpretation: pilot-only direction is curated better than Stage-A random and
raw random, but every update arm is worse than base no-update because the run
is intentionally tiny. This validates the runner and motivates a larger
equal-budget run; it is not a paper claim.

Scaled pilot `pilot_1024_lr1e5` completed after the smaller smoke run:

```text
train blocks: 1024 sequences per primary arm
eval blocks: 512 sequences
learning rate: 1e-5
optimizer steps: 128
primary seeds: 20260608, 20260609, 20260610
base_no_update mean NLL: 2.805226400
curated mean NLL: 2.798956268
Stage-A random mean NLL: 2.799728514
curated_minus_stageA_random_mean_nll: -0.000772247
curated_better_seed_count: 3/3
raw_random seed 20260608 NLL: 2.801262047
```

Interpretation: this is promising scaled-pilot evidence, not certification
evidence. Curated beats Stage-A random on all three primary seeds and both
primary arms improve over base no-update on the internal heldout. The effect is
small, and the LR was chosen after the first smoke run, so the next step is a
predeclared certification-scale run, not claiming success from this pilot.

Full-budget token blocks are ready:

```text
directory: outputs/slm_update_experiments/fineweb_edu_canonical_slm_update_v1/token_blocks_full
curated_equal_budget: 21679 sequences
stageA_random_equal_budget: 21679 sequences
raw_random_equal_budget: 21679 sequences
eval: 1289 sequences
```

Certification-scale run `cert_lr1e5_full` has started from the predeclared
plan `configs/slm_update_certification_plan_qwen25_0p5b_fineweb.json`.
Completed first primary seed:

```text
base_no_update NLL: 2.778654529
curated seed 20260608 NLL: 2.780961865
Stage-A random seed 20260608 NLL: 2.778531128
curated_minus_stageA_random_nll: +0.002430737
status: early_negative_signal_pause_recommended
report: outputs/slm_update_experiments/fineweb_edu_canonical_slm_update_v1/cert_lr1e5_full_certification_report.md
```

Interpretation: full-budget first seed contradicts the scaled pilot. Stage-A
random slightly beats base no-update, while curated is worse than both. Pause
before spending more full-run GPU time. Next debugging target is transfer from
1024-sequence pilot to 21679-sequence full-budget training: ordering effects,
training duration, LR schedule/overtraining, heldout composition, and whether
the curated subset is too narrow under full exposure.

The full-budget diagnosis found that selected-only curated remains higher
quality and lower redundancy, but is much shorter and distributionally farther
from the broad Stage-A heldout than Stage-A random. The first exploratory
release/training-construction follow-up interleaves a 50% selected core with
50% disjoint Stage-A coverage backfill at the same Qwen-token budget:

```text
coverage_backfilled_interleaved50_equal_budget seed 20260608 NLL: 2.777828628
Stage-A random seed 20260608 NLL:                             2.778531128
base_no_update NLL:                                           2.778654529
selected-only curated seed 20260608 NLL:                      2.780961865
```

This is the best current full-budget result and beats Stage-A random by
`0.000702499` NLL. It supports the release-layer mixture direction, but it is
not certification evidence because it was created after observing the
selected-only reversal and has only one completed full-budget seed. Read
`docs/slm_backfilled_full_result.md` and
`docs/slm_full_budget_shift_interpretation.md`.

Next target-SLM action: freeze the 50/50 interleaved mixture as a new
release/training-construction candidate, replicate it on remaining seeds, and
test it on untouched external/new heldout evidence. Do not add Utility or
target-SLM outcomes to the Stage-B selector objective.

That confirmatory protocol is now frozen:

```text
plan: configs/slm_backfill_confirmatory_plan_qwen25_0p5b_fineweb.json
fresh confirmatory seeds: 20260609, 20260610
primary eval: confirmatory_broad_stageA_eval
secondary diagnostic: confirmatory_coverage_stratified_stageA_eval
exploratory seed 20260608: excluded from confirmatory success count
exact UID overlap across frozen train/eval sets: 0
```

Run `46_validate_slm_confirmatory_contract.py` before fresh training. Read
`docs/slm_backfill_confirmatory_protocol.md`. The secondary holdout cannot
rescue a failed broad-primary result.

First fresh confirmatory seed `20260609` is complete:

```text
primary broad Stage-A delta, backfilled - Stage-A random: +0.000377098
secondary coverage-stratified delta:                      -0.000980644
```

The frozen candidate loses the primary and wins the secondary diagnostic.
Under the predeclared rule, confirmatory success is no longer possible because
both fresh seeds were required to win the primary. Stop seed `20260610`; do
not alter the rule or tune the ratio on these holdouts. Next research work is
to define the intended deployment distribution and a release-policy tradeoff,
then validate a new predeclared candidate on new evidence.

The Deployment Contract and release-policy layer is now implemented:

```text
candidate corpus + Deployment Contract -> supported training release or abstention
```

Current FineWeb decisions from the same evidence:

```text
broad_refresh -> stageA_broad
targeted_coverage_refresh -> coverage_backfilled
capability_preserving_update -> reject
```

These are scoped decisions. They do not change Stage B and do not certify
either release universally. Read `docs/deployment_contract_and_release_policy.md`.

The first provisional external retention guardrail is now complete on the
frozen WikiText103 validation/test holdout. Exact normalized-text overlap with
the FineWeb training arms is zero, but every updated model regresses against
the base model by roughly `+0.118` to `+0.120` mean NLL. The
capability-preserving contract therefore rejects both selected-only and
coverage-backfilled releases. This is provisional forgetting evidence, not a
task-benchmark, safety, near-duplicate-contamination, or deployment claim.

The retention-aware replay Pareto development run is also complete. All
training and evaluation ran on physical CUDA device 1, RTX 3070 Ti only.
WikiText `train` replay strongly improves the WikiText validation/test
retention outcome, but none of the tested replay ratios jointly passes both:

```text
target NLL < matched Stage-A random
external WikiText NLL <= base no-update
```

The final boundary check at `0.75%`, `1%`, and `1.5%` replay found no joint
pass. Stop fine-grained ratio tuning on this development evidence. Next work
should revise the training recipe or replay-source contract while keeping
Stage B unchanged. Read `docs/retention_replay_pareto_analysis.md`.

The matched training-recipe follow-up then produced the first joint-pass
development candidates. The selected confirmatory candidate is:

```text
99% target + 1% WikiText-train replay
learning rate = 5e-6
optimizer steps = 128
target gain vs recipe-matched Stage-A = +0.000152922 NLL
external regression vs base = -0.000169115 NLL
```

This is not certification evidence. The candidate, comparator, fresh seeds,
and joint success rule are frozen in
`configs/retention_recipe_confirmatory_plan_qwen25_0p5b_fineweb.json`.
Before training, create untouched target and external-retention holdouts and
run exact plus near-duplicate overlap audits.

That fresh confirmation is now complete. The frozen candidate passed seed
`20260612` but failed the target-primary comparison by `0.000011424` NLL on
seed `20260613`; external retention passed on both seeds. Because both fresh
seeds were required to joint-pass, the final status is:

```text
confirmatory_joint_not_supported
```

The replay-aware recipe has resolved the observed forgetting failure in this
provisional external-corpus check. The remaining problem is that the target
advantage over recipe-matched Stage-A random is not seed-stable. Do not tune
another ratio on the frozen holdouts.

The current Core-only follow-up candidate is `configs/core_proxy_length_recurrence_guard_probe.json`. It implements the `core_proxy_length_recurrence_guard` policy from the calibration report: lower learnability/repetition bonuses, higher useful-length support, conservative same-bucket learnability rebalance, and finer length buckets in `policy/subsets.py` for Stage-B distribution preservation and Stage-C matched baselines. Treat it as a targeted follow-up candidate, not a promoted canonical policy.

`19_run_utility_probe_power_sweep.py` treats profile mismatches as stale/incompatible. `25_build_stage_c_protocol_decision_report.py` also rejects the entire sweep report unless its top-level profile matches the active readiness report. This matters because older `canonical` sweep files can otherwise be accidentally interpreted as evidence for `paper_release_certification` or a candidate profile.

Latest Windows candidate evidence as of 2026-06-03:

- Full candidate run completed on all three datasets with CUDA physical device 1.
- `validate_outputs.py`: 298/298 pass after adding cross-report consistency checks across Utility transfer-gap, curation readiness, Stage-C protocol decision, strict-baseline control, curation decision, power-sweep, candidate-comparison, and profile-matched anti-memorization evidence reports. It now also checks that token-exposure caveat status propagates consistently across these reports. Missing-report autobuild fallback now infers the active report profile from `outputs/run_summary.json` instead of hard-coding `canonical`.
- Candidate comparison: promote candidate = false; targeted follow-up candidate = true.
- `tiny_textbooks`: Stage-C development pass under `core_proxy_length_recurrence_guard`, now explicitly classified as `stage_c_development_ready_with_token_exposure_caveat`; current-profile power sweep now has 8 compatible runs, 2 probe-valid runs, 6 selected>random runs, and 1 valid selected>random run. Best valid selected>random preset is `train_eval_hash_noise_b0`, but token-exposure caveats remain. A fresh CUDA device 1 current-profile anti-memorization run does not support selected (`delta_nll=-0.000391`, CI low `-0.001383`, MDE `0.00028689`, effect/MDE `-3.764344`; causal mode `weaker_selected_training_signal`). Next step is certification-grade Stage-C Utility/token-exposure protocol handling, not another Core proxy change.
- `wikitext103_subset`: canonical strict multi-matched Utility still fails, but a fresh CUDA device 1 rerun of the targeted anti-memorization baseline supports selected (`delta_nll=+0.001858`, CI low `+0.001254`, MDE `0.00027393`, effect/MDE `6.994378`). Current-profile power sweep now has 8 compatible runs, 2 probe-valid runs, 3 selected>random runs, and 2 valid selected>random runs. Best valid selected>random preset is `current_like_hash_noise_b0`. Treat this as strict-baseline/easy-NLL confound plus probe-protocol evidence; next step is strict baseline control revision/reporting, not selector tuning.
- `openwebtext2_subset`: power sweep has 9 compatible common-baseline runs, 3 probe-valid runs, 6 selected>random runs, and 3 valid selected>random runs. `stronger_probe_b0` still has the largest single selected>random margin, but `stronger_probe_b1` failed control ordering, so it is not the standardization target. A fresh CUDA device 1 run added `eval_power_b1`; together `eval_power_b0` and `eval_power_b1` form a replicated valid Stage-C protocol candidate family (`eval_power: [0, 1]`). Stable probe validity is still false overall, so treat this as `utility_probe_preset_instability` with a dataset-level replicated protocol candidate; do not use it as selector Utility evidence or a global default.

`19_run_utility_probe_power_sweep.py --aggregate-only` now infers datasets from existing sweep run files when `--datasets` is omitted. This prevents accidental report truncation to the tiny default after a targeted dataset sweep.

`25_build_stage_c_protocol_decision_report.py` now computes the valid selected>Stage-A-random preset intersection and the replicated valid-family intersection across datasets. Replicated-family detection is not hard-coded to `_b0`/`_b1`: it parses `family_b<number>` repeats and counts a family only when at least two compatible completed repeats are valid selected>random and no compatible completed repeat in that family failed. Current intersections are empty:

```text
openwebtext2_subset: eval_power_b0, stronger_probe_b0
openwebtext2_subset replicated family: eval_power [0, 1]
tiny_textbooks: train_eval_hash_noise_b0
wikitext103_subset: current_like_b0, current_like_hash_noise_b0
common replicated valid families across all datasets: none
```

Replication runs added on 2026-06-03 did not hold:

- `openwebtext2_subset__eval_power_b1`: passed probe controls and selected>Stage-A-random; together with `eval_power_b0`, this creates a replicated OpenWebText2-only Stage-C protocol candidate.
- `openwebtext2_subset__stronger_probe_b1`: positive control not separated; do not standardize `stronger_probe_b0` from the single b0 win.
- `tiny_textbooks__train_eval_hash_noise_b1`: selected>random remains positive, but positive/control ordering and token-stress caveat remain unresolved.
- `tiny_textbooks` current-profile anti-memorization diagnostic: selected loses to the repeat-pressure matched control, so anti-mem does not explain the token-exposure caveat or justify strict-control promotion for tiny.
- `wikitext103_subset__current_like_hash_noise_b1`: selected>random remains weakly positive, but positive control not separated.

Therefore no current Utility preset should be promoted as a global or replicated Stage-C default. Dataset-level preset candidates are protocol follow-ups only, not selector criteria.

`18_compare_candidate_profile.py` now reads `outputs/validation/stage_c_protocol_decision_report.json` as a promotion gate. Global candidate promotion is blocked unless that report shows a replicated global Utility family. Current candidate comparison explicitly reports:

```text
Stage-C protocol gate blocks promotion: True
reason: Stage-C protocol report has no replicated global Utility family.
```

`validate_outputs.py` validates candidate comparison reports and enforces that any promoted candidate must have the replicated global Utility-family gate satisfied. It also now checks that curation readiness, Utility transfer-gap, Stage-C protocol, strict-baseline control, power-sweep, and candidate-comparison reports are mutually consistent, so stale downstream reports should fail validation instead of silently carrying old decisions.

Current strict-baseline control report:

```text
fineweb_edu_sample: strict_control_supported_for_certification_candidate
openwebtext2_subset: strict_control_unresolved
tiny_textbooks: strict_control_unresolved
wikitext103_subset: strict_control_unresolved
certification_claim_allowed_dataset_count: 1
```

This means the framework now has one positive LM-training curation demonstration case. `fineweb_edu_sample` is classified by `27_build_curation_decision_report.py` as `accepted_for_training` / `certification_candidate`; the other three datasets remain negative or diagnostic stress cases under the canonical decision report.

Latest key reports:

```text
outputs/validation/candidate_profile_comparison_core_proxy_length_recurrence_guard.md
outputs/validation/curation_readiness_report.md
outputs/validation/utility_transfer_gap_report.md
outputs/validation/core_proxy_alignment_report.md
outputs/validation/core_proxy_calibration_report.md
outputs/validation/anti_memorization_probe_report.md
outputs/validation/utility_probe_power_sweep.md
outputs/validation/stage_c_protocol_decision_report.md
outputs/validation/strict_baseline_control_report.md
```

Dataset preparation:

```text
prepare_openwebtext2_subset.py
prepare_tiny_textbooks.py
prepare_wikitext103_subset.py
prepare_fineweb_edu_sample.py
prepare_reference_quality_model.py
```

`fineweb_edu_sample` is the added clean demonstration dataset. It is prepared from `HuggingFaceFW/fineweb-edu` config `sample-10BT` and is now the positive LM-training curation case alongside the existing stress-test datasets.

FineWeb-Edu smoke preparation has been tested with:

```bash
python prepare_fineweb_edu_sample.py --target-tokens 5000000 --limit 50000 --output validation/fixtures/fineweb_edu_sample
python 01_validate_inputs.py --datasets-config datasets_config.json
```

For the full demonstration run, prepare roughly 250M GPT-2 tokens:

```bash
python prepare_fineweb_edu_sample.py --target-tokens 250000000 --limit 500000 --output validation/fixtures/fineweb_edu_sample
```

`00_run_data_eval.py` defaults to dataset indexes `0 1` in dual-eval mode, so FineWeb-Edu must be explicitly selected with `--dataset-index 3` for FineWeb-only work or `--dataset-index 0 1 2 3` for the full four-dataset suite.

Latest FineWeb-Edu certification follow-up on Windows:

```text
profile: canonical
cuda device: 1
decision: accepted_for_training
training_use: certification_candidate
certification_claim_allowed: True
replicated Utility family: current_like_hash_noise
anti-memorization diagnostic: supports selected
token-exposure caveat: False
validation: 332/332 pass
```

Latest full paper-release certification run:

```text
profile: paper_release_certification
cuda device: 1
log: outputs/logs/paper_release_20260606_154230.log
fineweb_edu_sample decision: accepted_for_training
fineweb_edu_sample training_use: certification_candidate
fineweb_edu_sample strict min Utility gain: 0.000707
fineweb_edu_sample strict min delta NLL: 0.007599
fineweb_edu_sample strict min delta NLL CI low: 0.007290
profile-matched replicated Utility family: current_like_hash_noise
openwebtext2_subset: rejected_for_training
tiny_textbooks: rejected_for_training
wikitext103_subset: rejected_for_training
validation: 344/344 pass
```

During the paper-release follow-up, a profile-scope bug was found: `25_build_stage_c_protocol_decision_report.py` could reuse a canonical power-sweep family in a paper-release decision. The protocol report now requires a matching top-level power-sweep profile, and validation checks this scope. The FineWeb-Edu `current_like_hash_noise_b0/b1` runs were rerun under `paper_release_certification` before restoring the certification candidate claim.

Paper-ready evidence outputs:

```text
outputs/validation/paper_evidence_table.json
outputs/validation/paper_evidence_table.md
outputs/validation/paper_evidence_table.csv
```

Commands used for the final FineWeb-Edu follow-up:

```bash
python 22_run_anti_memorization_probe.py --profile canonical --dataset fineweb_edu_sample
python 19_run_utility_probe_power_sweep.py --profile canonical --datasets fineweb_edu_sample --presets current_like_hash_noise_b0 current_like_hash_noise_b1
python 21_build_utility_transfer_gap_report.py --profile canonical
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
python 27_build_curation_decision_report.py
python validate_outputs.py
```

## 9. Recommended First Checks After Clone

Run syntax/import check:

```bash
python -m py_compile 00_run_data_eval.py 01_validate_inputs.py 02_build_index.py 03_score_core_metrics.py 04_generate_subsets.py 05_build_dashboard.py 06_validate_outputs.py 07_run_property_benchmarks.py 08_build_metric_maturity_snapshot.py 13_run_paper_release.py 14_run_utility_causal_diagnostics.py 15_run_selector_baseline_audit.py 16_run_good_chunk_dropout_audit.py 17_run_policy_ablation_audit.py 18_compare_candidate_profile.py 19_run_utility_probe_power_sweep.py 20_build_curation_readiness_report.py 21_build_utility_transfer_gap_report.py 22_run_anti_memorization_probe.py 23_build_core_proxy_alignment_report.py 24_build_core_proxy_calibration_report.py 25_build_stage_c_protocol_decision_report.py 26_build_strict_baseline_control_report.py 27_build_curation_decision_report.py 28_build_paper_evidence_table.py 32_compare_utility_baselines.py 33_decompose_utility_matching.py 34_prepare_slm_update_experiment.py 35_freeze_slm_update_plan.py 36_prepare_slm_eval_holdout.py 37_run_slm_update_training.py 38_build_slm_update_pilot_report.py 39_build_slm_update_scaled_report.py 40_build_slm_certification_report.py 41_diagnose_slm_full_budget_shift.py 42_prepare_slm_backfilled_arm.py 43_build_slm_backfill_report.py 44_prepare_slm_confirmatory_holdouts.py 45_freeze_slm_backfill_confirmatory_plan.py 46_validate_slm_confirmatory_contract.py 47_build_slm_backfill_confirmatory_report.py 48_build_release_decision_report.py 49_build_fineweb_deployment_evidence.py release_policy.py data_eval_common.py validate_outputs.py
```

Run input validation smoke fixture:

```bash
python 01_validate_inputs.py --datasets-config validation/fixtures/mini_datasets_config.json
```

Do not expect full `04_generate_subsets.py` Utility validation to pass on the mini fixture. The mini fixture is intentionally tiny and may not contain enough Stage-A disjoint baseline pool for the full Utility protocol.

## 10. Full Pipeline After Dataset Restoration

Once real datasets are restored or regenerated:

```bash
python 01_validate_inputs.py
python 02_build_index.py
python 03_score_core_metrics.py
python 04_generate_subsets.py
python 05_build_dashboard.py
python 15_run_selector_baseline_audit.py
python 21_build_utility_transfer_gap_report.py
python 23_build_core_proxy_alignment_report.py
python 24_build_core_proxy_calibration_report.py
python 20_build_curation_readiness_report.py
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
python 27_build_curation_decision_report.py
python 06_validate_outputs.py
python 08_build_metric_maturity_snapshot.py
```

Or:

```bash
python 00_run_data_eval.py --flow full
```

Canonical paper-evidence plan:

```powershell
python run_canonical_paper_evidence.py
```

Canonical paper-evidence rebuild:

```powershell
python run_canonical_paper_evidence.py --execute
```

`13_run_paper_release.py` is retained only as a compatibility alias.

## 11. Utility Debugging Continuation Plan

Current priority order after the 2026-06-06 FineWeb-Edu certification follow-up:

### Step 0: Preserve the FineWeb-Edu positive case

FineWeb-Edu is currently the clean positive demonstration case:

```text
decision: accepted_for_training
training_use: certification_candidate
certification_claim_allowed: True
reported controls: baseline_multi_matched_stageA_random, baseline_anti_memorization_matched_stageA_random
replicated Utility family: current_like_hash_noise
```

Do not tune the Stage-B selector from this Utility result. Utility remains Stage C only. The next research use is to turn this into a paper-ready evidence table and, if needed, rerun the same evidence under the paper-release certification profile.

### Step 1: OpenWebText2 probe preset standardization

OpenWebText2 is not ready for Core proxy tuning because default Utility sensitivity is preset-dependent. The current replicated dataset-level protocol candidate is the `eval_power` family (`eval_power_b0`, `eval_power_b1`). `stronger_probe_b0` has a larger single-run margin but should not be standardized because `stronger_probe_b1` failed.

```bash
python 19_run_utility_probe_power_sweep.py --profile core_proxy_length_recurrence_guard --datasets openwebtext2_subset --presets eval_power_b0 eval_power_b1 --force
python 21_build_utility_transfer_gap_report.py --profile core_proxy_length_recurrence_guard
python 20_build_curation_readiness_report.py --profile core_proxy_length_recurrence_guard
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
python 27_build_curation_decision_report.py
python validate_outputs.py
```

Interpretation rule:

- compatible runs must use `common_stageA_baseline_disjoint_from_all_sensitivity_arms`
- if only some presets are probe-valid, do not use the dataset as selector Utility evidence
- if a replicated preset family consistently validates controls and selected>random, standardize that family as a reported Stage-C diagnostic/certification candidate, not as a selector objective

### Step 2: Wikitext strict-baseline control revision

Wikitext selected chunks are supported by the anti-memorization diagnostic but fail the default strict multi-matched baseline.

```bash
python 22_run_anti_memorization_probe.py --dataset wikitext103_subset --profile core_proxy_length_recurrence_guard --profiles configs/core_proxy_length_recurrence_guard_probe.json
python 21_build_utility_transfer_gap_report.py --profile core_proxy_length_recurrence_guard
python 23_build_core_proxy_alignment_report.py
python 24_build_core_proxy_calibration_report.py
python 20_build_curation_readiness_report.py --profile core_proxy_length_recurrence_guard
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
python 27_build_curation_decision_report.py
python validate_outputs.py
```

Interpretation rule:

- anti-memorization support is Stage-C diagnostic evidence only
- it can justify revising/reporting strict baseline controls
- it must not be added to the selector objective

### Step 3: TinyTextbooks certification follow-up

TinyTextbooks now passes Stage-C development validation under the candidate profile, but the report status must remain `stage_c_development_ready_with_token_exposure_caveat`. The remaining caveat is token-exposure inconclusiveness, so the next step is certification-grade Utility, not another selector/Core tweak.

### Step 4: Interpret Utility correctly

If probe control ordering fails:

```text
Do not call it selector failure.
Call it probe/protocol sensitivity failure.
```

If selected beats Stage-A random but not multi-matched baseline:

```text
Report curation benefit but not strict counterfactual benefit.
```

If selected and multi-matched baseline differ only by tiny deltas below MDE/CI:

```text
Report inconclusive strict counterfactual evidence.
```

Do not force thresholds just to make Utility pass. The point is to find a defensible protocol.

## 12. Important Research Decisions Already Made

Do not undo these without strong reason:

- Keep five canonical Core axes: Validity, Selection Value Evidence,
  Redundancy, Coverage, Utility. Preserve `Quality` only as a legacy alias.
- Keep Utility as required Core.
- Do not use Utility in selector objective.
- Keep Stage A/B/C separation.
- Treat Coverage and Utility as subset-level validators.
- Treat Utility failure carefully: it may indicate probe invalidity, curation weakness, strict counterfactual weakness, or transfer limitation.
- Do not claim that curation universally improves Utility unless strict evidence supports it.
- Do not overclaim domain coverage when only source/style/cluster support exists.
- Allow `insufficient_usable_data`; the framework must not force a training-use
  claim when the Stage-A pool cannot support selection plus disjoint validation.
- OpenWebText2 exposed and now has a fix for a Stage-B/Stage-C style-taxonomy
  mismatch. The canonical full-text taxonomy removes the false
  technical-reference concentration signal. A certification-budget,
  profile-matched anti-memorization diagnostic supports selected across all 16
  cells (`delta_nll=+0.002316`, CI low `+0.000386`, minimum effect/MDE
  `2.239582`). Treat the canonical strict failure as an easy-NLL baseline
  confound candidate; hold Stage B and revise/report Stage-C strict controls.
  Read
  `docs/openwebtext2_failure_analysis.md` before tuning its Core/Policy behavior.

## 13. Known Pitfalls

### Pitfall 1: Randomness can drop good chunks

Chunk selection is ratio/budget constrained. A good chunk can be excluded if:

- the target ratio is too small
- cluster quotas are already filled
- redundancy risk pushes it down
- coverage support favors another bucket
- tie-breaking/random order affects boundary cases

This is why good-chunk dropout audit exists:

```text
16_run_good_chunk_dropout_audit.py
```

### Pitfall 2: Quality can conflict with Coverage

Selecting only top-quality chunks can collapse distributional coverage. Stage B therefore includes coverage-preserving support and Stage C validates Coverage.

### Pitfall 3: Coverage can pass while Utility fails

This is expected. Coverage only says the subset preserves distributional structure. It does not prove the subset improves learning under a fixed probe.

### Pitfall 4: Utility deltas are small

Small LM probe deltas can be numerically tiny. A value like `-0.0003` does not automatically mean catastrophic failure. It means selected training produced slightly worse held-out NLL than the baseline under that protocol. Interpretation must consider CI, MDE, probe validity, and baseline fairness.

### Pitfall 5: Probe invalidity is not selector failure

If positive/random/negative control ordering fails, the Utility instrument itself is not reliable for selector judgment on that dataset/preset.

## 14. Paper/Research Positioning

The intended paper framing is:

```text
A Stage-Based Framework for Reliability-Aware Training Data Evaluation and Curation
```

Safe claim:

- The framework defines a reproducible Core-Metric-Policy contract.
- Validity, Quality, Redundancy, and Coverage are relatively stable.
- Utility is the hardest axis and exposes limitations that upstream metrics cannot reveal.
- The framework is valuable because it separates gates, selection signals, subset validators, and diagnostic evidence.

Avoid claiming:

- curation always improves Utility
- tiny-gpt2 is a universal Utility proxy
- quality/coverage imply learning benefit
- current datasets prove universal general-purpose transfer

## 15. Current Repository Hygiene Rule

Keep Git focused on experiment code and reproducibility config.

Do not commit:

- `outputs/`
- raw datasets
- model caches
- large scored/subset JSONL files
- generated dashboards/logs
- local IDE files
- temporary transfer zip files

The `.gitignore` is set up for this.

## 16. What To Tell A New Codex Session

Use this prompt after cloning:

```text
Read HANDOFF.md and README.md first. Continue the training-data evaluation framework from the current codebase. Preserve the Core-Metric-Policy and Stage A/B/C design. Utility is Stage C only and must not be added to selector objective. The next important work is to restore/regenerate datasets, rerun the common-baseline Utility sensitivity audit, and interpret Utility using probe sensitivity, selected > Stage-A random, and selected > multi-matched baseline evidence.
```

## 17. Latest Retention-Aware Target Stability Result

The replay-aware `99% target + 1% replay`, `lr=5e-6`, `128-step` candidate
consistently fixes the observed external forgetting problem, but the frozen
two-seed confirmatory rule remains unsupported because one fresh target result
is effectively zero.

Latest diagnostics:

- `59_run_target_effect_power_diagnostic.py`: paired evaluation has enough
  sensitivity; the failing seed is genuinely near zero.
- `60_run_training_trajectory_diagnostic.py`: at step `128`, three additional
  development seeds are target-positive and external-retention-safe.
- `61_run_target_holdout_shift_diagnostic.py`: development and confirmatory
  train tensors are identical; one near-zero model pair changes sign across
  target holdouts.
- `62_build_target_effect_stability_report.py`: synthesizes the evidence.

Current interpretation:

```text
The candidate has a small positive development effect, but its margin is too
close to zero to support a robust release claim across seed and holdout
variation.
```

Do not tune on the frozen confirmatory holdout. The next protocol revision must
predeclare a practical effect margin, seed replication count, distinct target
holdouts, and task-based outcomes. Utility remains Stage C only.

That future-only protocol is now frozen in
`configs/target_effect_release_protocol_v1.json`. It is non-retroactive and
does not change the current candidate's rejected confirmatory status.

## 18. Next Raw-Corpus Experiment

The next operational experiment is preregistered in
`docs/temporal_code_curation_preregistration.md` and
`configs/temporal_code_curation_protocol_v1.json`.

- Primary model: `Qwen/Qwen3-4B-Base`
- Transfer replication model: `bigcode/starcoder2-3b`
- Domain: permissively licensed Python merged-PR change bundles
- Eligible training dates: `2025-05-01` through `2025-12-31`
- Primary splits: time-disjoint and repository-disjoint
- Local training claim: matched QLoRA continued pretraining only
- Primary Utility: repository-disjoint executable task aggregate
- Utility and benchmark outcomes: Stage C only

The change-bundle candidate schema, repository assignment manifest, and
benchmark-quarantine manifest are implemented. A bounded smoke corpus has been
fetched, but no bundle or repository is approved for training.

Those pre-collection contracts are now implemented:

- `ingestion/code_change.py`
- `ingestion/temporal_code_manifests.py`
- `63_build_temporal_code_collection_manifests.py`
- `validation/test_temporal_code_ingestion.py`

The fixture report admits one clean bundle, rejects a repository/time-window
split mismatch, and quarantines a benchmark repository. The next task is an
authenticated metadata-only GitHub discovery collector that produces the real
repository candidate manifest before any training content is downloaded.

Authenticated discovery produced 499 candidates. A frozen bounded smoke plan
then selected one repository per split and fetched two sampled pull requests
per repository:

- train: `scrapy/scrapy`
- development: `pennyw0rth/netexec`
- confirmatory: `skrub-data/skrub`

The fetch produced 6 bundles and 35 file records. `71_audit_temporal_code_smoke_bundles.py`
confirmed:

- 5 / 6 bundles satisfy the change-bundle contract
- 6 / 6 satisfy repository-disjoint and time-window split rules
- 0 / 6 match the current SWE-bench quarantine identities or hashes
- 0 / 6 are Stage-0 release candidates

The empty development bundle has no allowed text files and is contract-invalid.
Code-aware PII calibration reduced quarantined files from 15 to 3 while
retaining email, high-confidence phone, and secret detection. The remaining
three quarantines are documentation files containing email addresses.
Deterministic filename/header generated-file detection completed for all 32
persisted-content files and found no generated files.

Frozen repository-specific smoke commands were verified in Docker for all 6
bundles and all 12 parent/merge checkouts. Test execution used no network, a
read-only root filesystem, dropped capabilities, no-new-privileges, and
bounded CPU, memory, PID, and timeout settings. Isolation feasibility changes
are recorded in `configs/temporal_code_smoke_test_commands_v1.json`; they did
not alter the selected repositories, bundles, or test targets.

Current audit:

- 3 / 6 bundles pass both the training-content and executable-evaluation gates
- 2 / 6 are quarantined by the corrected SWE-bench derived artifact manifest
- 1 / 6 is rejected because it has no allowed text files

The corrected v2 SWE-bench artifact manifest preserves the benchmark name and
includes token SimHash and normalized Python AST fingerprints without retaining
raw task content. The current bounded smoke has no eligible train bundle and
therefore correctly abstains as `insufficient_usable_data`.

The first Stage-A smoke exposed and fixed a critical adapter issue: generic
text normalization collapsed Python whitespace and removed angle-bracket
expressions. Repository-code Stage 0 now preserves source text exactly except
for line-ending normalization.

Syntax-aware Stage A is implemented in `ingestion/code_chunks.py` and
`74_run_temporal_code_stage_a_smoke.py`:

- Python code/test files chunk on top-level AST boundaries
- documentation chunks preserve paragraph groups
- each split is gated independently
- fuzzy near-duplicate diagnostics require both SimHash candidate proximity and
  verified token-shingle overlap, but do not reject at Stage A
- Quality, Coverage, Utility, and benchmark outcomes are forbidden inputs

Latest bounded Stage-A smoke after corrected quarantine:

- Stage-0 input records: 4
- syntax-aware chunks: 34
- Stage-A pass: 23
- Stage-A reject: 11
- train Stage-A pass: 0
- Stage-B operational decision: `insufficient_usable_data`

Run the bounded fetch and audit with:

```powershell
conda run --no-capture-output -n research python 70_fetch_temporal_code_smoke_bundles.py
conda run --no-capture-output -n research python 71_audit_temporal_code_smoke_bundles.py
conda run --no-capture-output -n research python 72_verify_temporal_code_test_commands.py
conda run --no-capture-output -n research python 73_prepare_temporal_code_stage0_candidates.py
conda run --no-capture-output -n research python 74_run_temporal_code_stage_a_smoke.py
conda run --no-capture-output -n research python validation\test_temporal_code_smoke_audit.py
```

The frozen train-only Stage-B contract is now implemented in
`ingestion/code_selection.py`, `75_run_temporal_code_stage_b_smoke.py`, and
`configs/temporal_code_curation_protocol_v1.json`.

The bounded Stage-B smoke now abstains because its corrected train pool is
empty. Empty-input index equivalence passes without crashing. The broad
tranche is the active Stage-B operational evidence.

Next work, in order:

1. Fetch a bounded broad-corpus tranche under the frozen 314-repository
   manifest and apply the same automated content, quarantine, split, Stage-0,
   and Stage-A gates used by the bounded smoke.
2. Build broad equal-token Stage-B selected and common disjoint Stage-A-random
   arms without using Utility, benchmark outcomes, or human/LLM labels.
3. Confirm Qwen3-4B-Base QLoRA feasibility and freeze the practical
   task-effect margin before Stage-C outcomes are observed.
4. Then run the equal-budget Stage-C/4B comparison; keep the common disjoint
   Stage-A baseline invariant across every sensitivity arm.

The indexed exact redundancy search is active. On all 175 broad-tranche train
Stage-A-pass chunks it matches all-pairs exactly: risk delta `0`, objective
delta `0`, selected symmetric difference `0`, and baseline symmetric
difference `0`.

Real-corpus proxy review expansion:

- broader discovery remains metadata-only: 499 candidates
- broad metadata/path-only enrichment completed: 314 / 499 candidates pass
- broad parent/merge commit reproducibility completed: 314 / 314 pass
- the versioned broad freeze rule produced a 314-repository manifest:
  train 250 / development 38 / confirmatory 26
- manifest membership is not content-gate, Stage-0, Stage-A, Stage-B, Stage-C,
  or training approval
- deterministic broad tranche frozen: 20 repositories
  (train 12 / development 4 / confirmatory 4), maximum 40 PR bundles
- broad tranche fetched: 40 bundles / 81 file records
- broad tranche pre-execution audit: 26 / 40 contract-valid, 40 / 40
  split-valid, 1 benchmark quarantine match, 6 PII-quarantined files
- broad training-content gate pass: 19 / 40
- broad executable-evaluation gate pass: 0 / 40
- pre-execution filtering leaves 19 bundles whose sole blocker is
  `test_command_not_verified`
- isolated Docker verification ran for all 19 candidates; none passed the
  frozen generic execution hypothesis, and host fallback remained forbidden
- first additional train repository: `bytedance/deer-flow`, contributing 10
  review-only Stage-A-pass chunks
- `mem0ai/mem0` was attempted under the same frozen rule but its sampled PRs
  yielded no allowed text files; the negative fetch result is preserved
- next additional train repository: `scikit-learn/scikit-learn`, contributing
  3 review-only Stage-A-pass chunks
- combined review corpus: 331 chunks across 3 repositories
- blind packet: 72 records, with scores, arms, paths, repositories, and
  sampling strata hidden
- optional diagnostic reviewer packets: 72 records each, currently unlabeled
- review completion is not required for Stage-B approval or Stage-C entry
- review outcomes cannot promote, reject, or tune the selector

All expansion content is review-only. Its test commands are unverified, and
it is not a Stage-0 release candidate or training-approved input. Read
`docs/metric_evidence_audit.md` before interpreting citations or Stage-B
parameters.

Latest local contract verification:

```text
validate_outputs.py: 270 / 270 pass after E2 guardrail, retention, and primary-source assessment integration
temporal-code focused regression suite: pass
git diff --check: pass
```

This verifies available implementation and evidence contracts only. It does
not promote the 254 broad Stage-A-pass chunks or the 94 broad Stage-B-selected chunks to
training approval.

## 2026-06-13 Path-Stratified Fresh-Tranche Update

The original broad tranche is retained as negative sampling-frame evidence.
Its Stage-C blockers must not be fixed by tuning Stage B.

A fresh path-only pre-collection frame is now frozen:

- v1 correctly abstained because development and confirmatory had zero
  `test_only` representative candidates
- v2 feasibility amendment used path-metadata availability only, before
  content fetch or Stage outcomes
- one PR per repository; `code_and_test` and `code_only` strata only
- 40 repositories: train 24 / development 8 / confirmatory 8
- fetch: 40 bundles / 244 file records
- content gate: 27 / 40 pass
- Stage 0: 106 release-candidate records
- Stage A: 868 pass, including 523 train chunks
- Stage B: 423 selected / 79 disjoint Stage-A-random, token ratio 0.999634
- documentation share 0.011472; largest bundle share 0.198853
- selected redundancy risk 0.192428 versus random 0.364932
- full selector differs from Quality-only

Isolated Docker verification produced one executable train bundle and one
executable development bundle. A separately frozen untouched confirmatory
expansion produced one executable confirmatory bundle. The combined
corpus-side readiness status is `ready_for_stage_c_smoke`.

This is not a Utility or training-benefit result. The Qwen3-4B feasibility
smoke is complete. Next:

1. Freeze the development practical effect margin.
2. Freeze development seeds and the executable evaluation aggregate.
3. Freeze general/code retention non-inferiority guardrails.
4. Run development Stage C before touching confirmatory outcomes.
5. Preserve human/LLM review as optional diagnostic only.

The single verified development executable bundle is not sufficient for a
task-distribution claim, regardless of training-seed count. An outcome-free
development expansion is frozen in
`outputs/temporal_code_collection/temporal_code_development_execution_expansion_plan.json`.
It includes all 11 remaining repository-disjoint development `code_and_test`
candidates and must be fetched and verified before the development evaluation
aggregate or practical effect margin is finalized.

The expansion has now been fetched and generically verified:

- fetched: 11 bundles / 132 file records
- collection gate: 7 / 11 pass
- frozen generic execution candidates: 7
- additional executable bundles verified: 0
- failure accounting: 13 commit build failures / 1 test failure
- total verified development executable bundles remains 1

The machine-readable decision is
`outputs/validation/temporal_code_development_expansion_readiness.json` with
status `development_stage_c_blocked_insufficient_executable_holdout`. This is
not a Stage-B or curation failure. Do not run development Utility yet. The next
admissible work is an outcome-independent repository-native execution-recipe
freeze or a metadata-only development sampling-frame expansion.

The outcome-independent repository-native recipe experiment has now run twice
as an explicitly post-generic-failure exploratory diagnostic:

- structured test/dev extras found: 5 / 11 repositories
- non-default project-declared Python images: 6 / 11 repositories
- generic build-pass commits: 1
- native-recipe build-pass commits: 5
- parent-and-merge executable bundles recovered: 0

The stopping decision is
`outputs/validation/temporal_code_native_execution_refinement_report.json`.
Do not add repository-specific exceptions on this same development pool:
further refinement would be outcome-guided overfitting. Development Utility
remains blocked. Move to a fresh metadata-only expansion, a reusable external
task harness, or explicit execution-support tiers.

The fresh metadata-only development expansion is also complete:

- used every remaining repository-disjoint development candidate with frozen
  `test_only` or `code_only` path evidence: 14 repositories
- fetched 14 bundles / 31 file records
- collection gate: 12 / 14 pass
- unchanged generic rule: 2 build-pass commits / 0 verified bundles
- unchanged native rule transfer: 4 build-pass commits / 2 individual
  test-pass commits / 0 parent-and-merge verified bundles
- total verified development executable bundles remains 1

The decision is
`outputs/validation/temporal_code_development_fresh_expansion_report.json`
with status `raw_repository_execution_support_insufficient`.

This demonstrates that candidate-corpus acquisition and executable-task
acquisition are separate capabilities. Do not broaden raw-repository discovery
solely to recover executable tasks. The next architecture step is to freeze a
prevalidated executable-task harness or explicit repository execution-support
tiers. Development Utility and confirmatory outcome inspection remain blocked.

Execution support is now represented as an axis orthogonal to training-content
eligibility:

- `C1`: training-content gates pass
- `E0`: no frozen isolated execution candidate
- `E1`: isolated command attempted but parent-and-merge verification absent
- `E2`: frozen isolated command passes on both parent and merge

Across the current 77 audited bundles, 54 are `C1`-eligible training-content
candidates and only 3 are executable Stage-C eligible. `C1/E0` and `C1/E1`
content must not be rejected merely because it lacks execution support, while
only `E2` may enter executable Stage C. Execution tier is forbidden from the
Stage-B selector.

The independent executable-task harness contract is frozen in
`configs/temporal_code_executable_task_harness_v1.json` and materialized at
`outputs/temporal_code_collection/temporal_code_executable_task_harness_plan.json`.
Harness tasks are evaluation-only, must be `E2`, and must remain repository-
and time-disjoint. Their task prompts, tests, patches, and solutions may never
enter training. An arbitrary minimum task count is forbidden; task count must
be computed from a frozen practical margin and desired confidence-interval
precision before development model outcomes.

Independent harness feasibility has now been profiled outcome-free:

- `E2` is generalized by task class; repository-patch and function-generation
  tasks have separate frozen reproducibility contracts.
- The primary paired executable-success practical margin is frozen at 0.05
  absolute, with five training seeds and a conservative one-sided 95% task-
  distribution requirement of 1,083 tasks.
- `104_acquire_swebench_harness_metadata.py` retained metadata only. From 500
  SWE-bench Verified tasks, the frozen repository/time rule yields 49
  development and 38 confirmatory candidates, zero repository overlap, and
  zero locally E2-prevalidated tasks. The 87 candidates are below the 1,083
  task precision requirement, so SWE-bench Verified cannot be the sole primary
  executable aggregate.
- EvalPlus 0.3.1 is installed in `research`; the official loaders expose 164
  HumanEval+ and 378 MBPP+ tasks. Native Windows remains incompatible with its
  Unix-only reliability guard, but the existing Docker Desktop Linux WSL2
  backend is operational. `105_prevalidate_evalplus_guardrail.py` builds a
  frozen image and runs with no network, read-only root, no capabilities, and
  no-new-privileges. All frozen reference controls pass and all fixed negative
  controls are rejected. The evaluator is now `E2`.
- `106_freeze_evalplus_guardrail_split.py` freezes 284 development and 258
  untouched confirmatory tasks. EvalPlus remains an external code guardrail,
  never the sole primary temporal aggregate.
- `107_freeze_temporal_code_retention_guardrails.py` freezes code, general-task,
  and general-text non-inferiority margins before model outcomes.
- `108_build_temporal_primary_source_assessment.py` correctly abstains: only
  two current project-created development/confirmatory temporal E2 tasks exist
  versus the frozen 1,083-task requirement. Current SWE-bench Verified,
  EvalPlus, and LiveCodeBench sources cannot replace the primary temporal
  distribution without weakening the study retroactively.

Next work, in order:

1. Scale forward acquisition of post-training-window temporal tasks under the
   frozen rule and prevalidate E2 support at acquisition time.
2. Abstain until the primary temporal task-distribution contract is met; do not
   lower the frozen 1,083-task requirement because current sources are sparse.
3. Run development Utility only after the primary temporal aggregate exists.
4. Keep EvalPlus, SWE-bench Verified, and LiveCodeBench as secondary/guardrail
   evidence according to their frozen roles.
5. Keep confirmatory outcomes untouched until the development decision and all
   margins are frozen.

The first forward E2 infrastructure pilot is now complete:

- contract: `configs/temporal_code_forward_e2_acquisition_v1.json`
- metadata discovery: `110_discover_temporal_code_forward_e2_pilot.py`
- outcome-free recipe freeze: `111_freeze_temporal_code_forward_e2_pilot_recipes.py`
- isolated parent-fail/merge-pass verification:
  `112_verify_temporal_code_forward_e2_pilot.py`
- productivity report: `113_build_temporal_code_forward_e2_productivity_report.py`
- pilot result: 16 metadata candidates / 5 execution candidates / 2 task-valid
  E2 tasks
- observed metadata-to-E2 yield: 0.125
- observed execution-to-E2 yield: 0.4
- point-estimate capacity for 1,083 E2 tasks: 8,664 metadata candidates and
  2,708 execution attempts

The pilot exposed and fixed a pre-execution validity flaw: broad test-directory
membership could select support modules such as `tests/models.py` or
`conftest.py`. The frozen feasibility amendment now permits only Python test
modules named `test_*.py` or `*_test.py` to establish task validity. This was
changed before forward-pilot execution outcomes were observed.

The two successful pilot tasks demonstrate that forward task-valid E2
acquisition is possible. They do not enter evaluation because the
infrastructure pilot intentionally used existing train repositories. The three
invalid candidates all failed on the merge-test side due to target/repository
or environment-recipe support, not because the parent test passed. The
capacity estimates are planning-only because the pilot is too small for an
inferential yield claim. Development Utility remains blocked and confirmatory
model outcomes remain untouched.

The first actual development-window snapshot is complete:

- fresh authenticated repository discovery: 917 repositories
- exclusions against the existing broad frozen manifest: 158
- snapshot-001 fresh frame: 200 repositories
- frozen time range: `2026-06-15..2026-06-15`
- metadata candidates: 0
- execution recipes / E2 executions: 0 / 0
- training-repository overlap: 0

Zero candidates on the first day of the window is valid acquisition evidence,
not a reason to loosen task validity. Snapshot-001 is preserved without
retroactive expansion. Before any later snapshot task metadata is read, the
continuing accumulation plan is frozen at
`outputs/temporal_code_collection/temporal_code_forward_development_accumulation_plan.json`.
The initial 759-repository continuing frame was then audited against the
pre-existing productivity estimate. Under the frozen one-task-per-repository
rule it could produce at most 759 metadata candidates and approximately 94 E2
tasks at the pilot yield, versus the 542-task development target. This was a
structural capacity blocker, not a task-outcome result.

Before reading later snapshot task metadata, repository discovery was expanded
under frozen source-coverage strata:

- unchanged `stars>=20` query exhausted to GitHub's supported depth
- separately frozen `stars 5..19` and `stars 0..4` query strata
- combined metadata-only repository discovery: 12,067
- frozen fresh future accumulation frame: 5,000
- existing broad-manifest overlap: 0
- benchmark-source overlap: 0
- point-estimate metadata-candidate requirement: 4,336
- expected E2 at pilot yield if each repository produces one candidate: 625

This resolves the structural repository-frame capacity blocker only. It does
not guarantee eligible task production, and expanded-frame task metadata has
not been read. Actual development task candidates and E2 tasks remain zero,
development Utility remains blocked, and confirmatory model outcomes remain
untouched.

The large-frame forward collector is now operational:

- `121_freeze_temporal_code_forward_collection_schedule.py` freezes 25
  deterministic shards of 200 repositories each.
- `122_collect_temporal_code_forward_snapshot_shard.py` writes immutable
  later-date shard snapshots and refuses to overwrite an existing identity.
- `123_build_temporal_code_forward_candidate_ledger.py` accumulates snapshots,
  deduplicates repeated observations, and enforces one earliest eligible task
  per repository before recipe metadata.
- `124_build_temporal_code_forward_operations_status.py` reports every gate and
  the next admissible action.
- `125_run_temporal_code_forward_operations.py` refreshes the operating state
  or collects one shard and rebuilds the ledger/status in one command.
- `126_freeze_temporal_code_forward_recipe_batch.py` freezes at most 25
  outcome-independent project-metadata recipes after candidates exist.
- `112_verify_temporal_code_forward_e2_pilot.py` now accepts both quarantined
  pilot recipes and actual development recipe batches while preserving the
  same isolated parent-fail/merge-pass semantics.

Fixture-based end-to-end validation covers strict test-module classification,
immutable shard collection, candidate discovery, and deterministic
one-task-per-repository ledger deduplication. Current operations state:

- schedule: 5,000 repositories / 25 shards
- immutable later-date snapshots: 0
- cumulative candidates: 0
- candidate gap: 542
- recipe freeze allowed: false
- E2 execution allowed: false
- development Utility allowed: false

Use:

```powershell
conda run --no-capture-output -n research python 125_run_temporal_code_forward_operations.py --action refresh
conda run --no-capture-output -n research python 125_run_temporal_code_forward_operations.py --action collect --available-through YYYY-MM-DD --shard-index N
```

Do not reuse a snapshot identity, overwrite immutable snapshots, or run recipes
before the candidate ledger is frozen.

The Qwen3-4B Stage-C feasibility smoke contract is frozen in
`configs/temporal_code_stage_c_smoke_qwen3_4b_v1.json`.

- target-token arm manifest: 409 curated records / 79 common Stage-A-random
  records / 22 raw-random records
- all arms pack to exactly 68 blocks x 2048 = 139,264 target tokens
- the common Stage-A baseline SHA is reused for every sensitivity comparison
- QLoRA dependencies are installed in conda environment `research`
- Qwen3-4B tokenizer and model weights are cached
- all three arms completed 8 QLoRA optimizer steps on the physical RTX 3070 Ti
  with the same seed, packed-token budget, and compute recipe
- the machine-readable feasibility report is
  `outputs/validation/temporal_code_stage_c_smoke_report.json`

The smoke is execution-feasibility evidence only. Its training-loss ordering
is not Utility. Do not tune Stage B from it, do not make a curation-benefit or
release claim from it, and do not inspect confirmatory outcomes.

## Retrospective GitHub Development Tasks

The future-only development acquisition bottleneck has been replaced with a
retrospective-development path using already-public post-base-release GitHub
PRs. Future tasks remain reserved for untouched confirmation.

Completed:

- `127_freeze_temporal_code_retrospective_development_schedule.py` froze a
  `2025-05-01..2026-05-31` development window before task metadata.
- `128_collect_temporal_code_retrospective_shard.py` and
  `129_run_temporal_code_retrospective_collection.py` scanned all 5,000 frozen
  repositories: 25 immutable shards, 1,666 strict metadata candidates, zero
  training overlap, and one task per repository.
- `126_freeze_temporal_code_forward_recipe_batch.py` froze the first 25
  execution recipes. All 47 frozen targets satisfy `test_*.py` or `*_test.py`.
- `112_verify_temporal_code_forward_e2_pilot.py` found 4 strict task-valid
  `E2` tasks. Failures were 8 merge-build failures, 12 merge-test failures,
  and 1 parent test that passed and was therefore non-discriminative.
- `130_build_temporal_code_retrospective_development_report.py` records the
  result and keeps every capacity projection explicitly planning-only.
- `131_freeze_temporal_code_retrospective_expansion_schedule.py` freezes the
  remaining unchanged-rule frame before reading its task metadata.
- `132_build_temporal_code_retrospective_combined_ledger.py` merges initial
  and expansion snapshots inside the frozen 11,822-repository disjoint
  universe and still enforces one task per repository.
- `133_build_temporal_code_retrospective_operations_status.py` records the
  single current operating state and Utility gate.

Current interpretation:

- Existing high-quality GitHub code can produce valid temporal executable
  development tasks; the approach is feasible.
- 1,666 candidates are not equivalent to 1,666 `E2` tasks.
- At the first-batch 16% rate, the current candidate pool projects to 266
  `E2`, below the frozen development target of 542.
- The combined discovery has 12,067 repositories. After excluding the initial
  5,000 and 245 additional training-overlap repositories, the expansion
  schedule contains 6,822 disjoint repositories in 35 shards.
- Scanning the full disjoint frame projects to about 630 `E2` tasks, but this
  is not a guarantee.
- Utility remains Stage C only and blocked. Selector rules are unchanged.
- The future confirmatory window `2026-09-01..2026-11-30` remains untouched.
- Current combined operations state: 60/60 metadata shards, 3,847 candidates,
  825 E2 attempts, 167 valid E2, and an actual valid-E2 gap of 375.
- Remaining candidate execution order is frozen by
  `134_freeze_temporal_code_retrospective_execution_order.py`; further batches
  must follow this outcome-independent hash order.
- `135_build_temporal_code_retrospective_e2_capacity_audit.py` currently says
  `retrospective_strict_e2_execution_should_continue`: observed valid-E2 rate
  is 20.24%, point-estimate full-frame yield is about 779, and the one-sided
  95% Wilson upper-bound projection is 851.
- The research direction has shifted: continue strict E2 only as a secondary
  executable-evidence track. The primary next experiment is the code-domain
  raw-vs-curated equal-budget training validation in
  `docs/code_domain_training_validation_protocol.md`.
- `126_freeze_temporal_code_forward_recipe_batch.py` now records
  tree-metadata 404/unavailable rows as `metadata_unavailable_repositories`
  and skips them before execution. This is a recipe-freeze metadata availability
  guard, not a task-validity weakening or E2 failure relabeling.
- Windows storage was moved off C: `Phase-1/outputs`,
  `Phase-1/tiny_textbooks_raw`, and `Phase-1/validation/fixtures` are junctions
  to `D:\UNLV-Research\Phase-1\...`. Docker Desktop's WSL disk directory
  `C:\Users\ksl11\AppData\Local\Docker\wsl\disk` is also a junction to
  `D:\DockerDesktop\wsl\disk`; Docker was verified after the move with
  server `linux 29.5.2`.
- `temporal_code_retrospective_recipe_batch_014.json` was rerun from the
  already frozen recipe and completed: 25 attempts, 10 valid E2.
- `temporal_code_retrospective_recipe_batch_015.json` completed after adding
  verifier checkpoint/resume support: 25 attempts, 9 valid E2.
- `temporal_code_retrospective_recipe_batch_016.json` completed: 25 attempts,
  5 valid E2.
- `temporal_code_retrospective_recipe_batch_017.json` completed: 25 attempts,
  5 valid E2.
- `temporal_code_retrospective_recipe_batch_018.json` completed: 25 attempts,
  11 valid E2.
- `temporal_code_retrospective_recipe_batch_019.json` completed: 25 attempts,
  5 valid E2.
- `temporal_code_retrospective_recipe_batch_020.json` completed: 25 attempts,
  6 valid E2.
- `temporal_code_retrospective_recipe_batch_021.json` completed: 25 attempts,
  5 valid E2.
- `temporal_code_retrospective_recipe_batch_022.json` completed: 25 attempts,
  7 valid E2.
- `temporal_code_retrospective_recipe_batch_023.json` completed: 25 attempts,
  5 valid E2.
- `temporal_code_retrospective_recipe_batch_024.json` completed: 25 attempts,
  6 valid E2.
- `temporal_code_retrospective_recipe_batch_025.json` completed: 25 attempts,
  4 valid E2.
- `temporal_code_retrospective_recipe_batch_026.json` completed after verifier
  checkpoint/resume recovered from an interrupted Docker run: 25 attempts, 6
  valid E2.
- `temporal_code_retrospective_recipe_batch_027.json` completed: 25 attempts,
  7 valid E2.
- `temporal_code_retrospective_recipe_batch_028.json` completed: 25 attempts,
  4 valid E2.
- `temporal_code_retrospective_recipe_batch_029.json` completed: 25 attempts,
  2 valid E2.
- `temporal_code_retrospective_recipe_batch_030.json` completed after the
  recipe freezer skipped 2 tree-metadata-unavailable candidates before
  execution: 25 attempts, 4 valid E2.
- `temporal_code_retrospective_recipe_batch_031.json` completed: 25 attempts,
  4 valid E2.
- `temporal_code_retrospective_recipe_batch_032.json` completed: 25 attempts,
  6 valid E2.

Next work:

1. Run the frozen Stage-C guardrails needed by
   `configs/code_domain_development_plan_qwen3_4b_v1.json`: EvalPlus
   HumanEval+/MBPP+ development split and general retention evidence. The
   current development decision remains abstained until these pass.
2. If guardrails pass, freeze the confirmatory protocol before reading any
   confirmatory outcomes. Do not change seeds, heldout slice, optimizer steps,
   token budgets, or margins after the development outcomes below.
3. Keep strict E2 batches as optional secondary executable evidence; do not
   weaken E2 validity if they are resumed.
4. Keep Utility and benchmark outcomes in Stage C only; never use them as a
   selector objective.

## Code-Domain Development QLoRA Result

Completed on Windows with conda env `research` and CUDA visible device 1
(`NVIDIA GeForce RTX 3070 Ti`):

- 20/20 frozen development QLoRA runs completed:
  raw-random, Stage-A-random, curated, and known-high-quality arms, each with
  seeds `11,23,37,53,71`.
- 21/21 frozen heldout NLL evaluations completed: base-no-update plus all 20
  adapters.
- Decision report:
  `outputs/validation/code_domain_development_decision_report.json`.
- Report script:
  `142_build_code_domain_development_decision_report.py`.
- Current decision status:
  `development_decision_abstain_missing_required_guardrails`.
- NLL gate status: `passed`.
- Base no-update mean NLL: `1.178296703`.
- Raw-random mean NLL: `1.157646260`.
- Stage-A-random mean NLL: `1.164799889`.
- Curated mean NLL: `1.153644232`.
- Known-high-quality mean NLL: `1.154043969`.
- Curated vs Stage-A-random mean NLL reduction: `0.011155657`, above the
  frozen required margin `0.005`.
- Curated vs raw-random mean NLL reduction: `0.004002028`, directionally
  better than raw-random.
- Claim boundary: this is a development heldout NLL result only. Promotion is
  still blocked/abstained until the frozen code and general-retention
  guardrails are present and passing.

## Code-Domain Stage-C Guardrail Progress

Completed on Windows with conda env `research` and CUDA visible device 1
(`NVIDIA GeForce RTX 3070 Ti`):

- General-text NLL retention guardrail is complete and passed:
  `outputs/validation/code_domain_general_text_guardrail_report.json`.
- Frozen general-text holdout: existing Wikitext-103 validation/test guardrail
  from
  `outputs/slm_update_experiments/fineweb_edu_canonical_slm_update_v1/external_guardrails/wikitext103_validation_test_guardrail.jsonl`.
- Packed Qwen3 token blocks:
  `outputs/code_domain_development_qwen3_4b_v1/general_text_guardrail/token_blocks/wikitext103_qwen3_blocks.pt`.
- Block SHA-256:
  `54a60115e1c409febdd4882a13c841fdaeb5ad9a64466fdee559798d2b5918c0`.
- Base no-update general-text mean NLL: `2.176154667`.
- Raw-random mean NLL: `2.171828114`; one-sided 95% upper NLL increase
  versus base: `-0.004052799`; passed.
- Stage-A-random mean NLL: `2.172770015`; one-sided 95% upper NLL increase
  versus base: `-0.003161307`; passed.
- Curated mean NLL: `2.169513446`; one-sided 95% upper NLL increase
  versus base: `-0.006321777`; passed.
- Known-high-quality mean NLL: `2.169039960`; one-sided 95% upper NLL
  increase versus base: `-0.006856794`; passed.
- `142_build_code_domain_development_decision_report.py` now reads the
  EvalPlus and general-text Stage-C guardrail reports instead of treating all
  guardrail evidence as absent.
- Current decision report remains
  `development_decision_abstain_missing_required_guardrails` because:
  EvalPlus development guardrail is still incomplete, and general-task
  retention suites (`HellaSwag`, `ARC-Challenge`, `PIQA`, `WinoGrande`) are
  not yet implemented/run.
- EvalPlus guardrail infrastructure exists and smoke evaluation succeeded via
  Docker with network disabled:
  `143_generate_code_domain_evalplus_samples.py`,
  `144_run_code_domain_evalplus_guardrail.py`,
  `145_build_code_domain_evalplus_guardrail_report.py`, and
  `validation/docker/evalplus/evaluate_samples.py`.
- Current EvalPlus report:
  `outputs/validation/code_domain_evalplus_guardrail_report.json`, status
  `evalplus_development_guardrail_incomplete`.
- EvalPlus resume/aggregation hardening was added:
  `144_run_code_domain_evalplus_guardrail.py` now only skips an existing
  result when its `task_count` matches the current sample JSONL line count,
  and `145_build_code_domain_evalplus_guardrail_report.py` now only accepts
  full frozen development split counts (`HumanEval+` 90, `MBPP+` 194).
- Full EvalPlus base-no-update generation and Docker evaluation completed:
  HumanEval+ pass rate `0.233333333`, MBPP+ pass rate `0.603092784`, macro
  pass rate `0.418213058`.
- Full EvalPlus `raw_random_equal_budget` generation and Docker evaluation
  completed for seeds `11` and `23`:
  seed 11 HumanEval+ `0.333333333`, MBPP+ `0.592783505`;
  seed 23 HumanEval+ `0.288888889`, MBPP+ `0.603092784`.
- Full EvalPlus development guardrail is now complete and passed:
  `outputs/validation/code_domain_evalplus_guardrail_report.json`, status
  `evalplus_development_guardrail_passed`, blockers `[]`.
- Full EvalPlus sample/result coverage: 42 sample files and 42 result files,
  covering base plus four trained arms across five seeds, each on HumanEval+
  development split 90 tasks and MBPP+ development split 194 tasks.
- EvalPlus macro pass rates:
  base-no-update `0.418213058`,
  raw-random `0.464444444`,
  Stage-A-random `0.444879725`,
  curated `0.476391753`,
  known-high-quality `0.484604811`.
- EvalPlus suite pass rates:
  base HumanEval+ `0.233333333`, MBPP+ `0.603092784`;
  raw-random HumanEval+ `0.328888889`, MBPP+ `0.600000000`;
  Stage-A-random HumanEval+ `0.286666667`, MBPP+ `0.603092784`;
  curated HumanEval+ `0.360000000`, MBPP+ `0.592783505`;
  known-high-quality HumanEval+ `0.373333333`, MBPP+ `0.595876289`.
- EvalPlus guardrail comparisons versus base all passed the frozen maximum
  allowed absolute regression `0.02`. Curated macro regression versus base is
  `-0.058178694` (negative means improvement); suite regressions are
  HumanEval+ `-0.126666667` and MBPP+ `0.010309278`.
- Development decision report was rebuilt after EvalPlus completion:
  `outputs/validation/code_domain_development_decision_report.json`.
  Current status remains
  `development_decision_abstain_missing_required_guardrails` only because the
  general-task retention suites (`HellaSwag`, `ARC-Challenge`, `PIQA`,
  `WinoGrande`) are still missing/not yet implemented. NLL gate, general-text
  retention, and EvalPlus are all passing.

## Code-Domain General-Task Retention Completion

Completed on Windows with conda env `research`, using both local GPUs:

- General-task lm-eval retention suite is complete and passed:
  `outputs/validation/code_domain_general_task_guardrail_report.json`, status
  `general_task_guardrail_passed`, blockers `[]`.
- Full result coverage: 21 full lm-eval result files under
  `outputs/code_domain_development_qwen3_4b_v1/general_task_guardrail/lm_eval`,
  covering base-no-update plus four trained arms across seeds
  `11,23,37,53,71`.
- Suites and primary metric policy:
  HellaSwag `acc_norm`, ARC-Challenge `acc_norm`, PIQA `acc`, WinoGrande
  `acc`. Diagnostic raw/normalized accuracy values are retained in the report
  where available.
- Frozen guardrail thresholds from
  `outputs/temporal_code_collection/temporal_code_retention_guardrail_plan.json`:
  max absolute regression per suite `0.02`, max macro regression `0.01`.
- Base no-update macro score: `0.672744357`.
- Macro scores versus base:
  raw-random `0.673145558`, regression `-0.000401201`, passed;
  Stage-A-random `0.674481783`, regression `-0.001737426`, passed;
  curated `0.674622247`, regression `-0.001877890`, passed;
  known-high-quality `0.674812682`, regression `-0.002068325`, passed.
- Curated suite scores/regressions:
  HellaSwag `0.719856602`, regression `0.002409879`, passed;
  ARC-Challenge `0.519283276`, regression `-0.004778157`, passed;
  PIQA `0.772687704`, regression `-0.001196953`, passed;
  WinoGrande `0.686661405`, regression `-0.003946330`, passed.
- `148_run_code_domain_general_task_guardrail.py` was hardened so CUDA cleanup
  runs in `finally`; this prevents failed/aborted lm-eval calls from leaving
  GPU memory unreleased.
- `149_build_code_domain_general_task_guardrail_report.py` was added.
- `142_build_code_domain_development_decision_report.py` now reads the
  general-task guardrail report.
- Development decision report was rebuilt:
  `outputs/validation/code_domain_development_decision_report.json`, status
  `development_decision_promote_to_confirmatory`.
- Current Stage-C development evidence status:
  heldout code NLL passed, EvalPlus development guardrail passed,
  general-text NLL retention passed, and general-task retention passed.

Next immediate work:

1. Freeze the confirmatory protocol snapshot from the now-passing development
   recipe without changing margins, seeds, guardrails, or selector objectives.
2. Run untouched confirmatory training/evaluation only after the frozen
   confirmatory contract is written.
3. Keep Utility and all benchmark/retention outcomes strictly in Stage C; do
   not feed them into Stage B selector objectives.

## Code-Domain Confirmatory Protocol Freeze

Completed on Windows with conda env `research`:

- Confirmatory protocol freeze script:
  `150_freeze_code_domain_confirmatory_protocol.py`.
- Frozen protocol:
  `configs/code_domain_confirmatory_protocol_qwen3_4b_v1.json`, status
  `frozen_before_confirmatory_training_outcomes`.
- Freeze report:
  `outputs/validation/code_domain_confirmatory_protocol_qwen3_4b_report.json`,
  status `confirmatory_protocol_frozen`, blockers `[]`.
- Contract test:
  `validation/test_code_domain_confirmatory_protocol.py`.
- Test result:
  `[code-domain-confirmatory] frozen protocol preserves outcome isolation and Stage-C-only Utility: pass`.
- Confirmatory training seeds are now bound to the already-frozen retention and
  EvalPlus contracts: `101,131,163,197,239`.
- Confirmatory heldout NLL slice is frozen at:
  `outputs/code_domain_confirmatory_qwen3_4b_v1/heldouts/confirmatory_code_nll_heldout.jsonl`.
- Confirmatory heldout summary:
  110 selected records, token-proxy count `61618`, selected from the
  confirmatory Stage-A pass pool using deterministic seed `20260621`.
- Training arms remain unchanged:
  base-no-update, raw-random, Stage-A-random, curated, known-high-quality.
- The primary confirmatory rule is now frozen before confirmatory model
  outcomes:
  curated must beat Stage-A-random by the predeclared absolute NLL margin
  `0.005`, curated must beat raw-random directionally, and all Stage-C
  confirmatory retention guardrails must pass.
- Utility, EvalPlus, general-task, and general-text outcomes remain Stage C
  evidence only and are forbidden as selector objectives.
- `confirmatory_outcomes_read` remains `false`.

Next immediate work:

1. Add/adapt confirmatory execution scripts so they read
   `configs/code_domain_confirmatory_protocol_qwen3_4b_v1.json` and write to
   `outputs/code_domain_confirmatory_qwen3_4b_v1`.
2. Run untouched confirmatory QLoRA training for all frozen arms and seeds.
3. Run confirmatory heldout NLL, EvalPlus confirmatory split, general-text NLL,
   and general-task retention.
4. Build the final confirmatory decision report. Do not alter the protocol
   after any confirmatory outcome is produced.

## Code-Domain Confirmatory Execution Start

Started on Windows with conda env `research`, using CUDA device `1`
(`NVIDIA GeForce RTX 3070 Ti`) for the first confirmatory execution probe:

- `141_run_code_domain_development_qlora.py` was extended to support both the
  original development plan schema and the frozen confirmatory protocol schema.
  It now reads `confirmatory_training_recipe`,
  `confirmatory_training_seeds`, and the frozen confirmatory heldout path when
  `configs/code_domain_confirmatory_protocol_qwen3_4b_v1.json` is supplied.
- New confirmatory decision builder:
  `151_build_code_domain_confirmatory_decision_report.py`.
- New contract test:
  `validation/test_code_domain_confirmatory_runner_contract.py`.
- Verification completed:
  - `python -m py_compile 141_run_code_domain_development_qlora.py 151_build_code_domain_confirmatory_decision_report.py validation\test_code_domain_confirmatory_runner_contract.py`
  - `python validation\test_code_domain_confirmatory_protocol.py`
  - `python validation\test_code_domain_confirmatory_runner_contract.py`
- Frozen confirmatory heldout eval blocks were prepared:
  `outputs/code_domain_confirmatory_qwen3_4b_v1/eval_blocks/confirmatory_code_nll_heldout.pt`.
  Manifest status is `confirmatory_eval_blocks_frozen`, with 34 blocks,
  sequence length 2048, and 69,632 packed tokens.
- First confirmatory QLoRA run completed:
  `raw_random_equal_budget`, seed `101`, optimizer steps `8`, status
  `confirmatory_qlora_completed`.
- First confirmatory heldout NLL evaluations completed:
  base-no-update mean NLL `1.0118654002161587`;
  raw-random seed `101` mean NLL `0.9990089903859531`.
- Confirmatory decision report exists at:
  `outputs/validation/code_domain_confirmatory_decision_report.json`.
  Current status is still `confirmatory_decision_incomplete`, with
  training coverage `1/20` and heldout NLL coverage `2/21`.

Important:

- Confirmatory outcomes have now been read only by the confirmatory execution
  and decision-report path. Do not change the frozen protocol, seed set,
  margins, token budgets, heldout slice, EvalPlus split, guardrail thresholds,
  or selector objective from this point forward.
- Utility remains Stage C validation only and must not be added to Stage B.

Next immediate work:

1. Continue confirmatory QLoRA training on CUDA device `1`:
   `train-missing --plan configs\code_domain_confirmatory_protocol_qwen3_4b_v1.json --output-dir outputs\code_domain_confirmatory_qwen3_4b_v1 --blocks-dir outputs\code_domain_qlora_smoke_qwen3_4b_v1\token_blocks`.
2. After each completed training batch, run heldout NLL with:
   `eval-missing --plan configs\code_domain_confirmatory_protocol_qwen3_4b_v1.json --output-dir outputs\code_domain_confirmatory_qwen3_4b_v1`.
3. Rebuild `outputs/validation/code_domain_confirmatory_decision_report.json`
   with `151_build_code_domain_confirmatory_decision_report.py`.
4. After NLL coverage is complete, adapt/run the confirmatory EvalPlus,
   general-text NLL, and general-task Stage-C guardrails using the frozen
   confirmatory split and thresholds.

## Code-Domain Confirmatory Heldout NLL Completion

Completed on Windows with conda env `research`, using CUDA device `1`
(`NVIDIA GeForce RTX 3070 Ti`) for QLoRA training and CUDA device `0`
(`NVIDIA GeForce RTX 4060 Ti`) for parallel heldout NLL evaluation:

- Confirmatory QLoRA training coverage is complete:
  20/20 trained-arm runs.
- Confirmatory heldout NLL coverage is complete:
  21/21 evaluations, including base-no-update.
- Updated decision report:
  `outputs/validation/code_domain_confirmatory_decision_report.json`.
- Current report status:
  `confirmatory_decision_reject_primary_margin_failure`.
- NLL gate status:
  `failed`.

Confirmatory heldout NLL means:

- base-no-update: `1.0118654002161587`.
- raw-random equal budget: `0.9987000654725466`.
- Stage-A-random equal budget: `1.0012560558669708`.
- curated equal budget: `0.997489395737648`.
- known-high-quality equal budget: `0.997341270481839`.

Frozen NLL gate interpretation:

- Curated beats Stage-A-random directionally on every frozen seed.
- Curated vs Stage-A-random mean NLL reduction:
  `0.0037666601293226964`.
- Frozen required reduction:
  `0.005`.
- Therefore the predeclared primary confirmatory margin fails.
- Curated also beats raw-random directionally:
  raw-random minus curated mean NLL reduction
  `0.001210669734898584`.
- Known-high-quality is still slightly better than curated on mean NLL:
  known-high-quality minus curated `-0.00014812525580909508`
  (negative means known-high-quality lower NLL).

Important interpretation:

- This is not an infrastructure failure; the confirmatory NLL experiment
  completed.
- The result is a positive directional curation signal but a negative
  confirmatory primary-margin result under the frozen protocol.
- Do not change the frozen margin, seeds, heldout, token budget, split, or
  selector objective in response to this outcome.
- Stage-C confirmatory guardrails are still missing:
  EvalPlus confirmatory, general-text NLL retention, and general-task
  retention. However, the current decision report already rejects on the
  primary NLL margin before those guardrails can rescue the claim.

Next immediate work:

1. Decide whether to run the remaining Stage-C confirmatory guardrails anyway
   for diagnostic completeness, clearly labeled as post-primary-failure
   evidence.
2. Analyze why development passed but confirmatory missed the frozen margin:
   heldout split shift, margin calibration, effect-size variance, Stage-A
   baseline behavior, and token-budget/training-step sensitivity.
3. Preserve the negative confirmatory result as scientifically valid evidence;
   any new recipe or margin must be treated as a new development cycle with a
   newly frozen untouched confirmatory protocol.

## Code-Domain Confirmatory Postmortem

Completed after the frozen confirmatory NLL failure:

- Postmortem script:
  `152_build_code_domain_confirmatory_postmortem.py`.
- JSON report:
  `outputs/validation/code_domain_confirmatory_postmortem_report.json`,
  status `confirmatory_postmortem_completed`.
- Markdown summary:
  `docs/code_domain_confirmatory_postmortem.md`.
- Contract test:
  `validation/test_code_domain_confirmatory_postmortem.py`.
- Test result:
  `[code-domain-confirmatory-postmortem] negative result and next-cycle separation: pass`.

Locked interpretation:

- The completed frozen confirmatory protocol remains a negative primary-margin
  result:
  `confirmatory_decision_reject_primary_margin_failure`.
- It is not an infrastructure failure: training coverage was 20/20 and
  heldout NLL coverage was 21/21.
- Directional curation signal replicated:
  curated beat Stage-A-random and raw-random on every frozen confirmatory seed.
- Primary practical margin did not replicate:
  frozen margin `0.005`, observed reduction `0.0037666601293226964`,
  gap to margin `0.0012333398706773037`.
- Descriptive 95% paired-seed CI for Stage-A-random minus curated is
  `[0.003470092276943865, 0.004063227981701572]`, below the frozen margin.

Development-to-confirmatory shift:

- Development primary reduction:
  `0.011155656973521166`.
- Confirmatory primary reduction:
  `0.0037666601293226964`.
- Confirmatory retained about `0.33764574675101267` of the development effect.
- Development base NLL:
  `1.1782967034313414`.
- Confirmatory base NLL:
  `1.0118654002161587`.

Heldout shift diagnosis:

- Development heldout: 175 records, 7 repositories, code/test ratio
  125/50, test ratio `0.2857142857142857`.
- Confirmatory heldout: 110 records, 5 repositories, code/test ratio
  64/46, test ratio `0.41818181818181815`.
- Repository Jaccard overlap between the two heldouts: `0.0`.
- Confirmatory mean token-proxy per record increased by
  `185.72935064935064`.

Next-cycle requirements:

- Do not change the completed confirmatory protocol, margin, seeds, heldout,
  token budget, split, or selector objective.
- Treat the current result as valid negative confirmatory evidence with a
  positive directional signal.
- Any improvement must start a new development cycle, followed by a newly
  frozen untouched confirmatory protocol.
- Recommended next-cycle fixes:
  stratified larger heldouts by repository and content type;
  development-only margin/power calibration;
  stronger Stage-B code-quality proxies without Utility/benchmark leakage;
  explicit Stage-A-random hardness diagnostics before confirmatory freeze.

## Code-Domain Next Development Cycle Design

Designed the next code-domain development cycle as a new v2 cycle, not as a
revision of the completed v1 confirmatory result.

New artifacts:

- Human-readable design:
  `docs/code_domain_next_development_cycle_design.md`.
- Machine-readable draft contract:
  `configs/code_domain_next_development_cycle_v2_design.json`, status
  `design_draft_not_executable_protocol`.
- Contract test:
  `validation/test_code_domain_next_cycle_design.py`.

Core design decisions:

- v1 remains locked as
  `confirmatory_decision_reject_primary_margin_failure`.
- v2 must rebuild or expand the raw-like Python candidate pool until split
  and stratification requirements can be checked.
- Minimum Stage-A-pass repository targets are train `>=30`, development
  heldout `>=10`, and confirmatory heldout `>=10`.
- No repository should contribute more than 25% of selected training or
  heldout tokens.
- Development and confirmatory heldouts must be stratified by content type;
  maximum allowed absolute test-ratio difference before freeze is `0.05`.
- Stage B v2 may strengthen code-local proxies such as AST granularity,
  test/code balance, API/import meaningfulness, generated/template risk,
  useful recurrence, soft redundancy, cluster coverage, and repository/path
  diversity caps.
- Utility, benchmarks, retention, development model outcomes, confirmatory
  model outcomes, human labels, and LLM labels remain forbidden Stage-B
  signals.
- Margin calibration is development-only for the new cycle. Exactly one
  primary v2 confirmatory success rule must be frozen before any v2
  confirmatory outcomes are read.
- Confirmatory freeze must include the common disjoint Stage-A baseline design
  for all Utility sensitivity arms.

Next concrete work:

1. Validate the new design contract.
2. Build the v2 candidate-pool readiness script/report against the current
   raw-like Python corpus.
3. If the current corpus fails the new repository and stratification
   thresholds, expand collection before running more GPU training.

## Code-Domain v2 Candidate-Pool Readiness

Added the v2 candidate-pool readiness builder:

- Script:
  `153_build_code_domain_v2_candidate_pool_readiness.py`.
- Report:
  `outputs/validation/code_domain_v2_candidate_pool_readiness_report.json`.
- Markdown summary:
  `docs/code_domain_v2_candidate_pool_readiness.md`.
- Contract test:
  `validation/test_code_domain_v2_candidate_pool_readiness.py`.

The script reads the current Stage-A pass pool, by default:

```text
outputs/temporal_code_collection/stage_a_path_stratified_tranche/
```

It checks the v2 design requirements from
`configs/code_domain_next_development_cycle_v2_design.json`:

- repository-disjoint train/development/confirmatory splits
- minimum Stage-A-pass repository counts
- maximum token share per repository
- development and confirmatory heldout token budgets
- development/confirmatory test-ratio difference
- required reporting of content-type, chunk-kind, token, repository, and
  top-repository-share profiles

The report is candidate-pool readiness only. It does not run Stage B, Stage C,
Utility, training, or confirmatory evaluation, and it keeps Utility and model
outcomes forbidden for Stage-B selection.

Current readiness result:

```text
status: candidate_pool_not_ready_for_v2_development_design
```

Split summary from the current path-stratified Stage-A pool:

```text
train:        523 chunks, 300598 token proxy, 15 repos, largest repo share 0.520203, test ratio 0.470363
development: 235 chunks,  93841 token proxy,  7 repos, largest repo share 0.259695, test ratio 0.272340
confirmatory:110 chunks,  61618 token proxy,  5 repos, largest repo share 0.780486, test ratio 0.418182
```

Blockers:

- train repo count is 15, below required 30.
- train largest repo token share is 0.520203, above cap 0.25.
- development repo count is 7, below required 10.
- development largest repo token share is 0.259695, slightly above cap 0.25.
- confirmatory repo count is 5, below required 10.
- confirmatory largest repo token share is 0.780486, above cap 0.25.
- confirmatory heldout token proxy is 61618, below required 65536.
- development/confirmatory test-ratio difference is 0.145842, above limit
  0.05.

Warning:

- development heldout token proxy is 93841, above minimum but below preferred
  131072.

Interpretation:

- The current Stage-A candidate pool is useful diagnostic evidence, but it is
  not shaped well enough for the v2 development cycle.
- Do not spend more GPU training on this pool as a v2 confirmatory candidate.
- Expand or rebuild the raw-like Python corpus, then rerun Stage 0, Stage A,
  and `153_build_code_domain_v2_candidate_pool_readiness.py`.

## Code-Domain v2 Expansion Collection Completion

Completed a v2 expansion collection pass after the first readiness failure.

New artifacts:

- Expansion tranche config:
  `configs/code_domain_v2_expansion_tranche_v1.json`.
- Expansion tranche freezer:
  `154_freeze_code_domain_v2_expansion_tranche.py`.
- Stage0 merge script:
  `155_merge_code_domain_v2_stage0_pools.py`.
- Balanced Stage-A pool builder:
  `156_build_code_domain_v2_balanced_stage_a_pool.py`.
- Expansion contract test:
  `validation/test_code_domain_v2_expansion_collection.py`.
- Balanced-pool contract test:
  `validation/test_code_domain_v2_balanced_pool.py`.

Expansion plan:

```text
outputs/temporal_code_collection/code_domain_v2_expansion_tranche_plan.json
status: frozen_before_expansion_content_fetch
repositories: 68
train: 20 code_and_test + 20 code_only
development: 8 code_and_test + 6 code_only
confirmatory: 12 code_and_test + 2 code_only
```

Content fetch:

```text
output: outputs/temporal_code_collection/code_domain_v2_expansion_bundles/
bundle_count: 68
file_record_count: 377
github_api_requests: 768
```

Bundle audit:

```text
stage0_release_candidate_count: 52 / 68
collection_gate_pass_count: 52
benchmark_quarantine_match_count: 11
pii_quarantined_file_count: 25
secret_quarantined_file_count: 11
generated_file_count: 5
```

Expansion Stage0:

```text
input_records: 209
release_candidate_records: 206
train release candidates: 128
development release candidates: 39
confirmatory release candidates: 39
```

Combined Stage0:

```text
output: outputs/temporal_code_collection/stage0_code_domain_v2_combined/
train release candidates: 188
development release candidates: 76
confirmatory release candidates: 48
duplicate_record_id_count: 0
```

Combined Stage A:

```text
output: outputs/temporal_code_collection/stage_a_code_domain_v2_combined/
stage0_input_records: 312
chunk_count: 3456
stage_a_pass_count: 3313
train pass chunks: 1913
development pass chunks: 602
confirmatory pass chunks: 798
```

The combined Stage-A pool still had two corpus-shape blockers:

```text
confirmatory largest repo share: 0.323902 > 0.25
development/confirmatory test-ratio difference: 0.226701 > 0.05
```

Therefore `156_build_code_domain_v2_balanced_stage_a_pool.py` built a
deterministic non-Utility balanced Stage-A pool using only Stage-A pass status,
repository identity, content type, token proxy count, and chunk UID hashes.
It does not use Utility, benchmark, retention, model outcomes, or review
labels.

Balanced Stage-A readiness now passes:

```text
source: outputs/temporal_code_collection/stage_a_code_domain_v2_balanced/
report: outputs/validation/code_domain_v2_candidate_pool_readiness_report.json
status: candidate_pool_ready_for_v2_development_design
blockers: []

train:        1913 chunks, 818060 token proxy, 47 repos, largest repo share 0.191150, test ratio 0.436487
development:  390 chunks, 140513 token proxy, 17 repos, largest repo share 0.180916, test ratio 0.300000
confirmatory: 659 chunks, 242784 token proxy, 14 repos, largest repo share 0.214001, test ratio 0.300455
```

Verification:

```text
[code-domain-v2-expansion] expansion collection contract: pass
[code-domain-v2-candidate-pool-readiness] corpus-shape contract: pass
[code-domain-v2-balanced-pool] balanced candidate pool readiness: pass
```

Next immediate work:

1. Freeze Stage-B v2 arms from
   `outputs/temporal_code_collection/stage_a_code_domain_v2_balanced/`.
2. Include required ablations from
   `configs/code_domain_next_development_cycle_v2_design.json`:
   full selector, quality-only, redundancy-only, no-coverage-support,
   no-test-code-balance, no-repository-diversity-cap, Stage-A-random, and
   raw-random.
3. Do not run more GPU training until Stage-B v2 arms and development heldouts
   are frozen.

## Code-Domain v2 Stage-B Arms Freeze

Completed the Stage-B v2 arm freeze before any v2 Stage-C training.

New artifacts:

- Freeze script:
  `157_freeze_code_domain_v2_stage_b_arms.py`.
- Contract test:
  `validation/test_code_domain_v2_stage_b_arms.py`.
- Output directory:
  `outputs/temporal_code_collection/stage_b_code_domain_v2/`.
- Report:
  `outputs/temporal_code_collection/stage_b_code_domain_v2/stage_b_v2_arms_report.json`,
  status `stage_b_v2_arms_frozen_before_stage_c`.

Primary frozen arms:

```text
curated_v2_equal_budget:
  records: 1424
  token_proxy_count: 327222
  repositories: 47
  content types: code 794, documentation 20, test 610

stageA_random_equal_budget:
  records: 300
  token_proxy_count: 327222
  repositories: 41
  content types: code 134, documentation 22, test 144

raw_random_equal_budget:
  records: 861
  token_proxy_count: 327334
  repositories: 47
  content types: code 471, documentation 28, test 362
```

Curated and Stage-A-random are selected-disjoint:

```text
curated_v2_stageA_random_disjoint: true
intersection_count: 0
```

Required ablations are frozen:

```text
full_selector:
  selected_chunks: 1424
  selected_token_proxy: 327222
  mean_code_quality_proxy: 0.887792
  mean_soft_redundancy_risk: 0.227030

quality_only:
  selected_chunks: 1533
  selected_token_proxy: 327222
  mean_code_quality_proxy: 0.888524
  mean_soft_redundancy_risk: 0.288692

redundancy_only:
  selected_chunks: 514
  selected_token_proxy: 327224
  mean_code_quality_proxy: 0.809545
  mean_soft_redundancy_risk: 0.031706

no_coverage_support:
  selected_chunks: 1512
  selected_token_proxy: 327223
  mean_code_quality_proxy: 0.886488
  mean_soft_redundancy_risk: 0.250286

no_test_code_balance:
  selected_chunks: 1424
  selected_token_proxy: 327222
  mean_code_quality_proxy: 0.887792
  mean_soft_redundancy_risk: 0.227030

no_repository_diversity_cap:
  selected_chunks: 1514
  selected_token_proxy: 327223
  mean_code_quality_proxy: 0.885915
  mean_soft_redundancy_risk: 0.249473
```

Verification:

```text
[code-domain-v2-stage-b] frozen arms and ablations: pass
```

Important interpretation:

- This is Stage-B arm construction only.
- No Utility, benchmark, retention, development model outcome, confirmatory
  outcome, human label, or LLM label was used in selection.
- The v2 confirmatory outcome remains unread.
- The next step is to freeze v2 development heldouts and a development training
  plan before any GPU training.

## Code-Domain v2 Development Plan Freeze

Completed the v2 development training-plan freeze after Stage-B arms and before
any v2 Stage-C GPU training.

New artifacts:

- Development-plan freeze script:
  `158_freeze_code_domain_v2_development_plan.py`.
- Contract test:
  `validation/test_code_domain_v2_development_plan.py`.
- Frozen development plan:
  `configs/code_domain_v2_development_plan_qwen3_4b.json`.
- Freeze report:
  `outputs/validation/code_domain_v2_development_plan_qwen3_4b_report.json`,
  status `v2_development_plan_frozen`.
- Token-block output directory:
  `outputs/code_domain_v2_development_qwen3_4b/token_blocks/`.
- Development NLL heldout:
  `outputs/code_domain_v2_development_qwen3_4b/heldouts/development_code_nll_heldout.jsonl`.
- Development NLL eval blocks:
  `outputs/code_domain_v2_development_qwen3_4b/eval_blocks/development_code_nll_heldout.pt`.

Frozen training arms:

```text
base_no_update
raw_random_equal_budget
stageA_random_equal_budget
curated_v2_equal_budget
known_high_quality_equal_budget
```

Frozen comparison structure:

```text
treatment: curated_v2_equal_budget
primary baseline: stageA_random_equal_budget
supporting baselines: raw_random_equal_budget, base_no_update
reference arm: known_high_quality_equal_budget
```

Frozen development run settings:

```text
model: Qwen/Qwen3-4B
development seeds: 11, 23, 37, 53, 71
optimizer_steps: 20
gradient_accumulation_steps: 8
training_token_budget_cap: 327222
common_packed_token_budget: 325632
sequence_length: 2048
blocks per trained arm: 159
```

Frozen token blocks:

```text
raw_random_equal_budget.pt:
  sha256: a7c4b083f91df4ccfe5e49308250bd71c87f4de621c121fb8e95aa40e6e74ec9

stageA_random_equal_budget.pt:
  sha256: 575b68339d4c04535dd8d275711d24381e01039703a89a1a9dbc36dd91a9996f

curated_v2_equal_budget.pt:
  sha256: 9dc825634e9621d0d4722cae417daced7474018d68c4e2357f7254a97c147930

known_high_quality_equal_budget.pt:
  sha256: f48eb9fe523a627729e48a87028670f43a173c11458a80d43df5805b5db5e011
```

Frozen development NLL heldout:

```text
source split: development
selected records: 342
selected token proxy: 126482
repository count: 17
content types: code 225, test 117
sha256: 95aa8973a26e469708faf5d1c053cdee023fef1c2c8891547cffe77ba475bdd8
```

Frozen development NLL eval blocks:

```text
blocks: 71
sequence_length: 2048
packed_tokens: 145408
consumed_tokens_before_packing: 147219
dropped_tail_tokens: 1811
sha256: 4e962d4602126c1c54b0746d34b53f7d7d0db1d259bb440ee55b4f6efcc27d57
confirmatory_outcomes_read: false
```

Verification:

```text
[code-domain-v2-stage-b] frozen arms and ablations: pass
[code-domain-v2-development-plan] frozen heldout, blocks, and training contract: pass
```

Important interpretation:

- This freezes Stage-C development inputs only.
- It does not read or use confirmatory outcomes.
- It does not change Stage-B objective with Utility, benchmark, retention,
  development model outcome, human label, or LLM label.
- The next step is to run v2 Stage-C development training/evaluation across
  the frozen development seeds, then inspect whether curated beats
  Stage-A-random directionally before deciding whether a confirmatory run is
  justified.

## Code-Domain v2 Stage-C Development Execution

Completed v2 Stage-C development QLoRA training and heldout NLL evaluation on
Windows with conda env `research`.

GPU execution note:

```text
Primary completion device: physical CUDA device 1, NVIDIA GeForce RTX 3070 Ti
RTX 4060 Ti was briefly used with explicit user approval, then released for
gaming. Remaining training/evaluation was completed on RTX 3070 Ti only.
```

Training/evaluation artifacts:

- QLoRA adapters:
  `outputs/code_domain_v2_development_qwen3_4b/qlora_runs/`
- Heldout NLL results:
  `outputs/code_domain_v2_development_qwen3_4b/heldout_nll/`
- Decision report:
  `outputs/validation/code_domain_v2_development_decision_report.json`
- Report status:
  `development_decision_promote_to_confirmatory`

Completed counts:

```text
training runs completed: 20 / 20
heldout NLL results completed: 21 / 21
confirmatory outcomes read: false
Utility scope: Stage C validation only; never selector objective
```

Development heldout NLL means:

```text
base_no_update: 1.2732299574663941

curated_v2_equal_budget:
  mean NLL: 1.2367234251868555
  sample std: 0.0009183310874709193

raw_random_equal_budget:
  mean NLL: 1.2378267587070735
  sample std: 0.00045848371377475364

stageA_random_equal_budget:
  mean NLL: 1.2438329681544236
  sample std: 0.0005125806968323734

known_high_quality_equal_budget:
  mean NLL: 1.2451677663225524
  sample std: 0.0005344613091267325
```

Primary development signal:

```text
Stage-A-random minus curated mean NLL delta: +0.00710954296756805
All five paired seed deltas are positive.
Per-seed deltas:
  seed 11: +0.007062
  seed 23: +0.006120
  seed 37: +0.008103
  seed 53: +0.007853
  seed 71: +0.006410
```

Supporting raw-random comparison:

```text
raw-random minus curated mean NLL delta: +0.001103333520217964
Curated beats raw-random on 4 / 5 paired seeds.
Seed 23 is the exception:
  seed 23 delta: -0.000683
```

Reference comparison:

```text
known-high-quality minus curated mean NLL delta: +0.00844434113569692
All five paired seed deltas are positive.
```

Guardrail evidence loaded by the decision report:

```text
EvalPlus development guardrail: passed
general-text NLL retention guardrail: passed
general-task retention guardrail: passed
```

Important interpretation:

- This is the first v2 development run where curated clearly beats the primary
  Stage-A-random baseline on all paired seeds.
- Curated also beats raw-random on the mean, but not on every seed.
- The decision report currently records promotion to confirmatory, but v2
  still needs a freshly frozen confirmatory protocol before any confirmatory
  outcomes are read.
- Because the v2 development plan used a development-calibration placeholder
  rather than a fixed practical margin, the next protocol must explicitly
  freeze the primary metric and confirmatory margin/rule before confirmatory
  training/evaluation.
- Do not treat this as a final paper claim or release claim. It is
  development-stage evidence that justifies freezing and running a new untouched
  confirmatory protocol.

## Code-Domain v2 Confirmatory Protocol Freeze

Completed the v2 confirmatory protocol freeze after the v2 development
decision and before any v2 confirmatory model outcomes were read.

New artifacts:

- Freeze script:
  `159_freeze_code_domain_v2_confirmatory_protocol.py`.
- Contract test:
  `validation/test_code_domain_v2_confirmatory_protocol.py`.
- Frozen protocol:
  `configs/code_domain_v2_confirmatory_protocol_qwen3_4b.json`.
- Freeze report:
  `outputs/validation/code_domain_v2_confirmatory_protocol_qwen3_4b_report.json`,
  status `v2_confirmatory_protocol_frozen`.
- Confirmatory heldout:
  `outputs/code_domain_v2_confirmatory_qwen3_4b/heldouts/confirmatory_code_nll_heldout.jsonl`.

Frozen confirmatory seeds:

```text
101, 131, 163, 197, 239
```

Frozen primary metric and rule:

```text
primary metric: confirmatory_code_nll_heldout mean NLL
treatment: curated_v2_equal_budget
primary baseline: stageA_random_equal_budget
required absolute NLL reduction: 0.003
paired seed rule: all curated_v2 seed-level NLLs must be lower than paired Stage-A-random NLLs
supporting raw-random rule: curated_v2 mean NLL must be lower than raw-random mean NLL
guardrail rule: all frozen confirmatory Stage-C guardrails must pass
```

Margin calibration:

```text
source: development-only primary paired seed deltas
development primary mean delta: 0.0071095429675679615
development primary sample std: 0.0008676009342821313
formula: ceil_to_0.0005(max(0.0025, 0.40 * development mean delta, 2.0 * development paired std))
frozen absolute NLL margin: 0.003
confirmatory outcomes used: false
```

Frozen confirmatory heldout:

```text
selected records: 377
selected token proxy: 131068
repository count: 13
source split: confirmatory
confirmatory outcomes read: false
```

Verification:

```text
[code-domain-v2-confirmatory] frozen protocol preserves margin, seeds, and outcome isolation: pass
```

Important interpretation:

- The primary metric, margin, seed set, heldout, and guardrail rules are now
  frozen before v2 confirmatory training/evaluation.
- The v2 confirmatory outcome remains unread.
- The next step is to prepare confirmatory eval blocks, run the frozen
  confirmatory training seeds, evaluate heldout NLL, and then build a
  confirmatory decision report without changing this protocol.

## Code-Domain v2 Confirmatory NLL Execution

Completed v2 confirmatory QLoRA training and frozen heldout NLL evaluation on
Windows with conda env `research`, using physical CUDA device 1
(`NVIDIA GeForce RTX 3070 Ti`).

Additional artifacts:

- Confirmatory eval blocks:
  `outputs/code_domain_v2_confirmatory_qwen3_4b/eval_blocks/confirmatory_code_nll_heldout.pt`.
- QLoRA adapters:
  `outputs/code_domain_v2_confirmatory_qwen3_4b/qlora_runs/`.
- Heldout NLL results:
  `outputs/code_domain_v2_confirmatory_qwen3_4b/heldout_nll/`.
- Decision builder:
  `160_build_code_domain_v2_confirmatory_decision_report.py`.
- Decision contract test:
  `validation/test_code_domain_v2_confirmatory_decision.py`.
- Decision report:
  `outputs/validation/code_domain_v2_confirmatory_decision_report.json`.

Confirmatory eval block freeze:

```text
blocks: 77
sequence_length: 2048
packed_tokens: 157696
consumed_tokens_before_packing: 158731
dropped_tail_tokens: 1035
sha256: f5a6c679ca35894066137520d692dc52b88000ae831307d497547a157049871b
confirmatory_outcomes_read during block freeze: false
```

Completed counts:

```text
training runs completed: 20 / 20
heldout NLL results completed: 21 / 21
```

Confirmatory heldout NLL means:

```text
base_no_update: 1.2341832828212094

curated_v2_equal_budget:
  mean NLL: 1.2016515642017513
  sample std: 0.0002306571606881428

raw_random_equal_budget:
  mean NLL: 1.2042412353026402
  sample std: 0.0007667165894477228

known_high_quality_equal_budget:
  mean NLL: 1.2051035645720247
  sample std: 0.0004520039534788804

stageA_random_equal_budget:
  mean NLL: 1.2087674524102892
  sample std: 0.0012339406107037335
```

Frozen primary NLL rule result:

```text
required absolute NLL margin: 0.003
Stage-A-random minus curated mean NLL delta: +0.007115888208537813
primary margin pass: true
all paired seed deltas positive: true
per-seed deltas:
  seed 101: +0.005757
  seed 131: +0.007772
  seed 163: +0.006446
  seed 197: +0.006362
  seed 239: +0.009242
```

Supporting comparisons:

```text
raw-random minus curated mean NLL delta: +0.002589671100888813
raw-random comparison: all five paired seed deltas positive

known-high-quality minus curated mean NLL delta: +0.003452000370273378
known-high-quality comparison: all five paired seed deltas positive
```

Decision report status:

```text
v2_confirmatory_decision_abstain_missing_required_guardrails
```

Interpretation:

- The frozen confirmatory NLL primary rule passed.
- Curated beats Stage-A-random by more than the frozen `0.003` margin and on
  every paired confirmatory seed.
- Curated also beats raw-random on mean NLL and on every paired confirmatory
  seed.
- This is not yet a final release/paper success claim because the frozen
  confirmatory EvalPlus and general-task retention guardrail reports are still
  incomplete. The general-text NLL retention guardrail is complete and passed.

Remaining required work before a final pass claim:

```text
1. Complete v2 confirmatory EvalPlus sample/result coverage and rebuild the
   EvalPlus guardrail report.
2. Complete v2 confirmatory general-task retention results and rebuild the
   general-task guardrail report.
3. Rebuild `outputs/validation/stage_c_guardrail_gap_report.json`.
4. Rebuild `outputs/validation/code_domain_v2_confirmatory_decision_report.json`.
```

Verification:

```text
[code-domain-v2-confirmatory-decision] frozen NLL decision contract: pass
```

Current Stage-C guardrail status:

```text
stage_c_guardrail_gap_report: stage_c_guardrail_gaps_open
incomplete guardrails: evalplus_confirmatory, general_task_retention
general_text_nll_retention: passed
general_task completed results:
  base_no_update_base_full.json
  raw_random_equal_budget_seed101_full.json
  raw_random_equal_budget_seed131_full.json
  raw_random_equal_budget_seed163_full.json
general_task missing trained-arm seed results: 17
EvalPlus execution blocker:
  Docker daemon is not running (`//./pipe/docker_engine` unavailable).
```

## 2026-06-26 Literature-Grounded Framework Reset

The framework direction was rechecked against public work from OpenAI, Google
DeepMind, Meta, Microsoft, Alibaba Qwen, DeepSeek, BigCode, Hugging Face, AI2,
and DataComp-LM.

Canonical decision:

- Keep the five Core axes: Validity, Selection Value Evidence, Redundancy,
  Coverage, and Utility.
- Do not claim intrinsic data-quality measurement.
- Treat exact/near deduplication, structural filtering, model-based selection,
  mixture control, and fixed-recipe downstream validation as literature-backed
  method families.
- Treat current SimHash/Jaccard/containment thresholds, AST risk `0.85`,
  Stage-B weights `0.8/0.2`, useful-recurrence formulas, style discounts, and
  fixed selection ratios as unvalidated frozen project hypotheses.
- Split Coverage into collapse-prevention retention support and
  Deployment-Contract target alignment.
- Keep Utility in Stage C only.
- Use DataComp-LM-style controlled training comparisons as the primary
  scientific validation pattern.

New canonical direction document:

```text
docs/literature_grounded_curation_direction.md
```

Revised order:

1. Freeze evidence classes and claim boundaries.
2. Repair and validate irreversible Stage-0/Stage-A behavior.
3. Calibrate Redundancy with labeled pair and saturation benchmarks.
4. Rebuild Stage B as preregistered, outcome-free policy arms.
5. Screen with fixed proxy-scale training.
6. Run equal-token/equal-compute Qwen3-4B development validation.
7. Close task, retention, forgetting, contamination, and stability guardrails.
8. Run untouched repository/time confirmation.
9. Freeze the paper claim.

Do not continue tuning Utility or the selector before the Stage-A and
Redundancy validity gaps are closed. Existing positive NLL results remain
evidence for the frozen code-domain experiment, but they do not validate every
Core metric or authorize a production-framework claim.

## 2026-06-26 Stage-A Representative Fix and Redundancy Benchmark

Completed:

- Stage-A duplicate representatives are selected only from chunks that pass
  local parseability, minimum-unit, and pathological-repetition gates.
- Representative choice is deterministic by `chunk_uid`, while returned rows
  preserve original input order.
- Raw and canonical-content exact-duplicate lineage records the representative UID.
- Input-permutation regression passes.
- Existing frozen-corpus read-only comparison:
  - broad tranche: `254 -> 254` pass, decision changes `0`;
  - path-stratified tranche: `868 -> 868` pass, decision changes `0`;
  - code-domain v2 combined: `3313 -> 3313` pass, two row decisions swap one
    exact-duplicate representative; aggregate membership count is unchanged.
- Added `173_build_redundancy_validity_benchmark.py` and
  `validation/fixtures/redundancy_validity_benchmark_cases.json`.

First benchmark result:

```text
pairs: 10
hard duplicates: 4
related-useful: 4
independent: 2

current Hamming 3 / Jaccard 0.75 / containment 0.88:
  precision: 1.0
  recall: 0.5
  F1: 0.666667
  false positives: 0
  false negatives: 2

bounded fixture-best:
  Hamming 18 / Jaccard 0.50 / containment 0.95
  precision: 1.0
  recall: 1.0
```

Do not promote the fixture-best threshold. The missed current cases are a
literal-varied generated test and a containment-1.0 boilerplate extension;
both are lost by the strict SimHash candidate gate.

Stage-B saturation result:

```text
group size 1: mean risk 0.0, mean structural matches 0
group size 2: mean risk 0.85, mean structural matches 1
group size 5: mean risk 0.85, mean structural matches 4
```

This confirms that current Stage-B evidence records saturation count but the
binding risk collapses every nonzero structural count to `0.85`.

Next order:

1. Expand the pair benchmark with repository-disjoint real code and
   documentation, stratified by content type and chunk length.
2. Measure threshold precision/recall and useful-data dropout on that expanded
   development benchmark.
3. Design a calibrated saturation-response function using match count and
   cluster size; keep it as a new ablation arm rather than silently replacing
   the frozen selector.
4. Separate useful recurrence from duplicate risk in benchmark labels and
   reporting.
5. Rerun Stage-B policy arms and proxy-scale fixed-recipe training.
6. Only then run the decisive Qwen3-4B development comparison and untouched
   confirmation.

## 2026-06-26 Real-Corpus Redundancy Calibration and Holdout

Development silver calibration:

```text
source repositories: 25
pairs: 111
hard duplicate: 75
nonduplicate controls: 36

current threshold:
  precision: 1.0
  recall: 0.626667
  near-only recall: 0.44
  useful-data dropout: 0.0
```

Independent silver holdout:

```text
source repositories: 13
calibration repository overlap: 0
pairs: 56

current:
  precision: 0.958333
  near-only recall: 0.384615
  dropout: 0.058824

zero-dropout development challenger:
  threshold: Hamming 5 / Jaccard 0.40 / containment 0.70
  holdout precision: 0.964286
  near-only recall: 0.538462
  dropout: 0.058824
```

No threshold arm passed the frozen holdout gate (`precision >= 0.98`,
`dropout <= 0.05`). The current threshold also exposed one semantic-change
false positive, showing that lexical identity can remain high after a
meaning-changing operator mutation.

Cluster dropout audit for the conservative challenger:

```text
additional current records removed: 13
additional token proxy removed: 8026
Stage-B-selected records removed: 4
Stage-B-selected token proxy removed: 1241
content types: code 1, test 12
mean lost Selection Value Evidence: 0.810276
```

Decision:

```text
hold_challenger
keep canonical Stage-A threshold unchanged
move saturation handling to Stage B soft evidence
```

Stage-B saturation ablations:

```text
binary_current:
  selected: 1424
  tests: 610
  repositories: 47

log_count:
  selected: 1428
  tests: 610
  repositories: 47
  concise selected: 386 vs 384
  overlap with current: 1416
  added / removed: 12 / 8
  selection Jaccard: 0.986072
```

`log_count` is frozen as the sole proxy-training candidate before model
outcomes. It is not canonical and cannot be promoted from proxy metrics.

Next order:

1. Materialize equal-token `binary_current`, `log_count`, and
   Stage-A-random proxy-training arms.
2. Freeze a 0.4B-1.5B proxy model, seeds, token budget, heldout, and detectable
   effect rule before training.
3. Run target heldout NLL plus general retention and template-saturation
   diagnostics.
4. Freeze the proxy decision; do not iterate until positive.
5. Only a passing proxy candidate may enter Qwen3-4B development.
6. Complete task, forgetting, contamination, and seed-stability guardrails.
7. Run untouched repository/time confirmation and freeze the paper claim.

## 2026-06-26 Redundancy Saturation Proxy Data Arms Frozen

Completed step 1 of the proxy-training sequence:

```text
output:
  outputs/temporal_code_collection/redundancy_saturation_proxy_arms_v1/

shared token-proxy training cap:
  327222

binary_current_equal_budget:
  records: 1424
  materialized token proxy: 327222
  repositories: 47
  code / documentation / test: 794 / 20 / 610

log_count_equal_budget:
  records: 1428
  materialized token proxy: 327223
  repositories: 47
  code / documentation / test: 798 / 20 / 610

stageA_random_common_disjoint_equal_budget:
  records: 337
  materialized token proxy: 327390
  repositories: 42
  code / documentation / test: 157 / 27 / 153
```

Selector relationship:

```text
intersection: 1416
binary only: 8
log_count only: 12
union: 1436
Jaccard: 0.986072
```

Common random disjointness:

```text
overlap with binary: 0
overlap with log_count: 0
overlap with selector union: 0
```

The three files are equal-budget data-arm inputs, not yet exact
model-tokenizer arms. Exact equality is deferred to the next step because the
proxy model/tokenizer has not been frozen. After that freeze, each file must be
packed in listed order to exactly the same tokenizer-token count, truncating
only the final consumed record where required.

Remaining order:

1. Pack all three arms to the identical tokenizer-token count and hash the
   packed blocks.
2. Run equal-compute proxy training.
3. Evaluate target heldout NLL, general retention, and template-saturation
   slices.
4. Freeze the proxy decision without repeated tuning.
5. Allow only a passing candidate into Qwen3-4B development.
6. Complete remaining guardrails and untouched confirmation.

## 2026-06-26 Redundancy Proxy Experiment Contract Frozen

Completed step 2 of the proxy-training sequence before reading any proxy
training outcome:

```text
config:
  configs/temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json

model:
  Qwen/Qwen2.5-0.5B
  revision: 060db6499f32faf8b98477b0a26969ef7d8b9987
  local snapshot and config/tokenizer/model hashes frozen

training:
  seeds: 11, 23, 37
  sequence length: 1024
  micro batch: 1
  gradient accumulation: 8
  optimizer steps: 40
  exact tokens per arm: 327680
  exact blocks per arm: 320

heldout:
  records: 342
  repositories: 17
  exact tokens: 146432
  exact blocks: 143
  train-repository overlap: 0
```

Raw Qwen2.5 tokenizer availability before the common cap:

```text
binary_current: 385818
log_count: 385812
common disjoint Stage-A random: 380059
```

Decision contract:

1. Curation effect is `NLL(common Stage-A random) - NLL(log_count)`.
   Its one-sided paired 95% lower confidence bound must be at least `0.002`,
   with at least two of three positive seed deltas.
2. Candidate non-inferiority is `NLL(log_count) - NLL(binary_current)`.
   Its one-sided paired 95% upper confidence bound must not exceed `0.002`,
   with at least two of three seeds non-worse.
3. Effects below `max(0.002, paired MDE95)` are inconclusive and cannot support
   a positive Utility claim.
4. The pre-frozen template-saturation diagnostic and all general-text,
   general-task, and code-retention guardrails are mandatory.
5. Missing required evidence means `abstain`. Qwen3-4B cannot be used to tune
   this proxy cycle.

The freeze report is stored inside
`validation/frozen_contracts/redundancy_proxy_experiment_freeze_report.json`
rather than `outputs/validation`. On this Windows checkout,
`outputs/validation` is a junction to `C:\UNLV-Research\...`, which caused the
previous `PermissionError` and is not a portable location for a source-level
pre-registration contract.

Validated:

```text
validation/test_redundancy_proxy_experiment_freeze.py: pass
validation/test_redundancy_saturation_proxy_arms.py: pass
validation/test_redundancy_saturation_proxy_candidate.py: pass
validation/test_redundancy_saturation_ablations.py: pass
```

Remaining order:

1. Materialize and hash the three exact `320 x 1024` training-block tensors
   plus the `143 x 1024` heldout tensor.
2. Freeze the template-saturation diagnostic artifact and retention inputs.
3. Run the three arms for seeds `11/23/37` at identical compute.
4. Evaluate the pre-registered comparisons and guardrails.
5. Freeze `promote`, `hold`, `reject`, or `abstain` without another proxy
   tuning loop.
6. Only `promote` may enter Qwen3-4B development.

## 2026-06-26 Redundancy Proxy Exact Blocks Materialized

Completed the exact-token materialization step without loading or training a
model:

```text
script:
  181_materialize_redundancy_proxy_blocks.py

manifest:
  validation/frozen_contracts/redundancy_proxy_packed_blocks_manifest.json

D-drive-backed artifact directory:
  outputs/redundancy_saturation_proxy_qwen25_0p5b_v1/token_blocks/

serialization:
  safetensors
  key: input_ids
  dtype: int32
  record order: frozen JSONL order
  boundary rule: append EOS after each nonempty record
  cap rule: stop exactly at the frozen token count
```

Artifacts:

```text
binary_current_equal_budget:
  shape: 320 x 1024
  tokens: 327680
  file SHA-256:
    ee2f51207216d1600f6b01277c0d11c036127bbc41eefc71c46a7b7dacac0afe
  tensor-content SHA-256:
    4966334ce46306880d7c3458eb0f498ad7e1489f01c49c71df6e475d4e38c746

log_count_equal_budget:
  shape: 320 x 1024
  tokens: 327680
  file SHA-256:
    88b45bf6e171a188430ad6a046a2c673450b6023a48f72a63c133ca195bbb279
  tensor-content SHA-256:
    6a601948bb455e5cdd123d150af7d7541c2a6376fad87e7fe7a8ae12124b9a98

stageA_random_common_disjoint_equal_budget:
  shape: 320 x 1024
  tokens: 327680
  file SHA-256:
    412075d0983bf9d5a6df3b8eee50a07c6101ba6bc8da64a0ea3b5441f21a6ef9
  tensor-content SHA-256:
    1b81bf2a7512efba090a7f658fbd315eb08226c25f4471fde6e7da557a9218cb

development_code_nll_heldout:
  shape: 143 x 1024
  tokens: 146432
  file SHA-256:
    b04d45dffe2be69a94ed24b53402de8fd7b5c0b4b7e7308fc5063998c8254561
  tensor-content SHA-256:
    8180fb033a54e63e01f2d0c73417045d33469d0a0f1a33f93e4f2ebf4fab356b
```

All four streams consume a partial final record to meet the exact token cap.
No padding or partial sequence block is present. The three training tensors
have equal shapes and distinct token-content hashes. Heldout repository
overlap remains zero.

The full materialization was run twice. Every serialized file hash and
tensor-content hash was identical across both runs.

Validated:

```text
validation/test_redundancy_proxy_packed_blocks.py: pass
validation/test_redundancy_proxy_experiment_freeze.py: pass
validation/test_redundancy_saturation_proxy_arms.py: pass
```

One validation boundary was corrected during this step: Qwen's reported
`vocab_size` excludes added special tokens, so valid EOS ID `151643` can equal
the base-vocabulary size. Token-range validation now uses `len(tokenizer)`,
while the manifest records both base vocabulary and full tokenizer size.

Remaining order:

1. Freeze the template-saturation mechanism diagnostic artifact and the exact
   general-text, general-task, and code-retention inputs.
2. Build or adapt the Qwen2.5-0.5B QLoRA runner to consume these immutable
   `safetensors` blocks with seed-controlled block shuffling.
3. Run binary, log-count, and common Stage-A-random arms for seeds `11/23/37`
   at identical compute.
4. Evaluate frozen heldout NLL, mechanism diagnostics, and retention
   guardrails.
5. Apply the pre-registered detectable-effect and non-inferiority rules once.
6. Freeze `promote`, `hold`, `reject`, or `abstain`; only `promote` may enter
   Qwen3-4B development.

## 2026-06-26 Proxy Mechanism and Retention Inputs Frozen

Completed the final outcome-free input freeze before proxy training:

```text
script:
  182_freeze_redundancy_proxy_evaluation_inputs.py

config:
  configs/temporal_code_redundancy_proxy_evaluation_inputs_v1.json

report:
  validation/frozen_contracts/
    redundancy_proxy_evaluation_inputs_freeze_report.json

status:
  redundancy_proxy_evaluation_inputs_frozen

blockers:
  none
```

Mechanism definition:

```text
match_count = 0: no structural recurrence
match_count = 1: single recurrence, reported but not saturation
match_count >= 2: high template saturation
match_count >= 4: severe template saturation
```

This boundary is deliberate. Reusable code patterns and tests can recur once
without being harmful duplication. The count-sensitive policy must react more
strongly as recurrence accumulates, while not increasing high-saturation
exposure in the exact training stream.

Mechanism result:

```text
log_count risk:
  increases before the bounded cap and is nondecreasing

binary_current risk:
  flat after the first match

exact-stream count>=2 token share:
  binary_current: 0
  log_count: 0

exact-stream count>=4 token share:
  binary_current: 0
  log_count: 0

repositories:
  binary_current: 47
  log_count: 47

test-token share:
  binary_current: 0.315067
  log_count: 0.316388
```

`log_count` exposes slightly more single-recurrence tokens. This is reported
as a diagnostic and not counted as saturation. The candidate passes because
it improves count-response fidelity, does not increase `count>=2` saturation,
and preserves repository/test support.

General-text retention:

```text
source:
  Wikitext103 validation/test frozen holdout
source SHA-256:
  dc5c0dcb9838f797faaa0d218faf6d825b8478f94b8c0edaefa83f996bd4e99e

Qwen2.5 blocks:
  496 x 1024
  packed tokens: 507904
  dropped tail: 97
  file:
    outputs/redundancy_saturation_proxy_qwen25_0p5b_v1/
      evaluation_inputs/wikitext103_qwen25_0p5b.safetensors
  file SHA-256:
    d77df0ef361257685e06d9542fae47d277da2ff09081941ed24397ab30ea6689

margin:
  one-sided 95% seed-level upper confidence bound on NLL increase <= 0.01
```

General-task retention:

```text
lm_eval version: 0.4.12
num_fewshot: 0
limit: full validation split

HellaSwag: 10042
ARC-Challenge: 299
PIQA: 1838
WinoGrande: 1267

per-suite maximum regression: 0.02
macro maximum regression: 0.01
```

Every task YAML/helper and cached validation Arrow file is hash-frozen.

Code retention:

```text
EvalPlus version: 0.3.1
development tasks: 284
HumanEval+: 90
MBPP+: 194
execution tier: E2
temperature: 0
samples per task: 1
per-suite and macro maximum regression: 0.02
```

The development task-ID order, HumanEval+/MBPP+ local caches, split plan,
prevalidation report, Docker image tag, and image ID are frozen. Confirmatory
task IDs remain unavailable to this proxy decision.

The freeze was regenerated twice. The config and Wikitext tensor hashes were
identical.

Validated:

```text
validation/test_redundancy_proxy_evaluation_inputs.py: pass
validation/test_redundancy_proxy_packed_blocks.py: pass
validation/test_redundancy_proxy_experiment_freeze.py: pass
```

Remaining order:

1. Implement the immutable-block Qwen2.5-0.5B QLoRA runner with seed-controlled
   block shuffling and adapter/result hash manifests.
2. Verify runner semantics without reading partial arm outcomes.
3. Run the three trained arms for seeds `11/23/37` at the frozen 40-step
   compute budget.
4. Evaluate base plus all adapters on heldout NLL, Wikitext NLL, general tasks,
   and EvalPlus development.
5. Apply the frozen paired detectable-effect, non-inferiority, mechanism, and
   retention rules once.
6. Freeze the proxy decision. Do not tune this proxy cycle after outcomes.

## 2026-06-26 Redundancy Proxy Training and Decision Complete

Runner:

```text
183_run_redundancy_proxy_qlora.py
physical GPU: CUDA device 1, NVIDIA GeForce RTX 3070 Ti
completed runs: 9 / 9
per run: 320 micro-steps, 40 optimizer steps, 327680 tokens
trainable LoRA parameters: 8798208
peak allocated GPU memory: 3.12-3.68 GB
```

NLL evaluator:

```text
184_evaluate_redundancy_proxy_nll.py
target blocks: 143 x 1024
general-text blocks: 496 x 1024
base plus adapters evaluated: 10 / 10
```

Mean target heldout NLL:

```text
base_no_update: 1.756293
binary_current: 1.697021
log_count: 1.697254
common disjoint Stage-A random: 1.711347
```

Primary curation effect:

```text
estimand:
  NLL(common Stage-A random) - NLL(log_count)

seed deltas:
  11: 0.013302
  23: 0.014551
  37: 0.014425

mean: 0.014093
one-sided 95% lower bound: 0.012934
paired MDE95: 0.001159
required lower bound: 0.002
result: pass
```

This is positive development evidence that the Stage-B-curated arm beats a
common disjoint Stage-A-random arm under matched tokens and compute.

Candidate comparison:

```text
estimand:
  NLL(log_count) - NLL(binary_current)

seed deltas:
  11: 0.000478
  23: 0.000036
  37: 0.000186

mean: 0.000233
one-sided 95% upper bound: 0.000612
non-inferiority margin: 0.002
statistical non-inferiority: pass
directional non-worse seeds: 0 / 3
required directional non-worse seeds: at least 2 / 3
directional promotion condition: fail
```

The candidate is not meaningfully harmful: its difference is below the
practical floor and inside the non-inferiority margin. But every paired seed
slightly favors `binary_current`, so there is no evidence to replace the
canonical policy.

General-text NLL:

```text
base_no_update: 2.854381
binary_current: 2.835880
log_count: 2.835908
common disjoint Stage-A random: 2.843166

maximum allowed upper-bound increase: 0.01
all arms: pass
```

Frozen decision:

```text
script:
  185_build_redundancy_proxy_decision.py

report:
  validation/frozen_contracts/redundancy_proxy_decision_report.json

candidate decision:
  hold_log_count_keep_binary_current_directional_nonworse_failed

log_count promotion allowed:
  false

Qwen3-4B development allowed for log_count:
  false

framework release:
  abstain_missing_general_task_and_evalplus_guardrails
```

General-task and EvalPlus were not run for `log_count` after the futility
boundary because the failed mandatory directional condition makes promotion
impossible regardless of those outcomes. This avoids spending compute on a
decision that cannot change.

Interpretation:

1. The curation framework produced a subset that clearly beats common
   Stage-A random on the frozen target heldout.
2. Count-sensitive `log_count` did not improve on the current binary policy.
3. Keep `binary_current` canonical.
4. Do not retune `log_count` from these outcomes.
5. Complete general-task and EvalPlus guardrails only for the canonical
   framework release path.
6. Any future saturation formula must start a new preregistered proxy cycle.

Validated:

```text
validation/test_redundancy_proxy_decision.py: pass
validation/test_redundancy_proxy_nll_analysis.py: pass
validation/test_redundancy_proxy_runner.py: pass
validation/test_redundancy_proxy_evaluation_inputs.py: pass
validation/test_redundancy_proxy_packed_blocks.py: pass
validation/test_redundancy_proxy_experiment_freeze.py: pass
```

General-task runner update:

- `148_run_code_domain_general_task_guardrail.py` now saves/merges task-level
  partial results. A task-specific run such as `--tasks piqa` can be reused by
  later full-task runs because completion checks inspect the actual
  `lm_eval_results.results` keys, not just the stored status string.
- `149_build_code_domain_general_task_guardrail_report.py` reports partial
  general-task results explicitly instead of treating them as generic status
  mismatches.
- Regression test:
  `validation/test_general_task_runner_incremental.py`.

## 2026-06-26 Canonical Binary Guardrails Complete

The rejected `log_count` candidate was not given more guardrail compute after
the frozen futility boundary. The canonical framework path remains
`binary_current_equal_budget`, so the remaining Stage-C checks were run only
for:

```text
base_no_update
binary_current_equal_budget seeds 11, 23, 37
```

Execution contract:

```text
186_freeze_redundancy_canonical_guardrails.py
configs/temporal_code_redundancy_canonical_guardrails_qwen25_0p5b_v1.json
```

General-task retention:

```text
report:
  validation/frozen_contracts/redundancy_canonical_general_task_guardrail_report.json

status:
  general_task_guardrail_passed

base macro:
  0.513293

binary_current macro:
  0.514972

macro absolute regression:
  -0.001680
```

Per-suite regressions all passed the frozen `0.02` suite margin:

```text
HellaSwag:      -0.002954
ARC-Challenge:  -0.007964
PIQA:            0.002358
WinoGrande:      0.001842
```

EvalPlus development E2:

```text
sample generation:
  143_generate_code_domain_evalplus_samples.py

execution:
  144_run_code_domain_evalplus_guardrail.py

report:
  validation/frozen_contracts/redundancy_canonical_evalplus_guardrail_report.json

status:
  evalplus_development_guardrail_passed

base macro pass@1:
  0.145934

binary_current mean macro pass@1:
  0.196449

macro absolute regression:
  -0.050515
```

Per-suite EvalPlus development results:

```text
HumanEval+ base:        0.044444
HumanEval+ binary mean: 0.111111
MBPP+ base:             0.247423
MBPP+ binary mean:      0.281787
```

Combined canonical decision:

```text
script:
  187_build_redundancy_canonical_guardrail_decision.py

report:
  validation/frozen_contracts/redundancy_canonical_guardrail_decision_report.json

status:
  canonical_qwen25_0p5b_development_guardrails_passed

release decision:
  abstain_not_a_production_release
```

Interpretation:

1. The canonical binary recurrence path has now passed the frozen Qwen2.5-0.5B
   development evidence stack: target heldout NLL vs common Stage-A random,
   Wikitext general-text NLL, general-task retention, and EvalPlus development
   E2.
2. This is real progress for the framework direction: the curated path did not
   merely shrink data; it improved the target code heldout and did not fail
   external retention guardrails.
3. At this historical checkpoint it was still not a production or paper-final
   release claim. Later blocks closed the target-size rerun, confirmatory
   guardrail, and paper-level reproducibility-packaging blockers for the
   bounded paper package; production deployment remains blocked separately.
4. Utility remains Stage C only and never enters the selector objective.

Implementation notes:

- `143_generate_code_domain_evalplus_samples.py` now accepts plan-defined arms,
  uses local `snapshot_path` when present, and supports deterministic
  left-padded batch generation with `--generation-batch-size`. The decoding
  contract remains greedy temperature-0 generation with the same max-token
  bound.
- `148_run_code_domain_general_task_guardrail.py` now accepts plan-defined arms
  and local model snapshots for the canonical Qwen2.5 proxy contract.

Validated:

```text
python validation/test_redundancy_canonical_guardrails.py
python validation/test_redundancy_canonical_guardrail_decision.py
python validation/test_general_task_runner_incremental.py
python -m py_compile 143_generate_code_domain_evalplus_samples.py 186_freeze_redundancy_canonical_guardrails.py 187_build_redundancy_canonical_guardrail_decision.py
```

## 2026-06-26 Target-Size Qwen3-4B Development Rerun

The frozen canonical binary recurrence path was carried into the target-size
Qwen3-4B development rerun. This is not a new selector candidate and does not
reintroduce `log_count`.

Frozen contract and blocks:

```text
script:
  188_freeze_redundancy_target_size_development.py

plan:
  configs/temporal_code_redundancy_target_size_development_qwen3_4b_v1.json

blocks:
  validation/frozen_contracts/redundancy_target_size_qwen3_4b_blocks_manifest.json

output dir:
  outputs/redundancy_target_size_qwen3_4b_v1/
```

Training contract:

```text
model:
  Qwen/Qwen3-4B-Base

local snapshot:
  D:\UNLV-Research\hf_cache\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539

trained arms:
  binary_current_equal_budget
  stageA_random_common_disjoint_equal_budget

base reference:
  base_no_update

seeds:
  11, 23, 37

exact train tokens per arm:
  327,680

heldout tokens:
  65,536

optimizer steps:
  40
```

Execution note:

- The first `train-missing` multi-run process completed `binary_current`
  seed11 but then failed to produce a second adapter for several hours while
  holding the 3070 Ti. The process was stopped and the remaining runs were
  executed one run per process.
- One-run-per-process execution completed all six adapters successfully on
  physical GPU 1, RTX 3070 Ti.
- `141_run_code_domain_development_qlora.py` now accepts local
  `snapshot_path` and `.safetensors` token blocks.

Target heldout NLL report:

```text
script:
  189_build_redundancy_target_size_development_report.py

report:
  validation/frozen_contracts/redundancy_target_size_qwen3_4b_development_report.json

status:
  target_size_development_passed
```

NLL results:

```text
base_no_update:
  1.374647

binary_current_equal_budget:
  seed11  1.342053
  seed23  1.341521
  seed37  1.341691
  mean    1.341755

stageA_random_common_disjoint_equal_budget:
  seed11  1.345532
  seed23  1.346688
  seed37  1.348146
  mean    1.346789

baseline_minus_binary mean:
  0.005034

required development margin:
  0.005000
```

Interpretation:

1. Target-size code heldout NLL passes the frozen mean-margin rule, but only
   narrowly.
2. All three paired seeds are directionally positive for `binary_current`, but
   seed11 does not individually clear the `0.005` margin.
3. This is meaningful target-size development evidence for the framework, not
   a production or paper-final claim.
4. Target-size general-text retention, general-task retention, and EvalPlus
   development guardrails are now observed in the current artifacts; target-size
   release support is no longer the active hard-gate blocker.
5. Utility remains Stage C only and was not used in Stage B.

Validated:

```text
python -m py_compile 188_freeze_redundancy_target_size_development.py 189_build_redundancy_target_size_development_report.py 141_run_code_domain_development_qlora.py
target-size validation functions:
  validation/test_redundancy_target_size_development.py
  validation/test_redundancy_target_size_development_report.py
```

## 2026-06-27 Paper-Claim Hardening Pass

The next paper-defense hardening chunk was implemented.

Claim boundary:

```text
docs/paper_claim_boundary_and_release_gate.md
```

The current claim remains development evidence only. Intrinsic Quality,
production-ready Core validity, comprehensive duplicate coverage, and
release-ready framework claims are not supported.

Scoring reproducibility:

- `03_score_core_metrics.py` remains the canonical single-process scorer.
- `191_score_core_metrics_parallel.py` is the Windows full-corpus runner. It
  uses dataset-level parallelism, shared scorer-cache inputs, atomic per-dataset
  `.tmp` outputs, and a final manifest write after all datasets complete.
- The scoring manifest includes `scoring_reproducibility` and
  `scoring_execution` when scoring is run against a valid index.
- The reproducibility surface hashes:
  - `03_score_core_metrics.py`
  - `191_score_core_metrics_parallel.py`
  - `signals/core.py`
  - `quality/reference_quality.py`
  - `data_eval_common.py`
  - `models/reference_quality_model.joblib`
  - `models/reference_quality_model.meta.json`
  - index DB input
- Current status: `outputs/index/index.sqlite` and
  `outputs/scored/scoring_manifest.json` are valid rebuilt Windows artifacts.
  Do not treat the old zero-byte-index caveat as active.

Block 2 validation commands:

```text
python -m py_compile index/build.py 191_score_core_metrics_parallel.py signals/core.py 03_score_core_metrics.py validation/test_index_pass2_batching.py
python validation/test_index_pass2_batching.py
python validation/test_scoring_reproducibility_manifest.py
python validation/test_scoring_schema_separation_audit.py
```

Utility leakage audit:

```text
script:
  164_build_selector_utility_leakage_audit.py

report:
  outputs/validation/selector_utility_leakage_audit.json

status:
  selector_utility_leakage_audit_passed
```

The audit now checks both selector surfaces:

```text
policy/subsets.py
ingestion/code_selection.py
```

It also scans the full temporal-code Stage-B evidence artifact by default and
checks its keys against an explicit allowlist. This is stronger than the
previous policy-only, 2,000-row sample string audit.

Hard paper/release gate:

```text
script:
  190_run_paper_claim_release_gate.py

report:
  outputs/validation/paper_claim_release_gate_report.json

current status:
  paper_curation_stage_claim_gate_passed
```

Current paper-claim blockers:

```text
none
```

Current production-deployment blockers:

```text
production_core_validity_not_supported
```

The previous v2 confirmatory, canonical guardrail, and target-size development
guardrail blockers are closed in the current artifacts. The remaining blocker is
not a paper-claim blocker; it is a production-deployment blocker. Current Core
behavior checks and engineering ledgers support the curation-stage framework
claim, but they are still scoped evidence rather than production-grade detector
certification.

Paper Method section:

```text
docs/paper_method_core_metric_policy.md
```

Limitations and threats-to-validity section:

```text
docs/paper_limitations_and_threats.md
```

Frozen paper comparison tables:

```text
outputs/validation/paper_comparison_tables.json
outputs/validation/paper_comparison_tables.md
outputs/validation/paper_comparison_tables.csv
```

Frozen paper reproducibility manifest:

```text
outputs/validation/paper_reproducibility_manifest.json
outputs/validation/paper_reproducibility_manifest.md
```

Remaining paper-submission packaging tasks:

```text
none
```

This split is intentional. The paper claim is the bounded curation-stage
framework claim. Production deployment remains a separate, blocked tier.

Validated:

```text
python validation/test_scoring_reproducibility_manifest.py
python validation/test_selector_utility_leakage_audit.py
python validation/test_selector_utility_leakage_audit_v2.py
python validation/test_paper_claim_release_gate.py
python 198_build_paper_reproducibility_manifest.py
python validation/test_paper_reproducibility_manifest.py
python 196_build_curation_stage_paper_package.py
python validation/test_curation_stage_paper_package.py
python validation/test_code_domain_stage_b_feature_shift.py
python -m py_compile 03_score_core_metrics.py 164_build_selector_utility_leakage_audit.py 190_run_paper_claim_release_gate.py
```
