# Training Data Evaluation Pipeline

This repository contains the canonical training-data evaluation pipeline used for the paper-release run. The active framework keeps four canonical Core axes:

- `Validity`
- `Selection Value Evidence`
- `Redundancy`
- `Coverage`

`Quality` remains only as a legacy field and artifact alias for Selection Value
Evidence. It is not a separate intrinsic construct and has no hard-reject
authority.

`Utility` is not a Core. It is an external downstream measurement used only by
the External Evaluation Protocol after a curation output is frozen.

Research framing note: read `docs/research_framing.md` before changing the
curation or Utility logic. Read
`docs/literature_grounded_curation_direction.md` before changing a Core
definition, threshold, selector objective, or validation arm. It records which
parts of the framework are supported by public evidence and which remain frozen
project hypotheses. Use `docs/framework_requirements_and_test_matrix.md` as the
canonical requirements, pipeline-ownership, test-matrix, and milestone
contract. Use `docs/lm_curation_operational_framework.md` and
`configs/lm_curation_operational_framework_v1.json` as the practical framework
target. The project goal is general LM-training-data curation:

```text
candidate corpus -> full curated pool -> optional budgeted training subset
                 -> supported LM-training release or explicit abstention
```

The active operational boundary is
`docs/operational_curation_boundary_v2.md`, encoded in
`configs/operational_curation_contract_v2.json`: Stage A-B-C is the curation
engine, while the External Evaluation Protocol is offline validation. The current 5M collection
target produced a 7.03M-token Raw-safe artifact, and its three-seed
natural-budget external validation is running. Its historical fixed-`0.4`
Stage-B output is an archived budgeted-subset instance, not a general reduction
policy.

Data collection is assumed to happen upstream. This framework decides which candidate data should actually be used for language-model training, and it must be allowed to emit `insufficient_usable_data`, `reject`, or `abstain` when the evidence does not support release. Time-window or continual-update curation is an application scenario, not the only scope. Utility is an External Evaluation Protocol measure and must not become a Stage-C selector objective.

The framework does not claim to measure intrinsic data quality. `Quality` is a
legacy compatibility label for pre-outcome Stage-C selection-value evidence over
observable information-density, structural-usefulness, and boilerplate-risk
signals. Its practical value must be tested downstream by the External Evaluation Protocol against
matched equal-token baselines and retention guardrails.

Current canonical evidence snapshot:

- Code natural-budget Stage C is a historical positive result: curated v2 uses
  60.8% fewer packed
  training tokens than raw, improves five-seed heldout NLL from `1.209983` to
  `1.200903`, and improves the same-protocol natural-budget EvalPlus macro pass
  rate from `51.08%` to `58.22%`. The Stage-B implementation hashes match the
  current code and the bounded curation-stage paper gate passes.
- The frozen independent LiveCodeBench pilot does not reproduce that gain.
  On 48 disjoint 2025 tasks at seed 101, base, raw-natural, and
  curated-natural each score `9/48` (`18.75%`) pass@1. Generated programs do
  differ across arms, but all correctness outcomes are identical. Treat this
  as neutral Stage-C transfer evidence, not as a positive result and not as
  permission to tune Stage B on LiveCodeBench.
- Math natural-budget Stage C is `abstain`: selector v2 over-filtered and
  worsened heldout NLL from `1.495650` to `1.527065`; selector v3 repairs that
  failure to `1.498987` while using 8.4% fewer packed training tokens than raw,
  but it still does not beat raw and still lacks GSM8K/MATH benchmark
  guardrails.
- The supported claim is a deployment-conditioned LM training-data curation
  framework with explicit pass/fail/abstain decisions. The unsupported claim is
  a universal data-quality detector or all-domain improvement guarantee.
- Domain/capability mix is now an optional Deployment Contract field and an
  audit surface, not a universal fixed ratio. The current Block-2 domain
  composition audit reports paper-domain arms only: raw Code/Math packed-token
  shares are `46.69%`/`53.31%`, while current curated-arm shares are
  `27.29%`/`72.71%`. This is observed composition evidence, not a certified
  joint production mixture.
- Math failure analysis is tracked in
  `docs/math_domain_failure_postmortem.md`; paper wording is bounded by
  `docs/paper_claim_redefinition.md`. The machine-readable claim contract is
  `configs/paper_claim_consistency_contract_v1.json`; verify it with
  `218_build_paper_claim_consistency_audit.py`.
  Domain composition is documented in
  `docs/domain_mix_contract_and_composition_audit.md` and verified with
  `219_build_domain_composition_audit.py`.
  Coverage/domain-mix scope is verified with
  `220_build_coverage_domain_mix_audit.py`; the current audit passes with a
  scope boundary and allows observed-composition claims only, not target-mix,
  Utility, intrinsic-quality, or universal-ratio claims.
- Stage-B policy scope is verified with
  `221_build_stage_b_policy_contract_audit.py`; the current audit passes and
  fixes Stage B as optional budget allocation over retained Stage-A survivors.
  It does not reject usable records, does not require shrinkage, and does not
  consume Utility.
- Canonical paper-evidence execution is now registered in
  `configs/canonical_execution_path_v1.json` and audited by
  `222_build_canonical_execution_registry.py`. The current registry passes with
  8 canonical lightweight rebuild scripts, 2 active entry points, 1 compatibility
  alias, and 216 historical/experimental
  numbered scripts outside the canonical path. This path rebuilds paper
  evidence and claim audits; it is not raw-data acquisition, GPU training, or a
  production release runner.

Only `00_run_data_eval.py` and `run_canonical_paper_evidence.py` are active
operator entry points. Numbered scripts outside the canonical path are
historical/experimental unless the registry explicitly marks them otherwise.

Run the complete canonical rebuild through one entry point:

```powershell
python run_canonical_paper_evidence.py --execute
```

An exit code of `2` means the evidence is correctly blocked by a release gate;
it is not a runner crash.
- The next empirical validation block is the Qwen3-4B Hugging Face
  mixed-corpus retest. It is frozen in
  `configs/hf_mixed_corpus_retest_protocol_qwen3_4b_v1.json` and audited by
  `223_build_hf_mixed_corpus_retest_protocol.py`. The protocol mixes raw-like
  Python code sources with known-high-quality reference sources while
  preserving source labels for audit and forbidding those labels from Stage-B
  selector inputs.

Curation is not required to reduce the corpus. Every non-quarantined Stage-B
pass belongs to the full curated pool. Stage C runs as a selection step only
when an explicit token or compute budget is smaller than that pool. With no
binding budget, `retain_all` is a correct output. A record marked
`budget_not_selected` remains retained and must never be described as rejected
or low quality.

The current execution model is Stage-based:

- Stage A: candidate-record provenance, normalization, and risk quarantine
- Stage B: chunk-level hard gate
- Stage C: optional chunk-level budget allocation
- External Evaluation Protocol: frozen-subset training, benchmarks, and release research

Historical files that contain `stage0`, `stage_a`, `stage_b`, or `stage_c` in
their names retain the prior notation for reproducibility. Their active mapping
is `legacy Stage 0 -> Stage A`, `legacy Stage A -> Stage B`, `legacy Stage B ->
Stage C`, and `legacy Stage C -> External Evaluation Protocol`.

Stage A currently has a versioned candidate-record and release/quarantine
contract in `ingestion/schema.py`. Validate its fixture with:

```bash
python 29_validate_stage0_contract.py
```

Run the legacy-named Stage-A normalization and quarantine processor with:

```bash
python 30_process_stage0_candidates.py
```

Run the labeled Stage-A hazard fixture benchmark with:

```bash
python 166_build_stage0_hazard_benchmark.py
python validation/test_stage0_hazard_benchmark.py
```

The current processor is an auditable heuristic baseline. The fixture benchmark
covers PII, secrets, benchmark-contamination, poisoning, rights restriction,
repository-code operator preservation, and one numeric false-positive guard.
It is still not production-grade real-corpus detector validation.

The final decision report can emit `insufficient_usable_data` when the Stage-A
pool cannot support both selection and a disjoint validation baseline.

The canonical per-record disposition model is:

```text
curation_disposition:
  retained | rejected | quarantined

training_budget_disposition:
  not_requested | selected_for_training_budget | budget_not_selected
```

No fixed rejection quota or target reduction ratio is permitted.

Run the operational decision regression contract with:

```bash
python validation/test_decision_contracts.py
```

Run the operational framework target contract with:

```bash
python validation/test_lm_curation_operational_framework.py
```

Build and validate the Core operational audit with:

```bash
python 161_build_core_operational_audit.py
python validation/test_core_operational_audit.py
```

Build and validate the Core construct-boundary review and selector Utility
leakage audit with:

```bash
python 163_build_core_construct_validity_review.py
python validation/test_core_construct_validity_review.py
python 164_build_selector_utility_leakage_audit.py
python validation/test_selector_utility_leakage_audit.py
python 165_build_core_behavior_audit_v2.py
python validation/test_core_behavior_audit_v2.py
python 167_build_coverage_domain_fixture_benchmark.py
python validation/test_coverage_domain_fixture_benchmark.py
python 168_build_scoring_schema_separation_audit.py
python validation/test_scoring_schema_separation_audit.py
python 169_build_real_corpus_stage0_coverage_audit.py
python validation/test_real_corpus_stage0_coverage_audit.py
python 170_build_stage0_detector_validation.py
python validation/test_stage0_detector_validation.py
python 172_build_stage0_detector_heldout_benchmark.py
python validation/test_stage0_detector_heldout_benchmark.py
python 173_build_redundancy_validity_benchmark.py
python validation/test_redundancy_validity_benchmark.py
python 192_build_core_claim_defense_report.py
python validation/test_core_claim_defense_report.py
python 193_build_stage0_risk_boundary_report.py
python validation/test_stage0_risk_boundary_report.py
python 194_build_stage_c_training_validation_report.py
python validation/test_stage_c_training_validation_report.py
python 195_build_confirmatory_decision_boundary_report.py
python validation/test_confirmatory_decision_boundary_report.py
python 197_build_paper_comparison_tables.py
python validation/test_paper_comparison_tables.py
python 198_build_paper_reproducibility_manifest.py
python validation/test_paper_reproducibility_manifest.py
python validation/test_stage_a_duplicate_representative.py
python 174_build_real_corpus_redundancy_calibration.py
python validation/test_real_corpus_redundancy_calibration.py
python 175_freeze_redundancy_silver_holdout.py
python 176_evaluate_redundancy_silver_holdout.py
python validation/test_redundancy_silver_holdout.py
python 177_build_redundancy_cluster_dropout_audit.py
python validation/test_redundancy_cluster_dropout_audit.py
python 178_run_redundancy_saturation_ablations.py
python validation/test_redundancy_saturation_ablations.py
python 179_freeze_redundancy_saturation_proxy_arms.py
python validation/test_redundancy_saturation_proxy_arms.py
```

Build and validate the canonical paper-evidence execution registry with:

```bash
python 222_build_canonical_execution_registry.py
python validation/test_canonical_execution_registry.py
```

Build and validate the Qwen3-4B HF mixed-corpus retest protocol with:

```bash
python 223_build_hf_mixed_corpus_retest_protocol.py
python validation/test_hf_mixed_corpus_retest_protocol.py
```

The operational audit checks role/stage/claim consistency. The construct review
records which Core concepts are defensible, replaces intrinsic Quality with
Selection Value Evidence, and preserves `Quality` only as a legacy alias. The
behavior audit runs the current expanded labeled and
metamorphic checks, including the frozen temporal-code Stage-B proxy fixture
contract, the heldout Stage-0 detector benchmark, and the current real-corpus
Stage-0/Coverage metadata audit, but it still reports remaining evidence gaps
rather than a production-grade Core-validity proof.

The Redundancy validity benchmark measures the current temporal-code Stage-A
threshold on labeled exact, near-duplicate, related-but-useful, and independent
pairs, and separately checks whether Stage-B soft risk responds to saturation
magnitude. The first bounded result is high-precision but low-recall
(`precision=1.0`, `recall=0.5`) and confirms that structural risk remains
binary after the first match. Its fixture-best threshold is diagnostic only
and must not replace the frozen production threshold without broader
repository-disjoint calibration.

`192_build_core_claim_defense_report.py` is the Block-3 claim ledger. It joins
the Core behavior audit, Redundancy benchmark, scoring schema separation,
selector Utility leakage audit, and paper release gate into one scoped decision:
current Core behavior evidence is useful for engineering defense, but it does
not support intrinsic Quality claims, Utility-in-selector claims, or a
release-ready framework claim.

`193_build_stage0_risk_boundary_report.py` is the Block-4 Stage-0 risk boundary
ledger. It joins the Stage-0 hazard fixture, detector validation precheck,
heldout detector benchmark, and real-corpus Stage-0/Coverage audit. It supports
project-defined quarantine behavior evidence, but blocks production-grade PII,
secret, license, benchmark-contamination, poisoning, legal-clearance, or
training-release safety claims.

`194_build_stage_c_training_validation_report.py` is the Block-5 Stage-C
training-validation ledger. It joins the v2 confirmatory NLL decision,
Stage-C guardrail gap report, canonical 0.5B guardrail decision, and Qwen3-4B
target-size development report. It supports the target-code NLL training-effect
claim for the curated arm. The current Stage-C and target-size guardrails are
closed, and the bounded curation-stage paper claim is supported. Production
deployment remains blocked by Core-validity scope limits.

`195_build_confirmatory_decision_boundary_report.py` records the confirmatory
decision boundary. The current natural-budget Stage-C rerun uses matching
Stage-B implementation hashes, five seeds, and complete EvalPlus evidence; the
bounded curation-stage paper gate passes while production remains blocked.

The repository-disjoint silver calibration now uses 25 source repositories and
111 metamorphic pairs. The former hard-near candidate has precision `1.0`,
recall `0.626667`, and near-only recall `0.44` on development silver evidence.
No threshold arm, including the current candidate, passed the independent
13-repository precision and useful-data-dropout holdout. Fuzzy near-duplicate
evidence is therefore Stage-B-only; Stage A retains only raw/canonical exact
duplicate rejection:
the best conservative challenger still had precision `0.964286` and dropout
`0.058824`. Cluster audit showed it would remove 13 additional current records,
including four Stage-B-selected records. Therefore the canonical Stage-A
threshold remains unchanged.

Count-sensitive structural saturation is instead being tested in Stage B.
`log_count` is frozen as the sole proxy-training candidate because it preserves
all 47 repositories and all 610 selected tests, keeps selection Jaccard
`0.986072` with the current selector, and was chosen without Utility or model
outcomes.

The proxy data arms are materialized under
`outputs/temporal_code_collection/redundancy_saturation_proxy_arms_v1/`:

- `binary_current_equal_budget.jsonl`: 1,424 records, 327,222 token proxies
- `log_count_equal_budget.jsonl`: 1,428 records, 327,223 token proxies
- `stageA_random_common_disjoint_equal_budget.jsonl`: 337 records, 327,390 token proxies

The shared pre-tokenizer training cap is 327,222 token proxies. The common
Stage-A-random arm has zero record overlap with the union of both selector
arms. Exact tokenizer-token packing is intentionally deferred until the proxy
model and tokenizer are frozen.

The proxy experiment is now pre-registered in
`configs/temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json`.
It freezes the cached `Qwen/Qwen2.5-0.5B` snapshot at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, seeds `11/23/37`, sequence
length `1024`, and exactly `327,680` training tokens (`320` blocks) per arm.
The frozen code/test heldout contains `146,432` evaluation tokens (`143`
blocks) from 17 repositories with zero train-repository overlap.

The proxy decision has separate requirements:

- `log_count` must show a detectable curation gain over the common disjoint
  Stage-A-random arm;
- `log_count` must be non-inferior to `binary_current`;
- the frozen template-saturation diagnostic and all retention guardrails must
  pass.

The practical absolute NLL floor is `0.002`. A smaller mean difference is
inconclusive, not evidence of improvement. Missing mechanism or retention
evidence forces `abstain`. No Qwen3-4B outcome may be used to revise this
proxy contract.

Rebuild and validate the freeze with:

```bash
python 180_freeze_redundancy_proxy_experiment.py
python validation/test_redundancy_proxy_experiment_freeze.py
```

The exact proxy tensors are materialized under
`outputs/redundancy_saturation_proxy_qwen25_0p5b_v1/token_blocks/` on the D
drive-backed outputs junction. They use `safetensors`, one `int32 input_ids`
tensor per file, and preserve the frozen JSONL order with EOS between records.

- `binary_current_equal_budget`: `320 x 1024`, SHA-256
  `ee2f51207216d1600f6b01277c0d11c036127bbc41eefc71c46a7b7dacac0afe`
- `log_count_equal_budget`: `320 x 1024`, SHA-256
  `88b45bf6e171a188430ad6a046a2c673450b6023a48f72a63c133ca195bbb279`
- `stageA_random_common_disjoint_equal_budget`: `320 x 1024`, SHA-256
  `412075d0983bf9d5a6df3b8eee50a07c6101ba6bc8da64a0ea3b5441f21a6ef9`
- `development_code_nll_heldout`: `143 x 1024`, SHA-256
  `b04d45dffe2be69a94ed24b53402de8fd7b5c0b4b7e7308fc5063998c8254561`

The serialized and tensor-content hashes remained identical after a second
full materialization. Validate with:

```bash
python 181_materialize_redundancy_proxy_blocks.py
python validation/test_redundancy_proxy_packed_blocks.py
```

The proxy mechanism and retention inputs are frozen in
`configs/temporal_code_redundancy_proxy_evaluation_inputs_v1.json`.

Mechanism precheck:

- `log_count` is monotonic until its bounded risk reaches `1.0`;
- binary risk remains flat after the first structural match;
- `match_count >= 2` and `match_count >= 4` token shares are both unchanged at
  zero between `log_count` and `binary_current`;
- all 47 repositories are retained;
- exact-stream test-token share is preserved.

A single recurrence (`match_count=1`) is reported but is not classified as
saturation. This avoids treating every repeated code pattern as waste.

Frozen retention inputs:

- Wikitext103 general-text NLL: `496 x 1024 = 507,904` Qwen2.5 tokens;
- general tasks: HellaSwag 10,042, ARC-Challenge 299, PIQA 1,838, and
  WinoGrande 1,267 validation examples under `lm_eval 0.4.12`;
- EvalPlus development guardrail: 284 tasks, comprising 90 HumanEval+ and 194
  MBPP+ tasks under `evalplus 0.3.1` and the prevalidated E2 Docker contract.

Task implementations, local dataset caches, task-ID order, evaluator datasets,
and the general-text tensor are hash-frozen. Missing evidence still forces
`abstain`; every retention outcome remains Stage C only.

```bash
python 182_freeze_redundancy_proxy_evaluation_inputs.py
python validation/test_redundancy_proxy_evaluation_inputs.py
```

The frozen proxy cycle is complete for target and general-text NLL:

- all 9 QLoRA runs completed on physical CUDA device 1, RTX 3070 Ti;
- every run consumed exactly 327,680 tokens in 320 micro-steps and 40
  optimizer steps;
- common Stage-A random minus `log_count` target NLL is `0.014093`, with a
  one-sided 95% lower bound of `0.012934`, so the curation-effect gate passes;
- `log_count` minus `binary_current` is `0.000233`, with an upper bound of
  `0.000612`, so it is statistically within the `0.002` non-inferiority
  margin;
- all three seed deltas nevertheless favor `binary_current`, so the frozen
  directional promotion condition fails;
- every arm improves Wikitext NLL relative to base, so the general-text
  retention guardrail passes.

Decision: hold `log_count`, keep `binary_current` canonical, and do not enter
Qwen3-4B development with this candidate. The positive curated-vs-random
result remains valid development evidence for the framework. General-task and
EvalPlus were not run for the rejected candidate after the futility boundary;
the complete framework release therefore remains `abstain`.

Decision artifact:
`validation/frozen_contracts/redundancy_proxy_decision_report.json`.

Canonical binary guardrails are now complete for the Qwen2.5-0.5B development
proxy:

- execution contract:
  `configs/temporal_code_redundancy_canonical_guardrails_qwen25_0p5b_v1.json`;
- general-task report:
  `validation/frozen_contracts/redundancy_canonical_general_task_guardrail_report.json`;
- EvalPlus development E2 report:
  `validation/frozen_contracts/redundancy_canonical_evalplus_guardrail_report.json`;
- combined decision:
  `validation/frozen_contracts/redundancy_canonical_guardrail_decision_report.json`.

Results:

- general-task retention passed: base macro `0.513293`, binary macro
  `0.514972`, macro regression `-0.001680`;
- EvalPlus development retention passed: base macro pass@1 `0.145934`,
  binary mean macro pass@1 `0.196449`, macro regression `-0.050515`;
- combined status:
  `canonical_qwen25_0p5b_development_guardrails_passed`;
- release decision remains `abstain_not_a_production_release`.

Interpretation: the canonical binary path now has positive development
evidence across target code NLL, general-text NLL, general-task retention, and
EvalPlus development E2. It supports continuing the framework with
`binary_current`, but it is not yet a paper-final or production release claim.
The earlier target-size rerun, untouched confirmatory guardrail, and
reproducibility-packaging blockers are closed for the bounded paper-claim
package. Production deployment remains blocked separately.

The target-size Qwen3-4B development rerun has now been frozen and executed for
the canonical binary path:

- freeze/materialization script:
  `188_freeze_redundancy_target_size_development.py`;
- target-size plan:
  `configs/temporal_code_redundancy_target_size_development_qwen3_4b_v1.json`;
- block manifest:
  `validation/frozen_contracts/redundancy_target_size_qwen3_4b_blocks_manifest.json`;
- development report:
  `validation/frozen_contracts/redundancy_target_size_qwen3_4b_development_report.json`.

The rerun uses `Qwen/Qwen3-4B-Base`, trains only
`binary_current_equal_budget` and the common disjoint
`stageA_random_common_disjoint_equal_budget` baseline, and evaluates
`base_no_update` on the same frozen 65,536-token target heldout. All runs were
executed on physical GPU 1, RTX 3070 Ti, one process per adapter.

Target heldout NLL:

- `base_no_update`: `1.374647`;
- `binary_current_equal_budget` mean across seeds 11/23/37: `1.341755`;
- common Stage-A random mean across seeds 11/23/37: `1.346789`;
- baseline-minus-binary mean: `0.005034`;
- frozen development margin: `0.005000`.

Status: `target_size_development_passed`. This is positive but narrow
target-size development evidence with required guardrails observed. It is not a
production claim; the separate bounded curation-stage paper gate now passes,
while Core metric-validity evidence gaps still block production deployment.

Paper-claim hardening is tracked in
`docs/paper_claim_boundary_and_release_gate.md`. The hard release/paper gate is
`190_run_paper_claim_release_gate.py`; unlike report builders, it exits
non-zero for abstain, reject, missing guardrails, incomplete evidence, or any
non-release decision. The current expected status is
`paper_curation_stage_claim_gate_passed`, which means the bounded paper claim is
now the curation-stage framework claim, not a production deployment claim.
The paper Method draft for this bounded claim is
`docs/paper_method_core_metric_policy.md`.
The paper limitations and production-boundary section is
`docs/paper_limitations_and_threats.md`.
The frozen paper comparison tables are generated by
`197_build_paper_comparison_tables.py`.
The frozen paper reproducibility manifest is generated by
`198_build_paper_reproducibility_manifest.py`.

Current paper-claim blockers:

```text
none
```

Current production-deployment blockers:

```text
production_core_validity_not_supported
```

Selector Utility leakage auditing now covers both `policy/subsets.py` and
`ingestion/code_selection.py`, and scans the full temporal-code Stage-B
evidence artifact against an explicit allowlist. Scoring manifests now include
hashes for the scorer source surface and reference-quality model artifacts when
`03_score_core_metrics.py` or the large-corpus Windows runner
`191_score_core_metrics_parallel.py` is run against a valid index.

Build and validate the code-domain Stage-B feature-shift diagnostic with:

```bash
python 162_build_code_domain_stage_b_feature_shift_report.py
python validation/test_code_domain_stage_b_feature_shift.py
```

Run the Stage-C semantic cluster backbone regression contract with:

```bash
python validation/test_cluster_backbone_contract.py
```

Build the OpenWebText2 selected/rejected slice diagnostic with:

```bash
python 31_build_openwebtext2_slice_diagnostic.py
```

See `docs/openwebtext2_failure_analysis.md` for the current interpretation.

## Active Code Surface

Keep day-to-day execution on these scripts:

1. `01_validate_inputs.py`
2. `02_build_index.py`
3. `03_score_core_metrics.py`
4. `191_score_core_metrics_parallel.py`
5. `04_generate_subsets.py`
6. `05_build_dashboard.py`
7. `15_run_selector_baseline_audit.py`
8. `21_build_utility_transfer_gap_report.py`
9. `23_build_core_proxy_alignment_report.py`
10. `24_build_core_proxy_calibration_report.py`
11. `20_build_curation_readiness_report.py`
12. `25_build_stage_c_protocol_decision_report.py`
13. `26_build_strict_baseline_control_report.py`
14. `161_build_core_operational_audit.py`
15. `162_build_code_domain_stage_b_feature_shift_report.py`
16. `163_build_core_construct_validity_review.py`
17. `164_build_selector_utility_leakage_audit.py`
18. `165_build_core_behavior_audit_v2.py`
19. `166_build_stage0_hazard_benchmark.py`
20. `167_build_coverage_domain_fixture_benchmark.py`
21. `168_build_scoring_schema_separation_audit.py`
22. `169_build_real_corpus_stage0_coverage_audit.py`
23. `170_build_stage0_detector_validation.py`
24. `172_build_stage0_detector_heldout_benchmark.py`
25. `192_build_core_claim_defense_report.py`
26. `193_build_stage0_risk_boundary_report.py`
27. `194_build_stage_c_training_validation_report.py`
28. `195_build_confirmatory_decision_boundary_report.py`
29. `196_build_curation_stage_paper_package.py`
30. `197_build_paper_comparison_tables.py`
31. `198_build_paper_reproducibility_manifest.py`
32. `27_build_curation_decision_report.py`
33. `28_build_paper_evidence_table.py`
34. `06_validate_outputs.py`
35. `07_run_property_benchmarks.py`
36. `08_build_metric_maturity_snapshot.py`
37. `run_canonical_paper_evidence.py`

Multi-step runners:

- `00_run_data_eval.py`: development/core pipeline runner
- `run_canonical_paper_evidence.py`: the manifest-driven paper-evidence runner
  for the seven canonical rebuild steps; `13_run_paper_release.py` is a
  compatibility alias

Utility and selector audit scripts:

- `14_run_utility_causal_diagnostics.py`: Utility sensitivity audit
- `15_run_selector_baseline_audit.py`: selector-vs-baseline audit
- `16_run_good_chunk_dropout_audit.py`: rejected useful-chunk dropout audit
- `17_run_policy_ablation_audit.py`: selector policy ablation audit
- `18_compare_candidate_profile.py`: candidate profile comparison
- `19_run_utility_probe_power_sweep.py`: Utility probe power sweep
- `20_build_curation_readiness_report.py`: dataset-level curation readiness and failure triage
- `21_build_utility_transfer_gap_report.py`: feature-space to LM-Utility transfer-gap triage
- `22_run_anti_memorization_probe.py`: targeted repeat-pressure matched Utility diagnostic
- `23_build_core_proxy_alignment_report.py`: Core proxy versus easy-NLL alignment diagnostic
- `24_build_core_proxy_calibration_report.py`: Core proxy calibration target diagnostic
- `25_build_stage_c_protocol_decision_report.py`: Stage-C protocol decision record; keeps dataset follow-ups from becoming selector criteria
- `26_build_strict_baseline_control_report.py`: Stage-C strict-baseline control decision record; separates canonical strict baselines from reported diagnostics
- `27_build_curation_decision_report.py`: final LM-training curation decision report; maps Stage A/B/C evidence to training-use decisions
- `28_build_paper_evidence_table.py`: reproducible JSON/Markdown/CSV paper evidence table built from final Stage A/B/C reports
- `196_build_curation_stage_paper_package.py`: paper-claim package for the bounded curation-stage framework tier and production boundary
- `197_build_paper_comparison_tables.py`: frozen raw-random, Stage-A-random, curated, reference, and ablation comparison tables for the paper package
- `198_build_paper_reproducibility_manifest.py`: frozen commands, configs, artifacts, and hardware/runtime notes for the paper package
- `161_build_core_operational_audit.py`: Core role/stage/claim-boundary audit; not metric validity proof
- `163_build_core_construct_validity_review.py`: Core construct-boundary review; rejects intrinsic Quality measurement and records required behavior evidence
- `164_build_selector_utility_leakage_audit.py`: AST/artifact audit that Stage-B selector/evidence does not consume Utility surrogate fields
- `165_build_core_behavior_audit_v2.py`: expanded labeled/metamorphic Core behavior audit; passes current checks but explicitly reports remaining evidence gaps
- `166_build_stage0_hazard_benchmark.py`: labeled Stage-0 hazard fixture benchmark for PII, secrets, rights, benchmark contamination, poisoning, and code-normalization false positives
- `167_build_coverage_domain_fixture_benchmark.py`: labeled Coverage/domain fixture benchmark that distinguishes explicit domain metadata from source-bucket fallback and collapse
- `168_build_scoring_schema_separation_audit.py`: Core-vs-diagnostic scoring schema audit; keeps `predictive_utility_proxy` out of `core_metrics`
- `169_build_real_corpus_stage0_coverage_audit.py`: current real-corpus Stage-0/Coverage metadata audit; closes the missing real-corpus metadata check but keeps explicit-domain and production-detector caveats visible
- `170_build_stage0_detector_validation.py`: labeled Stage-0 detector validation precheck with per-axis precision/recall for PII, secrets, benchmark contamination, poisoning, and rights
- `172_build_stage0_detector_heldout_benchmark.py`: heldout labeled Stage-0 detector benchmark; closes the development-fixture-only gap but remains project-defined rather than an external public detector benchmark
- `173_build_redundancy_validity_benchmark.py`: labeled Stage-A threshold sweep and Stage-B saturation diagnostic; reports known validity gaps without auto-promoting fixture-optimal thresholds
- `192_build_core_claim_defense_report.py`: Block-3 Core claim ledger; joins Core behavior, Redundancy validity, Utility leakage, schema separation, and release-gate blockers into one scoped claim decision
- `193_build_stage0_risk_boundary_report.py`: Block-4 Stage-0 risk ledger; joins hazard fixtures, detector validation, heldout detector benchmark, and real-corpus Stage-0/Coverage evidence into one scoped safety-boundary decision
- `194_build_stage_c_training_validation_report.py`: Block-5 Stage-C training ledger; joins v2 confirmatory NLL, guardrail evidence, canonical 0.5B guardrails, and Qwen3-4B target-size evidence into a bounded training-effect decision
- `195_build_confirmatory_decision_boundary_report.py`: Block-6 confirmatory decision ledger; joins NLL, guardrails, and the hard paper gate while preserving the separate production blocker
- `174_build_real_corpus_redundancy_calibration.py`: repository-disjoint real-corpus silver calibration by content type, length, and transformation
- `175_freeze_redundancy_silver_holdout.py`: freezes calibration-disjoint source repositories before threshold holdout evaluation
- `176_evaluate_redundancy_silver_holdout.py`: evaluates frozen threshold arms and applies precision/dropout gates
- `177_build_redundancy_cluster_dropout_audit.py`: measures additional corpus and Stage-B-selected data removed by a hard-gate challenger
- `178_run_redundancy_saturation_ablations.py`: compares count-sensitive Stage-B saturation hypotheses without changing the canonical selector
- `179_freeze_redundancy_saturation_proxy_arms.py`: materializes binary-current, log-count, and one common selector-union-disjoint Stage-A-random arm under a shared pre-tokenizer cap
- `148_run_code_domain_general_task_guardrail.py`: Stage-C general-task retention runner; supports task-level incremental saving/merging so long confirmatory evaluations can resume after partial completion
- `149_build_code_domain_general_task_guardrail_report.py`: builds the Stage-C general-task retention guardrail report and distinguishes partial results from missing results
- `32_compare_utility_baselines.py`: same-condition certification comparison of
  canonical, nuisance-matched, and anti-memorization Stage-C controls
- `33_decompose_utility_matching.py`: certification-grade one-factor matching
  decomposition with matched-selected common-support reporting
- `34_prepare_slm_update_experiment.py`: prepares frozen equal-budget arms and
  a manifest for the pre-registered target-SLM continued-training experiment
- `35_freeze_slm_update_plan.py`: freezes target model/tokenizer config and
  converts equal-word arms into matched target-token budgets
- `36_prepare_slm_eval_holdout.py`: prepares Stage-A heldout eval records
  disjoint from all equal-budget SLM training arms
- `37_run_slm_update_training.py`: prepares token blocks and runs target-SLM
  continued-pretraining/eval arms
- `38_build_slm_update_pilot_report.py`: summarizes pilot-only SLM update
  results without treating them as certification evidence
- `39_build_slm_update_scaled_report.py`: summarizes replicated scaled-pilot
  target-SLM results
- `40_build_slm_certification_report.py`: summarizes predeclared full-budget
  primary-arm results
- `41_diagnose_slm_full_budget_shift.py`: compares pilot/full arm composition
  and heldout alignment after a full-budget reversal
- `42_prepare_slm_backfilled_arm.py`: builds exploratory selected-core plus
  Stage-A coverage-backfill training arms
- `43_build_slm_backfill_report.py`: builds the exploratory full-budget
  coverage-backfill comparison and claim boundary
- `44_prepare_slm_confirmatory_holdouts.py`: freezes mutually disjoint broad
  and coverage-stratified untouched internal holdouts
- `45_freeze_slm_backfill_confirmatory_plan.py`: freezes the post-exploratory
  50/50 backfill candidate, fresh seeds, hashes, and success rule
- `46_validate_slm_confirmatory_contract.py`: validates frozen hashes, seeds,
  mixture ratio, and exact train/eval disjointness
- `48_build_release_decision_report.py`: applies a frozen Deployment Contract
  to Stage-C evidence and selects a scoped release or abstention
- `49_build_fineweb_deployment_evidence.py`: builds the current FineWeb
  release-policy evidence bundle
- `50_prepare_external_guardrail_holdout.py`: freezes the provisional external
  WikiText retention holdout and audits exact training/evaluation overlap
- `51_build_capability_guardrail_evidence.py`: adds external retention and
  forgetting evidence for the capability-preserving Deployment Contract
- `52_prepare_retention_replay_arms.py`: prepares target/general-replay
  training-construction development arms without changing Stage B
- `53_build_retention_replay_pareto_report.py`: reports target-gain versus
  external-retention Pareto evidence
- `54_build_retention_recipe_report.py`: compares recipe-matched target and
  retention outcomes
- `55_freeze_retention_recipe_candidate.py`: freezes the first joint-pass
  development recipe for fresh confirmatory evaluation
- `56_prepare_retention_confirmatory_holdouts.py`: freezes untouched target and
  external holdouts with exact and coarse near-duplicate controls
- `57_validate_retention_confirmatory_contract.py`: validates candidate,
  comparator, holdout hashes, and disjointness before training
- `58_build_retention_confirmatory_report.py`: applies the frozen two-seed joint
  success rule

Utility diagnostic baselines include Stage-A random, full random, style/length/quality-band matched Stage-A random, and the canonical multi-matched Stage-A baseline. When transfer-gap triage suggests the matched baseline is winning because it is longer or more repetition-heavy, use `22_run_anti_memorization_probe.py` to run the targeted anti-memorization matched Stage-A diagnostic arm. That arm is Stage-C diagnostic evidence only; it is not part of the selector objective or default full pipeline. For multiple datasets, `22_run_anti_memorization_probe.py --datasets ...` writes dataset-specific reports named `anti_memorization_probe_report_<dataset>.json`; `21_build_utility_transfer_gap_report.py` reads both the legacy default report and these dataset-specific reports. Anti-memorization reports are valid only for matching dataset and profile.

The default Stage-C run also reports `baseline_nuisance_matched_stageA_random`
as an operational-counterfactual candidate. It exactly matches length, style,
domain/source bucket, and repeat pressure while deliberately leaving Quality
and redundancy-risk selector targets unmatched. It is not canonical and cannot
control certification until certification-budget replicated evidence supports
promotion. Match coverage, selected/control disjointness, and no-fallback
construction are recorded in the baseline-pool diagnostics.

Run the three-way same-condition comparison with:

```bash
python 32_compare_utility_baselines.py --datasets openwebtext2_subset --profile style_taxonomy_alignment_probe --profiles configs/style_taxonomy_alignment_probe.json
python 32_compare_utility_baselines.py --datasets fineweb_edu_sample --profile canonical --profiles configs/curation_profiles.json
```

The current certification-condition result shows that baseline construction
can reverse the Utility sign on both the clean positive and raw-like stress
cases. No baseline is promoted. Read `docs/utility_baseline_comparison.md`
before changing Utility baseline roles or Stage B.

Run the matching decomposition with:

```bash
python 33_decompose_utility_matching.py --datasets openwebtext2_subset --profile style_taxonomy_alignment_probe --profiles configs/style_taxonomy_alignment_probe.json
python 33_decompose_utility_matching.py --datasets fineweb_edu_sample --profile canonical --profiles configs/curation_profiles.json
```

The decomposition shows that Quality conditioning is the consistent sign
change point and that restrictive matching can lose common support. Treat
matched controls as conditional mechanism diagnostics; the primary operational
effect remains selected versus an equal-budget disjoint Stage-A random control.

The final target-SLM validation protocol is documented in
`docs/slm_update_experiment_preregistration.md`. Use
`34_prepare_slm_update_experiment.py` to create the equal-budget training arms
from frozen curation outputs. The primary comparison is
`curated_equal_budget` versus `stageA_random_equal_budget`;
`raw_random_equal_budget`, `stageA_all_reference`, and `raw_all_reference` are
supporting operational and efficiency references. This step still keeps Utility
in Stage C only and does not feed target-model outcomes back into Stage-B
selection.

The active code-domain validation decision is documented in
`docs/code_domain_training_validation_protocol.md`. The main next validation is
raw-vs-curated equal-budget continued pretraining on raw-like permissive Python
code, using external code benchmarks and heldout NLL. The strict retrospective
E2 pipeline remains valuable secondary executable evidence, but it is no
longer the primary blocker for beginning raw-corpus training validation.

The first frozen target-SLM config is
`configs/slm_update_qwen25_0p5b_experiment.json`, using
`Qwen/Qwen2.5-0.5B` as the small base checkpoint. After arm generation, run
`35_freeze_slm_update_plan.py` to produce
`outputs/slm_update_experiments/<experiment>/frozen_training_plan.json`. The
current FineWeb-Edu plan uses a Qwen tokenizer matched budget of `22,199,800`
tokens for the primary `curated_equal_budget` versus
`stageA_random_equal_budget` comparison, with sequence length `1024` and long
records split into packed sequence blocks.

The first Windows target-SLM pilot completed with both visible GPUs
(`RTX 4060 Ti` and `RTX 3070 Ti`) using 256 training sequences per arm, 128
internal heldout eval sequences, one seed, and 32 optimizer steps. It is not
certification evidence. Pilot NLL ranking was:

```text
base_no_update:              2.816260
curated_equal_budget:        2.940322
stageA_random_equal_budget:  2.945509
raw_random_equal_budget:     2.949955
```

The pilot direction is `curated < Stage-A random < raw random` by NLL, but all
updates are worse than base because the run is intentionally tiny. Treat this
as runner validation and a reason to run the larger equal-budget experiment,
not as the paper result.

A larger scaled pilot named `pilot_1024_lr1e5` has also completed. It used
1024 training sequences per primary arm, 512 internal heldout eval sequences,
128 optimizer steps, learning rate `1e-5`, and three seeds for the primary
comparison. This is still pilot evidence because the learning rate was chosen
after the smaller smoke run and external benchmark/forgetting/contamination
checks are not complete.

```text
base_no_update mean NLL:              2.805226400
curated mean NLL across 3 seeds:      2.798956268
Stage-A random mean NLL across seeds: 2.799728514
curated - Stage-A random mean NLL:   -0.000772247
curated better seeds:                 3/3
raw-random seed0 NLL:                 2.801262047
```

Full-budget token blocks are prepared under
`outputs/slm_update_experiments/fineweb_edu_canonical_slm_update_v1/token_blocks_full`.
Each equal-budget training arm has `21,679` packed sequences, and the internal
heldout eval has `1,289` packed sequences. The next run should use this full
block directory for the certification-scale primary arms.

Certification-scale run `cert_lr1e5_full` has started from the predeclared
plan in `configs/slm_update_certification_plan_qwen25_0p5b_fineweb.json`.
The first complete seed reversed the scaled-pilot direction:

```text
base_no_update NLL:                         2.778654529
curated_equal_budget seed 20260608 NLL:     2.780961865
stageA_random_equal_budget seed 20260608:   2.778531128
curated - Stage-A random NLL:              +0.002430737
```

`40_build_slm_certification_report.py` therefore marks the current status as
`early_negative_signal_pause_recommended`. Do not claim certification success
from the scaled pilot. Before spending more full-run GPU time, inspect why the
full-budget update favors Stage-A random while the smaller pilot favored
curated.

The diagnosis is documented in `docs/slm_full_budget_shift_interpretation.md`.
An exploratory release/training-construction arm now mixes the selected core
with disjoint Stage-A coverage backfill while keeping the matched target-token
budget:

```text
coverage_backfilled_interleaved50_equal_budget seed 20260608 NLL: 2.777828628
Stage-A random seed 20260608 NLL:                             2.778531128
selected-only curated seed 20260608 NLL:                      2.780961865
```

The backfilled arm is the best current full-budget result, beating Stage-A
random by `0.000702499` NLL. This is exploratory one-seed evidence, not
certification: the arm was created after observing the selected-only reversal.
Read `docs/slm_backfilled_full_result.md`. Freeze and replicate the mixture on
untouched evaluation evidence before making a paper or deployment claim.

The confirmatory protocol is now frozen in
`configs/slm_backfill_confirmatory_plan_qwen25_0p5b_fineweb.json`. It uses
fresh seeds `20260609` and `20260610`; exploratory seed `20260608` is excluded
from confirmatory success counting. Read
`docs/slm_backfill_confirmatory_protocol.md` before running or interpreting the
fresh-seed experiments.

The first fresh confirmatory seed, `20260609`, is complete. The frozen 50/50
candidate loses to Stage-A random on the primary broad holdout by
`+0.000377098` NLL, while winning the secondary coverage-stratified diagnostic
by `-0.000980644` NLL. The frozen rule requires primary wins on both fresh
seeds, so the confirmatory direction is not supported and the remaining
expensive seed is stopped. This identifies a distribution-dependent tradeoff,
not a universal backfill win.

The release layer is now explicitly conditioned on a Deployment Contract. With
the same current FineWeb evidence, `broad_refresh` selects `stageA_broad`,
while `targeted_coverage_refresh` selects `coverage_backfilled`. This is a
scoped distribution tradeoff, not a contradiction or a Stage-B objective
change. Read `docs/deployment_contract_and_release_policy.md`.

`21_build_utility_transfer_gap_report.py` and `20_build_curation_readiness_report.py` also emit `framework_implication` fields that map dataset failures back to framework-level actions: hold selector/Core claims while probe evidence is uninterpretable, inspect Core/Policy proxy calibration when valid Utility sweeps do not support selected chunks, or revise strict baseline controls when repeat-pressure diagnostics support the selected subset. `23_build_core_proxy_alignment_report.py` summarizes the same evidence as a diagnostic-only Core proxy/easy-NLL tension report; `24_build_core_proxy_calibration_report.py` turns mismatch evidence into Core proxy audit targets. `25_build_stage_c_protocol_decision_report.py` records the final Stage-C follow-up decision per dataset so dataset-specific diagnostics do not become dataset-specific selector pass criteria. `26_build_strict_baseline_control_report.py` records whether canonical strict and reported diagnostic controls are sufficient for certification claims. `27_build_curation_decision_report.py` maps the Stage A/B/C evidence matrix into explicit LM-training-use decisions such as `needs_certification_utility`, `utility_probe_unstable`, and `strict_baseline_confounded`. None of these reports changes the selector objective.

The current Core-only follow-up candidate is stored in `configs/core_proxy_length_recurrence_guard_probe.json`. It reduces learnability/repetition bonuses, raises useful-length support, uses finer length buckets for selection/baseline matching, and remains a targeted follow-up candidate rather than a promoted canonical policy. Current Windows evidence: `fineweb_edu_sample` is the first positive LM-training curation case and is classified by `27_build_curation_decision_report.py` as `accepted_for_training` / `certification_candidate` under the canonical profile; it passes Stage A/B, coverage, selected > Stage-A-random Utility, selected > multi-matched Utility, token-exposure checks, the anti-memorization diagnostic, and a replicated `current_like_hash_noise` Utility family. `tiny_textbooks`, `openwebtext2_subset`, and `wikitext103_subset` remain diagnostic stress cases rather than certification-ready releases. OpenWebText2 now has strong profile-matched anti-memorization evidence supporting selected (`delta_nll=+0.002316`, CI low `+0.000386`, 16/16 positive cells), so its canonical strict failure is treated as an easy-NLL baseline-confound candidate and the selector remains on hold. Replication runs still do not produce a common replicated valid preset family across all datasets, so no Utility preset is promoted as a global default. The Stage-C protocol decision report computes both the valid-preset intersection and replicated-family intersection across datasets; replicated-family detection parses `family_b<number>` repeats and counts a family only when at least two compatible completed repeats are valid selected>random and no compatible completed repeat in that family failed.

The positive demonstration dataset is `fineweb_edu_sample`, prepared from `HuggingFaceFW/fineweb-edu` with config `sample-10BT`. Its role is a clean LM-training curation demonstration case: less synthetic/template-heavy than TinyTextbooks, less noisy than OpenWebText2, and closer to public pretraining-data curation literature. Use `prepare_fineweb_edu_sample.py` to materialize `validation/fixtures/fineweb_edu_sample`; generated data remains gitignored.

FineWeb-Edu preparation smoke run:

```bash
python prepare_fineweb_edu_sample.py --target-tokens 5000000 --limit 50000
python 01_validate_inputs.py --datasets-config datasets_config.json
```

Full-size demonstration preparation:

```bash
python prepare_fineweb_edu_sample.py --target-tokens 250000000 --limit 500000
```

`00_run_data_eval.py` defaults to the first two datasets in dual-eval mode. To include the FineWeb-Edu demonstration dataset, pass explicit indexes, for example `--dataset-index 0 1 2 3` for all four datasets or `--dataset-index 3` for a FineWeb-Edu-only preparation/scoring run.

`18_compare_candidate_profile.py` reads the Stage-C protocol decision report as a promotion gate. A candidate cannot be globally promoted unless the protocol report has a replicated global Utility family; without that, the candidate can only remain a targeted follow-up candidate even if some dataset-level Stage-C signals improve.

`validate_outputs.py` now checks cross-report consistency across Utility transfer-gap, curation readiness, Stage-C protocol decision, strict-baseline control, curation decision, power-sweep, candidate-comparison, profile-matched anti-memorization reports, target-SLM experiment manifests, and frozen SLM training plans, including token-exposure caveat status propagation. It also enforces that the semantic cluster backbone uses pairwise lexical separation evidence and that source/domain anchor purity remains diagnostic-only. This is meant to catch stale downstream reports after targeted Utility sweeps or sequential report regeneration. If a downstream report is missing, validation infers the active report profile from `outputs/run_summary.json` instead of hard-coding `canonical`.

Power-sweep aggregation treats profile mismatches as stale/incompatible, and `25_build_stage_c_protocol_decision_report.py` independently rejects a sweep report whose profile does not match the active readiness/profile report. This prevents older `canonical` sweep files from being reused as evidence for `paper_release_certification` or candidate profiles. In aggregate-only mode, omitted datasets are inferred from existing sweep run files to avoid accidentally truncating the report after a targeted dataset sweep.

Latest Windows verification after FineWeb-Edu certification follow-up:

```bash
python 22_run_anti_memorization_probe.py --profile canonical --dataset fineweb_edu_sample
python 19_run_utility_probe_power_sweep.py --profile canonical --datasets fineweb_edu_sample --presets current_like_hash_noise_b0 current_like_hash_noise_b1
python 21_build_utility_transfer_gap_report.py --profile canonical
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
python 27_build_curation_decision_report.py
python validate_outputs.py
```

Observed validation result: `332/332 pass`.

Latest full paper-release certification result on Windows:

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
other three datasets: rejected_for_training
validation: 344/344 pass
```

Paper-ready outputs:

```text
outputs/validation/paper_evidence_table.json
outputs/validation/paper_evidence_table.md
outputs/validation/paper_evidence_table.csv
outputs/validation/curation_stage_paper_package.json
outputs/validation/curation_stage_paper_package.md
outputs/validation/paper_comparison_tables.json
outputs/validation/paper_comparison_tables.md
outputs/validation/paper_comparison_tables.csv
docs/paper_method_core_metric_policy.md
docs/paper_limitations_and_threats.md
```

## Paper Release Run

Inspect the manifest-defined rebuild plan:

```powershell
python run_canonical_paper_evidence.py
```

Rebuild all current paper-evidence reports:

```powershell
python run_canonical_paper_evidence.py --execute
```

The runner writes `outputs/validation/canonical_paper_evidence_run_report.json`.
Exit code `2` records a blocked evidence decision; it does not mean the runner
crashed. `13_run_paper_release.py` remains a compatibility alias.

Paper-release guarantees checked before execution:

- no `synthetic_smoke`
- no `runtime_limits`
- `evaluation_mode=certification`
- `certification_scope=general_purpose`
- real canonical Utility probe: `sshleifer/tiny-gpt2`
- all-pairwise OOD Utility enforcement
- strict `min` Utility statistic with positive delta-NLL and positive CI requirement
- selector objective does not include Utility

## Development Run

Core development run:

```bash
python 00_run_data_eval.py
```

Full development run with property benchmarks and maturity snapshot:

```bash
python 00_run_data_eval.py --flow full
```

The development runner also prints live step progress and writes a persistent log. Latest log:

```bash
tail -f "$(cat outputs/logs/latest_data_eval.log)"
```

## Manual Step Run

Use this when you want to inspect one step at a time:

```bash
python 01_validate_inputs.py
python 02_build_index.py
python 191_score_core_metrics_parallel.py --workers 4
python 04_generate_subsets.py --profiles configs/paper_release.json
python 05_build_dashboard.py
python 15_run_selector_baseline_audit.py --profile paper_release_certification
python 21_build_utility_transfer_gap_report.py --profile paper_release_certification
python 23_build_core_proxy_alignment_report.py
python 24_build_core_proxy_calibration_report.py
python 20_build_curation_readiness_report.py --profile paper_release_certification
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
python 06_validate_outputs.py
python 08_build_metric_maturity_snapshot.py
```

For small debugging runs, `python 03_score_core_metrics.py` remains the
canonical single-process scorer. For full Windows corpus scoring, prefer
`191_score_core_metrics_parallel.py`; it writes per-dataset JSONL files through
atomic `.tmp` paths and writes `outputs/scored/scoring_manifest.json` only after
all dataset files complete.

## Main Configs

- `configs/paper_release.json`: paper-release certification profile
- `configs/curation_profiles.json`: development canonical profile
- `configs/learnability_rescue_probe.json`: Utility/learnability candidate profile
- `configs/utility_scope_smoke.json`: fast schema/regression smoke profile, not paper evidence
- `configs/metric_spec_with_citations.json`: canonical metric contract
- `configs/metric_spec_with_citations.md`: readable metric contract
- `datasets_config.json`: active dataset config

## Core Metrics

Chunk-level selection metrics:

- `structural_validity_gate`
- `reference_quality_score`
- `exact_duplicate_indicator`
- `shingle_near_duplicate_indicator`
- `shingle_near_duplicate_risk_score`

Subset-level validators:

- `subset_coverage_retention_score`
- `small_lm_probe_gain_score`

Diagnostics only:

- `structural_validity_score`
- `explanatory_quality_proxy`
- `tail_cluster_rarity_proxy`
- `predictive_utility_proxy`
- `fixed_token_probe_gain_score`

## Output Contract

Primary outputs:

- `outputs/index/index.sqlite`
- `outputs/index/index_manifest.json`
- `outputs/scored/scoring_manifest.json`
- `outputs/scored/<dataset>.jsonl`
- `outputs/scored/<dataset>.jsonl.tmp` can appear during parallel scoring only;
  treat it as incomplete and do not use it as evidence.
- `outputs/subsets/<profile>/<dataset>.jsonl`
- `outputs/run_manifest.json`
- `outputs/run_summary.json`
- `outputs/dashboard.html`
- `outputs/validation/full_validation_report.json`
- `outputs/validation/selector_baseline_audit.json`
- `outputs/validation/selector_baseline_audit.md`
- `outputs/validation/utility_transfer_gap_report.json`
- `outputs/validation/utility_transfer_gap_report.md`
- `outputs/validation/core_proxy_alignment_report.json`
- `outputs/validation/core_proxy_alignment_report.md`
- `outputs/validation/core_proxy_calibration_report.json`
- `outputs/validation/core_proxy_calibration_report.md`
- `outputs/validation/anti_memorization_probe_report.json`
- `outputs/validation/anti_memorization_probe_report.md`
- `outputs/validation/anti_memorization_probe_report_<dataset>.json`
- `outputs/validation/anti_memorization_probe_report_<dataset>.md`
- `outputs/validation/curation_readiness_report.json`
- `outputs/validation/curation_readiness_report.md`
- `outputs/validation/stage_c_protocol_decision_report.json`
- `outputs/validation/stage_c_protocol_decision_report.md`
- `outputs/validation/strict_baseline_control_report.json`
- `outputs/validation/strict_baseline_control_report.md`
- `outputs/metric_maturity_snapshot.json`

Useful result checks:

```bash
jq '.summary' outputs/validation/full_validation_report.json
```

```bash
jq '.tracked_metrics[] | {metric, maturity, action, validation:(.evidence.validation.label // .evidence.validation.certification_label // null), certification_ready:(.evidence.validation.certification_ready // null)}' outputs/metric_maturity_snapshot.json
```

```bash
jq '.profiles.paper_release_certification | to_entries[] | select(.key|startswith("_")|not) | {dataset:.key, stage_c_pass:.value.stage_c_core_validation.passed, fail_reasons:.value.stage_c_fail_reasons, coverage:.value.subset_coverage_retention_score, utility:.value.small_lm_probe_gain_score, strict_min:.value.stage_c_core_validation.utility_strict_min_gain, utility_pass:.value.stage_c_core_validation.utility_pass, coverage_pass:.value.stage_c_core_validation.coverage_pass}' outputs/run_summary.json
```

Build curation readiness/failure triage:

```bash
python 15_run_selector_baseline_audit.py --profile canonical
python 21_build_utility_transfer_gap_report.py --profile canonical
python 23_build_core_proxy_alignment_report.py
python 24_build_core_proxy_calibration_report.py
python 20_build_curation_readiness_report.py --profile canonical
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
```

Focused anti-memorization Utility diagnostic:

```bash
python 22_run_anti_memorization_probe.py --dataset wikitext103_subset --profile canonical
python 21_build_utility_transfer_gap_report.py --profile canonical
python 23_build_core_proxy_alignment_report.py
python 24_build_core_proxy_calibration_report.py
python 20_build_curation_readiness_report.py --profile canonical
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
```

For the active candidate profile, use profile-matched diagnostics:

```bash
python 22_run_anti_memorization_probe.py --dataset tiny_textbooks --profile core_proxy_length_recurrence_guard --profiles configs/core_proxy_length_recurrence_guard_probe.json --output outputs/validation/anti_memorization_probe_report_tiny_textbooks.json --md-output outputs/validation/anti_memorization_probe_report_tiny_textbooks.md
python 22_run_anti_memorization_probe.py --dataset wikitext103_subset --profile core_proxy_length_recurrence_guard --profiles configs/core_proxy_length_recurrence_guard_probe.json
python 21_build_utility_transfer_gap_report.py --profile core_proxy_length_recurrence_guard
python 23_build_core_proxy_alignment_report.py
python 24_build_core_proxy_calibration_report.py
python 20_build_curation_readiness_report.py --profile core_proxy_length_recurrence_guard
python 25_build_stage_c_protocol_decision_report.py
python 26_build_strict_baseline_control_report.py
```

To accumulate targeted anti-memorization diagnostics without overwriting the default report:

```bash
python 22_run_anti_memorization_probe.py --datasets tiny_textbooks wikitext103_subset --profile canonical
python 21_build_utility_transfer_gap_report.py --profile canonical
```

For a fast diagnostic-only run on a large selected subset, use budget and sample overrides. These runs are not certification evidence:

```bash
python 22_run_anti_memorization_probe.py --dataset tiny_textbooks --profile canonical --train-token-budget 2000 --eval-token-budget 1000 --bootstrap-rounds 20 --max-train-steps 16 --train-epochs 1.0 --train-audit-token-budget 128 --min-probe-bucket-count 1 --holdout-buckets 0 --seeds 17 --max-selected-records 1000 --output outputs/validation/anti_memorization_probe_report_tiny_textbooks.json --md-output outputs/validation/anti_memorization_probe_report_tiny_textbooks.md
```

## Repository Hygiene

Generated datasets, indexes, scored JSONL files, selected subsets, model caches, logs, and dashboards are intentionally excluded from Git. Recreate them by running the pipeline commands above.

## Retention-Aware Target Stability Diagnostics

The current frozen replay-aware recipe preserves the external retention
guardrail but does not satisfy its strict all-seed fresh-confirmation rule. The
latest diagnosis shows a small positive development target effect whose margin
can cross zero under seed and target-holdout variation.

GPU operating rule for this Windows host:

- Check `nvidia-smi` before launching Stage-C training/evaluation.
- Use physical CUDA device `1` / `NVIDIA GeForce RTX 3070 Ti` as the default
  local research GPU, including when the `RTX 4060 Ti` is idle.
- Use the `RTX 4060 Ti` only with explicit per-run user approval, or as an
  approved fallback if the `RTX 3070 Ti` is infeasible.

Run the development-only diagnostics on physical CUDA device 1:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
conda run --no-capture-output -n research python 59_run_target_effect_power_diagnostic.py
conda run --no-capture-output -n research python 60_run_training_trajectory_diagnostic.py
conda run --no-capture-output -n research python 61_run_target_holdout_shift_diagnostic.py
conda run --no-capture-output -n research python 62_build_target_effect_stability_report.py
```

These are Stage-C/release diagnostics only. They must not modify Stage-B
selection or use Utility as a selector objective.

Future candidates must follow
`configs/target_effect_release_protocol_v1.json`. The protocol is
non-retroactive and requires replicated seed-level, multi-holdout, retention,
and task evidence before a release claim.

## Temporal Code Curation Experiment

The first raw-corpus operational experiment is preregistered in
`docs/temporal_code_curation_preregistration.md` and
`configs/temporal_code_curation_protocol_v1.json`.

It uses `Qwen/Qwen3-4B-Base` as the primary model and collects permissively
licensed Python change bundles created after the model's public release.
Primary train, development, and confirmatory splits are both time-disjoint and
repository-disjoint. Local evidence is scoped to matched QLoRA continued
pretraining; full-parameter claims require a later replication.

Build and validate the pre-collection manifests:

```powershell
conda run --no-capture-output -n research python 63_build_temporal_code_collection_manifests.py
conda run --no-capture-output -n research python validation\test_temporal_code_ingestion.py
```

The builder freezes deterministic repository split assignments, benchmark
quarantine entries, and reason-coded bundle admission decisions before generic
Stage 0 or Core scoring.

Authenticated metadata-only repository discovery:

```powershell
gh auth login --web
conda run --no-capture-output -n research python 64_discover_temporal_code_repositories.py
```

This command does not fetch repository content or pull-request prose. Review
and enrich its candidates before freezing repository inclusion or collecting
content. Search metadata alone is never sufficient for the frozen manifest.
The collector uses the GitHub CLI credential store when available and never
writes the token to its output.

Run the preregistered bounded content-fetch smoke and audit:

```powershell
conda run --no-capture-output -n research python 70_fetch_temporal_code_smoke_bundles.py
conda run --no-capture-output -n research python 71_audit_temporal_code_smoke_bundles.py
conda run --no-capture-output -n research python validation\test_temporal_code_smoke_audit.py
```

The current smoke result contains 6 bundles and 35 file records. All 6 obey
the frozen repository/time split. Code-aware PII calibration reduced
quarantined files from 15 to 3, generated-file detection completed for all 32
persisted-content files, and frozen smoke commands passed in all 12 isolated
parent/merge checkouts.

The corrected SWE-bench derived artifact manifest quarantines two bounded-smoke
bundles. Three bundles pass both content and executable-evaluation gates; the
empty bundle is rejected. No eligible train bundle remains, so the bounded
smoke correctly returns `insufficient_usable_data` instead of forcing Stage B.

```powershell
conda run --no-capture-output -n research python 72_verify_temporal_code_test_commands.py
conda run --no-capture-output -n research python 71_audit_temporal_code_smoke_bundles.py
conda run --no-capture-output -n research python 73_prepare_temporal_code_stage0_candidates.py
```

The next protocol boundary is Python syntax-aware chunking for chunk-level
Stage A. Development and confirmatory records must remain isolated from
Stage-B tuning.

That Stage-A boundary is now implemented:

```powershell
conda run --no-capture-output -n research python 74_run_temporal_code_stage_a_smoke.py
```

Repository-code Stage 0 preserves source whitespace and operators. Python
code/test records chunk on top-level AST boundaries; documentation uses
paragraph groups. Raw/canonical exact duplicates, parseability, minimum units,
and pathological repetition are the only Stage-A hard gates. Fuzzy
near-duplicate evidence is reversible Stage-B input because the independent
holdout does not support irreversible rejection. Utility, Selection Value,
Coverage, and benchmark outcomes are excluded from Stage A.

Current bounded result after corrected quarantine: 34 chunks, 23 Stage-A
passes, 11 rejections, and zero train Stage-A-pass chunks.

The frozen train-only Stage-B smoke is now implemented:

```powershell
conda run --no-capture-output -n research python 75_run_temporal_code_stage_b_smoke.py
```

The bounded smoke has no eligible train input and now abstains as
`insufficient_usable_data`. The broad tranche provides the active Stage-B
operational evidence: 175 train Stage-A-pass chunks, 94 selected chunks, and a
49-chunk disjoint Stage-A-random arm at 99.9744% of the selected token-proxy
budget.

This proves bounded Stage-B engineering behavior only. Better selected proxy
scores are expected by construction and do not establish training benefit.
Initial labeled preservation/destruction checks pass `7/7`. Indexed exact
soft-overlap search matches all-pairs on the 175-chunk broad train pool with zero risk,
objective, selected-set, or baseline-set difference. Broad metadata/path-only
enrichment now passes 314 / 499 discovered candidates, and all 314 pass sampled
parent/merge commit reproducibility. The versioned broad freeze rule produces
a 314-repository manifest (train 250 / development 38 / confirmatory 26).
Manifest membership is not training approval. Optional blind review may
support failure analysis but cannot block or promote the policy.

The first deterministic broad tranche contains 20 repositories and at most 40
PR bundles. Fetching produced 40 bundles and 81 file records. Before executable
test verification, 26 bundles are contract-valid, all 40 are split-valid, one
benchmark quarantine match is detected, and six files are PII-quarantined.
Automatically derived test commands are frozen without outcome leakage and
was executed for 19 otherwise eligible bundles. None passed the frozen generic
execution hypothesis: 26 parent/merge builds failed and 12 built checkouts
failed tests. Execution eligibility remains separate from licensed
training-content eligibility; all executable Stage-C use remains blocked.

The broad tranche progresses through Stage 0/A/B: 23 Stage-0 release
candidates produce 254 Stage-A-pass chunks, including 175 train chunks. Stage
B selects 94 chunks and constructs a disjoint 49-chunk Stage-A-random arm at
99.9744% of the selected token-proxy budget. Stage C must not start from this
tranche: no bundle is executable-evaluation eligible, the train pool is
documentation-dominated and single-bundle-dominated, selected redundancy risk
exceeds random, and the full selector behaves identically to the Quality-only
ablation.

Two additional train repositories now contribute review-only content:
`bytedance/deer-flow` contributes 10 Stage-A-pass chunks and
`scikit-learn/scikit-learn` contributes 3. A frozen `mem0ai/mem0` attempt
yielded no allowed text files and remains recorded as a negative fetch result.
The combined optional diagnostic-review corpus contains 331 chunks across 3
repositories. Build the score-hidden packet with:

```powershell
conda run --no-capture-output -n research python 78_build_temporal_code_stage_b_blind_review.py --scored outputs\temporal_code_collection\proxy_review_expansion_4\combined_review_scored.jsonl
conda run --no-capture-output -n research python 79_analyze_temporal_code_stage_b_blind_review.py
```

The current packet contains 72 records and hides scores, selection arms,
repositories, paths, and sampling strata. It is an optional failure-analysis
diagnostic. Its completion cannot promote or tune Stage B and does not block
automated validation, corpus expansion, or Stage C. Review-only expansion
content is not training-approved and cannot enter Stage C.

Optional multi-review diagnostics support two independently ordered blind
packets plus disagreement-only adjudication:

```powershell
conda run --no-capture-output -n research python 82_build_temporal_code_stage_b_multi_reviewer_packets.py
conda run --no-capture-output -n research python 83_analyze_temporal_code_stage_b_multi_review.py
```

Proxy-review analysis remains blocked until both reviewers complete all 72
labels and every disagreement receives a blind adjudication label. This block
applies only to the optional diagnostic, not the framework execution path.
Use `docs/temporal_code_stage_b_blind_review_guide.md` for the frozen label
definitions and resumable review commands.

Read `docs/metric_evidence_audit.md` for the required distinction between
paper-backed methods, paper-aligned principles, frozen project hypotheses,
and engineering diagnostics.

Latest local verification: `validate_outputs.py` passes `270/270`. This is
contract verification, not a training release claim.

### Fresh path-stratified temporal-code tranche

The first broad tranche exposed a sampling-frame failure rather than a reason
to tune Stage B: its train pool was documentation- and single-bundle-dominated.
A fresh v2 frame was therefore frozen using changed-path metadata only, before
fetching file content or observing Stage outcomes. It permits one PR per
repository and samples `code_and_test` and `code_only` strata across all
splits.

Current fresh-tranche evidence:

- 40 repositories / 40 bundles / 244 fetched file records
- 27 training-content-eligible bundles
- 106 Stage-0 release-candidate records
- 868 Stage-A-pass chunks, including 523 train chunks
- 423 Stage-B-selected chunks and a disjoint 79-chunk Stage-A-random arm at
  99.9634% of the selected token-proxy budget
- train documentation share 1.15%; largest-bundle share 19.89%
- selected redundancy risk is lower than Stage-A random
- full selector differs behaviorally from Quality-only
- executable-evaluation candidates exist in train, development, and untouched
  confirmatory splits

The corpus-side readiness decision is now `ready_for_stage_c_smoke`. This does
not establish target-model benefit. Qwen3-4B equal-token/equal-compute Stage C,
raw-random target-token construction, Utility, retention, and confirmatory
effect evidence remain pending.

The Qwen3-4B Stage-C feasibility smoke was frozen before model execution. Its
three arms use the Qwen3 tokenizer and pack to an exactly matched 139,264
tokens each at sequence length 2048. The selected-disjoint Stage-A-random arm
is the common baseline for every sensitivity comparison.

All three frozen arms completed 8 QLoRA optimizer steps on a physical RTX 3070
Ti with the same seed, packed-token budget, and compute recipe. This establishes
Qwen3-4B Stage-C execution feasibility only. Training-loss ordering is an
execution diagnostic, not Utility, and cannot support selector tuning,
curation-benefit, or release claims. Before development Stage C, the practical
effect margin, development seeds and executable aggregate, and retention
non-inferiority guardrails must be frozen.

The current development executable holdout contains only one verified bundle.
Repeated training seeds cannot turn one task into a representative task
distribution. Before development Utility, an outcome-free expansion plan now
freezes all 11 remaining repository-disjoint development `code_and_test`
candidates. Selection uses path metadata only and forbids file content, test
outcomes, Stage outcomes, Utility, benchmarks, and review labels.

The expansion fetched 11 bundles / 132 file records. Seven bundles passed the
collection gate, but the frozen generic execution hypothesis verified none:
13 commit builds failed and one test execution failed. Development Stage C
therefore remains blocked with only one verified executable bundle in total.
This is execution-infrastructure evidence, not curation or Utility evidence.
The next admissible step is to freeze an outcome-independent repository-native
execution-recipe extraction rule or expand the metadata-only sampling frame.

The repository-native recipe experiment is now complete as a post-failure
exploratory diagnostic. Structured project metadata selected test/dev extras,
project roots, supported Python images, and ephemeral writable workspaces
without reading generic execution outcomes. Build-stage reach improved from 1
to 5 passing commits, but no bundle passed both parent and merge execution.

Further repository-specific recipe tuning on this same development pool is
forbidden because it would overfit execution outcomes. Development Utility
remains blocked. The next admissible choices are a fresh metadata-only
development sampling-frame expansion, a reusable external executable-task
harness, or explicit repository execution-support tiers.

A repository-disjoint fresh development expansion then consumed every
remaining metadata-eligible `test_only` or `code_only` repository without
using prior execution outcomes. Fourteen bundles were fetched and twelve
passed the collection gate. The unchanged native rule improved fresh
build-stage reach from 2 to 4 commits and reached two passing individual test
executions, but recovered zero bundles that passed both parent and merge.

This establishes an architecture boundary: raw collected repositories can be
valid training candidates without automatically being reproducible executable
evaluation tasks. Broadening raw-repository discovery solely to recover
executable tasks is no longer the recommended next step. Stage C needs a
separately prevalidated executable-task harness or explicit repository
execution-support tiers.

Execution-support tiers are now explicit and orthogonal to training-content
eligibility. Across the current 77 audited bundles, 54 are training-content
eligible while only 3 are executable Stage-C eligible. A `C1/E0` or `C1/E1`
bundle may remain a training-data candidate but cannot become an executable
task; only `E2`, verified on both parent and merge commits, may enter
executable Stage C. Execution tier is forbidden from Stage B.

An independent executable-task harness acquisition contract is now frozen.
Harness tasks are evaluation-only and may never enter training. They must be
`E2`, repository- and time-disjoint, contamination-quarantined, and acquired
before model outcomes. The required task count must be computed from a frozen
practical effect margin and desired confidence-interval precision; an arbitrary
minimum task count is forbidden.

The first independent-source feasibility profiles are complete without reading
model outcomes or retaining task prompts, patches, tests, or solutions:

- `E2` is now task-class-specific. Repository-patch tasks require the same
  frozen isolated command to pass parent and merge; function-generation tasks
  require a frozen isolated evaluator whose reference and negative controls
  pass.
- The frozen primary paired executable-success estimand uses a 5 percentage
  point practical margin, five training seeds, and a one-sided 95% task-
  distribution precision rule. Its conservative task-count requirement is
  1,083 tasks.
- SWE-bench Verified metadata profiling yields 49 development and 38 untouched
  confirmatory candidates across disjoint repositories, but only 87 total
  candidates and zero locally E2-prevalidated tasks. It is therefore a
  secondary repository-level guardrail candidate, not a sufficient sole
  primary aggregate.
- EvalPlus 0.3.1 is installed and exposes 164 HumanEval+ and 378 MBPP+ tasks.
  Native Windows remains incompatible with its Unix-only reliability guard,
  but the existing Docker Desktop Linux WSL2 backend now runs the frozen
  evaluator with no network, a read-only root filesystem, and all capabilities
  dropped. Reference controls pass and fixed negative controls are rejected, so
  the evaluator is `E2`.
- EvalPlus is frozen as an external code guardrail: 284 development tasks and
  258 untouched confirmatory tasks, with a 2 percentage point non-inferiority
  margin. It cannot replace the primary temporal executable aggregate.
- General retention guardrails are frozen before model outcomes: per-suite
  general-task regression at most 2 percentage points, macro regression at
  most 1 percentage point, and external general-text NLL increase at most 0.01.
- The primary temporal source assessment correctly abstains. Only two
  project-created development/confirmatory temporal E2 tasks exist versus the
  frozen 1,083-task requirement. SWE-bench Verified and EvalPlus are secondary
  guardrails; the current LiveCodeBench lite snapshot is `n<1K`, was last
  modified on 2025-06-05, and does not demonstrate untouched post-training-
  window confirmatory coverage.

The Windows runtime blocker and external-code E2 guardrail are resolved. The
remaining blocker is acquisition of a sufficiently powered primary temporal
E2 task distribution. This is not a Stage-B, curation-benefit, or curation-
failure result. Development Utility remains blocked and confirmatory outcomes
remain untouched.

### Forward temporal E2 acquisition pilot

The first forward-acquisition infrastructure pilot is complete. It validates
task semantics at collection time: frozen changed test modules must pass on
the merge commit and fail when overlaid on the first parent, while both
environments build successfully. Test-support files such as `conftest.py` do
not establish task validity, and pilot tasks from training repositories are
permanently quarantined from development and confirmatory evaluation.

Observed infrastructure productivity:

- 30 train repositories scanned using metadata only
- 16 metadata candidates frozen before execution
- 5 execution recipes frozen before execution
- 2 task-valid E2 repository-patch tasks
- metadata-to-E2 yield: 12.5%
- execution-to-E2 yield: 40%
- all 3 invalid execution candidates failed on the merge-test side
- pilot tasks authorized for evaluation: 0

At the observed point-estimate yield, acquiring the frozen 1,083-task primary
aggregate would require roughly 8,664 metadata candidates and 2,708 execution
attempts. These are planning-only estimates from a deliberately small
infrastructure pilot, not inferential yield or capacity claims. The result
establishes that forward E2 acquisition is feasible, but development Utility
must still abstain until the frozen primary task-distribution contract is met.

The first actual forward-development snapshot is also frozen and complete. An
authenticated search found 917 current Python repositories; 158 existing broad
manifest repositories were excluded, and a deterministic 200-repository fresh
frame was scanned for merged PRs from `2026-06-15` through `2026-06-15`.
Because this is the first day of the development window, the snapshot produced
zero task candidates. That is valid acquisition evidence: no recipes or E2
executions were fabricated, the same snapshot was not retroactively expanded,
and Utility remains blocked.

Before reading any later snapshot metadata, a higher-capacity accumulation
plan is now frozen using the earlier pilot productivity estimate rather than
the zero-candidate result. It keeps the same eligibility rule and expands
future-date scans without reading task metadata. A capacity audit found that
the original 759-repository frame could produce at most 759 one-per-repository
metadata candidates and only about 94 E2 tasks at the pilot yield, so it was
structurally underpowered.

The same `stars>=20` eligibility query was exhausted to GitHub's supported
depth, then lower-popularity `5..19` and `0..4` star source strata were frozen
before their metadata was read. Popularity is not a protocol eligibility
requirement, and broadening source coverage does not weaken task validity.
The combined metadata-only discovery contains 12,067 repositories. After broad
manifest and benchmark-source exclusion, the future accumulation frame is
frozen at 5,000 fresh repositories with zero overlap.

The 5,000-repository frame exceeds the planning-only 4,336 metadata-candidate
point estimate and therefore resolves the structural repository-frame capacity
blocker. It does not guarantee that repositories will produce eligible tasks:
actual task candidates and E2 tasks remain zero until later-date snapshots are
collected. Development Utility remains blocked.

### Forward collection operations

The 5,000-repository frame is now operational rather than merely documented.
It is deterministically divided into 25 immutable 200-repository shards. Each
later-date shard snapshot is written once, preserves repository-level errors,
and freezes candidate paths and commit identities before project metadata,
recipes, or execution outcomes.

The cumulative candidate ledger:

- deduplicates by repository, pull request, and merge commit
- enforces one primary task per repository
- deterministically keeps the earliest eligible task per repository
- remains frozen before recipe metadata and Docker execution
- never authorizes Utility or confirmatory outcome inspection

Refresh the schedule, ledger, and machine-readable operations status with:

```powershell
conda run --no-capture-output -n research python 125_run_temporal_code_forward_operations.py --action refresh
```

Collect one later-date shard with:

```powershell
conda run --no-capture-output -n research python 125_run_temporal_code_forward_operations.py --action collect --available-through YYYY-MM-DD --shard-index N
```

When candidates exist, `126_freeze_temporal_code_forward_recipe_batch.py`
freezes at most 25 outcome-independent project-metadata recipes. The same
isolated parent-fail/merge-pass verifier then processes the frozen batch.
Currently the operations status is
`forward_collection_operational_waiting_for_later_date_tasks`: 25 shards are
ready, but candidate, recipe, and E2 counts remain zero.

### Retrospective development acquisition

Already-public GitHub pull requests after the Qwen3-4B base release are now
used for development-task acquisition. This avoids waiting for future data
while preserving `2026-09-01..2026-11-30` as an untouched confirmatory window.
Evaluation repositories remain disjoint from the existing training broad
manifest, and the frozen strict parent-fail/merge-pass `E2` rule is unchanged.

Observed retrospective development progress:

- 5,000 disjoint repositories scanned across 25 immutable shards
- 1,666 strict metadata candidates, one earliest task per repository
- first frozen execution batch: 25 attempts / 4 task-valid `E2`
- first-batch `E2` rate: 16%
- failure stages: 8 merge builds, 12 merge tests, 1 non-discriminative parent
- projected current-pool `E2`: 266, versus the frozen development target 542
- Utility remains blocked and confirmatory outcomes remain untouched

The aggregate first-batch result justifies acquisition expansion, not task-rule
weakening. Before reading any additional task metadata,
`131_freeze_temporal_code_retrospective_expansion_schedule.py` froze every
remaining eligible repository from the pre-existing combined metadata-only
frame. Of 7,067 repositories outside the initial 5,000, 245 overlap the
training broad manifest and are excluded. The resulting expansion contains
6,822 training-disjoint repositories in 35 shards with unchanged eligibility,
strict `E2`, and one-task-per-repository rules.

At the first-batch point estimate, scanning the entire disjoint combined frame
would produce roughly 3,939 metadata candidates and 630 valid `E2` tasks.
This is planning evidence only, not an inferential capacity guarantee or a
curation-benefit claim.

`132_build_temporal_code_retrospective_combined_ledger.py` merges initial and
expansion snapshots while enforcing the frozen 11,822-repository disjoint
universe and one task per repository. `133_build_temporal_code_retrospective_operations_status.py`
then exposes the single current gate state. All 60 metadata shards now exist,
3,847 candidates are frozen, 825 have been executed, 167 are task-valid `E2`,
and the actual valid-`E2` gap is 375.

`134_freeze_temporal_code_retrospective_execution_order.py` freezes the
remaining execution order by a fixed hash seed before further E2 outcomes.
`135_build_temporal_code_retrospective_e2_capacity_audit.py` applies a
one-sided 95% Wilson upper-bound stopping rule. Current observed strict-E2
rate is 20.24%, with a point-estimate full-frame expectation of about 779 valid
E2 tasks and a conservative projected upper total of 851. The stopping audit
therefore says strict E2 execution can continue, but E2 is now a secondary
executable-evidence track rather than the main blocker for the raw-vs-curated
training validation.
