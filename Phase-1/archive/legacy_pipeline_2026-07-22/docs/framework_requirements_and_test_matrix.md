# Framework Requirements and Test Matrix

## 1. Document Status

This document is the canonical requirements, ownership, and validation contract for the training-data curation framework. `docs/lm_curation_operational_framework.md` and `configs/lm_curation_operational_framework_v1.json` define the current practical target: an operational curation layer that emits a supported training release or an explicit abstention.

The operational execution boundary is additionally fixed in
`docs/operational_curation_boundary_v2.md` and
`configs/operational_curation_contract_v2.json`: Stage 0-A-B materializes the
curated output, and Stage C is external offline validation that cannot mutate
or reselect a frozen output. A same-token random arm is optional research
control evidence, not a runtime curation requirement.

Use it to decide:

- what the framework must do
- which pipeline boundary owns each problem
- what evidence is required before a claim is allowed
- which experiments must be completed next

`docs/research_framing.md` defines the research motivation and claim boundary. This document turns that framing into testable requirements.
The active 30-day execution plan is `docs/30_day_paper_sprint_plan.md`, and
the production-readiness gate boundary is
`docs/production_readiness_gate_spec.md`.

## 2. Mission and Claim Boundary

The framework receives a candidate corpus produced by an upstream collection process and decides what should be used for language-model training.

```text
candidate corpus + Deployment Contract
-> full curated pool
-> optional budgeted training subset
-> supported training release or explicit abstention
```

The framework must support:

- from-scratch pretraining corpus construction
- continued pretraining on newly collected data
- domain adaptation
- periodic dataset refresh

The framework may claim that it can produce a supported training-use decision from a candidate corpus. It must not claim that every candidate corpus can be transformed into a useful training dataset.

The framework must not claim to measure intrinsic data quality. It operationalizes pre-outcome selection proxies and then tests whether those proxies produce downstream training benefit under Stage-C protocols.

The framework must not assume that curation reduces corpus size. Every
non-quarantined Stage-A pass belongs to the full curated pool. Stage B is
activated as competitive selection only when a declared training budget is
smaller than that pool.

The Deployment Contract may declare an optional domain or capability mixture
target and allowed drift. If no target mixture is declared, the framework must
report observed raw-vs-curated composition only and must not infer a universal
optimal ratio.

A valid outcome can therefore be:

```text
insufficient usable data
```

The following claim boundaries are mandatory:

- Utility is evaluated only in Stage C.
- Utility must never be a Stage-B selector objective.
- Dataset-specific Stage-C diagnostics must not become dataset-specific selector rules without an independent Core-level justification.
- A positive result on an already curated dataset does not prove raw-corpus operation.
- A Stage-C failure is evidence to diagnose, not permission to tune until the result becomes positive.
- Human or LLM review is optional diagnostic evidence and must not approve,
  reject, tune, or block Stage B or Stage-C entry.
- Literature citations support principles or reproduced methods only; they do
  not validate project-specific weights, thresholds, or implementations.

### Current Evidence Status

The current evidence supports a bounded deployment-conditioned curation claim,
not a universal data-quality claim.

- Code natural-budget validation is a historical positive case: curated v2 reduces
  packed training tokens by 60.8%, improves heldout NLL from `1.210000` to
  `1.201043`, and improves the paper-facing EvalPlus macro pass rate from
  `51.06%` to `57.87%` under the same natural-budget protocol. These are
  historical positive results produced before the current Stage-A implementation
  fingerprint and require a current-framework rerun before confirmatory use.
- Math natural-budget validation is `abstain`: selector v2 over-filtered and
  worsened heldout NLL from `1.495650` to `1.527065`; selector v3 restores the
  proof/theorem token mass that v2 over-dropped and repairs the NLL to
  `1.498987`, but it still does not beat raw. Its token blocks are frozen at
  `1,026,048` packed training tokens versus raw `1,120,256`.
- Math remains missing GSM8K/MATH benchmark guardrails, so v3 is a repair-only
  result, not a Math success claim.
- Production release remains blocked until required guardrails are validated.
- Forbidden claim: universal data-quality detector or all-domain improvement
  guarantee.
- Allowed claim: a Core-Metric-Policy curation-control framework that emits
  accept, reject, or abstain decisions under frozen Stage-C validation.
- Machine-checkable claim consistency is frozen in
  `configs/paper_claim_consistency_contract_v1.json` and audited by
  `218_build_paper_claim_consistency_audit.py`.
- Domain composition evidence is frozen in
  `configs/domain_mix_contract_v1.json` and audited by
  `219_build_domain_composition_audit.py`. Current Block-2 evidence reports
  paper-domain arms, not a joint production corpus: raw Code/Math packed-token
  shares are `46.69%`/`53.31%`, and current curated-arm shares are
  `27.29%`/`72.71%`.
- Coverage/domain-mix scope is frozen in
  `configs/coverage_domain_mix_contract_v1.json` and audited by
  `220_build_coverage_domain_mix_audit.py`. Current Block-3 evidence passes
  with a scope boundary: observed composition drift is reportable, but target
  mix satisfaction is not claimable because no target mix is declared.
- Stage-B policy scope is frozen in
  `configs/stage_b_policy_contract_v1.json` and audited by
  `221_build_stage_b_policy_contract_audit.py`. Current Block-4 evidence
  passes: Stage B is optional budget allocation, `retain_all` is valid when no
  budget binds, and `budget_not_selected` is retained data rather than
  rejection or a low-quality label.
- Canonical paper-evidence execution is frozen in
  `configs/canonical_execution_path_v1.json` and audited by
  `222_build_canonical_execution_registry.py`. Current Block-5 evidence
  identifies 7 lightweight canonical rebuild scripts and separates 215
  historical/experimental numbered scripts from the paper-evidence path.

## 3. Pipeline Boundaries

### Upstream Collection

Upstream collection discovers, fetches, and stores candidate documents. It owns source acquisition and crawl scheduling. It is outside the main curation claim, but it must provide provenance and legal metadata when available.

### Stage 0: Ingestion, Normalization, and Quarantine Boundary

Stage 0 converts source documents into auditable candidate chunks before Core scoring.

It owns:

- parsing and text extraction
- document-to-chunk conversion
- encoding normalization
- provenance retention
- language identification and routing
- PII, secret, licensing, and policy quarantine
- benchmark-contamination quarantine
- adversarial or poisoning quarantine

Stage 0 is a required real-world interface boundary. The repository now has a
versioned candidate-record contract, fixture validator, and bounded
temporal-code adapter. Repository-code normalization must preserve source
layout and operators exactly except for line-ending normalization; generic
prose normalization is forbidden for code payloads. Hazard detectors and
external production-detector scientific validation remains incomplete. Current
real-corpus Stage-0/Coverage metadata lineage is audited by
`169_build_real_corpus_stage0_coverage_audit.py`, and current labeled detector
precheck coverage is audited by `170_build_stage0_detector_validation.py`.

### Stage A: Chunk-Level Hard Gate

Stage A answers:

```text
Can this chunk be used at all?
```

It owns structural unusability, raw or canonical-content exact duplicates, and pathological repetition. Fuzzy near-duplicate evidence remains reversible Stage-B evidence until independent precision and useful-data-dropout gates pass. Stage A must not judge semantic usefulness or Utility.

For temporal Python change bundles, Stage A uses domain-specific Core
implementations while preserving the same ownership boundary:

- Python chunks use top-level AST boundaries and must parse independently.
- Documentation uses paragraph-group boundaries.
- Raw and canonical-content exact duplicate decisions are split-local.
- SimHash and token-shingle overlap produce fuzzy near-duplicate evidence for
  Stage B; they do not authorize Stage-A rejection under the current holdout.
- Cross-split duplicate checks are diagnostic only and cannot change
  train/development/confirmatory decisions.

### Stage B: Chunk-Level Selection

Stage B answers:

```text
If a binding budget exists, which usable chunks should receive that budget?
```

It owns optional budget allocation using Selection Value Evidence, soft
Redundancy risk, useful recurrence, length support, and coverage-preserving
support. `Quality` is only a legacy alias. Utility is prohibited from the
objective. When no binding budget exists, Stage B must emit `retain_all`.
Records outside a budgeted subset remain in the full curated pool and are
marked `budget_not_selected`, never rejected or low quality.

### Stage C: Subset-Level Validation

Stage C answers:

```text
Is the selected subset supported for the intended training use?
```

It owns Coverage retention, fixed-budget Utility, fair counterfactual baselines, probe validity, transfer checks, forgetting checks, and statistical robustness.

Every Utility sensitivity arm must share the same Stage-A baseline pool, and that pool must be disjoint from the union of all sensitivity arms.

Expected policy:

```text
common_stageA_baseline_disjoint_from_all_sensitivity_arms
```

### Decision and Release Layer

The decision layer converts Stage A/B/C evidence into an operational action.
The release layer then applies a predeclared Deployment Contract specifying the
target model, budget, objective, primary evaluation distribution, and
guardrails. It must not force a selected dataset to be released or treat one
release as universally best.

Required actions:

- `accept`
- `accept_with_caveat`
- `retain_all`
- `full_curated_pool`
- `budgeted_training_subset`
- `cap_or_downsample`
- `route_to_specialized_pool`
- `manual_review`
- `quarantine`
- `reject`
- `insufficient_usable_data`
- `selected_only`
- `coverage_backfilled`
- `stageA_broad`

`27_build_curation_decision_report.py` implements selected-subset training-use
decisions. `48_build_release_decision_report.py` applies a Deployment Contract
and emits `selected_only`, `coverage_backfilled`, `stageA_broad`, `reject`, or
`insufficient_usable_data`. Routing, capping, and quarantine remain
Stage-0/decision-layer follow-ups.

## 4. Responsibility Matrix

| Problem or decision | Primary owner | Required handling |
| --- | --- | --- |
| Fetch failures and source discovery | Upstream collection | Log, retry, or exclude before curation |
| Broken extraction, HTML residue, boilerplate segmentation | Stage 0 | Normalize, re-extract, or quarantine |
| Language mismatch or mixed-language routing | Stage 0 | Route or quarantine with retained metadata |
| PII, secrets, licensing restrictions | Stage 0 | Quarantine or reject before Core scoring |
| Benchmark contamination | Stage 0 | Quarantine and report contamination evidence |
| Poisoning or adversarial payloads | Stage 0 | Quarantine and require review |
| Empty, corrupted, symbol-heavy, non-language chunks | Stage A | Hard reject with reason |
| Raw and canonical-content exact duplicates | Stage A | Hard reject with duplicate lineage |
| Fuzzy near-duplicates | Stage B | Penalize or cap under a binding budget; never irreversible rejection under current evidence |
| Observable information density, structural usefulness, boilerplate risk | Stage B | Selection Value Evidence for optional budget allocation; no hard rejection |
| Soft redundancy and corpus saturation | Stage B | Penalize, cap, or downsample |
| Useful recurrence and rare valuable material | Stage B | Preserve when Core evidence supports it |
| Source, style, cluster, or explicit-domain balance | Stage B and Stage C | Support during selection; validate after selection |
| Training usefulness | Stage C | Fixed-budget Utility validation only |
| Probe instability, baseline confounding, token exposure | Stage C | Diagnose and report; do not silently tune selector |
| Target-model fit, transfer, and forgetting | Stage C | Validate for the intended training claim |
| Final release, route, quarantine, reject, or abstain | Decision layer | Emit explicit action, rationale, and caveats |
| Objective-specific training release | Release layer | Apply a frozen Deployment Contract; never feed Stage-C outcomes into Stage B |

## 5. Functional Requirements

### Implementation Placement

New behavior should be implemented at the owning boundary below. Do not place
real-world ingestion hazards into Stage-B scoring, and do not place Stage-C
Utility outcomes into Stage-B policy.

| Boundary | Current or planned implementation location | Validation location |
| --- | --- | --- |
| Stage 0 candidate schema and provenance | `ingestion/schema.py`; versioned candidate-record contract | `29_validate_stage0_contract.py` and `validation/fixtures/stage0_candidate_records.json` |
| Stage 0 normalization and quarantine | `ingestion/normalize.py` and `30_process_stage0_candidates.py`; external detector benchmark remains planned | `validation/fixtures/stage0_raw_candidates.json`, `166_build_stage0_hazard_benchmark.py`, `169_build_real_corpus_stage0_coverage_audit.py`, and `170_build_stage0_detector_validation.py` |
| Stage A structural and duplicate gates | Generic: `signals/core.py`, `03_score_core_metrics.py`, and Stage-A logic in `policy/subsets.py`; temporal code: `ingestion/code_chunks.py` and `74_run_temporal_code_stage_a_smoke.py` | Generic: `07_run_property_benchmarks.py`, `validate_outputs.py`, and rejection audits; temporal code: `validation/test_temporal_code_chunking.py` and `validation/test_temporal_code_stage_a_smoke.py` |
| Stage B selection policy | Generic: `policy/subsets.py` and profile configs under `configs/`; temporal code: frozen contract in `configs/temporal_code_curation_protocol_v1.json`, `ingestion/code_selection.py`, and `75_run_temporal_code_stage_b_smoke.py` | Generic: `15_run_selector_baseline_audit.py`, `16_run_good_chunk_dropout_audit.py`, and `17_run_policy_ablation_audit.py`; temporal code: `validation/test_temporal_code_stage_b.py`, `validation/test_temporal_code_stage_b_smoke.py`, and `validate_outputs.py` |
| Stage C Coverage and Utility | Subset validation in `policy/subsets.py`; diagnostics in `14_run_utility_causal_diagnostics.py`, `19_run_utility_probe_power_sweep.py`, and `22_run_anti_memorization_probe.py` | `20` through `26` reports and `validate_outputs.py` |
| Decision and release layer | `27_build_curation_decision_report.py` | `28_build_paper_evidence_table.py` and `validate_outputs.py` |
| Deployment Contract and release policy | `release_policy.py`, `48_build_release_decision_report.py`, and `configs/deployment_contract_*.json` | `validation/test_release_policy_contract.py` and generated release-decision reports |
| End-to-end target-SLM experiment | `34_prepare_slm_update_experiment.py`, `35_freeze_slm_update_plan.py`, `configs/slm_update_qwen25_0p5b_experiment.json`, and `docs/slm_update_experiment_preregistration.md`; do not overload the selector | SLM-update manifest and frozen-plan checks in `validate_outputs.py`; planned pre-registered G4 result report |
| Temporal code raw-corpus experiment | `configs/temporal_code_curation_protocol_v1.json`, `docs/temporal_code_curation_preregistration.md`, collectors `63` through `73`, and syntax-aware Stage A in `ingestion/code_chunks.py` / `74_run_temporal_code_stage_a_smoke.py`; code-domain Stage B remains planned | Temporal-code validation suite; bounded collection, Stage-0, and Stage-A reports; future Stage-B, executable-task, and 4B result reports |

| ID | Requirement | Verification |
| --- | --- | --- |
| FR-001 | Preserve the Core-Metric-Policy contract and record which metric and policy produced each decision. | Config validation and output-manifest checks |
| FR-002 | Preserve Stage A/B/C role separation. | Property tests and release guardrails |
| FR-003 | Keep Utility out of all Stage-B selector objectives and canonical Core scoring artifacts. | Config/code validation, `164_build_selector_utility_leakage_audit.py`, and `168_build_scoring_schema_separation_audit.py` |
| FR-004 | Retain source and transformation provenance from ingestion through final decision. | Lineage audit; currently incomplete for Stage 0 |
| FR-005 | Emit reason-coded Stage-A rejection decisions. | Stage-A rejection audit |
| FR-006 | Compare Stage B against Stage-A random and relevant non-Utility baselines. | Selector baseline audit |
| FR-007 | Validate subset Coverage without overclaiming domain or domain-mix coverage when explicit metadata or a declared mix contract is absent. | Coverage report, `167_build_coverage_domain_fixture_benchmark.py`, `169_build_real_corpus_stage0_coverage_audit.py`, `219_build_domain_composition_audit.py`, `220_build_coverage_domain_mix_audit.py`, and certification guard |
| FR-008 | Validate Stage-B policy semantics so optional budget allocation cannot be mistaken for quality rejection or mandatory corpus reduction. | `213_build_record_disposition_audit_report.py`, `221_build_stage_b_policy_contract_audit.py`, and selector Utility leakage audit |
| FR-008 | Use a common disjoint Stage-A baseline for every Utility sensitivity arm. | Utility protocol validation |
| FR-009 | Separate selected-vs-random curation benefit from selected-vs-matched strict counterfactual benefit. | Stage-C evidence report |
| FR-010 | Record CI, MDE, replication, probe validity, and caveats for Utility claims. | Stage-C protocol and strict-control reports |
| FR-011 | Emit an explicit operational decision, including abstention when usable data is insufficient. | Decision report; routing, capping, and quarantine actions incomplete |
| FR-012 | Prevent stale or profile-mismatched evidence from supporting a decision. | Cross-report validation |
| FR-013 | Support equal-token and equal-compute comparisons for target-model validation. | End-to-end experiment; not yet completed |
| FR-014 | Detect and report benchmark contamination, PII, licensing risk, and poisoning risk before release. | Stage-0 fixture benchmark, labeled detector precheck, and real-corpus lineage audit exist; external production detector benchmark remains incomplete |
| FR-015 | Require a Deployment Contract declaring the objective, primary evaluation distribution, comparator, and guardrails before choosing a training release. | Release-policy contract test and release-decision report |
| FR-016 | Allow the same Stage-C evidence to yield different scoped releases for different predeclared objectives without changing Stage B. | Broad-refresh and targeted-update regression fixtures |
| FR-017 | Reject or abstain when no candidate release satisfies required primary and guardrail evidence. | Release-policy contract test |
| FR-018 | For temporal raw-code experiments, freeze time and repository-disjoint splits before Core scoring and exclude benchmark sources and near-duplicates from training. | Temporal-code protocol test and future quarantine manifest |
| FR-019 | Do not assume that repository code licenses authorize issue or pull-request prose for training. | Stage-0 content-type authorization audit |
| FR-020 | Preserve repository-code source layout and operators through Stage 0; code payloads must not use generic prose normalization. | `validation/test_stage0_processing.py`, temporal-code Stage-0 adapter test, and Stage-A parseability report |
| FR-021 | For temporal code, use syntax-aware Stage-A chunks and make duplicate decisions independently within each frozen split. Only local-gate-pass chunks may become duplicate representatives; representative selection must be deterministic. Cross-split duplicate observations are diagnostic only. | `validation/test_temporal_code_chunking.py`, `validation/test_stage_a_duplicate_representative.py`, `validation/test_temporal_code_stage_a_smoke.py`, and bounded Stage-A report |
| FR-022 | Treat Quality as a legacy label for a pre-outcome selection-value proxy rather than intrinsic ground-truth quality, and report this boundary in framework contracts. | `configs/lm_curation_operational_framework_v1.json`, `163_build_core_construct_validity_review.py`, and `validation/test_lm_curation_operational_framework.py` |
| FR-023 | Separate harmful duplication from useful recurrence before claiming operational readiness. | Stage-B feature-shift reports and operational framework contract |
| FR-024 | Preserve concise but useful examples, tests, bug fixes, and API-usage chunks when Core evidence supports them. | Code-domain Stage-B feature reports and future selector ablations |
| FR-025 | Emit `abstain` when required Stage-C primary or guardrail evidence is missing. | Operational framework contract and decision reports |
| FR-026 | Preserve every non-quarantined Stage-A pass in the full curated pool. | Disposition contract and retain-all fixtures |
| FR-027 | Run competitive Stage-B selection only under a binding declared budget. | Stage-B selection-mode tests |
| FR-028 | Keep `budget_not_selected` distinct from `rejected` and `quarantined`. | Per-record disposition checks |
| FR-029 | Permit `retain_all` when the full curated pool fits the declared budget. | All-high-value corpus fixture |
| FR-030 | Separate the canonical paper-evidence rebuild path from historical experiments, raw-data acquisition, GPU training, and production release execution. | `configs/canonical_execution_path_v1.json`, `222_build_canonical_execution_registry.py`, and `validation/test_canonical_execution_registry.py` |
| FR-031 | Prohibit fixed rejection quotas and target reduction ratios. | Framework contract and policy audit |

## 6. Non-Functional Requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| NFR-001 | Reproducibility | Config, seed, model, dataset identity, and output manifests are recorded |
| NFR-002 | Auditability | Every rejection, selection, caveat, and release decision has traceable evidence |
| NFR-003 | Profile isolation | Evidence from one profile cannot certify another profile |
| NFR-004 | Canonical execution clarity | Current paper evidence can be rebuilt from an explicit lightweight script registry without implying full data regeneration or production release readiness |
| NFR-004 | Failure transparency | Failed and inconclusive cases remain visible in reports |
| NFR-005 | Resource comparability | Primary training comparisons use matched tokens and compute |
| NFR-006 | Statistical defensibility | Certification claims require replicated evidence and positive confidence criteria |
| NFR-007 | Operational safety | Quarantined content cannot enter training subsets by default |

## 7. Operational Use-Case Test Matrix

The test corpus for each row must include labeled fixtures and an expected decision. Passing only clean public datasets is not sufficient.

| ID | Candidate-corpus condition | Primary owner | Expected outcome | Minimum pass criterion | Current evidence |
| --- | --- | --- | --- | --- | --- |
| UC-01 | Clean, diverse, useful corpus | Stage B/C | Accept supported subset | Coverage and Utility certification criteria pass | FineWeb-Edu positive case; already curated input |
| UC-02 | Mixed clean text and broken extraction/markup | Stage 0/A | Repair or reject broken chunks; retain clean chunks | High recall on clean chunks and high rejection precision on broken chunks | Partial structural fixtures only |
| UC-03 | Duplicate dump or template-dominated corpus | Stage A/B | Deduplicate, cap saturation, preserve useful recurrence | Duplicate burden decreases without excessive valuable-chunk dropout | Partial Redundancy and dropout audits |
| UC-04 | Mostly low-quality but structurally valid corpus | Stage B/C | Select supported minority or abstain | No forced acceptance; decision reflects insufficient evidence/data | OpenWebText2 is a partial raw-like stress case |
| UC-05 | Narrow high-quality domain corpus | Stage B/C | Route to specialized pool or accept with scope caveat | No general-purpose claim without broad Coverage evidence | Not yet tested end to end |
| UC-06 | Multilingual or language-mismatched corpus | Stage 0/B/C | Route by language or quarantine unsupported portions | Language routing is auditable and target scope is explicit | Not yet implemented |
| UC-07 | Rare but valuable technical material mixed with common text | Stage B/C | Preserve valuable tail under budget | Tail retention improves without invalidating Utility/Coverage | Partial Coverage support; domain/source fixture benchmark passes 5/5 and current real-corpus repository/content/path metadata audit passes with explicit-domain caveat |
| UC-08 | PII, secrets, restricted-license, or benchmark-contaminated content | Stage 0 | Quarantine or reject | Zero known labeled hazards in released subset | Labeled fixture benchmark passes 10/10; detector precheck passes 13/13 with per-axis precision/recall 1.0; current real-corpus Stage-0 lineage audit passes with external-benchmark caveat |
| UC-09 | Adversarial, poisoned, or instruction-injection-like payloads | Stage 0/A/C | Quarantine and flag for review | High detection on labeled attacks; no default release | Not yet implemented |
| UC-10 | Candidate corpus with too little usable data | Decision layer | `insufficient_usable_data` | Framework abstains instead of forcing a training-use claim | Implemented with `validation/test_decision_contracts.py`; full input-to-report fixture pending |
| UC-11 | Entire corpus is usable and high-value, budget is sufficient | Stage A/B | Full pool retained; `retain_all` | 100% of Stage-A-pass records remain retained; no budget exclusions | Implemented in temporal-code Stage-B and Core behavior fixtures |
| UC-12 | Entire corpus is usable and high-value, budget is constrained | Stage B | Budgeted subset plus retained remainder | Unselected records are `budget_not_selected`, never rejected | Implemented in temporal-code Stage-B and Core behavior fixtures |
| UC-13 | Mostly usable corpus with a few hard failures | Stage A | Reject only reason-coded failures | High usable-data retention and no target rejection rate | Fixture expansion required |
| UC-14 | Novel format with uncertain selection evidence | Stage B | Retain or route; do not hard reject | Low selection score alone cannot remove it from curated pool | Fixture expansion required |
| UC-15 | Safety, rights, or contamination uncertainty | Stage 0 | Quarantine | Uncertainty does not become silent rejection or release | Partial detector fixtures |
| UC-16 | Duplicate-heavy but otherwise valuable corpus | Stage A/B | Remove duplicate copies while preserving deterministic eligible representatives and lineage | Duplicate burden falls without deleting all recurring patterns; saturation magnitude changes soft evidence | `173_build_redundancy_validity_benchmark.py`, deterministic representative regression, and future real-corpus calibration |

## 8. Scientific Validation Matrix

| Evidence level | Question answered | Required comparison | Claim allowed |
| --- | --- | --- | --- |
| Engineering correctness | Does the pipeline execute and preserve contracts? | Fixtures, schema checks, cross-report checks | Implementation works as specified |
| Automated metric behavior | Do Core metrics satisfy frozen metamorphic, invariance, destructive-control, and isolation contracts? | Synthetic transformations, executable controls, and ablations | Metric is usable as a frozen experimental policy component |
| Stage-B budget allocation behavior | Under a binding budget, does selected differ from retained budget-not-selected data as intended without false rejection semantics? | Selected vs budget-not-selected and Stage-A random | Selector changes measured pre-outcome evidence under this budget |
| Stage-C curation benefit | Does selected train better than random usable data? | Selected vs Stage-A random, equal budget | Curation benefit for this setting |
| Conditional mechanism effect | Does selected beat an alternative after conditioning on declared variables? | Selected vs exact/multi-matched Stage-A controls with common-support reporting | Conditional benefit for this setting; not the total curation effect |
| Operational counterfactual candidate | Does selected beat a control matched on easy-NLL nuisance factors without removing selector target gains? | Selected vs exact nuisance-matched Stage-A baseline | Diagnostic candidate only until replicated certification evidence |
| Protocol robustness | Is Utility evidence stable and unconfounded? | Common disjoint baseline, controls, repeats, CI/MDE | Certification-grade Stage-C evidence |
| Target-model end-to-end benefit | Does the curated corpus improve the intended SLM update? | Curated vs Stage-A random usable data, equal tokens/compute, multiple seeds; raw random and all-data arms as supporting references | Practical continued-pretraining claim |
| Cross-corpus generality | Does behavior hold across raw corpus conditions? | Operational use-case matrix | Broader framework claim |

Passing a lower evidence level never implies passing a higher one.

## 9. Current Evidence and Interpretation

| Dataset | Role in current evidence | Result | What it establishes | What it does not establish |
| --- | --- | --- | --- | --- |
| `fineweb_edu_sample` | Clean positive demonstration | Accepted; certification candidate | The framework can identify a subset supported by current Stage-C evidence | Raw-corpus transformation or universal benefit |
| `openwebtext2_subset` | Most raw-like current stress case | Rejected under paper-release profile | The framework can expose a Core-to-Utility gap and refuse training use | That raw mixed-quality data can yet be curated successfully |
| `tiny_textbooks` | Synthetic/template-heavy stress case | Rejected; token-exposure concerns observed in diagnostics | Utility controls reveal synthetic/token artifacts | General raw-web behavior |
| `wikitext103_subset` | Processed reference stress case | Rejected; prior strict-control diagnostics required care | Strict baseline design matters | Raw-corpus operation |

The latest locally generated validation report is the authority for the
current pass count. A full validation pass means the available implementation
and evidence contracts are internally consistent; it is not equivalent to all
scientific or operational requirements being satisfied.

The latest same-condition certification audit demonstrates that Utility
baseline choice changes the estimated effect even when the selected subset,
probe budget, seeds, and holdouts are fixed. FineWeb-Edu is positive against
canonical and anti-memorization controls but negative against the exact
nuisance control. OpenWebText2 is negative against canonical and nuisance
controls but positive against the anti-memorization control. No baseline is
promoted from this result. See `docs/utility_baseline_comparison.md`.

The follow-up matching decomposition identifies Quality conditioning as the
consistent sign-change point on both FineWeb-Edu and OpenWebText2. It also
shows that restrictive matching can sharply reduce common support. Because
Quality, redundancy, and distributional composition are partly outputs of
Stage B, their matched controls are conditional mechanism diagnostics rather
than substitutes for the primary total-effect comparison against Stage-A
random.

## 10. Primary End-to-End Experiment

The next decisive experiment should test the intended deployment scenario directly.

### Setup

- Choose a fixed 2024-era small language model checkpoint.
- Collect a provenance-rich raw 2025 candidate corpus through the upstream collection process.
- Freeze the framework configuration before target-model results are observed.
- Produce the framework-selected subset and all decision/caveat reports.

### Required Arms

| Arm | Purpose |
| --- | --- |
| Base checkpoint, no update | Measures update benefit and forgetting reference |
| Stage-A random equal-budget subset | Primary equal-budget operational baseline |
| Framework-selected subset | Primary treatment |
| Raw-random equal-budget subset | Raw/unfiltered stress baseline |
| Stage-A-all reference, when compute permits | Separates hard cleaning from Stage-B selection at larger budget |
| Raw-all reference, when compute permits | Measures value and cost of using all candidate data |
| Quality-only selector | Stage-B ablation |
| No-Coverage-support selector | Coverage ablation |

All primary training arms must use equal training-token and compute budgets. Use at least three seeds for the primary Stage-A-random versus Framework-selected comparison. Raw-random and all-data arms are supporting references unless separately pre-registered as primary.

### Required Evaluation

- held-out evaluation from the new-data distribution
- general capability benchmarks
- forgetting or regression on the base model's prior capabilities
- benchmark-contamination audit
- domain and source-slice analysis
- training stability and seed variance
- cost and retained-token efficiency

### Primary Success Criterion

The framework-selected arm must outperform the Stage-A-random equal-budget arm under equal tokens and compute on the pre-registered primary new-data evaluation, with replicated positive evidence, while staying within the pre-registered forgetting and safety limits.

Failure or abstention remains publishable evidence when the framework correctly identifies unsupported candidate data.

## 11. Milestones and Goals

### G0: Lock the Contract

Goal: Treat this document as the canonical requirement and test contract.

Done when:

- requirements and ownership are reviewed
- test IDs are referenced by future issues, code, and reports
- claim boundaries are unchanged without an explicit research decision

### G1: Establish the Real-World Input Boundary

Goal: Define and validate Stage 0 without expanding the framework into a crawler.

Done when:

- candidate document/chunk schema and provenance contract are defined
- quarantine decisions exist for PII, licensing, contamination, and poisoning
- UC-02, UC-06, UC-08, and UC-09 have labeled fixtures and reports

### G2: Explain the Raw-Like Failure

Goal: Determine why OpenWebText2 Core improvements do not transfer to Utility.

Done when:

- probe/protocol instability is separated from selector mismatch
- useful versus harmful retained/rejected slices are quantitatively audited;
  manual review may be reported only as an optional diagnostic
- the result identifies a framework-level action or a defensible abstention

### G3: Prove Raw-Corpus Decision Behavior

Goal: Run the framework on a newly collected, provenance-rich raw corpus.

Done when:

- all operational decisions are available
- the framework can accept, route, quarantine, reject, or abstain
- no fixed selection ratio forces an unsupported release

### G4: Validate the Intended SLM Update

Goal: Complete the equal-budget 2024-checkpoint plus 2025-data continued-pretraining experiment.

Done when:

- required arms and at least three primary seeds complete
- contamination and forgetting checks complete
- results support either a scoped benefit claim or an explicit negative finding

### G5: Freeze the Paper Claim

Goal: Align the paper claim exactly with completed evidence.

Minimum defensible claim today:

```text
The framework provides a reproducible, stage-separated curation and abstention
process that can identify supported and unsupported LM-training subsets under
reported Stage-C evidence.
```

Target claim after G4:

```text
For a pre-registered raw-corpus continued-pretraining setting, the framework
produces a curated subset that improves an SLM update over an equal-budget
random usable-data baseline while satisfying coverage, safety, and forgetting
constraints.
```

## 12. Immediate Execution Order

The evidence basis and detailed exit criteria are defined in
`docs/literature_grounded_curation_direction.md`. The canonical order is:

1. Freeze claim boundaries and classify every binding method, threshold, and
   weight by evidence class.
2. Repair irreversible Stage-0 and Stage-A boundaries, including duplicate
   representative selection and labeled false-positive/false-negative audits.
3. Calibrate Redundancy on exact, near-duplicate, related-but-useful,
   templated, saturated, and independent fixtures.
4. Keep fuzzy near-duplicate evidence out of Stage-A rejection until a candidate
   passes repository-disjoint precision, dropout, and cluster-level retention gates.
5. Rebuild Stage B as a small preregistered family of auditable policy arms,
   with Selection Value Evidence components and separate Coverage retention
   and target-alignment signals.
6. Screen policy arms using fixed proxy-scale training recipes without reading
   confirmatory outcomes.
7. Run the decisive equal-token/equal-compute Qwen3-4B continued-pretraining
   comparison against a common disjoint Stage-A-random baseline, with
   raw-random and Core ablations as supporting arms.
8. Complete EvalPlus or equivalent target-task evidence, general-task and
   general-text retention, forgetting, contamination, and seed-stability
   guardrails.
9. Run untouched repository/time confirmation without changing the frozen
   policy.
10. Freeze the paper claim from the completed evidence, including negative or
   abstention outcomes.

Current bounded-smoke boundary after corrected benchmark quarantine:

```text
3 content-eligible bundles
-> 4 Stage-0 records
-> 34 syntax-aware chunks
-> 23 Stage-A-pass chunks
-> 0 train Stage-A-pass chunks
-> Stage B: insufficient_usable_data
```

The bounded smoke correctly abstains rather than forcing selection from an
empty train pool. The active broad-tranche Stage-B evidence contains 175 train
Stage-A-pass chunks, 94 selected chunks, and a 49-chunk selected-disjoint
Stage-A-random arm at 99.9744% of the selected token-proxy budget. Indexed
exact redundancy search matches all-pairs decisions exactly on that broad
train pool. Stage C remains blocked: the pool is documentation-dominated and
single-bundle-dominated, no bundle is executable-evaluation eligible, selected
redundancy risk exceeds random, and the full selector behaves identically to
the Quality-only ablation.
