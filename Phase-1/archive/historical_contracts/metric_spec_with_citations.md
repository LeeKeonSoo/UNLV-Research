# Archived Metric Spec With Citations

> **Archive-only historical evidence.** This file documents the archived
> five-axis metric design and is not an active runtime contract. The active
> framework uses four Cores: Validity, Redundancy, Coverage, and Quality;
> Utility, NLL, and benchmarks are external evaluation only. See
> `docs/current_curation_framework.md` and `configs/curation_contract.json`.

This document is the human-readable companion to `metric_spec_with_citations.json`. The JSON file is the canonical contract.

## Core-Metric-Policy Structure

Canonical axes are `Validity`, `Selection Value Evidence`, `Redundancy`,
`Coverage`, and `Utility`. Any older `Quality` label below is a legacy field
or artifact alias for Selection Value Evidence.

```text
Core 5
├─ Validity: structural_validity_gate as Stage-A hard gate; structural_validity_score as diagnostic audit
├─ Quality: legacy axis; style/length-normalized reference_quality_score as Stage-B selection-value signal
├─ Redundancy: exact/near duplicate gates plus harmful-redundancy risk penalty
├─ Coverage: source, style, and semantic coverage support as Stage-C validators
└─ Utility: small_lm_probe_gain_score as Stage-C outcome validator only
```

Policy separation:

- Stage A removes hard-invalid or duplicate chunks.
- Stage B allocates a binding budget using Selection Value Evidence and Redundancy risk.
- Stage-A-pass records remain in the full curated pool even when they are budget-not-selected.
- When no binding budget exists, `retain_all` is a valid result.
- Selector preservation constraints protect rare clusters plus source/domain and style buckets.
- Stage C validates subset-level Coverage and Utility.
- Utility is not used in the canonical selector objective.
- `predictive_utility_proxy` and `fixed_token_probe_gain_score` are diagnostic/deprecated and do not determine canonical pass/fail.
- Domain coverage is claimed only when explicit domain metadata exists; otherwise the framework reports source-bucket fallback support.

## Metrics

### Validity

#### `structural_validity_gate`

- Role: `gate`
- Status: `paper_aligned`
- Claim: Binary hard-usability gate for text units that are decodable, non-corrupted, and structurally usable for language-model training.
- Implementation: `signals/core.py` / `CoreMetricScorer.structural_validity_gate`
- Definition: Pass/fail gate over hard structural invalidity only: empty or too-short units, encoding/control-character corruption, non-language fragments, excessive symbol noise, markup/extraction residue, and broken repetition runs. The gate records decision_scope=structural_usability_only and explicitly excludes semantic quality, duplicate status, coverage balance, and Utility outcomes.
- Allowed signals: decodability and encoding hygiene, control-character and symbol hygiene, minimum learnable-unit length, markup/extraction residue, hard broken repetition runs
- Prohibited signals: quality classifier output, duplicate indicators, coverage signals, utility outcomes, semantic informativeness judgments
- Acceptance tests:
  - structural_validity_gate(clean_structured) > structural_validity_gate(noisy_corrupted)
  - style-repetitive but readable educational/reference text remains valid and is reported through warning_rules, not violated_rules
  - structural_validity_gate records decision_scope=structural_usability_only
  - warning_rules may be present on valid chunks and do not fail the gate

#### `structural_validity_score`

- Role: `diagnostic`
- Status: `diagnostic`
- Claim: Soft support score and audit surface for the hard Validity gate; it is not a standalone quality or selection objective.
- Implementation: `signals/core.py` / `CoreMetricScorer.structural_validity_score`
- Definition: Continuous diagnostic score based only on structural usability features. Hard failures are listed in violated_rules; style or borderline concerns are listed in warning_rules and do not invalidate a chunk by themselves. This score is an audit surface for the binary gate, not a ranking objective.
- Allowed signals: form/structure ratios, character and sentence hygiene, hard invalidity rules, diagnostic warning rules
- Prohibited signals: quality classifier output, duplication indicators, coverage signals, utility outcomes
- Acceptance tests:
  - structural_validity_score(clean_structured) > structural_validity_score(noisy_corrupted)
  - warning_rules are diagnostic and do not directly determine pass/fail
  - structural_validity_score exposes hard_rule_count and warning_rule_count
  - structural_validity_score excludes reference_quality_score and duplicate indicators from its decision

### Selection Value Evidence (`Quality` legacy alias)

#### `reference_quality_score`

- Role: `selection_signal`
- Status: `paper_aligned`
- Claim: Frozen pre-outcome selection-value proxy for observable information density, structural usefulness, and boilerplate risk; not intrinsic data quality.
- Implementation: `signals/core.py` / `CoreMetricScorer.reference_quality_score`
- Definition: Legacy metric name for frozen observable pre-outcome selection-value evidence. It estimates information density, coherence-like structure, and useful content support for chunks that already pass Validity, while penalizing boilerplate and low-information patterns. It is not intrinsic or ground-truth data quality, is not a Utility outcome, and has no hard-reject authority. Low evidence may affect optional budget allocation but cannot remove a Stage-A-pass record from the full curated pool.
- Allowed signals: reference quality classifier features, coherence and useful content-density features, style bucket, length bucket, lexical diversity, and structural hygiene as calibration context only
- Prohibited signals: duplicate flags, coverage retention, utility probe outcome, validity hard-gate decision as a substitute for quality
- Acceptance tests:
  - reference_quality_score(explanatory) > reference_quality_score(shallow_procedural)
  - reference_quality_score(coherent_prose) > reference_quality_score(corrupted_readable)
  - reference_quality_score(informative_dense) > reference_quality_score(template_boilerplate)
  - reference_quality_score(clean_structured) > reference_quality_score(template_boilerplate)
  - quality audit reports by_style_bucket, by_domain_bucket_top, by_length_bucket, and valid_but_low_quality
  - reference_quality_score is used in Stage-B selection and not as a Stage-A hard gate
  - reference_quality_score details declare canonical_core_axis=Selection Value Evidence and hard_reject_authority=false
  - a low reference_quality_score does not remove a Stage-A-pass record from the full curated pool
  - when no training budget is constrained, Stage B may emit retain_all
  - reference_quality_score details include quality_calibration_policy=style_length_normalized_quality_v2
  - reference_quality_score details include style_length_normalized_quality, quality_evidence_score, and style_length_quality_correction
  - short informative non-boilerplate chunks may receive a conservative calibration floor, while boilerplate remains low selection-value
  - selector diagnostics report quality_band_distribution_balance with soft_top_quality_anti_collapse when enabled
  - quality bands split the high proxy-score region into 0.90-0.95, 0.95-0.99, and >=0.99 so top-tail concentration is measurable

### Redundancy

#### `exact_duplicate_indicator`

- Role: `gate`
- Status: `paper_backed`
- Claim: Detect exact duplicates from text hash collisions.
- Implementation: `signals/core.py` / `CoreMetricScorer.exact_duplicate_indicator`
- Definition: Binary duplicate indicator from corpus hash counts.
- Allowed signals: exact hash counts
- Prohibited signals: quality scores, coverage signals, utility outcomes
- Acceptance tests:
  - exact_duplicate_indicator(exact_duplicate) > exact_duplicate_indicator(clean_structured)

#### `shingle_near_duplicate_indicator`

- Role: `selection_signal`
- Status: `diagnostic` (no irreversible Stage-A authority)
- Claim: Binary fuzzy near-duplicate evidence from locality-sensitive shortlist plus overlap verification.
- Implementation: `signals/core.py` / `CoreMetricScorer.shingle_near_duplicate_indicator`
- Definition: Binary fuzzy near-duplicate evidence using local simhash buckets and verified shingle overlap thresholds. It cannot authorize irreversible Stage-A rejection until independent precision and useful-data-dropout gates pass.
- Allowed signals: simhash locality, shingle overlap
- Prohibited signals: quality scores, coverage signals, utility outcomes
- Acceptance tests:
  - shingle_near_duplicate_indicator(near_duplicate) > shingle_near_duplicate_indicator(clean_structured)

#### `shingle_near_duplicate_risk_score`

- Role: `selection_signal`
- Status: `paper_aligned`
- Claim: Continuous harmful-redundancy risk used as a Stage B redundancy penalty while preserving useful recurrence.
- Implementation: `signals/core.py` / `CoreMetricScorer.shingle_near_duplicate_risk_score`
- Definition: Continuous risk from verified overlap, SimHash prefix pressure, and intra-chunk repetition pressure. The canonical policy is `harmful_redundancy_minus_useful_recurrence_v1`: exact/near duplicate burden remains penalized, but definitional, example-driven, exercise-like, and technical-reference recurrence receives limited relief when verified overlap is low.
- Allowed signals: overlap/hash/repetition, local text-structure markers only to distinguish useful recurrence from harmful redundancy
- Prohibited signals: quality classifier output, coverage retention, utility outcomes
- Acceptance tests:
  - near_dup(near_duplicate) > near_dup(clean_structured)
  - shingle_near_duplicate_risk_score(intra_chunk_repetitive) > shingle_near_duplicate_risk_score(clean_structured)
  - shingle_near_duplicate_risk_score details include redundancy_policy=harmful_redundancy_minus_useful_recurrence_v1
  - shingle_near_duplicate_risk_score details include harmful_redundancy_risk and useful_recurrence_score

### Coverage

#### `subset_coverage_retention_score`

- Role: `subset_validator`
- Status: `paper_aligned`
- Claim: Evaluate subset coverage preservation, especially tail-cluster retention.
- Implementation: `policy/subsets.py` / `_coverage_retention`
- Definition: Subset-level composite of distribution similarity, rare-cluster retention ratio, retained rare-cluster count, and declared domain/capability-mix drift when metadata and a Deployment Contract support that axis. Coverage is reported as source coverage, style coverage, semantic cluster-backbone coverage, and observed composition drift. Domain coverage is only claimed when explicit domain metadata exists; target-mix satisfaction is only claimed when a target mix is declared and validated. Otherwise source buckets and observed domain arms are reported as fallback/diagnostic support and must not be over-claimed as semantic web-domain coverage or a universal domain ratio. A diagnostic learning-signal coverage report compares the selected subset against the canonical multi-matched Stage-A baseline for token novelty, phrase novelty, concept density, moderate difficulty share, and template density. This report is diagnostic only and does not replace Utility outcome validation.
- Allowed signals: cluster distribution, tail-cluster presence/count, source bucket distribution support, explicit domain metadata distribution support when available, style/format bucket distribution support, semantic cluster-backbone audit
- Prohibited signals: quality score, duplicate indicators, utility outcomes
- Acceptance tests:
  - subset_coverage_retention_score(selected_subset) is reported together with distribution_similarity, tail_cluster_retention, tail_cluster_retained_count, source_coverage_support, domain_coverage_support, style_coverage_support, and semantic_coverage_support
  - subset_coverage_retention_score is only treated as passing when cluster_backbone_audit.passed == true in certification mode
  - domain_coverage_support records support_scope so fallback source buckets are not over-claimed as explicit domain metadata
  - style_coverage_support, source/domain bucket support, and semantic_coverage_support are validated separately from the scalar coverage score
  - coverage_axis_components reports source, style, and semantic support payloads
  - coverage_details include learning_signal_coverage_diagnostic with selected, baseline, gaps_selected_minus_baseline, and risk_flags
  - learning_signal_coverage_diagnostic policy is diagnostic_only_not_selector_objective

### Utility

#### `small_lm_probe_gain_score`

- Role: `subset_validator`
- Status: `paper_aligned`
- Claim: Measure Utility as a control-validated small-LM evidence protocol that separates probe sensitivity, selected-over-random curation benefit, and selected-over-multi-matched strict counterfactual benefit.
- Implementation: `utility/lm_probe.py` / `score_selected_records`
- Definition: Utility is a Stage-C subset-level validation protocol rather than a selector objective. The protocol first runs a small-LM sensitivity audit with a common Stage-A baseline disjoint from all sensitivity arms. Destructive probe validity requires positive_control > baseline_stageA_random > corrupted_negative_control, where corrupted_negative_control destroys next-token structure with hash-noise text. Control ordering also reports MDE-aware margins so near-noise differences are treated as inconclusive rather than decisive failures. A separate token_shuffle_negative_control is reported as token-inventory stress; decisive failure there is token-exposure confounding, while below-MDE failure is a token-exposure inconclusive caveat, not the destructive sanity check itself. Same-dataset low-quality controls are retained as diagnostics because they can still improve NLL through vocabulary and domain-token exposure. The protocol then reports curation benefit as selected versus baseline_stageA_random under equal token budgets, and reports strict counterfactual benefit as selected versus baseline_multi_matched_stageA_random, where the multi-matched baseline is sampled from Stage-A records matched by quality band, length bucket, style bucket, and domain bucket with hierarchical fallback. In-domain and all-pairwise OOD cells are evaluated with paired held-out documents, delta-NLL, paired-bootstrap CI lower bound, minimum detectable effect, seed/bucket stability, and train-exposure audit fields. The canonical Stage-C strict pass remains tied to baseline_multi_matched_stageA_random, but reporting no longer collapses all Utility failures into one negative score: aggregate evidence reports probe_sensitivity_status, curation_benefit_status, strict_counterfactual_status, utility_probe_valid, utility_strict_pass, failure_reason, and evidence_tier. Evidence tiers include not_evaluable_utility_evidence, probe_valid_token_exposure_caveat, random_baseline_gain, random_baseline_gain_with_token_exposure_caveat, matched_baseline_inconclusive, matched_baseline_gain, and strict_certification_ready. baseline_stageA_random is the curation-benefit baseline; baseline_multi_matched_stageA_random is the strict counterfactual certification baseline; quality-, length-, style-, and full-random baselines remain diagnostic stress tests.
- Allowed signals: fixed-budget train/eval outcomes, held-out NLL/perplexity deltas, Stage-A feasible random baseline comparison, held-out NLL delta under equal train-token budget, probe stability diagnostics, equal target train exposure, multi-epoch small-LM adaptation protocol
- Prohibited signals: quality metrics, redundancy metrics, coverage metrics, utility surrogate feature vectors, raw full random baseline as canonical pass/fail baseline
- Acceptance tests:
  - small_lm_probe_gain_score(selected_subset) > 0 against baseline_multi_matched_stageA_random on required eval cells
  - baseline_quality_band_matched_stageA_random, baseline_length_matched_stageA_random, baseline_style_matched_stageA_random, baseline_stageA_random, and baseline_full_random are recorded as diagnostic stress outputs but excluded from canonical Utility pass/fail
  - utility protocol records canonical_baseline as baseline_multi_matched_stageA_random and diagnostic_baselines includes quality-band, length, style, Stage-A random, and full-random baselines
  - utility protocol records utility_pass_statistic as mean in development and min in certification
  - development reports include certification_shadow with separate protocol_readiness, in_domain_signal, and ood_signal sections
  - certification_shadow includes blocker_categories, scope_snapshots, strict_values, and worst_cells for in-domain and all-pairwise OOD Utility evidence
  - utility details include delta_nll, delta_nll_ci_low, relative_nll_gain, train token counts, per-baseline in_domain results, and out_of_domain results keyed by eval_dataset
  - utility details include selected/baseline effective_train_steps and estimated_seen_train_tokens for probe exposure auditing
  - selected and baseline probes use the same configured train_epochs within each eval cell
  - utility details include selected/baseline target_train_exposure_ratio and one_epoch_train_steps for exposure auditing
  - utility protocol records train_epochs and max_train_steps separately, with train_epochs >= 1.0
  - with three datasets, each train dataset reports two OOD eval datasets, producing six total OOD pairs per profile
  - small_lm_probe_gain_score aggregate reports stability_diagnostics with positive_run_fraction, ci_positive_fraction, and mean_delta_nll_to_std_ratio
  - certification_shadow reports stability_analysis for in_domain, ood, and combined_effective scopes
  - certification readiness fails protocol_readiness when selected or baseline training hits max_train_steps before reaching target train_epochs
  - small_lm_probe_gain_score aggregate reports signal_interpretation with strict_positive, inconclusive_numerical_drift, inconclusive_below_detectable_effect, inconclusive_ci_crosses_zero, or strict_negative status
  - numerical drift inside delta_nll_numerical_tolerance is reported as inconclusive rather than optimized away or treated as confirmed negative evidence
  - utility probe runs report paired_bootstrap=true and eval_pairing_policy=paired_same_eval_documents for each baseline/eval cell
  - small_lm_probe_gain_score aggregate reports minimum_detectable_delta_nll_95, effect_to_mde_ratio, and detectable_effect_fraction fields for Utility signal power analysis
  - certification protocol readiness checks actual holdout_bucket_count and ood_holdout_bucket_count against min_probe_bucket_count, not only configured seed count
  - certification dry-run profiles use certification mode, min pass statistic, positive CI, enforced all-pairwise OOD, at least four seeds, and at least four in-domain/OOD holdout buckets
  - stage_c_core_validation reports separate in_domain_utility_axis_pass, cross_domain_utility_axis_pass, domain_specific_utility_axis_pass, general_purpose_utility_axis_pass, and final_utility_axis_pass fields
  - utility evidence summary reports final_certification_scope, final_scope_certification_ready, domain_specific_certification_ready, and general_purpose_certification_ready
  - utility sensitivity audit writes outputs/validation/utility_sensitivity_audit.json with schema_version utility-sensitivity-audit-v1
  - utility sensitivity audit records destructive_probe_valid/order_pass for positive_control > baseline_stageA_random > corrupted_negative_control and records token_shuffle_negative_control as a separate token-inventory stress arm
  - utility aggregate reports probe_sensitivity_status, curation_benefit_status, and strict_counterfactual_status
  - utility aggregate reports utility_probe_valid and utility_strict_pass as separate fields
  - utility aggregate evidence_tier is one of not_evaluable_utility_evidence, probe_valid_token_exposure_caveat, random_baseline_gain, random_baseline_gain_with_token_exposure_caveat, matched_baseline_inconclusive, matched_baseline_gain, or strict_certification_ready
  - utility failure_reason is one of probe_not_evaluable, selected_below_stageA_random, random_gain_only, random_gain_only_with_token_exposure_caveat, matched_inconclusive, strict_negative, or pass
  - baseline_stageA_random is used as the curation benefit baseline, not as the strict certification baseline
  - baseline_multi_matched_stageA_random remains the strict counterfactual certification baseline
  - datasets with destructive_probe_valid=false are not reported as selector failures and selector_tuning_allowed=false is recorded

### Deprecated/Diagnostic

#### `fixed_token_probe_gain_score`

- Role: `subset_validator`
- Status: `deprecated_diagnostic`
- Claim: Deprecated compatibility alias for the old hybrid n-gram utility probe.
- Implementation: `utility/fixed_token_probe.py` / `score_selected_records`
- Definition: Legacy utility output retained for compatibility only; not canonical for Stage C utility decisions.
- Allowed signals: compatibility outputs
- Prohibited signals: used as canonical utility pass/fail
- Acceptance tests:
  - fixed_token_probe_gain_score is retained only as a deprecated compatibility diagnostic and is not used for canonical utility pass/fail

#### `explanatory_quality_proxy`

- Role: `selection_signal`
- Status: `diagnostic`
- Claim: Diagnostic proxy for explanatory style.
- Implementation: `signals/core.py` / `CoreMetricScorer.explanatory_quality_proxy`
- Definition: Heuristic/prototype similarity based explanatory-style estimate for diagnostics.
- Allowed signals: heuristics
- Acceptance tests:
  - explanatory_quality_proxy(explanatory) > explanatory_quality_proxy(shallow_procedural)

#### `tail_cluster_rarity_proxy`

- Role: `selection_signal`
- Status: `diagnostic`
- Claim: Diagnostic proxy for chunk rarity in cluster space.
- Implementation: `signals/core.py` / `CoreMetricScorer.tail_cluster_rarity_proxy`
- Definition: Cluster-size-based rarity estimate at chunk level.
- Allowed signals: cluster size
- Acceptance tests:
  - tail_cluster_rarity_proxy(tail_rare) > tail_cluster_rarity_proxy(head_common)

#### `predictive_utility_proxy`

- Role: `diagnostic`
- Status: `diagnostic`
- Claim: Diagnostic utility surrogate for comparison only.
- Implementation: `signals/core.py` / `CoreMetricScorer.predictive_utility_proxy`
- Definition: Diagnostic utility surrogate for development inspection only. It is explicitly excluded from canonical selector objectives, Stage-C Utility pass/fail, and certification readiness.
- Allowed signals: surrogate features
- Prohibited signals: used as final utility pass/fail, used in canonical selector objective, used as certification readiness evidence
- Acceptance tests:
  - predictive_utility_proxy(explanatory) > predictive_utility_proxy(shallow_procedural)
  - predictive_utility_proxy is diagnostic and absent from selector objective weights
  - predictive_utility_proxy is not used for Stage-C Utility pass/fail

## Paper Registry
- `gpt3_2020`: Language Models are Few-Shot Learners ()
- `c4_2020`: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer ()
- `fineweb_2024`: FineWeb: decanting the web for the finest text data at scale ()
- `dolma_2024`: Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research ()
- `paloma_2023`: PALOMA: A Benchmark for Evaluating Language Model Utility ()
- `datacomp_lm_2024`: DataComp-LM: In search of the next generation of training sets for language models ()
- `refinedweb_2023`: The RefinedWeb Dataset for Falcon LLM ()
