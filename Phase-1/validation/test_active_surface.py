#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_python_files_are_limited_to_curation_surface() -> None:
    expected = {
        "aggressive_structural_candidate_runner.py", "composition_audit.py", "curation_artifacts.py", "data_eval_common.py", "general_web_span_compaction.py", "hard_development_ablation.py", "hard_quality_profile.py", "hard_structural_runtime.py", "inline_license_comment_block_compaction.py", "inline_license_header_compaction.py", "mid_quality_estimator.py", "mode_development_ablation.py", "pretraining_audit.py", "python_code_evidence_audit.py", "quality_rule_development_matrix.py", "reason_code_audit.py", "repeated_span_template_inventory.py", "run_curation.py", "span_level_candidate_development_runner.py", "span_level_template_compaction.py", "stage_c2_development_runner.py", "stage_c2_model_relative_selector.py", "stage_c_selection.py", "weak_development_gate.py",
        "external_evaluation/__init__.py", "external_evaluation/confirmatory_qlora_training.py", "external_evaluation/evalplus_generator.py", "external_evaluation/official_suite_generator.py", "external_evaluation/preflight_code_7benchmark_pretraining_eligible_v3.py", "external_evaluation/preflight_record_disjoint_confirmatory.py", "external_evaluation/validation_integrity.py",
        "ingestion/__init__.py", "ingestion/candidate_processing.py", "ingestion/candidate_contract.py", "ingestion/input_adapter.py",
        "scripts/adapt_legacy_candidate_pool.py", "scripts/audit_python_template_families.py", "scripts/audit_reference_distribution_overlap.py", "scripts/build_calibrated_selector_reference_pool.py", "scripts/build_code_benchmark_snapshots.py", "scripts/build_core_rule_inventory.py", "scripts/build_external_validation_integrity_report.py", "scripts/build_historical_proxy_forensics.py", "scripts/build_historical_selector_decomposition.py", "scripts/build_python_syntax_inventory.py", "scripts/build_python_template_family_inventory.py", "scripts/build_reference_distribution_calibration.py", "scripts/build_reference_distribution_review_sample.py", "scripts/build_rule_opportunity_audit.py", "scripts/build_stage_c2_ablation_report.py", "scripts/collect_git_attribute_candidate_pool.py", "scripts/collect_huggingface_text_candidate_pool.py", "scripts/collect_json_batch_candidate_pool.py", "scripts/collect_the_stack_v2_repository_samples.py", "scripts/materialize_pretraining_eligible_v3_training_inputs.py", "scripts/preflight_calibrated_selector.py", "scripts/run_reference_distribution_probe.py",
        "validation/core_behavior_audit_v3.py", "validation/test_active_evalplus_generator.py", "validation/test_active_surface.py", "validation/test_aggressive_structural_candidate_runner.py", "validation/test_calibrated_selector_preflight.py", "validation/test_calibrated_selector_reference_pool.py", "validation/test_candidate_policy_gate.py", "validation/test_candidate_processing.py", "validation/test_code_7benchmark_preflight.py", "validation/test_code_7benchmark_pretraining_eligible_v3_execution.py", "validation/test_code_benchmark_snapshot_builder.py", "validation/test_code_evaluation_protocol.py", "validation/test_code_six_benchmark_execution_protocol.py", "validation/test_core_behavior_audit_v3.py", "validation/test_core_case_matrix.py", "validation/test_external_validation_integrity.py", "validation/test_policy_fixture_contract.py", "validation/test_policy_integrity_boundary.py", "validation/test_policy_registry_contract.py", "validation/test_reason_code_impact_audit.py", "validation/test_repeated_span_template_inventory.py", "validation/test_record_disjoint_external_protocol.py", "validation/test_core_policy_registry.py", "validation/test_core_policy_runtime_linkage.py", "validation/test_core_rule_inventory.py", "validation/test_curation_contract.py", "validation/test_curation_runtime.py", "validation/test_explicit_generated_artifact_rule.py", "validation/test_git_attribute_candidate_collection.py", "validation/test_git_attribute_stress_contract.py", "validation/test_historical_proxy_forensics.py", "validation/test_huggingface_text_candidate_collection.py", "validation/test_inline_license_header_compaction.py", "validation/test_json_batch_candidate_collection.py", "validation/test_legacy_candidate_pool_adapter.py", "validation/test_official_suite_generator.py", "validation/test_policy_profile_contract.py", "validation/test_pretraining_audit.py", "validation/test_pretraining_eligible_v3_training_materialization.py", "validation/test_python_syntax_inventory.py", "validation/test_python_template_family_false_positive_audit.py", "validation/test_python_template_family_inventory.py", "validation/test_reference_distribution_calibration.py", "validation/test_reference_distribution_overlap_audit.py", "validation/test_reference_distribution_probe.py", "validation/test_reference_distribution_policy_boundary.py", "validation/test_reference_distribution_review_sample.py", "validation/test_repository_code_hazard_handling.py", "validation/test_rule_opportunity_audit.py", "validation/test_source_agnostic_stage_a.py", "validation/test_source_contract.py", "validation/test_span_level_candidate_development_runner.py", "validation/test_span_level_template_compaction.py", "validation/test_stage_c2_development_runner.py", "validation/test_stage_c2_model_relative_selector.py", "validation/test_stage_c_selection.py",
    }
    expected.add("validation/test_inline_license_comment_block_compaction.py")
    expected.add("validation/test_python_code_evidence_audit.py")
    expected.add("validation/test_coverage_invariants.py")
    expected.add("validation/test_weak_structural_quality_rules.py")
    expected.add("validation/test_weak_development_gate.py")
    expected.add("validation/test_mid_quality_protocol.py")
    expected.add("validation/test_mid_quality_estimator.py")
    expected.add("validation/test_hard_quality_profile.py")
    expected.add("validation/test_hard_quality_protocol.py")
    expected.add("validation/test_mode_development_ablation.py")
    expected.add("validation/test_mode_development_ablation_protocol.py")
    expected.add("validation/test_normal_hard_mode_contract.py")
    expected.add("validation/test_hard_policy_inventory.py")
    expected.add("validation/test_hard_structural_runtime.py")
    expected.add("validation/test_hard_development_ablation.py")
    expected.add("validation/test_hard_confirmatory_scope.py")
    expected.add("validation/test_confirmatory_qlora_training.py")
    expected.add("validation/test_general_web_span_compaction.py")
    expected.add("validation/test_composition_axes.py")
    expected.add("validation/test_core_label_reconciliation.py")
    expected.add("validation/test_quality_candidate_package.py")
    expected.add("validation/test_quality_rule_development_matrix.py")
    expected.add("stage_c2_frozen_proxy_evidence.py")
    expected.add("validation/test_stage_c2_frozen_proxy_evidence.py")
    expected.add("stage_c2_proxy_lm_scoring.py")
    expected.add("validation/test_stage_c2_proxy_lm_scoring.py")
    expected.add("stage_c2_development_protocol_report.py")
    expected.add("validation/test_stage_c2_development_protocol_report.py")
    expected.add("scripts/build_stage_c2_case_audit.py")
    expected.add("validation/test_stage_c2_promotion_decision.py")
    expected.add("repeated_line_block_compaction.py")
    expected.add("validation/test_repeated_line_block_compaction.py")
    expected.add("validation/test_redundancy_candidate_package.py")
    expected.add("repeated_label_block_development_matrix.py")
    expected.add("validation/test_repeated_label_block_development_matrix.py")
    expected.add("validation/test_the_stack_v2_repository_collection.py")
    expected.add("validation/test_historical_selector_decomposition.py")
    expected.add("quality_retention.py")
    expected.add("quality_decision_contract.py")
    expected.add("quality_rule_evidence.py")
    expected.add("validation/test_quality_retention_decision.py")
    expected.add("validation/test_quality_candidate_scope.py")
    expected.add("validation/test_positive_quality_coverage_contract.py")
    expected.add("validation/test_content_routing_quality_coverage_contract_v2.py")
    expected.add("content_router.py")
    expected.add("validation/test_content_router_v2.py")
    expected.add("validity_recovery.py")
    expected.add("validation/test_validity_recovery_v1.py")
    expected.add("route_conditioned_quality.py")
    expected.add("validation/test_route_conditioned_quality_v2.py")
    expected.add("quality_evidence_gate.py")
    expected.add("validation/test_quality_route_evidence_gate_v2.py")
    expected.add("coverage_taxonomy.py")
    expected.add("validation/test_coverage_taxonomy_v1.py")
    expected.add("positive_quality_evidence.py")
    expected.add("validation/test_positive_quality_evidence.py")
    expected.add("validation/test_positive_quality_provider_registry.py")
    expected.add("scripts/score_qurater_development.py")
    expected.add("validation/test_qurater_development_scoring.py")
    expected.add("scripts/calibrate_qurater_general_prose.py")
    expected.add("validation/test_qurater_general_prose_calibration.py")
    expected.add("scripts/build_general_prose_evidence_v2.py")
    expected.add("scripts/audit_general_prose_evidence_v2.py")
    expected.add("validation/test_general_prose_evidence_v2.py")
    expected.add("validation/test_general_prose_evidence_audit_v2.py")
    expected.add("scripts/score_general_provider_candidates_v2.py")
    expected.add("scripts/audit_general_scalar_provider_v2.py")
    expected.add("validation/test_general_provider_candidate_scoring_v2.py")
    expected.add("validation/test_general_scalar_provider_audit_v2.py")
    expected.add("validation/test_general_provider_candidate_protocol_v2.py")
    expected.add("validation/test_general_provider_candidate_decision_v2.py")
    expected.add("code_positive_evidence.py")
    expected.add("validation/test_code_positive_evidence.py")
    expected.add("scripts/score_stack_edu_python_development.py")
    expected.add("validation/test_stack_edu_python_development_scoring.py")
    expected.add("scripts/build_post_cutoff_python_clean_controls.py")
    expected.add("validation/test_post_cutoff_python_clean_controls.py")
    expected.add("scripts/calibrate_stack_edu_python.py")
    expected.add("validation/test_stack_edu_python_calibration.py")
    expected.add("math_positive_evidence.py")
    expected.add("validation/test_math_positive_evidence.py")
    expected.add("scripts/score_math_positive_development.py")
    expected.add("validation/test_math_positive_development_scoring.py")
    expected.add("math_structural_evidence.py")
    expected.add("scripts/audit_math_hard_candidate_profile.py")
    expected.add("scripts/calibrate_math_complete_bundle.py")
    expected.add("scripts/calibrate_math_known_heads.py")
    expected.add("scripts/materialize_math_clean_controls.py")
    expected.add("scripts/score_math_structural_heads.py")
    expected.add("scripts/train_math_structural_heads.py")
    expected.add("validation/test_math_clean_control_materialization.py")
    expected.add("validation/test_math_complete_bundle_calibration.py")
    expected.add("validation/test_math_hard_candidate_audit.py")
    expected.add("validation/test_math_known_head_calibration.py")
    expected.add("validation/test_math_structural_evidence.py")
    expected.add("validation/test_score_math_structural_heads.py")
    expected.add("validation/test_train_math_structural_heads.py")
    expected.add("explicit_structural_coherence.py")
    expected.add("latex_control_units.py")
    expected.add("scripts/score_math_explicit_coherence.py")
    expected.add("validation/test_math_explicit_coherence_scoring.py")
    expected.add("route_quality_evidence_candidates.py")
    expected.add("validation/test_route_quality_evidence_candidates_v1.py")
    expected.add("validation/core_behavior_contracts.py")
    expected.add("validation/core_behavior_executors.py")
    expected.add("model_provider_contract.py")
    expected.add("corpus_profiler.py")
    expected.add("scripts/profile_corpus_audit_only.py")
    expected.add("validation/test_model_provider_contract_v1.py")
    expected.add("validation/test_corpus_profiler_audit_only_v1.py")
    expected.add("validity_v2.py")
    expected.add("validity_v2_audit.py")
    expected.add("scripts/build_validity_v2_audit.py")
    expected.add("validation/test_validity_v2.py")
    expected.add("redundancy_v2.py")
    expected.add("redundancy_v2_retrieval.py")
    expected.add("redundancy_v2_audit.py")
    expected.add("scripts/build_redundancy_v2_audit.py")
    expected.add("validation/test_redundancy_v2.py")
    expected.add("quality_effect_engine.py")
    expected.add("quality_effect_calibration.py")
    expected.add("scripts/build_quality_effect_engine_v2_audit.py")
    expected.add("validation/test_quality_effect_engine_v2.py")
    expected.add("coverage_contract.py")
    expected.add("coverage_metrics.py")
    expected.add("coverage_engine.py")
    expected.add("scripts/build_coverage_engine_v2_audit.py")
    expected.add("validation/test_coverage_engine_v2.py")
    expected.add("joint_selector_contract.py")
    expected.add("joint_selector_gates.py")
    expected.add("joint_selector_manifest.py")
    expected.add("joint_selector.py")
    expected.add("scripts/build_joint_selector_v1_audit.py")
    expected.add("validation/test_joint_selector_v1.py")
    expected.add("development_selection.py")
    expected.add("development_selection_contract.py")
    expected.add("development_selection_preflight.py")
    expected.add("scripts/build_development_selection_v1_audit.py")
    expected.add("validation/test_development_selection_v1.py")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if "archive" not in path.parts and ".omo" not in path.parts and "__pycache__" not in path.parts
    }
    assert actual == expected


def test_active_configs_are_limited_to_revised_contract() -> None:
    expected = {
        "aggressive_structural_candidate_ablation_preregistration.json", "aggressive_structural_candidate_v1.example.json", "calibrated_selector_contract.example.json", "core_policy_registry.json", "curation_contract.json", "curation_run_contract.example.json", "policy_card_contract.json", "policy_cards.json", "policy_profiles.json", "span_level_template_candidate_ablation_preregistration.json", "span_level_candidate_development_stage_c_selection.json", "stage_c2_model_relative_candidate_ablation_preregistration.json",
    }
    expected.add("explicit_coherence_fresh_control_protocol_v1.json")
    expected.add("explicit_coherence_fresh_control_protocol_v2.json")
    expected.add("explicit_coherence_fresh_control_protocol_v3.json")
    expected.add("code_7m_text_only_baseline_v1.json")
    expected.add("stage_c2_frozen_proxy_development_protocol.json")
    expected.add("stage_c2_promotion_decision.json")
    expected.add("mid_quality_protocol_v1.json")
    expected.add("hard_quality_protocol_v1.json")
    expected.add("mode_development_ablation_protocol_v1.json")
    expected.add("hard_policy_inventory_v1.json")
    expected.add("positive_quality_coverage_contract_v1.json")
    expected.add("content_routing_quality_coverage_contract_v2.json")
    expected.add("content_router_v2.json")
    expected.add("validity_recovery_v1.json")
    expected.add("route_conditioned_quality_v2.json")
    expected.add("quality_route_evidence_gate_v2.json")
    expected.add("coverage_taxonomy_v1.json")
    expected.add("positive_quality_evidence_v1.json")
    expected.add("positive_quality_provider_registry_v1.json")
    expected.add("route_quality_evidence_candidates_v1.json")
    expected.add("qurater_general_prose_development_v1.json")
    expected.add("qurater_provider_manifest_v1.json")
    expected.add("qurater_general_prose_development_bundle_v1.json")
    expected.add("general_prose_evidence_v2.json")
    expected.add("general_prose_evidence_bundle_v2.json")
    expected.add("general_provider_candidate_protocol_v2.json")
    expected.add("general_provider_candidate_decision_v2.json")
    expected.add("general_dclm_evidence_v2.json")
    expected.add("general_fineweb_edu_evidence_v2.json")
    expected.add("stack_edu_python_provider_manifest_v1.json")
    expected.add("stack_edu_python_development_bundle_v1.json")
    expected.add("stack_edu_python_clean_control_protocol_v1.json")
    expected.add("stack_edu_python_calibration_v1.json")
    expected.add("stack_edu_python_calibration_report_v1.json")
    expected.add("math_positive_provider_manifest_v1.json")
    expected.add("math_positive_development_bundle_v1.json")
    expected.add("model_provider_registry_v1.json")
    expected.add("corpus_profiler_contract_v1.json")
    expected.add("validity_v2.json")
    expected.add("redundancy_v2.json")
    expected.add("quality_effect_engine_v2.json")
    expected.add("coverage_engine_v2.json")
    expected.add("joint_selector_profiles_v1.json")
    expected.add("development_selection_v1.json")
    expected.update(
        {
            "math_complete_bundle_calibration_v1.json",
            "math_complete_bundle_calibration_v2.json",
            "math_complete_bundle_calibration_v3.json",
            "math_complete_bundle_calibration_v4.json",
            "math_complete_bundle_calibration_v5.json",
            "math_hard_candidate_fixture_gate_v1.json",
            "math_hard_candidate_profile_v1.json",
            "math_explicit_coherence_guard_v1.json",
            "math_explicit_coherence_guard_v2.json",
            "math_known_head_calibration_v1.json",
            "math_open_educational_clean_control_protocol_v1.json",
            "math_open_educational_clean_control_protocol_v2.json",
            "math_open_educational_clean_control_protocol_v3.json",
            "math_open_educational_clean_control_protocol_v4.json",
            "math_open_educational_clean_control_protocol_v5.json",
            "math_open_educational_clean_control_protocol_v6.json",
            "math_open_educational_clean_control_v1_route_audit.json",
            "math_quality_evidence_decision_v1.json",
            "math_quality_evidence_decision_v2.json",
            "math_quality_evidence_decision_v3.json",
            "math_structural_head_development_v1.json",
            "math_structural_head_development_v2.json",
            "math_structural_head_development_v3.json",
        }
    )
    actual = {path.name for path in (ROOT / "configs").glob("*.json")}
    assert actual == expected


if __name__ == "__main__":
    test_active_python_files_are_limited_to_curation_surface()
    test_active_configs_are_limited_to_revised_contract()
    print("[active-surface] revised curation files only: pass")
