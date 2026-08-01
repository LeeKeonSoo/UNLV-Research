#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "positive_quality_coverage_contract_v1.json"
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def load_contract() -> dict[str, JsonValue]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_is_frozen_but_not_runtime_active() -> None:
    contract = load_contract()

    assert contract["schema_version"] == "positive-quality-coverage-contract-v1"
    assert contract["status"] == "design_frozen_not_runtime_active"
    assert contract["runtime_activation"] is False
    assert contract["current_runtime_contract"] == "configs/curation_contract.json"


def test_quality_is_a_positive_three_way_retention_gate() -> None:
    contract = load_contract()
    quality = contract["quality"]

    assert quality["definition"] == "positive_retention_eligibility_not_intrinsic_quality"
    assert quality["routes"] == [
        "general_prose",
        "code",
        "math",
        "technical_documentation",
        "conversation_instruction",
        "unknown",
    ]
    assert quality["decisions"] == ["eligible_keep", "reject", "abstain"]
    assert quality["combination"] == "conjunctive_per_route_without_weighted_sum"
    assert quality["required_evidence"] == [
        "route_confidence",
        "substantive_payload",
        "coherence_completeness",
        "route_specific_evidence",
    ]
    assert quality["unknown_route_action"] == "abstain"


def test_quality_evidence_artifact_is_frozen_and_candidate_only() -> None:
    contract = load_contract()
    artifact = contract["quality"]["evidence_artifact"]

    assert artifact["config"] == "configs/positive_quality_evidence_v1.json"
    assert artifact["implementation"] == "positive_quality_evidence.py"
    assert artifact["provider_registry"] == "configs/positive_quality_provider_registry_v1.json"
    assert artifact["status"] == "candidate_only_not_stage_c_active"
    assert artifact["provider_scores_are_frozen_before_selection"] is True
    assert artifact["missing_or_invalid_evidence_action"] == "abstain"
    assert artifact["global_weighted_score_allowed"] is False


def test_normal_and_hard_differ_only_on_abstention() -> None:
    contract = load_contract()
    modes = contract["mode_actions"]

    assert modes["normal"] == {
        "eligible_keep": "retain",
        "reject": "not_select",
        "abstain": "retain",
    }
    assert modes["hard"] == {
        "eligible_keep": "retain",
        "reject": "not_select",
        "abstain": "not_select",
    }
    assert contract["not_selected_artifact_required"] is True


def test_coverage_classifies_representation_without_assigning_importance() -> None:
    contract = load_contract()
    coverage = contract["coverage"]

    assert coverage["definition"] == "multi_axis_representation_classification_and_loss_audit"
    assert coverage["axes"] == [
        "semantic_domain",
        "language_script",
        "format_genre",
        "content_morphology",
    ]
    assert coverage["classification"] == "multi_label_with_unknown"
    assert coverage["may_assign_cross_stratum_importance"] is False
    assert coverage["may_enforce_target_mix_or_quota"] is False
    assert coverage["may_rescue_quality_reject"] is False
    assert coverage["representative_pool"] == "quality_eligible_records_only"


def test_stage_c_order_prevents_low_quality_representative_lock_in() -> None:
    contract = load_contract()

    assert contract["stage_contract"]["stage_c_order"] == [
        "coverage_pre_tagging",
        "positive_quality_gate",
        "near_duplicate_and_scaffold_family_resolution",
        "coverage_post_audit",
        "reason_coded_materialization",
    ]


def test_thresholds_are_calibrated_without_a_retention_budget() -> None:
    contract = load_contract()
    calibration = contract["quality"]["calibration"]

    assert calibration["selection_rule"] == "most_compressive_feasible_threshold_per_route"
    assert calibration["uses_target_retention_fraction"] is False
    assert calibration["confidence_level"] == 0.95
    assert calibration["normal_clean_control_false_reject_upper_bound"] == 0.01
    assert calibration["hard_clean_control_false_reject_upper_bound"] == 0.05
    assert calibration["confirmatory_results_may_retune_thresholds"] is False


def test_runtime_inputs_exclude_feedback_and_source_shortcuts() -> None:
    contract = load_contract()
    forbidden = set(contract["runtime_forbidden_inputs"])

    assert {
        "Utility",
        "NLL",
        "benchmark_outcomes",
        "target_retention_fraction",
        "source_identity",
        "source_tier",
        "path",
        "declared_domain_quota",
        "dataset_identity",
    } <= forbidden


if __name__ == "__main__":
    test_contract_is_frozen_but_not_runtime_active()
    test_quality_is_a_positive_three_way_retention_gate()
    test_quality_evidence_artifact_is_frozen_and_candidate_only()
    test_normal_and_hard_differ_only_on_abstention()
    test_coverage_classifies_representation_without_assigning_importance()
    test_stage_c_order_prevents_low_quality_representative_lock_in()
    test_thresholds_are_calibrated_without_a_retention_budget()
    test_runtime_inputs_exclude_feedback_and_source_shortcuts()
    print("[positive-quality-coverage-contract] frozen design boundary: pass")
