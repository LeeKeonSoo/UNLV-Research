#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "positive_quality_provider_registry_v1.json"
QURATER_PROTOCOL = ROOT / "configs" / "qurater_general_prose_development_v1.json"
QURATER_PROVIDER = ROOT / "configs" / "qurater_provider_manifest_v1.json"
QURATER_BUNDLE = ROOT / "configs" / "qurater_general_prose_development_bundle_v1.json"
STACK_EDU_BUNDLE = ROOT / "configs" / "stack_edu_python_development_bundle_v1.json"
MATH_BUNDLE = ROOT / "configs" / "math_positive_development_bundle_v1.json"
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
KNOWN_ROUTES = {
    "general_prose",
    "code",
    "math",
    "technical_documentation",
    "conversation_instruction",
}
REQUIRED_HEADS = {
    "substantive_payload",
    "route_specific_evidence",
}


def load_registry() -> dict[str, JsonValue]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_cannot_activate_an_incomplete_provider_bundle() -> None:
    registry = load_registry()

    assert registry["schema_version"] == "positive-quality-provider-registry-v1"
    assert registry["runtime_activation"] is False
    assert registry["candidate_score_may_bypass_missing_heads"] is False
    assert registry["routing_precondition"]["quality_evidence"] is False
    routes = registry["routes"]
    assert set(routes) == KNOWN_ROUTES | {"unknown"}
    for route in KNOWN_ROUTES:
        entry = routes[route]
        assert entry["provider_bundle_complete"] is False
        assert entry["runtime_action"] == "abstain"
        assert set(entry["missing_heads"]) <= REQUIRED_HEADS
        assert entry["missing_heads"]


def test_provider_candidates_declare_scope_evidence_and_limitations() -> None:
    routes = load_registry()["routes"]

    for route in KNOWN_ROUTES:
        for candidate in routes[route]["candidates"]:
            assert candidate["provider_id"]
            assert candidate["artifact_url"].startswith("https://")
            assert candidate["primary_evidence_url"].startswith("https://")
            assert set(candidate["supported_heads"]) <= REQUIRED_HEADS
            assert candidate["limitations"]
            assert candidate["local_source_disjoint_calibration_required"] is True


def test_objective_mismatched_candidates_cannot_contribute_to_activation() -> None:
    routes = load_registry()["routes"]

    for route in KNOWN_ROUTES:
        for candidate in routes[route]["candidates"]:
            if candidate["purpose_alignment"] != "continued_pretraining":
                assert candidate["eligible_for_bundle"] is False


def test_unknown_route_is_a_fixed_abstention_boundary() -> None:
    unknown = load_registry()["routes"]["unknown"]

    assert unknown == {
        "provider_bundle_complete": False,
        "runtime_action": "abstain",
        "missing_heads": sorted(REQUIRED_HEADS),
        "candidates": [],
    }


def test_qurater_protocol_freezes_mapping_before_full_score_observation() -> None:
    protocol = json.loads(QURATER_PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["status"] == "preregistered_candidate_only"
    assert protocol["runtime_activation"] is False
    assert protocol["provider_revision"] == "bd61c778c2f42c6e406b7bac8064290ffc183ae1"
    assert protocol["head_mapping"] == {
        "route_confidence": "coverage_taxonomy_v1_general_knowledge_and_prose_binary",
        "substantive_payload": "qurater_facts_trivia_raw_logit",
        "coherence_completeness": "qurater_writing_style_raw_logit_coherence_only",
        "route_specific_evidence": "qurater_educational_value_raw_logit",
    }
    assert protocol["audit_only_dimensions"] == ["qurater_required_expertise_raw_logit"]
    assert protocol["target_retention_fraction_used"] is False
    assert protocol["external_results_visible"] is False


def test_qurater_provider_and_calibration_bundle_are_frozen_but_inactive() -> None:
    provider = json.loads(QURATER_PROVIDER.read_text(encoding="utf-8"))
    bundle = json.loads(QURATER_BUNDLE.read_text(encoding="utf-8"))

    assert provider["model_revision"] == "bd61c778c2f42c6e406b7bac8064290ffc183ae1"
    assert provider["training_dataset_revision"] == "b553523cf6e15b7af6744166ef144e9dc54695f0"
    assert provider["runtime_activation"] is False
    assert bundle["status"] == "calibrated_development_candidate_not_runtime_active"
    assert bundle["normal"]["selected_profile_id"] == "clean_q0"
    assert bundle["hard"]["selected_profile_id"] == "clean_q0.01"
    assert len(bundle["provider_manifest_sha256"]) == 64
    assert bundle["external_results_visible"] is False
    assert bundle["runtime_activation"] is False


def test_stack_edu_strict_calibration_failure_cannot_activate_code_route() -> None:
    registry = load_registry()
    candidate = registry["routes"]["code"]["candidates"][0]
    bundle = json.loads(STACK_EDU_BUNDLE.read_text(encoding="utf-8"))

    assert candidate["eligible_for_bundle"] is False
    assert candidate["candidate_mapping_status"] == "strict_clean_control_calibration_failed_no_runtime_authority"
    assert bundle["status"] == "strict_calibration_blocked_not_runtime_active"
    assert bundle["strict_calibration"]["normal_selected_profile"] is None
    assert bundle["strict_calibration"]["hard_selected_profile"] is None
    assert bundle["runtime_activation"] is False


def test_math_bundle_keeps_provider_heads_independent_and_abstains() -> None:
    registry = load_registry()
    candidates = registry["routes"]["math"]["candidates"]
    bundle = json.loads(MATH_BUNDLE.read_text(encoding="utf-8"))

    assert candidates[0]["supported_heads"] == []
    assert candidates[0]["supported_routing_preconditions"] == ["route_confidence"]
    assert candidates[1]["supported_heads"] == ["route_specific_evidence"]
    assert bundle["independent_evidence"]["substantive_payload"] is None
    assert bundle["independent_evidence"]["coherence_completeness"] is None
    assert bundle["runtime_activation"] is False


def test_general_scalar_candidates_are_frozen_rejections() -> None:
    candidates = load_registry()["routes"]["general_prose"]["candidates"]
    by_id = {candidate["provider_id"]: candidate for candidate in candidates}

    for provider_id in (
        "mlfoundations/dclm-fasttext-filter",
        "HuggingFaceFW/fineweb-edu-classifier",
    ):
        candidate = by_id[provider_id]
        assert candidate["eligible_for_bundle"] is False
        assert candidate["candidate_mapping_status"] == "rejected_source_transfer_and_semantic_stress"
        assert candidate["frozen_decision"] == "configs/general_provider_candidate_decision_v2.json"


if __name__ == "__main__":
    test_registry_cannot_activate_an_incomplete_provider_bundle()
    test_provider_candidates_declare_scope_evidence_and_limitations()
    test_objective_mismatched_candidates_cannot_contribute_to_activation()
    test_unknown_route_is_a_fixed_abstention_boundary()
    test_qurater_protocol_freezes_mapping_before_full_score_observation()
    test_qurater_provider_and_calibration_bundle_are_frozen_but_inactive()
    test_stack_edu_strict_calibration_failure_cannot_activate_code_route()
    test_math_bundle_keeps_provider_heads_independent_and_abstains()
    test_general_scalar_candidates_are_frozen_rejections()
    print("[positive-quality-provider-registry] incomplete bundles abstain: pass")
