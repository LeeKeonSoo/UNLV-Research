#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "configs" / "span_level_template_candidate_ablation_preregistration.json"


def test_candidate_policy_is_not_runnable_until_its_structural_gates_are_closed() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    candidate = next(policy for policy in registry["policies"] if policy["id"] == "stage_c_declared_dependency_copy_candidate")
    active_profile = next(profile for profile in profiles["profiles"] if profile["id"] == "safe_structural_v3")

    assert candidate["status"] == "candidate"
    assert candidate["activation_state"] == "candidate"
    assert candidate["required_metadata"] == ["artifact_context.dependency_copy=true"]
    assert candidate["reason_codes"] == []
    assert candidate["blocking_conditions"] == [
        "source_backed_dependency_copy_metadata_available",
        "executable_false_positive_fixture_passed",
        "reason_code_impact_audit_complete",
        "development_ablation_pre_registered",
        "confirmatory_evaluation_without_runtime_feedback",
    ]
    assert candidate["id"] not in active_profile["stage_c_policy_ids"]


def test_text_only_repeated_span_candidate_has_no_source_metadata_dependency() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    candidate = next(policy for policy in registry["policies"] if policy["id"] == "stage_c_repeated_span_template_candidate")
    active_profile = next(profile for profile in profiles["profiles"] if profile["id"] == "normal_structural_v1")

    assert candidate["core"] == "quality"
    assert candidate["status"] == "candidate"
    assert candidate["required_metadata"] == []
    assert candidate["reason_codes"] == ["repeated_exact_template_span_removed"]
    assert candidate["id"] not in active_profile["stage_c_policy_ids"]


def test_repeated_span_candidate_ablation_is_preregistered_without_runtime_feedback() -> None:
    preregistration = json.loads(ABLATION.read_text(encoding="utf-8"))

    assert preregistration["status"] == "preregistered_not_executed"
    assert preregistration["candidate_policy_id"] == "stage_c_repeated_span_template_candidate"
    assert preregistration["frozen_input_snapshot"]["required_fields"] == ["input_sha256", "policy_fingerprint", "stage_b_pass_sha256"]
    assert preregistration["arms"][0]["candidate_enabled"] is False
    assert preregistration["arms"][1]["candidate_enabled"] is True
    assert preregistration["external_evaluation"]["primary_comparison"] == "natural_budget_external_benchmark"
    assert preregistration["external_evaluation"]["equal_token_comparison"] == "not_primary_not_required"
    assert preregistration["external_evaluation"]["feedback_into_policy"] is False


def test_inline_license_header_candidate_is_text_only_and_not_active() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    candidate = next(policy for policy in registry["policies"] if policy["id"] == "stage_c_inline_license_header_candidate")
    active_profile = next(profile for profile in profiles["profiles"] if profile["id"] == "normal_structural_v1")

    assert candidate["core"] == "quality"
    assert candidate["status"] == "candidate"
    assert candidate["required_metadata"] == []
    assert candidate["reason_codes"] == ["inline_license_header_removed"]
    assert candidate["id"] not in active_profile["stage_c_policy_ids"]


def test_python_code_evidence_candidate_is_not_a_quality_score_or_runtime_rule() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    candidate = next(policy for policy in registry["policies"] if policy["id"] == "python_source_code_evidence_candidate")

    assert candidate["core"] == "validity"
    assert candidate["status"] == "candidate"
    assert candidate["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert candidate["required_metadata"] == ["language.code=python", "language.version"]
    assert candidate["reason_codes"] == [
        "python_syntax_error_source_candidate",
        "python_non_executable_stub_source_candidate",
    ]


def test_stage_c2_model_relative_candidate_is_frozen_evidence_only() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    candidate = next(policy for policy in registry["policies"] if policy["id"] == "stage_c2_model_relative_representative_candidate")

    assert candidate["core"] == "redundancy"
    assert candidate["status"] == "candidate"
    assert candidate["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert candidate["reason_codes"] == ["model_relative_redundant_family_member"]
