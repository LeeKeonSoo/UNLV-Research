#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_policy_registry_declares_activation_and_safety_evidence() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "validation" / "fixtures" / "core_case_matrix.json").read_text(encoding="utf-8"))
    matrix_ids = {case["id"] for case in matrix["cases"]}

    assert registry["schema_version"] == "core-policy-registry-v2"
    assert registry["authoritative_policy_definition"] == "configs/policy_cards.json"
    assert registry["activation_states"] == [
        "candidate",
        "active_structural",
        "development_validated",
        "confirmatory_validated",
        "retired",
    ]
    normal = registry["runtime_profile_authorization"]["normal_structural_v1"]
    assert normal["stage_a_policy"] == "text_only_v2"
    assert set(normal["excluded_policy_ids"]) >= {
        "stage_a_provenance_contract",
        "stage_a_risk_quarantine",
        "stage_c_coverage_guard",
        "stage_c_declared_dependency_copy_candidate",
    }

    for policy in registry["policies"]:
        if policy["status"] != "active":
            continue
        assert policy["activation_state"] == "active_structural"
        assert policy["policy_card_id"] == policy["id"]
        assert (ROOT / policy["false_positive_fixture"]).is_file()
        assert policy["case_matrix_scenario"] in matrix_ids
        assert policy["coverage_impact_validation"]
        assert policy["promotion_requirements"] == [
            "structural_fixture_passed",
            "reason_code_audit_complete",
            "development_ablation_pre_registered",
            "confirmatory_evaluation_without_runtime_feedback",
        ]


def test_registry_and_policy_cards_cannot_drift_on_runtime_decisions() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    cards = json.loads((ROOT / "configs" / "policy_cards.json").read_text(encoding="utf-8"))
    cards_by_id = {card["id"]: card for card in cards["cards"]}

    for policy in registry["policies"]:
        if policy["status"] != "active":
            continue
        card = cards_by_id[policy["policy_card_id"]]
        assert policy["version"] == card["version"]
        assert policy["reason_codes"] == card["reason_codes"]
        assert policy["runtime_implementation"] == card["runtime_implementation"]
