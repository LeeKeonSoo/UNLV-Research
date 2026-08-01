#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    assert registry["schema_version"] == "core-policy-registry-v2"
    assert registry["authoritative_policy_definition"] == "configs/policy_cards.json"
    policies = registry["policies"]
    assert registry["canonical_cores"] == ["validity", "redundancy", "coverage", "quality"]
    assert {policy["core"] for policy in policies} == set(registry["canonical_cores"])
    assert registry["legacy_core_aliases"] == {
        "selection_value_evidence": "quality",
        "structural_compression": "quality",
    }
    quality_contract = registry["core_decision_contracts"]["quality"]
    assert quality_contract["runtime_owner"] == "quality_retention.py"
    assert quality_contract["decisions"] == ["keep", "reject", "abstain_retain"]
    assert quality_contract["abstain_action"] == "retain"
    assert quality_contract["decision_contract"] == "quality_retention_deletion_authority_v2"
    assert quality_contract["missing_trace_action"] == "reject_decision_construction_fails"
    assert quality_contract["intrinsic_quality_score_used"] is False
    route_quality = registry["core_decision_contracts"]["candidate_route_conditioned_quality_v2"]
    assert route_quality["required_heads"] == ["substantive_payload", "route_specific_evidence"]
    assert route_quality["routing_precondition"]["quality_evidence"] is False
    assert route_quality["candidate_evidence_registry"] == "configs/route_quality_evidence_candidates_v1.json"
    assert set(route_quality["registered_route_evidence_status"]) == {
        "general_prose", "code_artifact", "mathematical_content",
        "technical_documentation", "conversation", "instruction", "table_structured_data",
    }
    active = [policy for policy in policies if policy["status"] == "active"]
    assert all(policy["reason_codes"] or policy["id"] == "stage_a_bom_normalization" for policy in active)
    assert all(policy["activation_state"] == "active_structural" for policy in active)
    assert all((ROOT / policy["false_positive_fixture"]).is_file() for policy in active)
    assert all(policy["case_matrix_scenario"] for policy in active)
    assert all(policy["coverage_impact_validation"] for policy in active)
    assert all((ROOT / policy["fixture"]).is_file() for policy in policies)
    print("[core-policy-registry] active and diagnostic policy evidence: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
