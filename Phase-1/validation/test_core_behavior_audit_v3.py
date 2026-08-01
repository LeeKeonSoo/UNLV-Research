#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.core_behavior_audit_v3 import build_audit


def main() -> int:
    report = build_audit(ROOT / "validation" / "fixtures" / "core_behavior_audit_v3_cases.json")
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "configs" / "curation_contract.json").read_text(encoding="utf-8"))
    active_policy_ids = {policy["id"] for policy in registry["policies"] if policy["status"] == "active"}
    audited_policy_ids = active_policy_ids | {"stage_c_coverage_guard"}
    declared_reason_codes = {
        policy["id"]: set(policy["reason_codes"])
        for policy in registry["policies"]
        if policy["status"] == "active" and policy["reason_codes"]
    }
    observed_positive_codes: dict[str, set[str]] = {}
    for case in report["cases"]:
        if case["expected_trigger"] and case["policy_id"] in declared_reason_codes:
            observed_positive_codes.setdefault(case["policy_id"], set()).add(case["expected_code"])
    assert report["schema_version"] == "core-behavior-audit-v3"
    assert set(report["policies"]) == audited_policy_ids
    assert observed_positive_codes == declared_reason_codes
    assert report["summary"]["false_positives"] == 0
    assert report["summary"]["false_negatives"] == 0
    assert report["summary"]["behavior_invariant_failures"] == 0
    assert set(report["cores"]) == {"validity", "quality", "redundancy", "coverage"}
    assert all(core["behavior_gate_passed"] for core in report["cores"].values())
    assert report["cores"]["validity"]["required_dimensions"] == [
        "reason_coded_action",
        "non_trigger_retention",
    ]
    assert report["cores"]["redundancy"]["required_dimensions"] == [
        "representative_linkage",
        "representative_survival",
        "non_trigger_retention",
    ]
    assert report["cores"]["quality"]["required_dimensions"] == [
        "typed_deletion_authority",
        "observable_trigger",
        "false_positive_boundary",
    ]
    assert report["cores"]["coverage"]["required_dimensions"] == [
        "materialization_invariant_authority",
        "representative_linkage_detection",
        "zero_survivor_detection",
    ]
    coverage_cases = [case for case in report["cases"] if case["core"] == "coverage"]
    assert {case["fixture_kind"] for case in coverage_cases} == {
        "labeled_positive",
        "adversarial_negative",
    }
    assert all(case["behavior_invariants_passed"] for case in report["cases"])
    fixture_kinds = {case["fixture_kind"] for case in report["cases"]}
    assert {"metamorphic_positive", "adversarial_positive", "adversarial_negative"} <= fixture_kinds
    assert report["case_matrix_reconciliation"] == {"passed": True, "mismatches": []}
    assert contract["core_behavior_audit"]["required_dimensions"] == {
        core: details["required_dimensions"] for core, details in report["cores"].items()
    }
    for policy in report["policies"].values():
        assert policy["positive_cases"] >= 1
        assert policy["negative_cases"] >= 1
    print("[core-behavior-audit-v3] labeled policy behavior and false-positive gate: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
