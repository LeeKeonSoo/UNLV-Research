#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.core_behavior_audit_v3 import build_audit


CONTRACT_PATH = ROOT / "validation" / "fixtures" / "policy_fixture_contract_v1.json"


def test_every_active_policy_has_executable_positive_and_false_positive_boundaries() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "configs" / "runtime_policy_registry_v1.json").read_text(encoding="utf-8"))
    behavior_path = ROOT / contract["behavior_fixture"]
    behavior_cases = json.loads(behavior_path.read_text(encoding="utf-8"))["cases"]
    behavior_by_id = {case["id"]: case for case in behavior_cases}
    report_by_id = {case["id"]: case for case in build_audit(behavior_path)["cases"]}
    contract_by_policy = {item["policy_id"]: item for item in contract["policies"]}
    active = registry["policies"]

    assert contract["schema_version"] == "policy-fixture-contract-v1"
    assert {policy["id"] for policy in active} <= set(contract_by_policy)
    for policy in active:
        fixture = contract_by_policy[policy["id"]]
        assert (ROOT / policy["false_positive_fixture"]).is_file()
        assert fixture["positive_case_ids"]
        assert fixture["false_positive_case_ids"]
        for case_id in fixture["positive_case_ids"]:
            assert behavior_by_id[case_id]["policy_id"] == policy["id"]
            assert report_by_id[case_id]["outcome"] == "true_positive"
        for case_id in [*fixture["false_positive_case_ids"], *fixture["adversarial_case_ids"]]:
            assert behavior_by_id[case_id]["policy_id"] == policy["id"]
            assert report_by_id[case_id]["outcome"] == "true_negative"


if __name__ == "__main__":
    test_every_active_policy_has_executable_positive_and_false_positive_boundaries()
    print("[policy-fixture-contract] executable rule boundaries: pass")
