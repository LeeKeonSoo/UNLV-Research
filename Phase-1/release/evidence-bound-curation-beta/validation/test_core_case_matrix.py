#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    registry = json.loads((ROOT / "configs" / "runtime_policy_registry_v1.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "validation" / "fixtures" / "core_case_matrix.json").read_text(encoding="utf-8"))
    policies = {policy["id"]: policy for policy in registry["policies"]}
    active_policy_ids = set(policies)
    matrix_policy_ids = {policy_id for case in matrix["cases"] for policy_id in case["policy_ids"]}
    assert matrix["schema_version"] == "core-case-matrix-v1"
    assert {case["core"] for case in matrix["cases"]} == {"validity", "redundancy", "quality", "coverage"}
    assert active_policy_ids <= matrix_policy_ids
    for case in matrix["cases"]:
        assert case["policy_ids"]
        for policy_id in case["policy_ids"]:
            if policy_id not in policies:
                continue
            assert policy_id in policies
            assert policies[policy_id]["core"] == case["core"]
            assert (ROOT / policies[policy_id]["positive_fixture"]).is_file()
    print("[core-case-matrix] every Core case maps to a policy and executable fixture: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
