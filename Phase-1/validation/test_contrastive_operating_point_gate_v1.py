#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_operating_point_gate import build_contrastive_operating_point_gate


def main() -> int:
    # Given: the frozen two-model audit and the three-role v2 protocol.
    report = build_contrastive_operating_point_gate(ROOT)

    # When: Block 10B checks every prerequisite for profile thresholds.
    assert report["schema_version"] == "contrastive-operating-point-gate-v1"
    assert report["status"] == "blocked_missing_empirical_inputs"
    assert report["required_routes"] == [
        "code_artifact",
        "mathematical_content",
        "general_prose",
    ]
    assert report["common_baseline_shared_by_all_arms"] is False
    assert report["baseline_disjoint_from_every_arm"] is False
    assert report["sensitivity_arm_count"] == 0
    assert report["qualified_three_role_provider"] is False
    assert report["external_natural_budget_evidence_present"] is False
    assert report["profile_operating_point_artifacts_present"] is False
    assert report["hard_subset_of_normal_verified"] is False
    route_gates = {item["route"]: item for item in report["route_gates"]}
    assert set(route_gates) == {
        "code_artifact",
        "mathematical_content",
        "general_prose",
    }
    assert all(item["observed_source_group_count"] == 2 for item in route_gates.values())
    assert all(item["observed_effect_bin_count"] == 0 for item in route_gates.values())
    assert all(item["ready"] is False for item in route_gates.values())
    decisions = {item["profile_id"]: item for item in report["operating_point_decisions"]}
    assert set(decisions) == {"normal", "hard"}
    assert all(item["status"] == "blocked" for item in decisions.values())
    assert all(item["threshold_emitted"] is False for item in decisions.values())
    assert all(item["artifact_sha256"] is None for item in decisions.values())
    assert "profile_operating_point_artifacts_missing" in report["blocker_codes"]
    assert "profile_monotonicity_evidence_missing" in report["blocker_codes"]
    assert report["runtime_activation_mutated"] is False
    assert report["benchmark_outcomes_read_at_runtime"] is False
    assert report["utility_read_at_runtime"] is False
    frozen = json.loads(
        (ROOT / "validation/frozen_contracts/contrastive_operating_point_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report == frozen
    print("[contrastive-operating-point-gate-v1] missing empirical inputs fail closed: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
