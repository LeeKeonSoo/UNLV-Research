#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_policy_ablation import build_policy_ablation


def main() -> int:
    report = build_policy_ablation(ROOT)
    frozen_path = ROOT / "validation/frozen_contracts/framework_policy_ablation_v1.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    assert report == frozen
    assert report["schema_version"] == "framework-policy-ablation-v1"
    assert report["status"] == "block_9_complete_no_hard_policy_promoted"
    assert report["development_admission"]["passed"] is True
    assert report["development_admission"]["benchmark_contaminated_records"] == 0
    assert report["development_admission"]["confirmatory_record_overlap"] == 0
    assert report["development_admission"]["confirmatory_text_overlap"] == 0

    decisions = {item["policy_id"]: item for item in report["policy_decisions"]}
    assert set(decisions) == {
        "redundancy.exact_text_family",
        "redundancy.symmetric_near_duplicate_candidate",
        "quality.teacher_panel_candidate",
    }
    exact = decisions["redundancy.exact_text_family"]
    assert exact["lifecycle_decision"] == "development_passed"
    assert exact["positive_units"] == 2400
    assert exact["false_positive_units"] == 0
    assert exact["representative_failures"] == 0

    near = decisions["redundancy.symmetric_near_duplicate_candidate"]
    assert near["lifecycle_decision"] == "blocked"
    assert near["threshold_emitted"] is False
    assert near["candidate_units"] == 860
    assert "near_positive_nonexact_equivalence_missing" in near["blocker_codes"]

    teacher_panel = decisions["quality.teacher_panel_candidate"]
    assert teacher_panel["lifecycle_decision"] == "blocked"
    assert teacher_panel["operating_point_emitted"] is False
    assert teacher_panel["teacher_count"] == 3
    assert teacher_panel["policy_count"] == 4
    assert teacher_panel["smoke_fixture_target"] == 512
    assert teacher_panel["protected_fixture_target"] == 800
    assert "quality_teacher_protected_fixture_gate_missing" in teacher_panel["blocker_codes"]

    assert report["hard_profile_development_ready"] is False
    assert report["block_10_authorized"] is False
    assert report["benchmark_outcomes_read"] is False
    assert report["utility_read"] is False
    assert report["runtime_activation_mutated"] is False

    registry = json.loads((ROOT / "configs/framework_objects_v1.json").read_text(encoding="utf-8"))
    exact_registry_policy = next(
        policy for policy in registry["policies"] if policy["id"] == "redundancy.exact_text_family"
    )
    assert exact_registry_policy["lifecycle"] == "development_passed"
    assert exact_registry_policy["evidence"] == [
        {
            "path": "validation/frozen_contracts/framework_policy_ablation_v1.json",
            "sha256": "d6d1f8caa55ca40c27fc43672a84645d2ec37f587678a76b69402cef3965346f",
        }
    ]
    print("[framework-policy-ablation-v1] exact passed; near and teacher panel blocked: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
