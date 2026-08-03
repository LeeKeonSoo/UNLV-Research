#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from near_duplicate_calibration import build_near_duplicate_calibration


def main() -> int:
    # Given: the frozen Block 10A protocol and its deterministic metamorphic cases.
    report = build_near_duplicate_calibration(ROOT)
    frozen_path = ROOT / "validation/frozen_contracts/near_duplicate_calibration_v1.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    # When: every preregistered candidate setting is evaluated.
    assert report == frozen

    # Then: unresolved semantic collisions prevent a deletion threshold.
    assert report["schema_version"] == "near-duplicate-calibration-v1"
    assert report["status"] == "blocked_threshold_not_identifiable"
    assert report["domains"] == ["code", "math", "general"]
    assert report["target_token_counts"] == [24, 64, 128, 256]
    assert report["positive_case_count"] == 12
    assert report["hard_negative_case_count"] == 12
    assert report["eligible_positive_case_count"] == 9
    assert report["passing_setting_count"] == 0
    assert report["semantic_false_positive_count"] > 0
    assert report["feature_collision_count"] > 0
    assert report["collision_domains"] == ["general"]
    assert set(report["positive_miss_domains"]) == {"code", "math"}
    assert report["threshold_emitted"] is False
    decisions = {item["profile_id"]: item for item in report["operating_point_decisions"]}
    assert set(decisions) == {"normal", "hard"}
    assert all(item["status"] == "blocked" for item in decisions.values())
    assert all(item["threshold_emitted"] is False for item in decisions.values())
    assert report["same_policy_family"] is True
    assert report["hard_subset_of_normal_required"] is True
    assert report["safe_family_edge_authorized"] is False
    assert report["runtime_activation_mutated"] is False
    assert report["benchmark_outcomes_read"] is False
    assert report["utility_read"] is False
    assert report["recommended_disposition"] == "require_external_equivalence_witness"
    registry = json.loads((ROOT / "configs/framework_objects_v1.json").read_text(encoding="utf-8"))
    near_policy = next(
        policy
        for policy in registry["policies"]
        if policy["id"] == "redundancy.symmetric_near_duplicate_candidate"
    )
    assert near_policy["lifecycle"] == "blocked"
    assert near_policy["evidence"] == [
        {
            "path": "validation/frozen_contracts/near_duplicate_calibration_v1.json",
            "sha256": "33f3f80f01b99abbe3abdd6085fb08823c754e820f23b4b51ffd0fa6d208ffec",
        }
    ]
    print("[near-duplicate-calibration-v1] semantic collisions block threshold-only deletion: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
