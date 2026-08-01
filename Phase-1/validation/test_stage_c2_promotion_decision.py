from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage_c2_is_archived_not_promoted_to_the_frozen_runtime_profile() -> None:
    # Given: the frozen promotion decision and active policy registry.
    decision = json.loads((ROOT / "configs" / "stage_c2_promotion_decision.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))

    # When: the candidate lifecycle is checked before a 7M run.
    candidate = next(policy for policy in registry["policies"] if policy["id"] == decision["candidate_policy_id"])
    active = registry["runtime_profile_authorization"]["normal_structural_v1"]

    # Then: the candidate has no runtime authorization or promotion claim.
    assert decision["decision"] == "not_promoted_candidate_archive"
    assert candidate["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert candidate["id"] not in active["authorized_policy_ids"]
    assert candidate["id"] not in active["enabled_policy_ids"]
    assert "known_high_quality_reference_false_positive_risk" in decision["blocking_evidence"]


if __name__ == "__main__":
    test_stage_c2_is_archived_not_promoted_to_the_frozen_runtime_profile()
    print("[stage-c2-promotion-decision] candidate archived, active profile frozen: pass")
