#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage_b_policy import propose_stage_b_removals


def test_redundancy_and_quality_traces_are_owned_by_stage_b() -> None:
    selected, removed, audit = propose_stage_b_removals(
        ({"chunk_uid": "a", "text": "Substantive payload remains.", "token_proxy": 3},),
        {
            "near_duplicate_compaction": {
                "candidate_enabled": False,
                "shingle_size": 5,
                "minimum_lexical_tokens": 40,
                "symmetric_overlap_threshold": 0.95,
            },
            "structural_scaffold_compaction": {"enabled": False},
            "structural_artifact_rules": {},
        },
    )

    assert removed == []
    assert "stage_b_policy" in selected[0]
    assert "stage_c_selection" not in selected[0]
    quality = selected[0]["quality_retention_decision"]
    assert all(
        not str(policy_id).startswith("stage_c_")
        for policy_id in quality["evaluated_policy_ids"]
    )
    assert audit["owner_stage"] == "Stage B"
    assert audit["core_ids"] == ["redundancy", "quality"]
    assert audit["proposal_is_final_membership"] is False


if __name__ == "__main__":
    test_redundancy_and_quality_traces_are_owned_by_stage_b()
    print("[stage-b-policy-ownership-v1] trace migration: pass")
