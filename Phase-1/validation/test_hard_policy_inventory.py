#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "configs" / "hard_policy_inventory_v1.json"
PROFILES = ROOT / "configs" / "policy_profiles.json"


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    hard = next(profile for profile in profiles["profiles"] if profile["id"] == "hard_structural_v1")

    assert inventory["schema_version"] == "hard-policy-inventory-v1"
    assert inventory["base_profile"] == "normal_structural_v1"
    assert inventory["runtime_authorization"] == "none_candidates_cannot_select_transform_or_remove"
    assert inventory["selected_initial_candidates"] == [
        "stage_b_inline_license_header_candidate",
        "stage_b_inline_license_comment_block_candidate",
        "stage_b_repeated_span_template_candidate",
    ]
    assert inventory["promotion_requirements"] == [
        "positive_false_positive_adversarial_clean_fixtures",
        "reason_code_and_token_delta",
        "residual_payload_validity",
        "representative_or_span_trace",
        "coverage_invariant",
        "code_math_general_development_ablation",
        "benchmark_disjoint_confirmatory_evaluation",
    ]
    excluded = {item["policy_id"]: item["reason"] for item in inventory["excluded_from_hard_v1"]}
    assert "stage_c2_model_relative_representative_candidate" in excluded
    assert "stage_c_strengthened_symmetric_near_duplicate_candidate" in excluded
    assert hard["inherits_profile"] == "normal_structural_v1"
    assert hard["candidate_policy_ids"] == inventory["selected_initial_candidates"]

    print("[hard-policy-inventory] deterministic payload-preserving Hard v1 scope: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
