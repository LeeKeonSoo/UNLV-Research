#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_POLICY_ID = "stage_c_repeated_label_block_candidate"


def test_repeated_label_block_candidate_has_complete_non_runtime_package() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    cards = json.loads((ROOT / "configs" / "policy_cards.json").read_text(encoding="utf-8"))
    registry_by_id = {policy["id"]: policy for policy in registry["policies"]}
    cards_by_id = {card["id"]: card for card in cards["cards"]}

    registry_policy = registry_by_id[CANDIDATE_POLICY_ID]
    card = cards_by_id[CANDIDATE_POLICY_ID]

    assert registry_policy["core"] == "redundancy"
    assert registry_policy["status"] == "retired"
    assert registry_policy["runtime_authorization"] == "none_retired_cannot_select_or_remove"
    assert registry_policy["fixture"] == "validation/test_repeated_line_block_compaction.py"
    assert registry_policy["coverage_impact_validation"] == "first_occurrence_and_residual_payload_required"
    assert card["reason_codes"] == ["repeated_label_block_removed"]
    assert card["allowed_inputs"] == ["chunk text", "declared Stage-B minimum_residual_chars"]
    assert card["empirical_status"] == "retired_zero_observed_spans_in_development"
    assert "benchmark_outcomes" in card["forbidden_inputs"]


if __name__ == "__main__":
    test_repeated_label_block_candidate_has_complete_non_runtime_package()
    print("[redundancy-candidate-package] repeated-label candidate package: pass")
