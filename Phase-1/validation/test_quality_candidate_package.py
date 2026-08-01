#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_POLICY_ID = "stage_c_explicit_web_control_span_candidate"


def test_web_control_candidate_has_complete_non_runtime_package() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    cards = json.loads((ROOT / "configs" / "policy_cards.json").read_text(encoding="utf-8"))
    registry_by_id = {policy["id"]: policy for policy in registry["policies"]}
    cards_by_id = {card["id"]: card for card in cards["cards"]}

    registry_policy = registry_by_id[CANDIDATE_POLICY_ID]
    card = cards_by_id[CANDIDATE_POLICY_ID]

    assert registry_policy["core"] == "quality"
    assert registry_policy["status"] == "candidate"
    assert registry_policy["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert registry_policy["policy_card_id"] == CANDIDATE_POLICY_ID
    assert registry_policy["fixture"] == "validation/test_general_web_span_compaction.py"
    assert registry_policy["coverage_impact_validation"] == "residual_payload_required_before_promotion"
    assert card["reason_codes"] == ["web_control_span_removed", "url_directory_span_removed"]
    assert card["allowed_inputs"] == ["chunk text", "declared Stage-B minimum_residual_chars"]
    assert "dialogue-like speaker turns remain" in card["negative_conditions"]
    assert "benchmark_outcomes" in card["forbidden_inputs"]


if __name__ == "__main__":
    test_web_control_candidate_has_complete_non_runtime_package()
    print("[quality-candidate-package] web-control candidate package: pass")
