#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_POLICY_ID = "stage_c_structural_scaffold"
def test_representative_family_rules_and_near_duplicates_belong_to_redundancy() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    cards = json.loads((ROOT / "configs" / "policy_cards.json").read_text(encoding="utf-8"))
    registry_by_id = {policy["id"]: policy for policy in registry["policies"]}
    cards_by_id = {card["id"]: card for card in cards["cards"]}

    assert registry_by_id[SCAFFOLD_POLICY_ID]["core"] == "redundancy"
    assert cards_by_id[SCAFFOLD_POLICY_ID]["core"] == "redundancy"

    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    for profile in profiles["profiles"]:
        if profile.get("user_facing_mode") not in {"normal", "hard"}:
            continue
        assert profile["runtime_policy"]["redundancy_v2"]["runtime_activation"] is True
        assert "stage_b_symmetric_near_duplicate" in profile["enabled_policy_ids"]


if __name__ == "__main__":
    test_representative_family_rules_and_near_duplicates_belong_to_redundancy()
    print("[core-label-reconciliation] Core labels and active near-duplicate boundary: pass")
