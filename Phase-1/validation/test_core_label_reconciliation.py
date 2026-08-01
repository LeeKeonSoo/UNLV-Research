#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_POLICY_ID = "stage_c_structural_scaffold"
FROZEN_PROTOCOLS = (
    "code_7m_normal_confirmatory_curation_v1.json",
    "code_7m_hard_confirmatory_curation_v1.json",
)


def test_representative_family_rules_are_redundancy_and_frozen_profiles_disable_near_duplicates() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    cards = json.loads((ROOT / "configs" / "policy_cards.json").read_text(encoding="utf-8"))
    registry_by_id = {policy["id"]: policy for policy in registry["policies"]}
    cards_by_id = {card["id"]: card for card in cards["cards"]}

    assert registry_by_id[SCAFFOLD_POLICY_ID]["core"] == "redundancy"
    assert cards_by_id[SCAFFOLD_POLICY_ID]["core"] == "redundancy"

    for protocol_name in FROZEN_PROTOCOLS:
        protocol = json.loads((ROOT / "protocols" / protocol_name).read_text(encoding="utf-8"))
        assert protocol["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is False


if __name__ == "__main__":
    test_representative_family_rules_are_redundancy_and_frozen_profiles_disable_near_duplicates()
    print("[core-label-reconciliation] core labels and frozen near-duplicate boundary: pass")
