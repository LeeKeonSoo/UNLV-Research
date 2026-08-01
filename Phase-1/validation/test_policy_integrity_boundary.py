#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_active_registry_policy_has_a_versioned_integrity_card() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    cards = json.loads((ROOT / "configs" / "policy_cards.json").read_text(encoding="utf-8"))
    active_ids = {policy["id"] for policy in registry["policies"] if policy["status"] == "active"}
    registry_ids = {policy["id"] for policy in registry["policies"]}
    registry_core_by_id = {policy["id"]: policy["core"] for policy in registry["policies"] if policy["status"] == "active"}
    card_by_id = {card["id"]: card for card in cards["cards"]}

    assert active_ids.issubset(card_by_id)
    assert set(card_by_id).issubset(registry_ids)
    assert set(registry["canonical_cores"]) == {"validity", "redundancy", "coverage", "quality"}
    for policy_id in active_ids:
        card = card_by_id[policy_id]
        assert card["core"] == registry_core_by_id[policy_id]
        assert card["version"]
        assert card["runtime_implementation"]
        assert card["deployment_scope"]
        assert card["negative_conditions"]
        assert set(cards["runtime_forbidden_inputs"]).issubset(card["forbidden_inputs"])
        expected_status = (
            "runtime_materialization_invariant_not_selector"
            if policy_id == "stage_c_coverage_guard"
            else "unvalidated_structural_policy"
        )
        assert card["empirical_status"] == expected_status


def test_boundary_contract_forbids_quality_and_runtime_evaluation_feedback() -> None:
    contract = json.loads((ROOT / "configs" / "curation_contract.json").read_text(encoding="utf-8"))

    assert contract["curation_engine"]["curated_output"] == "full_reason_coded_curated_pool"
    assert set(contract["runtime_forbidden_inputs"]) >= {
        "intrinsic_quality_score",
        "human_quality_label",
        "Utility",
        "NLL",
        "benchmark_outcomes",
        "target_retention_fraction",
        "source_pool_role",
        "source_tier",
        "composition",
    }
    assert contract["external_evaluation"]["may_mutate_or_reselect_frozen_output"] is False


if __name__ == "__main__":
    test_every_active_registry_policy_has_a_versioned_integrity_card()
    test_boundary_contract_forbids_quality_and_runtime_evaluation_feedback()
    print("[policy-integrity-boundary] versioned policy cards and runtime boundary: pass")
