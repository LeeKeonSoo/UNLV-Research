#!/usr/bin/env python3
"""Validate the operational LM-curation framework contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "lm_curation_operational_framework_v1.json"


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "active_operational_target"
    assert contract["utility_scope"] == "External Evaluation Protocol only; never Stage-C selector objective"

    selection_value = contract["core_interpretation"]["selection_value"]
    assert selection_value["role"] == "observable_pre_outcome_selection_evidence"
    assert "not intrinsic" in selection_value["claim_boundary"]
    assert "no Stage-B hard-reject authority" in selection_value["claim_boundary"]

    quality_alias = contract["core_interpretation"]["quality"]
    assert quality_alias["role"] == "legacy_alias_for_selection_value_evidence"
    assert "must not authorize rejection" in quality_alias["claim_boundary"]

    assert "utility" not in contract["core_interpretation"]

    stage_c_forbidden = set(contract["stage_contract"]["stage_c"]["forbidden"])
    for forbidden in (
        "Utility",
        "benchmark outcomes",
        "retention outcomes",
        "development model outcomes",
        "confirmatory model outcomes",
        "human review labels",
        "LLM review labels",
    ):
        assert forbidden in stage_c_forbidden

    actions = set(contract["stage_contract"]["decision_release"]["allowed_actions"])
    assert "insufficient_usable_data" in actions
    assert "abstain" in actions
    assert "accept" in actions
    assert "retain_all" in actions
    assert "full_curated_pool" in actions
    assert "budgeted_training_subset" in actions

    dispositions = contract["disposition_contract"]
    assert set(dispositions["curation_dispositions"]) == {
        "retained",
        "rejected",
        "quarantined",
    }
    assert "budget_not_selected" in dispositions["training_budget_dispositions"]
    invariant_blob = " ".join(dispositions["invariants"])
    assert "Budget-not-selected is not rejection" in invariant_blob
    assert "Retain-all is a valid expected outcome" in invariant_blob
    assert "No fixed rejection quota" in invariant_blob

    forbidden_claims = contract["forbidden_claims"]
    assert any("intrinsic data quality" in claim for claim in forbidden_claims)
    assert any("Every arbitrary candidate corpus" in claim for claim in forbidden_claims)
    assert any("Curation must always reduce" in claim for claim in forbidden_claims)
    assert any("budget-not-selected" in claim for claim in forbidden_claims)

    baselines = set(contract["external_evaluation_operational_readiness"]["required_baselines"])
    assert "raw_random_equal_budget" in baselines
    assert "stageA_random_equal_budget" in baselines

    improvements = set(contract["stage_c_operational_improvements"]["required_before_claiming_operational_readiness"])
    assert "split harmful duplication from useful recurrence" in improvements
    assert "preserve concise but useful examples, tests, bug fixes, and API usage chunks" in improvements

    print("[lm-curation-operational-framework] contract preserves practical framework target: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
