#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reference_distribution_is_diagnostic_only() -> None:
    registry = json.loads((ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    policy = next(item for item in registry["policies"] if item["id"] == "stage_c_reference_distribution_diagnostic")
    calibrated = next(item for item in profiles["profiles"] if item["id"] == "calibrated_selector_template_v1")

    assert policy["status"] == "diagnostic"
    assert policy["reason_codes"] == []
    assert policy["authorization"] == "diagnostic_only_no_selection_or_removal"
    assert policy["forbidden_inputs"] == ["utility", "benchmark_outcomes", "target_token_fraction"]
    assert calibrated["selector"]["reference_distribution_score_status"] == "excluded_from_stage_c_until_new_policy_card_passes"


if __name__ == "__main__":
    test_reference_distribution_is_diagnostic_only()
    print("[reference-distribution-policy-boundary] diagnostic-only selector boundary: pass")
