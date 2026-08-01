#!/usr/bin/env python3
"""Contract checks for orthogonal content and execution-support tiers."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(PROJECT_DIR / "outputs" / "validation" / "temporal_code_execution_support_report.json")
    contract = report["contract"]
    decision = report["decision"]
    assert report["status"] == "orthogonal_content_and_execution_tiers_operational"
    assert set(contract["orthogonal_axes"]) == {"training_content", "execution_support"}
    assert set(contract["task_class_e2_contracts"]) == {"repository_patch", "function_generation", "other"}
    assert contract["stage_entry_rules"]["stage_b_selector"] == "must not use execution tier"
    assert contract["stage_entry_rules"]["utility"] == "Stage C validation only; never selector objective"
    assert decision["training_content_may_be_preserved_without_executable_support"] is True
    assert decision["executable_stage_c_requires_e2"] is True
    assert decision["execution_tier_may_enter_stage_b"] is False
    assert decision["development_utility_may_start"] is False
    assert report["confirmatory_outcomes_read"] is False
    assert report["summary"]["training_content_eligible_count"] > report["summary"][
        "executable_stage_c_eligible_count"
    ]
    print("[temporal-code-execution-support] orthogonal content/execution tier boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
