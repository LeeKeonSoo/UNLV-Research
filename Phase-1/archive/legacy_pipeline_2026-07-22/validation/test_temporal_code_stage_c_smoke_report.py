#!/usr/bin/env python3
"""Contract checks for Qwen3-4B Stage-C smoke evidence."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(PROJECT_DIR / "outputs" / "validation" / "temporal_code_stage_c_smoke_report.json")
    summary = report["summary"]
    assert report["status"] == "qlora_stage_c_smoke_feasibility_pass"
    assert summary["all_arms_completed"] is True
    assert summary["equal_packed_token_budget"] is True
    assert summary["equal_optimizer_steps"] is True
    assert summary["equal_seed"] is True
    assert summary["common_stage_a_baseline_shared"] is True
    assert summary["curated_common_baseline_overlap_count"] == 0
    assert report["confirmatory_outcomes_read"] is False
    assert "not Utility" in report["training_loss_interpretation"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-stage-c-smoke-report] equal-budget QLoRA feasibility boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
