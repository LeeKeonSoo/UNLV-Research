#!/usr/bin/env python3
"""Contract checks for frozen temporal-code retention guardrails."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    plan = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
    )
    contract = plan["contract"]
    assert plan["status"] == "frozen_before_development_model_outcomes"
    assert contract["code_guardrail"]["maximum_allowed_absolute_regression_macro"] == 0.02
    assert contract["general_task_guardrail"]["maximum_allowed_absolute_regression_macro"] == 0.01
    assert contract["general_text_guardrail"]["maximum_allowed_mean_nll_increase"] == 0.01
    assert contract["decision_rule"]["all_guardrails_mandatory"] is True
    assert contract["decision_rule"]["missing_evidence_action"] == "abstain"
    assert len(contract["seed_contract"]["development_training_seeds"]) == 5
    assert len(contract["seed_contract"]["confirmatory_training_seeds"]) == 5
    assert plan["development_utility_may_start"] is False
    assert plan["confirmatory_outcomes_read"] is False
    assert "using retention outcomes in Stage B" in contract["forbidden_uses"]
    print("[temporal-code-retention] Stage-C non-inferiority guardrails frozen outcome-free: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
