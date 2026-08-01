#!/usr/bin/env python3
"""Regression tests for deployment-contract-conditioned release decisions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from release_policy import decide_release  # noqa: E402
from data_eval_common import load_json  # noqa: E402


def _contract(objective: str, evaluation: str, eligible: list[str], preference: list[str], guardrails=None):
    return {
        "schema_version": "deployment-contract-v1",
        "contract_name": objective,
        "objective_type": objective,
        "primary_outcome": {
            "evaluation": evaluation,
            "direction": "lower_is_better",
            "comparison_reference": "stageA_broad",
            "minimum_improvement": 0.0,
            "reference_requires_base_gain": objective == "broad_refresh",
        },
        "guardrails": guardrails or [],
        "eligible_release_actions": eligible,
        "preference_order": preference,
        "claim_scope": objective,
        "utility_scope": "Stage C validation only; never selector objective",
    }


def main() -> int:
    evidence = {
        "usable_data_sufficient": True,
        "arms": {
            "base_no_update": {"evaluations": {"broad": 2.0, "target": 2.0}},
            "selected_only": {"evaluations": {"broad": 2.2, "target": 2.2}},
            "coverage_backfilled": {"evaluations": {"broad": 1.99, "target": 1.8}},
            "stageA_broad": {"evaluations": {"broad": 1.98, "target": 1.9}},
        },
    }
    broad = _contract(
        "broad_refresh",
        "broad",
        ["selected_only", "coverage_backfilled", "stageA_broad", "reject", "insufficient_usable_data"],
        ["selected_only", "coverage_backfilled", "stageA_broad"],
    )
    broad_result = decide_release(broad, evidence)
    assert broad_result["release_action"] == "stageA_broad", broad_result

    targeted = _contract(
        "targeted_update",
        "target",
        ["selected_only", "coverage_backfilled", "reject", "insufficient_usable_data"],
        ["selected_only", "coverage_backfilled"],
        guardrails=[
            {
                "evaluation": "broad",
                "direction": "lower_is_better",
                "comparison_reference": "base_no_update",
                "maximum_regression": 0.0,
                "required": True,
            }
        ],
    )
    targeted_result = decide_release(targeted, evidence)
    assert targeted_result["release_action"] == "coverage_backfilled", targeted_result

    capability_preserving = _contract(
        "capability_preserving_update",
        "target",
        ["selected_only", "coverage_backfilled", "reject", "insufficient_usable_data"],
        ["selected_only", "coverage_backfilled"],
        guardrails=[
            {
                "evaluation": "external_general_capability",
                "direction": "higher_is_better",
                "comparison_reference": "base_no_update",
                "maximum_regression": 0.01,
                "required": True,
            }
        ],
    )
    capability_result = decide_release(capability_preserving, evidence)
    assert capability_result["release_action"] == "reject", capability_result

    capability_config = load_json(PROJECT_DIR / "configs" / "deployment_contract_capability_preserving_update.json")
    assert capability_config["primary_outcome"]["direction"] == "lower_is_better"
    nll_guardrails = {
        row["evaluation"]: row["direction"]
        for row in capability_config["guardrails"]
        if row["evaluation"] in {"general_capability_eval", "forgetting_regression_eval"}
    }
    assert set(nll_guardrails.values()) == {"lower_is_better"}, nll_guardrails

    insufficient = dict(evidence)
    insufficient["usable_data_sufficient"] = False
    insufficient_result = decide_release(broad, insufficient)
    assert insufficient_result["release_action"] == "insufficient_usable_data", insufficient_result

    print("[release-policy-contract] broad objective selects Stage-A broad: pass")
    print("[release-policy-contract] targeted objective selects coverage backfill: pass")
    print("[release-policy-contract] missing capability guardrail forces rejection: pass")
    print("[release-policy-contract] capability NLL directions are lower-is-better: pass")
    print("[release-policy-contract] insufficient usable data forces abstention: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
