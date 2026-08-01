#!/usr/bin/env python3
"""Freeze canonical Stage-C guardrail execution after the redundancy proxy decision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_EVALUATION_INPUTS = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json"
)
DEFAULT_DECISION = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_proxy_decision_report.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs" / "temporal_code_redundancy_canonical_guardrails_qwen25_0p5b_v1.json"
)


def build(
    experiment_path: Path,
    evaluation_inputs_path: Path,
    decision_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    experiment = load_json(experiment_path)
    evaluation_inputs = load_json(evaluation_inputs_path)
    decision = load_json(decision_path)
    blockers = []
    if decision.get("canonical_control") != "binary_current":
        blockers.append("binary_current_not_canonical")
    if decision.get("promotion_allowed") is not False:
        blockers.append("candidate_futility_boundary_not_frozen")
    if evaluation_inputs.get("confirmatory_outcomes_read") is not False:
        blockers.append("confirmatory_outcomes_already_read")

    source_sha256 = {
        str(experiment_path): sha256_file(experiment_path),
        str(evaluation_inputs_path): sha256_file(evaluation_inputs_path),
        str(decision_path): sha256_file(decision_path),
    }
    guardrails = {
        "schema_version": "temporal-code-redundancy-canonical-guardrails-v1",
        "status": (
            "frozen_before_canonical_guardrail_outcomes"
            if not blockers
            else "canonical_guardrail_freeze_blocked"
        ),
        "purpose": (
            "Evaluate the canonical binary recurrence selector path against base_no_update "
            "on frozen Stage-C general-task and EvalPlus development guardrails."
        ),
        "source_sha256": source_sha256,
        "target_model": experiment["target_model"],
        "training_arms": ["binary_current_equal_budget"],
        "excluded_arms": {
            "log_count_equal_budget": "candidate promotion is futile after the frozen directional non-worse failure",
            "stageA_random_common_disjoint_equal_budget": (
                "operational target-NLL baseline, not a release candidate requiring retention certification"
            ),
        },
        "training_recipe": {
            **experiment["training_recipe"],
            "development_training_seeds": experiment["training_recipe"]["seeds"],
        },
        "external_code_guardrails": {
            "maximum_allowed_absolute_regression_per_suite": evaluation_inputs[
                "code_retention"
            ]["maximum_allowed_absolute_regression_per_suite"],
            "maximum_allowed_absolute_regression_macro_vs_base": evaluation_inputs[
                "code_retention"
            ]["maximum_allowed_absolute_regression_macro"],
        },
        "general_task_guardrail": evaluation_inputs["general_task_retention"],
        "evalplus_guardrail": evaluation_inputs["code_retention"],
        "required_jobs": {
            "general_task": {
                "base_evaluated_once": True,
                "trained_arm_seed_jobs": 3,
                "tasks_per_job": 4,
            },
            "evalplus_development": {
                "base_evaluated_once": True,
                "trained_arm_seed_jobs": 3,
                "tasks_per_job": 284,
                "suites": ["HumanEval+", "MBPP+"],
                "isolated_execution_support_tier": "E2",
            },
        },
        "decision_rule": {
            "reference_arm": "base_no_update",
            "all_guardrails_mandatory": True,
            "missing_evidence_action": "abstain",
            "failed_guardrail_action": "reject_canonical_release",
            "development_evidence_cannot_be_called_confirmatory": True,
        },
        "forbidden_uses": [
            "feeding guardrail outcomes into Stage-B selection",
            "running further log_count guardrails after the frozen futility boundary",
            "changing tasks, margins, seeds, or model snapshot after outcomes",
            "calling development EvalPlus evidence confirmatory",
        ],
        "confirmatory_outcomes_read": False,
        "blockers": blockers,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Canonical Qwen2.5-0.5B development guardrail execution contract only. "
            "Passing does not by itself establish production release or cross-model generality."
        ),
    }
    save_json(output_path, guardrails)
    return guardrails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--evaluation-inputs", type=Path, default=DEFAULT_EVALUATION_INPUTS)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.experiment, args.evaluation_inputs, args.decision, args.output)
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
