#!/usr/bin/env python3
"""Build the frozen redundancy proxy candidate decision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_EVAL_INPUTS = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json"
)
DEFAULT_TRAINING = (
    ROOT
    / "outputs"
    / "redundancy_saturation_proxy_qwen25_0p5b_v1"
    / "qlora_runs"
    / "train_missing_summary.json"
)
DEFAULT_NLL = (
    ROOT
    / "outputs"
    / "redundancy_saturation_proxy_qwen25_0p5b_v1"
    / "nll_evaluation"
    / "nll_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_decision_report.json"
)


def build(
    plan_path: Path,
    eval_inputs_path: Path,
    training_path: Path,
    nll_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_inputs = load_json(eval_inputs_path)
    training = load_json(training_path)
    nll = load_json(nll_path)
    blockers = []
    if training.get("status") != "redundancy_proxy_training_complete":
        blockers.append("proxy_training_incomplete")
    if nll.get("status") != "redundancy_proxy_nll_summary_ready":
        blockers.append("proxy_nll_summary_missing")
    if eval_inputs["mechanism_diagnostic"]["status"] != (
        "template_saturation_mechanism_precheck_passed"
    ):
        blockers.append("mechanism_precheck_failed")

    curation = nll["curation_effect_random_minus_log_count"]
    noninferiority = nll["candidate_noninferiority_log_count_minus_binary"]
    margin = float(noninferiority["maximum_upper_bound"])
    statistical_noninferiority = noninferiority["one_sided_95_upper"] <= margin
    directional_nonworse = noninferiority["nonpositive_seed_count"] >= 2
    mechanism_pass = eval_inputs["mechanism_diagnostic"]["status"].endswith(
        "_passed"
    )
    general_text_pass = bool(nll["general_text_retention"]["passed"])

    promotion_requirements = {
        "curation_effect_vs_common_stageA_random": bool(curation["passed"]),
        "statistical_noninferiority_vs_binary_current": statistical_noninferiority,
        "directional_nonworse_in_at_least_two_seeds": directional_nonworse,
        "template_saturation_mechanism_precheck": mechanism_pass,
        "general_text_retention": general_text_pass,
        "general_task_retention": None,
        "evalplus_code_retention": None,
    }
    known_failure = not directional_nonworse
    if blockers:
        candidate_decision = "abstain_proxy_contract_incomplete"
    elif known_failure:
        candidate_decision = (
            "hold_log_count_keep_binary_current_directional_nonworse_failed"
        )
    elif None in promotion_requirements.values():
        candidate_decision = "abstain_missing_required_guardrails"
    elif all(promotion_requirements.values()):
        candidate_decision = "promote_log_count_to_qwen3_4b_development"
    else:
        candidate_decision = "reject_log_count_candidate"

    report = {
        "schema_version": "redundancy-proxy-decision-report-v1",
        "status": (
            "redundancy_proxy_candidate_decision_frozen"
            if not blockers
            else "redundancy_proxy_candidate_decision_blocked"
        ),
        "candidate": "log_count",
        "canonical_control": "binary_current",
        "candidate_decision": candidate_decision,
        "promotion_allowed": candidate_decision.startswith("promote_"),
        "qwen3_4b_development_allowed": candidate_decision.startswith("promote_"),
        "source_sha256": {
            str(plan_path): sha256_file(plan_path),
            str(eval_inputs_path): sha256_file(eval_inputs_path),
            str(training_path): sha256_file(training_path),
            str(nll_path): sha256_file(nll_path),
        },
        "target_nll_means": nll["target_nll_means"],
        "curation_effect": curation,
        "candidate_vs_binary": {
            **noninferiority,
            "statistical_noninferiority_passed": statistical_noninferiority,
            "directional_nonworse_passed": directional_nonworse,
            "interpretation": (
                "The candidate is within the frozen non-inferiority margin, but "
                "all three paired seed deltas favor binary_current. The effect is "
                "small and below the practical floor, so this is a hold rather "
                "than evidence that log_count is harmful."
            ),
        },
        "promotion_requirements": promotion_requirements,
        "futility_rule": {
            "triggered": known_failure,
            "reason": (
                "A mandatory pre-registered directional promotion condition has "
                "failed, so unrun guardrails cannot make promotion possible."
            ),
            "general_task_and_evalplus_for_candidate": (
                "not_run_after_futility_boundary"
            ),
        },
        "framework_evidence": {
            "curation_vs_common_stageA_random": (
                "positive_target_heldout_nll_evidence"
                if curation["passed"]
                else "not_supported"
            ),
            "general_text_retention": (
                "passed" if general_text_pass else "failed"
            ),
            "release_status": "abstain_missing_general_task_and_evalplus_guardrails",
            "interpretation": (
                "The proxy supports the value of Stage-B curation relative to "
                "common disjoint Stage-A random, but does not support replacing "
                "binary_current with log_count or making a framework release claim."
            ),
        },
        "blockers": blockers,
        "forbidden_next_actions": [
            "retuning log_count from these outcomes",
            "changing the frozen seed-direction rule retroactively",
            "promoting log_count to Qwen3-4B development",
            "using target or retention outcomes in Stage B",
        ],
        "required_next_work": [
            "keep binary_current canonical",
            "preserve the positive curated-vs-random proxy result as development evidence",
            "complete general-task and EvalPlus guardrails only for the canonical framework release path",
            "design any future saturation candidate as a new preregistered cycle",
        ],
        "utility_scope": plan["primary_comparison"]["utility_scope"],
        "claim_boundary": (
            "Frozen proxy candidate decision. It supports holding log_count and "
            "retaining binary_current; the full framework release remains abstain."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build redundancy proxy decision.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--eval-inputs", type=Path, default=DEFAULT_EVAL_INPUTS)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--nll", type=Path, default=DEFAULT_NLL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.plan,
        args.eval_inputs,
        args.training,
        args.nll,
        args.output,
    )
    print(report)
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
