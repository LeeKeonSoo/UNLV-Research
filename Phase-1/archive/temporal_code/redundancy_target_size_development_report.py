#!/usr/bin/env python3
"""Build the target-size canonical redundancy development decision report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "configs" / "temporal_code_redundancy_target_size_development_qwen3_4b_v1.json"
DEFAULT_BLOCKS = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_target_size_qwen3_4b_blocks_manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "redundancy_target_size_qwen3_4b_v1"
DEFAULT_OUTPUT = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_target_size_qwen3_4b_development_report.json"
)
DEFAULT_CANONICAL_GUARDRAIL_DECISION = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_canonical_guardrail_decision_report.json"
)
REQUIRED_GUARDRAIL_EVIDENCE = {
    "general_text_retention_nll": "general_text_retention",
    "general_task_retention": "general_task_retention",
    "evalplus_development": "evalplus_development_retention",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _ci(values: List[float]) -> Dict[str, Any]:
    if len(values) < 2:
        return {"mean": values[0] if values else None, "sd": None, "stderr": None}
    sd = statistics.stdev(values)
    stderr = sd / math.sqrt(len(values))
    # t critical for df=2, 95% two-sided. This is intentionally conservative
    # for the frozen three-seed development read.
    half_width = 4.302652729911275 * stderr
    return {
        "mean": _mean(values),
        "sd": sd,
        "stderr": stderr,
        "t_critical_95_two_sided_df2": 4.302652729911275,
        "ci95_low": _mean(values) - half_width,
        "ci95_high": _mean(values) + half_width,
    }


def _result_path(output_dir: Path, arm: str, seed: int | None) -> Path:
    name = "base_no_update" if seed is None else f"{arm}_seed{seed}"
    return output_dir / "heldout_nll" / f"{name}.json"


def _guardrail_evidence(required_guardrails: Dict[str, Any], decision_path: Path) -> Dict[str, Any]:
    decision = _load_json(decision_path) if decision_path.exists() else {}
    evidence = decision.get("evidence", {}) if isinstance(decision.get("evidence"), dict) else {}
    rows = {}
    missing = []
    failed = []
    for guardrail, evidence_key in REQUIRED_GUARDRAIL_EVIDENCE.items():
        if guardrail not in required_guardrails:
            continue
        row = evidence.get(evidence_key) if isinstance(evidence.get(evidence_key), dict) else None
        passed = bool(row and row.get("passed") is True)
        rows[guardrail] = {
            "evidence_key": evidence_key,
            "source_path": str(decision_path),
            "passed": passed,
            "evidence": row,
        }
        if row is None:
            missing.append(guardrail)
        elif not passed:
            failed.append(guardrail)
    return {"rows": rows, "missing": missing, "failed": failed}


def build(plan_path: Path, blocks_path: Path, output_dir: Path, output_path: Path) -> Dict[str, Any]:
    plan = _load_json(plan_path)
    blocks = _load_json(blocks_path)
    treatment = plan["primary_comparison"]["treatment"]
    baseline = plan["primary_comparison"]["primary_baseline"]
    seeds = [int(seed) for seed in plan["training_recipe"]["development_training_seeds"]]
    margin = float(plan["decision_rule"]["binary_vs_stageA_random_required_absolute_nll_reduction"])

    blockers = []
    if blocks.get("status") != "target_size_blocks_materialized":
        blockers.append("target_size_blocks_not_materialized")

    base_path = _result_path(output_dir, "base_no_update", None)
    if not base_path.exists():
        blockers.append("missing_base_no_update_nll")
    base = _load_json(base_path) if base_path.exists() else None

    rows = []
    for seed in seeds:
        treatment_path = _result_path(output_dir, treatment, seed)
        baseline_path = _result_path(output_dir, baseline, seed)
        if not treatment_path.exists():
            blockers.append(f"missing_treatment_nll_seed{seed}")
            continue
        if not baseline_path.exists():
            blockers.append(f"missing_baseline_nll_seed{seed}")
            continue
        treatment_row = _load_json(treatment_path)
        baseline_row = _load_json(baseline_path)
        if treatment_row.get("status") != "heldout_nll_completed":
            blockers.append(f"incomplete_treatment_nll_seed{seed}")
        if baseline_row.get("status") != "heldout_nll_completed":
            blockers.append(f"incomplete_baseline_nll_seed{seed}")
        rows.append(
            {
                "seed": seed,
                "treatment_mean_nll": float(treatment_row["mean_nll"]),
                "baseline_mean_nll": float(baseline_row["mean_nll"]),
                "baseline_minus_treatment_nll": float(baseline_row["mean_nll"])
                - float(treatment_row["mean_nll"]),
                "treatment_result_sha256": sha256_file(treatment_path),
                "baseline_result_sha256": sha256_file(baseline_path),
            }
        )

    deltas = [row["baseline_minus_treatment_nll"] for row in rows]
    treatment_values = [row["treatment_mean_nll"] for row in rows]
    baseline_values = [row["baseline_mean_nll"] for row in rows]
    mean_delta = _mean(deltas) if deltas else None
    target_nll_passed = bool(mean_delta is not None and mean_delta >= margin and not blockers)
    required_guardrails = plan["required_stage_c_evidence"]
    guardrail_evidence = _guardrail_evidence(required_guardrails, DEFAULT_CANONICAL_GUARDRAIL_DECISION)
    missing_guardrails = guardrail_evidence["missing"]
    failed_guardrails = guardrail_evidence["failed"]
    if blockers:
        status = "target_size_development_blocked"
    elif target_nll_passed and failed_guardrails:
        status = "target_size_guardrail_failed"
    elif target_nll_passed and missing_guardrails:
        status = "target_size_target_nll_passed_abstain_missing_guardrails"
    elif target_nll_passed:
        status = "target_size_development_passed"
    else:
        status = "target_size_target_nll_margin_failed"

    report = {
        "schema_version": "redundancy-target-size-qwen3-4b-development-report-v1",
        "status": status,
        "source_sha256": {
            str(plan_path): sha256_file(plan_path),
            str(blocks_path): sha256_file(blocks_path),
            str(base_path): sha256_file(base_path) if base_path.exists() else None,
            str(DEFAULT_CANONICAL_GUARDRAIL_DECISION): (
                sha256_file(DEFAULT_CANONICAL_GUARDRAIL_DECISION)
                if DEFAULT_CANONICAL_GUARDRAIL_DECISION.exists()
                else None
            ),
        },
        "target_model": plan["target_model"],
        "comparison": {
            "treatment": treatment,
            "baseline": baseline,
            "base_no_update_mean_nll": float(base["mean_nll"]) if base else None,
            "seeds": seeds,
            "rows": rows,
            "treatment_summary": _ci(treatment_values) if treatment_values else None,
            "baseline_summary": _ci(baseline_values) if baseline_values else None,
            "baseline_minus_treatment_summary": _ci(deltas) if deltas else None,
            "required_mean_nll_reduction": margin,
            "mean_margin_passed": target_nll_passed,
            "all_seed_direction_positive": all(value > 0 for value in deltas) if deltas else False,
            "all_seed_margin_passed": all(value >= margin for value in deltas) if deltas else False,
        },
        "guardrail_status": {
            "missing_guardrails": missing_guardrails,
            "failed_guardrails": failed_guardrails,
            "missing_guardrail_action": plan["decision_rule"]["missing_required_evidence_action"],
            "release_decision": (
                "release_supported"
                if target_nll_passed and not missing_guardrails and not failed_guardrails
                else (
                    "reject_target_size_promotion"
                    if failed_guardrails
                    else "abstain_not_a_production_release"
                )
            ),
        },
        "stage_c_guardrails": guardrail_evidence["rows"],
        "blockers": blockers,
        "utility_scope": plan["utility_scope"],
        "confirmatory_outcomes_read": False,
        "claim_boundary": (
            "Target-size development target-code NLL evidence only. General-text, "
            "general-task, and EvalPlus guardrails remain required before promotion."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build target-size redundancy development report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.plan, args.blocks, args.output_dir, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison": report["comparison"],
                "guardrail_status": report["guardrail_status"],
                "blockers": report["blockers"],
            },
            indent=2,
        )
    )
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
