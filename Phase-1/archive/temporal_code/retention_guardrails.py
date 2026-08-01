#!/usr/bin/env python3
"""Materialize frozen temporal-code Stage-C retention guardrails."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_stage_c_retention_guardrails_v1.json"
DEFAULT_EVALPLUS_SPLIT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"


def freeze(contract_path: Path, evalplus_split_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    evalplus_split = load_json(evalplus_split_path)
    if evalplus_split.get("status") != "frozen_e2_guardrail_split_before_model_outcomes":
        raise ValueError("Retention guardrails require a frozen E2 EvalPlus split.")
    report = {
        "schema_version": "temporal-code-retention-guardrail-plan-v1",
        "status": "frozen_before_development_model_outcomes",
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(evalplus_split_path): sha256_file(evalplus_split_path),
        },
        "current_evidence": {
            "evalplus_guardrail_split_status": evalplus_split["status"],
            "evalplus_task_count": evalplus_split["summary"]["task_count"],
            "development_model_outcomes_read": False,
            "confirmatory_model_outcomes_read": False,
        },
        "development_utility_may_start": False,
        "remaining_blockers": [
            "primary_temporal_executable_aggregate_not_frozen",
            "primary_temporal_development_and_confirmatory_e2_task_pools_not_acquired",
        ],
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze temporal-code Stage-C retention guardrails.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--evalplus-split", type=Path, default=DEFAULT_EVALPLUS_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.evalplus_split, args.output)
    print({"status": report["status"], "remaining_blockers": report["remaining_blockers"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
