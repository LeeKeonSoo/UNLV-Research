#!/usr/bin/env python3
"""Freeze EvalPlus development and confirmatory task IDs without model outcomes."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_evalplus_guardrail_split_v1.json"
DEFAULT_PREVALIDATION = OUTPUT_DIR / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"


def _bucket(dataset_name: str, task_id: str) -> int:
    value = f"{dataset_name}:{task_id}".encode("utf-8")
    return int(hashlib.sha256(value).hexdigest(), 16) % 100


def _assigned_split(bucket: int, contract: Dict[str, Any]) -> str:
    rule = contract["split_rule"]
    if int(rule["development_buckets"][0]) <= bucket <= int(rule["development_buckets"][1]):
        return "development"
    if int(rule["confirmatory_buckets"][0]) <= bucket <= int(rule["confirmatory_buckets"][1]):
        return "confirmatory"
    raise ValueError(bucket)


def freeze(contract_path: Path, prevalidation_path: Path, output_path: Path) -> Dict[str, Any]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    contract = load_json(contract_path)
    prevalidation = load_json(prevalidation_path)
    if prevalidation.get("status") != "e2_prevalidated":
        raise ValueError("EvalPlus guardrail split requires E2-prevalidated evaluator.")
    loaders = {"HumanEval+": get_human_eval_plus, "MBPP+": get_mbpp_plus}
    records = []
    for dataset_name, loader in loaders.items():
        for task_id in sorted(loader()):
            bucket = _bucket(dataset_name, task_id)
            records.append(
                {
                    "dataset": dataset_name,
                    "task_id": task_id,
                    "split_bucket": bucket,
                    "assigned_split": _assigned_split(bucket, contract),
                }
            )
    split_counts = Counter(row["assigned_split"] for row in records)
    suite_split_counts = Counter((row["dataset"], row["assigned_split"]) for row in records)
    report = {
        "schema_version": "temporal-code-evalplus-guardrail-split-plan-v1",
        "status": "frozen_e2_guardrail_split_before_model_outcomes",
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(prevalidation_path): sha256_file(prevalidation_path),
        },
        "summary": {
            "task_count": len(records),
            "split_counts": dict(sorted(split_counts.items())),
            "suite_split_counts": {
                f"{dataset}/{split}": count
                for (dataset, split), count in sorted(suite_split_counts.items())
            },
            "task_content_persisted": False,
            "model_outcomes_read": False,
        },
        "records": records,
        "development_utility_may_start": False,
        "development_utility_blockers": [
            "primary_temporal_executable_aggregate_not_acquired",
            "retention_non_inferiority_guardrails_not_complete",
        ],
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze EvalPlus external guardrail split.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--prevalidation", type=Path, default=DEFAULT_PREVALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.prevalidation, args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
