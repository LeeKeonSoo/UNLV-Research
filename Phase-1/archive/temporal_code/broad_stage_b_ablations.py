#!/usr/bin/env python3
"""Run frozen Stage-B selector ablations on the broad train tranche."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import select_stage_b


DEFAULT_STAGE_A = OUTPUT_DIR / "temporal_code_collection" / "stage_a_broad_tranche" / "train" / "stage_a_pass.jsonl"
DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_ABLATION = Path("configs") / "temporal_code_stage_b_ablation_protocol_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_broad_stage_b_ablations.json"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return round(mean(float(row["stage_b_evidence"][key]) for row in rows), 6) if rows else 0.0


def run(stage_a_path: Path, protocol_path: Path, ablation_path: Path, output_path: Path) -> Dict[str, Any]:
    records = list(_jsonl(stage_a_path))
    protocol = load_json(protocol_path)
    ablation = load_json(ablation_path)
    contract = protocol["stage_b_contract"]
    coverage = contract["coverage_support"]
    results = {}
    for name in ("full_selector", "quality_only", "redundancy_only", "no_coverage_support"):
        arm = ablation["arms"][name]
        coverage_enabled = arm["coverage_support"] != "disabled"
        result = select_stage_b(
            records,
            budget_fraction=float(contract["budget"]["fraction"]),
            quality_weight=float(arm["quality_weight"]),
            redundancy_weight=float(arm["redundancy_support_weight"]),
            coverage_axes=[str(value) for value in coverage["axes"]] if coverage_enabled else [],
            minimum_exemplars=int(coverage["minimum_exemplars_per_observed_value"]),
            baseline_seed=int(contract["stage_a_random_baseline"]["seed"]),
            distribution_axes=[str(value) for value in coverage["distribution_axes"]] if coverage_enabled else [],
            minimum_relative_token_share=float(coverage["minimum_relative_token_share"]) if coverage_enabled else 0.0,
            redundancy_search_mode=str(contract["objective"]["redundancy_search_mode"]),
        )
        results[name] = {
            "selected_chunks": len(result["selected"]),
            "selected_token_proxy": result["selected_token_proxy"],
            "mean_code_quality_proxy": _mean(result["selected"], "code_quality_proxy"),
            "mean_soft_redundancy_risk": _mean(result["selected"], "soft_redundancy_risk"),
            "mean_stage_b_objective_score": _mean(result["selected"], "stage_b_objective_score"),
            "selected_bundle_count": len(
                {str(row.get("bundle_id")) for row in result["selected"]}
            ),
            "selected_content_types": sorted(
                {str(row.get("content_type")) for row in result["selected"]}
            ),
        }
    report = {
        "schema_version": "temporal-code-broad-stage-b-ablations-v1",
        "status": "frozen_selector_ablations_complete_before_stage_c",
        "input_train_stage_a_pass_chunks": len(records),
        "arms": results,
        "raw_random_equal_token": {
            "status": "pending_target_tokenizer_construction",
            "reason": "Raw authorized payload must be matched with the frozen target-model tokenizer, not chunk token proxy.",
        },
        "forbidden_signals_observed": [],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Stage-B proxy ablations only; no target-model or Utility conclusion.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run broad temporal-code Stage-B selector ablations.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.stage_a, args.protocol, args.ablation, args.output)
    print(report["arms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
