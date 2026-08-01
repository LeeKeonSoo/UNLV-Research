#!/usr/bin/env python3
"""Run the frozen train-only temporal-code Stage-B engineering smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import select_stage_b


DEFAULT_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_smoke"
DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "configs" / "temporal_code_curation_protocol_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mean(records: List[Dict[str, Any]], field: str) -> float:
    return round(
        sum(float(row["stage_b_evidence"][field]) for row in records) / max(1, len(records)),
        6,
    )


def run(stage_a_dir: Path, protocol_path: Path, output_dir: Path) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    contract = protocol["stage_b_contract"]
    objective = contract["objective"]
    coverage = contract["coverage_support"]
    baseline_contract = contract["stage_a_random_baseline"]
    train = list(_jsonl(stage_a_dir / "train" / "stage_a_pass.jsonl"))
    if any(row.get("split") != "train" or row.get("stage_a_pass") is not True for row in train):
        raise ValueError("Stage B accepts train split Stage-A-pass chunks only")
    result = select_stage_b(
        train,
        budget_fraction=float(contract["budget"]["fraction"]),
        quality_weight=float(objective["code_quality_proxy_weight"]),
        redundancy_weight=float(objective["soft_redundancy_support_weight"]),
        coverage_axes=[str(value) for value in coverage["axes"]],
        minimum_exemplars=int(coverage["minimum_exemplars_per_observed_value"]),
        baseline_seed=int(baseline_contract["seed"]),
        distribution_axes=[str(value) for value in coverage.get("distribution_axes") or []],
        minimum_relative_token_share=float(coverage.get("minimum_relative_token_share") or 0.0),
        redundancy_search_mode=str(objective.get("redundancy_search_mode") or "indexed_exact"),
    )
    selected_ids = {row["chunk_uid"] for row in result["selected"]}
    baseline_ids = {row["chunk_uid"] for row in result["baseline"]}
    coverage_missing = {
        axis: sorted(set(result["coverage_all"][axis]) - set(result["coverage_selected"][axis]))
        for axis in coverage["axes"]
    }
    distribution_support = {}
    for axis in coverage.get("distribution_axes") or []:
        distribution_support[axis] = {}
        total_all = sum(result["coverage_tokens_all"][axis].values())
        total_selected = sum(result["coverage_tokens_selected"][axis].values())
        for value, all_tokens in result["coverage_tokens_all"][axis].items():
            selected_tokens = result["coverage_tokens_selected"][axis].get(value, 0)
            source_share = all_tokens / max(1, total_all)
            selected_share = selected_tokens / max(1, total_selected)
            distribution_support[axis][value] = {
                "source_token_share": round(source_share, 6),
                "selected_token_share": round(selected_share, 6),
                "relative_token_share_retained": round(selected_share / max(source_share, 1e-12), 6),
                "passed": selected_share / max(source_share, 1e-12)
                >= float(coverage.get("minimum_relative_token_share") or 0.0),
            }
    report = {
        "schema_version": "temporal-code-stage-b-smoke-report-v1",
        "operational_decision": (
            "insufficient_usable_data" if not train else "stage_b_engineering_candidate_constructed"
        ),
        "stage_b_contract": contract,
        "summary": {
            "input_train_stage_a_pass_chunks": len(train),
            "scored_chunks": len(result["scored"]),
            "selected_chunks": len(result["selected"]),
            "stage_a_random_disjoint_chunks": len(result["baseline"]),
            "budget_token_proxy": result["budget_token_proxy"],
            "selected_token_proxy": result["selected_token_proxy"],
            "stage_a_random_disjoint_token_proxy": result["baseline_token_proxy"],
            "baseline_to_selected_token_ratio": round(
                result["baseline_token_proxy"] / max(1, result["selected_token_proxy"]), 6
            ),
            "selected_and_baseline_disjoint": not bool(selected_ids.intersection(baseline_ids)),
        },
        "core_proxy_comparison": {
            "selected": {
                "mean_code_quality_proxy": _mean(result["selected"], "code_quality_proxy"),
                "mean_soft_redundancy_risk": _mean(result["selected"], "soft_redundancy_risk"),
                "mean_stage_b_objective_score": _mean(result["selected"], "stage_b_objective_score"),
            },
            "stage_a_random_disjoint": {
                "mean_code_quality_proxy": _mean(result["baseline"], "code_quality_proxy"),
                "mean_soft_redundancy_risk": _mean(result["baseline"], "soft_redundancy_risk"),
                "mean_stage_b_objective_score": _mean(result["baseline"], "stage_b_objective_score"),
            },
        },
        "coverage_support": {
            "all_train_stage_a_pass": result["coverage_all"],
            "selected": result["coverage_selected"],
            "stage_a_random_disjoint": result["coverage_baseline"],
            "selected_missing_observed_values": coverage_missing,
            "all_observed_values_retained": all(not values for values in coverage_missing.values()),
            "distribution_support": distribution_support,
            "all_distribution_floors_passed": all(
                row["passed"]
                for values in distribution_support.values()
                for row in values.values()
            ),
        },
        "isolation": {
            "selection_input_split": "train",
            "development_read": False,
            "confirmatory_read": False,
            "development_and_confirmatory_use": "forbidden for Stage-B scoring or selection",
        },
        "scalability_boundary": "Indexed exact soft-redundancy search is active; smoke equivalence to all-pairs is validated separately.",
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Bounded Stage-B Core-proxy engineering smoke only; no Stage-C, Utility, or training-release claim. "
            "An empty train Stage-A pool must abstain as insufficient_usable_data."
        ),
    }
    _write_jsonl(output_dir / "train_scored.jsonl", result["scored"])
    _write_jsonl(output_dir / "train_selected.jsonl", result["selected"])
    _write_jsonl(output_dir / "train_stage_a_random_disjoint.jsonl", result["baseline"])
    save_json(output_dir / "stage_b_smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run temporal-code Stage-B smoke.")
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(run(args.stage_a_dir, args.protocol, args.output_dir)["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
