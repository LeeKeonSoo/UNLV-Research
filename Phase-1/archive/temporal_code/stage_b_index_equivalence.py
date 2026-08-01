#!/usr/bin/env python3
"""Validate indexed Stage-B redundancy search against exact all-pairs results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import select_stage_b


DEFAULT_STAGE_A = OUTPUT_DIR / "temporal_code_collection" / "stage_a_smoke" / "train" / "stage_a_pass.jsonl"
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_DIR / "configs" / "temporal_code_curation_protocol_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_stage_b_index_equivalence.json"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def validate(records: list[Dict[str, Any]], protocol: Dict[str, Any]) -> Dict[str, Any]:
    contract = protocol["stage_b_contract"]
    objective = contract["objective"]
    coverage = contract["coverage_support"]
    common = {
        "budget_fraction": float(contract["budget"]["fraction"]),
        "quality_weight": float(objective["code_quality_proxy_weight"]),
        "redundancy_weight": float(objective["soft_redundancy_support_weight"]),
        "coverage_axes": [str(value) for value in coverage["axes"]],
        "minimum_exemplars": int(coverage["minimum_exemplars_per_observed_value"]),
        "baseline_seed": int(contract["stage_a_random_baseline"]["seed"]),
        "distribution_axes": [str(value) for value in coverage.get("distribution_axes") or []],
        "minimum_relative_token_share": float(coverage.get("minimum_relative_token_share") or 0.0),
    }
    all_pairs = select_stage_b(records, **common, redundancy_search_mode="all_pairs_exact")
    indexed = select_stage_b(records, **common, redundancy_search_mode="indexed_exact")
    all_by_id = {row["chunk_uid"]: row for row in all_pairs["scored"]}
    indexed_by_id = {row["chunk_uid"]: row for row in indexed["scored"]}
    maximum_risk_delta = max(
        abs(
            float(all_by_id[uid]["stage_b_evidence"]["soft_redundancy_risk"])
            - float(indexed_by_id[uid]["stage_b_evidence"]["soft_redundancy_risk"])
        )
        for uid in all_by_id
    ) if all_by_id else 0.0
    maximum_objective_delta = max(
        abs(
            float(all_by_id[uid]["stage_b_evidence"]["stage_b_objective_score"])
            - float(indexed_by_id[uid]["stage_b_evidence"]["stage_b_objective_score"])
        )
        for uid in all_by_id
    ) if all_by_id else 0.0
    all_selected = {row["chunk_uid"] for row in all_pairs["selected"]}
    indexed_selected = {row["chunk_uid"] for row in indexed["selected"]}
    all_baseline = {row["chunk_uid"] for row in all_pairs["baseline"]}
    indexed_baseline = {row["chunk_uid"] for row in indexed["baseline"]}
    passed = (
        maximum_risk_delta == 0.0
        and maximum_objective_delta == 0.0
        and all_selected == indexed_selected
        and all_baseline == indexed_baseline
    )
    return {
        "schema_version": "temporal-code-stage-b-index-equivalence-v1",
        "summary": {
            "record_count": len(records),
            "operational_decision": "insufficient_usable_data" if not records else "equivalence_validated",
            "maximum_redundancy_risk_delta": maximum_risk_delta,
            "maximum_objective_score_delta": maximum_objective_delta,
            "selected_symmetric_difference_count": len(all_selected.symmetric_difference(indexed_selected)),
            "baseline_symmetric_difference_count": len(all_baseline.symmetric_difference(indexed_baseline)),
            "passed": passed,
        },
        "indexed_candidate_audit": {
            "mean_candidate_count": round(
                sum(row["stage_b_evidence"]["lexical_candidate_count"] for row in indexed["scored"]) / max(1, len(records)),
                6,
            ),
            "all_pairs_candidate_count_per_record": max(0, len(records) - 1),
        },
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Engineering equivalence only; no Utility, training-benefit, or release claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate indexed temporal-code Stage-B equivalence.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate(list(_jsonl(args.stage_a)), load_json(args.protocol))
    save_json(args.output, report)
    print(report["summary"])
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
