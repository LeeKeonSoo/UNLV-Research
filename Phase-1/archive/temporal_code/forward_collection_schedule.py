#!/usr/bin/env python3
"""Freeze deterministic repository shards for forward development collection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_ACCUMULATION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json"


def freeze(contract_path: Path, accumulation_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    accumulation = load_json(accumulation_path)
    operations = contract["development_operations_contract"]
    identities = list(accumulation["accumulation_frame"]["repository_identities"])
    size = int(operations["repository_shard_size"])
    shards = [
        {
            "shard_index": index,
            "shard_id": f"{index:03d}",
            "repository_count": len(identities[offset : offset + size]),
            "repository_identities": identities[offset : offset + size],
        }
        for index, offset in enumerate(range(0, len(identities), size))
    ]
    if len(shards) != int(operations["expected_repository_shard_count"]):
        raise ValueError("Frozen repository frame does not match the expected shard count.")
    report = {
        "schema_version": "temporal-code-forward-collection-schedule-v1",
        "status": "frozen_before_later_snapshot_task_metadata",
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(accumulation_path): sha256_file(accumulation_path),
        },
        "summary": {
            "repository_count": len(identities),
            "shard_size": size,
            "shard_count": len(shards),
            "duplicate_repository_count": len(identities) - len(set(identities)),
        },
        "shards": shards,
        "task_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Forward collection schedule only; no task, E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze forward collection repository shards.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--accumulation", type=Path, default=DEFAULT_ACCUMULATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.accumulation, args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
