#!/usr/bin/env python3
"""Freeze retrospective-development shards before task metadata."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_ACCUMULATION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"
DEFAULT_BROAD = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retrospective_development_schedule.json"


def freeze(contract_path: Path, accumulation_path: Path, broad_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    accumulation = load_json(accumulation_path)
    broad = load_json(broad_path)
    helper = importlib.import_module("121_freeze_temporal_code_forward_collection_schedule")
    identities = list(accumulation["accumulation_frame"]["repository_identities"])
    broad_ids = set(broad["repositories"])
    overlap = sorted(set(identities).intersection(broad_ids))
    if overlap:
        raise ValueError("Retrospective development repositories overlap the existing broad manifest.")
    operations = contract["development_operations_contract"]
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
    report = {
        "schema_version": "temporal-code-retrospective-development-schedule-v1",
        "status": "frozen_before_retrospective_task_metadata",
        "contract": contract,
        "collection_window": {
            "start": contract["retrospective_development_contract"]["window_start"],
            "end": contract["retrospective_development_contract"]["window_end"],
        },
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(accumulation_path): sha256_file(accumulation_path),
            str(broad_path): sha256_file(broad_path),
        },
        "summary": {
            "repository_count": len(identities),
            "shard_size": size,
            "shard_count": len(shards),
            "training_repository_overlap_count": len(overlap),
            "duplicate_repository_count": len(identities) - len(set(identities)),
        },
        "shards": shards,
        "task_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Retrospective development schedule only; no task, E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze retrospective development schedule.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--accumulation", type=Path, default=DEFAULT_ACCUMULATION)
    parser.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.accumulation, args.broad, args.output)
    print({"status": report["status"], "window": report["collection_window"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
