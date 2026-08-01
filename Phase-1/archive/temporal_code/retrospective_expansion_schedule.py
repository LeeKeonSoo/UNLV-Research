#!/usr/bin/env python3
"""Freeze the remaining retrospective repository frame before task metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_INITIAL = COLLECTION / "temporal_code_retrospective_development_schedule.json"
DEFAULT_DISCOVERY = COLLECTION / "forward_development_repository_discovery_combined.json"
DEFAULT_BROAD = COLLECTION / "temporal_code_broad_repository_manifest.json"
DEFAULT_REPORT = OUTPUT_DIR / "validation" / "temporal_code_retrospective_development_report.json"
DEFAULT_OUTPUT = COLLECTION / "temporal_code_retrospective_expansion_schedule.json"


def freeze(
    contract_path: Path,
    initial_path: Path,
    discovery_path: Path,
    broad_path: Path,
    report_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    initial = load_json(initial_path)
    discovery = load_json(discovery_path)
    broad = load_json(broad_path)
    progress = load_json(report_path)
    initial_ids = {identity for shard in initial["shards"] for identity in shard["repository_identities"]}
    combined_ids = set(discovery["candidates"])
    broad_ids = set(broad["repositories"])
    remaining = sorted(combined_ids - initial_ids - broad_ids)
    overlap_initial = sorted(set(remaining).intersection(initial_ids))
    overlap_broad = sorted(set(remaining).intersection(broad_ids))
    if overlap_initial or overlap_broad:
        raise ValueError("Retrospective expansion must remain disjoint from initial and training repositories.")
    if not progress["decision"]["same_rules_full_remaining_frame_expansion_justified"]:
        raise ValueError("Progress report does not justify the unchanged-rule expansion.")
    size = int(contract["development_operations_contract"]["repository_shard_size"])
    shards = [
        {
            "shard_index": index,
            "shard_id": f"{index:03d}",
            "repository_count": len(remaining[offset : offset + size]),
            "repository_identities": remaining[offset : offset + size],
        }
        for index, offset in enumerate(range(0, len(remaining), size))
    ]
    report = {
        "schema_version": "temporal-code-retrospective-expansion-schedule-v1",
        "status": "frozen_after_first_e2_batch_and_before_remaining_repository_task_metadata",
        "contract": contract,
        "collection_window": initial["collection_window"],
        "adaptation_contract": {
            "basis": "aggregate first-batch E2 productivity shows feasibility and a point-estimate shortfall",
            "outcomes_used": ["aggregate first-batch E2 validity and failure-stage counts"],
            "outcomes_forbidden": ["Stage-A outcomes", "Stage-B outcomes", "Utility", "benchmark outcomes", "confirmatory outcomes"],
            "eligibility_rule_changes": "none",
            "task_validity_rule_changes": "none",
            "one_task_per_repository": True,
            "repository_choice_rule": "all remaining repositories from the pre-existing combined metadata-only discovery frame",
            "future_confirmatory_window_remains_untouched": True,
        },
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(initial_path): sha256_file(initial_path),
            str(discovery_path): sha256_file(discovery_path),
            str(broad_path): sha256_file(broad_path),
            str(report_path): sha256_file(report_path),
        },
        "summary": {
            "combined_repository_count": len(combined_ids),
            "initial_repository_count": len(initial_ids),
            "remaining_repository_count": len(remaining),
            "shard_size": size,
            "shard_count": len(shards),
            "initial_repository_overlap_count": len(overlap_initial),
            "training_repository_overlap_count": len(overlap_broad),
            "duplicate_repository_count": len(remaining) - len(set(remaining)),
        },
        "shards": shards,
        "remaining_task_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Retrospective expansion schedule only; no new task, E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze retrospective expansion schedule.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.initial, args.discovery, args.broad, args.report, args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
