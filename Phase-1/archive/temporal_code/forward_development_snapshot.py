#!/usr/bin/env python3
"""Freeze a training-disjoint forward development repository snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_change import normalize_repository_identity


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery.json"
DEFAULT_BROAD_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_BENCHMARK_SEED = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_snapshot_plan.json"


def _order(identity: str) -> tuple[str, str]:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest(), identity


def freeze(
    contract_path: Path,
    discovery_path: Path,
    broad_manifest_path: Path,
    benchmark_seed_path: Path,
    output_path: Path,
    available_through: str,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    discovery = load_json(discovery_path)
    broad = load_json(broad_manifest_path)
    benchmark = load_json(benchmark_seed_path)
    run_date = dt.date.today().isoformat()
    window = contract["future_primary_acquisition"]["development_window"]
    if not (window["start"] <= available_through <= min(window["end"], run_date)):
        raise ValueError("available-through must be inside the development window and must not exceed the run date.")
    excluded_broad = set(broad["repositories"])
    excluded_benchmark = {
        normalize_repository_identity(identity)
        for entry in benchmark["entries"]
        for identity in entry.get("repository_patterns") or []
    }
    candidates = []
    exclusions: Dict[str, str] = {}
    for identity, row in discovery["candidates"].items():
        if not row.get("eligible_for_metadata_enrichment"):
            exclusions[identity] = "repository_discovery_ineligible"
        elif identity in excluded_broad:
            exclusions[identity] = "existing_broad_manifest_repository"
        elif identity in excluded_benchmark:
            exclusions[identity] = "benchmark_source_repository"
        else:
            candidates.append(identity)
    maximum = int(contract["development_acquisition_snapshots"]["maximum_repositories_per_snapshot"])
    frame = sorted(candidates, key=_order)[:maximum]
    report = {
        "schema_version": "temporal-code-forward-development-snapshot-plan-v1",
        "status": "frozen_before_forward_development_task_metadata",
        "contract": contract,
        "snapshot": {
            "window_start": window["start"],
            "available_through": available_through,
            "run_date": run_date,
            "repository_frame_count": len(frame),
            "repository_identities": frame,
        },
        "exclusion_summary": {
            "existing_broad_manifest_repository_count": sum(
                reason == "existing_broad_manifest_repository" for reason in exclusions.values()
            ),
            "benchmark_source_repository_count": sum(
                reason == "benchmark_source_repository" for reason in exclusions.values()
            ),
            "repository_discovery_ineligible_count": sum(
                reason == "repository_discovery_ineligible" for reason in exclusions.values()
            ),
        },
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(discovery_path): sha256_file(discovery_path),
            str(broad_manifest_path): sha256_file(broad_manifest_path),
            str(benchmark_seed_path): sha256_file(benchmark_seed_path),
        },
        "training_repository_overlap_count": 0,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Outcome-free forward development repository snapshot only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a forward development repository snapshot.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--broad-manifest", type=Path, default=DEFAULT_BROAD_MANIFEST)
    parser.add_argument("--benchmark-seed", type=Path, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--available-through", required=True)
    args = parser.parse_args()
    report = freeze(
        args.contract,
        args.discovery,
        args.broad_manifest,
        args.benchmark_seed,
        args.output,
        args.available_through,
    )
    print({"status": report["status"], "snapshot": report["snapshot"], "exclusions": report["exclusion_summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
