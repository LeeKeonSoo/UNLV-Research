#!/usr/bin/env python3
"""Freeze the higher-capacity forward development accumulation plan."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_change import normalize_repository_identity


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery.json"
DEFAULT_BROAD = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_BENCHMARK = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_SNAPSHOT_REPORT = OUTPUT_DIR / "validation" / "temporal_code_forward_development_snapshot_report.json"
DEFAULT_PRODUCTIVITY = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_productivity_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"


def _order(identity: str) -> tuple[str, str]:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest(), identity


def freeze(
    contract_path: Path,
    discovery_path: Path,
    broad_path: Path,
    benchmark_path: Path,
    snapshot_report_path: Path,
    productivity_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    discovery = load_json(discovery_path)
    broad = load_json(broad_path)
    benchmark = load_json(benchmark_path)
    snapshot = load_json(snapshot_report_path)
    productivity = load_json(productivity_path)
    amendment = contract["development_accumulation_amendment"]
    broad_repositories = set(broad["repositories"])
    benchmark_repositories = {
        normalize_repository_identity(identity)
        for entry in benchmark["entries"]
        for identity in entry.get("repository_patterns") or []
    }
    eligible = sorted(
        (
            identity
            for identity, row in discovery["candidates"].items()
            if row.get("eligible_for_metadata_enrichment")
            and identity not in broad_repositories
            and identity not in benchmark_repositories
        ),
        key=_order,
    )
    capacity_amendment = contract.get("development_discovery_capacity_amendment") or {}
    cap = int(
        capacity_amendment.get(
            "maximum_repository_discovery_candidates",
            contract["development_acquisition_snapshots"]["maximum_repository_discovery_candidates"],
        )
    )
    frame = eligible[:cap]
    required_metadata = int(
        productivity["point_estimate_only"]["development_metadata_candidates_needed_for_542"]
    )
    report = {
        "schema_version": "temporal-code-forward-development-accumulation-plan-v1",
        "status": amendment["status"],
        "contract": contract,
        "accumulation_frame": {
            "repository_count": len(frame),
            "repository_identities": frame,
            "existing_broad_repository_overlap_count": len(set(frame).intersection(broad_repositories)),
            "benchmark_source_repository_overlap_count": len(set(frame).intersection(benchmark_repositories)),
            "prior_snapshot_candidate_count": snapshot["summary"]["metadata_candidate_count"],
            "development_window_end": contract["future_primary_acquisition"]["development_window"]["end"],
        },
        "capacity_context": {
            "point_estimate_metadata_candidates_needed_for_development": required_metadata,
            "current_repository_frame_count": len(frame),
            "frame_meets_point_estimate_candidate_capacity": len(frame) >= required_metadata,
            "frame_alone_guarantees_target": False,
            "estimate_role": "planning_only",
        },
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(discovery_path): sha256_file(discovery_path),
            str(broad_path): sha256_file(broad_path),
            str(benchmark_path): sha256_file(benchmark_path),
            str(snapshot_report_path): sha256_file(snapshot_report_path),
            str(productivity_path): sha256_file(productivity_path),
        },
        "next_snapshot_task_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Future development acquisition capacity plan only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze forward development accumulation.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--snapshot-report", type=Path, default=DEFAULT_SNAPSHOT_REPORT)
    parser.add_argument("--productivity", type=Path, default=DEFAULT_PRODUCTIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(
        args.contract,
        args.discovery,
        args.broad,
        args.benchmark,
        args.snapshot_report,
        args.productivity,
        args.output,
    )
    print(
        {
            "status": report["status"],
            "repository_count": report["accumulation_frame"]["repository_count"],
            "capacity": report["capacity_context"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
