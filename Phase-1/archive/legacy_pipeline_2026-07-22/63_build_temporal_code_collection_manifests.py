#!/usr/bin/env python3
"""Build pre-collection temporal-code split and benchmark-quarantine manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import bundle_protocol_eligibility
from ingestion.temporal_code_manifests import (
    benchmark_quarantine_decision,
    build_benchmark_quarantine_manifest,
    build_repository_split_manifest,
    bundle_split_eligibility,
)


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_REPOSITORIES = Path("validation") / "fixtures" / "temporal_code_repositories.json"
DEFAULT_BENCHMARKS = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_BUNDLES = Path("validation") / "fixtures" / "temporal_code_change_bundles.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection"


def build(
    protocol_path: Path,
    repositories_path: Path,
    benchmarks_path: Path,
    bundles_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    repositories = load_json(repositories_path)["repositories"]
    benchmark_entries = load_json(benchmarks_path)["entries"]
    bundles = load_json(bundles_path)["bundles"]
    split_manifest = build_repository_split_manifest(repositories, protocol)
    quarantine_manifest = build_benchmark_quarantine_manifest(benchmark_entries, protocol)
    decisions = {}
    for bundle in bundles:
        bundle_id = str(bundle["bundle_id"])
        payload = bundle_protocol_eligibility(bundle, protocol)
        split = bundle_split_eligibility(bundle, split_manifest, protocol)
        quarantine = benchmark_quarantine_decision(bundle, quarantine_manifest)
        blockers = list(payload["blockers"]) + list(split["blockers"])
        if quarantine["quarantine"]:
            blockers.append("benchmark_quarantine")
        decisions[bundle_id] = {
            "eligible_for_stage0": not blockers,
            "blockers": sorted(set(blockers)),
            "training_payload_count": len(payload["training_payloads"]),
            "split": split,
            "benchmark_quarantine": quarantine,
            "validation_errors": payload["validation_errors"],
        }
    report = {
        "schema_version": "temporal-code-precollection-report-v1",
        "protocol": str(protocol_path),
        "summary": {
            "repository_count": split_manifest["repository_count"],
            "bundle_count": len(decisions),
            "stage0_eligible_bundle_count": sum(1 for row in decisions.values() if row["eligible_for_stage0"]),
            "benchmark_quarantined_bundle_count": sum(
                1 for row in decisions.values() if row["benchmark_quarantine"]["quarantine"]
            ),
        },
        "bundle_decisions": decisions,
    }
    save_json(output_dir / "repository_split_manifest.json", split_manifest)
    save_json(output_dir / "benchmark_quarantine_manifest.json", quarantine_manifest)
    save_json(output_dir / "precollection_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal-code pre-collection manifests.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--repositories", type=Path, default=DEFAULT_REPOSITORIES)
    parser.add_argument("--benchmarks", type=Path, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build(args.protocol, args.repositories, args.benchmarks, args.bundles, args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
