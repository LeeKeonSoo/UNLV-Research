#!/usr/bin/env python3
"""Freeze forward E2 task-acquisition and infrastructure-pilot contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_PRIMARY_ASSESSMENT = OUTPUT_DIR / "validation" / "temporal_code_primary_executable_source_assessment.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_e2_acquisition_plan.json"


def freeze(contract_path: Path, manifest_path: Path, assessment_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    assessment = load_json(assessment_path)
    train_repositories = sorted(
        identity for identity, row in manifest["repositories"].items() if row["assigned_split"] == "train"
    )
    report = {
        "schema_version": "temporal-code-forward-e2-acquisition-plan-v1",
        "status": contract["status"],
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(manifest_path): sha256_file(manifest_path),
            str(assessment_path): sha256_file(assessment_path),
        },
        "pilot_repository_frame": {
            "eligible_train_repository_count": len(train_repositories),
            "repository_identities": train_repositories,
            "pilot_tasks_evaluation_authorized": False,
        },
        "current_primary_source_status": assessment["status"],
        "development_utility_may_start": False,
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze forward E2 acquisition contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--primary-assessment", type=Path, default=DEFAULT_PRIMARY_ASSESSMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.manifest, args.primary_assessment, args.output)
    print({"status": report["status"], "pilot_repository_frame": report["pilot_repository_frame"]["eligible_train_repository_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
