#!/usr/bin/env python3
"""Audit structural capacity of the frozen forward development frame."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_ACCUMULATION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"
DEFAULT_PRODUCTIVITY = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_productivity_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_forward_capacity_audit.json"


def build(contract_path: Path, accumulation_path: Path, productivity_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    accumulation = load_json(accumulation_path)
    productivity = load_json(productivity_path)
    repository_count = int(accumulation["accumulation_frame"]["repository_count"])
    metadata_needed = int(productivity["point_estimate_only"]["development_metadata_candidates_needed_for_542"])
    target_e2 = int(contract["future_primary_acquisition"]["development_window"]["target_task_count"])
    observed_yield = float(productivity["observed_pilot"]["metadata_to_e2_yield"])
    expected_e2 = math.floor(repository_count * observed_yield)
    report = {
        "schema_version": "temporal-code-forward-capacity-audit-v1",
        "status": "current_forward_repository_frame_structurally_underpowered",
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(accumulation_path): sha256_file(accumulation_path),
            str(productivity_path): sha256_file(productivity_path),
        },
        "capacity": {
            "current_fresh_repository_count": repository_count,
            "maximum_metadata_candidates_under_one_task_per_repository": repository_count,
            "point_estimate_metadata_candidates_needed": metadata_needed,
            "metadata_capacity_ratio": repository_count / metadata_needed,
            "expected_e2_at_pilot_yield": expected_e2,
            "development_e2_target": target_e2,
            "expected_e2_target_ratio": expected_e2 / target_e2,
            "structurally_sufficient_at_point_estimate": repository_count >= metadata_needed,
        },
        "decision": {
            "expanded_discovery_authorized_before_later_task_metadata": True,
            "eligibility_rule_changes_authorized": False,
            "task_validity_weakening_authorized": False,
            "one_task_per_repository_weakening_authorized": False,
            "development_utility_may_start": False,
            "confirmatory_outcomes_read": False,
            "next_action": "run the frozen capacity-driven expanded repository discovery",
        },
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Repository-frame capacity audit only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build forward repository-frame capacity audit.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--accumulation", type=Path, default=DEFAULT_ACCUMULATION)
    parser.add_argument("--productivity", type=Path, default=DEFAULT_PRODUCTIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.contract, args.accumulation, args.productivity, args.output)
    print({"status": report["status"], "capacity": report["capacity"], "decision": report["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
