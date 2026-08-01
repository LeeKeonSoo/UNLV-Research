#!/usr/bin/env python3
"""Build the no-outcome development executable-expansion readiness report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_PLAN = COLLECTION / "temporal_code_development_execution_expansion_plan.json"
DEFAULT_FETCH = COLLECTION / "development_execution_expansion_bundles" / "smoke_fetch_report.json"
DEFAULT_AUDIT = (
    COLLECTION / "development_execution_expansion_bundles" / "development_execution_expansion_audit_report.json"
)
DEFAULT_VERIFICATION = COLLECTION / "development_execution_expansion_verification.json"
DEFAULT_PATH_READINESS = OUTPUT_DIR / "validation" / "temporal_code_path_stratified_tranche_readiness.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_development_expansion_readiness.json"


def build(
    plan_path: Path,
    fetch_path: Path,
    audit_path: Path,
    verification_path: Path,
    path_readiness_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    fetch = load_json(fetch_path)
    audit = load_json(audit_path)
    verification = load_json(verification_path)
    path_readiness = load_json(path_readiness_path)
    existing_verified = int(
        path_readiness["summary"]["executable_evaluation_gate_pass_by_split"]["development"]
    )
    added_verified = int(audit["summary"]["executable_evaluation_gate_pass_count"])
    total_verified = existing_verified + added_verified
    report = {
        "schema_version": "temporal-code-development-expansion-readiness-v1",
        "status": "development_stage_c_blocked_insufficient_executable_holdout",
        "summary": {
            "frozen_candidate_repositories": int(plan["summary"]["repository_count"]),
            "fetched_bundles": int(fetch["summary"]["bundle_count"]),
            "fetched_file_records": int(fetch["summary"]["file_record_count"]),
            "collection_gate_pass_bundles": int(audit["summary"]["collection_gate_pass_count"]),
            "generic_execution_candidates": int(verification["summary"]["bundle_count"]),
            "generic_execution_verified_bundles": int(verification["summary"]["verified_bundle_count"]),
            "generic_execution_build_failed_commits": int(
                verification["summary"]["build_failed_commit_count"]
            ),
            "generic_execution_test_failed_commits": int(
                verification["summary"]["test_failed_commit_count"]
            ),
            "existing_verified_development_bundles": existing_verified,
            "total_verified_development_bundles": total_verified,
        },
        "decision": {
            "development_utility_may_start": False,
            "reason": (
                "Only one verified development executable bundle exists. Repeated training seeds do not "
                "replace a task distribution, and a task-aware practical-effect and inference contract "
                "cannot be finalized from one task."
            ),
            "allowed_next_actions": [
                "freeze outcome-independent repository-native execution recipe extraction before another execution attempt",
                "expand the development sampling frame using metadata-only rules before content or outcomes",
            ],
            "forbidden_reactions": [
                "infer curation or Stage-B failure from generic execution-hypothesis failure",
                "tune Stage B from execution or future Utility outcomes",
                "run development Utility before a task-count-aware decision contract is frozen",
                "inspect confirmatory outcomes",
            ],
        },
        "generic_execution_result_interpretation": (
            "The frozen generic execution hypothesis failed to add executable tasks. "
            "This is execution-infrastructure evidence, not Utility evidence."
        ),
        "confirmatory_outcomes_read": False,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "The development executable holdout is insufficient for Stage-C effect estimation. "
            "No curation-benefit, curation-failure, or release claim is established."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build development expansion readiness report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--fetch", type=Path, default=DEFAULT_FETCH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--path-readiness", type=Path, default=DEFAULT_PATH_READINESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.plan,
        args.fetch,
        args.audit,
        args.verification,
        args.path_readiness,
        args.output,
    )
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
