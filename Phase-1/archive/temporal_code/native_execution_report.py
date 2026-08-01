#!/usr/bin/env python3
"""Build the exploratory repository-native execution refinement report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_COMMANDS = COLLECTION / "temporal_code_development_native_test_commands_v1.json"
DEFAULT_GENERIC = COLLECTION / "development_execution_expansion_verification.json"
DEFAULT_NATIVE_V1 = COLLECTION / "development_native_execution_verification.json"
DEFAULT_NATIVE_V2 = COLLECTION / "development_native_execution_verification_v2.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_native_execution_refinement_report.json"


def _build_pass_count(report: Dict[str, Any]) -> int:
    return sum(
        result["build"]["exit_code"] == 0
        for row in report["decisions"]
        for result in row.get("commit_results") or []
    )


def _test_pass_count(report: Dict[str, Any]) -> int:
    return sum(
        result["test"]["exit_code"] == 0
        for row in report["decisions"]
        for result in row.get("commit_results") or []
    )


def build(
    commands_path: Path,
    generic_path: Path,
    native_v1_path: Path,
    native_v2_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    commands = load_json(commands_path)
    generic = load_json(generic_path)
    native_v1 = load_json(native_v1_path)
    native_v2 = load_json(native_v2_path)
    report = {
        "schema_version": "temporal-code-native-execution-refinement-report-v1",
        "status": "native_recipe_exploration_no_executable_recovery",
        "summary": {
            "candidate_bundles": int(native_v2["summary"]["bundle_count"]),
            "generic_build_pass_commits": _build_pass_count(generic),
            "native_v1_build_pass_commits": _build_pass_count(native_v1),
            "native_v2_build_pass_commits": _build_pass_count(native_v2),
            "generic_test_pass_commits": _test_pass_count(generic),
            "native_v1_test_pass_commits": _test_pass_count(native_v1),
            "native_v2_test_pass_commits": _test_pass_count(native_v2),
            "generic_verified_bundles": int(generic["summary"]["verified_bundle_count"]),
            "native_v1_verified_bundles": int(native_v1["summary"]["verified_bundle_count"]),
            "native_v2_verified_bundles": int(native_v2["summary"]["verified_bundle_count"]),
            "structured_optional_extra_count": int(commands["summary"]["structured_optional_extra_count"]),
            "nondefault_python_image_count": int(commands["summary"]["nondefault_python_image_count"]),
        },
        "interpretation": (
            "Metadata-derived native recipes improved build-stage reach but recovered no parent-and-merge "
            "executable bundle. Remaining failures require repository-specific dependency ecosystems, services, "
            "hardware, or task harnesses. This is execution-infrastructure evidence, not Utility evidence."
        ),
        "decision": {
            "development_utility_may_start": False,
            "continue_recipe_tuning_on_same_development_pool": False,
            "reason": (
                "Further recipe changes informed by this pool's execution failures would be outcome-guided "
                "development overfitting without establishing a representative executable task distribution."
            ),
            "allowed_next_actions": [
                "freeze a fresh metadata-only development sampling-frame expansion before content or execution outcomes",
                "freeze a reusable external executable-task harness independent of this development pool",
                "treat repository-native execution support as a separate collection capability with explicit support tiers",
            ],
            "forbidden_reactions": [
                "infer Stage-B, curation, or Utility failure from execution-recipe failure",
                "tune Stage B from execution or Utility outcomes",
                "continue repository-specific recipe exceptions on the same development pool",
                "inspect confirmatory outcomes",
            ],
        },
        "evidence_role": commands["contract"]["evidence_role"],
        "confirmatory_outcomes_read": False,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Repository-native metadata improves execution reach but does not yet create a sufficient development "
            "executable holdout. No Utility, curation-benefit, curation-failure, or release claim is established."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build native execution refinement report.")
    parser.add_argument("--commands", type=Path, default=DEFAULT_COMMANDS)
    parser.add_argument("--generic", type=Path, default=DEFAULT_GENERIC)
    parser.add_argument("--native-v1", type=Path, default=DEFAULT_NATIVE_V1)
    parser.add_argument("--native-v2", type=Path, default=DEFAULT_NATIVE_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.commands, args.generic, args.native_v1, args.native_v2, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
