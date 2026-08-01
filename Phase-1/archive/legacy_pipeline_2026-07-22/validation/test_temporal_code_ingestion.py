#!/usr/bin/env python3
"""Regression checks for temporal code change-bundle ingestion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402
from ingestion.code_change import (  # noqa: E402
    bundle_executable_evaluation_eligibility,
    bundle_protocol_eligibility,
    normalize_repository_identity,
)
from ingestion.temporal_code_manifests import (  # noqa: E402
    benchmark_quarantine_decision,
    build_benchmark_quarantine_manifest,
    build_repository_split_manifest,
    bundle_split_eligibility,
    temporal_split,
)


def main() -> int:
    protocol = load_json(PROJECT_DIR / "configs" / "temporal_code_curation_protocol_v1.json")
    repositories = load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_repositories.json")[
        "repositories"
    ]
    benchmark_entries = load_json(
        PROJECT_DIR / "validation" / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
    )["entries"]
    bundles = {
        row["bundle_id"]: row
        for row in load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_change_bundles.json")[
            "bundles"
        ]
    }
    split_manifest = build_repository_split_manifest(repositories, protocol)
    quarantine_manifest = build_benchmark_quarantine_manifest(benchmark_entries, protocol)

    assert normalize_repository_identity("https://github.com/Fixture/Repo0.git") == "fixture/repo0"
    clean = bundle_protocol_eligibility(bundles["fixture-train-clean"], protocol)
    assert clean["eligible"] is True, clean
    assert len(clean["training_payloads"]) == 2, clean
    assert all(payload["text"] != bundles["fixture-train-clean"]["prose"]["body"] for payload in clean["training_payloads"])
    assert any("vendored" in row["blockers"] for row in clean["file_eligibility"] if row["path"] == "vendor/helper.py")
    unverified = dict(bundles["fixture-train-clean"])
    unverified["execution_validation"] = dict(unverified["execution_validation"])
    unverified["execution_validation"]["test_command_verified"] = False
    assert bundle_protocol_eligibility(unverified, protocol)["eligible"] is True
    assert bundle_executable_evaluation_eligibility(unverified)["eligible"] is False
    assert "test_command_not_verified" in bundle_executable_evaluation_eligibility(unverified)["blockers"]
    disallowed = dict(bundles["fixture-train-clean"])
    disallowed["repository_rights"] = {"status": "allowed", "license": "GPL-3.0-only"}
    assert "repository_license_not_allowlisted" in bundle_protocol_eligibility(disallowed, protocol)["blockers"]

    clean_split = bundle_split_eligibility(bundles["fixture-train-clean"], split_manifest, protocol)
    assert clean_split["eligible"] is True, clean_split
    mismatch = bundle_split_eligibility(bundles["fixture-dev-window-repo-mismatch"], split_manifest, protocol)
    assert "repository_split_time_window_mismatch" in mismatch["blockers"], mismatch
    assert temporal_split("not-a-timestamp", protocol) is None

    quarantine = benchmark_quarantine_decision(bundles["fixture-benchmark-repository"], quarantine_manifest)
    assert quarantine["quarantine"] is True, quarantine
    reasons = {reason for match in quarantine["matches"] for reason in match["reasons"]}
    assert reasons == {"benchmark_repository_identity", "benchmark_exact_content_hash"}, reasons
    pending_task_artifacts = [
        row for row in quarantine_manifest["entries"] if row["task_artifact_manifest_status"] == "required_before_freeze"
    ]
    assert {row["benchmark"] for row in pending_task_artifacts} == {"SWE-bench"}
    print("[temporal-code-ingestion] training payload authorization and prose exclusion: pass")
    print("[temporal-code-ingestion] frozen repository split and benchmark quarantine: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
