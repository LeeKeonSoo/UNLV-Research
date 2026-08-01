#!/usr/bin/env python3
"""Regression checks for isolated temporal-code test verification."""

from __future__ import annotations

import importlib
import copy
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    verifier = importlib.import_module("72_verify_temporal_code_test_commands")
    auditor = importlib.import_module("71_audit_temporal_code_smoke_bundles")
    commands = load_json(PROJECT_DIR / "configs" / "temporal_code_smoke_test_commands_v1.json")
    assert commands["status"] == "refrozen_before_sixth_execution", commands
    assert commands["repository_commands"]["scrapy/scrapy"]["writable_workspace_copy"] is True, commands
    assert "pytest-twisted>=1.14.3" in commands["repository_commands"]["scrapy/scrapy"]["install_arguments"], commands
    assert "pexpect>=4.8.0" in commands["repository_commands"]["scrapy/scrapy"]["install_arguments"], commands
    isolation = commands["isolation_contract"]
    assert isolation["host_execution_forbidden"] is True, isolation
    assert isolation["test_network"] == "none", isolation
    assert isolation["root_filesystem"] == "read_only", isolation
    assert "/root:rw,noexec,nosuid,size=512m" in isolation["common_writable_tmpfs"], isolation

    dockerfile = verifier._dockerfile("python:3.11-slim", "https://github.com/fixture/repo", "a" * 40, ["-e", ".", "pytest"])
    assert "pytest -q" not in dockerfile, dockerfile
    assert "git checkout --detach" in dockerfile, dockerfile
    with tempfile.TemporaryDirectory() as directory:
        report = verifier.verify(commands, [], Path(directory), dry_run=True)
    assert report["summary"]["bundle_count"] == 0, report

    fixture = copy.deepcopy(
        load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_change_bundles.json")["bundles"][0]
    )
    fixture["execution_validation"]["test_command"] = "unverified"
    fixture["execution_validation"]["test_command_verified"] = False
    verified = {
        "schema_version": "temporal-code-smoke-test-verification-v1",
        "dry_run": False,
        "decisions": [
            {
                "bundle_id": fixture["bundle_id"],
                "test_command": ["python", "-m", "pytest", "-q"],
                "test_command_verified": True,
            }
        ],
    }
    overlaid = auditor.bundle_with_test_verification(fixture, verified)
    assert overlaid["execution_validation"]["test_command_verified"] is True, overlaid
    assert fixture["execution_validation"]["test_command_verified"] is False, fixture
    print("[temporal-code-test-verification] Docker-only isolation contract: pass")
    print("[temporal-code-test-verification] audit overlay preserves raw bundle: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
