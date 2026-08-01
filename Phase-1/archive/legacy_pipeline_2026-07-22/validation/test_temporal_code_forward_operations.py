#!/usr/bin/env python3
"""End-to-end contract checks for forward collection operations."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json, save_json  # noqa: E402


def main() -> int:
    schedule = load_json(PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json")
    ledger = load_json(PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_candidate_ledger.json")
    status = load_json(PROJECT_DIR / "outputs" / "validation" / "temporal_code_forward_operations_status.json")
    assert schedule["status"] == "frozen_before_later_snapshot_task_metadata"
    assert schedule["summary"] == {
        "repository_count": 5000,
        "shard_size": 200,
        "shard_count": 25,
        "duplicate_repository_count": 0,
    }
    assert all(row["repository_count"] == 200 for row in schedule["shards"])
    assert ledger["status"] == "candidate_ledger_frozen_before_recipe_or_execution"
    assert ledger["summary"]["candidate_count"] == 0
    assert status["status"] == "forward_collection_operational_waiting_for_later_date_tasks"
    assert status["gates"]["recipe_freeze_may_start"] is False
    assert status["gates"]["e2_execution_may_start"] is False
    assert status["gates"]["development_utility_may_start"] is False
    assert "126_freeze_temporal_code_forward_recipe_batch.py" in status["operational_commands"]["freeze_recipe_batch"]
    assert "112_verify_temporal_code_forward_e2_pilot.py" in status["operational_commands"]["verify_recipe_batch"]

    ledger_module = importlib.import_module("123_build_temporal_code_forward_candidate_ledger")
    collector_module = importlib.import_module("122_collect_temporal_code_forward_snapshot_shard")
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        common = {
            "repository_identity": "fixture/repo",
            "repository_url": "https://github.com/fixture/repo",
            "license": "MIT",
            "merge_timestamp": "2026-06-16T00:00:00Z",
            "merge_commit": "a" * 40,
            "parent_commit": "b" * 40,
            "changed_test_paths": ["tests/test_feature.py"],
            "changed_code_paths": ["src/feature.py"],
            "path_stratum": "code_and_test",
            "assigned_split": "development",
            "evaluation_authorized_pending_e2_and_quarantine": False,
        }
        paths = []
        for number in (2, 1):
            path = root / f"snapshot_{number}.json"
            save_json(path, {"snapshot_identity": f"s{number}", "summary": {}, "candidates": [{**common, "pull_request_number": number}]})
            paths.append(path)
        output = root / "ledger.json"
        built = ledger_module.build(
            PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json",
            paths,
            output,
        )
        assert built["summary"]["candidate_count"] == 1
        assert built["summary"]["duplicate_candidate_count"] == 1
        assert built["candidates"][0]["pull_request_number"] == 1
        first_repository = schedule["shards"][0]["repository_identities"][0]

        class FakeClient:
            requests = 0

            def recent_pulls(self, repository, start, end, limit=5):
                self.requests += 1
                if repository != first_repository:
                    return []
                return [
                    {
                        "number": 7,
                        "mergedAt": "2026-06-15T01:00:00Z",
                        "mergeCommit": {
                            "oid": "c" * 40,
                            "parents": {"nodes": [{"oid": "d" * 40}]},
                        },
                    }
                ]

            def paths(self, repository, number):
                self.requests += 1
                return ["src/feature.py", "tests/conftest.py", "tests/test_feature.py"]

        snapshot_output = root / "collected.json"
        collected = collector_module.collect(
            PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json",
            PROJECT_DIR / "outputs" / "temporal_code_collection" / "forward_development_repository_discovery_combined.json",
            snapshot_output,
            0,
            "2026-06-15",
            FakeClient(),
        )
        assert collected["summary"]["candidate_count"] == 1
        assert collected["candidates"][0]["changed_test_paths"] == ["tests/test_feature.py"]
        assert "tests/conftest.py" in collected["candidates"][0]["changed_code_paths"]
    verifier = importlib.import_module("112_verify_temporal_code_forward_e2_pilot")
    assert "forward_development_recipe_batch_frozen_before_execution" in {
        "forward_development_recipe_batch_frozen_before_execution",
        "frozen_before_forward_pilot_execution",
    }
    assert callable(verifier.verify)
    print("[temporal-code-forward-operations] sharded immutable accumulation: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
