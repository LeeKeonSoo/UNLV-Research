#!/usr/bin/env python3
"""Contract checks for frozen broad-tranche automated test commands."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def _verifier_module():
    path = PROJECT_DIR / "72_verify_temporal_code_test_commands.py"
    spec = importlib.util.spec_from_file_location("temporal_code_test_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    verifier = _verifier_module()
    assert verifier._execution_candidate_ids(
        {
            "decisions": [
                {"bundle_id": "run", "blockers": ["test_command_not_verified"]},
                {"bundle_id": "skip", "blockers": ["test_command_not_verified", "benchmark_quarantine_match"]},
            ]
        }
    ) == {"run"}
    path = PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_broad_test_commands_v1.json"
    if not path.exists():
        print("[temporal-code-broad-test-commands] generated commands absent; contract test skipped")
        return 0
    commands = load_json(path)
    assert commands["schema_version"] == "temporal-code-broad-test-commands-v1"
    assert commands["status"] == "frozen_before_execution"
    assert commands["summary"]["repository_count"] == 20
    assert commands["isolation_contract"]["host_execution_forbidden"] is True
    assert commands["isolation_contract"]["test_network"] == "none"
    forbidden = set(commands["forbidden_inputs"])
    assert {"Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(forbidden)
    print("[temporal-code-broad-test-commands] automated freeze and no-outcome-leak contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
