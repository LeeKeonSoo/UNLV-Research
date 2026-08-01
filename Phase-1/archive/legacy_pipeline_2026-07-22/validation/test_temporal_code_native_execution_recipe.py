#!/usr/bin/env python3
"""Contract checks for exploratory repository-native execution recipes."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    path = (
        PROJECT_DIR
        / "outputs"
        / "temporal_code_collection"
        / "temporal_code_development_native_test_commands_v1.json"
    )
    commands = load_json(path)
    assert commands["status"] == "refrozen_before_second_native_execution"
    assert commands["summary"]["repository_count"] == 11
    assert commands["contract"]["evidence_role"] == "post-generic-failure exploratory development diagnostic only"
    assert commands["contract"]["confirmatory_use"].startswith("forbidden")
    forbidden = set(commands["forbidden_inputs"])
    assert {"generic execution outcomes", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(
        forbidden
    )
    assert all(row["generic_execution_outcomes_read"] is False for row in commands["repository_commands"].values())
    assert all(row["writable_workspace_copy"] is True for row in commands["repository_commands"].values())
    assert commands["summary"]["nondefault_python_image_count"] >= 1
    assert commands["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-native-recipe] post-failure exploratory no-outcome-input contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
