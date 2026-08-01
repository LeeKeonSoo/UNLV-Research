#!/usr/bin/env python3
"""Contract checks for forward E2 task acquisition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def _module(name: str, filename: str):
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    plan = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_e2_acquisition_plan.json"
    )
    contract = plan["contract"]
    assert plan["status"] == "feasibility_amendment_frozen_before_forward_pilot_execution_outcomes"
    assert contract["task_validity_contract"]["merge_requirement"].startswith("frozen changed tests pass")
    assert contract["task_validity_contract"]["parent_requirement"].startswith("the same merge-commit tests fail")
    assert contract["task_validity_contract"]["executable_test_path_rule"].startswith("Python basename starts")
    assert contract["infrastructure_pilot"]["task_reuse_for_development_or_confirmatory"] is False
    assert contract["future_primary_acquisition"]["development_window"]["target_task_count"] == 542
    assert contract["future_primary_acquisition"]["confirmatory_window"]["target_task_count"] == 541
    assert contract["future_primary_acquisition"]["shortfall_action"] == "abstain_and_continue_forward_acquisition"
    assert "Utility" in contract["selection_forbids"]
    assert plan["development_utility_may_start"] is False
    verifier = _module("forward_e2_verifier", "112_verify_temporal_code_forward_e2_pilot.py")
    recipe = {
        "python_image": "python:3.11-slim",
        "repository_url": "https://github.com/fixture/repository",
        "merge_commit": "a" * 40,
        "parent_commit": "b" * 40,
        "install_arguments": ["-e", ".", "pytest"],
        "frozen_test_targets": ["tests/test_feature.py"],
    }
    parent = verifier._dockerfile(recipe, "parent")
    merge = verifier._dockerfile(recipe, "merge")
    assert "git -C /workspace show" in parent and "/tmp/test-overlay" in parent
    assert "git -C /workspace show" not in merge
    print("[temporal-code-forward-e2] parent-fail/merge-pass acquisition contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
