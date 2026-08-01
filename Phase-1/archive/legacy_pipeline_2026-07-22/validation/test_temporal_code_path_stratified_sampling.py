#!/usr/bin/env python3
"""Contract checks for path-only temporal-code sampling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def _module(name: str):
    path = PROJECT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    collector = _module("90_collect_temporal_code_pr_path_metadata.py")
    freezer = _module("91_freeze_temporal_code_path_stratified_tranche.py")
    suffixes = [".py", ".md", ".rst", ".toml", ".cfg", ".ini", ".txt"]
    assert collector.classify_changed_paths(["src/a.py", "tests/test_a.py"], suffixes)["path_stratum"] == (
        "code_and_test"
    )
    assert collector.classify_changed_paths(["src/a.py", "README.md"], suffixes)["path_stratum"] == "code_only"
    assert collector.classify_changed_paths(["tests/test_a.py"], suffixes)["path_stratum"] == "test_only"
    assert collector.classify_changed_paths(["README.md"], suffixes)["path_stratum"] == "documentation_only"
    assert freezer._quantile_indices(10, 4) == [0, 3, 6, 9]
    command_freezer = _module("87_freeze_temporal_code_broad_test_commands.py")
    excluded_names = {path.name for path in command_freezer._bundle_paths(PROJECT_DIR / "validation" / "fixtures")}
    assert "path_stratified_tranche_fetch_report.json" not in excluded_names
    assert "path_stratified_tranche_bundle_audit_report.json" not in excluded_names
    verifier = _module("72_verify_temporal_code_test_commands.py")
    candidate_ids = verifier._execution_candidate_ids(
        {
            "decisions": [
                {
                    "bundle_id": "eligible",
                    "collection_gate_pass": True,
                    "executable_evaluation_blockers": ["test_command_not_verified"],
                },
                {
                    "bundle_id": "content_blocked",
                    "collection_gate_pass": False,
                    "executable_evaluation_blockers": ["test_command_not_verified"],
                },
                {
                    "bundle_id": "other_exec_blocker",
                    "collection_gate_pass": True,
                    "executable_evaluation_blockers": ["test_command_not_verified", "other"],
                },
            ]
        }
    )
    assert candidate_ids == {"eligible"}
    for name in (
        "temporal_code_path_stratified_tranche_v1.json",
        "temporal_code_path_stratified_tranche_v2.json",
    ):
        contract = load_json(PROJECT_DIR / "configs" / name)
        forbidden = set(contract["selection_forbids"])
        assert {"file content", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(forbidden)
        assert contract["one_pull_request_per_repository"] is True
        assert contract["utility_scope"] == "Stage C validation only; never selector objective"
    output = (
        PROJECT_DIR
        / "outputs"
        / "temporal_code_collection"
        / "temporal_code_path_stratified_tranche_plan.json"
    )
    if output.exists():
        plan = load_json(output)
        assert plan["status"] in {"frozen_before_tranche_content_fetch", "insufficient_sampling_frame"}
        assert not plan["selected_repositories"] or all(
            len(row["sampled_prs"]) == 1
            for rows in plan["selected_repositories"].values()
            for row in rows
        )
    print("[temporal-code-path-stratified] path-only no-leak sampling contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
