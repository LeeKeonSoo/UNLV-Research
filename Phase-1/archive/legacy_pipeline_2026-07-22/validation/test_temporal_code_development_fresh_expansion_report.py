#!/usr/bin/env python3
"""Contract checks for fresh development expansion decision."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_development_fresh_expansion_report.json"
    )
    summary = report["summary"]
    decision = report["decision"]
    assert report["status"] == "raw_repository_execution_support_insufficient"
    assert summary["fresh_repository_count"] == 14
    assert summary["collection_gate_pass_bundles"] == 12
    assert summary["native_build_pass_commits"] >= summary["generic_build_pass_commits"]
    assert summary["native_verified_bundles"] == 0
    assert summary["total_verified_development_bundles"] == 1
    assert decision["development_utility_may_start"] is False
    assert decision["broaden_raw_repository_discovery_for_execution_recovery"] is False
    assert report["confirmatory_outcomes_read"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-development-fresh-report] execution-support architecture boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
