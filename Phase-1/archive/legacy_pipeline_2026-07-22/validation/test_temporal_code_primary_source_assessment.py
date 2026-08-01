#!/usr/bin/env python3
"""Contract checks for primary temporal executable-source assessment."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_primary_executable_source_assessment.json"
    )
    contract = report["contract"]
    assert report["status"] == (
        "primary_temporal_executable_distribution_not_currently_acquirable_from_frozen_sources"
    )
    assert report["summary"]["required_primary_task_count"] == 1083
    assert report["summary"]["current_primary_temporal_e2_task_count"] == 2
    assert report["summary"]["task_count_gap"] == 1081
    assert report["summary"]["evalplus_e2_guardrail_frozen"] is True
    assert report["summary"]["current_public_source_meets_primary_contract"] is False
    assert report["decision"]["development_utility_may_start"] is False
    assert report["decision"]["retroactive_contract_weakening_allowed"] is False
    assert "lower the frozen task-count requirement because current sources are insufficient" in contract[
        "forbidden_reactions"
    ]
    assert report["confirmatory_outcomes_read"] is False
    print("[temporal-code-primary-source] explicit forward-acquisition abstention: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
