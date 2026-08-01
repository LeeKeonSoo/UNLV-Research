#!/usr/bin/env python3
"""Contract checks for broad temporal-code tranche readiness evidence."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    path = PROJECT_DIR / "outputs" / "validation" / "temporal_code_broad_tranche_readiness.json"
    if not path.exists():
        print("[temporal-code-broad-tranche-readiness] generated report absent; contract test skipped")
        return 0
    report = load_json(path)
    assert report["schema_version"] == "temporal-code-broad-tranche-readiness-v1"
    assert report["engineering_contracts"]["stage_b_selected_and_baseline_disjoint"] is True
    assert report["engineering_contracts"]["indexed_all_pairs_equivalent"] is True
    assert report["status"] == "stage_b_operationally_valid_stage_c_not_ready"
    assert report["stage_c_blockers"]
    assert report["summary"]["executable_evaluation_gate_pass_by_split"] == {
        "train": 0,
        "development": 0,
        "confirmatory": 0,
    }
    assert "no_executable_development_holdout" in report["stage_c_blockers"]
    assert "no_executable_confirmatory_holdout" in report["stage_c_blockers"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-broad-tranche-readiness] operational pass and Stage-C blockers separated: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
