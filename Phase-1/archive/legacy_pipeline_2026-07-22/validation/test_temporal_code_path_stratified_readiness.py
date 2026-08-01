#!/usr/bin/env python3
"""Contract checks for the fresh path-stratified tranche readiness."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    path = PROJECT_DIR / "outputs" / "validation" / "temporal_code_path_stratified_tranche_readiness.json"
    report = load_json(path)
    assert report["status"] == "ready_for_stage_c_smoke"
    assert report["stage_c_blockers"] == []
    assert report["summary"]["executable_evaluation_gate_pass_by_split"] == {
        "train": 1,
        "development": 1,
        "confirmatory": 1,
    }
    diagnostics = report["distribution_diagnostics"]
    assert diagnostics["documentation_share"] < 0.02
    assert diagnostics["largest_bundle_share"] < 0.25
    assert diagnostics["selected_mean_soft_redundancy_risk"] < diagnostics[
        "stage_a_random_mean_soft_redundancy_risk"
    ]
    assert diagnostics["full_selector_matches_quality_only_ablation"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-path-stratified-readiness] corpus-side Stage-C smoke entry conditions pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
