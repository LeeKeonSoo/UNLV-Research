#!/usr/bin/env python3
"""Validate the frozen Redundancy silver holdout evaluation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "176_evaluate_redundancy_silver_holdout.py"
    spec = importlib.util.spec_from_file_location("redundancy_silver_holdout", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    source = (
        ROOT
        / "outputs"
        / "temporal_code_collection"
        / "stage_a_code_domain_v2_combined"
        / "train"
        / "stage_a_pass.jsonl"
    )
    holdout = ROOT / "configs" / "temporal_code_redundancy_silver_holdout_v1.json"
    if not source.exists() or not holdout.exists():
        print("[redundancy-silver-holdout] skipped: frozen source or holdout unavailable")
        return 0
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = module.build(
            source,
            holdout,
            ROOT / "configs" / "temporal_code_hard_near_duplicate_threshold_arms_v1.json",
            root / "pairs.jsonl",
            root / "report.json",
            root / "report.md",
        )
    assert report["calibration_repository_overlap"] == 0
    assert report["pair_count"] >= 40
    assert report["current_arm"] == "current"
    assert set(report["arm_results"]) == {
        "current",
        "zero_dropout_candidate",
        "low_dropout_candidate",
        "moderate_dropout_candidate",
        "high_recall_candidate",
    }
    assert report["promotion_blockers"]
    print("[redundancy-silver-holdout] frozen arms, disjoint sources, and promotion blockers: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
