#!/usr/bin/env python3
"""Validate Redundancy cluster-dropout audit contract."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "177_build_redundancy_cluster_dropout_audit.py"
    spec = importlib.util.spec_from_file_location("redundancy_cluster_dropout_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    stage0 = ROOT / "outputs" / "temporal_code_collection" / "stage0_code_domain_v2_combined"
    scored = (
        ROOT
        / "outputs"
        / "temporal_code_collection"
        / "stage_b_code_domain_v2"
        / "train_scored_full_selector.jsonl"
    )
    selected = scored.parent / "curated_v2_equal_budget.jsonl"
    if not stage0.exists() or not scored.exists() or not selected.exists():
        print("[redundancy-cluster-dropout] skipped: frozen corpus artifacts unavailable")
        return 0
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = module.build(
            stage0,
            ROOT / "configs" / "temporal_code_hard_near_duplicate_threshold_arms_v1.json",
            scored,
            selected,
            root / "report.json",
            root / "report.md",
        )
    assert report["status"] == "redundancy_cluster_dropout_audit_ready"
    assert report["arms"]["current"]["accepted_count"] > 0
    assert report["arms"]["zero_dropout_candidate"]["accepted_count"] > 0
    assert report["current_accepted_lost_by_challenger"]["lost_record_count"] >= 0
    assert report["utility_scope"].startswith("Stage C only")
    print("[redundancy-cluster-dropout] threshold-arm loss and Stage-B threat accounting: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
