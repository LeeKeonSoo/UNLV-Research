#!/usr/bin/env python3
"""Validate the real-corpus Redundancy silver calibration contract."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "174_build_real_corpus_redundancy_calibration.py"
    spec = importlib.util.spec_from_file_location("real_corpus_redundancy_calibration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    source = (
        ROOT
        / "outputs"
        / "temporal_code_collection"
        / "stage_a_code_domain_v2_combined"
        / "train"
        / "stage_a_pass.jsonl"
    )
    if not source.exists():
        print("[real-corpus-redundancy] skipped: frozen Stage-A source is unavailable")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = module.build(
            source,
            root / "pairs.jsonl",
            root / "report.json",
            root / "report.md",
            per_stratum=2,
        )
    assert report["status"] == "redundancy_real_corpus_silver_calibration_ready"
    assert report["source_metadata"]["source_count"] >= 10
    assert report["source_metadata"]["source_count"] == report["source_metadata"]["source_repository_count"]
    assert report["summary"]["pair_count"] >= 40
    assert report["summary"]["label_counts"]["hard_duplicate"] > 0
    assert report["summary"]["label_counts"]["nonduplicate"] > 0
    assert report["threshold_sweep_top20"]
    assert {"content_type", "length_bucket", "transformation"} == set(report["current_stratified"])
    print("[real-corpus-redundancy] repository-disjoint source strata and silver calibration: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
