#!/usr/bin/env python3
"""Validate the Redundancy calibration benchmark contract."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "173_build_redundancy_validity_benchmark.py"
    spec = importlib.util.spec_from_file_location("redundancy_validity_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = module.build(
            ROOT / "validation" / "fixtures" / "redundancy_validity_benchmark_cases.json",
            root / "report.json",
            root / "report.md",
        )
    assert report["schema_version"] == "redundancy-validity-benchmark-report-v1"
    assert report["summary"]["pair_count"] >= 10
    assert report["summary"]["hard_duplicate_count"] >= 4
    assert report["summary"]["related_useful_count"] >= 4
    assert report["threshold_sweep_top10"]
    assert report["saturation"]["match_count_strictly_increases"] is True
    assert "stage_b_soft_risk_not_saturation_magnitude_sensitive" in report["known_gaps"]
    by_id = {row["id"]: row for row in report["pairs"]}
    assert by_id["exact_python_copy"]["exact_match"] is True
    assert by_id["independent_parser_and_window"]["label"] == "independent"
    print("[redundancy-validity] labeled pair sweep and saturation diagnostic: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
