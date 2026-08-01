#!/usr/bin/env python3
"""Validate the Stage-0 detector validation precheck."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load("170_build_stage0_detector_validation.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "validation" / "fixtures" / "stage0_detector_validation_cases.json",
            tmp_path / "stage0_detector_validation_report.json",
            tmp_path / "stage0_detector_validation_report.md",
        )
    assert report["status"] == "stage0_detector_validation_precheck_passed_with_scope_caveats"
    assert not report["blockers"]
    assert report["summary"]["case_count"] >= 13
    for axis, values in report["axis_metrics"].items():
        assert values["false_positive_count"] == 0, (axis, values)
        assert values["false_negative_count"] == 0, (axis, values)
        assert values["recall"] == 1.0, (axis, values)
    assert "requires_external_labeled_detector_benchmark_before_production_claim" in report["remaining_evidence_gaps"]
    print("[stage0-detector-validation] labeled axis precision/recall precheck: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
