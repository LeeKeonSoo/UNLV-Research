#!/usr/bin/env python3
"""Validate Core construct-validity review."""

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
    module = _load("163_build_core_construct_validity_review.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "configs" / "lm_curation_operational_framework_v1.json",
            ROOT / "configs" / "metric_spec_with_citations.json",
            tmp_path / "core_construct_validity_review.json",
            tmp_path / "core_construct_validity_review.md",
        )
    assert report["status"] == "core_construct_validity_review_passed"
    assert report["decision"]["quality_as_intrinsic_core"] == "rejected"
    assert report["decision"]["canonical_axis_name"] == "Selection Value Evidence"
    assert report["decision"]["quality_axis_operational_name"] == "observable_pre_outcome_selection_evidence"
    by_core = {row["core"]: row for row in report["core_reviews"]}
    assert by_core["Selection Value Evidence"]["construct_status"] == "defensible_as_observable_evidence"
    assert by_core["Utility"]["construct_status"] == "defensible_as_protocol_bound_outcome"
    print("[core-construct-validity] Selection Value Evidence replaces intrinsic Quality: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
