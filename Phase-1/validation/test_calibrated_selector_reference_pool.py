#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_calibrated_selector_reference_pool import build_reference_pool


def test_reference_pool_is_source_declared_and_excluded_from_candidate_scope() -> None:
    rows = [
        {"record_id": "reference", "text": "def stable_reference():\n    return 1\n", "partition": {"source_tier": "known_high_quality_reference", "source_dataset": "fixture/reference"}},
        {"record_id": "raw", "text": "def raw_candidate():\n    return 2\n", "partition": {"source_tier": "raw_like", "source_dataset": "fixture/raw"}},
    ]

    selected, report = build_reference_pool(rows, "known_high_quality_reference")

    assert [row["record_id"] for row in selected] == ["reference"]
    assert report["summary"]["reference_records"] == 1
    assert report["summary"]["selector_candidate_records_excluding_reference"] == 1
    assert report["selector_boundary"]["reference_records_may_be_scored"] is False


if __name__ == "__main__":
    test_reference_pool_is_source_declared_and_excluded_from_candidate_scope()
    print("[calibrated-selector-reference-pool] source-declared reference boundary: pass")
