#!/usr/bin/env python3
"""Regression check for indexed temporal-code Stage-B redundancy equivalence."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    module = importlib.import_module("77_validate_temporal_code_stage_b_index_equivalence")
    fixture = load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_stage_b_proxy_cases.json")
    records = [
        {
            **row,
            "split": "train",
            "stage_a_pass": True,
            "bundle_id": "fixture-bundle",
            "repository_identity": "fixture/repo",
            "change_type": "modified",
            "chunk_kind": "function",
        }
        for row in fixture["records"]
    ]
    protocol = load_json(PROJECT_DIR / "configs" / "temporal_code_curation_protocol_v1.json")
    report = module.validate(records, protocol)
    assert report["summary"]["passed"] is True, report
    assert report["summary"]["selected_symmetric_difference_count"] == 0, report
    assert report["summary"]["baseline_symmetric_difference_count"] == 0, report
    print("[temporal-code-stage-b-index] indexed and all-pairs decisions equivalent: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
