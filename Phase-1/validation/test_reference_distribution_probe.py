#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_reference_distribution_probe import run_probe


def _reference(record_id: str, text: str) -> dict[str, object]:
    return {"record_id": record_id, "text": text}


def _raw(record_id: str, text: str) -> dict[str, object]:
    return {"record_id": record_id, "text": text, "partition": {"source_tier": "raw_like"}}


def test_reference_distribution_probe_reports_held_out_source_role_fit() -> None:
    reference_train = [
        _reference("reference-train-1", "class StableClient:\n    def request(self, payload):\n        return serialize(payload)"),
        _reference("reference-train-2", "def parse_schema(value):\n    return validate_contract(value)"),
    ]
    candidates = [
        _raw("raw-train-1", "print('temporary generated output')"),
        _raw("raw-train-2", "x = 'random fragment'; y = x * 3"),
        _raw("raw-heldout", "console.log('raw artifact')"),
    ]
    calibration = [
        {"origin_record_id": "reference-heldout", "source_role_label": "reference_distribution_member", "text": "def validate_schema(value):\n    return parse_contract(value)"},
        {"origin_record_id": "raw-heldout", "source_role_label": "raw_like_nonmember", "text": "console.log('raw artifact')"},
    ]

    report = run_probe(reference_train, candidates, calibration, split_salt="fixture")

    assert report["status"] == "diagnostic_probe_complete_not_a_selection_policy"
    assert report["training"]["reference_positive_records"] == 2
    assert report["training"]["raw_like_negative_records"] == 2
    assert report["held_out_calibration"]["records"] == 2
    assert report["selector_boundary"]["utility_read"] is False
    assert report["selector_boundary"]["benchmark_outcomes_read"] is False


if __name__ == "__main__":
    test_reference_distribution_probe_reports_held_out_source_role_fit()
    print("[reference-distribution-probe] held-out source-role diagnostic: pass")
