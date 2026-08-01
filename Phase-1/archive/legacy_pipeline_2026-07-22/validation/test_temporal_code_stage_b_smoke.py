#!/usr/bin/env python3
"""Regression checks for the train-only temporal-code Stage-B smoke runner."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(uid: str, bundle: str, content_type: str, text: str) -> dict:
    return {
        "chunk_uid": uid,
        "split": "train",
        "stage_a_pass": True,
        "bundle_id": bundle,
        "repository_identity": "fixture/repo",
        "path": f"{'tests' if content_type == 'test' else 'src/core'}/{uid}.py",
        "change_type": "modified",
        "content_type": content_type,
        "chunk_kind": "function",
        "text": text,
    }


def main() -> int:
    module = importlib.import_module("75_run_temporal_code_stage_b_smoke")
    protocol = {
        "stage_b_contract": {
            "input": "train split Stage-A-pass chunks only",
            "budget": {"fraction": 0.45},
            "objective": {
                "code_quality_proxy_weight": 0.8,
                "soft_redundancy_support_weight": 0.2,
                "forbidden_signals": ["Utility", "benchmark outcomes", "development outcomes", "confirmatory outcomes"],
            },
            "coverage_support": {
                "role": "selection constraint only; never part of the ranking objective",
                "axes": ["bundle_id", "content_type", "change_type", "path_family", "difficulty_band"],
                "minimum_exemplars_per_observed_value": 1,
                "distribution_axes": ["bundle_id", "content_type", "difficulty_band"],
                "minimum_relative_token_share": 0.5,
            },
            "stage_a_random_baseline": {"seed": 42, "must_be_disjoint_from_selected": True},
            "claim_boundary": "fixture",
        }
    }
    rows = [
        _row("a", "bundle-a", "code", "def transform(values):\n    return sorted(value.strip() for value in values if value)\n"),
        _row("b", "bundle-a", "code", "def identity(value):\n    return value\n"),
        _row("c", "bundle-b", "test", "def test_transform():\n    assert transform([' b ', 'a']) == ['a', 'b']\n"),
        _row("d", "bundle-b", "test", "def test_identity():\n    assert identity(1) == 1\n"),
        _row("e", "bundle-a", "code", "def parse(value):\n    if value is None:\n        return []\n    return value.split(',')\n"),
        _row("f", "bundle-b", "test", "def test_parse():\n    assert parse('a,b') == ['a', 'b']\n"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_jsonl(root / "stage_a" / "train" / "stage_a_pass.jsonl", rows)
        (root / "stage_a" / "development").mkdir(parents=True)
        (root / "stage_a" / "development" / "stage_a_pass.jsonl").write_text("not-json\n", encoding="utf-8")
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        report = module.run(root / "stage_a", protocol_path, root / "out")
        _write_jsonl(root / "empty" / "train" / "stage_a_pass.jsonl", [])
        empty_report = module.run(root / "empty", protocol_path, root / "empty-out")
    assert report["isolation"]["development_read"] is False, report
    assert report["isolation"]["confirmatory_read"] is False, report
    assert report["summary"]["selected_and_baseline_disjoint"] is True, report
    assert report["coverage_support"]["all_observed_values_retained"] is True, report
    assert report["coverage_support"]["all_distribution_floors_passed"] is True, report
    assert report["utility_scope"] == "Stage C validation only; never selector objective", report
    assert empty_report["operational_decision"] == "insufficient_usable_data", empty_report
    assert empty_report["summary"]["selected_chunks"] == 0
    print("[temporal-code-stage-b-smoke] train-only isolation and disjoint baseline: pass")
    print("[temporal-code-stage-b-smoke] coverage constraint and Utility exclusion: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
