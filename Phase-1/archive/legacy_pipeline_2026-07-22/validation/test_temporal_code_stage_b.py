#!/usr/bin/env python3
"""Regression checks for temporal-code Stage-B scoring and selection."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.code_selection import select_stage_b  # noqa: E402
from policy.dispositions import BUDGET_NOT_SELECTED, BUDGET_SELECTED  # noqa: E402


def _record(uid: str, text: str, kind: str = "code", bundle: str = "bundle-a") -> dict:
    return {
        "chunk_uid": uid,
        "split": "train",
        "stage_a_pass": True,
        "bundle_id": bundle,
        "repository_identity": "fixture/repo",
        "path": f"src/{uid}.py",
        "change_type": "modified",
        "content_type": kind,
        "chunk_kind": "function",
        "text": text,
    }


def main() -> int:
    rows = [
        _record("rich", "def normalize(values):\n    cleaned = [value.strip() for value in values if value]\n    return sorted(cleaned)\n"),
        _record("simple", "def value():\n    return 1\n"),
        _record("similar-a", "def parse(value):\n    if value is None:\n        return []\n    return value.split(',')\n"),
        _record("similar-b", "def parse_items(value):\n    if value is None:\n        return []\n    return value.split(',')\n"),
        _record("test", "def test_normalize():\n    assert normalize([' b ', 'a']) == ['a', 'b']\n", kind="test", bundle="bundle-b"),
    ]
    result = select_stage_b(
        rows,
        budget_fraction=0.55,
        quality_weight=0.8,
        redundancy_weight=0.2,
        coverage_axes=["bundle_id", "content_type", "change_type", "path_family", "difficulty_band"],
        minimum_exemplars=1,
        baseline_seed=42,
        distribution_axes=["bundle_id", "content_type", "difficulty_band"],
        minimum_relative_token_share=0.5,
    )
    selected_ids = {row["chunk_uid"] for row in result["selected"]}
    baseline_ids = {row["chunk_uid"] for row in result["baseline"]}
    assert selected_ids.isdisjoint(baseline_ids), result
    assert result["selected_token_proxy"] <= result["budget_token_proxy"], result
    assert all("utility" not in str(row.get("stage_b_evidence", {})).lower() for row in result["scored"]), result
    assert result["coverage_selected"]["bundle_id"].keys() == result["coverage_all"]["bundle_id"].keys(), result
    assert result["selection_mode"] == "budget_constrained"
    assert result["invariants"]["budget_not_selected_is_rejection"] is False
    assert len(result["curated_pool"]) == len(rows)
    assert all(
        row["curation_decision"]["curation_disposition"] == "retained"
        for row in result["curated_pool"]
    )
    assert all(
        row["curation_decision"]["training_budget_disposition"] == BUDGET_SELECTED
        for row in result["selected"]
    )
    assert all(
        row["curation_decision"]["training_budget_disposition"] == BUDGET_NOT_SELECTED
        and row["curation_decision"]["budget_exclusion_is_rejection"] is False
        for row in result["budget_not_selected"]
    )

    retain_all = select_stage_b(
        rows,
        budget_fraction=None,
        quality_weight=0.8,
        redundancy_weight=0.2,
        coverage_axes=["bundle_id", "content_type", "change_type", "path_family", "difficulty_band"],
        minimum_exemplars=1,
        baseline_seed=42,
    )
    assert retain_all["selection_mode"] == "retain_all"
    assert retain_all["budget_applied"] is False
    assert len(retain_all["selected"]) == len(rows)
    assert not retain_all["budget_not_selected"]
    assert not retain_all["baseline"]
    assert retain_all["selected_token_proxy"] == retain_all["full_curated_pool_token_proxy"]
    assert all(
        row["curation_decision"]["training_budget_disposition"] == "not_requested"
        for row in retain_all["selected"]
    )
    print("[temporal-code-stage-b] Core-only objective and disjoint baseline: pass")
    print("[temporal-code-stage-b] frozen coverage exemplars retained: pass")
    print("[temporal-code-stage-b] retain-all and budget-not-selected semantics: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
