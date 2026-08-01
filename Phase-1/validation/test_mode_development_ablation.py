#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mode_development_ablation import materialize_mode_development_arms


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_mode_ablation_materializes_disjoint_candidate_arms(tmp_path: Path) -> None:
    # Given: one frozen Weak corpus and complete Mid/Hard group decisions.
    weak_rows = [
        {"chunk_uid": "a", "text": "cookie preferences\naccept all\nreject all\nmanage preferences", "token_proxy": 8},
        {"chunk_uid": "b", "text": "def train(model): return model", "token_proxy": 10},
        {"chunk_uid": "c", "text": "def test(model): return model", "token_proxy": 10},
        {"chunk_uid": "d", "text": "A useful explanatory paragraph.", "token_proxy": 8},
    ]
    memberships = {"artifact": ["a"], "high-yield": ["b", "c"], "low-yield": ["d"]}
    mid_report = {
        "schema_version": "mid-quality-development-report-v1",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "groups": [
            {"group_id": "artifact", "decision": "candidate_remove"},
            {"group_id": "high-yield", "decision": "candidate_retain_positive"},
            {"group_id": "low-yield", "decision": "candidate_retain_positive"},
        ],
    }
    hard_plan = {
        "schema_version": "hard-quality-candidate-plan-v1",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "selected_groups": [{"group_id": "high-yield"}],
        "excluded_groups": [
            {"group_id": "artifact", "reason_code": "mid_quality_calibrated_non_positive_candidate"},
            {"group_id": "low-yield", "reason_code": "explicit_token_budget_exhausted"},
        ],
    }

    # When: candidate-only Weak, Mid, and Hard arms are materialized.
    report = materialize_mode_development_arms(
        weak_rows=weak_rows,
        group_memberships=memberships,
        mid_report=mid_report,
        hard_plan=hard_plan,
        output_dir=tmp_path,
    )

    # Then: each arm is explicit, auditable, and cannot authorize runtime selection.
    assert report["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert report["arms"]["weak"]["chunks"] == 4
    assert report["arms"]["mid"]["chunks"] == 3
    assert report["arms"]["hard"]["chunks"] == 2
    assert {row["chunk_uid"] for row in _read_jsonl(Path(report["arms"]["hard"]["dataset_path"]))} == {"b", "c"}
    assert report["arms"]["mid"]["removed_reasons"]["mid_quality_calibrated_non_positive_candidate"] == 1
    assert report["arms"]["hard"]["removed_reasons"]["explicit_token_budget_exhausted"] == 1
    assert report["composition_audit"]["authority"] == "audit_only"


def test_mode_ablation_rejects_ungrouped_weak_chunks(tmp_path: Path) -> None:
    # Given: a Weak chunk not represented in any frozen group membership.
    weak_rows = [{"chunk_uid": "missing", "text": "must not disappear", "token_proxy": 4}]

    # When / Then: materialization fails instead of silently dropping it.
    try:
        materialize_mode_development_arms(
            weak_rows=weak_rows,
            group_memberships={},
            mid_report={"schema_version": "mid-quality-development-report-v1", "runtime_authorization": "none_candidate_cannot_select_or_remove", "groups": []},
            hard_plan={"schema_version": "hard-quality-candidate-plan-v1", "runtime_authorization": "none_candidate_cannot_select_or_remove", "selected_groups": [], "excluded_groups": []},
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        assert "membership" in str(error)
    else:
        raise AssertionError("Every Weak chunk must have exactly one group membership")


def test_mode_ablation_rejects_hard_plan_that_reselects_mid_removal(tmp_path: Path) -> None:
    # Given: Mid has a calibrated removal group, but a malformed Hard plan reselects it.
    weak_rows = [{"chunk_uid": "artifact", "text": "cookie preferences", "token_proxy": 2}]
    memberships = {"artifact": ["artifact"]}
    mid_report = {
        "schema_version": "mid-quality-development-report-v1",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "groups": [{"group_id": "artifact", "decision": "candidate_remove"}],
    }
    hard_plan = {
        "schema_version": "hard-quality-candidate-plan-v1",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "selected_groups": [{"group_id": "artifact"}],
        "excluded_groups": [],
    }

    # When / Then: the containment boundary prevents Hard from undoing Mid removal.
    try:
        materialize_mode_development_arms(
            weak_rows=weak_rows,
            group_memberships=memberships,
            mid_report=mid_report,
            hard_plan=hard_plan,
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        assert "Mid" in str(error)
    else:
        raise AssertionError("Hard arm must remain a subset of Mid survivors")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        test_mode_ablation_materializes_disjoint_candidate_arms(root / "materialized")
        test_mode_ablation_rejects_ungrouped_weak_chunks(root / "invalid")
        test_mode_ablation_rejects_hard_plan_that_reselects_mid_removal(root / "containment")
    print("[mode-development-ablation] Weak/Mid/Hard candidate arms: pass")
