#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from span_level_template_compaction import build_candidate_impact_audit, build_plan, materialize_candidate_plan


FIXTURE = ROOT / "validation" / "fixtures" / "span_level_template_compaction_fixture_v1.json"


def test_span_compaction_plan_proposes_only_nonrepresentative_spans_with_sufficient_residual_payload() -> None:
    repeated = "This repeated boilerplate explains the stable transport contract shared by every generated client implementation."
    plan = build_plan(
        [
            {"record_id": "a", "text": f"{repeated}\n\nRecord A payload explains retry behavior and timeout recovery in detail."},
            {"record_id": "b", "text": f"{repeated}\n\nRecord B payload explains authentication behavior and token refresh in detail."},
        ],
        minimum_span_tokens=12,
        minimum_residual_tokens=8,
    )

    assert plan["status"] == "candidate_only_not_a_selection_policy"
    assert plan["proposed_span_removals"] == 1
    assert plan["records_with_proposed_compaction"] == ["b"]
    assert plan["proposals"][0]["representative_record_id"] == "a"
    assert plan["proposals"][0]["residual_token_proxy"] >= 8
    assert plan["selector_consumes_this_plan"] is False


def test_span_compaction_plan_does_not_remove_a_record_with_no_independent_payload() -> None:
    repeated = "This repeated boilerplate explains the stable transport contract shared by every generated client implementation."
    plan = build_plan(
        [
            {"record_id": "a", "text": f"{repeated}\n\nRepresentative payload remains detailed and independently useful for training."},
            {"record_id": "b", "text": repeated},
        ],
        minimum_span_tokens=12,
        minimum_residual_tokens=8,
    )

    assert plan["proposed_span_removals"] == 0
    assert plan["blocked_empty_or_short_residual_records"] == ["b"]


def test_candidate_materialization_removes_only_planned_span_and_emits_reversible_trace() -> None:
    repeated = "This repeated boilerplate explains the stable transport contract shared by every generated client implementation."
    rows = [
        {"record_id": "a", "text": f"{repeated}\n\nRecord A payload explains retry behavior and timeout recovery in detail."},
        {"record_id": "b", "text": f"{repeated}\n\nRecord B payload explains authentication behavior and token refresh in detail."},
    ]
    plan = build_plan(rows, minimum_span_tokens=12, minimum_residual_tokens=8)
    result = materialize_candidate_plan(rows, plan)

    assert result["status"] == "candidate_materialization_not_runtime_active"
    assert result["records"][0]["text"] == rows[0]["text"]
    assert repeated not in result["records"][1]["text"]
    assert "Record B payload" in result["records"][1]["text"]
    assert result["records"][1]["token_proxy"] == len(result["records"][1]["text"].split())
    transformation = result["transformations"][0]
    assert transformation["record_id"] == "b"
    assert transformation["chunk_uid"] == "b"
    assert transformation["reason_code"] == "repeated_exact_template_span_removed"
    assert transformation["span_sha256"] == plan["proposals"][0]["span_sha256"]
    assert transformation["representative_record_id"] == "a"
    assert transformation["representative_chunk_uid"] == "a"
    assert transformation["span_token_proxy"] == plan["proposals"][0]["span_token_proxy"]
    assert transformation["pre_token_proxy"] == 25
    assert transformation["post_token_proxy"] == 11
    assert result["runtime_authorization"] == "none_candidate_cannot_select_or_remove"


def test_candidate_materialization_keeps_near_match_and_payload_free_false_positives() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = fixture["rows"]
    plan = build_plan(
        rows,
        minimum_span_tokens=fixture["minimum_span_tokens"],
        minimum_residual_tokens=fixture["minimum_residual_tokens"],
    )
    result = materialize_candidate_plan(rows, plan)

    transformed = [item["record_id"] for item in result["transformations"]]
    texts = {row["chunk_uid"]: row["text"] for row in result["records"]}
    assert fixture["schema_version"] == "span-level-template-compaction-fixture-v1"
    assert transformed == fixture["expected_transformed_chunk_uids"]
    assert [item["chunk_uid"] for item in result["transformations"]] == fixture["expected_transformed_chunk_uids"]
    assert all("stable transport contract" in texts[chunk_uid] for chunk_uid in fixture["expected_retained_chunk_uids"])


def test_candidate_impact_audit_reports_text_only_delta_without_chunk_deletion() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    audit = build_candidate_impact_audit(
        fixture["rows"],
        minimum_span_tokens=fixture["minimum_span_tokens"],
        minimum_residual_tokens=fixture["minimum_residual_tokens"],
    )

    assert audit["authority"] == "candidate_only_text_structural_audit"
    assert audit["selector_consumes_this_audit"] is False
    assert audit["runtime_active"] is False
    assert audit["stage_b_pass_chunks_before"] == 4
    assert audit["stage_b_pass_chunks_after"] == 4
    assert audit["chunks_removed"] == 0
    assert audit["chunks_transformed"] == 1
    assert audit["lexical_token_proxy_removed"] > 0


def test_plan_counts_every_planned_occurrence_that_materialization_removes() -> None:
    repeated = "This repeated boilerplate explains the stable transport contract shared by every generated client implementation."
    rows = [
        {"chunk_uid": "a::0000", "text": f"{repeated}\n\nRepresentative payload remains detailed and independently useful for training."},
        {"chunk_uid": "b::0000", "text": f"{repeated}\n\n{repeated}\n\nCandidate payload remains detailed and independently useful for training."},
    ]
    plan = build_plan(rows, minimum_span_tokens=12, minimum_residual_tokens=8)
    materialized = materialize_candidate_plan(rows, plan)

    assert plan["proposed_span_removals"] == 2
    assert len(materialized["transformations"]) == 2


if __name__ == "__main__":
    test_span_compaction_plan_proposes_only_nonrepresentative_spans_with_sufficient_residual_payload()
    test_span_compaction_plan_does_not_remove_a_record_with_no_independent_payload()
    test_candidate_materialization_removes_only_planned_span_and_emits_reversible_trace()
    test_candidate_materialization_keeps_near_match_and_payload_free_false_positives()
    test_candidate_impact_audit_reports_text_only_delta_without_chunk_deletion()
    test_plan_counts_every_planned_occurrence_that_materialization_removes()
    print("[span-level-template-compaction] payload-preserving candidate plan: pass")
