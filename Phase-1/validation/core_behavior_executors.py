from __future__ import annotations

from typing import Any

from all_policy_stage_b import apply_quality_policy, apply_redundancy_policy
from ingestion.candidate_processing import process_candidate
from quality_operating_points import CurationMode
from quality_teacher_panel import PanelDecision
from quality_teacher_runtime import PanelPolicyResult
from redundancy_equivalence import RedundancyMode
from redundancy_v2 import RedundancySettings
from run_curation import _coverage_impact_audit, _stage_b_chunks
from stage_b_policy import propose_stage_b_removals


JsonMap = dict[str, Any]


def _expand_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("repeat_token"), str):
        return " ".join([value["repeat_token"]] * int(value.get("count") or 0))
    raise TypeError("Fixture text must be a string or repeat_token object.")


def _expand_chunks(chunks: list[JsonMap]) -> list[JsonMap]:
    return [{**chunk, "text": _expand_text(chunk["text"])} for chunk in chunks]


def _stage_b_records(records: list[JsonMap]) -> list[JsonMap]:
    return [
        {
            "record_id": record["record_id"],
            "text": _expand_text(record["text"]),
            "provenance": {
                "source_name": "fixture",
                "source_uri": "https://example.invalid",
                "collected_at": "2026-07-26T00:00:00Z",
            },
            "rights": {"license": "fixture-only"},
            "composition": {},
        }
        for record in records
    ]


def _composition_audit_stub() -> JsonMap:
    return {
        "delta_from_stage_b_pass": {
            "stage_c_curated": {"content_domain": {"token_share": {}}}
        },
    }


def execute_case(case: JsonMap) -> JsonMap:
    executor = case["executor"]
    expected_code = case["expected_code"]
    if executor == "stage_a_reason":
        record = process_candidate(case["raw"])
        triggered = expected_code in record["quarantine"]["reasons"]
        return _simple_event(triggered, "quarantine", expected_code)
    if executor == "stage_a_transformation":
        record = process_candidate(case["raw"])
        triggered = expected_code in record["transformations"]
        return _simple_event(triggered, "normalize", expected_code)
    if executor == "stage_b_reason":
        stage_b_policy = {
            "deduplicate_stage_a_text_exactly": True,
            **case["config"],
        }
        selected, rejected = _stage_b_chunks(
            _stage_b_records(case["records"]), stage_b_policy
        )
        matches = [
            row for row in rejected if expected_code in row["stage_b_hard_gate_reasons"]
        ]
        representative = (
            matches[0].get("stage_b_decision", {}).get("representative_chunk_uid")
            if matches
            else None
        )
        return {
            **_simple_event(bool(matches), "reject", expected_code),
            "representative_chunk_uid": representative,
            "representative_survived": representative in {
                row["chunk_uid"] for row in selected
            },
        }
    if executor == "stage_b_redundancy_v2":
        result = apply_redundancy_policy(
            _expand_chunks(case["chunks"]),
            mode=RedundancyMode(case["config"]["mode"]),
            settings=RedundancySettings(),
        )
        matches = [
            row
            for row in result.removals
            if row["stage_b_policy"].get("removed_reason") == expected_code
        ]
        match = matches[0] if matches else {}
        representative = match.get("stage_b_policy", {}).get(
            "representative_chunk_uid"
        )
        return {
            **_simple_event(bool(matches), "remove", expected_code),
            "representative_chunk_uid": representative,
            "representative_survived": representative in {
                row["chunk_uid"] for row in result.survivors
            },
        }
    if executor == "stage_b_structural_removal":
        selected, removed, _ = propose_stage_b_removals(
            _expand_chunks(case["chunks"]), case["config"]
        )
        matches = [
            row
            for row in removed
            if row["stage_b_policy"].get("removed_reason") == expected_code
        ]
        match = matches[0] if matches else {}
        representative = match.get("stage_b_policy", {}).get(
            "representative_chunk_uid"
        )
        return {
            **_simple_event(bool(matches), "remove", expected_code),
            "representative_chunk_uid": representative,
            "representative_survived": representative in {
                row["chunk_uid"] for row in selected
            },
            "quality_decision": match.get("quality_retention_decision"),
        }
    if executor == "stage_b_quality_ranker":
        payloads = case.get("policy_results") or [case["policy_result"]]
        panel_results = tuple(
            PanelPolicyResult(
                policy_id=str(payload["policy_id"]),
                decision=PanelDecision(payload["decision"]),
                first_pass=(),
                second_pass=None,
                decision_source="distilled_ranker",
                reason_codes=("controlled_ranker_fixture",),
                failure_probability=float(payload["failure_probability"]),
                normal_failure_threshold=float(payload["normal_failure_threshold"]),
                hard_failure_threshold=float(payload["hard_failure_threshold"]),
                prediction_confidence=float(payload["prediction_confidence"]),
                out_of_distribution=bool(payload.get("out_of_distribution", False)),
                ranker_artifact_sha256="f" * 64,
            )
            for payload in payloads
        )
        uid = "quality-fixture"
        result = apply_quality_policy(
            [{"chunk_uid": uid, "text": "fixture payload"}],
            results_by_chunk={uid: panel_results},
            mode=CurationMode(case["config"]["mode"]),
        )
        row = (result.not_selected or result.survivors)[0]
        decision = row["quality_stage_decision"]
        triggered = (
            decision["stage_b_action"] == "not_select"
            and decision["stage_b_reason_code"] == expected_code
        )
        return {
            **_simple_event(triggered, "not_select", expected_code),
            "quality_authority_kind": "distilled_ranker",
            "quality_decision": {
                **decision,
                "benchmark_outcomes_read": result.audit["benchmark_outcomes_read"],
                "utility_read": result.audit["utility_read"],
                "token_budget_read": result.audit["token_budget_read"],
            },
        }
    if executor == "coverage_invariant":
        return _coverage_event(case)
    raise ValueError(f"Unsupported fixture executor: {executor}")


def _simple_event(triggered: bool, action: str, reason_code: str) -> JsonMap:
    return {
        "triggered": triggered,
        "action": action if triggered else "retain",
        "reason_code": reason_code if triggered else None,
    }


def _coverage_event(case: JsonMap) -> JsonMap:
    audit = _coverage_impact_audit(
        passed=case["selected"],
        selected=case["selected"],
        rejected=case["rejected"],
        not_selected=case["not_selected"],
        span_transformations=[],
        minimum_residual_chars=40,
        composition_audit=_composition_audit_stub(),
    )
    return {
        "triggered": bool(audit["passed"]),
        "action": "audit",
        "reason_code": "coverage_invariants_passed" if audit["passed"] else None,
        "coverage": {
            "authority": audit["authority"],
            "selector_consumes_this_audit": audit["selector_consumes_this_audit"],
            "representative_linkage_passed": audit["representative_linkage"]["passed"],
            "zero_survivor_passed": audit["zero_survivor_invariant"]["passed"],
        },
    }
