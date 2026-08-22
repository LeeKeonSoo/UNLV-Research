from __future__ import annotations

from typing import Any

from all_policy_stage_b import apply_quality_policy, apply_redundancy_policy
from ingestion.candidate_processing import process_candidate
from quality_model_evidence import (
    QualityDecision,
    QualityPolicyEvidence,
    TeacherQualityPolicyEvidence,
)
from repeated_sentence_compaction import (
    REASON_CODE as REPEATED_SENTENCE_REASON_CODE,
    RepeatedSentenceSettings,
    compact_repeated_sentences,
)
from redundancy_equivalence import RedundancyMode
from redundancy_v2 import RedundancySettings
from run_curation import (
    _coverage_impact_audit,
    _stage_a_chunks,
    _stage_b_exact_duplicates,
)
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
    if executor == "stage_a_chunk_reason":
        selected, rejected = _stage_a_chunks(
            _stage_b_records(case["records"]),
            max_chunk_chars=int(case["config"]["max_chunk_chars"]),
        )
        matches = [
            row for row in rejected if expected_code in row["stage_a_hard_gate_reasons"]
        ]
        return {
            **_simple_event(bool(matches), "reject", expected_code),
            "representative_chunk_uid": None,
            "representative_survived": not matches and bool(selected),
        }
    if executor == "stage_b_exact_reason":
        stage_a_passed, stage_a_rejected = _stage_a_chunks(
            _stage_b_records(case["records"]),
            max_chunk_chars=int(case["config"]["max_chunk_chars"]),
        )
        if stage_a_rejected:
            raise AssertionError("Stage-B fixture unexpectedly failed Stage A")
        selected, rejected = _stage_b_exact_duplicates(stage_a_passed, enabled=True)
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
    if executor == "stage_b_repeated_sentence":
        result = compact_repeated_sentences(
            _expand_chunks(case["chunks"]),
            RepeatedSentenceSettings(**case["config"]),
        )
        matches = [
            item
            for item in result.transformations
            if item["reason_code"] == REPEATED_SENTENCE_REASON_CODE
        ]
        return {
            **_simple_event(bool(matches), "transform", expected_code),
            "representative_chunk_uid": (
                matches[0]["representative_chunk_uid"] if matches else None
            ),
            "representative_survived": bool(result.records),
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
            QualityPolicyEvidence(
                policy_id=str(payload["policy_id"]),
                decision=QualityDecision(payload["decision"]),
                reason_codes=("controlled_ranker_fixture",),
                class_probabilities=(
                    ("pass", 1.0 if payload["decision"] == "pass" else 0.0),
                    ("fail", 1.0 if payload["decision"] == "fail" else 0.0),
                    ("abstain", 1.0 if payload["decision"] == "abstain" else 0.0),
                ),
                failure_probability=float(payload["failure_probability"]),
                failure_threshold=float(payload["failure_threshold"]),
                prediction_confidence=float(payload["prediction_confidence"]),
                minimum_decision_confidence=0.7,
                out_of_distribution=bool(payload.get("out_of_distribution", False)),
                ranker_artifact_sha256="f" * 64,
            )
            for payload in payloads
        )
        uid = "quality-fixture"
        teacher_results = None
        if (
            case.get("teacher_evidence_complete")
            or case.get("teacher_positive_support")
            or case.get("teacher_failures")
        ):
            teacher_positive_support = frozenset(
                str(policy_id) for policy_id in case.get("teacher_positive_support") or ()
            )
            teacher_failures = frozenset(
                str(policy_id) for policy_id in case.get("teacher_failures") or ()
            )
            teacher_results = tuple(
                TeacherQualityPolicyEvidence(
                    policy_id=policy_id,
                    decision=(
                        QualityDecision.FAIL
                        if policy_id in teacher_failures
                        else QualityDecision.PASS
                        if policy_id in teacher_positive_support
                        else QualityDecision.ABSTAIN
                    ),
                    reason_codes=("controlled_teacher_fixture",),
                    observation_sha256="e" * 64,
                )
                for policy_id in (
                    "q1_correctness_evidence",
                    "q2_semantic_coherence",
                    "q3_substantive_payload",
                    "q4_learnable_relations",
                )
            )
        result = apply_quality_policy(
            [{"chunk_uid": uid, "text": "fixture payload"}],
            results_by_chunk={uid: panel_results},
            teacher_results_by_chunk=(
                None if teacher_results is None else {uid: teacher_results}
            ),
        )
        row = (result.not_selected or result.survivors)[0]
        decision = row["quality_stage_decision"]
        triggered = decision["stage_b_action"] == "not_select"
        return {
            "triggered": triggered,
            "action": decision["stage_b_action"],
            "reason_code": decision["stage_b_reason_code"],
            "quality_authority_kind": decision["decision_source"],
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
