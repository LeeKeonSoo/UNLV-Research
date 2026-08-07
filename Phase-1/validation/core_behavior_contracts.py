from __future__ import annotations

from typing import Any


JsonMap = dict[str, Any]
CORE_DIMENSIONS = {
    "validity": ["reason_coded_action", "non_trigger_retention"],
    "redundancy": [
        "representative_linkage",
        "representative_survival",
        "non_trigger_retention",
    ],
    "quality": [
        "typed_deletion_authority",
        "observable_trigger",
        "false_positive_boundary",
    ],
    "coverage": [
        "materialization_invariant_authority",
        "representative_linkage_detection",
        "zero_survivor_detection",
    ],
}


def behavior_invariants(core: str, expected: bool, event: JsonMap) -> JsonMap:
    if core == "validity":
        checks = {
            "reason_coded_action": not expected
            or event["action"] in {"quarantine", "normalize", "reject"},
            "non_trigger_retention": expected or event["action"] == "retain",
        }
    elif core == "redundancy":
        checks = {
            "representative_linkage": not expected
            or bool(event.get("representative_chunk_uid")),
            "representative_survival": not expected
            or bool(event.get("representative_survived")),
            "non_trigger_retention": expected or event["action"] == "retain",
        }
    elif core == "quality":
        checks = {
            "typed_deletion_authority": not expected or _quality_authority_passed(event),
            "observable_trigger": not expected or bool(event.get("reason_code")),
            "false_positive_boundary": expected or event["action"] == "retain",
        }
    elif core == "coverage":
        coverage = event.get("coverage") or {}
        checks = {
            "materialization_invariant_authority": coverage.get("authority") == "materialization_invariant"
            and coverage.get("selector_consumes_this_audit") is False,
            "representative_linkage_detection": coverage.get(
                "representative_linkage_passed"
            )
            is expected,
            "zero_survivor_detection": coverage.get("zero_survivor_passed") is True,
        }
    else:
        raise ValueError(f"Unsupported Core: {core}")
    return {"checks": checks, "passed": all(checks.values())}


def _quality_authority_passed(event: JsonMap) -> bool:
    decision = event.get("quality_decision") or {}
    if event.get("quality_authority_kind") == "teacher_panel":
        return (
            decision.get("stage_b_action") == "remove"
            and decision.get("stage_b_reason_code")
            in {"quality_normal_unanimous_fail", "quality_hard_stable_majority_fail"}
            and bool(decision.get("failed_policy_ids"))
            and decision.get("benchmark_outcomes_read") is False
            and decision.get("utility_read") is False
            and decision.get("token_budget_read") is False
        )
    routing = decision.get("routing_precondition") or {}
    required = (
        "policy_id",
        "policy_version",
        "reason_code",
        "trigger",
        "observed_evidence",
        "representative_fixture_id",
        "false_positive_fixture_id",
        "original_text_sha256",
        "policy_artifact_sha256",
    )
    return (
        decision.get("schema_version") == "quality-retention-decision-v2"
        and decision.get("decision") == "reject"
        and all(decision.get(field) for field in required)
        and decision.get("token_delta_proxy", 0) < 0
        and routing.get("quality_evidence") is False
        and routing.get("may_authorize_removal") is False
    )
