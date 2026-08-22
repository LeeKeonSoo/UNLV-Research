from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from content_router import route_content
from quality_decision_contract import (
    EvidenceObservation,
    QualityRetentionContractError,
    QualityRetentionDecision,
    RoutingPreconditionTrace,
)
from quality_rule_evidence import (
    EMPTY_HTML_SHELL_REASON,
    EXPLICIT_GENERATED_ARTIFACT_RE,
    EXPLICIT_GENERATED_ARTIFACT_REASON,
    FORBIDDEN_SCORE_KEYS,
    LICENSE_COMMENT_ONLY_REASON,
    POLICY_ARTIFACT_SHA256,
    POLICY_FIXTURES,
    POLICY_IDS,
    POLICY_VERSIONS,
    WEB_CHROME_ONLY_REASON,
    is_empty_html_shell,
    is_license_comment_only,
    is_web_chrome_only,
)


JsonMap = dict[str, Any]
QUALITY_REJECT = "reject"
QUALITY_KEEP = "keep"
QUALITY_ABSTAIN_RETAIN = "abstain_retain"

def _decision(
    *,
    row: Mapping[str, Any],
    decision: str,
    evaluated_policy_ids: list[str],
    policy_id: str | None = None,
    reason_code: str | None = None,
    trigger: str | None = None,
    non_trigger_boundary: str,
    evidence: str,
    additional_observations: tuple[EvidenceObservation, ...] = (),
) -> JsonMap:
    uid = str(row["chunk_uid"])
    text = str(row.get("text") or "")
    routing = route_content(text)
    routing_trace = RoutingPreconditionTrace(
        status=routing["route_status"],
        confidence=routing["route_confidence"],
        routes=tuple(routing["route_labels"]),
        router_version=routing["router_version"],
    )
    fixture_ids = POLICY_FIXTURES.get(policy_id) if policy_id is not None else None
    observations = (
        (EvidenceObservation("matched_rule_evidence", evidence),) + additional_observations
        if decision == QUALITY_REJECT
        else ()
    )
    typed = QualityRetentionDecision(
        decision=decision,
        chunk_uid=uid,
        policy_scope_route="common_all_routes",
        routing_precondition=routing_trace,
        evaluated_policy_ids=tuple(evaluated_policy_ids),
        non_trigger_boundary=non_trigger_boundary,
        evidence=evidence,
        original_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        policy_artifact_sha256=POLICY_ARTIFACT_SHA256,
        token_delta_proxy=(
            -int(row.get("token_proxy") or len(text.split()) or 1)
            if decision == QUALITY_REJECT
            else 0
        ),
        policy_id=policy_id,
        policy_version=POLICY_VERSIONS.get(policy_id) if policy_id is not None else None,
        reason_code=reason_code,
        trigger=trigger,
        observed_evidence=observations,
        representative_fixture_id=fixture_ids[0] if fixture_ids is not None else None,
        false_positive_fixture_id=fixture_ids[1] if fixture_ids is not None else None,
    )
    return typed.to_mapping()


def evaluate_quality_retention(
    rows: Iterable[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> tuple[dict[str, JsonMap], JsonMap]:
    forbidden = FORBIDDEN_SCORE_KEYS.intersection(settings)
    if forbidden:
        raise RuntimeError(
            "Quality retention accepts named evidence rules, not scalar scores or retention targets: "
            + ", ".join(sorted(forbidden))
        )
    materialized = [dict(row) for row in rows]
    enabled = [name for name in POLICY_IDS if bool(settings.get(name, False))]
    evaluated_policy_ids = [POLICY_IDS[name] for name in enabled]
    generated_record_observations: dict[str, tuple[EvidenceObservation, ...]] = {}
    if "explicit_generated_artifact" in enabled:
        for source_row in materialized:
            source_text = str(source_row.get("text") or "")
            if not EXPLICIT_GENERATED_ARTIFACT_RE.search(source_text):
                continue
            record_id = str(source_row.get("stage_a_record_id") or source_row["chunk_uid"])
            marker = EvidenceObservation(
                "source_record_marker_chunk",
                f"{source_row['chunk_uid']}:{hashlib.sha256(source_text.encode('utf-8')).hexdigest()}",
            )
            generated_record_observations[record_id] = (
                *generated_record_observations.get(record_id, ()),
                marker,
            )
    decisions: dict[str, JsonMap] = {}
    counts = {QUALITY_REJECT: 0, QUALITY_KEEP: 0, QUALITY_ABSTAIN_RETAIN: 0}
    reason_counts: dict[str, int] = {}
    for row in materialized:
        uid = str(row["chunk_uid"])
        record_id = str(row.get("stage_a_record_id") or uid)
        text = str(row.get("text") or "")
        if record_id in generated_record_observations:
            decision = _decision(
                row=row,
                decision=QUALITY_REJECT,
                evaluated_policy_ids=evaluated_policy_ids,
                policy_id=POLICY_IDS["explicit_generated_artifact"],
                reason_code=EXPLICIT_GENERATED_ARTIFACT_REASON,
                trigger="explicit_generated_and_noneditable_source_record_marker",
                non_trigger_boundary="generated_declaration_without_noneditable_marker_is_retained",
                evidence="source_record_contains_generated_and_do_not_edit_declaration",
                additional_observations=generated_record_observations[record_id],
            )
        elif "license_comment_only_chunk" in enabled and is_license_comment_only(text):
            decision = _decision(
                row=row,
                decision=QUALITY_REJECT,
                evaluated_policy_ids=evaluated_policy_ids,
                policy_id=POLICY_IDS["license_comment_only_chunk"],
                reason_code=LICENSE_COMMENT_ONLY_REASON,
                trigger="comment_only_chunk_with_explicit_license_marker",
                non_trigger_boundary="executable_or_explanatory_payload_is_retained",
                evidence="chunk_contains_only_comment_lines_with_explicit_license_marker",
            )
        elif "empty_html_shell" in enabled and is_empty_html_shell(text):
            decision = _decision(
                row=row,
                decision=QUALITY_REJECT,
                evaluated_policy_ids=evaluated_policy_ids,
                policy_id=POLICY_IDS["empty_html_shell"],
                reason_code=EMPTY_HTML_SHELL_REASON,
                trigger="complete_html_wrapper_with_no_visible_lexical_payload",
                non_trigger_boundary="html_with_visible_text_or_embedded_content_is_retained",
                evidence="complete_html_wrapper_without_visible_lexical_payload",
            )
        elif "web_chrome_only_chunk" in enabled and is_web_chrome_only(text):
            decision = _decision(
                row=row,
                decision=QUALITY_REJECT,
                evaluated_policy_ids=evaluated_policy_ids,
                policy_id=POLICY_IDS["web_chrome_only_chunk"],
                reason_code=WEB_CHROME_ONLY_REASON,
                trigger="complete_explicit_cookie_control_panel_without_explanatory_prose",
                non_trigger_boundary="cookie_or_privacy_terms_in_substantive_text_are_retained",
                evidence="all_nonblank_lines_are_explicit_cookie_control_markers",
            )
        else:
            decision = _decision(
                row=row,
                decision=QUALITY_ABSTAIN_RETAIN,
                evaluated_policy_ids=evaluated_policy_ids,
                non_trigger_boundary="no_active_quality_rejection_rule_has_sufficient_evidence",
                evidence="no_explicit_non_payload_evidence",
            )
        decisions[uid] = decision
        counts[str(decision["decision"])] += 1
        reason_code = decision.get("reason_code")
        if reason_code is not None:
            reason_counts[str(reason_code)] = reason_counts.get(str(reason_code), 0) + 1
    return decisions, {
        "contract": "quality_retention_deletion_authority_v2",
        "decision_scope": "deterministic_structural_prefilter_only",
        "policy_artifact_sha256": POLICY_ARTIFACT_SHA256,
        "decision_counts": counts,
        "reason_code_counts": reason_counts,
        "active_policy_ids": evaluated_policy_ids,
        "intrinsic_quality_score_used": False,
        "weighted_score_used": False,
        "utility_read": False,
        "benchmark_outcomes_read": False,
        "abstain_action": "continue_to_model_quality_gate",
    }
