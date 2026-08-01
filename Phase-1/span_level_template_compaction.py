from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


JsonMap = dict[str, Any]
LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _tokens(text: str) -> int:
    return len(LEXICAL_TOKEN_RE.findall(text))


def _row_id(row: JsonMap) -> str:
    """Use the Stage-B chunk identity when present, otherwise a record identity."""
    return str(row.get("chunk_uid") or row.get("record_id") or row.get("id") or "unknown")


def build_plan(
    rows: Iterable[JsonMap], *, minimum_span_tokens: int = 12, minimum_residual_tokens: int = 20,
    minimum_residual_chars: int | None = None,
) -> JsonMap:
    """Plan text-only span compaction without changing any candidate record."""
    if minimum_span_tokens < 1 or minimum_residual_tokens < 1 or minimum_residual_chars is not None and minimum_residual_chars < 1:
        raise ValueError("minimum span and residual tokens must be positive")
    paragraphs_by_record: dict[str, list[str]] = {}
    members_by_span: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        record_id = _row_id(row)
        paragraphs = _paragraphs(str(row.get("text") or ""))
        paragraphs_by_record[record_id] = paragraphs
        for normalized in {_normalize(paragraph) for paragraph in paragraphs}:
            if _tokens(normalized) >= minimum_span_tokens:
                members_by_span[normalized].add(record_id)
    removable_by_record: dict[str, set[str]] = defaultdict(set)
    representative_by_span: dict[str, str] = {}
    for normalized, members in members_by_span.items():
        if len(members) < 2:
            continue
        representative = min(members)
        representative_by_span[normalized] = representative
        for record_id in members:
            if record_id != representative:
                removable_by_record[record_id].add(normalized)
    proposals: list[JsonMap] = []
    blocked_records: list[str] = []
    for record_id, removable_spans in sorted(removable_by_record.items()):
        residual = [paragraph for paragraph in paragraphs_by_record[record_id] if _normalize(paragraph) not in removable_spans]
        residual_text = "\n\n".join(residual)
        residual_tokens = _tokens(residual_text)
        has_residual_payload = (
            len(residual_text) >= minimum_residual_chars if minimum_residual_chars is not None else residual_tokens >= minimum_residual_tokens
        )
        if not has_residual_payload:
            blocked_records.append(record_id)
            continue
        for occurrence_index, paragraph in enumerate(paragraphs_by_record[record_id]):
            normalized = _normalize(paragraph)
            if normalized not in removable_spans:
                continue
            proposals.append(
                {
                    "record_id": record_id,
                    "span_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "span_occurrence_index": occurrence_index,
                    "representative_record_id": representative_by_span[normalized],
                    "span_token_proxy": _tokens(normalized),
                    "residual_token_proxy": residual_tokens,
                }
            )
    return {
        "schema_version": "span-level-template-compaction-plan-v1",
        "status": "candidate_only_not_a_selection_policy",
        "method": "long_exact_normalized_paragraph_family_with_payload_preservation",
        "minimum_span_tokens": minimum_span_tokens,
        "minimum_residual_tokens": minimum_residual_tokens,
        "minimum_residual_chars": minimum_residual_chars,
        "candidate_family_count": sum(len(members) >= 2 for members in members_by_span.values()),
        "proposed_span_removals": len(proposals),
        "records_with_proposed_compaction": sorted({proposal["record_id"] for proposal in proposals}),
        "blocked_empty_or_short_residual_records": blocked_records,
        "proposals": proposals,
        "selector_consumes_this_plan": False,
        "claim_boundary": "The plan does not remove spans, records, or chunks. Promotion requires independent false-positive fixtures and frozen external validation.",
    }


def materialize_candidate_plan(rows: Iterable[JsonMap], plan: JsonMap) -> JsonMap:
    """Apply a candidate-only plan while preserving record-level removal evidence."""
    if plan.get("status") != "candidate_only_not_a_selection_policy":
        raise ValueError("Candidate materialization requires a candidate-only compaction plan")
    proposals = plan.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("Candidate compaction plan requires a proposal list")

    planned_by_record: dict[str, dict[str, list[JsonMap]]] = defaultdict(lambda: defaultdict(list))
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Candidate compaction proposal must be an object")
        record_id = str(proposal.get("record_id") or "")
        span_sha256 = str(proposal.get("span_sha256") or "")
        if not record_id or not span_sha256:
            raise ValueError("Candidate compaction proposal requires record and span identifiers")
        planned_by_record[record_id][span_sha256].append(proposal)

    materialized_records: list[JsonMap] = []
    transformations: list[JsonMap] = []
    for row in rows:
        record = dict(row)
        record_id = _row_id(record)
        original_text = str(record.get("text") or "")
        paragraphs = _paragraphs(original_text)
        planned = planned_by_record.get(record_id, {})
        removed: list[JsonMap] = []
        retained: list[str] = []
        for paragraph in paragraphs:
            normalized = _normalize(paragraph)
            span_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            proposals_for_span = planned.get(span_sha256, [])
            if not proposals_for_span:
                retained.append(paragraph)
            else:
                removed.append(proposals_for_span.pop(0))
        if not removed:
            materialized_records.append(record)
            continue

        compacted_text = "\n\n".join(retained)
        pre_token_proxy = _tokens(original_text)
        post_token_proxy = _tokens(compacted_text)
        minimum_residual_chars = plan.get("minimum_residual_chars")
        if isinstance(minimum_residual_chars, int):
            if len(compacted_text) < minimum_residual_chars:
                raise ValueError("Candidate materialization would violate Stage-B residual character threshold")
        elif post_token_proxy < int(plan["minimum_residual_tokens"]):
            raise ValueError("Candidate materialization would violate payload-preservation threshold")
        record["text"] = compacted_text
        record["token_proxy"] = len(compacted_text.split())
        materialized_records.append(record)
        for proposal in removed:
            transformations.append(
                {
                    "record_id": record_id,
                    "chunk_uid": record_id,
                    "reason_code": "repeated_exact_template_span_removed",
                    "span_sha256": proposal["span_sha256"],
                    "span_occurrence_index": proposal["span_occurrence_index"],
                    "representative_record_id": proposal["representative_record_id"],
                    "representative_chunk_uid": proposal["representative_record_id"],
                    "span_token_proxy": proposal["span_token_proxy"],
                    "pre_token_proxy": pre_token_proxy,
                    "post_token_proxy": post_token_proxy,
                }
            )

    return {
        "schema_version": "span-level-template-candidate-materialization-v1",
        "status": "candidate_materialization_not_runtime_active",
        "records": materialized_records,
        "transformations": transformations,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "claim_boundary": "This candidate materialization is not called by the active runtime. Promotion requires false-positive, impact-audit, and frozen external-validation gates.",
    }


def build_candidate_impact_audit(
    rows: Iterable[JsonMap], *, minimum_span_tokens: int = 12, minimum_residual_tokens: int = 20
) -> JsonMap:
    """Measure a candidate plan without permitting it to select or delete chunks."""
    input_rows = [dict(row) for row in rows]
    plan = build_plan(
        input_rows,
        minimum_span_tokens=minimum_span_tokens,
        minimum_residual_tokens=minimum_residual_tokens,
    )
    materialized = materialize_candidate_plan(input_rows, plan)
    lexical_before = sum(_tokens(str(row.get("text") or "")) for row in input_rows)
    lexical_after = sum(_tokens(str(row.get("text") or "")) for row in materialized["records"])
    transformed_chunks = {str(item["chunk_uid"]) for item in materialized["transformations"]}
    return {
        "schema_version": "span-level-template-candidate-impact-audit-v1",
        "authority": "candidate_only_text_structural_audit",
        "selector_consumes_this_audit": False,
        "runtime_active": False,
        "stage_b_pass_chunks_before": len(input_rows),
        "stage_b_pass_chunks_after": len(materialized["records"]),
        "chunks_removed": 0,
        "chunks_transformed": len(transformed_chunks),
        "candidate_span_removals": plan["proposed_span_removals"],
        "payload_protection_blocked_chunks": len(plan["blocked_empty_or_short_residual_records"]),
        "lexical_token_proxy_before": lexical_before,
        "lexical_token_proxy_after": lexical_after,
        "lexical_token_proxy_removed": lexical_before - lexical_after,
        "claim_boundary": "Reports a read-only candidate transformation delta. It does not alter the active curated output or authorize a selector decision.",
    }
