from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any


JsonMap = dict[str, Any]
ReasonExtractor = Callable[[JsonMap], list[str]]


def _token_proxy(row: JsonMap) -> int:
    value = row.get("token_proxy")
    if isinstance(value, int):
        return value
    return len(str(row.get("text") or "").split())


def _record_id(row: JsonMap) -> str:
    return str(row.get("stage_a_record_id") or row.get("record_id") or "unknown")


def _stage_impact(rows: Iterable[JsonMap], extract_reasons: ReasonExtractor) -> JsonMap:
    records_by_reason: dict[str, set[str]] = defaultdict(set)
    chunk_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    for row in rows:
        record_id = _record_id(row)
        tokens = _token_proxy(row)
        for reason in extract_reasons(row):
            records_by_reason[reason].add(record_id)
            chunk_counts[reason] += 1
            token_counts[reason] += tokens
    return {
        "reasons": {
            reason: {
                "records": len(records_by_reason[reason]),
                "chunks": chunk_counts[reason],
                "token_proxy": token_counts[reason],
            }
            for reason in sorted(records_by_reason)
        }
    }


def _stage_a_reasons(row: JsonMap) -> list[str]:
    quarantine = row.get("quarantine")
    if not isinstance(quarantine, dict):
        return []
    reasons = quarantine.get("reasons")
    return [str(reason) for reason in reasons] if isinstance(reasons, list) else []


def _stage_b_reasons(row: JsonMap) -> list[str]:
    reasons = row.get("stage_b_hard_gate_reasons")
    return [str(reason) for reason in reasons] if isinstance(reasons, list) else []


def _stage_c_reasons(row: JsonMap) -> list[str]:
    selections = (row.get("stage_c_selection"), row.get("stage_c2_selection"))
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        reason = selection.get("removed_reason")
        if isinstance(reason, str) and reason:
            return [reason]
    return []


def _span_transformation_impact(transformations: Iterable[JsonMap]) -> JsonMap:
    chunks_by_reason: dict[str, set[str]] = defaultdict(set)
    removed_tokens_by_reason: Counter[str] = Counter()
    for transformation in transformations:
        reason = transformation.get("reason_code")
        if not isinstance(reason, str) or not reason:
            continue
        chunk_uid = str(transformation.get("chunk_uid") or "unknown")
        span_tokens = transformation.get("span_token_proxy")
        header_tokens = transformation.get("header_token_proxy")
        block_tokens = transformation.get("block_token_proxy")
        token_proxy = next(
            (value for value in (span_tokens, header_tokens, block_tokens) if isinstance(value, int)),
            None,
        )
        if not isinstance(token_proxy, int):
            raise ValueError("Structural transformation requires an integer removed-unit token proxy")
        chunks_by_reason[reason].add(chunk_uid)
        removed_tokens_by_reason[reason] += token_proxy
    return {
        "reasons": {
            reason: {
                "chunks": len(chunks_by_reason[reason]),
                "token_proxy_removed": removed_tokens_by_reason[reason],
            }
            for reason in sorted(chunks_by_reason)
        }
    }


def build_reason_code_impact_audit(
    stage_a_quarantined: Iterable[JsonMap],
    stage_b_rejected: Iterable[JsonMap],
    stage_c_not_selected: Iterable[JsonMap],
    stage_c_transformations: Iterable[JsonMap] | None = None,
) -> JsonMap:
    """Summarize already-made A/B/C decisions without selection authority."""
    report = {
        "schema_version": "reason-code-impact-audit-v1",
        "authority": "audit_only",
        "selector_consumes_this_audit": False,
        "token_accounting": "A multi-reason row is charged to each of its reasons; reason totals are not additive across reasons.",
        "stages": {
            "stage_a_quarantine": _stage_impact(stage_a_quarantined, _stage_a_reasons),
            "stage_b_rejection": _stage_impact(stage_b_rejected, _stage_b_reasons),
            "stage_c_compaction": _stage_impact(stage_c_not_selected, _stage_c_reasons),
        },
    }
    if stage_c_transformations is not None:
        report["stages"]["stage_c_span_transformation"] = _span_transformation_impact(stage_c_transformations)
    return report
