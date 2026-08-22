from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from quality_model_evidence import (
    MissingQualityFallbackEvidenceError,
    QualityDecision,
    QualityPolicyEvidence,
    TeacherQualityPolicyEvidence,
    quality_evidence_from_mapping,
    quality_evidence_to_mapping,
)
from quality_operating_points import (
    QUALITY_POLICY_IDS,
    decide_quality_action,
    quality_requires_teacher,
)
from quality_teacher_observation_codec import (
    OBSERVATION_SCHEMA,
    quality_runtime_sha256,
    quality_task_id,
)


JsonMap = dict[str, object]
AGGREGATION_STRATEGY = "single_teacher_confirmed_fail"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[JsonMap]:
    rows: list[JsonMap] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Quality observation line {line_number} must be an object")
            rows.append(payload)
    return rows


def load_quality_fallback_evidence(
    path: Path,
    expected_text_by_uid: Mapping[str, str],
    *,
    expected_panel_sha256: str,
) -> dict[str, tuple[TeacherQualityPolicyEvidence, ...]]:
    if not path.exists():
        return {}
    artifact_sha256 = _sha256(path)
    expected_runtime_sha256 = quality_runtime_sha256()
    results: dict[str, tuple[TeacherQualityPolicyEvidence, ...]] = {}
    for row in _read_jsonl(path):
        if row.get("schema_version") != OBSERVATION_SCHEMA:
            raise RuntimeError("Unsupported Quality fallback observation schema")
        if row.get("aggregation_strategy") != AGGREGATION_STRATEGY:
            raise RuntimeError("Quality fallback requires confirmed-fail aggregation")
        if row.get("teacher_panel_sha256") != expected_panel_sha256:
            raise RuntimeError("Quality fallback teacher panel identity mismatch")
        if row.get("quality_runtime_sha256") != expected_runtime_sha256:
            raise RuntimeError("Quality fallback runtime identity mismatch")
        chunk_uid = str(row.get("chunk_uid") or "")
        if chunk_uid not in expected_text_by_uid:
            raise RuntimeError(f"Quality fallback references an unknown chunk: {chunk_uid}")
        if chunk_uid in results:
            raise RuntimeError(f"Duplicate Quality fallback observation: {chunk_uid}")
        text_sha256 = _text_sha256(expected_text_by_uid[chunk_uid])
        if row.get("text_sha256") != text_sha256:
            raise RuntimeError(f"Quality fallback text identity mismatch: {chunk_uid}")
        if row.get("task_id") != quality_task_id(
            expected_panel_sha256,
            expected_runtime_sha256,
            chunk_uid,
            text_sha256,
        ):
            raise RuntimeError(f"Quality fallback task identity mismatch: {chunk_uid}")
        policy_payloads = row.get("policy_results")
        if not isinstance(policy_payloads, list):
            raise RuntimeError(f"Quality fallback policy results are missing: {chunk_uid}")
        policies: list[TeacherQualityPolicyEvidence] = []
        for payload in policy_payloads:
            if not isinstance(payload, dict):
                raise RuntimeError(f"Quality fallback policy result is invalid: {chunk_uid}")
            policies.append(
                TeacherQualityPolicyEvidence(
                    policy_id=str(payload.get("policy_id") or ""),
                    decision=QualityDecision(str(payload.get("panel_decision") or "")),
                    reason_codes=tuple(
                        str(code)
                        for code in payload.get("decision_reason_codes") or ("teacher_panel_decision",)
                    ),
                    observation_sha256=artifact_sha256,
                )
            )
        policy_ids = tuple(policy.policy_id for policy in policies)
        if len(policy_ids) != len(QUALITY_POLICY_IDS) or set(policy_ids) != set(QUALITY_POLICY_IDS):
            raise RuntimeError(f"Quality fallback must contain Q1-Q4 exactly once: {chunk_uid}")
        results[chunk_uid] = tuple(policies)
    return results


def write_quality_fallback_requests(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    local_results_by_chunk: Mapping[str, tuple[QualityPolicyEvidence, ...]],
    available_teacher_uids: frozenset[str] = frozenset(),
) -> JsonMap:
    requests: list[JsonMap] = []
    for row in rows:
        chunk_uid = str(row["chunk_uid"])
        if chunk_uid in available_teacher_uids:
            continue
        local_results = local_results_by_chunk[chunk_uid]
        if not quality_requires_teacher(local_results):
            continue
        text = str(row["text"])
        requests.append(
            {
                "chunk_uid": chunk_uid,
                "text": text,
                "text_sha256": _text_sha256(text),
                "quality_declared_context": row.get("quality_declared_context"),
                "quality_attached_evidence": list(row.get("quality_attached_evidence") or ()),
                "local_quality_evidence": [
                    quality_evidence_to_mapping(result) for result in local_results
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=True, sort_keys=True) + "\n")
    return {
        "schema_version": "quality-fallback-request-audit-v1",
        "request_chunks": len(requests),
        "request_path": str(path),
        "request_sha256": _sha256(path),
        "decision_rule": "q2_and_q3_and_q4_positive_support",
        "transport_failure_action": "stop_without_membership_change",
    }


def write_quality_local_evidence(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    local_results_by_chunk: Mapping[str, tuple[QualityPolicyEvidence, ...]],
) -> JsonMap:
    counts = {"retain": 0, "not_select": 0, "luna_fallback_required": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source in rows:
            chunk_uid = str(source["chunk_uid"])
            text = str(source["text"])
            results = local_results_by_chunk[chunk_uid]
            try:
                decision = decide_quality_action(results, coverage_veto=False)
            except MissingQualityFallbackEvidenceError:
                action = "luna_fallback_required"
                reason_code = "quality_local_evidence_unsupported"
                decision_source = "distilled_ranker"
            else:
                action = decision.action.value
                reason_code = decision.reason_code
                decision_source = decision.decision_source
            counts[action] += 1
            handle.write(
                json.dumps(
                    {
                        "schema_version": "quality-local-evidence-audit-v1",
                        "chunk_uid": chunk_uid,
                        "text_sha256": _text_sha256(text),
                        "local_action": action,
                        "local_reason_code": reason_code,
                        "decision_source": decision_source,
                        "quality_policy_evidence": [
                            quality_evidence_to_mapping(result) for result in results
                        ],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "schema_version": "quality-local-evidence-audit-summary-v1",
        "input_chunks": len(rows),
        "decision_counts": counts,
        "output_path": str(path),
        "output_sha256": _sha256(path),
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "token_budget_read": False,
    }


__all__ = [
    "MissingQualityFallbackEvidenceError",
    "load_quality_fallback_evidence",
    "write_quality_local_evidence",
    "write_quality_fallback_requests",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Luna fallback requests from a local Quality-scored artifact."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read_jsonl(args.input)
    local_results: dict[str, tuple[QualityPolicyEvidence, ...]] = {}
    for row in rows:
        chunk_uid = str(row["chunk_uid"])
        payloads = row.get("quality_policy_evidence")
        if not isinstance(payloads, list):
            raise RuntimeError(f"Local Quality evidence is missing: {chunk_uid}")
        local_results[chunk_uid] = tuple(
            quality_evidence_from_mapping(payload)
            for payload in payloads
            if isinstance(payload, dict)
        )
    audit = write_quality_fallback_requests(args.output, rows, local_results)
    print(json.dumps(audit, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
