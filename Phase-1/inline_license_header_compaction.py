from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from quality_rule_evidence import COMMENT_LINE_RE


JsonMap = dict[str, Any]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
EXPLICIT_LICENSE_RE = re.compile(
    r"\b(?:spdx-license-identifier|licensed under|permission is hereby granted|all rights reserved)\b",
    re.IGNORECASE,
)


def _chunk_uid(row: JsonMap) -> str:
    return str(row.get("chunk_uid") or row.get("record_id") or row.get("id") or "unknown")


def _tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _prefix_license_header(text: str) -> tuple[str, str] | None:
    lines = text.splitlines()
    header_end = 0
    saw_comment = False
    for index, line in enumerate(lines):
        if not line.strip():
            header_end = index + 1
            continue
        if COMMENT_LINE_RE.match(line):
            saw_comment = True
            header_end = index + 1
            continue
        break
    header = "\n".join(lines[:header_end]).strip()
    residual = "\n".join(lines[header_end:]).strip()
    if not saw_comment or not EXPLICIT_LICENSE_RE.search(header):
        return None
    return header, residual


def build_plan(
    rows: Iterable[JsonMap], *, minimum_residual_tokens: int = 20, minimum_residual_chars: int | None = None
) -> JsonMap:
    """Plan a text-only prefix-license-header rewrite without activating it at runtime."""
    if minimum_residual_tokens < 1 or minimum_residual_chars is not None and minimum_residual_chars < 1:
        raise ValueError("Minimum residual payload threshold must be positive")
    proposals: list[JsonMap] = []
    blocked: list[str] = []
    for row in rows:
        chunk_uid = _chunk_uid(row)
        header_and_residual = _prefix_license_header(str(row.get("text") or ""))
        if header_and_residual is None:
            continue
        header, residual = header_and_residual
        residual_tokens = _tokens(residual)
        has_residual_payload = (
            len(residual) >= minimum_residual_chars if minimum_residual_chars is not None else residual_tokens >= minimum_residual_tokens
        )
        if not has_residual_payload:
            blocked.append(chunk_uid)
            continue
        proposals.append(
            {
                "chunk_uid": chunk_uid,
                "header_sha256": hashlib.sha256(header.encode("utf-8")).hexdigest(),
                "header_token_proxy": _tokens(header),
                "residual_token_proxy": residual_tokens,
            }
        )
    return {
        "schema_version": "inline-license-header-compaction-plan-v1",
        "status": "candidate_only_not_a_selection_policy",
        "method": "text_only_prefix_comment_license_header_with_payload_preservation",
        "minimum_residual_tokens": minimum_residual_tokens,
        "minimum_residual_chars": minimum_residual_chars,
        "candidate_header_removals": len(proposals),
        "blocked_no_payload_chunks": blocked,
        "proposals": proposals,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
    }


def materialize_candidate_plan(rows: Iterable[JsonMap], plan: JsonMap) -> JsonMap:
    """Materialize a planned prefix-header rewrite without enabling a runtime policy."""
    if plan.get("status") != "candidate_only_not_a_selection_policy":
        raise ValueError("Candidate materialization requires a candidate-only plan")
    proposal_by_chunk = {str(proposal["chunk_uid"]): proposal for proposal in plan["proposals"]}
    records: list[JsonMap] = []
    transformations: list[JsonMap] = []
    for row in rows:
        record = dict(row)
        chunk_uid = _chunk_uid(record)
        proposal = proposal_by_chunk.get(chunk_uid)
        if proposal is None:
            records.append(record)
            continue
        header_and_residual = _prefix_license_header(str(record.get("text") or ""))
        if header_and_residual is None:
            raise ValueError("Planned prefix license header is absent during materialization")
        _, residual = header_and_residual
        minimum_residual_chars = plan.get("minimum_residual_chars")
        if isinstance(minimum_residual_chars, int) and len(residual) < minimum_residual_chars:
            raise ValueError("Candidate materialization would violate Stage-B residual character threshold")
        record["text"] = residual
        record["token_proxy"] = len(residual.split())
        records.append(record)
        transformations.append(
            {
                "chunk_uid": chunk_uid,
                "reason_code": "inline_license_header_removed",
                "header_sha256": proposal["header_sha256"],
                "header_token_proxy": proposal["header_token_proxy"],
                "pre_token_proxy": len(str(row.get("text") or "").split()),
                "post_token_proxy": record["token_proxy"],
            }
        )
    return {
        "schema_version": "inline-license-header-candidate-materialization-v1",
        "status": "candidate_materialization_not_runtime_active",
        "records": records,
        "transformations": transformations,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
    }
