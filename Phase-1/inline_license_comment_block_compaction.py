from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from inline_license_header_compaction import EXPLICIT_LICENSE_RE
from quality_rule_evidence import COMMENT_LINE_RE


JsonMap = dict[str, Any]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _chunk_uid(row: JsonMap) -> str:
    return str(row.get("chunk_uid") or row.get("record_id") or row.get("id") or "unknown")


def _tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _license_blocks(lines: list[str]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for index, line in enumerate([*lines, ""]):
        if line.strip() and COMMENT_LINE_RE.match(line):
            if start is None:
                start = index
            continue
        if start is not None:
            block = "\n".join(lines[start:index])
            if EXPLICIT_LICENSE_RE.search(block):
                blocks.append((start, index))
            start = None
    return blocks


def build_plan(
    rows: Iterable[JsonMap], *, minimum_residual_tokens: int = 20, minimum_residual_chars: int | None = None
) -> JsonMap:
    """Plan text-only removal of explicit license comment blocks with payload preservation."""
    if minimum_residual_tokens < 1 or minimum_residual_chars is not None and minimum_residual_chars < 1:
        raise ValueError("Minimum residual payload threshold must be positive")
    proposals: list[JsonMap] = []
    blocked: list[str] = []
    for row in rows:
        chunk_uid = _chunk_uid(row)
        lines = str(row.get("text") or "").splitlines()
        blocks = _license_blocks(lines)
        if not blocks:
            continue
        removed_indexes = {index for start, end in blocks for index in range(start, end)}
        residual = "\n".join(line for index, line in enumerate(lines) if index not in removed_indexes).strip()
        residual_tokens = _tokens(residual)
        has_residual_payload = (
            len(residual) >= minimum_residual_chars if minimum_residual_chars is not None else residual_tokens >= minimum_residual_tokens
        )
        if not has_residual_payload:
            blocked.append(chunk_uid)
            continue
        for start, end in blocks:
            block = "\n".join(lines[start:end])
            proposals.append({"chunk_uid": chunk_uid, "line_start": start, "line_end": end, "block_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(), "block_token_proxy": _tokens(block), "residual_token_proxy": residual_tokens})
    return {"schema_version": "inline-license-comment-block-compaction-plan-v1", "status": "candidate_only_not_a_selection_policy", "minimum_residual_tokens": minimum_residual_tokens, "minimum_residual_chars": minimum_residual_chars, "candidate_block_removals": len(proposals), "blocked_no_payload_chunks": blocked, "proposals": proposals, "runtime_authorization": "none_candidate_cannot_select_or_remove"}


def materialize_candidate_plan(rows: Iterable[JsonMap], plan: JsonMap) -> JsonMap:
    """Apply only exact planned line ranges and retain every input chunk."""
    by_chunk: dict[str, list[JsonMap]] = defaultdict(list)
    for proposal in plan["proposals"]:
        by_chunk[str(proposal["chunk_uid"])].append(proposal)
    records: list[JsonMap] = []
    transformations: list[JsonMap] = []
    for row in rows:
        record = dict(row)
        chunk_uid = _chunk_uid(record)
        proposals = by_chunk.get(chunk_uid, [])
        if not proposals:
            records.append(record)
            continue
        lines = str(record.get("text") or "").splitlines()
        indexes = {index for proposal in proposals for index in range(int(proposal["line_start"]), int(proposal["line_end"]))}
        record["text"] = "\n".join(line for index, line in enumerate(lines) if index not in indexes).strip()
        minimum_residual_chars = plan.get("minimum_residual_chars")
        if isinstance(minimum_residual_chars, int) and len(record["text"]) < minimum_residual_chars:
            raise ValueError("Candidate materialization would violate Stage-B residual character threshold")
        record["token_proxy"] = len(record["text"].split())
        records.append(record)
        transformations.extend({"chunk_uid": chunk_uid, "reason_code": "inline_license_comment_block_removed", "block_sha256": proposal["block_sha256"], "block_token_proxy": proposal["block_token_proxy"], "pre_token_proxy": len(str(row.get("text") or "").split()), "post_token_proxy": record["token_proxy"]} for proposal in proposals)
    return {"schema_version": "inline-license-comment-block-candidate-materialization-v1", "status": "candidate_materialization_not_runtime_active", "records": records, "transformations": transformations, "runtime_authorization": "none_candidate_cannot_select_or_remove"}
