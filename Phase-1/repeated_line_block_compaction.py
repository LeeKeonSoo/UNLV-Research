"""Candidate-only compaction for repeated label-style navigation blocks."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


JsonMap = dict[str, Any]

NAVIGATION_MARKERS = frozenset(
    {
        "about",
        "contact",
        "gallery",
        "home",
        "login",
        "menu",
        "privacy",
        "register",
        "search",
        "sign in",
        "terms",
    }
)


def _is_label(line: str) -> bool:
    if not 1 <= len(line) <= 60 or not line[0].isalpha():
        return False
    return all(character.isalpha() or character.isspace() or character in "&'/-" for character in line)


def _is_navigation_block(block_lines: list[str]) -> bool:
    return len(block_lines) >= 3 and all(line.casefold() in NAVIGATION_MARKERS for line in block_lines)


def _blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end = start
        while end < len(lines) and _is_label(lines[end]):
            end += 1
        block_lines = lines[start:end]
        if _is_navigation_block(block_lines):
            block = "\n".join(block_lines).casefold()
            blocks.append((start, end, block))
        start = end + 1 if end == start else end
    return blocks


def build_plan(rows: Iterable[JsonMap], *, minimum_residual_chars: int) -> JsonMap:
    """Plan removal of only later exact repeated label blocks in one chunk."""
    proposals: list[JsonMap] = []
    for row in rows:
        lines = [line.strip() for line in str(row.get("text") or "").splitlines()]
        seen: set[str] = set()
        for start, end, block in _blocks(lines):
            if block not in seen:
                seen.add(block)
                continue
            residual = "\n".join(lines[:start] + lines[end:]).strip()
            if len(residual) < minimum_residual_chars:
                continue
            span = "\n".join(lines[start:end])
            proposals.append(
                {
                    "chunk_uid": str(row.get("chunk_uid") or "unknown"),
                    "start": start,
                    "end": end,
                    "reason_code": "repeated_label_block_removed",
                    "span_sha256": hashlib.sha256(span.encode()).hexdigest(),
                    "span_token_proxy": len(span.split()),
                    "representative_block_sha256": hashlib.sha256(block.encode()).hexdigest(),
                    "representative_occurrence": "earlier_in_same_chunk",
                }
            )
    return {"status": "candidate_only_not_runtime_active", "candidate_span_removals": len(proposals), "minimum_residual_chars": minimum_residual_chars, "proposals": proposals}


def materialize_candidate_plan(rows: Iterable[JsonMap], plan: JsonMap) -> JsonMap:
    """Apply a frozen candidate plan while preserving every chunk."""
    by_id: dict[str, list[JsonMap]] = {}
    for item in plan["proposals"]:
        by_id.setdefault(str(item["chunk_uid"]), []).append(item)
    records: list[JsonMap] = []
    transformations: list[JsonMap] = []
    for raw in rows:
        row = dict(raw)
        proposals = by_id.get(str(row.get("chunk_uid") or "unknown"), [])
        lines = str(row.get("text") or "").splitlines()
        remove = {index for item in proposals for index in range(int(item["start"]), int(item["end"]))}
        if proposals:
            row["text"] = "\n".join(line for index, line in enumerate(lines) if index not in remove).strip()
            transformations.extend({**item, "pre_token_proxy": len(str(raw.get("text") or "").split()), "post_token_proxy": len(row["text"].split())} for item in proposals)
        records.append(row)
    return {"status": "candidate_materialization_not_runtime_active", "records": records, "transformations": transformations}
