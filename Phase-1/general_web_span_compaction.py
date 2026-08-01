"""Candidate-only span compaction for explicit General-web structural residue."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any


JsonMap = dict[str, Any]
CONTROL_LINES = frozenset(
    {
        "cookie preferences",
        "accept all",
        "reject all",
        "manage preferences",
        "manage cookies",
        "privacy settings",
    }
)
URL_LINE_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
DIALOGUE_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9 _-]{0,30}:\s+\S+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")


def _chunk_uid(row: JsonMap) -> str:
    return str(row.get("chunk_uid") or row.get("record_id") or row.get("id") or "unknown")


def _normalized_line(line: str) -> str:
    return " ".join(line.casefold().split())


def _looks_like_dialogue(lines: list[str]) -> bool:
    nonblank = [line for line in lines if line.strip()]
    return len(nonblank) >= 3 and sum(bool(DIALOGUE_LINE_RE.match(line.strip())) for line in nonblank) >= 3


def _span_candidates(lines: list[str]) -> list[JsonMap]:
    """Return only explicit control or URL-directory line runs."""
    proposals: list[JsonMap] = []
    index = 0
    while index < len(lines):
        control_start = index
        while index < len(lines) and _normalized_line(lines[index]) in CONTROL_LINES:
            index += 1
        if index - control_start >= 2:
            proposals.append(
                {
                    "start_line": control_start,
                    "end_line": index,
                    "reason_code": "web_control_span_removed",
                    "trigger": "contiguous_explicit_web_control_lines",
                    "non_trigger_boundary": "privacy_or_cookie_terms_inside_explanatory_prose_are_retained",
                }
            )
            continue
        index = control_start
        directory_start = index
        while index < len(lines) and bool(URL_LINE_RE.fullmatch(lines[index].strip())):
            index += 1
        if index - directory_start >= 3:
            proposals.append(
                {
                    "start_line": directory_start,
                    "end_line": index,
                    "reason_code": "url_directory_span_removed",
                    "trigger": "three_or_more_contiguous_url_only_lines",
                    "non_trigger_boundary": "reference_lists_with_titles_or_explanations_are_retained",
                }
            )
            continue
        index = directory_start + 1
    return proposals


def _remove_spans(lines: list[str], proposals: list[JsonMap]) -> list[str]:
    removed_lines = {
        line_index
        for proposal in proposals
        for line_index in range(int(proposal["start_line"]), int(proposal["end_line"]))
    }
    retained = [line for index, line in enumerate(lines) if index not in removed_lines]
    while retained and not retained[0].strip():
        retained.pop(0)
    while retained and not retained[-1].strip():
        retained.pop()
    return retained


def _token_proxy(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def build_plan(
    rows: Iterable[JsonMap],
    *,
    minimum_residual_chars: int,
    token_counter: Callable[[str], int] = _token_proxy,
) -> JsonMap:
    """Plan payload-preserving General-web span removals without runtime activation."""
    if minimum_residual_chars < 1:
        raise ValueError("General-web compaction requires a positive residual boundary")
    proposals: list[JsonMap] = []
    blocked: list[str] = []
    for row in rows:
        chunk_uid = _chunk_uid(row)
        text = str(row.get("text") or "")
        lines = text.splitlines()
        if _looks_like_dialogue(lines):
            continue
        candidates = _span_candidates(lines)
        if not candidates:
            continue
        residual = "\n".join(_remove_spans(lines, candidates)).strip()
        if len(residual) < minimum_residual_chars:
            blocked.append(chunk_uid)
            continue
        for candidate in candidates:
            span_text = "\n".join(lines[int(candidate["start_line"]) : int(candidate["end_line"])])
            proposals.append(
                {
                    **candidate,
                    "chunk_uid": chunk_uid,
                    "span_sha256": hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    "span_token_proxy": token_counter(span_text),
                    "residual_token_proxy": token_counter(residual),
                }
            )
    return {
        "schema_version": "general-web-span-compaction-plan-v1",
        "status": "candidate_only_not_a_selection_policy",
        "method": "explicit_web_control_and_url_directory_span_removal_with_dialogue_and_residual_protection",
        "minimum_residual_chars": minimum_residual_chars,
        "candidate_span_removals": len(proposals),
        "blocked_no_payload_chunks": blocked,
        "proposals": proposals,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
    }


def materialize_candidate_plan(
    rows: Iterable[JsonMap],
    plan: JsonMap,
    *,
    token_counter: Callable[[str], int] = _token_proxy,
) -> JsonMap:
    """Materialize a frozen candidate plan without enabling an active runtime policy."""
    if plan.get("status") != "candidate_only_not_a_selection_policy":
        raise ValueError("General-web candidate materialization requires a candidate-only plan")
    proposals_by_chunk: dict[str, list[JsonMap]] = defaultdict(list)
    for proposal in plan["proposals"]:
        proposals_by_chunk[str(proposal["chunk_uid"])].append(proposal)
    records: list[JsonMap] = []
    transformations: list[JsonMap] = []
    for row in rows:
        record = dict(row)
        chunk_uid = _chunk_uid(record)
        proposals = proposals_by_chunk.get(chunk_uid, [])
        if not proposals:
            records.append(record)
            continue
        lines = str(record.get("text") or "").splitlines()
        residual = "\n".join(_remove_spans(lines, proposals)).strip()
        if len(residual) < int(plan["minimum_residual_chars"]):
            raise ValueError("General-web candidate materialization would violate the residual boundary")
        record["text"] = residual
        record["token_proxy"] = token_counter(residual)
        records.append(record)
        for proposal in proposals:
            transformations.append(
                {
                    "chunk_uid": chunk_uid,
                    "reason_code": proposal["reason_code"],
                    "span_sha256": proposal["span_sha256"],
                    "span_token_proxy": proposal["span_token_proxy"],
                    "pre_token_proxy": token_counter(str(row.get("text") or "")),
                    "post_token_proxy": record["token_proxy"],
                }
            )
    return {
        "schema_version": "general-web-span-candidate-materialization-v1",
        "status": "candidate_materialization_not_runtime_active",
        "records": records,
        "transformations": transformations,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
    }
