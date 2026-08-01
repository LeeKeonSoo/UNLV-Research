from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


JsonMap = dict[str, Any]
LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _normalized_span(text: str) -> str:
    return " ".join(text.split()).casefold()


def _paragraphs(text: str) -> list[str]:
    return [paragraph for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _token_count(text: str) -> int:
    return len(LEXICAL_TOKEN_RE.findall(text))


def build_inventory(rows: Iterable[JsonMap], minimum_lexical_tokens: int = 12) -> JsonMap:
    """Inventory long exact repeated paragraphs without selecting or removing text."""
    if minimum_lexical_tokens < 1:
        raise ValueError("minimum_lexical_tokens must be positive")
    members_by_span: dict[str, list[str]] = defaultdict(list)
    original_by_span: dict[str, str] = {}
    for row in rows:
        record_id = str(row.get("record_id") or row.get("chunk_uid") or row.get("id") or "unknown")
        spans_in_record: set[str] = set()
        for paragraph in _paragraphs(str(row.get("text") or "")):
            normalized = _normalized_span(paragraph)
            if _token_count(normalized) < minimum_lexical_tokens:
                continue
            if normalized in spans_in_record:
                continue
            spans_in_record.add(normalized)
            members_by_span[normalized].append(record_id)
            original_by_span.setdefault(normalized, paragraph.strip())
    families = [
        {
            "span_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "member_count": len(record_ids),
            "record_ids": sorted(record_ids),
            "span_token_proxy": _token_count(normalized),
            "span_preview": original_by_span[normalized][:240],
        }
        for normalized, record_ids in members_by_span.items()
        if len(set(record_ids)) >= 2
    ]
    families.sort(key=lambda family: (-family["member_count"], family["span_sha256"]))
    return {
        "schema_version": "repeated-span-template-inventory-v1",
        "status": "diagnostic_only_not_a_selection_policy",
        "method": "long_exact_normalized_paragraph_match",
        "minimum_lexical_tokens": minimum_lexical_tokens,
        "repeated_span_family_count": len(families),
        "families": families,
        "selector_consumes_this_inventory": False,
        "claim_boundary": "A repeated exact span is a candidate for future span-level compaction research, not authorization to remove a record or chunk.",
    }
