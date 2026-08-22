from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonMap: TypeAlias = dict[str, JsonValue]
REASON_CODE = "redundancy_intra_chunk_exact_sentence_repeat_compacted"
_LEXICAL_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r".+?[.!?]+(?=\s|$)", flags=re.DOTALL)


@dataclass(frozen=True, slots=True)
class RepeatedSentenceConfigError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


@dataclass(frozen=True, slots=True)
class RepeatedSentenceSettings:
    minimum_occurrences: int
    minimum_lexical_tokens: int
    minimum_residual_chars: int

    def __post_init__(self) -> None:
        if self.minimum_occurrences < 3:
            raise RepeatedSentenceConfigError(
                "repeated_sentence_minimum_occurrences_below_three"
            )
        if self.minimum_lexical_tokens < 1:
            raise RepeatedSentenceConfigError(
                "repeated_sentence_minimum_lexical_tokens_invalid"
            )
        if self.minimum_residual_chars < 1:
            raise RepeatedSentenceConfigError(
                "repeated_sentence_minimum_residual_chars_invalid"
            )


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    start: int
    end: int
    normalized: str
    lexical_tokens: int


@dataclass(frozen=True, slots=True)
class RepeatedSentenceResult:
    records: tuple[JsonMap, ...]
    transformations: tuple[JsonMap, ...]
    blocked_chunk_uids: tuple[str, ...]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _sentence_spans(text: str, minimum_lexical_tokens: int) -> tuple[SentenceSpan, ...]:
    spans: list[SentenceSpan] = []
    for match in _SENTENCE_RE.finditer(text):
        normalized = _normalize(match.group())
        lexical_tokens = len(_LEXICAL_TOKEN_RE.findall(normalized))
        if lexical_tokens < minimum_lexical_tokens:
            continue
        spans.append(
            SentenceSpan(
                start=match.start(),
                end=match.end(),
                normalized=normalized,
                lexical_tokens=lexical_tokens,
            )
        )
    return tuple(spans)


def _compact_text(text: str, spans: tuple[SentenceSpan, ...], removed: set[int]) -> str:
    parts: list[str] = []
    cursor = 0
    for index, span in enumerate(spans):
        parts.append(text[cursor : span.start])
        if index not in removed:
            parts.append(text[span.start : span.end])
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts).strip()


def _compact_row(
    row: JsonMap, settings: RepeatedSentenceSettings
) -> tuple[JsonMap, tuple[JsonMap, ...], bool]:
    record = dict(row)
    text_value = record.get("text")
    text = text_value if isinstance(text_value, str) else ""
    spans = _sentence_spans(text, settings.minimum_lexical_tokens)
    family_counts = Counter(span.normalized for span in spans)
    representative_by_family: dict[str, int] = {}
    removed_indices: set[int] = set()
    for index, span in enumerate(spans):
        if family_counts[span.normalized] < settings.minimum_occurrences:
            continue
        representative = representative_by_family.setdefault(span.normalized, index)
        if index != representative:
            removed_indices.add(index)
    if not removed_indices:
        return record, (), False

    compacted = _compact_text(text, spans, removed_indices)
    if len(compacted) < settings.minimum_residual_chars:
        return record, (), True

    chunk_uid = str(record.get("chunk_uid") or "unknown")
    pre_token_proxy = len(text.split())
    post_token_proxy = len(compacted.split())
    transformations: list[JsonMap] = []
    for index in sorted(removed_indices):
        span = spans[index]
        transformations.append(
            {
                "chunk_uid": chunk_uid,
                "reason_code": REASON_CODE,
                "span_sha256": hashlib.sha256(
                    span.normalized.encode("utf-8")
                ).hexdigest(),
                "representative_chunk_uid": chunk_uid,
                "representative_occurrence_index": representative_by_family[
                    span.normalized
                ],
                "removed_occurrence_index": index,
                "span_token_proxy": span.lexical_tokens,
                "pre_token_proxy": pre_token_proxy,
                "post_token_proxy": post_token_proxy,
            }
        )
    record["text"] = compacted
    record["token_proxy"] = post_token_proxy
    record["stage_b_redundancy_span_transformations"] = transformations
    return record, tuple(transformations), False


def compact_repeated_sentences(
    rows: Iterable[JsonMap], settings: RepeatedSentenceSettings
) -> RepeatedSentenceResult:
    """Compact exact repeated sentence families while retaining one occurrence."""
    records: list[JsonMap] = []
    transformations: list[JsonMap] = []
    blocked: list[str] = []
    for row in rows:
        record, row_transformations, was_blocked = _compact_row(row, settings)
        records.append(record)
        transformations.extend(row_transformations)
        if was_blocked:
            blocked.append(str(record.get("chunk_uid") or "unknown"))
    return RepeatedSentenceResult(
        records=tuple(records),
        transformations=tuple(transformations),
        blocked_chunk_uids=tuple(blocked),
    )


__all__ = [
    "REASON_CODE",
    "RepeatedSentenceConfigError",
    "RepeatedSentenceResult",
    "RepeatedSentenceSettings",
    "compact_repeated_sentences",
]
