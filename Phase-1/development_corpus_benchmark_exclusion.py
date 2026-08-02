from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from development_corpus_admission_contract import (
    BenchmarkArtifactEvidence,
    BenchmarkArtifactFormat,
    BenchmarkArtifactSpec,
    DevelopmentCorpusAdmissionRegistry,
    DevelopmentCorpusAdmissionError,
)
from development_corpus_inventory_contract import InventoryDomain


type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type TokenSequence = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SegmentFingerprint:
    benchmark_id: str
    tokens: TokenSequence
    sha256: str
    lexical_token_count: int


@dataclass(frozen=True, slots=True)
class DomainBenchmarkIndex:
    exact: dict[str, tuple[SegmentFingerprint, ...]]
    anchors: dict[TokenSequence, tuple[SegmentFingerprint, ...]]


@dataclass(frozen=True, slots=True)
class BenchmarkIndex:
    evidence: tuple[BenchmarkArtifactEvidence, ...]
    by_domain: dict[InventoryDomain, DomainBenchmarkIndex]


@dataclass(frozen=True, slots=True)
class BenchmarkSegmentMatch:
    benchmark_id: str
    match_kind: Literal["exact_text", "segment_containment"]
    segment_sha256: str
    segment_lexical_token_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> TokenSequence:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE))


def token_sha256(tokens: TokenSequence) -> str:
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def _lexical_count(tokens: TokenSequence) -> int:
    return sum(bool(re.search(r"\w", token, flags=re.UNICODE)) for token in tokens)


def _text_segments(value: JsonValue, minimum: int) -> set[TokenSequence]:
    segments: set[TokenSequence] = set()
    match value:
        case str() as text:
            tokens = tokenize(text)
            if _lexical_count(tokens) >= minimum:
                segments.add(tokens)
        case list() as values:
            for item in values:
                segments.update(_text_segments(item, minimum))
        case dict() as mapping:
            for item in mapping.values():
                segments.update(_text_segments(item, minimum))
        case int() | float() | bool() | None:
            pass
        case unreachable:
            assert_never(unreachable)
    return segments


def _json_values(spec: BenchmarkArtifactSpec) -> tuple[JsonValue, ...]:
    path = Path(spec.path)
    if sha256_file(path) != spec.expected_file_sha256:
        raise DevelopmentCorpusAdmissionError(f"benchmark_artifact_hash_mismatch:{spec.benchmark_id}")
    match spec.artifact_format:
        case BenchmarkArtifactFormat.JSON:
            return (json.loads(path.read_text(encoding="utf-8")),)
        case BenchmarkArtifactFormat.JSONL:
            with path.open(encoding="utf-8") as handle:
                return tuple(json.loads(line) for line in handle if line.strip())
        case unreachable:
            assert_never(unreachable)


def build_benchmark_index(registry: DevelopmentCorpusAdmissionRegistry) -> BenchmarkIndex:
    segments_by_domain: dict[InventoryDomain, list[SegmentFingerprint]] = defaultdict(list)
    evidence: list[BenchmarkArtifactEvidence] = []
    for spec in registry.benchmark_artifacts:
        segments: set[TokenSequence] = set()
        for value in _json_values(spec):
            segments.update(_text_segments(value, registry.minimum_exact_segment_lexical_tokens))
        if not segments:
            raise DevelopmentCorpusAdmissionError(f"benchmark_artifact_has_no_eligible_segments:{spec.benchmark_id}")
        segments_by_domain[spec.domain].extend(
            SegmentFingerprint(spec.benchmark_id, tokens, token_sha256(tokens), _lexical_count(tokens))
            for tokens in segments
        )
        evidence.append(
            BenchmarkArtifactEvidence(
                benchmark_id=spec.benchmark_id,
                domain=spec.domain,
                file_sha256=spec.expected_file_sha256,
                eligible_segment_count=len(segments),
            )
        )
    by_domain: dict[InventoryDomain, DomainBenchmarkIndex] = {}
    width = registry.minimum_containment_segment_lexical_tokens
    for domain, segments in segments_by_domain.items():
        exact: dict[str, list[SegmentFingerprint]] = defaultdict(list)
        anchors: dict[TokenSequence, list[SegmentFingerprint]] = defaultdict(list)
        for segment in segments:
            exact[segment.sha256].append(segment)
            if segment.lexical_token_count >= width:
                anchors[segment.tokens[:width]].append(segment)
        by_domain[domain] = DomainBenchmarkIndex(
            exact={key: tuple(values) for key, values in exact.items()},
            anchors={key: tuple(values) for key, values in anchors.items()},
        )
    return BenchmarkIndex(tuple(evidence), by_domain)


def match_benchmark_segments(
    tokens: TokenSequence,
    domain_index: DomainBenchmarkIndex,
    width: int,
) -> tuple[BenchmarkSegmentMatch, ...]:
    found: dict[tuple[str, str, str], BenchmarkSegmentMatch] = {}
    text_hash = token_sha256(tokens)
    for segment in domain_index.exact.get(text_hash, ()):
        key = (segment.benchmark_id, "exact_text", segment.sha256)
        found[key] = BenchmarkSegmentMatch(*key, segment.lexical_token_count)
    for index in range(max(0, len(tokens) - width + 1)):
        for segment in domain_index.anchors.get(tokens[index:index + width], ()):
            if tokens[index:index + len(segment.tokens)] == segment.tokens:
                key = (segment.benchmark_id, "segment_containment", segment.sha256)
                found[key] = BenchmarkSegmentMatch(*key, segment.lexical_token_count)
    return tuple(found[key] for key in sorted(found))


__all__ = [
    "BenchmarkIndex", "BenchmarkSegmentMatch", "build_benchmark_index",
    "match_benchmark_segments", "sha256_file", "token_sha256", "tokenize",
]
