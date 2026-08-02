from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from content_router import route_content
from model_provider_contract import ProviderRegistry


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
LENGTH_BOUNDS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


class CorpusProfileError(RuntimeError):
    """Raised when an input corpus cannot be audited unambiguously."""


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    tokenizer_id: str
    revision: str
    add_special_tokens: bool
    append_eos_per_record: bool


class TokenCounter(Protocol):
    identity: TokenizerIdentity

    def count(self, text: str) -> int: ...


class JsonRecord(RootModel[dict[str, JsonValue]]):
    pass


class FrozenReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputArtifact(FrozenReport):
    path: str
    sha256: str = Field(min_length=64, max_length=64)
    byte_count: int = Field(ge=0)
    record_count: int = Field(ge=0)


class NoSelectionInvariants(FrozenReport):
    records_read: int = Field(ge=0)
    records_accounted: int = Field(ge=0)
    records_selected: None = None
    records_removed: int = Field(default=0, ge=0, le=0)
    output_dataset_written: bool = False
    ranking_emitted: bool = False


class LengthProfile(FrozenReport):
    total_characters: int = Field(ge=0)
    total_lines: int = Field(ge=0)
    total_paragraphs: int = Field(ge=0)
    whitespace_token_proxy: int = Field(ge=0)
    character_histogram: dict[str, int]


class DuplicateOpportunity(FrozenReport):
    normalization: str
    family_count: int = Field(ge=0)
    excess_record_count: int = Field(ge=0)
    excess_whitespace_token_proxy: int = Field(ge=0)
    target_token_delta_lower_bound: int | None
    target_token_delta_upper_bound: int | None
    action: str


class TokenizerProfile(FrozenReport):
    available: bool
    tokenizer_id: str | None
    revision: str | None
    add_special_tokens: bool | None
    append_eos_per_record: bool | None
    total_tokens: int | None


class RoutingProfile(FrozenReport):
    route_labels: dict[str, int]
    route_status: dict[str, int]
    route_confidence: dict[str, int]
    content_format: dict[str, int]
    structural_state: dict[str, int]
    language_script: dict[str, int]
    semantic_domain: dict[str, int]


class ProviderProfile(FrozenReport):
    registry_schema_version: str
    registered_provider_ids: tuple[str, ...]
    lifecycle_counts: dict[str, int]
    provider_outputs_executed: bool = False
    selection_authority: bool = False


class CorpusProfileReport(FrozenReport):
    schema_version: str = "audit-only-corpus-profile-v1"
    status: str
    authority: str = "measurement_only"
    inputs: tuple[InputArtifact, ...]
    invariants: NoSelectionInvariants
    lengths: LengthProfile
    exact_duplicate_opportunity: DuplicateOpportunity
    target_tokenizer: TokenizerProfile
    routing: RoutingProfile
    providers: ProviderProfile
    unavailable_measurements: tuple[str, ...]
    bounded_memory_strategy: str


def _extract_text(record: dict[str, JsonValue], text_fields: tuple[str, ...]) -> str:
    populated = [(field, record[field]) for field in text_fields if record.get(field) not in (None, "")]
    if len(populated) != 1:
        names = ", ".join(field for field, _ in populated) or "none"
        raise CorpusProfileError(f"Input record must populate exactly one declared text field; found: {names}")
    value = populated[0][1]
    if not isinstance(value, str):
        raise CorpusProfileError(f"Declared text field {populated[0][0]} must contain a string")
    return value


def _length_bucket(length: int) -> str:
    for lower, upper in zip(LENGTH_BOUNDS, LENGTH_BOUNDS[1:], strict=True):
        if lower <= length < upper:
            return f"{lower}-{upper - 1}"
    return f"{LENGTH_BOUNDS[-1]}+"


def _normalized_digest(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _increment_axis(counter: Counter[str], labels: list[str]) -> None:
    counter.update(labels)


def _parse_line(raw_line: bytes, path: Path, line_number: int) -> dict[str, JsonValue]:
    try:
        return JsonRecord.model_validate_json(raw_line).root
    except ValidationError as error:
        raise CorpusProfileError(f"Invalid JSON object at {path}:{line_number}: {error}") from error


def _provider_profile(registry: ProviderRegistry) -> ProviderProfile:
    lifecycle = Counter(provider.lifecycle.value for provider in registry.providers)
    return ProviderProfile(
        registry_schema_version=registry.schema_version,
        registered_provider_ids=tuple(provider.provider_id for provider in registry.providers),
        lifecycle_counts=dict(sorted(lifecycle.items())),
    )


def profile_jsonl(
    input_paths: tuple[Path, ...],
    text_fields: tuple[str, ...],
    provider_registry: ProviderRegistry,
    token_counter: TokenCounter | None,
) -> CorpusProfileReport:
    if not input_paths or not text_fields:
        raise CorpusProfileError("At least one input path and one text field are required")
    artifacts: list[InputArtifact] = []
    lengths: Counter[str] = Counter()
    axes = {name: Counter[str]() for name in ("route_labels", "route_status", "route_confidence", "content_format", "structural_state", "language_script", "semantic_domain")}
    records = characters = lines = paragraphs = whitespace_tokens = target_tokens = 0
    with TemporaryDirectory(prefix="unlv-corpus-profile-") as directory:
        database_path = Path(directory) / "exact-digests.sqlite3"
        with closing(sqlite3.connect(database_path)) as database:
            database.execute("CREATE TABLE digest_counts (digest TEXT PRIMARY KEY, records INTEGER NOT NULL, whitespace_tokens INTEGER NOT NULL, target_tokens INTEGER NOT NULL, minimum_target_tokens INTEGER NOT NULL, maximum_target_tokens INTEGER NOT NULL)")
            for path in input_paths:
                if not path.is_file():
                    raise CorpusProfileError(f"Input JSONL does not exist: {path}")
                file_hash = hashlib.sha256()
                file_records = file_bytes = 0
                with path.open("rb") as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        file_hash.update(raw_line)
                        file_bytes += len(raw_line)
                        if not raw_line.strip():
                            continue
                        text = _extract_text(_parse_line(raw_line, path, line_number), text_fields)
                        lexical_count = len(text.split())
                        current_target = token_counter.count(text) if token_counter is not None else 0
                        digest = _normalized_digest(text)
                        database.execute(
                            "INSERT INTO digest_counts VALUES (?, 1, ?, ?, ?, ?) ON CONFLICT(digest) DO UPDATE SET records=records+1, whitespace_tokens=whitespace_tokens+excluded.whitespace_tokens, target_tokens=target_tokens+excluded.target_tokens, minimum_target_tokens=MIN(minimum_target_tokens, excluded.minimum_target_tokens), maximum_target_tokens=MAX(maximum_target_tokens, excluded.maximum_target_tokens)",
                            (digest, lexical_count, current_target, current_target, current_target),
                        )
                        routing = route_content(text)
                        _increment_axis(axes["route_labels"], routing["route_labels"])
                        axes["route_status"].update((routing["route_status"],))
                        axes["route_confidence"].update((routing["route_confidence"],))
                        for axis in ("content_format", "structural_state", "language_script", "semantic_domain"):
                            _increment_axis(axes[axis], routing[axis]["labels"])
                        records += 1
                        file_records += 1
                        characters += len(text)
                        lines += max(1, len(text.splitlines()))
                        paragraphs += max(1, len(re.split(r"\n\s*\n", text.strip())))
                        whitespace_tokens += lexical_count
                        target_tokens += current_target
                        lengths[_length_bucket(len(text))] += 1
                artifacts.append(InputArtifact(path=str(path.resolve()), sha256=file_hash.hexdigest(), byte_count=file_bytes, record_count=file_records))
            family_count, excess_records, excess_proxy, target_lower, target_upper = cast(
                tuple[int, int, int, int, int],
                database.execute("SELECT COUNT(*), COALESCE(SUM(records-1),0), COALESCE(SUM(whitespace_tokens - whitespace_tokens/records),0), COALESCE(SUM(target_tokens-maximum_target_tokens),0), COALESCE(SUM(target_tokens-minimum_target_tokens),0) FROM digest_counts WHERE records > 1").fetchone(),
            )
    identity = token_counter.identity if token_counter is not None else None
    return CorpusProfileReport(
        status="audit_only_complete",
        inputs=tuple(artifacts),
        invariants=NoSelectionInvariants(records_read=records, records_accounted=records),
        lengths=LengthProfile(total_characters=characters, total_lines=lines, total_paragraphs=paragraphs, whitespace_token_proxy=whitespace_tokens, character_histogram=dict(lengths)),
        exact_duplicate_opportunity=DuplicateOpportunity(normalization="unicode_casefold_plus_whitespace_collapse_sha256", family_count=family_count, excess_record_count=excess_records, excess_whitespace_token_proxy=excess_proxy, target_token_delta_lower_bound=target_lower if token_counter is not None else None, target_token_delta_upper_bound=target_upper if token_counter is not None else None, action="report_only_no_representative_selected"),
        target_tokenizer=TokenizerProfile(available=identity is not None, tokenizer_id=identity.tokenizer_id if identity else None, revision=identity.revision if identity else None, add_special_tokens=identity.add_special_tokens if identity else None, append_eos_per_record=identity.append_eos_per_record if identity else None, total_tokens=target_tokens if identity else None),
        routing=RoutingProfile(**{name: dict(sorted(counter.items())) for name, counter in axes.items()}),
        providers=_provider_profile(provider_registry),
        unavailable_measurements=("near_duplicate_relation_opportunity", "semantic_cluster_stability", "model_based_quality_evidence"),
        bounded_memory_strategy="streaming_jsonl_plus_sqlite_digest_counts_plus_fixed_histograms",
    )


class TransformersTokenCounter:
    def __init__(self, tokenizer_path: str, revision: str, append_eos_per_record: bool) -> None:
        from transformers import AutoTokenizer

        self.identity = TokenizerIdentity(tokenizer_path, revision, False, append_eos_per_record)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=revision, local_files_only=True)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False)) + int(self.identity.append_eos_per_record)
