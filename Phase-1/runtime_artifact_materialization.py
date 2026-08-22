from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from semantic_coverage_corpus_runner import (
    ProviderRunSpec,
    SemanticCoverageRunConfig,
    SourceSpec,
    audit_corpus,
    encode_provider,
    prepare_corpus,
)


class RuntimeArtifactError(RuntimeError):
    """Raised when automatic runtime artifacts cannot be materialized."""


class RuntimeArtifactUnit(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    chunk_uid: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RuntimeArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    universe: tuple[RuntimeArtifactUnit, ...] = Field(min_length=2)
    output_root: Path
    cache_dir: Path
    provider_registry: Path
    providers: dict[str, ProviderRunSpec]
    neighbor_count: int = Field(default=8, ge=1)
    block_size: int = Field(default=512, ge=1)
    graph_device: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runtime_graph(self) -> "RuntimeArtifactRequest":
        if set(self.providers) != {"primary", "audit"}:
            raise RuntimeArtifactError("Automatic artifacts require primary and audit providers")
        if self.neighbor_count >= len(self.universe):
            raise RuntimeArtifactError("neighbor_count must be smaller than the corpus")
        return self


PrepareCorpus = Callable[[SemanticCoverageRunConfig], Path]
EncodeProvider = Callable[[SemanticCoverageRunConfig, str], Path]
AuditCorpus = Callable[[SemanticCoverageRunConfig], Path]


@dataclass(frozen=True, slots=True)
class ArtifactPipeline:
    prepare: PrepareCorpus
    encode: EncodeProvider
    audit: AuditCorpus


@dataclass(frozen=True, slots=True)
class RuntimeArtifactBundle:
    source_path: Path
    quality_embedding_manifest: Path
    coverage_corpus: Path
    coverage_graph: Path
    coverage_audit: Path
    primary_provider_id: str
    cache_hit: bool = False


class CachedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeArtifactCache(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runtime-artifact-cache-v1"]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[CachedArtifact, ...] = Field(min_length=1)


DEFAULT_PIPELINE = ArtifactPipeline(
    prepare=prepare_corpus,
    encode=encode_provider,
    audit=audit_corpus,
)

CACHE_MANIFEST_NAME: Final = "runtime_artifact_cache.json"
CACHE_TEMP_NAME: Final = "runtime_artifact_cache.tmp"
REQUIRED_ARTIFACTS: Final = frozenset(
    {
        "stage_b_universe.jsonl",
        "corpus.jsonl",
        "corpus_manifest.json",
        "primary/embedding_manifest.json",
        "primary/embeddings.npz",
        "audit/embedding_manifest.json",
        "audit/embeddings.npz",
        "semantic_coverage_graph.json",
        "semantic_coverage_empirical_audit.json",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_sha256(request: RuntimeArtifactRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bundle(request: RuntimeArtifactRequest, *, cache_hit: bool) -> RuntimeArtifactBundle:
    return RuntimeArtifactBundle(
        source_path=request.output_root / "stage_b_universe.jsonl",
        quality_embedding_manifest=request.output_root
        / "primary"
        / "embedding_manifest.json",
        coverage_corpus=request.output_root / "corpus.jsonl",
        coverage_graph=request.output_root / "semantic_coverage_graph.json",
        coverage_audit=request.output_root
        / "semantic_coverage_empirical_audit.json",
        primary_provider_id=request.providers["primary"].provider_id,
        cache_hit=cache_hit,
    )


def _cached_bundle(
    request: RuntimeArtifactRequest,
    request_sha256: str,
) -> RuntimeArtifactBundle | None:
    cache_path = request.output_root / CACHE_MANIFEST_NAME
    try:
        cache = RuntimeArtifactCache.model_validate_json(
            cache_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if cache.request_sha256 != request_sha256:
        return None
    relative_paths = {artifact.relative_path for artifact in cache.artifacts}
    if not REQUIRED_ARTIFACTS <= relative_paths:
        return None
    root = request.output_root.resolve()
    for artifact in cache.artifacts:
        path = (root / artifact.relative_path).resolve()
        if path != root and root not in path.parents:
            return None
        try:
            actual_sha256 = _sha256(path)
        except OSError:
            return None
        if actual_sha256 != artifact.sha256:
            return None
    return _bundle(request, cache_hit=True)


def _write_cache(request: RuntimeArtifactRequest, request_sha256: str) -> None:
    artifacts = tuple(
        CachedArtifact(
            relative_path=path.relative_to(request.output_root).as_posix(),
            sha256=_sha256(path),
        )
        for path in sorted(request.output_root.rglob("*"))
        if path.is_file()
        and path.name not in {CACHE_MANIFEST_NAME, CACHE_TEMP_NAME}
    )
    relative_paths = {artifact.relative_path for artifact in artifacts}
    if not REQUIRED_ARTIFACTS <= relative_paths:
        missing = ", ".join(sorted(REQUIRED_ARTIFACTS - relative_paths))
        raise RuntimeArtifactError(f"Automatic artifact pipeline omitted: {missing}")
    cache = RuntimeArtifactCache(
        schema_version="runtime-artifact-cache-v1",
        request_sha256=request_sha256,
        artifacts=artifacts,
    )
    cache_path = request.output_root / CACHE_MANIFEST_NAME
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(
        cache.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)


def _encode_providers(
    config: SemanticCoverageRunConfig,
    pipeline: ArtifactPipeline,
) -> None:
    aliases = ("primary", "audit")
    devices = tuple(
        config.providers[alias].device.strip().casefold()
        for alias in aliases
    )
    distinct_cuda_devices = (
        len(set(devices)) == len(devices)
        and all(device.startswith("cuda:") for device in devices)
    )
    if not distinct_cuda_devices:
        for alias in aliases:
            pipeline.encode(config, alias)
        return
    with ThreadPoolExecutor(
        max_workers=len(aliases),
        thread_name_prefix="runtime-embedding",
    ) as executor:
        futures = tuple(
            executor.submit(pipeline.encode, config, alias)
            for alias in aliases
        )
        for future in futures:
            future.result()


def materialize_runtime_artifacts(
    request: RuntimeArtifactRequest,
    pipeline: ArtifactPipeline = DEFAULT_PIPELINE,
) -> RuntimeArtifactBundle:
    request.output_root.mkdir(parents=True, exist_ok=True)
    request_sha256 = _request_sha256(request)
    cached = _cached_bundle(request, request_sha256)
    if cached is not None:
        return cached
    source_path = request.output_root / "stage_b_universe.jsonl"
    with source_path.open("w", encoding="utf-8", newline="\n") as handle:
        for unit in request.universe:
            handle.write(
                json.dumps(unit.model_dump(), ensure_ascii=False, sort_keys=True) + "\n"
            )

    config = SemanticCoverageRunConfig(
        schema_version="semantic-coverage-corpus-run-v1",
        status="frozen_before_embedding",
        sources=(SourceSpec(path=source_path, text_fields=("text",)),),
        output_root=request.output_root,
        cache_dir=request.cache_dir,
        provider_registry=request.provider_registry,
        providers=request.providers,
        neighbor_count=request.neighbor_count,
        block_size=request.block_size,
        graph_device=request.graph_device,
    )
    pipeline.prepare(config)
    _encode_providers(config, pipeline)
    pipeline.audit(config)
    _write_cache(request, request_sha256)
    return _bundle(request, cache_hit=False)


__all__ = [
    "ArtifactPipeline",
    "RuntimeArtifactBundle",
    "RuntimeArtifactError",
    "RuntimeArtifactRequest",
    "materialize_runtime_artifacts",
]
