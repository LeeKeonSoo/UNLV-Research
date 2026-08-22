from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_router import route_content
from model_provider_contract import ProviderRole, load_provider_registry
from semantic_coverage_empirical_audit import (
    EmpiricalCoverageTag,
    build_empirical_audit_bundle,
)
from semantic_embedding_runtime import (
    EmbeddingDocument,
    EmbeddingProviderSpec,
    PoolingMode,
    encode_documents,
)
from semantic_neighbor_runtime import consensus_support_strata


class CoverageCorpusRunError(RuntimeError):
    pass


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    text_fields: tuple[str, ...] = Field(min_length=1)
    max_records: int | None = Field(default=None, ge=2)


class ProviderRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    pooling: PoolingMode
    max_length: int = Field(ge=8)
    batch_size: int = Field(ge=1)
    device: str = Field(min_length=1)
    append_eos: bool
    model_path: Path | None = None


class SemanticCoverageRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    status: str
    sources: tuple[SourceSpec, ...] = Field(min_length=1)
    output_root: Path
    cache_dir: Path
    provider_registry: Path
    providers: dict[str, ProviderRunSpec]
    neighbor_count: int = Field(ge=1)
    block_size: int = Field(ge=1)
    graph_device: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "SemanticCoverageRunConfig":
        if self.schema_version != "semantic-coverage-corpus-run-v1":
            raise CoverageCorpusRunError("Unsupported semantic Coverage run schema")
        if self.status != "frozen_before_embedding":
            raise CoverageCorpusRunError("Run configuration must be frozen before embedding")
        if set(self.providers) != {"primary", "audit"}:
            raise CoverageCorpusRunError("Exactly primary and audit providers are required")
        return self


def load_run_config(path: Path) -> SemanticCoverageRunConfig:
    return SemanticCoverageRunConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(row: dict[str, object], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise CoverageCorpusRunError(f"No nonempty text field among {fields}")


def _uid(row: dict[str, object], source_index: int, line_number: int) -> str:
    for field in ("chunk_uid", "candidate_id", "record_id", "stage_a_record_id", "id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return f"source-{source_index}::line-{line_number}"


def _source_rows(source: SourceSpec, source_index: int) -> list[tuple[str, str, dict[str, object]]]:
    rows: list[tuple[str, str, dict[str, object]]] = []
    with source.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            uid = _uid(row, source_index, line_number)
            rows.append((hashlib.sha256(uid.encode()).hexdigest(), uid, row))
    if source.max_records is not None:
        rows = sorted(rows)[: source.max_records]
    return rows


def prepare_corpus(config: SemanticCoverageRunConfig) -> Path:
    config.output_root.mkdir(parents=True, exist_ok=True)
    corpus_path = config.output_root / "corpus.jsonl"
    seen: set[str] = set()
    count = 0
    with corpus_path.open("w", encoding="utf-8", newline="\n") as output:
        for source_index, source in enumerate(config.sources):
            for _, uid, row in _source_rows(source, source_index):
                text = _text(row, source.text_fields)
                if uid in seen:
                    raise CoverageCorpusRunError(f"Duplicate corpus UID: {uid}")
                seen.add(uid)
                routed = route_content(text)
                normalized = {
                    "uid": uid,
                    "text": text,
                    "token_proxy": int(row.get("token_proxy") or len(text.split())),
                    "route_labels": routed["route_labels"],
                    "script_labels": routed["language_script"]["labels"],
                    "format_labels": routed["content_format"]["labels"],
                }
                output.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                count += 1
    if count <= config.neighbor_count:
        raise CoverageCorpusRunError("Corpus must contain more records than neighbor_count")
    manifest = {
        "schema_version": "semantic-coverage-corpus-manifest-v1",
        "record_count": count,
        "corpus_file": corpus_path.name,
        "corpus_sha256": _sha256(corpus_path),
        "selector_visible_fields": ["uid", "text"],
        "audit_only_fields": ["token_proxy", "route_labels", "script_labels", "format_labels"],
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    path = config.output_root / "corpus_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _rows(config: SemanticCoverageRunConfig) -> list[dict[str, object]]:
    path = config.output_root / "corpus.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _provider(config: SemanticCoverageRunConfig, alias: str):
    run = config.providers[alias]
    registry = load_provider_registry(config.provider_registry)
    matches = [item for item in registry.providers if item.provider_id == run.provider_id]
    if len(matches) != 1 or matches[0].role is not ProviderRole.SEMANTIC:
        raise CoverageCorpusRunError(f"Unknown semantic provider: {run.provider_id}")
    provider = matches[0]
    if len(provider.artifacts) != 1:
        raise CoverageCorpusRunError("Semantic provider must bind exactly one model artifact")
    return run, provider, provider.artifacts[0]


def encode_provider(config: SemanticCoverageRunConfig, alias: str) -> Path:
    run, provider, artifact = _provider(config, alias)
    rows = _rows(config)
    manifest = json.loads((config.output_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    return encode_documents(
        EmbeddingProviderSpec(
            provider.provider_id,
            provider.identity_sha256(),
            str(run.model_path or artifact.model_id),
            artifact.revision,
            run.pooling,
            run.max_length,
            run.batch_size,
            run.device,
            config.cache_dir,
            run.append_eos,
            run.model_path is not None,
        ),
        tuple(EmbeddingDocument(str(row["uid"]), str(row["text"])) for row in rows),
        str(manifest["corpus_sha256"]),
        config.output_root / alias,
    )


def _embedding(config: SemanticCoverageRunConfig, alias: str):
    manifest_path = config.output_root / alias / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors_path = manifest_path.parent / str(manifest["vectors_file"])
    if _sha256(vectors_path) != manifest["vectors_sha256"]:
        raise CoverageCorpusRunError(f"Embedding vector hash mismatch: {alias}")
    with np.load(vectors_path, allow_pickle=False) as stored:
        uids = tuple(str(uid) for uid in stored["uids"].tolist())
        vectors = np.asarray(stored["vectors"], dtype=np.float32)
    return manifest, uids, vectors


def audit_corpus(config: SemanticCoverageRunConfig) -> Path:
    corpus = json.loads((config.output_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    rows = _rows(config)
    primary, uids, primary_vectors = _embedding(config, "primary")
    audit, audit_uids, audit_vectors = _embedding(config, "audit")
    if uids != audit_uids or uids != tuple(str(row["uid"]) for row in rows):
        raise CoverageCorpusRunError("Provider and corpus UID universes differ")
    if any(item["corpus_sha256"] != corpus["corpus_sha256"] for item in (primary, audit)):
        raise CoverageCorpusRunError("Provider and corpus identities differ")
    audit_bundle = build_empirical_audit_bundle(
        uids=uids,
        primary_vectors=primary_vectors,
        audit_vectors=audit_vectors,
        neighbor_count=config.neighbor_count,
        block_size=config.block_size,
        device=config.graph_device,
        corpus_sha256=corpus["corpus_sha256"],
        primary_identity_sha256=primary["provider_identity_sha256"],
        audit_identity_sha256=audit["provider_identity_sha256"],
        tags=tuple(
            EmpiricalCoverageTag(
                str(row["uid"]), tuple(row["route_labels"]), tuple(row["script_labels"])
            )
            for row in rows
        ),
    )
    report = audit_bundle.report
    primary_graph = audit_bundle.primary_graph
    audit_graph = audit_bundle.audit_graph
    stable, uncertain = consensus_support_strata(primary_graph, audit_graph)
    position = {uid: index for index, uid in enumerate(uids)}
    edge_pairs = sorted(
        {
            tuple(sorted((uid, neighbor)))
            for graph in (primary_graph, audit_graph)
            for uid, neighbors in graph.items()
            for neighbor in neighbors
        }
    )
    similarities = [
        {
            "left_uid": left,
            "right_uid": right,
            "similarity": max(
                0.0,
                float(
                    (
                        primary_vectors[position[left]] @ primary_vectors[position[right]]
                        + audit_vectors[position[left]] @ audit_vectors[position[right]]
                    )
                    / 2.0
                ),
            ),
        }
        for left, right in edge_pairs
    ]
    graph_path = config.output_root / "semantic_coverage_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic-coverage-graph-v1",
                "corpus_sha256": corpus["corpus_sha256"],
                "graph_sha256": report.graph_sha256,
                "primary_provider_id": primary["provider_id"],
                "primary_provider_identity_sha256": primary["provider_identity_sha256"],
                "audit_provider_id": audit["provider_id"],
                "audit_provider_identity_sha256": audit["provider_identity_sha256"],
                "stable_strata": [sorted(group) for group in stable],
                "uncertain_strata": [sorted(group) for group in uncertain],
                "similarities": similarities,
                "embedding_similarity_alone_may_delete": False,
                "benchmark_outcomes_read": False,
                "utility_read": False,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    report_path = config.output_root / "semantic_coverage_empirical_audit.json"
    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path
