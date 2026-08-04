from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SemanticEmbeddingArtifactError(RuntimeError):
    pass


class PoolingMode(str, Enum):
    LAST_TOKEN = "last_token"
    CLS = "cls"


@dataclass(frozen=True, slots=True)
class EmbeddingArtifact:
    provider_id: str
    provider_identity_sha256: str
    corpus_sha256: str
    pooling: PoolingMode
    max_length: int
    uids: tuple[str, ...]
    vectors: NDArray[np.float32]
    model_files_sha256: str | None = None
    model_id: str | None = None
    revision: str | None = None
    truncated_records: int = 0
    maximum_observed_tokens: int = 0
    windowed_records: int = 0
    total_windows: int = 0

    def __post_init__(self) -> None:
        if not self.provider_id or not SHA256_RE.fullmatch(self.provider_identity_sha256):
            raise SemanticEmbeddingArtifactError("Embedding artifacts require provider identity")
        if not SHA256_RE.fullmatch(self.corpus_sha256) or self.max_length < 1:
            raise SemanticEmbeddingArtifactError("Embedding artifacts require corpus identity")
        if len(self.uids) != len(set(self.uids)) or self.vectors.shape[0] != len(self.uids):
            raise SemanticEmbeddingArtifactError("Embedding UID and vector universes must match")
        if self.vectors.ndim != 2 or not np.isfinite(self.vectors).all():
            raise SemanticEmbeddingArtifactError("Embedding vectors must be a finite matrix")
        if min(self.truncated_records, self.maximum_observed_tokens, self.windowed_records, self.total_windows) < 0:
            raise SemanticEmbeddingArtifactError("Embedding token audit cannot be negative")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_model_snapshot(snapshot: Path) -> str:
    files = tuple(
        path for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".model", ".safetensors"}
    )
    if not files:
        raise SemanticEmbeddingArtifactError("Resolved model snapshot contains no frozen files")
    payload = tuple(
        (path.relative_to(snapshot).as_posix(), path.stat().st_size, _sha256_file(path))
        for path in files
    )
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_embedding_artifact(artifact: EmbeddingArtifact, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(artifact.vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise SemanticEmbeddingArtifactError("Embedding vectors must have nonzero norm")
    vectors = np.asarray(artifact.vectors / norms, dtype=np.float32)
    vectors_path = output_dir / "embeddings.npz"
    np.savez(vectors_path, uids=np.asarray(artifact.uids), vectors=vectors)
    manifest = {
        "schema_version": "semantic-embedding-artifact-v1",
        "provider_id": artifact.provider_id,
        "provider_identity_sha256": artifact.provider_identity_sha256,
        "corpus_sha256": artifact.corpus_sha256,
        "pooling": artifact.pooling.value,
        "max_length": artifact.max_length,
        "record_count": len(artifact.uids),
        "dimensions": int(vectors.shape[1]),
        "vectors_file": vectors_path.name,
        "vectors_sha256": _sha256_file(vectors_path),
        "model_files_sha256": artifact.model_files_sha256,
        "model_id": artifact.model_id,
        "revision": artifact.revision,
        "truncated_records": artifact.truncated_records,
        "maximum_observed_tokens": artifact.maximum_observed_tokens,
        "windowed_records": artifact.windowed_records,
        "total_windows": artifact.total_windows,
        "long_document_strategy": "exhaustive_nonoverlapping_window_mean",
        "normalized": True,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    path = output_dir / "embedding_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
