#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from threading import Barrier, BrokenBarrierError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_artifact_materialization import (
    ArtifactPipeline,
    RuntimeArtifactRequest,
    materialize_runtime_artifacts,
)


def test_auto_materialization_connects_existing_embedding_and_graph_pipeline() -> None:
    calls: list[str] = []

    def prepare(config) -> Path:
        calls.append("prepare")
        corpus = config.output_root / "corpus.jsonl"
        source = config.sources[0].path
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        corpus.write_text(
            "".join(
                json.dumps({"uid": row["chunk_uid"], "text": row["text"]}) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        manifest = config.output_root / "corpus_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    def encode(config, alias: str) -> Path:
        calls.append(f"encode:{alias}")
        output = config.output_root / alias
        output.mkdir(parents=True, exist_ok=True)
        vectors = output / "embeddings.npz"
        vectors.write_bytes(f"{alias}-vectors".encode("utf-8"))
        manifest = output / "embedding_manifest.json"
        manifest.write_text(
            json.dumps({"vectors_file": vectors.name}),
            encoding="utf-8",
        )
        return manifest

    def audit(config) -> Path:
        calls.append("audit")
        graph = config.output_root / "semantic_coverage_graph.json"
        graph.write_text("{}", encoding="utf-8")
        report = config.output_root / "semantic_coverage_empirical_audit.json"
        report.write_text("{}", encoding="utf-8")
        return report

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        request = RuntimeArtifactRequest(
                universe=(
                    {"chunk_uid": "a", "text": "first payload"},
                    {"chunk_uid": "b", "text": "second payload"},
                ),
                output_root=root / "artifacts",
                cache_dir=root / "cache",
                provider_registry=ROOT / "configs" / "model_provider_registry_v1.json",
                providers={
                    "primary": {
                        "provider_id": "qwen3-embedding-0.6b-semantic-candidate",
                        "pooling": "last_token",
                        "max_length": 128,
                        "batch_size": 2,
                        "device": "cuda:0",
                        "append_eos": False,
                    },
                    "audit": {
                        "provider_id": "bge-m3-semantic-audit-candidate",
                        "pooling": "cls",
                        "max_length": 128,
                        "batch_size": 2,
                        "device": "cuda:0",
                        "append_eos": False,
                    },
                },
                neighbor_count=1,
                block_size=8,
                graph_device="cpu",
            )
        bundle = materialize_runtime_artifacts(
            request,
            ArtifactPipeline(prepare=prepare, encode=encode, audit=audit),
        )
        cached_bundle = materialize_runtime_artifacts(
            request,
            ArtifactPipeline(prepare=prepare, encode=encode, audit=audit),
        )
        assert calls == ["prepare", "encode:primary", "encode:audit", "audit"]
        bundle.coverage_graph.write_text('{"tampered": true}', encoding="utf-8")
        recomputed_bundle = materialize_runtime_artifacts(
            request,
            ArtifactPipeline(prepare=prepare, encode=encode, audit=audit),
        )

        source_rows = [
            json.loads(line)
            for line in bundle.source_path.read_text(encoding="utf-8").splitlines()
        ]

    assert calls == [
        "prepare",
        "encode:primary",
        "encode:audit",
        "audit",
        "prepare",
        "encode:primary",
        "encode:audit",
        "audit",
    ]
    assert bundle.cache_hit is False
    assert cached_bundle.cache_hit is True
    assert recomputed_bundle.cache_hit is False
    assert cached_bundle.coverage_graph == bundle.coverage_graph
    assert [row["chunk_uid"] for row in source_rows] == ["a", "b"]
    assert bundle.quality_embedding_manifest.name == "embedding_manifest.json"
    assert bundle.coverage_corpus.name == "corpus.jsonl"
    assert bundle.coverage_graph.name == "semantic_coverage_graph.json"


def test_distinct_cuda_providers_encode_concurrently() -> None:
    barrier = Barrier(2)

    def prepare(config) -> Path:
        source = config.sources[0].path
        corpus = config.output_root / "corpus.jsonl"
        corpus.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = config.output_root / "corpus_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    def encode(config, alias: str) -> Path:
        try:
            barrier.wait(timeout=2.0)
        except BrokenBarrierError as error:
            raise AssertionError("Distinct CUDA providers did not overlap") from error
        output = config.output_root / alias
        output.mkdir(parents=True, exist_ok=True)
        vectors = output / "embeddings.npz"
        vectors.write_bytes(alias.encode("utf-8"))
        manifest = output / "embedding_manifest.json"
        manifest.write_text(
            json.dumps({"vectors_file": vectors.name}),
            encoding="utf-8",
        )
        return manifest

    def audit(config) -> Path:
        graph = config.output_root / "semantic_coverage_graph.json"
        graph.write_text("{}", encoding="utf-8")
        report = config.output_root / "semantic_coverage_empirical_audit.json"
        report.write_text("{}", encoding="utf-8")
        return report

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle = materialize_runtime_artifacts(
            RuntimeArtifactRequest(
                universe=(
                    {"chunk_uid": "a", "text": "first payload"},
                    {"chunk_uid": "b", "text": "second payload"},
                ),
                output_root=root / "artifacts",
                cache_dir=root / "cache",
                provider_registry=ROOT
                / "configs"
                / "model_provider_registry_v1.json",
                providers={
                    "primary": {
                        "provider_id": "qwen3-embedding-0.6b-semantic-candidate",
                        "pooling": "last_token",
                        "max_length": 128,
                        "batch_size": 2,
                        "device": "cuda:0",
                        "append_eos": False,
                    },
                    "audit": {
                        "provider_id": "bge-m3-semantic-audit-candidate",
                        "pooling": "cls",
                        "max_length": 128,
                        "batch_size": 2,
                        "device": "cuda:1",
                        "append_eos": False,
                    },
                },
                neighbor_count=1,
                block_size=8,
                graph_device="cpu",
            ),
            ArtifactPipeline(prepare=prepare, encode=encode, audit=audit),
        )

    assert bundle.cache_hit is False


if __name__ == "__main__":
    test_auto_materialization_connects_existing_embedding_and_graph_pipeline()
    test_distinct_cuda_providers_encode_concurrently()
    print("[runtime-artifact-materialization-v1] automatic artifact pipeline: pass")
