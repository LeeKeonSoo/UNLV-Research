#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_coverage_corpus_runner import (
    _rows,
    audit_corpus,
    load_run_config,
    prepare_corpus,
)
from semantic_embedding_runtime import EmbeddingArtifact, PoolingMode, write_embedding_artifact


def test_prepare_and_audit_create_hash_linked_corpus_evidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jsonl"
        rows = [
            {"id": "a", "text": "import math\ndef solve(x):\n    return x + 1"},
            {"id": "b", "text": "import math\ndef answer(x):\n    return x + 1"},
            {"id": "c", "text": "The theorem follows from the matrix equation and proof."},
        ]
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "semantic-coverage-corpus-run-v1",
                    "status": "frozen_before_embedding",
                    "sources": [{"path": str(source), "text_fields": ["text"]}],
                    "output_root": str(root / "output"),
                    "cache_dir": str(root / "cache"),
                    "provider_registry": str(ROOT / "configs" / "model_provider_registry_v1.json"),
                    "providers": {
                        "primary": {"provider_id": "qwen3-embedding-0.6b-semantic-candidate", "pooling": "last_token", "max_length": 128, "batch_size": 2, "device": "cpu", "append_eos": True},
                        "audit": {"provider_id": "bge-m3-semantic-audit-candidate", "pooling": "cls", "max_length": 128, "batch_size": 2, "device": "cpu", "append_eos": False},
                    },
                    "neighbor_count": 1,
                    "block_size": 2,
                    "graph_device": "cpu",
                }
            ),
            encoding="utf-8",
        )
        config = load_run_config(config_path)
        corpus_manifest_path = prepare_corpus(config)
        corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
        providers = (
            ("primary", "qwen3-embedding-0.6b-semantic-candidate", "1" * 64),
            ("audit", "bge-m3-semantic-audit-candidate", "2" * 64),
        )
        for alias, provider_id, identity in providers:
            write_embedding_artifact(
                EmbeddingArtifact(
                    provider_id,
                    identity,
                    corpus_manifest["corpus_sha256"],
                    PoolingMode.CLS,
                    128,
                    tuple(item["uid"] for item in _read_jsonl(root / "output" / "corpus.jsonl")),
                    vectors,
                ),
                root / "output" / alias,
            )

        report_path = audit_corpus(config)
        report = json.loads(report_path.read_text(encoding="utf-8"))

        assert corpus_manifest["record_count"] == 3
        assert report["deterministic_replay"] is True
        assert report["implementation_gate_passed"] is True
        assert report["scientific_promotion_gate_passed"] is False
        assert (root / "output" / "semantic_coverage_graph.json").is_file()
        graph = json.loads((root / "output" / "semantic_coverage_graph.json").read_text(encoding="utf-8"))
        assert graph["similarities"]
        assert all(0.0 <= edge["similarity"] <= 1.0 for edge in graph["similarities"])
        assert report["benchmark_outcomes_read"] is False


def test_corpus_reader_preserves_unicode_line_separator_inside_text() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jsonl"
        text = "A theorem statement\u2028followed by its proof."
        source.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in (
                    {"id": "math-a", "text": text},
                    {"id": "math-b", "text": "A second independent proof."},
                )
            ),
            encoding="utf-8",
        )
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "semantic-coverage-corpus-run-v1",
                    "status": "frozen_before_embedding",
                    "sources": [{"path": str(source), "text_fields": ["text"]}],
                    "output_root": str(root / "output"),
                    "cache_dir": str(root / "cache"),
                    "provider_registry": str(
                        ROOT / "configs" / "model_provider_registry_v1.json"
                    ),
                    "providers": {
                        "primary": {
                            "provider_id": "qwen3-embedding-0.6b-semantic-candidate",
                            "pooling": "last_token",
                            "max_length": 128,
                            "batch_size": 1,
                            "device": "cpu",
                            "append_eos": False,
                        },
                        "audit": {
                            "provider_id": "bge-m3-semantic-audit-candidate",
                            "pooling": "cls",
                            "max_length": 128,
                            "batch_size": 1,
                            "device": "cpu",
                            "append_eos": False,
                        },
                    },
                    "neighbor_count": 1,
                    "block_size": 2,
                    "graph_device": "cpu",
                }
            ),
            encoding="utf-8",
        )
        config = load_run_config(config_path)
        prepare_corpus(config)

        rows = _rows(config)

        assert len(rows) == 2
        assert rows[0]["text"] == text


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    test_prepare_and_audit_create_hash_linked_corpus_evidence()
    test_corpus_reader_preserves_unicode_line_separator_inside_text()
    print("[semantic-coverage-corpus-runner-v1] hash-linked empirical artifacts: pass")
