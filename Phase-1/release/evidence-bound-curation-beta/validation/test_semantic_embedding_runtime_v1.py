#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_embedding_runtime import (
    EmbeddingArtifact,
    PoolingMode,
    token_windows,
    pool_hidden_states,
    write_embedding_artifact,
)


class _FixtureTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool):
        assert truncation is False
        ids = list(range(len(text.split())))
        return {"input_ids": [101, *ids] if add_special_tokens else ids}

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 1



def test_qwen_last_token_and_bge_cls_pooling_are_explicit() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]],
        ]
    )
    right_padded = torch.tensor([[1, 1, 0], [1, 1, 1]])

    qwen = pool_hidden_states(hidden, right_padded, PoolingMode.LAST_TOKEN)
    bge = pool_hidden_states(hidden, right_padded, PoolingMode.CLS)

    assert torch.equal(qwen, torch.tensor([[2.0, 0.0], [0.0, 6.0]]))
    assert torch.equal(bge, torch.tensor([[1.0, 0.0], [0.0, 4.0]]))


def test_embedding_artifact_roundtrip_preserves_frozen_identity() -> None:
    artifact = EmbeddingArtifact(
        provider_id="fixture-provider",
        provider_identity_sha256="1" * 64,
        corpus_sha256="2" * 64,
        pooling=PoolingMode.CLS,
        max_length=128,
        uids=("a", "b"),
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = write_embedding_artifact(artifact, Path(directory))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(Path(directory) / manifest["vectors_file"]) as vectors:
            saved_uids = vectors["uids"].tolist()
            saved_norms = np.linalg.norm(vectors["vectors"], axis=1)

    assert manifest["provider_identity_sha256"] == "1" * 64
    assert manifest["corpus_sha256"] == "2" * 64
    assert manifest["pooling"] == "cls"
    assert saved_uids == ["a", "b"]
    assert np.allclose(saved_norms, 1.0)


def test_long_documents_are_exhaustively_windowed_with_qwen_eos_per_window() -> None:
    windows, observed = token_windows(
        _FixtureTokenizer(), "one two three four five six seven eight nine", 5, True
    )

    assert observed == 9
    assert len(windows) == 3
    assert all(len(window["input_ids"]) <= 5 for window in windows)
    assert all(window["input_ids"][-1] == 99 for window in windows)


if __name__ == "__main__":
    test_qwen_last_token_and_bge_cls_pooling_are_explicit()
    test_embedding_artifact_roundtrip_preserves_frozen_identity()
    test_long_documents_are_exhaustively_windowed_with_qwen_eos_per_window()
    print("[semantic-embedding-runtime-v1] frozen pooling and artifact: pass")
