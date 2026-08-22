#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_neighbor_runtime import blockwise_knn, consensus_support_strata, mutual_neighbor_graph


def test_blockwise_knn_is_exact_and_excludes_self() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float32,
    )

    indices, scores = blockwise_knn(vectors, neighbor_count=1, block_size=2, device="cpu")

    assert indices.tolist() == [[1], [0], [3], [2]]
    assert np.all(scores > 0.99)


def test_mutual_graph_keeps_only_reciprocal_neighbors() -> None:
    indices = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    graph = mutual_neighbor_graph(("a", "b", "c"), indices[:, :1])

    assert graph == {"a": frozenset({"b"}), "b": frozenset({"a"}), "c": frozenset()}


def test_consensus_support_uses_shared_local_edges_without_singleton_vetoes() -> None:
    primary = {"a": frozenset({"b"}), "b": frozenset({"a"}), "c": frozenset({"b"})}
    audit = {"a": frozenset({"b"}), "b": frozenset({"a", "c"}), "c": frozenset({"b"})}

    stable, uncertain = consensus_support_strata(primary, audit)

    assert stable == (frozenset({"a", "b"}), frozenset({"b", "c"}))
    assert uncertain == ()


if __name__ == "__main__":
    test_blockwise_knn_is_exact_and_excludes_self()
    test_mutual_graph_keeps_only_reciprocal_neighbors()
    test_consensus_support_uses_shared_local_edges_without_singleton_vetoes()
    print("[semantic-neighbor-runtime-v1] blockwise exact mutual kNN: pass")
