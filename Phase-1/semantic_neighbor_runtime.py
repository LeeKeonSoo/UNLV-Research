from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import NDArray


class SemanticNeighborRuntimeError(RuntimeError):
    pass


def _normalized(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise SemanticNeighborRuntimeError("Neighbor search requires a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise SemanticNeighborRuntimeError("Neighbor vectors require nonzero norm")
    return np.asarray(matrix / norms, dtype=np.float32)


def blockwise_knn(
    vectors: NDArray[np.float32],
    *,
    neighbor_count: int,
    block_size: int,
    device: str,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    matrix = _normalized(vectors)
    count = matrix.shape[0]
    if not 1 <= neighbor_count < count or block_size < 1:
        raise SemanticNeighborRuntimeError("Invalid neighbor or block size")
    reference = torch.from_numpy(matrix).to(device)
    all_indices: list[NDArray[np.int64]] = []
    all_scores: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            scores = reference[start:stop] @ reference.T
            rows = torch.arange(stop - start, device=device)
            columns = torch.arange(start, stop, device=device)
            scores[rows, columns] = -torch.inf
            values, indices = torch.topk(
                scores, neighbor_count, dim=1, largest=True, sorted=True
            )
            all_indices.append(indices.cpu().numpy().astype(np.int64, copy=False))
            all_scores.append(values.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(all_indices), np.concatenate(all_scores)


def mutual_neighbor_graph(
    uids: tuple[str, ...], indices: NDArray[np.int64]
) -> dict[str, frozenset[str]]:
    if len(uids) != len(set(uids)) or indices.shape[0] != len(uids):
        raise SemanticNeighborRuntimeError("Neighbor indices must match unique UIDs")
    directed = {
        uid: frozenset(uids[int(index)] for index in indices[row])
        for row, uid in enumerate(uids)
    }
    return {
        uid: frozenset(other for other in directed[uid] if uid in directed[other])
        for uid in uids
    }


def connected_components(
    graph: Mapping[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    remaining = set(graph)
    result: dict[str, frozenset[str]] = {}
    while remaining:
        frontier = [min(remaining)]
        members: set[str] = set()
        while frontier:
            uid = frontier.pop()
            if uid in members:
                continue
            members.add(uid)
            frontier.extend(graph[uid] - members)
        component = frozenset(members)
        remaining -= component
        for uid in component:
            result[uid] = component
    return result


def neighbor_graph_sha256(
    uids: tuple[str, ...],
    primary: Mapping[str, frozenset[str]],
    audit: Mapping[str, frozenset[str]],
    neighbor_count: int,
) -> str:
    payload = {
        "uids": uids,
        "neighbor_count": neighbor_count,
        "primary": {uid: sorted(primary[uid]) for uid in uids},
        "audit": {uid: sorted(audit[uid]) for uid in uids},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def consensus_support_strata(
    primary: Mapping[str, frozenset[str]],
    audit: Mapping[str, frozenset[str]],
) -> tuple[tuple[frozenset[str], ...], tuple[frozenset[str], ...]]:
    stable: set[frozenset[str]] = set()
    uncertain: set[frozenset[str]] = set()
    for uid in sorted(primary):
        shared = primary[uid] & audit[uid]
        disagreement = primary[uid] | audit[uid]
        if shared:
            stable.add(frozenset({uid, *shared}))
        elif disagreement:
            uncertain.add(frozenset({uid, *disagreement}))
    key = lambda members: tuple(sorted(members))
    return tuple(sorted(stable, key=key)), tuple(sorted(uncertain, key=key))


def mean_neighbor_jaccard(
    primary: Mapping[str, frozenset[str]], audit: Mapping[str, frozenset[str]]
) -> float:
    scores = neighbor_jaccard_by_uid(primary, audit)
    return sum(scores.values()) / len(scores)


def neighbor_jaccard_by_uid(
    primary: Mapping[str, frozenset[str]], audit: Mapping[str, frozenset[str]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for uid in sorted(primary):
        union = primary[uid] | audit[uid]
        result[uid] = len(primary[uid] & audit[uid]) / len(union) if union else 1.0
    return result
