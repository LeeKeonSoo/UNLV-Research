from __future__ import annotations

import math

from coverage_contract import CoverageChunk, FrozenSimilarity


def similarity(left_uid: str, right_uid: str, edges: tuple[FrozenSimilarity, ...]) -> float:
    if left_uid == right_uid:
        return 1.0
    key = tuple(sorted((left_uid, right_uid)))
    for edge in edges:
        if tuple(sorted((edge.left_uid, edge.right_uid))) == key:
            return edge.similarity
    return 0.0


def facility_location_value(
    universe: tuple[str, ...],
    selected: frozenset[str],
    edges: tuple[FrozenSimilarity, ...],
) -> float:
    if not selected:
        return 0.0
    return sum(max(similarity(uid, representative, edges) for representative in selected) for uid in universe)


def choose_by_marginal_gain(
    candidates: frozenset[str],
    selected: frozenset[str],
    universe: tuple[str, ...],
    edges: tuple[FrozenSimilarity, ...],
) -> tuple[str, float]:
    baseline = facility_location_value(universe, selected, edges)
    best_uid = ""
    best_gain = -math.inf
    for uid in sorted(candidates):
        gain = facility_location_value(universe, selected | {uid}, edges) - baseline
        if gain > best_gain:
            best_uid = uid
            best_gain = gain
    return best_uid, best_gain


def effective_sample_size(chunks: tuple[CoverageChunk, ...], selected: frozenset[str]) -> float:
    weights = tuple(chunk.token_count for chunk in chunks if chunk.uid in selected)
    if not weights:
        return 0.0
    return sum(weights) ** 2 / sum(weight**2 for weight in weights)


def nearest_representative_radius(
    universe: tuple[str, ...],
    selected: frozenset[str],
    edges: tuple[FrozenSimilarity, ...],
) -> float:
    if not universe:
        return 0.0
    if not selected:
        return 1.0
    return max(1.0 - max(similarity(uid, representative, edges) for representative in selected) for uid in universe)


def jensen_shannon_divergence(raw_mass: tuple[float, ...], selected_mass: tuple[float, ...]) -> float:
    raw_total = sum(raw_mass)
    selected_total = sum(selected_mass)
    if raw_total <= 0.0:
        return 0.0
    if selected_total <= 0.0:
        return 1.0
    raw = tuple(value / raw_total for value in raw_mass)
    selected = tuple(value / selected_total for value in selected_mass)
    midpoint = tuple((left + right) / 2.0 for left, right in zip(raw, selected, strict=True))

    def divergence(distribution: tuple[float, ...]) -> float:
        return sum(value * math.log2(value / middle) for value, middle in zip(distribution, midpoint, strict=True) if value > 0.0)

    return (divergence(raw) + divergence(selected)) / 2.0
