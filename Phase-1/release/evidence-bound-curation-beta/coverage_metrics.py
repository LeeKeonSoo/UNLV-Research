from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from coverage_contract import CoverageChunk, FrozenSimilarity


def _adjacency(edges: tuple[FrozenSimilarity, ...]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in edges:
        result[edge.left_uid][edge.right_uid] = edge.similarity
        result[edge.right_uid][edge.left_uid] = edge.similarity
    return dict(result)


def similarity(left_uid: str, right_uid: str, edges: tuple[FrozenSimilarity, ...]) -> float:
    if left_uid == right_uid:
        return 1.0
    key = tuple(sorted((left_uid, right_uid)))
    for edge in edges:
        if tuple(sorted((edge.left_uid, edge.right_uid))) == key:
            return edge.similarity
    return 0.0


def _facility_location_value(
    universe: tuple[str, ...],
    selected: frozenset[str],
    adjacency: dict[str, dict[str, float]],
) -> float:
    return sum(
        1.0
        if uid in selected
        else max(
            (score for neighbor, score in adjacency.get(uid, {}).items() if neighbor in selected),
            default=0.0,
        )
        for uid in universe
    )


@dataclass(frozen=True, slots=True)
class FacilityLocationIndex:
    universe: tuple[str, ...]
    adjacency: dict[str, dict[str, float]]

    @classmethod
    def build(
        cls,
        universe: tuple[str, ...],
        edges: tuple[FrozenSimilarity, ...],
    ) -> "FacilityLocationIndex":
        return cls(universe=universe, adjacency=_adjacency(edges))

    def choose(
        self,
        candidates: frozenset[str],
        selected: frozenset[str],
    ) -> tuple[str, float]:
        baseline = _facility_location_value(
            self.universe,
            selected,
            self.adjacency,
        )
        best_uid = ""
        best_gain = -math.inf
        for uid in sorted(candidates):
            gain = (
                _facility_location_value(
                    self.universe,
                    selected | {uid},
                    self.adjacency,
                )
                - baseline
            )
            if gain > best_gain:
                best_uid = uid
                best_gain = gain
        return best_uid, best_gain

    def nearest_radius(self, selected: frozenset[str]) -> float:
        if not self.universe:
            return 0.0
        if not selected:
            return 1.0
        minimum_similarity = min(
            1.0
            if uid in selected
            else max(
                (
                    score
                    for neighbor, score in self.adjacency.get(uid, {}).items()
                    if neighbor in selected
                ),
                default=0.0,
            )
            for uid in self.universe
        )
        return 1.0 - minimum_similarity


def choose_by_marginal_gain(
    candidates: frozenset[str],
    selected: frozenset[str],
    universe: tuple[str, ...],
    edges: tuple[FrozenSimilarity, ...],
) -> tuple[str, float]:
    return FacilityLocationIndex.build(universe, edges).choose(candidates, selected)


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
    return FacilityLocationIndex.build(universe, edges).nearest_radius(selected)


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
