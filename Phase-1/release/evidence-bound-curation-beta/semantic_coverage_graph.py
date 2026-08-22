from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Final

from coverage_contract import CoverageStratum, CoverageView, FrozenSimilarity, StratumState


SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class SemanticCoverageContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SemanticEmbedding:
    uid: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.uid or not self.values or any(not math.isfinite(value) for value in self.values):
            raise SemanticCoverageContractError("Embeddings require an ID and finite values")
        if sum(value * value for value in self.values) <= 0.0:
            raise SemanticCoverageContractError("Embedding vectors must have nonzero norm")


@dataclass(frozen=True, slots=True)
class CoverageTag:
    uid: str
    route_labels: tuple[str, ...]
    script_labels: tuple[str, ...]
    format_labels: tuple[str, ...]
    stable: bool


@dataclass(frozen=True, slots=True)
class SemanticCoverageGraphRequest:
    primary_provider_id: str
    primary_provider_identity_sha256: str
    audit_provider_id: str
    audit_provider_identity_sha256: str
    primary_embeddings: tuple[SemanticEmbedding, ...]
    audit_embeddings: tuple[SemanticEmbedding, ...]
    neighbor_count: int

    def __post_init__(self) -> None:
        if not self.primary_provider_id or not self.audit_provider_id:
            raise SemanticCoverageContractError("Semantic providers require IDs")
        if self.primary_provider_id == self.audit_provider_id:
            raise SemanticCoverageContractError("Primary and audit providers must be independent")
        if not SHA256_RE.fullmatch(self.primary_provider_identity_sha256) or not SHA256_RE.fullmatch(
            self.audit_provider_identity_sha256
        ):
            raise SemanticCoverageContractError("Semantic providers require frozen identities")
        primary_ids = tuple(item.uid for item in self.primary_embeddings)
        audit_ids = tuple(item.uid for item in self.audit_embeddings)
        if len(primary_ids) != len(set(primary_ids)) or set(primary_ids) != set(audit_ids):
            raise SemanticCoverageContractError("Primary and audit embedding universes must match")
        if len(primary_ids) < 2 or not 1 <= self.neighbor_count < len(primary_ids):
            raise SemanticCoverageContractError("Neighbor count must be within the embedding universe")
        dimensions = {len(item.values) for item in (*self.primary_embeddings, *self.audit_embeddings)}
        if len(dimensions) != 1:
            raise SemanticCoverageContractError("All embeddings must share one dimension")


@dataclass(frozen=True, slots=True)
class SemanticCoverageGraphResult:
    strata: tuple[CoverageStratum, ...]
    similarities: tuple[FrozenSimilarity, ...]
    primary_provider_identity_sha256: str
    audit_provider_identity_sha256: str
    graph_evidence_sha256: str
    benchmark_outcomes_read: bool = False
    utility_read: bool = False


def _cosine(left: SemanticEmbedding, right: SemanticEmbedding) -> float:
    numerator = sum(a * b for a, b in zip(left.values, right.values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left.values))
    right_norm = math.sqrt(sum(value * value for value in right.values))
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _mutual_graph(
    embeddings: tuple[SemanticEmbedding, ...], neighbor_count: int
) -> dict[str, frozenset[str]]:
    by_uid = {item.uid: item for item in embeddings}
    nearest = {
        uid: tuple(
            candidate
            for candidate, _ in sorted(
                (
                    (candidate, _cosine(item, by_uid[candidate]))
                    for candidate in by_uid
                    if candidate != uid
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )[:neighbor_count]
        )
        for uid, item in by_uid.items()
    }
    return {
        uid: frozenset(candidate for candidate in nearest[uid] if uid in nearest[candidate])
        for uid in by_uid
    }


def _components(graph: dict[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    remaining = set(graph)
    while remaining:
        root = min(remaining)
        frontier = [root]
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


def _evidence_hash(request: SemanticCoverageGraphRequest) -> str:
    payload = {
        "primary": request.primary_provider_identity_sha256,
        "audit": request.audit_provider_identity_sha256,
        "k": request.neighbor_count,
        "primary_embeddings": [(item.uid, item.values) for item in request.primary_embeddings],
        "audit_embeddings": [(item.uid, item.values) for item in request.audit_embeddings],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_semantic_coverage_graph(
    request: SemanticCoverageGraphRequest,
) -> SemanticCoverageGraphResult:
    primary = _mutual_graph(request.primary_embeddings, request.neighbor_count)
    audit = _mutual_graph(request.audit_embeddings, request.neighbor_count)
    primary_components = _components(primary)
    audit_components = _components(audit)
    evidence = _evidence_hash(request)
    stable: set[frozenset[str]] = set()
    uncertain: set[frozenset[str]] = set()
    for uid in sorted(primary):
        primary_component = primary_components[uid]
        audit_component = audit_components[uid]
        if primary_component == audit_component and len(primary_component) >= 2:
            stable.add(primary_component)
        else:
            uncertain.add(primary_component | audit_component)
    strata = tuple(
        CoverageStratum(
            f"semantic-stable-{index}",
            CoverageView.SEMANTIC_SKILL,
            members,
            StratumState.STABLE,
            evidence,
        )
        for index, members in enumerate(sorted(stable, key=lambda item: tuple(sorted(item))))
    ) + tuple(
        CoverageStratum(
            f"semantic-uncertain-{index}",
            CoverageView.UNCERTAIN_INTERSECTION,
            members,
            StratumState.UNCERTAIN,
            evidence,
        )
        for index, members in enumerate(sorted(uncertain, key=lambda item: tuple(sorted(item))))
    )
    primary_by_uid = {item.uid: item for item in request.primary_embeddings}
    audit_by_uid = {item.uid: item for item in request.audit_embeddings}
    edge_pairs = {
        tuple(sorted((uid, neighbor)))
        for graph in (primary, audit)
        for uid, neighbors in graph.items()
        for neighbor in neighbors
    }
    similarities = tuple(
        FrozenSimilarity(
            left,
            right,
            max(
                0.0,
                (
                    _cosine(primary_by_uid[left], primary_by_uid[right])
                    + _cosine(audit_by_uid[left], audit_by_uid[right])
                )
                / 2.0,
            ),
            evidence,
        )
        for left, right in sorted(edge_pairs)
    )
    return SemanticCoverageGraphResult(
        strata,
        similarities,
        request.primary_provider_identity_sha256,
        request.audit_provider_identity_sha256,
        evidence,
    )


def build_multiview_strata(
    tags: tuple[CoverageTag, ...], evidence_artifact_sha256: str
) -> tuple[CoverageStratum, ...]:
    if not SHA256_RE.fullmatch(evidence_artifact_sha256):
        raise SemanticCoverageContractError("Coverage tags require frozen evidence")
    if len({tag.uid for tag in tags}) != len(tags):
        raise SemanticCoverageContractError("Coverage tag identifiers must be unique")
    groups: dict[tuple[CoverageView, str], set[str]] = {}
    uncertain: set[str] = set()
    for tag in tags:
        if not tag.stable or "unknown" in {*tag.route_labels, *tag.script_labels, *tag.format_labels}:
            uncertain.add(tag.uid)
            continue
        for view, labels in (
            (CoverageView.CONTENT_ROUTE, tag.route_labels),
            (CoverageView.LANGUAGE_SCRIPT, tag.script_labels),
            (CoverageView.FORMAT_MORPHOLOGY, tag.format_labels),
        ):
            for label in labels:
                groups.setdefault((view, label), set()).add(tag.uid)
    stable_strata = tuple(
        CoverageStratum(
            f"{view.value}:{label}",
            view,
            frozenset(members),
            StratumState.STABLE,
            evidence_artifact_sha256,
        )
        for (view, label), members in sorted(groups.items(), key=lambda item: (item[0][0].value, item[0][1]))
    )
    if not uncertain:
        return stable_strata
    return stable_strata + (
        CoverageStratum(
            "uncertain:routing",
            CoverageView.UNCERTAIN_INTERSECTION,
            frozenset(uncertain),
            StratumState.UNCERTAIN,
            evidence_artifact_sha256,
        ),
    )
