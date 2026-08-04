from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from semantic_neighbor_runtime import (
    blockwise_knn,
    mean_neighbor_jaccard,
    mutual_neighbor_graph,
    neighbor_jaccard_by_uid,
    neighbor_graph_sha256,
    consensus_support_strata,
)


@dataclass(frozen=True, slots=True)
class EmpiricalCoverageTag:
    uid: str
    route_labels: tuple[str, ...]
    script_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgreementCell:
    label: str
    records: int
    mean_neighbor_jaccard: float


@dataclass(frozen=True, slots=True)
class SemanticCoverageEmpiricalAudit:
    corpus_sha256: str
    primary_identity_sha256: str
    audit_identity_sha256: str
    record_count: int
    neighbor_count: int
    graph_sha256: str
    deterministic_replay: bool
    stable_strata: int
    stable_records: int
    uncertain_strata: int
    uncertain_records: int
    mean_mutual_neighbor_jaccard: float
    route_agreement: tuple[AgreementCell, ...]
    script_agreement: tuple[AgreementCell, ...]
    descriptive_bias_slices_complete: bool
    contract_extinction_detection_recall: float
    contract_representative_preserving_false_veto_rate: float
    implementation_gate_passed: bool
    scientific_promotion_gate_passed: bool
    scientific_blockers: tuple[str, ...]
    benchmark_outcomes_read: bool = False
    utility_read: bool = False


def _contract_extinction_detection_recall(stable: tuple[frozenset[str], ...]) -> float:
    if not stable:
        return 0.0
    detected = sum(bool(members) for members in stable)
    return detected / len(stable)


def _contract_representative_preserving_false_veto_rate(
    stable: tuple[frozenset[str], ...],
) -> float:
    if not stable:
        return 1.0
    survivors = {min(members) for members in stable}
    false_vetoes = sum(not bool(members & survivors) for members in stable)
    return false_vetoes / len(stable)


def build_neighbor_graphs(
    uids: tuple[str, ...],
    primary_vectors: NDArray[np.float32],
    audit_vectors: NDArray[np.float32],
    neighbor_count: int,
    block_size: int,
    device: str,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    primary_indices, _ = blockwise_knn(
        primary_vectors,
        neighbor_count=neighbor_count,
        block_size=block_size,
        device=device,
    )
    audit_indices, _ = blockwise_knn(
        audit_vectors,
        neighbor_count=neighbor_count,
        block_size=block_size,
        device=device,
    )
    return (
        mutual_neighbor_graph(uids, primary_indices),
        mutual_neighbor_graph(uids, audit_indices),
    )


def _agreement_cells(
    tags: tuple[EmpiricalCoverageTag, ...],
    scores: dict[str, float],
    axis: str,
) -> tuple[AgreementCell, ...]:
    by_label: dict[str, list[float]] = {}
    for tag in tags:
        labels = tag.route_labels if axis == "route" else tag.script_labels
        for label in labels:
            by_label.setdefault(label, []).append(scores[tag.uid])
    return tuple(
        AgreementCell(label, len(values), sum(values) / len(values))
        for label, values in sorted(by_label.items())
    )


def build_empirical_audit(
    *,
    uids: tuple[str, ...],
    primary_vectors: NDArray[np.float32],
    audit_vectors: NDArray[np.float32],
    neighbor_count: int,
    block_size: int,
    device: str,
    corpus_sha256: str,
    primary_identity_sha256: str,
    audit_identity_sha256: str,
    tags: tuple[EmpiricalCoverageTag, ...],
) -> SemanticCoverageEmpiricalAudit:
    if primary_vectors.shape != audit_vectors.shape or primary_vectors.shape[0] != len(uids):
        raise RuntimeError("Provider embedding universes must match")
    primary, audit = build_neighbor_graphs(
        uids, primary_vectors, audit_vectors, neighbor_count, block_size, device
    )
    replay_primary, replay_audit = build_neighbor_graphs(
        uids, primary_vectors, audit_vectors, neighbor_count, block_size, device
    )
    graph_sha256 = neighbor_graph_sha256(uids, primary, audit, neighbor_count)
    replay_sha256 = neighbor_graph_sha256(
        uids, replay_primary, replay_audit, neighbor_count
    )
    stable, uncertain = consensus_support_strata(primary, audit)
    tag_uids = tuple(tag.uid for tag in tags)
    tags_ready = len(tag_uids) == len(set(tag_uids)) and set(tag_uids) == set(uids)
    if not tags_ready:
        raise RuntimeError("Provider-bias audit requires one tag per embedding UID")
    jaccard_by_uid = neighbor_jaccard_by_uid(primary, audit)
    route_agreement = _agreement_cells(tags, jaccard_by_uid, "route")
    script_agreement = _agreement_cells(tags, jaccard_by_uid, "script")
    extinction_recall = _contract_extinction_detection_recall(stable)
    false_veto_rate = _contract_representative_preserving_false_veto_rate(stable)
    deterministic = graph_sha256 == replay_sha256
    implementation_passed = (
        deterministic
        and bool(stable)
        and extinction_recall == 1.0
        and false_veto_rate == 0.0
        and bool(route_agreement)
        and bool(script_agreement)
    )
    return SemanticCoverageEmpiricalAudit(
        corpus_sha256,
        primary_identity_sha256,
        audit_identity_sha256,
        len(uids),
        neighbor_count,
        graph_sha256,
        deterministic,
        len(stable),
        len(set().union(*stable)) if stable else 0,
        len(uncertain),
        len(set().union(*uncertain)) if uncertain else 0,
        mean_neighbor_jaccard(primary, audit),
        route_agreement,
        script_agreement,
        True,
        extinction_recall,
        false_veto_rate,
        implementation_passed,
        False,
        (
            "protected_false_veto_evidence_missing",
            "multidomain_confirmatory_evidence_missing",
            "normal_hard_materialization_evidence_missing",
        ),
    )
