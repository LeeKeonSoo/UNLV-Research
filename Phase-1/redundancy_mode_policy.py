from __future__ import annotations

import hashlib
from dataclasses import dataclass

from redundancy_equivalence import (
    DeclaredEquivalenceWitness,
    EquivalenceAuthority,
    RedundancyMode,
    RedundancyModeContractError,
    WitnessDecision,
    WitnessKind,
    evaluate_equivalence_authority,
)
from redundancy_v2 import RedundancySettings, RedundancyUnit, RelationType, tokenize
from redundancy_v2_retrieval import retrieve_candidate_pairs


@dataclass(frozen=True, slots=True)
class RedundancyRemovalProposal:
    removed_uid: str
    representative_uid: str
    mode: RedundancyMode
    reason_code: str
    witness_kind: WitnessKind
    evidence_sha256: str
    family_id: str
    removed_token_count: int
    coverage_veto_required: bool = True
    benchmark_outcomes_read: bool = False
    utility_read: bool = False


@dataclass(frozen=True, slots=True)
class RedundancyFamilyProposal:
    family_id: str
    representative_uid: str
    member_uids: tuple[str, ...]
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class RedundancyPlan:
    mode: RedundancyMode
    input_uids: tuple[str, ...]
    proposed_survivor_uids: tuple[str, ...]
    removals: tuple[RedundancyRemovalProposal, ...]
    families: tuple[RedundancyFamilyProposal, ...]
    authority_decisions: tuple[EquivalenceAuthority, ...]
    representative_selection: str = "payload_superset_then_stable_uid"
    coverage_veto_required: bool = True


def _resolve(uid: str, targets: dict[str, str]) -> str:
    visited: set[str] = set()
    while uid in targets:
        if uid in visited:
            raise RedundancyModeContractError(
                "Redundancy representative linkage contains a cycle"
            )
        visited.add(uid)
        uid = targets[uid]
    return uid


def _candidate_pairs(
    units: tuple[RedundancyUnit, ...],
    settings: RedundancySettings,
    declared: dict[tuple[str, str], DeclaredEquivalenceWitness],
    exhaustive: bool,
) -> set[tuple[str, str]]:
    if exhaustive:
        pairs = {
            tuple(sorted((left.uid, right.uid)))
            for index, left in enumerate(units)
            for right in units[index + 1 :]
        }
    else:
        pairs = {
            (pair.left_uid, pair.right_uid)
            for pair in retrieve_candidate_pairs(units, settings)
        }
    pairs.update(declared)
    return pairs


def _representative_targets(
    by_uid: dict[str, RedundancyUnit],
    decisions: tuple[EquivalenceAuthority, ...],
    mode: RedundancyMode,
) -> tuple[dict[str, str], dict[str, EquivalenceAuthority]]:
    symmetric = {
        WitnessKind.EXACT_TEXT,
        WitnessKind.FORMATTING_CANONICAL,
        WitnessKind.TOKEN_PRESERVING_PROSE_REFLOW,
        WitnessKind.DECLARED_EQUIVALENCE_VERIFIER,
    }
    parents = {uid: uid for uid in by_uid}

    def find(uid: str) -> str:
        while parents[uid] != uid:
            uid = parents[uid]
        return uid

    for decision in decisions:
        if (
            decision.authorized(mode)
            and decision.witness is not None
            and decision.witness.kind in symmetric
        ):
            left_root = find(decision.relation.left_uid)
            right_root = find(decision.relation.right_uid)
            parents[max(left_root, right_root)] = min(left_root, right_root)
    targets = {uid: find(uid) for uid in by_uid if find(uid) != uid}

    # Bounded near-substitute edges are intentionally pairwise. They do not
    # receive a silent transitive closure because A~B and B~C does not prove A~C.
    occupied_near_roots: set[str] = set()
    near_decisions = sorted(
        (
            decision
            for decision in decisions
            if decision.authorized(mode)
            and decision.witness is not None
            and decision.witness.kind is WitnessKind.BOUNDED_NEAR_SUBSTITUTE
        ),
        key=lambda decision: (
            decision.relation.evidence.changed_ratio,
            decision.relation.left_uid,
            decision.relation.right_uid,
        ),
    )
    for decision in near_decisions:
        left_root = find(decision.relation.left_uid)
        right_root = find(decision.relation.right_uid)
        if left_root == right_root or left_root in occupied_near_roots or right_root in occupied_near_roots:
            continue
        representative = min(left_root, right_root)
        removed = max(left_root, right_root)
        targets[removed] = representative
        occupied_near_roots.update((left_root, right_root))

    containment: dict[str, list[tuple[str, EquivalenceAuthority]]] = {}
    for decision in decisions:
        if (
            not decision.authorized(mode)
            or decision.witness is None
            or decision.witness.kind is not WitnessKind.EXACT_TOKEN_CONTAINMENT
        ):
            continue
        child = (
            decision.relation.left_uid
            if decision.relation.relation is RelationType.CONTAINED_PAYLOAD
            else decision.relation.right_uid
        )
        child_root = find(child)
        representative_root = find(decision.witness.preferred_representative_uid)
        if child_root != representative_root:
            containment.setdefault(child_root, []).append((representative_root, decision))

    chosen_containment: dict[str, EquivalenceAuthority] = {}
    for child, options in containment.items():
        representative, decision = min(
            options,
            key=lambda item: (-len(tokenize(by_uid[item[0]].text)), item[0]),
        )
        targets[child] = representative
        chosen_containment[child] = decision
    return targets, chosen_containment


def _build_removals(
    by_uid: dict[str, RedundancyUnit],
    decisions: tuple[EquivalenceAuthority, ...],
    mode: RedundancyMode,
    targets: dict[str, str],
    containment: dict[str, EquivalenceAuthority],
) -> tuple[tuple[RedundancyRemovalProposal, ...], tuple[RedundancyFamilyProposal, ...]]:
    members_by_representative: dict[str, set[str]] = {}
    for removed_uid in targets:
        representative = _resolve(removed_uid, targets)
        members_by_representative.setdefault(representative, {representative}).add(removed_uid)

    family_by_representative: dict[str, RedundancyFamilyProposal] = {}
    for representative, members in sorted(members_by_representative.items()):
        evidence_hashes = tuple(
            sorted(
                decision.witness.evidence_sha256
                for decision in decisions
                if decision.authorized(mode)
                and decision.witness is not None
                and decision.relation.left_uid in members
                and decision.relation.right_uid in members
            )
        )
        evidence = hashlib.sha256("\0".join(evidence_hashes).encode()).hexdigest()
        member_uids = tuple(sorted(members))
        family_id = hashlib.sha256(
            "\0".join((*member_uids, representative, evidence)).encode()
        ).hexdigest()
        family_by_representative[representative] = RedundancyFamilyProposal(
            family_id=family_id,
            representative_uid=representative,
            member_uids=member_uids,
            evidence_sha256=evidence,
        )

    removals: list[RedundancyRemovalProposal] = []
    for uid in sorted(targets):
        representative = _resolve(uid, targets)
        containment_decision = containment.get(uid)
        related = tuple(
            decision
            for decision in decisions
            if decision.authorized(mode)
            and decision.witness is not None
            and ({decision.relation.left_uid, decision.relation.right_uid} & {uid, representative})
        )
        kind = (
            WitnessKind.EXACT_TOKEN_CONTAINMENT
            if containment_decision is not None
            else min(
                (
                    decision.witness.kind
                    for decision in related
                    if decision.witness is not None
                ),
                key=lambda item: item.value,
            )
        )
        family = family_by_representative[representative]
        removals.append(
            RedundancyRemovalProposal(
                removed_uid=uid,
                representative_uid=representative,
                mode=mode,
                reason_code=(
                    "redundancy_contained_payload_nonrepresentative"
                    if kind is WitnessKind.EXACT_TOKEN_CONTAINMENT
                    else "redundancy_equivalent_family_nonrepresentative"
                ),
                witness_kind=kind,
                evidence_sha256=family.evidence_sha256,
                family_id=family.family_id,
                removed_token_count=len(tokenize(by_uid[uid].text)),
            )
        )
    return tuple(removals), tuple(family_by_representative.values())


def build_redundancy_plan(
    units: tuple[RedundancyUnit, ...],
    settings: RedundancySettings,
    mode: RedundancyMode,
    declared_witnesses: tuple[DeclaredEquivalenceWitness, ...] = (),
    *,
    exhaustive: bool = False,
) -> RedundancyPlan:
    by_uid = {unit.uid: unit for unit in units}
    if len(by_uid) != len(units):
        raise RedundancyModeContractError(
            "Redundancy plan unit identifiers must be unique"
        )
    declared = {witness.pair(): witness for witness in declared_witnesses}
    if len(declared) != len(declared_witnesses):
        raise RedundancyModeContractError(
            "Declared equivalence witness pairs must be unique"
        )
    pairs = _candidate_pairs(units, settings, declared, exhaustive)
    if any(left not in by_uid or right not in by_uid for left, right in pairs):
        raise RedundancyModeContractError(
            "Declared equivalence witness references an unknown unit"
        )
    decisions = tuple(
        evaluate_equivalence_authority(
            by_uid[left], by_uid[right], settings, declared.get((left, right))
        )
        for left, right in sorted(pairs)
    )
    targets, containment = _representative_targets(by_uid, decisions, mode)
    removals, families = _build_removals(
        by_uid, decisions, mode, targets, containment
    )
    removed = {proposal.removed_uid for proposal in removals}
    return RedundancyPlan(
        mode=mode,
        input_uids=tuple(sorted(by_uid)),
        proposed_survivor_uids=tuple(sorted(set(by_uid) - removed)),
        removals=removals,
        families=families,
        authority_decisions=decisions,
    )


__all__ = [
    "DeclaredEquivalenceWitness",
    "RedundancyFamilyProposal",
    "RedundancyMode",
    "WitnessDecision",
    "WitnessKind",
    "build_redundancy_plan",
    "evaluate_equivalence_authority",
]
