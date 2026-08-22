from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from redundancy_v2 import (
    RedundancyRelation,
    RedundancySettings,
    RedundancyUnit,
    RelationType,
    classify_relation,
    tokenize,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
STRUCTURAL_REFLOW_RISK = re.compile(
    r"(?:[{};|]| {2,}\n|^\s+|^\s*(?:[-*+] |\d+[.)] ))",
    re.MULTILINE,
)


class RedundancyModeContractError(ValueError):
    pass


class RedundancyMode(StrEnum):
    NORMAL = "normal"
    HARD = "hard"
    FRAMEWORK = "framework"


class WitnessDecision(StrEnum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    ABSTAIN = "abstain"


class WitnessKind(StrEnum):
    EXACT_TEXT = "exact_text"
    FORMATTING_CANONICAL = "formatting_canonical"
    TOKEN_PRESERVING_PROSE_REFLOW = "token_preserving_prose_reflow"
    BOUNDED_NEAR_SUBSTITUTE = "bounded_near_substitute"
    DECLARED_EQUIVALENCE_VERIFIER = "declared_equivalence_verifier"
    EXACT_TOKEN_CONTAINMENT = "exact_token_containment"


class DeclaredEquivalenceWitness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_uid: str = Field(min_length=1)
    right_uid: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    decision: WitnessDecision
    artifact_sha256: Sha256
    benchmark_outcomes_read: Literal[False] = False
    utility_read: Literal[False] = False

    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.left_uid, self.right_uid)))


@dataclass(frozen=True, slots=True)
class EquivalenceWitness:
    kind: WitnessKind
    evidence_sha256: str
    preferred_representative_uid: str
    verifier_id: str | None = None
    verifier_version: str | None = None
    source_artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EquivalenceAuthority:
    relation: RedundancyRelation
    witness: EquivalenceWitness | None
    normal_authority: bool
    hard_authority: bool
    authority_reason_code: str

    def authorized(self, mode: RedundancyMode) -> bool:
        match mode:
            case RedundancyMode.NORMAL:
                return self.normal_authority
            case RedundancyMode.HARD:
                return self.hard_authority
            case RedundancyMode.FRAMEWORK:
                return self.hard_authority
            case unreachable:
                assert_never(unreachable)


def _hash_evidence(
    kind: WitnessKind,
    left: RedundancyUnit,
    right: RedundancyUnit,
    source: str = "",
) -> str:
    payload = "\0".join(
        (
            kind.value,
            left.uid,
            right.uid,
            hashlib.sha256(left.text.encode()).hexdigest(),
            hashlib.sha256(right.text.encode()).hexdigest(),
            source,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _stable_uid(left: RedundancyUnit, right: RedundancyUnit) -> str:
    return min(left.uid, right.uid)


def _prose_reflow(
    left: RedundancyUnit,
    right: RedundancyUnit,
    settings: RedundancySettings,
) -> bool:
    if min(len(tokenize(left.text)), len(tokenize(right.text))) < settings.near_min_tokens:
        return False
    if tokenize(left.text) != tokenize(right.text) or left.text == right.text:
        return False
    return not STRUCTURAL_REFLOW_RISK.search(left.text) and not STRUCTURAL_REFLOW_RISK.search(
        right.text
    )


def evaluate_equivalence_authority(
    left: RedundancyUnit,
    right: RedundancyUnit,
    settings: RedundancySettings,
    declared: DeclaredEquivalenceWitness | None = None,
    *,
    semantic_candidate: bool = False,
) -> EquivalenceAuthority:
    relation = classify_relation(left, right, settings, semantic_candidate=semantic_candidate)
    expected_pair = tuple(sorted((left.uid, right.uid)))
    if declared is not None and declared.pair() != expected_pair:
        raise RedundancyModeContractError(
            "Declared equivalence witness pair does not match the evaluated units"
        )
    stable = _stable_uid(left, right)
    if relation.relation is RelationType.EXACT_EQUIVALENT:
        kind = WitnessKind.EXACT_TEXT
        witness = EquivalenceWitness(kind, _hash_evidence(kind, left, right), stable)
        return EquivalenceAuthority(
            relation, witness, True, True, "redundancy_exact_equivalence_witness"
        )
    if relation.relation is RelationType.FORMATTING_EQUIVALENT:
        kind = WitnessKind.FORMATTING_CANONICAL
        witness = EquivalenceWitness(kind, _hash_evidence(kind, left, right), stable)
        return EquivalenceAuthority(
            relation, witness, True, True, "redundancy_formatting_equivalence_witness"
        )
    if relation.relation is RelationType.CONTAINED_PAYLOAD:
        kind = WitnessKind.EXACT_TOKEN_CONTAINMENT
        witness = EquivalenceWitness(kind, _hash_evidence(kind, left, right), right.uid)
        return EquivalenceAuthority(
            relation, witness, True, True, "redundancy_exact_containment_witness"
        )
    if relation.relation is RelationType.SUPERSET_PAYLOAD:
        kind = WitnessKind.EXACT_TOKEN_CONTAINMENT
        witness = EquivalenceWitness(kind, _hash_evidence(kind, left, right), left.uid)
        return EquivalenceAuthority(
            relation, witness, True, True, "redundancy_exact_containment_witness"
        )

    declared_blockers = set(relation.evidence.substantive_difference_codes) - {
        "api_signature_changed"
    }
    if (
        declared is not None
        and declared.decision is WitnessDecision.EQUIVALENT
        and min(relation.evidence.left_token_count, relation.evidence.right_token_count)
        >= settings.near_min_tokens
        and not declared_blockers
    ):
        kind = WitnessKind.DECLARED_EQUIVALENCE_VERIFIER
        witness = EquivalenceWitness(
            kind,
            _hash_evidence(kind, left, right, declared.artifact_sha256),
            stable,
            declared.verifier_id,
            declared.verifier_version,
            declared.artifact_sha256,
        )
        return EquivalenceAuthority(
            relation, witness, False, True, "redundancy_declared_equivalence_witness"
        )
    if relation.evidence.substantive_difference_codes:
        return EquivalenceAuthority(
            relation, None, False, False, "redundancy_substantive_difference_protected"
        )
    if relation.relation is RelationType.NEAR_SUBSTITUTE and _prose_reflow(
        left, right, settings
    ):
        kind = WitnessKind.TOKEN_PRESERVING_PROSE_REFLOW
        witness = EquivalenceWitness(kind, _hash_evidence(kind, left, right), stable)
        return EquivalenceAuthority(
            relation, witness, True, True, "redundancy_token_preserving_reflow_witness"
        )
    if relation.relation is RelationType.NEAR_SUBSTITUTE:
        kind = WitnessKind.BOUNDED_NEAR_SUBSTITUTE
        witness = EquivalenceWitness(kind, _hash_evidence(kind, left, right), stable)
        return EquivalenceAuthority(
            relation, witness, True, True, "redundancy_bounded_near_substitute_witness"
        )
    return EquivalenceAuthority(
        relation, None, False, False, "redundancy_equivalence_unproved_retain"
    )
