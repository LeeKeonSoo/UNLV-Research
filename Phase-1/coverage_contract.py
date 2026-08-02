from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import assert_never


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CoverageContractError(RuntimeError):
    """Raised when Coverage evidence violates the frozen contract."""


class CoverageView(str, Enum):
    REDUNDANCY_FAMILY = "redundancy_family"
    CONTENT_ROUTE = "content_route"
    LANGUAGE_SCRIPT = "language_script"
    FORMAT_MORPHOLOGY = "format_morphology"
    SEMANTIC_SKILL = "semantic_skill"
    UNCERTAIN_INTERSECTION = "uncertain_intersection"


class StratumState(str, Enum):
    STABLE = "stable"
    UNCERTAIN = "uncertain"


class ExclusionKind(str, Enum):
    VALIDITY_INVALID = "validity_invalid"
    QUALITY_SUPPORTED_NONPOSITIVE = "quality_supported_nonpositive"


class CoverageStatus(str, Enum):
    PASS = "pass"
    VETO_CANDIDATE = "veto_candidate"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class CoverageChunk:
    uid: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.uid or self.token_count < 1:
            raise CoverageContractError("Coverage chunks require an ID and positive token count")


@dataclass(frozen=True, slots=True)
class CoverageStratum:
    stratum_id: str
    view: CoverageView
    member_uids: frozenset[str]
    state: StratumState
    evidence_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.stratum_id or not self.member_uids:
            raise CoverageContractError("Coverage strata require an ID and at least one member")
        if not SHA256_RE.fullmatch(self.evidence_artifact_sha256):
            raise CoverageContractError("Coverage strata require a frozen evidence artifact")
        if self.view is CoverageView.UNCERTAIN_INTERSECTION and self.state is not StratumState.UNCERTAIN:
            raise CoverageContractError("Uncertain intersections require an uncertain state")
        if self.view is not CoverageView.UNCERTAIN_INTERSECTION and self.state is not StratumState.STABLE:
            raise CoverageContractError("Only uncertain intersections may use an uncertain state")


@dataclass(frozen=True, slots=True)
class RepresentativeFamily:
    family_id: str
    member_uids: frozenset[str]
    evidence_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.family_id or len(self.member_uids) < 2:
            raise CoverageContractError("Representative families require an ID and at least two members")
        if not SHA256_RE.fullmatch(self.evidence_artifact_sha256):
            raise CoverageContractError("Representative families require a frozen evidence artifact")


@dataclass(frozen=True, slots=True)
class FrozenSimilarity:
    left_uid: str
    right_uid: str
    similarity: float
    evidence_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.left_uid or not self.right_uid or self.left_uid == self.right_uid:
            raise CoverageContractError("Similarity edges require two distinct chunk IDs")
        if not math.isfinite(self.similarity) or not 0.0 <= self.similarity <= 1.0:
            raise CoverageContractError("Similarity must be finite and within [0, 1]")
        if not SHA256_RE.fullmatch(self.evidence_artifact_sha256):
            raise CoverageContractError("Similarity edges require a frozen evidence artifact")


@dataclass(frozen=True, slots=True)
class ExclusionEvidence:
    chunk_uid: str
    kind: ExclusionKind
    policy_id: str
    reason_code: str
    evidence_artifact_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.chunk_uid or not self.policy_id or not self.reason_code:
            raise CoverageContractError("Exclusions require chunk, policy, and reason identifiers")
        if not self.evidence_artifact_hashes or any(
            not SHA256_RE.fullmatch(artifact) for artifact in self.evidence_artifact_hashes
        ):
            raise CoverageContractError("Exclusions require frozen evidence artifacts")
        if len(self.evidence_artifact_hashes) != len(set(self.evidence_artifact_hashes)):
            raise CoverageContractError("Exclusion evidence artifacts must be unique")
        match self.kind:
            case ExclusionKind.VALIDITY_INVALID:
                if not self.reason_code.startswith("validity_"):
                    raise CoverageContractError("Validity exclusions require a typed Validity reason")
            case ExclusionKind.QUALITY_SUPPORTED_NONPOSITIVE:
                expected = (
                    self.policy_id == "stage_c_calibrated_quality_effect_candidate"
                    and self.reason_code == "quality_nonpositive_effect_supported"
                    and len(self.evidence_artifact_hashes) == 5
                )
                if not expected:
                    raise CoverageContractError("Quality exclusions require the complete calibrated effect trace")
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class CoverageRequest:
    chunks: tuple[CoverageChunk, ...]
    proposed_survivors: frozenset[str]
    strata: tuple[CoverageStratum, ...]
    redundancy_families: tuple[RepresentativeFamily, ...]
    similarities: tuple[FrozenSimilarity, ...]
    exclusions: tuple[ExclusionEvidence, ...]
    provider_id: str
    provider_identity_sha256: str

    def __post_init__(self) -> None:
        chunk_ids = tuple(chunk.uid for chunk in self.chunks)
        universe = set(chunk_ids)
        if not chunk_ids or len(chunk_ids) != len(universe):
            raise CoverageContractError("Coverage universe requires unique chunks")
        if not self.proposed_survivors <= universe:
            raise CoverageContractError("Proposed survivors must belong to the Coverage universe")
        if not self.provider_id or not SHA256_RE.fullmatch(self.provider_identity_sha256):
            raise CoverageContractError("Coverage requests require a frozen semantic-provider identity")
        stratum_ids = tuple(stratum.stratum_id for stratum in self.strata)
        if len(stratum_ids) != len(set(stratum_ids)):
            raise CoverageContractError("Coverage stratum IDs must be unique")
        if any(not stratum.member_uids <= universe for stratum in self.strata):
            raise CoverageContractError("Coverage strata cannot reference chunks outside the universe")
        family_ids = tuple(family.family_id for family in self.redundancy_families)
        if len(family_ids) != len(set(family_ids)):
            raise CoverageContractError("Representative family IDs must be unique")
        if any(not family.member_uids <= universe for family in self.redundancy_families):
            raise CoverageContractError("Representative families cannot reference chunks outside the universe")
        family_members = tuple(uid for family in self.redundancy_families for uid in family.member_uids)
        if len(family_members) != len(set(family_members)):
            raise CoverageContractError("Representative families must be disjoint")
        excluded_ids = tuple(exclusion.chunk_uid for exclusion in self.exclusions)
        if len(excluded_ids) != len(set(excluded_ids)) or not set(excluded_ids) <= universe:
            raise CoverageContractError("Every exclusion must identify one unique universe chunk")
        if self.proposed_survivors & set(excluded_ids):
            raise CoverageContractError("Proposed survivors cannot carry independent exclusion evidence")
        edge_keys = tuple(tuple(sorted((edge.left_uid, edge.right_uid))) for edge in self.similarities)
        if len(edge_keys) != len(set(edge_keys)):
            raise CoverageContractError("Similarity edges must be unique and symmetric by contract")
        if any(edge.left_uid not in universe or edge.right_uid not in universe for edge in self.similarities):
            raise CoverageContractError("Similarity edges cannot reference chunks outside the universe")


@dataclass(frozen=True, slots=True)
class RepresentativeChoice:
    family_id: str
    representative_uid: str
    marginal_gain: float
    selection_method: str = "facility_location_marginal_gain_then_uid"


@dataclass(frozen=True, slots=True)
class CoverageViewReport:
    view: CoverageView
    target_strata: int
    proposed_covered_strata: int
    protected_covered_strata: int
    proposed_support_recall: float
    protected_support_recall: float
    token_mass_jensen_shannon_divergence: float


@dataclass(frozen=True, slots=True)
class TokenMassReport:
    raw_tokens: int
    proposed_tokens: int
    protected_tokens: int
    proposed_retention_ratio: float
    protected_retention_ratio: float


@dataclass(frozen=True, slots=True)
class CoverageDecision:
    status: CoverageStatus
    reason_code: str
    protected_uids: tuple[str, ...]
    family_representatives: tuple[RepresentativeChoice, ...]
    extinct_before_protection: tuple[str, ...]
    permitted_extinctions: tuple[str, ...]
    view_reports: tuple[CoverageViewReport, ...]
    token_report: TokenMassReport
    nearest_representative_radius: float
    effective_sample_size: float
    evidence_artifact_hashes: tuple[str, ...]
    may_mutate_curated_membership: bool = False
    fixed_quota_used: bool = False
    source_identity_used: bool = False
    benchmark_outcomes_read: bool = False
    utility_read: bool = False
