from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal


DecisionName = Literal["keep", "reject", "abstain_retain"]
RoutingStatus = Literal["routed", "mixed", "unknown", "out_of_distribution"]
RoutingConfidence = Literal["closed_evidence", "ambiguous_evidence", "none", "unsupported"]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class QualityRetentionContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class RoutingPreconditionTrace:
    status: RoutingStatus
    confidence: RoutingConfidence
    routes: tuple[str, ...]
    router_version: str
    quality_evidence: bool = False
    may_authorize_removal: bool = False

    def __post_init__(self) -> None:
        if not self.router_version:
            raise QualityRetentionContractError("Routing precondition requires a router version")
        if self.quality_evidence or self.may_authorize_removal:
            raise QualityRetentionContractError("Routing precondition cannot become Quality or removal evidence")


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    code: str
    value: str

    def __post_init__(self) -> None:
        if not self.code or not self.value:
            raise QualityRetentionContractError("Evidence observations require non-empty code and value")


@dataclass(frozen=True, slots=True)
class QualityRetentionDecision:
    decision: DecisionName
    chunk_uid: str
    policy_scope_route: str
    routing_precondition: RoutingPreconditionTrace
    evaluated_policy_ids: tuple[str, ...]
    non_trigger_boundary: str
    evidence: str
    original_text_sha256: str
    policy_artifact_sha256: str
    token_delta_proxy: int
    policy_id: str | None = None
    policy_version: str | None = None
    reason_code: str | None = None
    trigger: str | None = None
    observed_evidence: tuple[EvidenceObservation, ...] = ()
    representative_fixture_id: str | None = None
    false_positive_fixture_id: str | None = None
    schema_version: str = "quality-retention-decision-v2"
    intrinsic_quality_claim: bool = False
    weighted_score_used: bool = False
    utility_read: bool = False
    benchmark_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if self.decision not in {"keep", "reject", "abstain_retain"}:
            raise QualityRetentionContractError(f"Unsupported Quality retention decision: {self.decision}")
        if not isinstance(self.routing_precondition, RoutingPreconditionTrace):
            raise QualityRetentionContractError("Decision requires a typed routing precondition trace")
        if not self.chunk_uid or not self.policy_scope_route or not self.non_trigger_boundary:
            raise QualityRetentionContractError("Decision identity, scope, and non-trigger boundary are required")
        for field_name, digest in (
            ("original text", self.original_text_sha256),
            ("policy artifact", self.policy_artifact_sha256),
        ):
            if not SHA256_RE.fullmatch(digest):
                raise QualityRetentionContractError(f"{field_name} must use a lowercase SHA-256 digest")
        rejection_fields = (
            self.policy_id,
            self.policy_version,
            self.reason_code,
            self.trigger,
            self.representative_fixture_id,
            self.false_positive_fixture_id,
        )
        if self.decision == "reject":
            if not all(rejection_fields) or not self.observed_evidence or self.token_delta_proxy >= 0:
                raise QualityRetentionContractError(
                    "Reject decisions require policy/version/reason/trigger, observed evidence, both fixtures, and negative token delta"
                )
        elif any(rejection_fields) or self.observed_evidence or self.token_delta_proxy != 0:
            raise QualityRetentionContractError("Non-reject decisions cannot carry deletion authority")

    def to_mapping(self) -> dict[str, object]:
        result = asdict(self)
        result["evaluated_policy_ids"] = list(self.evaluated_policy_ids)
        routing = result["routing_precondition"]
        if isinstance(routing, dict):
            routing["routes"] = list(self.routing_precondition.routes)
        result["observed_evidence"] = [asdict(item) for item in self.observed_evidence]
        return result
