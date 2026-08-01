from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, assert_never

from content_router import ROUTER_VERSION, RouteStatus, route_content


QualityRoute = Literal[
    "general_prose",
    "code_artifact",
    "mathematical_content",
    "technical_documentation",
    "conversation",
    "instruction",
    "table_structured_data",
]
HeadName = Literal["substantive_payload", "route_specific_evidence"]
HeadOutcome = Literal["pass", "negative", "indeterminate", "missing", "out_of_scope"]
QualityDecisionName = Literal["eligible_keep", "reject", "abstain_retain"]
RoutingPreconditionOutcome = Literal["pass", "indeterminate"]

KNOWN_ROUTES: Final[tuple[QualityRoute, ...]] = (
    "general_prose",
    "code_artifact",
    "mathematical_content",
    "technical_documentation",
    "conversation",
    "instruction",
    "table_structured_data",
)
HEAD_NAMES: Final[tuple[HeadName, ...]] = (
    "substantive_payload",
    "route_specific_evidence",
)
HEAD_OUTCOMES: Final[tuple[HeadOutcome, ...]] = (
    "pass",
    "negative",
    "indeterminate",
    "missing",
    "out_of_scope",
)
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class EvidenceContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class EvidenceHead:
    name: HeadName
    outcome: HeadOutcome
    evidence_id: str
    provider_version: str
    artifact_sha256: str
    negative_boundary_id: str | None = None

    def __post_init__(self) -> None:
        if self.name not in HEAD_NAMES:
            raise EvidenceContractError(f"Unsupported Quality evidence head: {self.name}")
        if self.outcome not in HEAD_OUTCOMES:
            raise EvidenceContractError(f"Unsupported Quality evidence outcome: {self.outcome}")
        if not self.evidence_id or not self.provider_version:
            raise EvidenceContractError("Evidence ID and provider version must be non-empty")
        if not SHA256_RE.fullmatch(self.artifact_sha256):
            raise EvidenceContractError("Evidence artifact must use a lowercase SHA-256 digest")
        if self.outcome == "negative" and not self.negative_boundary_id:
            raise EvidenceContractError("A negative outcome requires a named negative boundary")
        if self.outcome != "negative" and self.negative_boundary_id is not None:
            raise EvidenceContractError("A named negative boundary is valid only for a negative outcome")


@dataclass(frozen=True, slots=True)
class RouteEvidenceBundle:
    route: QualityRoute
    substantive_payload: EvidenceHead
    route_specific_evidence: EvidenceHead
    profile_id: str
    profile_sha256: str

    def __post_init__(self) -> None:
        if self.route not in KNOWN_ROUTES:
            raise EvidenceContractError(f"Unsupported Quality route: {self.route}")
        if self.substantive_payload.name != "substantive_payload":
            raise EvidenceContractError("Substantive payload head has the wrong declared name")
        if self.route_specific_evidence.name != "route_specific_evidence":
            raise EvidenceContractError("Route-specific head has the wrong declared name")
        if not self.profile_id or not SHA256_RE.fullmatch(self.profile_sha256):
            raise EvidenceContractError("A frozen profile ID and lowercase SHA-256 digest are required")


@dataclass(frozen=True, slots=True)
class QualityUnit:
    text: str
    evidence_bundles: tuple[RouteEvidenceBundle, ...]

    def __post_init__(self) -> None:
        routes = tuple(bundle.route for bundle in self.evidence_bundles)
        if len(routes) != len(set(routes)):
            raise EvidenceContractError("Each Quality route may have at most one evidence bundle")


@dataclass(frozen=True, slots=True)
class QualityDecision:
    decision: QualityDecisionName
    reason_code: str
    router_version: str
    route_status: RouteStatus
    routing_precondition: RoutingPreconditionOutcome
    routed_routes: tuple[QualityRoute, ...]
    evaluated_routes: tuple[QualityRoute, ...]
    qualifying_routes: tuple[QualityRoute, ...]
    negative_routes: tuple[QualityRoute, ...]
    evidence_artifact_hashes: tuple[str, ...]
    authority: str = "candidate_quality_only"
    may_mutate_curated_membership: bool = False


@dataclass(frozen=True, slots=True)
class DecisionContext:
    routing_status: RouteStatus
    routed: tuple[QualityRoute, ...]
    bundles: tuple[RouteEvidenceBundle, ...]


@dataclass(frozen=True, slots=True)
class DecisionDraft:
    name: QualityDecisionName
    reason: str
    qualifying: tuple[QualityRoute, ...] = ()
    negative: tuple[QualityRoute, ...] = ()


def _abstain_reason(status: RouteStatus) -> str:
    match status:
        case "mixed":
            return "quality_routing_mixed"
        case "unknown":
            return "quality_routing_unknown"
        case "out_of_distribution":
            return "quality_routing_out_of_distribution"
        case "routed":
            return "quality_evidence_incomplete"
        case unreachable:
            assert_never(unreachable)


def _decision(
    context: DecisionContext,
    draft: DecisionDraft,
) -> QualityDecision:
    hashes = tuple(
        sorted(
            {
                digest
                for bundle in context.bundles
                for digest in (
                    bundle.profile_sha256,
                    bundle.substantive_payload.artifact_sha256,
                    bundle.route_specific_evidence.artifact_sha256,
                )
            }
        )
    )
    return QualityDecision(
        decision=draft.name,
        reason_code=draft.reason,
        router_version=ROUTER_VERSION,
        route_status=context.routing_status,
        routing_precondition="pass" if context.routing_status == "routed" else "indeterminate",
        routed_routes=context.routed,
        evaluated_routes=tuple(bundle.route for bundle in context.bundles),
        qualifying_routes=draft.qualifying,
        negative_routes=draft.negative,
        evidence_artifact_hashes=hashes,
    )


def evaluate_route_conditioned_quality(unit: QualityUnit) -> QualityDecision:
    routing = route_content(unit.text)
    routed = tuple(route for route in KNOWN_ROUTES if route in routing["route_labels"])
    if routing["route_status"] != "routed":
        return _decision(
            DecisionContext(routing["route_status"], routed, ()),
            DecisionDraft("abstain_retain", _abstain_reason(routing["route_status"])),
        )
    by_route = {bundle.route: bundle for bundle in unit.evidence_bundles}
    evaluated = tuple(by_route[route] for route in routed if route in by_route)
    qualifying = tuple(
        bundle.route
        for bundle in evaluated
        if bundle.substantive_payload.outcome == "pass"
        and bundle.route_specific_evidence.outcome == "pass"
    )
    negative = tuple(
        bundle.route
        for bundle in evaluated
        if "negative"
        in {bundle.substantive_payload.outcome, bundle.route_specific_evidence.outcome}
    )
    if negative:
        return _decision(
            DecisionContext(routing["route_status"], routed, evaluated),
            DecisionDraft(
                "reject",
                "quality_calibrated_negative_boundary",
                negative=negative,
            ),
        )
    if qualifying:
        return _decision(
            DecisionContext(routing["route_status"], routed, evaluated),
            DecisionDraft(
                "eligible_keep",
                "quality_both_evidence_heads_passed",
                qualifying=qualifying,
            ),
        )
    return _decision(
        DecisionContext(routing["route_status"], routed, evaluated),
        DecisionDraft("abstain_retain", "quality_evidence_incomplete"),
    )
