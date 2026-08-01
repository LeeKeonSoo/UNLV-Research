from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Mapping


EvidenceGateStatus = Literal[
    "evidence_ready_candidate",
    "blocked_artifact_integrity",
    "blocked_source_disjointness",
    "blocked_source_transfer",
    "blocked_missing_stress_evidence",
    "blocked_external_feedback_leakage",
]

SUPPORTED_ROUTES: Final = frozenset(
    {"general_prose", "code_artifact", "mathematical_content"}
)
REQUIRED_BOOLEAN_FIELDS: Final = (
    "artifacts_frozen",
    "source_and_hash_disjoint",
    "strict_source_transfer",
    "adversarial_and_format_fixtures",
    "provider_bias_stress",
    "route_holdout_stress",
    "external_results_hidden",
)


@dataclass(frozen=True, slots=True)
class EvidenceGateContractError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class RouteEvidenceGate:
    route: str
    artifacts_frozen: bool
    source_and_hash_disjoint: bool
    strict_source_transfer: bool
    adversarial_and_format_fixtures: bool
    provider_bias_stress: bool
    route_holdout_stress: bool
    external_results_hidden: bool

    def __post_init__(self) -> None:
        if self.route not in SUPPORTED_ROUTES:
            raise EvidenceGateContractError(f"Unsupported evidence route: {self.route}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "RouteEvidenceGate":
        route = raw.get("route")
        if not isinstance(route, str):
            raise EvidenceGateContractError("Evidence route must be a string")
        values: dict[str, bool] = {}
        for field in REQUIRED_BOOLEAN_FIELDS:
            value = raw.get(field)
            if not isinstance(value, bool):
                raise EvidenceGateContractError(f"Evidence gate {field} must be Boolean")
            values[field] = value
        return cls(route=route, **values)


@dataclass(frozen=True, slots=True)
class RouteEvidenceGateResult:
    route: str
    status: EvidenceGateStatus
    passed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    runtime_authorized: bool = False


def evaluate_route_evidence_gate(gate: RouteEvidenceGate) -> RouteEvidenceGateResult:
    values = {field: getattr(gate, field) for field in REQUIRED_BOOLEAN_FIELDS}
    passed = tuple(field for field, value in values.items() if value)
    failed = tuple(field for field, value in values.items() if not value)
    if not gate.artifacts_frozen:
        status: EvidenceGateStatus = "blocked_artifact_integrity"
    elif not gate.source_and_hash_disjoint:
        status = "blocked_source_disjointness"
    elif not gate.external_results_hidden:
        status = "blocked_external_feedback_leakage"
    elif not gate.strict_source_transfer:
        status = "blocked_source_transfer"
    elif not all(
        (
            gate.adversarial_and_format_fixtures,
            gate.provider_bias_stress,
            gate.route_holdout_stress,
        )
    ):
        status = "blocked_missing_stress_evidence"
    else:
        status = "evidence_ready_candidate"
    return RouteEvidenceGateResult(
        route=gate.route,
        status=status,
        passed_gates=passed,
        failed_gates=failed,
    )
