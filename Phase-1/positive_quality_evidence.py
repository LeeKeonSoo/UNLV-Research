from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, Literal


Route = Literal[
    "general_prose",
    "code",
    "math",
    "technical_documentation",
    "conversation_instruction",
    "unknown",
]
Decision = Literal["eligible_keep", "reject", "abstain"]
CalibrationRole = Literal["clean_control", "candidate_pool"]
KNOWN_ROUTES: Final[tuple[Route, ...]] = (
    "general_prose",
    "code",
    "math",
    "technical_documentation",
    "conversation_instruction",
)
ALL_ROUTES: Final[tuple[Route, ...]] = (*KNOWN_ROUTES, "unknown")


@dataclass(frozen=True, slots=True)
class EvidenceContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _bounded_probability(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EvidenceContractError(f"{field} must be finite and within [0, 1]")


def _finite_score(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise EvidenceContractError(f"{field} must be finite")


@dataclass(frozen=True, slots=True)
class RouteEvidence:
    route: Route
    route_confidence: float
    substantive_payload: float
    coherence_completeness: float
    route_specific_evidence: float

    def __post_init__(self) -> None:
        if self.route not in ALL_ROUTES:
            raise EvidenceContractError(f"Unsupported Quality route: {self.route}")
        for field in (
            "route_confidence",
            "substantive_payload",
            "coherence_completeness",
            "route_specific_evidence",
        ):
            _finite_score(float(getattr(self, field)), field)


@dataclass(frozen=True, slots=True)
class ChunkEvidence:
    chunk_uid: str
    routes: tuple[RouteEvidence, ...]
    provider_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.chunk_uid:
            raise EvidenceContractError("chunk_uid must be non-empty")
        if len(self.provider_manifest_sha256) < 8:
            raise EvidenceContractError("provider_manifest_sha256 must identify a frozen provider")
        route_names = [route.route for route in self.routes]
        if len(route_names) != len(set(route_names)):
            raise EvidenceContractError("Each Quality route may appear at most once per chunk")


@dataclass(frozen=True, slots=True)
class RouteThresholds:
    route: Route
    route_confidence: float
    substantive_payload: float
    coherence_completeness: float
    route_specific_evidence: float

    def __post_init__(self) -> None:
        if self.route not in KNOWN_ROUTES:
            raise EvidenceContractError("Thresholds may be declared only for known routes")
        for field in (
            "route_confidence",
            "substantive_payload",
            "coherence_completeness",
            "route_specific_evidence",
        ):
            _finite_score(float(getattr(self, field)), field)


@dataclass(frozen=True, slots=True)
class ThresholdProfile:
    profile_id: str
    routes: tuple[RouteThresholds, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise EvidenceContractError("profile_id must be non-empty")
        names = [route.route for route in self.routes]
        if len(names) != len(set(names)):
            raise EvidenceContractError("Threshold profile routes must be unique")


@dataclass(frozen=True, slots=True)
class QualityDecision:
    decision: Decision
    reason_code: str
    qualifying_routes: tuple[Route, ...]
    evaluated_routes: tuple[Route, ...]
    provider_manifest_sha256: str
    threshold_profile_id: str


def _passes(evidence: RouteEvidence, thresholds: RouteThresholds) -> bool:
    return (
        evidence.route_confidence >= thresholds.route_confidence
        and evidence.substantive_payload >= thresholds.substantive_payload
        and evidence.coherence_completeness >= thresholds.coherence_completeness
        and evidence.route_specific_evidence >= thresholds.route_specific_evidence
    )


def evaluate_positive_quality(
    evidence: ChunkEvidence,
    profile: ThresholdProfile,
    explicit_reject_reason: str | None = None,
) -> QualityDecision:
    if explicit_reject_reason:
        return QualityDecision(
            "reject",
            explicit_reject_reason,
            (),
            tuple(route.route for route in evidence.routes),
            evidence.provider_manifest_sha256,
            profile.profile_id,
        )
    thresholds = {route.route: route for route in profile.routes}
    qualifying = tuple(
        route.route
        for route in evidence.routes
        if route.route in thresholds and _passes(route, thresholds[route.route])
    )
    return QualityDecision(
        "eligible_keep" if qualifying else "abstain",
        "positive_route_evidence_passed" if qualifying else "positive_route_evidence_insufficient",
        qualifying,
        tuple(route.route for route in evidence.routes),
        evidence.provider_manifest_sha256,
        profile.profile_id,
    )


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    evidence: ChunkEvidence
    token_count: int
    role: CalibrationRole
    expected_route: Route | None
    source_group: str

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise EvidenceContractError("Calibration token_count must be positive")
        if self.role == "clean_control" and self.expected_route not in KNOWN_ROUTES:
            raise EvidenceContractError("Every clean control requires one known expected route")
        if not self.source_group:
            raise EvidenceContractError("Calibration source_group must be non-empty")


@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    provider_training_source_groups: frozenset[str]
    calibration_source_groups: frozenset[str]
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.provider_training_source_groups & self.calibration_source_groups:
            raise EvidenceContractError("Provider training and calibration source groups must be disjoint")
        if not 0.5 < self.confidence_level < 1.0:
            raise EvidenceContractError("confidence_level must be within (0.5, 1.0)")


@dataclass(frozen=True, slots=True)
class ProfileCalibration:
    profile_id: str
    feasible: bool
    excluded_candidate_tokens: int
    route_false_reject_upper_bounds: tuple[tuple[Route, float], ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    selected_profile_id: str | None
    profiles: tuple[ProfileCalibration, ...]
    selection_rule: str = "most_compressive_feasible_threshold_profile"
    target_retention_fraction_used: bool = False


def _wilson_upper_bound(failures: int, trials: int, confidence_level: float) -> float:
    if trials == 0:
        return 1.0
    z = NormalDist().inv_cdf(confidence_level)
    proportion = failures / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
    return min(1.0, (center + radius) / denominator)


def wilson_upper_bound(failures: int, trials: int, confidence_level: float = 0.95) -> float:
    """Return a one-sided Wilson upper bound for a false-reject rate."""
    if failures < 0 or trials < 0 or failures > trials:
        raise EvidenceContractError("Wilson counts require 0 <= failures <= trials")
    if not 0.5 < confidence_level < 1.0:
        raise EvidenceContractError("confidence_level must be within (0.5, 1.0)")
    return _wilson_upper_bound(failures, trials, confidence_level)


def wilson_lower_bound(successes: int, trials: int, confidence_level: float = 0.95) -> float:
    """Return a one-sided Wilson lower bound for a success rate."""
    if successes < 0 or trials < 0 or successes > trials:
        raise EvidenceContractError("Wilson counts require 0 <= successes <= trials")
    if not 0.5 < confidence_level < 1.0:
        raise EvidenceContractError("confidence_level must be within (0.5, 1.0)")
    return 1.0 - _wilson_upper_bound(trials - successes, trials, confidence_level)


def calibrate_threshold_profiles(
    rows: tuple[CalibrationRow, ...],
    profiles: tuple[ThresholdProfile, ...],
    manifest: CalibrationManifest,
    false_reject_upper_bound: float,
) -> CalibrationResult:
    _bounded_probability(false_reject_upper_bound, "false_reject_upper_bound")
    if not profiles:
        raise EvidenceContractError("At least one threshold profile is required")
    observed_calibration_sources = {row.source_group for row in rows if row.role == "clean_control"}
    if not observed_calibration_sources <= manifest.calibration_source_groups:
        raise EvidenceContractError("Clean controls contain undeclared calibration source groups")
    reports: list[ProfileCalibration] = []
    for profile in profiles:
        decisions = tuple(evaluate_positive_quality(row.evidence, profile) for row in rows)
        bounds: list[tuple[Route, float]] = []
        for route in (threshold.route for threshold in profile.routes):
            route_controls = [
                (row, decision)
                for row, decision in zip(rows, decisions, strict=True)
                if row.role == "clean_control" and row.expected_route == route
            ]
            misses = sum(route not in decision.qualifying_routes for _, decision in route_controls)
            bounds.append((route, _wilson_upper_bound(misses, len(route_controls), manifest.confidence_level)))
        excluded_tokens = sum(
            row.token_count
            for row, decision in zip(rows, decisions, strict=True)
            if row.role == "candidate_pool" and decision.decision != "eligible_keep"
        )
        reports.append(
            ProfileCalibration(
                profile.profile_id,
                bool(bounds) and all(bound <= false_reject_upper_bound for _, bound in bounds),
                excluded_tokens,
                tuple(bounds),
            )
        )
    feasible = [report for report in reports if report.feasible]
    selected = max(feasible, key=lambda report: (report.excluded_candidate_tokens, report.profile_id)) if feasible else None
    return CalibrationResult(selected.profile_id if selected else None, tuple(reports))
