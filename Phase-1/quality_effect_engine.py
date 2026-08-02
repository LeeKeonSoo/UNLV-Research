from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal, assert_never

from model_provider_contract import ProviderLifecycle, ProviderManifest, ProviderRole
from quality_effect_calibration import (
    EffectCalibrationBundle,
    EffectCalibrationReport,
    EffectDirection,
    EffectInterval,
    EvidenceBin,
    QualityEffectContractError,
    calibrate_effect_bins,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RouteState = Literal["routed", "mixed", "unknown", "out_of_distribution"]


class QualityEffectDecisionName(str, Enum):
    ELIGIBLE_KEEP = "eligible_keep"
    REJECT_CANDIDATE = "reject_candidate"
    ABSTAIN_RETAIN = "abstain_retain"


@dataclass(frozen=True, slots=True)
class QualityEffectObservation:
    chunk_uid: str
    route_state: RouteState
    route: str | None
    provider_id: str | None
    provider_identity_sha256: str | None
    bin_id: str | None
    observation_artifact_sha256: str | None

    def __post_init__(self) -> None:
        if not self.chunk_uid:
            raise QualityEffectContractError("Quality observations require a chunk identifier")
        provider_fields = (
            self.route,
            self.provider_id,
            self.provider_identity_sha256,
            self.bin_id,
            self.observation_artifact_sha256,
        )
        present = all(value is not None for value in provider_fields)
        match self.route_state:
            case "routed":
                if not present:
                    raise QualityEffectContractError("Routed observations require complete provider-bin evidence")
                if not SHA256_RE.fullmatch(self.provider_identity_sha256 or "") or not SHA256_RE.fullmatch(
                    self.observation_artifact_sha256 or ""
                ):
                    raise QualityEffectContractError("Routed observations require lowercase SHA-256 identities")
            case "mixed" | "unknown" | "out_of_distribution":
                if any(value is not None for value in provider_fields):
                    raise QualityEffectContractError("Uncertain routes cannot carry provider-bin evidence")
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class QualityEvaluationRequest:
    observation: QualityEffectObservation


@dataclass(frozen=True, slots=True)
class QualityEffectDecision:
    decision: QualityEffectDecisionName
    reason_code: str
    chunk_uid: str
    effect_direction: EffectDirection | None
    evidence_artifact_hashes: tuple[str, ...]
    may_mutate_curated_membership: bool = False
    benchmark_outcomes_read: bool = False
    utility_read: bool = False
    source_reputation_read: bool = False


def _decision(
    request: QualityEvaluationRequest,
    decision: QualityEffectDecisionName,
    reason: str,
    direction: EffectDirection | None = None,
    hashes: tuple[str, ...] = (),
) -> QualityEffectDecision:
    return QualityEffectDecision(decision, reason, request.observation.chunk_uid, direction, hashes)


def evaluate_quality_effect(
    request: QualityEvaluationRequest,
    calibration: EffectCalibrationReport,
    provider: ProviderManifest,
) -> QualityEffectDecision:
    observation = request.observation
    if observation.route_state != "routed":
        return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_route_uncertain")
    if provider.role is not ProviderRole.QUALITY or provider.lifecycle is not ProviderLifecycle.ACTIVE or not provider.policy_contribution_authority:
        return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_provider_not_active")
    if observation.route not in provider.supported_routes:
        return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_provider_route_unsupported")
    if not calibration.passed:
        return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_calibration_gates_failed")
    expected_identity = provider.identity_sha256()
    if (
        observation.provider_id != provider.provider_id
        or calibration.provider_id != provider.provider_id
        or observation.provider_identity_sha256 != expected_identity
        or calibration.provider_identity_sha256 != expected_identity
    ):
        return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_provider_identity_mismatch")
    effect = next(
        (
            candidate
            for candidate in calibration.bins
            if candidate.bin_id == observation.bin_id and candidate.route == observation.route
        ),
        None,
    )
    if effect is None:
        return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_evidence_bin_missing")
    hashes = tuple(
        sorted(
            {
                expected_identity,
                calibration.effect_metric_artifact_sha256,
                calibration.common_baseline_artifact_sha256,
                effect.artifact_sha256,
                observation.observation_artifact_sha256 or "",
            }
        )
    )
    match effect.direction:
        case EffectDirection.SUPPORTED_NONPOSITIVE:
            return _decision(request, QualityEffectDecisionName.REJECT_CANDIDATE, "quality_nonpositive_effect_supported", effect.direction, hashes)
        case EffectDirection.SUPPORTED_POSITIVE:
            return _decision(request, QualityEffectDecisionName.ELIGIBLE_KEEP, "quality_positive_effect_supported", effect.direction, hashes)
        case EffectDirection.UNCERTAIN:
            return _decision(request, QualityEffectDecisionName.ABSTAIN_RETAIN, "quality_effect_uncertain", effect.direction, hashes)
        case unreachable:
            assert_never(unreachable)
