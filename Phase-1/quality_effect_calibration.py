from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QualityEffectContractError(RuntimeError):
    """Raised when Quality effect evidence violates the frozen contract."""


class EffectDirection(str, Enum):
    SUPPORTED_NONPOSITIVE = "supported_nonpositive"
    UNCERTAIN = "uncertain"
    SUPPORTED_POSITIVE = "supported_positive"


@dataclass(frozen=True, slots=True)
class EffectInterval:
    point: float
    lower: float
    upper: float
    samples: int

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.point, self.lower, self.upper)):
            raise QualityEffectContractError("Effect intervals must be finite")
        if not self.lower <= self.point <= self.upper:
            raise QualityEffectContractError("Effect interval must contain its point estimate")
        if self.samples < 3:
            raise QualityEffectContractError("Effect intervals require at least three independent observations")


@dataclass(frozen=True, slots=True)
class EvidenceBin:
    route: str
    bin_id: str
    bin_order: int
    development: EffectInterval
    heldout: EffectInterval
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.route or not self.bin_id or self.bin_order < 0:
            raise QualityEffectContractError("Evidence bins require a route, ID, and nonnegative order")
        if not SHA256_RE.fullmatch(self.artifact_sha256):
            raise QualityEffectContractError("Evidence-bin artifacts require lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class EffectCalibrationBundle:
    provider_id: str
    provider_identity_sha256: str
    effect_metric_id: str
    effect_metric_artifact_sha256: str
    common_baseline_artifact_sha256: str
    bins: tuple[EvidenceBin, ...]
    provider_training_source_groups: frozenset[str]
    development_source_groups: frozenset[str]
    heldout_source_groups: frozenset[str]
    all_arms_share_common_baseline: bool
    common_baseline_disjoint_from_all_bins: bool
    external_results_hidden: bool
    provider_bias_stress_passed: bool
    route_holdout_stress_passed: bool

    def __post_init__(self) -> None:
        if not self.provider_id or not SHA256_RE.fullmatch(self.provider_identity_sha256):
            raise QualityEffectContractError("Calibration requires a frozen provider identity")
        if not self.effect_metric_id:
            raise QualityEffectContractError("Calibration requires a frozen effect metric ID")
        if not SHA256_RE.fullmatch(self.effect_metric_artifact_sha256) or not SHA256_RE.fullmatch(
            self.common_baseline_artifact_sha256
        ):
            raise QualityEffectContractError("Metric and common-baseline artifacts require lowercase SHA-256")
        if not self.bins:
            raise QualityEffectContractError("Calibration requires evidence bins")
        ids = tuple(effect.bin_id for effect in self.bins)
        route_orders = tuple((effect.route, effect.bin_order) for effect in self.bins)
        if len(ids) != len(set(ids)) or len(route_orders) != len(set(route_orders)):
            raise QualityEffectContractError("Evidence-bin IDs and route orders must be unique")
        if any(count < 3 for count in Counter(effect.route for effect in self.bins).values()):
            raise QualityEffectContractError("Each calibrated route requires at least three ordered bins")
        groups = (
            self.provider_training_source_groups,
            self.development_source_groups,
            self.heldout_source_groups,
        )
        if any(not group for group in groups):
            raise QualityEffectContractError("Every calibration split requires a declared source group")
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1 :]):
            raise QualityEffectContractError("Provider, development, and held-out source groups must be disjoint")


@dataclass(frozen=True, slots=True)
class CalibratedEffectBin:
    route: str
    bin_id: str
    bin_order: int
    fitted_development_effect: float
    direction: EffectDirection
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class EffectCalibrationReport:
    provider_id: str
    provider_identity_sha256: str
    effect_metric_id: str
    effect_metric_artifact_sha256: str
    common_baseline_artifact_sha256: str
    bins: tuple[CalibratedEffectBin, ...]
    failed_gates: tuple[str, ...]
    passed: bool
    weighted_quality_formula_used: bool = False
    target_retention_fraction_used: bool = False
    benchmark_outcomes_used: bool = False
    calibration_method: str = "sample_weighted_isotonic_bin_effects"
    effect_unit: str = "risk_reduction_per_target_token"


def _isotonic_fit(effects: tuple[EvidenceBin, ...]) -> tuple[float, ...]:
    blocks: list[list[float]] = []
    for effect in effects:
        blocks.append([1.0, float(effect.development.samples), effect.development.point * effect.development.samples])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][1] > blocks[-1][2] / blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1], left[2] + right[2]])
    fitted: list[float] = []
    for count, weight, total in blocks:
        fitted.extend([total / weight] * int(count))
    return tuple(fitted)


def _direction(effect: EvidenceBin) -> EffectDirection:
    if effect.development.upper <= 0.0 and effect.heldout.upper <= 0.0:
        return EffectDirection.SUPPORTED_NONPOSITIVE
    if effect.development.lower > 0.0 and effect.heldout.lower > 0.0:
        return EffectDirection.SUPPORTED_POSITIVE
    return EffectDirection.UNCERTAIN


def calibrate_effect_bins(bundle: EffectCalibrationBundle) -> EffectCalibrationReport:
    failed: list[str] = []
    if not bundle.external_results_hidden:
        failed.append("external_feedback_leakage")
    if not bundle.provider_bias_stress_passed:
        failed.append("provider_bias_stress_missing")
    if not bundle.route_holdout_stress_passed:
        failed.append("route_holdout_stress_missing")
    if not bundle.all_arms_share_common_baseline:
        failed.append("common_baseline_not_shared")
    if not bundle.common_baseline_disjoint_from_all_bins:
        failed.append("common_baseline_not_disjoint")
    calibrated: list[CalibratedEffectBin] = []
    direction_order = {
        EffectDirection.SUPPORTED_NONPOSITIVE: 0,
        EffectDirection.UNCERTAIN: 1,
        EffectDirection.SUPPORTED_POSITIVE: 2,
    }
    for route in sorted({effect.route for effect in bundle.bins}):
        ordered = tuple(sorted((effect for effect in bundle.bins if effect.route == route), key=lambda effect: effect.bin_order))
        fitted = _isotonic_fit(ordered)
        heldout_points = tuple(effect.heldout.point for effect in ordered)
        if any(left > right for left, right in zip(heldout_points, heldout_points[1:])):
            failed.append(f"heldout_nonmonotonic:{route}")
        route_directions = tuple(_direction(effect) for effect in ordered)
        ranks = tuple(direction_order[direction] for direction in route_directions)
        if any(left > right for left, right in zip(ranks, ranks[1:])):
            failed.append(f"direction_nonmonotonic:{route}")
        for effect, fitted_value, direction in zip(ordered, fitted, route_directions, strict=True):
            if not effect.heldout.lower <= fitted_value <= effect.heldout.upper:
                failed.append(f"heldout_interval_miss:{effect.bin_id}")
            calibrated.append(
                CalibratedEffectBin(
                    route=effect.route,
                    bin_id=effect.bin_id,
                    bin_order=effect.bin_order,
                    fitted_development_effect=fitted_value,
                    direction=direction,
                    artifact_sha256=effect.artifact_sha256,
                )
            )
    unique_failures = tuple(dict.fromkeys(failed))
    return EffectCalibrationReport(
        provider_id=bundle.provider_id,
        provider_identity_sha256=bundle.provider_identity_sha256,
        effect_metric_id=bundle.effect_metric_id,
        effect_metric_artifact_sha256=bundle.effect_metric_artifact_sha256,
        common_baseline_artifact_sha256=bundle.common_baseline_artifact_sha256,
        bins=tuple(calibrated),
        failed_gates=unique_failures,
        passed=not unique_failures,
    )
