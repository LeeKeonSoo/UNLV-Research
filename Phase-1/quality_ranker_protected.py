from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quality_ranker_artifact import sha256_file


@dataclass(frozen=True, slots=True)
class ObservationArtifact:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TeacherObservation:
    chunk_uid: str
    text_sha256: str
    policy_decisions: tuple[tuple[str, str], ...]

    def decision_for(self, policy_id: str) -> str | None:
        return next(
            (decision for observed_id, decision in self.policy_decisions if observed_id == policy_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class ObservationUniverse:
    panel_sha256: str
    runtime_sha256: str
    aggregation_strategy: str
    observations: tuple[TeacherObservation, ...]
    artifacts: tuple[ObservationArtifact, ...]

    def by_uid(self) -> dict[str, TeacherObservation]:
        return {observation.chunk_uid: observation for observation in self.observations}


@dataclass(frozen=True, slots=True)
class ProtectedThresholdConfig:
    maximum_false_positive_rate: float
    minimum_negative_count: int


@dataclass(frozen=True, slots=True)
class ThresholdVerification:
    candidate_threshold: float | None
    activated_threshold: float | None
    negative_count: int
    false_positive_count: int
    false_positive_upper_bound: float | None
    status: str

    def as_mapping(self) -> dict[str, str | int | float | None]:
        return {
            "candidate_threshold": self.candidate_threshold,
            "activated_threshold": self.activated_threshold,
            "negative_count": self.negative_count,
            "false_positive_count": self.false_positive_count,
            "false_positive_upper_bound": self.false_positive_upper_bound,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ProtectedObservationError(RuntimeError):
    reason_code: str
    overlap_count: int = 0

    def __str__(self) -> str:
        return f"{self.reason_code}:{self.overlap_count}"


def load_observation_universe(paths: tuple[Path, ...]) -> ObservationUniverse:
    if not paths:
        raise ProtectedObservationError("protected_observation_paths_missing")
    panel_ids: set[str] = set()
    runtime_ids: set[str] = set()
    aggregation_strategies: set[str] = set()
    observations: dict[str, TeacherObservation] = {}
    artifacts: list[ObservationArtifact] = []
    for path in paths:
        artifacts.append(ObservationArtifact(path=str(path), sha256=sha256_file(path)))
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                aggregation_strategy = str(
                    row.get("aggregation_strategy") or "three_teacher_stable_majority"
                )
                match aggregation_strategy:
                    case "three_teacher_stable_majority":
                        required_teachers = 3
                    case "single_teacher_confirmed_fail":
                        required_teachers = 1
                    case _:
                        raise ProtectedObservationError(
                            "unknown_teacher_aggregation_strategy"
                        )
                if len(tuple(row.get("available_teacher_ids") or ())) != required_teachers:
                    continue
                panel_ids.add(str(row["teacher_panel_sha256"]))
                runtime_ids.add(str(row["quality_runtime_sha256"]))
                aggregation_strategies.add(aggregation_strategy)
                observation = TeacherObservation(
                    chunk_uid=str(row["chunk_uid"]),
                    text_sha256=str(row["text_sha256"]),
                    policy_decisions=tuple(
                        (str(result["policy_id"]), str(result["panel_decision"]))
                        for result in row.get("policy_results") or ()
                    ),
                )
                existing = observations.get(observation.chunk_uid)
                if existing is not None and existing != observation:
                    raise ProtectedObservationError("conflicting_teacher_observation")
                observations[observation.chunk_uid] = observation
    if (
        len(panel_ids) != 1
        or len(runtime_ids) != 1
        or len(aggregation_strategies) != 1
        or not observations
    ):
        raise ProtectedObservationError("teacher_observation_universe_missing")
    return ObservationUniverse(
        panel_sha256=next(iter(panel_ids)),
        runtime_sha256=next(iter(runtime_ids)),
        aggregation_strategy=next(iter(aggregation_strategies)),
        observations=tuple(observations[uid] for uid in sorted(observations)),
        artifacts=tuple(artifacts),
    )


def require_disjoint_observations(
    calibration: ObservationUniverse,
    protected: ObservationUniverse,
) -> None:
    if calibration.panel_sha256 != protected.panel_sha256:
        raise ProtectedObservationError("protected_observation_panel_mismatch")
    if calibration.runtime_sha256 != protected.runtime_sha256:
        raise ProtectedObservationError("protected_observation_runtime_mismatch")
    if calibration.aggregation_strategy != protected.aggregation_strategy:
        raise ProtectedObservationError("protected_observation_aggregation_mismatch")
    calibration_uids = {item.chunk_uid for item in calibration.observations}
    protected_uids = {item.chunk_uid for item in protected.observations}
    uid_overlap = calibration_uids & protected_uids
    calibration_hashes = {item.text_sha256 for item in calibration.observations}
    protected_hashes = {item.text_sha256 for item in protected.observations}
    text_overlap = calibration_hashes & protected_hashes
    overlap_count = len(uid_overlap | text_overlap)
    if overlap_count:
        raise ProtectedObservationError("protected_observation_overlap", overlap_count)


def wilson_upper_bound(false_positives: int, negatives: int) -> float:
    if negatives < 1 or not 0 <= false_positives <= negatives:
        raise ProtectedObservationError("invalid_false_positive_counts")
    z = 1.6448536269514722
    rate = false_positives / negatives
    denominator = 1.0 + (z * z / negatives)
    center = rate + (z * z / (2.0 * negatives))
    radius = z * math.sqrt(
        (rate * (1.0 - rate) / negatives) + (z * z / (4.0 * negatives * negatives))
    )
    return (center + radius) / denominator


def verify_threshold(
    candidate: float | None,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    config: ProtectedThresholdConfig,
) -> ThresholdVerification:
    negatives = labels == 0
    negative_count = int(np.count_nonzero(negatives))
    if candidate is None:
        return ThresholdVerification(None, None, negative_count, 0, None, "candidate_missing")
    if negative_count < config.minimum_negative_count:
        return ThresholdVerification(
            candidate,
            None,
            negative_count,
            0,
            None,
            "insufficient_protected_negatives",
        )
    false_positives = int(np.count_nonzero((probabilities >= candidate) & negatives))
    upper_bound = wilson_upper_bound(false_positives, negative_count)
    activated = (
        candidate if upper_bound <= config.maximum_false_positive_rate else None
    )
    return ThresholdVerification(
        candidate,
        activated,
        negative_count,
        false_positives,
        upper_bound,
        "verified" if activated is not None else "false_positive_bound_failed",
    )
