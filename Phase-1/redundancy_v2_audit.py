from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from redundancy_v2 import RedundancySettings, RedundancyUnit, classify_relation


class RedundancyAuditError(RuntimeError):
    """Raised when a Redundancy fixture violates the audit contract."""


@dataclass(frozen=True, slots=True)
class RedundancyAuditCase:
    case_id: str
    role: str
    left: RedundancyUnit
    right: RedundancyUnit
    expected_relation: str
    semantic_candidate: bool = False

    def __post_init__(self) -> None:
        if self.role not in {"safe_family_positive", "retain_control", "candidate_only"}:
            raise RedundancyAuditError(f"Unsupported Redundancy fixture role: {self.role}")


@dataclass(frozen=True, slots=True)
class RedundancyCaseAudit:
    case_id: str
    role: str
    expected_relation: str
    observed_relation: str
    reason_code: str
    safe_family_edge: bool
    substantive_difference_codes: tuple[str, ...]
    changed_left_count: int
    changed_right_count: int
    changed_ratio: float
    repeated_span_hashes: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class RedundancyAuditReport:
    safe_family_positive_count: int
    safe_family_false_negative_count: int
    safe_family_false_negative_upper_bound: float
    retain_control_count: int
    retain_safe_family_false_positive_count: int
    retain_safe_family_false_positive_upper_bound: float
    candidate_only_count: int
    relation_mismatch_count: int
    case_results: tuple[RedundancyCaseAudit, ...]
    confidence_level: float
    passed: bool
    authority: str = "fixture_behavior_audit_only"
    runtime_activation: bool = False


def _wilson_upper_bound(failures: int, trials: int, confidence_level: float) -> float:
    if trials == 0:
        return 1.0
    z = NormalDist().inv_cdf(confidence_level)
    proportion = failures / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
    return min(1.0, (center + radius) / denominator)


def build_redundancy_audit(
    cases: tuple[RedundancyAuditCase, ...],
    settings: RedundancySettings,
    confidence_level: float,
) -> RedundancyAuditReport:
    if not 0.5 < confidence_level < 1.0:
        raise RedundancyAuditError("confidence_level must be within (0.5, 1.0)")
    safe_positives = [case for case in cases if case.role == "safe_family_positive"]
    retain_controls = [case for case in cases if case.role == "retain_control"]
    candidate_only = [case for case in cases if case.role == "candidate_only"]
    safe_false_negatives = 0
    retain_false_positives = 0
    mismatches = 0
    results: list[RedundancyCaseAudit] = []
    for case in cases:
        relation = classify_relation(
            case.left,
            case.right,
            settings,
            semantic_candidate=case.semantic_candidate,
        )
        mismatch = relation.relation.value != case.expected_relation
        mismatches += mismatch
        if case.role == "safe_family_positive":
            safe_false_negatives += not relation.safe_family_edge
        if case.role == "retain_control":
            retain_false_positives += relation.safe_family_edge
        passed = not mismatch and (case.role != "safe_family_positive" or relation.safe_family_edge) and (
            case.role != "retain_control" or not relation.safe_family_edge
        )
        results.append(
            RedundancyCaseAudit(
                case_id=case.case_id,
                role=case.role,
                expected_relation=case.expected_relation,
                observed_relation=relation.relation.value,
                reason_code=relation.reason_code,
                safe_family_edge=relation.safe_family_edge,
                substantive_difference_codes=relation.evidence.substantive_difference_codes,
                changed_left_count=relation.evidence.changed_left_count,
                changed_right_count=relation.evidence.changed_right_count,
                changed_ratio=relation.evidence.changed_ratio,
                repeated_span_hashes=relation.evidence.repeated_span_hashes,
                passed=passed,
            )
        )
    return RedundancyAuditReport(
        safe_family_positive_count=len(safe_positives),
        safe_family_false_negative_count=safe_false_negatives,
        safe_family_false_negative_upper_bound=_wilson_upper_bound(safe_false_negatives, len(safe_positives), confidence_level),
        retain_control_count=len(retain_controls),
        retain_safe_family_false_positive_count=retain_false_positives,
        retain_safe_family_false_positive_upper_bound=_wilson_upper_bound(retain_false_positives, len(retain_controls), confidence_level),
        candidate_only_count=len(candidate_only),
        relation_mismatch_count=mismatches,
        case_results=tuple(results),
        confidence_level=confidence_level,
        passed=bool(safe_positives) and bool(retain_controls) and safe_false_negatives == 0 and retain_false_positives == 0 and mismatches == 0,
    )
