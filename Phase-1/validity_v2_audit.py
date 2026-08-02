from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from validity_v2 import ValidityInput, ValidityV2Decision, evaluate_validity_v2


class ValidityAuditError(RuntimeError):
    """Raised when a Validity audit case is malformed."""


@dataclass(frozen=True, slots=True)
class ValidityAuditCase:
    case_id: str
    role: str
    input_unit: ValidityInput
    expected_status: str
    expected_action: str
    expected_reason: str | None

    def __post_init__(self) -> None:
        if self.role not in {"clean_control", "positive"}:
            raise ValidityAuditError(f"Unsupported fixture role: {self.role}")


@dataclass(frozen=True, slots=True)
class ReasonAudit:
    reason_code: str
    positive_trials: int
    false_negatives: int
    false_negative_upper_bound: float


@dataclass(frozen=True, slots=True)
class CaseAudit:
    case_id: str
    role: str
    expected_status: str
    observed_status: str
    expected_action: str
    observed_action: str
    expected_reason: str | None
    observed_reasons: tuple[str, ...]
    transformation_codes: tuple[str, ...]
    original_field_hashes: tuple[tuple[str, str], ...]
    original_bytes_sha256: str | None
    recovered_sha256: str
    source_record_sha256: str | None
    passed: bool


@dataclass(frozen=True, slots=True)
class ValidityAuditReport:
    clean_control_count: int
    clean_false_positive_count: int
    clean_false_positive_upper_bound: float
    positive_count: int
    positive_false_negative_count: int
    positive_false_negative_upper_bound: float
    per_reason: tuple[ReasonAudit, ...]
    case_results: tuple[CaseAudit, ...]
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


def _case_audit(case: ValidityAuditCase, decision: ValidityV2Decision, passed: bool) -> CaseAudit:
    return CaseAudit(
        case_id=case.case_id,
        role=case.role,
        expected_status=case.expected_status,
        observed_status=decision.status.value,
        expected_action=case.expected_action,
        observed_action=decision.action.value,
        expected_reason=case.expected_reason,
        observed_reasons=decision.reason_codes,
        transformation_codes=tuple(step.code for step in decision.transformation_trace),
        original_field_hashes=decision.original_field_hashes,
        original_bytes_sha256=decision.original_bytes_sha256,
        recovered_sha256=decision.recovered_sha256,
        source_record_sha256=decision.source_record_sha256,
        passed=passed,
    )


def build_validity_audit(cases: tuple[ValidityAuditCase, ...], confidence_level: float) -> ValidityAuditReport:
    if not 0.5 < confidence_level < 1.0:
        raise ValidityAuditError("confidence_level must be within (0.5, 1.0)")
    clean = [case for case in cases if case.role == "clean_control"]
    positives = [case for case in cases if case.role == "positive"]
    clean_failures = 0
    positive_failures = 0
    reason_counts: dict[str, tuple[int, int]] = {}
    case_results: list[CaseAudit] = []
    for case in clean:
        decision = evaluate_validity_v2(case.input_unit)
        passed = decision.status.value == case.expected_status and decision.action.value == case.expected_action
        clean_failures += not passed
        case_results.append(_case_audit(case, decision, passed))
    for case in positives:
        decision = evaluate_validity_v2(case.input_unit)
        missed = decision.status.value != case.expected_status or decision.action.value != case.expected_action or (
            case.expected_reason is not None and case.expected_reason not in decision.reason_codes
        )
        positive_failures += missed
        case_results.append(_case_audit(case, decision, not missed))
        if case.expected_reason is not None:
            trials, failures = reason_counts.get(case.expected_reason, (0, 0))
            reason_counts[case.expected_reason] = (trials + 1, failures + int(missed))
    per_reason = tuple(
        ReasonAudit(reason, trials, failures, _wilson_upper_bound(failures, trials, confidence_level))
        for reason, (trials, failures) in sorted(reason_counts.items())
    )
    clean_bound = _wilson_upper_bound(clean_failures, len(clean), confidence_level)
    positive_bound = _wilson_upper_bound(positive_failures, len(positives), confidence_level)
    return ValidityAuditReport(
        clean_control_count=len(clean),
        clean_false_positive_count=clean_failures,
        clean_false_positive_upper_bound=clean_bound,
        positive_count=len(positives),
        positive_false_negative_count=positive_failures,
        positive_false_negative_upper_bound=positive_bound,
        per_reason=per_reason,
        case_results=tuple(case_results),
        confidence_level=confidence_level,
        passed=bool(clean) and bool(positives) and clean_failures == 0 and positive_failures == 0,
    )
