from __future__ import annotations

import math
from dataclasses import dataclass


class QualificationContractError(RuntimeError):
    """Raised when a qualification observation is mathematically invalid."""


def _binomial_cdf(observed: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if observed == trials else 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = (
        math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            + index * log_p
            + (trials - index) * log_q
        )
        for index in range(observed + 1)
    )
    return math.fsum(terms)


def exact_one_sided_upper_bound(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    if trials <= 0:
        raise QualificationContractError("Trial count must be positive")
    if successes < 0 or successes > trials:
        raise QualificationContractError("Success count cannot exceed trials or be negative")
    if not 0.0 < confidence < 1.0:
        raise QualificationContractError("Confidence must be strictly between zero and one")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    low = successes / trials
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


@dataclass(frozen=True, slots=True)
class ProtectedFixtureGate:
    sample_count: int
    false_removal_count: int
    maximum_upper_bound: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise QualificationContractError("Protected fixture count must be positive")
        if self.false_removal_count < 0 or self.false_removal_count > self.sample_count:
            raise QualificationContractError("False-removal count cannot exceed protected fixture count")
        if not 0.0 < self.maximum_upper_bound < 1.0:
            raise QualificationContractError("Maximum upper bound must be strictly between zero and one")

    def upper_bound(self, confidence: float) -> float:
        return exact_one_sided_upper_bound(
            successes=self.false_removal_count,
            trials=self.sample_count,
            confidence=confidence,
        )

    def passes(self, confidence: float) -> bool:
        return self.upper_bound(confidence) <= self.maximum_upper_bound
