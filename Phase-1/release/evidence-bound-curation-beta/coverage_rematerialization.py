from __future__ import annotations

from dataclasses import dataclass, replace
from typing import assert_never

from coverage_contract import CoverageDecision, CoverageRequest, CoverageStatus
from coverage_engine import evaluate_coverage
from model_provider_contract import ProviderManifest


class CoverageRematerializationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CoverageRematerialization:
    initial_decision: CoverageDecision
    final_decision: CoverageDecision
    required_retain_uids: tuple[str, ...]
    final_survivor_uids: tuple[str, ...]
    rematerialization_applied: bool
    silent_restore: bool = False
    benchmark_outcomes_read: bool = False
    utility_read: bool = False


def rematerialize_with_coverage(
    request: CoverageRequest,
    provider: ProviderManifest,
) -> CoverageRematerialization:
    initial = evaluate_coverage(request, provider)
    match initial.status:
        case CoverageStatus.PASS:
            return CoverageRematerialization(
                initial,
                initial,
                (),
                tuple(sorted(request.proposed_survivors)),
                False,
            )
        case CoverageStatus.ABSTAIN:
            universe = tuple(sorted(chunk.uid for chunk in request.chunks))
            return CoverageRematerialization(initial, initial, (), universe, False)
        case CoverageStatus.VETO_CANDIDATE:
            required = initial.required_retain_uids
        case unreachable:
            assert_never(unreachable)
    revised = replace(
        request,
        proposed_survivors=request.proposed_survivors | frozenset(required),
    )
    final = evaluate_coverage(revised, provider)
    if final.status is not CoverageStatus.PASS:
        raise CoverageRematerializationError(
            "Coverage veto rematerialization must pass a second complete evaluation"
        )
    return CoverageRematerialization(
        initial,
        final,
        required,
        tuple(sorted(revised.proposed_survivors)),
        True,
    )
