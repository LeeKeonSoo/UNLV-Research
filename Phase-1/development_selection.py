from __future__ import annotations

import math
from dataclasses import dataclass
from typing import assert_never

from development_selection_contract import (
    ArmRole,
    CorpusDomain,
    CorpusScenario,
    CorpusSliceEvidence,
    DevelopmentArm,
    DevelopmentGateEvidence,
    DevelopmentProtocol,
    DevelopmentSelectionBundle,
    DevelopmentSelectionContractError,
    DevelopmentSelectionResult,
    DevelopmentSelectionStatus,
    hash_json,
    load_development_protocol,
)


@dataclass(frozen=True, slots=True)
class _SelectionChoice:
    blockers: tuple[str, ...]
    normal: DevelopmentArm | None
    hard: DevelopmentArm | None


def _wilson_upper(failures: int, count: int, z: float) -> float:
    if count <= 0 or failures < 0 or failures > count:
        return math.inf
    proportion = failures / count
    denominator = 1 + z * z / count
    center = proportion + z * z / (2 * count)
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
    return (center + radius) / denominator


def _arm_eligible(arm: DevelopmentArm, bundle: DevelopmentSelectionBundle) -> bool:
    return (
        0 < arm.exact_natural_tokens < bundle.base_exact_natural_tokens
        and arm.development_gain_lcb_per_token >= 0
        and arm.all_sensitivity_arms_share_one_common_baseline
        and arm.common_baseline_disjoint_from_all_arms
        and arm.development_and_heldout_effect_arms_disjoint
        and arm.minimum_support_recall == 1
        and arm.extinct_supported_strata == 0
        and arm.unknown_mixed_extinct_strata == 0
        and arm.representative_linkage_complete
        and arm.valid_residuals_complete
        and arm.removal_trace_count == arm.complete_removal_trace_count
    )


def _dominates(left: DevelopmentArm, right: DevelopmentArm) -> bool:
    no_worse = (
        left.exact_natural_tokens <= right.exact_natural_tokens
        and left.development_gain_lcb_per_token >= right.development_gain_lcb_per_token
        and left.maximum_coverage_js_divergence <= right.maximum_coverage_js_divergence
    )
    strict = (
        left.exact_natural_tokens < right.exact_natural_tokens
        or left.development_gain_lcb_per_token > right.development_gain_lcb_per_token
        or left.maximum_coverage_js_divergence < right.maximum_coverage_js_divergence
    )
    return no_worse and strict


def _frontier(arms: tuple[DevelopmentArm, ...]) -> tuple[DevelopmentArm, ...]:
    return tuple(arm for arm in arms if not any(_dominates(other, arm) for other in arms if other.arm_id != arm.arm_id))


def _profile_hash(bundle: DevelopmentSelectionBundle, arm: DevelopmentArm) -> str:
    return hash_json({"arm_id": arm.arm_id, "policies": (*arm.required_policy_ids, *arm.hard_extension_policy_ids), "protocol": bundle.protocol.identity_sha256(), "evidence": arm.evidence_artifact_hashes})


def _result(bundle: DevelopmentSelectionBundle, choice: _SelectionChoice) -> DevelopmentSelectionResult:
    frozen = not choice.blockers and choice.normal is not None and choice.hard is not None
    payload = {"protocol": bundle.protocol.identity_sha256(), "blockers": choice.blockers, "normal": choice.normal.arm_id if choice.normal else None, "hard": choice.hard.arm_id if choice.hard else None, "base": bundle.base_input_manifest_sha256}
    return DevelopmentSelectionResult(
        status=DevelopmentSelectionStatus.FROZEN if frozen else DevelopmentSelectionStatus.BLOCKED,
        blocker_codes=choice.blockers,
        normal_arm_id=choice.normal.arm_id if frozen and choice.normal else None,
        hard_arm_id=choice.hard.arm_id if frozen and choice.hard else None,
        normal_profile_sha256=_profile_hash(bundle, choice.normal) if frozen and choice.normal else None,
        hard_profile_sha256=_profile_hash(bundle, choice.hard) if frozen and choice.hard else None,
        protocol_sha256=bundle.protocol.identity_sha256(),
        manifest_sha256=hash_json(payload),
    )


def _split_by_role(arms: tuple[DevelopmentArm, ...]) -> tuple[tuple[DevelopmentArm, ...], tuple[DevelopmentArm, ...]]:
    normal: list[DevelopmentArm] = []
    hard: list[DevelopmentArm] = []
    for arm in arms:
        match arm.role:
            case ArmRole.NORMAL:
                normal.append(arm)
            case ArmRole.HARD_EXTENSION:
                hard.append(arm)
            case unreachable:
                assert_never(unreachable)
    return tuple(normal), tuple(hard)


def select_development_profiles(bundle: DevelopmentSelectionBundle) -> DevelopmentSelectionResult:
    bundle.validate_contract()
    gates = bundle.gates
    blockers: list[str] = []
    for ready, code in ((gates.redundancy_ready, "redundancy_gate_not_ready"), (gates.quality_ready, "quality_gate_not_ready"), (gates.coverage_ready, "coverage_gate_not_ready"), (gates.external_results_hidden, "external_results_visible"), (gates.provider_bias_stress_passed, "provider_bias_stress_failed"), (gates.route_holdout_stress_passed, "route_holdout_stress_failed")):
        if not ready:
            blockers.append(code)
    upper = _wilson_upper(gates.clean_control_false_positives, gates.clean_control_count, bundle.protocol.one_sided_confidence_z)
    if upper > bundle.protocol.maximum_clean_false_positive_upper_bound:
        blockers.append("clean_control_false_positive_bound_failed")
    eligible = tuple(arm for arm in bundle.arms if _arm_eligible(arm, bundle))
    normal_candidates, hard_candidates = _split_by_role(eligible)
    normal = _frontier(normal_candidates)
    hard = _frontier(hard_candidates)
    if not normal:
        blockers.append("normal_operating_point_missing")
    if not hard:
        blockers.append("hard_operating_point_missing")
    if blockers:
        return _result(bundle, _SelectionChoice(tuple(sorted(set(blockers))), None, None))
    selected_normal = sorted(normal, key=lambda arm: (-arm.development_gain_lcb_per_token, arm.exact_natural_tokens, arm.arm_id))[0]
    selected_hard = sorted(hard, key=lambda arm: (arm.exact_natural_tokens, -arm.development_gain_lcb_per_token, arm.arm_id))[0]
    if selected_hard.exact_natural_tokens > selected_normal.exact_natural_tokens:
        return _result(bundle, _SelectionChoice(("hard_retains_more_than_normal",), None, None))
    return _result(bundle, _SelectionChoice((), selected_normal, selected_hard))


__all__ = [
    "ArmRole", "CorpusDomain", "CorpusScenario", "CorpusSliceEvidence",
    "DevelopmentArm", "DevelopmentGateEvidence", "DevelopmentProtocol",
    "DevelopmentSelectionBundle", "DevelopmentSelectionContractError",
    "DevelopmentSelectionResult", "DevelopmentSelectionStatus",
    "load_development_protocol", "select_development_profiles",
]
