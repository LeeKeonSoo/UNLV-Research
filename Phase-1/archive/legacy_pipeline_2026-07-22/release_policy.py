"""Deployment-contract validation and release-policy decisions."""

from __future__ import annotations

from typing import Any, Dict, List


OBJECTIVE_TYPES = {
    "broad_refresh",
    "targeted_update",
    "capability_preserving_update",
}
RELEASE_ACTIONS = {
    "selected_only",
    "coverage_backfilled",
    "stageA_broad",
    "reject",
    "insufficient_usable_data",
}


def _numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _improvement(candidate: float, reference: float, direction: str) -> float:
    if direction == "lower_is_better":
        return reference - candidate
    if direction == "higher_is_better":
        return candidate - reference
    raise ValueError(f"Unsupported metric direction: {direction}")


def validate_deployment_contract(contract: Dict[str, Any]) -> None:
    if contract.get("schema_version") != "deployment-contract-v1":
        raise ValueError("Unsupported deployment contract schema")
    if contract.get("objective_type") not in OBJECTIVE_TYPES:
        raise ValueError(f"Unsupported objective_type: {contract.get('objective_type')}")
    primary = contract.get("primary_outcome") if isinstance(contract.get("primary_outcome"), dict) else {}
    if not str(primary.get("evaluation") or ""):
        raise ValueError("Deployment contract requires primary_outcome.evaluation")
    if primary.get("direction") not in {"lower_is_better", "higher_is_better"}:
        raise ValueError("Deployment contract requires a supported primary_outcome.direction")
    if not str(primary.get("comparison_reference") or ""):
        raise ValueError("Deployment contract requires primary_outcome.comparison_reference")
    eligible = list(contract.get("eligible_release_actions") or [])
    if not eligible or any(action not in RELEASE_ACTIONS for action in eligible):
        raise ValueError("Deployment contract has invalid eligible_release_actions")
    preference = list(contract.get("preference_order") or [])
    if any(action not in eligible for action in preference):
        raise ValueError("preference_order must contain only eligible_release_actions")
    if str(contract.get("utility_scope") or "") != "Stage C validation only; never selector objective":
        raise ValueError("Deployment contract violates Utility scope")


def decide_release(contract: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    validate_deployment_contract(contract)
    if not bool(evidence.get("usable_data_sufficient", True)):
        return {
            "release_action": "insufficient_usable_data",
            "supported": False,
            "rationale": "The usable data pool cannot support the declared deployment contract.",
            "candidate_assessments": {},
            "claim_scope": "abstention",
        }

    arms = evidence.get("arms") if isinstance(evidence.get("arms"), dict) else {}
    primary = contract["primary_outcome"]
    evaluation = str(primary["evaluation"])
    direction = str(primary["direction"])
    reference_arm = str(primary["comparison_reference"])
    minimum_improvement = float(primary.get("minimum_improvement") or 0.0)
    reference_value = _numeric(((arms.get(reference_arm) or {}).get("evaluations") or {}).get(evaluation))
    if reference_value is None:
        raise ValueError(f"Missing primary reference evidence: {reference_arm}.{evaluation}")

    assessments: Dict[str, Any] = {}
    for action in contract.get("eligible_release_actions") or []:
        if action in {"reject", "insufficient_usable_data"}:
            continue
        arm = arms.get(action) if isinstance(arms.get(action), dict) else {}
        candidate_value = _numeric((arm.get("evaluations") or {}).get(evaluation))
        reasons: List[str] = []
        primary_gain = None
        if candidate_value is None:
            reasons.append("missing_primary_outcome")
        else:
            primary_gain = _improvement(candidate_value, reference_value, direction)
            if action != reference_arm and primary_gain < minimum_improvement:
                reasons.append("primary_improvement_below_contract")
            if action == reference_arm and bool(primary.get("reference_requires_base_gain")):
                base_value = _numeric(((arms.get("base_no_update") or {}).get("evaluations") or {}).get(evaluation))
                if base_value is None:
                    reasons.append("missing_base_reference")
                else:
                    gain_over_base = _improvement(candidate_value, base_value, direction)
                    if gain_over_base < float(primary.get("minimum_reference_gain_over_base") or 0.0):
                        reasons.append("reference_gain_over_base_below_contract")
        guardrail_rows = []
        for guardrail in contract.get("guardrails") or []:
            guard_eval = str(guardrail.get("evaluation") or "")
            guard_direction = str(guardrail.get("direction") or "")
            if not guard_eval or guard_direction not in {"lower_is_better", "higher_is_better"}:
                raise ValueError("Guardrails require evaluation and supported direction")
            guard_reference = str(guardrail.get("comparison_reference") or "base_no_update")
            candidate_guard = _numeric((arm.get("evaluations") or {}).get(guard_eval))
            reference_guard = _numeric(((arms.get(guard_reference) or {}).get("evaluations") or {}).get(guard_eval))
            passed = candidate_guard is not None and reference_guard is not None
            improvement = None
            if passed:
                improvement = _improvement(candidate_guard, reference_guard, guard_direction)
                passed = improvement >= -float(guardrail.get("maximum_regression") or 0.0)
            if not passed and bool(guardrail.get("required", True)):
                reasons.append(f"guardrail_failed:{guard_eval}")
            guardrail_rows.append(
                {
                    "evaluation": guard_eval,
                    "comparison_reference": guard_reference,
                    "improvement": improvement,
                    "passed": passed,
                }
            )
        assessments[action] = {
            "eligible": not reasons,
            "primary_value": candidate_value,
            "primary_reference_value": reference_value,
            "primary_improvement": primary_gain,
            "guardrails": guardrail_rows,
            "reasons": reasons,
        }

    chosen = next(
        (action for action in contract.get("preference_order") or [] if (assessments.get(action) or {}).get("eligible")),
        None,
    )
    if chosen is None:
        return {
            "release_action": "reject",
            "supported": False,
            "rationale": "No candidate release satisfies the declared primary outcome and guardrails.",
            "candidate_assessments": assessments,
            "claim_scope": str(contract.get("claim_scope") or "declared deployment objective"),
        }
    return {
        "release_action": chosen,
        "supported": True,
        "rationale": f"{chosen} is the highest-preference release satisfying the declared primary outcome and guardrails.",
        "candidate_assessments": assessments,
        "claim_scope": str(contract.get("claim_scope") or "declared deployment objective"),
    }
