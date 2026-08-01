from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Final


JsonMap = dict[str, Any]
FORBIDDEN_DECLARATION_FIELDS: Final = frozenset(
    {
        "retention_fraction",
        "target_retention_fraction",
        "quality",
        "quality_score",
        "utility",
        "nll",
        "benchmark",
        "benchmark_outcomes",
        "source",
        "source_identity",
        "domain",
    }
)


def _required_text(declaration: JsonMap, field: str) -> str:
    value = declaration.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Hard Quality declaration requires non-empty {field}")
    return value


def _declared_budget(declaration: JsonMap) -> int:
    value = declaration.get("max_training_tokens")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("Hard Quality declaration requires positive integer max_training_tokens")
    return value


def _number(group: JsonMap, field: str) -> float:
    value = group.get(field)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise RuntimeError(f"Hard Quality group requires finite numeric {field}")
    return float(value)


def _token_count(group: JsonMap) -> int:
    value = group.get("token_count")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("Hard Quality group requires positive integer token_count")
    return value


def build_hard_quality_candidate_plan(*, groups: Iterable[JsonMap], declaration: JsonMap) -> JsonMap:
    """Create an opt-in, explicit-token-budget candidate plan from frozen Mid evidence."""
    forbidden = sorted(FORBIDDEN_DECLARATION_FIELDS.intersection(declaration))
    if forbidden:
        raise RuntimeError(f"Hard Quality declaration contains forbidden retention or runtime inputs: {', '.join(forbidden)}")
    budget = _declared_budget(declaration)
    model_id = _required_text(declaration, "model_id")
    tokenizer_sha256 = _required_text(declaration, "tokenizer_sha256")
    recipe_fingerprint = _required_text(declaration, "training_recipe_fingerprint")
    intended_use = _required_text(declaration, "intended_use")

    seen_group_ids: set[str] = set()
    eligible: list[JsonMap] = []
    excluded: list[JsonMap] = []
    for raw_group in groups:
        group = dict(raw_group)
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError("Hard Quality group requires non-empty group_id")
        if group_id in seen_group_ids:
            raise RuntimeError(f"Hard Quality candidate plan received duplicate group_id: {group_id}")
        seen_group_ids.add(group_id)
        effect_estimate = _number(group, "effect_estimate")
        upper_bound = _number(group, "upper_confidence_bound")
        token_count = _token_count(group)
        if upper_bound <= 0.0:
            excluded.append({"group_id": group_id, "token_count": token_count, "reason_code": "mid_quality_calibrated_non_positive_candidate"})
            continue
        if effect_estimate <= 0.0:
            excluded.append({"group_id": group_id, "token_count": token_count, "reason_code": "non_positive_expected_marginal_contribution"})
            continue
        eligible.append(
            {
                "group_id": group_id,
                "token_count": token_count,
                "effect_estimate": effect_estimate,
                "upper_confidence_bound": upper_bound,
                "expected_contribution_per_token": effect_estimate / token_count,
            }
        )
    if not seen_group_ids:
        raise RuntimeError("Hard Quality candidate plan requires at least one group")

    selected: list[JsonMap] = []
    remaining_tokens = budget
    for group in sorted(eligible, key=lambda item: (-float(item["expected_contribution_per_token"]), str(item["group_id"]))):
        if int(group["token_count"]) <= remaining_tokens:
            selected.append(group)
            remaining_tokens -= int(group["token_count"])
            continue
        excluded.append({"group_id": group["group_id"], "token_count": group["token_count"], "reason_code": "explicit_token_budget_exhausted"})
    return {
        "schema_version": "hard-quality-candidate-plan-v1",
        "status": "opt_in_candidate_only_not_runtime_active",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "declaration": {
            "model_id": model_id,
            "tokenizer_sha256": tokenizer_sha256,
            "training_recipe_fingerprint": recipe_fingerprint,
            "intended_use": intended_use,
        },
        "budget": {
            "kind": "explicit_user_declared_max_training_tokens",
            "declared_max_training_tokens": budget,
            "selected_token_count": budget - remaining_tokens,
            "unallocated_token_capacity": remaining_tokens,
        },
        "ranking": "descending_expected_marginal_contribution_per_token_then_group_id",
        "selected_groups": selected,
        "excluded_groups": sorted(excluded, key=lambda item: str(item["group_id"])),
    }
