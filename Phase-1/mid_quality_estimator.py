from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Any, Final


JsonMap = dict[str, Any]
FORBIDDEN_GROUP_FIELDS: Final = frozenset(
    {
        "quality",
        "quality_score",
        "human_quality_label",
        "utility",
        "nll",
        "current_corpus_NLL_as_runtime_input",
        "benchmark",
        "benchmark_outcomes",
        "source",
        "source_identity",
        "domain",
        "target_retention_fraction",
        "budget",
    }
)


def _numeric_samples(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise RuntimeError(f"Mid Quality estimator requires at least two numeric {field}")
    if not all(isinstance(item, int | float) and math.isfinite(float(item)) for item in value):
        raise RuntimeError(f"Mid Quality estimator requires finite numeric {field}")
    return tuple(float(item) for item in value)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RuntimeError("Mid Quality estimator cannot compute a quantile from no values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _bootstrap_means(samples: Sequence[float], replicates: int, generator: random.Random) -> tuple[float, ...]:
    sample_count = len(samples)
    return tuple(sum(samples[generator.randrange(sample_count)] for _ in range(sample_count)) / sample_count for _ in range(replicates))


def _validate_parameters(confidence_level: float, bootstrap_replicates: int) -> None:
    if not 0.5 < confidence_level < 1.0:
        raise RuntimeError("Mid Quality estimator confidence level must be in (0.5, 1.0)")
    if bootstrap_replicates < 100:
        raise RuntimeError("Mid Quality estimator requires at least 100 bootstrap replicates")


def _decision(effect_estimate: float, upper_confidence_bound: float) -> str:
    if upper_confidence_bound <= 0.0:
        return "candidate_remove"
    if effect_estimate > 0.0:
        return "candidate_retain_positive"
    return "candidate_retain_uncertain"


def build_mid_quality_development_report(
    *,
    groups: Iterable[JsonMap],
    null_control_effect_samples: list[float],
    confidence_level: float,
    bootstrap_replicates: int,
    random_seed: int,
) -> JsonMap:
    """Summarize candidate-only group ablations with conservative control calibration."""
    _validate_parameters(confidence_level, bootstrap_replicates)
    control_samples = _numeric_samples(null_control_effect_samples, "null_control_effect_samples")
    control_bootstrap = _bootstrap_means(control_samples, bootstrap_replicates, random.Random(random_seed))
    calibration_margin = max(0.0, _quantile(control_bootstrap, confidence_level))

    seen_group_ids: set[str] = set()
    summaries: list[JsonMap] = []
    for raw_group in groups:
        group = dict(raw_group)
        forbidden = sorted(FORBIDDEN_GROUP_FIELDS.intersection(group))
        if forbidden:
            raise RuntimeError(f"Mid Quality group contains forbidden runtime inputs: {', '.join(forbidden)}")
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError("Mid Quality estimator requires a non-empty group_id")
        if group_id in seen_group_ids:
            raise RuntimeError(f"Mid Quality estimator received duplicate group_id: {group_id}")
        seen_group_ids.add(group_id)
        effect_samples = _numeric_samples(group.get("effect_samples"), "effect_samples")
        effect_estimate = sum(effect_samples) / len(effect_samples)
        group_seed = random_seed + sum(ord(character) for character in group_id)
        raw_upper_bound = _quantile(
            _bootstrap_means(effect_samples, bootstrap_replicates, random.Random(group_seed)),
            confidence_level,
        )
        upper_confidence_bound = raw_upper_bound + calibration_margin
        summaries.append(
            {
                "group_id": group_id,
                "effect_estimate": effect_estimate,
                "raw_upper_confidence_bound": raw_upper_bound,
                "calibration_margin": calibration_margin,
                "upper_confidence_bound": upper_confidence_bound,
                "decision": _decision(effect_estimate, upper_confidence_bound),
                "reason_code": "mid_quality_group_ablation_candidate",
            }
        )
    if not summaries:
        raise RuntimeError("Mid Quality estimator requires at least one group")
    return {
        "schema_version": "mid-quality-development-report-v1",
        "status": "candidate_only_development_artifact",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "target": "benchmark_disjoint_heldout_continuation_loss",
        "selection_rule": "negative_only_after_calibrated_upper_confidence_bound_is_non_positive",
        "calibration": {
            "method": "null_control_bootstrap_margin",
            "confidence_level": confidence_level,
            "bootstrap_replicates": bootstrap_replicates,
            "random_seed": random_seed,
            "margin": calibration_margin,
        },
        "groups": sorted(summaries, key=lambda row: str(row["group_id"])),
    }
