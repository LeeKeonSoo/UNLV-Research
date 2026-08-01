#!/usr/bin/env python3
"""Threshold profiles and subset generation logic."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from tqdm import tqdm

from data_eval_common import (
    DEFAULT_PROFILE_CONFIG,
    PROFILE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SCORED_DIR,
    SUBSETS_DIR,
    UTILITY_PROBE_RESULTS_PATH,
    iter_jsonl_records_resilient,
    load_json,
    repeated_token_ratio,
    save_json,
)
from reports.summary import write_run_reports
from policy.dispositions import annotate_retained_pool, disposition_summary
from policy.stage_b_budget import fit_word_budget, resolve_stage_b_budget as _resolve_stage_b_budget
from signals.core import style_bucket_from_text
from utility.lm_probe import aggregate_probe_runs, build_probe_context, score_selected_records


INDEX_DB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "index" / "index.sqlite"
SCORING_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "outputs" / "scored" / "scoring_manifest.json"
UTILITY_SENSITIVITY_AUDIT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "validation" / "utility_sensitivity_audit.json"
_UTILITY_SENSITIVITY_AUDIT_CACHE: Dict[str, Any] | None = None


def _progress(message: str) -> None:
    print(f"[04] {message}", flush=True)


def _elapsed_seconds(started: float) -> str:
    return f"{time.perf_counter() - started:.1f}s"


def load_profiles(path: Path = DEFAULT_PROFILE_CONFIG) -> Dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported profile schema: {payload.get('schema_version')}")
    return payload


def _iter_scored_records(path: Path) -> Iterator[Dict[str, Any]]:
    yield from iter_jsonl_records_resilient(path)


def _stage_a_gate(profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    raw = profile.get("stage_a_gate") or {}
    if raw:
        floors = dict(raw.get("metric_floors") or {})
        ceilings = dict(raw.get("metric_ceilings") or {})
    else:
        floors = dict(profile.get("metric_floors") or {})
        ceilings = dict(profile.get("metric_ceilings") or {})
    return {
        "metric_floors": {str(k): float(v) for k, v in floors.items()},
        "metric_ceilings": {str(k): float(v) for k, v in ceilings.items()},
    }


def _stage_b_rank(profile: Dict[str, Any]) -> Dict[str, Any]:
    raw = profile.get("stage_b_rank") or {}
    weights = dict(
        raw.get("weights")
        or profile.get("weights")
        or {"quality": 0.8, "redundancy": 0.2}
    )
    if "redundancy" not in weights and "efficiency" in weights:
        weights["redundancy"] = weights.get("efficiency")
    quality_w = float(weights.get("quality", 0.7))
    redundancy_w = float(weights.get("redundancy", 0.3))
    total = quality_w + redundancy_w
    if total <= 0.0:
        quality_w, redundancy_w, total = 0.7, 0.3, 1.0
    normalized = {
        "quality": quality_w / total,
        "redundancy": redundancy_w / total,
    }
    metric_ceilings = profile.get("metric_ceilings") or {}
    explicit_near_dup_ceiling = raw.get(
        "near_duplicate_risk_ceiling",
        metric_ceilings.get("shingle_near_duplicate_risk_score", 1.0),
    )
    quantile_ceiling = raw.get("near_duplicate_risk_quantile_ceiling")
    return {
        "selection_threshold": float(raw.get("selection_threshold", profile.get("selection_threshold", 0.0))),
        "near_duplicate_risk_ceiling": float(explicit_near_dup_ceiling) if explicit_near_dup_ceiling is not None else 1.0,
        "near_duplicate_risk_quantile_ceiling": (
            float(quantile_ceiling) if quantile_ceiling is not None else None
        ),
        "near_duplicate_risk_quantile_sample_size": int(raw.get("near_duplicate_risk_quantile_sample_size", 60000)),
        "weights": normalized,
    }


def _selector_config(profile: Dict[str, Any]) -> Dict[str, Any]:
    raw = profile.get("selector") or {}
    objective_weights = dict(raw.get("objective_weights") or {})
    if not objective_weights:
        # Backward compatibility fallback.
        stage_b = _stage_b_rank(profile)
        objective_weights = {
            "quality": float(stage_b["weights"].get("quality", 0.75)),
            "redundancy_risk": float(stage_b["weights"].get("redundancy", 0.25)),
        }
    quality_w = max(0.0, float(objective_weights.get("quality", 0.6)))
    # risk weight is subtractive in objective.
    risk_w = max(0.0, float(objective_weights.get("redundancy_risk", 0.2)))
    denom = quality_w + risk_w
    if denom <= 0.0:
        quality_w, risk_w = 0.75, 0.25
        denom = 1.0
    normalized = {
        "quality": quality_w / denom,
        "redundancy_risk": risk_w / denom,
    }

    penalties = dict(raw.get("constraint_penalties") or {})
    adjustments = dict(raw.get("selection_adjustments") or {})
    return {
        "objective_weights": normalized,
        "constraint_penalties": {
            "rare_cluster_bonus": float(penalties.get("rare_cluster_bonus", 0.12)),
            "small_cluster_bonus": float(penalties.get("small_cluster_bonus", 0.06)),
            "penalty_growth": float(penalties.get("penalty_growth", 1.35)),
            "threshold_relax_step": float(penalties.get("threshold_relax_step", 0.015)),
        },
        "selection_adjustments": {
            "useful_length_bonus": float(adjustments.get("useful_length_bonus", 0.0)),
            "lexical_diversity_bonus": float(adjustments.get("lexical_diversity_bonus", 0.0)),
            "useful_recurrence_bonus": float(adjustments.get("useful_recurrence_bonus", 0.0)),
            "learnability_support_bonus": float(adjustments.get("learnability_support_bonus", 0.0)),
            "pattern_recurrence_bonus": float(adjustments.get("pattern_recurrence_bonus", 0.0)),
            "quality_tail_penalty": float(adjustments.get("quality_tail_penalty", 0.0)),
            "boilerplate_penalty": float(adjustments.get("boilerplate_penalty", 0.0)),
        },
        "iteration_cap": int(raw.get("iteration_cap", 10)),
        "hash_sampling_seed": int(raw.get("hash_sampling_seed", 42)),
        "min_selected_ratio": float(raw.get("min_selected_ratio", 0.0)),
        "min_selected_tokens": int(raw.get("min_selected_tokens", 0)),
        "rare_cluster_min_count": int(raw.get("rare_cluster_min_count", 1)),
        "preserve_domain_bucket_exemplars": bool(raw.get("preserve_domain_bucket_exemplars", True)),
        "domain_bucket_min_count": int(raw.get("domain_bucket_min_count", 1)),
        "preserve_style_bucket_exemplars": bool(raw.get("preserve_style_bucket_exemplars", True)),
        "style_bucket_min_count": int(raw.get("style_bucket_min_count", 4)),
        "preserve_domain_distribution": bool(raw.get("preserve_domain_distribution", True)),
        "preserve_style_distribution": bool(raw.get("preserve_style_distribution", True)),
        "preserve_length_distribution": bool(raw.get("preserve_length_distribution", True)),
        # Kept off by default because preserving low-quality bands can directly
        # fight the Quality core. Quality-band matching is still measured as a
        # Utility diagnostic baseline.
        "preserve_quality_band_distribution": bool(raw.get("preserve_quality_band_distribution", False)),
        "diagnose_quality_band_distribution": bool(raw.get("diagnose_quality_band_distribution", True)),
        "quality_band_distribution_min_quality": float(raw.get("quality_band_distribution_min_quality", 0.0)),
        "quality_band_rebalance_mode": str(raw.get("quality_band_rebalance_mode") or "soft_cap"),
        "quality_band_max_swap_ratio": float(raw.get("quality_band_max_swap_ratio", 0.08)),
        "quality_top_band_max_share": float(raw.get("quality_top_band_max_share", 0.08)),
        "enable_learnability_rebalance": bool(raw.get("enable_learnability_rebalance", False)),
        "learnability_rebalance_max_swap_ratio": float(raw.get("learnability_rebalance_max_swap_ratio", 0.0)),
        "learnability_rebalance_min_gain": float(raw.get("learnability_rebalance_min_gain", 0.08)),
        "learnability_rebalance_min_quality": float(raw.get("learnability_rebalance_min_quality", 0.80)),
        "learnability_rebalance_preserve_buckets": list(
            raw.get("learnability_rebalance_preserve_buckets") or ["domain", "style", "length"]
        ),
        "learning_signal_diagnostic_sample_size": int(raw.get("learning_signal_diagnostic_sample_size", 3000)),
        "matched_baseline_pool_multiplier": int(raw.get("matched_baseline_pool_multiplier", 4)),
    }


def _runtime_limits(profile: Dict[str, Any]) -> Dict[str, Any]:
    raw = profile.get("runtime_limits") or {}
    max_records = int(raw.get("max_records_per_dataset") or 0)
    return {
        "max_records_per_dataset": max(0, max_records),
        "disable_progress": bool(raw.get("disable_progress", False)),
    }


def _apply_mode_overrides(raw: Dict[str, Any], evaluation_mode: str) -> Dict[str, Any]:
    merged = dict(raw)
    mode_key = f"{evaluation_mode}_overrides"
    overrides = raw.get(mode_key)
    if isinstance(overrides, dict):
        merged.update(overrides)
    return merged


def _stage_c_validation(profile: Dict[str, Any]) -> Dict[str, Any]:
    raw = profile.get("stage_c_validation") or {}
    certification_overrides = raw.get("certification_overrides") if isinstance(raw.get("certification_overrides"), dict) else {}
    evaluation_mode = str(raw.get("evaluation_mode") or "").strip().lower()
    if evaluation_mode not in {"development", "certification"}:
        legacy_dual_eval = (
            bool(raw.get("enforce_ood_utility_pass"))
            or bool(raw.get("require_dual_eval_pass"))
        )
        evaluation_mode = "certification" if legacy_dual_eval else "development"
    raw = _apply_mode_overrides(raw, evaluation_mode)

    if evaluation_mode == "certification":
        enforce_ood = True
        compute_ood = True
        enforce_coverage_backbone = True
    else:
        enforce_ood = False
        compute_ood = True
        enforce_coverage_backbone = False
    pass_statistic = str(raw.get("utility_pass_statistic") or ("min" if evaluation_mode == "certification" else "mean")).strip().lower()
    if pass_statistic not in {"mean", "min"}:
        pass_statistic = "min" if evaluation_mode == "certification" else "mean"
    certification_scope = str(
        raw.get("certification_scope")
        or ("general_purpose" if evaluation_mode == "certification" else "domain_specific")
    ).strip().lower()
    if certification_scope not in {"domain_specific", "general_purpose"}:
        certification_scope = "general_purpose" if evaluation_mode == "certification" else "domain_specific"

    certification_requirements = {
        "utility_pass_statistic": "min",
        "min_small_lm_probe_gain_score": float(
            certification_overrides.get(
                "min_small_lm_probe_gain_score",
                raw.get("min_small_lm_probe_gain_score", raw.get("min_fixed_token_probe_gain_score", 0.0)),
            )
        ),
        "min_small_lm_probe_relative_gain": float(
            certification_overrides.get(
                "min_small_lm_probe_relative_gain",
                raw.get("min_small_lm_probe_relative_gain", raw.get("min_fixed_token_probe_relative_gain", 0.0)),
            )
        ),
        "require_utility_ci_gain_positive": bool(certification_overrides.get("require_utility_ci_gain_positive", True)),
        "require_utility_delta_nll_positive": bool(certification_overrides.get("require_utility_delta_nll_positive", True)),
        "enforce_ood_utility_pass": bool(certification_overrides.get("enforce_ood_utility_pass", True)),
        "delta_nll_numerical_tolerance": float(certification_overrides.get("delta_nll_numerical_tolerance", 1e-5)),
    }

    # Explicit flags remain supported, but evaluation_mode is authoritative.
    # If explicit values disagree with mode, prefer mode semantics.
    return {
        "evaluation_mode": evaluation_mode,
        "certification_scope": certification_scope,
        "utility_metric": str(raw.get("utility_metric") or "small_lm_probe_gain_score"),
        "utility_pass_statistic": pass_statistic,
        "min_subset_coverage_retention_score": float(raw.get("min_subset_coverage_retention_score", 0.0)),
        "min_rare_cluster_retention": float(raw.get("min_rare_cluster_retention", 0.0)),
        "min_rare_cluster_retained_count": int(raw.get("min_rare_cluster_retained_count", 0)),
        "enforce_domain_coverage_support": bool(raw.get("enforce_domain_coverage_support", True)),
        "min_domain_coverage_distribution_similarity": float(raw.get("min_domain_coverage_distribution_similarity", 0.9)),
        "min_domain_coverage_retained_bucket_ratio": float(raw.get("min_domain_coverage_retained_bucket_ratio", 0.95)),
        "enforce_style_coverage_support": bool(raw.get("enforce_style_coverage_support", True)),
        "min_style_coverage_distribution_similarity": float(raw.get("min_style_coverage_distribution_similarity", 0.9)),
        "min_style_coverage_retained_bucket_ratio": float(raw.get("min_style_coverage_retained_bucket_ratio", 0.95)),
        "min_small_lm_probe_gain_score": float(raw.get("min_small_lm_probe_gain_score", raw.get("min_fixed_token_probe_gain_score", 0.0))),
        "min_small_lm_probe_relative_gain": float(raw.get("min_small_lm_probe_relative_gain", raw.get("min_fixed_token_probe_relative_gain", 0.0))),
        "require_utility_ci_gain_positive": bool(raw.get("require_utility_ci_gain_positive", False)),
        "require_utility_delta_nll_positive": bool(raw.get("require_utility_delta_nll_positive", False)),
        "enforce_ood_utility_pass": bool(enforce_ood),
        "compute_ood_utility_report": bool(compute_ood),
        "enforce_coverage_backbone_pass": bool(raw.get("enforce_coverage_backbone_pass", enforce_coverage_backbone)),
        "certification_requirements": certification_requirements,
        # Backward-compatible alias for downstream readers.
        "require_dual_eval_pass": bool(enforce_ood),
    }


def _utility_probe_config(profile: Dict[str, Any], *, evaluation_mode: str = "development") -> Dict[str, Any]:
    base_raw = profile.get("utility_probe") or {}
    certification_overrides = (
        base_raw.get("certification_overrides") if isinstance(base_raw.get("certification_overrides"), dict) else {}
    )
    raw = _apply_mode_overrides(base_raw, str(evaluation_mode or "development"))
    mode = str(raw.get("mode") or "full").strip().lower()
    if mode not in {"fast", "full", "synthetic_smoke"}:
        mode = "full"
    dual_eval_required = bool(raw.get("dual_eval_required", True))
    ood_dataset = raw.get("ood_eval_dataset")
    holdout_buckets = raw.get("holdout_buckets")
    if holdout_buckets is None:
        holdout_buckets = [int(raw.get("holdout_bucket", 0))]
    holdout_modulo = int(raw.get("holdout_modulo", 17))
    holdout_buckets = sorted({int(v) for v in holdout_buckets})
    holdout_buckets = [b for b in holdout_buckets if 0 <= b < holdout_modulo]
    if not holdout_buckets:
        holdout_buckets = [0]
    ood_holdout_buckets = raw.get("ood_holdout_buckets")
    if ood_holdout_buckets is None:
        ood_holdout_buckets = holdout_buckets
    ood_holdout_buckets = sorted({int(v) for v in ood_holdout_buckets})
    ood_holdout_buckets = [b for b in ood_holdout_buckets if 0 <= b < holdout_modulo]
    if not ood_holdout_buckets:
        ood_holdout_buckets = list(holdout_buckets)
    is_fast_like_mode = mode in {"fast", "synthetic_smoke"}
    default_seeds = [17, 29] if is_fast_like_mode else [17, 29, 41, 53]
    user_seeds = raw.get("seeds", raw.get("bootstrap_seeds"))
    if not isinstance(user_seeds, list) or not user_seeds:
        user_seeds = default_seeds
    seeds = sorted({int(x) for x in user_seeds})
    if not seeds:
        seeds = default_seeds
    train_budget_default = 32_000 if is_fast_like_mode else 96_000
    eval_budget_default = 12_000 if is_fast_like_mode else 24_000
    bootstrap_default = 80 if is_fast_like_mode else 120
    max_train_steps_default = 192 if is_fast_like_mode else 448
    certification_max_train_steps_default = 384 if is_fast_like_mode else 896
    train_epochs_default = 1.5 if is_fast_like_mode else 2.0
    certification_train_epochs_default = 2.0 if is_fast_like_mode else 3.0
    certification_seeds = certification_overrides.get("seeds", raw.get("seeds", raw.get("bootstrap_seeds", seeds)))
    if not isinstance(certification_seeds, list) or not certification_seeds:
        certification_seeds = seeds
    certification_requirements = {
        "train_token_budget": int(
            certification_overrides.get("train_token_budget", certification_overrides.get("fixed_token_budget", raw.get("train_token_budget", raw.get("fixed_token_budget", train_budget_default))))
        ),
        "eval_token_budget": int(certification_overrides.get("eval_token_budget", raw.get("eval_token_budget", eval_budget_default))),
        "ood_eval_token_budget": int(
            certification_overrides.get(
                "ood_eval_token_budget",
                certification_overrides.get("eval_token_budget", raw.get("ood_eval_token_budget", raw.get("eval_token_budget", eval_budget_default))),
            )
        ),
        "bootstrap_samples": int(
            certification_overrides.get("bootstrap_samples", certification_overrides.get("bootstrap_rounds", raw.get("bootstrap_samples", raw.get("bootstrap_rounds", bootstrap_default))))
        ),
        "seed_count": int(len({int(x) for x in certification_seeds})),
        "min_probe_bucket_count": int(certification_overrides.get("min_probe_bucket_count", raw.get("min_probe_bucket_count", 1))),
        "max_train_steps": int(
            certification_overrides.get("max_train_steps", raw.get("max_train_steps", certification_max_train_steps_default))
        ),
        "train_epochs": max(
            1.0,
            float(certification_overrides.get("train_epochs", raw.get("train_epochs", certification_train_epochs_default))),
        ),
    }
    return {
        "evaluation_mode": str(evaluation_mode or "development"),
        "mode": mode,
        "dual_eval_required": dual_eval_required,
        "model_name": str(raw.get("model_name") or "sshleifer/tiny-gpt2"),
        "train_token_budget": int(raw.get("train_token_budget", raw.get("fixed_token_budget", train_budget_default))),
        "eval_token_budget": int(raw.get("eval_token_budget", eval_budget_default)),
        "ood_eval_token_budget": int(raw.get("ood_eval_token_budget", raw.get("eval_token_budget", eval_budget_default))),
        "bootstrap_samples": int(raw.get("bootstrap_samples", raw.get("bootstrap_rounds", bootstrap_default))),
        "seeds": seeds,
        "holdout_modulo": holdout_modulo,
        "holdout_buckets": holdout_buckets,
        "ood_holdout_buckets": ood_holdout_buckets,
        "min_probe_bucket_count": int(raw.get("min_probe_bucket_count", 1)),
        "baseline_sampling_ratio": float(raw.get("baseline_sampling_ratio", 1.0)),
        "max_length": int(raw.get("max_length", 128)),
        "train_batch_size": int(raw.get("train_batch_size", 4)),
        "eval_batch_size": int(raw.get("eval_batch_size", 4)),
        "train_audit_token_budget": int(raw.get("train_audit_token_budget", 4096 if not is_fast_like_mode else 1024)),
        "learning_rate": float(raw.get("learning_rate", 5e-5)),
        "max_train_steps": int(raw.get("max_train_steps", max_train_steps_default)),
        "train_epochs": max(1.0, float(raw.get("train_epochs", train_epochs_default))),
        "seed": int(raw.get("seed", 42)),
        "sampling_hash_seed": int(raw.get("sampling_hash_seed", 42)),
        "ood_eval_dataset": str(ood_dataset) if ood_dataset is not None else None,
        "certification_requirements": certification_requirements,
    }


def _axis_scores(record: Dict[str, Any]) -> Dict[str, float]:
    metrics = record["core_metrics"]
    diagnostic = record.get("diagnostic_metrics") or {}
    selection_value = float(metrics["reference_quality_score"]["score"])
    near_risk = float(metrics["shingle_near_duplicate_risk_score"]["score"])
    redundancy = max(0.0, min(1.0, 1.0 - near_risk))
    return {
        "validity_gate": float(metrics["structural_validity_gate"]["score"]),
        "selection_value": selection_value,
        "quality": selection_value,
        "redundancy": float(redundancy),
        "redundancy_risk": near_risk,
        "experimental_quality": float((diagnostic.get("explanatory_quality_proxy") or {}).get("score") or 0.0),
        "experimental_tail_rarity": float((diagnostic.get("tail_cluster_rarity_proxy") or {}).get("score") or 0.0),
    }


def _objective_components(record: Dict[str, Any]) -> Dict[str, float]:
    axes = _axis_scores(record)
    quality_details = (((record.get("core_metrics") or {}).get("reference_quality_score") or {}).get("details") or {})
    redundancy_details = (((record.get("core_metrics") or {}).get("shingle_near_duplicate_risk_score") or {}).get("details") or {})
    word_count = int(record.get("word_count") or quality_details.get("word_count") or 0)
    if word_count <= 16:
        useful_length_support = 0.0
    elif word_count <= 64:
        useful_length_support = (float(word_count) - 16.0) / 48.0
    elif word_count <= 384:
        useful_length_support = 1.0
    elif word_count <= 768:
        useful_length_support = 1.0 - (0.25 * ((float(word_count) - 384.0) / 384.0))
    else:
        useful_length_support = 0.65
    lexical_diversity = max(0.0, min(1.0, float(quality_details.get("lexical_diversity") or 0.0)))
    boilerplate_penalty = max(0.0, min(1.0, float(quality_details.get("boilerplate_hits") or 0.0) / 2.0))
    useful_recurrence = max(0.0, min(1.0, float(redundancy_details.get("useful_recurrence_score") or 0.0)))
    repeat_pressure = max(0.0, min(1.0, float(redundancy_details.get("intra_chunk_repeat_pressure") or 0.0)))
    lexical_balance_support = max(0.0, 1.0 - (abs(float(lexical_diversity) - 0.42) / 0.42))
    pattern_recurrence_support = max(0.0, 1.0 - (abs(float(repeat_pressure) - 0.32) / 0.32))
    selection_value = float(axes["selection_value"])
    if selection_value < 0.65:
        selection_value_learnability_support = 0.0
    elif selection_value < 0.84:
        selection_value_learnability_support = (selection_value - 0.65) / 0.19
    elif selection_value <= 0.94:
        selection_value_learnability_support = 1.0
    else:
        selection_value_learnability_support = max(0.35, 1.0 - ((selection_value - 0.94) / 0.08) * 0.65)
    selection_value_tail_penalty = max(0.0, min(1.0, (selection_value - 0.97) / 0.03))
    learnability_support = max(
        0.0,
        min(
            1.0,
            (0.35 * selection_value_learnability_support)
            + (0.25 * pattern_recurrence_support)
            + (0.20 * useful_recurrence)
            + (0.10 * lexical_balance_support)
            + (0.10 * useful_length_support),
        ),
    )
    return {
        "selection_value": selection_value,
        "quality": selection_value,
        "redundancy_risk": float(axes["redundancy_risk"]),
        "useful_length_support": float(max(0.0, min(1.0, useful_length_support))),
        "lexical_diversity": float(lexical_diversity),
        "useful_recurrence": float(useful_recurrence),
        "repeat_pressure": float(repeat_pressure),
        "lexical_balance_support": float(lexical_balance_support),
        "pattern_recurrence_support": float(pattern_recurrence_support),
        "selection_value_learnability_support": float(selection_value_learnability_support),
        "quality_learnability_support": float(selection_value_learnability_support),
        "selection_value_tail_penalty": float(selection_value_tail_penalty),
        "quality_tail_penalty": float(selection_value_tail_penalty),
        "learnability_support": float(learnability_support),
        "boilerplate_penalty": float(boilerplate_penalty),
    }


def _structured_text_relief(record: Dict[str, Any]) -> Dict[str, float]:
    # Stage-B distribution preservation and Stage-C coverage validation must
    # use the same full-chunk style taxonomy. A truncated preview can silently
    # overwrite the scored style and create false coverage support.
    style_bucket = _style_bucket_from_scored_record(record)
    # Structured-text false-positive mitigation now lives in the metric-level
    # near-duplicate risk scorer. Keep the selector payload stable, but do not
    # apply an additional policy-layer relief on top of the calibrated metric.
    relief = 0.0
    return {
        "style_bucket": style_bucket,
        "redundancy_risk_relief": float(relief),
    }


def _passes_gates(record: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    metrics = record["core_metrics"]
    gate = _stage_a_gate(profile)
    for metric_name, threshold in gate["metric_floors"].items():
        if float(metrics[metric_name]["score"]) < float(threshold):
            return False
    for metric_name, threshold in gate["metric_ceilings"].items():
        if float(metrics[metric_name]["score"]) > float(threshold):
            return False
    return True


def _passes_stage_a_validity_exactdup(record: Dict[str, Any], stage_a_gate: Dict[str, Dict[str, float]]) -> bool:
    metrics = record["core_metrics"]
    validity_floor = float(stage_a_gate["metric_floors"].get("structural_validity_gate", 0.0))
    exact_dup_ceiling = float(stage_a_gate["metric_ceilings"].get("exact_duplicate_indicator", 0.0))
    return (
        float(metrics["structural_validity_gate"]["score"]) >= validity_floor
        and float(metrics["exact_duplicate_indicator"]["score"]) <= exact_dup_ceiling
    )


def _resolve_stage_b_config(stage_b_like: Dict[str, Any]) -> Dict[str, Any]:
    # Backward-compatible: callers may pass a full profile or an already-resolved stage_b dict.
    if (
        "weights" in stage_b_like
        and "selection_threshold" in stage_b_like
        and "near_duplicate_risk_ceiling" in stage_b_like
    ):
        return stage_b_like
    return _stage_b_rank(stage_b_like)


def _selection_score(record: Dict[str, Any], stage_b_like: Dict[str, Any]) -> float:
    comps = _objective_components(record)
    stage_b_rank = _resolve_stage_b_config(stage_b_like)
    relief = _structured_text_relief(record)
    effective_risk = max(0.0, comps["redundancy_risk"] - float(relief["redundancy_risk_relief"]))
    # Backward-compatible score for analytics and benchmarks.
    q_w = float(stage_b_rank["weights"].get("quality", 0.6))
    red_w = float(stage_b_rank["weights"].get("redundancy", 0.2))
    score = (q_w * comps["quality"]) + (red_w * (1.0 - effective_risk))
    return round(float(score), 6)


def _passes_stage_b(record: Dict[str, Any], stage_b_like: Dict[str, Any], rank_score: float) -> bool:
    metrics = record["core_metrics"]
    stage_b_rank = _resolve_stage_b_config(stage_b_like)
    relief = _structured_text_relief(record)
    effective_risk = max(
        0.0,
        float(metrics["shingle_near_duplicate_risk_score"]["score"]) - float(relief["redundancy_risk_relief"]),
    )
    if effective_risk > float(stage_b_rank["near_duplicate_risk_ceiling"]):
        return False
    return float(rank_score) >= float(stage_b_rank["selection_threshold"])


def _dataset_cluster_counts(conn: sqlite3.Connection) -> Dict[str, Counter[int]]:
    out: Dict[str, Counter[int]] = defaultdict(Counter)
    for dataset, cluster_id, count in conn.execute(
        "SELECT dataset, cluster_id, COUNT(*) FROM chunks GROUP BY dataset, cluster_id"
    ):
        out[str(dataset)][int(cluster_id)] = int(count)
    return out


def _cluster_id(record: Dict[str, Any]) -> int:
    return int(record["diagnostics"]["cluster_id"])


def _rare_cluster_cutoff(original_clusters: Counter[int], quantile: float) -> int:
    if not original_clusters:
        return 0
    sizes = sorted(int(v) for v in original_clusters.values())
    idx = max(0, min(len(sizes) - 1, int(len(sizes) * quantile) - 1))
    return int(sizes[idx])


def _coverage_strategy(profile: Dict[str, Any], original_clusters: Counter[int]) -> Dict[str, Any]:
    raw = dict(profile.get("coverage_strategy") or {})
    enabled = bool(raw.get("ensure_rare_cluster_exemplars"))
    quantile = float(raw.get("rare_cluster_quantile") or 0.25)
    cutoff = _rare_cluster_cutoff(original_clusters, quantile) if enabled else 0
    rare_clusters = {
        int(cluster_id)
        for cluster_id, size in original_clusters.items()
        if enabled and int(size) <= cutoff
    }
    return {
        "enabled": enabled,
        "rare_cluster_quantile": quantile,
        "rare_cluster_cutoff": cutoff,
        "rare_clusters": rare_clusters,
        "rare_exemplar_min_validity": float(raw.get("rare_exemplar_min_validity") or 0.0),
        "rare_exemplar_min_reference_quality": float(raw.get("rare_exemplar_min_reference_quality") or 0.0),
        "rare_exemplar_max_exact_duplicate_indicator": float(raw.get("rare_exemplar_max_exact_duplicate_indicator") or 0.0),
        "rare_exemplar_relaxed_near_dup_ceiling": float(raw.get("rare_exemplar_relaxed_near_dup_ceiling") or 1.0),
    }


def _passes_rare_exemplar_filters(record: Dict[str, Any], strategy: Dict[str, Any]) -> bool:
    if not strategy.get("enabled"):
        return False
    metrics = record["core_metrics"]
    return (
        metrics["structural_validity_gate"]["score"] >= strategy["rare_exemplar_min_validity"]
        and metrics["reference_quality_score"]["score"] >= strategy["rare_exemplar_min_reference_quality"]
        and metrics["exact_duplicate_indicator"]["score"] <= strategy["rare_exemplar_max_exact_duplicate_indicator"]
        and metrics["shingle_near_duplicate_indicator"]["score"] <= strategy["rare_exemplar_relaxed_near_dup_ceiling"]
    )


def _domain_bucket_from_row(metadata_json: Any, source: Any, input_source: Any) -> str:
    metadata: Dict[str, Any] = {}
    if metadata_json:
        try:
            loaded = json.loads(str(metadata_json))
            if isinstance(loaded, dict):
                metadata = loaded
        except json.JSONDecodeError:
            metadata = {}

    for key in ("domain", "source_domain", "site", "host"):
        value = str(metadata.get(key) or "").strip().lower()
        if value:
            return value

    for key in ("url", "source_url", "page_url"):
        value = str(metadata.get(key) or "").strip()
        if value:
            sanitized = value.lower().split("://", 1)[-1].split("/", 1)[0]
            sanitized = sanitized.split("?", 1)[0].split("#", 1)[0]
            if sanitized:
                return sanitized

    source_value = str(source or "").strip()
    if source_value:
        source_name = Path(source_value).name.strip().lower()
        if source_name:
            return source_name

    input_source_value = str(input_source or "").strip()
    if input_source_value:
        input_name = Path(input_source_value).name.strip().lower()
        if input_name:
            return input_name

    return "unknown"


def _dataset_domain_counts(conn: sqlite3.Connection, *, dataset: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    rows = conn.execute(
        "SELECT metadata_json, source, input_source FROM chunks WHERE dataset = ?",
        (str(dataset),),
    ).fetchall()
    for metadata_json, source, input_source in rows:
        counts[_domain_bucket_from_row(metadata_json, source, input_source)] += 1
    return counts


def _selected_domain_counts(
    conn: sqlite3.Connection,
    *,
    chunk_uids: List[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not chunk_uids:
        return counts
    batch_size = 800
    for i in range(0, len(chunk_uids), batch_size):
        batch = chunk_uids[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT metadata_json, source, input_source FROM chunks WHERE chunk_uid IN ({placeholders})",
            batch,
        ).fetchall()
        for metadata_json, source, input_source in rows:
            counts[_domain_bucket_from_row(metadata_json, source, input_source)] += 1
    return counts


def _selected_domain_counts_from_scored_records(selected_records: List[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in selected_records:
        counts[_domain_bucket_from_scored_record(record)] += 1
    return counts


def _distribution_bucket_support(
    selected_counts: Counter[str],
    original_counts: Counter[str],
    *,
    support_scope: str = "distribution_bucket",
    support_label: str = "bucket",
) -> Dict[str, Any]:
    if not original_counts:
        return {
            "support_scope": support_scope,
            "support_label": support_label,
            "distribution_similarity": 0.0,
            "source_bucket_count": 0,
            "selected_bucket_count": 0,
            "retained_bucket_count": 0,
            "retained_bucket_ratio": 0.0,
            "dominant_source_bucket": None,
            "dominant_selected_bucket": None,
        }

    total_orig = sum(original_counts.values())
    total_sel = sum(selected_counts.values())
    if total_sel == 0:
        return {
            "support_scope": support_scope,
            "support_label": support_label,
            "distribution_similarity": 0.0,
            "source_bucket_count": int(len(original_counts)),
            "selected_bucket_count": 0,
            "retained_bucket_count": 0,
            "retained_bucket_ratio": 0.0,
            "dominant_source_bucket": max(original_counts, key=original_counts.get),
            "dominant_selected_bucket": None,
        }

    orig_p = {k: v / total_orig for k, v in original_counts.items()}
    sel_p = {k: v / total_sel for k, v in selected_counts.items()}
    keys = set(orig_p) | set(sel_p)
    tvd = 0.5 * sum(abs(orig_p.get(k, 0.0) - sel_p.get(k, 0.0)) for k in keys)
    retained_bucket_count = sum(1 for key in original_counts if selected_counts.get(key, 0) > 0)
    retained_bucket_ratio = retained_bucket_count / max(len(original_counts), 1)
    return {
        "support_scope": support_scope,
        "support_label": support_label,
        "distribution_similarity": round(max(0.0, 1.0 - tvd), 6),
        "source_bucket_count": int(len(original_counts)),
        "selected_bucket_count": int(len(selected_counts)),
        "retained_bucket_count": int(retained_bucket_count),
        "retained_bucket_ratio": round(float(retained_bucket_ratio), 6),
        "dominant_source_bucket": max(original_counts, key=original_counts.get),
        "dominant_selected_bucket": max(selected_counts, key=selected_counts.get) if selected_counts else None,
    }


def _source_bucket_support_scope(counts: Counter[str]) -> str:
    buckets = [str(bucket) for bucket in counts]
    if not buckets:
        return "unavailable"
    fallback_like = 0
    for bucket in buckets:
        if bucket == "unknown" or bucket.endswith(".json") or bucket.startswith("batch_"):
            fallback_like += 1
    if fallback_like == len(buckets):
        return "source_bucket_fallback"
    if fallback_like > 0:
        return "mixed_domain_and_source_bucket"
    return "explicit_domain_metadata"


def _semantic_coverage_support(coverage: Dict[str, Any], cluster_backbone_audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "support_scope": "semantic_cluster_backbone",
        "support_label": "semantic_cluster",
        "distribution_similarity": coverage.get("distribution_similarity", 0.0),
        "rare_cluster_retention": coverage.get("rare_cluster_retention", 0.0),
        "rare_cluster_count": coverage.get("rare_cluster_count", 0),
        "rare_cluster_retained_count": coverage.get("rare_cluster_retained_count", 0),
        "cluster_backbone_pass": bool(cluster_backbone_audit.get("passed")),
        "cluster_backbone_readiness": str(cluster_backbone_audit.get("readiness") or ""),
        "coherence_proxy": cluster_backbone_audit.get("coherence_proxy"),
        "separation_margin": cluster_backbone_audit.get("separation_margin"),
        "style_purity_proxy": cluster_backbone_audit.get("style_purity_proxy"),
        "domain_purity_proxy": cluster_backbone_audit.get("domain_purity_proxy"),
    }


def _bucket_support_pass(
    support: Dict[str, Any],
    *,
    min_distribution_similarity: float,
    min_retained_bucket_ratio: float,
) -> bool:
    return bool(
        float(support.get("distribution_similarity") or 0.0) >= float(min_distribution_similarity)
        and float(support.get("retained_bucket_ratio") or 0.0) >= float(min_retained_bucket_ratio)
    )


def _style_bucket_from_text(text: Any) -> str:
    return style_bucket_from_text(str(text or ""))


def _style_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    selection = record.get("selection") or {}
    if selection.get("style_bucket"):
        return str(selection.get("style_bucket"))
    for metric_group in ("core_metrics", "diagnostic_metrics"):
        details = (((record.get(metric_group) or {}).get("structural_validity_gate") or {}).get("details") or {})
        if details.get("style_bucket"):
            return str(details.get("style_bucket"))
        details = (((record.get(metric_group) or {}).get("structural_validity_score") or {}).get("details") or {})
        if details.get("style_bucket"):
            return str(details.get("style_bucket"))
    return _style_bucket_from_text(record.get("text") or "")


def _length_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    word_count = int(record.get("word_count") or 0)
    if word_count <= 48:
        return "len_000_048"
    if word_count <= 64:
        return "len_049_064"
    if word_count <= 96:
        return "len_065_096"
    if word_count <= 128:
        return "len_097_128"
    if word_count <= 192:
        return "len_129_192"
    if word_count <= 256:
        return "len_193_256"
    if word_count <= 384:
        return "len_257_384"
    if word_count <= 512:
        return "len_385_512"
    return "len_513_plus"


def _quality_score_from_scored_record(record: Dict[str, Any]) -> float:
    payload = (record.get("core_metrics") or {}).get("reference_quality_score") or {}
    try:
        return float(payload.get("score"))
    except (TypeError, ValueError):
        return 0.0


def _quality_band_from_scored_record(record: Dict[str, Any]) -> str:
    score = _quality_score_from_scored_record(record)
    if score < 0.65:
        return "quality_lt_0_65"
    if score < 0.80:
        return "quality_0_65_0_80"
    if score < 0.90:
        return "quality_0_80_0_90"
    if score < 0.95:
        return "quality_0_90_0_95"
    if score < 0.99:
        return "quality_0_95_0_99"
    return "quality_ge_0_99"


def _domain_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    provenance = record.get("provenance") or {}
    metadata = provenance.get("metadata") if isinstance(provenance, dict) else {}
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    input_source = provenance.get("input_source") if isinstance(provenance, dict) else None
    return _domain_bucket_from_row(metadata_json, record.get("source"), input_source)


def _multi_matched_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    return "|".join(
        (
            f"quality={_quality_band_from_scored_record(record)}",
            f"length={_length_bucket_from_scored_record(record)}",
            f"style={_style_bucket_from_scored_record(record)}",
            f"domain={_domain_bucket_from_scored_record(record)}",
        )
    )


def _repeat_pressure_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    redundancy_details = (((record.get("core_metrics") or {}).get("shingle_near_duplicate_risk_score") or {}).get("details") or {})
    repeat = float(redundancy_details.get("intra_chunk_repeat_pressure") or 0.0)
    if repeat < 0.24:
        return "repeat_lt_0_24"
    if repeat < 0.36:
        return "repeat_0_24_0_36"
    if repeat < 0.52:
        return "repeat_0_36_0_52"
    if repeat < 0.70:
        return "repeat_0_52_0_70"
    return "repeat_ge_0_70"


def _anti_memorization_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    return "|".join(
        (
            _multi_matched_bucket_from_scored_record(record),
            f"repeat={_repeat_pressure_bucket_from_scored_record(record)}",
        )
    )


def _nuisance_matched_bucket_from_scored_record(record: Dict[str, Any]) -> str:
    return "|".join(
        (
            f"length={_length_bucket_from_scored_record(record)}",
            f"style={_style_bucket_from_scored_record(record)}",
            f"domain={_domain_bucket_from_scored_record(record)}",
            f"repeat={_repeat_pressure_bucket_from_scored_record(record)}",
        )
    )


def _dataset_style_counts(conn: sqlite3.Connection, *, dataset: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    rows = conn.execute(
        "SELECT text FROM chunks WHERE dataset = ?",
        (str(dataset),),
    ).fetchall()
    for (text,) in rows:
        counts[_style_bucket_from_text(text)] += 1
    return counts


def _selected_style_counts_from_text_map(
    selected_records: List[Dict[str, Any]],
    text_map: Dict[str, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in selected_records:
        counts[_style_bucket_from_text(text_map.get(str(record["chunk_uid"]), ""))] += 1
    return counts


def _style_taxonomy_alignment_diagnostic(
    stage_b_selected_style_counts: Counter[str],
    selected_style_counts: Counter[str],
    selector_diagnostics: Dict[str, Any] | None,
) -> Dict[str, Any]:
    iterations = list((selector_diagnostics or {}).get("iterations") or [])
    quota_diagnostics = (iterations[-1].get("quota_diagnostics") or {}) if iterations else {}
    quota_snapshot_counts = Counter(
        {
            str(bucket): int(count)
            for bucket, count in ((quota_diagnostics.get("style_distribution_balance") or {}).get("selected_bucket_counts_after") or {}).items()
        }
    )
    selector_counts = Counter({str(bucket): int(count) for bucket, count in stage_b_selected_style_counts.items()})
    stage_c_counts = Counter({str(bucket): int(count) for bucket, count in selected_style_counts.items()})
    return {
        "contract": "stage_b_selected_style_equals_stage_c_full_text_recount",
        "aligned": bool(selector_counts) and selector_counts == stage_c_counts,
        "selector_selected_bucket_counts": dict(sorted(selector_counts.items())),
        "stage_c_full_text_bucket_counts": dict(sorted(stage_c_counts.items())),
        "selector_quota_snapshot_bucket_counts": dict(sorted(quota_snapshot_counts.items())),
        "absolute_count_difference": int(
            sum(abs(selector_counts[bucket] - stage_c_counts[bucket]) for bucket in set(selector_counts) | set(stage_c_counts))
        ),
    }


def _preview_text_map_from_scored_records(selected_records: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for record in selected_records:
        provenance = record.get("provenance") or {}
        out[str(record["chunk_uid"])] = str(provenance.get("text_preview") or "")
    return out


def _coverage_retention(selected_clusters: Counter[int], original_clusters: Counter[int]) -> Dict[str, float]:
    if not original_clusters:
        return {
            "distribution_similarity": 0.0,
            "rare_cluster_retention": 0.0,
            "rare_cluster_count": 0,
            "rare_cluster_retained_count": 0,
            "score": 0.0,
        }
    total_orig = sum(original_clusters.values())
    total_sel = sum(selected_clusters.values())
    if total_sel == 0:
        return {
            "distribution_similarity": 0.0,
            "rare_cluster_retention": 0.0,
            "rare_cluster_count": 0,
            "rare_cluster_retained_count": 0,
            "score": 0.0,
        }

    orig_p = {k: v / total_orig for k, v in original_clusters.items()}
    sel_p = {k: v / total_sel for k, v in selected_clusters.items()}
    keys = set(orig_p) | set(sel_p)
    tvd = 0.5 * sum(abs(orig_p.get(k, 0.0) - sel_p.get(k, 0.0)) for k in keys)
    distribution_similarity = max(0.0, 1.0 - tvd)

    rare_cutoff = sorted(original_clusters.values())[max(0, len(original_clusters) // 4 - 1)] if original_clusters else 0
    rare_clusters = {k for k, v in original_clusters.items() if v <= rare_cutoff}
    if not rare_clusters:
        rare_retention = 1.0
        retained = 0
    else:
        retained = sum(1 for k in rare_clusters if selected_clusters.get(k, 0) > 0)
        rare_retention = retained / len(rare_clusters)

    # Coverage-preserving behavior is primarily about keeping rare/tail regions
    # alive after filtering; distribution similarity remains a secondary guardrail.
    score = 0.3 * distribution_similarity + 0.7 * rare_retention
    return {
        "distribution_similarity": round(distribution_similarity, 6),
        "rare_cluster_retention": round(rare_retention, 6),
        "rare_cluster_count": int(len(rare_clusters)),
        "rare_cluster_retained_count": int(retained),
        "score": round(score, 6),
    }


def _cluster_text_sets(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    cluster_id: int,
    limit: int = 3,
) -> List[set[str]]:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE dataset = ? AND cluster_id = ? ORDER BY chunk_uid LIMIT ?",
        (dataset, int(cluster_id), int(limit)),
    ).fetchall()
    out: List[set[str]] = []
    for (text,) in rows:
        tokens = {tok for tok in str(text).lower().split() if tok.isalpha() and len(tok) >= 4}
        if tokens:
            out.append(tokens)
    return out


def _cluster_sample_rows(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    cluster_id: int,
    limit: int = 3,
) -> List[Tuple[str, str, str, str]]:
    rows = conn.execute(
        "SELECT text, metadata_json, source, input_source FROM chunks WHERE dataset = ? AND cluster_id = ? ORDER BY chunk_uid LIMIT ?",
        (dataset, int(cluster_id), int(limit)),
    ).fetchall()
    return [
        (
            str(text or ""),
            str(metadata_json or ""),
            str(source or ""),
            str(input_source or ""),
        )
        for text, metadata_json, source, input_source in rows
    ]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def _ensure_coverage_backbone_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_dataset_uid ON chunks(dataset, chunk_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_dataset_cluster_uid ON chunks(dataset, cluster_id, chunk_uid)")
    conn.commit()


def _cluster_backbone_audit(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    original_clusters: Counter[int],
    seed: int,
    cluster_sample_limit: int = 24,
    rows_per_cluster: int = 8,
    between_pairs_per_cluster_pair: int = 4,
) -> Dict[str, Any]:
    _ensure_coverage_backbone_indexes(conn)
    if not original_clusters:
        return {
            "passed": False,
            "cluster_count": 0,
            "rare_cluster_count": 0,
            "coherence_proxy": 0.0,
            "separation_proxy": 0.0,
            "separation_margin": 0.0,
            "size_entropy": 0.0,
        }

    cluster_ids = sorted(original_clusters)
    rng = random.Random(seed)
    if len(cluster_ids) > cluster_sample_limit:
        rng.shuffle(cluster_ids)
        cluster_ids = sorted(cluster_ids[:cluster_sample_limit])

    within_scores: List[float] = []
    sampled_token_sets: Dict[int, List[set[str]]] = {}
    style_purity_scores: List[float] = []
    domain_purity_scores: List[float] = []
    for cluster_id in cluster_ids:
        token_sets = _cluster_text_sets(conn, dataset=dataset, cluster_id=cluster_id, limit=rows_per_cluster)
        sampled_token_sets[cluster_id] = token_sets
        if len(token_sets) >= 2:
            for i in range(len(token_sets)):
                for j in range(i + 1, len(token_sets)):
                    within_scores.append(_jaccard(token_sets[i], token_sets[j]))
        sample_rows = _cluster_sample_rows(conn, dataset=dataset, cluster_id=cluster_id, limit=rows_per_cluster)
        if sample_rows:
            style_counts = Counter(_style_bucket_from_text(text) for text, _, _, _ in sample_rows)
            domain_counts = Counter(
                _domain_bucket_from_row(metadata_json, source, input_source)
                for _, metadata_json, source, input_source in sample_rows
            )
            style_purity_scores.append(max(style_counts.values()) / max(sum(style_counts.values()), 1))
            domain_purity_scores.append(max(domain_counts.values()) / max(sum(domain_counts.values()), 1))

    between_scores: List[float] = []
    for i in range(len(cluster_ids)):
        left = sampled_token_sets.get(cluster_ids[i]) or []
        for j in range(i + 1, len(cluster_ids)):
            right = sampled_token_sets.get(cluster_ids[j]) or []
            pair_count = min(len(left), len(right), between_pairs_per_cluster_pair)
            for pair_idx in range(pair_count):
                between_scores.append(_jaccard(left[pair_idx], right[pair_idx]))

    comparison_count = min(len(within_scores), len(between_scores))
    comparison_rng = random.Random(seed + 1)
    matched_within = list(within_scores)
    matched_between = list(between_scores)
    comparison_rng.shuffle(matched_within)
    comparison_rng.shuffle(matched_between)
    within_gt_between_fraction = (
        float(
            sum(
                within > between
                for within, between in zip(
                    matched_within[:comparison_count],
                    matched_between[:comparison_count],
                )
            )
            / comparison_count
        )
        if comparison_count
        else 0.0
    )

    total = float(sum(original_clusters.values()))
    probs = [float(v) / total for v in original_clusters.values() if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    rare_cutoff = _rare_cluster_cutoff(original_clusters, 0.25)
    rare_cluster_count = sum(1 for size in original_clusters.values() if int(size) <= rare_cutoff)
    coherence_proxy = float(sum(within_scores) / len(within_scores)) if within_scores else 0.0
    separation_proxy = float(sum(between_scores) / len(between_scores)) if between_scores else 0.0
    separation_margin = coherence_proxy - separation_proxy
    style_purity_proxy = float(sum(style_purity_scores) / len(style_purity_scores)) if style_purity_scores else 0.0
    domain_purity_proxy = float(sum(domain_purity_scores) / len(domain_purity_scores)) if domain_purity_scores else 0.0
    structural_pass = bool(len(original_clusters) >= 8 and rare_cluster_count >= 2 and entropy >= 1.5)
    coherence_pass = bool(coherence_proxy >= 0.02)
    lexical_separation_pass = bool(separation_margin >= 0.005 and within_gt_between_fraction >= 0.55)
    # Anchor purity remains diagnostic-only: source/input-file buckets are not
    # guaranteed to be semantic domain labels and must not bypass lexical evidence.
    anchor_purity_pass = bool(domain_purity_proxy >= 0.85 and style_purity_proxy >= 0.65)
    passed = bool(structural_pass and coherence_pass and lexical_separation_pass)
    failure_reasons: List[str] = []
    if len(original_clusters) < 8:
        failure_reasons.append("cluster_count_lt_8")
    if rare_cluster_count < 2:
        failure_reasons.append("rare_cluster_count_lt_2")
    if entropy < 1.5:
        failure_reasons.append("size_entropy_lt_1_5")
    if coherence_proxy < 0.02:
        failure_reasons.append("coherence_proxy_lt_0_02")
    if not lexical_separation_pass:
        failure_reasons.append("pairwise_lexical_separation_failed")
    return {
        "passed": passed,
        "readiness": "certification_ready" if passed else "development_only",
        "failure_reasons": failure_reasons,
        "structural_pass": bool(structural_pass),
        "coherence_pass": bool(coherence_pass),
        "lexical_separation_pass": bool(lexical_separation_pass),
        "anchor_purity_pass": bool(anchor_purity_pass),
        "anchor_purity_role": "diagnostic_only",
        "cluster_count": int(len(original_clusters)),
        "rare_cluster_count": int(rare_cluster_count),
        "coherence_proxy": round(coherence_proxy, 6),
        "separation_proxy": round(separation_proxy, 6),
        "separation_margin": round(separation_margin, 6),
        "within_gt_between_fraction": round(within_gt_between_fraction, 6),
        "within_pair_count": int(len(within_scores)),
        "between_pair_count": int(len(between_scores)),
        "matched_comparison_count": int(comparison_count),
        "style_purity_proxy": round(style_purity_proxy, 6),
        "domain_purity_proxy": round(domain_purity_proxy, 6),
        "size_entropy": round(entropy, 6),
        "sampled_cluster_count": int(len(cluster_ids)),
        "rows_per_cluster": int(rows_per_cluster),
        "between_pairs_per_cluster_pair": int(between_pairs_per_cluster_pair),
    }


def _objective_score_with_constraints(
    *,
    record: Dict[str, Any],
    components: Dict[str, float],
    selector_cfg: Dict[str, Any],
    strategy: Dict[str, Any],
) -> float:
    weights = selector_cfg["objective_weights"]
    penalties = selector_cfg["constraint_penalties"]
    adjustments = selector_cfg.get("selection_adjustments") or {}
    cluster_size = int((record.get("diagnostics") or {}).get("cluster_size") or 1)
    cluster_id = _cluster_id(record)
    rare_bonus = penalties["rare_cluster_bonus"] if cluster_id in strategy["rare_clusters"] else 0.0
    # Encourage small clusters while preserving monotonicity.
    small_cluster_bonus = penalties["small_cluster_bonus"] * (1.0 / math.sqrt(max(cluster_size, 1)))
    relief = _structured_text_relief(record)
    effective_risk = max(0.0, components["redundancy_risk"] - float(relief["redundancy_risk_relief"]))
    selection_value_weight = (
        float(weights["selection_value"])
        if "selection_value" in weights
        else float(weights["quality"])
    )
    return float(
        (selection_value_weight * components["selection_value"])
        - (weights["redundancy_risk"] * effective_risk)
        + rare_bonus
        + small_cluster_bonus
        + (float(adjustments.get("useful_length_bonus") or 0.0) * components["useful_length_support"])
        + (float(adjustments.get("lexical_diversity_bonus") or 0.0) * components["lexical_diversity"])
        + (float(adjustments.get("useful_recurrence_bonus") or 0.0) * components["useful_recurrence"])
        + (float(adjustments.get("learnability_support_bonus") or 0.0) * components["learnability_support"])
        + (float(adjustments.get("pattern_recurrence_bonus") or 0.0) * components["pattern_recurrence_support"])
        - (float(adjustments.get("quality_tail_penalty") or 0.0) * components["quality_tail_penalty"])
        - (float(adjustments.get("boilerplate_penalty") or 0.0) * components["boilerplate_penalty"])
    )


def _target_selected_count(source_records: int, selector_cfg: Dict[str, Any]) -> int:
    min_selected_ratio = float(selector_cfg.get("min_selected_ratio") or 0.0)
    return max(0, int(math.ceil(float(source_records) * min_selected_ratio)))


def _top_up_bucket_exemplars(
    *,
    selected: List[Dict[str, Any]],
    eligible_records: List[Dict[str, Any]],
    bucket_name: str,
    bucket_fn: Any,
    min_count: int,
    accepted_by: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    min_count = max(0, int(min_count))
    if min_count <= 0:
        return selected, {
            "enabled": False,
            "bucket_name": bucket_name,
            "min_count": 0,
            "eligible_bucket_count": 0,
            "selected_bucket_count_before": 0,
            "selected_bucket_count_after": 0,
            "added_records": 0,
        }

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in eligible_records:
        by_bucket[str(bucket_fn(record) or "unknown")].append(record)
    for records in by_bucket.values():
        records.sort(
            key=lambda r: (
                -float((r.get("selection") or {}).get("stage_b_rank_score") or 0.0),
                str(r.get("chunk_uid") or ""),
            )
        )

    selected_uids = {str(r.get("chunk_uid") or "") for r in selected}
    selected_counts: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in selected)
    selected_bucket_count_before = sum(1 for bucket in by_bucket if selected_counts.get(bucket, 0) > 0)
    added_records = 0
    for bucket in sorted(by_bucket):
        target = min(min_count, len(by_bucket[bucket]))
        deficit = max(0, target - int(selected_counts.get(bucket, 0)))
        if deficit <= 0:
            continue
        for record in by_bucket[bucket]:
            if deficit <= 0:
                break
            uid = str(record.get("chunk_uid") or "")
            if uid in selected_uids:
                continue
            record["selection"]["accepted"] = True
            record["selection"]["accepted_by"] = accepted_by
            selected.append(record)
            selected_uids.add(uid)
            selected_counts[bucket] += 1
            added_records += 1
            deficit -= 1

    selected_bucket_count_after = sum(1 for bucket in by_bucket if selected_counts.get(bucket, 0) > 0)
    return selected, {
        "enabled": True,
        "bucket_name": bucket_name,
        "min_count": int(min_count),
        "eligible_bucket_count": int(len(by_bucket)),
        "selected_bucket_count_before": int(selected_bucket_count_before),
        "selected_bucket_count_after": int(selected_bucket_count_after),
        "added_records": int(added_records),
    }


def _bucket_target_counts(
    *,
    eligible_records: List[Dict[str, Any]],
    bucket_fn: Any,
    target_count: int,
) -> Dict[str, int]:
    bucket_counts: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in eligible_records)
    total = int(sum(bucket_counts.values()))
    if total <= 0 or target_count <= 0:
        return {bucket: 0 for bucket in bucket_counts}
    targets: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    allocated = 0
    for bucket, count in bucket_counts.items():
        raw_target = (float(count) / float(total)) * float(target_count)
        floor_target = int(math.floor(raw_target))
        targets[bucket] = floor_target
        allocated += floor_target
        remainders.append((raw_target - float(floor_target), bucket))
    leftover = max(0, int(target_count) - int(allocated))
    for _, bucket in sorted(remainders, key=lambda item: (-item[0], item[1]))[:leftover]:
        targets[bucket] += 1
    return targets


def _distribution_similarity_from_counts(
    *,
    observed_counts: Counter[str],
    target_counts: Dict[str, int],
) -> float:
    buckets = sorted(set(observed_counts) | set(target_counts))
    observed_total = float(sum(observed_counts.values()))
    target_total = float(sum(target_counts.values()))
    if observed_total <= 0.0 or target_total <= 0.0:
        return 0.0
    tvd = 0.5 * sum(
        abs((float(observed_counts.get(bucket, 0)) / observed_total) - (float(target_counts.get(bucket, 0)) / target_total))
        for bucket in buckets
    )
    return max(0.0, 1.0 - tvd)


def _bucket_distribution_diagnostic(
    *,
    selected: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
    bucket_name: str,
    bucket_fn: Any,
) -> Dict[str, Any]:
    target_counts = _bucket_target_counts(
        eligible_records=reference_records,
        bucket_fn=bucket_fn,
        target_count=int(len(selected)),
    )
    selected_counts: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in selected)
    return {
        "enabled": False,
        "diagnostic_only": True,
        "bucket_name": bucket_name,
        "target_count": int(len(selected)),
        "swap_count": 0,
        "eligible_bucket_count": int(len(target_counts)),
        "reference_record_count": int(len(reference_records)),
        "distribution_similarity_before": round(
            _distribution_similarity_from_counts(
                observed_counts=selected_counts,
                target_counts=target_counts,
            ),
            6,
        ),
        "distribution_similarity_after": round(
            _distribution_similarity_from_counts(
                observed_counts=selected_counts,
                target_counts=target_counts,
            ),
            6,
        ),
        "target_bucket_counts": {str(bucket): int(count) for bucket, count in sorted(target_counts.items())},
        "selected_bucket_counts_after": {str(bucket): int(count) for bucket, count in sorted(selected_counts.items())},
    }


def _rebalance_bucket_distribution(
    *,
    selected: List[Dict[str, Any]],
    eligible_records: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]] | None,
    bucket_name: str,
    bucket_fn: Any,
    target_count: int,
    enabled: bool,
    protected_acceptors: set[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not enabled:
        return selected, {
            "enabled": False,
            "bucket_name": bucket_name,
            "target_count": 0,
            "swap_count": 0,
            "distribution_similarity_before": 0.0,
            "distribution_similarity_after": 0.0,
            "eligible_bucket_count": 0,
        }
    protected_acceptors = protected_acceptors or set()
    selected_map: Dict[str, Dict[str, Any]] = {str(record.get("chunk_uid") or ""): record for record in selected}
    selected_uids = set(selected_map.keys())
    target_source = reference_records if reference_records is not None else eligible_records
    target_counts = _bucket_target_counts(
        eligible_records=target_source,
        bucket_fn=bucket_fn,
        target_count=int(target_count),
    )
    if not target_counts:
        return selected, {
            "enabled": True,
            "bucket_name": bucket_name,
            "target_count": int(target_count),
            "swap_count": 0,
            "distribution_similarity_before": 0.0,
            "distribution_similarity_after": 0.0,
            "eligible_bucket_count": 0,
        }

    selected_counts_before: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in selected_map.values())
    removable_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    addition_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in selected_map.values():
        bucket = str(bucket_fn(record) or "unknown")
        accepted_by = str((record.get("selection") or {}).get("accepted_by") or "")
        if accepted_by in protected_acceptors:
            continue
        removable_by_bucket[bucket].append(record)
    for records in removable_by_bucket.values():
        records.sort(
            key=lambda r: (
                float((r.get("selection") or {}).get("stage_b_rank_score") or 0.0),
                str(r.get("chunk_uid") or ""),
            )
        )

    for record in eligible_records:
        uid = str(record.get("chunk_uid") or "")
        if uid in selected_uids:
            continue
        bucket = str(bucket_fn(record) or "unknown")
        addition_by_bucket[bucket].append(record)
    for records in addition_by_bucket.values():
        records.sort(
            key=lambda r: (
                -float((r.get("selection") or {}).get("stage_b_rank_score") or 0.0),
                str(r.get("chunk_uid") or ""),
            )
        )

    selected_counts = Counter(selected_counts_before)
    swap_count = 0
    while True:
        deficits = {
            bucket: max(0, int(target_counts.get(bucket, 0)) - int(selected_counts.get(bucket, 0)))
            for bucket in target_counts
        }
        donors = [
            bucket
            for bucket in set(selected_counts) | set(target_counts)
            if int(selected_counts.get(bucket, 0)) > int(target_counts.get(bucket, 0))
            and removable_by_bucket.get(bucket)
        ]
        recipients = [bucket for bucket, deficit in deficits.items() if deficit > 0 and addition_by_bucket.get(bucket)]
        if not donors or not recipients:
            break
        recipient_bucket = max(recipients, key=lambda bucket: (deficits[bucket], str(bucket)))
        donor_bucket = max(
            donors,
            key=lambda bucket: (
                int(selected_counts.get(bucket, 0)) - int(target_counts.get(bucket, 0)),
                str(bucket),
            ),
        )
        donor_record = removable_by_bucket[donor_bucket].pop(0)
        add_record = addition_by_bucket[recipient_bucket].pop(0)
        donor_uid = str(donor_record.get("chunk_uid") or "")
        add_uid = str(add_record.get("chunk_uid") or "")
        if donor_uid not in selected_map or add_uid in selected_map:
            continue
        add_record["selection"]["accepted"] = True
        add_record["selection"]["accepted_by"] = f"{bucket_name}_distribution_rebalance"
        selected_map.pop(donor_uid, None)
        selected_map[add_uid] = add_record
        selected_counts[donor_bucket] -= 1
        selected_counts[recipient_bucket] += 1
        swap_count += 1

    selected_after = list(selected_map.values())
    selected_counts_after: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in selected_after)
    return selected_after, {
        "enabled": True,
        "bucket_name": bucket_name,
        "target_count": int(target_count),
        "swap_count": int(swap_count),
        "eligible_bucket_count": int(len(target_counts)),
        "reference_record_count": int(len(target_source)),
        "distribution_similarity_before": round(
            _distribution_similarity_from_counts(
                observed_counts=selected_counts_before,
                target_counts=target_counts,
            ),
            6,
        ),
        "distribution_similarity_after": round(
            _distribution_similarity_from_counts(
                observed_counts=selected_counts_after,
                target_counts=target_counts,
            ),
            6,
        ),
        "target_bucket_counts": {str(bucket): int(count) for bucket, count in sorted(target_counts.items())},
        "selected_bucket_counts_before": {str(bucket): int(count) for bucket, count in sorted(selected_counts_before.items())},
        "selected_bucket_counts_after": {str(bucket): int(count) for bucket, count in sorted(selected_counts_after.items())},
    }


def _soft_cap_quality_band_distribution(
    *,
    selected: List[Dict[str, Any]],
    eligible_records: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
    target_count: int,
    min_quality: float,
    top_band_max_share: float,
    max_swap_ratio: float,
    protected_acceptors: set[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    protected_acceptors = protected_acceptors or set()
    target_count = int(target_count)
    selected_map: Dict[str, Dict[str, Any]] = {str(record.get("chunk_uid") or ""): record for record in selected}
    selected_uids = set(selected_map.keys())
    reference_records = [record for record in reference_records if _quality_score_from_scored_record(record) >= float(min_quality)]
    target_counts = _bucket_target_counts(
        eligible_records=reference_records,
        bucket_fn=_quality_band_from_scored_record,
        target_count=target_count,
    )
    selected_counts_before: Counter[str] = Counter(_quality_band_from_scored_record(record) for record in selected_map.values())
    top_band = "quality_ge_0_99"
    top_cap = max(0, int(math.ceil(float(target_count) * max(0.0, float(top_band_max_share)))))
    if target_counts:
        # Never force a larger top-tail share than the reference mixture already asks for.
        top_cap = min(top_cap, int(target_counts.get(top_band, 0)))
    max_swaps = max(0, int(math.floor(float(target_count) * max(0.0, float(max_swap_ratio)))))

    removable_top: List[Dict[str, Any]] = []
    for record in selected_map.values():
        if _quality_band_from_scored_record(record) != top_band:
            continue
        accepted_by = str((record.get("selection") or {}).get("accepted_by") or "")
        if accepted_by in protected_acceptors:
            continue
        removable_top.append(record)
    removable_top.sort(
        key=lambda r: (
            float((r.get("selection") or {}).get("stage_b_rank_score") or 0.0),
            str(r.get("chunk_uid") or ""),
        )
    )

    addition_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in eligible_records:
        uid = str(record.get("chunk_uid") or "")
        if uid in selected_uids:
            continue
        quality = _quality_score_from_scored_record(record)
        if quality < float(min_quality):
            continue
        bucket = _quality_band_from_scored_record(record)
        if bucket == top_band:
            continue
        addition_by_bucket[bucket].append(record)
    for bucket, records in addition_by_bucket.items():
        # Prefer the strongest records inside underrepresented non-top bands.
        records.sort(
            key=lambda r: (
                -float((r.get("selection") or {}).get("stage_b_rank_score") or 0.0),
                str(r.get("chunk_uid") or ""),
            )
        )

    selected_counts = Counter(selected_counts_before)
    swap_count = 0
    while (
        selected_counts.get(top_band, 0) > top_cap
        and removable_top
        and swap_count < max_swaps
    ):
        deficits = {
            bucket: max(0, int(target_counts.get(bucket, 0)) - int(selected_counts.get(bucket, 0)))
            for bucket in target_counts
            if bucket != top_band
        }
        recipients = [bucket for bucket, deficit in deficits.items() if deficit > 0 and addition_by_bucket.get(bucket)]
        if not recipients:
            break
        recipient_bucket = max(recipients, key=lambda bucket: (deficits[bucket], bucket))
        donor_record = removable_top.pop(0)
        add_record = addition_by_bucket[recipient_bucket].pop(0)
        donor_uid = str(donor_record.get("chunk_uid") or "")
        add_uid = str(add_record.get("chunk_uid") or "")
        if donor_uid not in selected_map or add_uid in selected_map:
            continue
        add_record["selection"]["accepted"] = True
        add_record["selection"]["accepted_by"] = "quality_band_soft_cap_rebalance"
        selected_map.pop(donor_uid, None)
        selected_map[add_uid] = add_record
        selected_counts[top_band] -= 1
        selected_counts[recipient_bucket] += 1
        selected_uids.discard(donor_uid)
        selected_uids.add(add_uid)
        swap_count += 1

    selected_after = list(selected_map.values())
    selected_counts_after: Counter[str] = Counter(_quality_band_from_scored_record(record) for record in selected_after)
    return selected_after, {
        "enabled": True,
        "bucket_name": "quality_band",
        "target_count": int(target_count),
        "rebalance_mode": "soft_cap",
        "swap_count": int(swap_count),
        "max_swaps": int(max_swaps),
        "top_band": top_band,
        "top_band_cap": int(top_cap),
        "top_band_count_before": int(selected_counts_before.get(top_band, 0)),
        "top_band_count_after": int(selected_counts_after.get(top_band, 0)),
        "top_band_share_before": round(float(selected_counts_before.get(top_band, 0)) / float(max(len(selected), 1)), 6),
        "top_band_share_after": round(float(selected_counts_after.get(top_band, 0)) / float(max(len(selected_after), 1)), 6),
        "eligible_bucket_count": int(len(target_counts)),
        "reference_record_count": int(len(reference_records)),
        "distribution_similarity_before": round(
            _distribution_similarity_from_counts(
                observed_counts=selected_counts_before,
                target_counts=target_counts,
            ),
            6,
        ),
        "distribution_similarity_after": round(
            _distribution_similarity_from_counts(
                observed_counts=selected_counts_after,
                target_counts=target_counts,
            ),
            6,
        ),
        "target_bucket_counts": {str(bucket): int(count) for bucket, count in sorted(target_counts.items())},
        "selected_bucket_counts_before": {str(bucket): int(count) for bucket, count in sorted(selected_counts_before.items())},
        "selected_bucket_counts_after": {str(bucket): int(count) for bucket, count in sorted(selected_counts_after.items())},
    }


def _learnability_score_from_record(record: Dict[str, Any]) -> float:
    components = (record.get("selection") or {}).get("objective_components") or {}
    if "learnability_support" not in components:
        try:
            components = _objective_components(record)
        except Exception:
            components = {}
    return max(0.0, min(1.0, float(components.get("learnability_support") or 0.0)))


def _learnability_rebalance_bucket(record: Dict[str, Any], bucket_fields: List[str]) -> str:
    parts: List[str] = []
    for field in bucket_fields:
        field_name = str(field)
        if field_name == "domain":
            value = _domain_bucket_from_scored_record(record)
        elif field_name == "style":
            value = _style_bucket_from_scored_record(record)
        elif field_name == "length":
            value = _length_bucket_from_scored_record(record)
        elif field_name == "quality_band":
            value = _quality_band_from_scored_record(record)
        else:
            value = "unknown"
        parts.append(f"{field_name}={value}")
    return "|".join(parts)


def _mean_record_metric(records: List[Dict[str, Any]], metric_fn) -> float:
    if not records:
        return 0.0
    return float(sum(float(metric_fn(record)) for record in records) / float(len(records)))


def _rebalance_learnability_support(
    *,
    selected: List[Dict[str, Any]],
    eligible_records: List[Dict[str, Any]],
    selector_cfg: Dict[str, Any],
    protected_acceptors: set[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    protected_acceptors = protected_acceptors or set()
    enabled = bool(selector_cfg.get("enable_learnability_rebalance"))
    max_swap_ratio = max(0.0, float(selector_cfg.get("learnability_rebalance_max_swap_ratio") or 0.0))
    bucket_fields = [str(item) for item in (selector_cfg.get("learnability_rebalance_preserve_buckets") or [])]
    if not bucket_fields:
        bucket_fields = ["domain", "style", "length"]
    before_mean = _mean_record_metric(selected, _learnability_score_from_record)
    before_quality = _mean_record_metric(selected, _quality_score_from_scored_record)
    if not enabled or not selected or max_swap_ratio <= 0.0:
        return selected, {
            "enabled": bool(enabled),
            "swap_count": 0,
            "max_swaps": 0,
            "preserve_buckets": bucket_fields,
            "mean_learnability_before": round(before_mean, 6),
            "mean_learnability_after": round(before_mean, 6),
            "mean_quality_before": round(before_quality, 6),
            "mean_quality_after": round(before_quality, 6),
            "policy": "disabled" if not enabled else "no_swap_budget",
        }

    selected_map: Dict[str, Dict[str, Any]] = {str(record.get("chunk_uid") or ""): record for record in selected}
    selected_uids = set(selected_map.keys())
    max_swaps = max(1, int(math.floor(float(len(selected)) * max_swap_ratio)))
    min_gain = max(0.0, float(selector_cfg.get("learnability_rebalance_min_gain") or 0.08))
    min_quality = max(0.0, float(selector_cfg.get("learnability_rebalance_min_quality") or 0.80))

    removable: List[Dict[str, Any]] = []
    for record in selected_map.values():
        accepted_by = str((record.get("selection") or {}).get("accepted_by") or "")
        if accepted_by in protected_acceptors:
            continue
        removable.append(record)
    removable.sort(
        key=lambda record: (
            _learnability_score_from_record(record),
            float((record.get("selection") or {}).get("stage_b_rank_score") or 0.0),
            str(record.get("chunk_uid") or ""),
        )
    )

    additions_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in eligible_records:
        uid = str(record.get("chunk_uid") or "")
        if uid in selected_uids:
            continue
        if _quality_score_from_scored_record(record) < min_quality:
            continue
        bucket = _learnability_rebalance_bucket(record, bucket_fields)
        additions_by_bucket[bucket].append(record)
    for records in additions_by_bucket.values():
        records.sort(
            key=lambda record: (
                -_learnability_score_from_record(record),
                -float((record.get("selection") or {}).get("stage_b_rank_score") or 0.0),
                str(record.get("chunk_uid") or ""),
            )
        )

    swap_count = 0
    gain_sum = 0.0
    skipped_no_bucket = 0
    skipped_low_gain = 0
    touched_buckets: Counter[str] = Counter()
    for donor in removable:
        if swap_count >= max_swaps:
            break
        donor_uid = str(donor.get("chunk_uid") or "")
        if donor_uid not in selected_map:
            continue
        bucket = _learnability_rebalance_bucket(donor, bucket_fields)
        candidates = additions_by_bucket.get(bucket) or []
        if not candidates:
            skipped_no_bucket += 1
            continue
        donor_score = _learnability_score_from_record(donor)
        add_record = None
        while candidates:
            candidate = candidates.pop(0)
            candidate_uid = str(candidate.get("chunk_uid") or "")
            if candidate_uid in selected_map:
                continue
            candidate_score = _learnability_score_from_record(candidate)
            if candidate_score < donor_score + min_gain:
                skipped_low_gain += 1
                break
            add_record = candidate
            break
        if add_record is None:
            continue

        add_uid = str(add_record.get("chunk_uid") or "")
        donor["selection"]["accepted"] = False
        donor["selection"]["accepted_by"] = "learnability_rebalance_removed"
        add_record["selection"]["accepted"] = True
        add_record["selection"]["accepted_by"] = "learnability_rebalance"
        selected_map.pop(donor_uid, None)
        selected_map[add_uid] = add_record
        selected_uids.discard(donor_uid)
        selected_uids.add(add_uid)
        swap_count += 1
        gain_sum += _learnability_score_from_record(add_record) - donor_score
        touched_buckets[bucket] += 1

    selected_after = list(selected_map.values())
    after_mean = _mean_record_metric(selected_after, _learnability_score_from_record)
    after_quality = _mean_record_metric(selected_after, _quality_score_from_scored_record)
    return selected_after, {
        "enabled": True,
        "policy": "same_bucket_learnability_swap",
        "swap_count": int(swap_count),
        "max_swaps": int(max_swaps),
        "max_swap_ratio": round(float(max_swap_ratio), 6),
        "min_gain": round(float(min_gain), 6),
        "min_quality": round(float(min_quality), 6),
        "preserve_buckets": bucket_fields,
        "mean_learnability_before": round(before_mean, 6),
        "mean_learnability_after": round(after_mean, 6),
        "mean_learnability_delta": round(float(after_mean - before_mean), 6),
        "mean_quality_before": round(before_quality, 6),
        "mean_quality_after": round(after_quality, 6),
        "mean_quality_delta": round(float(after_quality - before_quality), 6),
        "mean_swap_gain": round(float(gain_sum) / float(max(swap_count, 1)), 6),
        "skipped_no_matching_bucket": int(skipped_no_bucket),
        "skipped_below_min_gain": int(skipped_low_gain),
        "touched_bucket_count": int(len(touched_buckets)),
        "top_touched_buckets": dict(touched_buckets.most_common(8)),
    }


def _cluster_quota_selection(
    *,
    eligible_records: List[Dict[str, Any]],
    distribution_reference_records: List[Dict[str, Any]],
    original_clusters: Counter[int],
    strategy: Dict[str, Any],
    selector_cfg: Dict[str, Any],
    target_count: int,
    min_selected_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not eligible_records:
        return [], {
            "target_selected_count": int(target_count),
            "selected_records": 0,
            "selected_word_count": 0,
            "quota_clusters": 0,
            "rare_cluster_floor_count": 0,
            "capacity_shortfall": int(max(0, target_count)),
        }

    by_cluster: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in eligible_records:
        by_cluster[_cluster_id(record)].append(record)
    for records in by_cluster.values():
        records.sort(
            key=lambda r: (
                -float(r["selection"].get("stage_b_rank_score") or 0.0),
                str(r.get("chunk_uid") or ""),
            )
        )

    eligible_clusters = {cluster_id for cluster_id, records in by_cluster.items() if records}
    rare_clusters = sorted(cluster_id for cluster_id in strategy["rare_clusters"] if cluster_id in eligible_clusters)
    effective_target = max(int(target_count), len(rare_clusters))
    quotas: Dict[int, int] = {cluster_id: 0 for cluster_id in eligible_clusters}

    # Reserve slots for rare clusters so the selector does not collapse
    # into head-only regions before Stage C ever runs.
    rare_cluster_min_count = max(1, int(selector_cfg.get("rare_cluster_min_count") or 1))
    for cluster_id in rare_clusters:
        quotas[cluster_id] = min(rare_cluster_min_count, len(by_cluster[cluster_id]))

    reserved_rare_slots = sum(quotas.get(cluster_id, 0) for cluster_id in rare_clusters)
    remaining = max(0, effective_target - reserved_rare_slots)
    if remaining > 0 and eligible_clusters:
        eligible_mass = float(sum(original_clusters.get(cluster_id, len(by_cluster[cluster_id])) for cluster_id in eligible_clusters))
        if eligible_mass <= 0.0:
            eligible_mass = float(len(eligible_clusters))

        fractional: List[Tuple[float, int]] = []
        allocated = 0
        for cluster_id in sorted(eligible_clusters):
            capacity_remaining = max(0, len(by_cluster[cluster_id]) - quotas[cluster_id])
            if capacity_remaining <= 0:
                continue
            weight = float(original_clusters.get(cluster_id, len(by_cluster[cluster_id]))) / eligible_mass
            raw_quota = remaining * weight
            extra = min(capacity_remaining, int(math.floor(raw_quota)))
            quotas[cluster_id] += extra
            allocated += extra
            remainder = raw_quota - float(extra)
            fractional.append((remainder, cluster_id))

        leftover = max(0, remaining - allocated)
        if leftover > 0:
            fractional.sort(key=lambda item: (-item[0], item[1]))
            idx = 0
            while leftover > 0 and fractional:
                _, cluster_id = fractional[idx % len(fractional)]
                capacity_remaining = max(0, len(by_cluster[cluster_id]) - quotas[cluster_id])
                if capacity_remaining > 0:
                    quotas[cluster_id] += 1
                    leftover -= 1
                idx += 1
                if idx > len(fractional) * max(1, remaining + 1):
                    break

    selected: List[Dict[str, Any]] = []
    for cluster_id in sorted(quotas):
        records = by_cluster[cluster_id]
        if cluster_id in strategy["rare_clusters"]:
            preferred = [record for record in records if _passes_rare_exemplar_filters(record, strategy)]
            preferred_uids = {str(record.get("chunk_uid") or "") for record in preferred}
            fallback = [record for record in records if str(record.get("chunk_uid") or "") not in preferred_uids]
            records = preferred + fallback
        take = min(int(quotas[cluster_id]), len(records))
        for position, record in enumerate(records[:take]):
            record["selection"]["accepted"] = True
            record["selection"]["accepted_by"] = "selector_quota"
            if cluster_id in strategy["rare_clusters"] and position == 0:
                record["selection"]["accepted_by"] = "rare_cluster_exemplar"
            selected.append(record)

    domain_anchor_diag = {
        "enabled": False,
        "bucket_name": "domain",
        "min_count": 0,
        "eligible_bucket_count": 0,
        "selected_bucket_count_before": 0,
        "selected_bucket_count_after": 0,
        "added_records": 0,
    }
    style_anchor_diag = {
        "enabled": False,
        "bucket_name": "style",
        "min_count": 0,
        "eligible_bucket_count": 0,
        "selected_bucket_count_before": 0,
        "selected_bucket_count_after": 0,
        "added_records": 0,
    }
    if bool(selector_cfg.get("preserve_domain_bucket_exemplars", True)):
        selected, domain_anchor_diag = _top_up_bucket_exemplars(
            selected=selected,
            eligible_records=eligible_records,
            bucket_name="domain",
            bucket_fn=_domain_bucket_from_scored_record,
            min_count=int(selector_cfg.get("domain_bucket_min_count") or 1),
            accepted_by="domain_bucket_exemplar",
        )
    if bool(selector_cfg.get("preserve_style_bucket_exemplars", True)):
        selected, style_anchor_diag = _top_up_bucket_exemplars(
            selected=selected,
            eligible_records=eligible_records,
            bucket_name="style",
            bucket_fn=lambda r: str((r.get("selection") or {}).get("style_bucket") or "unknown"),
            min_count=int(selector_cfg.get("style_bucket_min_count") or 4),
            accepted_by="style_bucket_exemplar",
        )

    protected_acceptors = {
        "rare_cluster_exemplar",
        "domain_bucket_exemplar",
        "style_bucket_exemplar",
    }
    distribution_balance_diag: Dict[str, Dict[str, Any]] = {}
    for bucket_name, bucket_fn, enabled_key in (
        ("domain", _domain_bucket_from_scored_record, "preserve_domain_distribution"),
        ("style", _style_bucket_from_scored_record, "preserve_style_distribution"),
        ("length", _length_bucket_from_scored_record, "preserve_length_distribution"),
        ("quality_band", _quality_band_from_scored_record, "preserve_quality_band_distribution"),
    ):
        reference_records = distribution_reference_records
        if bucket_name == "quality_band":
            min_quality = float(selector_cfg.get("quality_band_distribution_min_quality") or 0.0)
            reference_records = [
                record for record in distribution_reference_records if _quality_score_from_scored_record(record) >= min_quality
            ]
        enabled = bool(selector_cfg.get(enabled_key, False))
        if bucket_name == "quality_band" and not enabled and bool(selector_cfg.get("diagnose_quality_band_distribution", True)):
            bucket_diag = _bucket_distribution_diagnostic(
                selected=selected,
                reference_records=reference_records,
                bucket_name=bucket_name,
                bucket_fn=bucket_fn,
            )
        elif bucket_name == "quality_band" and enabled and str(selector_cfg.get("quality_band_rebalance_mode") or "soft_cap") == "soft_cap":
            selected, bucket_diag = _soft_cap_quality_band_distribution(
                selected=selected,
                eligible_records=eligible_records,
                reference_records=reference_records,
                target_count=int(len(selected)),
                min_quality=float(selector_cfg.get("quality_band_distribution_min_quality") or 0.0),
                top_band_max_share=float(selector_cfg.get("quality_top_band_max_share") or 0.08),
                max_swap_ratio=float(selector_cfg.get("quality_band_max_swap_ratio") or 0.08),
                protected_acceptors=protected_acceptors,
            )
        else:
            selected, bucket_diag = _rebalance_bucket_distribution(
                selected=selected,
                eligible_records=eligible_records,
                reference_records=reference_records,
                bucket_name=bucket_name,
                bucket_fn=bucket_fn,
                target_count=int(len(selected)),
                enabled=enabled,
                protected_acceptors=protected_acceptors,
            )
        if bucket_name == "quality_band":
            bucket_diag["min_quality"] = round(float(selector_cfg.get("quality_band_distribution_min_quality") or 0.0), 6)
            bucket_diag["policy"] = (
                "diagnostic_only_not_preserved"
                if not enabled
                else (
                    "soft_top_quality_anti_collapse"
                    if str(selector_cfg.get("quality_band_rebalance_mode") or "soft_cap") == "soft_cap"
                    else "mid_high_quality_distribution_preserved"
                )
            )
            bucket_diag["purpose"] = (
                "Prevent top-quality tail collapse without forcing exact quality-band distribution matching."
            )
        distribution_balance_diag[bucket_name] = bucket_diag

    selected, learnability_rebalance_diag = _rebalance_learnability_support(
        selected=selected,
        eligible_records=eligible_records,
        selector_cfg=selector_cfg,
        protected_acceptors=protected_acceptors,
    )

    if min_selected_tokens > 0:
        selected_word_count = sum(int(r.get("word_count") or 0) for r in selected)
        if selected_word_count < min_selected_tokens:
            selected_uids = {str(r["chunk_uid"]) for r in selected}
            remaining_records = [
                record
                for record in eligible_records
                if str(record["chunk_uid"]) not in selected_uids
            ]
            remaining_records.sort(
                key=lambda r: (
                    -float(r["selection"].get("stage_b_rank_score") or 0.0),
                    str(r.get("chunk_uid") or ""),
                )
            )
            for record in remaining_records:
                record["selection"]["accepted"] = True
                record["selection"]["accepted_by"] = "token_budget_topup"
                selected.append(record)
                selected_word_count += int(record.get("word_count") or 0)
                if selected_word_count >= min_selected_tokens:
                    break

    selected_word_count = sum(int(r.get("word_count") or 0) for r in selected)
    return selected, {
        "target_selected_count": int(effective_target),
        "selected_records": int(len(selected)),
        "selected_word_count": int(selected_word_count),
        "quota_clusters": int(len(quotas)),
        "rare_cluster_floor_count": int(reserved_rare_slots),
        "rare_cluster_min_count": int(rare_cluster_min_count),
        "domain_bucket_anchor": domain_anchor_diag,
        "style_bucket_anchor": style_anchor_diag,
        "distribution_balance": distribution_balance_diag,
        "domain_distribution_balance": distribution_balance_diag.get("domain", {}),
        "style_distribution_balance": distribution_balance_diag.get("style", {}),
        "length_distribution_balance": distribution_balance_diag.get("length", {}),
        "quality_band_distribution_balance": distribution_balance_diag.get("quality_band", {}),
        "learnability_rebalance": learnability_rebalance_diag,
        "capacity_shortfall": int(max(0, effective_target - len(selected))),
    }


def _satisfies_selector_constraints(
    coverage: Dict[str, float],
    selected_count: int,
    source_records: int,
    selected_word_count: int,
    selector_cfg: Dict[str, Any],
    stage_c: Dict[str, Any],
) -> bool:
    coverage_ok = bool(
        float(coverage["score"]) >= float(stage_c["min_subset_coverage_retention_score"])
        and float(coverage["rare_cluster_retention"]) >= float(stage_c["min_rare_cluster_retention"])
        and int(coverage["rare_cluster_retained_count"]) >= int(stage_c["min_rare_cluster_retained_count"])
    )
    min_selected_ratio = float(selector_cfg.get("min_selected_ratio") or 0.0)
    min_selected_tokens = int(selector_cfg.get("min_selected_tokens") or 0)
    selected_ratio = float(selected_count) / float(max(source_records, 1))
    size_ok = bool(selected_ratio >= min_selected_ratio and selected_word_count >= min_selected_tokens)
    return bool(coverage_ok and size_ok)


def _select_with_objective_constraints(
    *,
    candidates: List[Dict[str, Any]],
    stage_b: Dict[str, Any],
    selector_cfg: Dict[str, Any],
    strategy: Dict[str, Any],
    original_clusters: Counter[int],
    source_records: int,
    stage_c: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not candidates:
        return [], {
            "iterations": [],
            "final_iteration": 0,
            "coverage_constraints_satisfied": False,
            "selector_constraints_satisfied": False,
            "constraint_violations": {
                "min_subset_coverage_retention_score": True,
                "min_rare_cluster_retention": True,
                "min_rare_cluster_retained_count": True,
                "min_selected_ratio": True,
                "min_selected_tokens": True,
            },
        }

    penalties = dict(selector_cfg["constraint_penalties"])
    threshold = float(stage_b["selection_threshold"])
    iteration_cap = max(1, int(selector_cfg["iteration_cap"]))
    iterations: List[Dict[str, Any]] = []
    best_selected: List[Dict[str, Any]] = []
    best_coverage = {"score": 0.0, "rare_cluster_retention": 0.0, "rare_cluster_retained_count": 0}
    best_selected_word_count = 0

    for i in range(iteration_cap):
        for record in candidates:
            record["selection"]["accepted"] = False
            record["selection"]["accepted_by"] = None
            record["selection"]["accepted_without_threshold"] = False
            record["selection"]["stage_b_fail_reason"] = None
        eligible_records: List[Dict[str, Any]] = []
        for record in candidates:
            components = _objective_components(record)
            relief = _structured_text_relief(record)
            effective_risk = max(0.0, components["redundancy_risk"] - float(relief["redundancy_risk_relief"]))
            objective = _objective_score_with_constraints(
                record=record,
                components=components,
                selector_cfg={
                    "objective_weights": selector_cfg["objective_weights"],
                    "constraint_penalties": penalties,
                    "selection_adjustments": selector_cfg.get("selection_adjustments") or {},
                },
                strategy=strategy,
            )
            record["selection"]["objective_components"] = {
                "selection_value": round(float(components["selection_value"]), 6),
                "quality": round(float(components["quality"]), 6),
                "redundancy_risk": round(float(components["redundancy_risk"]), 6),
                "effective_redundancy_risk": round(float(effective_risk), 6),
                "useful_length_support": round(float(components["useful_length_support"]), 6),
                "lexical_diversity": round(float(components["lexical_diversity"]), 6),
                "useful_recurrence": round(float(components["useful_recurrence"]), 6),
                "repeat_pressure": round(float(components["repeat_pressure"]), 6),
                "lexical_balance_support": round(float(components["lexical_balance_support"]), 6),
                "pattern_recurrence_support": round(float(components["pattern_recurrence_support"]), 6),
                "selection_value_learnability_support": round(float(components["selection_value_learnability_support"]), 6),
                "quality_learnability_support": round(float(components["quality_learnability_support"]), 6),
                "selection_value_tail_penalty": round(float(components["selection_value_tail_penalty"]), 6),
                "quality_tail_penalty": round(float(components["quality_tail_penalty"]), 6),
                "learnability_support": round(float(components["learnability_support"]), 6),
                "boilerplate_penalty": round(float(components["boilerplate_penalty"]), 6),
            }
            record["selection"]["style_bucket"] = relief["style_bucket"]
            record["selection"]["structured_text_relief"] = round(float(relief["redundancy_risk_relief"]), 6)
            record["selection"]["stage_b_rank_score"] = round(float(objective), 6)
            record["selection"]["stage_b_rank_passed"] = bool(objective >= threshold)
            if effective_risk > float(stage_b["near_duplicate_risk_ceiling"]):
                record["selection"]["stage_b_rank_passed"] = False
                record["selection"]["stage_b_fail_reason"] = "near_duplicate_risk_ceiling"
                continue
            if objective >= threshold:
                eligible_records.append(record)

        target_selected_count = _target_selected_count(source_records, selector_cfg)
        min_selected_tokens = int(selector_cfg.get("min_selected_tokens") or 0)
        selected, quota_diag = _cluster_quota_selection(
            eligible_records=eligible_records,
            distribution_reference_records=candidates,
            original_clusters=original_clusters,
            strategy=strategy,
            selector_cfg=selector_cfg,
            target_count=target_selected_count,
            min_selected_tokens=min_selected_tokens,
        )
        selected_clusters: Counter[int] = Counter()
        for record in selected:
            selected_clusters[_cluster_id(record)] += 1
        selected_word_count = int(quota_diag["selected_word_count"])
        selected_ratio = float(len(selected)) / float(max(source_records, 1))
        coverage = _coverage_retention(selected_clusters, original_clusters)
        ok = _satisfies_selector_constraints(
            coverage,
            len(selected),
            source_records,
            selected_word_count,
            selector_cfg,
            stage_c,
        )
        if coverage["score"] >= best_coverage.get("score", 0.0):
            best_selected = list(selected)
            best_coverage = dict(coverage)
            best_selected_word_count = int(selected_word_count)

        min_selected_ratio = float(selector_cfg.get("min_selected_ratio") or 0.0)
        violated = {
            "min_subset_coverage_retention_score": coverage["score"] < float(stage_c["min_subset_coverage_retention_score"]),
            "min_rare_cluster_retention": coverage["rare_cluster_retention"] < float(stage_c["min_rare_cluster_retention"]),
            "min_rare_cluster_retained_count": int(coverage["rare_cluster_retained_count"]) < int(stage_c["min_rare_cluster_retained_count"]),
            "min_selected_ratio": selected_ratio < min_selected_ratio,
            "min_selected_tokens": selected_word_count < min_selected_tokens,
        }
        iterations.append(
            {
                "iteration": i + 1,
                "selection_threshold": round(float(threshold), 6),
                "constraint_penalties": {k: round(float(v), 6) for k, v in penalties.items()},
                "selected_records": int(len(selected)),
                "selected_ratio": round(float(selected_ratio), 6),
                "selected_word_count": int(selected_word_count),
                "eligible_records": int(len(eligible_records)),
                "min_selected_ratio": round(float(min_selected_ratio), 6),
                "min_selected_tokens": int(min_selected_tokens),
                "quota_diagnostics": quota_diag,
                "coverage": coverage,
                "coverage_constraints_satisfied": bool(
                    not violated["min_subset_coverage_retention_score"]
                    and not violated["min_rare_cluster_retention"]
                    and not violated["min_rare_cluster_retained_count"]
                ),
                "selector_constraints_satisfied": bool(ok),
                "constraint_violations": violated,
            }
        )
        if ok:
            return selected, {
                "iterations": iterations,
                "final_iteration": i + 1,
                "coverage_constraints_satisfied": True,
                "selector_constraints_satisfied": True,
                "constraint_violations": violated,
            }

        penalties["rare_cluster_bonus"] *= float(max(1.0, penalties.get("penalty_growth", 1.0)))
        penalties["small_cluster_bonus"] *= float(max(1.0, penalties.get("penalty_growth", 1.0)))
        threshold -= float(penalties.get("threshold_relax_step", 0.01))

    violated = {
        "min_subset_coverage_retention_score": best_coverage["score"] < float(stage_c["min_subset_coverage_retention_score"]),
        "min_rare_cluster_retention": best_coverage["rare_cluster_retention"] < float(stage_c["min_rare_cluster_retention"]),
        "min_rare_cluster_retained_count": int(best_coverage["rare_cluster_retained_count"]) < int(stage_c["min_rare_cluster_retained_count"]),
        "min_selected_ratio": (float(len(best_selected)) / float(max(source_records, 1))) < float(selector_cfg.get("min_selected_ratio") or 0.0),
        "min_selected_tokens": int(best_selected_word_count) < int(selector_cfg.get("min_selected_tokens") or 0),
    }
    if iterations:
        last = iterations[-1].get("constraint_violations") or {}
        violated["min_selected_ratio"] = bool(last.get("min_selected_ratio", True))
        violated["min_selected_tokens"] = bool(last.get("min_selected_tokens", True))
    return best_selected, {
        "iterations": iterations,
        "final_iteration": int(len(iterations)),
        "coverage_constraints_satisfied": False,
        "selector_constraints_satisfied": False,
        "constraint_violations": violated,
    }


def _estimate_metric_quantile(
    source_path: Path,
    metric_name: str,
    quantile: float,
    sample_size: int = 60000,
    seed: int = 42,
) -> float:
    q = max(0.0, min(1.0, float(quantile)))
    cap = max(5000, int(sample_size))
    rng = random.Random(seed)
    reservoir: List[float] = []
    seen = 0
    for record in _iter_scored_records(source_path):
        metrics = record.get("core_metrics") or {}
        payload = metrics.get(metric_name) or {}
        try:
            value = float(payload.get("score"))
        except (TypeError, ValueError):
            continue
        seen += 1
        if len(reservoir) < cap:
            reservoir.append(value)
            continue
        slot = rng.randrange(seen)
        if slot < cap:
            reservoir[slot] = value
    if not reservoir:
        return 1.0
    reservoir.sort()
    idx = int(round(q * (len(reservoir) - 1)))
    idx = max(0, min(len(reservoir) - 1, idx))
    return float(reservoir[idx])


def _float_metric(payload: Dict[str, Any], key: str, *, fallback_key: str | None = None, default: float = 0.0) -> float:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    if value is None:
        return float(default)
    return float(value)


def _utility_pass_field(mean_key: str, pass_statistic: str) -> str:
    if pass_statistic == "min" and not mean_key.endswith("_min"):
        return f"{mean_key}_min"
    return mean_key


def _utility_result_value(probe_result: Dict[str, Any], mean_key: str, pass_statistic: str) -> float:
    key = _utility_pass_field(mean_key, pass_statistic)
    return _float_metric(probe_result, key, fallback_key=mean_key)


CANONICAL_UTILITY_BASELINE = "baseline_multi_matched_stageA_random"
CURATION_BENEFIT_BASELINE = "baseline_stageA_random"
STRICT_COUNTERFACTUAL_BASELINE = CANONICAL_UTILITY_BASELINE
OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE = "baseline_nuisance_matched_stageA_random"
ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE = "baseline_anti_memorization_matched_stageA_random"
DIAGNOSTIC_UTILITY_BASELINES = (
    CURATION_BENEFIT_BASELINE,
    "baseline_full_random",
    OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE,
    "baseline_multi_matched_stageA_random",
    "baseline_style_matched_stageA_random",
    "baseline_length_matched_stageA_random",
    "baseline_quality_band_matched_stageA_random",
)
MATCHED_DIAGNOSTIC_BASELINES = {
    OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE: _nuisance_matched_bucket_from_scored_record,
    "baseline_multi_matched_stageA_random": _multi_matched_bucket_from_scored_record,
    "baseline_style_matched_stageA_random": _style_bucket_from_scored_record,
    "baseline_length_matched_stageA_random": _length_bucket_from_scored_record,
    "baseline_quality_band_matched_stageA_random": _quality_band_from_scored_record,
}


def _load_utility_sensitivity_audit() -> Dict[str, Any]:
    global _UTILITY_SENSITIVITY_AUDIT_CACHE
    if _UTILITY_SENSITIVITY_AUDIT_CACHE is not None:
        return _UTILITY_SENSITIVITY_AUDIT_CACHE
    if not UTILITY_SENSITIVITY_AUDIT_PATH.exists():
        _UTILITY_SENSITIVITY_AUDIT_CACHE = {}
        return _UTILITY_SENSITIVITY_AUDIT_CACHE
    try:
        payload = json.loads(UTILITY_SENSITIVITY_AUDIT_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    _UTILITY_SENSITIVITY_AUDIT_CACHE = payload if isinstance(payload, dict) else {}
    return _UTILITY_SENSITIVITY_AUDIT_CACHE


def _utility_sensitivity_for_dataset(dataset: str) -> Dict[str, Any]:
    payload = _load_utility_sensitivity_audit()
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, dict):
        return {}
    value = datasets.get(str(dataset))
    return value if isinstance(value, dict) else {}


def _stable_hash_score(value: str, *, seed: int) -> float:
    digest = hashlib.sha1(f"{int(seed)}:{value}".encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _fingerprint_uids(uids: Iterable[str]) -> str:
    hasher = hashlib.sha1()
    for uid in sorted(str(uid) for uid in uids):
        hasher.update(uid.encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def _matched_bucket_baseline_pool(
    *,
    baseline_records: List[Dict[str, Any]],
    selected_records: List[Dict[str, Any]],
    bucket_fn: Any,
    seed: int,
    pool_multiplier: int,
    exclude_selected: bool = True,
) -> Tuple[set[str], Dict[str, Any]]:
    selected_counts: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in selected_records)
    selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    excluded_selected_records = 0
    for record in baseline_records:
        uid = str(record.get("chunk_uid") or "")
        if exclude_selected and uid in selected_uids:
            excluded_selected_records += 1
            continue
        by_bucket[str(bucket_fn(record) or "unknown")].append(record)
    for records in by_bucket.values():
        records.sort(
            key=lambda record: (
                _stable_hash_score(str(record.get("chunk_uid") or ""), seed=seed),
                str(record.get("chunk_uid") or ""),
            )
        )

    chosen_uids: set[str] = set()
    bucket_targets: Dict[str, int] = {}
    bucket_available: Dict[str, int] = {}
    bucket_selected: Dict[str, int] = {}
    multiplier = max(1, int(pool_multiplier))
    for bucket, selected_count in sorted(selected_counts.items()):
        available = by_bucket.get(bucket, [])
        bucket_available[bucket] = int(len(available))
        target = min(len(available), max(int(selected_count), int(selected_count) * multiplier, 64))
        bucket_targets[bucket] = int(target)
        for record in available[:target]:
            uid = str(record.get("chunk_uid") or "")
            if uid:
                chosen_uids.add(uid)
        bucket_selected[bucket] = int(min(target, len(available)))

    return chosen_uids, {
        "bucket_count": int(len(selected_counts)),
        "selected_reference_count": int(len(selected_records)),
        "baseline_reference_count": int(len(baseline_records)),
        "matched_pool_count": int(len(chosen_uids)),
        "pool_multiplier": int(multiplier),
        "exclude_selected": bool(exclude_selected),
        "excluded_selected_records": int(excluded_selected_records),
        "bucket_targets": bucket_targets,
        "bucket_available": bucket_available,
        "bucket_selected": bucket_selected,
        "matched_bucket_count": int(sum(1 for bucket in selected_counts if bucket_available.get(bucket, 0) > 0)),
        "matched_bucket_ratio": round(
            sum(1 for bucket in selected_counts if bucket_available.get(bucket, 0) > 0) / max(len(selected_counts), 1),
            6,
        ),
        "matched_selected_reference_count": int(
            sum(selected_count for bucket, selected_count in selected_counts.items() if bucket_available.get(bucket, 0) > 0)
        ),
        "matched_selected_reference_ratio": round(
            sum(selected_count for bucket, selected_count in selected_counts.items() if bucket_available.get(bucket, 0) > 0)
            / max(sum(selected_counts.values()), 1),
            6,
        ),
    }


def _multi_matched_stagea_baseline_pool(
    *,
    baseline_records: List[Dict[str, Any]],
    selected_records: List[Dict[str, Any]],
    seed: int,
    pool_multiplier: int,
    exclude_selected: bool = True,
) -> Tuple[set[str], Dict[str, Any]]:
    def keys(record: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            _quality_band_from_scored_record(record),
            _length_bucket_from_scored_record(record),
            _style_bucket_from_scored_record(record),
            _domain_bucket_from_scored_record(record),
        )

    def level_keys(key: Tuple[str, str, str, str]) -> Tuple[str, str, str, str, str]:
        quality, length, style, domain = key
        return (
            f"exact::{quality}|{length}|{style}|{domain}",
            f"quality_length_style::{quality}|{length}|{style}",
            f"quality_length::{quality}|{length}",
            f"quality::{quality}",
            "global::*",
        )

    selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
    selected_counts: Counter[Tuple[str, str, str, str]] = Counter(keys(record) for record in selected_records)
    candidate_levels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    excluded_selected_records = 0
    for record in baseline_records:
        uid = str(record.get("chunk_uid") or "")
        if exclude_selected and uid in selected_uids:
            excluded_selected_records += 1
            continue
        for level_key in level_keys(keys(record)):
            candidate_levels[level_key].append(record)
    for records in candidate_levels.values():
        records.sort(
            key=lambda record: (
                _stable_hash_score(str(record.get("chunk_uid") or ""), seed=seed),
                str(record.get("chunk_uid") or ""),
            )
        )

    chosen_uids: set[str] = set()
    bucket_targets: Dict[str, int] = {}
    bucket_selected: Dict[str, int] = {}
    bucket_available_exact: Dict[str, int] = {}
    fallback_selected_by_level: Counter[str] = Counter()
    multiplier = max(1, int(pool_multiplier))
    level_cursors: Dict[str, int] = defaultdict(int)

    def take_from_level(level_key: str, remaining: int) -> int:
        if remaining <= 0:
            return 0
        records = candidate_levels.get(level_key, [])
        cursor = int(level_cursors.get(level_key, 0))
        chosen = 0
        level_name = level_key.split("::", 1)[0]
        while cursor < len(records) and chosen < remaining:
            record = records[cursor]
            cursor += 1
            uid = str(record.get("chunk_uid") or "")
            if not uid or uid in chosen_uids:
                continue
            chosen_uids.add(uid)
            chosen += 1
            fallback_selected_by_level[level_name] += 1
        level_cursors[level_key] = cursor
        return chosen

    for key, selected_count in sorted(selected_counts.items()):
        bucket_key = "|".join(key)
        target = max(int(selected_count), int(selected_count) * multiplier, 64)
        bucket_targets[bucket_key] = int(target)
        chosen_for_bucket = 0
        exact_key, *fallback_keys = level_keys(key)
        bucket_available_exact[bucket_key] = int(len(candidate_levels.get(exact_key, [])))
        for level_key in (exact_key, *fallback_keys):
            if chosen_for_bucket >= target:
                break
            chosen_for_bucket += take_from_level(level_key, target - chosen_for_bucket)
        bucket_selected[bucket_key] = int(chosen_for_bucket)

    def compact(mapping: Dict[str, int], *, limit: int = 200) -> Dict[str, int]:
        ordered_items = sorted(mapping.items(), key=lambda item: (-int(item[1]), item[0]))
        return {str(k): int(v) for k, v in ordered_items[:limit]}

    return chosen_uids, {
        "bucket_count": int(len(selected_counts)),
        "selected_reference_count": int(len(selected_records)),
        "baseline_reference_count": int(len(baseline_records)),
        "matched_pool_count": int(len(chosen_uids)),
        "pool_multiplier": int(multiplier),
        "exclude_selected": bool(exclude_selected),
        "excluded_selected_records": int(excluded_selected_records),
        "matching_policy": "quality_length_style_domain_with_hierarchical_fallback",
        "fallback_order": [
            "exact",
            "quality_length_style",
            "quality_length",
            "quality",
            "global",
        ],
        "fallback_selected_by_level": {str(k): int(v) for k, v in sorted(fallback_selected_by_level.items())},
        "bucket_diagnostics_truncated": bool(len(bucket_targets) > 200),
        "bucket_diagnostics_limit": 200,
        "bucket_targets": compact(bucket_targets),
        "bucket_available_exact": compact(bucket_available_exact),
        "bucket_selected": compact(bucket_selected),
    }


def _diagnostic_matched_baseline_pools(
    *,
    baseline_records: List[Dict[str, Any]],
    selected_records: List[Dict[str, Any]],
    seed: int,
    pool_multiplier: int,
    exclude_selected: bool = True,
) -> Dict[str, Dict[str, Any]]:
    pools: Dict[str, Dict[str, Any]] = {}
    for baseline_name, bucket_fn in MATCHED_DIAGNOSTIC_BASELINES.items():
        if baseline_name == OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE:
            uids, diagnostics = _matched_bucket_baseline_pool(
                baseline_records=baseline_records,
                selected_records=selected_records,
                bucket_fn=_nuisance_matched_bucket_from_scored_record,
                seed=int(seed),
                pool_multiplier=int(pool_multiplier),
                exclude_selected=bool(exclude_selected),
            )
            diagnostics.update(
                {
                    "matching_policy": "exact_length_style_domain_repeat_pressure",
                    "matched_variables": ["length", "style", "domain", "repeat_pressure"],
                    "excluded_selector_target_variables": ["quality", "redundancy_risk"],
                    "fallback_order": [],
                }
            )
        elif baseline_name == CANONICAL_UTILITY_BASELINE:
            uids, diagnostics = _multi_matched_stagea_baseline_pool(
                baseline_records=baseline_records,
                selected_records=selected_records,
                seed=int(seed),
                pool_multiplier=int(pool_multiplier),
                exclude_selected=bool(exclude_selected),
            )
        else:
            uids, diagnostics = _matched_bucket_baseline_pool(
                baseline_records=baseline_records,
                selected_records=selected_records,
                bucket_fn=bucket_fn,
                seed=int(seed),
                pool_multiplier=int(pool_multiplier),
                exclude_selected=bool(exclude_selected),
            )
        pools[baseline_name] = {
            "allowed_uids": uids,
            "fingerprint": _fingerprint_uids(uids),
            "diagnostics": diagnostics,
        }
    return pools


def _utility_axis_pass(
    probe_result: Dict[str, Any],
    stage_c: Dict[str, Any],
) -> Dict[str, Any]:
    pass_statistic = str(stage_c.get("utility_pass_statistic") or "min").strip().lower()
    if pass_statistic not in {"mean", "min"}:
        pass_statistic = "min"
    score = _utility_result_value(probe_result, "small_lm_probe_gain_score", pass_statistic)
    rel_gain = _utility_result_value(probe_result, "relative_nll_gain", pass_statistic)
    delta_nll = _utility_result_value(probe_result, "delta_nll", pass_statistic)
    ci_low = _float_metric(probe_result, "delta_nll_ci_low")
    score_pass = score >= float(stage_c["min_small_lm_probe_gain_score"])
    relative_gain_pass = rel_gain >= float(stage_c["min_small_lm_probe_relative_gain"])
    delta_nll_pass = (delta_nll > 0.0) if bool(stage_c["require_utility_delta_nll_positive"]) else True
    ci_pass = (ci_low > 0.0) if bool(stage_c["require_utility_ci_gain_positive"]) else True
    axis_pass = bool(score_pass and relative_gain_pass and delta_nll_pass and ci_pass)
    return {
        "pass": axis_pass,
        "score_pass": bool(score_pass),
        "relative_gain_pass": bool(relative_gain_pass),
        "delta_nll_pass": bool(delta_nll_pass),
        "ci_pass": bool(ci_pass),
        "pass_statistic": pass_statistic,
        "score_value": round(score, 6),
        "relative_gain_value": round(rel_gain, 6),
        "delta_nll_value": round(delta_nll, 6),
        "delta_nll_ci_low": round(ci_low, 6),
    }


def _utility_axis_pass_by_baselines(
    baseline_results: Dict[str, Dict[str, Any]],
    stage_c: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_pass: Dict[str, Dict[str, Any]] = {}
    for baseline_name, result in baseline_results.items():
        baseline_pass[baseline_name] = _utility_axis_pass(result, stage_c)
    canonical = baseline_pass.get(CANONICAL_UTILITY_BASELINE)
    if canonical is None:
        raise RuntimeError(f"Missing canonical utility baseline: {CANONICAL_UTILITY_BASELINE}")
    return {
        "pass": bool(canonical["pass"]),
        "score_pass": bool(canonical["score_pass"]),
        "relative_gain_pass": bool(canonical["relative_gain_pass"]),
        "delta_nll_pass": bool(canonical["delta_nll_pass"]),
        "ci_pass": bool(canonical["ci_pass"]),
        "canonical_baseline": CANONICAL_UTILITY_BASELINE,
        "diagnostic_baselines": list(DIAGNOSTIC_UTILITY_BASELINES),
        "by_baseline": baseline_pass,
    }


def _utility_curation_benefit_status(stage_a_random_result: Dict[str, Any]) -> Dict[str, Any]:
    delta = _float_metric(stage_a_random_result, "delta_nll")
    ci_low = _float_metric(stage_a_random_result, "delta_nll_ci_low")
    mde = _float_metric(
        stage_a_random_result,
        "minimum_detectable_delta_nll_95_max",
        fallback_key="minimum_detectable_delta_nll_95",
    )
    selected_beats_random_mean = bool(delta > 0.0)
    selected_beats_random_ci = bool(ci_low > 0.0)
    selected_beats_random_mde = bool(delta > max(0.0, mde))
    if selected_beats_random_ci or selected_beats_random_mde:
        status = "random_baseline_gain"
    elif selected_beats_random_mean:
        status = "random_baseline_inconclusive"
    else:
        status = "no_random_baseline_gain"
    return {
        "baseline": CURATION_BENEFIT_BASELINE,
        "status": status,
        "selected_beats_random": selected_beats_random_mean,
        "selected_beats_random_ci": selected_beats_random_ci,
        "selected_beats_random_mde": selected_beats_random_mde,
        "delta_nll": round(delta, 6),
        "delta_nll_ci_low": round(ci_low, 6),
        "minimum_detectable_delta_nll_95": round(mde, 8),
        "interpretation": "Curation benefit compares selected against feasible Stage-A random, not against a matched counterfactual.",
    }


def _utility_strict_counterfactual_status(
    *,
    final_scope_certification_ready: bool,
    utility_axis_pass: bool,
    combined_signal_status: Dict[str, Any],
    strict_values: Dict[str, Any],
) -> Dict[str, Any]:
    min_delta = float(strict_values.get("min_delta_nll") or 0.0)
    ci_low = float(strict_values.get("min_delta_nll_ci_low") or 0.0)
    effect_to_mde = float(strict_values.get("min_effect_to_mde_ratio") or 0.0)
    signal_status = str(combined_signal_status.get("status") or "unknown")
    if final_scope_certification_ready:
        status = "strict_certification_ready"
    elif utility_axis_pass:
        status = "matched_baseline_gain"
    elif signal_status.startswith("inconclusive") or (min_delta > 0.0 and (ci_low <= 0.0 or effect_to_mde < 1.0)):
        status = "matched_baseline_inconclusive"
    else:
        status = "strict_negative"
    return {
        "baseline": STRICT_COUNTERFACTUAL_BASELINE,
        "status": status,
        "selected_beats_multi_matched": bool(status in {"matched_baseline_gain", "strict_certification_ready"}),
        "strict_pass": bool(utility_axis_pass),
        "certification_ready": bool(final_scope_certification_ready),
        "signal_status": signal_status,
        "min_delta_nll": round(min_delta, 6),
        "min_delta_nll_ci_low": round(ci_low, 6),
        "min_effect_to_mde_ratio": round(effect_to_mde, 6),
        "interpretation": "Strict counterfactual benefit compares selected against the multi-matched Stage-A baseline.",
    }


def _utility_operational_counterfactual_candidate_status(result: Dict[str, Any]) -> Dict[str, Any]:
    delta = _float_metric(result, "delta_nll")
    min_delta = _float_metric(result, "delta_nll_min", fallback_key="delta_nll")
    ci_low = _float_metric(result, "delta_nll_ci_low")
    mde = _float_metric(
        result,
        "minimum_detectable_delta_nll_95_max",
        fallback_key="minimum_detectable_delta_nll_95",
    )
    positive_run_fraction = _float_metric(
        result.get("stability_diagnostics") or {},
        "positive_run_fraction",
    )
    if min_delta > 0.0 and ci_low > 0.0 and (mde <= 0.0 or min_delta > mde):
        status = "candidate_strict_positive"
    elif delta > 0.0:
        status = "candidate_inconclusive_positive_mean"
    else:
        status = "candidate_negative"
    return {
        "baseline": OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE,
        "role": "operational_counterfactual_candidate_not_canonical",
        "status": status,
        "selected_beats_candidate_mean": bool(delta > 0.0),
        "selected_beats_candidate_strict": bool(status == "candidate_strict_positive"),
        "delta_nll": round(delta, 6),
        "min_delta_nll": round(min_delta, 6),
        "delta_nll_ci_low": round(ci_low, 6),
        "minimum_detectable_delta_nll_95": round(mde, 8),
        "positive_run_fraction": round(positive_run_fraction, 6),
        "interpretation": (
            "This candidate matches length, style, domain, and repeat pressure while leaving selector target "
            "variables unmatched. It is reported for protocol comparison and does not control certification."
        ),
    }


def _utility_probe_sensitivity_status(dataset: str) -> Dict[str, Any]:
    audit = _utility_sensitivity_for_dataset(dataset)
    sensitivity = audit.get("probe_sensitivity") if isinstance(audit, dict) else None
    root_cause = audit.get("root_cause_decision") if isinstance(audit, dict) else None
    if not isinstance(sensitivity, dict):
        return {
            "available": False,
            "probe_valid": None,
            "status": "not_evaluated",
            "selector_tuning_allowed": None,
            "source": str(UTILITY_SENSITIVITY_AUDIT_PATH),
        }
    probe_valid = bool(sensitivity.get("destructive_probe_valid", sensitivity.get("probe_valid", sensitivity.get("order_pass"))))
    status = str(sensitivity.get("utility_evidence_status") or ("probe_valid" if probe_valid else "probe_not_evaluable"))
    return {
        "available": True,
        "probe_valid": probe_valid,
        "destructive_probe_valid": probe_valid,
        "status": status,
        "selector_tuning_allowed": False,
        "selector_policy_action": (root_cause or {}).get("selector_policy_action", "hold"),
        "utility_scope": (root_cause or {}).get("utility_scope", "Stage C diagnostic only; never selector objective"),
        "selector_tuning_caveat": (root_cause or {}).get("selector_tuning_caveat"),
        "source": str(UTILITY_SENSITIVITY_AUDIT_PATH),
        "expected_order": sensitivity.get("expected_order"),
        "positive_gt_random": bool(sensitivity.get("positive_gt_random")),
        "random_gt_negative": bool(sensitivity.get("random_gt_negative", sensitivity.get("random_gt_destructive_negative"))),
        "random_gt_destructive_negative": bool(sensitivity.get("random_gt_destructive_negative", sensitivity.get("random_gt_negative"))),
        "selected_gt_random": bool(sensitivity.get("selected_gt_random")),
        "token_inventory_stress_pass": sensitivity.get("token_inventory_stress_pass"),
        "token_exposure_confounded": bool(sensitivity.get("token_exposure_confounded")),
        "token_exposure_inconclusive": bool(sensitivity.get("token_exposure_inconclusive")),
        "control_margins": sensitivity.get("control_margins") or {},
        "canonical_negative_control": sensitivity.get("canonical_negative_control"),
        "destructive_negative_control": sensitivity.get("destructive_negative_control"),
        "token_inventory_stress_control": sensitivity.get("token_inventory_stress_control"),
        "delta_nll_by_arm": sensitivity.get("delta_nll_by_arm") or {},
        "root_cause": (root_cause or {}).get("primary_hypothesis"),
    }


def _utility_evidence_tier(
    *,
    probe_sensitivity_status: Dict[str, Any],
    curation_benefit_status: Dict[str, Any],
    strict_counterfactual_status: Dict[str, Any],
) -> str:
    if probe_sensitivity_status.get("probe_valid") is False:
        return "not_evaluable_utility_evidence"
    strict_status = str(strict_counterfactual_status.get("status") or "")
    curation_status = str(curation_benefit_status.get("status") or "")
    if strict_status == "strict_certification_ready":
        return "strict_certification_ready"
    if strict_status == "matched_baseline_gain":
        return "matched_baseline_gain"
    token_exposure_caveat = bool(
        probe_sensitivity_status.get("token_exposure_confounded")
        or probe_sensitivity_status.get("token_exposure_inconclusive")
    )
    if curation_status == "random_baseline_gain":
        return (
            "random_baseline_gain_with_token_exposure_caveat"
            if token_exposure_caveat
            else "random_baseline_gain"
        )
    if token_exposure_caveat:
        return "probe_valid_token_exposure_caveat"
    return "matched_baseline_inconclusive"


def _utility_failure_reason(
    *,
    probe_sensitivity_status: Dict[str, Any],
    curation_benefit_status: Dict[str, Any],
    strict_counterfactual_status: Dict[str, Any],
) -> str:
    if probe_sensitivity_status.get("probe_valid") is False:
        return "probe_not_evaluable"
    if strict_counterfactual_status.get("status") == "strict_certification_ready":
        return "pass"
    if strict_counterfactual_status.get("status") == "matched_baseline_gain":
        return "pass"
    token_exposure_caveat = bool(
        probe_sensitivity_status.get("token_exposure_confounded")
        or probe_sensitivity_status.get("token_exposure_inconclusive")
    )
    if curation_benefit_status.get("status") == "random_baseline_gain":
        return (
            "random_gain_only_with_token_exposure_caveat"
            if token_exposure_caveat
            else "random_gain_only"
        )
    if strict_counterfactual_status.get("status") == "matched_baseline_inconclusive":
        return "matched_inconclusive"
    if probe_sensitivity_status.get("selected_gt_random") is False:
        return "selected_below_stageA_random"
    return "strict_negative"


def _utility_protocol_summary(stage_c: Dict[str, Any], probe_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evaluation_mode": str(stage_c.get("evaluation_mode") or "development"),
        "certification_scope": str(stage_c.get("certification_scope") or "domain_specific"),
        "probe_mode": str(probe_cfg.get("mode") or "full"),
        "probe_model_name": str(probe_cfg.get("model_name") or ""),
        "utility_pass_statistic": str(stage_c.get("utility_pass_statistic") or "min"),
        "train_token_budget": int(probe_cfg.get("train_token_budget") or 0),
        "train_audit_token_budget": int(probe_cfg.get("train_audit_token_budget") or 0),
        "eval_token_budget": int(probe_cfg.get("eval_token_budget") or 0),
        "ood_eval_token_budget": int(probe_cfg.get("ood_eval_token_budget") or 0),
        "bootstrap_samples": int(probe_cfg.get("bootstrap_samples") or 0),
        "max_train_steps": int(probe_cfg.get("max_train_steps") or 0),
        "train_epochs": float(probe_cfg.get("train_epochs") or 0.0),
        "seed_count": int(len(probe_cfg.get("seeds") or [])),
        "seeds": list(probe_cfg.get("seeds") or []),
        "holdout_modulo": int(probe_cfg.get("holdout_modulo") or 0),
        "holdout_buckets": list(probe_cfg.get("holdout_buckets") or []),
        "ood_holdout_buckets": list(probe_cfg.get("ood_holdout_buckets") or []),
        "holdout_bucket_count": int(len(probe_cfg.get("holdout_buckets") or [])),
        "ood_holdout_bucket_count": int(len(probe_cfg.get("ood_holdout_buckets") or [])),
        "min_probe_bucket_count": int(probe_cfg.get("min_probe_bucket_count") or 0),
        "enforce_ood_utility_pass": bool(stage_c.get("enforce_ood_utility_pass")),
        "compute_ood_utility_report": bool(stage_c.get("compute_ood_utility_report")),
        "require_utility_ci_gain_positive": bool(stage_c.get("require_utility_ci_gain_positive")),
        "require_utility_delta_nll_positive": bool(stage_c.get("require_utility_delta_nll_positive")),
        "min_small_lm_probe_gain_score": float(stage_c.get("min_small_lm_probe_gain_score") or 0.0),
        "min_small_lm_probe_relative_gain": float(stage_c.get("min_small_lm_probe_relative_gain") or 0.0),
        "canonical_baseline": CANONICAL_UTILITY_BASELINE,
        "diagnostic_baselines": list(DIAGNOSTIC_UTILITY_BASELINES),
        "certification_requirements": {
            "stage_c": stage_c.get("certification_requirements") or {},
            "probe": probe_cfg.get("certification_requirements") or {},
        },
    }


def _utility_certification_shadow(
    *,
    stage_c: Dict[str, Any],
    probe_cfg: Dict[str, Any],
    utility_protocol: Dict[str, Any],
    in_domain_results: Dict[str, Dict[str, Any]],
    ood_results: Dict[str, Dict[str, Dict[str, Any]]] | None,
) -> Dict[str, Any]:
    stage_req = dict(stage_c.get("certification_requirements") or {})
    probe_req = dict(probe_cfg.get("certification_requirements") or {})
    protocol_blockers: List[str] = []
    signal_blockers: List[str] = []
    blocker_categories = {
        "protocol": protocol_blockers,
        "signal": signal_blockers,
    }

    canonical_in_domain = in_domain_results.get(CANONICAL_UTILITY_BASELINE)
    ood_by_dataset = ood_results or {}
    canonical_ood_by_dataset = {
        str(eval_dataset): baselines[CANONICAL_UTILITY_BASELINE]
        for eval_dataset, baselines in ood_by_dataset.items()
        if isinstance(baselines, dict) and isinstance(baselines.get(CANONICAL_UTILITY_BASELINE), dict)
    }
    min_score_threshold = float(stage_req.get("min_small_lm_probe_gain_score", 0.0))
    min_relative_threshold = float(stage_req.get("min_small_lm_probe_relative_gain", 0.0))
    delta_tolerance = max(0.0, float(stage_req.get("delta_nll_numerical_tolerance", 1e-5)))

    def strict_values(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {
                "min_small_lm_probe_gain_score": 0.0,
                "min_relative_nll_gain": 0.0,
                "min_delta_nll": 0.0,
                "min_delta_nll_ci_low": 0.0,
            }
        return {
            "min_small_lm_probe_gain_score": round(min(_utility_result_value(result, "small_lm_probe_gain_score", "min") for result in results), 6),
            "min_relative_nll_gain": round(min(_utility_result_value(result, "relative_nll_gain", "min") for result in results), 6),
            "min_delta_nll": round(min(_utility_result_value(result, "delta_nll", "min") for result in results), 6),
            "min_delta_nll_ci_low": round(min((_float_metric(result, "delta_nll_ci_low") for result in results), default=0.0), 6),
            "max_minimum_detectable_delta_nll_95": round(
                max((_float_metric(result, "minimum_detectable_delta_nll_95_max", fallback_key="minimum_detectable_delta_nll_95") for result in results), default=0.0),
                8,
            ),
            "min_effect_to_mde_ratio": round(
                min((_float_metric(result, "effect_to_mde_ratio_min", fallback_key="effect_to_mde_ratio") for result in results), default=0.0),
                6,
            ),
            "min_detectable_effect_fraction": round(
                min((_float_metric(result, "detectable_effect_fraction") for result in results), default=0.0),
                6,
            ),
        }

    def strict_passes(values: Dict[str, float]) -> Dict[str, bool]:
        return {
            "score_pass": bool(values["min_small_lm_probe_gain_score"] >= min_score_threshold and values["min_small_lm_probe_gain_score"] > 0.0),
            "relative_gain_pass": bool(values["min_relative_nll_gain"] >= min_relative_threshold),
            "delta_nll_pass": bool((values["min_delta_nll"] > 0.0) if bool(stage_req.get("require_utility_delta_nll_positive", True)) else True),
            "ci_pass": bool((values["min_delta_nll_ci_low"] > 0.0) if bool(stage_req.get("require_utility_ci_gain_positive", True)) else True),
            "detectable_effect_pass": bool(
                values["min_delta_nll"] > 0.0
                and (
                    values.get("max_minimum_detectable_delta_nll_95", 0.0) <= 0.0
                    or values["min_delta_nll"] > values.get("max_minimum_detectable_delta_nll_95", 0.0)
                    or values.get("min_effect_to_mde_ratio", 0.0) >= 1.0
                )
            ),
        }

    def worst_cell(scope: str, aggregate: Dict[str, Any] | None, *, eval_dataset: str | None = None) -> Dict[str, Any] | None:
        if not isinstance(aggregate, dict):
            return None
        runs = aggregate.get("per_bucket_runs") or []
        if not runs:
            return None
        worst = min(runs, key=lambda run: _utility_result_value(run, "small_lm_probe_gain_score", "mean"))
        return {
            "scope": scope,
            "baseline_variant": str(worst.get("baseline_variant") or aggregate.get("baseline_variant") or CANONICAL_UTILITY_BASELINE),
            "eval_dataset": str(worst.get("eval_dataset") or aggregate.get("eval_dataset") or ""),
            "seed": int(worst.get("bootstrap_seed") or 0),
            "holdout_bucket": int(worst.get("holdout_bucket") or 0),
            "small_lm_probe_gain_score": round(_utility_result_value(worst, "small_lm_probe_gain_score", "mean"), 6),
            "relative_nll_gain": round(_utility_result_value(worst, "relative_nll_gain", "mean"), 6),
            "delta_nll": round(_utility_result_value(worst, "delta_nll", "mean"), 6),
            "delta_nll_ci_low": round(_float_metric(worst, "delta_nll_ci_low"), 6),
            "baseline_nll": round(_float_metric(worst, "baseline_nll"), 6),
            "selected_nll": round(_float_metric(worst, "selected_nll"), 6),
            "selected_train_tokens": int(worst.get("selected_train_tokens") or 0),
            "baseline_train_tokens": int(worst.get("baseline_train_tokens") or 0),
            "pair": f"{scope}:{eval_dataset}" if eval_dataset else scope,
        }

    def scope_snapshot(scope: str, results_by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        results = [result for result in results_by_name.values() if isinstance(result, dict)]
        values = strict_values(results)
        passes = strict_passes(values)
        scope_blockers: List[str] = []
        if not results:
            scope_blockers.append(f"{scope}_missing_canonical_baseline")
        if not passes["score_pass"]:
            scope_blockers.append(f"{scope}_strict_min_gain_not_positive_or_below_threshold")
        if not passes["relative_gain_pass"]:
            scope_blockers.append(f"{scope}_strict_min_relative_gain_below_threshold")
        if not passes["delta_nll_pass"]:
            scope_blockers.append(f"{scope}_strict_min_delta_nll_not_positive")
        if not passes["ci_pass"]:
            scope_blockers.append(f"{scope}_strict_min_ci_low_not_positive")
        if not passes["detectable_effect_pass"]:
            scope_blockers.append(f"{scope}_effect_below_minimum_detectable_effect")
        worst_candidates = [
            worst_cell(scope, result, eval_dataset=name)
            for name, result in results_by_name.items()
            if isinstance(result, dict)
        ]
        worst_candidates = [candidate for candidate in worst_candidates if candidate is not None]
        worst = min(
            worst_candidates,
            key=lambda item: float(item.get("small_lm_probe_gain_score") or 0.0),
            default=None,
        )
        return {
            "strict_metric_pass": bool(all(passes.values()) and results),
            "strict_values": values,
            "strict_passes": passes,
            "worst_cell": worst,
            "blockers": scope_blockers,
            "numerical_drift_band": bool(abs(values["min_delta_nll"]) <= delta_tolerance),
            "result_count": int(len(results)),
            "eval_datasets": sorted(str(name) for name in results_by_name.keys()),
        }

    def legacy_scope_snapshot(scope: str, aggregate: Dict[str, Any] | None) -> Dict[str, Any]:
        results = [aggregate] if isinstance(aggregate, dict) else []
        values = strict_values(results)
        passes = strict_passes(values)
        scope_blockers: List[str] = []
        if not results:
            scope_blockers.append(f"{scope}_missing_canonical_baseline")
        if not passes["score_pass"]:
            scope_blockers.append(f"{scope}_strict_min_gain_not_positive_or_below_threshold")
        if not passes["relative_gain_pass"]:
            scope_blockers.append(f"{scope}_strict_min_relative_gain_below_threshold")
        if not passes["delta_nll_pass"]:
            scope_blockers.append(f"{scope}_strict_min_delta_nll_not_positive")
        if not passes["ci_pass"]:
            scope_blockers.append(f"{scope}_strict_min_ci_low_not_positive")
        if not passes["detectable_effect_pass"]:
            scope_blockers.append(f"{scope}_effect_below_minimum_detectable_effect")
        return {
            "strict_metric_pass": bool(all(passes.values()) and results),
            "strict_values": values,
            "strict_passes": passes,
            "worst_cell": worst_cell(scope, aggregate),
            "blockers": scope_blockers,
            "numerical_drift_band": bool(abs(values["min_delta_nll"]) <= delta_tolerance),
            "result_count": int(len(results)),
            "eval_datasets": [scope] if results else [],
        }

    in_domain_snapshot = legacy_scope_snapshot("in_domain", canonical_in_domain)
    ood_snapshot = scope_snapshot("ood", canonical_ood_by_dataset)

    if canonical_in_domain is None:
        signal_blockers.append("missing_canonical_in_domain_baseline")
        effective_results: List[Dict[str, Any]] = []
    else:
        effective_results = [canonical_in_domain]

    if bool(stage_req.get("enforce_ood_utility_pass", True)):
        if not canonical_ood_by_dataset:
            signal_blockers.append("missing_canonical_ood_baseline")
        else:
            effective_results.extend(canonical_ood_by_dataset.values())

    combined_values = strict_values(effective_results)
    combined_passes = strict_passes(combined_values)
    score_pass = combined_passes["score_pass"]
    relative_gain_pass = combined_passes["relative_gain_pass"]
    delta_nll_pass = combined_passes["delta_nll_pass"]
    ci_pass = combined_passes["ci_pass"]
    detectable_effect_pass = combined_passes["detectable_effect_pass"]

    if not score_pass:
        signal_blockers.append("strict_min_gain_not_positive_or_below_threshold")
    if not relative_gain_pass:
        signal_blockers.append("strict_min_relative_gain_below_threshold")
    if not delta_nll_pass:
        signal_blockers.append("strict_min_delta_nll_not_positive")
    if not ci_pass:
        signal_blockers.append("strict_min_ci_low_not_positive")
    if not detectable_effect_pass:
        signal_blockers.append("effect_below_minimum_detectable_effect")

    def _stability_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        diagnostics = [
            result.get("stability_diagnostics") or {}
            for result in results
            if isinstance(result, dict) and isinstance(result.get("stability_diagnostics"), dict)
        ]
        if not diagnostics:
            return {
                "available": False,
                "result_count": 0,
                "min_positive_run_fraction": None,
                "min_ci_positive_fraction": None,
                "min_mean_delta_nll_to_std_ratio": None,
                "max_delta_nll_std": None,
                "strict_min_negative_count": 0,
                "ci_crosses_zero_count": 0,
                "noise_dominated": None,
            }
        min_positive = min(float(d.get("positive_run_fraction") or 0.0) for d in diagnostics)
        min_ci_positive = min(float(d.get("ci_positive_fraction") or 0.0) for d in diagnostics)
        min_snr = min(float(d.get("mean_delta_nll_to_std_ratio") or 0.0) for d in diagnostics)
        max_std = max(float(d.get("delta_nll_std") or 0.0) for d in diagnostics)
        strict_negative_count = sum(1 for d in diagnostics if bool(d.get("strict_min_negative")))
        ci_cross_count = sum(1 for d in diagnostics if bool(d.get("ci_crosses_zero")))
        return {
            "available": True,
            "result_count": int(len(diagnostics)),
            "min_positive_run_fraction": round(min_positive, 6),
            "min_ci_positive_fraction": round(min_ci_positive, 6),
            "min_mean_delta_nll_to_std_ratio": round(min_snr, 6),
            "max_delta_nll_std": round(max_std, 8),
            "strict_min_negative_count": int(strict_negative_count),
            "ci_crosses_zero_count": int(ci_cross_count),
            "noise_dominated": bool(min_snr < 1.0 or min_positive < 0.75 or ci_cross_count > 0),
        }

    stability_analysis = {
        "in_domain": _stability_summary([canonical_in_domain] if canonical_in_domain else []),
        "ood": _stability_summary(list(canonical_ood_by_dataset.values())),
        "combined_effective": _stability_summary(effective_results),
    }

    def _step_cap_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected_count = sum(int(result.get("selected_step_cap_reached_count") or 0) for result in results if isinstance(result, dict))
        baseline_count = sum(int(result.get("baseline_step_cap_reached_count") or 0) for result in results if isinstance(result, dict))
        return {
            "selected_step_cap_reached_count": int(selected_count),
            "baseline_step_cap_reached_count": int(baseline_count),
            "step_cap_reached": bool(selected_count > 0 or baseline_count > 0),
        }

    step_cap_analysis = {
        "in_domain": _step_cap_summary([canonical_in_domain] if canonical_in_domain else []),
        "ood": _step_cap_summary(list(canonical_ood_by_dataset.values())),
        "combined_effective": _step_cap_summary(effective_results),
    }

    protocol_pass = True
    if str(stage_c.get("evaluation_mode") or "") != "certification":
        protocol_pass = False
        protocol_blockers.append("not_certification_mode")
    if str(stage_c.get("utility_pass_statistic") or "") != "min":
        protocol_pass = False
        protocol_blockers.append("pass_statistic_not_min")
    if not bool(stage_c.get("require_utility_ci_gain_positive")):
        protocol_pass = False
        protocol_blockers.append("ci_requirement_disabled")
    if not bool(stage_c.get("require_utility_delta_nll_positive")):
        protocol_pass = False
        protocol_blockers.append("delta_nll_requirement_disabled")
    if bool(stage_req.get("enforce_ood_utility_pass", True)) and not bool(stage_c.get("enforce_ood_utility_pass")):
        protocol_pass = False
        protocol_blockers.append("ood_pass_not_enforced")

    probe_protocol_pass = True
    probe_checks = {
        "train_token_budget": int(utility_protocol.get("train_token_budget") or 0) >= int(probe_req.get("train_token_budget") or 0),
        "eval_token_budget": int(utility_protocol.get("eval_token_budget") or 0) >= int(probe_req.get("eval_token_budget") or 0),
        "ood_eval_token_budget": int(utility_protocol.get("ood_eval_token_budget") or 0) >= int(probe_req.get("ood_eval_token_budget") or 0),
        "bootstrap_samples": int(utility_protocol.get("bootstrap_samples") or 0) >= int(probe_req.get("bootstrap_samples") or 0),
        "max_train_steps": int(utility_protocol.get("max_train_steps") or 0) >= int(probe_req.get("max_train_steps") or 0),
        "train_epochs": float(utility_protocol.get("train_epochs") or 0.0) >= float(probe_req.get("train_epochs") or 0.0),
        "seed_count": int(utility_protocol.get("seed_count") or 0) >= int(probe_req.get("seed_count") or 0),
        "min_probe_bucket_count": int(utility_protocol.get("min_probe_bucket_count") or 0) >= int(probe_req.get("min_probe_bucket_count") or 0),
        "holdout_bucket_count": int(utility_protocol.get("holdout_bucket_count") or 0) >= int(probe_req.get("min_probe_bucket_count") or 0),
        "ood_holdout_bucket_count": (
            int(utility_protocol.get("ood_holdout_bucket_count") or 0) >= int(probe_req.get("min_probe_bucket_count") or 0)
            if bool(stage_req.get("enforce_ood_utility_pass", True))
            else True
        ),
    }
    for check_name, passed in probe_checks.items():
        if not passed:
            probe_protocol_pass = False
            protocol_blockers.append(f"probe_{check_name}_below_certification_requirement")
    if bool((step_cap_analysis.get("combined_effective") or {}).get("step_cap_reached")):
        probe_protocol_pass = False
        protocol_blockers.append("probe_step_cap_reached_before_target_train_epochs")

    signal_blockers.extend(in_domain_snapshot["blockers"])
    signal_blockers.extend(ood_snapshot["blockers"])
    strict_metric_pass = bool(
        score_pass
        and relative_gain_pass
        and delta_nll_pass
        and ci_pass
        and detectable_effect_pass
        and effective_results
    )
    signal_pass = bool(in_domain_snapshot["strict_metric_pass"] and ood_snapshot["strict_metric_pass"])

    def _scope_signal_status(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        values = snapshot.get("strict_values") or {}
        passes = snapshot.get("strict_passes") or {}
        min_gain = float(values.get("min_small_lm_probe_gain_score") or 0.0)
        min_rel_gain = float(values.get("min_relative_nll_gain") or 0.0)
        min_delta = float(values.get("min_delta_nll") or 0.0)
        min_ci_low = float(values.get("min_delta_nll_ci_low") or 0.0)
        min_effect_to_mde = float(values.get("min_effect_to_mde_ratio") or 0.0)
        if bool(snapshot.get("strict_metric_pass")):
            status = "strict_positive"
            reason = "Strict min gain, relative gain, delta-NLL, and CI lower bound are all positive."
        elif bool(snapshot.get("numerical_drift_band")):
            status = "inconclusive_numerical_drift"
            reason = "Worst-cell delta-NLL is within the configured numerical drift band, so this is not confirmed negative evidence."
        elif min_delta > 0.0 and min_effect_to_mde < 1.0:
            status = "inconclusive_below_detectable_effect"
            reason = "Strict delta-NLL is positive, but the observed effect is below the current paired-bootstrap minimum detectable effect."
        elif min_delta > 0.0 and min_ci_low <= 0.0:
            status = "inconclusive_ci_crosses_zero"
            reason = "Strict delta-NLL is positive, but the bootstrap CI lower bound crosses zero."
        elif min_delta <= 0.0:
            status = "strict_negative"
            reason = "Worst-cell delta-NLL is negative outside the numerical drift band."
        else:
            status = "inconclusive_threshold"
            reason = "Signal is positive on some dimensions but does not satisfy every strict certification threshold."
        return {
            "status": status,
            "reason": reason,
            "strict_metric_pass": bool(snapshot.get("strict_metric_pass")),
            "numerical_drift_band": bool(snapshot.get("numerical_drift_band")),
            "score_pass": bool(passes.get("score_pass")),
            "relative_gain_pass": bool(passes.get("relative_gain_pass")),
            "delta_nll_pass": bool(passes.get("delta_nll_pass")),
            "ci_pass": bool(passes.get("ci_pass")),
            "detectable_effect_pass": bool(passes.get("detectable_effect_pass")),
            "min_small_lm_probe_gain_score": round(min_gain, 6),
            "min_relative_nll_gain": round(min_rel_gain, 6),
            "min_delta_nll": round(min_delta, 6),
            "min_delta_nll_ci_low": round(min_ci_low, 6),
            "min_effect_to_mde_ratio": round(min_effect_to_mde, 6),
            "max_minimum_detectable_delta_nll_95": round(float(values.get("max_minimum_detectable_delta_nll_95") or 0.0), 8),
            "min_detectable_effect_fraction": round(float(values.get("min_detectable_effect_fraction") or 0.0), 6),
        }

    in_domain_signal_status = _scope_signal_status(in_domain_snapshot)
    ood_signal_status = _scope_signal_status(ood_snapshot)
    combined_signal_status = _scope_signal_status(
        {
            "strict_metric_pass": strict_metric_pass,
            "strict_values": combined_values,
            "strict_passes": combined_passes,
            "numerical_drift_band": bool(abs(combined_values["min_delta_nll"]) <= delta_tolerance),
        }
    )
    if (
        in_domain_signal_status["status"].startswith("inconclusive")
        or ood_signal_status["status"].startswith("inconclusive")
        or combined_signal_status["status"].startswith("inconclusive")
    ):
        signal_blockers.append("strict_signal_inconclusive_not_confirmed_negative")

    certification_ready = bool(signal_pass and protocol_pass and probe_protocol_pass)
    evidence_tier = "development_only"
    if bool(in_domain_snapshot["strict_metric_pass"]):
        evidence_tier = "in_domain_strict_signal"
    if bool(in_domain_snapshot["strict_metric_pass"]) and bool(ood_snapshot["strict_metric_pass"]):
        evidence_tier = "cross_domain_strict_signal"
    if certification_ready:
        evidence_tier = "certification_ready"
    protocol_readiness = {
        "pass": bool(protocol_pass and probe_protocol_pass),
        "stage_c_protocol_pass": bool(protocol_pass),
        "probe_protocol_pass": bool(probe_protocol_pass),
        "blockers": sorted(set(protocol_blockers)),
        "checks": {
            "evaluation_mode_certification": str(stage_c.get("evaluation_mode") or "") == "certification",
            "utility_pass_statistic_min": str(stage_c.get("utility_pass_statistic") or "") == "min",
            "ci_requirement_enabled": bool(stage_c.get("require_utility_ci_gain_positive")),
            "delta_nll_requirement_enabled": bool(stage_c.get("require_utility_delta_nll_positive")),
            "ood_pass_enforced": bool(stage_c.get("enforce_ood_utility_pass")),
            "probe_step_cap_not_reached": not bool((step_cap_analysis.get("combined_effective") or {}).get("step_cap_reached")),
            **{f"probe_{name}": bool(passed) for name, passed in probe_checks.items()},
        },
        "step_cap_analysis": step_cap_analysis,
    }
    in_domain_signal = {
        **in_domain_snapshot,
        "pass": bool(in_domain_snapshot["strict_metric_pass"]),
        "blockers": sorted(set(in_domain_snapshot["blockers"])),
    }
    ood_signal = {
        **ood_snapshot,
        "pass": bool(ood_snapshot["strict_metric_pass"]),
        "blockers": sorted(set(ood_snapshot["blockers"])),
    }
    protocol_ready = bool(protocol_pass and probe_protocol_pass)
    in_domain_certification_ready = bool(protocol_ready and in_domain_signal["pass"])
    cross_domain_certification_ready = bool(protocol_ready and ood_signal["pass"])
    domain_specific_certification_ready = bool(in_domain_certification_ready)
    general_purpose_certification_ready = bool(
        protocol_ready
        and in_domain_signal["pass"]
        and ood_signal["pass"]
    )
    all_blockers = sorted(set(protocol_blockers + signal_blockers))
    return {
        "certification_ready": certification_ready,
        "in_domain_certification_ready": in_domain_certification_ready,
        "cross_domain_certification_ready": cross_domain_certification_ready,
        "domain_specific_certification_ready": domain_specific_certification_ready,
        "general_purpose_certification_ready": general_purpose_certification_ready,
        "strict_metric_pass": strict_metric_pass,
        "signal_pass": signal_pass,
        "protocol_pass": bool(protocol_pass),
        "probe_protocol_pass": bool(probe_protocol_pass),
        "evidence_tier": evidence_tier,
        "blockers": all_blockers,
        "blocker_categories": {k: sorted(set(v)) for k, v in blocker_categories.items()},
        "protocol_readiness": protocol_readiness,
        "in_domain_signal": in_domain_signal,
        "ood_signal": ood_signal,
        "requirements": {
            "stage_c": stage_req,
            "probe": probe_req,
        },
        "strict_values": combined_values,
        "strict_passes": combined_passes,
        "scope_snapshots": {
            "in_domain": in_domain_snapshot,
            "ood": ood_snapshot,
        },
        "stability_analysis": stability_analysis,
        "step_cap_analysis": step_cap_analysis,
        "signal_interpretation": {
            "combined": combined_signal_status,
            "in_domain": in_domain_signal_status,
            "ood": ood_signal_status,
            "delta_nll_numerical_tolerance": round(float(delta_tolerance), 8),
            "interpretation_policy": (
                "Strict positive evidence can support certification; numerical-drift or CI-crossing cells are "
                "reported as inconclusive rather than optimized away or treated as confirmed negative evidence."
            ),
        },
        "worst_cells": {
            "in_domain": in_domain_snapshot.get("worst_cell"),
            "ood": ood_snapshot.get("worst_cell"),
        },
        "probe_protocol_checks": probe_checks,
    }


def _score_with_probe_buckets(
    conn: sqlite3.Connection,
    *,
    context_cache: Dict[Tuple[Any, ...], Any],
    selected_sequence_cache: Dict[Tuple[Any, ...], tuple[List[List[int]], int, int, float]],
    selected_records: List[Dict[str, Any]],
    text_map: Dict[str, str],
    baseline_variant: str,
    baseline_allowed_uids: set[str] | None,
    baseline_uid_fingerprint: str,
    train_dataset: str,
    eval_dataset: str,
    probe_cfg: Dict[str, Any],
    eval_token_budget: int,
    holdout_buckets: List[int],
    progress_label: str | None = None,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    selected_pairs = [(r["chunk_uid"], text_map.get(r["chunk_uid"], "")) for r in selected_records]
    selected_hasher = hashlib.sha1()
    for record in selected_records:
        selected_hasher.update(str(record["chunk_uid"]).encode("utf-8", errors="replace"))
    selected_fingerprint = selected_hasher.hexdigest()
    if str(probe_cfg.get("mode") or "") == "synthetic_smoke":
        baseline_nll = 5.0
        base_delta = 0.02 if str(train_dataset) == str(eval_dataset) else 0.012
        variant_penalty = {
            CANONICAL_UTILITY_BASELINE: 0.0,
            "baseline_multi_matched_stageA_random": 0.0,
            "baseline_quality_band_matched_stageA_random": 0.0005,
            "baseline_stageA_random": 0.003,
            "baseline_full_random": 0.004,
            "baseline_style_matched_stageA_random": 0.001,
            "baseline_length_matched_stageA_random": 0.0015,
        }.get(str(baseline_variant), 0.002)
        delta_nll = max(0.001, base_delta - variant_penalty)
        ci_low = max(0.0005, delta_nll * 0.65)
        ci_high = delta_nll * 1.35
        mde = max(0.00025, delta_nll * 0.25)
        for bucket in holdout_buckets:
            for bootstrap_seed in list(probe_cfg.get("seeds") or [int(probe_cfg["seed"])]):
                selected_nll = baseline_nll - delta_nll
                runs.append(
                    {
                        "baseline_variant": str(baseline_variant),
                        "mode": "synthetic_smoke",
                        "train_dataset": str(train_dataset),
                        "eval_dataset": str(eval_dataset),
                        "bootstrap_seed": int(bootstrap_seed),
                        "holdout_bucket": int(bucket),
                        "small_lm_probe_gain_score": round(delta_nll / baseline_nll, 6),
                        "fixed_token_probe_gain_score": round(delta_nll / baseline_nll, 6),
                        "relative_nll_gain": round(delta_nll / baseline_nll, 6),
                        "delta_nll": round(delta_nll, 6),
                        "delta_nll_ci_low": round(ci_low, 6),
                        "delta_nll_ci_high": round(ci_high, 6),
                        "baseline_nll": round(baseline_nll, 6),
                        "selected_nll": round(selected_nll, 6),
                        "eval_docs": 8,
                        "eval_tokens": int(eval_token_budget),
                        "selected_train_tokens": int(probe_cfg["train_token_budget"]),
                        "baseline_train_tokens": int(probe_cfg["train_token_budget"]),
                        "train_audit_token_budget": int(probe_cfg.get("train_audit_token_budget") or 0),
                        "selected_train_audit_tokens": int(probe_cfg.get("train_audit_token_budget") or 0),
                        "baseline_train_audit_tokens": int(probe_cfg.get("train_audit_token_budget") or 0),
                        "selected_train_audit_pre_nll": round(baseline_nll, 6),
                        "selected_train_audit_post_nll": round(selected_nll, 6),
                        "selected_train_audit_delta_nll": round(delta_nll, 6),
                        "selected_train_audit_relative_gain": round(delta_nll / baseline_nll, 6),
                        "baseline_train_audit_pre_nll": round(baseline_nll, 6),
                        "baseline_train_audit_post_nll": round(baseline_nll - (delta_nll * 0.5), 6),
                        "baseline_train_audit_delta_nll": round(delta_nll * 0.5, 6),
                        "baseline_train_audit_relative_gain": round((delta_nll * 0.5) / baseline_nll, 6),
                        "selected_minus_baseline_train_audit_delta_nll": round(delta_nll * 0.5, 6),
                        "causal_failure_mode": "positive_learning_signal",
                        "selected_effective_train_steps": int(probe_cfg["max_train_steps"]),
                        "baseline_effective_train_steps": int(probe_cfg["max_train_steps"]),
                        "selected_target_train_steps": int(probe_cfg["max_train_steps"]),
                        "baseline_target_train_steps": int(probe_cfg["max_train_steps"]),
                        "selected_one_epoch_train_steps": int(probe_cfg["max_train_steps"]),
                        "baseline_one_epoch_train_steps": int(probe_cfg["max_train_steps"]),
                        "selected_estimated_seen_train_tokens": int(probe_cfg["train_token_budget"]),
                        "baseline_estimated_seen_train_tokens": int(probe_cfg["train_token_budget"]),
                        "selected_train_token_exposure_ratio": 1.0,
                        "baseline_train_token_exposure_ratio": 1.0,
                        "selected_target_train_exposure_ratio": 1.0,
                        "baseline_target_train_exposure_ratio": 1.0,
                        "train_epochs": float(probe_cfg.get("train_epochs") or 1.0),
                        "selected_step_cap_reached": False,
                        "baseline_step_cap_reached": False,
                        "paired_bootstrap": True,
                        "eval_pairing_policy": "paired_same_eval_documents",
                        "paired_bootstrap_delta_nll_std": round(mde / 1.96, 8),
                        "minimum_detectable_delta_nll_95": round(mde, 8),
                        "minimum_detectable_relative_gain_95": round(mde / baseline_nll, 8),
                        "effect_to_mde_ratio": round(delta_nll / mde, 6),
                        "detectable_effect": True,
                        "selected_fingerprint": selected_fingerprint,
                    }
                )
        return aggregate_probe_runs(
            runs,
            mode="synthetic_smoke_multi_bucket",
            train_dataset=str(train_dataset),
            eval_dataset=str(eval_dataset),
        )
    total_probe_runs = len(holdout_buckets) * len(list(probe_cfg.get("seeds") or [int(probe_cfg["seed"])]))
    completed_probe_runs = 0
    for bucket in holdout_buckets:
        for bootstrap_seed in list(probe_cfg.get("seeds") or [int(probe_cfg["seed"])]):
            completed_probe_runs += 1
            run_label = (
                f"{progress_label or f'{train_dataset}->{eval_dataset}'} "
                f"baseline={baseline_variant} bucket={int(bucket)} seed={int(bootstrap_seed)} "
                f"run={completed_probe_runs}/{total_probe_runs}"
            )
            run_started = time.perf_counter()
            _progress(f"utility probe start: {run_label}")
            # Vary train/eval sampling across probe seeds as well as model initialization.
            # Otherwise the min statistic can be dominated by one fixed random baseline draw.
            run_sampling_hash_seed = (
                int(probe_cfg["sampling_hash_seed"])
                + (int(bootstrap_seed) * 1009)
                + (int(bucket) * 9176)
            )
            context_key = (
                baseline_variant,
                baseline_uid_fingerprint,
                train_dataset,
                eval_dataset,
                probe_cfg["train_token_budget"],
                eval_token_budget,
                probe_cfg["holdout_modulo"],
                int(bucket),
                probe_cfg["model_name"],
                probe_cfg["max_length"],
                probe_cfg["train_batch_size"],
                probe_cfg["eval_batch_size"],
                probe_cfg["learning_rate"],
                probe_cfg["max_train_steps"],
                probe_cfg["train_epochs"],
                probe_cfg.get("train_audit_token_budget", 0),
                run_sampling_hash_seed,
            )
            context = context_cache.get(context_key)
            if context is None:
                context_started = time.perf_counter()
                _progress(f"utility context build start: {run_label}")
                context = build_probe_context(
                    conn,
                    baseline_variant=str(baseline_variant),
                    baseline_allowed_uids=baseline_allowed_uids,
                    baseline_uid_fingerprint=str(baseline_uid_fingerprint),
                    dataset=str(train_dataset),
                    eval_dataset=str(eval_dataset),
                    token_budget=int(probe_cfg["train_token_budget"]),
                    eval_token_budget=int(eval_token_budget),
                    holdout_modulo=int(probe_cfg["holdout_modulo"]),
                    holdout_bucket=int(bucket),
                    model_name=str(probe_cfg["model_name"]),
                    max_length=int(probe_cfg["max_length"]),
                    train_batch_size=int(probe_cfg["train_batch_size"]),
                    eval_batch_size=int(probe_cfg["eval_batch_size"]),
                    learning_rate=float(probe_cfg["learning_rate"]),
                    max_train_steps=int(probe_cfg["max_train_steps"]),
                    train_epochs=float(probe_cfg["train_epochs"]),
                    train_audit_token_budget=int(probe_cfg.get("train_audit_token_budget") or 0),
                    sampling_hash_seed=int(run_sampling_hash_seed),
                )
                context_cache[context_key] = context
                _progress(f"utility context build done: {run_label} elapsed={_elapsed_seconds(context_started)}")
            else:
                _progress(f"utility context cache hit: {run_label}")
            score_started = time.perf_counter()
            _progress(f"utility train/eval start: {run_label}")
            run = score_selected_records(
                context,
                selected_pairs,
                bootstrap_rounds=int(probe_cfg["bootstrap_samples"]),
                seed=int(bootstrap_seed),
                selected_fingerprint=selected_fingerprint,
                selected_sequence_cache=selected_sequence_cache,
            )
            runs.append(run)
            _progress(
                "utility probe done: "
                f"{run_label} gain={float(run.get('small_lm_probe_gain_score') or 0.0):.6f} "
                f"delta_nll={float(run.get('delta_nll') or 0.0):.6f} "
                f"score_elapsed={_elapsed_seconds(score_started)} total_elapsed={_elapsed_seconds(run_started)}"
            )

    if len(runs) < int(probe_cfg.get("min_probe_bucket_count") or 1):
        raise RuntimeError(
            f"{train_dataset}->{eval_dataset}: probe bucket runs insufficient "
            f"(required={probe_cfg.get('min_probe_bucket_count')}, got={len(runs)})"
        )
    return aggregate_probe_runs(
        runs,
        mode=f"{probe_cfg.get('mode', 'full')}_multi_bucket",
        train_dataset=str(train_dataset),
        eval_dataset=str(eval_dataset),
    )


def _fetch_texts(conn: sqlite3.Connection, chunk_uids: List[str]) -> Dict[str, str]:
    if not chunk_uids:
        return {}
    out: Dict[str, str] = {}
    batch_size = 800
    for i in range(0, len(chunk_uids), batch_size):
        batch = chunk_uids[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT chunk_uid, text FROM chunks WHERE chunk_uid IN ({placeholders})",
            batch,
        ).fetchall()
        for chunk_uid, text in rows:
            out[str(chunk_uid)] = str(text)
    return out


_LEARNING_SIGNAL_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "you",
    "your",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "will",
    "they",
    "their",
    "there",
    "which",
    "when",
    "what",
    "how",
    "into",
    "than",
    "then",
    "about",
    "also",
}


def _learning_signal_tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", str(text).lower())
        if token not in _LEARNING_SIGNAL_STOPWORDS
    ]


def _stable_sample_records(
    records: List[Dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    allowed_uids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    sample_size = max(0, int(sample_size))
    if sample_size <= 0:
        return []
    filtered = [
        record
        for record in records
        if allowed_uids is None or str(record.get("chunk_uid") or "") in allowed_uids
    ]
    filtered.sort(
        key=lambda r: (
            hashlib.sha256(f"{seed}:{r.get('chunk_uid')}".encode("utf-8")).hexdigest(),
            str(r.get("chunk_uid") or ""),
        )
    )
    return filtered[:sample_size]


def _learning_signal_summary(records: List[Dict[str, Any]], text_map: Dict[str, str]) -> Dict[str, Any]:
    if not records:
        return {
            "sample_records": 0,
            "token_count": 0,
            "unique_token_ratio": 0.0,
            "unique_bigram_ratio": 0.0,
            "rare_token_density": 0.0,
            "concept_density": 0.0,
            "moderate_difficulty_share": 0.0,
            "template_density": 0.0,
            "mean_quality": 0.0,
            "mean_redundancy_risk": 0.0,
            "mean_predictive_utility_proxy": 0.0,
        }
    total_tokens = 0
    unique_tokens: set[str] = set()
    total_bigrams = 0
    unique_bigrams: set[tuple[str, str]] = set()
    rare_token_hits = 0
    concept_hits = 0
    template_hits = 0
    moderate_count = 0
    quality_sum = 0.0
    redundancy_sum = 0.0
    predictive_sum = 0.0
    for record in records:
        uid = str(record.get("chunk_uid") or "")
        text = str(text_map.get(uid) or ((record.get("provenance") or {}).get("text_preview") or ""))
        tokens = _learning_signal_tokens(text)
        token_count = len(tokens)
        total_tokens += token_count
        unique_tokens.update(tokens)
        bigrams = list(zip(tokens, tokens[1:]))
        total_bigrams += len(bigrams)
        unique_bigrams.update(bigrams)
        rare_token_hits += sum(1 for token in tokens if len(token) >= 9)
        concept_hits += sum(
            1
            for token in tokens
            if len(token) >= 6
            and (
                token.endswith("tion")
                or token.endswith("ment")
                or token.endswith("ity")
                or token.endswith("ism")
                or token.endswith("ics")
                or token.endswith("ive")
            )
        )
        q = _quality_score_from_scored_record(record)
        redundancy = float(((record.get("core_metrics") or {}).get("shingle_near_duplicate_risk_score") or {}).get("score") or 0.0)
        q_details = (((record.get("core_metrics") or {}).get("reference_quality_score") or {}).get("details") or {})
        predictive = float(((record.get("diagnostic_metrics") or {}).get("predictive_utility_proxy") or {}).get("score") or 0.0)
        repeat = repeated_token_ratio(text)
        boilerplate_hits = int(q_details.get("boilerplate_hits") or 0)
        template_hits += int(boilerplate_hits > 0 or repeat >= 0.36)
        moderate_count += int(0.78 <= q <= 0.97 and 0.02 <= redundancy <= 0.38 and token_count >= 32)
        quality_sum += q
        redundancy_sum += redundancy
        predictive_sum += predictive
    n = max(len(records), 1)
    token_den = max(total_tokens, 1)
    bigram_den = max(total_bigrams, 1)
    return {
        "sample_records": int(len(records)),
        "token_count": int(total_tokens),
        "unique_token_ratio": round(float(len(unique_tokens)) / float(token_den), 6),
        "unique_bigram_ratio": round(float(len(unique_bigrams)) / float(bigram_den), 6),
        "rare_token_density": round(float(rare_token_hits) / float(token_den), 6),
        "concept_density": round(float(concept_hits) / float(token_den), 6),
        "moderate_difficulty_share": round(float(moderate_count) / float(n), 6),
        "template_density": round(float(template_hits) / float(n), 6),
        "mean_quality": round(float(quality_sum) / float(n), 6),
        "mean_redundancy_risk": round(float(redundancy_sum) / float(n), 6),
        "mean_predictive_utility_proxy": round(float(predictive_sum) / float(n), 6),
    }


def _learning_signal_gap_diagnostic(
    *,
    conn: sqlite3.Connection,
    selected_records: List[Dict[str, Any]],
    selected_text_map: Dict[str, str],
    baseline_records: List[Dict[str, Any]],
    baseline_allowed_uids: set[str],
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    selected_sample = _stable_sample_records(selected_records, sample_size=sample_size, seed=seed)
    baseline_sample = _stable_sample_records(
        baseline_records,
        sample_size=sample_size,
        seed=seed + 104729,
        allowed_uids=baseline_allowed_uids,
    )
    baseline_text_map = _fetch_texts(conn, [str(record["chunk_uid"]) for record in baseline_sample])
    selected_summary = _learning_signal_summary(selected_sample, selected_text_map)
    baseline_summary = _learning_signal_summary(baseline_sample, baseline_text_map)
    gap_fields = (
        "unique_token_ratio",
        "unique_bigram_ratio",
        "rare_token_density",
        "concept_density",
        "moderate_difficulty_share",
        "template_density",
        "mean_quality",
        "mean_redundancy_risk",
        "mean_predictive_utility_proxy",
    )
    gaps = {
        field: round(float(selected_summary.get(field) or 0.0) - float(baseline_summary.get(field) or 0.0), 6)
        for field in gap_fields
    }
    risk_flags = []
    if gaps["unique_bigram_ratio"] < -0.02:
        risk_flags.append("selected_lower_phrase_novelty")
    if gaps["concept_density"] < -0.005:
        risk_flags.append("selected_lower_concept_density")
    if gaps["moderate_difficulty_share"] < -0.05:
        risk_flags.append("selected_lower_moderate_difficulty")
    if gaps["template_density"] > 0.02:
        risk_flags.append("selected_higher_template_density")
    if gaps["mean_quality"] > 0.04 and gaps["unique_bigram_ratio"] <= 0.0:
        risk_flags.append("quality_gain_without_phrase_novelty_gain")
    return {
        "policy": "diagnostic_only_not_selector_objective",
        "baseline": CANONICAL_UTILITY_BASELINE,
        "sample_size_target": int(sample_size),
        "selected": selected_summary,
        "baseline": baseline_summary,
        "gaps_selected_minus_baseline": gaps,
        "risk_flags": risk_flags,
    }


def generate_subsets(
    profiles_path: Path = DEFAULT_PROFILE_CONFIG,
    scoring_manifest_path: Path = SCORING_MANIFEST_PATH,
    index_db_path: Path = INDEX_DB_PATH,
    dataset_names: List[str] | None = None,
) -> Dict[str, Any]:
    profiles = load_profiles(profiles_path)
    scoring_manifest = load_json(scoring_manifest_path)
    conn = sqlite3.connect(str(index_db_path))
    cluster_counts: Dict[str, Counter[int]] | None = None

    profile_summaries: Dict[str, Any] = {}
    utility_probe_context_cache: Dict[Tuple[Any, ...], Any] = {}
    utility_selected_sequence_cache: Dict[Tuple[Any, ...], tuple[List[List[int]], int, int, float]] = {}
    utility_probe_payload: Dict[str, Any] = {
        "schema_version": "small-lm-probe-v1",
        "profiles_path": str(profiles_path),
        "index_db_path": str(index_db_path),
        "datasets": {},
        "profiles": {},
    }

    for profile_name, profile in profiles["profiles"].items():
        profile_started = time.perf_counter()
        profile_dir = SUBSETS_DIR / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        stage_a = _stage_a_gate(profile)
        stage_b_profile = _stage_b_rank(profile)
        stage_c = _stage_c_validation(profile)
        probe_cfg = _utility_probe_config(profile, evaluation_mode=str(stage_c.get("evaluation_mode") or "development"))
        selector_cfg = _selector_config(profile)
        runtime_limits = _runtime_limits(profile)
        if "evaluation_mode" not in (profile.get("stage_c_validation") or {}) and bool(probe_cfg.get("dual_eval_required", True)):
            stage_c["enforce_ood_utility_pass"] = True
            stage_c["require_dual_eval_pass"] = True
        available_dataset_names = sorted(scoring_manifest["datasets"].keys())
        active_dataset_names = [str(name) for name in dataset_names] if dataset_names else available_dataset_names
        missing_dataset_names = sorted(set(active_dataset_names) - set(available_dataset_names))
        if missing_dataset_names:
            raise ValueError(f"Requested datasets are not present in the scoring manifest: {missing_dataset_names}")
        _progress(
            f"profile start: {profile_name} datasets={len(active_dataset_names)} "
            f"mode={stage_c.get('evaluation_mode')} utility_mode={probe_cfg.get('mode')}"
        )
        profile_summary = {
            "selection_threshold": float(stage_b_profile["selection_threshold"]),
            "weights": dict(stage_b_profile["weights"]),
            "metric_floors": dict(stage_a["metric_floors"]),
            "metric_ceilings": dict(stage_a["metric_ceilings"]),
            "stage_a_gate": stage_a,
            "stage_b_rank": stage_b_profile,
            "stage_c_validation": stage_c,
            "utility_probe": probe_cfg,
            "selector": selector_cfg,
            "datasets": {},
        }

        for dataset in active_dataset_names:
            dataset_meta = scoring_manifest["datasets"][dataset]
            dataset_started = time.perf_counter()
            selected: List[Dict[str, Any]] = []
            selected_clusters: Counter[int] = Counter()
            source_path = Path(dataset_meta["path"])
            total_records = int(dataset_meta["records"])
            max_records = int(runtime_limits.get("max_records_per_dataset") or 0)
            source_records_for_profile = min(total_records, max_records) if max_records > 0 else total_records
            runtime_limited = max_records > 0
            _progress(
                f"dataset start: profile={profile_name} dataset={dataset} "
                f"records={source_records_for_profile} original_records={total_records}"
            )
            if runtime_limited:
                original_clusters = Counter()
            else:
                _progress(f"dataset references start: profile={profile_name} dataset={dataset}")
                if cluster_counts is None:
                    cluster_counts = _dataset_cluster_counts(conn)
                original_clusters = cluster_counts.get(dataset, Counter())
            original_domain_counts = Counter() if runtime_limited else _dataset_domain_counts(conn, dataset=str(dataset))
            original_style_counts = Counter() if runtime_limited else _dataset_style_counts(conn, dataset=str(dataset))
            _progress(
                f"dataset references done: profile={profile_name} dataset={dataset} "
                f"clusters={len(original_clusters)} domain_buckets={len(original_domain_counts)} "
                f"style_buckets={len(original_style_counts)}"
            )
            processed_records = 0
            stage_b = dict(stage_b_profile)
            selector_diagnostics: Dict[str, Any] | None = None
            risk_quantile = stage_b.get("near_duplicate_risk_quantile_ceiling")
            if risk_quantile is not None:
                quantile_started = time.perf_counter()
                _progress(f"near-dup quantile start: profile={profile_name} dataset={dataset} q={risk_quantile}")
                quantile_ceiling = _estimate_metric_quantile(
                    source_path,
                    "shingle_near_duplicate_risk_score",
                    float(risk_quantile),
                    sample_size=int(stage_b.get("near_duplicate_risk_quantile_sample_size") or 60000),
                    seed=int(probe_cfg.get("seed") or 42),
                )
                stage_b["near_duplicate_risk_ceiling"] = min(
                    float(stage_b["near_duplicate_risk_ceiling"]),
                    float(quantile_ceiling),
                )
                stage_b["near_duplicate_risk_quantile_ceiling_value"] = round(float(quantile_ceiling), 6)
                _progress(
                    f"near-dup quantile done: profile={profile_name} dataset={dataset} "
                    f"value={float(quantile_ceiling):.6f} elapsed={_elapsed_seconds(quantile_started)}"
                )
            candidates: List[Dict[str, Any]] = []
            stage_a_baseline_records: List[Dict[str, Any]] = []
            all_dataset_uids: set[str] = set()
            stage_a_exactdup_uids: set[str] = set()
            scan_started = time.perf_counter()
            for record in tqdm(
                _iter_scored_records(source_path),
                total=source_records_for_profile,
                desc=f"[04] {profile_name}:{dataset}",
                unit="chunk",
                disable=bool(runtime_limits.get("disable_progress")),
            ):
                if max_records > 0 and processed_records >= max_records:
                    break
                processed_records += 1
                if runtime_limited:
                    original_clusters[_cluster_id(record)] += 1
                    original_domain_counts[_domain_bucket_from_scored_record(record)] += 1
                    original_style_counts[_style_bucket_from_scored_record(record)] += 1
                all_dataset_uids.add(str(record["chunk_uid"]))
                stage_a_pass = _passes_gates(record, profile)
                stage_a_baseline_pass = _passes_stage_a_validity_exactdup(record, stage_a)
                record["selection"] = {
                    "profile": profile_name,
                    "axis_scores": _axis_scores(record),
                    "stage_a_gate_passed": bool(stage_a_pass),
                    "stage_b_rank_score": None,
                    "stage_b_rank_passed": False,
                    "accepted": False,
                    "accepted_by": None,
                }
                if stage_a_baseline_pass:
                    uid = str(record["chunk_uid"])
                    stage_a_exactdup_uids.add(uid)
                    stage_a_baseline_records.append(record)
                if not stage_a_pass:
                    continue
                candidates.append(record)
            _progress(
                f"chunk scan done: profile={profile_name} dataset={dataset} "
                f"processed={processed_records} candidates={len(candidates)} "
                f"stageA_baseline={len(stage_a_baseline_records)} elapsed={_elapsed_seconds(scan_started)}"
            )

            selection_started = time.perf_counter()
            _progress(f"selection start: profile={profile_name} dataset={dataset} candidates={len(candidates)}")
            strategy = _coverage_strategy(profile, original_clusters)
            cluster_backbone_audit = (
                {
                    "passed": True,
                    "runtime_limited": True,
                    "coherence_proxy": 1.0,
                    "separation_margin": 1.0,
                    "style_purity_proxy": 1.0,
                    "domain_purity_proxy": 1.0,
                    "cluster_count": int(len(original_clusters)),
                    "sample_scope": "runtime_limited_scored_records",
                }
                if runtime_limited
                else _cluster_backbone_audit(
                    conn,
                    dataset=str(dataset),
                    original_clusters=original_clusters,
                    seed=int(probe_cfg.get("seed") or 42),
                )
            )
            _progress(
                f"coverage backbone done: profile={profile_name} dataset={dataset} "
                f"passed={bool(cluster_backbone_audit.get('passed'))}"
            )
            stage_b_budget = _resolve_stage_b_budget(
                profile,
                total_word_count=sum(int(record.get("word_count") or 0) for record in candidates),
            )
            if stage_b_budget.binding:
                selected, selector_diagnostics = _select_with_objective_constraints(
                    candidates=candidates,
                    stage_b=stage_b,
                    selector_cfg=selector_cfg,
                    strategy=strategy,
                    original_clusters=original_clusters,
                    source_records=source_records_for_profile,
                    stage_c=stage_c,
                )
                selected = fit_word_budget(
                    selected,
                    word_count=lambda record: int(record.get("word_count") or 0),
                    word_limit=int(stage_b_budget.word_limit or 0),
                )
                selector_diagnostics["budget_binding"] = True
                selector_diagnostics["word_limit"] = stage_b_budget.word_limit
                selector_diagnostics["selected_word_count_after_budget"] = sum(
                    int(record.get("word_count") or 0) for record in selected
                )
            else:
                selected = list(candidates)
                for record in selected:
                    record["selection"]["accepted"] = True
                    record["selection"]["accepted_by"] = "retain_all_no_binding_budget"
                selector_diagnostics = {
                    "selection_mode": "retain_all",
                    "budget_binding": False,
                    "word_limit": None,
                    "iterations": [],
                    "coverage_constraints_satisfied": True,
                    "selector_constraints_satisfied": True,
                    "constraint_violations": {},
                }
            selected_uids = {str(record["chunk_uid"]) for record in selected}
            curated_pool_records = annotate_retained_pool(
                candidates,
                selected_ids=selected_uids,
                budget_applied=stage_b_budget.binding,
            )
            curated_by_uid = {
                str(record["chunk_uid"]): record
                for record in curated_pool_records
            }
            selected = [
                {
                    **curated_by_uid[str(record["chunk_uid"])],
                    "selection": record["selection"],
                }
                for record in selected
            ]
            budget_not_selected_records = [
                record
                for record in curated_pool_records
                if (record.get("curation_decision") or {}).get(
                    "training_budget_disposition"
                )
                == "budget_not_selected"
            ]
            for record in selected:
                selected_clusters[_cluster_id(record)] += 1
            _progress(
                f"selection done: profile={profile_name} dataset={dataset} "
                f"selected={len(selected)} ratio={len(selected) / max(source_records_for_profile, 1):.6f} "
                f"elapsed={_elapsed_seconds(selection_started)}"
            )

            if processed_records != source_records_for_profile:
                raise RuntimeError(
                    f"{profile_name}:{dataset} processed_records mismatch "
                    f"(expected={source_records_for_profile}, processed={processed_records})"
                )

            synthetic_runtime_smoke = runtime_limited and str(probe_cfg.get("mode") or "") == "synthetic_smoke"
            text_started = time.perf_counter()
            _progress(f"text fetch start: profile={profile_name} dataset={dataset} selected={len(selected)}")
            text_map = (
                _preview_text_map_from_scored_records(selected)
                if synthetic_runtime_smoke
                else _fetch_texts(conn, [r["chunk_uid"] for r in selected])
            )
            missing_text_uids = [r["chunk_uid"] for r in selected if r["chunk_uid"] not in text_map]
            if missing_text_uids:
                sample_missing = ",".join(missing_text_uids[:5])
                raise RuntimeError(
                    f"{profile_name}:{dataset} missing texts for {len(missing_text_uids)} selected chunks "
                    f"(sample={sample_missing})"
                )
            _progress(
                f"text fetch done: profile={profile_name} dataset={dataset} "
                f"texts={len(text_map)} elapsed={_elapsed_seconds(text_started)}"
            )
            out_path = profile_dir / f"{dataset}.jsonl"
            tmp_out_path = profile_dir / f".{dataset}.jsonl.tmp"
            write_started = time.perf_counter()
            _progress(f"subset write start: profile={profile_name} dataset={dataset} path={out_path}")
            with tmp_out_path.open("w", encoding="utf-8") as f:
                for record in selected:
                    payload = dict(record)
                    payload["text"] = text_map.get(record["chunk_uid"], "")
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            tmp_out_path.replace(out_path)
            _progress(f"subset write done: profile={profile_name} dataset={dataset} elapsed={_elapsed_seconds(write_started)}")

            coverage_started = time.perf_counter()
            _progress(f"coverage metrics start: profile={profile_name} dataset={dataset}")
            coverage = _coverage_retention(selected_clusters, original_clusters)
            selected_domain_counts = (
                _selected_domain_counts_from_scored_records(selected)
                if synthetic_runtime_smoke
                else _selected_domain_counts(conn, chunk_uids=[r["chunk_uid"] for r in selected])
            )
            domain_coverage_support = _distribution_bucket_support(
                selected_domain_counts,
                original_domain_counts,
                support_scope=_source_bucket_support_scope(original_domain_counts),
                support_label="source_or_domain_bucket",
            )
            selected_style_counts = _selected_style_counts_from_text_map(selected, text_map)
            stage_b_selected_style_counts = Counter(_style_bucket_from_scored_record(record) for record in selected)
            style_coverage_support = _distribution_bucket_support(
                selected_style_counts,
                original_style_counts,
                support_scope="style_bucket",
                support_label="style_bucket",
            )
            style_taxonomy_alignment = _style_taxonomy_alignment_diagnostic(
                stage_b_selected_style_counts,
                selected_style_counts,
                selector_diagnostics,
            )
            source_coverage_support = dict(domain_coverage_support)
            source_coverage_support["support_label"] = "source_bucket"
            semantic_coverage_support = _semantic_coverage_support(coverage, cluster_backbone_audit)
            retained_rare_clusters = sum(1 for cluster_id in strategy["rare_clusters"] if selected_clusters.get(cluster_id, 0) > 0)
            rare_cluster_exemplars_added = sum(
                1
                for r in selected
                if str((r.get("selection") or {}).get("accepted_by") or "") == "rare_cluster_exemplar"
            )
            _progress(
                f"coverage metrics done: profile={profile_name} dataset={dataset} "
                f"coverage={float(coverage['score']):.6f} rare_retention={float(coverage['rare_cluster_retention']):.6f} "
                f"elapsed={_elapsed_seconds(coverage_started)}"
            )
            if not stage_a_exactdup_uids:
                raise RuntimeError(
                    f"{profile_name}:{dataset} stageA(validity+exactdup) baseline pool is empty."
                )
            baseline_started = time.perf_counter()
            _progress(f"baseline pools start: profile={profile_name} dataset={dataset}")
            selected_uids = {str(record["chunk_uid"]) for record in selected}
            all_dataset_control_uids = {uid for uid in all_dataset_uids if uid not in selected_uids}
            stage_a_control_uids = {uid for uid in stage_a_exactdup_uids if uid not in selected_uids}
            if not all_dataset_control_uids:
                raise RuntimeError(f"{profile_name}:{dataset} full-random disjoint control pool is empty.")
            if not stage_a_control_uids:
                raise RuntimeError(f"{profile_name}:{dataset} Stage-A disjoint control pool is empty.")
            all_dataset_control_fingerprint = _fingerprint_uids(all_dataset_control_uids)
            stage_a_control_fingerprint = _fingerprint_uids(stage_a_control_uids)
            matched_baseline_pools = _diagnostic_matched_baseline_pools(
                baseline_records=stage_a_baseline_records,
                selected_records=selected,
                seed=int(selector_cfg.get("hash_sampling_seed") or probe_cfg.get("sampling_hash_seed") or 42),
                pool_multiplier=int(selector_cfg.get("matched_baseline_pool_multiplier") or 4),
                exclude_selected=True,
            )
            learning_signal_started = time.perf_counter()
            _progress(f"learning-signal diagnostic start: profile={profile_name} dataset={dataset}")
            canonical_pool = matched_baseline_pools.get(CANONICAL_UTILITY_BASELINE) or {}
            learning_signal_coverage_diagnostic = _learning_signal_gap_diagnostic(
                conn=conn,
                selected_records=selected,
                selected_text_map=text_map,
                baseline_records=stage_a_baseline_records,
                baseline_allowed_uids=set(canonical_pool.get("allowed_uids") or set()),
                sample_size=int(selector_cfg.get("learning_signal_diagnostic_sample_size") or 3000),
                seed=int(selector_cfg.get("hash_sampling_seed") or probe_cfg.get("sampling_hash_seed") or 42),
            )
            _progress(
                f"learning-signal diagnostic done: profile={profile_name} dataset={dataset} "
                f"flags={len(learning_signal_coverage_diagnostic.get('risk_flags') or [])} "
                f"elapsed={_elapsed_seconds(learning_signal_started)}"
            )
            in_domain_baseline_specs: Dict[str, Tuple[set[str] | None, str]] = {
                "baseline_full_random": (all_dataset_control_uids, all_dataset_control_fingerprint),
                "baseline_stageA_random": (stage_a_control_uids, stage_a_control_fingerprint),
            }
            for baseline_name, pool in matched_baseline_pools.items():
                in_domain_baseline_specs[baseline_name] = (
                    pool["allowed_uids"],
                    str(pool["fingerprint"]),
                )
            _progress(
                f"baseline pools done: profile={profile_name} dataset={dataset} "
                f"in_domain_baselines={len(in_domain_baseline_specs)} "
                f"full_pool={len(all_dataset_control_uids)} stageA_pool={len(stage_a_control_uids)} "
                f"elapsed={_elapsed_seconds(baseline_started)}"
            )
            in_domain_results = {}
            in_domain_started = time.perf_counter()
            _progress(
                f"utility in-domain start: profile={profile_name} dataset={dataset} "
                f"baselines={len(in_domain_baseline_specs)} "
                f"buckets={len(probe_cfg['holdout_buckets'])} seeds={len(probe_cfg.get('seeds') or [probe_cfg['seed']])}"
            )
            for baseline_name, (allowed_uids, fingerprint) in in_domain_baseline_specs.items():
                baseline_eval_started = time.perf_counter()
                _progress(f"utility in-domain baseline start: profile={profile_name} dataset={dataset} baseline={baseline_name}")
                in_domain_results[baseline_name] = _score_with_probe_buckets(
                    conn,
                    context_cache=utility_probe_context_cache,
                    selected_sequence_cache=utility_selected_sequence_cache,
                    selected_records=selected,
                    text_map=text_map,
                    baseline_variant=baseline_name,
                    baseline_allowed_uids=allowed_uids,
                    baseline_uid_fingerprint=fingerprint,
                    train_dataset=str(dataset),
                    eval_dataset=str(dataset),
                    probe_cfg=probe_cfg,
                    eval_token_budget=int(probe_cfg["eval_token_budget"]),
                    holdout_buckets=list(probe_cfg["holdout_buckets"]),
                    progress_label=f"profile={profile_name} train={dataset} eval={dataset}",
                )
                baseline_result = in_domain_results[baseline_name]
                _progress(
                    f"utility in-domain baseline done: profile={profile_name} dataset={dataset} "
                    f"baseline={baseline_name} gain={float(baseline_result.get('small_lm_probe_gain_score') or 0.0):.6f} "
                    f"delta_nll={float(baseline_result.get('delta_nll') or 0.0):.6f} "
                    f"elapsed={_elapsed_seconds(baseline_eval_started)}"
                )
            in_domain_pass = _utility_axis_pass_by_baselines(in_domain_results, stage_c)
            _progress(
                f"utility in-domain done: profile={profile_name} dataset={dataset} "
                f"pass={bool(in_domain_pass.get('pass'))} elapsed={_elapsed_seconds(in_domain_started)}"
            )
            utility_mode = "in_domain_only"
            ood_eval_datasets: List[str] = []
            ood_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
            ood_pass_by_dataset: Dict[str, Dict[str, Any]] = {}
            should_compute_ood = bool(stage_c["compute_ood_utility_report"]) and len(active_dataset_names) > 1
            should_enforce_ood = bool(stage_c["enforce_ood_utility_pass"])
            if should_compute_ood:
                utility_mode = "in_domain_required_ood_report"
                ood_eval_datasets = [str(name) for name in active_dataset_names if str(name) != str(dataset)]
                ood_started = time.perf_counter()
                _progress(
                    f"utility OOD start: profile={profile_name} train={dataset} "
                    f"eval_datasets={ood_eval_datasets} baselines={len(in_domain_baseline_specs)} "
                    f"buckets={len(probe_cfg['ood_holdout_buckets'])} seeds={len(probe_cfg.get('seeds') or [probe_cfg['seed']])}"
                )
                for ood_dataset in ood_eval_datasets:
                    ood_pair_started = time.perf_counter()
                    _progress(f"utility OOD pair start: profile={profile_name} train={dataset} eval={ood_dataset}")
                    ood_baseline_specs: Dict[str, Tuple[set[str] | None, str]] = {
                        "baseline_full_random": (all_dataset_control_uids, all_dataset_control_fingerprint),
                        "baseline_stageA_random": (stage_a_control_uids, stage_a_control_fingerprint),
                    }
                    for baseline_name, pool in matched_baseline_pools.items():
                        ood_baseline_specs[baseline_name] = (
                            pool["allowed_uids"],
                            str(pool["fingerprint"]),
                        )

                    pair_results = {}
                    for baseline_name, (allowed_uids, fingerprint) in ood_baseline_specs.items():
                        ood_baseline_started = time.perf_counter()
                        _progress(
                            f"utility OOD baseline start: profile={profile_name} train={dataset} "
                            f"eval={ood_dataset} baseline={baseline_name}"
                        )
                        pair_results[baseline_name] = _score_with_probe_buckets(
                            conn,
                            context_cache=utility_probe_context_cache,
                            selected_sequence_cache=utility_selected_sequence_cache,
                            selected_records=selected,
                            text_map=text_map,
                            baseline_variant=baseline_name,
                            baseline_allowed_uids=allowed_uids,
                            baseline_uid_fingerprint=fingerprint,
                            train_dataset=str(dataset),
                            eval_dataset=str(ood_dataset),
                            probe_cfg=probe_cfg,
                            eval_token_budget=int(probe_cfg["ood_eval_token_budget"]),
                            holdout_buckets=list(probe_cfg["ood_holdout_buckets"]),
                            progress_label=f"profile={profile_name} train={dataset} eval={ood_dataset}",
                        )
                        ood_baseline_result = pair_results[baseline_name]
                        _progress(
                            f"utility OOD baseline done: profile={profile_name} train={dataset} eval={ood_dataset} "
                            f"baseline={baseline_name} gain={float(ood_baseline_result.get('small_lm_probe_gain_score') or 0.0):.6f} "
                            f"delta_nll={float(ood_baseline_result.get('delta_nll') or 0.0):.6f} "
                            f"elapsed={_elapsed_seconds(ood_baseline_started)}"
                        )
                    ood_results[str(ood_dataset)] = pair_results
                    ood_pass_by_dataset[str(ood_dataset)] = _utility_axis_pass_by_baselines(pair_results, stage_c)
                    _progress(
                        f"utility OOD pair done: profile={profile_name} train={dataset} eval={ood_dataset} "
                        f"pass={bool(ood_pass_by_dataset[str(ood_dataset)].get('pass'))} "
                        f"elapsed={_elapsed_seconds(ood_pair_started)}"
                    )
                if should_enforce_ood:
                    utility_mode = "dual_eval_strict"
                _progress(
                    f"utility OOD done: profile={profile_name} train={dataset} "
                    f"pairs={len(ood_results)} elapsed={_elapsed_seconds(ood_started)}"
                )
            elif should_enforce_ood:
                utility_mode = "dual_eval_strict"

            aggregate_started = time.perf_counter()
            _progress(f"stage-C aggregate start: profile={profile_name} dataset={dataset}")
            canonical_in_domain_results = [in_domain_results[CANONICAL_UTILITY_BASELINE]]
            canonical_ood_results = (
                [
                    baselines[CANONICAL_UTILITY_BASELINE]
                    for baselines in ood_results.values()
                    if isinstance(baselines, dict) and CANONICAL_UTILITY_BASELINE in baselines
                ]
                if should_enforce_ood
                else []
            )
            effective_results = canonical_in_domain_results + canonical_ood_results
            reported_results = list(in_domain_results.values()) + [
                result for baselines in ood_results.values() for result in baselines.values()
            ]
            utility_pass_statistic = str(stage_c.get("utility_pass_statistic") or "min")
            small_lm_probe_gain_score = min(
                _utility_result_value(r, "small_lm_probe_gain_score", utility_pass_statistic)
                for r in effective_results
            )

            coverage_score_pass = float(coverage["score"]) >= float(stage_c["min_subset_coverage_retention_score"])
            coverage_tail_retention_pass = float(coverage["rare_cluster_retention"]) >= float(stage_c["min_rare_cluster_retention"])
            coverage_tail_count_pass = int(coverage["rare_cluster_retained_count"]) >= int(stage_c["min_rare_cluster_retained_count"])
            coverage_backbone_pass = bool(cluster_backbone_audit.get("passed"))
            coverage_semantic_support_pass = bool(semantic_coverage_support.get("cluster_backbone_pass"))
            coverage_backbone_enforced = bool(stage_c.get("enforce_coverage_backbone_pass", False))
            coverage_domain_support_pass = _bucket_support_pass(
                domain_coverage_support,
                min_distribution_similarity=float(stage_c["min_domain_coverage_distribution_similarity"]),
                min_retained_bucket_ratio=float(stage_c["min_domain_coverage_retained_bucket_ratio"]),
            )
            coverage_style_support_pass = _bucket_support_pass(
                style_coverage_support,
                min_distribution_similarity=float(stage_c["min_style_coverage_distribution_similarity"]),
                min_retained_bucket_ratio=float(stage_c["min_style_coverage_retained_bucket_ratio"]),
            )
            coverage_domain_support_enforced = bool(stage_c.get("enforce_domain_coverage_support", True))
            coverage_style_support_enforced = bool(stage_c.get("enforce_style_coverage_support", True))
            coverage_pass = bool(
                coverage_score_pass
                and coverage_tail_retention_pass
                and coverage_tail_count_pass
                and (coverage_backbone_pass if coverage_backbone_enforced else True)
                and (coverage_domain_support_pass if coverage_domain_support_enforced else True)
                and (coverage_style_support_pass if coverage_style_support_enforced else True)
            )

            expected_ood_pair_count = max(0, len(active_dataset_names) - 1) if (should_compute_ood or should_enforce_ood) else 0
            observed_ood_pair_count = len(ood_results)
            ood_required_missing = bool(
                should_enforce_ood
                and (
                    expected_ood_pair_count == 0
                    or observed_ood_pair_count < expected_ood_pair_count
                    or len(ood_pass_by_dataset) < expected_ood_pair_count
                )
            )
            ood_enforced = bool(should_enforce_ood and not ood_required_missing)

            def _all_ood_pass(pass_key: str) -> bool:
                if not should_enforce_ood:
                    return True
                if ood_required_missing:
                    return False
                return all(bool(payload.get(pass_key)) for payload in ood_pass_by_dataset.values())

            def _all_ood_baseline_pass(baseline_key: str) -> bool:
                if not should_enforce_ood:
                    return True
                if ood_required_missing:
                    return False
                return all(
                    bool(((payload.get("by_baseline") or {}).get(baseline_key) or {}).get("pass"))
                    for payload in ood_pass_by_dataset.values()
                )

            def _failed_ood_pairs(baseline_key: str) -> List[str]:
                return [
                    str(eval_dataset)
                    for eval_dataset, payload in ood_pass_by_dataset.items()
                    if not bool(((payload.get("by_baseline") or {}).get(baseline_key) or {}).get("pass"))
                ]

            cross_domain_utility_score_pass = bool(
                observed_ood_pair_count > 0
                and not ood_required_missing
                and all(bool(payload.get("score_pass")) for payload in ood_pass_by_dataset.values())
            )
            cross_domain_utility_relative_gain_pass = bool(
                observed_ood_pair_count > 0
                and not ood_required_missing
                and all(bool(payload.get("relative_gain_pass")) for payload in ood_pass_by_dataset.values())
            )
            cross_domain_utility_delta_nll_pass = bool(
                observed_ood_pair_count > 0
                and not ood_required_missing
                and all(bool(payload.get("delta_nll_pass")) for payload in ood_pass_by_dataset.values())
            )
            cross_domain_utility_ci_pass = bool(
                observed_ood_pair_count > 0
                and not ood_required_missing
                and all(bool(payload.get("ci_pass")) for payload in ood_pass_by_dataset.values())
            )
            cross_domain_utility_axis_pass = bool(
                observed_ood_pair_count > 0
                and not ood_required_missing
                and all(bool(payload.get("pass")) for payload in ood_pass_by_dataset.values())
            )
            domain_specific_utility_axis_pass = bool(in_domain_pass["pass"])
            general_purpose_utility_axis_pass = bool(domain_specific_utility_axis_pass and cross_domain_utility_axis_pass)
            final_certification_scope = str(stage_c.get("certification_scope") or "domain_specific")
            final_uses_cross_domain = final_certification_scope == "general_purpose"
            utility_score_pass = bool(
                in_domain_pass["score_pass"]
                and (cross_domain_utility_score_pass if final_uses_cross_domain else True)
            )
            utility_relative_gain_pass = bool(
                in_domain_pass["relative_gain_pass"]
                and (cross_domain_utility_relative_gain_pass if final_uses_cross_domain else True)
            )
            utility_delta_nll_pass = bool(
                in_domain_pass["delta_nll_pass"]
                and (cross_domain_utility_delta_nll_pass if final_uses_cross_domain else True)
            )
            utility_ci_pass = bool(
                in_domain_pass["ci_pass"]
                and (cross_domain_utility_ci_pass if final_uses_cross_domain else True)
            )
            utility_axis_pass = bool(
                general_purpose_utility_axis_pass if final_uses_cross_domain else domain_specific_utility_axis_pass
            )
            stage_c_pass = bool(coverage_pass and utility_axis_pass)

            baseline_failures = {
                "failed_vs_multi_matched_stageA_random": not bool(
                    in_domain_pass["by_baseline"].get(CANONICAL_UTILITY_BASELINE, {}).get("pass")
                    and _all_ood_baseline_pass(CANONICAL_UTILITY_BASELINE)
                ),
                "failed_ood_pairs_vs_multi_matched_stageA_random": _failed_ood_pairs(CANONICAL_UTILITY_BASELINE),
            }
            stress_baseline_failures = {
                "failed_vs_stageA_random": not bool(
                    in_domain_pass["by_baseline"].get("baseline_stageA_random", {}).get("pass")
                    and _all_ood_baseline_pass("baseline_stageA_random")
                ),
                "failed_ood_pairs_vs_stageA_random": _failed_ood_pairs("baseline_stageA_random"),
                "failed_vs_full_random": not bool(
                    in_domain_pass["by_baseline"].get("baseline_full_random", {}).get("pass")
                    and _all_ood_baseline_pass("baseline_full_random")
                ),
                "failed_ood_pairs_vs_full_random": _failed_ood_pairs("baseline_full_random"),
            }
            diagnostic_baseline_failures = {
                f"failed_vs_{baseline_name}": not bool(payload.get("pass"))
                for baseline_name, payload in sorted((in_domain_pass.get("by_baseline") or {}).items())
                if baseline_name != CANONICAL_UTILITY_BASELINE
            }

            baseline_minima = {}
            baseline_pass_values = {}
            for baseline_key in sorted(in_domain_results.keys()):
                baseline_effective_results = [in_domain_results[baseline_key]] + (
                    [
                        baselines[baseline_key]
                        for baselines in ood_results.values()
                        if isinstance(baselines, dict) and baseline_key in baselines
                    ]
                    if should_enforce_ood
                    else []
                )
                baseline_pass_values[baseline_key] = {
                    "small_lm_probe_gain_score": round(
                        min(
                            _utility_result_value(r, "small_lm_probe_gain_score", utility_pass_statistic)
                            for r in baseline_effective_results
                        ),
                        6,
                    ),
                    "relative_nll_gain": round(
                        min(_utility_result_value(r, "relative_nll_gain", utility_pass_statistic) for r in baseline_effective_results),
                        6,
                    ),
                    "delta_nll": round(
                        min(_utility_result_value(r, "delta_nll", utility_pass_statistic) for r in baseline_effective_results),
                        6,
                    ),
                }
                baseline_minima[baseline_key] = {
                    "fixed_token_probe_gain_score": round(
                        min(
                            _utility_result_value(r, "small_lm_probe_gain_score", "min")
                            for r in baseline_effective_results
                        ),
                        6,
                    ),
                    "small_lm_probe_gain_score": round(
                        min(
                            _utility_result_value(r, "small_lm_probe_gain_score", "min")
                            for r in baseline_effective_results
                        ),
                        6,
                    ),
                    "min_relative_nll_gain": round(
                        min(_utility_result_value(r, "relative_nll_gain", "min") for r in baseline_effective_results),
                        6,
                    ),
                    "min_delta_nll": round(
                        min(_utility_result_value(r, "delta_nll", "min") for r in baseline_effective_results),
                        6,
                    ),
                    "min_delta_nll_ci_low": round(
                        min(_float_metric(r, "delta_nll_ci_low") for r in baseline_effective_results),
                        6,
                    ),
                }

            utility_protocol = _utility_protocol_summary(stage_c, probe_cfg)
            utility_certification_shadow = _utility_certification_shadow(
                stage_c=stage_c,
                probe_cfg=probe_cfg,
                utility_protocol=utility_protocol,
                in_domain_results=in_domain_results,
                ood_results=ood_results,
            )
            protocol_ready = bool((utility_certification_shadow.get("protocol_readiness") or {}).get("pass"))
            in_domain_certification_ready = bool(utility_certification_shadow.get("in_domain_certification_ready"))
            cross_domain_certification_ready = bool(utility_certification_shadow.get("cross_domain_certification_ready"))
            domain_specific_certification_ready = bool(
                utility_certification_shadow.get("domain_specific_certification_ready")
            )
            general_purpose_certification_ready = bool(
                utility_certification_shadow.get("general_purpose_certification_ready")
            )
            final_scope_certification_ready = bool(
                general_purpose_certification_ready if final_uses_cross_domain else domain_specific_certification_ready
            )
            canonical_result = in_domain_results.get(CANONICAL_UTILITY_BASELINE) or {}
            canonical_causal_audit = canonical_result.get("causal_utility_audit") or {}
            matched_baseline_deltas = {}
            for baseline_name in sorted(MATCHED_DIAGNOSTIC_BASELINES):
                result = in_domain_results.get(baseline_name) or {}
                matched_baseline_deltas[baseline_name] = {
                    "small_lm_probe_gain_score": result.get("small_lm_probe_gain_score"),
                    "delta_nll": result.get("delta_nll"),
                    "delta_vs_canonical_gain": round(
                        float(result.get("small_lm_probe_gain_score") or 0.0)
                        - float(canonical_result.get("small_lm_probe_gain_score") or 0.0),
                        6,
                    ),
                    "delta_vs_canonical_delta_nll": round(
                        float(result.get("delta_nll") or 0.0) - float(canonical_result.get("delta_nll") or 0.0),
                        6,
                    ),
                }
            stage_a_random_result = in_domain_results.get("baseline_stageA_random") or {}
            distribution_shift_stress = {
                "stageA_random_gain": stage_a_random_result.get("small_lm_probe_gain_score"),
                "stageA_random_delta_nll": stage_a_random_result.get("delta_nll"),
                "delta_vs_canonical_gain": round(
                    float(stage_a_random_result.get("small_lm_probe_gain_score") or 0.0)
                    - float(canonical_result.get("small_lm_probe_gain_score") or 0.0),
                    6,
                ),
                "delta_vs_canonical_delta_nll": round(
                    float(stage_a_random_result.get("delta_nll") or 0.0) - float(canonical_result.get("delta_nll") or 0.0),
                    6,
                ),
            }
            stability_combined = (utility_certification_shadow.get("stability_analysis") or {}).get("combined_effective") or {}
            strict_values = utility_certification_shadow.get("strict_values") or {}
            canonical_delta = float(canonical_result.get("delta_nll") or 0.0)
            failure_mode = "pass"
            if not bool(utility_certification_shadow.get("signal_pass")):
                if bool(stability_combined.get("noise_dominated")) and abs(canonical_delta) <= 0.0025:
                    failure_mode = "weak_negative_or_noise_dominated_signal"
                elif any(
                    (payload.get("delta_nll") is not None)
                    and float(payload.get("delta_nll") or 0.0) > canonical_delta
                    for payload in in_domain_results.values()
                    if isinstance(payload, dict)
                ):
                    failure_mode = "distribution_shift_sensitive"
                else:
                    failure_mode = "negative_lm_transfer_signal"
            utility_failure_analysis = {
                "failure_mode": failure_mode,
                "canonical_in_domain_delta_nll": round(canonical_delta, 6),
                "canonical_in_domain_gain": canonical_result.get("small_lm_probe_gain_score"),
                "strict_min_delta_nll": strict_values.get("min_delta_nll"),
                "strict_min_gain": strict_values.get("min_small_lm_probe_gain_score"),
                "noise_dominated": bool(stability_combined.get("noise_dominated")) if stability_combined.get("noise_dominated") is not None else None,
                "matched_baseline_deltas": matched_baseline_deltas,
                "distribution_shift_stress": distribution_shift_stress,
                "learning_signal_coverage_diagnostic": learning_signal_coverage_diagnostic,
                "causal_utility_audit": canonical_causal_audit,
                "matched_baseline_pool_diagnostics": {
                    baseline_name: pool.get("diagnostics") for baseline_name, pool in sorted(matched_baseline_pools.items())
                },
                "interpretation": (
                    "Utility failure is treated as a subset-level learning-signal issue; "
                    "multi-matched Stage-A remains canonical; nuisance-matched Stage-A is an operational counterfactual candidate."
                ),
            }
            worst_cells = utility_certification_shadow.get("worst_cells") or {}
            worst_in_domain_cell = worst_cells.get("in_domain") or {}
            worst_ood_cell = worst_cells.get("ood") or {}
            blocker_categories = utility_certification_shadow.get("blocker_categories") or {}
            protocol_blockers = list(blocker_categories.get("protocol") or [])
            signal_blockers = list(blocker_categories.get("signal") or [])
            signal_interpretation = utility_certification_shadow.get("signal_interpretation") or {}
            combined_signal_interpretation = signal_interpretation.get("combined") or {}
            probe_sensitivity_status = _utility_probe_sensitivity_status(str(dataset))
            curation_benefit_status = _utility_curation_benefit_status(stage_a_random_result)
            strict_counterfactual_status = _utility_strict_counterfactual_status(
                final_scope_certification_ready=final_scope_certification_ready,
                utility_axis_pass=utility_axis_pass,
                combined_signal_status=combined_signal_interpretation,
                strict_values=strict_values,
            )
            operational_counterfactual_candidate_status = _utility_operational_counterfactual_candidate_status(
                in_domain_results.get(OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE) or {}
            )
            evidence_aware_tier = _utility_evidence_tier(
                probe_sensitivity_status=probe_sensitivity_status,
                curation_benefit_status=curation_benefit_status,
                strict_counterfactual_status=strict_counterfactual_status,
            )
            utility_failure_reason = _utility_failure_reason(
                probe_sensitivity_status=probe_sensitivity_status,
                curation_benefit_status=curation_benefit_status,
                strict_counterfactual_status=strict_counterfactual_status,
            )
            utility_probe_valid = (
                bool(probe_sensitivity_status.get("probe_valid"))
                if probe_sensitivity_status.get("probe_valid") is not None
                else None
            )
            utility_strict_pass = bool(
                utility_probe_valid is not False
                and strict_counterfactual_status.get("status") in {"matched_baseline_gain", "strict_certification_ready"}
            )
            utility_failure_analysis.update(
                {
                    "evidence_aware_failure_reason": utility_failure_reason,
                    "probe_sensitivity_status": probe_sensitivity_status,
                    "curation_benefit_status": curation_benefit_status,
                    "strict_counterfactual_status": strict_counterfactual_status,
                    "operational_counterfactual_candidate_status": operational_counterfactual_candidate_status,
                    "interpretation": (
                        "Utility is interpreted as an evidence protocol: first validate probe sensitivity, "
                        "then test selected > Stage-A random for curation benefit, and only then use "
                        "selected > multi-matched Stage-A as strict counterfactual certification evidence."
                    ),
                }
            )
            utility_evidence_summary = {
                "development_pass": bool(utility_axis_pass),
                "certification_ready": bool(utility_certification_shadow.get("certification_ready")),
                "final_scope_certification_ready": bool(final_scope_certification_ready),
                "in_domain_certification_ready": bool(in_domain_certification_ready),
                "cross_domain_certification_ready": bool(cross_domain_certification_ready),
                "domain_specific_certification_ready": bool(domain_specific_certification_ready),
                "general_purpose_certification_ready": bool(general_purpose_certification_ready),
                "protocol_ready": bool(protocol_ready),
                "signal_pass": bool(utility_certification_shadow.get("signal_pass")),
                "in_domain_signal_pass": bool((utility_certification_shadow.get("in_domain_signal") or {}).get("pass")),
                "ood_signal_pass": bool((utility_certification_shadow.get("ood_signal") or {}).get("pass")),
                "in_domain_utility_axis_pass": bool(domain_specific_utility_axis_pass),
                "cross_domain_utility_axis_pass": bool(cross_domain_utility_axis_pass),
                "domain_specific_utility_axis_pass": bool(domain_specific_utility_axis_pass),
                "general_purpose_utility_axis_pass": bool(general_purpose_utility_axis_pass),
                "final_utility_axis_pass": bool(utility_axis_pass),
                "final_certification_scope": final_certification_scope,
                "utility_probe_valid": utility_probe_valid,
                "utility_strict_pass": bool(utility_strict_pass),
                "probe_sensitivity_status": probe_sensitivity_status,
                "curation_benefit_status": curation_benefit_status,
                "strict_counterfactual_status": strict_counterfactual_status,
                "operational_counterfactual_candidate_status": operational_counterfactual_candidate_status,
                "evidence_tier": evidence_aware_tier,
                "legacy_certification_evidence_tier": str(utility_certification_shadow.get("evidence_tier") or "development_only"),
                "failure_reason": utility_failure_reason,
                "signal_status": str(combined_signal_interpretation.get("status") or "unknown"),
                "signal_status_reason": str(combined_signal_interpretation.get("reason") or ""),
                "in_domain_signal_status": str((signal_interpretation.get("in_domain") or {}).get("status") or "unknown"),
                "ood_signal_status": str((signal_interpretation.get("ood") or {}).get("status") or "unknown"),
                "failure_mode": failure_mode,
                "causal_failure_mode": str(canonical_causal_audit.get("dominant_failure_mode") or "unresolved"),
                "causal_failure_interpretation": str(canonical_causal_audit.get("interpretation") or ""),
                "noise_dominated": (
                    bool(stability_combined.get("noise_dominated"))
                    if stability_combined.get("noise_dominated") is not None
                    else None
                ),
                "canonical_baseline": CANONICAL_UTILITY_BASELINE,
                "canonical_mean_gain": canonical_result.get("small_lm_probe_gain_score"),
                "canonical_in_domain_delta_nll": round(canonical_delta, 6),
                "strict_min_gain": strict_values.get("min_small_lm_probe_gain_score"),
                "strict_min_relative_nll_gain": strict_values.get("min_relative_nll_gain"),
                "strict_min_delta_nll": strict_values.get("min_delta_nll"),
                "strict_min_delta_nll_ci_low": strict_values.get("min_delta_nll_ci_low"),
                "max_minimum_detectable_delta_nll_95": strict_values.get("max_minimum_detectable_delta_nll_95"),
                "min_effect_to_mde_ratio": strict_values.get("min_effect_to_mde_ratio"),
                "min_detectable_effect_fraction": strict_values.get("min_detectable_effect_fraction"),
                "worst_in_domain_gain": worst_in_domain_cell.get("small_lm_probe_gain_score"),
                "worst_in_domain_delta_nll": worst_in_domain_cell.get("delta_nll"),
                "worst_in_domain_pair": worst_in_domain_cell.get("pair") or worst_in_domain_cell.get("eval_dataset"),
                "worst_ood_gain": worst_ood_cell.get("small_lm_probe_gain_score"),
                "worst_ood_delta_nll": worst_ood_cell.get("delta_nll"),
                "worst_ood_pair": worst_ood_cell.get("pair") or worst_ood_cell.get("eval_dataset"),
                "ood_pair_count": int(observed_ood_pair_count),
                "ood_expected_pair_count": int(expected_ood_pair_count),
                "protocol_blocker_count": int(len(protocol_blockers)),
                "signal_blocker_count": int(len(signal_blockers)),
                "protocol_blockers": sorted(set(str(item) for item in protocol_blockers)),
                "signal_blockers": sorted(set(str(item) for item in signal_blockers)),
                "certification_blockers": list(utility_certification_shadow.get("blockers") or []),
                "signal_interpretation": signal_interpretation,
                "causal_utility_audit": canonical_causal_audit,
            }
            baseline_control_policy = {
                "treatment_control_disjoint": True,
                "selected_uid_count": int(len(selected_uids)),
                "full_random_control_uid_count": int(len(all_dataset_control_uids)),
                "stageA_random_control_uid_count": int(len(stage_a_control_uids)),
                "matched_baseline_controls_exclude_selected": True,
                "canonical_baseline": CANONICAL_UTILITY_BASELINE,
                "canonical_matching_policy": str(
                    (matched_baseline_pools.get(CANONICAL_UTILITY_BASELINE) or {})
                    .get("diagnostics", {})
                    .get("matching_policy")
                    or "bucket_matched"
                ),
                "canonical_matched_pool_count": int(
                    (matched_baseline_pools.get(CANONICAL_UTILITY_BASELINE) or {})
                    .get("diagnostics", {})
                    .get("matched_pool_count")
                    or 0
                ),
                "canonical_baseline_excludes_selected": bool(
                    (matched_baseline_pools.get(CANONICAL_UTILITY_BASELINE) or {}).get("diagnostics", {}).get("exclude_selected")
                ),
                "canonical_baseline_excluded_selected_records": int(
                    (matched_baseline_pools.get(CANONICAL_UTILITY_BASELINE) or {}).get("diagnostics", {}).get("excluded_selected_records") or 0
                ),
            }
            aggregate_utility = {
                "mode": utility_mode,
                "evaluation_mode": str(stage_c.get("evaluation_mode") or "development"),
                "final_certification_scope": final_certification_scope,
                "protocol": utility_protocol,
                "pass_statistic": utility_pass_statistic,
                "canonical_baseline": CANONICAL_UTILITY_BASELINE,
                "diagnostic_baselines": list(DIAGNOSTIC_UTILITY_BASELINES),
                "score": round(small_lm_probe_gain_score, 6),
                "small_lm_probe_gain_score": round(small_lm_probe_gain_score, 6),
                "fixed_token_probe_gain_score": round(small_lm_probe_gain_score, 6),
                "reported_small_lm_probe_gain_score_min": round(
                    min(_utility_result_value(r, "small_lm_probe_gain_score", "min") for r in effective_results),
                    6,
                ),
                "stress_reported_small_lm_probe_gain_score_min": round(
                    min(_utility_result_value(r, "small_lm_probe_gain_score", "min") for r in reported_results),
                    6,
                ),
                "min_relative_nll_gain": round(
                    min(_utility_result_value(r, "relative_nll_gain", "min") for r in effective_results),
                    6,
                ),
                "min_delta_nll": round(min(_utility_result_value(r, "delta_nll", "min") for r in effective_results), 6),
                "min_delta_nll_ci_low": round(min(_float_metric(r, "delta_nll_ci_low") for r in effective_results), 6),
                "max_minimum_detectable_delta_nll_95": strict_values.get("max_minimum_detectable_delta_nll_95"),
                "min_effect_to_mde_ratio": strict_values.get("min_effect_to_mde_ratio"),
                "min_detectable_effect_fraction": strict_values.get("min_detectable_effect_fraction"),
                "in_domain_dataset": str(dataset),
                "ood_eval_dataset": str(ood_eval_datasets[0]) if ood_eval_datasets else None,
                "ood_eval_datasets": list(ood_eval_datasets),
                "ood_pair_count": int(observed_ood_pair_count),
                "ood_expected_pair_count": int(expected_ood_pair_count),
                "pairwise_ood_results": ood_results,
                "ood_eval_reported": bool(ood_results),
                "ood_eval_enforced": bool(ood_enforced),
                "ood_required_missing": bool(ood_required_missing),
                "in_domain_utility_axis_pass": bool(domain_specific_utility_axis_pass),
                "cross_domain_utility_axis_pass": bool(cross_domain_utility_axis_pass),
                "domain_specific_utility_axis_pass": bool(domain_specific_utility_axis_pass),
                "general_purpose_utility_axis_pass": bool(general_purpose_utility_axis_pass),
                "final_utility_axis_pass": bool(utility_axis_pass),
                "utility_probe_valid": utility_probe_valid,
                "utility_strict_pass": bool(utility_strict_pass),
                "probe_sensitivity_status": probe_sensitivity_status,
                "curation_benefit_status": curation_benefit_status,
                    "strict_counterfactual_status": strict_counterfactual_status,
                    "operational_counterfactual_candidate_status": operational_counterfactual_candidate_status,
                "utility_failure_reason": utility_failure_reason,
                "final_scope_certification_ready": bool(final_scope_certification_ready),
                "in_domain_certification_ready": bool(in_domain_certification_ready),
                "cross_domain_certification_ready": bool(cross_domain_certification_ready),
                "domain_specific_certification_ready": bool(domain_specific_certification_ready),
                "general_purpose_certification_ready": bool(general_purpose_certification_ready),
                "baseline_pass_values": baseline_pass_values,
                "baseline_minima": baseline_minima,
                "failed_vs_multi_matched_stageA_random": bool(baseline_failures["failed_vs_multi_matched_stageA_random"]),
                "stress_failed_vs_stageA_random": bool(stress_baseline_failures["failed_vs_stageA_random"]),
                "failed_by_baseline": baseline_failures,
                "stress_failed_by_baseline": stress_baseline_failures,
                "diagnostic_failed_by_baseline": diagnostic_baseline_failures,
                "stress_failed_vs_full_random": bool(stress_baseline_failures["failed_vs_full_random"]),
                "strict_values": utility_certification_shadow.get("strict_values") or {},
                "worst_cells": worst_cells,
                "stability_analysis": utility_certification_shadow.get("stability_analysis") or {},
                "evidence_tier": evidence_aware_tier,
                "legacy_certification_evidence_tier": utility_certification_shadow.get("evidence_tier"),
                "protocol_blocker_count": utility_evidence_summary["protocol_blocker_count"],
                "signal_blocker_count": utility_evidence_summary["signal_blocker_count"],
                "utility_evidence_summary": utility_evidence_summary,
                "baseline_control_policy": baseline_control_policy,
                "utility_failure_analysis": utility_failure_analysis,
                "causal_utility_audit": canonical_causal_audit,
                "certification_shadow": utility_certification_shadow,
            }
            utility_probe_details = {}
            utility_probe_details["mode"] = utility_mode
            utility_probe_details["evaluation_mode"] = str(stage_c.get("evaluation_mode") or "development")
            utility_probe_details["final_certification_scope"] = final_certification_scope
            utility_probe_details["protocol"] = utility_protocol
            utility_probe_details["in_domain"] = in_domain_results
            utility_probe_details["out_of_domain"] = ood_results
            utility_probe_details["aggregate"] = aggregate_utility

            dataset_summary = {
                "processed_records": processed_records,
                "execution_scope": (
                    "experimental_budgeted_subset_validation"
                    if stage_b_budget.binding
                    else "full_curated_pool_retain_all_validation"
                ),
                "full_curated_pool_records": len(curated_pool_records),
                "selected_records": len(selected),
                "budget_not_selected_records": len(budget_not_selected_records),
                "stage_a_rejected_records": max(
                    0,
                    int(processed_records) - int(len(curated_pool_records)),
                ),
                "stage_b_selection_mode": stage_b_budget.mode,
                "stage_b_budget": {
                    "binding": stage_b_budget.binding,
                    "word_limit": stage_b_budget.word_limit,
                },
                "disposition_summary": disposition_summary(curated_pool_records),
                "disposition_invariants": {
                    "stage_a_pass_implies_curated_pool_membership": True,
                    "budget_not_selected_is_rejection": False,
                    "quality_score_has_hard_reject_authority": False,
                    "retain_all_supported_by_operational_stage_b": True,
                },
                "curated_pool_reference": {
                    "source_scored_path": str(source_path),
                    "membership_rule": "passes frozen Stage-A hard gates",
                    "materialized_training_subset_path": str(out_path),
                    "note": (
                        "This generic runner is an experimental budgeted-subset "
                        "validation path. The selected JSONL is not the full curated pool."
                    ),
                },
                "source_records": int(source_records_for_profile),
                "original_source_records": int(dataset_meta["records"]),
                "runtime_limited_source_records": int(source_records_for_profile),
                "selection_ratio": round(len(selected) / max(int(source_records_for_profile), 1), 6),
                "subset_coverage_retention_score": coverage["score"],
                "small_lm_probe_gain_score": round(small_lm_probe_gain_score, 6),
                "fixed_token_probe_gain_score": round(small_lm_probe_gain_score, 6),
                "stage_c_core_validation": {
                    "passed": stage_c_pass,
                    "evaluation_mode": str(stage_c.get("evaluation_mode") or "development"),
                    "final_certification_scope": final_certification_scope,
                    "coverage_pass": bool(coverage_pass),
                    "coverage_score_pass": bool(coverage_score_pass),
                    "coverage_tail_retention_pass": bool(coverage_tail_retention_pass),
                    "coverage_tail_count_pass": bool(coverage_tail_count_pass),
                    "coverage_backbone_pass": bool(coverage_backbone_pass),
                    "coverage_backbone_enforced": bool(coverage_backbone_enforced),
                    "coverage_semantic_support_pass": bool(coverage_semantic_support_pass),
                    "coverage_semantic_support_enforced": bool(coverage_backbone_enforced),
                    "coverage_domain_support_pass": bool(coverage_domain_support_pass),
                    "coverage_domain_support_enforced": bool(coverage_domain_support_enforced),
                    "coverage_style_support_pass": bool(coverage_style_support_pass),
                    "coverage_style_support_enforced": bool(coverage_style_support_enforced),
                    "utility_pass_statistic": utility_pass_statistic,
                    "utility_score_pass": bool(utility_score_pass),
                    "utility_relative_gain_pass": bool(utility_relative_gain_pass),
                    "utility_delta_nll_pass": bool(utility_delta_nll_pass),
                    "utility_ci_pass": bool(utility_ci_pass),
                    "utility_axis_pass": bool(utility_axis_pass),
                    "utility_probe_valid": utility_probe_valid,
                    "utility_strict_pass": bool(utility_strict_pass),
                    "utility_failure_reason": utility_failure_reason,
                    "utility_probe_sensitivity_status": probe_sensitivity_status,
                    "utility_curation_benefit_status": curation_benefit_status,
                    "utility_strict_counterfactual_status": strict_counterfactual_status,
                    "utility_operational_counterfactual_candidate_status": operational_counterfactual_candidate_status,
                    "in_domain_utility_axis_pass": bool(domain_specific_utility_axis_pass),
                    "cross_domain_utility_axis_pass": bool(cross_domain_utility_axis_pass),
                    "domain_specific_utility_axis_pass": bool(domain_specific_utility_axis_pass),
                    "general_purpose_utility_axis_pass": bool(general_purpose_utility_axis_pass),
                    "final_utility_axis_pass": bool(utility_axis_pass),
                    "utility_failures_by_baseline": baseline_failures,
                    "utility_stress_failures_by_baseline": stress_baseline_failures,
                    "utility_canonical_baseline": CANONICAL_UTILITY_BASELINE,
                    "utility_diagnostic_baselines": list(DIAGNOSTIC_UTILITY_BASELINES),
                    "utility_mode": utility_mode,
                    "utility_ood_required_missing": bool(ood_required_missing),
                    "utility_ood_pair_count": int(observed_ood_pair_count),
                    "utility_ood_expected_pair_count": int(expected_ood_pair_count),
                    "utility_ood_eval_datasets": list(ood_eval_datasets),
                    "utility_certification_ready": bool(utility_certification_shadow["certification_ready"]),
                    "utility_final_scope_certification_ready": bool(final_scope_certification_ready),
                    "utility_in_domain_certification_ready": bool(in_domain_certification_ready),
                    "utility_cross_domain_certification_ready": bool(cross_domain_certification_ready),
                    "utility_domain_specific_certification_ready": bool(domain_specific_certification_ready),
                    "utility_general_purpose_certification_ready": bool(general_purpose_certification_ready),
                    "utility_evidence_tier": evidence_aware_tier,
                    "utility_legacy_certification_evidence_tier": str(utility_certification_shadow["evidence_tier"]),
                    "utility_strict_min_gain": utility_evidence_summary["strict_min_gain"],
                    "utility_strict_min_delta_nll": utility_evidence_summary["strict_min_delta_nll"],
                    "utility_strict_min_delta_nll_ci_low": utility_evidence_summary["strict_min_delta_nll_ci_low"],
                    "utility_worst_in_domain_gain": utility_evidence_summary["worst_in_domain_gain"],
                    "utility_worst_ood_gain": utility_evidence_summary["worst_ood_gain"],
                    "utility_worst_ood_pair": utility_evidence_summary["worst_ood_pair"],
                    "utility_certification_blockers": list(utility_certification_shadow["blockers"]),
                    "utility_protocol_blockers": list((utility_certification_shadow.get("blocker_categories") or {}).get("protocol") or []),
                    "utility_signal_blockers": list((utility_certification_shadow.get("blocker_categories") or {}).get("signal") or []),
                },
                "coverage_details": {
                    **coverage,
                    "source_coverage_support": source_coverage_support,
                    "domain_coverage_support": domain_coverage_support,
                    "style_coverage_support": style_coverage_support,
                    "style_taxonomy_alignment": style_taxonomy_alignment,
                    "semantic_coverage_support": semantic_coverage_support,
                    "coverage_axis_components": {
                        "source": source_coverage_support,
                        "style": style_coverage_support,
                        "semantic": semantic_coverage_support,
                        "learning_signal": learning_signal_coverage_diagnostic,
                    },
                    "learning_signal_coverage_diagnostic": learning_signal_coverage_diagnostic,
                    "domain_coverage_support_thresholds": {
                        "min_distribution_similarity": float(stage_c["min_domain_coverage_distribution_similarity"]),
                        "min_retained_bucket_ratio": float(stage_c["min_domain_coverage_retained_bucket_ratio"]),
                    },
                    "style_coverage_support_thresholds": {
                        "min_distribution_similarity": float(stage_c["min_style_coverage_distribution_similarity"]),
                        "min_retained_bucket_ratio": float(stage_c["min_style_coverage_retained_bucket_ratio"]),
                    },
                },
                "cluster_backbone_audit": cluster_backbone_audit,
                "utility_probe_details": utility_probe_details,
                "stage_b_rank_effective": stage_b,
                "selector_diagnostics": selector_diagnostics or {},
                "coverage_strategy_details": {
                    "enabled": bool(strategy["enabled"]),
                    "rare_cluster_quantile": strategy["rare_cluster_quantile"],
                    "rare_cluster_cutoff": strategy["rare_cluster_cutoff"],
                    "rare_cluster_count": len(strategy["rare_clusters"]),
                    "rare_cluster_retained": retained_rare_clusters,
                    "rare_cluster_exemplars_added": rare_cluster_exemplars_added,
                    "rare_exemplar_min_validity": strategy["rare_exemplar_min_validity"],
                    "rare_exemplar_min_reference_quality": strategy["rare_exemplar_min_reference_quality"],
                    "rare_exemplar_max_exact_duplicate_indicator": strategy["rare_exemplar_max_exact_duplicate_indicator"],
                    "rare_exemplar_relaxed_near_dup_ceiling": strategy["rare_exemplar_relaxed_near_dup_ceiling"],
                    "preserve_domain_bucket_exemplars": bool(selector_cfg.get("preserve_domain_bucket_exemplars")),
                    "domain_bucket_min_count": int(selector_cfg.get("domain_bucket_min_count") or 0),
                    "preserve_style_bucket_exemplars": bool(selector_cfg.get("preserve_style_bucket_exemplars")),
                    "style_bucket_min_count": int(selector_cfg.get("style_bucket_min_count") or 0),
                    "rare_cluster_min_count": int(selector_cfg.get("rare_cluster_min_count") or 1),
                    "preserve_domain_distribution": bool(selector_cfg.get("preserve_domain_distribution")),
                    "preserve_style_distribution": bool(selector_cfg.get("preserve_style_distribution")),
                    "preserve_length_distribution": bool(selector_cfg.get("preserve_length_distribution")),
                    "preserve_quality_band_distribution": bool(selector_cfg.get("preserve_quality_band_distribution")),
                    "diagnose_quality_band_distribution": bool(selector_cfg.get("diagnose_quality_band_distribution")),
                    "quality_band_distribution_min_quality": float(selector_cfg.get("quality_band_distribution_min_quality") or 0.0),
                    "stage_b_distribution_reference_scope": "stage_a_usable_candidates",
                    "stage_c_coverage_reference_scope": "original_dataset_distribution",
                    "quality_band_policy": (
                        "soft_top_quality_anti_collapse"
                        if bool(selector_cfg.get("preserve_quality_band_distribution"))
                        else "diagnostic_only_not_coverage"
                    ),
                    "quality_band_rebalance_mode": str(selector_cfg.get("quality_band_rebalance_mode") or "soft_cap"),
                    "quality_band_max_swap_ratio": float(selector_cfg.get("quality_band_max_swap_ratio") or 0.0),
                    "quality_top_band_max_share": float(selector_cfg.get("quality_top_band_max_share") or 0.0),
                    "selection_adjustments": dict(selector_cfg.get("selection_adjustments") or {}),
                },
                "output_path": str(out_path),
            }
            profile_summary["datasets"][dataset] = dataset_summary
            utility_probe_payload["datasets"].setdefault(profile_name, {})[dataset] = {
                "small_lm_probe_gain_score": round(small_lm_probe_gain_score, 6),
                "fixed_token_probe_gain_score": round(small_lm_probe_gain_score, 6),
                "details": utility_probe_details,
            }
            utility_probe_payload["profiles"].setdefault(profile_name, {})[dataset] = utility_probe_payload["datasets"][profile_name][dataset]
            _progress(
                f"stage-C aggregate done: profile={profile_name} dataset={dataset} "
                f"coverage_pass={coverage_pass} utility_pass={utility_axis_pass} "
                f"stage_c_pass={stage_c_pass} elapsed={_elapsed_seconds(aggregate_started)}"
            )
            _progress(f"dataset done: profile={profile_name} dataset={dataset} elapsed={_elapsed_seconds(dataset_started)}")

        profile_summaries[profile_name] = profile_summary
        _progress(f"profile done: {profile_name} elapsed={_elapsed_seconds(profile_started)}")

    _progress(f"write utility probe results start: path={UTILITY_PROBE_RESULTS_PATH}")
    save_json(UTILITY_PROBE_RESULTS_PATH, utility_probe_payload)
    _progress("write run reports start")
    run_manifest = write_run_reports(
        profiles_path=profiles_path,
        index_db_path=index_db_path,
        scoring_manifest_path=scoring_manifest_path,
        scoring_manifest=scoring_manifest,
        profile_summaries=profile_summaries,
        utility_probe_results_path=UTILITY_PROBE_RESULTS_PATH,
    )
    _progress("write run reports done")
    conn.close()
    return run_manifest
