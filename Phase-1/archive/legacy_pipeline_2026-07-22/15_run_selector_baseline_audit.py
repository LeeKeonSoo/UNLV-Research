#!/usr/bin/env python3
"""Audit what the Stage-B selector changes relative to random and matched baselines."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, SCORED_DIR, iter_jsonl_records_resilient
from policy.subsets import (
    _domain_bucket_from_scored_record,
    _length_bucket_from_scored_record,
    _objective_components,
    _quality_band_from_scored_record,
    _stable_hash_score,
    _style_bucket_from_scored_record,
)


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "selector_baseline_audit.md"


def _metric_score(record: Dict[str, Any], group: str, metric: str) -> float:
    payload = (record.get(group) or {}).get(metric) or {}
    try:
        return float(payload.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stage_a_pass(record: Dict[str, Any]) -> bool:
    return bool(
        _metric_score(record, "core_metrics", "structural_validity_gate") >= 1.0
        and _metric_score(record, "core_metrics", "exact_duplicate_indicator") <= 0.0
        and _metric_score(record, "core_metrics", "shingle_near_duplicate_indicator") <= 0.0
    )


def _resolve_profile(run_summary: Dict[str, Any], requested_profile: str) -> str:
    profiles = run_summary.get("profiles") or {}
    if requested_profile in profiles:
        return requested_profile
    names = [
        str(name)
        for name, payload in profiles.items()
        if not str(name).startswith("_") and isinstance(payload, dict)
    ]
    if len(names) == 1:
        fallback = names[0]
        print(
            f"[15] requested profile={requested_profile!r} not found; using only available profile={fallback!r}",
            flush=True,
        )
        return fallback
    raise RuntimeError(f"Requested profile {requested_profile!r} not found. Available profiles: {names}")


def _load_selected_uids(run_summary: Dict[str, Any], profile: str, dataset: str) -> set[str]:
    profile_payload = (run_summary.get("profiles") or {}).get(profile) or {}
    meta = profile_payload.get(dataset) or {}
    path = Path(str(meta.get("output_path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"selected subset missing for {profile}:{dataset}: {path}")
    return {
        str(record.get("chunk_uid") or "")
        for record in iter_jsonl_records_resilient(path)
        if str(record.get("chunk_uid") or "")
    }


def _choose_by_stable_hash(records: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    ordered = sorted(
        records,
        key=lambda record: (
            _stable_hash_score(str(record.get("chunk_uid") or ""), seed=seed),
            str(record.get("chunk_uid") or ""),
        ),
    )
    return list(ordered[: min(n, len(ordered))])


def _multi_key(record: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _quality_band_from_scored_record(record),
        _length_bucket_from_scored_record(record),
        _style_bucket_from_scored_record(record),
        _domain_bucket_from_scored_record(record),
    )


def _multi_level_keys(key: tuple[str, str, str, str]) -> tuple[str, str, str, str, str]:
    quality, length, style, domain = key
    return (
        f"exact::{quality}|{length}|{style}|{domain}",
        f"quality_length_style::{quality}|{length}|{style}",
        f"quality_length::{quality}|{length}",
        f"quality::{quality}",
        "global::*",
    )


def _build_fast_multi_matched_sample(
    *,
    baseline_records: Sequence[Dict[str, Any]],
    selected_records: Sequence[Dict[str, Any]],
    seed: int,
    pool_multiplier: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
    selected_counts = Counter(_multi_key(record) for record in selected_records)
    candidate_levels: Dict[str, List[Dict[str, Any]]] = {}
    excluded_selected_records = 0
    for record in baseline_records:
        uid = str(record.get("chunk_uid") or "")
        if uid in selected_uids:
            excluded_selected_records += 1
            continue
        quality, length, style, domain = _multi_key(record)
        keys = (
            f"exact::{quality}|{length}|{style}|{domain}",
            f"quality_length_style::{quality}|{length}|{style}",
            f"quality_length::{quality}|{length}",
            f"quality::{quality}",
        )
        for level_key in keys:
            candidate_levels.setdefault(level_key, []).append(record)
    for level_key, records in candidate_levels.items():
        records.sort(
            key=lambda record: (
                _stable_hash_score(str(record.get("chunk_uid") or ""), seed=seed),
                str(record.get("chunk_uid") or ""),
            )
        )
    chosen: List[Dict[str, Any]] = []
    chosen_uids: set[str] = set()
    level_cursors: Dict[str, int] = {}
    bucket_targets: Dict[str, int] = {}
    bucket_available_exact: Dict[str, int] = {}
    bucket_selected: Dict[str, int] = {}
    fallback_selected_by_level: Counter[str] = Counter()
    multiplier = max(1, int(pool_multiplier))
    for key, selected_count in sorted(selected_counts.items()):
        bucket_key = "|".join(key)
        target = max(int(selected_count), int(selected_count) * multiplier, 64)
        bucket_targets[bucket_key] = int(target)
        level_keys = _multi_level_keys(key)[:-1]
        bucket_available_exact[bucket_key] = int(len(candidate_levels.get(level_keys[0], [])))
        chosen_for_bucket = 0
        for level_key in level_keys:
            if chosen_for_bucket >= target:
                break
            level_name = level_key.split("::", 1)[0]
            candidates = candidate_levels.get(level_key, [])
            cursor = int(level_cursors.get(level_key, 0))
            while cursor < len(candidates) and chosen_for_bucket < target:
                record = candidates[cursor]
                cursor += 1
                if chosen_for_bucket >= target:
                    break
                uid = str(record.get("chunk_uid") or "")
                if not uid or uid in chosen_uids:
                    continue
                chosen.append(record)
                chosen_uids.add(uid)
                chosen_for_bucket += 1
                fallback_selected_by_level[level_name] += 1
            level_cursors[level_key] = cursor
        bucket_selected[bucket_key] = int(chosen_for_bucket)
    selected_n = len(selected_records)
    global_fill_count = 0
    if len(chosen) < selected_n:
        remaining = [
            record
            for record in baseline_records
            if str(record.get("chunk_uid") or "") not in selected_uids
            and str(record.get("chunk_uid") or "") not in chosen_uids
        ]
        fill = _choose_by_stable_hash(remaining, selected_n - len(chosen), seed + 17)
        global_fill_count = len(fill)
        chosen.extend(fill)
    sample = _choose_by_stable_hash(chosen, selected_n, seed + 11)

    def compact(mapping: Dict[Any, int], limit: int = 200) -> Dict[str, int]:
        return {
            str(k): int(v)
            for k, v in sorted(mapping.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
        }

    diagnostics = {
        "matching_policy": "quality_length_style_domain_with_hierarchical_fallback",
        "selected_reference_count": int(len(selected_records)),
        "baseline_reference_count": int(len(baseline_records)),
        "matched_pool_count": int(len(chosen)),
        "sample_count": int(len(sample)),
        "pool_multiplier": int(multiplier),
        "exclude_selected": True,
        "excluded_selected_records": int(excluded_selected_records),
        "selected_bucket_count": int(len(selected_counts)),
        "fallback_selected_by_level": compact(fallback_selected_by_level),
        "global_final_fill_count": int(global_fill_count),
        "bucket_diagnostics_truncated": bool(len(bucket_targets) > 200),
        "bucket_diagnostics_limit": 200,
        "bucket_targets": compact(bucket_targets),
        "bucket_available_exact": compact(bucket_available_exact),
        "bucket_selected": compact(bucket_selected),
    }
    return sample, diagnostics


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _record_features(record: Dict[str, Any]) -> Dict[str, float]:
    core = record.get("core_metrics") or {}
    diagnostic = record.get("diagnostic_metrics") or {}
    validity_details = ((core.get("structural_validity_gate") or {}).get("details") or {})
    quality_details = ((core.get("reference_quality_score") or {}).get("details") or {})
    redundancy_details = ((core.get("shingle_near_duplicate_risk_score") or {}).get("details") or {})
    objective = _objective_components(record)
    return {
        "word_count": _safe_float(record.get("word_count")),
        "quality": _metric_score(record, "core_metrics", "reference_quality_score"),
        "redundancy_risk": _metric_score(record, "core_metrics", "shingle_near_duplicate_risk_score"),
        "exact_duplicate": _metric_score(record, "core_metrics", "exact_duplicate_indicator"),
        "near_duplicate": _metric_score(record, "core_metrics", "shingle_near_duplicate_indicator"),
        "validity_warning_count": _safe_float(validity_details.get("warning_rule_count")),
        "validity_hard_rule_count": _safe_float(validity_details.get("hard_rule_count")),
        "lexical_diversity": _safe_float(quality_details.get("lexical_diversity")),
        "boilerplate_hits": _safe_float(quality_details.get("boilerplate_hits")),
        "useful_recurrence_score": _safe_float(redundancy_details.get("useful_recurrence_score")),
        "intra_chunk_repeat_pressure": _safe_float(redundancy_details.get("intra_chunk_repeat_pressure")),
        "learnability_support": _safe_float(objective.get("learnability_support")),
        "quality_learnability_support": _safe_float(objective.get("quality_learnability_support")),
        "useful_length_support": _safe_float(objective.get("useful_length_support")),
        "quality_tail_penalty": _safe_float(objective.get("quality_tail_penalty")),
        "diagnostic_predictive_utility": _metric_score(record, "diagnostic_metrics", "predictive_utility_proxy"),
        "diagnostic_tail_rarity": _metric_score(record, "diagnostic_metrics", "tail_cluster_rarity_proxy"),
        "diagnostic_explanatory_quality": _metric_score(record, "diagnostic_metrics", "explanatory_quality_proxy"),
    }


def _numeric_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}
    feature_rows = [_record_features(record) for record in records]
    keys = sorted(feature_rows[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in feature_rows], dtype=float)
        summary[key] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
            "p10": round(float(np.quantile(values, 0.10)), 6),
            "p50": round(float(np.quantile(values, 0.50)), 6),
            "p90": round(float(np.quantile(values, 0.90)), 6),
        }
    return summary


def _bucket_counts(records: Sequence[Dict[str, Any]], bucket_fn: Any, *, limit: int = 15) -> Dict[str, Any]:
    counts = Counter(str(bucket_fn(record)) for record in records)
    total = max(1, sum(counts.values()))
    top = [
        {"bucket": bucket, "count": int(count), "share": round(float(count) / total, 6)}
        for bucket, count in counts.most_common(limit)
    ]
    return {
        "total_buckets": int(len(counts)),
        "top": top,
    }


def _distribution_summaries(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "quality_band": _bucket_counts(records, _quality_band_from_scored_record),
        "length_bucket": _bucket_counts(records, _length_bucket_from_scored_record),
        "style_bucket": _bucket_counts(records, _style_bucket_from_scored_record),
        "domain_bucket": _bucket_counts(records, _domain_bucket_from_scored_record),
    }


def _compare_numeric(
    selected_summary: Dict[str, Dict[str, float]],
    baseline_summary: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    comparison: Dict[str, Dict[str, float]] = {}
    for key in sorted(set(selected_summary) & set(baseline_summary)):
        selected = selected_summary[key]
        baseline = baseline_summary[key]
        selected_mean = float(selected.get("mean") or 0.0)
        baseline_mean = float(baseline.get("mean") or 0.0)
        pooled_std = math.sqrt(
            max(0.0, (float(selected.get("std") or 0.0) ** 2 + float(baseline.get("std") or 0.0) ** 2) / 2.0)
        )
        delta = selected_mean - baseline_mean
        comparison[key] = {
            "selected_mean": round(selected_mean, 6),
            "baseline_mean": round(baseline_mean, 6),
            "delta_selected_minus_baseline": round(delta, 6),
            "standardized_delta": round(delta / pooled_std, 6) if pooled_std > 0 else 0.0,
        }
    return comparison


def _mean_delta(comparison: Dict[str, Dict[str, float]], key: str) -> float:
    return _safe_float((comparison.get(key) or {}).get("delta_selected_minus_baseline"))


def _classify_difference(comparison: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    quality_delta = _mean_delta(comparison, "quality")
    learnability_delta = _mean_delta(comparison, "learnability_support")
    risk_delta = _mean_delta(comparison, "redundancy_risk")
    word_delta = _mean_delta(comparison, "word_count")
    material = [
        key
        for key, payload in comparison.items()
        if abs(float(payload.get("standardized_delta") or 0.0)) >= 0.10
    ]
    if quality_delta > 0.01 and learnability_delta > 0.01 and risk_delta <= 0.01:
        verdict = "selected_meaningfully_stronger"
    elif abs(quality_delta) <= 0.005 and abs(learnability_delta) <= 0.005 and abs(risk_delta) <= 0.005:
        verdict = "selected_near_baseline"
    elif quality_delta > 0.0 and learnability_delta <= 0.0:
        verdict = "quality_gain_without_learnability_gain"
    elif quality_delta <= 0.0 and learnability_delta <= 0.0:
        verdict = "selected_not_stronger_on_core_learning_features"
    else:
        verdict = "mixed_effects"
    return {
        "verdict": verdict,
        "quality_delta": round(quality_delta, 6),
        "learnability_delta": round(learnability_delta, 6),
        "redundancy_risk_delta": round(risk_delta, 6),
        "word_count_delta": round(word_delta, 3),
        "material_standardized_differences": material[:20],
    }


def _summarize_arm(records: Sequence[Dict[str, Any]], *, source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "records": int(len(records)),
        "numeric": _numeric_summary(records),
        "distributions": _distribution_summaries(records),
    }


def _top_comparison_rows(comparison: Dict[str, Dict[str, float]], *, limit: int = 10) -> List[Dict[str, Any]]:
    rows = []
    for key, payload in comparison.items():
        rows.append(
            {
                "feature": key,
                **payload,
            }
        )
    rows.sort(key=lambda row: abs(float(row.get("standardized_delta") or 0.0)), reverse=True)
    return rows[:limit]


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| Feature | Selected Mean | Baseline Mean | Delta | Std. Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {selected_mean:.6f} | {baseline_mean:.6f} | {delta_selected_minus_baseline:.6f} | {standardized_delta:.3f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Selector Baseline Audit",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Seed: `{report['seed']}`",
        f"- Baselines: `stageA_random`, `multi_matched_stageA_random`",
        "",
        "This report answers whether Stage B selection creates a subset that is measurably different from feasible random and strict matched baselines.",
        "",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        utility = payload.get("latest_utility_evidence") or {}
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Selected records: `{payload.get('selected_records')}`",
                f"- Selected sample records: `{payload.get('selected_sample_records')}`",
                f"- Stage-A records: `{payload.get('stage_a_records')}`",
                f"- Utility evidence tier: `{utility.get('evidence_tier')}`",
                f"- Utility failure reason: `{utility.get('failure_reason')}`",
                "",
            ]
        )
        for baseline_name, comparison_payload in (payload.get("comparisons") or {}).items():
            verdict = comparison_payload.get("verdict") or {}
            lines.extend(
                [
                    f"### selected vs {baseline_name}",
                    "",
                    f"- Baseline records used: `{comparison_payload.get('baseline_records')}`",
                    f"- Verdict: `{verdict.get('verdict')}`",
                    f"- Quality delta: `{verdict.get('quality_delta')}`",
                    f"- Learnability delta: `{verdict.get('learnability_delta')}`",
                    f"- Redundancy risk delta: `{verdict.get('redundancy_risk_delta')}`",
                    "",
                    _markdown_table(comparison_payload.get("top_numeric_differences") or []),
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(
    profile: str,
    datasets: Sequence[str] | None,
    seed: int,
    pool_multiplier: int,
    sample_size: int,
) -> Dict[str, Any]:
    run_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    profile = _resolve_profile(run_summary, profile)
    profile_payload = (run_summary.get("profiles") or {}).get(profile) or {}
    dataset_names = list(datasets or [name for name in profile_payload.keys() if not str(name).startswith("_")])
    report: Dict[str, Any] = {
        "schema_version": "selector-baseline-audit-v1",
        "profile": profile,
        "seed": int(seed),
        "pool_multiplier": int(pool_multiplier),
        "arm_sample_size": int(sample_size),
        "purpose": "Quantify whether Stage-B selected subsets differ from Stage-A random and multi-matched Stage-A baselines.",
        "datasets": {},
    }
    for dataset in dataset_names:
        print(f"[15] dataset start: {dataset}", flush=True)
        scored_path = SCORED_DIR / f"{dataset}.jsonl"
        selected_uids = _load_selected_uids(run_summary, profile, str(dataset))
        scored_records = list(iter_jsonl_records_resilient(scored_path))
        selected_records = [record for record in scored_records if str(record.get("chunk_uid") or "") in selected_uids]
        stage_a_records = [_ for _ in scored_records if _stage_a_pass(_)]
        selected_count = len(selected_records)
        arm_n = min(max(1, int(sample_size)), selected_count)
        selected_sample = _choose_by_stable_hash(selected_records, arm_n, seed + 3)
        selected_uid_set = {str(record.get("chunk_uid") or "") for record in selected_records}
        stage_a_candidates = [record for record in stage_a_records if str(record.get("chunk_uid") or "") not in selected_uid_set]
        stage_a_random = _choose_by_stable_hash(stage_a_candidates, arm_n, seed)
        multi_matched, multi_matched_diagnostics = _build_fast_multi_matched_sample(
            baseline_records=stage_a_records,
            selected_records=selected_sample,
            seed=seed,
            pool_multiplier=pool_multiplier,
        )
        arms = {
            "selected": _summarize_arm(selected_sample, source="selected_subset_sample"),
            "stageA_random": _summarize_arm(stage_a_random, source="stage_a_random_excluding_selected"),
            "multi_matched_stageA_random": _summarize_arm(multi_matched, source="multi_matched_pool_excluding_selected"),
        }
        selected_numeric = arms["selected"]["numeric"]
        comparisons = {}
        for baseline_name in ("stageA_random", "multi_matched_stageA_random"):
            numeric_comparison = _compare_numeric(selected_numeric, arms[baseline_name]["numeric"])
            comparisons[baseline_name] = {
                "baseline_records": int(arms[baseline_name]["records"]),
                "numeric_comparison": numeric_comparison,
                "top_numeric_differences": _top_comparison_rows(numeric_comparison),
                "verdict": _classify_difference(numeric_comparison),
            }
        meta = profile_payload.get(str(dataset)) or {}
        aggregate = ((meta.get("utility_probe_details") or {}).get("aggregate") or {})
        evidence = aggregate.get("utility_evidence_summary") or {}
        report["datasets"][str(dataset)] = {
            "selected_records": int(selected_count),
            "selected_sample_records": int(len(selected_sample)),
            "stage_a_records": int(len(stage_a_records)),
            "stage_a_candidate_records_excluding_selected": int(len(stage_a_candidates)),
            "multi_matched_pool_records": int(multi_matched_diagnostics.get("matched_pool_count") or 0),
            "multi_matched_pool_diagnostics": multi_matched_diagnostics,
            "latest_utility_evidence": {
                "evidence_tier": evidence.get("evidence_tier"),
                "failure_reason": evidence.get("failure_reason"),
                "utility_probe_valid": evidence.get("utility_probe_valid"),
                "utility_strict_pass": evidence.get("utility_strict_pass"),
                "curation_benefit_status": (evidence.get("curation_benefit_status") or {}).get("status"),
                "strict_counterfactual_status": (evidence.get("strict_counterfactual_status") or {}).get("status"),
            },
            "arms": arms,
            "comparisons": comparisons,
        }
        stage_a_random_count = len(stage_a_random)
        multi_matched_count = len(multi_matched)
        random_verdict = comparisons["stageA_random"]["verdict"]["verdict"]
        matched_verdict = comparisons["multi_matched_stageA_random"]["verdict"]["verdict"]
        del (
            scored_records,
            selected_records,
            stage_a_records,
            stage_a_candidates,
            stage_a_random,
            multi_matched,
            selected_sample,
            arms,
            comparisons,
        )
        gc.collect()
        print(
            f"[15] dataset done: {dataset} "
            f"selected={selected_count} stageA_random={stage_a_random_count} multi_matched={multi_matched_count} "
            f"vs_random={random_verdict} "
            f"vs_matched={matched_verdict}",
            flush=True,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit selected subsets against Stage-A random and multi-matched baselines.")
    parser.add_argument("--profile", default="canonical")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-multiplier", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_audit(
        profile=str(args.profile),
        datasets=args.datasets,
        seed=int(args.seed),
        pool_multiplier=int(args.pool_multiplier),
        sample_size=int(args.sample_size),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[15] write json start: {args.json_output}", flush=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[15] write json done: {args.json_output}", flush=True)
    print(f"[15] write markdown start: {args.md_output}", flush=True)
    _write_markdown(report, args.md_output)
    print(f"[15] write markdown done: {args.md_output}", flush=True)
    print(f"[15] selector baseline audit json: {args.json_output}", flush=True)
    print(f"[15] selector baseline audit md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
