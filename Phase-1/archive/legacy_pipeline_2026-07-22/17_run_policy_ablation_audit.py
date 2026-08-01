#!/usr/bin/env python3
"""Audit selector policy ablations without rerunning the small-LM Utility probe.

The goal is to isolate whether coverage balancing and top-quality anti-collapse
are suppressing high-quality chunks before we spend time on full Utility runs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from data_eval_common import DEFAULT_PROFILE_CONFIG, OUTPUT_DIR, load_json, iter_jsonl_records_resilient
from policy.subsets import (
    _cluster_id,
    _coverage_retention,
    _coverage_strategy,
    _distribution_bucket_support,
    _domain_bucket_from_scored_record,
    _estimate_metric_quantile,
    _length_bucket_from_scored_record,
    _objective_components,
    _passes_gates,
    _quality_band_from_scored_record,
    _quality_score_from_scored_record,
    _select_with_objective_constraints,
    _selector_config,
    _stage_a_gate,
    _stage_b_rank,
    _stage_c_validation,
    _style_bucket_from_scored_record,
    SCORING_MANIFEST_PATH,
)

DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "policy_ablation_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "policy_ablation_audit.md"


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


def _metric_score(record: Dict[str, Any], group: str, metric: str) -> float:
    payload = (record.get(group) or {}).get(metric) or {}
    return _safe_float(payload.get("score"), 0.0)


def _stage_a_pass_hard(record: Dict[str, Any]) -> bool:
    return bool(
        _metric_score(record, "core_metrics", "structural_validity_gate") >= 1.0
        and _metric_score(record, "core_metrics", "exact_duplicate_indicator") <= 0.0
        and _metric_score(record, "core_metrics", "shingle_near_duplicate_indicator") <= 0.0
    )


def _record_features(record: Dict[str, Any]) -> Dict[str, float]:
    objective = _objective_components(record)
    return {
        "quality": _metric_score(record, "core_metrics", "reference_quality_score"),
        "redundancy_risk": _metric_score(record, "core_metrics", "shingle_near_duplicate_risk_score"),
        "learnability_support": _safe_float(objective.get("learnability_support")),
        "quality_learnability_support": _safe_float(objective.get("quality_learnability_support")),
        "quality_tail_penalty": _safe_float(objective.get("quality_tail_penalty")),
        "word_count": _safe_float(record.get("word_count")),
        "diagnostic_predictive_utility": _metric_score(record, "diagnostic_metrics", "predictive_utility_proxy"),
    }


def _mean(records: Sequence[Dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(sum(_record_features(record)[key] for record in records) / float(len(records)))


def _quantile(records: Sequence[Dict[str, Any]], key: str, q: float) -> float:
    if not records:
        return 0.0
    arr = np.asarray([_record_features(record)[key] for record in records], dtype=float)
    return float(np.quantile(arr, q))


def _counter_share(counter: Counter[Any], total: int, limit: int = 12) -> List[Dict[str, Any]]:
    denom = max(1, int(total))
    return [
        {"key": str(key), "count": int(count), "share": round(float(count) / float(denom), 6)}
        for key, count in counter.most_common(limit)
    ]


def _distribution_similarity(selected: Sequence[Dict[str, Any]], original_counts: Counter[str], bucket_fn: Any) -> float:
    selected_counts: Counter[str] = Counter(str(bucket_fn(record) or "unknown") for record in selected)
    return float((_distribution_bucket_support(selected_counts, original_counts) or {}).get("distribution_similarity") or 0.0)


def _selected_summary(
    *,
    selected: Sequence[Dict[str, Any]],
    original_clusters: Counter[int],
    original_domain_counts: Counter[str],
    original_style_counts: Counter[str],
    original_length_counts: Counter[str],
    source_records: int,
) -> Dict[str, Any]:
    selected_clusters = Counter(_cluster_id(record) for record in selected)
    coverage = _coverage_retention(selected_clusters, original_clusters)
    qband_counts = Counter(_quality_band_from_scored_record(record) for record in selected)
    high_quality = sum(1 for record in selected if _quality_score_from_scored_record(record) >= 0.90)
    very_high_quality = sum(1 for record in selected if _quality_score_from_scored_record(record) >= 0.95)
    top_tail = sum(1 for record in selected if _quality_score_from_scored_record(record) >= 0.99)
    return {
        "selected_records": int(len(selected)),
        "selected_ratio": round(float(len(selected)) / float(max(source_records, 1)), 6),
        "coverage_score": round(float(coverage.get("score") or 0.0), 6),
        "rare_cluster_retention": round(float(coverage.get("rare_cluster_retention") or 0.0), 6),
        "rare_cluster_retained_count": int(coverage.get("rare_cluster_retained_count") or 0),
        "mean_quality": round(_mean(selected, "quality"), 6),
        "p90_quality": round(_quantile(selected, "quality", 0.90), 6),
        "mean_learnability_support": round(_mean(selected, "learnability_support"), 6),
        "mean_redundancy_risk": round(_mean(selected, "redundancy_risk"), 6),
        "mean_predictive_utility_proxy": round(_mean(selected, "diagnostic_predictive_utility"), 6),
        "high_quality_share": round(float(high_quality) / float(max(len(selected), 1)), 6),
        "very_high_quality_share": round(float(very_high_quality) / float(max(len(selected), 1)), 6),
        "top_tail_quality_share": round(float(top_tail) / float(max(len(selected), 1)), 6),
        "domain_distribution_similarity": round(_distribution_similarity(selected, original_domain_counts, _domain_bucket_from_scored_record), 6),
        "style_distribution_similarity": round(_distribution_similarity(selected, original_style_counts, _style_bucket_from_scored_record), 6),
        "length_distribution_similarity": round(_distribution_similarity(selected, original_length_counts, _length_bucket_from_scored_record), 6),
        "quality_band_counts_top": _counter_share(qband_counts, len(selected)),
    }


def _summary_delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "selected_ratio",
        "coverage_score",
        "rare_cluster_retention",
        "mean_quality",
        "mean_learnability_support",
        "mean_redundancy_risk",
        "mean_predictive_utility_proxy",
        "high_quality_share",
        "very_high_quality_share",
        "top_tail_quality_share",
        "domain_distribution_similarity",
        "style_distribution_similarity",
        "length_distribution_similarity",
    ]
    return {key: round(float(current.get(key) or 0.0) - float(baseline.get(key) or 0.0), 6) for key in keys}


def _variant_profiles(base_profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    variants: Dict[str, Dict[str, Any]] = {}

    def clone() -> Dict[str, Any]:
        return copy.deepcopy(base_profile)

    variants["canonical"] = clone()

    p = clone()
    p.setdefault("selector", {})["preserve_quality_band_distribution"] = False
    p.setdefault("selector", {})["diagnose_quality_band_distribution"] = True
    p.setdefault("selector", {})["quality_band_rebalance_mode"] = "diagnostic_only"
    variants["no_top_quality_anti_collapse"] = p

    p = clone()
    p.setdefault("selector", {})["quality_top_band_max_share"] = 0.20
    p.setdefault("selector", {})["quality_band_max_swap_ratio"] = 0.02
    variants["relaxed_top_quality_cap"] = p

    p = clone()
    p.setdefault("coverage_strategy", {})["ensure_rare_cluster_exemplars"] = False
    p.setdefault("selector", {})["preserve_domain_bucket_exemplars"] = False
    p.setdefault("selector", {})["preserve_style_bucket_exemplars"] = False
    p.setdefault("selector", {})["preserve_domain_distribution"] = False
    p.setdefault("selector", {})["preserve_style_distribution"] = False
    p.setdefault("selector", {})["preserve_length_distribution"] = False
    p.setdefault("selector", {})["preserve_quality_band_distribution"] = False
    variants["quality_first_coverage_light"] = p

    p = clone()
    p.setdefault("selector", {})["preserve_quality_band_distribution"] = False
    p.setdefault("selector", {})["quality_band_rebalance_mode"] = "diagnostic_only"
    p.setdefault("selector", {})["enable_learnability_rebalance"] = True
    p.setdefault("selector", {})["learnability_rebalance_max_swap_ratio"] = 0.10
    p.setdefault("selector", {})["learnability_rebalance_min_gain"] = 0.04
    p.setdefault("selector", {})["learnability_rebalance_min_quality"] = 0.82
    variants["learnability_rescue_no_anti_collapse"] = p

    p = clone()
    adjustments = p.setdefault("selector", {}).setdefault("selection_adjustments", {})
    adjustments["learnability_support_bonus"] = 0.03
    adjustments["useful_recurrence_bonus"] = 0.02
    adjustments["pattern_recurrence_bonus"] = 0.01
    adjustments["useful_length_bonus"] = 0.08
    p.setdefault("selector", {})["enable_learnability_rebalance"] = True
    p.setdefault("selector", {})["learnability_rebalance_max_swap_ratio"] = 0.02
    p.setdefault("selector", {})["learnability_rebalance_min_gain"] = 0.12
    p.setdefault("selector", {})["learnability_rebalance_min_quality"] = 0.84
    variants["core_proxy_length_recurrence_guard"] = p

    p = clone()
    p.setdefault("selector", {}).setdefault("objective_weights", {})["redundancy_risk"] = 0.28
    adjustments = p.setdefault("selector", {}).setdefault("selection_adjustments", {})
    adjustments["learnability_support_bonus"] = 0.02
    adjustments["useful_recurrence_bonus"] = 0.0
    adjustments["pattern_recurrence_bonus"] = 0.0
    adjustments["useful_length_bonus"] = 0.09
    p.setdefault("selector", {})["enable_learnability_rebalance"] = False
    variants["core_proxy_no_recurrence_relief"] = p

    return variants


def _prepare_candidates(
    *,
    scored_path: Path,
    profile: Dict[str, Any],
    stage_b: Dict[str, Any],
    max_records: int,
) -> tuple[List[Dict[str, Any]], Counter[int], Counter[str], Counter[str], Counter[str], int]:
    candidates: List[Dict[str, Any]] = []
    original_clusters: Counter[int] = Counter()
    original_domain_counts: Counter[str] = Counter()
    original_style_counts: Counter[str] = Counter()
    original_length_counts: Counter[str] = Counter()
    processed = 0
    for record in iter_jsonl_records_resilient(scored_path):
        if max_records > 0 and processed >= max_records:
            break
        processed += 1
        original_clusters[_cluster_id(record)] += 1
        original_domain_counts[_domain_bucket_from_scored_record(record)] += 1
        original_style_counts[_style_bucket_from_scored_record(record)] += 1
        original_length_counts[_length_bucket_from_scored_record(record)] += 1
        record["selection"] = {
            "profile": "policy_ablation",
            "axis_scores": {},
            "stage_a_gate_passed": False,
            "stage_b_rank_score": None,
            "stage_b_rank_passed": False,
            "accepted": False,
            "accepted_by": None,
        }
        if not _passes_gates(record, profile):
            continue
        candidates.append(record)
    return candidates, original_clusters, original_domain_counts, original_style_counts, original_length_counts, processed


def _high_quality_recovery(
    *,
    candidates: Sequence[Dict[str, Any]],
    canonical_uids: set[str],
    variant_uids: set[str],
) -> Dict[str, Any]:
    canonical_rejected_high_quality = {
        str(record.get("chunk_uid") or "")
        for record in candidates
        if str(record.get("chunk_uid") or "") not in canonical_uids
        and _quality_score_from_scored_record(record) >= 0.90
        and _safe_float(_objective_components(record).get("learnability_support")) >= 0.60
    }
    recovered = canonical_rejected_high_quality & variant_uids
    lost_from_canonical = {
        uid for uid in canonical_uids if uid not in variant_uids
    }
    return {
        "canonical_rejected_high_quality_pool": int(len(canonical_rejected_high_quality)),
        "recovered_high_quality_records": int(len(recovered)),
        "recovered_high_quality_share": round(float(len(recovered)) / float(max(len(canonical_rejected_high_quality), 1)), 6),
        "lost_canonical_records": int(len(lost_from_canonical)),
        "jaccard_vs_canonical": round(float(len(canonical_uids & variant_uids)) / float(max(len(canonical_uids | variant_uids), 1)), 6),
    }


def build_audit(*, profile_name: str, datasets: Sequence[str] | None, max_records: int) -> Dict[str, Any]:
    config = load_json(DEFAULT_PROFILE_CONFIG)
    base_profile = ((config.get("profiles") or {}).get(profile_name) or {})
    if not base_profile:
        raise RuntimeError(f"profile {profile_name!r} not found in {DEFAULT_PROFILE_CONFIG}")
    scoring_manifest = load_json(SCORING_MANIFEST_PATH)
    dataset_names = list(datasets or sorted((scoring_manifest.get("datasets") or {}).keys()))
    variants = _variant_profiles(base_profile)
    report: Dict[str, Any] = {
        "schema_version": "policy-ablation-audit-v1",
        "profile": profile_name,
        "purpose": "Compare selector policy variants before rerunning expensive Utility probes.",
        "max_records_per_dataset": int(max_records),
        "sample_policy": "prefix-bounded scored-record audit; use --max-records 0 for expensive full audit",
        "variants": {
            "canonical": "current policy",
            "no_top_quality_anti_collapse": "disable quality-band anti-collapse while keeping other selector policies",
            "relaxed_top_quality_cap": "keep anti-collapse but allow larger q>=0.99 share and fewer swaps",
            "quality_first_coverage_light": "disable coverage/distribution preservation to measure the quality-first extreme",
            "learnability_rescue_no_anti_collapse": "disable anti-collapse and add stronger same-bucket learnability rescue",
            "core_proxy_length_recurrence_guard": "reduce learnability/repetition bonuses and increase useful-length support for tiny Core-proxy calibration",
            "core_proxy_no_recurrence_relief": "remove recurrence relief/learnability rescue and increase redundancy pressure for tiny Core-proxy calibration",
        },
        "datasets": {},
    }

    for dataset in dataset_names:
        meta = (scoring_manifest.get("datasets") or {}).get(str(dataset)) or {}
        scored_path = Path(str(meta.get("path") or ""))
        if not scored_path.exists():
            raise FileNotFoundError(f"scored dataset missing: {scored_path}")
        print(f"[17] dataset start: {dataset}", flush=True)

        # Use the canonical risk quantile for all variants so the ablation isolates selector policy.
        stage_b_for_scan = _stage_b_rank(base_profile)
        risk_quantile = stage_b_for_scan.get("near_duplicate_risk_quantile_ceiling")
        if risk_quantile is not None:
            quantile_ceiling = _estimate_metric_quantile(
                scored_path,
                "shingle_near_duplicate_risk_score",
                float(risk_quantile),
                sample_size=int(stage_b_for_scan.get("near_duplicate_risk_quantile_sample_size") or 60000),
                seed=42,
            )
            stage_b_for_scan["near_duplicate_risk_ceiling"] = min(
                float(stage_b_for_scan["near_duplicate_risk_ceiling"]),
                float(quantile_ceiling),
            )
            stage_b_for_scan["near_duplicate_risk_quantile_ceiling_value"] = round(float(quantile_ceiling), 6)

        candidates, original_clusters, domain_counts, style_counts, length_counts, processed = _prepare_candidates(
            scored_path=scored_path,
            profile=base_profile,
            stage_b=stage_b_for_scan,
            max_records=int(max_records),
        )
        print(f"[17] candidates: {dataset} processed={processed} candidates={len(candidates)}", flush=True)
        dataset_report: Dict[str, Any] = {
            "processed_records": int(processed),
            "stage_a_candidates": int(len(candidates)),
            "variant_results": {},
        }
        canonical_selected_uids: set[str] = set()
        canonical_summary: Dict[str, Any] | None = None

        for variant_name, variant_profile in variants.items():
            print(f"[17] variant start: {dataset}:{variant_name}", flush=True)
            stage_b = copy.deepcopy(stage_b_for_scan)
            selector_cfg = _selector_config(variant_profile)
            stage_c = _stage_c_validation(variant_profile)
            strategy = _coverage_strategy(variant_profile, original_clusters)
            selected, selector_diag = _select_with_objective_constraints(
                candidates=candidates,
                stage_b=stage_b,
                selector_cfg=selector_cfg,
                strategy=strategy,
                original_clusters=original_clusters,
                source_records=int(processed),
                stage_c=stage_c,
            )
            selected_uids = {str(record.get("chunk_uid") or "") for record in selected}
            summary = _selected_summary(
                selected=selected,
                original_clusters=original_clusters,
                original_domain_counts=domain_counts,
                original_style_counts=style_counts,
                original_length_counts=length_counts,
                source_records=int(processed),
            )
            last_iteration = ((selector_diag.get("iterations") or [])[-1] if selector_diag.get("iterations") else {})
            quota_diag = last_iteration.get("quota_diagnostics") or {}
            summary["selector_constraints_satisfied"] = bool(selector_diag.get("selector_constraints_satisfied"))
            summary["coverage_constraints_satisfied"] = bool(selector_diag.get("coverage_constraints_satisfied"))
            summary["quality_band_policy"] = ((quota_diag.get("quality_band_distribution_balance") or {}).get("policy") or "unknown")
            summary["quality_band_swap_count"] = int((quota_diag.get("quality_band_distribution_balance") or {}).get("swap_count") or 0)
            summary["learnability_swap_count"] = int((quota_diag.get("learnability_rebalance") or {}).get("swap_count") or 0)
            summary["accepted_by_counts"] = _counter_share(
                Counter(str((record.get("selection") or {}).get("accepted_by") or "unknown") for record in selected),
                len(selected),
            )
            if variant_name == "canonical":
                canonical_selected_uids = set(selected_uids)
                canonical_summary = dict(summary)
                summary["delta_vs_canonical"] = {key: 0.0 for key in _summary_delta(summary, summary)}
                summary["high_quality_recovery_vs_canonical"] = {
                    "canonical_rejected_high_quality_pool": 0,
                    "recovered_high_quality_records": 0,
                    "recovered_high_quality_share": 0.0,
                    "lost_canonical_records": 0,
                    "jaccard_vs_canonical": 1.0,
                }
            else:
                assert canonical_summary is not None
                summary["delta_vs_canonical"] = _summary_delta(summary, canonical_summary)
                summary["high_quality_recovery_vs_canonical"] = _high_quality_recovery(
                    candidates=candidates,
                    canonical_uids=canonical_selected_uids,
                    variant_uids=selected_uids,
                )
            dataset_report["variant_results"][variant_name] = summary
            print(
                f"[17] variant done: {dataset}:{variant_name} selected={summary['selected_records']} "
                f"coverage={summary['coverage_score']:.3f} quality={summary['mean_quality']:.3f} "
                f"learnability={summary['mean_learnability_support']:.3f} top_tail={summary['top_tail_quality_share']:.3f}",
                flush=True,
            )
            gc.collect()
        report["datasets"][str(dataset)] = dataset_report
        gc.collect()
    return report


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Policy Ablation Audit",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Max records per dataset: `{report.get('max_records_per_dataset')}`",
        "- Utility probe: not rerun in this audit.",
        "",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.extend([
            f"## {dataset}",
            "",
            f"- Processed records: `{payload.get('processed_records')}`",
            f"- Stage-A candidates: `{payload.get('stage_a_candidates')}`",
            "",
            "| Variant | Selected | Coverage | Quality | Learnability | Predictive proxy | Redundancy risk | Top-tail share | Domain sim | Style sim | Length sim | HQ recovered | Jaccard |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for variant, result in (payload.get("variant_results") or {}).items():
            recovery = result.get("high_quality_recovery_vs_canonical") or {}
            lines.append(
                f"| {variant} | {result.get('selected_records')} | {float(result.get('coverage_score') or 0):.3f} | "
                f"{float(result.get('mean_quality') or 0):.3f} | {float(result.get('mean_learnability_support') or 0):.3f} | "
                f"{float(result.get('mean_predictive_utility_proxy') or 0):.3f} | "
                f"{float(result.get('mean_redundancy_risk') or 0):.3f} | {float(result.get('top_tail_quality_share') or 0):.3f} | "
                f"{float(result.get('domain_distribution_similarity') or 0):.3f} | {float(result.get('style_distribution_similarity') or 0):.3f} | "
                f"{float(result.get('length_distribution_similarity') or 0):.3f} | {recovery.get('recovered_high_quality_records', 0)} | "
                f"{float(recovery.get('jaccard_vs_canonical') or 0):.3f} |"
            )
        lines.append("")
        lines.append("### Deltas vs Canonical")
        lines.append("")
        lines.append("| Variant | dCoverage | dQuality | dLearnability | dPredictive | dRedundancy | dTopTail | dDomainSim | dStyleSim | dLengthSim |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for variant, result in (payload.get("variant_results") or {}).items():
            if variant == "canonical":
                continue
            delta = result.get("delta_vs_canonical") or {}
            lines.append(
                f"| {variant} | {float(delta.get('coverage_score') or 0):+.4f} | {float(delta.get('mean_quality') or 0):+.4f} | "
                f"{float(delta.get('mean_learnability_support') or 0):+.4f} | {float(delta.get('mean_predictive_utility_proxy') or 0):+.4f} | "
                f"{float(delta.get('mean_redundancy_risk') or 0):+.4f} | "
                f"{float(delta.get('top_tail_quality_share') or 0):+.4f} | {float(delta.get('domain_distribution_similarity') or 0):+.4f} | "
                f"{float(delta.get('style_distribution_similarity') or 0):+.4f} | {float(delta.get('length_distribution_similarity') or 0):+.4f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selector policy ablation audit without small-LM Utility reruns.")
    parser.add_argument("--profile", default="canonical")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-records", type=int, default=150000, help="Bounded audit sample per dataset. Use 0 only for expensive full scored dataset runs.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_audit(profile_name=str(args.profile), datasets=args.datasets, max_records=int(args.max_records))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, args.md_output)
    print(f"[17] policy ablation audit json: {args.json_output}", flush=True)
    print(f"[17] policy ablation audit md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
