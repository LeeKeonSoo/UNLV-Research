#!/usr/bin/env python3
"""Audit high-quality Stage-A chunks that were not selected by Stage B."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, SCORED_DIR, iter_jsonl_records_resilient
from policy.subsets import (
    _cluster_id,
    _domain_bucket_from_scored_record,
    _length_bucket_from_scored_record,
    _objective_components,
    _quality_band_from_scored_record,
    _stable_hash_score,
    _style_bucket_from_scored_record,
)


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "good_chunk_dropout_audit.json"
DEFAULT_EXAMPLES_OUTPUT = OUTPUT_DIR / "validation" / "good_chunk_dropout_examples.jsonl"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "good_chunk_dropout_audit.md"


def _metric_score(record: Dict[str, Any], group: str, metric: str) -> float:
    payload = (record.get(group) or {}).get(metric) or {}
    try:
        return float(payload.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
            f"[16] requested profile={requested_profile!r} not found; using only available profile={fallback!r}",
            flush=True,
        )
        return fallback
    raise RuntimeError(f"Requested profile {requested_profile!r} not found. Available profiles: {names}")


def _load_selected_records(run_summary: Dict[str, Any], profile: str, dataset: str) -> List[Dict[str, Any]]:
    meta = ((run_summary.get("profiles") or {}).get(profile) or {}).get(dataset) or {}
    path = Path(str(meta.get("output_path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"selected subset missing for {profile}:{dataset}: {path}")
    return list(iter_jsonl_records_resilient(path))


def _feature_row(record: Dict[str, Any]) -> Dict[str, float]:
    objective = _objective_components(record)
    quality_details = (((record.get("core_metrics") or {}).get("reference_quality_score") or {}).get("details") or {})
    redundancy_details = (
        ((record.get("core_metrics") or {}).get("shingle_near_duplicate_risk_score") or {}).get("details") or {}
    )
    validity_details = (((record.get("core_metrics") or {}).get("structural_validity_gate") or {}).get("details") or {})
    return {
        "quality": _metric_score(record, "core_metrics", "reference_quality_score"),
        "learnability_support": _safe_float(objective.get("learnability_support")),
        "quality_learnability_support": _safe_float(objective.get("quality_learnability_support")),
        "redundancy_risk": _metric_score(record, "core_metrics", "shingle_near_duplicate_risk_score"),
        "word_count": _safe_float(record.get("word_count")),
        "quality_tail_penalty": _safe_float(objective.get("quality_tail_penalty")),
        "lexical_diversity": _safe_float(quality_details.get("lexical_diversity")),
        "useful_recurrence_score": _safe_float(redundancy_details.get("useful_recurrence_score")),
        "intra_chunk_repeat_pressure": _safe_float(redundancy_details.get("intra_chunk_repeat_pressure")),
        "validity_warning_count": _safe_float(validity_details.get("warning_rule_count")),
        "diagnostic_predictive_utility": _metric_score(record, "diagnostic_metrics", "predictive_utility_proxy"),
        "diagnostic_tail_rarity": _metric_score(record, "diagnostic_metrics", "tail_cluster_rarity_proxy"),
    }


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "p50": round(float(np.quantile(arr, 0.50)), 6),
        "p75": round(float(np.quantile(arr, 0.75)), 6),
        "p90": round(float(np.quantile(arr, 0.90)), 6),
        "p95": round(float(np.quantile(arr, 0.95)), 6),
    }


def _selected_profile(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [_feature_row(record) for record in records]
    cluster_counts = Counter(_cluster_id(record) for record in records)
    return {
        "records": int(len(records)),
        "quality": _quantiles([row["quality"] for row in rows]),
        "learnability_support": _quantiles([row["learnability_support"] for row in rows]),
        "redundancy_risk": _quantiles([row["redundancy_risk"] for row in rows]),
        "quality_band_counts": Counter(_quality_band_from_scored_record(record) for record in records),
        "domain_counts": Counter(_domain_bucket_from_scored_record(record) for record in records),
        "style_counts": Counter(_style_bucket_from_scored_record(record) for record in records),
        "length_counts": Counter(_length_bucket_from_scored_record(record) for record in records),
        "cluster_counts": cluster_counts,
    }


def _latest_quota_diagnostics(run_summary: Dict[str, Any], profile: str, dataset: str) -> Dict[str, Any]:
    meta = ((run_summary.get("profiles") or {}).get(profile) or {}).get(dataset) or {}
    iterations = ((meta.get("selector_diagnostics") or {}).get("iterations") or [])
    if not iterations:
        return {}
    return (iterations[-1].get("quota_diagnostics") or {})


def _saturation_reason(
    *,
    bucket_name: str,
    bucket_value: str,
    quota_diagnostics: Dict[str, Any],
) -> str | None:
    if bucket_name == "quality_band":
        payload = quota_diagnostics.get("quality_band_distribution_balance") or {}
    elif bucket_name in {"domain", "style", "length"}:
        payload = (
            quota_diagnostics.get(f"{bucket_name}_distribution_balance")
            or (quota_diagnostics.get("distribution_balance") or {}).get(bucket_name)
            or {}
        )
    else:
        return None
    target = (payload.get("target_bucket_counts") or {}).get(bucket_value)
    selected = (payload.get("selected_bucket_counts_after") or {}).get(bucket_value)
    selected_before = (payload.get("selected_bucket_counts_before") or {}).get(bucket_value)
    if target is None or selected is None or selected_before is None:
        return None
    # Only call it a dropout pressure when the policy actively reduced an overfull bucket.
    if int(selected_before) > int(target) and int(selected) <= int(selected_before):
        return f"{bucket_name}_bucket_saturated"
    return None


def _classify_dropout(
    record: Dict[str, Any],
    *,
    selected_profile: Dict[str, Any],
    quota_diagnostics: Dict[str, Any],
    original_cluster_counts: Counter[int],
    selected_cluster_counts: Counter[int],
    selection_ratio: float,
) -> List[str]:
    row = _feature_row(record)
    reasons: List[str] = []
    qband = _quality_band_from_scored_record(record)
    domain = _domain_bucket_from_scored_record(record)
    style = _style_bucket_from_scored_record(record)
    length = _length_bucket_from_scored_record(record)
    for bucket_name, bucket_value in (
        ("quality_band", qband),
        ("domain", domain),
        ("style", style),
        ("length", length),
    ):
        reason = _saturation_reason(bucket_name=bucket_name, bucket_value=bucket_value, quota_diagnostics=quota_diagnostics)
        if reason:
            reasons.append(reason)
    quality_balance = quota_diagnostics.get("quality_band_distribution_balance") or {}
    if (
        bool(quality_balance.get("enabled"))
        and qband == str(quality_balance.get("top_band"))
        and int(quality_balance.get("top_band_count_before") or 0) > int(quality_balance.get("top_band_cap") or 0)
    ):
        reasons.append("top_quality_anti_collapse")
    if row["redundancy_risk"] >= float((selected_profile.get("redundancy_risk") or {}).get("p90") or 1.0):
        reasons.append("high_redundancy_risk_relative_to_selected")
    if row["quality_tail_penalty"] >= 0.50:
        reasons.append("quality_tail_penalty")
    cluster_id = _cluster_id(record)
    original_cluster_count = max(1, int(original_cluster_counts.get(cluster_id, 0)))
    selected_cluster_count = int(selected_cluster_counts.get(cluster_id, 0))
    expected_cluster_selected = max(1.0, float(original_cluster_count) * float(selection_ratio))
    if selected_cluster_count >= expected_cluster_selected:
        reasons.append("cluster_saturated")
    if not reasons:
        reasons.append("selection_ratio_capacity_or_tie_break")
    return sorted(set(reasons))


def _dropout_priority(record: Dict[str, Any]) -> float:
    row = _feature_row(record)
    return (
        (1.00 * row["quality"])
        + (0.75 * row["learnability_support"])
        + (0.20 * row["diagnostic_predictive_utility"])
        - (0.35 * row["redundancy_risk"])
    )


def _compact_counter(counter: Counter[Any], limit: int = 20, denominator: int | None = None) -> List[Dict[str, Any]]:
    total = max(1, int(denominator) if denominator is not None else sum(counter.values()))
    return [
        {"key": str(key), "count": int(count), "share": round(float(count) / total, 6)}
        for key, count in counter.most_common(limit)
    ]


def _example_payload(record: Dict[str, Any], reasons: Sequence[str], rank_score: float) -> Dict[str, Any]:
    row = _feature_row(record)
    provenance = record.get("provenance") or {}
    return {
        "chunk_uid": str(record.get("chunk_uid") or ""),
        "dataset": str(record.get("dataset") or ""),
        "source": str(record.get("source") or ""),
        "doc_id": str(record.get("doc_id") or ""),
        "chunk_id": record.get("chunk_id"),
        "rank_score": round(float(rank_score), 6),
        "reasons": list(reasons),
        "quality_band": _quality_band_from_scored_record(record),
        "domain_bucket": _domain_bucket_from_scored_record(record),
        "style_bucket": _style_bucket_from_scored_record(record),
        "length_bucket": _length_bucket_from_scored_record(record),
        "cluster_id": _cluster_id(record),
        "features": {k: round(float(v), 6) for k, v in row.items()},
        "text_preview": str(provenance.get("text_preview") or "")[:500],
    }


def _push_top(heap: List[tuple[float, str, Dict[str, Any]]], payload: Dict[str, Any], limit: int) -> None:
    key = (float(payload["rank_score"]), str(payload["chunk_uid"]), payload)
    if len(heap) < limit:
        heapq.heappush(heap, key)
    elif key[:2] > heap[0][:2]:
        heapq.heapreplace(heap, key)


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Good Chunk Dropout Audit",
        "",
        f"- Profile: `{report['profile']}`",
        f"- High-quality rule: quality >= `{report['high_quality_min_quality']}` and learnability_support >= `{report['high_quality_min_learnability']}`",
        "",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Stage-A records: `{payload['stage_a_records']}`",
                f"- Selected records: `{payload['selected_records']}`",
                f"- Rejected Stage-A records: `{payload['rejected_stage_a_records']}`",
                f"- High-quality rejected records: `{payload['high_quality_rejected_records']}`",
                f"- Very-high-quality rejected records: `{payload['very_high_quality_rejected_records']}`",
                "",
                "### Reason Counts",
                "",
                "| Reason | Count | Share |",
                "|---|---:|---:|",
            ]
        )
        for item in payload.get("reason_counts", []):
            lines.append(f"| {item['key']} | {item['count']} | {item['share']:.3f} |")
        lines.extend(["", "### Top Rejected Examples", ""])
        for ex in payload.get("top_examples", [])[:10]:
            features = ex.get("features") or {}
            lines.extend(
                [
                    f"- `{ex['chunk_uid']}`",
                    f"  - reasons: `{', '.join(ex.get('reasons') or [])}`",
                    f"  - quality: `{features.get('quality')}`, learnability: `{features.get('learnability_support')}`, redundancy risk: `{features.get('redundancy_risk')}`",
                    f"  - bucket: `{ex.get('quality_band')}` / `{ex.get('style_bucket')}` / `{ex.get('length_bucket')}`",
                    f"  - preview: {ex.get('text_preview') or ''}",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(
    *,
    profile: str,
    datasets: Sequence[str] | None,
    min_quality: float,
    min_learnability: float,
    example_limit: int,
) -> Dict[str, Any]:
    run_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    profile = _resolve_profile(run_summary, profile)
    profile_payload = (run_summary.get("profiles") or {}).get(profile) or {}
    dataset_names = list(datasets or [name for name in profile_payload.keys() if not str(name).startswith("_")])
    report: Dict[str, Any] = {
        "schema_version": "good-chunk-dropout-audit-v1",
        "profile": profile,
        "high_quality_min_quality": float(min_quality),
        "high_quality_min_learnability": float(min_learnability),
        "purpose": "Find Stage-A-passing high-quality chunks that were not selected and classify likely dropout causes.",
        "datasets": {},
    }
    examples_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in dataset_names:
        print(f"[16] dataset start: {dataset}", flush=True)
        selected_records = _load_selected_records(run_summary, profile, str(dataset))
        selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
        selected_profile = _selected_profile(selected_records)
        selected_cluster_counts = selected_profile["cluster_counts"]
        quota_diagnostics = _latest_quota_diagnostics(run_summary, profile, str(dataset))
        selection_ratio = float((profile_payload.get(str(dataset)) or {}).get("selection_ratio") or 0.0)
        stage_a_records = 0
        rejected_stage_a_records = 0
        high_quality_rejected_records = 0
        very_high_quality_rejected_records = 0
        original_cluster_counts: Counter[int] = Counter()
        reason_counts: Counter[str] = Counter()
        qband_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        style_counts: Counter[str] = Counter()
        length_counts: Counter[str] = Counter()
        top_heap: List[tuple[float, str, Dict[str, Any]]] = []
        high_quality_feature_values: Dict[str, List[float]] = defaultdict(list)
        scored_path = SCORED_DIR / f"{dataset}.jsonl"
        # First pass: cluster counts for saturation estimates.
        for record in iter_jsonl_records_resilient(scored_path):
            if not _stage_a_pass(record):
                continue
            stage_a_records += 1
            original_cluster_counts[_cluster_id(record)] += 1
        # Second pass: high-quality rejected chunks and dropout reasons.
        for record in iter_jsonl_records_resilient(scored_path):
            if not _stage_a_pass(record):
                continue
            uid = str(record.get("chunk_uid") or "")
            if uid in selected_uids:
                continue
            rejected_stage_a_records += 1
            row = _feature_row(record)
            if row["quality"] < float(min_quality) or row["learnability_support"] < float(min_learnability):
                continue
            high_quality_rejected_records += 1
            if row["quality"] >= 0.95:
                very_high_quality_rejected_records += 1
            reasons = _classify_dropout(
                record,
                selected_profile=selected_profile,
                quota_diagnostics=quota_diagnostics,
                original_cluster_counts=original_cluster_counts,
                selected_cluster_counts=selected_cluster_counts,
                selection_ratio=selection_ratio,
            )
            reason_counts.update(reasons)
            qband_counts[_quality_band_from_scored_record(record)] += 1
            domain_counts[_domain_bucket_from_scored_record(record)] += 1
            style_counts[_style_bucket_from_scored_record(record)] += 1
            length_counts[_length_bucket_from_scored_record(record)] += 1
            for key, value in row.items():
                high_quality_feature_values[key].append(float(value))
            payload = _example_payload(record, reasons, _dropout_priority(record))
            _push_top(top_heap, payload, int(example_limit))
        feature_summary = {
            key: {
                "mean": round(float(np.mean(values)), 6),
                "p50": round(float(np.quantile(values, 0.50)), 6),
                "p90": round(float(np.quantile(values, 0.90)), 6),
            }
            for key, values in high_quality_feature_values.items()
            if values
        }
        top_examples = [
            item[2] for item in sorted(top_heap, key=lambda item: (item[0], item[1]), reverse=True)
        ]
        examples_by_dataset[str(dataset)] = top_examples
        report["datasets"][str(dataset)] = {
            "stage_a_records": int(stage_a_records),
            "selected_records": int(len(selected_records)),
            "rejected_stage_a_records": int(rejected_stage_a_records),
            "high_quality_rejected_records": int(high_quality_rejected_records),
            "very_high_quality_rejected_records": int(very_high_quality_rejected_records),
            "high_quality_rejected_share_of_rejected_stage_a": round(
                float(high_quality_rejected_records) / max(1, rejected_stage_a_records), 6
            ),
            "selected_profile_quantiles": {
                "quality": selected_profile["quality"],
                "learnability_support": selected_profile["learnability_support"],
                "redundancy_risk": selected_profile["redundancy_risk"],
            },
            "high_quality_rejected_feature_summary": feature_summary,
            "reason_counts": _compact_counter(reason_counts, denominator=high_quality_rejected_records),
            "quality_band_counts": _compact_counter(qband_counts),
            "domain_counts_top": _compact_counter(domain_counts),
            "style_counts": _compact_counter(style_counts),
            "length_counts": _compact_counter(length_counts),
            "top_examples": top_examples[: min(20, int(example_limit))],
        }
        print(
            f"[16] dataset done: {dataset} "
            f"rejected_stageA={rejected_stage_a_records} high_quality_rejected={high_quality_rejected_records} "
            f"top_reason={(reason_counts.most_common(1)[0][0] if reason_counts else '-')}",
            flush=True,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit high-quality chunks rejected by Stage-B selection.")
    parser.add_argument("--profile", default="canonical")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--min-quality", type=float, default=0.90)
    parser.add_argument("--min-learnability", type=float, default=0.60)
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--examples-output", type=Path, default=DEFAULT_EXAMPLES_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_audit(
        profile=str(args.profile),
        datasets=args.datasets,
        min_quality=float(args.min_quality),
        min_learnability=float(args.min_learnability),
        example_limit=int(args.example_limit),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.examples_output.open("w", encoding="utf-8") as f:
        for dataset, payload in (report.get("datasets") or {}).items():
            for example in payload.get("top_examples") or []:
                f.write(json.dumps({"dataset": dataset, **example}, ensure_ascii=False) + "\n")
    _write_markdown(report, args.md_output)
    print(f"[16] good chunk dropout audit json: {args.json_output}", flush=True)
    print(f"[16] good chunk dropout examples: {args.examples_output}", flush=True)
    print(f"[16] good chunk dropout audit md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
