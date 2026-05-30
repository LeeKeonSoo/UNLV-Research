#!/usr/bin/env python3
"""Property-based metric benchmark runner for scored datasets."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np

from data_eval_common import (
    ALL_METRICS,
    DEFAULT_PROFILE_CONFIG,
    METRIC_SPEC_PATH,
    OUTPUT_DIR,
    RUN_MANIFEST_PATH,
    fingerprint_files,
    iter_jsonl_records_resilient,
    load_json,
    save_json,
)
from policy.subsets import (
    _cluster_id,
    _coverage_strategy,
    _passes_gates,
    _passes_stage_b,
    _passes_rare_exemplar_filters,
    _stage_a_gate,
    _stage_b_rank,
    _selection_score,
    _style_bucket_from_text,
)


PROPERTY_BENCHMARK_SCHEMA_VERSION = "property-benchmark-v5"
DEFAULT_SCORED_DIR = OUTPUT_DIR / "scored"
DEFAULT_OUT_DIR = OUTPUT_DIR / "validation" / "property_benchmarks"
_ASSERTION_RE = re.compile(
    r"^(?P<lhs_metric>[a-z_]+)\((?P<lhs_bucket>[a-z_]+)\)\s*>\s*(?P<rhs_metric>[a-z_]+)\((?P<rhs_bucket>[a-z_]+)\)$"
)
_METRIC_ALIASES = {
    "validity_gate": "structural_validity_gate",
    "validity": "structural_validity_score",
    "quality": "reference_quality_score",
    "near_dup": "shingle_near_duplicate_risk_score",
}


@dataclass
class RunningStat:
    count: int = 0
    total: float = 0.0
    min: float | None = None
    max: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def as_dict(self) -> Dict[str, Any]:
        mean = self.total / self.count if self.count else None
        return {
            "count": self.count,
            "mean": round(float(mean), 6) if mean is not None else None,
            "min": round(float(self.min), 6) if self.min is not None else None,
            "max": round(float(self.max), 6) if self.max is not None else None,
        }


@dataclass
class BucketAggregate:
    name: str
    count: int = 0
    metric_stats: Dict[str, RunningStat] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def add(self, record: Dict[str, Any], rng: random.Random, sample_limit: int) -> None:
        self.count += 1
        self.seen += 1
        for metric_name, value in _record_metric_values(record).items():
            self.metric_stats.setdefault(metric_name, RunningStat()).add(value)
        sample = {
            "chunk_uid": record["chunk_uid"],
            "dataset": record["dataset"],
            "source": record["source"],
            "text_preview": record["provenance"].get("text_preview", ""),
            "core_metrics": {
                metric: round(float(record["core_metrics"][metric]["score"]), 6)
                for metric in record["core_metrics"]
            },
            "diagnostic_metrics": {
                metric: round(float(record["diagnostic_metrics"][metric]["score"]), 6)
                for metric in record.get("diagnostic_metrics", {})
            },
        }
        if len(self.samples) < sample_limit:
            self.samples.append(sample)
            return
        j = rng.randint(1, self.seen)
        if j <= sample_limit:
            self.samples[j - 1] = sample

    def as_dict(self, criteria: str) -> Dict[str, Any]:
        return {
            "count": self.count,
            "criteria": criteria,
            "metric_stats": {metric: stat.as_dict() for metric, stat in self.metric_stats.items()},
            "samples": self.samples,
        }


def _iter_scored_records(path: Path) -> Iterator[Dict[str, Any]]:
    yield from iter_jsonl_records_resilient(path)


def _record_metric_values(record: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in record.get("core_metrics") or {}:
        out[metric] = float(record["core_metrics"][metric]["score"])
    for metric in record.get("diagnostic_metrics") or {}:
        out[metric] = float(record["diagnostic_metrics"][metric]["score"])
    return out


def _metric_payload(record: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    if metric_name in (record.get("core_metrics") or {}):
        return record["core_metrics"][metric_name]
    if metric_name in (record.get("diagnostic_metrics") or {}):
        return record["diagnostic_metrics"][metric_name]
    raise KeyError(metric_name)


def _score_array(path: Path, getter) -> np.ndarray:
    values: List[float] = []
    for record in _iter_scored_records(path):
        values.append(float(getter(record)))
    return np.array(values, dtype=np.float32)


def _cluster_size(record: Dict[str, Any]) -> int:
    return int((_metric_payload(record, "tail_cluster_rarity_proxy").get("details") or {}).get("cluster_size") or 0)


def _prefix_bucket_count(record: Dict[str, Any]) -> int:
    details = _metric_payload(record, "shingle_near_duplicate_risk_score").get("details") or {}
    return int(details.get("simhash_prefix_bucket_count") or 0)


def _quantiles(path: Path) -> Dict[str, Dict[str, float]]:
    validity = _score_array(path, lambda r: _metric_payload(r, "structural_validity_score")["score"])
    quality = _score_array(path, lambda r: _metric_payload(r, "reference_quality_score")["score"])
    near_dup = _score_array(path, lambda r: _metric_payload(r, "shingle_near_duplicate_risk_score")["score"])
    coverage = _score_array(path, lambda r: _metric_payload(r, "tail_cluster_rarity_proxy")["score"])
    cluster_size = _score_array(path, _cluster_size)
    prefix_bucket = _score_array(path, _prefix_bucket_count)

    def q(arr: np.ndarray, levels: Iterable[float]) -> Dict[str, float]:
        return {str(level): round(float(np.quantile(arr, level)), 6) for level in levels}

    return {
        "validity": q(validity, (0.05, 0.5, 0.75)),
        "quality": q(quality, (0.5, 0.6, 0.75)),
        "near_duplicate": q(near_dup, (0.1, 0.25, 0.95, 0.99)),
        "coverage": q(coverage, (0.1, 0.5, 0.9)),
        "cluster_size": q(cluster_size, (0.1, 0.5, 0.9)),
        "prefix_bucket_count": q(prefix_bucket, (0.9, 0.95, 0.99)),
    }


def _bucket_criteria_strings(q: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    return {
        "clean_structured": (
            f"validity_gate==1 and validity>={q['validity']['0.75']} and quality>={q['quality']['0.6']} "
            f"and exact_duplicate_indicator==0 and near_duplicate<={q['near_duplicate']['0.1']} "
            "and procedural_penalty<=0.15 and bullet_penalty<=0.15 and glossary_penalty==0"
        ),
        "noisy_corrupted": (
            f"validity<={q['validity']['0.05']} and validity_gate==0 and "
            "(alpha_ratio<0.75 or markup_residue_ratio>0.05 or symbol_ratio>0.10 "
            "or sentence_count<1 or word_count<20)"
        ),
        "exact_duplicate": "exact_duplicate_indicator>0",
        "near_duplicate": (
            f"exact_duplicate_indicator==0 and near_duplicate>={q['near_duplicate']['0.95']} "
            f"and simhash_prefix_bucket_count>={int(round(q['prefix_bucket_count']['0.95']))}"
        ),
        "head_common": (
            f"cluster_size>={int(round(q['cluster_size']['0.9']))} and coverage<={q['coverage']['0.1']}"
        ),
        "tail_rare": (
            f"cluster_size<={int(round(q['cluster_size']['0.1']))} and validity>={q['validity']['0.5']}"
        ),
        "explanatory": (
            f"quality>={q['quality']['0.75']} and validity>={q['validity']['0.5']} "
            "and explanatory_signal>=0.25 and procedural_penalty<=0.15 and glossary_penalty==0"
        ),
        "shallow_procedural": (
            f"validity>={q['validity']['0.5']} and quality<={q['quality']['0.5']} "
            "and (procedural_penalty>=0.3 or bullet_penalty>=0.2 or glossary_penalty==1 or conclusion_penalty==1)"
        ),
        "coherent_prose": (
            f"validity>={q['validity']['0.5']} and quality>={q['quality']['0.75']} "
            "and style_bucket==general_prose and procedural_penalty<=0.15 and bullet_penalty<=0.1"
        ),
        "corrupted_readable": (
            f"validity>={q['validity']['0.5']} and quality<={q['quality']['0.5']} "
            "and (alpha_ratio<0.9 or repeated_token_ratio>=0.25 or markup_residue_ratio>0.0)"
        ),
        "informative_dense": (
            f"validity>={q['validity']['0.5']} and quality>={q['quality']['0.75']} "
            "and (info_density>=0.45 or concept_ratio>=0.18 or explanatory_signal>=0.25)"
        ),
        "template_boilerplate": (
            f"validity>={q['validity']['0.5']} and quality<={q['quality']['0.6']} "
            "and (procedural_penalty>=0.35 or bullet_penalty>=0.25 or glossary_penalty==1 "
            "or conclusion_penalty==1 or list_density_penalty>=0.25)"
        ),
        "non_prose_structured": (
            f"validity>={q['validity']['0.5']} and style_bucket in "
            "{structured_list,instructional,technical_reference}"
        ),
        "intra_chunk_repetitive": (
            f"validity>={q['validity']['0.5']} and exact_duplicate_indicator==0 "
            "and repeated_token_ratio>=0.35"
        ),
    }


def _bucket_membership(record: Dict[str, Any], q: Dict[str, Dict[str, float]]) -> List[str]:
    core = record["core_metrics"]
    validity = _metric_payload(record, "structural_validity_score")
    validity_gate = _metric_payload(record, "structural_validity_gate")
    quality = _metric_payload(record, "reference_quality_score")
    near = _metric_payload(record, "shingle_near_duplicate_risk_score")
    coverage = _metric_payload(record, "tail_cluster_rarity_proxy")
    exact = _metric_payload(record, "exact_duplicate_indicator")
    diagnostic_quality = _metric_payload(record, "explanatory_quality_proxy")
    predictive = _metric_payload(record, "predictive_utility_proxy")

    validity_score = float(validity["score"])
    quality_score = float(quality["score"])
    near_score = float(near["score"])
    coverage_score = float(coverage["score"])
    exact_score = float(exact["score"])
    cluster_size = _cluster_size(record)
    prefix_bucket_count = _prefix_bucket_count(record)

    v_details = validity.get("details") or {}
    q_details = diagnostic_quality.get("details") or {}
    p_details = predictive.get("details") or {}

    alpha = float(v_details.get("alpha_ratio") or 0.0)
    repeated = float(v_details.get("repeated_token_ratio") or 0.0)
    markup = float(v_details.get("markup_residue_ratio") or 0.0)
    valid_flag = bool(validity.get("valid"))
    validity_gate_score = float(validity_gate["score"])
    explanatory = float(q_details.get("explanatory_signal") or 0.0)
    info_density = float(q_details.get("info_density") or 0.0)
    procedural = float(q_details.get("procedural_penalty") or 0.0)
    bullet = float(q_details.get("bullet_penalty") or 0.0)
    glossary = int(q_details.get("glossary_penalty") or 0)
    conclusion = int(q_details.get("conclusion_penalty") or 0)
    concept_ratio = float((p_details.get("concept_ratio") or 0.0))
    list_density_penalty = float((p_details.get("list_density_penalty") or 0.0))
    style_bucket = _style_bucket_from_text((record.get("provenance") or {}).get("text_preview") or "")

    buckets: List[str] = []

    if (
        validity_gate_score > 0.0
        and
        validity_score >= q["validity"]["0.75"]
        and quality_score >= q["quality"]["0.6"]
        and exact_score == 0.0
        and near_score <= q["near_duplicate"]["0.1"]
        and procedural <= 0.15
        and bullet <= 0.15
        and glossary == 0
    ):
        buckets.append("clean_structured")

    if (
        validity_score <= q["validity"]["0.05"]
        and not valid_flag
        and (alpha < 0.75 or markup > 0.05 or v_details.get("symbol_ratio", 0.0) > 0.10 or v_details.get("sentence_count", 0) < 1 or v_details.get("word_count", 0) < 20)
    ):
        buckets.append("noisy_corrupted")

    if exact_score > 0.0:
        buckets.append("exact_duplicate")

    if (
        exact_score == 0.0
        and near_score >= q["near_duplicate"]["0.95"]
        and prefix_bucket_count >= int(round(q["prefix_bucket_count"]["0.95"]))
    ):
        buckets.append("near_duplicate")

    if cluster_size >= int(round(q["cluster_size"]["0.9"])) and coverage_score <= q["coverage"]["0.1"]:
        buckets.append("head_common")

    if cluster_size <= int(round(q["cluster_size"]["0.1"])) and validity_score >= q["validity"]["0.5"]:
        buckets.append("tail_rare")

    if (
        quality_score >= q["quality"]["0.75"]
        and validity_score >= q["validity"]["0.5"]
        and explanatory >= 0.25
        and procedural <= 0.15
        and glossary == 0
    ):
        buckets.append("explanatory")

    if (
        validity_score >= q["validity"]["0.5"]
        and quality_score <= q["quality"]["0.5"]
        and (procedural >= 0.3 or bullet >= 0.2 or glossary == 1 or conclusion == 1)
    ):
        buckets.append("shallow_procedural")

    if (
        validity_score >= q["validity"]["0.5"]
        and quality_score >= q["quality"]["0.75"]
        and style_bucket == "general_prose"
        and procedural <= 0.15
        and bullet <= 0.1
    ):
        buckets.append("coherent_prose")

    if (
        validity_score >= q["validity"]["0.5"]
        and quality_score <= q["quality"]["0.5"]
        and (alpha < 0.9 or repeated >= 0.25 or markup > 0.0)
    ):
        buckets.append("corrupted_readable")

    if (
        validity_score >= q["validity"]["0.5"]
        and quality_score >= q["quality"]["0.75"]
        and (info_density >= 0.45 or concept_ratio >= 0.18 or explanatory >= 0.25)
        and procedural <= 0.2
        and list_density_penalty <= 0.25
    ):
        buckets.append("informative_dense")

    if (
        validity_score >= q["validity"]["0.5"]
        and quality_score <= q["quality"]["0.6"]
        and (procedural >= 0.35 or bullet >= 0.25 or glossary == 1 or conclusion == 1 or list_density_penalty >= 0.25)
    ):
        buckets.append("template_boilerplate")

    if (
        validity_score >= q["validity"]["0.5"]
        and style_bucket in {"structured_list", "instructional", "technical_reference"}
    ):
        buckets.append("non_prose_structured")

    if (
        validity_score >= q["validity"]["0.5"]
        and exact_score == 0.0
        and repeated >= 0.35
    ):
        buckets.append("intra_chunk_repetitive")

    return buckets


def _domain_bucket_from_record(record: Dict[str, Any]) -> str:
    provenance = record.get("provenance") or {}
    metadata = provenance.get("metadata") if isinstance(provenance, dict) else {}
    if isinstance(metadata, dict):
        for key in ("domain", "source_domain", "site", "host"):
            value = str(metadata.get(key) or "").strip().lower()
            if value:
                return value
        for key in ("url", "source_url", "page_url"):
            value = str(metadata.get(key) or "").strip()
            if value:
                host = value.lower().split("://", 1)[-1].split("/", 1)[0]
                host = host.split("?", 1)[0].split("#", 1)[0]
                if host:
                    return host
    source = str(record.get("source") or "").strip().lower()
    return Path(source).name if source else "unknown"


def _length_bucket(word_count: int) -> str:
    if word_count < 40:
        return "short"
    if word_count < 120:
        return "medium"
    if word_count < 260:
        return "long"
    return "very_long"


def _bucket_mean_gap(summary: Dict[str, Dict[str, Any]], *, min_count: int = 50) -> Dict[str, Any]:
    eligible = {
        key: float(value["mean_quality"])
        for key, value in summary.items()
        if int(value.get("count") or 0) >= int(min_count) and value.get("mean_quality") is not None
    }
    if len(eligible) < 2:
        return {
            "min_count": int(min_count),
            "eligible_bucket_count": int(len(eligible)),
            "max_gap": None,
            "lowest_bucket": None,
            "highest_bucket": None,
        }
    lowest_bucket = min(eligible, key=eligible.get)
    highest_bucket = max(eligible, key=eligible.get)
    return {
        "min_count": int(min_count),
        "eligible_bucket_count": int(len(eligible)),
        "max_gap": round(float(eligible[highest_bucket] - eligible[lowest_bucket]), 6),
        "lowest_bucket": lowest_bucket,
        "highest_bucket": highest_bucket,
    }


def _quality_domain_shift_audit(scored_path: Path) -> Dict[str, Any]:
    style_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    style_quality_total: Counter[str] = Counter()
    domain_quality_total: Counter[str] = Counter()
    length_quality_total: Counter[str] = Counter()
    style_validity_total: Counter[str] = Counter()
    domain_validity_total: Counter[str] = Counter()
    length_validity_total: Counter[str] = Counter()
    style_explanatory_total: Counter[str] = Counter()
    domain_explanatory_total: Counter[str] = Counter()
    length_explanatory_total: Counter[str] = Counter()
    clean_valid_count = 0
    clean_valid_low_quality_count = 0

    for record in _iter_scored_records(scored_path):
        preview = ((record.get("provenance") or {}).get("text_preview") or "")
        style_bucket = _style_bucket_from_text(preview)
        domain_bucket = _domain_bucket_from_record(record)
        validity_payload = _metric_payload(record, "structural_validity_score")
        validity_details = validity_payload.get("details") or {}
        word_count = int(validity_details.get("word_count") or record.get("word_count") or 0)
        length_bucket = _length_bucket(word_count)
        quality_score = float(_metric_payload(record, "reference_quality_score")["score"])
        validity_score = float(validity_payload["score"])
        explanatory_score = float(_metric_payload(record, "explanatory_quality_proxy")["score"])

        for bucket, counts, quality_total, validity_total, explanatory_total in (
            (style_bucket, style_counts, style_quality_total, style_validity_total, style_explanatory_total),
            (domain_bucket, domain_counts, domain_quality_total, domain_validity_total, domain_explanatory_total),
            (length_bucket, length_counts, length_quality_total, length_validity_total, length_explanatory_total),
        ):
            counts[bucket] += 1
            quality_total[bucket] += quality_score
            validity_total[bucket] += validity_score
            explanatory_total[bucket] += explanatory_score

        if bool(validity_payload.get("valid")):
            clean_valid_count += 1
            if quality_score < 0.35:
                clean_valid_low_quality_count += 1

    def summarize(
        counts: Counter[str],
        quality_total: Counter[str],
        validity_total: Counter[str],
        explanatory_total: Counter[str],
        *,
        limit: int | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        keys = sorted(counts, key=lambda key: (-counts[key], key))
        if limit is not None:
            keys = keys[: int(limit)]
        out: Dict[str, Dict[str, Any]] = {}
        for bucket in keys:
            count = counts[bucket]
            out[bucket] = {
                "count": int(count),
                "mean_quality": round(float(quality_total[bucket] / count), 6),
                "mean_validity": round(float(validity_total[bucket] / count), 6),
                "mean_explanatory_proxy": round(float(explanatory_total[bucket] / count), 6),
            }
        return out

    by_style = summarize(style_counts, style_quality_total, style_validity_total, style_explanatory_total)
    by_domain_top = summarize(
        domain_counts,
        domain_quality_total,
        domain_validity_total,
        domain_explanatory_total,
        limit=25,
    )
    by_length = summarize(length_counts, length_quality_total, length_validity_total, length_explanatory_total)
    by_domain_all = summarize(domain_counts, domain_quality_total, domain_validity_total, domain_explanatory_total)

    prose_mean = by_style.get("general_prose", {}).get("mean_quality")
    non_prose = {
        k: v for k, v in by_style.items() if k in {"structured_list", "instructional", "technical_reference", "conversational"}
    }
    non_prose_mean = None
    if non_prose:
        total = sum(v["mean_quality"] * v["count"] for v in non_prose.values())
        denom = sum(v["count"] for v in non_prose.values())
        if denom:
            non_prose_mean = round(float(total / denom), 6)

    style_gap = _bucket_mean_gap(by_style, min_count=50)
    domain_gap = _bucket_mean_gap(by_domain_all, min_count=50)
    length_gap = _bucket_mean_gap(by_length, min_count=50)
    calibration_risks = []
    if style_gap.get("max_gap") is not None and float(style_gap["max_gap"]) > 0.20:
        calibration_risks.append("large_style_quality_gap")
    if domain_gap.get("max_gap") is not None and float(domain_gap["max_gap"]) > 0.20:
        calibration_risks.append("large_domain_quality_gap")
    if length_gap.get("max_gap") is not None and float(length_gap["max_gap"]) > 0.20:
        calibration_risks.append("large_length_quality_gap")
    low_quality_clean_rate = clean_valid_low_quality_count / max(clean_valid_count, 1)
    if low_quality_clean_rate > 0.10:
        calibration_risks.append("many_valid_chunks_low_quality")

    return {
        "by_style_bucket": by_style,
        "by_domain_bucket_top": by_domain_top,
        "by_length_bucket": by_length,
        "general_prose_mean_quality": prose_mean,
        "non_prose_mean_quality": non_prose_mean,
        "quality_gap_prose_vs_non_prose": (
            round(float(prose_mean - non_prose_mean), 6)
            if prose_mean is not None and non_prose_mean is not None
            else None
        ),
        "max_style_quality_gap": style_gap,
        "max_domain_quality_gap": domain_gap,
        "max_length_quality_gap": length_gap,
        "valid_but_low_quality": {
            "valid_count": int(clean_valid_count),
            "low_quality_count": int(clean_valid_low_quality_count),
            "low_quality_rate": round(float(low_quality_clean_rate), 6),
            "quality_threshold": 0.35,
        },
        "calibration_risks": calibration_risks,
    }


def _redundancy_behavior_audit(scored_path: Path) -> Dict[str, Any]:
    style_counts: Counter[str] = Counter()
    style_risk_total: Counter[str] = Counter()
    style_indicator_total: Counter[str] = Counter()
    intra_repeat_count = 0
    intra_repeat_risk_total = 0.0
    false_positive_candidates: Counter[str] = Counter()

    for record in _iter_scored_records(scored_path):
        preview = ((record.get("provenance") or {}).get("text_preview") or "")
        style_bucket = _style_bucket_from_text(preview)
        style_counts[style_bucket] += 1
        risk = float(_metric_payload(record, "shingle_near_duplicate_risk_score")["score"])
        indicator = float(_metric_payload(record, "shingle_near_duplicate_indicator")["score"])
        exact = float(_metric_payload(record, "exact_duplicate_indicator")["score"])
        repeated = float((_metric_payload(record, "structural_validity_score").get("details") or {}).get("repeated_token_ratio") or 0.0)
        style_risk_total[style_bucket] += risk
        style_indicator_total[style_bucket] += indicator
        if exact == 0.0 and repeated >= 0.35:
            intra_repeat_count += 1
            intra_repeat_risk_total += risk
        if exact == 0.0 and risk >= 0.25 and style_bucket in {"structured_list", "technical_reference", "instructional"}:
            false_positive_candidates[style_bucket] += 1

    style_summary = {
        bucket: {
            "count": int(count),
            "mean_near_duplicate_risk": round(float(style_risk_total[bucket] / count), 6),
            "mean_near_duplicate_indicator": round(float(style_indicator_total[bucket] / count), 6),
        }
        for bucket, count in style_counts.items()
    }
    return {
        "by_style_bucket": style_summary,
        "intra_chunk_repetition": {
            "count": int(intra_repeat_count),
            "mean_near_duplicate_risk": round(float(intra_repeat_risk_total / intra_repeat_count), 6)
            if intra_repeat_count
            else None,
        },
        "false_positive_candidates_by_style": dict(false_positive_candidates),
    }


def _validity_behavior_audit(scored_path: Path) -> Dict[str, Any]:
    rule_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    decision_scope_counts: Counter[str] = Counter()
    invalid_count = 0
    warning_only_valid_count = 0
    learnable_unit_failures = 0
    repetition_only_failures = 0
    repetition_only_false_negative_candidates = 0
    repetition_only_quality_total = 0.0
    repetition_only_validity_total = 0.0
    style_repetition_warnings = 0

    for record in _iter_scored_records(scored_path):
        validity = _metric_payload(record, "structural_validity_score")
        quality = _metric_payload(record, "reference_quality_score")
        details = validity.get("details") or {}
        rules = list(details.get("violated_rules") or [])
        warnings = list(details.get("warning_rules") or [])
        decision_scope_counts[str(details.get("decision_scope") or "unknown")] += 1
        for rule in rules:
            rule_counts[rule] += 1
        for warning in warnings:
            warning_counts[warning] += 1
        if "empty_or_too_short" in rules or not bool(details.get("learnable_unit_pass", True)):
            learnable_unit_failures += 1
        if any(warning in {"style_repetition_pattern", "soft_repetition_warning"} for warning in warnings):
            style_repetition_warnings += 1
        if not validity.get("valid"):
            invalid_count += 1
            non_repeat_rules = [rule for rule in rules if rule != "hard_broken_repetition"]
            if not non_repeat_rules and "hard_broken_repetition" in rules:
                repetition_only_failures += 1
                repetition_only_quality_total += float(quality["score"])
                repetition_only_validity_total += float(validity["score"])
                if float(validity["score"]) >= 0.94 and float(quality["score"]) >= 0.55:
                    repetition_only_false_negative_candidates += 1
        elif warnings:
            warning_only_valid_count += 1

    return {
        "violated_rule_counts": dict(rule_counts),
        "warning_rule_counts": dict(warning_counts),
        "decision_scope_counts": dict(decision_scope_counts),
        "hard_invalid_count": int(invalid_count),
        "warning_only_valid_count": int(warning_only_valid_count),
        "learnable_unit_failures": int(learnable_unit_failures),
        "hard_warning_boundary": {
            "warnings_do_not_fail_gate": True,
            "warning_only_valid_count": int(warning_only_valid_count),
            "hard_invalid_count": int(invalid_count),
        },
        "repetition_only_failures": {
            "count": int(repetition_only_failures),
            "mean_quality": round(float(repetition_only_quality_total / repetition_only_failures), 6)
            if repetition_only_failures
            else None,
            "mean_validity": round(float(repetition_only_validity_total / repetition_only_failures), 6)
            if repetition_only_failures
            else None,
            "false_negative_candidates": int(repetition_only_false_negative_candidates),
        },
        "style_repetition_warnings": int(style_repetition_warnings),
    }


def _parse_assertion(expr: str) -> Dict[str, str] | None:
    match = _ASSERTION_RE.match(expr.strip())
    if not match:
        return None
    payload = match.groupdict()
    payload["lhs_metric"] = _METRIC_ALIASES.get(payload["lhs_metric"], payload["lhs_metric"])
    payload["rhs_metric"] = _METRIC_ALIASES.get(payload["rhs_metric"], payload["rhs_metric"])
    if payload["lhs_metric"] != payload["rhs_metric"]:
        return None
    return payload


def _unique_assertions(spec: Dict[str, Any]) -> List[str]:
    assertions: List[str] = []
    seen_keys: set[tuple[str, str, str]] = set()
    suite = spec.get("property_benchmark_suite") or {}
    known_buckets = set((suite.get("buckets") or {}).keys())
    for expr in suite.get("assertions") or []:
        parsed = _parse_assertion(expr)
        if parsed is None and any(metric_name in expr for metric_name in ("subset_coverage_retention_score", "small_lm_probe_gain_score")):
            continue
        key = None
        if parsed:
            key = (parsed["lhs_metric"], parsed["lhs_bucket"], parsed["rhs_bucket"])
        if expr not in assertions and (key is None or key not in seen_keys):
            assertions.append(expr)
            if key is not None:
                seen_keys.add(key)
    for meta in (spec.get("metrics") or {}).values():
        for expr in meta.get("acceptance_tests") or []:
            parsed = _parse_assertion(expr)
            if parsed is None and any(metric_name in expr for metric_name in ("subset_coverage_retention_score", "small_lm_probe_gain_score")):
                continue
            if (
                parsed
                and parsed["lhs_metric"] not in {"subset_coverage_retention_score", "small_lm_probe_gain_score"}
                and parsed["lhs_metric"] in ALL_METRICS
                and parsed["lhs_bucket"] in known_buckets
                and parsed["rhs_bucket"] in known_buckets
                and expr not in assertions
                and (parsed["lhs_metric"], parsed["lhs_bucket"], parsed["rhs_bucket"]) not in seen_keys
            ):
                assertions.append(expr)
                seen_keys.add((parsed["lhs_metric"], parsed["lhs_bucket"], parsed["rhs_bucket"]))
    return assertions


def _subset_assertions(dataset_name: str) -> List[Dict[str, Any]]:
    if not RUN_MANIFEST_PATH.exists():
        return []
    run_manifest = load_json(RUN_MANIFEST_PATH)
    profiles = run_manifest.get("profiles") or {}
    assertions: List[Dict[str, Any]] = []
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        meta = ((profile.get("datasets") or {}).get(dataset_name) or {})
        if not meta:
            continue
        coverage_score = float(meta.get("subset_coverage_retention_score") or 0.0)
        stage_c = meta.get("stage_c_core_validation") or {}
        utility_details = meta.get("utility_probe_details") or {}
        aggregate = utility_details.get("aggregate") or {}
        baseline_minima = aggregate.get("baseline_minima") or {}
        protocol = utility_details.get("protocol") or {}
        assertions.append(
            {
                "expression": f"subset_coverage_retention_score({profile_name}) includes backbone and support audit",
                "metric": "subset_coverage_retention_score",
                "supported": True,
                "passed": isinstance(meta.get("cluster_backbone_audit"), dict)
                and isinstance((meta.get("cluster_backbone_audit") or {}).get("passed"), bool)
                and isinstance((((meta.get("coverage_details") or {}).get("domain_coverage_support") or {}).get("distribution_similarity")), (int, float))
                and isinstance((((meta.get("coverage_details") or {}).get("style_coverage_support") or {}).get("distribution_similarity")), (int, float)),
                "lhs_mean": round(coverage_score, 6),
                "rhs_mean": None,
                "margin": None,
            }
        )
        assertions.append(
            {
                "expression": f"small_lm_probe_gain_score({profile_name}) uses dual-baseline held-out probe",
                "metric": "small_lm_probe_gain_score",
                "supported": True,
                "passed": bool(
                    isinstance(aggregate, dict)
                    and isinstance(baseline_minima.get("baseline_full_random"), dict)
                    and isinstance(baseline_minima.get("baseline_stageA_random"), dict)
                    and isinstance(stage_c.get("utility_failures_by_baseline"), dict)
                ),
                "lhs_mean": round(float(meta.get("small_lm_probe_gain_score", meta.get("fixed_token_probe_gain_score") or 0.0)), 6),
                "rhs_mean": None,
                "margin": None,
            }
        )
        assertions.append(
            {
                "expression": f"small_lm_probe_gain_score({profile_name}) records explicit protocol metadata",
                "metric": "small_lm_probe_gain_score",
                "supported": True,
                "passed": isinstance(protocol.get("probe_model_name"), str)
                and isinstance(protocol.get("train_token_budget"), int)
                and isinstance(protocol.get("eval_token_budget"), int)
                and isinstance(protocol.get("seed_count"), int),
                "lhs_mean": round(float(meta.get("small_lm_probe_gain_score", meta.get("fixed_token_probe_gain_score") or 0.0)), 6),
                "rhs_mean": None,
                "margin": None,
            }
        )
    return assertions


def _profile_gate_diagnostics(scored_path: Path, profiles_path: Path) -> Dict[str, Any]:
    profiles = (load_json(profiles_path) or {}).get("profiles") or {}
    original_clusters: Counter[int] = Counter()
    for record in _iter_scored_records(scored_path):
        original_clusters[_cluster_id(record)] += 1
    strategies = {
        name: _coverage_strategy(profile, original_clusters)
        for name, profile in profiles.items()
    }
    stats = {
        name: {
            "total": 0,
            "pass_floors": 0,
            "pass_ceilings": 0,
            "pass_all_gates": 0,
            "selected_after_threshold": 0,
            "rare_cluster_exemplars_added": 0,
            "fail_reasons": {},
            "score_after_gates_min": None,
            "score_after_gates_max": None,
        }
        for name in profiles
    }
    fail_reasons = {name: {} for name in profiles}
    selected_clusters = {name: Counter() for name in profiles}
    rare_cluster_candidates = {name: {} for name in profiles}

    for record in _iter_scored_records(scored_path):
        metrics = record["core_metrics"]
        cluster_id = _cluster_id(record)

        for profile_name, profile in profiles.items():
            current = stats[profile_name]
            strategy = strategies[profile_name]
            stage_a = _stage_a_gate(profile)
            stage_b = _stage_b_rank(profile)
            current["total"] += 1
            floor_ok = True
            ceiling_ok = True
            for metric_name, threshold in stage_a["metric_floors"].items():
                if metrics[metric_name]["score"] < float(threshold):
                    fail_reasons[profile_name][f"floor:{metric_name}"] = fail_reasons[profile_name].get(f"floor:{metric_name}", 0) + 1
                    floor_ok = False
            for metric_name, threshold in stage_a["metric_ceilings"].items():
                if metrics[metric_name]["score"] > float(threshold):
                    fail_reasons[profile_name][f"ceiling:{metric_name}"] = fail_reasons[profile_name].get(f"ceiling:{metric_name}", 0) + 1
                    ceiling_ok = False
            if floor_ok:
                current["pass_floors"] += 1
            if ceiling_ok:
                current["pass_ceilings"] += 1
            stage_a_ok = _passes_gates(record, profile)
            if floor_ok and ceiling_ok and stage_a_ok:
                current["pass_all_gates"] += 1
                score = _selection_score(record, profile)
                current["score_after_gates_min"] = score if current["score_after_gates_min"] is None else min(current["score_after_gates_min"], score)
                current["score_after_gates_max"] = score if current["score_after_gates_max"] is None else max(current["score_after_gates_max"], score)
                if _passes_stage_b(record, profile, score):
                    current["selected_after_threshold"] += 1
                    selected_clusters[profile_name][cluster_id] += 1
                    continue
                if float(metrics["shingle_near_duplicate_risk_score"]["score"]) > float(stage_b["near_duplicate_risk_ceiling"]):
                    key = "stage_b:near_duplicate_risk_ceiling"
                    fail_reasons[profile_name][key] = fail_reasons[profile_name].get(key, 0) + 1
                elif score < float(stage_b["selection_threshold"]):
                    key = "stage_b:selection_threshold"
                    fail_reasons[profile_name][key] = fail_reasons[profile_name].get(key, 0) + 1
            if cluster_id in strategy["rare_clusters"] and _passes_rare_exemplar_filters(record, strategy):
                score = _selection_score(record, profile)
                best = rare_cluster_candidates[profile_name].get(cluster_id)
                if best is None or score > best:
                    rare_cluster_candidates[profile_name][cluster_id] = score

    for profile_name, strategy in strategies.items():
        for cluster_id in strategy["rare_clusters"]:
            if selected_clusters[profile_name].get(cluster_id, 0) > 0:
                continue
            if cluster_id not in rare_cluster_candidates[profile_name]:
                continue
            stats[profile_name]["selected_after_threshold"] += 1
            stats[profile_name]["rare_cluster_exemplars_added"] += 1

    for profile_name in stats:
        stats[profile_name]["fail_reasons"] = dict(
            sorted(fail_reasons[profile_name].items(), key=lambda kv: kv[1], reverse=True)
        )
    return stats


def benchmark_scored_dataset(
    scored_path: Path,
    dataset_name: str,
    metric_spec_path: Path = METRIC_SPEC_PATH,
    profiles_path: Path = DEFAULT_PROFILE_CONFIG,
    sample_limit: int = 5,
    min_assertion_bucket_size: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    spec = load_json(metric_spec_path)
    quantiles = _quantiles(scored_path)
    criteria = _bucket_criteria_strings(quantiles)
    rng = random.Random(seed)

    buckets = {
        name: BucketAggregate(name=name)
        for name in (spec.get("property_benchmark_suite") or {}).get("buckets", {})
    }

    near_risk_values: List[float] = []
    near_indicator_values: List[float] = []
    for record in _iter_scored_records(scored_path):
        near_indicator_values.append(float(_metric_payload(record, "shingle_near_duplicate_indicator")["score"]))
        near_risk_values.append(float(_metric_payload(record, "shingle_near_duplicate_risk_score")["score"]))
        for bucket_name in _bucket_membership(record, quantiles):
            if bucket_name in buckets:
                buckets[bucket_name].add(record, rng=rng, sample_limit=sample_limit)

    assertions: List[Dict[str, Any]] = []
    for expr in _unique_assertions(spec):
        parsed = _parse_assertion(expr)
        if parsed is None:
            assertions.append({"expression": expr, "supported": False, "passed": False, "reason": "unparsed"})
            continue
        metric = parsed["lhs_metric"]
        lhs_bucket = buckets.get(parsed["lhs_bucket"])
        rhs_bucket = buckets.get(parsed["rhs_bucket"])
        lhs_stat = (lhs_bucket.metric_stats.get(metric) if lhs_bucket else None)
        rhs_stat = (rhs_bucket.metric_stats.get(metric) if rhs_bucket else None)
        lhs_mean = lhs_stat.total / lhs_stat.count if lhs_stat and lhs_stat.count else None
        rhs_mean = rhs_stat.total / rhs_stat.count if rhs_stat and rhs_stat.count else None
        supported = bool(
            lhs_stat
            and rhs_stat
            and lhs_stat.count >= min_assertion_bucket_size
            and rhs_stat.count >= min_assertion_bucket_size
        )
        passed = bool(supported and lhs_mean is not None and rhs_mean is not None and lhs_mean > rhs_mean)
        assertions.append(
            {
                "expression": expr,
                "metric": metric,
                "lhs_bucket": parsed["lhs_bucket"],
                "rhs_bucket": parsed["rhs_bucket"],
                "lhs_count": lhs_stat.count if lhs_stat else 0,
                "rhs_count": rhs_stat.count if rhs_stat else 0,
                "lhs_mean": round(float(lhs_mean), 6) if lhs_mean is not None else None,
                "rhs_mean": round(float(rhs_mean), 6) if rhs_mean is not None else None,
                "margin": round(float(lhs_mean - rhs_mean), 6) if lhs_mean is not None and rhs_mean is not None else None,
                "supported": supported,
                "passed": passed,
            }
        )

    near_risk_arr = np.array(near_risk_values, dtype=np.float32)
    near_indicator_arr = np.array(near_indicator_values, dtype=np.float32)
    q25 = float(np.quantile(near_risk_arr, 0.25))
    q50 = float(np.quantile(near_risk_arr, 0.5))
    near_risk_distribution = {
        "q25": round(q25, 6),
        "q50": round(q50, 6),
        "q90": round(float(np.quantile(near_risk_arr, 0.9)), 6),
        "saturated": bool(q25 >= 0.95 or q50 >= 0.99),
    }
    near_indicator_distribution = {
        "q25": round(float(np.quantile(near_indicator_arr, 0.25)), 6),
        "q50": round(float(np.quantile(near_indicator_arr, 0.5)), 6),
        "q90": round(float(np.quantile(near_indicator_arr, 0.9)), 6),
        "saturated": bool(
            float(np.quantile(near_indicator_arr, 0.25)) >= 0.95
            or float(np.quantile(near_indicator_arr, 0.5)) >= 0.99
        ),
    }

    subset_assertions = _subset_assertions(dataset_name)
    all_assertions = assertions + subset_assertions
    assertion_passed = sum(1 for x in all_assertions if x.get("supported") and x.get("passed"))
    assertion_supported = sum(1 for x in all_assertions if x.get("supported"))
    assertion_failed = assertion_supported - assertion_passed

    report = {
        "schema_version": PROPERTY_BENCHMARK_SCHEMA_VERSION,
        "dataset": dataset_name,
        "scored_path": str(scored_path),
        "metric_spec_path": str(metric_spec_path),
        "metric_spec_fingerprint": fingerprint_files([metric_spec_path]),
        "profiles_path": str(profiles_path),
        "quantiles": quantiles,
        "near_duplicate_distribution": near_risk_distribution,
        "near_duplicate_indicator_distribution": near_indicator_distribution,
        "buckets": {
            name: bucket.as_dict(criteria=criteria[name])
            for name, bucket in buckets.items()
        },
        "chunk_level_assertions": assertions,
        "subset_level_assertions": subset_assertions,
        "assertions": all_assertions,
        "profile_gate_diagnostics": _profile_gate_diagnostics(scored_path, profiles_path),
        "diagnostic_audits": {
            "validity_behavior": _validity_behavior_audit(scored_path),
            "quality_domain_shift": _quality_domain_shift_audit(scored_path),
            "redundancy_behavior": _redundancy_behavior_audit(scored_path),
        },
        "summary": {
            "supported_assertions": assertion_supported,
            "passed_assertions": assertion_passed,
            "failed_assertions": assertion_failed,
        },
    }
    return report


def write_benchmark_report(report: Dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['dataset']}_property_benchmark_report.json"
    save_json(path, report)
    return path
