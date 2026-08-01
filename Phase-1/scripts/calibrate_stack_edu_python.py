#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from positive_quality_evidence import (
    CalibrationManifest,
    CalibrationRow,
    ChunkEvidence,
    RouteEvidence,
    RouteThresholds,
    ThresholdProfile,
    calibrate_threshold_profiles,
    evaluate_positive_quality,
    wilson_upper_bound,
)


@dataclass(frozen=True, slots=True)
class CalibrationInputError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class StackEduScore:
    record_id: str
    normalized_text_sha256: str
    source_group: str
    token_count: int
    score: float

    def __post_init__(self) -> None:
        if not self.record_id or len(self.normalized_text_sha256) != 64:
            raise CalibrationInputError("Every row requires a stable ID and normalized-text SHA-256")
        if not self.source_group or self.token_count <= 0 or not math.isfinite(self.score):
            raise CalibrationInputError("Every row requires a source group, positive tokens, and finite score")


def _lower_quantile(values: tuple[float, ...], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise CalibrationInputError("Quantiles require values and a probability within [0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def build_source_balanced_profiles(
    clean: tuple[StackEduScore, ...],
    quantiles: tuple[float, ...],
) -> tuple[ThresholdProfile, ...]:
    by_source: dict[str, list[float]] = {}
    for row in clean:
        by_source.setdefault(row.source_group, []).append(row.score)
    if len(by_source) < 2:
        raise CalibrationInputError("Source-balanced calibration requires at least two clean-control groups")
    profiles = []
    for quantile in quantiles:
        threshold = min(_lower_quantile(tuple(values), quantile) for values in by_source.values())
        profiles.append(
            ThresholdProfile(
                f"source_balanced_q{quantile:g}",
                (RouteThresholds("code", 1.0, 1.0, 1.0, threshold),),
            )
        )
    return tuple(profiles)


def select_strict_profile(
    profiles: tuple[dict[str, Any], ...],
    false_reject_upper_bound: float,
) -> str | None:
    feasible = [
        profile
        for profile in profiles
        if max(
            float(profile["pooled_wilson_upper_bound"]),
            float(profile["max_source_wilson_upper_bound"]),
            float(profile["max_leave_one_source_out_wilson_upper_bound"]),
        )
        <= false_reject_upper_bound
    ]
    if not feasible:
        return None
    selected = max(feasible, key=lambda row: (int(row["excluded_candidate_tokens"]), str(row["profile_id"])))
    return str(selected["profile_id"])


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _load_text(path: Path, default_source: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            record_id, text = raw.get("record_id"), raw.get("text")
            token_count = raw.get("token_count")
            if not isinstance(record_id, str) or not isinstance(text, str) or not isinstance(token_count, int):
                raise CalibrationInputError("Text rows require record_id, text, and exact token_count")
            rows[record_id] = {
                "hash": raw.get("normalized_text_sha256") or _normalized_hash(text),
                "source_group": raw.get("source_group") or default_source,
                "token_count": token_count,
            }
    return rows


def _load_scores(score_path: Path, text_path: Path, default_source: str) -> tuple[StackEduScore, ...]:
    texts = _load_text(text_path, default_source)
    rows = []
    with score_path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            score = raw.get("route_specific_evidence")
            record_id = raw.get("record_id")
            if score is None:
                continue
            if not isinstance(record_id, str) or record_id not in texts or not isinstance(score, int | float):
                raise CalibrationInputError("Every provider score must join one source record")
            text = texts[record_id]
            rows.append(StackEduScore(record_id, text["hash"], text["source_group"], text["token_count"], float(score)))
    return tuple(rows)


def _evidence(row: StackEduScore, provider_hash: str) -> ChunkEvidence:
    return ChunkEvidence(row.record_id, (RouteEvidence("code", 1.0, 1.0, 1.0, row.score),), provider_hash)


def _strict_profile_reports(
    clean: tuple[StackEduScore, ...],
    candidate: tuple[StackEduScore, ...],
    profiles: tuple[ThresholdProfile, ...],
    quantiles: tuple[float, ...],
    provider_hash: str,
    confidence: float,
) -> tuple[dict[str, Any], ...]:
    by_source: dict[str, tuple[StackEduScore, ...]] = {
        source: tuple(row for row in clean if row.source_group == source)
        for source in sorted({row.source_group for row in clean})
    }
    reports = []
    for profile, quantile in zip(profiles, quantiles, strict=True):
        threshold = profile.routes[0].route_specific_evidence
        clean_misses = sum(row.score < threshold for row in clean)
        per_source = {}
        leave_one_out = {}
        for held_source, held_rows in by_source.items():
            misses = sum(row.score < threshold for row in held_rows)
            per_source[held_source] = {
                "failures": misses,
                "trials": len(held_rows),
                "wilson_upper_bound": wilson_upper_bound(misses, len(held_rows), confidence),
            }
            training_groups = [rows for source, rows in by_source.items() if source != held_source]
            held_threshold = min(_lower_quantile(tuple(row.score for row in rows), quantile) for rows in training_groups)
            held_misses = sum(row.score < held_threshold for row in held_rows)
            leave_one_out[held_source] = {
                "threshold": held_threshold,
                "failures": held_misses,
                "trials": len(held_rows),
                "wilson_upper_bound": wilson_upper_bound(held_misses, len(held_rows), confidence),
            }
        decisions = tuple(evaluate_positive_quality(_evidence(row, provider_hash), profile) for row in candidate)
        reports.append(
            {
                "profile_id": profile.profile_id,
                "threshold": threshold,
                "pooled_failures": clean_misses,
                "pooled_trials": len(clean),
                "pooled_wilson_upper_bound": wilson_upper_bound(clean_misses, len(clean), confidence),
                "max_source_wilson_upper_bound": max(row["wilson_upper_bound"] for row in per_source.values()),
                "max_leave_one_source_out_wilson_upper_bound": max(
                    row["wilson_upper_bound"] for row in leave_one_out.values()
                ),
                "per_source": per_source,
                "leave_one_source_out": leave_one_out,
                "excluded_candidate_records": sum(decision.decision != "eligible_keep" for decision in decisions),
                "excluded_candidate_tokens": sum(
                    row.token_count
                    for row, decision in zip(candidate, decisions, strict=True)
                    if decision.decision != "eligible_keep"
                ),
            }
        )
    return tuple(reports)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate source-balanced Stack-Edu Python candidate thresholds.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = config["calibration_inputs"]
    clean = _load_scores(Path(inputs["clean_scores"]), Path(inputs["clean_text"]), "clean-control")
    candidate = _load_scores(Path(inputs["candidate_scores"]), Path(inputs["candidate_text"]), "github-code-candidate")
    if {row.record_id for row in clean} & {row.record_id for row in candidate}:
        raise CalibrationInputError("Clean and candidate record IDs overlap")
    if {row.normalized_text_sha256 for row in clean} & {row.normalized_text_sha256 for row in candidate}:
        raise CalibrationInputError("Clean and candidate normalized-text hashes overlap")
    quantiles = tuple(float(value) for value in config["threshold_quantiles"])
    profiles = build_source_balanced_profiles(clean, quantiles)
    provider_hash = str(config["provider_manifest_sha256"])
    confidence = float(config["confidence_level"])
    calibration_rows = tuple(
        CalibrationRow(_evidence(row, provider_hash), row.token_count, "clean_control", "code", row.source_group)
        for row in clean
    ) + tuple(
        CalibrationRow(_evidence(row, provider_hash), row.token_count, "candidate_pool", None, row.source_group)
        for row in candidate
    )
    manifest = CalibrationManifest(
        frozenset({str(config["provider_training_source_group"])}),
        frozenset(row.source_group for row in clean),
        confidence,
    )
    pooled_normal = calibrate_threshold_profiles(calibration_rows, profiles, manifest, float(config["normal_bound"]))
    pooled_hard = calibrate_threshold_profiles(calibration_rows, profiles, manifest, float(config["hard_bound"]))
    strict = _strict_profile_reports(clean, candidate, profiles, quantiles, provider_hash, confidence)
    normal = select_strict_profile(strict, float(config["normal_bound"]))
    hard = select_strict_profile(strict, float(config["hard_bound"]))
    report = {
        "schema_version": "stack-edu-python-calibration-report-v1",
        "status": "blocked_no_strictly_feasible_profile" if normal is None and hard is None else "development_candidate_only",
        "runtime_activation": False,
        "provider_id": config["provider_id"],
        "provider_revision": config["provider_revision"],
        "clean_records": len(clean),
        "clean_source_groups": sorted({row.source_group for row in clean}),
        "candidate_scored_records": len(candidate),
        "candidate_scored_tokens": sum(row.token_count for row in candidate),
        "profiles": strict,
        "pooled_only_selected_profiles": {
            "normal": pooled_normal.selected_profile_id,
            "hard": pooled_hard.selected_profile_id,
        },
        "strict_selected_profiles": {"normal": normal, "hard": hard},
        "strict_gate": "pooled_and_each_source_and_leave_one_source_out_wilson_upper_bounds_must_pass",
        "target_retention_fraction_used": False,
        "external_results_visible": False,
        "conclusion": "Provider thresholds are not runtime-authorized unless a strict profile is selected.",
        "threshold_profiles": [asdict(profile) for profile in profiles],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "normal": normal, "hard": hard}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
