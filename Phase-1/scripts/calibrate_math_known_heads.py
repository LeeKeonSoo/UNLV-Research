#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from positive_quality_evidence import wilson_upper_bound


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class MathKnownScore:
    record_id: str
    normalized_text_sha256: str
    source_group: str
    token_count: int
    route_confidence: float
    route_specific_evidence: float

    def __post_init__(self) -> None:
        values = (self.route_confidence, self.route_specific_evidence)
        if not self.record_id or len(self.normalized_text_sha256) != 64 or not self.source_group:
            raise ValueError("Math calibration rows require stable identity, hash, and source group")
        if self.token_count <= 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("Math calibration rows require positive tokens and finite scores")


@dataclass(frozen=True, slots=True)
class KnownHeadProfile:
    profile_id: str
    route_quantile: float
    usefulness_quantile: float
    route_threshold: float
    usefulness_threshold: float


@dataclass(frozen=True, slots=True)
class StrictProfileReport:
    profile_id: str
    route_threshold: float
    usefulness_threshold: float
    pooled_wilson_upper_bound: float
    max_source_wilson_upper_bound: float
    max_leave_one_source_out_wilson_upper_bound: float
    clean_failures: int
    clean_trials: int
    excluded_candidate_records: int
    excluded_candidate_tokens: int
    per_source: dict[str, dict[str, float | int]]
    leave_one_source_out: dict[str, dict[str, float | int]]


def _lower_quantile(values: tuple[float, ...], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("Quantiles require values and a probability within [0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _source_threshold(clean: tuple[MathKnownScore, ...], field: str, quantile: float) -> float:
    sources = sorted({row.source_group for row in clean})
    if len(sources) < 2:
        raise ValueError("Source-balanced calibration requires at least two source groups")
    return min(_lower_quantile(tuple(float(getattr(row, field)) for row in clean if row.source_group == source), quantile) for source in sources)


def build_source_balanced_profiles(
    clean: tuple[MathKnownScore, ...], quantiles: tuple[float, ...]
) -> tuple[KnownHeadProfile, ...]:
    profiles = []
    for route_quantile in quantiles:
        for usefulness_quantile in quantiles:
            profiles.append(
                KnownHeadProfile(
                    f"route_q{route_quantile:g}__usefulness_q{usefulness_quantile:g}",
                    route_quantile,
                    usefulness_quantile,
                    _source_threshold(clean, "route_confidence", route_quantile),
                    _source_threshold(clean, "route_specific_evidence", usefulness_quantile),
                )
            )
    return tuple(profiles)


def _passes(row: MathKnownScore, route_threshold: float, usefulness_threshold: float) -> bool:
    return row.route_confidence >= route_threshold and row.route_specific_evidence >= usefulness_threshold


def build_strict_reports(
    clean: tuple[MathKnownScore, ...],
    candidate: tuple[MathKnownScore, ...],
    profiles: tuple[KnownHeadProfile, ...],
    confidence: float,
) -> tuple[StrictProfileReport, ...]:
    by_source = {source: tuple(row for row in clean if row.source_group == source) for source in sorted({row.source_group for row in clean})}
    reports = []
    for profile in profiles:
        failures = sum(not _passes(row, profile.route_threshold, profile.usefulness_threshold) for row in clean)
        per_source: dict[str, dict[str, float | int]] = {}
        leave_one_out: dict[str, dict[str, float | int]] = {}
        for held_source, held_rows in by_source.items():
            held_failures = sum(not _passes(row, profile.route_threshold, profile.usefulness_threshold) for row in held_rows)
            per_source[held_source] = {
                "failures": held_failures,
                "trials": len(held_rows),
                "wilson_upper_bound": wilson_upper_bound(held_failures, len(held_rows), confidence),
            }
            training = tuple(row for source, rows in by_source.items() if source != held_source for row in rows)
            route_threshold = _source_threshold(training, "route_confidence", profile.route_quantile)
            usefulness_threshold = _source_threshold(training, "route_specific_evidence", profile.usefulness_quantile)
            held_out_failures = sum(not _passes(row, route_threshold, usefulness_threshold) for row in held_rows)
            leave_one_out[held_source] = {
                "route_threshold": route_threshold,
                "usefulness_threshold": usefulness_threshold,
                "failures": held_out_failures,
                "trials": len(held_rows),
                "wilson_upper_bound": wilson_upper_bound(held_out_failures, len(held_rows), confidence),
            }
        excluded = tuple(row for row in candidate if not _passes(row, profile.route_threshold, profile.usefulness_threshold))
        reports.append(
            StrictProfileReport(
                profile.profile_id,
                profile.route_threshold,
                profile.usefulness_threshold,
                wilson_upper_bound(failures, len(clean), confidence),
                max(float(item["wilson_upper_bound"]) for item in per_source.values()),
                max(float(item["wilson_upper_bound"]) for item in leave_one_out.values()),
                failures,
                len(clean),
                len(excluded),
                sum(row.token_count for row in excluded),
                per_source,
                leave_one_out,
            )
        )
    return tuple(reports)


def select_strict_profile(reports: tuple[StrictProfileReport, ...], bound: float) -> str | None:
    feasible = tuple(
        report
        for report in reports
        if max(
            report.pooled_wilson_upper_bound,
            report.max_source_wilson_upper_bound,
            report.max_leave_one_source_out_wilson_upper_bound,
        )
        <= bound
    )
    if not feasible:
        return None
    return max(feasible, key=lambda report: (report.excluded_candidate_tokens, report.profile_id)).profile_id


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _load_scores(text_path: Path, score_path: Path, default_source: str) -> tuple[MathKnownScore, ...]:
    texts: dict[str, tuple[str, str, int]] = {}
    with text_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            record_id, text = row.get("record_id"), row.get("text")
            token_count = row.get("token_count", row.get("token_proxy"))
            if not isinstance(record_id, str) or not isinstance(text, str) or not isinstance(token_count, int):
                raise ValueError("Text rows require record_id, text, and token count")
            texts[record_id] = (str(row.get("source_group") or default_source), str(row.get("normalized_text_sha256") or _normalized_hash(text)), token_count)
    scores = []
    with score_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or record_id not in texts:
                raise ValueError("Every score must join one text record")
            source_group, digest, token_count = texts[record_id]
            scores.append(MathKnownScore(record_id, digest, source_group, token_count, float(row["route_confidence"]), float(row["route_specific_evidence"])))
    return tuple(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the two available Math positive-evidence heads.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    clean = _load_scores(Path(config["clean_text"]), Path(config["clean_scores"]), "math-clean-control")
    candidate = _load_scores(Path(config["candidate_text"]), Path(config["candidate_scores"]), "openwebmath-candidate")
    if {row.record_id for row in clean} & {row.record_id for row in candidate}:
        raise ValueError("Clean and candidate record IDs overlap")
    if {row.normalized_text_sha256 for row in clean} & {row.normalized_text_sha256 for row in candidate}:
        raise ValueError("Clean and candidate normalized-text hashes overlap")
    quantiles = tuple(float(value) for value in config["lower_quantiles"])
    profiles = build_source_balanced_profiles(clean, quantiles)
    reports = build_strict_reports(clean, candidate, profiles, float(config["confidence_level"]))
    normal = select_strict_profile(reports, float(config["normal_false_reject_wilson_upper_bound"]))
    hard = select_strict_profile(reports, float(config["hard_false_reject_wilson_upper_bound"]))
    report: dict[str, JsonValue] = {
        "schema_version": "math-known-head-calibration-report-v1",
        "status": "partial_known_heads_only_no_runtime_authority",
        "clean_records": len(clean),
        "clean_source_groups": sorted({row.source_group for row in clean}),
        "candidate_records": len(candidate),
        "candidate_tokens": sum(row.token_count for row in candidate),
        "strict_selected_profiles": {"normal": normal, "hard": hard},
        "profiles": [asdict(profile) for profile in reports],
        "missing_heads": config["missing_heads"],
        "target_retention_fraction_used": False,
        "external_results_visible": False,
        "runtime_activation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "normal": normal, "hard": hard}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
