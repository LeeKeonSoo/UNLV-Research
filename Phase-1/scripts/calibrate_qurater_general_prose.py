#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


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
)
from scripts.score_qurater_development import score_metadata


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class CalibrationInputError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class GeneralProseScore:
    chunk_uid: str
    normalized_text_sha256: str
    route_eligible: bool
    token_count: int
    writing_style: float
    required_expertise: float
    facts_trivia: float
    educational_value: float

    def __post_init__(self) -> None:
        scores = (self.writing_style, self.required_expertise, self.facts_trivia, self.educational_value)
        if not self.chunk_uid or len(self.normalized_text_sha256) != 64:
            raise CalibrationInputError("Every score requires a stable ID and SHA-256 text hash")
        if self.token_count <= 0 or not all(math.isfinite(value) for value in scores):
            raise CalibrationInputError("Every score requires positive tokens and finite logits")


def _lower_quantile(values: tuple[float, ...], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise CalibrationInputError("Quantiles require values and a probability within [0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def build_threshold_profiles(
    clean: tuple[GeneralProseScore, ...],
    quantiles: tuple[float, ...],
) -> tuple[ThresholdProfile, ...]:
    in_scope = tuple(row for row in clean if row.route_eligible)
    if not in_scope:
        raise CalibrationInputError("General-prose calibration requires route-eligible clean controls")
    profiles: list[ThresholdProfile] = []
    for quantile in quantiles:
        thresholds = RouteThresholds(
            "general_prose",
            1.0,
            _lower_quantile(tuple(row.facts_trivia for row in in_scope), quantile),
            _lower_quantile(tuple(row.writing_style for row in in_scope), quantile),
            _lower_quantile(tuple(row.educational_value for row in in_scope), quantile),
        )
        profiles.append(ThresholdProfile(f"clean_q{quantile:g}", (thresholds,)))
    return tuple(profiles)


def ensure_disjoint(
    clean: tuple[GeneralProseScore, ...],
    candidate: tuple[GeneralProseScore, ...],
) -> None:
    clean_ids = {row.chunk_uid for row in clean}
    clean_hashes = {row.normalized_text_sha256 for row in clean}
    if clean_ids & {row.chunk_uid for row in candidate}:
        raise CalibrationInputError("Clean and candidate chunk IDs overlap")
    if clean_hashes & {row.normalized_text_sha256 for row in candidate}:
        raise CalibrationInputError("Clean and candidate normalized-text hashes overlap")


def _text_by_uid(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CalibrationInputError("Text JSONL rows must be objects")
            uid = row.get("chunk_uid")
            text = row.get("text")
            if isinstance(uid, str) and isinstance(text, str):
                records[uid] = text
    return records


def _numeric(mapping: dict[str, JsonValue], field: str) -> float:
    value = mapping.get(field)
    if not isinstance(value, int | float):
        raise CalibrationInputError(f"Missing numeric QuRater field: {field}")
    return float(value)


def _load_scores(score_path: Path, text_path: Path) -> tuple[GeneralProseScore, ...]:
    texts = _text_by_uid(text_path)
    rows: list[GeneralProseScore] = []
    with score_path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise CalibrationInputError("Score JSONL rows must be objects")
            uid = raw.get("chunk_uid")
            score_values = raw.get("raw_scores")
            if not isinstance(uid, str) or not isinstance(score_values, dict) or uid not in texts:
                raise CalibrationInputError("Score row must join one source chunk")
            text = texts[uid]
            metadata = score_metadata(text)
            token_count = raw.get("scored_tokens")
            if not isinstance(token_count, int):
                raise CalibrationInputError("Score row requires scored_tokens")
            rows.append(
                GeneralProseScore(
                    uid,
                    metadata["normalized_text_sha256"],
                    metadata["general_informational_prose"],
                    token_count,
                    _numeric(score_values, "writing_style"),
                    _numeric(score_values, "required_expertise"),
                    _numeric(score_values, "facts_trivia"),
                    _numeric(score_values, "educational_value"),
                )
            )
    return tuple(rows)


def _evidence(row: GeneralProseScore, provider_revision: str) -> ChunkEvidence:
    return ChunkEvidence(
        row.chunk_uid,
        (
            RouteEvidence(
                "general_prose",
                1.0 if row.route_eligible else 0.0,
                row.facts_trivia,
                row.writing_style,
                row.educational_value,
            ),
        ),
        provider_revision,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate QuRater General-prose candidate thresholds.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = config["calibration_inputs"]
    clean = _load_scores(Path(inputs["clean_control_scores"]), Path(inputs["clean_control_text"]))
    candidate = _load_scores(Path(inputs["candidate_scores"]), Path(inputs["candidate_text"]))
    ensure_disjoint(clean, candidate)
    clean_in_scope = tuple(row for row in clean if row.route_eligible)
    quantiles = tuple(float(value) for value in config["candidate_threshold_profiles"]["lower_quantiles"])
    profiles = build_threshold_profiles(clean, quantiles)
    provider_revision = str(config["provider_revision"])
    rows = tuple(
        CalibrationRow(_evidence(row, provider_revision), row.token_count, "clean_control", "general_prose", "fineweb-edu-clean")
        for row in clean_in_scope
    ) + tuple(
        CalibrationRow(_evidence(row, provider_revision), row.token_count, "candidate_pool", None, "fineweb-broad")
        for row in candidate
        if row.route_eligible
    )
    manifest = CalibrationManifest(
        frozenset({str(inputs["provider_training_source_group"])}),
        frozenset({"fineweb-edu-clean"}),
        float(config["candidate_threshold_profiles"]["confidence_level"]),
    )
    normal = calibrate_threshold_profiles(
        rows,
        profiles,
        manifest,
        float(config["candidate_threshold_profiles"]["normal_false_reject_wilson_upper_bound"]),
    )
    hard = calibrate_threshold_profiles(
        rows,
        profiles,
        manifest,
        float(config["candidate_threshold_profiles"]["hard_false_reject_wilson_upper_bound"]),
    )
    report = {
        "schema_version": "qurater-general-prose-calibration-report-v1",
        "status": "development_only_no_runtime_authority",
        "provider_id": config["provider_id"],
        "provider_revision": provider_revision,
        "clean_controls": len(clean),
        "route_eligible_clean_controls": len(clean_in_scope),
        "candidate_records": len(candidate),
        "route_eligible_candidates": sum(row.route_eligible for row in candidate),
        "candidate_tokens": sum(row.token_count for row in candidate),
        "route_eligible_candidate_tokens": sum(row.token_count for row in candidate if row.route_eligible),
        "threshold_profiles": [asdict(profile) for profile in profiles],
        "normal": asdict(normal),
        "hard": asdict(hard),
        "target_retention_fraction_used": False,
        "external_results_visible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "normal": normal.selected_profile_id, "hard": hard.selected_profile_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
