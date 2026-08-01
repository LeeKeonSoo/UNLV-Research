#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
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
HEAD_NAMES = ("route_confidence", "substantive_payload", "coherence_completeness", "route_specific_evidence")


@dataclass(frozen=True, slots=True)
class CompleteMathScore:
    record_id: str
    normalized_text_sha256: str
    source_group: str
    token_count: int
    route_confidence: float
    substantive_payload: float
    coherence_completeness: float
    route_specific_evidence: float

    def __post_init__(self) -> None:
        scores = self.scores()
        if not self.record_id or len(self.normalized_text_sha256) != 64 or not self.source_group:
            raise CalibrationContractError("Complete Math rows require stable identity, hash, and source group")
        if self.token_count <= 0 or not all(math.isfinite(score) for score in scores):
            raise CalibrationContractError("Complete Math rows require positive tokens and finite scores")

    def scores(self) -> tuple[float, float, float, float]:
        return self.route_confidence, self.substantive_payload, self.coherence_completeness, self.route_specific_evidence


@dataclass(frozen=True, slots=True)
class CompleteProfile:
    profile_id: str
    quantiles: tuple[float, float, float, float]
    thresholds: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CalibrationContractError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _lower_quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _threshold(rows: tuple[CompleteMathScore, ...], head_index: int, quantile: float) -> float:
    sources = sorted({row.source_group for row in rows})
    return min(_lower_quantile(tuple(row.scores()[head_index] for row in rows if row.source_group == source), quantile) for source in sources)


def build_source_balanced_profiles(
    clean: tuple[CompleteMathScore, ...], quantiles: tuple[float, ...]
) -> tuple[CompleteProfile, ...]:
    if len({row.source_group for row in clean}) < 2:
        raise CalibrationContractError("Four-head calibration requires at least two clean-control source groups")
    profiles = []
    for combination in itertools.product(quantiles, repeat=4):
        frozen = tuple(float(value) for value in combination)
        thresholds = tuple(_threshold(clean, index, frozen[index]) for index in range(4))
        profile_id = "__".join(f"{name}_q{value:g}" for name, value in zip(HEAD_NAMES, frozen, strict=True))
        profiles.append(CompleteProfile(profile_id, frozen, thresholds))
    return tuple(profiles)


def _passes(row: CompleteMathScore, thresholds: tuple[float, float, float, float]) -> bool:
    return all(score >= threshold for score, threshold in zip(row.scores(), thresholds, strict=True))


def _profile_summary(
    profile: CompleteProfile,
    clean: tuple[CompleteMathScore, ...],
    candidate: tuple[CompleteMathScore, ...],
    confidence: float,
) -> dict[str, JsonValue]:
    by_source = {source: tuple(row for row in clean if row.source_group == source) for source in sorted({row.source_group for row in clean})}
    failures = sum(not _passes(row, profile.thresholds) for row in clean)
    source_bounds = []
    leave_out_bounds = []
    for held_source, held_rows in by_source.items():
        misses = sum(not _passes(row, profile.thresholds) for row in held_rows)
        source_bounds.append(wilson_upper_bound(misses, len(held_rows), confidence))
        training = tuple(row for source, rows in by_source.items() if source != held_source for row in rows)
        held_thresholds = tuple(_threshold(training, index, profile.quantiles[index]) for index in range(4))
        held_misses = sum(not _passes(row, held_thresholds) for row in held_rows)
        leave_out_bounds.append(wilson_upper_bound(held_misses, len(held_rows), confidence))
    excluded = tuple(row for row in candidate if not _passes(row, profile.thresholds))
    return {
        "profile_id": profile.profile_id,
        "quantiles": list(profile.quantiles),
        "thresholds": dict(zip(HEAD_NAMES, profile.thresholds, strict=True)),
        "clean_failures": failures,
        "clean_trials": len(clean),
        "pooled_wilson_upper_bound": wilson_upper_bound(failures, len(clean), confidence),
        "max_source_wilson_upper_bound": max(source_bounds),
        "max_leave_one_source_out_wilson_upper_bound": max(leave_out_bounds),
        "excluded_candidate_records": len(excluded),
        "excluded_candidate_tokens": sum(row.token_count for row in excluded),
    }


def leave_one_source_out_diagnostics(
    rows: tuple[CompleteMathScore, ...], profile: CompleteProfile, confidence: float
) -> dict[str, dict[str, JsonValue]]:
    by_source = {
        source: tuple(row for row in rows if row.source_group == source)
        for source in sorted({row.source_group for row in rows})
    }
    diagnostics: dict[str, dict[str, JsonValue]] = {}
    for held_source, held_rows in by_source.items():
        training = tuple(row for source, source_rows in by_source.items() if source != held_source for row in source_rows)
        thresholds = tuple(_threshold(training, index, profile.quantiles[index]) for index in range(4))
        per_head = {
            name: sum(row.scores()[index] < thresholds[index] for row in held_rows)
            for index, name in enumerate(HEAD_NAMES)
        }
        failures = sum(not _passes(row, thresholds) for row in held_rows)
        diagnostics[held_source] = {
            "trials": len(held_rows),
            "failures": failures,
            "wilson_false_reject_upper_bound": wilson_upper_bound(failures, len(held_rows), confidence),
            "recalibrated_thresholds": dict(zip(HEAD_NAMES, thresholds, strict=True)),
            "per_head_failures": per_head,
        }
    return diagnostics


def _select(reports: tuple[dict[str, JsonValue], ...], bound: float) -> str | None:
    feasible = tuple(
        report
        for report in reports
        if max(float(report["pooled_wilson_upper_bound"]), float(report["max_source_wilson_upper_bound"]), float(report["max_leave_one_source_out_wilson_upper_bound"])) <= bound
    )
    if not feasible:
        return None
    selected = max(feasible, key=lambda report: (int(report["excluded_candidate_tokens"]), str(report["profile_id"])))
    return str(selected["profile_id"])


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _load_text(path: Path, default_source: str) -> dict[str, tuple[str, str, int]]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            record_id, text = raw.get("record_id"), raw.get("text")
            tokens = raw.get("token_count", raw.get("token_proxy"))
            if not isinstance(record_id, str) or not isinstance(text, str) or not isinstance(tokens, int):
                raise CalibrationContractError("Text rows require record_id, text, and token count")
            rows[record_id] = (str(raw.get("source_group") or default_source), str(raw.get("normalized_text_sha256") or _normalized_hash(text)), tokens)
    return rows


def _load_score_map(path: Path, fields: tuple[str, str]) -> dict[str, tuple[float, float]]:
    scores = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            record_id = raw.get("record_id")
            if not isinstance(record_id, str):
                raise CalibrationContractError("Score rows require record_id")
            scores[record_id] = (float(raw[fields[0]]), float(raw[fields[1]]))
    return scores


def load_complete_scores(text_path: Path, known_path: Path, structural_path: Path, default_source: str) -> tuple[CompleteMathScore, ...]:
    texts = _load_text(text_path, default_source)
    known = _load_score_map(known_path, ("route_confidence", "route_specific_evidence"))
    structural = _load_score_map(structural_path, ("substantive_payload", "coherence_completeness"))
    record_ids = sorted(set(texts) & set(known) & set(structural))
    return tuple(CompleteMathScore(record_id, texts[record_id][1], texts[record_id][0], texts[record_id][2], known[record_id][0], structural[record_id][0], structural[record_id][1], known[record_id][1]) for record_id in record_ids)


def select_declared_calibration_sources(
    rows: tuple[CompleteMathScore, ...], calibration_sources: frozenset[str], allowed_other_sources: frozenset[str]
) -> tuple[CompleteMathScore, ...]:
    observed = {row.source_group for row in rows}
    if observed != calibration_sources | allowed_other_sources:
        raise CalibrationContractError("Observed clean source groups do not match the declared calibration/input roles")
    return tuple(row for row in rows if row.source_group in calibration_sources)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a complete four-head Math positive-evidence candidate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    loaded_clean = load_complete_scores(Path(config["clean_text"]), Path(config["clean_known_head_scores"]), Path(config["clean_structural_scores"]), "math-clean-control")
    candidate = load_complete_scores(Path(config["candidate_text"]), Path(config["candidate_known_head_scores"]), Path(config["candidate_structural_scores"]), "openwebmath-candidate")
    expected_sources = frozenset(str(value) for value in config["calibration_source_groups"])
    allowed_other_sources = frozenset(str(value) for value in config.get("non_calibration_source_groups_present", ()))
    clean = select_declared_calibration_sources(loaded_clean, expected_sources, allowed_other_sources)
    if {row.normalized_text_sha256 for row in clean} & {row.normalized_text_sha256 for row in candidate}:
        raise CalibrationContractError("Clean and candidate normalized-text hashes overlap")
    profiles = build_source_balanced_profiles(clean, tuple(float(value) for value in config["lower_quantiles"]))
    reports = tuple(_profile_summary(profile, clean, candidate, float(config["confidence_level"])) for profile in profiles)
    normal = _select(reports, float(config["normal_false_reject_wilson_upper_bound"]))
    hard = _select(reports, float(config["hard_false_reject_wilson_upper_bound"]))
    report: dict[str, JsonValue] = {
        "schema_version": "math-complete-bundle-calibration-report-v1",
        "status": "complete_candidate_pending_fixture_and_ablation_gates" if normal or hard else "blocked_no_strictly_feasible_profile",
        "clean_records": len(clean),
        "clean_source_groups": sorted(expected_sources),
        "candidate_records": len(candidate),
        "candidate_tokens": sum(row.token_count for row in candidate),
        "strict_selected_profiles": {"normal": normal, "hard": hard},
        "profiles": list(reports),
        "q0_leave_one_source_out_diagnostics": leave_one_source_out_diagnostics(
            clean, next(profile for profile in profiles if profile.quantiles == (0.0, 0.0, 0.0, 0.0)), float(config["confidence_level"])
        ),
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
