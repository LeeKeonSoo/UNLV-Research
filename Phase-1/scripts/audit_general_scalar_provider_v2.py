#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from positive_quality_evidence import wilson_upper_bound


@dataclass(frozen=True, slots=True)
class ScalarScore:
    chunk_uid: str
    source_group: str
    token_count: int
    score: float


def _lower_quantile(values: tuple[float, ...], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("A lower quantile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _source_balanced_threshold(rows: tuple[ScalarScore, ...], quantile: float) -> float:
    sources = sorted({row.source_group for row in rows})
    if len(sources) < 2:
        raise ValueError("Strict General calibration requires at least two source groups")
    return min(
        _lower_quantile(tuple(row.score for row in rows if row.source_group == source), quantile)
        for source in sources
    )


def _bound(failures: int, trials: int, confidence: float) -> float:
    return wilson_upper_bound(failures, trials, confidence)


def build_scalar_profile_report(
    rows: tuple[ScalarScore, ...],
    quantile: float,
    confidence: float,
) -> dict[str, object]:
    threshold = _source_balanced_threshold(rows, quantile)
    sources = sorted({row.source_group for row in rows})
    failures = sum(row.score < threshold for row in rows)
    per_source: dict[str, dict[str, int | float]] = {}
    leave_out: dict[str, dict[str, int | float]] = {}
    for source in sources:
        held = tuple(row for row in rows if row.source_group == source)
        held_failures = sum(row.score < threshold for row in held)
        per_source[source] = {
            "trials": len(held),
            "failures": held_failures,
            "wilson_upper_bound": _bound(held_failures, len(held), confidence),
        }
        training = tuple(row for row in rows if row.source_group != source)
        training_sources = {row.source_group for row in training}
        held_threshold = (
            _source_balanced_threshold(training, quantile)
            if len(training_sources) >= 2
            else _lower_quantile(tuple(row.score for row in training), quantile)
        )
        loso_failures = sum(row.score < held_threshold for row in held)
        leave_out[source] = {
            "threshold": held_threshold,
            "trials": len(held),
            "failures": loso_failures,
            "wilson_upper_bound": _bound(loso_failures, len(held), confidence),
        }
    ordered = sorted(rows, key=lambda row: (row.token_count, row.chunk_uid))
    quartiles: dict[str, dict[str, int | float]] = {}
    for quartile in range(4):
        subset = tuple(row for index, row in enumerate(ordered) if min(3, index * 4 // len(ordered)) == quartile)
        misses = sum(row.score < threshold for row in subset)
        quartiles[f"q{quartile + 1}"] = {
            "trials": len(subset),
            "failures": misses,
            "wilson_upper_bound": _bound(misses, len(subset), confidence),
        }
    return {
        "profile_id": f"source_balanced_q{quantile:g}",
        "quantile": quantile,
        "threshold": threshold,
        "evidence_head": "route_specific_evidence",
        "complete_quality_bundle": False,
        "pooled_failures": failures,
        "clean_trials": len(rows),
        "pooled_wilson_upper_bound": _bound(failures, len(rows), confidence),
        "per_source": per_source,
        "max_source_wilson_upper_bound": max(float(value["wilson_upper_bound"]) for value in per_source.values()),
        "leave_one_source_out": leave_out,
        "max_leave_one_source_out_wilson_upper_bound": max(
            float(value["wilson_upper_bound"]) for value in leave_out.values()
        ),
        "length_quartiles": quartiles,
        "max_length_quartile_wilson_upper_bound": max(
            float(value["wilson_upper_bound"]) for value in quartiles.values()
        ),
    }


def select_scalar_profile(reports: tuple[Mapping[str, object], ...], bound: float) -> str | None:
    fields = (
        "pooled_wilson_upper_bound",
        "max_source_wilson_upper_bound",
        "max_leave_one_source_out_wilson_upper_bound",
        "max_length_quartile_wilson_upper_bound",
    )
    feasible = [report for report in reports if max(float(report[field]) for field in fields) <= bound]
    return str(max(feasible, key=lambda report: float(report["quantile"]))["profile_id"]) if feasible else None


def decide_scalar_provider_status(
    normal_profile: str | None,
    hard_profile: str | None,
    stress_report: Mapping[str, object],
    stress_wilson_upper_bound: float,
) -> dict[str, str | list[str]]:
    blocking: list[str] = []
    if normal_profile is None and hard_profile is None:
        blocking.append("source_transfer")
    if float(stress_report["max_format_flip_wilson_upper_bound"]) > stress_wilson_upper_bound:
        blocking.append("format_invariance")
    semantic = stress_report["semantic_destruction"]
    if not isinstance(semantic, Mapping):
        raise ValueError("Scalar provider stress report requires semantic_destruction metrics")
    if float(semantic["wilson_upper_bound"]) > stress_wilson_upper_bound:
        blocking.append("semantic_destruction")
    if not blocking:
        status = "component_candidate_pass_incomplete_quality_bundle"
    elif len(blocking) == 1:
        status = f"blocked_{blocking[0]}"
    else:
        status = "blocked_multiple_gates"
    return {"status": status, "blocking_gates": blocking}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _scores(text_rows: list[dict[str, object]], score_rows: list[dict[str, object]]) -> tuple[ScalarScore, ...]:
    metadata = {str(row["chunk_uid"]): row for row in text_rows}
    return tuple(
        ScalarScore(
            str(score["chunk_uid"]),
            str(metadata[str(score["chunk_uid"])]["source_group"]),
            int(score["character_count"]),
            float(score["score"]),
        )
        for score in score_rows
    )


def _stress_report(
    clean: tuple[ScalarScore, ...],
    fixtures: list[dict[str, object]],
    stress: tuple[ScalarScore, ...],
    threshold: float,
    confidence: float,
) -> dict[str, object]:
    clean_map = {row.chunk_uid: row for row in clean}
    stress_map = {row.chunk_uid: row for row in stress}
    format_counts: dict[str, list[int]] = {}
    semantic_trials = semantic_violations = 0
    for fixture in fixtures:
        variant = str(fixture["variant"])
        base = clean_map[str(fixture["base_chunk_uid"])]
        scored = stress_map[str(fixture["chunk_uid"])]
        if variant.startswith("format_") and base.score >= threshold:
            counts = format_counts.setdefault(variant, [0, 0])
            counts[0] += 1
            counts[1] += int(scored.score < threshold)
        elif variant == "semantic_destruction_token_permutation":
            semantic_trials += 1
            semantic_violations += int(scored.score >= base.score)
    formats = {
        name: {"trials": values[0], "decision_flips": values[1], "wilson_upper_bound": _bound(values[1], values[0], confidence)}
        for name, values in format_counts.items()
    }
    return {
        "format_invariance": formats,
        "max_format_flip_wilson_upper_bound": max(float(value["wilson_upper_bound"]) for value in formats.values()),
        "semantic_destruction": {
            "trials": semantic_trials,
            "outscore_violations": semantic_violations,
            "wilson_upper_bound": _bound(semantic_violations, semantic_trials, confidence),
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a scalar General Quality provider candidate.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in config["outputs"].items()}
    clean = _scores(_load_jsonl(paths["clean_controls"]), _load_jsonl(paths["clean_scores"]))
    fixtures = _load_jsonl(paths["stress_fixtures"])
    stress = _scores(fixtures, _load_jsonl(paths["stress_scores"]))
    confidence = float(config["confidence_level"])
    profiles = tuple(build_scalar_profile_report(clean, float(q), confidence) for q in config["lower_quantiles"])
    normal = select_scalar_profile(profiles, float(config["normal_false_reject_wilson_upper_bound"]))
    hard = select_scalar_profile(profiles, float(config["hard_false_reject_wilson_upper_bound"]))
    stress_report = _stress_report(clean, fixtures, stress, float(profiles[0]["threshold"]), confidence)
    decision = decide_scalar_provider_status(
        normal,
        hard,
        stress_report,
        float(config["stress_wilson_upper_bound"]),
    )
    status = str(decision["status"])
    report = {
        "schema_version": "general-scalar-provider-audit-v2",
        "status": status,
        "provider": config["provider"],
        "controls": len(clean),
        "source_groups": sorted({row.source_group for row in clean}),
        "evidence_head": "route_specific_evidence",
        "complete_quality_bundle": False,
        "strict_selected_profiles": {"normal": normal, "hard": hard},
        "blocking_gates": decision["blocking_gates"],
        "profiles": profiles,
        "stress": stress_report,
        "component_runtime_activation": False,
        "route_runtime_activation": False,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "audit_report"
        },
        "target_retention_fraction_used": False,
        "external_results_visible": False,
    }
    paths["audit_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "normal": normal, "hard": hard, "controls": len(clean)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
