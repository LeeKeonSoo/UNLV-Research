#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from positive_quality_evidence import wilson_upper_bound


@dataclass(frozen=True, slots=True)
class GeneralScore:
    chunk_uid: str
    source_group: str
    token_count: int
    facts_trivia: float
    educational_value: float


def _lower_quantile(values: tuple[float, ...], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("A lower quantile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def source_balanced_thresholds(rows: tuple[GeneralScore, ...], quantile: float) -> tuple[float, float]:
    sources = sorted({row.source_group for row in rows})
    if len(sources) < 2:
        raise ValueError("Strict General calibration requires at least two source groups")
    return (
        min(_lower_quantile(tuple(row.facts_trivia for row in rows if row.source_group == source), quantile) for source in sources),
        min(_lower_quantile(tuple(row.educational_value for row in rows if row.source_group == source), quantile) for source in sources),
    )


def _fails(row: GeneralScore, thresholds: tuple[float, float]) -> bool:
    return row.facts_trivia < thresholds[0] or row.educational_value < thresholds[1]


def _bound(failures: int, trials: int, confidence: float) -> float:
    return wilson_upper_bound(failures, trials, confidence)


def build_profile_report(rows: tuple[GeneralScore, ...], quantile: float, confidence: float) -> dict[str, object]:
    thresholds = source_balanced_thresholds(rows, quantile)
    sources = sorted({row.source_group for row in rows})
    failures = sum(_fails(row, thresholds) for row in rows)
    per_source: dict[str, dict[str, object]] = {}
    leave_out: dict[str, dict[str, object]] = {}
    for source in sources:
        held = tuple(row for row in rows if row.source_group == source)
        held_failures = sum(_fails(row, thresholds) for row in held)
        per_source[source] = {
            "trials": len(held), "failures": held_failures,
            "wilson_upper_bound": _bound(held_failures, len(held), confidence),
        }
        training = tuple(row for row in rows if row.source_group != source)
        held_thresholds = source_balanced_thresholds(training, quantile) if len({row.source_group for row in training}) >= 2 else (
            _lower_quantile(tuple(row.facts_trivia for row in training), quantile),
            _lower_quantile(tuple(row.educational_value for row in training), quantile),
        )
        loso_failures = sum(_fails(row, held_thresholds) for row in held)
        leave_out[source] = {
            "thresholds": {"substantive_payload": held_thresholds[0], "route_specific_evidence": held_thresholds[1]},
            "trials": len(held), "failures": loso_failures,
            "wilson_upper_bound": _bound(loso_failures, len(held), confidence),
        }
    ordered = sorted(rows, key=lambda row: (row.token_count, row.chunk_uid))
    quartiles: dict[str, dict[str, object]] = {}
    for quartile in range(4):
        subset = tuple(row for index, row in enumerate(ordered) if min(3, index * 4 // len(ordered)) == quartile)
        misses = sum(_fails(row, thresholds) for row in subset)
        quartiles[f"q{quartile + 1}"] = {
            "trials": len(subset), "failures": misses,
            "wilson_upper_bound": _bound(misses, len(subset), confidence),
        }
    return {
        "profile_id": f"source_balanced_q{quantile:g}", "quantile": quantile,
        "thresholds": {"substantive_payload": thresholds[0], "route_specific_evidence": thresholds[1]},
        "pooled_failures": failures, "clean_trials": len(rows),
        "pooled_wilson_upper_bound": _bound(failures, len(rows), confidence),
        "per_source": per_source,
        "max_source_wilson_upper_bound": max(float(value["wilson_upper_bound"]) for value in per_source.values()),
        "leave_one_source_out": leave_out,
        "max_leave_one_source_out_wilson_upper_bound": max(float(value["wilson_upper_bound"]) for value in leave_out.values()),
        "length_quartiles": quartiles,
        "max_length_quartile_wilson_upper_bound": max(float(value["wilson_upper_bound"]) for value in quartiles.values()),
    }


def select_strict_profile(reports: tuple[Mapping[str, object], ...], bound: float) -> str | None:
    fields = (
        "pooled_wilson_upper_bound", "max_source_wilson_upper_bound",
        "max_leave_one_source_out_wilson_upper_bound", "max_length_quartile_wilson_upper_bound",
    )
    feasible = [report for report in reports if max(float(report[field]) for field in fields) <= bound]
    return str(max(feasible, key=lambda report: float(report["quantile"]))["profile_id"]) if feasible else None


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _scores(text_rows: list[dict[str, object]], score_rows: list[dict[str, object]]) -> tuple[GeneralScore, ...]:
    metadata = {str(row["chunk_uid"]): row for row in text_rows}
    result = []
    for score in score_rows:
        uid = str(score["chunk_uid"])
        raw = score["raw_scores"]
        meta = metadata[uid]
        result.append(GeneralScore(uid, str(meta["source_group"]), int(score["scored_tokens"]), float(raw["facts_trivia"]), float(raw["educational_value"])))
    return tuple(result)


def _stress_report(clean: tuple[GeneralScore, ...], fixture_rows: list[dict[str, object]], stress: tuple[GeneralScore, ...], thresholds: tuple[float, float], confidence: float) -> dict[str, object]:
    clean_map = {row.chunk_uid: row for row in clean}
    stress_map = {row.chunk_uid: row for row in stress}
    format_counts: dict[str, list[int]] = {}
    semantic_trials = semantic_violations = 0
    for fixture in fixture_rows:
        variant = str(fixture["variant"])
        base = clean_map[str(fixture["base_chunk_uid"])]
        scored = stress_map[str(fixture["chunk_uid"])]
        if variant.startswith("format_") and not _fails(base, thresholds):
            counts = format_counts.setdefault(variant, [0, 0])
            counts[0] += 1
            counts[1] += int(_fails(scored, thresholds))
        elif variant == "semantic_destruction_token_permutation":
            semantic_trials += 1
            semantic_violations += int(scored.facts_trivia >= base.facts_trivia and scored.educational_value >= base.educational_value)
    formats = {
        name: {"trials": values[0], "decision_flips": values[1], "wilson_upper_bound": _bound(values[1], values[0], confidence)}
        for name, values in format_counts.items()
    }
    return {
        "format_invariance": formats,
        "max_format_flip_wilson_upper_bound": max(float(value["wilson_upper_bound"]) for value in formats.values()),
        "semantic_destruction": {
            "trials": semantic_trials, "both_head_outscore_violations": semantic_violations,
            "wilson_upper_bound": _bound(semantic_violations, semantic_trials, confidence),
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source transfer and stress behavior for General Quality v2.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in config["outputs"].items()}
    clean_text = _load_jsonl(paths["clean_controls"])
    clean = _scores(clean_text, _load_jsonl(paths["clean_scores"]))
    fixtures = _load_jsonl(paths["stress_fixtures"])
    stress = _scores(fixtures, _load_jsonl(paths["stress_scores"]))
    confidence = float(config["confidence_level"])
    profiles = tuple(build_profile_report(clean, float(q), confidence) for q in config["lower_quantiles"])
    normal = select_strict_profile(profiles, float(config["normal_false_reject_wilson_upper_bound"]))
    hard = select_strict_profile(profiles, float(config["hard_false_reject_wilson_upper_bound"]))
    diagnostic = profiles[0]
    threshold_values = diagnostic["thresholds"]
    stress_report = _stress_report(clean, fixtures, stress, (float(threshold_values["substantive_payload"]), float(threshold_values["route_specific_evidence"])), confidence)
    status = "blocked_source_transfer" if normal is None and hard is None else "blocked_provider_bias_scope"
    report = {
        "schema_version": "general-prose-evidence-audit-v2", "status": status,
        "controls": len(clean), "source_groups": sorted({row.source_group for row in clean}),
        "strict_selected_profiles": {"normal": normal, "hard": hard}, "profiles": profiles,
        "stress": stress_report,
        "documented_provider_bias_dimensions": {
            "source_topic_length_format_measured": True,
            "social_role_closed": False, "region_closed": False, "language_closed": False,
            "comprehensive_provider_bias_gate": False,
        },
        "artifacts": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items() if name != "audit_report"},
        "target_retention_fraction_used": False, "external_results_visible": False, "runtime_activation": False,
    }
    paths["audit_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "normal": normal, "hard": hard, "controls": len(clean)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
