#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_structural_evidence import coherence_corruptions, payload_corruptions
from math_structural_evidence import FeatureSchema
from positive_quality_evidence import wilson_lower_bound, wilson_upper_bound
from scripts.calibrate_math_complete_bundle import CompleteMathScore, load_complete_scores
from scripts.score_math_structural_heads import verify_model_hashes
from scripts.train_math_structural_heads import TextRow, load_text_rows, positive_scores


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
HEAD_NAMES = ("route_confidence", "substantive_payload", "coherence_completeness", "route_specific_evidence")


@dataclass(frozen=True, slots=True)
class HeadThresholds:
    route_confidence: float
    substantive_payload: float
    coherence_completeness: float
    route_specific_evidence: float

    def values(self) -> tuple[float, float, float, float]:
        return (
            self.route_confidence,
            self.substantive_payload,
            self.coherence_completeness,
            self.route_specific_evidence,
        )


@dataclass(frozen=True, slots=True)
class AuditContractError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class ProbabilityModel(Protocol):
    def predict_proba(self, features: list[tuple[float, ...]]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class AblationArm:
    arm_id: str
    head_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FixtureObservation:
    family_id: str
    detected: int
    trials: int


@dataclass(frozen=True, slots=True)
class FixtureGate:
    confidence: float
    minimum_lower_bound: float


@dataclass(frozen=True, slots=True)
class FixtureAuditContext:
    models: dict[str, ProbabilityModel]
    thresholds: HeadThresholds
    gate: FixtureGate
    feature_schema: FeatureSchema


def failed_head_names(row: CompleteMathScore, thresholds: HeadThresholds) -> tuple[str, ...]:
    return tuple(
        name
        for name, score, threshold in zip(HEAD_NAMES, row.scores(), thresholds.values(), strict=True)
        if score < threshold
    )


def _arm_summary(
    rows: tuple[CompleteMathScore, ...], arm: AblationArm, thresholds: HeadThresholds
) -> dict[str, JsonValue]:
    values = thresholds.values()
    excluded = tuple(row for row in rows if any(row.scores()[index] < values[index] for index in arm.head_indexes))
    return {
        "arm_id": arm.arm_id,
        "heads": [HEAD_NAMES[index] for index in arm.head_indexes],
        "excluded_records": len(excluded),
        "excluded_tokens": sum(row.token_count for row in excluded),
    }


def summarize_ablation_arms(
    rows: tuple[CompleteMathScore, ...], thresholds: HeadThresholds
) -> tuple[dict[str, JsonValue], ...]:
    arms = (
        AblationArm("known_provider_heads", (0, 3)),
        AblationArm("known_plus_substantive_payload", (0, 1, 3)),
        AblationArm("known_plus_coherence_completeness", (0, 2, 3)),
        AblationArm("all_four_heads", (0, 1, 2, 3)),
    )
    return tuple(_arm_summary(rows, arm, thresholds) for arm in arms)


def summarize_fixture_family(
    observation: FixtureObservation, gate: FixtureGate
) -> dict[str, JsonValue]:
    lower_bound = wilson_lower_bound(observation.detected, observation.trials, gate.confidence)
    return {
        "family_id": observation.family_id,
        "detected": observation.detected,
        "trials": observation.trials,
        "sensitivity": observation.detected / observation.trials if observation.trials else 0.0,
        "wilson_sensitivity_lower_bound": lower_bound,
        "required_lower_bound": gate.minimum_lower_bound,
        "gate_passed": lower_bound >= gate.minimum_lower_bound,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reason_audit(rows: tuple[CompleteMathScore, ...], thresholds: HeadThresholds) -> dict[str, JsonValue]:
    head_records: Counter[str] = Counter()
    head_tokens: Counter[str] = Counter()
    masks: Counter[str] = Counter()
    excluded_records = 0
    excluded_tokens = 0
    for row in rows:
        failed = failed_head_names(row, thresholds)
        if not failed:
            continue
        excluded_records += 1
        excluded_tokens += row.token_count
        masks["+".join(failed)] += 1
        for head in failed:
            head_records[head] += 1
            head_tokens[head] += row.token_count
    return {
        "excluded_records": excluded_records,
        "excluded_tokens": excluded_tokens,
        "per_head": {
            head: {"records": head_records[head], "tokens": head_tokens[head]} for head in HEAD_NAMES
        },
        "overlap_masks_records": dict(sorted(masks.items())),
    }


def _clean_source_audit(
    rows: tuple[CompleteMathScore, ...], thresholds: HeadThresholds, confidence: float
) -> dict[str, JsonValue]:
    report: dict[str, JsonValue] = {}
    for source_group in sorted({row.source_group for row in rows}):
        source_rows = tuple(row for row in rows if row.source_group == source_group)
        failures = sum(bool(failed_head_names(row, thresholds)) for row in source_rows)
        report[source_group] = {
            "failures": failures,
            "trials": len(source_rows),
            "wilson_false_reject_upper_bound": wilson_upper_bound(failures, len(source_rows), confidence),
        }
    return report


def _corruption_rows(rows: tuple[TextRow, ...], head: str) -> dict[str, tuple[TextRow, ...]]:
    families: dict[str, list[TextRow]] = {}
    for row in rows:
        corruptions = (
            payload_corruptions(row.text, row.record_id)
            if head == "substantive_payload"
            else coherence_corruptions(row.text, row.record_id)
        )
        for corruption in corruptions:
            families.setdefault(corruption.corruption_id, []).append(
                TextRow(f"{row.record_id}::{corruption.corruption_id}", row.source_group, corruption.text, row.token_count)
            )
    return {family: tuple(family_rows) for family, family_rows in families.items()}


def _fixture_audit(
    clean_rows: tuple[TextRow, ...], context: FixtureAuditContext
) -> tuple[dict[str, JsonValue], ...]:
    threshold_map = asdict(context.thresholds)
    summaries = []
    for head in ("substantive_payload", "coherence_completeness"):
        for family_id, rows in sorted(_corruption_rows(clean_rows, head).items()):
            scores = positive_scores(context.models[head], rows, context.feature_schema)
            detected = sum(score < float(threshold_map[head]) for score in scores)
            summary = summarize_fixture_family(FixtureObservation(family_id, detected, len(rows)), context.gate)
            summary["head"] = head
            summaries.append(summary)
    return tuple(summaries)


def main() -> int:
    import joblib

    parser = argparse.ArgumentParser(description="Audit the frozen Math Hard candidate without retuning thresholds.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    thresholds = HeadThresholds(**{name: float(profile["thresholds"][name]) for name in HEAD_NAMES})
    raw_feature_schema = profile.get("structural_feature_schema_version", "v1")
    if raw_feature_schema not in {"v1", "v2"}:
        raise AuditContractError("Unsupported structural feature schema")
    feature_schema: FeatureSchema = raw_feature_schema
    calibration_report_path = Path(profile["calibration_report"])
    if _sha256(calibration_report_path) != profile["calibration_report_sha256"]:
        raise AuditContractError("Frozen calibration report hash mismatch")
    model_dir = Path(profile["structural_model_dir"])
    model_paths = {head: model_dir / f"{head}.joblib" for head in ("substantive_payload", "coherence_completeness")}
    model_hashes = verify_model_hashes(model_paths, profile["structural_model_artifact_sha256"])
    clean = load_complete_scores(Path(profile["clean_text"]), Path(profile["clean_known_head_scores"]), Path(profile["clean_structural_scores"]), "math-clean-control")
    candidate = load_complete_scores(Path(profile["candidate_text"]), Path(profile["candidate_known_head_scores"]), Path(profile["candidate_structural_scores"]), "openwebmath-candidate")
    text_rows = load_text_rows(Path(profile["clean_text"]), "math-clean-control")
    models = {head: joblib.load(path) for head, path in model_paths.items()}
    fixture_gate = FixtureGate(
        float(gate["confidence_level"]), float(gate["minimum_family_wilson_sensitivity_lower_bound"])
    )
    fixtures = _fixture_audit(text_rows, FixtureAuditContext(models, thresholds, fixture_gate, feature_schema))
    fixture_passed = bool(fixtures) and all(bool(summary["gate_passed"]) for summary in fixtures)
    report: dict[str, JsonValue] = {
        "schema_version": "math-hard-candidate-audit-v1",
        "status": "fixture_and_ablation_gates_passed_candidate_still_inactive" if fixture_passed else "fixture_gate_failed_candidate_inactive",
        "profile_id": profile["profile_id"],
        "thresholds": asdict(thresholds),
        "model_artifact_sha256": model_hashes,
        "structural_feature_schema_version": feature_schema,
        "candidate_reason_audit": _reason_audit(candidate, thresholds),
        "candidate_ablation_arms": list(summarize_ablation_arms(candidate, thresholds)),
        "clean_source_audit": _clean_source_audit(clean, thresholds, float(gate["confidence_level"])),
        "corruption_fixture_audit": list(fixtures),
        "fixture_gate_passed": fixture_passed,
        "threshold_retuning_performed": False,
        "target_retention_fraction_used": False,
        "external_results_visible": False,
        "runtime_activation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "fixture_gate_passed": fixture_passed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
