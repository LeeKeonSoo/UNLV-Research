#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explicit_structural_coherence import CoherenceGuardVersion, explicit_coherence_corruptions, explicit_coherence_evidence
from positive_quality_evidence import wilson_lower_bound, wilson_upper_bound


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
InputRole = Literal["clean_control", "fresh_control", "candidate_pool"]


@dataclass(frozen=True, slots=True)
class TextRow:
    record_id: str
    source_group: str
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class BaseStructuralScore:
    record_id: str
    substantive_payload: float
    provider_artifact_sha256: JsonValue


@dataclass(frozen=True, slots=True)
class ExplicitScore:
    row: TextRow
    substantive_payload: float
    coherence_completeness: float
    reason_codes: tuple[str, ...]
    provider_artifact_sha256: JsonValue


@dataclass(frozen=True, slots=True)
class ExplicitCoherenceScoringError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class GuardGate:
    confidence: float
    minimum_sensitivity: float
    maximum_false_reject: float


@dataclass(frozen=True, slots=True)
class GuardAuditContext:
    role: InputRole
    gate: GuardGate
    version: CoherenceGuardVersion


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_text(path: Path) -> tuple[TextRow, ...]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            record_id, text = raw.get("record_id"), raw.get("text")
            token_count = raw.get("token_count", raw.get("token_proxy"))
            if not isinstance(record_id, str) or not isinstance(text, str) or not isinstance(token_count, int):
                raise ExplicitCoherenceScoringError("Text rows require record_id, text, and integer token count")
            rows.append(TextRow(record_id, str(raw.get("source_group") or "unknown"), text, token_count))
    return tuple(rows)


def _load_base_scores(path: Path) -> dict[str, BaseStructuralScore]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            record_id = raw.get("record_id")
            if not isinstance(record_id, str):
                raise ExplicitCoherenceScoringError("Base score rows require record_id")
            payload = float(raw["substantive_payload"])
            if not math.isfinite(payload):
                raise ExplicitCoherenceScoringError("Substantive-payload scores must be finite")
            rows[record_id] = BaseStructuralScore(record_id, payload, raw.get("provider_artifact_sha256"))
    return rows


def score_rows(
    text_rows: tuple[TextRow, ...], base_scores: dict[str, BaseStructuralScore], version: CoherenceGuardVersion = "v2"
) -> tuple[ExplicitScore, ...]:
    if {row.record_id for row in text_rows} != set(base_scores):
        raise ExplicitCoherenceScoringError("Text and base structural score record IDs must match exactly")
    scored = []
    for row in text_rows:
        evidence = explicit_coherence_evidence(row.text, version)
        base = base_scores[row.record_id]
        scored.append(
            ExplicitScore(
                row,
                base.substantive_payload,
                1.0 if evidence.outcome == "guard_passed" else 0.0,
                evidence.reason_codes,
                base.provider_artifact_sha256,
            )
        )
    return tuple(scored)


def _fixture_report(
    rows: tuple[TextRow, ...], gate: GuardGate, version: CoherenceGuardVersion
) -> list[JsonValue]:
    families: dict[str, list[bool]] = {}
    for row in rows:
        for corrupted in explicit_coherence_corruptions(row.text):
            families.setdefault(corrupted.corruption_id, []).append(
                explicit_coherence_evidence(corrupted.text, version).outcome == "explicit_defect"
            )
    return [
        {
            "family_id": family,
            "detected": sum(detections),
            "trials": len(detections),
            "wilson_sensitivity_lower_bound": wilson_lower_bound(sum(detections), len(detections), gate.confidence),
            "required_lower_bound": gate.minimum_sensitivity,
            "gate_passed": wilson_lower_bound(sum(detections), len(detections), gate.confidence) >= gate.minimum_sensitivity,
        }
        for family, detections in sorted(families.items())
    ]


def build_guard_report(
    scores: tuple[ExplicitScore, ...], context: GuardAuditContext
) -> dict[str, JsonValue]:
    reason_records = Counter(reason for score in scores for reason in score.reason_codes)
    defects = tuple(score for score in scores if score.coherence_completeness == 0.0)
    report: dict[str, JsonValue] = {
        "records": len(scores),
        "tokens": sum(score.row.token_count for score in scores),
        "explicit_defect_records": len(defects),
        "explicit_defect_tokens": sum(score.row.token_count for score in defects),
        "reason_code_records": dict(sorted(reason_records.items())),
        "role": context.role,
    }
    if context.role == "candidate_pool":
        report["status"] = "candidate_scored_no_runtime_authority"
        return report
    by_source: dict[str, JsonValue] = {}
    for source_group in sorted({score.row.source_group for score in scores}):
        source_scores = tuple(score for score in scores if score.row.source_group == source_group)
        failures = sum(score.coherence_completeness == 0.0 for score in source_scores)
        by_source[source_group] = {
            "failures": failures,
            "trials": len(source_scores),
            "wilson_false_reject_upper_bound": wilson_upper_bound(failures, len(source_scores), context.gate.confidence),
        }
    fixtures = _fixture_report(tuple(score.row for score in scores), context.gate, context.version)
    source_gate = bool(by_source) and all(
        float(source["wilson_false_reject_upper_bound"]) <= context.gate.maximum_false_reject
        for source in by_source.values()
    )
    fixture_gate = bool(fixtures) and all(bool(fixture["gate_passed"]) for fixture in fixtures)
    gates_passed = source_gate and fixture_gate
    match context.role:
        case "clean_control":
            passed_status = "development_gates_passed_pending_fresh_controls"
        case "fresh_control":
            passed_status = "fresh_control_gates_passed"
        case "candidate_pool":
            raise ExplicitCoherenceScoringError("Candidate pools do not enter clean-control gate reporting")
    report.update(
        {
            "clean_source_audit": by_source,
            "corruption_fixture_audit": fixtures,
            "source_false_reject_gate_passed": source_gate,
            "fixture_gate_passed": fixture_gate,
            "status": passed_status if gates_passed else "clean_control_gate_failed",
        }
    )
    return report


def _write_scores(path: Path, scores: tuple[ExplicitScore, ...], contract_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for score in scores:
            target.write(
                json.dumps(
                    {
                        "schema_version": "math-explicit-coherence-score-v1",
                        "record_id": score.row.record_id,
                        "source_group": score.row.source_group,
                        "token_count": score.row.token_count,
                        "substantive_payload": score.substantive_payload,
                        "coherence_completeness": score.coherence_completeness,
                        "coherence_reason_codes": list(score.reason_codes),
                        "provider_artifact_sha256": score.provider_artifact_sha256,
                        "coherence_contract_sha256": contract_hash,
                        "status": "development_candidate_no_runtime_authority",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace learned Math coherence with the frozen explicit guard.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-structural-scores", type=Path, required=True)
    parser.add_argument("--role", choices=("clean_control", "fresh_control", "candidate_pool"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    role: InputRole = args.role
    raw_version = str(contract["schema_version"]).rsplit("-", 1)[-1]
    if raw_version not in {"v1", "v2"}:
        raise ExplicitCoherenceScoringError("Unsupported explicit coherence contract version")
    version: CoherenceGuardVersion = raw_version
    scores = score_rows(_load_text(args.input), _load_base_scores(args.base_structural_scores), version)
    gate = contract["fixture_gate"]
    report = build_guard_report(
        scores,
        GuardAuditContext(
            role,
            GuardGate(
                float(gate["confidence_level"]),
                float(gate["minimum_family_wilson_sensitivity_lower_bound"]),
                float(gate["maximum_each_source_false_reject_wilson_upper_bound"]),
            ),
            version,
        ),
    )
    contract_hash = _sha256(args.contract)
    _write_scores(args.output, scores, contract_hash)
    report.update(
        {
            "schema_version": "math-explicit-coherence-report-v1",
            "contract_sha256": contract_hash,
            "base_structural_scores_sha256": _sha256(args.base_structural_scores),
            "target_retention_fraction_used": False,
            "external_results_visible": False,
            "runtime_activation": False,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": report["records"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
