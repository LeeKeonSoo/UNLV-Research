#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

DEFAULT_HAZARD = OUTPUT_DIR / "validation" / "stage0_hazard_benchmark_report.json"
DEFAULT_VALIDATION = OUTPUT_DIR / "validation" / "stage0_detector_validation_report.json"
DEFAULT_HELDOUT = OUTPUT_DIR / "validation" / "stage0_detector_heldout_benchmark_report.json"
DEFAULT_REAL_CORPUS = OUTPUT_DIR / "validation" / "real_corpus_stage0_coverage_audit.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage0_risk_boundary_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage0_risk_boundary_report.md"
AXES = (
    "pii_detected",
    "secret_detected",
    "benchmark_contamination",
    "poisoning_suspected",
    "rights_allowed",
)


def _as_map(value: JsonValue) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _source(path: Path) -> JsonMap:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _load(path: Path) -> JsonMap:
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def _axis_metric(report: JsonMap, axis: str, metric: str) -> JsonValue:
    axis_metrics = _as_map(report.get("axis_metrics"))
    values = _as_map(axis_metrics.get(axis))
    return values.get(metric)


def _real_stage0(real_corpus: JsonMap) -> JsonMap:
    stage0 = _as_map(real_corpus.get("stage0"))
    return {
        "release_candidate_count": stage0.get("release_candidate_count"),
        "quarantined_candidate_count": stage0.get("quarantined_candidate_count"),
        "quarantine_reason_counts": stage0.get("quarantine_reason_counts"),
        "hazard_true_counts": stage0.get("hazard_true_counts"),
        "rights_status_counts": stage0.get("rights_status_counts"),
        "missing_required_provenance_counts": stage0.get("missing_required_provenance_counts"),
    }


def _risk_axes(validation: JsonMap, heldout: JsonMap) -> JsonMap:
    axes: JsonMap = {}
    for axis in AXES:
        axes[axis] = {
            "development_fixture_precision": _axis_metric(validation, axis, "precision"),
            "development_fixture_recall": _axis_metric(validation, axis, "recall"),
            "development_false_positive_count": _axis_metric(validation, axis, "false_positive_count"),
            "development_false_negative_count": _axis_metric(validation, axis, "false_negative_count"),
            "heldout_fixture_precision": _axis_metric(heldout, axis, "precision"),
            "heldout_fixture_recall": _axis_metric(heldout, axis, "recall"),
            "heldout_false_positive_count": _axis_metric(heldout, axis, "false_positive_count"),
            "heldout_false_negative_count": _axis_metric(heldout, axis, "false_negative_count"),
            "allowed_claim": "project_defined_quarantine_behavior_check",
        }
    return axes


def build(
    hazard_path: Path,
    validation_path: Path,
    heldout_path: Path,
    real_corpus_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> JsonMap:
    hazard = _load(hazard_path)
    validation = _load(validation_path)
    heldout = _load(heldout_path)
    real_corpus = _load(real_corpus_path)
    missing = [
        name
        for name, path in {
            "stage0_hazard": hazard_path,
            "stage0_detector_validation": validation_path,
            "stage0_detector_heldout": heldout_path,
            "real_corpus_stage0_coverage": real_corpus_path,
        }.items()
        if not path.exists()
    ]
    report = {
        "schema_version": "stage0-risk-boundary-report-v1",
        "status": (
            "stage0_risk_boundary_blocked_missing_inputs"
            if missing
            else "stage0_risk_boundary_scoped_not_production_ready"
        ),
        "sources": {
            "stage0_hazard": _source(hazard_path),
            "stage0_detector_validation": _source(validation_path),
            "stage0_detector_heldout": _source(heldout_path),
            "real_corpus_stage0_coverage": _source(real_corpus_path),
        },
        "missing_inputs": missing,
        "source_status": {
            "stage0_hazard": hazard.get("status"),
            "stage0_detector_validation": validation.get("status"),
            "stage0_detector_heldout": heldout.get("status"),
            "real_corpus_stage0_coverage": real_corpus.get("status"),
        },
        "risk_axes": _risk_axes(validation, heldout),
        "real_corpus_stage0": _real_stage0(real_corpus),
        "claim_decision": {
            "stage0_quarantine_behavior_supported": not missing,
            "production_detector_claim_supported": False,
            "legal_rights_clearance_claim_supported": False,
            "benchmark_contamination_exhaustive_claim_supported": False,
            "poisoning_robustness_claim_supported": False,
            "training_release_safety_claim_supported": False,
        },
        "allowed_claims": [
            "project_defined_stage0_quarantine_behavior_passed_on_development_and_heldout_fixtures",
            "current_real_corpus_stage0_lineage_and_quarantine_counts_are_reported",
            "coverage_metadata_supports_observable_retention_not_true_domain_or_legal_claims",
        ],
        "forbidden_claims": [
            "production_grade_pii_secret_license_or_poisoning_detector",
            "external_public_detector_benchmark_certification",
            "legal_clearance_or_license_compliance_opinion",
            "exhaustive_benchmark_contamination_removal",
            "adversarial_poisoning_robustness",
        ],
        "remaining_evidence_gaps": [
            "external_public_detector_benchmark_missing",
            "license_rights_policy_not_legal_clearance",
            "benchmark_contamination_registry_or_hash_matching_incomplete",
            "poisoning_detection_not_adversarially_validated",
            "real_corpus_hazard_counts_are_lineage_evidence_not_detector_certification",
        ],
        "utility_scope": "Stage C validation only; Stage-0 risk boundary is pre-training quarantine evidence.",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Stage-0 Risk Boundary Report",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Claim Decision",
        "",
    ]
    for key, value in _as_map(report.get("claim_decision")).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Risk Axes", ""])
    for axis, raw_payload in _as_map(report.get("risk_axes")).items():
        payload = _as_map(raw_payload)
        lines.append(
            f"- `{axis}`: dev recall `{payload.get('development_fixture_recall')}`, "
            f"heldout recall `{payload.get('heldout_fixture_recall')}`"
        )
    lines.extend(["", "## Real Corpus Stage-0", ""])
    for key, value in _as_map(report.get("real_corpus_stage0")).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend([f"- `{item}`" for item in _as_list(report.get("forbidden_claims"))])
    lines.extend(["", "## Remaining Evidence Gaps", ""])
    lines.extend([f"- `{item}`" for item in _as_list(report.get("remaining_evidence_gaps"))])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Stage-0 risk boundary report.")
    parser.add_argument("--hazard", type=Path, default=DEFAULT_HAZARD)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--real-corpus", type=Path, default=DEFAULT_REAL_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.hazard, args.validation, args.heldout, args.real_corpus, args.output, args.md_output)
    print({"status": report.get("status"), "missing_inputs": report.get("missing_inputs")})
    return 0 if not _as_list(report.get("missing_inputs")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
