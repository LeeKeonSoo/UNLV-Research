#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

DEFAULT_CORE_BEHAVIOR = OUTPUT_DIR / "validation" / "core_behavior_audit_v2.json"
DEFAULT_REDUNDANCY = OUTPUT_DIR / "validation" / "redundancy_validity_benchmark_report.json"
DEFAULT_SCHEMA_SEPARATION = OUTPUT_DIR / "validation" / "scoring_schema_separation_audit.json"
DEFAULT_SELECTOR_LEAKAGE = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.json"
DEFAULT_RELEASE_GATE = OUTPUT_DIR / "validation" / "paper_claim_release_gate_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "core_claim_defense_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "core_claim_defense_report.md"
AUX_EVIDENCE_PATHS: dict[str, Path] = {
    "redundancy_real_corpus_calibration": OUTPUT_DIR / "validation" / "redundancy_real_corpus_calibration_report.json",
    "redundancy_silver_holdout": OUTPUT_DIR / "validation" / "redundancy_silver_holdout_report.json",
    "redundancy_cluster_dropout": OUTPUT_DIR / "validation" / "redundancy_cluster_dropout_audit.json",
    "redundancy_saturation_ablation": OUTPUT_DIR / "validation" / "redundancy_saturation_ablation_report.json",
    "stage0_detector_heldout": OUTPUT_DIR / "validation" / "stage0_detector_heldout_benchmark_report.json",
    "real_corpus_stage0_coverage": OUTPUT_DIR / "validation" / "real_corpus_stage0_coverage_audit.json",
    "coverage_domain_fixture": OUTPUT_DIR / "validation" / "coverage_domain_fixture_benchmark_report.json",
}


def _as_map(value: JsonValue) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _string_items(value: JsonValue) -> list[str]:
    return [str(item) for item in _as_list(value)]


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


def _all_checks_passed(core: JsonMap, axis: str) -> bool:
    checks = _as_map(core.get("core_checks"))
    rows = _as_list(checks.get(axis))
    return bool(rows) and all(bool(_as_map(row).get("passed")) for row in rows)


def _selection_hard_reject_authority(core: JsonMap) -> bool | None:
    checks = _as_map(core.get("core_checks"))
    rows = _as_list(checks.get("Selection Value Evidence"))
    for row in rows:
        mapped = _as_map(row)
        if mapped.get("name") != "selection_value_evidence_declares_no_hard_reject_authority":
            continue
        evidence = _as_map(mapped.get("evidence"))
        value = evidence.get("hard_reject_authority")
        return bool(value) if isinstance(value, bool) else None
    return None


def build(
    core_behavior_path: Path,
    redundancy_path: Path,
    schema_separation_path: Path,
    selector_leakage_path: Path,
    release_gate_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> JsonMap:
    core = _load(core_behavior_path)
    redundancy = _load(redundancy_path)
    aux = {name: _load(path) for name, path in AUX_EVIDENCE_PATHS.items()}
    schema = _load(schema_separation_path)
    leakage = _load(selector_leakage_path)
    release_gate = _load(release_gate_path)
    missing = [
        name
        for name, path in {
            "core_behavior": core_behavior_path,
            "redundancy_validity": redundancy_path,
            "scoring_schema_separation": schema_separation_path,
            "selector_utility_leakage": selector_leakage_path,
            "paper_claim_release_gate": release_gate_path,
        }.items()
        if not path.exists()
    ]
    redundancy_summary = _as_map(redundancy.get("summary"))
    current_threshold = _as_map(redundancy_summary.get("current_threshold"))
    report = {
        "schema_version": "core-claim-defense-report-v1",
        "status": (
            "core_claim_defense_blocked_missing_inputs"
            if missing
            else "core_claim_defense_scoped_not_release_ready"
        ),
        "sources": {
            "core_behavior": _source(core_behavior_path),
            "redundancy_validity": _source(redundancy_path),
            **{name: _source(path) for name, path in AUX_EVIDENCE_PATHS.items()},
            "scoring_schema_separation": _source(schema_separation_path),
            "selector_utility_leakage": _source(selector_leakage_path),
            "paper_claim_release_gate": _source(release_gate_path),
        },
        "missing_inputs": missing,
        "core_axes": {
            "Validity": {
                "allowed_claim": "structural_usability_gate_behavior",
                "behavior_checks_passed": _all_checks_passed(core, "Validity"),
                "not_supported": ["semantic_correctness", "license_safety", "downstream_learning_utility"],
            },
            "Selection Value Evidence": {
                "allowed_claim": "pre_outcome_selection_value_proxy",
                "legacy_alias": "Quality",
                "hard_reject_authority": _selection_hard_reject_authority(core),
                "behavior_checks_passed": _all_checks_passed(core, "Selection Value Evidence"),
                "not_supported": ["intrinsic_quality_measurement", "human_preference_ground_truth"],
            },
            "Redundancy": {
                "allowed_claim": "high_precision_conservative_duplicate_control",
                "behavior_checks_passed": _all_checks_passed(core, "Redundancy"),
                "current_fixture_precision": current_threshold.get("precision"),
                "current_fixture_recall": current_threshold.get("recall"),
                "current_fixture_f1": current_threshold.get("f1"),
                "known_gaps": _as_list(redundancy.get("known_gaps")),
                "evidence_ledger": {
                    "fixture_status": redundancy.get("status"),
                    "real_corpus_calibration_status": aux["redundancy_real_corpus_calibration"].get("status"),
                    "silver_holdout_status": aux["redundancy_silver_holdout"].get("status"),
                    "cluster_dropout_status": aux["redundancy_cluster_dropout"].get("status"),
                    "cluster_dropout_decision": aux["redundancy_cluster_dropout"].get("decision"),
                    "saturation_ablation_status": aux["redundancy_saturation_ablation"].get("status"),
                    "saturation_decision": aux["redundancy_saturation_ablation"].get("decision"),
                },
                "claim_boundary": (
                    "Current canonical threshold is defensible as a high-precision conservative "
                    "duplicate-control policy. It is not a recall-complete deduplication metric."
                ),
                "not_supported": [
                    "universal_semantic_clone_detection", "recall_complete_deduplication",
                    "relaxed_threshold_promotion_without_useful_dropout_guardrail"],
            },
            "Coverage": {
                "allowed_claim": "observable_source_style_path_content_cluster_retention",
                "behavior_checks_passed": _all_checks_passed(core, "Coverage"),
                "fixture_status": aux["coverage_domain_fixture"].get("status"),
                "real_corpus_status": aux["real_corpus_stage0_coverage"].get("status"),
                "remaining_scope_gaps": [
                    gap
                    for gap in _string_items(core.get("remaining_evidence_gaps"))
                    if "coverage" in gap or "domain" in gap
                ],
                "not_supported": ["true_domain_coverage_without_explicit_metadata"],
            },
            "Stage 0 Risk Boundary": {
                "allowed_claim": "project_defined_hazard_quarantine_behavior",
                "heldout_status": aux["stage0_detector_heldout"].get("status"),
                "benchmark_scope": aux["stage0_detector_heldout"].get("benchmark_scope"),
                "remaining_scope_gaps": _string_items(aux["stage0_detector_heldout"].get("remaining_evidence_gaps")),
                "not_supported": ["production_grade_external_detector_validation"],
            },
            "Utility": {
                "allowed_claim": "stage_c_downstream_validation_only",
                "selector_leakage_status": leakage.get("status"),
                "scoring_schema_separation_status": schema.get("status"),
                "not_supported": ["selector_objective_signal", "stage_b_tuning_signal"],
            },
        },
        "claim_decision": {
            "paper_claim_tier": "curation_stage_research_framework",
            "curation_stage_framework_claim_supported": True,
            "curation_responsibility_evidence_supported": True,
            "production_deployment_claim_supported": False,
            "intrinsic_quality_claim_supported": False,
            "utility_in_selector_supported": False,
            "current_allowed_surface": "curation_stage_research_framework",
            "production_blocker_categories": [
                "core_metric_validity_is_behavioral_not_production_grade", "redundancy_is_high_precision_not_recall_complete",
                "stage0_detectors_are_project_defined_not_external_public_benchmarks", "coverage_lacks_explicit_domain_metadata_for_true_domain_claims"],
        },
        "release_gate_status": release_gate.get("status"),
        "release_gate_blockers": _as_list(release_gate.get("blockers")),
        "production_blockers": _as_list(release_gate.get("production_blockers")),
        "required_next_evidence": [
            "broaden_core_behavior_fixtures_with_repository_disjoint_real_cases", "calibrate_redundancy_thresholds_by_content_type_and_chunk_length",
            "prove_stage0_detectors_on_external_or_larger_labeled_benchmarks",
            "keep_utility_outcomes_out_of_stage_b_selection_and_core_scoring_claims"],
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    claim_decision = _as_map(report.get("claim_decision"))
    core_axes = _as_map(report.get("core_axes"))
    lines = [
        "# Core Claim Defense Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Claim Decision",
        "",
    ]
    for key, value in claim_decision.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Core Axes", ""])
    for axis, raw_payload in core_axes.items():
        payload = _as_map(raw_payload)
        lines.append(f"### {axis}")
        lines.append(f"- Allowed claim: `{payload.get('allowed_claim')}`")
        if "behavior_checks_passed" in payload:
            lines.append(f"- Behavior checks passed: `{payload.get('behavior_checks_passed')}`")
        if axis == "Redundancy":
            lines.append(f"- Current fixture precision: `{payload.get('current_fixture_precision')}`")
            lines.append(f"- Current fixture recall: `{payload.get('current_fixture_recall')}`")
            lines.append(f"- Claim boundary: {payload.get('claim_boundary')}")
            lines.extend([f"- Known gap: `{gap}`" for gap in _string_items(payload.get("known_gaps"))])
            ledger = _as_map(payload.get("evidence_ledger"))
            for key, value in ledger.items():
                lines.append(f"- Evidence `{key}`: `{value}`")
        if axis == "Coverage":
            lines.append(f"- Fixture status: `{payload.get('fixture_status')}`")
            lines.append(f"- Real-corpus status: `{payload.get('real_corpus_status')}`")
            lines.extend([f"- Remaining scope gap: `{gap}`" for gap in _string_items(payload.get("remaining_scope_gaps"))])
        if axis == "Stage 0 Risk Boundary":
            lines.append(f"- Heldout status: `{payload.get('heldout_status')}`")
            lines.append(f"- Benchmark scope: `{payload.get('benchmark_scope')}`")
            lines.extend([f"- Remaining scope gap: `{gap}`" for gap in _string_items(payload.get("remaining_scope_gaps"))])
        lines.extend([f"- Not supported: `{item}`" for item in _string_items(payload.get("not_supported"))])
        lines.append("")
    for title, key in (("Release Gate Blockers", "release_gate_blockers"), ("Production Blockers", "production_blockers"), ("Required Next Evidence", "required_next_evidence")):
        lines.extend(["", f"## {title}", ""])
        lines.extend([f"- `{item}`" for item in _string_items(report.get(key))] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Core claim defense report.")
    parser.add_argument("--core-behavior", type=Path, default=DEFAULT_CORE_BEHAVIOR)
    parser.add_argument("--redundancy", type=Path, default=DEFAULT_REDUNDANCY)
    parser.add_argument("--schema-separation", type=Path, default=DEFAULT_SCHEMA_SEPARATION)
    parser.add_argument("--selector-leakage", type=Path, default=DEFAULT_SELECTOR_LEAKAGE)
    parser.add_argument("--release-gate", type=Path, default=DEFAULT_RELEASE_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.core_behavior,
        args.redundancy,
        args.schema_separation,
        args.selector_leakage,
        args.release_gate,
        args.output,
        args.md_output,
    )
    print({"status": report.get("status"), "release_gate_status": report.get("release_gate_status")})
    return 0 if not _as_list(report.get("missing_inputs")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
