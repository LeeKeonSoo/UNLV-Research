#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "raw_corpus_matrix_v1.json"
REPORT_PATH = OUTPUT_DIR / "validation" / "raw_corpus_matrix_report.json"
MD_REPORT_PATH = OUTPUT_DIR / "validation" / "raw_corpus_matrix_report.md"
REQUIRED_FIELDS = {
    "record_id", "text_or_code", "source_dataset", "source_config", "source_split", "source_uri",
    "repository_or_origin", "collected_at", "license_family", "content_type", "source_tier", "dedup_key",
    "benchmark_exclusion_status", "token_proxy",
}
REQUIRED_CONDITIONS = {"clean_retain_all", "raw_mixed", "risk_heavy"}


def _as_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _blockers(config: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    fields = {str(field) for field in config.get("required_record_fields", [])}
    conditions = _as_map(config.get("conditions"))
    blinding = _as_map(config.get("stage_b_blinding"))
    benchmark = _as_map(config.get("benchmark_exclusion"))
    if config.get("status") != "frozen_before_materialization":
        blockers.append("matrix_not_frozen_before_materialization")
    if REQUIRED_FIELDS - fields:
        blockers.append("required_provenance_fields_missing")
    if set(conditions) != REQUIRED_CONDITIONS:
        blockers.append("required_corpus_conditions_missing")
    for name, condition in conditions.items():
        mix = _as_map(_as_map(condition).get("source_mix"))
        if abs(sum(float(value) for value in mix.values()) - 1.0) > 1e-9:
            blockers.append(f"invalid_source_mix:{name}")
    if blinding.get("source_tier_available_to_stage_b") is not False:
        blockers.append("source_tier_not_blinded_from_stage_b")
    if blinding.get("known_reference_label_available_to_stage_b") is not False:
        blockers.append("known_reference_label_not_blinded_from_stage_b")
    if benchmark.get("task_hash_or_registry_required") is not True:
        blockers.append("benchmark_registry_not_required")
    if benchmark.get("contamination_or_uncertainty_action") != "quarantine":
        blockers.append("contamination_uncertainty_not_quarantined")
    return blockers


def build() -> dict[str, Any]:
    config = _as_map(load_json(CONFIG_PATH))
    blockers = _blockers(config)
    fields = {str(field) for field in config.get("required_record_fields", [])}
    report = {
        "schema_version": "raw-corpus-matrix-report-v1",
        "status": "raw_corpus_matrix_contract_frozen_materialization_pending" if not blockers else "raw_corpus_matrix_contract_blocked",
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "blockers": blockers,
        "conditions": config.get("conditions"),
        "required_record_fields": {field: field in fields for field in sorted(REQUIRED_FIELDS)},
        "stage_b_blinding": config.get("stage_b_blinding"),
        "benchmark_exclusion": config.get("benchmark_exclusion"),
        "materialization_requirements": config.get("materialization_requirements"),
        "materialization_status": "pending_upstream_data_acquisition_and_frozen_source_manifests",
        "utility_scope": config.get("utility_scope"),
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    conditions = _as_map(report.get("conditions"))
    lines = ["# Raw Corpus Matrix", "", f"Status: `{report['status']}`", "", "## Conditions", ""]
    lines.extend(f"- `{name}`: `{_as_map(value).get('expected_policy_outcome')}`" for name, value in conditions.items())
    lines.extend(["", f"Materialization: `{report['materialization_status']}`", ""])
    return "\n".join(lines)


def main() -> int:
    report = build()
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
