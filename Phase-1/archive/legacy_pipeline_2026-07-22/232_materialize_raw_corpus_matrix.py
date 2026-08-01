#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parent
MATRIX_CONFIG_PATH = ROOT / "configs" / "raw_corpus_matrix_v1.json"
RAW_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "train" / "release_candidates.jsonl"
REFERENCE_PATH = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "known_high_quality_reference_pool" / "known_high_quality_raw_records.jsonl"
QUARANTINE_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "train" / "quarantined_candidates.jsonl"
OUTPUT_ROOT = OUTPUT_DIR / "raw_corpus_matrix_v1"
REPORT_PATH = OUTPUT_DIR / "validation" / "raw_corpus_matrix_materialization_report.json"
MD_REPORT_PATH = OUTPUT_DIR / "validation" / "raw_corpus_matrix_materialization_report.md"
REQUIRED_PROVENANCE = ("source_name", "source_uri", "collected_at")
CONDITION_TOTALS = {"clean_retain_all": 100, "raw_mixed": 250, "risk_heavy": 200}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_sample(rows: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{salt}:{row['record_id']}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]


def _eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if (row.get("release_eligibility") or {}).get("eligible") is True]


def _payload(row: dict[str, Any], condition: str, tier: str) -> dict[str, Any]:
    partition = row.get("partition") or {}
    provenance = row.get("provenance") or {}
    return {
        "record_id": row["record_id"],
        "text": row["text"],
        "token_proxy": len(str(row["text"]).split()),
        "dedup_key": provenance.get("normalized_sha256"),
        "content_type": partition.get("content_type", "code"),
        "benchmark_exclusion_status": "not_detected_in_stage0",
        "provenance": provenance,
        "rights": row.get("rights"),
        "hazards": row.get("hazards"),
        "partition": partition,
        "matrix_condition": condition,
        "audit_provenance": {
            "source_dataset": "temporal_public_github_change_bundles" if tier == "raw_like" else "known_high_quality_reference_pool",
            "source_config": "code_domain_v2" if tier == "raw_like" else "code_domain_known_high_quality_reference_pool_v1",
            "source_split": partition.get("split"),
            "source_tier": tier,
            "repository_or_origin": partition.get("repository_identity") or provenance.get("source_name"),
            "license_family": (row.get("rights") or {}).get("license"),
        },
    }


def _missing_fields(rows: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        provenance = row.get("provenance") or {}
        for field in REQUIRED_PROVENANCE:
            if not str(provenance.get(field) or "").strip():
                missing.append(f"{row.get('record_id')}:{field}")
        if not str(((row.get("rights") or {}).get("license")) or "").strip():
            missing.append(f"{row.get('record_id')}:license")
    return missing


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _condition_rows(
    raw: list[dict[str, Any]], reference: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, list[tuple[dict[str, Any], str]]]:
    conditions = config["conditions"]
    selections: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for name, total in CONDITION_TOTALS.items():
        mix = conditions[name]["source_mix"]
        raw_count = round(total * float(mix.get("raw_like", 0.0)))
        reference_count = total - raw_count
        selections[name] = [
            *[(row, "raw_like") for row in _stable_sample(raw, raw_count, name)],
            *[(row, "known_high_quality_reference") for row in _stable_sample(reference, reference_count, name)],
        ]
    return selections


def build(raw_path: Path, reference_path: Path, quarantine_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_json(MATRIX_CONFIG_PATH)
    raw = _eligible(_read_jsonl(raw_path))
    reference = _eligible(_read_jsonl(reference_path))
    quarantined = _read_jsonl(quarantine_path)
    conditions = _condition_rows(raw, reference, config)
    blockers: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, pairs in conditions.items():
        selected = [row for row, _ in pairs]
        missing = _missing_fields(selected)
        if missing:
            blockers.extend(missing)
        payloads = [_payload(row, name, tier) for row, tier in pairs]
        payload_path = output_root / name / "release_candidates.jsonl"
        _write_jsonl(payload_path, payloads)
        tier_counts = {tier: sum(1 for _, current in pairs if current == tier) for tier in sorted({tier for _, tier in pairs})}
        summaries[name] = {
            "release_candidates_path": str(payload_path),
            "release_candidates_sha256": sha256_file(payload_path),
            "eligible_record_count": len(payloads),
            "token_proxy": sum(int(row["token_proxy"]) for row in payloads),
            "source_tier_counts": tier_counts,
            "quarantined_record_count": len(quarantined) if name == "risk_heavy" else 0,
        }
    risk_quarantine_path = output_root / "risk_heavy" / "quarantined_candidates.jsonl"
    _write_jsonl(risk_quarantine_path, quarantined)
    report = {
        "schema_version": "raw-corpus-matrix-materialization-report-v1",
        "status": "raw_corpus_matrix_materialized" if not blockers else "raw_corpus_matrix_materialization_blocked",
        "matrix_config_path": str(MATRIX_CONFIG_PATH),
        "matrix_config_sha256": sha256_file(MATRIX_CONFIG_PATH),
        "blockers": blockers,
        "conditions": summaries,
        "stage_b_blinding": {
            "source_tier_available_to_stage_b": False,
            "known_reference_label_available_to_stage_b": False,
            "audit_metadata_location": "audit_provenance only",
        },
        "provenance_audit": {
            "missing_required_field_count": len(blockers),
            "raw_source_records": len(raw),
            "reference_source_records": len(reference),
            "risk_heavy_quarantine_path": str(risk_quarantine_path),
            "risk_heavy_quarantine_sha256": sha256_file(risk_quarantine_path),
        },
        "training_readiness": {
            "stage_a_materialized": False,
            "stage_b_materialized": False,
            "primary_study_ready": False,
            "required_next_action": "run Stage A and frozen Stage B on each materialized condition before preparing training arms",
        },
        "source_paths": {
            "raw": str(raw_path),
            "reference": str(reference_path),
            "quarantine": str(quarantine_path),
        },
        "utility_scope": "Stage C validation only; never selector objective",
    }
    if output_root == OUTPUT_ROOT:
        save_json(REPORT_PATH, report)
        MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Raw Corpus Matrix Materialization", "", f"Status: `{report['status']}`", ""]
    for name, values in report["conditions"].items():
        lines.append(f"- `{name}`: `{values['eligible_record_count']}` eligible records, `{values['token_proxy']}` token proxy")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = build(RAW_PATH, REFERENCE_PATH, QUARANTINE_PATH, OUTPUT_ROOT)
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    print({"status": report["status"], "blockers": len(report["blockers"])})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
