#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

ROOT: Final = Path(__file__).resolve().parent
CONFIG_PATH: Final = ROOT / "configs" / "hf_mixed_corpus_retest_protocol_qwen3_4b_v1.json"
REPORT_PATH: Final = OUTPUT_DIR / "validation" / "hf_mixed_corpus_retest_protocol_report.json"
MD_REPORT_PATH: Final = OUTPUT_DIR / "validation" / "hf_mixed_corpus_retest_protocol_report.md"


def _required_field_status(fields: list[JsonValue]) -> JsonMap:
    required = {
        "source_dataset",
        "source_config",
        "source_split",
        "source_tier",
        "license_family",
        "repository_or_origin",
        "token_proxy",
    }
    present = {str(field) for field in fields}
    missing = sorted(required - present)
    return {"passed": not missing, "missing": missing}


def _blockers(config: JsonMap) -> list[str]:
    blockers = []
    mixture = config["candidate_mixture"]
    controls = config["selector_leakage_controls"]
    required_audits = config["required_audits"]
    field_status = _required_field_status(mixture["required_record_fields"])
    if mixture["source_labels_preserved"] is not True:
        blockers.append("source_labels_not_preserved")
    if mixture["source_labels_forbidden_as_selector_features"] is not True:
        blockers.append("source_labels_not_forbidden_as_selector_features")
    if controls["source_tier_label_available_to_stage_b"] is not False:
        blockers.append("source_tier_label_leaks_to_stage_b")
    if controls["hf_dataset_identity_available_to_stage_b"] is not False:
        blockers.append("hf_dataset_identity_leaks_to_stage_b")
    if controls["known_high_quality_label_available_to_stage_b"] is not False:
        blockers.append("known_high_quality_label_leaks_to_stage_b")
    if field_status["passed"] is not True:
        blockers.append("required_record_fields_missing")
    for audit_name, enabled in required_audits.items():
        if enabled is not True:
            blockers.append(f"required_audit_disabled:{audit_name}")
    return blockers


def _render_markdown(report: JsonMap) -> str:
    candidate_mixture = report["candidate_mixture"]
    primary_mix = candidate_mixture["primary_mix"]
    lines = [
        "# HF Mixed-Corpus Retest Protocol",
        "",
        f"Status: `{report['status']}`",
        f"Model: `{report['model']['name']}`",
        f"Primary mix: raw-like `{primary_mix['raw_like_fraction']}`, reference `{primary_mix['known_high_quality_reference_fraction']}`",
        "",
        "## HF Sources",
        "",
        "Raw-like:",
    ]
    for source in report["hf_sources"]["raw_like"]:
        lines.append(f"- `{source['dataset']}`")
    lines.extend(["", "Reference:"])
    for source in report["hf_sources"]["known_high_quality_reference"]:
        lines.append(f"- `{source['dataset']}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Source labels are preserved for audit.",
            "- Source labels and known-reference identity are forbidden as Stage-B selector features.",
            "- Utility remains Stage C only.",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> JsonMap:
    config = load_json(CONFIG_PATH)
    blockers = _blockers(config)
    report = {
        **config,
        "status": "hf_mixed_corpus_retest_protocol_frozen" if not blockers else "hf_mixed_corpus_retest_protocol_blocked",
        "blockers": blockers,
        "source_sha256": {str(CONFIG_PATH): sha256_file(CONFIG_PATH)},
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, indent=2))
    return 0 if report["status"] == "hf_mixed_corpus_retest_protocol_frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
