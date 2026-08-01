#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

VALIDATION_DIR = OUTPUT_DIR / "validation"
DEFAULT_GATE = VALIDATION_DIR / "paper_claim_release_gate_report.json"
DEFAULT_CORE = VALIDATION_DIR / "core_claim_defense_report.json"
DEFAULT_STAGE_C = VALIDATION_DIR / "stage_c_training_validation_report.json"
DEFAULT_BOUNDARY = VALIDATION_DIR / "confirmatory_decision_boundary_report.json"
DEFAULT_METHOD = Path("docs") / "paper_method_core_metric_policy.md"
DEFAULT_LIMITATIONS = Path("docs") / "paper_limitations_and_threats.md"
DEFAULT_TABLES = VALIDATION_DIR / "paper_comparison_tables.json"
DEFAULT_REPRO = VALIDATION_DIR / "paper_reproducibility_manifest.json"
DEFAULT_OUTPUT = VALIDATION_DIR / "curation_stage_paper_package.json"
DEFAULT_MD_OUTPUT = VALIDATION_DIR / "curation_stage_paper_package.md"


def _load(path: Path) -> JsonMap:
    payload = load_json(path) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _source(path: Path) -> JsonMap:
    return {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}


def _as_map(value: JsonValue) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _strings(value: JsonValue) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _remaining_before_submission(
    method_ready: bool, tables_ready: bool, limitations_ready: bool, repro_ready: bool
) -> list[str]:
    remaining = []
    if not method_ready:
        remaining.append("write_method_section_for_core_metric_policy_and_stage_a_b_c_boundaries")
    if not tables_ready:
        remaining.append("freeze_tables_for_raw_stageA_random_curated_and_ablation_comparisons")
    if not limitations_ready:
        remaining.append("write_limitations_and_threats_to_validity_with_production_boundary")
    if not repro_ready:
        remaining.append("freeze_reproducibility_manifest_with_commands_configs_artifacts_and_hardware_notes")
    return remaining


def _completed_before_submission(
    method_ready: bool, tables_ready: bool, limitations_ready: bool, repro_ready: bool
) -> list[str]:
    completed = []
    if method_ready:
        completed.append("write_method_section_for_core_metric_policy_and_stage_a_b_c_boundaries")
    if tables_ready:
        completed.append("freeze_tables_for_raw_stageA_random_curated_and_ablation_comparisons")
    if limitations_ready:
        completed.append("write_limitations_and_threats_to_validity_with_production_boundary")
    if repro_ready:
        completed.append("freeze_reproducibility_manifest_with_commands_configs_artifacts_and_hardware_notes")
    return completed


def _artifact_ready(path: Path, status: str, remaining_field: str) -> bool:
    payload = _load(path)
    return payload.get("status") == status and _as_map(payload.get("summary")).get(remaining_field) == []


def build(output_path: Path, md_output_path: Path) -> JsonMap:
    gate = _load(DEFAULT_GATE)
    core = _load(DEFAULT_CORE)
    stage_c = _load(DEFAULT_STAGE_C)
    boundary = _load(DEFAULT_BOUNDARY)
    method_ready = DEFAULT_METHOD.exists()
    limitations_ready = DEFAULT_LIMITATIONS.exists()
    tables_ready = _artifact_ready(DEFAULT_TABLES, "paper_comparison_tables_frozen", "remaining_required_tables")
    repro_ready = _artifact_ready(
        DEFAULT_REPRO, "paper_reproducibility_manifest_frozen", "remaining_required_manifest_items"
    )
    missing = [
        name
        for name, path in {
            "method_section": DEFAULT_METHOD,
            "limitations_section": DEFAULT_LIMITATIONS,
            "paper_comparison_tables": DEFAULT_TABLES,
            "paper_reproducibility_manifest": DEFAULT_REPRO,
            "paper_claim_gate": DEFAULT_GATE,
            "core_claim_defense": DEFAULT_CORE,
            "stage_c_training_validation": DEFAULT_STAGE_C,
            "confirmatory_decision_boundary": DEFAULT_BOUNDARY,
        }.items()
        if not path.exists()
    ]
    gate_supported = gate.get("status") == "paper_curation_stage_claim_gate_passed" and gate.get("supported") is True
    core_decision = _as_map(core.get("claim_decision"))
    stage_c_decision = _as_map(stage_c.get("claim_decision"))
    boundary_decision = _as_map(boundary.get("claim_decision"))
    paper_supported = bool(
        not missing
        and gate_supported
        and core_decision.get("curation_stage_framework_claim_supported") is True
        and stage_c_decision.get("curation_stage_paper_claim_supported") is True
        and boundary.get("final_decision") == "curation_stage_claim_pass"
    )
    package_ready = paper_supported and method_ready and tables_ready and limitations_ready and repro_ready
    package = {
        "schema_version": "curation-stage-paper-package-v1",
        "status": "curation_stage_paper_package_ready" if package_ready else "curation_stage_paper_package_blocked",
        "paper_claim": {
            "tier": gate.get("paper_claim_tier") or "curation_stage_research_framework",
            "supported": paper_supported,
            "statement": (
                "A curation-stage framework for language-model training data: collected candidate "
                "corpora are structurally gated, risk-quarantined, ranked under budget constraints, "
                "checked for redundancy and coverage, and validated through Stage-C training evidence."
            ),
        },
        "forbidden_claims": [
            "intrinsic_data_quality_measurement",
            "production_ready_universal_filter",
            "utility_as_stage_b_selector_objective",
            "legal_or_license_clearance_certification",
        ],
        "production_boundary": {
            "supported": False,
            "status": gate.get("production_status"),
            "blockers": _strings(gate.get("production_blockers")),
        },
        "method_section": {
            "ready": method_ready,
            "path": str(DEFAULT_METHOD),
            "scope": "Core-Metric-Policy and Stage 0/A/B/C curation-stage method",
        },
        "limitations_section": {
            "ready": limitations_ready,
            "path": str(DEFAULT_LIMITATIONS),
            "scope": "Limitations, threats to validity, and production boundary",
        },
        "comparison_tables": {
            "ready": tables_ready,
            "path": str(DEFAULT_TABLES),
            "scope": "Raw, Stage-A-random, curated, reference, and ablation comparison tables",
        },
        "reproducibility_manifest": {
            "ready": repro_ready,
            "path": str(DEFAULT_REPRO),
            "scope": "Commands, configs, artifacts, and hardware/runtime notes",
        },
        "evidence_table": [
            {
                "surface": "Core responsibility evidence",
                "status": core.get("status"),
                "supported_claim": core_decision.get("current_allowed_surface"),
                "production_gap": ", ".join(_strings(core_decision.get("production_blocker_categories"))),
            },
            {
                "surface": "Stage-C training validation",
                "status": stage_c.get("status"),
                "supported_claim": "target-code NLL training-effect evidence under frozen equal-token comparisons",
                "production_gap": ", ".join(_strings(stage_c.get("remaining_evidence_gaps"))),
            },
            {
                "surface": "Confirmatory decision boundary",
                "status": boundary.get("status"),
                "supported_claim": boundary.get("final_decision"),
                "production_gap": ", ".join(_strings(boundary.get("production_blockers"))),
            },
            {
                "surface": "Paper claim gate",
                "status": gate.get("status"),
                "supported_claim": gate.get("claim_boundary"),
                "production_gap": ", ".join(_strings(gate.get("production_blockers"))),
            },
        ],
        "completed_before_submission": _completed_before_submission(method_ready, tables_ready, limitations_ready, repro_ready),
        "remaining_before_submission": _remaining_before_submission(method_ready, tables_ready, limitations_ready, repro_ready),
        "sources": {
            "method_section": _source(DEFAULT_METHOD),
            "limitations_section": _source(DEFAULT_LIMITATIONS),
            "paper_comparison_tables": _source(DEFAULT_TABLES),
            "paper_reproducibility_manifest": _source(DEFAULT_REPRO),
            "paper_claim_gate": _source(DEFAULT_GATE),
            "core_claim_defense": _source(DEFAULT_CORE),
            "stage_c_training_validation": _source(DEFAULT_STAGE_C),
            "confirmatory_decision_boundary": _source(DEFAULT_BOUNDARY),
        },
        "missing_inputs": missing,
    }
    save_json(output_path, package)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(package), encoding="utf-8")
    return package


def _render_markdown(package: JsonMap) -> str:
    paper = _as_map(package.get("paper_claim"))
    production = _as_map(package.get("production_boundary"))
    lines = [
        "# Curation-Stage Paper Package",
        "",
        f"Status: `{package.get('status')}`",
        f"Paper tier: `{paper.get('tier')}`",
        f"Paper claim supported: `{paper.get('supported')}`",
        f"Production supported: `{production.get('supported')}`",
        "",
        "## Method Section",
        "",
        f"Ready: `{_as_map(package.get('method_section')).get('ready')}`",
        f"Path: `{_as_map(package.get('method_section')).get('path')}`",
        "",
        "## Limitations Section",
        "",
        f"Ready: `{_as_map(package.get('limitations_section')).get('ready')}`",
        f"Path: `{_as_map(package.get('limitations_section')).get('path')}`",
        "",
        "## Comparison Tables",
        "",
        f"Ready: `{_as_map(package.get('comparison_tables')).get('ready')}`",
        f"Path: `{_as_map(package.get('comparison_tables')).get('path')}`",
        "",
        "## Reproducibility Manifest",
        "",
        f"Ready: `{_as_map(package.get('reproducibility_manifest')).get('ready')}`",
        f"Path: `{_as_map(package.get('reproducibility_manifest')).get('path')}`",
        "",
        "## Paper Claim",
        "",
        str(paper.get("statement")),
        "",
        "## Evidence Table",
        "",
        "| Surface | Status | Supported claim | Production gap |",
        "| --- | --- | --- | --- |",
    ]
    for row in _as_list(package.get("evidence_table")):
        item = _as_map(row)
        lines.append(
            f"| {item.get('surface')} | `{item.get('status')}` | {item.get('supported_claim')} | {item.get('production_gap') or '-'} |"
        )
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend([f"- `{item}`" for item in _strings(package.get("forbidden_claims"))])
    lines.extend(["", "## Completed Before Submission", ""])
    lines.extend([f"- `{item}`" for item in _strings(package.get("completed_before_submission"))])
    lines.extend(["", "## Remaining Before Submission", ""])
    lines.extend([f"- `{item}`" for item in _strings(package.get("remaining_before_submission"))])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the curation-stage paper claim package.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    package = build(args.output, args.md_output)
    print({"status": package.get("status"), "paper_supported": _as_map(package.get("paper_claim")).get("supported")})
    return 0 if not package.get("missing_inputs") else 2


if __name__ == "__main__":
    raise SystemExit(main())
