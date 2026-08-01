#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, save_json, sha256_file

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

VALIDATION_DIR = OUTPUT_DIR / "validation"
SCORED_DIR = OUTPUT_DIR / "scored"
DEFAULT_OUTPUT = VALIDATION_DIR / "paper_reproducibility_manifest.json"
DEFAULT_MD_OUTPUT = VALIDATION_DIR / "paper_reproducibility_manifest.md"

SOURCE_SCRIPTS: tuple[tuple[str, Path], ...] = (
    ("score_core_metrics", Path("03_score_core_metrics.py")),
    ("paper_claim_release_gate", Path("190_run_paper_claim_release_gate.py")),
    ("core_claim_defense", Path("192_build_core_claim_defense_report.py")),
    ("stage_c_training_validation", Path("194_build_stage_c_training_validation_report.py")),
    ("confirmatory_decision_boundary", Path("195_build_confirmatory_decision_boundary_report.py")),
    ("curation_stage_paper_package", Path("196_build_curation_stage_paper_package.py")),
    ("paper_comparison_tables", Path("197_build_paper_comparison_tables.py")),
)
CONFIGS: tuple[tuple[str, Path], ...] = (
    ("lm_curation_operational_framework", Path("configs/lm_curation_operational_framework_v1.json")),
    ("metric_evidence_audit", Path("configs/metric_evidence_audit.json")),
    ("training_arm_freeze", Path("configs/code_domain_training_arm_freeze_v1.json")),
    ("confirmatory_protocol", Path("configs/code_domain_v2_confirmatory_protocol_qwen3_4b.json")),
    ("redundancy_guardrails", Path("configs/temporal_code_redundancy_canonical_guardrails_qwen25_0p5b_v1.json")),
)
ARTIFACTS: tuple[tuple[str, Path], ...] = (
    ("scoring_manifest", SCORED_DIR / "scoring_manifest.json"),
    ("paper_claim_release_gate", VALIDATION_DIR / "paper_claim_release_gate_report.json"),
    ("core_claim_defense", VALIDATION_DIR / "core_claim_defense_report.json"),
    ("stage_c_training_validation", VALIDATION_DIR / "stage_c_training_validation_report.json"),
    ("confirmatory_decision_boundary", VALIDATION_DIR / "confirmatory_decision_boundary_report.json"),
    ("paper_comparison_tables", VALIDATION_DIR / "paper_comparison_tables.json"),
    ("curation_stage_paper_package", VALIDATION_DIR / "curation_stage_paper_package.json"),
)
RECURSIVE_ARTIFACT_NAMES = {"curation_stage_paper_package"}
DOCS: tuple[tuple[str, Path], ...] = (
    ("research_framing", Path("docs/research_framing.md")),
    ("operational_framework", Path("docs/lm_curation_operational_framework.md")),
    ("claim_boundary", Path("docs/paper_claim_boundary_and_release_gate.md")),
    ("method_section", Path("docs/paper_method_core_metric_policy.md")),
    ("limitations_section", Path("docs/paper_limitations_and_threats.md")),
)


def _source(path: Path) -> JsonMap:
    return {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}


def _sources(items: tuple[tuple[str, Path], ...]) -> dict[str, JsonMap]:
    sources = {name: _source(path) for name, path in items}
    for name in RECURSIVE_ARTIFACT_NAMES:
        if name in sources:
            sources[name]["sha256"] = None
            sources[name]["hash_note"] = "Omitted to avoid circular hash with the paper package."
    return sources


def _missing(group: str, sources: dict[str, JsonMap]) -> list[str]:
    return [f"{group}:{name}" for name, meta in sources.items() if meta.get("exists") is not True]


def _commands() -> list[JsonMap]:
    return [
        {"purpose": "score core metrics", "command": "conda run -n research python 03_score_core_metrics.py"},
        {"purpose": "build comparison tables", "command": "conda run -n research python 197_build_paper_comparison_tables.py"},
        {"purpose": "build paper package", "command": "conda run -n research python 196_build_curation_stage_paper_package.py"},
        {
            "purpose": "validate paper package",
            "command": "conda run -n research python validation\\test_curation_stage_paper_package.py",
        },
        {
            "purpose": "validate reproducibility manifest",
            "command": "conda run -n research python validation\\test_paper_reproducibility_manifest.py",
        },
    ]


def build(output_path: Path, md_output_path: Path) -> JsonMap:
    source_scripts = _sources(SOURCE_SCRIPTS)
    configs = _sources(CONFIGS)
    artifacts = _sources(ARTIFACTS)
    docs = _sources(DOCS)
    missing_inputs = (
        _missing("source_script", source_scripts)
        + _missing("config", configs)
        + _missing("artifact", artifacts)
        + _missing("doc", docs)
    )
    remaining = [] if not missing_inputs else ["restore_missing_manifest_inputs_before_submission"]
    manifest = {
        "schema_version": "paper-reproducibility-manifest-v1",
        "status": "paper_reproducibility_manifest_frozen" if not remaining else "paper_reproducibility_manifest_blocked",
        "summary": {
            "source_script_count": len(source_scripts),
            "config_count": len(configs),
            "artifact_count": len(artifacts),
            "doc_count": len(docs),
            "remaining_required_manifest_items": remaining,
        },
        "environment": {
            "os": "Windows",
            "conda_environment": "research",
            "default_cuda_visible_devices": "1",
            "primary_gpu": "NVIDIA GeForce RTX 3070 Ti",
            "secondary_gpu_policy": "NVIDIA GeForce RTX 4060 Ti requires explicit approval or approved fallback.",
            "working_directory": str(Path.cwd()),
        },
        "hardware_notes": [
            "Default local research runs use CUDA_VISIBLE_DEVICES=1 for the RTX 3070 Ti.",
            "The RTX 4060 Ti is approval-gated so interactive workloads can keep that GPU free.",
            "Paper claims are reproducibility-bounded and do not assert production deployment readiness.",
        ],
        "commands": _commands(),
        "source_scripts": source_scripts,
        "configs": configs,
        "artifacts": artifacts,
        "docs": docs,
        "missing_inputs": missing_inputs,
    }
    save_json(output_path, manifest)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(manifest), encoding="utf-8")
    return manifest


def _render_markdown(manifest: JsonMap) -> str:
    env = manifest["environment"]
    summary = manifest["summary"]
    lines = [
        "# Paper Reproducibility Manifest",
        "",
        f"Status: `{manifest['status']}`",
        f"Conda environment: `{env['conda_environment']}`",
        f"Default CUDA_VISIBLE_DEVICES: `{env['default_cuda_visible_devices']}`",
        f"Primary GPU: `{env['primary_gpu']}`",
        f"Remaining required items: `{summary['remaining_required_manifest_items']}`",
        "",
        "## Commands",
        "",
    ]
    for item in manifest["commands"]:
        lines.append(f"- `{item['command']}`")
    lines.extend(["", "## Hardware Notes", ""])
    for note in manifest["hardware_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Missing Inputs", ""])
    missing_inputs = manifest["missing_inputs"]
    lines.extend([f"- `{item}`" for item in missing_inputs] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the paper reproducibility manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output, args.md_output)
    print({"status": manifest["status"], "missing_inputs": manifest["missing_inputs"]})
    return 0 if not manifest["missing_inputs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
