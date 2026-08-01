#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from curation_artifacts import save_json, sha256_file
from reason_code_audit import build_reason_code_impact_audit
from span_level_template_compaction import build_candidate_impact_audit, build_plan, materialize_candidate_plan
from stage_c_selection import select_chunks


JsonMap = dict[str, Any]
POLICY_FILES = (
    "configs/curation_contract.json",
    "configs/core_policy_registry.json",
    "configs/policy_profiles.json",
    "configs/span_level_template_candidate_ablation_preregistration.json",
    "span_level_template_compaction.py",
    "span_level_candidate_development_runner.py",
)


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _text_only_chunks(rows: Iterable[JsonMap]) -> list[JsonMap]:
    """Remove selector-visible metadata before the candidate development arm runs."""
    chunks: list[JsonMap] = []
    for row in rows:
        chunk = dict(row)
        visible = dict(chunk.get("stage_c_selector_visible") or {})
        visible.update(
            {
                "declared_language": False,
                "declared_language_version": False,
                "declared_content_type": False,
                "declared_path": False,
                "declared_artifact_context": False,
                "source_name": False,
                "source_pool_role": False,
                "composition": False,
                "utility": False,
                "benchmark_outcomes": False,
            }
        )
        chunk["stage_c_policy_metadata"] = {}
        chunk["stage_c_selector_visible"] = visible
        chunks.append(chunk)
    return chunks


def _policy_fingerprint() -> JsonMap:
    root = Path(__file__).resolve().parent
    return {path: sha256_file(root / path) for path in POLICY_FILES}


def materialize_development_candidate(
    *,
    stage_b_path: Path,
    frozen_input_path: Path,
    output_dir: Path,
    stage_c_selection: JsonMap,
    candidate_enabled: bool = True,
    minimum_span_tokens: int = 12,
    minimum_residual_tokens: int = 20,
) -> JsonMap:
    """Materialize a non-runtime candidate arm from a frozen Stage-B chunk snapshot."""
    if "coverage_guard" in stage_c_selection:
        raise RuntimeError("Development candidate arm forbids metadata-based coverage guard")
    source_chunks = _text_only_chunks(_read_jsonl(stage_b_path))
    impact_audit = build_candidate_impact_audit(
        source_chunks,
        minimum_span_tokens=minimum_span_tokens,
        minimum_residual_tokens=minimum_residual_tokens,
    )
    plan = build_plan(
        source_chunks,
        minimum_span_tokens=minimum_span_tokens,
        minimum_residual_tokens=minimum_residual_tokens,
    )
    candidate_materialization = (
        materialize_candidate_plan(source_chunks, plan)
        if candidate_enabled
        else {"records": source_chunks, "transformations": []}
    )
    selected, not_selected, selection_audit = select_chunks(candidate_materialization["records"], stage_c_selection)
    reason_audit = build_reason_code_impact_audit(
        stage_a_quarantined=[],
        stage_b_rejected=[],
        stage_c_not_selected=not_selected,
        stage_c_transformations=candidate_materialization["transformations"],
    )
    paths = {
        "preselection_chunks": output_dir / "stage_c_candidate_preselection_chunks.jsonl",
        "curated_chunks": output_dir / "stage_c_candidate_curated_chunks.jsonl",
        "not_selected_chunks": output_dir / "stage_c_candidate_not_selected_chunks.jsonl",
        "transformations": output_dir / "stage_c_candidate_transformations.jsonl",
    }
    _write_jsonl(paths["preselection_chunks"], candidate_materialization["records"])
    _write_jsonl(paths["curated_chunks"], selected)
    _write_jsonl(paths["not_selected_chunks"], not_selected)
    _write_jsonl(paths["transformations"], candidate_materialization["transformations"])
    report = {
        "schema_version": "span-level-template-development-run-v1",
        "status": "development_candidate_materialization_complete_not_runtime_active",
        "runtime_active": False,
        "candidate_policy_id": "stage_c_repeated_span_template_candidate",
        "candidate_enabled": candidate_enabled,
        "candidate_allowed_inputs": ["chunk text"],
        "frozen_input_snapshot": {
            "input_sha256": sha256_file(frozen_input_path),
            "stage_b_pass_sha256": sha256_file(stage_b_path),
            "policy_fingerprint": _policy_fingerprint(),
        },
        "candidate_impact_audit": impact_audit,
        "reason_code_impact_audit": reason_audit,
        "stage_c_selection": selection_audit,
        "summary": {
            "stage_b_pass_chunks": len(source_chunks),
            "candidate_preselection_chunks": len(candidate_materialization["records"]),
            "stage_c_curated_chunks": len(selected),
            "stage_c_not_selected_chunks": len(not_selected),
            "candidate_transformations": len(candidate_materialization["transformations"]),
        },
        "claim_boundary": "A development-only candidate arm. It cannot activate or tune the active runtime and external evaluation remains post hoc.",
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
    }
    save_json(output_dir / "candidate_development_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a frozen, development-only span-compaction candidate arm.")
    parser.add_argument("--stage-b-pass", type=Path, required=True)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-c-selection-config", type=Path, required=True)
    parser.add_argument("--minimum-span-tokens", type=int, default=12)
    parser.add_argument("--minimum-residual-tokens", type=int, default=20)
    parser.add_argument("--candidate-enabled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    stage_c_selection = json.loads(args.stage_c_selection_config.read_text(encoding="utf-8-sig"))
    if not isinstance(stage_c_selection, dict):
        raise RuntimeError("Stage-C selection config must be a JSON object")
    report = materialize_development_candidate(
        stage_b_path=args.stage_b_pass,
        frozen_input_path=args.frozen_input,
        output_dir=args.output_dir,
        stage_c_selection=stage_c_selection,
        candidate_enabled=args.candidate_enabled,
        minimum_span_tokens=args.minimum_span_tokens,
        minimum_residual_tokens=args.minimum_residual_tokens,
    )
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
