#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks
from ingestion.code_selection import select_stage_b


ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX_ROOT = OUTPUT_DIR / "raw_corpus_matrix_v1"
DEFAULT_PROTOCOL_PATH = ROOT / "configs" / "temporal_code_curation_protocol_v1.json"
DEFAULT_OUTPUT_ROOT = OUTPUT_DIR / "raw_corpus_matrix_v1_stages"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "validation" / "raw_corpus_matrix_stages_report.json"
CONDITIONS = ("clean_retain_all", "raw_mixed", "risk_heavy")
FORBIDDEN_SELECTOR_KEYS = {
    "audit_provenance",
    "provenance",
    "rights",
    "hazards",
    "source_tier",
    "source_dataset",
    "source_config",
    "source_split",
    "source_uri",
    "license_family",
    "repository_or_origin",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stage_a_chunks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    unchunkable: list[dict[str, Any]] = []
    for row in rows:
        partition = row.get("partition") if isinstance(row.get("partition"), dict) else {}
        result = syntax_aware_chunks(row)
        if not result["parseable"]:
            unchunkable.append(
                {
                    "record_id": row["record_id"],
                    "path": partition.get("path"),
                    "stage_a_blockers": ["source_record_not_parseable"],
                }
            )
            continue
        for index, chunk in enumerate(result["chunks"]):
            chunks.append(
                {
                    "chunk_uid": f"{row['record_id']}::chunk-{index:04d}",
                    "record_id": row["record_id"],
                    "split": "train",
                    "bundle_id": partition.get("bundle_id"),
                    "repository_identity": partition.get("repository_identity"),
                    "path": partition.get("path"),
                    "change_type": partition.get("change_type"),
                    "content_type": partition.get("content_type"),
                    "chunking_mode": result["chunking_mode"],
                    "chunk_kind": chunk["kind"],
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                    "text": chunk["text"],
                }
            )
    return apply_stage_a_hard_gates(chunks), unchunkable


def _selector_inputs(stage_a_pass: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "chunk_uid",
        "record_id",
        "split",
        "bundle_id",
        "repository_identity",
        "path",
        "change_type",
        "content_type",
        "chunking_mode",
        "chunk_kind",
        "start_line",
        "end_line",
        "text",
        "stage_a_pass",
        "stage_a_blockers",
        "stage_a_evidence",
    }
    return [{key: value for key, value in row.items() if key in allowed} for row in stage_a_pass]


def _condition_stage_b(
    name: str, stage_a_pass: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    contract = protocol["stage_b_contract"]
    selector_input = _selector_inputs(stage_a_pass)
    budget_fraction = None if name == "clean_retain_all" else float(contract["budget"]["fraction"])
    result = select_stage_b(
        selector_input,
        budget_fraction=budget_fraction,
        quality_weight=float(contract["objective"]["code_quality_proxy_weight"]),
        redundancy_weight=float(contract["objective"]["soft_redundancy_support_weight"]),
        coverage_axes=[str(value) for value in contract["coverage_support"]["axes"]],
        minimum_exemplars=int(contract["coverage_support"]["minimum_exemplars_per_observed_value"]),
        baseline_seed=int(contract["stage_a_random_baseline"]["seed"]),
        distribution_axes=[str(value) for value in contract["coverage_support"]["distribution_axes"]],
        minimum_relative_token_share=float(contract["coverage_support"]["minimum_relative_token_share"]),
        redundancy_search_mode=str(contract["objective"]["redundancy_search_mode"]),
    )
    return {"selector_input": selector_input, "result": result}


def run(matrix_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    condition_reports: dict[str, dict[str, Any]] = {}
    frozen_conditions: dict[str, dict[str, str]] = {}
    seen_keys: set[str] = set()
    for name in CONDITIONS:
        rows = _read_jsonl(matrix_root / name / "release_candidates.jsonl")
        decisions, unchunkable = _stage_a_chunks(rows)
        passed = [row for row in decisions if row["stage_a_pass"]]
        rejected = [row for row in decisions if not row["stage_a_pass"]]
        stage_b = _condition_stage_b(name, passed, protocol)
        selector_input = stage_b["selector_input"]
        result = stage_b["result"]
        seen_keys.update(key for row in selector_input for key in row)
        target = output_root / name
        paths = {
            "stage_a_pass": target / "stage_a_pass.jsonl",
            "stage_a_rejected": target / "stage_a_rejected.jsonl",
            "stage_a_unchunkable": target / "stage_a_unchunkable.jsonl",
            "stage_b_scored": target / "stage_b_scored.jsonl",
            "stage_b_selected": target / "stage_b_selected.jsonl",
            "stage_b_baseline": target / "stage_b_baseline.jsonl",
            "stage_b_budget_not_selected": target / "stage_b_budget_not_selected.jsonl",
        }
        _write_jsonl(paths["stage_a_pass"], passed)
        _write_jsonl(paths["stage_a_rejected"], rejected)
        _write_jsonl(paths["stage_a_unchunkable"], unchunkable)
        _write_jsonl(paths["stage_b_scored"], result["scored"])
        _write_jsonl(paths["stage_b_selected"], result["selected"])
        _write_jsonl(paths["stage_b_baseline"], result["baseline"])
        _write_jsonl(paths["stage_b_budget_not_selected"], result["budget_not_selected"])
        frozen_conditions[name] = {
            "release_candidates_sha256": sha256_file(matrix_root / name / "release_candidates.jsonl"),
            **{f"{label}_sha256": sha256_file(path) for label, path in paths.items()},
        }
        condition_reports[name] = {
            "input_release_candidate_count": len(rows),
            "stage_a_chunk_count": len(decisions),
            "stage_a_pass_count": len(passed),
            "stage_a_rejected_count": len(rejected),
            "stage_a_unchunkable_count": len(unchunkable),
            "stage_a_rejection_reason_counts": dict(
                sorted(Counter(reason for row in rejected for reason in row["stage_a_blockers"]).items())
            ),
            "stage_b_selection_mode": result["selection_mode"],
            "stage_b_budget_applied": result["budget_applied"],
            "stage_b_selected_count": len(result["selected"]),
            "stage_b_budget_not_selected_count": len(result["budget_not_selected"]),
            "stage_b_selected_token_proxy": result["selected_token_proxy"],
            "stage_b_baseline_count": len(result["baseline"]),
            "stage_b_baseline_token_proxy": result["baseline_token_proxy"],
            "budget_not_selected_is_rejection": result["invariants"]["budget_not_selected_is_rejection"],
        }
    report = {
        "schema_version": "raw-corpus-matrix-stages-report-v1",
        "status": "raw_corpus_matrix_stages_materialized",
        "matrix_root": str(matrix_root),
        "protocol_path": str(protocol_path),
        "conditions": condition_reports,
        "frozen_input_manifest": {
            "stage_b_policy_sha256": sha256_file(protocol_path),
            "conditions": frozen_conditions,
        },
        "stage_b_blinding_audit": {
            "selector_input_keys": sorted(seen_keys),
            "forbidden_keys": sorted(FORBIDDEN_SELECTOR_KEYS),
            "forbidden_key_seen": bool(seen_keys.intersection(FORBIDDEN_SELECTOR_KEYS)),
            "source_tier_available_to_stage_b": False,
            "known_reference_label_available_to_stage_b": False,
            "benchmark_outcomes_available_to_stage_b": False,
        },
        "training_readiness": {
            "stage_a_materialized": True,
            "stage_b_materialized": True,
            "primary_study_ready": False,
            "required_next_action": "freeze split-isolated training arms and execute the preregistered multi-seed Stage-C study",
        },
        "utility_scope": "Stage C validation only; never selector objective",
    }
    if output_root == DEFAULT_OUTPUT_ROOT:
        save_json(DEFAULT_REPORT_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = run(args.matrix_root, args.protocol, args.output_root)
    save_json(DEFAULT_REPORT_PATH, report)
    print({"status": report["status"], "conditions": len(report["conditions"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
