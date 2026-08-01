#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from data_eval_common import load_json, save_json, sha256_file
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks
from ingestion.code_selection import select_stage_b


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_ROOT = Path("D:/UNLV-Research/code_5m_corpus_v2")
DEFAULT_INPUT = DEFAULT_CORPUS_ROOT / "stage0_output" / "release_candidates.jsonl"
DEFAULT_POLICY = ROOT / "configs" / "temporal_code_curation_protocol_v1.json"
DEFAULT_OUTPUT = DEFAULT_CORPUS_ROOT / "stages"
SELECTOR_FIELDS = frozenset(
    {
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
)
FORBIDDEN_SELECTOR_FIELDS = frozenset(
    {
        "source_tier",
        "source_dataset",
        "provenance",
        "rights",
        "hazards",
        "utility",
        "benchmark_outcomes",
    }
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stage_a_chunks(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    unchunkable: list[dict[str, Any]] = []
    for row in rows:
        partition = row.get("partition") if isinstance(row.get("partition"), dict) else {}
        result = syntax_aware_chunks(row)
        if not result["parseable"]:
            unchunkable.append(
                {"record_id": row["record_id"], "path": partition.get("path"), "stage_a_blockers": ["source_record_not_parseable"]}
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


def selector_inputs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key in SELECTOR_FIELDS} for row in rows]


def _selector_result(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    contract = policy["stage_b_contract"]
    return select_stage_b(
        selector_inputs(rows),
        budget_fraction=float(contract["budget"]["fraction"]),
        quality_weight=float(contract["objective"]["code_quality_proxy_weight"]),
        redundancy_weight=float(contract["objective"]["soft_redundancy_support_weight"]),
        coverage_axes=[str(value) for value in contract["coverage_support"]["axes"]],
        minimum_exemplars=int(contract["coverage_support"]["minimum_exemplars_per_observed_value"]),
        baseline_seed=int(contract["stage_a_random_baseline"]["seed"]),
        distribution_axes=[str(value) for value in contract["coverage_support"]["distribution_axes"]],
        minimum_relative_token_share=float(contract["coverage_support"]["minimum_relative_token_share"]),
        redundancy_search_mode=str(contract["objective"]["redundancy_search_mode"]),
    )


def run(input_path: Path, policy_path: Path, output_dir: Path) -> dict[str, Any]:
    records = _read_jsonl(input_path)
    decisions, unchunkable = stage_a_chunks(records)
    passed = [row for row in decisions if row["stage_a_pass"]]
    rejected = [row for row in decisions if not row["stage_a_pass"]]
    result = _selector_result(passed, load_json(policy_path))
    selected = result["selected"]
    baseline = result["baseline"]
    selected_ids = {str(row["chunk_uid"]) for row in selected}
    baseline_ids = {str(row["chunk_uid"]) for row in baseline}
    overlap = selected_ids.intersection(baseline_ids)
    paths = {
        "stage_a_pass": output_dir / "stage_a_pass.jsonl",
        "stage_a_rejected": output_dir / "stage_a_rejected.jsonl",
        "stage_a_unchunkable": output_dir / "stage_a_unchunkable.jsonl",
        "stage_b_scored": output_dir / "stage_b_scored.jsonl",
        "stage_b_selected": output_dir / "stage_b_selected.jsonl",
        "stage_b_baseline": output_dir / "stage_b_baseline.jsonl",
        "stage_b_budget_not_selected": output_dir / "stage_b_budget_not_selected.jsonl",
    }
    _write_jsonl(paths["stage_a_pass"], passed)
    _write_jsonl(paths["stage_a_rejected"], rejected)
    _write_jsonl(paths["stage_a_unchunkable"], unchunkable)
    _write_jsonl(paths["stage_b_scored"], result["scored"])
    _write_jsonl(paths["stage_b_selected"], selected)
    _write_jsonl(paths["stage_b_baseline"], baseline)
    _write_jsonl(paths["stage_b_budget_not_selected"], result["budget_not_selected"])
    selector_keys = {key for row in selector_inputs(passed) for key in row}
    report = {
        "schema_version": "code-5m-stages-report-v1",
        "status": "code_5m_stages_materialized" if not overlap else "code_5m_stages_blocked",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "summary": {
            "release_candidate_record_count": len(records),
            "stage_a_chunk_count": len(decisions),
            "stage_a_pass_count": len(passed),
            "stage_a_rejected_count": len(rejected),
            "stage_a_unchunkable_count": len(unchunkable),
            "stage_a_rejection_reason_counts": dict(sorted(Counter(reason for row in rejected for reason in row["stage_a_blockers"]).items())),
            "stage_b_selected_count": len(selected),
            "stage_b_selected_token_proxy": result["selected_token_proxy"],
            "stage_b_baseline_count": len(baseline),
            "stage_b_baseline_token_proxy": result["baseline_token_proxy"],
            "selected_baseline_overlap_count": len(overlap),
        },
        "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "stage_b_blinding_audit": {
            "selector_input_keys": sorted(selector_keys),
            "forbidden_key_seen": bool(selector_keys.intersection(FORBIDDEN_SELECTOR_FIELDS)),
            "source_tier_available_to_stage_b": False,
            "utility_available_to_stage_b": False,
            "benchmark_outcomes_available_to_stage_b": False,
        },
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_dir / "stages_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage A and Stage B for the frozen 5M corpus.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.input, args.policy, args.output_dir)
    print(report["summary"])
    return 0 if report["status"] == "code_5m_stages_materialized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
