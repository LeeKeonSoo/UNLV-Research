#!/usr/bin/env python3
"""Run split-isolated syntax-aware Stage-A hard gates on temporal-code smoke records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, save_json
from ingestion.code_chunks import (
    apply_stage_a_hard_gates,
    hard_near_duplicate_evidence,
    syntax_aware_chunks,
    token_shingles,
)


DEFAULT_STAGE0_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_smoke"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_smoke"
SPLITS = ("train", "development", "confirmatory")


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunks_for_split(stage0_dir: Path, split: str) -> Dict[str, Any]:
    chunks = []
    unchunkable = []
    for record in _jsonl(stage0_dir / split / "release_candidates.jsonl"):
        partition = record.get("partition") if isinstance(record.get("partition"), dict) else {}
        if partition.get("split") != split:
            raise ValueError(f"Stage-0 partition mismatch for {record.get('record_id')}")
        result = syntax_aware_chunks(record)
        if not result["parseable"]:
            unchunkable.append(
                {
                    "record_id": record["record_id"],
                    "path": partition.get("path"),
                    "parse_error": result["parse_error"],
                    "stage_a_blockers": ["source_record_not_parseable"],
                }
            )
            continue
        for index, chunk in enumerate(result["chunks"]):
            chunks.append(
                {
                    "chunk_uid": f"{record['record_id']}::chunk-{index:04d}",
                    "record_id": record["record_id"],
                    "split": split,
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
    decisions = apply_stage_a_hard_gates(chunks)
    return {"decisions": decisions, "unchunkable": unchunkable}


def run(stage0_dir: Path, output_dir: Path) -> Dict[str, Any]:
    split_reports = {}
    all_decisions = []
    for split in SPLITS:
        result = chunks_for_split(stage0_dir, split)
        decisions = result["decisions"]
        passed = [row for row in decisions if row["stage_a_pass"]]
        rejected = [row for row in decisions if not row["stage_a_pass"]]
        reason_counts = Counter(
            reason
            for row in [*rejected, *result["unchunkable"]]
            for reason in row["stage_a_blockers"]
        )
        _write_jsonl(output_dir / split / "stage_a_pass.jsonl", passed)
        _write_jsonl(output_dir / split / "stage_a_rejected.jsonl", rejected)
        _write_jsonl(output_dir / split / "stage_a_unchunkable.jsonl", result["unchunkable"])
        split_reports[split] = {
            "input_stage0_records": sum(1 for _ in _jsonl(stage0_dir / split / "release_candidates.jsonl")),
            "chunk_count": len(decisions),
            "stage_a_pass_count": len(passed),
            "stage_a_rejected_count": len(rejected),
            "source_record_unchunkable_count": len(result["unchunkable"]),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
        }
        all_decisions.extend(decisions)
    cross_split_exact = []
    cross_split_hard_near = []
    prepared = [
        {
            "chunk_uid": row["chunk_uid"],
            "split": row["split"],
            "text_sha256": row["stage_a_evidence"]["text_sha256"],
            "token_simhash64": row["stage_a_evidence"].get("token_simhash64"),
            "shingles": token_shingles(row["text"]),
        }
        for row in all_decisions
    ]
    for index, left in enumerate(prepared):
        for right in prepared[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            if left["text_sha256"] == right["text_sha256"]:
                cross_split_exact.append([left["chunk_uid"], right["chunk_uid"]])
                continue
            evidence = hard_near_duplicate_evidence(left, right)
            if evidence["match"]:
                cross_split_hard_near.append(
                    {
                        "left_chunk_uid": left["chunk_uid"],
                        "right_chunk_uid": right["chunk_uid"],
                        "simhash_distance": evidence["simhash_distance"],
                        "jaccard": round(float(evidence["jaccard"]), 6),
                        "containment": round(float(evidence["containment"]), 6),
                    }
                )
    report = {
        "schema_version": "temporal-code-stage-a-smoke-report-v1",
        "stage_a_contract": {
            "unit": "syntax-aware chunk",
            "hard_gates": [
                "Python chunk parseability",
                "minimum learnable unit",
                "exact duplicate within split",
                "hard near-duplicate within split",
                "pathological repetition",
            ],
            "forbidden_signals": ["semantic quality", "coverage", "Utility", "benchmark outcomes"],
            "split_isolation": "Each split is chunked and gated independently; cross-split observations cannot change decisions.",
        },
        "summary": {
            "stage0_input_records": sum(row["input_stage0_records"] for row in split_reports.values()),
            "chunk_count": sum(row["chunk_count"] for row in split_reports.values()),
            "stage_a_pass_count": sum(row["stage_a_pass_count"] for row in split_reports.values()),
            "stage_a_rejected_count": sum(row["stage_a_rejected_count"] for row in split_reports.values()),
            "source_record_unchunkable_count": sum(
                row["source_record_unchunkable_count"] for row in split_reports.values()
            ),
            "split_counts": split_reports,
        },
        "confirmatory_use_boundary": (
            "Confirmatory Stage-A outcomes validate the frozen hard-gate contract only and must not tune Stage A or Stage B."
        ),
        "cross_split_diagnostic_only": {
            "decision_use": "forbidden",
            "exact_duplicate_pair_count": len(cross_split_exact),
            "hard_near_duplicate_pair_count": len(cross_split_hard_near),
            "exact_duplicate_pair_examples": cross_split_exact[:20],
            "hard_near_duplicate_pair_examples": cross_split_hard_near[:20],
        },
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Bounded Stage-A smoke only; no Stage-B selection, Stage-C validation, or training claim.",
    }
    save_json(output_dir / "stage_a_smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run temporal-code syntax-aware Stage-A smoke.")
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args.stage0_dir, args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
