#!/usr/bin/env python3
"""Materialize a frozen, candidate-only repeated-navigation-block development arm."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from curation_artifacts import save_json, sha256_file
from reason_code_audit import build_reason_code_impact_audit
from repeated_line_block_compaction import build_plan, materialize_candidate_plan


JsonMap = dict[str, Any]
TokenCounter = Callable[[str], int]


def _copy_rows(rows: Iterable[JsonMap]) -> list[JsonMap]:
    return [dict(row) for row in rows]


def _token_total(rows: Iterable[JsonMap], token_counter: TokenCounter) -> int:
    return sum(token_counter(str(row.get("text") or "")) for row in rows)


def _coverage(source_rows: list[JsonMap], candidate_rows: list[JsonMap], transformations: list[JsonMap], minimum_residual_chars: int) -> JsonMap:
    source_ids = {str(row.get("chunk_uid") or "unknown") for row in source_rows}
    candidate_ids = {str(row.get("chunk_uid") or "unknown") for row in candidate_rows}
    candidate_by_id = {str(row.get("chunk_uid") or "unknown"): row for row in candidate_rows}
    transformed_ids = {str(item["chunk_uid"]) for item in transformations}
    residual_payload_passed = all(
        len(str(candidate_by_id[chunk_uid].get("text") or "").strip()) >= minimum_residual_chars
        for chunk_uid in transformed_ids
    )
    first_occurrence_linkage_passed = all(
        item.get("representative_occurrence") == "earlier_in_same_chunk"
        and isinstance(item.get("representative_block_sha256"), str)
        and bool(item["representative_block_sha256"])
        for item in transformations
    )
    return {
        "authority": "audit_only",
        "whole_chunk_preservation_passed": source_ids == candidate_ids,
        "residual_payload_passed": residual_payload_passed,
        "first_occurrence_linkage_passed": first_occurrence_linkage_passed,
        "passed": source_ids == candidate_ids and residual_payload_passed and first_occurrence_linkage_passed,
    }


def run_candidate_matrix(rows: Iterable[JsonMap], *, minimum_residual_chars: int, token_counter: TokenCounter) -> JsonMap:
    """Compare one candidate span-only arm against its frozen input without runtime authority."""
    if minimum_residual_chars < 1:
        raise ValueError("minimum_residual_chars must be positive")
    source_rows = _copy_rows(rows)
    if not source_rows:
        raise ValueError("Repeated-label candidate matrix requires at least one Stage-B pass chunk")
    plan = build_plan(source_rows, minimum_residual_chars=minimum_residual_chars)
    materialized = materialize_candidate_plan(source_rows, plan)
    curated_rows = materialized["records"]
    transformations = materialized["transformations"]
    input_tokens = _token_total(source_rows, token_counter)
    curated_tokens = _token_total(curated_rows, token_counter)
    return {
        "schema_version": "repeated-label-block-development-matrix-v1",
        "status": "development_candidate_complete_not_runtime_active",
        "runtime_active": False,
        "candidate_policy_id": "stage_c_repeated_label_block_candidate",
        "allowed_inputs": ["chunk text", "declared Stage-B minimum_residual_chars"],
        "forbidden_selector_inputs": ["intrinsic_quality_score", "Utility", "NLL", "benchmark_outcomes", "source_identity", "composition", "target_retention_fraction"],
        "summary": {
            "input_chunks": len(source_rows),
            "curated_chunks": len(curated_rows),
            "candidate_span_removals": len(transformations),
            "input_tokens": input_tokens,
            "curated_tokens": curated_tokens,
            "token_delta": curated_tokens - input_tokens,
        },
        "reason_code_impact_audit": build_reason_code_impact_audit([], [], [], transformations),
        "coverage": _coverage(source_rows, curated_rows, transformations, minimum_residual_chars),
        "claim_boundary": "Candidate-only structural evidence. The result cannot activate Normal or Hard, tune a policy from benchmark outcomes, or establish downstream effectiveness.",
    }


def _token_counter(tokenizer_path: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a frozen repeated-label-block candidate development matrix.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--minimum-residual-chars", type=int, default=40)
    args = parser.parse_args()
    report = run_candidate_matrix(
        _read_jsonl(args.input),
        minimum_residual_chars=args.minimum_residual_chars,
        token_counter=_token_counter(args.tokenizer),
    )
    report["input"] = {
        "path": str(args.input),
        "sha256": sha256_file(args.input),
        "tokenizer_path": args.tokenizer,
        "token_count_kind": "frozen_tokenizer",
    }
    save_json(args.output, report)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
