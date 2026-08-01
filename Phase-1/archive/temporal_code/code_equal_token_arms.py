#!/usr/bin/env python3
"""Freeze code-domain equal-token training arms from raw, Stage-A, Stage-B, and reference pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks
from ingestion.code_selection import token_proxy_count


DEFAULT_CONFIG = Path("configs") / "code_domain_training_arm_freeze_v1.json"
DEFAULT_STAGE0_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_path_stratified_tranche"
DEFAULT_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_path_stratified_tranche"
DEFAULT_STAGE_B_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_path_stratified_tranche"
DEFAULT_REFERENCE_DIR = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "known_high_quality_reference_pool"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "equal_token_arms"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_order(records: Iterable[Dict[str, Any]], seed: int, label: str) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{label}:{row['chunk_uid']}".encode("utf-8")).hexdigest(),
    )


def _with_token_counts(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in records:
        evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
        count = int(evidence.get("token_proxy_count") or row.get("token_proxy_count") or token_proxy_count(str(row.get("text") or "")))
        result.append({**row, "token_proxy_count": count})
    return result


def _take_until_budget(records: Iterable[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    selected = []
    total = 0
    for row in records:
        count = int(row["token_proxy_count"])
        selected.append(row)
        total += count
        if total >= cap:
            break
    return selected


def _raw_chunks(stage0_dir: Path) -> Dict[str, Any]:
    chunks = []
    unchunkable = []
    for record in _jsonl(stage0_dir / "train" / "release_candidates.jsonl"):
        partition = record.get("partition") if isinstance(record.get("partition"), dict) else {}
        if partition.get("split") != "train":
            raise ValueError(f"Expected train split raw candidate, got {partition.get('split')}")
        result = syntax_aware_chunks(record)
        if not result["parseable"]:
            unchunkable.append({"record_id": record["record_id"], "parse_error": result["parse_error"]})
            continue
        for index, chunk in enumerate(result["chunks"]):
            chunks.append(
                {
                    "chunk_uid": f"raw::{record['record_id']}::chunk-{index:04d}",
                    "record_id": record["record_id"],
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
                    "raw_pool_stage": "stage0_chunkable_before_stage_a",
                }
            )
    decisions = apply_stage_a_hard_gates(chunks)
    return {"chunks": _with_token_counts(decisions), "unchunkable": unchunkable}


def _arm_record(row: Dict[str, Any], arm: str, source_pool: str) -> Dict[str, Any]:
    return {
        "arm": arm,
        "chunk_uid": row["chunk_uid"],
        "text": row["text"],
        "token_proxy_count": int(row["token_proxy_count"]),
        "source_pool": source_pool,
        "provenance": {
            "record_id": row.get("record_id"),
            "bundle_id": row.get("bundle_id"),
            "repository_identity": row.get("repository_identity"),
            "path": row.get("path"),
            "split": row.get("split"),
            "content_type": row.get("content_type"),
            "change_type": row.get("change_type"),
        },
        "stage_a_pass": row.get("stage_a_pass"),
        "stage_a_blockers": row.get("stage_a_blockers"),
        "stage_b_selection": row.get("stage_b_selection"),
        "stage_b_baseline": row.get("stage_b_baseline"),
        "stage_b_evidence": row.get("stage_b_evidence"),
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_count": len(records),
        "token_proxy_count": sum(int(row["token_proxy_count"]) for row in records),
        "repository_count": len({str((row.get("provenance") or {}).get("repository_identity")) for row in records}),
        "content_type_counts": dict(
            sorted(
                {
                    value: sum(1 for row in records if (row.get("provenance") or {}).get("content_type") == value)
                    for value in {str((row.get("provenance") or {}).get("content_type")) for row in records}
                }.items()
            )
        ),
    }


def freeze(
    config_path: Path,
    stage0_dir: Path,
    stage_a_dir: Path,
    stage_b_dir: Path,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    config = load_json(config_path)
    seed = int(config["seed"])
    raw = _raw_chunks(stage0_dir)
    raw_pool = _with_token_counts(raw["chunks"])
    stage_a_pool = _with_token_counts(_jsonl(stage_a_dir / "train" / "stage_a_pass.jsonl"))
    curated_pool = _with_token_counts(_jsonl(stage_b_dir / "train_selected.jsonl"))
    stage_a_random_pool = _with_token_counts(_jsonl(stage_b_dir / "train_stage_a_random_disjoint.jsonl"))
    reference_pool_path = reference_dir / "known_high_quality_stage_a_pass.jsonl"
    if not reference_pool_path.exists():
        raise FileNotFoundError(f"Known-high-quality reference pool missing: {reference_pool_path}")
    reference_pool = _with_token_counts(_jsonl(reference_pool_path))

    code_tokens = {
        "raw_chunkable": sum(row["token_proxy_count"] for row in raw_pool if row.get("content_type") == "code"),
        "stageA_pass": sum(row["token_proxy_count"] for row in stage_a_pool if row.get("content_type") == "code"),
        "curated": sum(row["token_proxy_count"] for row in curated_pool if row.get("content_type") == "code"),
        "stageA_random": sum(row["token_proxy_count"] for row in stage_a_random_pool if row.get("content_type") == "code"),
        "known_high_quality": sum(row["token_proxy_count"] for row in reference_pool),
    }
    blockers = [
        f"insufficient_python_code_tokens:{name}:{tokens}"
        for name, tokens in sorted(code_tokens.items())
        if tokens < int(config["minimum_python_code_token_proxy"])
    ]
    cap = min(
        sum(row["token_proxy_count"] for row in curated_pool),
        sum(row["token_proxy_count"] for row in stage_a_random_pool),
        sum(row["token_proxy_count"] for row in raw_pool),
        sum(row["token_proxy_count"] for row in reference_pool),
    )
    if cap < int(config["minimum_equal_arm_token_proxy"]):
        blockers.append(f"equal_token_cap_below_minimum:{cap}")

    arms = {
        "raw_random_equal_budget": _take_until_budget(_stable_order(raw_pool, seed, "raw"), cap),
        "stageA_random_equal_budget": _take_until_budget(stage_a_random_pool, cap),
        "curated_equal_budget": _take_until_budget(curated_pool, cap),
        "known_high_quality_equal_budget": _take_until_budget(_stable_order(reference_pool, seed, "known_hq"), cap),
    }
    materialized = {
        name: [_arm_record(row, name, name.replace("_equal_budget", "")) for row in rows]
        for name, rows in arms.items()
    }
    underfilled_arms = {
        name: sum(int(row["token_proxy_count"]) for row in rows)
        for name, rows in materialized.items()
        if sum(int(row["token_proxy_count"]) for row in rows) < cap
    }
    blockers.extend(
        f"arm_materialized_below_training_cap:{name}:{tokens}"
        for name, tokens in sorted(underfilled_arms.items())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in materialized.items():
        _write_jsonl(output_dir / f"{name}.jsonl", records)

    selected_ids = {row["chunk_uid"] for row in materialized["curated_equal_budget"]}
    baseline_ids = {row["chunk_uid"] for row in materialized["stageA_random_equal_budget"]}
    report = {
        "schema_version": "code-domain-equal-token-training-arms-report-v1",
        "config_schema": config["schema_version"],
        "status": "training_arms_frozen" if not blockers else "training_arms_frozen_with_blockers",
        "source_sha256": {
            str(config_path): sha256_file(config_path),
            str(stage0_dir / "train" / "release_candidates.jsonl"): sha256_file(stage0_dir / "train" / "release_candidates.jsonl"),
            str(stage_a_dir / "train" / "stage_a_pass.jsonl"): sha256_file(stage_a_dir / "train" / "stage_a_pass.jsonl"),
            str(stage_b_dir / "train_selected.jsonl"): sha256_file(stage_b_dir / "train_selected.jsonl"),
            str(stage_b_dir / "train_stage_a_random_disjoint.jsonl"): sha256_file(stage_b_dir / "train_stage_a_random_disjoint.jsonl"),
            str(reference_pool_path): sha256_file(reference_pool_path),
        },
        "summary": {
            "training_token_budget_cap": cap,
            "materialized_rows_may_exceed_cap": True,
            "training_must_truncate_or_pack_to_cap": True,
            "raw_chunkable_pool_chunks": len(raw_pool),
            "raw_unchunkable_records": len(raw["unchunkable"]),
            "stage_a_pool_chunks": len(stage_a_pool),
            "curated_pool_chunks": len(curated_pool),
            "stage_a_random_pool_chunks": len(stage_a_random_pool),
            "known_high_quality_pool_chunks": len(reference_pool),
            "code_token_proxy_by_pool": code_tokens,
            "curated_stageA_random_disjoint": not bool(selected_ids.intersection(baseline_ids)),
            "blockers": blockers,
        },
        "arms": {name: _summarize(records) for name, records in materialized.items()},
        "training_payload_contract": config["training_payload"],
        "selection_forbids": config["selection_forbids"],
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    save_json(output_dir / "equal_token_training_arms_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain equal-token training arms.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--stage-b-dir", type=Path, default=DEFAULT_STAGE_B_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = freeze(args.config, args.stage0_dir, args.stage_a_dir, args.stage_b_dir, args.reference_dir, args.output_dir)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
