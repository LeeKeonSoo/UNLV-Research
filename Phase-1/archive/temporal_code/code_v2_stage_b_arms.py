#!/usr/bin/env python3
"""Freeze code-domain v2 Stage-B arms before Stage-C training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks
from ingestion.code_selection import select_stage_b, token_proxy_count


DEFAULT_DESIGN = Path("configs") / "code_domain_next_development_cycle_v2_design.json"
DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_STAGE0_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined"
DEFAULT_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_balanced"
DEFAULT_REFERENCE_DIR = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "known_high_quality_reference_pool"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2"
PROJECT_DIR = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATHS = (Path("ingestion/code_chunks.py"), Path("ingestion/code_selection.py"))


def _implementation_sha256() -> Dict[str, str]:
    return {str(path).replace("\\", "/"): sha256_file(PROJECT_DIR / path) for path in IMPLEMENTATION_PATHS}


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _token(row: Dict[str, Any]) -> int:
    evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
    return int(evidence.get("token_proxy_count") or row.get("token_proxy_count") or token_proxy_count(str(row.get("text") or "")))


def _stable_order(rows: Iterable[Dict[str, Any]], seed: int, label: str) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{label}:{row['chunk_uid']}".encode("utf-8")).hexdigest())


def _take_until_budget(rows: Iterable[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    total = 0
    for row in rows:
        selected.append(row)
        total += _token(row)
        if total >= budget:
            break
    return selected


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return round(mean(float(row["stage_b_evidence"][key]) for row in rows), 6) if rows else 0.0


def _arm_record(row: Dict[str, Any], arm: str, source_pool: str) -> Dict[str, Any]:
    return {
        "arm": arm,
        "chunk_uid": row["chunk_uid"],
        "text": row.get("text"),
        "token_proxy_count": _token(row),
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
        "raw_pool_stage": row.get("raw_pool_stage"),
    }


def _summarize_arm(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    repos = {str((row.get("provenance") or {}).get("repository_identity") or "missing") for row in rows}
    content_types = {}
    for row in rows:
        value = str((row.get("provenance") or {}).get("content_type") or "missing")
        content_types[value] = content_types.get(value, 0) + 1
    return {
        "records": len(rows),
        "token_proxy_count": sum(int(row["token_proxy_count"]) for row in rows),
        "repository_count": len(repos),
        "content_type_counts": dict(sorted(content_types.items())),
    }


def _raw_chunks(stage0_dir: Path) -> Dict[str, Any]:
    chunks: List[Dict[str, Any]] = []
    unchunkable: List[Dict[str, Any]] = []
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
    for row in decisions:
        row["token_proxy_count"] = _token(row)
    return {"chunks": decisions, "unchunkable": unchunkable}


def _arm_specs(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    coverage = contract["coverage_support"]
    base_axes = [str(value) for value in coverage["axes"]]
    base_distribution = [str(value) for value in coverage["distribution_axes"]]
    return {
        "full_selector": {
            "quality_weight": 0.8,
            "redundancy_weight": 0.2,
            "coverage_axes": base_axes,
            "distribution_axes": base_distribution,
        },
        "quality_only": {
            "quality_weight": 1.0,
            "redundancy_weight": 0.0,
            "coverage_axes": base_axes,
            "distribution_axes": base_distribution,
        },
        "redundancy_only": {
            "quality_weight": 0.0,
            "redundancy_weight": 1.0,
            "coverage_axes": base_axes,
            "distribution_axes": base_distribution,
        },
        "no_coverage_support": {
            "quality_weight": 0.8,
            "redundancy_weight": 0.2,
            "coverage_axes": [],
            "distribution_axes": [],
        },
        "no_test_code_balance": {
            "quality_weight": 0.8,
            "redundancy_weight": 0.2,
            "coverage_axes": [axis for axis in base_axes if axis != "content_type"],
            "distribution_axes": [axis for axis in base_distribution if axis != "content_type"],
        },
        "no_repository_diversity_cap": {
            "quality_weight": 0.8,
            "redundancy_weight": 0.2,
            "coverage_axes": [axis for axis in base_axes if axis != "bundle_id"],
            "distribution_axes": [axis for axis in base_distribution if axis != "bundle_id"],
        },
    }


def freeze(
    design_path: Path,
    protocol_path: Path,
    stage0_dir: Path,
    stage_a_dir: Path,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    design = load_json(design_path)
    protocol = load_json(protocol_path)
    contract = protocol["stage_b_contract"]
    coverage = contract["coverage_support"]
    baseline_seed = int(contract["stage_a_random_baseline"]["seed"])
    records = list(_jsonl(stage_a_dir / "train" / "stage_a_pass.jsonl"))
    if any(row.get("split") != "train" or row.get("stage_a_pass") is not True for row in records):
        raise ValueError("Stage-B v2 freeze accepts train split Stage-A-pass chunks only.")

    output_dir.mkdir(parents=True, exist_ok=True)
    arm_results: Dict[str, Dict[str, Any]] = {}
    full_result: Dict[str, Any] | None = None
    specs = _arm_specs(contract)
    for arm, spec in specs.items():
        result = select_stage_b(
            records,
            budget_fraction=float(contract["budget"]["fraction"]),
            quality_weight=float(spec["quality_weight"]),
            redundancy_weight=float(spec["redundancy_weight"]),
            coverage_axes=list(spec["coverage_axes"]),
            minimum_exemplars=int(coverage["minimum_exemplars_per_observed_value"]),
            baseline_seed=baseline_seed,
            distribution_axes=list(spec["distribution_axes"]),
            minimum_relative_token_share=float(coverage["minimum_relative_token_share"]) if spec["distribution_axes"] else 0.0,
            redundancy_search_mode=str(contract["objective"]["redundancy_search_mode"]),
        )
        if arm == "full_selector":
            full_result = result
            _write_jsonl(output_dir / "train_scored_full_selector.jsonl", result["scored"])
            _write_jsonl(output_dir / "curated_v2_equal_budget.jsonl", [_arm_record(row, "curated_v2_equal_budget", arm) for row in result["selected"]])
            _write_jsonl(output_dir / "stageA_random_equal_budget.jsonl", [_arm_record(row, "stageA_random_equal_budget", "stageA_random_disjoint") for row in result["baseline"]])
        else:
            _write_jsonl(output_dir / f"{arm}_selected.jsonl", [_arm_record(row, arm, arm) for row in result["selected"]])
        arm_results[arm] = {
            "selected_chunks": len(result["selected"]),
            "selected_token_proxy": int(result["selected_token_proxy"]),
            "budget_token_proxy": int(result["budget_token_proxy"]),
            "mean_code_quality_proxy": _mean(result["selected"], "code_quality_proxy"),
            "mean_soft_redundancy_risk": _mean(result["selected"], "soft_redundancy_risk"),
            "mean_stage_b_objective_score": _mean(result["selected"], "stage_b_objective_score"),
            "coverage_axes": spec["coverage_axes"],
            "distribution_axes": spec["distribution_axes"],
        }

    if full_result is None:
        raise RuntimeError("Full selector result was not produced.")
    raw = _raw_chunks(stage0_dir)
    raw_pool = raw["chunks"]
    raw_random = _take_until_budget(_stable_order(raw_pool, baseline_seed, "raw_random_v2"), int(full_result["selected_token_proxy"]))
    _write_jsonl(output_dir / "raw_random_equal_budget.jsonl", [_arm_record(row, "raw_random_equal_budget", "raw_stage0_chunkable_before_stage_a") for row in raw_random])
    reference_path = reference_dir / "known_high_quality_stage_a_pass.jsonl"
    reference_pool = [{**row, "token_proxy_count": _token(row)} for row in _jsonl(reference_path)]
    known_hq = _take_until_budget(
        _stable_order(reference_pool, baseline_seed, "known_high_quality_v2"),
        int(full_result["selected_token_proxy"]),
    )
    _write_jsonl(
        output_dir / "known_high_quality_equal_budget.jsonl",
        [_arm_record(row, "known_high_quality_equal_budget", "known_high_quality_reference_pool") for row in known_hq],
    )

    selected_ids = {row["chunk_uid"] for row in full_result["selected"]}
    baseline_ids = {row["chunk_uid"] for row in full_result["baseline"]}
    report = {
        "schema_version": "code-domain-v2-stage-b-arms-freeze-v1",
        "status": "stage_b_v2_arms_frozen_before_stage_c",
        "source_sha256": {
            str(design_path): sha256_file(design_path),
            str(protocol_path): sha256_file(protocol_path),
            str(stage_a_dir / "train" / "stage_a_pass.jsonl"): sha256_file(stage_a_dir / "train" / "stage_a_pass.jsonl"),
            str(stage0_dir / "train" / "release_candidates.jsonl"): sha256_file(stage0_dir / "train" / "release_candidates.jsonl"),
            str(reference_path): sha256_file(reference_path),
        },
        "implementation_sha256": _implementation_sha256(),
        "inputs": {
            "stage_a_dir": str(stage_a_dir),
            "stage0_dir": str(stage0_dir),
            "train_stage_a_pass_chunks": len(records),
            "raw_train_chunkable_chunks": len(raw_pool),
            "raw_train_unchunkable_records": len(raw["unchunkable"]),
            "known_high_quality_pool_chunks": len(reference_pool),
        },
        "primary_arms": {
            "curated_v2_equal_budget": _summarize_arm(list(_jsonl(output_dir / "curated_v2_equal_budget.jsonl"))),
            "stageA_random_equal_budget": _summarize_arm(list(_jsonl(output_dir / "stageA_random_equal_budget.jsonl"))),
            "raw_random_equal_budget": _summarize_arm(list(_jsonl(output_dir / "raw_random_equal_budget.jsonl"))),
            "known_high_quality_equal_budget": _summarize_arm(list(_jsonl(output_dir / "known_high_quality_equal_budget.jsonl"))),
        },
        "ablations": arm_results,
        "disjointness": {
            "curated_v2_stageA_random_disjoint": not bool(selected_ids.intersection(baseline_ids)),
            "intersection_count": len(selected_ids.intersection(baseline_ids)),
        },
        "required_ablations_from_design": design["stage_b_v2_proxy_plan"]["required_ablations"],
        "selector_signal_policy": design["selector_signal_policy"],
        "selection_forbids": design["selector_signal_policy"]["forbidden_stage_b_signals"],
        "utility_scope": design["stage_boundaries"]["utility_scope"],
        "confirmatory_outcomes_read_for_v2": design["confirmatory_outcomes_read_for_v2"],
        "claim_boundary": "Stage-B v2 arm freeze only; no Stage-C, Utility, training, confirmatory, release, or paper success claim.",
    }
    save_json(output_dir / "stage_b_v2_arms_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain v2 Stage-B arms.")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = freeze(args.design, args.protocol, args.stage0_dir, args.stage_a_dir, args.reference_dir, args.output_dir)
    print(
        {
            "status": report["status"],
            "primary_arms": report["primary_arms"],
            "disjointness": report["disjointness"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
