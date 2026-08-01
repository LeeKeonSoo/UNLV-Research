#!/usr/bin/env python3
"""Freeze binary, log-count, and common-disjoint Stage-A-random proxy arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import select_stage_b


DEFAULT_STAGE_A = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_a_code_domain_v2_balanced"
    / "train"
    / "stage_a_pass.jsonl"
)
DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_CANDIDATE = Path("configs") / "temporal_code_redundancy_saturation_proxy_candidate_v1.json"
DEFAULT_OUTPUT_DIR = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "redundancy_saturation_proxy_arms_v1"
)


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _token(row: Dict[str, Any]) -> int:
    evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
    return int(evidence.get("token_proxy_count") or row.get("token_proxy_count") or 0)


def _stable_order(rows: Iterable[Dict[str, Any]], seed: int, label: str) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{label}:{row['chunk_uid']}".encode("utf-8")
        ).hexdigest(),
    )


def _take_to_materialized_cap(rows: Iterable[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    selected = []
    total = 0
    for row in rows:
        if total >= cap:
            break
        selected.append(row)
        total += _token(row)
    return selected


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
            "chunk_kind": row.get("chunk_kind"),
        },
        "stage_a_pass": row.get("stage_a_pass"),
        "stage_a_blockers": row.get("stage_a_blockers"),
        "stage_b_selection": row.get("stage_b_selection"),
        "stage_b_baseline": row.get("stage_b_baseline"),
        "stage_b_evidence": row.get("stage_b_evidence"),
        "training_cap_contract": {
            "budget_basis": "token_proxy_before_target_tokenizer_freeze",
            "target_token_proxy_cap": None,
            "target_tokenizer_exact_packing_deferred": True,
        },
    }


def _summarize(rows: List[Dict[str, Any]], training_cap: int) -> Dict[str, Any]:
    materialized = sum(int(row["token_proxy_count"]) for row in rows)
    return {
        "record_count": len(rows),
        "materialized_token_proxy": materialized,
        "training_token_proxy_cap": training_cap,
        "materialized_covers_cap": materialized >= training_cap,
        "last_record_requires_partial_consumption": materialized > training_cap,
        "excess_materialized_token_proxy": max(0, materialized - training_cap),
        "repository_count": len(
            {
                str((row.get("provenance") or {}).get("repository_identity") or "")
                for row in rows
            }
        ),
        "content_type_counts": dict(
            sorted(
                Counter(
                    str((row.get("provenance") or {}).get("content_type") or "unknown")
                    for row in rows
                ).items()
            )
        ),
    }


def freeze(
    stage_a_path: Path,
    protocol_path: Path,
    candidate_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    records = list(_jsonl(stage_a_path))
    protocol = load_json(protocol_path)["stage_b_contract"]
    candidate = load_json(candidate_path)
    coverage = protocol["coverage_support"]
    common = {
        "budget_fraction": float(protocol["budget"]["fraction"]),
        "quality_weight": 0.8,
        "redundancy_weight": 0.2,
        "coverage_axes": [str(value) for value in coverage["axes"]],
        "minimum_exemplars": int(coverage["minimum_exemplars_per_observed_value"]),
        "baseline_seed": int(protocol["stage_a_random_baseline"]["seed"]),
        "distribution_axes": [str(value) for value in coverage["distribution_axes"]],
        "minimum_relative_token_share": float(coverage["minimum_relative_token_share"]),
        "redundancy_search_mode": str(protocol["objective"]["redundancy_search_mode"]),
    }
    binary = select_stage_b(
        records,
        structural_saturation_mode="binary_current",
        **common,
    )
    log_count = select_stage_b(
        records,
        structural_saturation_mode=str(candidate["candidate"]),
        **common,
    )
    binary_ids = {str(row["chunk_uid"]) for row in binary["selected"]}
    log_count_ids = {str(row["chunk_uid"]) for row in log_count["selected"]}
    selector_union = binary_ids.union(log_count_ids)
    training_cap = min(
        int(binary["selected_token_proxy"]),
        int(log_count["selected_token_proxy"]),
    )

    remaining = [row for row in binary["scored"] if str(row["chunk_uid"]) not in selector_union]
    random_rows = _take_to_materialized_cap(
        _stable_order(
            remaining,
            int(protocol["stage_a_random_baseline"]["seed"]),
            "redundancy_saturation_common_disjoint",
        ),
        training_cap,
    )
    random_ids = {str(row["chunk_uid"]) for row in random_rows}

    materialized = {
        "binary_current_equal_budget": [
            _arm_record(row, "binary_current_equal_budget", "stageA_pass_binary_current")
            for row in binary["selected"]
        ],
        "log_count_equal_budget": [
            _arm_record(row, "log_count_equal_budget", "stageA_pass_log_count")
            for row in log_count["selected"]
        ],
        "stageA_random_common_disjoint_equal_budget": [
            _arm_record(
                {
                    **row,
                    "stage_b_baseline": {
                        "arm": "stageA_random_common_disjoint_equal_budget",
                        "seed": int(protocol["stage_a_random_baseline"]["seed"]),
                        "disjoint_from": [
                            "binary_current_equal_budget",
                            "log_count_equal_budget",
                        ],
                    },
                },
                "stageA_random_common_disjoint_equal_budget",
                "stageA_pass_complement_of_selector_union",
            )
            for row in random_rows
        ],
    }
    for rows in materialized.values():
        for row in rows:
            row["training_cap_contract"]["target_token_proxy_cap"] = training_cap

    blockers = []
    if random_ids.intersection(selector_union):
        blockers.append("common_random_overlaps_selector_union")
    for name, rows in materialized.items():
        if sum(int(row["token_proxy_count"]) for row in rows) < training_cap:
            blockers.append(f"arm_does_not_cover_training_cap:{name}")
    if binary_ids == log_count_ids:
        blockers.append("candidate_selector_identical_to_binary_control")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in materialized.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)

    report = {
        "schema_version": "redundancy-saturation-proxy-arms-freeze-v1",
        "status": (
            "redundancy_saturation_proxy_arms_frozen"
            if not blockers
            else "redundancy_saturation_proxy_arms_frozen_with_blockers"
        ),
        "source_sha256": {
            str(stage_a_path): sha256_file(stage_a_path),
            str(protocol_path): sha256_file(protocol_path),
            str(candidate_path): sha256_file(candidate_path),
        },
        "input": {
            "stage_a_path": str(stage_a_path),
            "stage_a_record_count": len(records),
            "stage_a_token_proxy": sum(_token(row) for row in binary["scored"]),
        },
        "training_budget": {
            "basis": "shared token-proxy cap before proxy model/tokenizer freeze",
            "cap": training_cap,
            "target_tokenizer_exact_packing_deferred": True,
            "packing_rule_next_step": (
                "After proxy model/tokenizer freeze, pack each arm in listed order and "
                "consume exactly the same tokenizer-token count, truncating the final record if required."
            ),
        },
        "arms": {
            name: _summarize(rows, training_cap)
            for name, rows in materialized.items()
        },
        "selection_relationship": {
            "binary_selected_count": len(binary_ids),
            "log_count_selected_count": len(log_count_ids),
            "selector_union_count": len(selector_union),
            "selector_intersection_count": len(binary_ids.intersection(log_count_ids)),
            "binary_only_count": len(binary_ids - log_count_ids),
            "log_count_only_count": len(log_count_ids - binary_ids),
            "selector_jaccard": round(
                len(binary_ids.intersection(log_count_ids)) / max(1, len(selector_union)),
                6,
            ),
        },
        "disjointness": {
            "common_random_binary_overlap_count": len(random_ids.intersection(binary_ids)),
            "common_random_log_count_overlap_count": len(random_ids.intersection(log_count_ids)),
            "common_random_selector_union_overlap_count": len(random_ids.intersection(selector_union)),
            "common_random_disjoint_from_both_selectors": not bool(
                random_ids.intersection(selector_union)
            ),
        },
        "blockers": blockers,
        "selection_forbids": candidate["forbidden_inputs"],
        "utility_scope": "Stage C only; not used to construct these arms",
        "claim_boundary": (
            "Data-arm freeze only. Token-proxy equality does not replace exact tokenizer-token "
            "packing, training, Utility, retention, or target-model evidence."
        ),
    }
    save_json(output_dir / "proxy_arms_freeze_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Redundancy saturation proxy arms.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = freeze(args.stage_a, args.protocol, args.candidate, args.output_dir)
    print(
        {
            "status": report["status"],
            "training_budget": report["training_budget"],
            "arms": report["arms"],
            "disjointness": report["disjointness"],
            "blockers": report["blockers"],
        }
    )
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
