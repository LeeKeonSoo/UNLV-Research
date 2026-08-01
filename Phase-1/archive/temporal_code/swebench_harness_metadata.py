#!/usr/bin/env python3
"""Acquire outcome-free SWE-bench harness metadata without retaining task content."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_change import normalize_repository_identity


DEFAULT_CONTRACT = Path("configs") / "temporal_code_swebench_metadata_acquisition_v1.json"
DEFAULT_HARNESS_CONTRACT = Path("configs") / "temporal_code_executable_task_harness_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "swebench_harness_metadata_profile.json"
DATASET_SERVER = "https://datasets-server.huggingface.co"


def _rows(dataset: str, config: str, split: str) -> Iterable[Dict[str, Any]]:
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100}
        )
        with urllib.request.urlopen(f"{DATASET_SERVER}/rows?{query}", timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                yield row
        offset += len(rows)
        if not rows or offset >= int(payload.get("num_rows_total") or offset):
            break


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _bucket(repository_identity: str) -> int:
    digest = hashlib.sha256(repository_identity.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def _assign_split(repository_identity: str, created_at: datetime | None, rule: Dict[str, Any]) -> str:
    if created_at is None:
        return "excluded_missing_timestamp"
    bucket = _bucket(repository_identity)
    development_end = _parse_timestamp(rule["development_task_end_inclusive"])
    confirmatory_start = _parse_timestamp(rule["confirmatory_task_start_inclusive"])
    development_buckets = rule["development_repository_buckets"]
    confirmatory_buckets = rule["confirmatory_repository_buckets"]
    if (
        int(development_buckets[0]) <= bucket <= int(development_buckets[1])
        and development_end is not None
        and created_at <= development_end
    ):
        return "development"
    if (
        int(confirmatory_buckets[0]) <= bucket <= int(confirmatory_buckets[1])
        and confirmatory_start is not None
        and created_at >= confirmatory_start
    ):
        return "confirmatory"
    return "excluded_split_or_time_rule"


def _required_task_count(harness: Dict[str, Any]) -> Dict[str, Any]:
    rule = harness["sample_size_rule"]
    z_one_sided_95 = 1.6448536269514722
    half_width = float(rule["desired_task_distribution_half_width"])
    variance_bound = float(rule["conservative_variance_bound_for_paired_difference"])
    required = int(math.ceil((z_one_sided_95**2) * variance_bound / (half_width**2)))
    return {
        "z_one_sided_95": z_one_sided_95,
        "desired_half_width": half_width,
        "paired_difference_variance_bound": variance_bound,
        "required_task_count": required,
        "practical_effect_margin_absolute": float(rule["practical_effect_margin_absolute"]),
        "training_seed_count": int(rule["training_seed_count"]),
    }


def acquire(contract_path: Path, harness_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    harness = load_json(harness_path)
    source = contract["source"]
    split_rule = contract["split_rule"]
    records = []
    source_fields = set()
    for row in _rows(source["dataset"], source["config"], source["split"]):
        source_fields.update(str(key) for key in row)
        identity = normalize_repository_identity(str(row.get("repo") or ""))
        created_at = _parse_timestamp(row.get("created_at"))
        if not identity:
            assigned_split = "excluded_missing_repository_identity"
            bucket = None
        else:
            assigned_split = _assign_split(identity, created_at, split_rule)
            bucket = _bucket(identity)
        records.append(
            {
                "instance_id": str(row.get("instance_id") or ""),
                "repository_identity": identity,
                "base_commit": str(row.get("base_commit") or "").lower(),
                "created_at": created_at.isoformat().replace("+00:00", "Z") if created_at else None,
                "version": str(row.get("version") or ""),
                "assigned_split": assigned_split,
                "split_bucket": bucket,
            }
        )
    forbidden = set(contract["forbidden_persisted_fields"])
    persisted = {key for row in records for key in row}
    if forbidden & persisted:
        raise ValueError(f"Forbidden task fields would be persisted: {sorted(forbidden & persisted)}")
    split_counts = Counter(row["assigned_split"] for row in records)
    repository_sets = {
        split: {row["repository_identity"] for row in records if row["assigned_split"] == split}
        for split in ("development", "confirmatory")
    }
    precision = _required_task_count(harness)
    eligible_count = int(split_counts["development"] + split_counts["confirmatory"])
    profile = {
        "schema_version": "temporal-code-swebench-harness-metadata-profile-v1",
        "status": "outcome_free_source_profile_complete",
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(harness_path): sha256_file(harness_path),
        },
        "source_summary": {
            "row_count": len(records),
            "source_fields_observed_not_persisted": sorted(source_fields),
            "raw_task_content_persisted": False,
            "model_outcomes_read": False,
        },
        "split_summary": {
            "counts": dict(sorted(split_counts.items())),
            "development_repository_count": len(repository_sets["development"]),
            "confirmatory_repository_count": len(repository_sets["confirmatory"]),
            "repository_overlap_count": len(repository_sets["development"] & repository_sets["confirmatory"]),
            "eligible_task_count": eligible_count,
        },
        "precision_analysis": {
            **precision,
            "eligible_task_count": eligible_count,
            "eligible_count_meets_required_task_count": eligible_count >= precision["required_task_count"],
        },
        "e2_analysis": {
            "candidate_task_count": eligible_count,
            "e2_verified_task_count": 0,
            "verified_membership_is_not_e2": True,
            "required_next_step": "run the frozen task-class-specific repository-patch execution harness",
        },
        "records": records,
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire outcome-free SWE-bench harness metadata.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--harness-contract", type=Path, default=DEFAULT_HARNESS_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    profile = acquire(args.contract, args.harness_contract, args.output)
    print(
        {
            "status": profile["status"],
            "split_summary": profile["split_summary"],
            "precision_analysis": profile["precision_analysis"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
