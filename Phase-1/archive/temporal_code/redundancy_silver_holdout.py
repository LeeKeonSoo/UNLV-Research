#!/usr/bin/env python3
"""Freeze repository-disjoint source chunks for Redundancy silver holdout."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import token_proxy_count


DEFAULT_STAGE_A = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_a_code_domain_v2_combined"
    / "train"
    / "stage_a_pass.jsonl"
)
DEFAULT_CALIBRATION = OUTPUT_DIR / "validation" / "redundancy_real_corpus_calibration_report.json"
DEFAULT_OUTPUT = Path("configs") / "temporal_code_redundancy_silver_holdout_v1.json"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _length_bucket(text: str) -> str:
    count = token_proxy_count(text)
    if count < 80:
        return "small"
    if count < 240:
        return "medium"
    return "large"


def build(stage_a_path: Path, calibration_path: Path, output_path: Path, *, per_stratum: int) -> Dict[str, Any]:
    rows = list(_jsonl(stage_a_path))
    calibration = load_json(calibration_path)
    used_repositories = set()
    by_uid = {str(row["chunk_uid"]): row for row in rows}
    for uid in calibration["source_metadata"]["source_chunk_uids"]:
        row = by_uid.get(str(uid))
        if row is not None:
            used_repositories.add(str(row.get("repository_identity") or ""))

    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        repository = str(row.get("repository_identity") or "")
        if not repository or repository in used_repositories:
            continue
        groups[
            (
                str(row.get("content_type") or "unknown"),
                _length_bucket(str(row.get("text") or "")),
            )
        ].append(row)

    selected = []
    holdout_repositories: set[str] = set()
    for stratum in sorted(groups):
        count = 0
        for row in sorted(groups[stratum], key=lambda value: str(value["chunk_uid"])):
            repository = str(row.get("repository_identity") or "")
            if repository in holdout_repositories:
                continue
            selected.append(row)
            holdout_repositories.add(repository)
            count += 1
            if count >= per_stratum:
                break

    payload = {
        "schema_version": "temporal-code-redundancy-silver-holdout-v1",
        "status": "frozen_before_threshold_arm_holdout_evaluation",
        "stage_a_source": str(stage_a_path),
        "calibration_report": str(calibration_path),
        "calibration_repository_overlap": len(used_repositories.intersection(holdout_repositories)),
        "source_count": len(selected),
        "source_repository_count": len(holdout_repositories),
        "source_strata": dict(
            sorted(
                Counter(
                    f"{row.get('content_type')}::{_length_bucket(str(row.get('text') or ''))}"
                    for row in selected
                ).items()
            )
        ),
        "source_chunk_uids": [str(row["chunk_uid"]) for row in selected],
        "transformations": [
            "exact_copy",
            "format_only",
            "containment_extension",
            "semantic_change_control_when_available",
            "cross_repository_independent"
        ],
        "threshold_outcomes_read": False,
        "utility_consumed": False,
        "benchmark_outcomes_consumed": False
    }
    save_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Redundancy silver holdout sources.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()
    report = build(
        args.stage_a,
        args.calibration,
        args.output,
        per_stratum=max(1, args.per_stratum),
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
