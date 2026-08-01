#!/usr/bin/env python3
"""Freeze repository-disjoint source-role calibration data for a future selector."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]
REFERENCE_TIER = "known_high_quality_reference"
RAW_TIER = "raw_like"


def _tier(row: JsonMap) -> str:
    partition = row.get("partition")
    if not isinstance(partition, dict):
        return "unlabeled"
    value = partition.get("source_tier")
    return str(value) if isinstance(value, str) and value else "unlabeled"


def _repository(row: JsonMap) -> str | None:
    partition = row.get("partition")
    if not isinstance(partition, dict):
        return None
    value = partition.get("repository_identity")
    return str(value) if isinstance(value, str) and value else None


def _stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _calibration_row(row: JsonMap, label: str) -> JsonMap:
    return {
        "origin_record_id": str(row["record_id"]),
        "source_role_label": label,
        "repository_identity": _repository(row),
        "text": str(row["text"]),
    }


def build_calibration(
    rows: Iterable[JsonMap], *, held_out_repository_count: int, split_salt: str
) -> tuple[list[JsonMap], list[JsonMap], JsonMap]:
    """Build a source-role calibration split without treating its labels as Quality."""
    all_rows = list(rows)
    references = [row for row in all_rows if _tier(row) == REFERENCE_TIER and _repository(row) is not None]
    raw_candidates = [row for row in all_rows if _tier(row) == RAW_TIER]
    reference_repositories = sorted({_repository(row) for row in references if _repository(row) is not None}, key=lambda value: _stable_key(value, split_salt))
    if held_out_repository_count < 1 or held_out_repository_count >= len(reference_repositories):
        raise RuntimeError("held_out_repository_count must leave at least one reference repository for training.")
    held_out_repositories = set(reference_repositories[:held_out_repository_count])
    train = [row for row in references if _repository(row) not in held_out_repositories]
    positives = [row for row in references if _repository(row) in held_out_repositories]
    negatives = sorted(raw_candidates, key=lambda row: _stable_key(str(row.get("record_id") or ""), split_salt))[: len(positives)]
    if len(negatives) != len(positives):
        raise RuntimeError("Raw-like pool does not contain enough records for balanced calibration.")
    calibration = [
        *[_calibration_row(row, "reference_distribution_member") for row in positives],
        *[_calibration_row(row, "raw_like_nonmember") for row in negatives],
    ]
    train_repositories = sorted({_repository(row) for row in train if _repository(row) is not None})
    return train, calibration, {
        "schema_version": "reference-distribution-calibration-v1",
        "status": "source_role_calibration_frozen_pending_selector_implementation",
        "selection_hypothesis": "reference_distribution_membership_for_declared_code_scope",
        "label_boundary": "Labels describe declared source role, not intrinsic Quality, semantic necessity, or downstream Utility.",
        "reference_source_tier": REFERENCE_TIER,
        "candidate_source_tier": RAW_TIER,
        "split": {
            "unit": "repository_identity",
            "salt": split_salt,
            "held_out_repositories": sorted(held_out_repositories),
            "training_repositories": train_repositories,
        },
        "repository_overlap": sorted(held_out_repositories & set(train_repositories)),
        "summary": {
            "input_records": len(all_rows),
            "reference_training_records": len(train),
            "calibration_positive_records": len(positives),
            "calibration_negative_records": len(negatives),
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_token_fraction_read": False,
        },
    }


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze reference-distribution calibration artifacts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-train-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--held-out-repository-count", type=int, default=2)
    parser.add_argument("--split-salt", default="calibrated-selector-v1")
    args = parser.parse_args()
    train, calibration, report = build_calibration(
        _read_jsonl(args.input),
        held_out_repository_count=args.held_out_repository_count,
        split_salt=args.split_salt,
    )
    _write_jsonl(args.reference_train_output, train)
    _write_jsonl(args.calibration_output, calibration)
    report["artifacts"] = {
        "reference_train": {"path": str(args.reference_train_output), "sha256": _sha256(args.reference_train_output)},
        "held_out_calibration": {"path": str(args.calibration_output), "sha256": _sha256(args.calibration_output)},
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
