#!/usr/bin/env python3
"""Explain what the retired priority proxy removed without authorizing a new rule."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any, TypedDict

from build_core_rule_inventory import _evidence


JsonMap = dict[str, Any]
FEATURES = (
    "length_support",
    "structural_richness",
    "lexical_or_identifier_diversity",
    "pass_through_assignment_ratio",
    "score",
)
CANDIDATES = (
    "strong_generated_marker_candidate",
    "one_or_two_line_minified_candidate",
    "pathological_line_repetition_candidate",
    "license_or_comment_only_candidate",
    "vendored_dependency_path_candidate",
    "generated_path_candidate",
    "minified_asset_path_candidate",
    "lockfile_path_candidate",
)


class GroupReport(TypedDict):
    chunks: int
    candidate_counts: dict[str, int]
    priority_feature_summary: dict[str, dict[str, float | int]]


def read_jsonl(path: Path) -> list[JsonMap]:
    """Load one curation JSONL artifact in its recorded order."""
    rows: list[JsonMap] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def proxy_rejected(rows: Iterable[JsonMap]) -> list[JsonMap]:
    """Return only historical rows rejected by the retired weighted threshold."""
    result: list[JsonMap] = []
    for row in rows:
        selection = row.get("stage_b_selection")
        if not isinstance(selection, dict):
            continue
        if selection.get("removed_reason") == "operational_priority_below_frozen_threshold":
            result.append(row)
    return result


def priority_values(rows: Iterable[JsonMap], feature: str) -> list[float]:
    """Collect one historical proxy feature when that feature was recorded."""
    values: list[float] = []
    for row in rows:
        selection = row.get("stage_b_selection")
        if not isinstance(selection, dict):
            continue
        priority = selection.get("operational_priority")
        if not isinstance(priority, dict):
            continue
        value = priority.get(feature)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def group_report(rows: list[JsonMap]) -> GroupReport:
    """Summarize explicit structural candidates and historical proxy features."""
    candidate_counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("stage_c_policy_metadata")
        policy_metadata = metadata if isinstance(metadata, dict) else {}
        evidence = _evidence(str(row.get("text") or ""), policy_metadata)
        for candidate in CANDIDATES:
            if evidence[candidate]:
                candidate_counts[candidate] += 1
    feature_summary: dict[str, dict[str, float | int]] = {}
    for feature in FEATURES:
        values = priority_values(rows, feature)
        if values:
            feature_summary[feature] = {
                "count": len(values),
                "mean": round(mean(values), 6),
                "minimum": round(min(values), 6),
                "maximum": round(max(values), 6),
            }
    return {
        "chunks": len(rows),
        "candidate_counts": dict(candidate_counts),
        "priority_feature_summary": feature_summary,
    }


def build_report(historical_pass: list[JsonMap], historical_rejected: list[JsonMap], current_retained: list[JsonMap]) -> JsonMap:
    """Build the bounded comparison that prevents proxy-output reinterpretation."""
    historical_proxy_rejected = proxy_rejected(historical_rejected)
    current_ids = {str(row.get("chunk_uid")) for row in current_retained}
    overlap = sum(str(row.get("chunk_uid")) in current_ids for row in historical_proxy_rejected)
    hard_gate_rejected = len(historical_rejected) - len(historical_proxy_rejected)
    return {
        "schema_version": "historical-proxy-forensics-v1",
        "status": "diagnostic_only_proxy_not_reactivated",
        "historical_groups": {
            "priority_accepted": group_report(historical_pass),
            "priority_rejected": group_report(historical_proxy_rejected),
            "hard_gate_rejected": hard_gate_rejected,
        },
        "current_v3": {"retained": group_report(current_retained)},
        "cross_version_overlap": {
            "historical_priority_rejected_chunks": len(historical_proxy_rejected),
            "also_retained_by_current_v3": overlap,
            "retained_share": round(overlap / len(historical_proxy_rejected), 6) if historical_proxy_rejected else 0.0,
        },
        "interpretation_boundary": [
            "Historical priority score features are descriptive evidence only and do not authorize removal.",
            "A candidate becomes active only after an explicit reason code, required metadata, false-positive fixture, Case Matrix scenario, and coverage-impact test are added.",
            "Candidate prevalence in a proxy-rejected group does not establish that every matching chunk is unnecessary or harmful for language-model training.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build forensic evidence for the retired weighted priority proxy.")
    parser.add_argument("--historical-pass", type=Path, required=True)
    parser.add_argument("--historical-rejected", type=Path, required=True)
    parser.add_argument("--current-retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        read_jsonl(args.historical_pass),
        read_jsonl(args.historical_rejected),
        read_jsonl(args.current_retained),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "overlap": report["cross_version_overlap"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
