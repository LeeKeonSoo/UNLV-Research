#!/usr/bin/env python3
"""Separate explicit structural evidence from historical proxy-only exclusions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any, Final, Literal, TypedDict

from build_core_rule_inventory import _evidence


JsonMap = dict[str, Any]
Family = Literal[
    "hard_gate_rejection",
    "explicit_license_comment_only_candidate",
    "explicit_generated_do_not_edit_candidate",
    "minified_shape_candidate",
    "partial_or_unparseable_chunk_diagnostic",
    "proxy_only_no_explicit_evidence",
]
FAMILIES: Final[tuple[Family, ...]] = (
    "hard_gate_rejection",
    "explicit_license_comment_only_candidate",
    "explicit_generated_do_not_edit_candidate",
    "minified_shape_candidate",
    "partial_or_unparseable_chunk_diagnostic",
    "proxy_only_no_explicit_evidence",
)
SCORE_FEATURES: Final[tuple[str, ...]] = (
    "length_support",
    "structural_richness",
    "lexical_or_identifier_diversity",
    "pass_through_assignment_ratio",
    "score",
)


class FamilyTotals(TypedDict):
    chunks: int
    token_proxy: int


def read_jsonl(path: Path) -> list[JsonMap]:
    """Read a JSONL artifact as records while preserving its recorded order."""
    rows: list[JsonMap] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def family_for(row: JsonMap) -> Family:
    """Classify a historical row without treating diagnostic evidence as removal authority."""
    hard_gate_reasons = row.get("stage_b_hard_gate_reasons")
    if isinstance(hard_gate_reasons, list) and hard_gate_reasons:
        return "hard_gate_rejection"
    metadata = row.get("stage_c_policy_metadata")
    policy_metadata = metadata if isinstance(metadata, dict) else {}
    evidence = _evidence(str(row.get("text") or ""), policy_metadata)
    if bool(evidence["strong_generated_marker_candidate"]):
        return "explicit_generated_do_not_edit_candidate"
    if bool(evidence["license_or_comment_only_candidate"]):
        return "explicit_license_comment_only_candidate"
    if bool(evidence["one_or_two_line_minified_candidate"]):
        return "minified_shape_candidate"
    if evidence["python_parse_status"] == "syntax_error":
        return "partial_or_unparseable_chunk_diagnostic"
    return "proxy_only_no_explicit_evidence"


def token_proxy(row: JsonMap) -> int:
    """Return the recorded historical token proxy, rejecting malformed artifacts."""
    value = row.get("token_proxy")
    if isinstance(value, int) and value >= 0:
        return value
    raise TypeError(f"Missing non-negative integer token_proxy for {row.get('chunk_uid')}")


def score_summary(rows: Iterable[JsonMap]) -> dict[str, float]:
    """Summarize recorded proxy features descriptively, never as a new objective."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        selection = row.get("stage_b_selection")
        if not isinstance(selection, dict):
            continue
        priority = selection.get("operational_priority")
        if not isinstance(priority, dict):
            continue
        for feature in SCORE_FEATURES:
            value = priority.get(feature)
            if isinstance(value, int | float):
                values[feature].append(float(value))
    return {feature: round(mean(feature_values), 6) for feature, feature_values in values.items() if feature_values}


def group_summary(rows: Iterable[JsonMap]) -> JsonMap:
    """Report exclusive structural families and recorded score features for one historical group."""
    materialized_rows = list(rows)
    chunks: Counter[Family] = Counter()
    tokens: Counter[Family] = Counter()
    for row in materialized_rows:
        family = family_for(row)
        chunks[family] += 1
        tokens[family] += token_proxy(row)
    families: dict[str, FamilyTotals] = {
        family: {"chunks": chunks[family], "token_proxy": tokens[family]} for family in FAMILIES
    }
    return {
        "chunks": len(materialized_rows),
        "token_proxy": sum(token_proxy(row) for row in materialized_rows),
        "families": families,
        "recorded_proxy_feature_means": score_summary(materialized_rows),
    }


def build_report(selected: Iterable[JsonMap], rejected: Iterable[JsonMap]) -> JsonMap:
    """Build a diagnostic decomposition that cannot authorize runtime removal."""
    return {
        "schema_version": "historical-selector-decomposition-v1",
        "status": "diagnostic_only_no_runtime_rule_authorized",
        "interpretation": [
            "Historical priority selection is a discovery signal, not a quality label.",
            "Only explicit candidates may advance to policy-card and fixture review.",
            "Partial or unparsable chunks are diagnostics because chunk boundaries can split valid source.",
            "The proxy-only family is evidence that further family discovery is required, not authority to delete it.",
        ],
        "groups": {
            "historical_selected": group_summary(selected),
            "historical_rejected": group_summary(rejected),
        },
    }


def main() -> int:
    """Write the historical selection decomposition report."""
    parser = argparse.ArgumentParser(description="Decompose historical priority selection into observable structural families.")
    parser.add_argument("--historical-selected", type=Path, required=True)
    parser.add_argument("--historical-rejected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(read_jsonl(args.historical_selected), read_jsonl(args.historical_rejected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "groups": report["groups"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
