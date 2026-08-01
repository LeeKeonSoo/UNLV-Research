#!/usr/bin/env python3
"""Analyze frozen independent labels against hidden temporal-code Stage-B evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_PACKET = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review" / "blind_review_packet.json"
DEFAULT_KEY = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review" / "blind_review_key.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review_analysis.json"
QUALITY_ORDER = {"downrank": 0, "neutral": 1, "preserve": 2}
REDUNDANCY_ORDER = {"unique": 0, "related": 1, "saturated": 2}


def analyze(packet: Dict[str, Any], key: Dict[str, Any]) -> Dict[str, Any]:
    records = packet.get("records") if isinstance(packet.get("records"), list) else []
    key_by_id = {row["review_id"]: row for row in key.get("records") or []}
    incomplete = []
    for row in records:
        fields = row.get("review_fields") if isinstance(row.get("review_fields"), dict) else {}
        if (
            fields.get("quality_label") not in QUALITY_ORDER
            or fields.get("redundancy_label") not in REDUNDANCY_ORDER
            or fields.get("confidence") not in {"low", "medium", "high"}
        ):
            incomplete.append(row.get("review_id"))
    if incomplete:
        return {
            "schema_version": "temporal-code-stage-b-blind-review-analysis-v1",
            "status": "blocked_incomplete_independent_review",
            "summary": {
                "review_record_count": len(records),
                "completed_review_count": len(records) - len(incomplete),
                "incomplete_review_count": len(incomplete),
            },
            "incomplete_review_ids": incomplete,
            "proxy_promotion_allowed": False,
            "utility_scope": "Stage C validation only; never selector objective",
            "claim_boundary": "No real-corpus proxy-validity claim while independent review is incomplete.",
        }
    joined = [(row, key_by_id[row["review_id"]]) for row in records]
    repository_count = len({hidden.get("repository_identity") for _, hidden in joined})
    quality_aligned = sum(
        (QUALITY_ORDER[row["review_fields"]["quality_label"]] >= 1)
        == (float(hidden["stage_b_evidence"]["code_quality_proxy"]) >= 0.7)
        for row, hidden in joined
    )
    redundancy_aligned = sum(
        (REDUNDANCY_ORDER[row["review_fields"]["redundancy_label"]] >= 1)
        == (float(hidden["stage_b_evidence"]["soft_redundancy_risk"]) >= 0.25)
        for row, hidden in joined
    )
    return {
        "schema_version": "temporal-code-stage-b-blind-review-analysis-v1",
        "status": "independent_review_complete_initial_real_corpus_evidence",
        "summary": {
            "review_record_count": len(records),
            "completed_review_count": len(records),
            "quality_threshold_agreement": round(quality_aligned / max(1, len(records)), 6),
            "redundancy_threshold_agreement": round(redundancy_aligned / max(1, len(records)), 6),
            "quality_label_counts": dict(sorted(Counter(row["review_fields"]["quality_label"] for row in records).items())),
            "redundancy_label_counts": dict(sorted(Counter(row["review_fields"]["redundancy_label"] for row in records).items())),
            "reviewed_repository_count": repository_count,
        },
        "proxy_promotion_allowed": False,
        "promotion_blockers": [
            "limited_repository_scope" if repository_count < 3 else "independent_repository_replication_not_completed",
            "independent_review_replication_not_completed",
            "Stage-C Utility remains separate and unrun",
        ],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Initial blind real-corpus direction evidence only; no broad metric-validity, Utility, or release claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze temporal-code Stage-B blind review.")
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(load_json(args.packet), load_json(args.key))
    save_json(args.output, report)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
