#!/usr/bin/env python3
"""Build independently ordered, score-hidden Stage-B review packets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_MASTER = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review" / "blind_review_packet.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "validation" / "temporal_code_stage_b_multi_review"
REVIEWERS = ("reviewer_a", "reviewer_b")


def _order_key(review_id: str, reviewer: str) -> str:
    return hashlib.sha256(f"temporal-code-stage-b-v1:{reviewer}:{review_id}".encode("utf-8")).hexdigest()


def _blank_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "review_id": row["review_id"],
        "content_type": row["content_type"],
        "change_type": row["change_type"],
        "chunk_kind": row["chunk_kind"],
        "text": row["text"],
        "review_fields": {
            "quality_label": None,
            "redundancy_label": None,
            "confidence": None,
            "notes": None,
        },
    }


def _packet(master: Dict[str, Any], reviewer: str) -> Dict[str, Any]:
    records = [_blank_record(row) for row in master["records"]]
    records.sort(key=lambda row: _order_key(row["review_id"], reviewer))
    return {
        "schema_version": "temporal-code-stage-b-independent-reviewer-packet-v1",
        "status": "awaiting_independent_review",
        "reviewer_id": reviewer,
        "review_contract": {
            "quality_labels": ["preserve", "neutral", "downrank"],
            "redundancy_labels": ["unique", "related", "saturated"],
            "confidence_labels": ["low", "medium", "high"],
            "scores_arms_repositories_paths_and_strata_hidden": True,
            "reviewers_must_not_discuss_labels_before_both_packets_are_frozen": True,
            "reviewer_must_not_open_master_key": True,
        },
        "summary": {"review_record_count": len(records)},
        "records": records,
        "claim_boundary": "Independent blind labels only; no proxy-validity, Utility, or release claim.",
    }


def _adjudication_packet(master: Dict[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for row in sorted(master["records"], key=lambda item: _order_key(item["review_id"], "adjudicator")):
        record = _blank_record(row)
        record["adjudication_fields"] = record.pop("review_fields")
        records.append(record)
    return {
        "schema_version": "temporal-code-stage-b-blind-adjudication-packet-v1",
        "status": "inactive_until_two_independent_reviews_are_frozen",
        "review_contract": {
            "only_disagreement_records_will_be_used": True,
            "adjudicator_must_not_open_master_key": True,
            "scores_arms_repositories_paths_and_strata_hidden": True,
        },
        "summary": {"candidate_record_count": len(records)},
        "records": records,
        "claim_boundary": "Blank adjudication packet only; no proxy-validity, Utility, or release claim.",
    }


def build(master: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = {reviewer: _packet(master, reviewer) for reviewer in REVIEWERS}
    for reviewer, packet in packets.items():
        save_json(output_dir / f"{reviewer}_packet.json", packet)
    save_json(output_dir / "adjudication_packet.json", _adjudication_packet(master))
    manifest = {
        "schema_version": "temporal-code-stage-b-multi-review-manifest-v1",
        "status": "awaiting_two_independent_reviews",
        "reviewer_count": len(REVIEWERS),
        "review_record_count": len(master["records"]),
        "required_completed_labels": len(REVIEWERS) * len(master["records"]),
        "reviewers": list(REVIEWERS),
        "adjudication_rule": "Use adjudication labels only where the two independent reviewers disagree.",
        "proxy_analysis_gate": "Do not inspect the hidden key or compute proxy alignment until both reviews and required adjudication labels are frozen.",
        "claim_boundary": "Review protocol only; no proxy-validity, Utility, or release claim.",
    }
    save_json(output_dir / "multi_review_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal-code Stage-B multi-review packets.")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build(load_json(args.master), args.output_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
