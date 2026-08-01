#!/usr/bin/env python3
"""Analyze two independent blind reviews with disagreement adjudication."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_DIR = OUTPUT_DIR / "validation" / "temporal_code_stage_b_multi_review"
DEFAULT_KEY = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review" / "blind_review_key.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_stage_b_multi_review_analysis.json"
QUALITY_ORDER = {"downrank": 0, "neutral": 1, "preserve": 2}
REDUNDANCY_ORDER = {"unique": 0, "related": 1, "saturated": 2}
CONFIDENCE = {"low", "medium", "high"}


def _by_id(packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["review_id"]: row for row in packet.get("records") or []}


def _complete(fields: Dict[str, Any]) -> bool:
    return (
        fields.get("quality_label") in QUALITY_ORDER
        and fields.get("redundancy_label") in REDUNDANCY_ORDER
        and fields.get("confidence") in CONFIDENCE
    )


def _kappa(left: Iterable[str], right: Iterable[str]) -> float:
    left_values, right_values = list(left), list(right)
    total = len(left_values)
    if not total:
        return 0.0
    observed = sum(a == b for a, b in zip(left_values, right_values)) / total
    left_counts, right_counts = Counter(left_values), Counter(right_values)
    expected = sum((left_counts[label] / total) * (right_counts[label] / total) for label in set(left_counts) | set(right_counts))
    return 1.0 if expected == 1.0 and observed == 1.0 else round((observed - expected) / max(1e-12, 1.0 - expected), 6)


def analyze(
    reviewer_a: Dict[str, Any],
    reviewer_b: Dict[str, Any],
    adjudication: Dict[str, Any],
    key: Dict[str, Any],
) -> Dict[str, Any]:
    a, b, adj = _by_id(reviewer_a), _by_id(reviewer_b), _by_id(adjudication)
    ids = sorted(set(a) | set(b))
    if set(a) != set(b):
        return _blocked("blocked_reviewer_record_mismatch", ids, [], [])
    incomplete_a = [rid for rid in ids if not _complete(a[rid].get("review_fields") or {})]
    incomplete_b = [rid for rid in ids if not _complete(b[rid].get("review_fields") or {})]
    if incomplete_a or incomplete_b:
        return _blocked("blocked_incomplete_independent_reviews", ids, incomplete_a, incomplete_b)

    disagreements = [
        rid
        for rid in ids
        if (
            a[rid]["review_fields"]["quality_label"] != b[rid]["review_fields"]["quality_label"]
            or a[rid]["review_fields"]["redundancy_label"] != b[rid]["review_fields"]["redundancy_label"]
        )
    ]
    incomplete_adj = [
        rid for rid in disagreements if rid not in adj or not _complete(adj[rid].get("adjudication_fields") or {})
    ]
    agreement = {
        "quality_exact_agreement": round(
            sum(a[rid]["review_fields"]["quality_label"] == b[rid]["review_fields"]["quality_label"] for rid in ids) / max(1, len(ids)), 6
        ),
        "redundancy_exact_agreement": round(
            sum(a[rid]["review_fields"]["redundancy_label"] == b[rid]["review_fields"]["redundancy_label"] for rid in ids) / max(1, len(ids)), 6
        ),
        "quality_cohen_kappa": _kappa(
            (a[rid]["review_fields"]["quality_label"] for rid in ids),
            (b[rid]["review_fields"]["quality_label"] for rid in ids),
        ),
        "redundancy_cohen_kappa": _kappa(
            (a[rid]["review_fields"]["redundancy_label"] for rid in ids),
            (b[rid]["review_fields"]["redundancy_label"] for rid in ids),
        ),
        "disagreement_record_count": len(disagreements),
    }
    if incomplete_adj:
        report = _blocked("blocked_pending_disagreement_adjudication", ids, [], [])
        report["agreement"] = agreement
        report["disagreement_review_ids"] = disagreements
        report["incomplete_adjudication_ids"] = incomplete_adj
        return report

    key_by_id = _by_id(key)
    consensus = {}
    for rid in ids:
        if rid in disagreements:
            consensus[rid] = adj[rid]["adjudication_fields"]
        else:
            consensus[rid] = a[rid]["review_fields"]
    quality_aligned = sum(
        (QUALITY_ORDER[consensus[rid]["quality_label"]] >= 1)
        == (float(key_by_id[rid]["stage_b_evidence"]["code_quality_proxy"]) >= 0.7)
        for rid in ids
    )
    redundancy_aligned = sum(
        (REDUNDANCY_ORDER[consensus[rid]["redundancy_label"]] >= 1)
        == (float(key_by_id[rid]["stage_b_evidence"]["soft_redundancy_risk"]) >= 0.25)
        for rid in ids
    )
    return {
        "schema_version": "temporal-code-stage-b-multi-review-analysis-v1",
        "status": "multi_review_complete_initial_real_corpus_evidence",
        "summary": {
            "review_record_count": len(ids),
            "completed_independent_label_count": len(ids) * 2,
            "adjudicated_record_count": len(disagreements),
            "quality_threshold_agreement": round(quality_aligned / max(1, len(ids)), 6),
            "redundancy_threshold_agreement": round(redundancy_aligned / max(1, len(ids)), 6),
            "reviewed_repository_count": len({key_by_id[rid]["repository_identity"] for rid in ids}),
        },
        "inter_reviewer_agreement": agreement,
        "proxy_promotion_allowed": False,
        "promotion_blockers": ["independent_repository_replication_not_completed", "Stage-C Utility remains separate and unrun"],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Initial multi-review real-corpus evidence only; no broad metric-validity, Utility, or release claim.",
    }


def _blocked(status: str, ids: list[str], incomplete_a: list[str], incomplete_b: list[str]) -> Dict[str, Any]:
    return {
        "schema_version": "temporal-code-stage-b-multi-review-analysis-v1",
        "status": status,
        "summary": {
            "review_record_count": len(ids),
            "reviewer_a_completed_count": len(ids) - len(incomplete_a),
            "reviewer_b_completed_count": len(ids) - len(incomplete_b),
        },
        "reviewer_a_incomplete_ids": incomplete_a,
        "reviewer_b_incomplete_ids": incomplete_b,
        "proxy_promotion_allowed": False,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "No proxy-validity claim until two independent reviews and required adjudication are frozen.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze temporal-code Stage-B multi-review labels.")
    parser.add_argument("--reviewer-a", type=Path, default=DEFAULT_DIR / "reviewer_a_packet.json")
    parser.add_argument("--reviewer-b", type=Path, default=DEFAULT_DIR / "reviewer_b_packet.json")
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_DIR / "adjudication_packet.json")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(load_json(args.reviewer_a), load_json(args.reviewer_b), load_json(args.adjudication), load_json(args.key))
    save_json(args.output, report)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
