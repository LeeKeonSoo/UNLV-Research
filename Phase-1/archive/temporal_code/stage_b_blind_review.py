#!/usr/bin/env python3
"""Build a score-hidden real-corpus review packet for temporal-code Stage-B proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, save_json


DEFAULT_SCORED = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke" / "train_scored.jsonl"
DEFAULT_SELECTED = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke" / "train_selected.jsonl"
DEFAULT_BASELINE = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke" / "train_stage_a_random_disjoint.jsonl"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review"
SEED = 42


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _stable_key(uid: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{uid}".encode("utf-8")).hexdigest()


def _choose(rows: List[Dict[str, Any]], count: int, salt: str, used: set[str]) -> List[Dict[str, Any]]:
    chosen = []
    for row in sorted(rows, key=lambda item: _stable_key(str(item["chunk_uid"]), salt)):
        if row["chunk_uid"] in used:
            continue
        chosen.append(row)
        used.add(row["chunk_uid"])
        if len(chosen) >= count:
            break
    return chosen


def _review_id(uid: str) -> str:
    return f"review-{hashlib.sha256(f'{SEED}:{uid}'.encode('utf-8')).hexdigest()[:12]}"


def build(scored_path: Path, selected_path: Path, baseline_path: Path, output_dir: Path) -> Dict[str, Any]:
    scored = list(_jsonl(scored_path))
    selected_ids = {row["chunk_uid"] for row in _jsonl(selected_path)}
    baseline_ids = {row["chunk_uid"] for row in _jsonl(baseline_path)}
    ordered = sorted(scored, key=lambda row: float(row["stage_b_evidence"]["stage_b_objective_score"]))
    used: set[str] = set()
    sampled: List[tuple[str, Dict[str, Any]]] = []

    def add(stratum: str, rows: List[Dict[str, Any]], count: int) -> None:
        sampled.extend((stratum, row) for row in _choose(rows, count, stratum, used))

    add("objective_low", ordered[: max(1, len(ordered) // 5)], 8)
    add("objective_high", ordered[-max(1, len(ordered) // 5) :], 8)
    add("high_soft_redundancy", sorted(scored, key=lambda row: -float(row["stage_b_evidence"]["soft_redundancy_risk"])), 8)
    add("pass_through_present", [row for row in scored if float(row["stage_b_evidence"]["pass_through_assignment_ratio"]) > 0], 5)
    for content_type in ("code", "test", "documentation"):
        add(f"content_type_{content_type}", [row for row in scored if row["content_type"] == content_type], 5)
    for repository_identity in sorted({str(row["repository_identity"]) for row in scored}):
        add(
            f"repository_{hashlib.sha256(repository_identity.encode('utf-8')).hexdigest()[:8]}",
            [row for row in scored if row["repository_identity"] == repository_identity],
            5,
        )
    add("selected_arm_hidden", [row for row in scored if row["chunk_uid"] in selected_ids], 8)
    add("baseline_arm_hidden", [row for row in scored if row["chunk_uid"] in baseline_ids], 8)

    packet_records = []
    key_records = []
    for stratum, row in sampled:
        review_id = _review_id(str(row["chunk_uid"]))
        packet_records.append(
            {
                "review_id": review_id,
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
        )
        key_records.append(
            {
                "review_id": review_id,
                "sampling_stratum": stratum,
                "chunk_uid": row["chunk_uid"],
                "repository_identity": row["repository_identity"],
                "bundle_id": row["bundle_id"],
                "path": row["path"],
                "arm": "selected" if row["chunk_uid"] in selected_ids else ("stage_a_random_disjoint" if row["chunk_uid"] in baseline_ids else "neither"),
                "stage_b_evidence": row["stage_b_evidence"],
            }
        )
    packet = {
        "schema_version": "temporal-code-stage-b-blind-review-packet-v1",
        "status": "awaiting_independent_review",
        "review_contract": {
            "quality_labels": ["preserve", "neutral", "downrank"],
            "redundancy_labels": ["unique", "related", "saturated"],
            "confidence_labels": ["low", "medium", "high"],
            "minimum_completed_reviews_before_analysis": len(packet_records),
            "scores_and_selection_arms_hidden": True,
            "sampling_strata_hidden": True,
            "reviewer_must_not_open_key_before_labels_are_frozen": True,
        },
        "summary": {
            "source_record_count": len(scored),
            "review_record_count": len(packet_records),
            "stratum_counts_hidden_from_reviewer": dict(sorted(Counter(stratum for stratum, _ in sampled).items())),
            "content_type_counts": dict(sorted(Counter(row["content_type"] for row in packet_records).items())),
        },
        "records": packet_records,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Blind review packet only; no real-corpus proxy-validity claim until labels are independently frozen and analyzed.",
    }
    key = {
        "schema_version": "temporal-code-stage-b-blind-review-key-v1",
        "warning": "Do not inspect before independent labels are frozen.",
        "records": key_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "blind_review_packet.json", packet)
    save_json(output_dir / "blind_review_key.json", key)
    lines = [
        "# Temporal-Code Stage-B Blind Review",
        "",
        "Status: `awaiting_independent_review`",
        "",
        "Label each record without inspecting the separate key or Stage-B scores.",
        "",
        "- Quality: `preserve`, `neutral`, or `downrank`",
        "- Redundancy: `unique`, `related`, or `saturated`",
        "- Confidence: `low`, `medium`, or `high`",
        "",
    ]
    for row in packet_records:
        lines.extend(
            [
                f"## {row['review_id']}",
                "",
                f"Content type: `{row['content_type']}`  ",
                f"Change type: `{row['change_type']}`  ",
                f"Chunk kind: `{row['chunk_kind']}`",
                "",
                "```text",
                row["text"],
                "```",
                "",
                "Quality label:  ",
                "Redundancy label:  ",
                "Confidence:  ",
                "Notes:",
                "",
            ]
        )
    (output_dir / "blind_review_packet.md").write_text("\n".join(lines), encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal-code Stage-B blind review packet.")
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build(args.scored, args.selected, args.baseline, args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
