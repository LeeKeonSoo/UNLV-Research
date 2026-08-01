#!/usr/bin/env python3
"""Enter, freeze, and adjudicate temporal-code Stage-B blind reviews."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_DIR = OUTPUT_DIR / "validation" / "temporal_code_stage_b_multi_review"
QUALITY = {"preserve", "neutral", "downrank"}
REDUNDANCY = {"unique", "related", "saturated"}
CONFIDENCE = {"low", "medium", "high"}


def _fields(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("review_fields")
    if isinstance(value, dict):
        return value
    value = row.get("adjudication_fields")
    if isinstance(value, dict):
        return value
    raise ValueError(f"Record {row.get('review_id')} has no label fields.")


def _is_complete(fields: Dict[str, Any]) -> bool:
    return (
        fields.get("quality_label") in QUALITY
        and fields.get("redundancy_label") in REDUNDANCY
        and fields.get("confidence") in CONFIDENCE
    )


def status(packet: Dict[str, Any]) -> Dict[str, Any]:
    records = packet.get("records") if isinstance(packet.get("records"), list) else []
    complete = [row for row in records if _is_complete(_fields(row))]
    return {
        "status": packet.get("status"),
        "record_count": len(records),
        "completed_count": len(complete),
        "incomplete_count": len(records) - len(complete),
        "next_incomplete_review_id": next((row.get("review_id") for row in records if not _is_complete(_fields(row))), None),
    }


def show_record(packet: Dict[str, Any], review_id: str | None = None) -> Dict[str, Any]:
    records = packet.get("records") if isinstance(packet.get("records"), list) else []
    row = (
        next((item for item in records if item.get("review_id") == review_id), None)
        if review_id
        else next((item for item in records if not _is_complete(_fields(item))), None)
    )
    if row is None:
        return {"status": packet.get("status"), "message": "No matching incomplete record."}
    return {
        "review_id": row.get("review_id"),
        "content_type": row.get("content_type"),
        "change_type": row.get("change_type"),
        "chunk_kind": row.get("chunk_kind"),
        "text": row.get("text"),
        "current_fields": _fields(row),
    }


def set_label(
    packet_path: Path,
    review_id: str,
    quality: str,
    redundancy: str,
    confidence: str,
    notes: str | None,
) -> Dict[str, Any]:
    if quality not in QUALITY or redundancy not in REDUNDANCY or confidence not in CONFIDENCE:
        raise ValueError("Invalid quality, redundancy, or confidence label.")
    packet = load_json(packet_path)
    if str(packet.get("status", "")).startswith("frozen"):
        raise RuntimeError("Frozen review packets cannot be edited.")
    row = next((item for item in packet.get("records") or [] if item.get("review_id") == review_id), None)
    if row is None:
        raise KeyError(f"Unknown review_id: {review_id}")
    _fields(row).update(
        {
            "quality_label": quality,
            "redundancy_label": redundancy,
            "confidence": confidence,
            "notes": notes,
        }
    )
    save_json(packet_path, packet)
    return status(packet)


def freeze_review(
    packet_path: Path,
    reviewer_attestation: str,
    *,
    attest_independent: bool,
    attest_no_key: bool,
) -> Dict[str, Any]:
    packet = load_json(packet_path)
    current = status(packet)
    if current["incomplete_count"]:
        raise RuntimeError(f"Cannot freeze incomplete review: {current['incomplete_count']} labels remain.")
    if not attest_independent or not attest_no_key:
        raise RuntimeError("Both independent-review and no-hidden-key attestations are required.")
    packet["status"] = "frozen_independent_review"
    packet["freeze_attestation"] = {
        "reviewer_attestation": reviewer_attestation,
        "independent_without_discussion": True,
        "hidden_key_not_opened": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(packet_path, packet)
    manifest = {
        "schema_version": "temporal-code-stage-b-review-freeze-v1",
        "reviewer_id": packet.get("reviewer_id"),
        "packet_path": str(packet_path),
        "packet_sha256": sha256_file(packet_path),
        "completed_count": current["completed_count"],
        "freeze_attestation": packet["freeze_attestation"],
        "claim_boundary": "Frozen independent blind labels only; hidden-key analysis remains gated.",
    }
    save_json(packet_path.with_name(f"{packet_path.stem}_freeze.json"), manifest)
    return manifest


def _label_pair(row: Dict[str, Any]) -> tuple[Any, Any]:
    fields = _fields(row)
    return fields.get("quality_label"), fields.get("redundancy_label")


def activate_adjudication(reviewer_a_path: Path, reviewer_b_path: Path, template_path: Path) -> Dict[str, Any]:
    reviewer_a, reviewer_b, template = load_json(reviewer_a_path), load_json(reviewer_b_path), load_json(template_path)
    if reviewer_a.get("status") != "frozen_independent_review" or reviewer_b.get("status") != "frozen_independent_review":
        raise RuntimeError("Both independent reviews must be frozen before adjudication.")
    a_by_id = {row["review_id"]: row for row in reviewer_a["records"]}
    b_by_id = {row["review_id"]: row for row in reviewer_b["records"]}
    if set(a_by_id) != set(b_by_id):
        raise RuntimeError("Reviewer packets contain different review IDs.")
    disagreements = {rid for rid in a_by_id if _label_pair(a_by_id[rid]) != _label_pair(b_by_id[rid])}
    template["records"] = [row for row in template["records"] if row["review_id"] in disagreements]
    template["status"] = "awaiting_blind_disagreement_adjudication" if disagreements else "frozen_no_disagreements"
    template["summary"] = {
        "independent_review_record_count": len(a_by_id),
        "disagreement_record_count": len(disagreements),
        "completed_adjudication_count": 0,
    }
    save_json(template_path, template)
    return status(template)


def freeze_adjudication(packet_path: Path, adjudicator_attestation: str, *, attest_no_key: bool) -> Dict[str, Any]:
    packet = load_json(packet_path)
    current = status(packet)
    if current["incomplete_count"]:
        raise RuntimeError(f"Cannot freeze incomplete adjudication: {current['incomplete_count']} labels remain.")
    if not attest_no_key:
        raise RuntimeError("The no-hidden-key attestation is required.")
    packet["status"] = "frozen_blind_disagreement_adjudication"
    packet["freeze_attestation"] = {
        "adjudicator_attestation": adjudicator_attestation,
        "hidden_key_not_opened": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(packet_path, packet)
    return {
        "packet_path": str(packet_path),
        "packet_sha256": sha256_file(packet_path),
        "completed_count": current["completed_count"],
        "status": packet["status"],
    }


def _packet_path(name: str) -> Path:
    paths = {
        "reviewer_a": DEFAULT_DIR / "reviewer_a_packet.json",
        "reviewer_b": DEFAULT_DIR / "reviewer_b_packet.json",
        "adjudication": DEFAULT_DIR / "adjudication_packet.json",
    }
    return paths[name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage temporal-code Stage-B blind reviews.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--packet", choices=["reviewer_a", "reviewer_b", "adjudication"], required=True)
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--packet", choices=["reviewer_a", "reviewer_b", "adjudication"], required=True)
    show_parser.add_argument("--review-id")
    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--packet", choices=["reviewer_a", "reviewer_b", "adjudication"], required=True)
    label_parser.add_argument("--review-id", required=True)
    label_parser.add_argument("--quality", choices=sorted(QUALITY), required=True)
    label_parser.add_argument("--redundancy", choices=sorted(REDUNDANCY), required=True)
    label_parser.add_argument("--confidence", choices=sorted(CONFIDENCE), required=True)
    label_parser.add_argument("--notes")
    freeze_parser = subparsers.add_parser("freeze-review")
    freeze_parser.add_argument("--packet", choices=["reviewer_a", "reviewer_b"], required=True)
    freeze_parser.add_argument("--reviewer-attestation", required=True)
    freeze_parser.add_argument("--attest-independent", action="store_true")
    freeze_parser.add_argument("--attest-no-key", action="store_true")
    subparsers.add_parser("activate-adjudication")
    freeze_adj_parser = subparsers.add_parser("freeze-adjudication")
    freeze_adj_parser.add_argument("--adjudicator-attestation", required=True)
    freeze_adj_parser.add_argument("--attest-no-key", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        result = status(load_json(_packet_path(args.packet)))
    elif args.command == "show":
        result = show_record(load_json(_packet_path(args.packet)), args.review_id)
    elif args.command == "label":
        result = set_label(_packet_path(args.packet), args.review_id, args.quality, args.redundancy, args.confidence, args.notes)
    elif args.command == "freeze-review":
        result = freeze_review(
            _packet_path(args.packet),
            args.reviewer_attestation,
            attest_independent=args.attest_independent,
            attest_no_key=args.attest_no_key,
        )
    elif args.command == "activate-adjudication":
        result = activate_adjudication(_packet_path("reviewer_a"), _packet_path("reviewer_b"), _packet_path("adjudication"))
    else:
        result = freeze_adjudication(
            _packet_path("adjudication"),
            args.adjudicator_attestation,
            attest_no_key=args.attest_no_key,
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
