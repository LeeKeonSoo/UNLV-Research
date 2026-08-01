#!/usr/bin/env python3
"""Prepare review-only additional-repository chunks without granting training eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, save_json
from ingestion.code_change import bundle_training_payload
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks
from ingestion.code_selection import score_stage_b
from ingestion.normalize import process_candidate


DEFAULT_BUNDLES = OUTPUT_DIR / "temporal_code_collection" / "proxy_review_expansion_bundles" / "train"
DEFAULT_PRIMARY_STAGE_A = OUTPUT_DIR / "temporal_code_collection" / "stage_a_smoke" / "train" / "stage_a_pass.jsonl"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "proxy_review_expansion"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(bundle_dir: Path, primary_stage_a_path: Path, output_dir: Path) -> Dict[str, Any]:
    candidates = []
    for bundle_path in sorted(bundle_dir.glob("*.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        payload = bundle_training_payload(bundle)
        for index, item in enumerate(payload["training_payloads"]):
            candidates.append(
                process_candidate(
                    {
                        "id": f"review-only::{bundle['bundle_id']}::{index:03d}::{item['path']}",
                        "text": item["text"],
                        "provenance": {
                            "source_name": bundle["repository_identity"],
                            "source_uri": f"{bundle['provenance']['source_urls'][0]}#{item['path']}",
                            "collected_at": bundle["provenance"]["collected_at"],
                        },
                        "language": {"code": "python" if item["path"].lower().endswith(".py") else "en", "confidence": 1.0},
                        "rights": {"status": "allowed", "license": item["license"]},
                        "pii_context": "repository_code",
                        "partition": {
                            "split": "train",
                            "bundle_id": bundle["bundle_id"],
                            "repository_identity": bundle["repository_identity"],
                            "path": item["path"],
                            "change_type": item["change_type"],
                            "content_type": item["content_type"],
                            "review_only": True,
                        },
                    },
                    index=index,
                )
            )
    release_like = [row for row in candidates if row["release_eligibility"]["eligible"]]
    chunks = []
    unchunkable = []
    for record in release_like:
        result = syntax_aware_chunks(record)
        if not result["parseable"]:
            unchunkable.append({"record_id": record["record_id"], "parse_error": result["parse_error"]})
            continue
        partition = record["partition"]
        for index, chunk in enumerate(result["chunks"]):
            chunks.append(
                {
                    "chunk_uid": f"review-only::{record['record_id']}::chunk-{index:04d}",
                    "record_id": record["record_id"],
                    "split": "train",
                    "stage_a_pass": True,
                    "bundle_id": partition["bundle_id"],
                    "repository_identity": partition["repository_identity"],
                    "path": partition["path"],
                    "change_type": partition["change_type"],
                    "content_type": partition["content_type"],
                    "chunk_kind": chunk["kind"],
                    "text": chunk["text"],
                    "review_only": True,
                }
            )
    decisions = apply_stage_a_hard_gates(chunks)
    review_pass = [{**row, "review_only": True} for row in decisions if row["stage_a_pass"]]
    combined = [*list(_jsonl(primary_stage_a_path)), *review_pass]
    combined_scored = score_stage_b(combined, quality_weight=0.8, redundancy_weight=0.2, redundancy_search_mode="indexed_exact")
    _write_jsonl(output_dir / "review_only_stage_a_pass.jsonl", review_pass)
    _write_jsonl(output_dir / "combined_review_scored.jsonl", combined_scored)
    report = {
        "schema_version": "temporal-code-proxy-review-expansion-report-v1",
        "summary": {
            "candidate_records": len(candidates),
            "locally_release_eligible_records": len(release_like),
            "review_only_stage_a_pass_chunks": len(review_pass),
            "review_only_stage_a_rejected_chunks": len(decisions) - len(review_pass),
            "unchunkable_records": len(unchunkable),
            "combined_review_scored_chunks": len(combined_scored),
            "combined_repository_count": len({row["repository_identity"] for row in combined_scored}),
        },
        "review_only_boundary": {
            "training_approval": False,
            "stage0_release_candidate": False,
            "test_command_verified": False,
            "purpose": "blind Stage-B Core-proxy review only",
        },
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Review-only content preparation; no training, Stage-C, or release claim.",
    }
    save_json(output_dir / "proxy_review_expansion_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare temporal-code proxy-review expansion.")
    parser.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    parser.add_argument("--primary-stage-a", type=Path, default=DEFAULT_PRIMARY_STAGE_A)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = prepare(args.bundles, args.primary_stage_a, args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
