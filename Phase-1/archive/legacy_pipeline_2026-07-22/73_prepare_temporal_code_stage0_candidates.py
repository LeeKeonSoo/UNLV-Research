#!/usr/bin/env python3
"""Convert collection-approved temporal-code payloads into split-preserving Stage-0 records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import bundle_training_payload
from ingestion.normalize import process_candidate


DEFAULT_AUDIT = OUTPUT_DIR / "temporal_code_collection" / "smoke_bundles" / "smoke_bundle_audit_report.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_smoke"


def raw_candidates(audit: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for decision in audit.get("decisions") or []:
        if decision.get("stage0_release_candidate") is not True:
            continue
        bundle = load_json(Path(decision["bundle_path"]))
        payload = bundle_training_payload(bundle)
        if not payload["eligible"]:
            raise ValueError(f"Audited bundle has no eligible training payload: {bundle['bundle_id']}")
        for index, item in enumerate(payload["training_payloads"]):
            path = item["path"]
            content_type = item["content_type"]
            yield {
                "id": f"{bundle['bundle_id']}::{index:03d}::{path}",
                "text": item["text"],
                "provenance": {
                    "source_name": bundle["repository_identity"],
                    "source_uri": f"{bundle['provenance']['source_urls'][0]}#{path}",
                    "collected_at": bundle["provenance"]["collected_at"],
                },
                "language": {
                    "code": "python" if path.lower().endswith(".py") else "en",
                    "confidence": 1.0,
                },
                "rights": {"status": "allowed", "license": item["license"]},
                "pii_context": "repository_code",
                "partition": {
                    "split": decision["assigned_split"],
                    "bundle_id": bundle["bundle_id"],
                    "repository_identity": bundle["repository_identity"],
                    "path": path,
                    "change_type": item["change_type"],
                    "content_type": content_type,
                    "parent_commit": bundle["parent_commit"],
                    "merge_commit": bundle["merge_commit"],
                },
            }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(audit: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    processed = [process_candidate(row, index=index) for index, row in enumerate(raw_candidates(audit))]
    split_rows = {}
    for split in ("train", "development", "confirmatory"):
        rows = [row for row in processed if (row.get("partition") or {}).get("split") == split]
        release = [row for row in rows if row["release_eligibility"]["eligible"]]
        quarantine = [row for row in rows if not row["release_eligibility"]["eligible"]]
        _write_jsonl(output_dir / split / "release_candidates.jsonl", release)
        _write_jsonl(output_dir / split / "quarantined_candidates.jsonl", quarantine)
        split_rows[split] = {
            "input_records": len(rows),
            "release_candidate_records": len(release),
            "quarantined_records": len(quarantine),
        }
    report = {
        "schema_version": "temporal-code-stage0-smoke-report-v1",
        "source_audit_schema": audit["schema_version"],
        "summary": {
            "input_records": len(processed),
            "release_candidate_records": sum(row["release_eligibility"]["eligible"] for row in processed),
            "quarantined_records": sum(not row["release_eligibility"]["eligible"] for row in processed),
            "split_counts": split_rows,
        },
        "confirmatory_use_boundary": (
            "Confirmatory records are processed only for frozen contract validation; "
            "their outcomes must not tune Stage A, Stage B, or the collection policy."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Stage-0 smoke records only; no Stage-A, Stage-B, Stage-C, or training claim.",
    }
    save_json(output_dir / "stage0_smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare temporal-code Stage-0 smoke candidates.")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = prepare(load_json(args.audit), args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
