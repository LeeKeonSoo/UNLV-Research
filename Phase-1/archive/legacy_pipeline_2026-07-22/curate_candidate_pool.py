#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from curation_artifacts import load_json, save_json, sha256_file
from ingestion.candidate_processing import process_candidate


JsonMap = dict[str, Any]


def load_config(path: Path) -> JsonMap:
    config = load_json(path)
    if config.get("status") != "frozen_before_stage_a_b_c_materialization":
        raise RuntimeError(f"Unexpected curation config status: {config.get('status')}")
    return config


def _read_jsonl(paths: Iterable[Path]) -> list[JsonMap]:
    rows: list[JsonMap] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _token_proxy(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, max_chunk_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph]
        if len(paragraph) > max_chunk_chars:
            words = paragraph.split()
            pieces = []
            while words:
                piece: list[str] = []
                while words and len(" ".join([*piece, words[0]])) <= max_chunk_chars:
                    piece.append(words.pop(0))
                if not piece:
                    piece.append(words.pop(0))
                pieces.append(" ".join(piece))
        for piece in pieces:
            proposal = piece if not current else f"{current}\n\n{piece}"
            if current and len(proposal) > max_chunk_chars:
                chunks.append(current)
                current = piece
            else:
                current = proposal
    if current:
        chunks.append(current)
    return chunks


def _partition_domain(record: JsonMap) -> str:
    partition = record.get("partition")
    if isinstance(partition, dict):
        return str(partition.get("domain") or "unspecified")
    return "unspecified"


def _audit_provenance(record: JsonMap) -> JsonMap:
    provenance = record["provenance"]
    rights = record["rights"]
    partition = record.get("partition")
    source_pool_role = None
    if isinstance(partition, dict):
        source_pool_role = partition.get("source_pool_role")
    return {
        "source_name": provenance["source_name"],
        "source_pool_role": source_pool_role,
        "license": rights["license"],
    }


def _stage_b_chunks(released: Iterable[JsonMap], config: JsonMap) -> tuple[list[JsonMap], list[JsonMap]]:
    hard_gate = config["stage_b"]
    minimum = int(hard_gate["minimum_chunk_chars"])
    maximum = int(hard_gate["max_chunk_chars"])
    seen: set[str] = set()
    passed: list[JsonMap] = []
    rejected: list[JsonMap] = []
    for record in released:
        for index, text in enumerate(chunk_text(str(record["text"]), maximum)):
            normalized = " ".join(text.split())
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            reasons: list[str] = []
            if len(text) < minimum:
                reasons.append("text_too_short")
            if digest in seen:
                reasons.append("normalized_exact_duplicate")
            seen.add(digest)
            chunk = {
                "chunk_uid": f"{record['record_id']}::{index:04d}",
                "text": text,
                "token_proxy": _token_proxy(text),
                "domain": _partition_domain(record),
                "stage_a_record_id": record["record_id"],
                "stage_b_hard_gate_reasons": reasons,
                "audit_only_provenance": _audit_provenance(record),
                "stage_c_selector_visible": {
                    "source_name": False,
                    "source_pool_role": False,
                    "utility": False,
                    "benchmark_outcomes": False,
                },
            }
            if reasons:
                rejected.append(chunk)
            else:
                passed.append(chunk)
    return passed, rejected


def materialize(config_path: Path) -> JsonMap:
    config = load_config(config_path)
    input_paths = [Path(str(path)) for path in config["input"]["candidate_files"]]
    output_dir = Path(str(config["output_dir"]))
    candidates = _read_jsonl(input_paths)
    processed = [process_candidate(candidate, index=index) for index, candidate in enumerate(candidates)]
    released = [row for row in processed if row["release_eligibility"]["eligible"]]
    quarantined = [row for row in processed if not row["release_eligibility"]["eligible"]]
    passed, rejected = _stage_b_chunks(released, config)
    budget = config["stage_c"]["training_budget_token_proxy"]
    if budget is not None:
        raise RuntimeError("This runtime supports retain_all only; declare a separate budget-selection contract.")
    paths = {
        "stage_a_release": output_dir / "stage_a_release_candidates.jsonl",
        "stage_a_quarantine": output_dir / "stage_a_quarantined_candidates.jsonl",
        "stage_b_pass": output_dir / "stage_b_pass_chunks.jsonl",
        "stage_b_rejected": output_dir / "stage_b_rejected_chunks.jsonl",
        "stage_c_curated": output_dir / "stage_c_curated_chunks.jsonl",
    }
    _write_jsonl(paths["stage_a_release"], released)
    _write_jsonl(paths["stage_a_quarantine"], quarantined)
    _write_jsonl(paths["stage_b_pass"], passed)
    _write_jsonl(paths["stage_b_rejected"], rejected)
    _write_jsonl(paths["stage_c_curated"], passed)
    report = {
        "schema_version": "curation-materialization-report-v1",
        "status": "curation_materialization_complete",
        "stage_contract": {
            "stage_a": "candidate_provenance_normalization_and_risk_quarantine",
            "stage_b": "chunk_level_hard_gate",
            "stage_c": "retain_all_without_binding_budget",
            "external_evaluation": "not_started",
        },
        "summary": {
            "input_records": len(candidates),
            "stage_a_release_records": len(released),
            "stage_a_quarantined_records": len(quarantined),
            "stage_b_pass_chunks": len(passed),
            "stage_b_rejected_chunks": len(rejected),
            "stage_c_curated_chunks": len(passed),
            "stage_c_curated_token_proxy": sum(int(row["token_proxy"]) for row in passed),
            "stage_a_quarantine_reasons": dict(Counter(reason for row in quarantined for reason in row["quarantine"]["reasons"])),
            "stage_b_rejection_reasons": dict(Counter(reason for row in rejected for reason in row["stage_b_hard_gate_reasons"])),
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "source_pool_role_read": False,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "source_sha256": {str(config_path): sha256_file(config_path)},
    }
    save_json(output_dir / "curation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a frozen Stage A/B/C curation output.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = materialize(args.config)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
