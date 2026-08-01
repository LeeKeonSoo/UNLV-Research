#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "code_5m_corpus_acquisition_v2.json"


def eligibility_reason(row: dict[str, Any], admission: dict[str, Any]) -> str | None:
    content = str(row.get("content") or "")
    if not content:
        return "missing_content"
    size = int(row.get("size") or 0)
    if size < int(admission["min_file_bytes"]):
        return "file_too_small"
    if size > int(admission["max_file_bytes"]):
        return "file_too_large"
    path = str(row.get("max_stars_repo_path") or "").lower()
    if any(fragment in path for fragment in admission["exclude_path_fragments"]):
        return "excluded_path"
    try:
        licenses = ast.literal_eval(str(row.get("max_stars_repo_licenses") or "[]"))
    except (SyntaxError, ValueError):
        return "invalid_license_metadata"
    if not isinstance(licenses, list) or not set(map(str, licenses)).intersection(admission["allowed_spdx_licenses"]):
        return "license_not_allowed"
    return None


def _token_count(tokenizer: Any, content: str) -> int:
    return len(tokenizer(content, add_special_tokens=False)["input_ids"]) + 1


def _record(row: dict[str, Any], source: dict[str, Any], token_count: int, content_hash: str) -> dict[str, Any]:
    repository = str(row["max_stars_repo_name"])
    path = str(row["max_stars_repo_path"])
    return {
        "record_id": f"the-stack-dedup::{content_hash}",
        "text": str(row["content"]),
        "token_count_with_eos": token_count,
        "source_dataset": source["dataset"],
        "source_revision": source["revision"],
        "source_shard": source["data_dir"],
        "repository_or_origin": repository,
        "path": path,
        "license": row["max_stars_repo_licenses"],
        "content_sha256": content_hash,
        "dedup_cluster": content_hash,
        "collected_at": datetime.now(UTC).isoformat(),
        "source_tier": "raw_like",
        "benchmark_exclusion_status": "pending_stage_0_audit",
        "audit_metadata": {
            "source_content_hexsha": row["hexsha"],
            "repository_head_hexsha": row["max_stars_repo_head_hexsha"],
            "stars": row["max_stars_count"],
        },
    }


def collect(config_path: Path, output_root: Path) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = load_json(config_path)
    source = config["primary_source"]
    admission = source["admission_rule"]
    target = int(config["target"]["raw_like_source_target_tokens"])
    cap = int(target * float(admission["repository_concentration_cap_fraction"]))
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"]["tokenizer_id"], local_files_only=True)
    dataset = load_dataset(source["dataset"], data_dir=source["data_dir"], split=source["split"], streaming=True, revision=source["revision"])
    stream = dataset.shuffle(seed=int(source["seed"]), buffer_size=10000)
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "raw_like_candidates.jsonl"
    rejected = Counter()
    repository_tokens: Counter[str] = Counter()
    content_hashes: set[str] = set()
    accepted = 0
    tokens = 0
    scanned = 0
    with data_path.open("w", encoding="utf-8") as handle:
        for row in stream:
            scanned += 1
            reason = eligibility_reason(row, admission)
            if reason is not None:
                rejected[reason] += 1
                continue
            content = str(row["content"])
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in content_hashes:
                rejected["exact_duplicate"] += 1
                continue
            count = _token_count(tokenizer, content)
            repository = str(row["max_stars_repo_name"])
            if repository_tokens[repository] + count > cap:
                rejected["repository_concentration_cap"] += 1
                continue
            if tokens + count > target:
                continue
            handle.write(json.dumps(_record(row, source, count, content_hash), ensure_ascii=False, sort_keys=True) + "\n")
            content_hashes.add(content_hash)
            repository_tokens[repository] += count
            accepted += 1
            tokens += count
            if tokens >= target:
                break
    report = {
        "schema_version": "code-5m-raw-like-collection-report-v1",
        "status": "raw_like_collection_complete" if tokens >= target else "raw_like_collection_incomplete",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "target_tokens": target,
        "collected_tokens": tokens,
        "accepted_records": accepted,
        "scanned_records": scanned,
        "rejected_by_reason": dict(sorted(rejected.items())),
        "repository_count": len(repository_tokens),
        "max_repository_tokens": max(repository_tokens.values(), default=0),
        "repository_token_cap": cap,
        "stage_b_isolation": config["stage_b_isolation"],
    }
    save_json(output_root / "raw_like_collection_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_json(args.config)
    output_root = args.output_root or Path(config["output"]["root"])
    report = collect(args.config, output_root)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "raw_like_collection_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
