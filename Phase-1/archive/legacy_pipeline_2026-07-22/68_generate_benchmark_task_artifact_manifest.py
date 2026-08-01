#!/usr/bin/env python3
"""Generate benchmark task-artifact quarantine rules without retaining raw task content."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import normalize_repository_identity
from ingestion.code_fingerprints import derived_fingerprints


DEFAULT_SEED = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "benchmark_task_artifact_manifest.json"
DATASET_SERVER = "https://datasets-server.huggingface.co"
HASH_FIELDS = ("problem_statement", "patch", "test_patch")
COMMIT_FIELDS = ("base_commit", "environment_setup_commit")


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _get_json_with_retry(url: str, *, max_retries: int = 12) -> Dict[str, Any]:
    retries = 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retries += 1
            if exc.code not in {429, 500, 502, 503, 504} or retries > max_retries:
                raise
            retry_after = int(exc.headers.get("Retry-After") or 0)
            time.sleep(max(retry_after, min(120, 5 * (2 ** min(retries - 1, 5)))))


def _rows(dataset: str, config: str, split: str, *, delay_seconds: float) -> Iterable[Dict[str, Any]]:
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100}
        )
        payload = _get_json_with_retry(f"{DATASET_SERVER}/rows?{params}")
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                yield row
        offset += len(rows)
        total = int(payload.get("num_rows_total") or offset)
        if not rows or offset >= total:
            break
        if delay_seconds:
            time.sleep(delay_seconds)


def generate(seed: Dict[str, Any], *, benchmarks: set[str] | None, delay_seconds: float) -> Dict[str, Any]:
    benchmark_rows = []
    for entry in seed["entries"]:
        name = str(entry["benchmark"])
        if benchmarks and name not in benchmarks:
            continue
        sources = entry.get("dataset_sources") or []
        if not sources:
            continue
        repositories: Dict[str, Dict[str, set[str]]] = {}
        source_reports = []
        for source in sources:
            for split in source["splits"]:
                count = 0
                error = None
                try:
                    for row in _rows(source["dataset"], source["config"], split, delay_seconds=delay_seconds):
                        identity = normalize_repository_identity(str(row.get("repo") or ""))
                        if not identity:
                            continue
                        rule = repositories.setdefault(
                            identity,
                            {
                                "commit_oids": set(),
                                "normalized_sha256": set(),
                                "token_simhash64": set(),
                                "python_ast_sha256": set(),
                            },
                        )
                        for field in COMMIT_FIELDS:
                            value = row.get(field)
                            if isinstance(value, str) and value:
                                rule["commit_oids"].add(value.lower())
                        for field in HASH_FIELDS:
                            raw_value = row.get(field)
                            value = _sha256(raw_value)
                            if value:
                                rule["normalized_sha256"].add(value)
                            if isinstance(raw_value, str) and raw_value:
                                fingerprints = derived_fingerprints(raw_value)
                                for fingerprint_name, fingerprint in fingerprints.items():
                                    rule[fingerprint_name].add(fingerprint)
                        count += 1
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                source_reports.append(
                    {
                        "dataset": source["dataset"],
                        "config": source["config"],
                        "split": split,
                        "rows_processed": count,
                        "error": error,
                    }
                )
        rules = [
            {
                "repository_identity": identity,
                "commit_oids": sorted(values["commit_oids"]),
                "normalized_sha256": sorted(values["normalized_sha256"]),
                "token_simhash64": sorted(values["token_simhash64"]),
                "python_ast_sha256": sorted(values["python_ast_sha256"]),
            }
            for identity, values in sorted(repositories.items())
        ]
        complete = all(report["error"] is None for report in source_reports)
        benchmark_rows.append(
            {
                "benchmark": name,
                "status": "complete" if complete else "incomplete",
                "source_reports": source_reports,
                "repository_count": len(rules),
                "task_artifact_rules": rules,
                "raw_problem_patch_or_test_content_persisted": False,
            }
        )
    return {
        "schema_version": "benchmark-task-artifact-manifest-v2",
        "scope": "derived_quarantine_identities_hashes_and_near_duplicate_fingerprints_only",
        "fingerprint_contract": {
            "token_simhash64": {"maximum_hamming_distance": 3},
            "python_ast_sha256": {"normalization": "identifiers_and_literals"},
        },
        "benchmarks": benchmark_rows,
        "summary": {
            "benchmark_count": len(benchmark_rows),
            "complete_benchmark_count": sum(1 for row in benchmark_rows if row["status"] == "complete"),
            "incomplete_benchmark_count": sum(1 for row in benchmark_rows if row["status"] != "complete"),
            "repository_rule_count": sum(row["repository_count"] for row in benchmark_rows),
        },
        "claim_boundary": "Quarantine artifact identities only; raw benchmark task content is not retained.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark task-artifact quarantine manifest.")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmarks", nargs="*")
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()
    report = generate(
        load_json(args.seed),
        benchmarks=set(args.benchmarks) if args.benchmarks else None,
        delay_seconds=max(0.0, args.delay_seconds),
    )
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
