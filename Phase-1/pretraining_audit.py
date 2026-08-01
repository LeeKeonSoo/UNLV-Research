#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from curation_artifacts import save_json, sha256_file


JsonMap = dict[str, Any]
SHINGLE_SIZE = 16
MIN_SEGMENT_CHARS = 80


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _token_proxy(text: str) -> int:
    return len(text.split())


def _source_pool(row: JsonMap) -> str:
    partition = row.get("partition")
    if not isinstance(partition, dict):
        return "unlabeled"
    source_tier = partition.get("source_tier")
    if isinstance(source_tier, str) and source_tier:
        return source_tier
    reference_pool = partition.get("reference_pool")
    return str(reference_pool) if reference_pool else "unlabeled"


def _source_dataset(row: JsonMap) -> str:
    partition = row.get("partition")
    if isinstance(partition, dict) and partition.get("source_dataset"):
        return str(partition["source_dataset"])
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and provenance.get("source_name"):
        return str(provenance["source_name"])
    return "unlabeled"


def _distribution(rows: Iterable[JsonMap], labeler: Callable[[JsonMap], str]) -> JsonMap:
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for row in rows:
        label = str(labeler(row))
        counts[label] += 1
        tokens[label] += _token_proxy(str(row.get("text") or ""))
    return {
        label: {"records": counts[label], "whitespace_token_proxy": tokens[label]}
        for label in sorted(counts)
    }


def build_source_snapshot(candidate_path: Path) -> JsonMap:
    rows = _read_jsonl(candidate_path)
    return {
        "schema_version": "pretraining-source-snapshot-v1",
        "input": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "summary": {
            "records": len(rows),
            "whitespace_token_proxy": sum(_token_proxy(str(row.get("text") or "")) for row in rows),
        },
        "by_source_pool": _distribution(rows, _source_pool),
        "by_source_dataset": _distribution(rows, _source_dataset),
        "claim_boundary": "Source composition is provenance audit metadata, not a selector feature or intrinsic quality label.",
    }


def _task_segments(task: JsonMap) -> list[str]:
    fields = ("prompt", "canonical_solution", "test", "assertion", "text", "code")
    return [
        _normalized(str(task[field]))
        for field in fields
        if isinstance(task.get(field), str) and len(_normalized(str(task[field]))) >= MIN_SEGMENT_CHARS
    ]


def _shingles(text: str) -> set[str]:
    tokens = text.split()
    return {
        hashlib.sha256(" ".join(tokens[index : index + SHINGLE_SIZE]).encode("utf-8")).hexdigest()
        for index in range(max(0, len(tokens) - SHINGLE_SIZE + 1))
    }


def _load_benchmark(path: Path) -> tuple[str, JsonMap, dict[str, set[str]], set[str]]:
    snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
    benchmark_id = str(snapshot["benchmark_id"])
    tasks = snapshot.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError(f"Benchmark snapshot requires a tasks list: {path}")
    shingle_tasks: dict[str, set[str]] = defaultdict(set)
    exact_segments: set[str] = set()
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("task_id") or "unknown-task")
        for segment in _task_segments(raw_task):
            exact_segments.add(hashlib.sha256(segment.encode("utf-8")).hexdigest())
            for shingle in _shingles(segment):
                shingle_tasks[shingle].add(task_id)
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "snapshot_revision": snapshot.get("snapshot_revision"),
        "tasks": len(tasks),
    }
    return benchmark_id, metadata, shingle_tasks, exact_segments


def build_benchmark_exclusion_audit(
    *,
    candidate_path: Path,
    benchmark_paths: list[Path],
    required_benchmark_ids: list[str],
    audited_candidate_path: Path,
) -> JsonMap:
    benchmarks = [_load_benchmark(path) for path in benchmark_paths]
    metadata = {benchmark_id: details for benchmark_id, details, _, _ in benchmarks}
    available_ids = set(metadata)
    missing = sorted(set(required_benchmark_ids) - available_ids)
    rows = _read_jsonl(candidate_path)
    retained: list[JsonMap] = []
    matches: list[JsonMap] = []
    for row in rows:
        normalized = _normalized(str(row.get("text") or ""))
        exact = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        row_shingles = _shingles(normalized)
        row_matches: list[JsonMap] = []
        for benchmark_id, _, benchmark_shingles, exact_segments in benchmarks:
            matched_tasks = sorted({task_id for shingle in row_shingles for task_id in benchmark_shingles.get(shingle, set())})
            if exact in exact_segments or matched_tasks:
                row_matches.append(
                    {
                        "benchmark_id": benchmark_id,
                        "match_type": "exact_segment" if exact in exact_segments else "shared_16_token_shingle",
                        "task_ids": matched_tasks,
                    }
                )
        if row_matches:
            matches.append({"record_id": str(row.get("record_id") or "unknown"), "matches": row_matches})
        else:
            retained.append(row)
    audited_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with audited_candidate_path.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    status = "benchmark_exclusion_complete" if not missing else "benchmark_exclusion_incomplete"
    return {
        "schema_version": "benchmark-exclusion-audit-v1",
        "status": status,
        "pretraining_eligible": not missing,
        "candidate_input": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "audited_output": {"path": str(audited_candidate_path), "sha256": sha256_file(audited_candidate_path)},
        "required_benchmark_ids": required_benchmark_ids,
        "available_benchmarks": metadata,
        "missing_required_benchmarks": missing,
        "summary": {"input_records": len(rows), "excluded_records": len(matches), "retained_records": len(retained)},
        "excluded_records": matches,
        "method": "exact_normalized_segment_or_shared_16_token_shingle_v1",
        "claim_boundary": "Incomplete benchmark coverage blocks training eligibility; a clean audit is not evidence of absence beyond supplied snapshots.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build input snapshot and benchmark exclusion audit.")
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--snapshot-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--audited-candidate-output", required=True, type=Path)
    parser.add_argument("--benchmark-snapshot", action="append", default=[], type=Path)
    parser.add_argument("--required-benchmark-id", action="append", default=[])
    args = parser.parse_args()
    save_json(args.snapshot_output, build_source_snapshot(args.candidate_path))
    audit = build_benchmark_exclusion_audit(
        candidate_path=args.candidate_path,
        benchmark_paths=args.benchmark_snapshot,
        required_benchmark_ids=args.required_benchmark_id,
        audited_candidate_path=args.audited_candidate_output,
    )
    save_json(args.audit_output, audit)
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
