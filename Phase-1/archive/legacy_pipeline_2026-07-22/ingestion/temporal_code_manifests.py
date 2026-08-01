"""Build frozen repository-split and benchmark-quarantine manifests."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Dict, Iterable, List

from ingestion.code_change import normalize_repository_identity
from ingestion.code_fingerprints import simhash_hamming_distance


REPOSITORY_SPLIT_MANIFEST_VERSION = "temporal-code-repository-split-manifest-v1"
BENCHMARK_QUARANTINE_MANIFEST_VERSION = "temporal-code-benchmark-quarantine-manifest-v1"
SPLIT_BUCKETS = {
    "train": (0, 80),
    "development": (80, 90),
    "confirmatory": (90, 100),
}


def repository_bucket(repository_identity: str) -> int:
    normalized = normalize_repository_identity(repository_identity)
    return int(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), 16) % 100


def assigned_split(repository_identity: str) -> str:
    bucket = repository_bucket(repository_identity)
    for split, (low, high) in SPLIT_BUCKETS.items():
        if low <= bucket < high:
            return split
    raise AssertionError(bucket)


def _day(value: str) -> date:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def temporal_split(merge_timestamp: str, protocol: Dict[str, Any]) -> str | None:
    try:
        value = _day(merge_timestamp)
    except (TypeError, ValueError):
        return None
    collection = protocol["collection_contract"]
    windows = {
        "train": collection["training_window"],
        "development": collection["development_holdout_window"],
        "confirmatory": collection["frozen_confirmatory_holdout_window"],
    }
    for split, window in windows.items():
        if _day(window["start"]) <= value <= _day(window["end"]):
            return split
    return None


def build_repository_split_manifest(
    repositories: Iterable[Dict[str, Any]],
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for repository in repositories:
        identity = normalize_repository_identity(str(repository.get("repository_identity") or ""))
        if not identity or "/" not in identity:
            raise ValueError(f"Invalid repository identity: {identity!r}")
        if identity in rows:
            raise ValueError(f"Duplicate repository identity: {identity}")
        rows[identity] = {
            "repository_identity": identity,
            "repository_url": repository.get("repository_url"),
            "bucket": repository_bucket(identity),
            "assigned_split": assigned_split(identity),
            "license": repository.get("license"),
            "assignment_frozen_before_core_scoring": True,
        }
    counts = {split: sum(1 for row in rows.values() if row["assigned_split"] == split) for split in SPLIT_BUCKETS}
    return {
        "schema_version": REPOSITORY_SPLIT_MANIFEST_VERSION,
        "protocol_name": protocol["protocol_name"],
        "assignment_algorithm": "sha256(normalized_repository_identity) modulo 100",
        "bucket_ranges": {split: [low, high - 1] for split, (low, high) in SPLIT_BUCKETS.items()},
        "repository_count": len(rows),
        "split_counts": counts,
        "repositories": dict(sorted(rows.items())),
    }


def bundle_split_eligibility(bundle: Dict[str, Any], split_manifest: Dict[str, Any], protocol: Dict[str, Any]) -> Dict[str, Any]:
    identity = normalize_repository_identity(str(bundle.get("repository_identity") or ""))
    repository = (split_manifest.get("repositories") or {}).get(identity)
    blockers: List[str] = []
    if repository is None:
        blockers.append("repository_not_in_frozen_split_manifest")
        assigned = None
    else:
        assigned = repository["assigned_split"]
    observed = temporal_split(str(bundle.get("merge_timestamp") or ""), protocol)
    if observed is None:
        blockers.append("outside_frozen_time_windows")
    if assigned and observed and assigned != observed:
        blockers.append("repository_split_time_window_mismatch")
    return {
        "eligible": not blockers,
        "repository_identity": identity,
        "assigned_split": assigned,
        "observed_temporal_split": observed,
        "blockers": sorted(set(blockers)),
    }


def build_benchmark_quarantine_manifest(entries: Iterable[Dict[str, Any]], protocol: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    seen = set()
    for entry in entries:
        benchmark = str(entry.get("benchmark") or "").strip()
        source_url = str(entry.get("source_url") or "").strip()
        repository_patterns = sorted(
            {
                normalize_repository_identity(value)
                for value in entry.get("repository_patterns") or []
                if normalize_repository_identity(value)
            }
        )
        task_artifact_rules = []
        for rule in entry.get("task_artifact_rules") or []:
            if not isinstance(rule, dict):
                continue
            task_artifact_rules.append(
                {
                    "repository_identity": normalize_repository_identity(str(rule.get("repository_identity") or "")),
                    "commit_oids": sorted({str(value).lower() for value in rule.get("commit_oids") or [] if value}),
                    "normalized_sha256": sorted(
                        {str(value).lower() for value in rule.get("normalized_sha256") or [] if value}
                    ),
                    "token_simhash64": sorted(
                        {str(value).lower() for value in rule.get("token_simhash64") or [] if value}
                    ),
                    "python_ast_sha256": sorted(
                        {str(value).lower() for value in rule.get("python_ast_sha256") or [] if value}
                    ),
                }
            )
        text_hashes = sorted({str(value).lower() for value in entry.get("text_sha256") or [] if value})
        signature = (benchmark.lower(), source_url.lower())
        if not benchmark or not source_url:
            raise ValueError("Benchmark quarantine entries require benchmark and source_url.")
        if signature in seen:
            raise ValueError(f"Duplicate benchmark quarantine entry: {benchmark}")
        seen.add(signature)
        rows.append(
            {
                "benchmark": benchmark,
                "source_url": source_url,
                "repository_patterns": repository_patterns,
                "text_sha256": text_hashes,
                "task_artifact_rules": task_artifact_rules,
                "task_artifact_manifest_status": str(entry.get("task_artifact_manifest_status") or "not_applicable"),
                "dataset_sources": list(entry.get("dataset_sources") or []),
                "quarantine_required": True,
            }
        )
    return {
        "schema_version": BENCHMARK_QUARANTINE_MANIFEST_VERSION,
        "protocol_name": protocol["protocol_name"],
        "frozen_before_training": True,
        "matching_contract": list(protocol["benchmark_quarantine"]["checks"]),
        "entries": rows,
    }


def benchmark_quarantine_decision(bundle: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    identity = normalize_repository_identity(str(bundle.get("repository_identity") or ""))
    normalized_hashes = {
        str(item.get("normalized_sha256") or "").lower()
        for item in bundle.get("content_signatures") or []
        if isinstance(item, dict)
    }
    bundle_commit_oids = {
        str(bundle.get("parent_commit") or "").lower(),
        str(bundle.get("merge_commit") or "").lower(),
    }
    bundle_simhashes = {
        str(item.get("token_simhash64") or "").lower()
        for item in bundle.get("content_signatures") or []
        if isinstance(item, dict) and item.get("token_simhash64")
    }
    bundle_ast_hashes = {
        str(item.get("python_ast_sha256") or "").lower()
        for item in bundle.get("content_signatures") or []
        if isinstance(item, dict) and item.get("python_ast_sha256")
    }
    matches = []
    for entry in manifest.get("entries") or []:
        reasons = []
        if identity in set(entry.get("repository_patterns") or []):
            reasons.append("benchmark_repository_identity")
        if normalized_hashes.intersection(set(entry.get("text_sha256") or [])):
            reasons.append("benchmark_exact_content_hash")
        for rule in entry.get("task_artifact_rules") or []:
            if identity == rule.get("repository_identity") and bundle_commit_oids.intersection(
                set(rule.get("commit_oids") or [])
            ):
                reasons.append("benchmark_task_commit_identity")
            if normalized_hashes.intersection(set(rule.get("normalized_sha256") or [])):
                reasons.append("benchmark_task_content_hash")
            if bundle_ast_hashes.intersection(set(rule.get("python_ast_sha256") or [])):
                reasons.append("benchmark_task_ast_structure_hash")
            rule_simhashes = set(rule.get("token_simhash64") or [])
            if any(
                simhash_hamming_distance(left, right) <= 3
                for left in bundle_simhashes
                for right in rule_simhashes
            ):
                reasons.append("benchmark_task_token_simhash_near_duplicate")
        if reasons:
            matches.append({"benchmark": entry["benchmark"], "reasons": reasons})
    return {
        "quarantine": bool(matches),
        "repository_identity": identity,
        "matches": matches,
    }
