#!/usr/bin/env python3
"""Fetch a frozen known-high-quality Python reference pool from GitHub."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import PurePosixPath, Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import generated_file_detection, normalize_repository_identity, path_exclusion_reason
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks
from ingestion.code_selection import token_proxy_count
from ingestion.normalize import detect_hazards, process_candidate


DEFAULT_CONFIG = Path("configs") / "code_domain_known_high_quality_reference_pool_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "known_high_quality_reference_pool"
GITHUB_API = "https://api.github.com"


class GitHubClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requests = 0

    def get(self, path: str, *, allow_unauthenticated_retry: bool = True) -> Any:
        request = urllib.request.Request(f"{GITHUB_API}{path}")
        request.add_header("Accept", "application/vnd.github+json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("User-Agent", "unlv-code-domain-reference-pool/1.0")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401 and self.token and allow_unauthenticated_retry:
                self.token = ""
                return self.get(path, allow_unauthenticated_retry=False)
            raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail[:500]}") from exc
        self.requests += 1
        return payload

    def file_text(self, repository: str, path: str, ref: str, maximum_bytes: int) -> Dict[str, Any]:
        raw_url = (
            "https://raw.githubusercontent.com/"
            f"{repository}/{urllib.parse.quote(ref, safe='')}/{urllib.parse.quote(path, safe='/')}"
        )
        request = urllib.request.Request(raw_url)
        request.add_header("User-Agent", "unlv-code-domain-reference-pool/1.0")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read(maximum_bytes + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"text": None, "blocker": "content_not_found"}
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub raw HTTP {exc.code}: {detail[:500]}") from exc
        size = len(raw)
        if size > maximum_bytes:
            return {"text": None, "blocker": "file_too_large", "size": size}
        if b"\x00" in raw:
            return {"text": None, "blocker": "binary_content", "size": size}
        text = raw.decode("utf-8", errors="replace")
        hazards = detect_hazards(text, pii_context="repository_code")
        if hazards["secret_detected"] or hazards["pii_detected"]:
            return {"text": None, "blocker": "hazardous_content_not_persisted", "size": size, "hazards": hazards}
        return {"text": text, "blocker": None, "size": size, "hazards": hazards}


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _path_allowed(path_value: str, config: Dict[str, Any]) -> bool:
    path = PurePosixPath(path_value.replace("\\", "/"))
    lower_parts = {part.lower() for part in path.parts}
    excluded = {str(part).lower() for part in config["exclude_path_parts"]}
    if lower_parts.intersection(excluded):
        return False
    if path_exclusion_reason(path_value):
        return False
    return any(str(path).lower().endswith(suffix) for suffix in config["allowed_suffixes"])


def _stable_file_order(repository: str, commit: str, path_value: str, seed: int) -> str:
    raw = f"{seed}:{repository}:{commit}:{path_value}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chunk_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = syntax_aware_chunks(record)
    if not result["parseable"]:
        return []
    partition = record["partition"]
    chunks = []
    for index, chunk in enumerate(result["chunks"]):
        chunks.append(
            {
                "chunk_uid": f"{record['record_id']}::chunk-{index:04d}",
                "record_id": record["record_id"],
                "split": "reference",
                "bundle_id": partition["bundle_id"],
                "repository_identity": partition["repository_identity"],
                "path": partition["path"],
                "change_type": "snapshot",
                "content_type": "code",
                "chunking_mode": result["chunking_mode"],
                "chunk_kind": chunk["kind"],
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "text": chunk["text"],
                "reference_pool": "known_high_quality",
            }
        )
    return chunks


def fetch(config: Dict[str, Any], client: GitHubClient, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    maximum_files = int(config["maximum_files_per_repository"])
    maximum_bytes = int(config["maximum_file_bytes"])
    raw_records: List[Dict[str, Any]] = []
    repository_reports = {}
    blockers: List[str] = []

    for repo_config in config["repositories"]:
        repository = normalize_repository_identity(repo_config["repository_identity"])
        repo_payload = client.get(f"/repos/{repository}")
        license_key = str(((repo_payload.get("license") or {}).get("spdx_id")) or "")
        default_branch = str(repo_payload.get("default_branch") or "main")
        branch_payload = client.get(f"/repos/{repository}/branches/{urllib.parse.quote(default_branch)}")
        commit = str(((branch_payload.get("commit") or {}).get("sha")) or "")
        if license_key != str(repo_config["expected_license"]):
            blockers.append(f"{repository}:license_mismatch:{license_key}")
            repository_reports[repository] = {"status": "skipped_license_mismatch", "license": license_key}
            continue
        tree = client.get(f"/repos/{repository}/git/trees/{urllib.parse.quote(commit)}?recursive=1")
        paths = [
            str(row.get("path") or "")
            for row in tree.get("tree", [])
            if row.get("type") == "blob" and _path_allowed(str(row.get("path") or ""), config)
        ]
        selected_paths = sorted(
            paths,
            key=lambda path_value: _stable_file_order(repository, commit, path_value, seed),
        )[:maximum_files]
        fetched = 0
        fetch_blockers = {}
        for path_value in selected_paths:
            content = client.file_text(repository, path_value, commit, maximum_bytes)
            if content.get("blocker"):
                fetch_blockers[content["blocker"]] = fetch_blockers.get(content["blocker"], 0) + 1
                continue
            text = str(content["text"] or "")
            generated = generated_file_detection(path_value, text)
            if generated["generated"]:
                fetch_blockers["generated_file"] = fetch_blockers.get("generated_file", 0) + 1
                continue
            raw_records.append(
                process_candidate(
                    {
                        "id": f"known-hq::{repository}::{commit[:12]}::{path_value}",
                        "text": text,
                        "provenance": {
                            "source_name": repository,
                            "source_uri": f"https://github.com/{repository}/blob/{commit}/{path_value}",
                            "collected_at": "2026-06-19T00:00:00+09:00",
                        },
                        "language": {"code": "python", "confidence": 1.0},
                        "rights": {"status": "allowed", "license": license_key},
                        "pii_context": "repository_code",
                        "partition": {
                            "split": "reference",
                            "bundle_id": f"known_hq__{repository.replace('/', '__')}__{commit[:12]}",
                            "repository_identity": repository,
                            "path": path_value,
                            "change_type": "snapshot",
                            "content_type": "code",
                            "reference_pool": "known_high_quality",
                            "snapshot_commit": commit,
                            "quality_basis": repo_config["quality_basis"],
                        },
                    },
                    index=len(raw_records),
                )
            )
            fetched += 1
        repository_reports[repository] = {
            "status": "fetched",
            "license": license_key,
            "default_branch": default_branch,
            "snapshot_commit": commit,
            "eligible_path_count": len(paths),
            "selected_path_count": len(selected_paths),
            "fetched_record_count": fetched,
            "fetch_blockers": dict(sorted(fetch_blockers.items())),
        }

    chunks = [chunk for record in raw_records for chunk in _chunk_record(record)]
    decisions = apply_stage_a_hard_gates(chunks)
    passed = [
        {**row, "token_proxy_count": token_proxy_count(str(row.get("text") or ""))}
        for row in decisions
        if row["stage_a_pass"]
    ]
    rejected = [row for row in decisions if not row["stage_a_pass"]]
    _write_jsonl(output_dir / "known_high_quality_raw_records.jsonl", raw_records)
    _write_jsonl(output_dir / "known_high_quality_stage_a_pass.jsonl", passed)
    _write_jsonl(output_dir / "known_high_quality_stage_a_rejected.jsonl", rejected)
    report = {
        "schema_version": "code-domain-known-high-quality-reference-pool-report-v1",
        "config_schema": config["schema_version"],
        "status": "reference_pool_materialized" if passed and not blockers else "reference_pool_materialized_with_blockers",
        "summary": {
            "repository_count": len(config["repositories"]),
            "raw_record_count": len(raw_records),
            "chunk_count": len(decisions),
            "stage_a_pass_count": len(passed),
            "stage_a_rejected_count": len(rejected),
            "stage_a_pass_token_proxy": sum(row["token_proxy_count"] for row in passed),
            "github_api_requests": client.requests,
            "blockers": sorted(blockers),
        },
        "repositories": repository_reports,
        "forbidden_uses": config["forbidden_uses"],
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    save_json(output_dir / "known_high_quality_reference_pool_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch known-high-quality Python reference pool.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    discovery = __import__("64_discover_temporal_code_repositories")
    token, _ = discovery.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    report = fetch(load_json(args.config), GitHubClient(token), args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
