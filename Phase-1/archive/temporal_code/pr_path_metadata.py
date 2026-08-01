#!/usr/bin/env python3
"""Collect changed-path metadata without fetching PR prose or file content."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_change import path_exclusion_reason


DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_pr_path_metadata.json"
GITHUB_API = "https://api.github.com"
MAXIMUM_PATH_ROWS = 100


def _is_test_path(path: str) -> bool:
    value = path.lower().replace("\\", "/")
    name = value.rsplit("/", 1)[-1]
    return value.endswith(".py") and (
        value.startswith("test")
        or "/test" in value
        or name.startswith("test_")
        or name == "conftest.py"
    )


def classify_changed_paths(paths: Iterable[str], allowed_suffixes: Iterable[str]) -> Dict[str, Any]:
    suffixes = tuple(str(value).lower() for value in allowed_suffixes)
    counts: Counter[str] = Counter()
    retained = []
    for raw_path in paths:
        path = str(raw_path).replace("\\", "/")
        lower = path.lower()
        if not lower.endswith(suffixes) or path_exclusion_reason(path):
            continue
        retained.append(path)
        if _is_test_path(path):
            counts["test"] += 1
        elif lower.endswith(".py"):
            counts["code"] += 1
        elif lower.endswith((".md", ".rst", ".txt")):
            counts["documentation"] += 1
        else:
            counts["configuration"] += 1
    if counts["code"] and counts["test"]:
        stratum = "code_and_test"
    elif counts["code"]:
        stratum = "code_only"
    elif counts["test"]:
        stratum = "test_only"
    elif counts["documentation"]:
        stratum = "documentation_only"
    elif retained:
        stratum = "configuration_only"
    else:
        stratum = "no_allowed_paths"
    return {
        "path_stratum": stratum,
        "allowed_path_count": len(retained),
        "content_type_path_counts": dict(sorted(counts.items())),
        "allowed_paths": sorted(retained),
    }


class GitHubPathMetadataClient:
    def __init__(self, token: str, *, delay_seconds: float = 0.0) -> None:
        self.token = token
        self.delay_seconds = delay_seconds
        self.requests = 0

    def pull_file_paths(self, repository: str, number: int) -> List[str]:
        request = urllib.request.Request(
            f"{GITHUB_API}/repos/{repository}/pulls/{number}/files?per_page={MAXIMUM_PATH_ROWS}"
        )
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("User-Agent", "unlv-temporal-code-curation/1.0")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail[:500]}") from exc
        self.requests += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if not isinstance(payload, list):
            raise RuntimeError("Pull-request file metadata must be a list.")
        return [str(row.get("filename") or "") for row in payload if isinstance(row, dict)]


def collect(manifest: Dict[str, Any], client: GitHubPathMetadataClient) -> Dict[str, Any]:
    limits = manifest["freeze_contract"]["content_fetch_limits"]
    allowed_suffixes = limits["allowed_file_suffixes"]
    repositories: Dict[str, Any] = {}
    stratum_counts: Counter[str] = Counter()
    incomplete_count = 0
    error_count = 0
    for identity, repository in manifest["repositories"].items():
        pulls = []
        for sample in repository["sampled_prs"]:
            number = int(sample["number"])
            try:
                paths = client.pull_file_paths(identity, number)
                classification = classify_changed_paths(paths, allowed_suffixes)
                complete = len(paths) < MAXIMUM_PATH_ROWS
                error = None
            except RuntimeError as exc:
                paths = []
                classification = classify_changed_paths(paths, allowed_suffixes)
                complete = False
                error = str(exc)
                error_count += 1
            incomplete_count += int(not complete)
            stratum_counts[classification["path_stratum"]] += 1
            pulls.append(
                {
                    "number": number,
                    "mergedAt": sample["mergedAt"],
                    "mergeCommit": sample["mergeCommit"],
                    "path_metadata_complete": complete,
                    "path_row_count": len(paths),
                    "error": error,
                    **classification,
                }
            )
        repositories[identity] = {
            "repository_identity": identity,
            "assigned_split": repository["assigned_split"],
            "tree_path_count": repository["tree_path_count"],
            "pull_requests": pulls,
        }
        print({"repository": identity, "pull_requests": len(pulls)})
    return {
        "schema_version": "temporal-code-pr-path-metadata-v1",
        "status": "collected_before_tranche_content_fetch",
        "scope": "changed_paths_only_no_prose_no_file_content",
        "summary": {
            "repository_count": len(repositories),
            "pull_request_count": sum(len(row["pull_requests"]) for row in repositories.values()),
            "path_metadata_incomplete_count": incomplete_count,
            "request_error_count": error_count,
            "path_stratum_counts": dict(sorted(stratum_counts.items())),
            "github_api_requests": client.requests,
        },
        "repositories": repositories,
        "forbidden_fields_collected": [],
        "claim_boundary": "Path-only pre-collection metadata; no content, Stage-B, Utility, or training claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect prose-free changed-path metadata.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    discovery = __import__("64_discover_temporal_code_repositories")
    token, _ = discovery.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    manifest = load_json(args.manifest)
    report = collect(manifest, GitHubPathMetadataClient(token, delay_seconds=max(0.0, args.delay_seconds)))
    report["source_manifest_sha256"] = sha256_file(args.manifest)
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
