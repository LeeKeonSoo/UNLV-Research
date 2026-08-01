#!/usr/bin/env python3
"""Enrich repository candidates using path-only trees and prose-free PR metadata."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_CANDIDATES = OUTPUT_DIR / "temporal_code_collection" / "repository_candidate_manifest_authenticated.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report.json"
GITHUB_API = "https://api.github.com"
TEST_PATH_MARKERS = (
    "test/",
    "tests/",
    "testing/",
    "pytest.ini",
    "tox.ini",
    "noxfile.py",
    "conftest.py",
)
PYTHON_PROJECT_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "poetry.lock",
)


def _window(protocol: Dict[str, Any], split: str) -> Dict[str, str]:
    key = {
        "train": "training_window",
        "development": "development_holdout_window",
        "confirmatory": "frozen_confirmatory_holdout_window",
    }[split]
    return protocol["collection_contract"][key]


def select_balanced_candidates(manifest: Dict[str, Any], max_repositories: int) -> List[Dict[str, Any]]:
    split_rows = {"train": [], "development": [], "confirmatory": []}
    assignments = manifest["preliminary_split_manifest"]["repositories"]
    for identity, candidate in manifest["candidates"].items():
        if not candidate["eligible_for_metadata_enrichment"]:
            continue
        split_rows[assignments[identity]["assigned_split"]].append(candidate)
    for rows in split_rows.values():
        rows.sort(key=lambda row: (-int(row["stars"]), row["repository_identity"]))
    result: List[Dict[str, Any]] = []
    splits = ("train", "development", "confirmatory")
    while len(result) < max_repositories and any(split_rows.values()):
        for split in splits:
            if split_rows[split] and len(result) < max_repositories:
                result.append(split_rows[split].pop(0))
    return result


class GitHubEnrichmentClient:
    def __init__(self, token: str, *, delay_seconds: float = 0.2) -> None:
        self.token = token
        self.delay_seconds = delay_seconds
        self.requests = 0

    def _request(self, request: urllib.request.Request) -> Dict[str, Any]:
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
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub API response must be an object.")
        return payload

    def get_tree_paths(self, repository_identity: str, branch: str) -> List[str]:
        request = urllib.request.Request(f"{GITHUB_API}/repos/{repository_identity}/git/trees/{branch}?recursive=1")
        payload = self._request(request)
        return [
            str(item["path"])
            for item in payload.get("tree") or []
            if isinstance(item, dict) and item.get("type") == "blob" and isinstance(item.get("path"), str)
        ]

    def merged_pr_metadata(
        self,
        repository_identity: str,
        *,
        start: str,
        end: str,
        limit: int,
    ) -> Dict[str, Any]:
        query = """
        query($searchQuery: String!, $limit: Int!) {
          search(query: $searchQuery, type: ISSUE, first: $limit) {
            issueCount
            nodes {
              ... on PullRequest {
                number
                mergedAt
                baseRefName
                headRefOid
                mergeCommit { oid }
              }
            }
          }
        }
        """
        variables = {
            "searchQuery": f"repo:{repository_identity} is:pr is:merged merged:{start}..{end}",
            "limit": limit,
        }
        request = urllib.request.Request(
            f"{GITHUB_API}/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
            method="POST",
        )
        payload = self._request(request)
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL errors: {payload['errors']}")
        search = ((payload.get("data") or {}).get("search") or {})
        nodes = [node for node in search.get("nodes") or [] if isinstance(node, dict)]
        return {"issue_count": int(search.get("issueCount") or 0), "samples": nodes}


def _tree_evidence(paths: List[str]) -> Dict[str, Any]:
    normalized = [path.replace("\\", "/").lower() for path in paths]
    test_paths = sorted(
        {
            path
            for path in normalized
            if any(path == marker or path.startswith(marker) or f"/{marker}" in path for marker in TEST_PATH_MARKERS)
        }
    )
    project_markers = sorted({path for path in normalized if path.rsplit("/", 1)[-1] in PYTHON_PROJECT_MARKERS})
    python_files = sum(1 for path in normalized if path.endswith(".py"))
    return {
        "tree_path_count": len(paths),
        "python_file_count": python_files,
        "test_path_count": len(test_paths),
        "test_path_samples": test_paths[:20],
        "python_project_marker_count": len(project_markers),
        "python_project_marker_samples": project_markers[:20],
        "test_suite_path_evidence": bool(test_paths),
        "python_project_evidence": python_files > 0 and bool(project_markers),
    }


def enrich_repository(
    candidate: Dict[str, Any],
    split: str,
    protocol: Dict[str, Any],
    client: GitHubEnrichmentClient,
    *,
    pr_sample_limit: int,
) -> Dict[str, Any]:
    identity = candidate["repository_identity"]
    blockers: List[str] = []
    try:
        tree = _tree_evidence(client.get_tree_paths(identity, str(candidate["default_branch"])))
    except RuntimeError as exc:
        tree = {"error": str(exc)}
        blockers.append("tree_metadata_unavailable")
    window = _window(protocol, split)
    try:
        pulls = client.merged_pr_metadata(
            identity,
            start=window["start"],
            end=window["end"],
            limit=pr_sample_limit,
        )
    except RuntimeError as exc:
        pulls = {"issue_count": 0, "samples": [], "error": str(exc)}
        blockers.append("merged_pr_metadata_unavailable")
    if not tree.get("test_suite_path_evidence"):
        blockers.append("no_test_suite_path_evidence")
    if not tree.get("python_project_evidence"):
        blockers.append("no_python_project_path_evidence")
    if int(pulls.get("issue_count") or 0) <= 0:
        blockers.append("no_merged_pr_in_assigned_window")
    if not pulls.get("samples"):
        blockers.append("no_sampled_merge_commit_identity")
    return {
        "repository_identity": identity,
        "assigned_split": split,
        "window": window,
        "tree_evidence": tree,
        "merged_pr_evidence": pulls,
        "prose_fields_requested": False,
        "code_content_requested": False,
        "eligible_for_reproducibility_probe": not blockers,
        "eligible_for_frozen_repository_manifest": False,
        "blockers": sorted(set(blockers)),
        "next_gate": "Probe sampled parent/merge commit checkout reproducibility before freezing inclusion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich temporal-code repository candidates.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-repositories", type=int, default=30)
    parser.add_argument("--pr-sample-limit", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    args = parser.parse_args()

    discovery = __import__("64_discover_temporal_code_repositories")
    token, token_source = discovery.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    protocol = load_json(args.protocol)
    candidates = load_json(args.candidates)
    selected = select_balanced_candidates(candidates, max(1, args.max_repositories))
    assignments = candidates["preliminary_split_manifest"]["repositories"]
    client = GitHubEnrichmentClient(token, delay_seconds=max(0.0, args.delay_seconds))
    rows = []
    for candidate in selected:
        identity = candidate["repository_identity"]
        rows.append(
            enrich_repository(
                candidate,
                assignments[identity]["assigned_split"],
                protocol,
                client,
                pr_sample_limit=max(1, min(20, args.pr_sample_limit)),
            )
        )
        print(
            {
                "repository": identity,
                "split": rows[-1]["assigned_split"],
                "eligible_for_reproducibility_probe": rows[-1]["eligible_for_reproducibility_probe"],
                "blockers": rows[-1]["blockers"],
            }
        )
    report = {
        "schema_version": "temporal-code-repository-enrichment-report-v1",
        "protocol_name": protocol["protocol_name"],
        "authentication": {
            "authenticated": True,
            "source": token_source,
            "token_not_persisted_in_output": True,
        },
        "scope": "path_only_tree_and_prose_free_pr_metadata",
        "summary": {
            "repository_count": len(rows),
            "eligible_for_reproducibility_probe_count": sum(
                1 for row in rows if row["eligible_for_reproducibility_probe"]
            ),
            "frozen_repository_count": 0,
            "github_api_requests": client.requests,
        },
        "repositories": {row["repository_identity"]: row for row in rows},
        "next_gate": "Checkout reproducibility and benchmark-quarantine enrichment before frozen inclusion.",
    }
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
