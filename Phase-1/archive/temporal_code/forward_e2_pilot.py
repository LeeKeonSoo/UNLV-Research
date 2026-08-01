#!/usr/bin/env python3
"""Discover outcome-free recent PR candidates for the forward E2 infrastructure pilot."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_change import path_exclusion_reason


DEFAULT_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_e2_acquisition_plan.json"
DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_e2_pilot_candidates.json"
GITHUB_API = "https://api.github.com"
MAX_REQUEST_ATTEMPTS = 6
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
TRANSIENT_NETWORK_ERRORS = (
    http.client.RemoteDisconnected,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
    urllib.error.URLError,
)


def _is_test(path: str) -> bool:
    value = path.lower().replace("\\", "/")
    name = value.rsplit("/", 1)[-1]
    return value.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _classify(paths: List[str]) -> Dict[str, Any]:
    retained = [path for path in paths if path.lower().endswith(".py") and not path_exclusion_reason(path)]
    tests = sorted(path for path in retained if _is_test(path))
    code = sorted(path for path in retained if not _is_test(path))
    if tests and code:
        stratum = "code_and_test"
    elif tests:
        stratum = "test_only"
    elif code:
        stratum = "code_only"
    else:
        stratum = "no_allowed_paths"
    return {"path_stratum": stratum, "changed_test_paths": tests, "changed_code_paths": code}


class Client:
    def __init__(self, token: str, delay_seconds: float) -> None:
        self.token = token
        self.delay_seconds = delay_seconds
        self.requests = 0

    def _request(self, request: urllib.request.Request) -> Any:
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("User-Agent", "unlv-temporal-forward-e2/1.0")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                secondary_limit = exc.code == 403 and "secondary rate limit" in detail.lower()
                transient = secondary_limit or exc.code in TRANSIENT_HTTP_CODES
                if not transient or attempt == MAX_REQUEST_ATTEMPTS - 1:
                    raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail[:500]}") from exc
                retry_after = int(exc.headers.get("Retry-After") or min(120, 5 * (2**attempt)))
                time.sleep(max(1, retry_after))
            except TRANSIENT_NETWORK_ERRORS as exc:
                if attempt == MAX_REQUEST_ATTEMPTS - 1:
                    raise RuntimeError(f"GitHub API transient network failure after retries: {exc}") from exc
                time.sleep(min(60, 2 * (2**attempt)))
        self.requests += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return payload

    def recent_pulls(self, repository: str, start: str, end: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = """
        query($searchQuery: String!, $limit: Int!) {
          search(query: $searchQuery, type: ISSUE, first: $limit) {
            nodes {
              ... on PullRequest {
                number
                mergedAt
                mergeCommit { oid parents(first: 1) { nodes { oid } } }
              }
            }
          }
        }
        """
        variables = {"searchQuery": f"repo:{repository} is:pr is:merged merged:{start}..{end}", "limit": limit}
        request = urllib.request.Request(
            f"{GITHUB_API}/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
            method="POST",
        )
        payload = self._request(request)
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL errors: {payload['errors']}")
        return [row for row in (((payload.get("data") or {}).get("search") or {}).get("nodes") or []) if isinstance(row, dict)]

    def paths(self, repository: str, number: int) -> List[str]:
        request = urllib.request.Request(f"{GITHUB_API}/repos/{repository}/pulls/{number}/files?per_page=100")
        payload = self._request(request)
        if not isinstance(payload, list):
            raise RuntimeError("Pull file metadata must be a list.")
        return [str(row.get("filename") or "") for row in payload if isinstance(row, dict)]


def _repository_order(identities: List[str]) -> List[str]:
    return sorted(identities, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))


def discover(plan_path: Path, manifest_path: Path, output_path: Path, client: Client) -> Dict[str, Any]:
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    contract = plan["contract"]
    pilot = contract["infrastructure_pilot"]
    frame = _repository_order(plan["pilot_repository_frame"]["repository_identities"])[
        : int(pilot["maximum_metadata_repositories"])
    ]
    allowed = set(pilot["allowed_path_strata"])
    candidates = []
    repository_reports = []
    for repository in frame:
        errors = []
        pulls = []
        try:
            pulls = client.recent_pulls(repository, pilot["window_start"], pilot["window_end"])
        except RuntimeError as exc:
            errors.append(str(exc))
        selected = None
        for pull in pulls:
            merge = pull.get("mergeCommit") or {}
            parents = ((merge.get("parents") or {}).get("nodes") or [])
            try:
                paths = client.paths(repository, int(pull["number"]))
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            classification = _classify(paths)
            if classification["path_stratum"] not in allowed or not parents:
                continue
            selected = {
                "repository_identity": repository,
                "repository_url": manifest["repositories"][repository]["repository_url"],
                "license": manifest["repositories"][repository]["license"],
                "pull_request_number": int(pull["number"]),
                "merge_timestamp": pull["mergedAt"],
                "merge_commit": str(merge.get("oid") or ""),
                "parent_commit": str(parents[0].get("oid") or ""),
                **classification,
                "pilot_task_evaluation_authorized": False,
            }
            candidates.append(selected)
            break
        repository_reports.append(
            {"repository_identity": repository, "pull_count": len(pulls), "selected": selected is not None, "errors": errors}
        )
        print({"repository": repository, "pulls": len(pulls), "selected": selected is not None})
    candidates = sorted(candidates, key=lambda row: (row["repository_identity"], row["pull_request_number"]))
    report = {
        "schema_version": "temporal-code-forward-e2-pilot-candidates-v1",
        "status": "frozen_candidates_before_project_metadata_or_execution_outcomes",
        "source_sha256": {str(plan_path): sha256_file(plan_path), str(manifest_path): sha256_file(manifest_path)},
        "summary": {
            "repository_frame_count": len(frame),
            "candidate_count": len(candidates),
            "code_and_test_count": sum(row["path_stratum"] == "code_and_test" for row in candidates),
            "test_only_count": sum(row["path_stratum"] == "test_only" for row in candidates),
            "github_api_requests": client.requests,
        },
        "candidates": candidates,
        "repository_reports": repository_reports,
        "forbidden_fields_collected": [],
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover recent forward E2 pilot candidates.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()
    discovery = __import__("64_discover_temporal_code_repositories")
    token, _ = discovery.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    report = discover(args.plan, args.manifest, args.output, Client(token, max(0.0, args.delay_seconds)))
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
