#!/usr/bin/env python3
"""Probe sampled merge-commit reproducibility without fetching code content."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_smoke30.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "commit_reproducibility_report_smoke30.json"
GITHUB_GRAPHQL = "https://api.github.com/graphql"


class CommitIdentityClient:
    def __init__(self, token: str, *, delay_seconds: float = 0.2) -> None:
        self.token = token
        self.delay_seconds = delay_seconds
        self.requests = 0

    def commit_identities(self, repository_identity: str, shas: List[str]) -> Dict[str, Any]:
        owner, name = repository_identity.split("/", 1)
        aliases = []
        variables: Dict[str, Any] = {"owner": owner, "name": name}
        declarations = ["$owner: String!", "$name: String!"]
        for index, sha in enumerate(shas):
            key = f"sha{index}"
            declarations.append(f"${key}: GitObjectID!")
            variables[key] = sha
            aliases.append(
                f'c{index}: object(oid: ${key}) {{ ... on Commit {{ oid parents(first: 3) {{ nodes {{ oid }} }} }} }}'
            )
        query = (
            f"query({', '.join(declarations)}) {{ "
            f"repository(owner: $owner, name: $name) {{ {' '.join(aliases)} }} }}"
        )
        request = urllib.request.Request(
            GITHUB_GRAPHQL,
            data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
            method="POST",
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
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL errors: {payload['errors']}")
        repository = ((payload.get("data") or {}).get("repository") or {})
        return {
            sha: repository.get(f"c{index}")
            for index, sha in enumerate(shas)
        }


def probe_repository(row: Dict[str, Any], client: CommitIdentityClient) -> Dict[str, Any]:
    blockers: List[str] = []
    samples = row["merged_pr_evidence"].get("samples") or []
    merge_shas = [
        str((sample.get("mergeCommit") or {}).get("oid") or "")
        for sample in samples
        if isinstance(sample, dict)
    ]
    merge_shas = [sha for sha in merge_shas if sha]
    if not merge_shas:
        blockers.append("no_sampled_merge_commit_identity")
        identities: Dict[str, Any] = {}
    else:
        try:
            identities = client.commit_identities(row["repository_identity"], merge_shas)
        except RuntimeError as exc:
            identities = {}
            blockers.append("commit_identity_query_failed")
            query_error = str(exc)
        else:
            query_error = None
    checks = []
    for sha in merge_shas:
        commit = identities.get(sha)
        parents = ((commit or {}).get("parents") or {}).get("nodes") or []
        parent_oids = [str(parent.get("oid") or "") for parent in parents if isinstance(parent, dict)]
        exists = bool((commit or {}).get("oid"))
        has_parent = bool(parent_oids)
        checks.append(
            {
                "merge_commit": sha,
                "commit_identity_fetchable": exists,
                "parent_commit_identities": parent_oids,
                "parent_identity_fetchable": has_parent,
            }
        )
        if not exists:
            blockers.append("sampled_merge_commit_not_fetchable")
        if not has_parent:
            blockers.append("sampled_parent_commit_not_fetchable")
    return {
        "repository_identity": row["repository_identity"],
        "assigned_split": row["assigned_split"],
        "sampled_commit_checks": checks,
        "query_error": query_error,
        "prose_fields_requested": False,
        "code_content_requested": False,
        "eligible_for_quarantine_review": not blockers,
        "eligible_for_frozen_repository_manifest": False,
        "blockers": sorted(set(blockers)),
        "next_gate": "Benchmark quarantine and license/content-type review before frozen inclusion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe temporal-code commit identities.")
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-repositories", type=int, default=30)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    args = parser.parse_args()

    discovery = __import__("64_discover_temporal_code_repositories")
    token, token_source = discovery.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    enrichment = load_json(args.enrichment)
    selected = [
        row
        for row in enrichment["repositories"].values()
        if row["eligible_for_reproducibility_probe"]
    ][: max(1, args.max_repositories)]
    client = CommitIdentityClient(token, delay_seconds=max(0.0, args.delay_seconds))
    rows = []
    for row in selected:
        result = probe_repository(row, client)
        rows.append(result)
        print(
            {
                "repository": result["repository_identity"],
                "eligible_for_quarantine_review": result["eligible_for_quarantine_review"],
                "blockers": result["blockers"],
            }
        )
    report = {
        "schema_version": "temporal-code-commit-reproducibility-report-v1",
        "authentication": {
            "authenticated": True,
            "source": token_source,
            "token_not_persisted_in_output": True,
        },
        "scope": "commit_and_parent_identities_only_no_code_or_prose",
        "summary": {
            "repository_count": len(rows),
            "eligible_for_quarantine_review_count": sum(1 for row in rows if row["eligible_for_quarantine_review"]),
            "frozen_repository_count": 0,
            "github_api_requests": client.requests,
        },
        "repositories": {row["repository_identity"]: row for row in rows},
        "next_gate": "Benchmark quarantine and license/content-type review before frozen inclusion.",
    }
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
