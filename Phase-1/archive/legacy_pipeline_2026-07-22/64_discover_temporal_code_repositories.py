#!/usr/bin/env python3
"""Discover metadata-only GitHub repository candidates for temporal code collection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import normalize_repository_identity
from ingestion.temporal_code_manifests import build_repository_split_manifest


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_BENCHMARK_SEED = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "repository_candidate_manifest.json"
GITHUB_API = "https://api.github.com"
LICENSE_QUERY_NAMES = {
    "Apache-2.0": "apache-2.0",
    "MIT": "mit",
    "BSD-2-Clause": "bsd-2-clause",
    "BSD-3-Clause": "bsd-3-clause",
    "ISC": "isc",
}
GH_CANDIDATE_PATHS = (
    Path(r"C:\Program Files\GitHub CLI\gh.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub CLI" / "gh.exe",
)


def build_search_queries(protocol: Dict[str, Any], *, min_stars: int, max_stars: int | None = None) -> List[str]:
    start = protocol["collection_contract"]["training_window"]["start"]
    queries = []
    star_clause = f"stars:{int(min_stars)}..{int(max_stars)}" if max_stars is not None else f"stars:>={int(min_stars)}"
    for license_name in protocol["collection_contract"]["allowed_licenses"]:
        query_license = LICENSE_QUERY_NAMES.get(str(license_name))
        if not query_license:
            continue
        queries.append(
            f"language:Python fork:false archived:false pushed:>={start} "
            f"{star_clause} license:{query_license}"
        )
    return queries


def resolve_github_token() -> tuple[str | None, str]:
    environment = os.environ.get("GITHUB_TOKEN")
    if environment:
        return environment, "environment"
    for path in GH_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        result = subprocess.run(
            [str(path), "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        token = result.stdout.strip()
        if result.returncode == 0 and token:
            return token, "github_cli"
    return None, "none"


def _benchmark_repository_patterns(seed: Dict[str, Any]) -> set[str]:
    return {
        normalize_repository_identity(pattern)
        for entry in seed.get("entries") or []
        for pattern in entry.get("repository_patterns") or []
        if normalize_repository_identity(pattern)
    }


def repository_candidate(item: Dict[str, Any], benchmark_patterns: set[str]) -> Dict[str, Any]:
    identity = normalize_repository_identity(str(item.get("full_name") or ""))
    license_info = item.get("license") if isinstance(item.get("license"), dict) else {}
    blockers = []
    if not identity or "/" not in identity:
        blockers.append("invalid_repository_identity")
    if item.get("fork") is True:
        blockers.append("fork")
    if item.get("archived") is True:
        blockers.append("archived")
    if identity in benchmark_patterns:
        blockers.append("benchmark_repository")
    if not license_info.get("spdx_id"):
        blockers.append("missing_license")
    return {
        "repository_identity": identity,
        "repository_url": item.get("html_url"),
        "api_url": item.get("url"),
        "default_branch": item.get("default_branch"),
        "license": license_info.get("spdx_id"),
        "stars": int(item.get("stargazers_count") or 0),
        "forks": int(item.get("forks_count") or 0),
        "open_issues": int(item.get("open_issues_count") or 0),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "pushed_at": item.get("pushed_at"),
        "metadata_only": True,
        "discovery_status": "metadata_discovered_pending_enrichment" if not blockers else "metadata_discovery_excluded",
        "eligible_for_metadata_enrichment": not blockers,
        "eligible_for_frozen_repository_manifest": False,
        "required_before_freeze": [
            "test suite or executable validation command confirmation",
            "merged pull-request availability in the assigned temporal window",
            "parent and merge commit reproducibility",
            "benchmark-quarantine enrichment",
            "license and content-type authorization review",
        ],
        "blockers": sorted(set(blockers)),
    }


class GitHubMetadataClient:
    def __init__(self, token: str | None, *, user_agent: str, delay_seconds: float) -> None:
        self.token = token
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.requests = 0

    def get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(f"{GITHUB_API}{path}?{query}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("User-Agent", self.user_agent)
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    remaining = int(response.headers.get("X-RateLimit-Remaining") or 1)
                    reset = int(response.headers.get("X-RateLimit-Reset") or 0)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                secondary_limit = exc.code == 403 and "secondary rate limit" in detail.lower()
                if not secondary_limit or attempt == 3:
                    raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail[:500]}") from exc
                retry_after = int(exc.headers.get("Retry-After") or 60 * (attempt + 1))
                time.sleep(max(1, retry_after))
        self.requests += 1
        if remaining <= 1 and reset:
            time.sleep(max(0, reset - int(time.time()) + 1))
        elif self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub API response must be an object.")
        return payload


def discover(
    protocol: Dict[str, Any],
    benchmark_seed: Dict[str, Any],
    client: GitHubMetadataClient,
    *,
    per_query: int,
    pages_per_query: int,
    min_stars: int,
    max_stars: int | None = None,
) -> Dict[str, Any]:
    benchmark_patterns = _benchmark_repository_patterns(benchmark_seed)
    candidates: Dict[str, Dict[str, Any]] = {}
    queries = build_search_queries(protocol, min_stars=min_stars, max_stars=max_stars)
    for query in queries:
        for page in range(1, pages_per_query + 1):
            payload = client.get_json(
                "/search/repositories",
                {"q": query, "sort": "updated", "order": "desc", "per_page": per_query, "page": page},
            )
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = repository_candidate(item, benchmark_patterns)
                if row["repository_identity"]:
                    candidates[row["repository_identity"]] = row
            if len(items) < per_query:
                break
    enrichment_candidates = [row for row in candidates.values() if row["eligible_for_metadata_enrichment"]]
    preliminary_split_manifest = build_repository_split_manifest(enrichment_candidates, protocol)
    return {
        "schema_version": "temporal-code-repository-candidate-manifest-v1",
        "protocol_name": protocol["protocol_name"],
        "metadata_only": True,
        "queries": queries,
        "github_api_requests": client.requests,
        "summary": {
            "candidate_count": len(candidates),
            "metadata_enrichment_candidate_count": len(enrichment_candidates),
            "excluded_repository_count": len(candidates) - len(enrichment_candidates),
            "frozen_repository_count": 0,
            "preliminary_split_counts": preliminary_split_manifest["split_counts"],
        },
        "candidates": dict(sorted(candidates.items())),
        "preliminary_split_manifest": preliminary_split_manifest,
        "next_gate": (
            "Enrich and review repository metadata before freezing inclusion. "
            "Do not fetch training content from this discovery manifest alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover metadata-only GitHub repositories for later enrichment.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--benchmark-seed", type=Path, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-query", type=int, default=20)
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--min-stars", type=int, default=20)
    parser.add_argument("--max-stars", type=int)
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    token, token_source = resolve_github_token()
    if not token and not args.allow_unauthenticated:
        raise SystemExit(
            "Authenticated GitHub access is required. Run `gh auth login --web`, set GITHUB_TOKEN, "
            "or explicitly supply --allow-unauthenticated for a non-freezable smoke run."
        )
    client = GitHubMetadataClient(token, user_agent="unlv-temporal-code-curation/1.0", delay_seconds=args.delay_seconds)
    report = discover(
        load_json(args.protocol),
        load_json(args.benchmark_seed),
        client,
        per_query=max(1, min(100, args.per_query)),
        pages_per_query=max(1, args.pages_per_query),
        min_stars=max(0, args.min_stars),
        max_stars=max(0, args.max_stars) if args.max_stars is not None else None,
    )
    report["authentication"] = {
        "authenticated": bool(token),
        "source": token_source,
        "token_not_persisted_in_output": True,
    }
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
