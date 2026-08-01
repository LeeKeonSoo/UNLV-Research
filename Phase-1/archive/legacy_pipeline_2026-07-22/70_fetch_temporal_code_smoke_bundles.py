#!/usr/bin/env python3
"""Fetch bounded smoke change bundles while excluding prose and hazardous file content."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import CHANGE_BUNDLE_SCHEMA_VERSION, generated_file_detection, path_exclusion_reason
from ingestion.normalize import detect_hazards


DEFAULT_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_smoke_fetch_plan.json"
DEFAULT_REPRODUCIBILITY = OUTPUT_DIR / "temporal_code_collection" / "commit_reproducibility_report_smoke30.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "smoke_bundles"
GITHUB_API = "https://api.github.com"


class GitHubContentClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requests = 0

    def get(self, path: str) -> Any:
        request = urllib.request.Request(f"{GITHUB_API}{path}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("User-Agent", "unlv-temporal-code-curation/1.0")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail[:500]}") from exc
        self.requests += 1
        return payload

    def file_text(self, repository: str, path: str, ref: str, maximum_bytes: int) -> Dict[str, Any]:
        encoded_path = urllib.parse.quote(path, safe="/")
        payload = self.get(f"/repos/{repository}/contents/{encoded_path}?ref={urllib.parse.quote(ref)}")
        if not isinstance(payload, dict):
            return {"text": None, "blocker": "content_not_found"}
        size = int(payload.get("size") or 0)
        if size > maximum_bytes:
            return {"text": None, "blocker": "file_too_large", "size": size}
        if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
            return {"text": None, "blocker": "non_base64_or_non_file_content", "size": size}
        raw = base64.b64decode(payload["content"])
        if b"\x00" in raw:
            return {"text": None, "blocker": "binary_content", "size": size}
        text = raw.decode("utf-8", errors="replace")
        hazards = detect_hazards(text, pii_context="repository_code")
        if hazards["secret_detected"] or hazards["pii_detected"]:
            return {"text": None, "blocker": "hazardous_content_not_persisted", "size": size, "hazards": hazards}
        return {"text": text, "blocker": None, "size": size, "hazards": hazards}


def _parent_map(reproducibility: Dict[str, Any], repository: str) -> Dict[str, str]:
    result = {}
    for check in reproducibility["repositories"][repository]["sampled_commit_checks"]:
        parents = check["parent_commit_identities"]
        if parents:
            result[check["merge_commit"]] = parents[0]
    return result


def _content_type(path: str) -> str:
    lower = path.lower().replace("\\", "/")
    if "/test" in lower or lower.startswith("test") or lower.startswith("tests/"):
        return "test"
    if lower.endswith((".md", ".rst", ".txt")):
        return "documentation"
    if lower.endswith(".py"):
        return "code"
    return "configuration"


def _repository_plans(plan: Dict[str, Any]):
    for split, value in plan["selected_repositories"].items():
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            yield split, row


def fetch(plan: Dict[str, Any], reproducibility: Dict[str, Any], client: GitHubContentClient, output_dir: Path) -> Dict[str, Any]:
    limits = plan["content_fetch_limits"]
    suffixes = tuple(limits["allowed_file_suffixes"])
    maximum_files = int(limits["maximum_changed_files_per_pull_request"])
    maximum_bytes = int(limits["maximum_file_bytes"])
    maximum_prs = int(limits["maximum_pull_requests_per_repository"])
    decisions = []
    for split, repository_plan in _repository_plans(plan):
        repository = repository_plan["repository_identity"]
        parents = _parent_map(reproducibility, repository)
        for sample in repository_plan["sampled_prs"][:maximum_prs]:
            number = int(sample["number"])
            merge_commit = str((sample.get("mergeCommit") or {}).get("oid") or "")
            parent_commit = parents.get(merge_commit)
            blockers = []
            if not parent_commit:
                blockers.append("parent_commit_not_available")
            file_rows: List[Dict[str, Any]] = []
            pull_files = client.get(f"/repos/{repository}/pulls/{number}/files?per_page={maximum_files}")
            for item in pull_files if isinstance(pull_files, list) else []:
                path = str(item.get("filename") or "")
                if not path.lower().endswith(suffixes):
                    continue
                if path_exclusion_reason(path):
                    continue
                status = str(item.get("status") or "modified")
                before = {"text": None, "blocker": "no_parent_commit"}
                if parent_commit and status != "added":
                    before = client.file_text(repository, str(item.get("previous_filename") or path), parent_commit, maximum_bytes)
                after = {"text": None, "blocker": "deleted_file"}
                if status != "removed":
                    after = client.file_text(repository, path, merge_commit, maximum_bytes)
                hazard_rows = [
                    value.get("hazards") or {}
                    for value in (before, after)
                    if value.get("blocker") == "hazardous_content_not_persisted"
                ]
                generated_detection = generated_file_detection(path, before.get("text"), after.get("text"))
                file_rows.append(
                    {
                        "path": path,
                        "change_type": {"added": "added", "removed": "deleted", "renamed": "renamed"}.get(
                            status, "modified"
                        ),
                        "content_type": _content_type(path),
                        "before_text": before.get("text"),
                        "after_text": after.get("text"),
                        "rights": {"status": "allowed", "license": repository_plan["license"]},
                        "generated": generated_detection["generated"],
                        "vendored": False,
                        "binary": before.get("blocker") == "binary_content" or after.get("blocker") == "binary_content",
                        "secret_detected": any(row.get("secret_detected") for row in hazard_rows),
                        "pii_detected": any(row.get("pii_detected") for row in hazard_rows),
                        "pii_detection_context": "repository_code",
                        "hazard_scan": {
                            "before": before.get("hazards"),
                            "after": after.get("hazards"),
                        },
                        "generated_detection_status": generated_detection["status"],
                        "generated_detection_evidence": generated_detection["evidence"],
                        "fetch_blockers": sorted(
                            {value["blocker"] for value in (before, after) if value.get("blocker")}
                        ),
                    }
                )
            bundle = {
                "schema_version": CHANGE_BUNDLE_SCHEMA_VERSION,
                "bundle_id": f"{repository.replace('/', '__')}__pr{number}",
                "repository_identity": repository,
                "repository_url": repository_plan["repository_url"],
                "repository_rights": {"status": "allowed", "license": repository_plan["license"]},
                "parent_commit": parent_commit or "",
                "merge_commit": merge_commit,
                "merge_timestamp": sample["mergedAt"],
                "provenance": {
                    "collector": "70_fetch_temporal_code_smoke_bundles.py",
                    "collector_version": "v1",
                    "collected_at": "2026-06-11T00:00:00+09:00",
                    "source_urls": [f"https://github.com/{repository}/pull/{number}"],
                },
                "execution_validation": {
                    "test_suite_present": True,
                    "test_command": "unverified",
                    "test_command_verified": False,
                    "parent_checkout_reproducible": bool(parent_commit),
                    "merge_checkout_reproducible": bool(merge_commit),
                },
                "prose": {"title": None, "body": None, "training_authorized": False},
                "files": file_rows,
                "content_signatures": [],
            }
            if not file_rows:
                blockers.append("no_allowed_text_files_fetched")
            save_json(output_dir / split / f"{bundle['bundle_id']}.json", bundle)
            decisions.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "split": split,
                    "files_persisted": len(file_rows),
                    "blockers": sorted(set(blockers)),
                    "stage0_release_candidate": False,
                }
            )
    plan_schema = plan.get("schema_version")
    broad_tranche = plan_schema == "temporal-code-broad-tranche-plan-v1"
    path_stratified_tranche = plan_schema == "temporal-code-path-stratified-tranche-plan-v1"
    bounded_tranche = broad_tranche or path_stratified_tranche
    report = {
        "schema_version": (
            "temporal-code-path-stratified-tranche-fetch-report-v1"
            if path_stratified_tranche
            else (
                "temporal-code-broad-tranche-fetch-report-v1"
                if broad_tranche
                else "temporal-code-smoke-fetch-report-v1"
            )
        ),
        "plan_status": plan["status"],
        "summary": {
            "bundle_count": len(decisions),
            "file_record_count": sum(row["files_persisted"] for row in decisions),
            "stage0_release_candidate_count": 0,
            "github_api_requests": client.requests,
        },
        "decisions": decisions,
        "claim_boundary": (
            "Fetched bounded-tranche bundles only; test commands are unverified and no bundle is a release candidate."
            if bounded_tranche
            else "Fetched smoke bundles only; test commands are unverified and no bundle is a release candidate."
        ),
    }
    report_name = (
        "path_stratified_tranche_fetch_report.json"
        if path_stratified_tranche
        else ("broad_tranche_fetch_report.json" if broad_tranche else "smoke_fetch_report.json")
    )
    save_json(output_dir / report_name, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch bounded temporal-code smoke bundles.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--reproducibility", type=Path, default=DEFAULT_REPRODUCIBILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    discovery = __import__("64_discover_temporal_code_repositories")
    token, _ = discovery.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    report = fetch(load_json(args.plan), load_json(args.reproducibility), GitHubContentClient(token), args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
