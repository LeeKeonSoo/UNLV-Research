#!/usr/bin/env python3
"""Freeze deterministic repository-native execution recipes from project metadata."""

from __future__ import annotations

import argparse
import base64
import configparser
import json
import subprocess
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_CONTRACT = Path("configs") / "temporal_code_native_execution_recipe_v1.json"
DEFAULT_PLAN = COLLECTION / "temporal_code_development_execution_expansion_plan.json"
DEFAULT_GENERIC_COMMANDS = COLLECTION / "temporal_code_development_expansion_test_commands_v1.json"
DEFAULT_ENRICHMENT = COLLECTION / "repository_enrichment_report_broad499.json"
DEFAULT_OUTPUT = COLLECTION / "temporal_code_development_native_test_commands_v1.json"
GITHUB_API = "https://api.github.com"


def _resolve_token() -> str | None:
    discovery = __import__("64_discover_temporal_code_repositories")
    token, _ = discovery.resolve_github_token()
    return token


class GitHubMetadataClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requests = 0

    def text(self, repository: str, path: str, ref: str) -> str | None:
        encoded_path = urllib.parse.quote(path, safe="/")
        url = f"{GITHUB_API}/repos/{repository}/contents/{encoded_path}?ref={urllib.parse.quote(ref)}"
        request = urllib.request.Request(url)
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
            raise
        self.requests += 1
        if not isinstance(payload, dict) or payload.get("encoding") != "base64":
            return None
        raw = base64.b64decode(str(payload.get("content") or ""))
        if len(raw) > 262144 or b"\x00" in raw:
            return None
        return raw.decode("utf-8", errors="replace")


def _parents(path: str) -> Iterable[str]:
    current = PurePosixPath(path).parent
    while True:
        value = "" if str(current) == "." else str(current)
        yield value
        if not value:
            break
        current = current.parent


def _candidate_roots(markers: list[str]) -> set[str]:
    project_markers = {"pyproject.toml", "setup.cfg", "setup.py"}
    return {
        "" if str(PurePosixPath(path).parent) == "." else str(PurePosixPath(path).parent)
        for path in markers
        if PurePosixPath(path).name.lower() in project_markers
    } or {""}


def _choose_root(roots: set[str], targets: list[str]) -> str:
    def score(root: str) -> tuple[int, int, str]:
        prefix = f"{root}/" if root else ""
        covered = sum(target == root or target.startswith(prefix) for target in targets)
        return (-covered, len(PurePosixPath(root).parts), root)

    return min(roots, key=score)


def _join(root: str, name: str) -> str:
    return f"{root}/{name}" if root else name


def _optional_extra(metadata: Dict[str, str], priority: list[str]) -> tuple[str | None, str | None]:
    pyproject = metadata.get("pyproject.toml")
    if pyproject:
        try:
            payload = tomllib.loads(pyproject)
            extras = ((payload.get("project") or {}).get("optional-dependencies") or {})
            for name in priority:
                if name in extras:
                    return name, "pyproject_optional_dependencies"
        except Exception:
            pass
    setup_cfg = metadata.get("setup.cfg")
    if setup_cfg:
        parser = configparser.ConfigParser()
        try:
            parser.read_string(setup_cfg)
            section = "options.extras_require"
            for name in priority:
                if parser.has_option(section, name):
                    return name, "setup_cfg_extras_require"
        except Exception:
            pass
    return None, None


def _python_image(metadata: Dict[str, str]) -> tuple[str, str | None]:
    pyproject = metadata.get("pyproject.toml")
    requires_python = None
    if pyproject:
        try:
            payload = tomllib.loads(pyproject)
            requires_python = str((payload.get("project") or {}).get("requires-python") or "") or None
        except Exception:
            pass
    if requires_python:
        for minor in ("3.10", "3.11", "3.12", "3.13"):
            if f">={minor}" in requires_python or f"=={minor}" in requires_python or f"~={minor}" in requires_python:
                return f"python:{minor}-slim", requires_python
    return "python:3.11-slim", requires_python


def _recipe(
    repository: str,
    markers: list[str],
    targets: list[str],
    merge_commit: str,
    client: GitHubMetadataClient,
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    root = _choose_root(_candidate_roots(markers), targets)
    allowed = [str(value) for value in contract["allowed_metadata_files"]]
    metadata = {}
    for name in allowed:
        path = _join(root, name)
        text = client.text(repository, path, merge_commit)
        if text is not None:
            metadata[name] = text
    extra, extra_source = _optional_extra(
        metadata, [str(value) for value in contract["dependency_rule"]["structured_optional_dependency_priority"]]
    )
    project_spec = root or "."
    install_arguments = ["-e", f"{project_spec}[{extra}]" if extra else project_spec]
    source = extra_source or "editable_project_fallback"
    selected_requirement = None
    if not extra:
        for name in contract["dependency_rule"]["requirement_file_priority"]:
            if str(name) in metadata:
                selected_requirement = _join(root, str(name))
                install_arguments.extend(["-r", selected_requirement])
                source = "frozen_requirement_file"
                break
    install_arguments.append("pytest")
    python_image, requires_python = _python_image(metadata)
    return {
        "python_image": python_image,
        "requires_python": requires_python,
        "install_arguments": install_arguments,
        "test_arguments": [
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "-p",
            "no:cacheprovider",
            *targets,
        ],
        "frozen_test_targets": targets,
        "test_target_source": "preexisting_frozen_changed_test_paths",
        "project_root": root or ".",
        "dependency_source": source,
        "selected_optional_extra": extra,
        "selected_requirement_file": selected_requirement,
        "metadata_paths_read": sorted(_join(root, name) for name in metadata),
        "generic_execution_outcomes_read": False,
        "writable_tmpfs": [],
        "writable_workspace_copy": True,
    }


def freeze(
    contract_path: Path,
    plan_path: Path,
    generic_commands_path: Path,
    enrichment_path: Path,
    output_path: Path,
    client: GitHubMetadataClient,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    plan = load_json(plan_path)
    generic = load_json(generic_commands_path)
    enrichment = load_json(enrichment_path)
    commands = {}
    plan_rows = {
        row["repository_identity"]: row
        for rows in plan["selected_repositories"].values()
        for row in rows
    }
    for repository, generic_row in sorted(generic["repository_commands"].items()):
        row = plan_rows[repository]
        merge_commit = str((row["sampled_prs"][0].get("mergeCommit") or {}).get("oid") or "")
        markers = list(enrichment["repositories"][repository]["tree_evidence"]["python_project_marker_samples"])
        commands[repository] = _recipe(
            repository,
            markers,
            list(generic_row["frozen_test_targets"]),
            merge_commit,
            client,
            contract,
        )
    report = {
        "schema_version": "temporal-code-native-test-commands-v1",
        "status": "refrozen_before_second_native_execution",
        "contract": contract,
        "application_plan": str(plan_path),
        "application_role": (
            "initial_exploratory_development_application"
            if str(plan_path).replace("\\", "/") == str(contract["applies_to_plan"]).replace("\\", "/")
            else "repository_disjoint_fresh_development_transfer"
        ),
        "python_image": generic["python_image"],
        "repository_commands": commands,
        "isolation_contract": generic["isolation_contract"],
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(plan_path): sha256_file(plan_path),
            str(generic_commands_path): sha256_file(generic_commands_path),
            str(enrichment_path): sha256_file(enrichment_path),
        },
        "summary": {
            "repository_count": len(commands),
            "structured_optional_extra_count": sum(
                row["selected_optional_extra"] is not None for row in commands.values()
            ),
            "requirement_file_count": sum(
                row["selected_requirement_file"] is not None for row in commands.values()
            ),
            "editable_fallback_count": sum(
                row["dependency_source"] == "editable_project_fallback" for row in commands.values()
            ),
            "nondefault_python_image_count": sum(
                row["python_image"] != generic["python_image"] for row in commands.values()
            ),
            "writable_workspace_copy_count": sum(
                row["writable_workspace_copy"] is True for row in commands.values()
            ),
            "github_api_requests": client.requests,
        },
        "forbidden_inputs": contract["selection_forbids"],
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze repository-native development execution recipes.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--generic-commands", type=Path, default=DEFAULT_GENERIC_COMMANDS)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    token = _resolve_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    report = freeze(
        args.contract,
        args.plan,
        args.generic_commands,
        args.enrichment,
        args.output,
        GitHubMetadataClient(token),
    )
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
