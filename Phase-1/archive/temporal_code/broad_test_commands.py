#!/usr/bin/env python3
"""Freeze automated broad-tranche test-command hypotheses before execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_POLICY = Path("configs") / "temporal_code_broad_test_command_policy_v1.json"
DEFAULT_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_tranche_plan.json"
DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_broad499.json"
DEFAULT_BUNDLE_DIR = OUTPUT_DIR / "temporal_code_collection" / "broad_tranche_bundles"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_test_commands_v1.json"


def _bundle_paths(bundle_dir: Path) -> Iterable[Path]:
    excluded = {
        "smoke_fetch_report.json",
        "smoke_bundle_audit_report.json",
        "broad_tranche_fetch_report.json",
        "broad_tranche_bundle_audit_report.json",
        "path_stratified_tranche_fetch_report.json",
        "path_stratified_tranche_bundle_audit_report.json",
        "confirmatory_execution_expansion_audit_report.json",
        "development_execution_expansion_audit_report.json",
        "development_fresh_expansion_audit_report.json",
    }
    return (path for path in sorted(bundle_dir.rglob("*.json")) if path.name not in excluded)


def _is_python_test_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return normalized.endswith(".py") and (
        "/test" in normalized or normalized.startswith("test") or name.startswith("test_")
    )


def freeze(
    policy_path: Path,
    plan_path: Path,
    enrichment_path: Path,
    bundle_dir: Path,
    output_path: Path,
) -> Dict[str, Any]:
    policy = load_json(policy_path)
    plan = load_json(plan_path)
    enrichment = load_json(enrichment_path)
    changed_test_paths: Dict[str, set[str]] = {}
    for path in _bundle_paths(bundle_dir):
        bundle = load_json(path)
        repository = bundle["repository_identity"]
        changed_test_paths.setdefault(repository, set()).update(
            row["path"] for row in bundle.get("files") or [] if _is_python_test_path(str(row.get("path") or ""))
        )
    repositories = {
        row["repository_identity"]
        for rows in plan["selected_repositories"].values()
        for row in rows
    }
    commands = {}
    maximum_targets = int(policy["maximum_test_targets"])
    for repository in sorted(repositories):
        enrichment_row = enrichment["repositories"][repository]
        targets = sorted(changed_test_paths.get(repository) or set())
        source = "changed_test_paths"
        if not targets:
            targets = sorted(
                path
                for path in enrichment_row["tree_evidence"]["test_path_samples"]
                if _is_python_test_path(path)
            )
            source = "enrichment_test_path_samples"
        targets = targets[:maximum_targets]
        arguments = list(policy["fallback_test_arguments"])
        arguments.extend(targets)
        commands[repository] = {
            "install_arguments": list(policy["install_arguments"]),
            "test_arguments": arguments,
            "test_target_source": source if targets else "pytest_discovery_fallback",
            "frozen_test_targets": targets,
            "writable_tmpfs": [],
        }
    report = {
        "schema_version": "temporal-code-broad-test-commands-v1",
        "status": "frozen_before_execution",
        "python_image": policy["python_image"],
        "repository_commands": commands,
        "isolation_contract": policy["isolation_contract"],
        "source_sha256": {
            str(policy_path): sha256_file(policy_path),
            str(plan_path): sha256_file(plan_path),
            str(enrichment_path): sha256_file(enrichment_path),
        },
        "summary": {
            "repository_count": len(commands),
            "changed_test_target_repository_count": sum(
                row["test_target_source"] == "changed_test_paths" for row in commands.values()
            ),
            "enrichment_test_target_repository_count": sum(
                row["test_target_source"] == "enrichment_test_path_samples" for row in commands.values()
            ),
            "pytest_discovery_fallback_repository_count": sum(
                row["test_target_source"] == "pytest_discovery_fallback" for row in commands.values()
            ),
        },
        "forbidden_inputs": policy["forbidden_inputs"],
        "claim_boundary": policy["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze broad-tranche automated test commands.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.policy, args.plan, args.enrichment, args.bundle_dir, args.output)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
