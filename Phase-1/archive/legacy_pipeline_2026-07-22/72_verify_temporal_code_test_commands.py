#!/usr/bin/env python3
"""Verify frozen smoke test commands in constrained Docker containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_COMMANDS = Path("configs") / "temporal_code_smoke_test_commands_v1.json"
DEFAULT_BUNDLE_DIR = OUTPUT_DIR / "temporal_code_collection" / "smoke_bundles"
DEFAULT_WORK_DIR = OUTPUT_DIR / "temporal_code_collection" / "test_verification_work"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "smoke_test_command_verification.json"
OUTPUT_TAIL_CHARACTERS = 4000


def _output_tail(value: str) -> str:
    return value[-OUTPUT_TAIL_CHARACTERS:]


def _run(command: List[str], *, timeout: int, cwd: Path | None = None) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}"
        return {
            "exit_code": result.returncode,
            "timed_out": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest(),
            "output_tail": _output_tail(output),
        }
    except subprocess.TimeoutExpired as exc:
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        return {
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(str(output).encode("utf-8", errors="replace")).hexdigest(),
            "output_tail": _output_tail(str(output)),
        }


def _dockerfile(
    image: str,
    repository_url: str,
    commit: str,
    install_arguments: Iterable[str],
    post_install_command: Iterable[str] | None = None,
) -> str:
    install = " ".join(shlex.quote(value) for value in install_arguments)
    lines = [
        f"FROM {image}",
        "ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1",
        "RUN apt-get update && apt-get install -y --no-install-recommends build-essential git "
        "&& rm -rf /var/lib/apt/lists/*",
        f"RUN git clone --filter=blob:none {shlex.quote(repository_url)} /workspace "
        f"&& cd /workspace && git checkout --detach {shlex.quote(commit)}",
        "WORKDIR /workspace",
        f"RUN python -m pip install --no-cache-dir {install}",
    ]
    if post_install_command:
        lines.append(f"RUN {' '.join(shlex.quote(value) for value in post_install_command)}")
    lines.append("")
    return "\n".join(lines)


def _bundle_paths(bundle_dir: Path, eligible_bundle_ids: set[str] | None = None) -> Iterable[Path]:
    for path in sorted(bundle_dir.rglob("*.json")):
        if path.name not in {
            "smoke_fetch_report.json",
            "smoke_bundle_audit_report.json",
            "broad_tranche_fetch_report.json",
            "broad_tranche_bundle_audit_report.json",
            "path_stratified_tranche_fetch_report.json",
            "path_stratified_tranche_bundle_audit_report.json",
            "confirmatory_execution_expansion_audit_report.json",
            "development_execution_expansion_audit_report.json",
            "development_fresh_expansion_audit_report.json",
        }:
            if eligible_bundle_ids is None:
                yield path
                continue
            bundle = load_json(path)
            if bundle.get("bundle_id") in eligible_bundle_ids:
                yield path


def _execution_candidate_ids(audit: Dict[str, Any]) -> set[str]:
    result = set()
    for row in audit.get("decisions") or []:
        if "collection_gate_pass" in row or "executable_evaluation_blockers" in row:
            eligible = row.get("collection_gate_pass") is True and set(
                row.get("executable_evaluation_blockers") or []
            ) == {"test_command_not_verified"}
        else:
            eligible = set(row.get("blockers") or []) == {"test_command_not_verified"}
        if eligible:
            result.add(str(row["bundle_id"]))
    return result


def verify(
    commands: Dict[str, Any],
    bundle_paths: Iterable[Path],
    work_dir: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    if commands.get("status") not in {
        "frozen_before_execution",
        "refrozen_before_second_execution",
        "refrozen_before_third_execution",
        "refrozen_before_fourth_execution",
        "refrozen_before_fifth_execution",
        "refrozen_before_sixth_execution",
        "frozen_before_native_recipe_execution",
        "refrozen_before_second_native_execution",
    }:
        raise ValueError("Test commands must be frozen before execution.")
    isolation = commands["isolation_contract"]
    if isolation.get("host_execution_forbidden") is not True:
        raise ValueError("Host execution must remain forbidden.")
    if not dry_run:
        engine = _run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30)
        if engine["exit_code"] != 0:
            raise RuntimeError("Docker engine is required; host fallback is forbidden.")
    work_dir.mkdir(parents=True, exist_ok=True)
    decisions = []
    for path in bundle_paths:
        bundle = load_json(path)
        repository = bundle["repository_identity"]
        command = commands["repository_commands"].get(repository)
        if command is None:
            decisions.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "repository_identity": repository,
                    "status": "blocked",
                    "blockers": ["frozen_test_command_missing"],
                    "test_command_verified": False,
                }
            )
            continue
        commit_results = []
        for role, commit in (("parent", bundle["parent_commit"]), ("merge", bundle["merge_commit"])):
            image_tag = f"unlv-temporal-smoke:{hashlib.sha256(f'{repository}:{commit}'.encode()).hexdigest()[:16]}"
            dockerfile_dir = work_dir / bundle["bundle_id"] / role
            dockerfile_dir.mkdir(parents=True, exist_ok=True)
            dockerfile_path = dockerfile_dir / "Dockerfile"
            dockerfile_path.write_text(
                _dockerfile(
                    command.get("python_image", commands["python_image"]),
                    bundle["repository_url"],
                    commit,
                    command["install_arguments"],
                    command.get("post_install_command"),
                ),
                encoding="utf-8",
            )
            build_command = ["docker", "build", "--pull", "-t", image_tag, str(dockerfile_dir)]
            test_command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(isolation["pids_limit"]),
                "--memory",
                str(isolation["memory"]),
                "--cpus",
                str(isolation["cpus"]),
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
            ]
            for mount in [*isolation["common_writable_tmpfs"], *command.get("writable_tmpfs", [])]:
                test_command.extend(["--tmpfs", mount])
            if command.get("writable_workspace_copy") is True:
                test_command.extend(["--tmpfs", "/run/workspace:rw,noexec,nosuid,size=1g"])
                arguments = " ".join(shlex.quote(value) for value in command["test_arguments"])
                test_command.extend(
                    [
                        image_tag,
                        "sh",
                        "-lc",
                        f"cp -a /workspace/. /run/workspace/ && cd /run/workspace && exec python {arguments}",
                    ]
                )
            else:
                test_command.extend([image_tag, "python", *command["test_arguments"]])
            if dry_run:
                build_result = {"exit_code": None, "timed_out": False, "duration_seconds": 0, "output_sha256": None}
                test_result = dict(build_result)
            else:
                build_result = _run(build_command, timeout=1800)
                test_result = (
                    _run(test_command, timeout=int(isolation["timeout_seconds"]))
                    if build_result["exit_code"] == 0
                    else {"exit_code": None, "timed_out": False, "duration_seconds": 0, "output_sha256": None}
                )
            commit_results.append(
                {
                    "role": role,
                    "commit": commit,
                    "image_tag": image_tag,
                    "build": build_result,
                    "test": test_result,
                    "passed": build_result["exit_code"] == 0 and test_result["exit_code"] == 0,
                }
            )
        verified = all(row["passed"] for row in commit_results) and not dry_run
        decisions.append(
            {
                "bundle_id": bundle["bundle_id"],
                "repository_identity": repository,
                "status": "dry_run" if dry_run else ("passed" if verified else "failed"),
                "test_command": ["python", *command["test_arguments"]],
                "commit_results": commit_results,
                "blockers": [] if verified else ["isolated_parent_merge_test_command_not_verified"],
                "test_command_verified": verified,
            }
        )
    return {
        "schema_version": "temporal-code-smoke-test-verification-v1",
        "command_manifest_status": commands["status"],
        "dry_run": dry_run,
        "summary": {
            "bundle_count": len(decisions),
            "verified_bundle_count": sum(row["test_command_verified"] for row in decisions),
            "failed_or_unverified_bundle_count": sum(not row["test_command_verified"] for row in decisions),
            "build_failed_commit_count": sum(
                result["build"]["exit_code"] not in {0, None}
                for row in decisions
                for result in row.get("commit_results") or []
            ),
            "test_failed_commit_count": sum(
                result["build"]["exit_code"] == 0 and result["test"]["exit_code"] not in {0, None}
                for row in decisions
                for result in row.get("commit_results") or []
            ),
        },
        "decisions": decisions,
        "isolation_contract": isolation,
        "claim_boundary": commands["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frozen smoke test commands in Docker.")
    parser.add_argument("--commands", type=Path, default=DEFAULT_COMMANDS)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit",
        type=Path,
        help="Optional pre-execution audit; only bundles blocked solely by test_command_not_verified are run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.stem}.dry_run.json")
    eligible_bundle_ids = _execution_candidate_ids(load_json(args.audit)) if args.audit else None
    report = verify(
        load_json(args.commands),
        _bundle_paths(args.bundle_dir, eligible_bundle_ids),
        args.work_dir,
        dry_run=args.dry_run,
    )
    report["pre_execution_filter"] = {
        "audit_path": str(args.audit) if args.audit else None,
        "eligible_bundle_count": len(eligible_bundle_ids) if eligible_bundle_ids is not None else None,
        "rule": (
            "run only training-content-eligible bundles whose sole executable-evaluation blocker "
            "is test_command_not_verified"
        ),
    }
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
