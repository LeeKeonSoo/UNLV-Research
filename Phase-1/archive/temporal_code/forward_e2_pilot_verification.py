#!/usr/bin/env python3
"""Verify parent-fail/merge-pass forward E2 task semantics in isolated Docker."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_RECIPES = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_e2_pilot_recipes.json"
DEFAULT_WORK_DIR = OUTPUT_DIR / "temporal_code_collection" / "forward_e2_pilot_work"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_pilot_report.json"
OUTPUT_TAIL_CHARACTERS = 4000
DOCKER_TRANSIENT_PATTERNS = (
    "Internal Server Error for API route",
    "dockerDesktopLinuxEngine",
    "docker_engine",
    "Cannot connect to the Docker daemon",
    "connection refused",
    "The pipe has been ended",
)
DOCKER_TRANSIENT_ATTEMPTS = 4


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.partial.json")


def _is_transient_docker_failure(result: Dict[str, Any]) -> bool:
    output = str(result.get("output_tail") or "")
    return result.get("exit_code") not in {0, None} and any(pattern in output for pattern in DOCKER_TRANSIENT_PATTERNS)


def _run_once(command: List[str], timeout: int) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
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
            "output_tail": output[-OUTPUT_TAIL_CHARACTERS:],
        }
    except subprocess.TimeoutExpired as exc:
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        return {
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(str(output).encode("utf-8", errors="replace")).hexdigest(),
            "output_tail": str(output)[-OUTPUT_TAIL_CHARACTERS:],
        }


def _run(command: List[str], timeout: int) -> Dict[str, Any]:
    result = _run_once(command, timeout)
    if command and command[0] == "docker":
        for attempt in range(1, DOCKER_TRANSIENT_ATTEMPTS):
            if not _is_transient_docker_failure(result):
                break
            time.sleep(min(60, 5 * (2**attempt)))
            result = _run_once(command, timeout)
    return result


def _dockerfile(recipe: Dict[str, Any], role: str) -> str:
    repository_url = shlex.quote(recipe["repository_url"])
    merge = shlex.quote(recipe["merge_commit"])
    parent = shlex.quote(recipe["parent_commit"])
    install = " ".join(shlex.quote(value) for value in recipe["install_arguments"])
    lines = [
        f"FROM {recipe['python_image']}",
        "ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1",
        "RUN apt-get update && apt-get install -y --no-install-recommends build-essential git "
        "&& rm -rf /var/lib/apt/lists/*",
        f"RUN git clone --filter=blob:none {repository_url} /workspace",
    ]
    if role == "merge":
        lines.append(f"RUN cd /workspace && git checkout --detach {merge}")
    else:
        overlay_commands = []
        for path in recipe["frozen_test_targets"]:
            quoted_path = shlex.quote(path)
            quoted_parent = shlex.quote(str(Path(path).parent).replace("\\", "/"))
            overlay_commands.append(
                f"mkdir -p /tmp/test-overlay/{quoted_parent} && "
                f"git -C /workspace show {merge}:{quoted_path} > /tmp/test-overlay/{quoted_path}"
            )
        lines.append(f"RUN cd /workspace && git checkout --detach {merge}")
        lines.append("RUN " + " && ".join(overlay_commands))
        lines.append(
            f"RUN cd /workspace && git checkout --detach {parent} && cp -a /tmp/test-overlay/. /workspace/"
        )
    lines.extend(["WORKDIR /workspace", f"RUN python -m pip install --no-cache-dir {install}", ""])
    return "\n".join(lines)


def _test_command(tag: str, recipe: Dict[str, Any], isolation: Dict[str, Any]) -> List[str]:
    arguments = " ".join(shlex.quote(value) for value in recipe["test_arguments"])
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", str(isolation["pids_limit"]),
        "--memory", str(isolation["memory"]), "--cpus", str(isolation["cpus"]),
        "--tmpfs", "/root:rw,noexec,nosuid,size=512m", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g",
        "--tmpfs", "/run/workspace:rw,nosuid,size=1g", tag, "sh", "-lc",
        f"cp -a /workspace/. /run/workspace/ && cd /run/workspace && exec python {arguments}",
    ]
    return command


def _build_report(
    payload: Dict[str, Any],
    recipes_path: Path,
    decisions: List[Dict[str, Any]],
    is_pilot: bool,
    status: str,
) -> Dict[str, Any]:
    candidate_count = int(payload["summary"]["candidate_count"])
    verified_count = sum(row["task_valid_e2"] for row in decisions)
    return {
        "schema_version": (
            "temporal-code-forward-e2-pilot-report-v1"
            if is_pilot
            else "temporal-code-forward-development-e2-batch-report-v1"
        ),
        "status": status,
        "source_sha256": {str(recipes_path): sha256_file(recipes_path)},
        "summary": {
            "metadata_candidate_count": candidate_count,
            "execution_candidate_count": len(decisions),
            "task_valid_e2_count": verified_count,
            "metadata_to_e2_yield": verified_count / candidate_count if candidate_count else 0.0,
            "execution_to_e2_yield": verified_count / len(decisions) if decisions else 0.0,
            "pilot_tasks_evaluation_authorized_count": 0,
            "development_tasks_authorized_pending_quarantine_count": verified_count if not is_pilot else 0,
        },
        "decisions": decisions,
        "decision": {
            "development_utility_may_start": False,
            "pilot_tasks_may_enter_development_or_confirmatory": False,
            "next_action": (
                "use pilot yield only to plan forward acquisition capacity"
                if is_pilot
                else "merge E2 decisions into development readiness only after contamination quarantine"
            ),
        },
        "confirmatory_outcomes_read": False,
        "utility_scope": payload["utility_scope"],
        "claim_boundary": payload["claim_boundary"],
    }


def _load_checkpoint(checkpoint_path: Path, recipes_path: Path) -> List[Dict[str, Any]]:
    if not checkpoint_path.exists():
        return []
    checkpoint = load_json(checkpoint_path)
    source = checkpoint.get("source_sha256") or {}
    if source.get(str(recipes_path)) != sha256_file(recipes_path):
        raise ValueError(f"Checkpoint source does not match frozen recipe: {checkpoint_path}")
    return list(checkpoint.get("decisions") or [])


def verify(recipes_path: Path, work_dir: Path, output_path: Path, checkpoint_path: Path | None = None) -> Dict[str, Any]:
    payload = load_json(recipes_path)
    accepted_statuses = {
        "frozen_before_forward_pilot_execution",
        "forward_development_recipe_batch_frozen_before_execution",
    }
    if payload["status"] not in accepted_statuses:
        raise ValueError("Forward recipes must be frozen before execution.")
    is_pilot = payload["status"] == "frozen_before_forward_pilot_execution"
    isolation = payload["contract"]["isolation_contract"]
    engine = _run(["docker", "version", "--format", "{{.Server.Version}}"], 30)
    if engine["exit_code"] != 0:
        raise RuntimeError(
            "Docker Linux backend is required; host execution is forbidden. "
            f"Last output: {engine['output_tail'][-500:]}"
        )
    checkpoint = checkpoint_path or _checkpoint_path(output_path)
    decisions = _load_checkpoint(checkpoint, recipes_path)
    completed = {row["repository_identity"] for row in decisions}
    for repository, recipe in payload["repository_recipes"].items():
        if repository in completed:
            print({"repository": repository, "task_valid_e2": "checkpoint"})
            continue
        roles = {}
        for role in ("merge", "parent_with_merge_tests"):
            tag = f"unlv-forward-e2:{hashlib.sha256(f'{repository}:{recipe[role.split('_')[0] + '_commit']}:{role}'.encode()).hexdigest()[:16]}"
            directory = work_dir / repository.replace("/", "__") / role
            directory.mkdir(parents=True, exist_ok=True)
            dockerfile = directory / "Dockerfile"
            dockerfile.write_text(_dockerfile(recipe, "merge" if role == "merge" else "parent"), encoding="utf-8")
            build = _run(["docker", "build", "--pull", "-t", tag, str(directory)], 1800)
            if _is_transient_docker_failure(build):
                raise RuntimeError(
                    "Transient Docker infrastructure failure after retries during build; "
                    "batch execution aborted before classifying repository validity."
                )
            test = _run(_test_command(tag, recipe, isolation), int(isolation["timeout_seconds"])) if build["exit_code"] == 0 else {
                "exit_code": None, "timed_out": False, "duration_seconds": 0, "output_sha256": None, "output_tail": ""
            }
            if _is_transient_docker_failure(test):
                raise RuntimeError(
                    "Transient Docker infrastructure failure after retries during test; "
                    "batch execution aborted before classifying repository validity."
                )
            roles[role] = {"image_tag": tag, "build": build, "test": test}
        merge_pass = roles["merge"]["build"]["exit_code"] == 0 and roles["merge"]["test"]["exit_code"] == 0
        parent_build = roles["parent_with_merge_tests"]["build"]["exit_code"] == 0
        parent_expected_fail = (
            parent_build
            and roles["parent_with_merge_tests"]["test"]["timed_out"] is False
            and roles["parent_with_merge_tests"]["test"]["exit_code"] not in {0, None}
        )
        verified = merge_pass and parent_expected_fail
        if verified:
            failure_stage = None
        elif roles["merge"]["build"]["exit_code"] != 0:
            failure_stage = "merge_build_failed"
        elif roles["merge"]["test"]["timed_out"]:
            failure_stage = "merge_test_timeout"
        elif roles["merge"]["test"]["exit_code"] != 0:
            failure_stage = "merge_test_failed"
        elif not parent_build:
            failure_stage = "parent_overlay_build_failed"
        elif roles["parent_with_merge_tests"]["test"]["timed_out"]:
            failure_stage = "parent_test_timeout"
        elif roles["parent_with_merge_tests"]["test"]["exit_code"] == 0:
            failure_stage = "parent_test_passed_not_discriminative"
        else:
            failure_stage = "unexpected_invalid_state"
        decisions.append(
            {
                "repository_identity": repository,
                "pull_request_number": recipe["pull_request_number"],
                "merge_pass": merge_pass,
                "parent_overlay_build_pass": parent_build,
                "parent_expected_test_failure": parent_expected_fail,
                "task_valid_e2": verified,
                "failure_stage": failure_stage,
                "roles": roles,
                "pilot_task_evaluation_authorized": False if is_pilot else None,
                "development_task_authorized_pending_quarantine": verified if not is_pilot else False,
            }
        )
        print({"repository": repository, "task_valid_e2": verified})
        partial_status = "forward_e2_infrastructure_pilot_in_progress" if is_pilot else "forward_development_e2_batch_in_progress"
        save_json(checkpoint, _build_report(payload, recipes_path, decisions, is_pilot, partial_status))
    report = _build_report(
        payload,
        recipes_path,
        decisions,
        is_pilot,
        "forward_e2_infrastructure_pilot_complete" if is_pilot else "forward_development_e2_batch_complete",
    )
    save_json(output_path, report)
    if checkpoint.exists() and checkpoint != output_path:
        checkpoint.unlink()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify forward E2 pilot task semantics.")
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-output", type=Path)
    args = parser.parse_args()
    report = verify(args.recipes, args.work_dir, args.output, args.checkpoint_output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
