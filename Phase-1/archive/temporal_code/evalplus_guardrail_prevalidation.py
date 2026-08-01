#!/usr/bin/env python3
"""Prevalidate EvalPlus reference and negative controls without model outcomes."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_evalplus_guardrail_prevalidation_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
DEFAULT_DOCKER_CONTEXT = Path("validation") / "docker" / "evalplus"


def _docker_daemon_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _run_docker_prevalidation(contract: Dict[str, Any], docker_context: Path) -> Dict[str, Any]:
    protocol = contract["isolated_execution_protocol"]
    image_tag = str(protocol["image_tag"])
    build = subprocess.run(
        ["docker", "build", "--pull", "-t", image_tag, str(docker_context)],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if build.returncode != 0:
        return {"ok": False, "phase": "build", "error": (build.stderr or build.stdout)[-4000:]}
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            str(protocol["memory_limit"]),
            "--pids-limit",
            str(protocol["pid_limit"]),
            image_tag,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if run.returncode != 0:
        return {"ok": False, "phase": "run", "error": (run.stderr or run.stdout)[-4000:]}
    lines = [line.strip() for line in run.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {"ok": False, "phase": "parse", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "phase": "complete",
        "image_tag": image_tag,
        "image_id": inspect.stdout.strip() if inspect.returncode == 0 else None,
        "datasets": payload["datasets"],
    }


def _evaluate_task(dataset: str, task: Dict[str, Any]) -> Dict[str, Any]:
    from evalplus.eval import PASS, untrusted_check
    from evalplus.gen.util import trusted_exec

    reference = str(task["prompt"]) + str(task["canonical_solution"])
    negative = f"def {task['entry_point']}(*args, **kwargs):\n    return None\n"
    controls = {}
    for test_name in ("base", "plus"):
        inputs = task[f"{test_name}_input"]
        expected, ref_time = trusted_exec(
            reference,
            inputs,
            task["entry_point"],
            record_time=True,
        )
        reference_status, _ = untrusted_check(
            dataset,
            reference,
            inputs,
            task["entry_point"],
            expected=expected,
            atol=task["atol"],
            ref_time=ref_time,
            fast_check=True,
        )
        negative_status, _ = untrusted_check(
            dataset,
            negative,
            inputs,
            task["entry_point"],
            expected=expected,
            atol=task["atol"],
            ref_time=ref_time,
            fast_check=True,
        )
        controls[test_name] = {
            "reference_status": reference_status,
            "negative_status": negative_status,
            "reference_pass": reference_status == PASS,
            "negative_rejected": negative_status != PASS,
        }
    return {
        "task_id": str(task["task_id"]),
        "base": controls["base"],
        "plus": controls["plus"],
        "reference_all_pass": all(row["reference_pass"] for row in controls.values()),
        "negative_rejected_by_any_suite": any(row["negative_rejected"] for row in controls.values()),
    }


def prevalidate(contract_path: Path, output_path: Path, docker_context: Path) -> Dict[str, Any]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    contract = load_json(contract_path)
    resource_module_available = importlib.util.find_spec("resource") is not None
    docker_available = _docker_daemon_available()
    loaders = {
        "HumanEval+": get_human_eval_plus,
        "MBPP+": get_mbpp_plus,
    }
    docker_result = _run_docker_prevalidation(contract, docker_context) if docker_available else {"ok": False}
    dataset_reports = {}
    if docker_result.get("ok"):
        for dataset_name, row in docker_result["datasets"].items():
            dataset_reports[dataset_name] = {
                **row,
                "smoke_task_ids": row.pop("task_ids"),
                "executed_task_count": int(row["smoke_task_count"]),
                "rows": [],
            }
    else:
        for dataset_name, dataset_contract in contract["datasets"].items():
            tasks = loaders[dataset_name]()
            selected_ids = sorted(tasks)[: int(dataset_contract["smoke_task_count"])]
            rows = (
                [_evaluate_task(dataset_name.lower(), tasks[task_id]) for task_id in selected_ids]
                if resource_module_available
                else []
            )
            dataset_reports[dataset_name] = {
                "available_task_count": len(tasks),
                "smoke_task_count": len(selected_ids),
                "smoke_task_ids": selected_ids,
                "executed_task_count": len(rows),
                "reference_control_pass_count": sum(row["reference_all_pass"] for row in rows),
                "negative_control_rejected_count": sum(row["negative_rejected_by_any_suite"] for row in rows),
                "rows": rows,
            }
    controls_executed = (bool(docker_result.get("ok")) or resource_module_available) and all(
        report["executed_task_count"] == report["smoke_task_count"] for report in dataset_reports.values()
    )
    controls_pass = controls_executed and all(
        report["reference_control_pass_count"] == report["smoke_task_count"]
        and report["negative_control_rejected_count"] == report["smoke_task_count"]
        for report in dataset_reports.values()
    )
    execution_tier = "E2" if controls_pass and docker_result.get("ok") else "E1"
    if docker_available and not docker_result.get("ok"):
        status = "isolated_backend_prevalidation_failed"
        required_next_step = f"fix frozen Docker prevalidation phase: {docker_result.get('phase')}"
    elif controls_pass and docker_result.get("ok"):
        status = "e2_prevalidated"
        required_next_step = "freeze the guardrail split and aggregate before model outcomes"
    elif not resource_module_available:
        status = "platform_runtime_blocked_before_semantic_controls"
        required_next_step = "run the frozen prevalidation in WSL2/Linux with an isolated execution backend"
    elif controls_pass and not docker_available:
        status = "semantic_controls_pass_isolation_blocked"
        required_next_step = "enable an isolated execution backend and repeat the frozen prevalidation"
    elif controls_pass:
        status = "e2_prevalidated"
        required_next_step = "freeze the guardrail split and aggregate before model outcomes"
    else:
        status = "semantic_controls_failed"
        required_next_step = "investigate failed reference or negative controls"
    report = {
        "schema_version": "temporal-code-evalplus-guardrail-prevalidation-report-v1",
        "status": status,
        "contract": contract,
        "source_sha256": {str(contract_path): sha256_file(contract_path)},
        "environment": {
            "evalplus_version": importlib.metadata.version("evalplus"),
            "platform": platform.platform(),
            "resource_module_available": resource_module_available,
            "docker_daemon_available": docker_available,
            "isolated_backend": "docker_linux" if docker_result.get("ok") else None,
            "isolated_image_tag": docker_result.get("image_tag"),
            "isolated_image_id": docker_result.get("image_id"),
            "docker_prevalidation_error": docker_result.get("error"),
            "model_generated_code_executed": False,
        },
        "datasets": dataset_reports,
        "decision": {
            "semantic_controls_executed": controls_executed,
            "semantic_controls_pass": controls_pass,
            "execution_support_tier": execution_tier,
            "may_enter_stage_c_guardrail": execution_tier == "E2",
            "may_replace_primary_temporal_executable_aggregate": False,
            "development_utility_may_start": False,
            "required_next_step": required_next_step,
        },
        "task_content_persisted": False,
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prevalidate EvalPlus guardrail controls.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docker-context", type=Path, default=DEFAULT_DOCKER_CONTEXT)
    args = parser.parse_args()
    report = prevalidate(args.contract, args.output, args.docker_context)
    print({"status": report["status"], "decision": report["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
