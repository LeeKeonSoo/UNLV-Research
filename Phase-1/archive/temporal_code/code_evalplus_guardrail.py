#!/usr/bin/env python3
"""Run isolated Docker EvalPlus evaluation for generated code-domain samples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PREVALIDATION = OUTPUT_DIR / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_DOCKER_CONTEXT = Path("validation") / "docker" / "evalplus"
DATASET_SLUGS = ("humaneval", "mbpp")
PROJECT_DIR = Path(__file__).resolve().parents[2]


def _parse_csv(value: str | None, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def _ensure_image(prevalidation_path: Path, docker_context: Path, rebuild: bool) -> Dict[str, Any]:
    prevalidation = load_json(prevalidation_path)
    protocol = prevalidation["contract"]["isolated_execution_protocol"]
    image_tag = str(protocol["image_tag"])
    inspect = _run(["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"], timeout=30)
    if inspect.returncode == 0 and inspect.stdout.strip() and not rebuild:
        return {"image_tag": image_tag, "image_id": inspect.stdout.strip(), "rebuilt": False}
    build = _run(["docker", "build", "--pull", "-t", image_tag, str(docker_context)], timeout=1200)
    if build.returncode != 0:
        raise RuntimeError((build.stderr or build.stdout)[-4000:])
    inspect = _run(["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"], timeout=30)
    if inspect.returncode != 0:
        raise RuntimeError((inspect.stderr or inspect.stdout)[-4000:])
    return {"image_tag": image_tag, "image_id": inspect.stdout.strip(), "rebuilt": True}


def _result_path(results_dir: Path, sample_path: Path) -> Path:
    return results_dir / f"{sample_path.stem}_eval.json"


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _result_matches_sample(result_path: Path, sample_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        row = load_json(result_path)
    except json.JSONDecodeError:
        return False
    if row.get("status") != "evalplus_samples_evaluated":
        return False
    return int(row.get("task_count") or -1) == _jsonl_count(sample_path)


def evaluate_missing(
    prevalidation_path: Path,
    output_dir: Path,
    docker_context: Path,
    datasets: List[str],
    max_evals: int | None,
    rebuild_image: bool,
) -> Dict[str, Any]:
    image = _ensure_image(prevalidation_path, docker_context, rebuild_image)
    root = (output_dir / "evalplus_guardrail").resolve()
    samples_dir = root / "samples"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    runner_dir = (PROJECT_DIR / "validation" / "docker" / "evalplus").resolve()
    executed = []
    skipped = []
    sample_paths = sorted(
        path for path in samples_dir.glob("*.jsonl")
        if any(path.name.startswith(f"{slug}_") for slug in datasets)
    )
    for sample_path in sample_paths:
        result_path = _result_path(results_dir, sample_path)
        if _result_matches_sample(result_path, sample_path):
            skipped.append({"sample": str(sample_path), "status": "already_complete"})
            continue
        if max_evals is not None and len(executed) >= max_evals:
            continue
        dataset_slug = sample_path.name.split("_", 1)[0]
        container_sample = f"/work/samples/{sample_path.name}"
        container_result = f"/work/results/{result_path.name}"
        cmd = [
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
            "4g",
            "--pids-limit",
            "512",
            "-v",
            f"{root}:/work:rw",
            "-v",
            f"{runner_dir}:/runner:ro",
            "--entrypoint",
            "python",
            image["image_tag"],
            "/runner/evaluate_samples.py",
            "--dataset",
            dataset_slug,
            "--samples",
            container_sample,
            "--output",
            container_result,
        ]
        run = _run(cmd, timeout=3600)
        if run.returncode != 0:
            raise RuntimeError((run.stderr or run.stdout)[-4000:])
        result = load_json(result_path)
        executed.append(
            {
                "sample": str(sample_path),
                "result": str(result_path),
                "dataset": result["dataset"],
                "task_count": result["task_count"],
                "pass_rate": result["pass_rate"],
            }
        )
        print(json.dumps(executed[-1], sort_keys=True))
    summary = {
        "schema_version": "code-domain-evalplus-evaluate-missing-summary-v1",
        "status": "evalplus_evaluate_missing_completed",
        "image": image,
        "source_sha256": {str(prevalidation_path): sha256_file(prevalidation_path)},
        "executed": executed,
        "skipped": skipped,
        "remaining": [
            str(path)
            for path in sample_paths
            if not _result_matches_sample(_result_path(results_dir, path), path)
        ],
        "confirmatory_outcomes_read": False,
    }
    save_json(root / "evaluate_missing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EvalPlus Docker guardrail evaluation.")
    parser.add_argument("--prevalidation", type=Path, default=DEFAULT_PREVALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docker-context", type=Path, default=DEFAULT_DOCKER_CONTEXT)
    parser.add_argument("--datasets", default="humaneval,mbpp")
    parser.add_argument("--max-evals", type=int)
    parser.add_argument("--rebuild-image", action="store_true")
    args = parser.parse_args()
    evaluate_missing(
        args.prevalidation,
        args.output_dir,
        args.docker_context,
        _parse_csv(args.datasets, DATASET_SLUGS),
        args.max_evals,
        args.rebuild_image,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
