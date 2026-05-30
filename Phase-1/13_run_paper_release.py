#!/usr/bin/env python3
"""Run the paper-release pipeline with release-mode guardrails."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_CONFIG = PROJECT_DIR / "configs" / "paper_release.json"
LOG_DIR = PROJECT_DIR / "outputs" / "logs"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def preflight(profiles_path: Path) -> list[str]:
    payload = _load_json(profiles_path)
    errors: list[str] = []
    if payload.get("schema_version") != "curation-profiles-v2":
        errors.append(f"unsupported schema_version={payload.get('schema_version')!r}")

    profiles = payload.get("profiles") or {}
    if not profiles:
        errors.append("no profiles defined")
        return errors

    for profile_name, profile in profiles.items():
        prefix = f"{profile_name}: "
        stage_c = profile.get("stage_c_validation") or {}
        utility = profile.get("utility_probe") or {}
        selector = profile.get("selector") or {}
        runtime_limits = profile.get("runtime_limits")
        utility_mode = str(utility.get("mode") or "").strip().lower()
        model_name = str(utility.get("model_name") or "").strip()
        objective_weights = selector.get("objective_weights") or {}

        if runtime_limits:
            errors.append(prefix + "runtime_limits must be absent for paper release")
        if utility_mode != "full":
            errors.append(prefix + f"utility_probe.mode must be 'full', got {utility_mode!r}")
        if "synthetic" in model_name.lower() or utility_mode == "synthetic_smoke":
            errors.append(prefix + "synthetic utility probes are forbidden in paper release")
        if model_name != "sshleifer/tiny-gpt2":
            errors.append(prefix + f"canonical model must be sshleifer/tiny-gpt2, got {model_name!r}")
        if str(stage_c.get("evaluation_mode") or "") != "certification":
            errors.append(prefix + "stage_c_validation.evaluation_mode must be certification")
        if str(stage_c.get("certification_scope") or "") != "general_purpose":
            errors.append(prefix + "stage_c_validation.certification_scope must be general_purpose")
        if str(stage_c.get("utility_pass_statistic") or "") != "min":
            errors.append(prefix + "utility_pass_statistic must be min")
        if not bool(stage_c.get("require_utility_ci_gain_positive")):
            errors.append(prefix + "require_utility_ci_gain_positive must be true")
        if not bool(stage_c.get("require_utility_delta_nll_positive")):
            errors.append(prefix + "require_utility_delta_nll_positive must be true")
        if not bool(stage_c.get("enforce_ood_utility_pass")):
            errors.append(prefix + "enforce_ood_utility_pass must be true")
        if not bool(stage_c.get("compute_ood_utility_report")):
            errors.append(prefix + "compute_ood_utility_report must be true")
        if not bool(stage_c.get("enforce_coverage_backbone_pass")):
            errors.append(prefix + "enforce_coverage_backbone_pass must be true")
        if "utility" in objective_weights or "diagnostic_predictive_utility" in objective_weights:
            errors.append(prefix + "selector objective must not include Utility")

        seeds = _as_list(utility.get("seeds"))
        holdout_buckets = _as_list(utility.get("holdout_buckets"))
        ood_holdout_buckets = _as_list(utility.get("ood_holdout_buckets"))
        if len(set(seeds)) < 4:
            errors.append(prefix + "paper release requires at least four utility seeds")
        if len(set(holdout_buckets)) < 4:
            errors.append(prefix + "paper release requires at least four in-domain holdout buckets")
        if len(set(ood_holdout_buckets)) < 4:
            errors.append(prefix + "paper release requires at least four OOD holdout buckets")
        if int(utility.get("train_token_budget") or 0) < 192_000:
            errors.append(prefix + "train_token_budget must be at least 192000")
        if int(utility.get("eval_token_budget") or 0) < 48_000:
            errors.append(prefix + "eval_token_budget must be at least 48000")
        if int(utility.get("ood_eval_token_budget") or 0) < 48_000:
            errors.append(prefix + "ood_eval_token_budget must be at least 48000")

    return errors


def _run_step(command: list[str], log_file: Path) -> None:
    command_text = " ".join(command)
    print(f"\n[paper-release] run: {command_text}", flush=True)
    started = time.time()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[run] {command_text}\n")
        log.flush()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        elapsed = time.time() - started
        status = "done" if returncode == 0 else "failed"
        message = f"[{status}] returncode={returncode} elapsed_sec={elapsed:.1f}\n"
        print(message, end="", flush=True)
        log.write(message)
        log.flush()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-release mode.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Run the full release pipeline after preflight.")
    args = parser.parse_args()

    profiles_path = args.profiles.resolve()
    errors = preflight(profiles_path)
    if errors:
        print("[paper-release] preflight failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"[paper-release] preflight passed: {profiles_path}")
    print("[paper-release] mode: real small-LM, full scored datasets, certification, general-purpose OOD")
    if not args.execute:
        print("[paper-release] full run not started. Add --execute to run 03 -> 04 -> 05 -> 06 -> 08.")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"paper_release_{time.strftime('%Y%m%d_%H%M%S')}.log"
    (LOG_DIR / "latest_paper_release.log").write_text(str(log_file), encoding="utf-8")
    print(f"[paper-release] log: {log_file}")
    _run_step([sys.executable, "03_score_core_metrics.py"], log_file)
    _run_step([sys.executable, "04_generate_subsets.py", "--profiles", str(profiles_path)], log_file)
    _run_step([sys.executable, "05_build_dashboard.py"], log_file)
    _run_step([sys.executable, "06_validate_outputs.py"], log_file)
    _run_step([sys.executable, "08_build_metric_maturity_snapshot.py"], log_file)
    print(f"[paper-release] complete: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
