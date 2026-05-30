#!/usr/bin/env python3
"""Run Utility probe sensitivity/power sweeps.

This script does not change selector outputs. It repeatedly runs the official
Utility sensitivity audit under several probe budgets and holdout buckets, then
aggregates whether probe controls become interpretable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from data_eval_common import OUTPUT_DIR

DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "utility_probe_power_sweep.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "utility_probe_power_sweep.md"
DEFAULT_RUN_DIR = OUTPUT_DIR / "validation" / "utility_probe_power_sweep_runs"
REQUIRED_BASELINE_POLICY = "common_stageA_baseline_disjoint_from_all_sensitivity_arms"

SWEEP_PRESETS: Dict[str, Dict[str, Any]] = {
    "current_like_b0": {
        "train_token_budget": 12000,
        "eval_token_budget": 6000,
        "bootstrap_rounds": 80,
        "max_train_steps": 96,
        "train_epochs": 1.0,
        "holdout_bucket": 0,
        "seed": 17,
        "arm_pool_size": 20000,
    },
    "eval_power_b0": {
        "train_token_budget": 12000,
        "eval_token_budget": 12000,
        "bootstrap_rounds": 120,
        "max_train_steps": 96,
        "train_epochs": 1.0,
        "holdout_bucket": 0,
        "seed": 17,
        "arm_pool_size": 30000,
    },
    "train_eval_power_b0": {
        "train_token_budget": 24000,
        "eval_token_budget": 12000,
        "bootstrap_rounds": 120,
        "max_train_steps": 192,
        "train_epochs": 1.5,
        "holdout_bucket": 0,
        "seed": 17,
        "arm_pool_size": 30000,
    },
    "train_eval_power_b1": {
        "train_token_budget": 24000,
        "eval_token_budget": 12000,
        "bootstrap_rounds": 120,
        "max_train_steps": 192,
        "train_epochs": 1.5,
        "holdout_bucket": 1,
        "seed": 29,
        "arm_pool_size": 30000,
    },
    "stronger_probe_b0": {
        "train_token_budget": 36000,
        "eval_token_budget": 18000,
        "bootstrap_rounds": 160,
        "max_train_steps": 288,
        "train_epochs": 2.0,
        "holdout_bucket": 0,
        "seed": 41,
        "arm_pool_size": 40000,
    },
}


def _cmd_for_run(*, profile: str, dataset: str, preset_name: str, preset: Dict[str, Any], output_path: Path) -> List[str]:
    return [
        sys.executable,
        "14_run_utility_causal_diagnostics.py",
        "--profile",
        profile,
        "--datasets",
        dataset,
        "--arm-pool-size",
        str(int(preset["arm_pool_size"])),
        "--train-token-budget",
        str(int(preset["train_token_budget"])),
        "--eval-token-budget",
        str(int(preset["eval_token_budget"])),
        "--bootstrap-rounds",
        str(int(preset["bootstrap_rounds"])),
        "--max-train-steps",
        str(int(preset["max_train_steps"])),
        "--train-epochs",
        str(float(preset["train_epochs"])),
        "--holdout-bucket",
        str(int(preset["holdout_bucket"])),
        "--seed",
        str(int(preset["seed"])),
        "--output",
        str(output_path),
        "--no-update-artifacts",
    ]


def _load_run(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_payload = next(iter((payload.get("datasets") or {}).values()), {})
    sensitivity = dataset_payload.get("probe_sensitivity") or {}
    root = dataset_payload.get("root_cause_decision") or {}
    arms = {str(item.get("arm")): item for item in dataset_payload.get("arm_results") or []}
    protocol = payload.get("protocol") or {}
    protocol_policy = protocol.get("baseline_policy")
    arm_policies = sorted({str(item.get("baseline_policy") or "") for item in arms.values()})
    arm_fingerprints = sorted({str(item.get("baseline_uid_fingerprint") or "") for item in arms.values()})
    common_fingerprint = str(dataset_payload.get("common_baseline_uid_fingerprint") or "")
    compatible = (
        protocol_policy == REQUIRED_BASELINE_POLICY
        and arm_policies == [REQUIRED_BASELINE_POLICY]
        and bool(common_fingerprint)
        and arm_fingerprints == [common_fingerprint]
    )
    def arm_delta(name: str) -> float | None:
        value = (arms.get(name) or {}).get("delta_nll")
        return float(value) if value is not None else None
    def arm_mde(name: str) -> float | None:
        value = (arms.get(name) or {}).get("minimum_detectable_delta_nll_95")
        return float(value) if value is not None else None
    negative_arm = str(sensitivity.get("canonical_negative_control") or "")
    if not negative_arm:
        negative_arm = "corrupted_negative_control" if "corrupted_negative_control" in arms else "negative_control"
    return {
        "exists": True,
        "path": str(path),
        "compatible": bool(compatible),
        "compatibility_reason": (
            "ok"
            if compatible
            else "stale_or_invalid_baseline_policy; rerun with current 14_run_utility_causal_diagnostics.py"
        ),
        "profile": payload.get("profile"),
        "protocol": protocol,
        "baseline_policy": protocol_policy,
        "arm_baseline_policies": arm_policies,
        "common_baseline_uid_fingerprint": common_fingerprint,
        "arm_baseline_uid_fingerprints": arm_fingerprints,
        "probe_valid": bool(sensitivity.get("probe_valid")),
        "positive_gt_random": bool(sensitivity.get("positive_gt_random")),
        "random_gt_negative": bool(sensitivity.get("random_gt_negative")),
        "selected_gt_random": bool(sensitivity.get("selected_gt_random")),
        "root_cause": (root.get("primary_hypothesis") if isinstance(root, dict) else None),
        "selector_tuning_allowed": bool(root.get("selector_tuning_allowed")) if isinstance(root, dict) else False,
        "delta_nll_by_arm": sensitivity.get("delta_nll_by_arm") or {},
        "arm_mde": {name: arm_mde(name) for name in sorted(arms)},
        "positive_minus_random": None if arm_delta("positive_control") is None or arm_delta("stageA_random") is None else round(arm_delta("positive_control") - arm_delta("stageA_random"), 8),
        "random_minus_negative": None if arm_delta("stageA_random") is None or arm_delta(negative_arm) is None else round(arm_delta("stageA_random") - arm_delta(negative_arm), 8),
        "selected_minus_random": None if arm_delta("selected") is None or arm_delta("stageA_random") is None else round(arm_delta("selected") - arm_delta("stageA_random"), 8),
        "canonical_negative_control": negative_arm,
    }


def _decision(dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    existing = [payload for payload in dataset_results.values() if payload.get("exists")]
    incompatible = [payload for payload in existing if not payload.get("compatible")]
    completed = [payload for payload in existing if payload.get("compatible")]
    valid = [payload for payload in completed if payload.get("probe_valid")]
    selected_gt_random = [payload for payload in completed if payload.get("selected_gt_random")]
    stable_valid = len(valid) >= max(2, len(completed) // 2 + 1) if completed else False
    any_valid = bool(valid)
    return {
        "existing_runs": int(len(existing)),
        "compatible_runs": int(len(completed)),
        "incompatible_runs": int(len(incompatible)),
        "completed_runs": int(len(completed)),
        "probe_valid_runs": int(len(valid)),
        "selected_gt_random_runs": int(len(selected_gt_random)),
        "any_probe_valid": any_valid,
        "stable_probe_valid": bool(stable_valid),
        "recommended_next_action": (
            "Use this dataset for selector Utility evidence only after choosing a stable probe preset."
            if stable_valid
            else "Do not use this dataset as selector Utility evidence; probe/protocol sensitivity remains unresolved."
            if completed
            else "Existing sweep files are stale/incompatible; rerun with --force."
            if incompatible
            else "No completed sweep runs."
        ),
    }


def aggregate_results(*, run_dir: Path, datasets: Sequence[str], presets: Sequence[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": "utility-probe-power-sweep-v1",
        "purpose": "Determine whether Utility probe validity/signal power improves with larger budgets and alternate holdout buckets.",
        "run_dir": str(run_dir),
        "datasets": {},
    }
    for dataset in datasets:
        per_preset = {}
        for preset_name in presets:
            path = run_dir / f"{dataset}__{preset_name}.json"
            per_preset[preset_name] = _load_run(path)
        report["datasets"][str(dataset)] = {
            "runs": per_preset,
            "decision": _decision(per_preset),
        }
    return report


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Utility Probe Power Sweep",
        "",
        "This audit tests whether Utility probe validity improves by increasing train/eval budget or changing holdout bucket.",
        "",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        decision = payload.get("decision") or {}
        lines.extend([
            f"## {dataset}",
            "",
            f"- Existing runs: `{decision.get('existing_runs')}`",
            f"- Compatible runs: `{decision.get('compatible_runs')}`",
            f"- Incompatible/stale runs: `{decision.get('incompatible_runs')}`",
            f"- Probe-valid runs: `{decision.get('probe_valid_runs')}`",
            f"- Selected > random runs: `{decision.get('selected_gt_random_runs')}`",
            f"- Stable probe valid: `{decision.get('stable_probe_valid')}`",
            f"- Recommended action: {decision.get('recommended_next_action')}",
            "",
            "| Preset | Compatible | Baseline policy | Probe valid | Negative arm | Pos>Rand | Rand>Neg | Sel>Rand | Pos-Rand | Rand-Neg | Sel-Rand | Root cause |",
            "|---|---|---|---|---|---|---|---|---:|---:|---:|---|",
        ])
        for preset_name, run in (payload.get("runs") or {}).items():
            if not run.get("exists"):
                lines.append(f"| {preset_name} | missing | - | - | - | - | - | - | - | - | - | - |")
                continue
            if not run.get("compatible"):
                lines.append(
                    f"| {preset_name} | False | {run.get('baseline_policy') or '-'} | - | - | - | - | - | - | - | - | {run.get('compatibility_reason')} |"
                )
                continue
            lines.append(
                f"| {preset_name} | True | {run.get('baseline_policy') or '-'} | {run.get('probe_valid')} | {run.get('canonical_negative_control')} | {run.get('positive_gt_random')} | "
                f"{run.get('random_gt_negative')} | {run.get('selected_gt_random')} | "
                f"{float(run.get('positive_minus_random') or 0):+.8f} | "
                f"{float(run.get('random_minus_negative') or 0):+.8f} | "
                f"{float(run.get('selected_minus_random') or 0):+.8f} | {run.get('root_cause')} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/aggregate Utility probe power sweeps.")
    parser.add_argument("--profile", default="learnability_rescue_no_anti_collapse")
    parser.add_argument("--datasets", nargs="*", default=["tiny_textbooks"])
    parser.add_argument("--presets", nargs="*", default=list(SWEEP_PRESETS.keys()))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    presets = [name for name in args.presets if name in SWEEP_PRESETS]
    if not presets:
        raise RuntimeError(f"No valid presets requested. Available: {sorted(SWEEP_PRESETS)}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for dataset in args.datasets:
            for preset_name in presets:
                output_path = args.run_dir / f"{dataset}__{preset_name}.json"
                if output_path.exists() and not args.force:
                    existing = _load_run(output_path)
                    if existing.get("compatible"):
                        print(f"[19] skip existing: dataset={dataset} preset={preset_name} path={output_path}", flush=True)
                        continue
                    print(
                        f"[19] rerun stale/incompatible: dataset={dataset} preset={preset_name} "
                        f"reason={existing.get('compatibility_reason')}",
                        flush=True,
                    )
                cmd = _cmd_for_run(
                    profile=str(args.profile),
                    dataset=str(dataset),
                    preset_name=str(preset_name),
                    preset=SWEEP_PRESETS[preset_name],
                    output_path=output_path,
                )
                started = time.perf_counter()
                print(f"[19] run start: dataset={dataset} preset={preset_name}", flush=True)
                print("[19] command: " + " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
                print(f"[19] run done: dataset={dataset} preset={preset_name} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    report = aggregate_results(run_dir=args.run_dir, datasets=args.datasets, presets=presets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, args.md_output)
    print(f"[19] sweep json: {args.output}", flush=True)
    print(f"[19] sweep md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
