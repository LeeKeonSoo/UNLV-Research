#!/usr/bin/env python3
"""Synthesize target-effect power, trajectory, and holdout-shift diagnostics."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"


def _summary(values: Dict[int, float]) -> Dict[str, Any]:
    gains = list(values.values())
    return {
        "seed_count": len(gains),
        "mean": statistics.mean(gains),
        "sample_stdev": statistics.stdev(gains) if len(gains) > 1 else 0.0,
        "minimum": min(gains),
        "maximum": max(gains),
        "positive_seed_count": sum(1 for value in gains if value > 0.0),
        "by_seed": {str(seed): value for seed, value in sorted(values.items())},
    }


def build(experiment_dir: Path) -> Dict[str, Any]:
    power = load_json(experiment_dir / "target_effect_power_diagnostic.json")
    trajectory = load_json(experiment_dir / "retention_training_trajectory_diagnostic.json")
    holdout = load_json(experiment_dir / "target_holdout_shift_diagnostic.json")

    development = {
        int(row["seed"]): float(row["target_improvement_vs_matched_stageA"])
        for row in holdout["rows"]
        if row["holdout"] == "development_target"
    }
    for row in trajectory["summary"]["paired_rows"]:
        if int(row["step"]) == 128:
            development[int(row["seed"])] = float(row["target_improvement_vs_matched_stageA"])
    fresh_holdout = {
        int(row["seed"]): float(row["target_improvement_vs_matched_stageA"])
        for row in holdout["rows"]
        if row["holdout"] == "fresh_confirmatory_target"
    }
    shifts = [float(row["holdout_shift_delta"]) for row in holdout["by_seed"]]
    report = {
        "schema_version": "target-effect-stability-report-v1",
        "status": "small_positive_development_effect_not_robust_at_strict_zero_boundary",
        "evidence": {
            "paired_power": {
                "interpretation": power["interpretation"],
                "seed_rows": power["seed_rows"],
            },
            "train_blocks_identical": holdout["train_blocks_identical"],
            "development_target_step128": _summary(development),
            "fresh_target_cross_evaluation": _summary(fresh_holdout),
            "holdout_shift": {
                "mean": statistics.mean(shifts),
                "mean_absolute": statistics.mean(abs(value) for value in shifts),
                "minimum": min(shifts),
                "maximum": max(shifts),
                "sign_consistent_seed_count": sum(1 for row in holdout["by_seed"] if row["sign_consistent"]),
                "seed_count": len(shifts),
            },
            "development_trajectory_step_summary": trajectory["summary"]["step_summary"],
        },
        "conclusion": (
            "The paired evaluator can detect effects at the observed scale, and identical train blocks rule out "
            "block construction as the cause. The replay-aware candidate has a consistently small positive effect "
            "on the previously used development target, but the margin is close enough to zero that a fresh "
            "holdout and training-seed combination can cross the strict sign boundary."
        ),
        "next_protocol_decisions": [
            "Do not tune another recipe on the frozen confirmatory holdout.",
            "Predeclare a practical target-effect margin and a training-seed replication count for future candidates.",
            "Require distributionally distinct target holdouts and task-based outcomes before a release claim.",
            "Keep retention replay because it consistently addresses the observed external forgetting failure.",
        ],
        "framework_interpretation": (
            "This is a Stage-C rejection of an insufficiently robust release claim, not evidence that Stage-B "
            "curation or the framework is malfunctioning."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Diagnostic synthesis only; no deployment or certification claim.",
    }
    save_json(experiment_dir / "target_effect_stability_report.json", report)
    dev = report["evidence"]["development_target_step128"]
    fresh = report["evidence"]["fresh_target_cross_evaluation"]
    lines = [
        "# Target Effect Stability Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Evidence Summary",
        "",
        "| Evaluation | Seeds | Mean gain | Stdev | Positive seeds | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Development target, step 128 | {dev['seed_count']} | {dev['mean']:.9f} | "
            f"{dev['sample_stdev']:.9f} | {dev['positive_seed_count']} | "
            f"{dev['minimum']:.9f} | {dev['maximum']:.9f} |"
        ),
        (
            f"| Fresh target cross-evaluation | {fresh['seed_count']} | {fresh['mean']:.9f} | "
            f"{fresh['sample_stdev']:.9f} | {fresh['positive_seed_count']} | "
            f"{fresh['minimum']:.9f} | {fresh['maximum']:.9f} |"
        ),
        "",
        "## Conclusion",
        "",
        report["conclusion"],
        "",
        "## Next Protocol Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["next_protocol_decisions"])
    lines.extend(["", "## Framework Interpretation", "", report["framework_interpretation"], ""])
    (experiment_dir / "target_effect_stability_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build target-effect stability synthesis.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    args = parser.parse_args()
    report = build(args.experiment_dir)
    print({"status": report["status"], "development": report["evidence"]["development_target_step128"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
