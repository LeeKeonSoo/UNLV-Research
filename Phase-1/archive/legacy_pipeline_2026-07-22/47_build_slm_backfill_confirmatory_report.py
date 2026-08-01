#!/usr/bin/env python3
"""Build the frozen coverage-backfill confirmatory report."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path(__file__).resolve().parent / "configs" / "slm_backfill_confirmatory_plan_qwen25_0p5b_fineweb.json"


def _optional(path: Path) -> Dict[str, Any]:
    return load_json(path) if path.exists() else {}


def _nll(payload: Dict[str, Any]) -> float | None:
    value = payload.get("mean_nll")
    return float(value) if isinstance(value, (int, float)) else None


def build_report(experiment_dir: Path, plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_dir = experiment_dir / "eval_results"
    rows: List[Dict[str, Any]] = []
    primary_deltas: List[float] = []
    secondary_deltas: List[float] = []
    for seed in plan.get("confirmatory_seeds") or []:
        seed = int(seed)
        backfilled_primary = _nll(_optional(eval_dir / f"confirm_seed{seed}_backfilled_interleaved50_confirmatory_broad_stageA_eval.json"))
        stagea_primary = _nll(_optional(eval_dir / f"confirm_seed{seed}_stageA_random_confirmatory_broad_stageA_eval.json"))
        backfilled_secondary = _nll(
            _optional(eval_dir / f"confirm_seed{seed}_backfilled_interleaved50_confirmatory_coverage_stratified_stageA_eval.json")
        )
        stagea_secondary = _nll(
            _optional(eval_dir / f"confirm_seed{seed}_stageA_random_confirmatory_coverage_stratified_stageA_eval.json")
        )
        primary_delta = (
            backfilled_primary - stagea_primary
            if backfilled_primary is not None and stagea_primary is not None
            else None
        )
        secondary_delta = (
            backfilled_secondary - stagea_secondary
            if backfilled_secondary is not None and stagea_secondary is not None
            else None
        )
        if primary_delta is not None:
            primary_deltas.append(primary_delta)
        if secondary_delta is not None:
            secondary_deltas.append(secondary_delta)
        rows.append(
            {
                "seed": seed,
                "complete": primary_delta is not None and secondary_delta is not None,
                "backfilled_primary_nll": backfilled_primary,
                "stageA_random_primary_nll": stagea_primary,
                "backfilled_minus_stageA_primary_nll": primary_delta,
                "backfilled_secondary_nll": backfilled_secondary,
                "stageA_random_secondary_nll": stagea_secondary,
                "backfilled_minus_stageA_secondary_nll": secondary_delta,
            }
        )
    completed = [row for row in rows if row["complete"]]
    primary_failure_observed = any(float(delta) >= 0.0 for delta in primary_deltas)
    all_complete = len(completed) == len(rows)
    all_primary_wins = all(float(delta) < 0.0 for delta in primary_deltas) and all_complete
    status = "incomplete"
    if primary_failure_observed:
        status = "confirmatory_primary_not_supported_stop_recommended"
    elif all_primary_wins:
        status = "confirmatory_direction_supported"
    summary = {
        "planned_fresh_seeds": len(rows),
        "completed_fresh_seeds": len(completed),
        "primary_backfilled_win_count": sum(1 for delta in primary_deltas if delta < 0.0),
        "primary_stageA_random_win_count": sum(1 for delta in primary_deltas if delta >= 0.0),
        "secondary_backfilled_win_count": sum(1 for delta in secondary_deltas if delta < 0.0),
        "primary_delta_mean": float(statistics.mean(primary_deltas)) if primary_deltas else None,
        "secondary_delta_mean": float(statistics.mean(secondary_deltas)) if secondary_deltas else None,
    }
    report = {
        "schema_version": "slm-backfill-confirmatory-report-v1",
        "status": status,
        "plan": str(plan_path),
        "primary_success_rule": plan.get("primary_success_rule"),
        "summary": summary,
        "seed_rows": rows,
        "interpretation": (
            "The frozen 50/50 backfill candidate is not supported as a broad Stage-A primary improvement. "
            "It improves the coverage-stratified secondary diagnostic on the first fresh seed, showing a "
            "distribution-dependent tradeoff rather than a universal release-policy win."
            if primary_failure_observed
            else "The confirmatory result is incomplete."
        ),
        "decision": (
            "Stop the remaining expensive confirmatory seed because the frozen all-fresh-seed primary success "
            "rule can no longer pass. Do not retune the ratio on these holdouts."
            if primary_failure_observed
            else "Complete remaining frozen runs."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": plan.get("claim_boundary"),
    }
    output_json = experiment_dir / "confirm_backfill_report.json"
    output_md = experiment_dir / "confirm_backfill_report.md"
    save_json(output_json, report)
    lines = [
        "# Coverage-Backfill Confirmatory Report",
        "",
        f"Status: `{status}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {value:.9f} |" if isinstance(value, float) else f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Fresh Seeds",
            "",
            "| Seed | Complete | Backfilled primary | Stage-A primary | Primary delta | Backfilled secondary | Stage-A secondary | Secondary delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        cells = []
        for key in (
            "backfilled_primary_nll",
            "stageA_random_primary_nll",
            "backfilled_minus_stageA_primary_nll",
            "backfilled_secondary_nll",
            "stageA_random_secondary_nll",
            "backfilled_minus_stageA_secondary_nll",
        ):
            value = row[key]
            cells.append(f"{value:.9f}" if isinstance(value, float) else "missing")
        lines.append(f"| {row['seed']} | {row['complete']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Interpretation", "", report["interpretation"], "", "## Decision", "", report["decision"], ""])
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frozen coverage-backfill confirmatory report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    report = build_report(args.experiment_dir, args.plan)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
