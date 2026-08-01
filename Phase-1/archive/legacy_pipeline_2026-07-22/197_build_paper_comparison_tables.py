#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

VALIDATION_DIR = OUTPUT_DIR / "validation"
DEFAULT_V2 = VALIDATION_DIR / "code_domain_v2_confirmatory_decision_report.json"
DEFAULT_STAGE_B_ABLATION = VALIDATION_DIR / "temporal_code_path_stratified_stage_b_ablations.json"
DEFAULT_REDUNDANCY_ABLATION = VALIDATION_DIR / "redundancy_saturation_ablation_report.json"
DEFAULT_OUTPUT = VALIDATION_DIR / "paper_comparison_tables.json"
DEFAULT_MD_OUTPUT = VALIDATION_DIR / "paper_comparison_tables.md"
DEFAULT_CSV_OUTPUT = VALIDATION_DIR / "paper_comparison_tables.csv"


def _load(path: Path) -> JsonMap:
    payload = load_json(path) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _source(path: Path) -> JsonMap:
    return {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}


def _as_map(value: JsonValue) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _number(value: JsonValue) -> float | int | None:
    return value if isinstance(value, (int, float)) else None


def _stage_c_arms(v2: JsonMap) -> JsonMap:
    summary = _as_map(v2.get("summary"))
    arms = _as_map(summary.get("arm_summaries"))
    return {
        name: {
            "seeds": _as_list(_as_map(row).get("seeds")),
            "mean_nll": _number(_as_map(row).get("mean_nll")),
            "sample_std_nll": _number(_as_map(row).get("sample_std_nll")),
            "min_nll": _number(_as_map(row).get("min_nll")),
            "max_nll": _number(_as_map(row).get("max_nll")),
        }
        for name, row in arms.items()
    }


def _stage_c_pairs(v2: JsonMap) -> JsonMap:
    gate = _as_map(_as_map(v2.get("summary")).get("nll_gate"))
    pairs = _as_map(gate.get("paired_deltas"))
    return {
        "curated_vs_stageA_random": {
            "baseline": "stageA_random_equal_budget",
            "mean_nll_reduction": _number(gate.get("curated_vs_stageA_random_mean_nll_reduction")),
            "margin_required": _number(gate.get("primary_margin_required_absolute_nll_reduction")),
            "margin_pass": gate.get("curated_vs_stageA_random_margin_pass") is True,
            "all_seed_direction_pass": gate.get("curated_vs_stageA_random_all_paired_seed_pass") is True,
            "paired_seed_deltas": _as_map(_as_map(pairs.get("stageA_random_minus_curated")).get("per_seed_delta")),
        },
        "curated_vs_raw_random": {
            "baseline": "raw_random_equal_budget",
            "mean_nll_reduction": _number(gate.get("curated_vs_raw_random_mean_nll_reduction")),
            "direction_pass": gate.get("curated_vs_raw_random_direction_pass") is True,
            "paired_seed_deltas": _as_map(_as_map(pairs.get("raw_random_minus_curated")).get("per_seed_delta")),
        },
        "curated_vs_known_high_quality": {
            "baseline": "known_high_quality_equal_budget",
            "mean_nll_reduction": _number(gate.get("known_high_quality_minus_curated_mean_nll")),
            "all_seed_direction_pass": _as_map(pairs.get("known_high_quality_minus_curated")).get(
                "all_seed_deltas_positive"
            )
            is True,
            "paired_seed_deltas": _as_map(_as_map(pairs.get("known_high_quality_minus_curated")).get("per_seed_delta")),
        },
    }


def _stage_b_ablation_rows(stage_b: JsonMap) -> JsonMap:
    arms = _as_map(stage_b.get("arms"))
    return {
        name: {
            "selected_chunks": _number(_as_map(row).get("selected_chunks")),
            "selected_token_proxy": _number(_as_map(row).get("selected_token_proxy")),
            "mean_code_quality_proxy": _number(_as_map(row).get("mean_code_quality_proxy")),
            "mean_soft_redundancy_risk": _number(_as_map(row).get("mean_soft_redundancy_risk")),
            "mean_stage_b_objective_score": _number(_as_map(row).get("mean_stage_b_objective_score")),
            "selected_bundle_count": _number(_as_map(row).get("selected_bundle_count")),
        }
        for name, row in arms.items()
    }


def _redundancy_rows(redundancy: JsonMap) -> JsonMap:
    arms = _as_map(redundancy.get("arms"))
    return {
        name: {
            "selected_count": _number(_as_map(row).get("selected_count")),
            "selected_token_proxy": _number(_as_map(row).get("selected_token_proxy")),
            "mean_soft_redundancy_risk": _number(_as_map(row).get("mean_soft_redundancy_risk")),
            "mean_code_quality_proxy": _number(_as_map(row).get("mean_code_quality_proxy")),
            "repository_count": _number(_as_map(row).get("repository_count")),
            "jaccard_with_current": _number(_as_map(row).get("jaccard_with_current")),
        }
        for name, row in arms.items()
    }


def build(output_path: Path, md_output_path: Path, csv_output_path: Path) -> JsonMap:
    v2 = _load(DEFAULT_V2)
    stage_b = _load(DEFAULT_STAGE_B_ABLATION)
    redundancy = _load(DEFAULT_REDUNDANCY_ABLATION)
    missing = [
        name
        for name, path in {
            "v2_confirmatory_decision": DEFAULT_V2,
            "stage_b_ablation": DEFAULT_STAGE_B_ABLATION,
            "redundancy_ablation": DEFAULT_REDUNDANCY_ABLATION,
        }.items()
        if not path.exists()
    ]
    stage_c_arm_table = _stage_c_arms(v2)
    stage_b_ablation_table = _stage_b_ablation_rows(stage_b)
    remaining = [] if not missing and stage_c_arm_table and stage_b_ablation_table else ["complete_table_sources"]
    report = {
        "schema_version": "paper-comparison-tables-v1",
        "status": "paper_comparison_tables_frozen" if not remaining else "paper_comparison_tables_blocked",
        "claim_boundary": {
            "scope": "Frozen paper tables for raw, Stage-A-random, curated, reference, and ablation comparisons.",
            "utility_scope": "Stage C only; never selector objective",
            "not_supported": "Ablation tables are outcome-free diagnostics and do not prove Utility by themselves.",
        },
        "summary": {
            "stage_c_arm_count": len(stage_c_arm_table),
            "ablation_arm_count": len(stage_b_ablation_table),
            "redundancy_ablation_arm_count": len(_as_map(redundancy.get("arms"))),
            "remaining_required_tables": remaining,
        },
        "stage_c_arm_table": stage_c_arm_table,
        "stage_c_pairwise_table": _stage_c_pairs(v2),
        "stage_b_ablation_table": stage_b_ablation_table,
        "redundancy_ablation_table": _redundancy_rows(redundancy),
        "sources": {
            "v2_confirmatory_decision": _source(DEFAULT_V2),
            "stage_b_ablation": _source(DEFAULT_STAGE_B_ABLATION),
            "redundancy_ablation": _source(DEFAULT_REDUNDANCY_ABLATION),
        },
        "missing_inputs": missing,
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_csv(report, csv_output_path)
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Paper Comparison Tables",
        "",
        f"Status: `{report.get('status')}`",
        f"Utility scope: `{_as_map(report.get('claim_boundary')).get('utility_scope')}`",
        "",
        "## Stage-C NLL Arms",
        "",
        "| Arm | Mean NLL | Std | Min | Max | Seeds |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm, row in _as_map(report.get("stage_c_arm_table")).items():
        item = _as_map(row)
        lines.append(
            f"| {arm} | {item.get('mean_nll')} | {item.get('sample_std_nll')} | "
            f"{item.get('min_nll')} | {item.get('max_nll')} | {', '.join(str(seed) for seed in _as_list(item.get('seeds')))} |"
        )
    lines.extend(["", "## Stage-C Pairwise Effects", "", "| Comparison | Mean NLL reduction | Pass |", "| --- | ---: | --- |"])
    for name, row in _as_map(report.get("stage_c_pairwise_table")).items():
        item = _as_map(row)
        passed = item.get("margin_pass") if "margin_pass" in item else item.get("direction_pass")
        lines.append(f"| {name} | {item.get('mean_nll_reduction')} | `{passed}` |")
    lines.extend(["", "## Stage-B Ablations", "", "| Arm | Selected | Tokens | Quality proxy | Redundancy risk | Objective |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for arm, row in _as_map(report.get("stage_b_ablation_table")).items():
        item = _as_map(row)
        lines.append(
            f"| {arm} | {item.get('selected_chunks')} | {item.get('selected_token_proxy')} | "
            f"{item.get('mean_code_quality_proxy')} | {item.get('mean_soft_redundancy_risk')} | "
            f"{item.get('mean_stage_b_objective_score')} |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(report: JsonMap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "name", "metric", "value"])
        writer.writeheader()
        for table_name in ["stage_c_arm_table", "stage_c_pairwise_table", "stage_b_ablation_table"]:
            for name, row in _as_map(report.get(table_name)).items():
                for metric, value in _as_map(row).items():
                    writer.writerow({"table": table_name, "name": name, "metric": metric, "value": value})


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze paper comparison tables.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    args = parser.parse_args()
    report = build(args.output, args.md_output, args.csv_output)
    print({"status": report.get("status"), "remaining": _as_map(report.get("summary")).get("remaining_required_tables")})
    return 0 if not report.get("missing_inputs") else 2


if __name__ == "__main__":
    raise SystemExit(main())
