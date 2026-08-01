#!/usr/bin/env python3
"""Run outcome-free Stage-B structural saturation ablations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import select_stage_b


DEFAULT_STAGE_A = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_a_code_domain_v2_balanced"
    / "train"
    / "stage_a_pass.jsonl"
)
DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_ABLATION = Path("configs") / "temporal_code_redundancy_saturation_ablation_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_saturation_ablation_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_saturation_ablation_report.md"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    values = [float((row.get("stage_b_evidence") or {}).get(key) or 0.0) for row in rows]
    return round(mean(values), 6) if values else 0.0


def _summary(result: Dict[str, Any]) -> Dict[str, Any]:
    selected = result["selected"]
    selected_ids = {str(row["chunk_uid"]) for row in selected}
    return {
        "selected_ids": selected_ids,
        "selected_count": len(selected),
        "selected_token_proxy": int(result["selected_token_proxy"]),
        "mean_soft_redundancy_risk": _mean(selected, "soft_redundancy_risk"),
        "mean_structural_redundancy_risk": _mean(selected, "soft_structural_redundancy_risk"),
        "mean_structural_match_count": _mean(selected, "soft_structural_match_count"),
        "mean_code_quality_proxy": _mean(selected, "code_quality_proxy"),
        "mean_objective": _mean(selected, "stage_b_objective_score"),
        "content_type_counts": dict(
            sorted(Counter(str(row.get("content_type") or "unknown") for row in selected).items())
        ),
        "repository_count": len(
            {str(row.get("repository_identity") or "") for row in selected}
        ),
        "high_saturation_selected_count": sum(
            int((row.get("stage_b_evidence") or {}).get("soft_structural_match_count") or 0) >= 4
            for row in selected
        ),
        "concise_selected_count": sum(
            int((row.get("stage_b_evidence") or {}).get("token_proxy_count") or 0) < 80
            for row in selected
        ),
    }


def build(
    stage_a_path: Path,
    protocol_path: Path,
    ablation_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    records = list(_jsonl(stage_a_path))
    protocol = load_json(protocol_path)["stage_b_contract"]
    ablation = load_json(ablation_path)
    coverage = protocol["coverage_support"]
    results = {}
    for mode in ablation["arms"]:
        selection = select_stage_b(
            records,
            budget_fraction=float(protocol["budget"]["fraction"]),
            quality_weight=0.8,
            redundancy_weight=0.2,
            coverage_axes=[str(value) for value in coverage["axes"]],
            minimum_exemplars=int(coverage["minimum_exemplars_per_observed_value"]),
            baseline_seed=int(protocol["stage_a_random_baseline"]["seed"]),
            distribution_axes=[str(value) for value in coverage["distribution_axes"]],
            minimum_relative_token_share=float(coverage["minimum_relative_token_share"]),
            redundancy_search_mode=str(protocol["objective"]["redundancy_search_mode"]),
            structural_saturation_mode=mode,
        )
        results[mode] = _summary(selection)

    current_ids = results["binary_current"].pop("selected_ids")
    for name, row in results.items():
        selected_ids = row.pop("selected_ids", current_ids if name == "binary_current" else set())
        if name == "binary_current":
            selected_ids = current_ids
        row["overlap_with_current_count"] = len(selected_ids.intersection(current_ids))
        row["added_vs_current_count"] = len(selected_ids - current_ids)
        row["removed_vs_current_count"] = len(current_ids - selected_ids)
        row["jaccard_with_current"] = round(
            len(selected_ids.intersection(current_ids))
            / max(1, len(selected_ids.union(current_ids))),
            6,
        )

    report = {
        "schema_version": "redundancy-saturation-ablation-report-v1",
        "status": "redundancy_saturation_ablations_ready_for_proxy_training_decision",
        "claim_boundary": (
            "Outcome-free Stage-B feature and selection shift only. "
            "No Utility, benchmark, or target-model outcome is consumed."
        ),
        "stage_a_source": str(stage_a_path),
        "protocol": str(protocol_path),
        "ablation_contract": str(ablation_path),
        "input_record_count": len(records),
        "arms": results,
        "decision": (
            "Keep binary_current canonical. Select at most one count-sensitive arm for "
            "fixed-recipe proxy-model development after concise/test/coverage checks."
        ),
        "next_actions": [
            "compare lost and added examples for each count-sensitive arm",
            "reject arms that disproportionately remove concise tests or repositories",
            "freeze at most one proxy-training candidate before model outcomes",
            "run equal-token proxy-model ablation against binary_current and Stage-A random",
        ],
        "utility_scope": "Stage C only; not consumed by this report",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Redundancy Saturation Ablations",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "| Arm | Selected | Risk | Structural Risk | Match Count | High Saturation | Concise | Jaccard Current |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in report["arms"].items():
        lines.append(
            f"| `{name}` | `{row['selected_count']}` | `{row['mean_soft_redundancy_risk']}` | "
            f"`{row['mean_structural_redundancy_risk']}` | `{row['mean_structural_match_count']}` | "
            f"`{row['high_saturation_selected_count']}` | `{row['concise_selected_count']}` | "
            f"`{row['jaccard_with_current']}` |"
        )
    lines.extend(["", "## Decision", "", str(report["decision"]), "", "## Next Actions", ""])
    lines.extend([f"- {value}" for value in report["next_actions"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Redundancy saturation ablations.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.stage_a, args.protocol, args.ablation, args.output, args.md_output)
    print({"status": report["status"], "arms": report["arms"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
