#!/usr/bin/env python3
"""Build the broad temporal-code tranche operational readiness report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_AUDIT = OUTPUT_DIR / "temporal_code_collection" / "broad_tranche_bundles" / "broad_tranche_bundle_audit_report.json"
DEFAULT_TESTS = OUTPUT_DIR / "temporal_code_collection" / "broad_tranche_test_verification.json"
DEFAULT_STAGE0 = OUTPUT_DIR / "temporal_code_collection" / "stage0_broad_tranche" / "stage0_smoke_report.json"
DEFAULT_STAGE_A = OUTPUT_DIR / "temporal_code_collection" / "stage_a_broad_tranche" / "stage_a_smoke_report.json"
DEFAULT_STAGE_B = OUTPUT_DIR / "temporal_code_collection" / "stage_b_broad_tranche" / "stage_b_smoke_report.json"
DEFAULT_INDEX = OUTPUT_DIR / "validation" / "temporal_code_stage_b_index_equivalence_broad_tranche.json"
DEFAULT_ABLATIONS = OUTPUT_DIR / "validation" / "temporal_code_broad_stage_b_ablations.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_broad_tranche_readiness.json"


def build(
    audit_path: Path,
    additional_executable_audit_path: Path | None,
    tests_path: Path,
    stage0_path: Path,
    stage_a_path: Path,
    stage_b_path: Path,
    index_path: Path,
    ablations_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    audit = load_json(audit_path)
    additional_executable_audit = (
        load_json(additional_executable_audit_path)
        if additional_executable_audit_path is not None and additional_executable_audit_path.exists()
        else {"decisions": []}
    )
    tests = load_json(tests_path)
    stage0 = load_json(stage0_path)
    stage_a = load_json(stage_a_path)
    stage_b = load_json(stage_b_path)
    index = load_json(index_path)
    ablations = load_json(ablations_path)
    train_coverage = stage_b["coverage_support"]["all_train_stage_a_pass"]
    content_counts = train_coverage["content_type"]
    bundle_counts = train_coverage["bundle_id"]
    train_count = int(stage_b["summary"]["input_train_stage_a_pass_chunks"])
    documentation_share = int(content_counts.get("documentation") or 0) / max(1, train_count)
    largest_bundle, largest_bundle_count = max(bundle_counts.items(), key=lambda item: item[1])
    largest_bundle_share = int(largest_bundle_count) / max(1, train_count)
    selected_redundancy = float(stage_b["core_proxy_comparison"]["selected"]["mean_soft_redundancy_risk"])
    baseline_redundancy = float(
        stage_b["core_proxy_comparison"]["stage_a_random_disjoint"]["mean_soft_redundancy_risk"]
    )
    executable_decisions = [
        *(audit.get("decisions") or []),
        *(additional_executable_audit.get("decisions") or []),
    ]
    executable_by_split = {
        split: sum(
            row.get("assigned_split") == split and row.get("executable_evaluation_gate_pass") is True
            for row in executable_decisions
        )
        for split in ("train", "development", "confirmatory")
    }
    blockers = []
    if sum(executable_by_split.values()) == 0:
        blockers.append("no_executable_evaluation_eligible_bundles")
    if executable_by_split["development"] == 0:
        blockers.append("no_executable_development_holdout")
    if executable_by_split["confirmatory"] == 0:
        blockers.append("no_executable_confirmatory_holdout")
    if documentation_share > 0.8:
        blockers.append("train_stage_a_pool_documentation_dominated")
    if largest_bundle_share > 0.5:
        blockers.append("train_stage_a_pool_single_bundle_dominated")
    if selected_redundancy > baseline_redundancy:
        blockers.append("selected_soft_redundancy_risk_exceeds_stage_a_random")
    full_arm = ablations["arms"]["full_selector"]
    quality_arm = ablations["arms"]["quality_only"]
    full_matches_quality_only = all(
        full_arm[key] == quality_arm[key]
        for key in (
            "selected_chunks",
            "selected_token_proxy",
            "mean_code_quality_proxy",
            "mean_soft_redundancy_risk",
            "selected_bundle_count",
        )
    )
    if full_matches_quality_only:
        blockers.append("full_selector_behavior_matches_quality_only_ablation")
    report = {
        "schema_version": "temporal-code-broad-tranche-readiness-v1",
        "status": "stage_b_operationally_valid_stage_c_not_ready" if blockers else "ready_for_stage_c_smoke",
        "summary": {
            "collection_gate_pass_bundles": audit["summary"]["collection_gate_pass_count"],
            "executable_evaluation_gate_pass_bundles": sum(executable_by_split.values()),
            "test_verification_candidates": tests["summary"]["bundle_count"],
            "test_verified_bundles": tests["summary"]["verified_bundle_count"],
            "executable_evaluation_gate_pass_by_split": executable_by_split,
            "stage0_release_candidate_records": stage0["summary"]["release_candidate_records"],
            "stage_a_pass_chunks": stage_a["summary"]["stage_a_pass_count"],
            "train_stage_a_pass_chunks": train_count,
            "stage_b_selected_chunks": stage_b["summary"]["selected_chunks"],
            "stage_b_random_chunks": stage_b["summary"]["stage_a_random_disjoint_chunks"],
            "baseline_to_selected_token_ratio": stage_b["summary"]["baseline_to_selected_token_ratio"],
        },
        "distribution_diagnostics": {
            "documentation_share": round(documentation_share, 6),
            "largest_bundle": largest_bundle,
            "largest_bundle_share": round(largest_bundle_share, 6),
            "selected_mean_soft_redundancy_risk": selected_redundancy,
            "stage_a_random_mean_soft_redundancy_risk": baseline_redundancy,
            "full_selector_matches_quality_only_ablation": full_matches_quality_only,
            "redundancy_only_mean_soft_redundancy_risk": ablations["arms"]["redundancy_only"][
                "mean_soft_redundancy_risk"
            ],
            "no_coverage_selected_bundle_count": ablations["arms"]["no_coverage_support"][
                "selected_bundle_count"
            ],
        },
        "engineering_contracts": {
            "stage_b_selected_and_baseline_disjoint": stage_b["summary"]["selected_and_baseline_disjoint"],
            "stage_b_all_observed_values_retained": stage_b["coverage_support"]["all_observed_values_retained"],
            "stage_b_all_distribution_floors_passed": stage_b["coverage_support"][
                "all_distribution_floors_passed"
            ],
            "indexed_all_pairs_equivalent": index["summary"]["passed"],
        },
        "stage_c_blockers": blockers,
        "next_actions": (
            [
                "Do not tune Stage B from this tranche result.",
                "Freeze Qwen3-4B Stage-C smoke budgets, seeds, and practical effect margin before outcomes.",
                "Construct selected, common disjoint Stage-A-random, and raw-random arms with the frozen target tokenizer.",
                "Run development Stage C before inspecting untouched confirmatory outcomes.",
            ]
            if not blockers
            else [
                "Do not tune Stage B from this tranche result.",
                "Freeze and run the existing Stage-B ablations on the broad train pool.",
                "Collect a larger frozen tranche or enforce a pre-collection change-type sampling frame before outcomes.",
                "Build executable-evaluation-eligible development and confirmatory holdouts separately from training-content eligibility.",
                "Run Qwen3-4B Stage C only after the intended code-versus-documentation claim is explicit.",
            ]
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Broad Stage-0/A/B operational behavior is demonstrated. Target-model benefit, executable Utility, "
            "and a representative code-corpus claim are not established."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build broad temporal-code tranche readiness.")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--additional-executable-audit",
        type=Path,
        help="Optional separately frozen holdout audit used only for executable-evaluation readiness.",
    )
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument("--stage0", type=Path, default=DEFAULT_STAGE0)
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--stage-b", type=Path, default=DEFAULT_STAGE_B)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--ablations", type=Path, default=DEFAULT_ABLATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.audit,
        args.additional_executable_audit,
        args.tests,
        args.stage0,
        args.stage_a,
        args.stage_b,
        args.index,
        args.ablations,
        args.output,
    )
    print({"status": report["status"], "stage_c_blockers": report["stage_c_blockers"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
