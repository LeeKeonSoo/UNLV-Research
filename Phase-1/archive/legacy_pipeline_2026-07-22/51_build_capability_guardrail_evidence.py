#!/usr/bin/env python3
"""Merge provisional external retention evidence into FineWeb deployment evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json
from importlib import import_module


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "fineweb_capability_guardrail_evidence.json"


def _nll(path: Path) -> float:
    payload = load_json(path)
    value = payload.get("mean_nll")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing mean_nll: {path}")
    return float(value)


def build_evidence(experiment_dir: Path) -> Dict[str, Any]:
    base_builder = import_module("49_build_fineweb_deployment_evidence")
    evidence: Dict[str, Any] = base_builder.build_evidence(experiment_dir)
    eval_dir = experiment_dir / "eval_results"
    manifest = load_json(experiment_dir / "external_guardrails" / "external_guardrail_holdout_manifest.json")
    overlap_counts = manifest.get("exact_normalized_text_overlap_counts") or {}
    holdout_records = int(manifest.get("record_count") or 0)
    result_paths = {
        "base_no_update": eval_dir / "guardrail_base_wikitext103.json",
        "selected_only": eval_dir / "guardrail_selected_only_wikitext103.json",
        "coverage_backfilled": eval_dir / "guardrail_coverage_backfilled_wikitext103.json",
        "stageA_broad": eval_dir / "guardrail_stageA_broad_wikitext103.json",
    }
    external_nlls = {arm: _nll(path) for arm, path in result_paths.items()}
    base_nll = external_nlls["base_no_update"]
    target_eval = "confirmatory_coverage_stratified_stageA_eval"

    for arm, payload in evidence["arms"].items():
        evaluations = payload.setdefault("evaluations", {})
        evaluations["target_domain_eval"] = float(evaluations[target_eval])
        evaluations["general_capability_eval"] = external_nlls[arm]
        evaluations["forgetting_regression_eval"] = external_nlls[arm] - base_nll
        overlap_count = 0 if arm == "base_no_update" else int(overlap_counts.get(arm) or 0)
        evaluations["training_eval_exact_overlap_rate"] = (
            float(overlap_count) / holdout_records if holdout_records else None
        )

    evidence["schema_version"] = "deployment-evidence-v1"
    evidence["evidence_identity"] = "fineweb_edu_qwen25_0p5b_confirmatory_plus_wikitext_guardrail_v1"
    evidence["evidence_boundary"].update(
        {
            "external_guardrails_complete": False,
            "provisional_external_retention_complete": True,
            "external_holdout_manifest": str(
                experiment_dir / "external_guardrails" / "external_guardrail_holdout_manifest.json"
            ),
            "exact_training_eval_overlap_pass": bool(manifest.get("exact_overlap_pass")),
            "missing_for_deployment_claim": [
                "task-based general capability benchmarks",
                "near-duplicate benchmark contamination audit",
                "safety evaluation",
                "multiple fresh training seeds",
            ],
        }
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Build capability-guardrail deployment evidence.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evidence = build_evidence(args.experiment_dir)
    save_json(args.output, evidence)
    print({"evidence_identity": evidence["evidence_identity"], "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
