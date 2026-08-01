#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from data_eval_common import load_json, sha256_file


ROOT: Final = Path(__file__).resolve().parents[1]
NATURAL_BUDGET_REPORT: Final = (
    ROOT / "outputs" / "validation" / "code_domain_natural_budget_current_framework_stage_c_summary_report.json"
)
EXTERNAL_CONFIRMATION_REPORT: Final = (
    ROOT / "outputs" / "validation" / "code_livecodebench_confirmation_summary_report.json"
)
STAGE_B_ARMS_REPORT: Final = (
    ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "stage_b_v2_arms_report.json"
)
IMPLEMENTATION_PATHS: Final = (
    Path("ingestion/code_chunks.py"),
    Path("ingestion/code_selection.py"),
)
REPORT_PATH: Final = ROOT / "outputs" / "validation" / "code_paper_evidence_report.json"
MD_REPORT_PATH: Final = ROOT / "outputs" / "validation" / "code_paper_evidence_report.md"


def main() -> int:
    natural = load_json(NATURAL_BUDGET_REPORT)
    external = load_json(EXTERNAL_CONFIRMATION_REPORT)
    stage_b_arms = load_json(STAGE_B_ARMS_REPORT)
    frozen_implementation = stage_b_arms.get("implementation_sha256") or {}
    current_implementation = {
        str(path).replace("\\", "/"): sha256_file(ROOT / path)
        for path in IMPLEMENTATION_PATHS
    }
    missing_frozen_hashes = [
        path for path in current_implementation if path not in frozen_implementation
    ]
    mismatched_hashes = [
        path
        for path, current_hash in current_implementation.items()
        if path in frozen_implementation and frozen_implementation[path] != current_hash
    ]
    implementation_matches = not missing_frozen_hashes and not mismatched_hashes
    evidence_status = (
        "code_paper_evidence_ready"
        if implementation_matches
        else "code_paper_evidence_stale_framework_rerun_required"
    )
    paper_decision = "pass" if implementation_matches else "historical_positive_rerun_required"
    raw = natural["arms"]["raw_full_natural"]
    curated = natural["arms"]["curated_v2_natural"]
    reduction = natural["natural_budget_reduction_curated_vs_raw"]
    deltas = natural["deltas_curated_minus_raw"]
    report = {
        "schema_version": "code-paper-evidence-report-v1",
        "status": evidence_status,
        "claim": (
            "code_positive_natural_budget_stage_c"
            if implementation_matches
            else "historical_code_positive_requires_current_framework_rerun"
        ),
        "framework_compatibility": {
            "current_artifacts_match": implementation_matches,
            "current_implementation_sha256": current_implementation,
            "frozen_implementation_sha256": frozen_implementation,
            "missing_frozen_implementation_hashes": missing_frozen_hashes,
            "mismatched_implementation_hashes": mismatched_hashes,
            "required_action": "none" if implementation_matches else "rerun_stage_a_stage_b_and_stage_c",
        },
        "protocol_id": natural["schema_version"],
        "protocol_lineage": {
            "source_report": str(NATURAL_BUDGET_REPORT.relative_to(ROOT)),
            "seed_scope": natural["seed_scope"],
            "raw_arm": "raw_full_natural",
            "curated_arm": "curated_v2_natural",
            "mixed_protocol_values_forbidden": True,
        },
        "utility_scope": "Stage C validation only; never selector objective",
        "nll": {
            "raw_mean_nll": raw["mean_nll"],
            "curated_mean_nll": curated["mean_nll"],
            "curated_minus_raw": deltas["mean_nll_lower_is_better"],
            "raw_packed_training_tokens": raw["packed_training_tokens"],
            "curated_packed_training_tokens": curated["packed_training_tokens"],
            "packed_token_reduction_fraction": reduction["packed_training_token_reduction_fraction"],
            "result": paper_decision,
        },
        "evalplus": {
            "raw_macro_pass_rate": raw["evalplus"]["macro_pass_rate"],
            "curated_macro_pass_rate": curated["evalplus"]["macro_pass_rate"],
            "curated_minus_raw": deltas["evalplus_macro_pass_rate_higher_is_better"],
            "evaluation_scope": "natural_budget_same_arms_same_seed_scope",
        },
        "external_confirmation": {
            "status": external["status"],
            "claim": external["claim"],
            "raw_mean_pass_rate": external["raw_mean_pass_rate"],
            "curated_mean_pass_rate": external["curated_mean_pass_rate"],
            "mean_pass_rate_delta": external["mean_pass_rate_delta"],
            "pooled_paired": external["pooled_paired"],
            "paper_use": external["interpretation"]["paper_use"],
        },
        "paper_table_row": {
            "domain": "Code",
            "budget": "natural",
            "protocol_id": natural["schema_version"],
            "raw_tokens": raw["packed_training_tokens"],
            "curated_tokens": curated["packed_training_tokens"],
            "token_reduction": round(reduction["packed_training_token_reduction_fraction"], 6),
            "raw_nll": round(raw["mean_nll"], 6),
            "curated_nll": round(curated["mean_nll"], 6),
            "evalplus_raw": round(raw["evalplus"]["macro_pass_rate"], 6),
            "evalplus_curated": round(curated["evalplus"]["macro_pass_rate"], 6),
            "decision": paper_decision,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Code Paper Evidence Report",
                "",
                f"Status: `{evidence_status}`",
                f"Decision: `{paper_decision}`",
                "",
                "| Metric | Raw | Curated | Delta |",
                "| --- | ---: | ---: | ---: |",
                f"| Packed training tokens | {raw['packed_training_tokens']} | {curated['packed_training_tokens']} | -{reduction['packed_training_token_reduction_fraction']:.1%} |",
                f"| Heldout NLL | {raw['mean_nll']:.6f} | {curated['mean_nll']:.6f} | {deltas['mean_nll_lower_is_better']:.6f} |",
                f"| EvalPlus macro pass rate | {raw['evalplus']['macro_pass_rate']:.4%} | {curated['evalplus']['macro_pass_rate']:.4%} | +{deltas['evalplus_macro_pass_rate_higher_is_better']:.4%} |",
                f"| LiveCodeBench confirmation pass@1 | {external['raw_mean_pass_rate']:.4%} | {external['curated_mean_pass_rate']:.4%} | +{external['mean_pass_rate_delta']:.4%} |",
                "",
                f"External transfer: `{external['claim']}`.",
                "Utility scope: Stage C validation only; never selector objective.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[code-paper-evidence] {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
