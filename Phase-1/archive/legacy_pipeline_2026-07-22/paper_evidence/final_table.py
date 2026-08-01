#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
VALIDATION_DIR: Final = ROOT / "outputs" / "validation"
REPORT_PATH: Final = VALIDATION_DIR / "final_paper_evidence_table.json"
MD_REPORT_PATH: Final = VALIDATION_DIR / "final_paper_evidence_table.md"


def _load(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def main() -> int:
    code = _load("outputs/validation/code_paper_evidence_report.json")
    external = _load("outputs/validation/code_livecodebench_confirmation_summary_report.json")
    code_framework_current = bool(code["framework_compatibility"]["current_artifacts_match"])
    code_base = _load("outputs/code_domain_natural_budget_qwen3_4b/heldout_nll/base_no_update.json")
    math = _load("outputs/validation/math_domain_selector_v3_stage_c_summary_report.json")
    humaneval_base = _load("outputs/code_domain_natural_budget_qwen3_4b/evalplus_guardrail/results/humaneval_base_no_update_base_eval.json")
    mbpp_base = _load("outputs/code_domain_natural_budget_qwen3_4b/evalplus_guardrail/results/mbpp_base_no_update_base_eval.json")
    math_arms = math["arms"]
    code_base_evalplus = (humaneval_base["pass_rate"] + mbpp_base["pass_rate"]) / 2.0

    rows = [
        {
            "domain": "Code",
            "arm": "base_no_update",
            "protocol_id": code["protocol_id"],
            "packed_training_tokens": 0,
            "mean_nll": code_base["mean_nll"],
            "evalplus_macro_pass_rate": code_base_evalplus,
            "decision": "reference_base",
        },
        {
            "domain": "Code",
            "arm": "raw_full_natural",
            "protocol_id": code["protocol_id"],
            "packed_training_tokens": code["nll"]["raw_packed_training_tokens"],
            "mean_nll": code["nll"]["raw_mean_nll"],
            "evalplus_macro_pass_rate": code["evalplus"]["raw_macro_pass_rate"],
            "decision": "reference_raw",
        },
        {
            "domain": "Code",
            "arm": "curated_v2_natural",
            "protocol_id": code["protocol_id"],
            "packed_training_tokens": code["nll"]["curated_packed_training_tokens"],
            "mean_nll": code["nll"]["curated_mean_nll"],
            "evalplus_macro_pass_rate": code["evalplus"]["curated_macro_pass_rate"],
            "decision": "pass" if code_framework_current else "historical_positive_rerun_required",
        },
        {
            "domain": "Math",
            "arm": "base_no_update",
            "protocol_id": math["schema_version"],
            "packed_training_tokens": 0,
            "mean_nll": math_arms["base_no_update"]["mean_nll"],
            "evalplus_macro_pass_rate": None,
            "decision": "reference_base",
        },
        {
            "domain": "Math",
            "arm": "raw_full_natural",
            "protocol_id": math["schema_version"],
            "packed_training_tokens": math_arms["raw_full_natural"]["packed_training_tokens"],
            "mean_nll": math_arms["raw_full_natural"]["mean_nll"],
            "evalplus_macro_pass_rate": None,
            "decision": "reference_raw",
        },
        {
            "domain": "Math",
            "arm": "curated_math_v2_natural",
            "protocol_id": math["schema_version"],
            "packed_training_tokens": math_arms["curated_math_v2_natural"]["packed_training_tokens"],
            "mean_nll": math_arms["curated_math_v2_natural"]["mean_nll"],
            "evalplus_macro_pass_rate": None,
            "decision": "fail",
        },
        {
            "domain": "Math",
            "arm": "curated_math_v3_natural",
            "protocol_id": math["schema_version"],
            "packed_training_tokens": math_arms["curated_math_v3_natural"]["packed_training_tokens"],
            "mean_nll": math_arms["curated_math_v3_natural"]["mean_nll"],
            "evalplus_macro_pass_rate": None,
            "decision": "repair_only_abstain",
        },
    ]
    report = {
        "schema_version": "final-paper-evidence-table-v1",
        "status": (
            "final_paper_evidence_table_frozen"
            if code_framework_current
            else "final_paper_evidence_table_blocked_current_framework_rerun"
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "domain_decisions": {
            "Code": "pass" if code_framework_current else "rerun_required",
            "Math": "abstain",
            "Production": "blocked",
            "UniversalAllDomain": "not_supported",
        },
        "external_transfer": {
            "benchmark": "LiveCodeBench code_generation_lite",
            "status": external["status"],
            "claim": external["claim"],
            "raw_mean_pass_rate": external["raw_mean_pass_rate"],
            "curated_mean_pass_rate": external["curated_mean_pass_rate"],
            "mean_pass_rate_delta": external["mean_pass_rate_delta"],
            "pooled_paired": external["pooled_paired"],
        },
        "rows": rows,
        "interpretation": [
            (
                "Code curated uses fewer tokens than raw and improves NLL plus EvalPlus under the current framework."
                if code_framework_current
                else "Historical Code evidence is positive, but the current Stage-A implementation requires a full rerun."
            ),
            "The frozen multi-seed LiveCodeBench confirmation is externally inconclusive and is a limitation, not transfer-gain evidence.",
            "Math v2 over-filtered and failed; Math v3 repairs v2 but still does not beat raw natural-budget training.",
            "Base rows are references for whether fine-tuning changed the model; raw-vs-curated remains the primary curation comparison.",
            "The paper claim is a deployment-conditioned curation-stage framework claim, not a universal all-domain improvement or production claim.",
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Final Paper Evidence Table",
        "",
        f"Status: `{report['status']}`",
        "",
        "Domain decisions:",
        "",
        *[f"- `{domain}`: `{decision}`" for domain, decision in report["domain_decisions"].items()],
        "",
        "| Domain | Arm | Packed train tokens | Mean NLL | EvalPlus macro | Decision |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        evalplus = "" if row["evalplus_macro_pass_rate"] is None else f"{row['evalplus_macro_pass_rate']:.4%}"
        lines.append(
            f"| {row['domain']} | {row['arm']} | {row['packed_training_tokens']} | {row['mean_nll']:.6f} | {evalplus} | {row['decision']} |"
        )
    lines.extend(["", "Utility scope: Stage C validation only; never selector objective.", ""])
    lines.extend(
        (
            "## External Transfer",
            "",
            f"LiveCodeBench status: `{report['external_transfer']['status']}`",
            f"Raw mean pass@1: `{report['external_transfer']['raw_mean_pass_rate']:.4%}`",
            f"Curated mean pass@1: `{report['external_transfer']['curated_mean_pass_rate']:.4%}`",
            f"Mean delta: `{report['external_transfer']['mean_pass_rate_delta']:.4%}`",
            "",
        )
    )
    MD_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[final-paper-evidence-table] {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
