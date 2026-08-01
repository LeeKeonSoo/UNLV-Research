#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "benchmark_sensitivity_protocol_v1.json"
EXTERNAL_REPORT_PATH = OUTPUT_DIR / "validation" / "code_livecodebench_confirmation_summary_report.json"
REPORT_PATH = OUTPUT_DIR / "validation" / "benchmark_sensitivity_protocol_report.json"
MD_REPORT_PATH = OUTPUT_DIR / "validation" / "benchmark_sensitivity_protocol_report.md"


def _as_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _blockers(config: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    isolation = _as_map(config.get("frozen_outcome_isolation"))
    primary = _as_map(config.get("equal_token_primary"))
    ci = _as_map(primary.get("paired_ci"))
    external = _as_map(config.get("external_transfer"))
    rules = _as_map(external.get("sample_size_rule"))
    if config.get("status") != "frozen_for_future_studies_after_existing_code_results":
        blockers.append("protocol_not_frozen_for_future_studies")
    if isolation.get("stage_b_policy_change_permitted") is not False:
        blockers.append("stage_b_policy_change_not_forbidden")
    if isolation.get("existing_outcomes_may_change_selector") is not False:
        blockers.append("existing_outcomes_may_change_selector")
    if int(primary.get("minimum_paired_training_seeds") or 0) < 5:
        blockers.append("insufficient_primary_seed_requirement")
    if ci.get("method") != "paired_bootstrap_percentile" or int(ci.get("bootstrap_resamples") or 0) < 10000:
        blockers.append("paired_ci_contract_incomplete")
    if not isinstance(primary.get("metric_families"), list) or len(primary["metric_families"]) < 2:
        blockers.append("primary_metric_families_incomplete")
    if int(rules.get("minimum_total_tasks") or 0) < 300 or int(rules.get("minimum_non_easy_tasks") or 0) < 200:
        blockers.append("external_transfer_power_rule_incomplete")
    if "using any Stage-C metric, CI, benchmark, or sensitivity outcome in Stage B" not in config.get("forbidden_uses", []):
        blockers.append("stage_c_selector_leakage_forbidden_use_missing")
    return blockers


def _external_transfer() -> dict[str, Any]:
    if not EXTERNAL_REPORT_PATH.exists():
        return {"exists": False, "current_status": None, "claim": None, "task_count": None}
    report = _as_map(load_json(EXTERNAL_REPORT_PATH))
    protocol = _as_map(report.get("protocol"))
    return {
        "exists": True,
        "current_status": report.get("status"),
        "claim": report.get("claim"),
        "task_count": protocol.get("task_count"),
        "base_pass_rate": protocol.get("base_no_update_pass_rate"),
    }


def build() -> dict[str, Any]:
    config = _as_map(load_json(CONFIG_PATH))
    blockers = _blockers(config)
    external = _external_transfer()
    external_contract = _as_map(config.get("external_transfer"))
    external["sample_size_rule"] = _as_map(external_contract.get("sample_size_rule"))
    external["below_powered_task_floor"] = bool(
        isinstance(external.get("task_count"), int)
        and external["task_count"] < int(external["sample_size_rule"].get("minimum_total_tasks") or 0)
    )
    status = "benchmark_sensitivity_protocol_frozen_external_transfer_inconclusive"
    if blockers:
        status = "benchmark_sensitivity_protocol_blocked"
    report = {
        "schema_version": "benchmark-sensitivity-protocol-report-v1",
        "status": status,
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "blockers": blockers,
        "frozen_outcome_isolation": config.get("frozen_outcome_isolation"),
        "equal_token_primary": config.get("equal_token_primary"),
        "natural_budget_supporting": config.get("natural_budget_supporting"),
        "external_transfer": external,
        "sensitivity_matrix": config.get("sensitivity_matrix"),
        "forbidden_uses": config.get("forbidden_uses"),
        "interpretation": "The current external neutral result remains a limitation. This contract freezes powered future sensitivity evidence without changing Stage B.",
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    external = _as_map(report.get("external_transfer"))
    primary = _as_map(report.get("equal_token_primary"))
    ci = _as_map(primary.get("paired_ci"))
    return "\n".join(
        [
            "# Benchmark Sensitivity Protocol",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Equal-Token Primary",
            "",
            f"- Minimum paired seeds: `{primary.get('minimum_paired_training_seeds')}`",
            f"- Paired CI: `{ci.get('method')}`, `{ci.get('confidence_level')}`",
            "",
            "## External Transfer",
            "",
            f"- Current status: `{external.get('current_status')}`",
            f"- Current task count: `{external.get('task_count')}`",
            f"- Below powered task floor: `{external.get('below_powered_task_floor')}`",
            f"- Claim: `{external.get('claim')}`",
            "",
            report["interpretation"],
            "",
        ]
    )


def main() -> int:
    report = build()
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
