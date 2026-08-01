from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from run_curation import materialize


JsonMap = dict[str, Any]
ALL_WEAK_QUALITY_RULES = {
    "explicit_generated_artifact": True,
    "license_comment_only_chunk": True,
    "empty_html_shell": True,
    "web_chrome_only_chunk": True,
}
RUNTIME_INPUTS = ["chunk text"]
FORBIDDEN_RUNTIME_INPUTS = ["Utility", "NLL", "benchmark_outcomes", "target_retention_fraction"]


def _write_json(path: Path, value: JsonMap) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[JsonMap]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records), encoding="utf-8")


def _reason_codes(report: JsonMap) -> list[str]:
    stages = report["reason_code_impact_audit"]["stages"]
    codes: set[str] = set()
    for stage in stages.values():
        reasons = stage.get("reasons") if isinstance(stage, dict) else None
        if isinstance(reasons, dict):
            codes.update(str(code) for code in reasons)
    return sorted(codes)


def _run_arm(work_dir: Path, scenario_id: str, records: list[JsonMap], artifact_rules: JsonMap) -> JsonMap:
    input_path = work_dir / f"{scenario_id}.jsonl"
    config_path = work_dir / f"{scenario_id}-config.json"
    output_dir = work_dir / f"{scenario_id}-output"
    _write_jsonl(input_path, records)
    _write_json(
        config_path,
        {
            "schema_version": "curation-run-contract-v1",
            "status": "frozen_before_stage_a_b_c_materialization",
            "input": {"candidate_files": [str(input_path)], "text_fields": ["text"], "defaults": {}},
            "output_dir": str(output_dir),
            "stage_a": {"policy": "text_only_v2"},
            "stage_b": {"max_chunk_chars": 6000, "minimum_chunk_chars": 40},
            "stage_c_selection": {
                "near_duplicate_compaction": {"candidate_enabled": False},
                "structural_artifact_rules": artifact_rules,
            },
            "stage_c": {"no_binding_budget_action": "selection_without_binding_budget"},
            "claim_boundary": "development structural fixture only",
        },
    )
    return materialize(config_path)


def _scenario_result(work_dir: Path, scenario: JsonMap) -> JsonMap:
    scenario_id = str(scenario["id"])
    records = list(scenario["records"])
    baseline = _run_arm(work_dir, f"{scenario_id}-baseline", records, {})
    all_rules = _run_arm(work_dir, f"{scenario_id}-all-rules", records, ALL_WEAK_QUALITY_RULES)
    baseline_reasons = _reason_codes(baseline)
    all_rule_reasons = _reason_codes(all_rules)
    expected_reasons = {str(reason) for reason in scenario.get("expected_all_rule_reason_codes", [])}
    clean_retention_required = bool(scenario.get("clean_retention_required", False))
    baseline_tokens = int(baseline["summary"]["stage_c_curated_token_proxy"])
    all_rules_tokens = int(all_rules["summary"]["stage_c_curated_token_proxy"])
    return {
        "id": scenario_id,
        "domain": str(scenario["domain"]),
        "scenario_type": str(scenario["scenario_type"]),
        "baseline_removed_reason_codes": baseline_reasons,
        "all_rules_removed_reason_codes": all_rule_reasons,
        "baseline_curated_token_proxy": baseline_tokens,
        "all_rules_curated_token_proxy": all_rules_tokens,
        "all_rules_token_delta_proxy": all_rules_tokens - baseline_tokens,
        "expected_reason_codes_observed": expected_reasons.issubset(all_rule_reasons),
        "coverage_invariant_passed": bool(all_rules["coverage_impact_audit"]["passed"]),
        "clean_retention_passed": (
            all_rules["summary"]["stage_c_curated_chunks"] == baseline["summary"]["stage_c_curated_chunks"]
            and all_rules_tokens == baseline_tokens
            and not all_rule_reasons
            if clean_retention_required
            else True
        ),
    }


def build_development_report(fixture_path: Path, output_path: Path) -> JsonMap:
    """Run frozen structural fixtures without exposing evaluation outcomes to curation."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "weak-development-gate-fixtures-v1":
        raise RuntimeError("Unexpected Weak development gate fixture schema")
    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        scenarios = [_scenario_result(work_dir, scenario) for scenario in fixture["scenarios"]]
    passed = all(
        scenario["expected_reason_codes_observed"]
        and scenario["coverage_invariant_passed"]
        and scenario["clean_retention_passed"]
        for scenario in scenarios
    )
    report = {
        "schema_version": "weak-development-gate-report-v1",
        "status": "weak_development_gate_passed" if passed else "weak_development_gate_failed",
        "runtime_inputs": RUNTIME_INPUTS,
        "forbidden_runtime_inputs": FORBIDDEN_RUNTIME_INPUTS,
        "external_evaluation_read": False,
        "scenarios": scenarios,
        "claim_boundary": "Structural fixture behavior and rule-on/off token deltas only; no downstream-effectiveness claim.",
    }
    _write_json(output_path, report)
    return report
