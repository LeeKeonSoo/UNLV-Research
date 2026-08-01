from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from run_curation import materialize


JsonMap = dict[str, Any]
RUNTIME_FORBIDDEN_INPUTS = ["Utility", "NLL", "benchmark_outcomes", "target_retention_fraction"]


def _write_json(path: Path, value: JsonMap) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _run_arm(work_dir: Path, scenario: JsonMap, mode: str) -> JsonMap:
    scenario_id = str(scenario["id"])
    input_path = work_dir / f"{scenario_id}-{mode}.jsonl"
    output_dir = work_dir / f"{scenario_id}-{mode}"
    config_path = work_dir / f"{scenario_id}-{mode}.json"
    _write_jsonl(input_path, scenario["records"])
    config: JsonMap = {
        "schema_version": "curation-run-contract-v1",
        "status": "frozen_before_stage_a_b_c_materialization",
        "curation_mode": mode,
        "input": {"candidate_files": [str(input_path)], "text_fields": ["text"], "defaults": {}},
        "output_dir": str(output_dir),
        "stage_a": {"policy": "text_only_v2"},
        "stage_b": {"max_chunk_chars": 6000, "minimum_chunk_chars": 40},
        "stage_c_selection": {"near_duplicate_compaction": {"candidate_enabled": False}},
        "stage_c": {"no_binding_budget_action": "selection_without_binding_budget"},
        "claim_boundary": "development structural fixture only",
    }
    if mode == "hard":
        config["execution_scope"] = "development"
    _write_json(config_path, config)
    return materialize(config_path)


def _span_reasons(report: JsonMap) -> list[str]:
    stages = report["reason_code_impact_audit"]["stages"]
    transformations = stages.get("stage_c_span_transformation", {})
    reasons = transformations.get("reasons", {}) if isinstance(transformations, dict) else {}
    return sorted(str(reason) for reason in reasons)


def _scenario_result(work_dir: Path, scenario: JsonMap) -> JsonMap:
    normal = _run_arm(work_dir, scenario, "normal")
    hard = _run_arm(work_dir, scenario, "hard")
    expected_reasons = sorted(str(reason) for reason in scenario["expected_hard_reason_codes"])
    hard_reasons = _span_reasons(hard)
    normal_tokens = int(normal["summary"]["stage_c_curated_token_proxy"])
    hard_tokens = int(hard["summary"]["stage_c_curated_token_proxy"])
    clean_retention_required = bool(scenario["clean_retention_required"])
    clean_retention_passed = (
        normal["summary"]["stage_c_curated_chunks"] == hard["summary"]["stage_c_curated_chunks"]
        and normal_tokens == hard_tokens
        and not hard_reasons
    )
    return {
        "id": str(scenario["id"]),
        "domain": str(scenario["domain"]),
        "clean_retention_required": clean_retention_required,
        "normal_curated_chunks": int(normal["summary"]["stage_c_curated_chunks"]),
        "hard_curated_chunks": int(hard["summary"]["stage_c_curated_chunks"]),
        "normal_token_proxy": normal_tokens,
        "hard_token_proxy": hard_tokens,
        "hard_token_delta_proxy": hard_tokens - normal_tokens,
        "hard_span_transformations": int(hard["summary"]["stage_c_hard_span_transformations"]),
        "hard_span_reason_codes": hard_reasons,
        "expected_hard_reason_codes": expected_reasons,
        "expected_reason_codes_observed": set(expected_reasons).issubset(hard_reasons),
        "coverage_invariant_passed": bool(hard["coverage_impact_audit"]["passed"]),
        "clean_retention_passed": clean_retention_passed if clean_retention_required else True,
        "external_evaluation_read": False,
    }


def build_development_report(fixture_path: Path, output_path: Path) -> JsonMap:
    """Run frozen Normal-vs-Hard structural fixtures without benchmark feedback."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "hard-development-ablation-fixtures-v1":
        raise RuntimeError("Unexpected Hard development-ablation fixture schema")
    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        scenarios = [_scenario_result(work_dir, scenario) for scenario in fixture["scenarios"]]
    passed = all(
        item["expected_reason_codes_observed"]
        and item["coverage_invariant_passed"]
        and item["clean_retention_passed"]
        for item in scenarios
    )
    report = {
        "schema_version": "hard-development-ablation-report-v1",
        "status": "hard_development_ablation_passed" if passed else "hard_development_ablation_failed",
        "profiles": {"normal": "normal_structural_v1", "hard": "hard_structural_v1"},
        "hard_execution_scope": "development_only_pending_n4_ablation",
        "runtime_forbidden_inputs": RUNTIME_FORBIDDEN_INPUTS,
        "external_evaluation_read": False,
        "scenarios": scenarios,
        "claim_boundary": "Fixture-only Normal-vs-Hard structural evidence; it does not promote Hard for production or establish downstream effectiveness.",
    }
    _write_json(output_path, report)
    return report
