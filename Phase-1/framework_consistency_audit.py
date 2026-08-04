from __future__ import annotations

import json
from pathlib import Path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_consistency_audit(root: Path, coverage_report: Path | None = None) -> dict:
    manifest = _json(root / "configs/curation_framework_v1.json")
    contract = _json(root / "configs/curation_contract.json")
    objects = _json(root / "configs/framework_objects_v1.json")
    profiles = _json(root / "configs/framework_profiles_v1.json")
    release = _json(root / "validation/frozen_contracts/framework_release_validation_v1.json")
    model_providers = _json(root / "configs/model_provider_registry_v1.json")
    runtime = (root / "run_curation.py").read_text(encoding="utf-8")
    ownership = {stage["id"]: stage["core_ids"] for stage in manifest["stages"]}
    contract_ownership = {
        "stage_a": contract["stage_a"]["core_ids"],
        "stage_b": contract["stage_b"]["core_ids"],
        "stage_c": contract["stage_c"]["core_ids"],
    }
    policy_lifecycle = {
        policy["id"]: policy["lifecycle"] for policy in objects["policies"]
    }
    profile_lifecycle = {
        item["policy_id"]: item["lifecycle"] for item in profiles["policy_lifecycles"]
    }
    metric_ids = {metric["id"] for metric in objects["metrics"]}
    method_ids = {method["id"] for method in objects["methods"]}
    provider_ids = {provider["id"] for provider in objects["providers"]}
    coverage_policy = next(
        policy for policy in objects["policies"] if policy["id"] == "coverage.representative_guard"
    )
    provider_lifecycle = {
        provider["provider_id"]: provider["lifecycle"] for provider in model_providers["providers"]
    }
    checks = {
        "manifest_contract_stage_ownership": ownership == contract_ownership,
        "runtime_stage_b_trace_owner": '"stage_b_policy": selection_audit' in runtime,
        "runtime_stage_c_contract": '"stage_c": "coverage_veto_and_final_materialization"' in runtime,
        "quality_lifecycle_registry_alignment": (
            policy_lifecycle.get("quality.teacher_panel_v2")
            == profile_lifecycle.get("quality.teacher_panel_v2")
            == "blocked"
        ),
        "profiles_release_disabled": all(not profile["release_enabled"] for profile in profiles["profiles"]),
        "external_evaluation_hidden": manifest["external_evaluation"]["selector_visible"] is False,
        "budget_forbidden": (
            manifest["profile_contract"]["fixed_retention_fraction_allowed"] is False
            and manifest["profile_contract"]["maximum_token_budget_allowed"] is False
        ),
        "semantic_coverage_typed_lineage": (
            "coverage.semantic_materialization" in method_ids
            and "coverage.semantic_support_extinction" in metric_ids
            and "coverage.semantic_consensus_v3" in provider_ids
            and "coverage.semantic_support_extinction" in coverage_policy["metric_ids"]
        ),
        "semantic_provider_lifecycle_alignment": (
            provider_lifecycle.get("qwen3-embedding-0.6b-semantic-candidate")
            == "runtime_experiment"
            and provider_lifecycle.get("bge-m3-semantic-audit-candidate") == "audit_only"
        ),
    }
    coverage = None
    if coverage_report is not None and coverage_report.is_file():
        coverage = _json(coverage_report)
    implementation_ready = all(checks.values())
    coverage_implemented = coverage is not None and coverage.get("implementation_gate_passed") is True
    release_eligible = release["framework_release"] == "eligible"
    coverage_promoted = coverage is not None and coverage.get("scientific_promotion_gate_passed") is True
    blockers = []
    if not release_eligible:
        blockers.append("framework_release_blocked")
    if not coverage_promoted:
        blockers.append("semantic_coverage_scientific_promotion_missing")
    return {
        "schema_version": "framework-consistency-audit-v2",
        "implementation_consistency": "passed" if implementation_ready else "failed",
        "confirmatory_candidate_ready": implementation_ready and coverage_implemented,
        "paper_claim_ready": implementation_ready and release_eligible and coverage_promoted,
        "production_release_ready": release_eligible and coverage_promoted,
        "stage_ownership": ownership,
        "quality_teacher_policy_lifecycle": policy_lifecycle.get("quality.teacher_panel_v2"),
        "checks": checks,
        "coverage_evidence": coverage,
        "release_blockers": release["release_blockers"],
        "readiness_blockers": blockers,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
