#!/usr/bin/env python3
"""Audit fetched temporal-code smoke bundles without approving them for release."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_change import bundle_executable_evaluation_eligibility, bundle_protocol_eligibility
from ingestion.code_fingerprints import derived_fingerprints
from ingestion.normalize import normalize_text
from ingestion.temporal_code_manifests import (
    benchmark_quarantine_decision,
    build_benchmark_quarantine_manifest,
    build_repository_split_manifest,
    bundle_split_eligibility,
)


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_smoke_fetch_plan.json"
DEFAULT_BENCHMARK_SEED = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_ARTIFACTS = OUTPUT_DIR / "temporal_code_collection" / "benchmark_task_artifact_manifest_swebench.json"
DEFAULT_BUNDLE_DIR = OUTPUT_DIR / "temporal_code_collection" / "smoke_bundles"
DEFAULT_OUTPUT = DEFAULT_BUNDLE_DIR / "smoke_bundle_audit_report.json"
DEFAULT_TEST_VERIFICATION = OUTPUT_DIR / "temporal_code_collection" / "smoke_test_command_verification.json"


def _artifact_entries(seed_entries: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    derived = {row["benchmark"]: row for row in artifacts.get("benchmarks") or [] if row.get("status") == "complete"}
    entries = []
    for seed in seed_entries:
        row = dict(seed)
        artifact = derived.get(row["benchmark"])
        if artifact:
            row["task_artifact_rules"] = artifact["task_artifact_rules"]
            row["task_artifact_manifest_status"] = "complete"
        entries.append(row)
    return entries


def _repositories_from_plan(plan: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for value in plan["selected_repositories"].values():
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            yield {
                "repository_identity": row["repository_identity"],
                "repository_url": row["repository_url"],
                "license": row["license"],
            }


def bundle_with_content_signatures(bundle: Dict[str, Any]) -> Dict[str, Any]:
    audited = dict(bundle)
    signatures = []
    for file_record in bundle.get("files") or []:
        for state in ("before_text", "after_text"):
            text = file_record.get(state)
            if not isinstance(text, str) or not text.strip():
                continue
            normalized = normalize_text(text)
            signatures.append(
                {
                    "path": file_record.get("path"),
                    "state": state.removesuffix("_text"),
                    "normalized_sha256": normalized["normalized_sha256"],
                    **derived_fingerprints(text),
                }
            )
    audited["content_signatures"] = signatures
    return audited


def bundle_with_test_verification(bundle: Dict[str, Any], verification: Dict[str, Any] | None) -> Dict[str, Any]:
    if not verification or verification.get("dry_run") is True:
        return dict(bundle)
    decision = next(
        (
            row
            for row in verification.get("decisions") or []
            if row.get("bundle_id") == bundle.get("bundle_id") and row.get("test_command_verified") is True
        ),
        None,
    )
    if decision is None:
        return dict(bundle)
    audited = dict(bundle)
    execution = dict(bundle.get("execution_validation") or {})
    execution["test_command"] = " ".join(decision["test_command"])
    execution["test_command_verified"] = True
    execution["verification_report_schema"] = verification.get("schema_version")
    audited["execution_validation"] = execution
    return audited


def audit(
    protocol: Dict[str, Any],
    plan: Dict[str, Any],
    benchmark_seed: Dict[str, Any],
    artifacts: Dict[str, Any],
    bundle_paths: Iterable[Path],
    test_verification: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    split_manifest = build_repository_split_manifest(_repositories_from_plan(plan), protocol)
    quarantine_manifest = build_benchmark_quarantine_manifest(
        _artifact_entries(benchmark_seed["entries"], artifacts),
        protocol,
    )
    decisions = []
    for path in sorted(bundle_paths):
        bundle = bundle_with_content_signatures(
            bundle_with_test_verification(load_json(path), test_verification)
        )
        protocol_decision = bundle_protocol_eligibility(bundle, protocol)
        executable_decision = bundle_executable_evaluation_eligibility(bundle)
        split_decision = bundle_split_eligibility(bundle, split_manifest, protocol)
        quarantine_decision = benchmark_quarantine_decision(bundle, quarantine_manifest)
        blockers = list(protocol_decision["blockers"])
        blockers.extend(split_decision["blockers"])
        if quarantine_decision["quarantine"]:
            blockers.append("benchmark_quarantine_match")
        file_fetch_blockers = sorted(
            {
                blocker
                for row in bundle.get("files") or []
                for blocker in row.get("fetch_blockers") or []
            }
        )
        generated_detection_complete = all(
            row.get("generated_detection_status") == "completed_heuristic_v1"
            for row in bundle.get("files") or []
            if isinstance(row.get("after_text"), str) and row.get("after_text", "").strip()
        )
        if not generated_detection_complete:
            blockers.append("generated_file_detection_not_verified")
        fingerprint_contract = artifacts.get("fingerprint_contract") or {}
        if not {"token_simhash64", "python_ast_sha256"}.issubset(fingerprint_contract):
            blockers.append("benchmark_token_ast_near_duplicate_checks_not_completed")
        collection_gate_pass = not blockers
        decisions.append(
            {
                "bundle_id": bundle.get("bundle_id"),
                "repository_identity": bundle.get("repository_identity"),
                "bundle_path": str(path),
                "assigned_split": split_decision["assigned_split"],
                "observed_temporal_split": split_decision["observed_temporal_split"],
                "file_count": len(bundle.get("files") or []),
                "training_payload_count": len(protocol_decision["training_payloads"]),
                "normalized_content_signature_count": len(bundle["content_signatures"]),
                "file_fetch_blockers": file_fetch_blockers,
                "pii_quarantined_file_count": sum(bool(row.get("pii_detected")) for row in bundle.get("files") or []),
                "secret_quarantined_file_count": sum(
                    bool(row.get("secret_detected")) for row in bundle.get("files") or []
                ),
                "generated_file_count": sum(bool(row.get("generated")) for row in bundle.get("files") or []),
                "suppressed_phone_candidate_count": sum(
                    int((((row.get("hazard_scan") or {}).get(state) or {}).get("diagnostics") or {}).get(
                        "phone_suppressed_count"
                    ) or 0)
                    for row in bundle.get("files") or []
                    for state in ("before", "after")
                ),
                "benchmark_quarantine": quarantine_decision,
                "blockers": sorted(set(blockers)),
                "collection_gate_pass": collection_gate_pass,
                "executable_evaluation_gate_pass": collection_gate_pass and executable_decision["eligible"],
                "executable_evaluation_blockers": executable_decision["blockers"],
                "stage0_release_candidate": not blockers,
            }
        )
    release_candidates = [row for row in decisions if row["stage0_release_candidate"]]
    plan_schema = plan.get("schema_version")
    broad_tranche = plan_schema == "temporal-code-broad-tranche-plan-v1"
    path_stratified_tranche = plan_schema == "temporal-code-path-stratified-tranche-plan-v1"
    bounded_tranche = broad_tranche or path_stratified_tranche
    return {
        "schema_version": (
            "temporal-code-path-stratified-tranche-bundle-audit-v1"
            if path_stratified_tranche
            else (
                "temporal-code-broad-tranche-bundle-audit-v1"
                if broad_tranche
                else "temporal-code-smoke-bundle-audit-v1"
            )
        ),
        "plan_status": plan["status"],
        "summary": {
            "bundle_count": len(decisions),
            "contract_valid_bundle_count": sum(
                "invalid_change_bundle_contract" not in row["blockers"] for row in decisions
            ),
            "split_valid_bundle_count": sum(
                not {
                    "repository_not_in_frozen_split_manifest",
                    "outside_frozen_time_windows",
                    "repository_split_time_window_mismatch",
                }.intersection(row["blockers"])
                for row in decisions
            ),
            "benchmark_quarantine_match_count": sum(row["benchmark_quarantine"]["quarantine"] for row in decisions),
            "pii_quarantined_file_count": sum(row["pii_quarantined_file_count"] for row in decisions),
            "secret_quarantined_file_count": sum(row["secret_quarantined_file_count"] for row in decisions),
            "generated_file_count": sum(row["generated_file_count"] for row in decisions),
            "suppressed_phone_candidate_count": sum(row["suppressed_phone_candidate_count"] for row in decisions),
            "collection_gate_pass_count": sum(row["collection_gate_pass"] for row in decisions),
            "executable_evaluation_gate_pass_count": sum(
                row["executable_evaluation_gate_pass"] for row in decisions
            ),
            "stage0_release_candidate_count": len(release_candidates),
        },
        "decisions": decisions,
        "required_before_stage0_release": [
            "retain and review deterministic generated-file detection evidence",
            "verify executable test commands in isolated parent and merge checkouts",
            "resolve or explicitly accept every file fetch blocker",
            "retain benchmark quarantine and split checks before generic Stage 0",
            "implement token and AST near-duplicate benchmark quarantine before generic Stage 0",
        ],
        "execution_boundary": (
            "Executable-test verification gates executable evaluation eligibility, not licensed training-content "
            "eligibility. Execution failures remain Stage-C/evaluation blockers."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Bounded-tranche bundle audit only; no bundle or repository is approved for training."
            if bounded_tranche
            else "Smoke bundle audit only; no bundle or repository is approved for training."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fetched temporal-code smoke bundles.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--benchmark-seed", type=Path, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-verification", type=Path, default=DEFAULT_TEST_VERIFICATION)
    args = parser.parse_args()
    report = audit(
        load_json(args.protocol),
        load_json(args.plan),
        load_json(args.benchmark_seed),
        load_json(args.artifacts),
        (
            path
            for path in args.bundle_dir.rglob("*.json")
            if path.name not in {
                "smoke_fetch_report.json",
                "broad_tranche_fetch_report.json",
                "path_stratified_tranche_fetch_report.json",
                args.output.name,
            }
        ),
        load_json(args.test_verification) if args.test_verification.exists() else None,
    )
    save_json(args.output, report)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
