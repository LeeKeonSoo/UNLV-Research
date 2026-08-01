#!/usr/bin/env python3
"""Build a conservative readiness report before freezing code repositories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.temporal_code_manifests import build_benchmark_quarantine_manifest


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_BENCHMARK_SEED = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "repository_candidate_manifest_authenticated.json"
DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_smoke30.json"
DEFAULT_REPRODUCIBILITY = OUTPUT_DIR / "temporal_code_collection" / "commit_reproducibility_report_smoke30.json"
DEFAULT_SMOKE_AUDIT = (
    OUTPUT_DIR / "temporal_code_collection" / "smoke_bundles" / "smoke_bundle_audit_report.json"
)
DEFAULT_TEST_VERIFICATION = OUTPUT_DIR / "temporal_code_collection" / "smoke_test_command_verification.json"
DEFAULT_STAGE0_REPORT = OUTPUT_DIR / "temporal_code_collection" / "stage0_smoke" / "stage0_smoke_report.json"
DEFAULT_FROZEN_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "collection_readiness_report.json"
DEFAULT_ARTIFACT_MANIFESTS = [
    OUTPUT_DIR / "temporal_code_collection" / "benchmark_task_artifact_manifest_swebench.json",
]


def build(
    protocol_path: Path,
    benchmark_seed_path: Path,
    discovery_path: Path,
    enrichment_path: Path,
    reproducibility_path: Path,
    output_path: Path,
    artifact_manifest_paths: list[Path],
    smoke_audit_path: Path | None = None,
    test_verification_path: Path | None = None,
    stage0_report_path: Path | None = None,
    frozen_manifest_path: Path | None = None,
) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    quarantine = build_benchmark_quarantine_manifest(load_json(benchmark_seed_path)["entries"], protocol)
    discovery = load_json(discovery_path)
    enrichment = load_json(enrichment_path)
    reproducibility = load_json(reproducibility_path)
    smoke_audit = load_json(smoke_audit_path) if smoke_audit_path and smoke_audit_path.exists() else None
    test_verification = (
        load_json(test_verification_path)
        if test_verification_path and test_verification_path.exists()
        else None
    )
    stage0_report = load_json(stage0_report_path) if stage0_report_path and stage0_report_path.exists() else None
    frozen_manifest = (
        load_json(frozen_manifest_path)
        if frozen_manifest_path and frozen_manifest_path.exists()
        else None
    )
    completed_artifacts = set()
    artifact_reports = []
    for path in artifact_manifest_paths:
        if not path.exists():
            continue
        artifact = load_json(path)
        artifact_reports.append(
            {
                "path": str(path),
                "summary": artifact["summary"],
            }
        )
        completed_artifacts.update(
            row["benchmark"] for row in artifact["benchmarks"] if row["status"] == "complete"
        )
    pending_artifacts = [
        row["benchmark"]
        for row in quarantine["entries"]
        if row["task_artifact_manifest_status"] == "required_before_freeze" and row["benchmark"] not in completed_artifacts
    ]
    blockers = []
    if pending_artifacts:
        blockers.append("benchmark_task_artifact_manifests_incomplete")
    discovery_eligible = discovery["summary"]["metadata_enrichment_candidate_count"]
    enrichment_count = enrichment["summary"]["repository_count"]
    enrichment_pass = enrichment["summary"]["eligible_for_reproducibility_probe_count"]
    reproducibility_count = reproducibility["summary"]["repository_count"]
    reproducibility_pass = reproducibility["summary"]["eligible_for_quarantine_review_count"]
    if enrichment_count < discovery_eligible:
        blockers.append("repository_enrichment_coverage_incomplete")
    if reproducibility_count < enrichment_pass:
        blockers.append("commit_reproducibility_coverage_incomplete")

    smoke_collection_gate_pass = (
        smoke_audit is not None and smoke_audit["summary"]["collection_gate_pass_count"] > 0
    )
    smoke_test_commands_verified = (
        test_verification is not None
        and test_verification.get("dry_run") is False
        and test_verification["summary"]["failed_or_unverified_bundle_count"] == 0
    )
    smoke_stage0_pass = (
        stage0_report is not None and stage0_report["summary"]["release_candidate_records"] > 0
    )
    smoke_feasibility_validated = (
        smoke_collection_gate_pass and smoke_test_commands_verified and smoke_stage0_pass
    )
    status = (
        "broad_repository_manifest_frozen"
        if not blockers and frozen_manifest is not None
        else "ready_to_freeze_repository_manifest"
        if not blockers
        else "smoke_feasibility_validated_broad_manifest_not_ready"
        if smoke_feasibility_validated
        else "not_ready_to_freeze_repository_manifest"
    )
    report = {
        "schema_version": "temporal-code-collection-readiness-report-v2",
        "status": status,
        "evidence": {
            "authenticated_discovery_candidates": discovery["summary"]["candidate_count"],
            "metadata_enrichment_candidates": discovery_eligible,
            "enrichment_sample_repositories": enrichment_count,
            "enrichment_sample_pass": enrichment_pass,
            "reproducibility_sample_repositories": reproducibility_count,
            "reproducibility_sample_pass": reproducibility_pass,
            "smoke_feasibility": {
                "validated": smoke_feasibility_validated,
                "collection_gate_pass_count": (
                    smoke_audit["summary"]["collection_gate_pass_count"] if smoke_audit else 0
                ),
                "test_command_verified_count": (
                    test_verification["summary"]["verified_bundle_count"] if test_verification else 0
                ),
                "stage0_release_candidate_records": (
                    stage0_report["summary"]["release_candidate_records"] if stage0_report else 0
                ),
            },
            "full_repository_benchmark_exclusions": {
                row["benchmark"]: row["repository_patterns"] for row in quarantine["entries"]
            },
            "pending_task_artifact_manifests": pending_artifacts,
            "completed_task_artifact_manifests": sorted(completed_artifacts),
            "task_artifact_reports": artifact_reports,
        },
        "blockers": blockers,
        "frozen_repository_count": (
            frozen_manifest["summary"]["frozen_repository_count"] if frozen_manifest else 0
        ),
        "next_actions": (
            [
                "Fetch a bounded broad-corpus tranche under the frozen manifest and fetch limits.",
                "Run the same automated content, quarantine, split, Stage-0, and Stage-A gates on the tranche.",
                "Build equal-token Stage-B selected and common disjoint Stage-A-random arms.",
                "Confirm Qwen3-4B-Base QLoRA feasibility before Stage-C target-model comparison.",
            ]
            if frozen_manifest
            else [
                "Complete balanced repository enrichment across the frozen discovery candidate manifest.",
                "Complete commit-identity reproducibility probes for every enrichment-pass repository.",
                "Freeze the broad repository manifest before fetching broad change-bundle content.",
                "Run the same automated content, quarantine, and isolated-test gates on the broad collection.",
            ]
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Bounded smoke feasibility may be validated, but no broad repository manifest or training "
            "claim exists until every broad-freeze blocker is cleared."
        ),
    }
    save_json(output_path, report)
    lines = [
        "# Temporal Code Collection Readiness",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Authenticated discovery candidates: {report['evidence']['authenticated_discovery_candidates']}",
        "",
        f"Enrichment smoke pass: {report['evidence']['enrichment_sample_pass']} / "
        f"{report['evidence']['enrichment_sample_repositories']}",
        "",
        f"Commit-identity smoke pass: {report['evidence']['reproducibility_sample_pass']} / "
        f"{report['evidence']['reproducibility_sample_repositories']}",
        "",
        f"Bounded smoke feasibility validated: {report['evidence']['smoke_feasibility']['validated']}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in blockers)
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.append("")
    output_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal code collection readiness.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--benchmark-seed", type=Path, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--reproducibility", type=Path, default=DEFAULT_REPRODUCIBILITY)
    parser.add_argument("--smoke-audit", type=Path, default=DEFAULT_SMOKE_AUDIT)
    parser.add_argument("--test-verification", type=Path, default=DEFAULT_TEST_VERIFICATION)
    parser.add_argument("--stage0-report", type=Path, default=DEFAULT_STAGE0_REPORT)
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-manifests", nargs="*", type=Path, default=DEFAULT_ARTIFACT_MANIFESTS)
    args = parser.parse_args()
    report = build(
        args.protocol,
        args.benchmark_seed,
        args.discovery,
        args.enrichment,
        args.reproducibility,
        args.output,
        args.artifact_manifests,
        args.smoke_audit,
        args.test_verification,
        args.stage0_report,
        args.frozen_manifest,
    )
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
