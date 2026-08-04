from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]
EXPECTED_STAGES = {
    "stage_a": "source_agnostic_text_normalization_and_integrity_handling",
    "stage_b": "redundancy_and_quality_removal_proposals",
    "stage_c": "coverage_veto_and_final_materialization",
    "external_evaluation": "not_started",
}


def _json(path: Path) -> JsonMap:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _curated_ids(report: JsonMap) -> frozenset[str]:
    declaration = report["outputs"]["stage_c_curated"]
    output = Path(str(declaration["path"]))
    if _sha256(output) != declaration["sha256"]:
        raise RuntimeError(f"Curated output hash mismatch: {output}")
    return frozenset(
        str(json.loads(line)["chunk_uid"])
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _arm_checks(report: JsonMap) -> dict[str, bool]:
    coverage = report.get("stage_c_coverage") or {}
    selector = report.get("selector_boundary") or {}
    return {
        "materialization_complete": report.get("status") == "curation_materialization_complete",
        "stage_contract_exact": report.get("stage_contract") == EXPECTED_STAGES,
        "coverage_graph_consumed": coverage.get("semantic_graph_consumed") is True,
        "coverage_recheck_passed": (
            coverage.get("final_status") == "pass"
            and coverage.get("complete_recheck_passed") is True
        ),
        "coverage_non_deleting": coverage.get("may_create_new_removal") is False,
        "runtime_forbidden_inputs_unread": all(
            selector.get(key) is False
            for key in ("utility_read", "benchmark_outcomes_read", "source_pool_role_read")
        ),
    }


def _exact_token_checks(
    training: JsonMap | None, normal: JsonMap, hard: JsonMap
) -> tuple[bool, dict[str, int]]:
    if training is None or training.get("status") != "tokenizer_materialization_complete":
        return False, {}
    arms = training.get("arms") or {}
    required = ("raw_audited_natural", "normal_natural", "hard_natural")
    counts = {
        arm: int(arms[arm]["stream_tokens"])
        for arm in required
        if arm in arms and int(arms[arm].get("stream_tokens", 0)) > 0
    }
    natural = training.get("selector_boundary") or {}
    files_valid = all(
        all(
            Path(str(arms[arm][path_key])).is_file()
            and _sha256(Path(str(arms[arm][path_key]))) == arms[arm][hash_key]
            for path_key, hash_key in (
                ("source_path", "source_sha256"),
                ("arm_path", "arm_sha256"),
                ("blocks_path", "blocks_sha256"),
            )
        )
        for arm in required
        if arm in arms
    )
    source_chain = (
        arms.get("normal_natural", {}).get("source_sha256")
        == normal["outputs"]["stage_c_curated"]["sha256"]
        and arms.get("hard_natural", {}).get("source_sha256")
        == hard["outputs"]["stage_c_curated"]["sha256"]
    )
    boundary = all(
        natural.get(key) is False
        for key in ("utility_read", "benchmark_outcomes_read", "target_token_fraction_read")
    )
    return len(counts) == len(required) and boundary and files_valid and source_chain, counts


def build_final_experiment_preflight(
    normal_report_path: Path,
    hard_report_path: Path,
    release_report_path: Path,
    training_report_path: Path | None = None,
) -> JsonMap:
    normal = _json(normal_report_path)
    hard = _json(hard_report_path)
    release = _json(release_report_path)
    training = _json(training_report_path) if training_report_path and training_report_path.is_file() else None
    checks = {"normal": _arm_checks(normal), "hard": _arm_checks(hard)}
    normal_ids = _curated_ids(normal)
    hard_ids = _curated_ids(hard)
    mode_monotonicity = hard_ids <= normal_ids
    exact_ready, exact_counts = _exact_token_checks(training, normal, hard)
    materialization_ready = all(all(values.values()) for values in checks.values()) and mode_monotonicity
    release_eligible = release.get("framework_release") == "eligible"
    coverage_promoted = all(
        (report.get("stage_c_coverage") or {}).get("scientific_promotion_claimed") is True
        for report in (normal, hard)
    )
    return {
        "schema_version": "final-experiment-preflight-v1",
        "framework_materialization_ready": materialization_ready,
        "external_confirmatory_ready": materialization_ready and exact_ready,
        "paper_claim_ready": materialization_ready and exact_ready and release_eligible and coverage_promoted,
        "production_release_ready": release_eligible and coverage_promoted,
        "arm_checks": checks,
        "hard_subset_or_equal_normal": mode_monotonicity,
        "retained_chunks": {"normal": len(normal_ids), "hard": len(hard_ids)},
        "exact_tokenizer_counts": exact_counts,
        "release_blockers": list(release.get("release_blockers") or []),
        "scientific_blockers": [
            blocker
            for blocker, blocked in (
                ("exact_tokenizer_materialization_missing", not exact_ready),
                ("framework_policy_release_blocked", not release_eligible),
                ("semantic_coverage_scientific_promotion_missing", not coverage_promoted),
            )
            if blocked
        ],
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit final curation and external experiment readiness.")
    parser.add_argument("--normal-report", type=Path, required=True)
    parser.add_argument("--hard-report", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_final_experiment_preflight(
        args.normal_report, args.hard_report, args.release_report, args.training_report
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "framework_materialization_ready", "external_confirmatory_ready",
        "paper_claim_ready", "production_release_ready"
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
