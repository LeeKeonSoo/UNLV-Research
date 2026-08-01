#!/usr/bin/env python3
"""Review whether each Core axis represents a defensible construct."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_FRAMEWORK = Path("configs") / "lm_curation_operational_framework_v1.json"
DEFAULT_METRIC_SPEC = Path("configs") / "metric_spec_with_citations.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "core_construct_validity_review.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "core_construct_validity_review.md"


CORE_REVIEWS: Dict[str, Dict[str, Any]] = {
    "Validity": {
        "construct_status": "defensible_when_scoped",
        "defensible_construct": "structural usability for training ingestion",
        "invalid_construct": "semantic correctness, usefulness, or quality",
        "minimum_behavior_evidence_needed": [
            "reject corrupted/empty/markup-residue chunks",
            "pass parseable short-but-usable code examples",
            "separate hard invalidity from warning-only style issues",
        ],
    },
    "Selection Value Evidence": {
        "construct_status": "defensible_as_observable_evidence",
        "defensible_construct": "pre-outcome selection-value proxy",
        "invalid_construct": "intrinsic or ground-truth data quality",
        "minimum_behavior_evidence_needed": [
            "rank boilerplate below concise useful examples",
            "avoid treating length or AST richness as quality by itself",
            "report selected-vs-rejected proxy shifts before Stage C",
            "validate downstream effect only in Stage C",
            "retain all Stage-A-pass records when no budget constraint applies",
            "never interpret budget-not-selected as rejected or low quality",
        ],
    },
    "Redundancy": {
        "construct_status": "defensible_when_split",
        "defensible_construct": "harmful duplication and saturation risk with useful recurrence preserved",
        "invalid_construct": "all repetition is harmful",
        "minimum_behavior_evidence_needed": [
            "reject exact duplicates",
            "penalize high-overlap near duplicates",
            "preserve useful recurrence such as API idioms, tests, examples, and definitions",
        ],
    },
    "Coverage": {
        "construct_status": "defensible_only_as_observable_retention",
        "defensible_construct": "source/style/path/content-type/cluster retention",
        "invalid_construct": "true semantic or domain coverage without metadata",
        "minimum_behavior_evidence_needed": [
            "detect source and content-type collapse",
            "detect path-family or template concentration",
            "separate explicit domain metadata from source fallback",
        ],
    },
    "Utility": {
        "construct_status": "defensible_as_protocol_bound_outcome",
        "defensible_construct": "downstream training effect under a frozen protocol",
        "invalid_construct": "universal data usefulness",
        "minimum_behavior_evidence_needed": [
            "equal-token matched baselines",
            "heldout target-domain NLL",
            "external benchmark and retention guardrails",
            "abstain when mandatory evidence is missing",
        ],
    },
}


def _blob(payload: Any) -> str:
    if isinstance(payload, dict):
        return " ".join(_blob(v) for v in payload.values())
    if isinstance(payload, list):
        return " ".join(_blob(v) for v in payload)
    return str(payload).lower()


def build(framework_path: Path, metric_spec_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    framework = load_json(framework_path)
    metric_spec = load_json(metric_spec_path)
    blockers: List[str] = []
    warnings: List[str] = []

    selection_value = framework["core_interpretation"]["selection_value"]
    if selection_value.get("role") != "observable_pre_outcome_selection_evidence":
        blockers.append("selection_value_core_role_mismatch")
    selection_boundary = str(selection_value.get("claim_boundary") or "").lower()
    if "not intrinsic" not in selection_boundary:
        blockers.append("selection_value_boundary_does_not_reject_intrinsic_quality")
    if "no stage-a hard-reject authority" not in selection_boundary:
        blockers.append("selection_value_boundary_missing_no_hard_reject_authority")

    reference_quality = (metric_spec.get("metrics") or {}).get("reference_quality_score") or {}
    quality_blob = _blob(reference_quality)
    if "not intrinsic" not in quality_blob or "not a utility outcome" not in quality_blob:
        blockers.append("reference_quality_score_claim_boundary_too_strong")
    if "ground-truth" not in quality_blob:
        warnings.append("reference_quality_score_should_explicitly_reject_ground_truth_quality")

    utility = framework["core_interpretation"]["utility"]
    if "never" not in str(utility.get("claim_boundary") or "").lower():
        blockers.append("utility_boundary_missing_never_selector_objective")

    rows = []
    for core, review in CORE_REVIEWS.items():
        rows.append(
            {
                "core": core,
                **review,
                "decision": (
                    "rename_or_reframe_required"
                    if core == "Selection Value Evidence"
                    else "keep_with_scope"
                ),
            }
        )

    report = {
        "schema_version": "core-construct-validity-review-v1",
        "status": "core_construct_validity_review_passed" if not blockers else "core_construct_validity_review_failed",
        "source_paths": {
            "framework_contract": str(framework_path),
            "metric_spec": str(metric_spec_path),
        },
        "core_reviews": rows,
        "blockers": blockers,
        "warnings": warnings,
        "decision": {
            "quality_as_intrinsic_core": "rejected",
            "canonical_axis_name": "Selection Value Evidence",
            "quality_axis_runtime_label": "Quality (legacy alias only)",
            "quality_axis_operational_name": "observable_pre_outcome_selection_evidence",
            "core_policy": (
                "A Core is retained only when it represents a defensible operational "
                "failure mode or validation responsibility. If a metric cannot represent "
                "that construct behaviorally, the metric must be demoted to diagnostic."
            ),
        },
        "claim_boundary": (
            "This review does not prove metric validity. It records which Core constructs "
            "are defensible and what behavior evidence is still required."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Core Construct Validity Review",
        "",
        f"Status: `{report['status']}`",
        "",
        report["claim_boundary"],
        "",
        "## Decision",
        "",
        f"- Quality as intrinsic Core: `{report['decision']['quality_as_intrinsic_core']}`",
        f"- Canonical axis: `{report['decision']['canonical_axis_name']}`",
        f"- Runtime label retained for compatibility: `{report['decision']['quality_axis_runtime_label']}`",
        f"- Operational name: `{report['decision']['quality_axis_operational_name']}`",
        "",
        "## Core Reviews",
        "",
        "| Core | Construct Status | Defensible Construct | Invalid Construct | Decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["core_reviews"]:
        lines.append(
            f"| {row['core']} | `{row['construct_status']}` | {row['defensible_construct']} | {row['invalid_construct']} | `{row['decision']}` |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- `{b}`" for b in report["blockers"]] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- `{w}`" for w in report["warnings"]] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Core construct-validity review.")
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--metric-spec", type=Path, default=DEFAULT_METRIC_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.framework, args.metric_spec, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"], "warnings": report["warnings"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
