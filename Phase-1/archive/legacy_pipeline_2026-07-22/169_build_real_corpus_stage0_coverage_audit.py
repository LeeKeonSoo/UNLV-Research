#!/usr/bin/env python3
"""Audit real-corpus Stage-0 and Coverage metadata support."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, save_json


DEFAULT_STAGE0_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined"
DEFAULT_STAGE_A_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_balanced" / "train" / "stage_a_pass.jsonl"
DEFAULT_STAGE_B_SELECTED_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2" / "curated_v2_equal_budget.jsonl"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "real_corpus_stage0_coverage_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "real_corpus_stage0_coverage_audit.md"
SPLITS = ("train", "development", "confirmatory")
REQUIRED_STAGE0_PROVENANCE_KEYS = ("source_name", "source_uri", "collected_at", "original_sha256", "normalized_sha256")


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _provenance(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("provenance")
    return value if isinstance(value, dict) else {}


def _coverage_buckets(row: Dict[str, Any]) -> Dict[str, Any]:
    evidence = row.get("stage_b_evidence")
    if not isinstance(evidence, dict):
        return {}
    buckets = evidence.get("coverage_buckets")
    return buckets if isinstance(buckets, dict) else {}


def _field(row: Dict[str, Any], name: str, default: str = "missing") -> str:
    if row.get(name) not in (None, ""):
        return str(row.get(name))
    provenance = _provenance(row)
    if provenance.get(name) not in (None, ""):
        return str(provenance.get(name))
    buckets = _coverage_buckets(row)
    if buckets.get(name) not in (None, ""):
        return str(buckets.get(name))
    return default


def _path_family(row: Dict[str, Any]) -> str:
    # Prefer the raw/provenance path so Stage-A and selected arms are compared
    # at the same granularity even if selector evidence stores finer buckets.
    path = _field(row, "path")
    if path != "missing":
        normalized = path.replace("\\", "/").strip("/")
        if not normalized:
            return "root"
        return normalized.split("/", 1)[0] if "/" in normalized else "root"
    buckets = _coverage_buckets(row)
    if buckets.get("path_family") not in (None, ""):
        return str(buckets["path_family"])
    return "missing"


def _domain_support_scope(rows: Iterable[Dict[str, Any]]) -> str:
    explicit = 0
    fallback = 0
    for row in rows:
        if _field(row, "domain", "") or _field(row, "domain_bucket", ""):
            explicit += 1
        elif _field(row, "repository_identity") != "missing" or _field(row, "source") != "missing":
            fallback += 1
    if explicit and not fallback:
        return "explicit_domain_metadata"
    if explicit and fallback:
        return "mixed_domain_and_source_bucket"
    if fallback:
        return "source_or_repository_bucket_fallback"
    return "no_domain_or_source_support"


def _distribution(rows: List[Dict[str, Any]], axis: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if axis == "path_family":
            counts[_path_family(row)] += 1
        else:
            counts[_field(row, axis)] += 1
    return counts


def _distribution_support(original: Counter[str], selected: Counter[str]) -> Dict[str, Any]:
    original_total = sum(original.values())
    selected_total = sum(selected.values())
    retained = sorted(bucket for bucket in original if selected.get(bucket, 0) > 0)
    missing = sorted(bucket for bucket in original if selected.get(bucket, 0) == 0)
    retained_ratio = (len(retained) / len(original)) if original else 0.0
    similarity = 0.0
    if original_total and selected_total:
        buckets = set(original) | set(selected)
        similarity = sum(
            min(original.get(bucket, 0) / original_total, selected.get(bucket, 0) / selected_total)
            for bucket in buckets
        )
    return {
        "original_bucket_count": len(original),
        "selected_bucket_count": len(selected),
        "retained_bucket_ratio": round(retained_ratio, 6),
        "distribution_similarity": round(similarity, 6),
        "missing_selected_buckets": missing[:20],
        "original_counts_top": dict(original.most_common(10)),
        "selected_counts_top": dict(selected.most_common(10)),
    }


def _metadata_summary(rows: List[Dict[str, Any]], axes: List[str]) -> Dict[str, Any]:
    total = len(rows)
    summary: Dict[str, Any] = {"record_count": total}
    for axis in axes:
        counts = _distribution(rows, axis)
        missing = int(counts.get("missing", 0))
        top_count = counts.most_common(1)[0][1] if counts else 0
        summary[axis] = {
            "observed_bucket_count": len([bucket for bucket in counts if bucket != "missing"]),
            "missing_count": missing,
            "missing_rate": round((missing / total) if total else 0.0, 6),
            "top_bucket_share": round((top_count / total) if total else 0.0, 6),
            "top_counts": dict(counts.most_common(10)),
        }
    return summary


def _stage0_summary(stage0_dir: Path) -> Dict[str, Any]:
    release_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        release = list(_jsonl(stage0_dir / split / "release_candidates.jsonl"))
        quarantine = list(_jsonl(stage0_dir / split / "quarantined_candidates.jsonl"))
        release_rows.extend(release)
        quarantine_rows.extend(quarantine)
        split_counts[split] = {"release_candidates": len(release), "quarantined_candidates": len(quarantine)}

    rights = Counter(str((row.get("rights") or {}).get("status") or "missing") for row in release_rows + quarantine_rows)
    release_eligible = sum(1 for row in release_rows if bool((row.get("release_eligibility") or {}).get("eligible")))
    release_ineligible = len(release_rows) - release_eligible
    quarantine_reasons: Counter[str] = Counter()
    hazard_true: Counter[str] = Counter()
    missing_provenance: Counter[str] = Counter()
    source_pools = Counter(str(row.get("code_domain_v2_source_pool") or "missing") for row in release_rows + quarantine_rows)
    for row in release_rows + quarantine_rows:
        provenance = _provenance(row)
        for key in REQUIRED_STAGE0_PROVENANCE_KEYS:
            if provenance.get(key) in (None, ""):
                missing_provenance[key] += 1
        for reason in (row.get("quarantine") or {}).get("reasons") or []:
            quarantine_reasons[str(reason)] += 1
        for key, value in (row.get("hazards") or {}).items():
            if isinstance(value, bool) and value:
                hazard_true[str(key)] += 1
    return {
        "stage0_dir": str(stage0_dir),
        "split_counts": split_counts,
        "release_candidate_count": len(release_rows),
        "quarantined_candidate_count": len(quarantine_rows),
        "release_eligible_count": release_eligible,
        "release_ineligible_count": release_ineligible,
        "rights_status_counts": dict(sorted(rights.items())),
        "source_pool_counts": dict(sorted(source_pools.items())),
        "quarantine_reason_counts": dict(sorted(quarantine_reasons.items())),
        "hazard_true_counts": dict(sorted(hazard_true.items())),
        "missing_required_provenance_counts": dict(sorted(missing_provenance.items())),
    }


def _coverage_summary(stage_a_rows: List[Dict[str, Any]], selected_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    axes = ["repository_identity", "bundle_id", "content_type", "change_type", "path_family"]
    supports = {
        axis: _distribution_support(_distribution(stage_a_rows, axis), _distribution(selected_rows, axis)) for axis in axes
    }
    return {
        "stage_a": _metadata_summary(stage_a_rows, axes),
        "selected": _metadata_summary(selected_rows, axes),
        "support_scope": _domain_support_scope(stage_a_rows),
        "true_domain_coverage_claim_allowed": _domain_support_scope(stage_a_rows) == "explicit_domain_metadata",
        "distribution_support": supports,
    }


def build(stage0_dir: Path, stage_a_path: Path, selected_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    blockers: List[str] = []
    caveats: List[str] = []
    if not stage0_dir.exists():
        blockers.append(f"stage0_dir_missing:{stage0_dir}")
    if not stage_a_path.exists():
        blockers.append(f"stage_a_path_missing:{stage_a_path}")
    if not selected_path.exists():
        blockers.append(f"selected_path_missing:{selected_path}")
    if blockers:
        report = {
            "schema_version": "real-corpus-stage0-coverage-audit-v1",
            "status": "real_corpus_stage0_coverage_audit_failed",
            "blockers": blockers,
            "caveats": caveats,
        }
        save_json(output_path, report)
        md_output_path.parent.mkdir(parents=True, exist_ok=True)
        md_output_path.write_text(_render_markdown(report), encoding="utf-8")
        return report

    stage0 = _stage0_summary(stage0_dir)
    stage_a_rows = list(_jsonl(stage_a_path))
    selected_rows = list(_jsonl(selected_path))
    coverage = _coverage_summary(stage_a_rows, selected_rows)

    if stage0["release_candidate_count"] <= 0:
        blockers.append("stage0_release_candidates_empty")
    if stage_a_rows and not selected_rows:
        blockers.append("stage_b_selected_rows_empty")
    if not stage_a_rows:
        blockers.append("stage_a_rows_empty")
    if stage0["release_ineligible_count"]:
        blockers.append("stage0_release_file_contains_ineligible_records")
    if stage0["missing_required_provenance_counts"]:
        blockers.append("stage0_required_provenance_incomplete")

    for axis, axis_summary in coverage["stage_a"].items():
        if isinstance(axis_summary, dict) and axis_summary.get("missing_rate", 0.0) > 0.0:
            caveats.append(f"stage_a_{axis}_metadata_missing")
    for axis, support in coverage["distribution_support"].items():
        if axis in {"content_type", "path_family"} and support["retained_bucket_ratio"] < 1.0:
            caveats.append(f"selected_missing_stage_a_{axis}_buckets")
        if axis == "repository_identity" and support["retained_bucket_ratio"] < 0.8:
            caveats.append("selected_repository_retention_below_80_percent")
    if not coverage["true_domain_coverage_claim_allowed"]:
        caveats.append("true_domain_coverage_not_claimable_without_explicit_domain_metadata")
    caveats.extend(
        [
            "real_corpus_audit_is_metadata_support_not_metric_validity_proof",
            "stage0_hazard_counts_do_not_replace_production_detector_validation",
        ]
    )

    status = (
        "real_corpus_stage0_coverage_audit_failed"
        if blockers
        else "real_corpus_stage0_coverage_audit_passed_with_scope_caveats"
    )
    report = {
        "schema_version": "real-corpus-stage0-coverage-audit-v1",
        "status": status,
        "claim_boundary": (
            "Audits whether the current real corpus carries enough observable metadata for Stage-0 "
            "and Coverage claims. It does not prove intrinsic quality, production hazard-detector "
            "validity, or Stage-C Utility."
        ),
        "inputs": {
            "stage0_dir": str(stage0_dir),
            "stage_a_path": str(stage_a_path),
            "selected_path": str(selected_path),
        },
        "stage0": stage0,
        "coverage": coverage,
        "blockers": blockers,
        "caveats": sorted(set(caveats)),
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Real-Corpus Stage-0 Coverage Audit",
        "",
        f"Status: `{report['status']}`",
        "",
    ]
    if "claim_boundary" in report:
        lines.extend([str(report["claim_boundary"]), ""])
    if "stage0" in report:
        stage0 = report["stage0"]
        lines.extend(
            [
                "## Stage 0",
                "",
                f"- Release candidates: `{stage0['release_candidate_count']}`",
                f"- Quarantined candidates: `{stage0['quarantined_candidate_count']}`",
                f"- Rights status: `{stage0['rights_status_counts']}`",
                f"- Quarantine reasons: `{stage0['quarantine_reason_counts']}`",
                "",
            ]
        )
    if "coverage" in report:
        coverage = report["coverage"]
        lines.extend(
            [
                "## Coverage Metadata",
                "",
                f"- Support scope: `{coverage['support_scope']}`",
                f"- True domain coverage claim allowed: `{coverage['true_domain_coverage_claim_allowed']}`",
                "",
                "| Axis | Original Buckets | Selected Buckets | Retained Ratio | Similarity |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for axis, support in coverage["distribution_support"].items():
            lines.append(
                f"| `{axis}` | `{support['original_bucket_count']}` | `{support['selected_bucket_count']}` | "
                f"`{support['retained_bucket_ratio']}` | `{support['distribution_similarity']}` |"
            )
        lines.append("")
    lines.extend(["## Blockers", ""])
    lines.extend([f"- `{blocker}`" for blocker in report.get("blockers") or []] or ["- None"])
    lines.extend(["", "## Caveats", ""])
    lines.extend([f"- `{caveat}`" for caveat in report.get("caveats") or []] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real-corpus Stage-0 and Coverage metadata audit.")
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--stage-a-path", type=Path, default=DEFAULT_STAGE_A_PATH)
    parser.add_argument("--selected-path", type=Path, default=DEFAULT_STAGE_B_SELECTED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.stage0_dir, args.stage_a_path, args.selected_path, args.output, args.md_output)
    print({"status": report["status"], "blockers": report.get("blockers") or [], "caveats": report.get("caveats") or []})
    return 0 if not report.get("blockers") else 2


if __name__ == "__main__":
    raise SystemExit(main())
