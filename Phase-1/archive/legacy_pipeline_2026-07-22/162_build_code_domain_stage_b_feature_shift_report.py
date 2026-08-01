#!/usr/bin/env python3
"""Build selected-vs-budget-not-selected Stage-B diagnostics for code-domain v2."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_STAGE_B_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2"
DEFAULT_SCORED = DEFAULT_STAGE_B_DIR / "train_scored_full_selector.jsonl"
DEFAULT_SELECTED = DEFAULT_STAGE_B_DIR / "curated_v2_equal_budget.jsonl"
DEFAULT_STAGE_B_REPORT = DEFAULT_STAGE_B_DIR / "stage_b_v2_arms_report.json"
DEFAULT_FRAMEWORK = Path("configs") / "lm_curation_operational_framework_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_stage_b_feature_shift_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_stage_b_feature_shift_report.md"


NUMERIC_FEATURES = [
    "token_proxy_count",
    "length_support",
    "structural_richness",
    "lexical_or_identifier_diversity",
    "code_quality_proxy",
    "ast_node_count",
    "semantic_token_proxy_count",
    "pass_through_assignment_ratio",
    "soft_lexical_redundancy_risk",
    "soft_structural_redundancy_risk",
    "soft_redundancy_risk",
    "soft_redundancy_support",
    "stage_b_objective_score",
]

BUCKET_FEATURES = [
    "content_type",
    "chunk_kind",
    "difficulty_band",
    "length_bucket",
    "quality_bucket",
    "redundancy_risk_bucket",
    "path_family",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}


def _coverage(row: Dict[str, Any]) -> Dict[str, str]:
    evidence = _evidence(row)
    buckets = evidence.get("coverage_buckets") if isinstance(evidence.get("coverage_buckets"), dict) else {}
    return {str(key): str(value) for key, value in buckets.items()}


def _text(row: Dict[str, Any]) -> str:
    return str(row.get("text") or "")


def _token_count(row: Dict[str, Any]) -> int:
    evidence = _evidence(row)
    return int(evidence.get("token_proxy_count") or row.get("token_proxy_count") or len(_text(row).split()))


def _length_bucket(tokens: int) -> str:
    if tokens < 64:
        return "lt_64"
    if tokens < 128:
        return "64_127"
    if tokens < 256:
        return "128_255"
    if tokens < 512:
        return "256_511"
    return "ge_512"


def _quality_bucket(score: float) -> str:
    if score < 0.60:
        return "quality_lt_0_60"
    if score < 0.75:
        return "quality_0_60_0_75"
    if score < 0.85:
        return "quality_0_75_0_85"
    if score < 0.92:
        return "quality_0_85_0_92"
    return "quality_ge_0_92"


def _risk_bucket(score: float) -> str:
    if score < 0.05:
        return "risk_lt_0_05"
    if score < 0.20:
        return "risk_0_05_0_20"
    if score < 0.50:
        return "risk_0_20_0_50"
    return "risk_ge_0_50"


def _identifier_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)


def _has_api_usage(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"^\s*(from\s+\S+\s+import|import\s+\S+)", text, flags=re.MULTILINE)
        or re.search(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\(", text)
        or any(marker in lowered for marker in ("client.", "session.", "request.", "response.", "api", "endpoint"))
    )


def _has_test_or_regression_signal(row: Dict[str, Any]) -> bool:
    text = _text(row).lower()
    path = str(row.get("path") or "").lower()
    content_type = str(row.get("content_type") or _coverage(row).get("content_type") or "").lower()
    return bool(
        content_type == "test"
        or "test" in path
        or any(
            marker in text
            for marker in (
                "assert ",
                "pytest",
                "unittest",
                "expected",
                "raises",
                "regression",
                "bug",
                "fix",
                "issue",
                "mock",
                "fixture",
            )
        )
    )


def _has_concise_example_signal(row: Dict[str, Any]) -> bool:
    text = _text(row)
    lowered = text.lower()
    tokens = _identifier_tokens(text)
    has_code_shape = bool(
        re.search(r"^\s*(def|class|async def)\s+", text, flags=re.MULTILINE)
        or re.search(r"^\s*(from\s+\S+\s+import|import\s+\S+)", text, flags=re.MULTILINE)
        or "assert " in lowered
    )
    has_example_word = any(marker in lowered for marker in ("example", "usage", "demo", "sample", "fixture"))
    return bool(has_code_shape or has_example_word or len(set(tokens)) >= 5)


def _has_template_or_boilerplate_risk(row: Dict[str, Any]) -> bool:
    text = _text(row).lower()
    evidence = _evidence(row)
    pass_through = _safe_float(evidence.get("pass_through_assignment_ratio"))
    structural_risk = _safe_float(evidence.get("soft_structural_redundancy_risk"))
    return bool(
        pass_through >= 0.25
        or structural_risk >= 0.70
        or any(marker in text for marker in ("generated by", "do not edit", "copyright", "license"))
    )


def _feature_tags(row: Dict[str, Any]) -> Dict[str, bool]:
    tokens = _token_count(row)
    concise = tokens <= 160
    test_or_regression = _has_test_or_regression_signal(row)
    api_usage = _has_api_usage(_text(row))
    concise_example = concise and _has_concise_example_signal(row)
    concise_useful = concise and (test_or_regression or api_usage or concise_example)
    return {
        "concise_useful_candidate": concise_useful,
        "concise_test_or_regression_candidate": concise and test_or_regression,
        "api_usage_candidate": api_usage,
        "bugfix_or_regression_test_signal": test_or_regression,
        "concise_example_support": concise_example,
        "template_or_boilerplate_risk": _has_template_or_boilerplate_risk(row),
    }


def _row_feature_view(row: Dict[str, Any], selected: bool) -> Dict[str, Any]:
    evidence = _evidence(row)
    coverage = _coverage(row)
    tokens = _token_count(row)
    quality = _safe_float(evidence.get("code_quality_proxy"))
    risk = _safe_float(evidence.get("soft_redundancy_risk"))
    view = {
        "selected": selected,
        "chunk_uid": str(row.get("chunk_uid")),
        "repository_identity": str(row.get("repository_identity") or (row.get("provenance") or {}).get("repository_identity") or ""),
        "path": str(row.get("path") or (row.get("provenance") or {}).get("path") or ""),
        "content_type": str(row.get("content_type") or coverage.get("content_type") or "unknown"),
        "chunk_kind": str(row.get("chunk_kind") or "unknown"),
        "difficulty_band": str(coverage.get("difficulty_band") or "unknown"),
        "path_family": str(coverage.get("path_family") or "unknown"),
        "length_bucket": _length_bucket(tokens),
        "quality_bucket": _quality_bucket(quality),
        "redundancy_risk_bucket": _risk_bucket(risk),
    }
    for feature in NUMERIC_FEATURES:
        if feature == "token_proxy_count":
            view[feature] = float(tokens)
        else:
            view[feature] = _safe_float(evidence.get(feature))
    view.update(_feature_tags(row))
    return view


def _numeric_summary(rows: List[Dict[str, Any]], feature: str) -> Dict[str, Any]:
    values = [float(row[feature]) for row in rows]
    return {
        "mean": round(_mean(values) or 0.0, 6),
        "median": round(_median(values) or 0.0, 6),
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def _bucket_counts(rows: List[Dict[str, Any]], feature: str) -> Dict[str, Any]:
    counts = Counter(str(row.get(feature) or "unknown") for row in rows)
    total = len(rows)
    return {
        key: {"count": count, "share": round(_rate(count, total), 6)}
        for key, count in sorted(counts.items())
    }


def _share(rows: List[Dict[str, Any]], feature: str) -> float:
    return _rate(sum(1 for row in rows if bool(row.get(feature))), len(rows))


def _boolean_shift(
    selected: List[Dict[str, Any]],
    budget_not_selected: List[Dict[str, Any]],
    feature: str,
) -> Dict[str, Any]:
    selected_share = _share(selected, feature)
    not_selected_share = _share(budget_not_selected, feature)
    return {
        "selected_share": round(selected_share, 6),
        "budget_not_selected_share": round(not_selected_share, 6),
        "selected_minus_budget_not_selected": round(selected_share - not_selected_share, 6),
        "selected_count": sum(1 for row in selected if bool(row.get(feature))),
        "budget_not_selected_count": sum(
            1 for row in budget_not_selected if bool(row.get(feature))
        ),
        "legacy_aliases": {
            "rejected_share": round(not_selected_share, 6),
            "selected_minus_rejected": round(selected_share - not_selected_share, 6),
            "rejected_count": sum(
                1 for row in budget_not_selected if bool(row.get(feature))
            ),
        },
    }


def _top_examples(rows: List[Dict[str, Any]], feature: str, *, selected: bool, limit: int = 8) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if bool(row.get(feature))]
    filtered.sort(key=lambda row: (float(row.get("stage_b_objective_score") or 0.0), str(row.get("chunk_uid"))))
    if selected:
        chosen = list(reversed(filtered[-limit:]))
    else:
        chosen = filtered[:limit]
    return [
        {
            "chunk_uid": row["chunk_uid"],
            "path": row["path"],
            "content_type": row["content_type"],
            "chunk_kind": row["chunk_kind"],
            "token_proxy_count": int(row["token_proxy_count"]),
            "code_quality_proxy": round(float(row["code_quality_proxy"]), 6),
            "soft_redundancy_risk": round(float(row["soft_redundancy_risk"]), 6),
            "stage_b_objective_score": round(float(row["stage_b_objective_score"]), 6),
        }
        for row in chosen
    ]


def _risk_flags(
    selected: List[Dict[str, Any]],
    budget_not_selected: List[Dict[str, Any]],
    boolean_shifts: Dict[str, Dict[str, Any]],
    numeric: Dict[str, Dict[str, Any]],
) -> List[str]:
    flags: List[str] = []
    concise = boolean_shifts["concise_useful_candidate"]
    if (
        concise["budget_not_selected_share"] >= 0.10
        and concise["selected_minus_budget_not_selected"] <= -0.05
    ):
        flags.append("concise_useful_candidates_under_selected")
    test = boolean_shifts["concise_test_or_regression_candidate"]
    if (
        test["budget_not_selected_share"] >= 0.05
        and test["selected_minus_budget_not_selected"] <= -0.04
    ):
        flags.append("concise_test_or_regression_candidates_under_selected")
    template = boolean_shifts["template_or_boilerplate_risk"]
    if template["selected_share"] > template["budget_not_selected_share"] + 0.05:
        flags.append("template_or_boilerplate_risk_over_selected")
    if (
        numeric["token_proxy_count"]["selected"]["mean"]
        > numeric["token_proxy_count"]["budget_not_selected"]["mean"] * 1.75
    ):
        flags.append("selector_strongly_prefers_longer_chunks")
    if (
        numeric["ast_node_count"]["selected"]["mean"]
        > numeric["ast_node_count"]["budget_not_selected"]["mean"] * 2.0
    ):
        flags.append("selector_strongly_prefers_ast_rich_chunks")
    return flags


def _summarize(
    scored_rows: List[Dict[str, Any]],
    selected_uids: set[str],
) -> Dict[str, Any]:
    views = [_row_feature_view(row, str(row.get("chunk_uid")) in selected_uids) for row in scored_rows]
    selected = [row for row in views if row["selected"]]
    budget_not_selected = [row for row in views if not row["selected"]]
    numeric: Dict[str, Dict[str, Any]] = {}
    for feature in NUMERIC_FEATURES:
        numeric[feature] = {
            "selected": _numeric_summary(selected, feature),
            "budget_not_selected": _numeric_summary(budget_not_selected, feature),
        }
        numeric[feature]["selected_minus_budget_not_selected_mean"] = round(
            numeric[feature]["selected"]["mean"]
            - numeric[feature]["budget_not_selected"]["mean"],
            6,
        )
        numeric[feature]["legacy_aliases"] = {
            "rejected": numeric[feature]["budget_not_selected"],
            "selected_minus_rejected_mean": numeric[feature][
                "selected_minus_budget_not_selected_mean"
            ],
        }

    buckets = {
        feature: {
            "selected": _bucket_counts(selected, feature),
            "budget_not_selected": _bucket_counts(budget_not_selected, feature),
        }
        for feature in BUCKET_FEATURES
    }
    boolean_features = [
        "concise_useful_candidate",
        "concise_test_or_regression_candidate",
        "api_usage_candidate",
        "bugfix_or_regression_test_signal",
        "concise_example_support",
        "template_or_boilerplate_risk",
    ]
    boolean_shifts = {
        feature: _boolean_shift(selected, budget_not_selected, feature)
        for feature in boolean_features
    }
    flags = _risk_flags(selected, budget_not_selected, boolean_shifts, numeric)
    return {
        "pool_counts": {
            "scored_stage_a_pass_records": len(views),
            "selected_records": len(selected),
            "budget_not_selected_records": len(budget_not_selected),
            "selected_share": round(_rate(len(selected), len(views)), 6),
            "curation_disposition": "retained",
            "budget_not_selected_is_rejection": False,
            "legacy_aliases": {
                "rejected_records": len(budget_not_selected),
            },
        },
        "numeric_feature_shifts": numeric,
        "bucket_feature_distributions": buckets,
        "operational_signal_shifts": boolean_shifts,
        "diagnostic_examples": {
            "selected_concise_useful_examples": _top_examples(selected, "concise_useful_candidate", selected=True),
            "budget_not_selected_concise_useful_examples": _top_examples(
                budget_not_selected,
                "concise_useful_candidate",
                selected=False,
            ),
            "budget_not_selected_concise_test_or_regression_examples": _top_examples(
                budget_not_selected,
                "concise_test_or_regression_candidate",
                selected=False,
            ),
        },
        "risk_flags": flags,
    }


def build(
    scored_path: Path,
    selected_path: Path,
    stage_b_report_path: Path,
    framework_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    scored_rows = _read_jsonl(scored_path)
    selected_rows = _read_jsonl(selected_path)
    selected_uids = {str(row.get("chunk_uid")) for row in selected_rows}
    summary = _summarize(scored_rows, selected_uids)
    stage_b_report = load_json(stage_b_report_path)
    framework = load_json(framework_path)
    missing_selected = sorted(selected_uids - {str(row.get("chunk_uid")) for row in scored_rows})
    blockers = []
    if missing_selected:
        blockers.append(f"selected_uids_missing_from_scored_pool:{len(missing_selected)}")
    if stage_b_report.get("status") != "stage_b_v2_arms_frozen_before_stage_c":
        blockers.append(f"stage_b_report_status_mismatch:{stage_b_report.get('status')}")
    if framework.get("utility_scope") != "Stage C validation only; never selector objective":
        blockers.append("framework_utility_scope_mismatch")

    status = "code_domain_stage_b_feature_shift_report_ready" if not blockers else "code_domain_stage_b_feature_shift_report_blocked"
    report = {
        "schema_version": "code-domain-stage-b-feature-shift-report-v2",
        "status": status,
        "source_sha256": {
            str(scored_path): sha256_file(scored_path),
            str(selected_path): sha256_file(selected_path),
            str(stage_b_report_path): sha256_file(stage_b_report_path),
            str(framework_path): sha256_file(framework_path),
        },
        "summary": summary,
        "blockers": blockers,
        "interpretation": {
            "scope": "Stage-B selected-vs-budget-not-selected feature-shift diagnostic for code-domain v2.",
            "all_compared_records_remain_in_full_curated_pool": True,
            "budget_not_selected_is_rejection": False,
            "not_utility_evidence": True,
            "not_selector_tuning_permission": True,
            "core_boundary": "Selection Value Evidence and Redundancy support optional budget allocation; Utility remains Stage C only.",
        },
        "next_actions": [
            "If concise useful candidates are under-selected, add an outcome-free concise-useful preservation diagnostic before any selector change.",
            "If AST-rich or long chunks dominate, separate structural richness from learnable code usefulness in the next development cycle.",
            "If template risk is over-selected, strengthen generated/template/vendored risk before Stage C.",
        ],
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Code-Domain Stage-B Feature Shift Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "This is a Stage-B diagnostic only. It is not Utility evidence and does not permit selector tuning from Stage-C outcomes.",
        "",
        "## Pool",
        "",
        f"- Scored Stage-A-pass records: {summary['pool_counts']['scored_stage_a_pass_records']}",
        f"- Selected records: {summary['pool_counts']['selected_records']}",
        f"- Budget-not-selected records: {summary['pool_counts']['budget_not_selected_records']}",
        "- All records in both groups remain retained in the full curated pool.",
        f"- Selected share: {summary['pool_counts']['selected_share']}",
        "",
        "## Operational Signals",
        "",
        "| Signal | Selected Share | Budget-Not-Selected Share | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for feature, row in summary["operational_signal_shifts"].items():
        lines.append(
            f"| `{feature}` | {row['selected_share']} | {row['budget_not_selected_share']} | {row['selected_minus_budget_not_selected']} |"
        )
    lines.extend(["", "## Numeric Mean Shifts", "", "| Feature | Selected Mean | Budget-Not-Selected Mean | Difference |", "| --- | ---: | ---: | ---: |"])
    for feature, row in summary["numeric_feature_shifts"].items():
        lines.append(
            f"| `{feature}` | {row['selected']['mean']} | {row['budget_not_selected']['mean']} | {row['selected_minus_budget_not_selected_mean']} |"
        )
    lines.extend(["", "## Risk Flags", ""])
    if summary["risk_flags"]:
        lines.extend(f"- `{flag}`" for flag in summary["risk_flags"])
    else:
        lines.append("- None")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain Stage-B selected-vs-budget-not-selected feature-shift report.")
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--stage-b-report", type=Path, default=DEFAULT_STAGE_B_REPORT)
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.scored, args.selected, args.stage_b_report, args.framework, args.output, args.md_output)
    print({"status": report["status"], "risk_flags": report["summary"]["risk_flags"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
