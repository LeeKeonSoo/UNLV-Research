#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]

CONFIG_PATH = Path("configs") / "math_domain_selector_v3_materialization.json"
REDESIGN_CONTRACT = Path("configs") / "math_domain_selector_v3_redesign_contract.json"
FIXTURES_PATH = Path("validation") / "fixtures" / "math_failure_selector_cases.json"
VALIDATION_REPORT = OUTPUT_DIR / "validation" / "math_domain_selector_v3_materialization_report.json"
VALIDATION_MD_REPORT = OUTPUT_DIR / "validation" / "math_domain_selector_v3_materialization_report.md"


def _jsonl(path: Path) -> list[JsonMap]:
    rows: list[JsonMap] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _token(row: JsonMap) -> int:
    value = row.get("token_proxy_count", row.get("token_proxy", 0))
    if isinstance(value, bool):
        return 0
    return int(value) if isinstance(value, int | float | str) and str(value).strip() else 0


def _summary(rows: list[JsonMap]) -> JsonMap:
    buckets = sorted({str(row.get("style_bucket", "missing")) for row in rows})
    return {
        "records": len(rows),
        "token_proxy_count": sum(_token(row) for row in rows),
        "style_bucket_counts": {bucket: sum(1 for row in rows if row.get("style_bucket") == bucket) for bucket in buckets},
        "style_token_counts": {bucket: sum(_token(row) for row in rows if row.get("style_bucket") == bucket) for bucket in buckets},
    }


def _arm(row: JsonMap, arm: str, source: str, reason: str) -> JsonMap:
    return {
        "arm": arm,
        "chunk_uid": row["chunk_uid"],
        "text": row["text"],
        "token_proxy_count": _token(row),
        "source_pool": source,
        "domain": "math",
        "stage_a_pass": row.get("stage_a_pass"),
        "style_bucket": row.get("style_bucket"),
        "stage_b_evidence": {
            "selector_v3_policy": "preoutcome_broader_curated_pool",
            "style_bucket": row.get("style_bucket"),
            "token_proxy_count": _token(row),
        },
        "stage_b_selection_reason": reason,
    }


def _fixture_categories_complete(contract: JsonMap, fixtures: list[JsonMap]) -> bool:
    required = set(contract["required_fixture_categories"])
    observed = {str(row.get("category")) for row in fixtures}
    mapped = set(contract["preoutcome_feature_map"])
    return required <= observed and required <= mapped


def _retention_fraction(selected: list[JsonMap], stage_a: list[JsonMap], style: str | None = None) -> float:
    selected_tokens = sum(_token(row) for row in selected if style is None or row.get("style_bucket") == style)
    stage_a_tokens = sum(_token(row) for row in stage_a if style is None or row.get("style_bucket") == style)
    return selected_tokens / stage_a_tokens if stage_a_tokens else 1.0


def build(config_path: Path, output_path: Path, md_output_path: Path) -> JsonMap:
    config = load_json(config_path)
    contract = load_json(REDESIGN_CONTRACT)
    fixtures = load_json(FIXTURES_PATH)
    input_dir = Path(str(config["input_materialization_v2"]))
    output_dir = Path(str(config["output_dir"]))
    stage_a = _jsonl(input_dir / "stageA_full_natural.jsonl")
    raw = _jsonl(input_dir / "raw_full_natural.jsonl")
    curated_v2 = _jsonl(input_dir / "curated_math_v2_natural.jsonl")
    excluded = set(config["stage_b"]["excluded_style_buckets"])
    broader = [
        _arm(row, "curated_math_v3_natural", "stage_a_broader_curated_pool", "retain_stage_a_except_excluded_style")
        for row in stage_a
        if str(row.get("style_bucket")) not in excluded
    ]
    retain_all = [_arm(row, "retain_all_if_budget_allows", "stage_a_full_pool", "retain_all_no_budget_constraint") for row in stage_a]
    arms = {
        "raw_full_natural": raw,
        "stageA_full_natural": stage_a,
        "curated_math_v2_natural": curated_v2,
        "retain_all_if_budget_allows": retain_all,
        "broader_curated_pool": broader,
        "curated_math_v3_natural": broader,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in arms.items():
        normalized = rows if name in {"raw_full_natural", "stageA_full_natural", "curated_math_v2_natural"} else rows
        _write_jsonl(output_dir / f"{name}.jsonl", normalized)
    retention = {
        "stage_a_token_retention_fraction": _retention_fraction(broader, stage_a),
        "proof_or_theorem_token_retention_fraction": _retention_fraction(broader, stage_a, "proof_or_theorem"),
    }
    blockers = [
        name
        for name, passed in {
            "preoutcome_mapping_incomplete": _fixture_categories_complete(contract, fixtures),
            "stage_a_retention_too_low": retention["stage_a_token_retention_fraction"]
            >= float(config["stage_b"]["minimum_stage_a_token_retention_fraction"]),
            "proof_or_theorem_retention_too_low": retention["proof_or_theorem_token_retention_fraction"]
            >= float(config["stage_b"]["minimum_proof_or_theorem_token_retention_fraction"]),
            "utility_outcomes_forbidden": config["utility_scope"] == "Stage C validation only; never selector objective",
        }.items()
        if not passed
    ]
    report = {
        "schema_version": "math-domain-selector-v3-materialization-report-v1",
        "status": "math_selector_v3_materialized" if not blockers else "math_selector_v3_blocked",
        "utility_outcomes_read": False,
        "v2_failure_preserved": contract["source_failure"]["math_v2_decision"] == "failed_stage_c_validation",
        "preoutcome_mapping_complete": _fixture_categories_complete(contract, fixtures),
        "v3_ready_for_stage_c_freeze": not blockers,
        "selection_mode": config["stage_b"]["selection_mode"],
        "arms": {name: _summary(rows) for name, rows in arms.items()},
        "retention_checks": retention,
        "blockers": blockers,
        "selector_forbidden_signals": config["stage_b"]["selector_forbidden_signals"],
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
        "source_sha256": {
            str(config_path): sha256_file(config_path),
            str(REDESIGN_CONTRACT): sha256_file(REDESIGN_CONTRACT),
            str(FIXTURES_PATH): sha256_file(FIXTURES_PATH),
            str(input_dir / "stageA_full_natural.jsonl"): sha256_file(input_dir / "stageA_full_natural.jsonl"),
            str(input_dir / "curated_math_v2_natural.jsonl"): sha256_file(input_dir / "curated_math_v2_natural.jsonl"),
        },
        "next_required_step": "Freeze Math selector v3 Stage-C protocol and run natural-budget training/evaluation before claiming success.",
    }
    save_json(output_dir / "math_selector_v3_materialization_report.json", report)
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Math Selector v3 Materialization Report",
        "",
        f"Status: `{report['status']}`",
        f"Selection mode: `{report['selection_mode']}`",
        f"V3 ready for Stage-C freeze: `{report['v3_ready_for_stage_c_freeze']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Retention Checks",
        "",
        f"- Stage-A token retention: `{report['retention_checks']['stage_a_token_retention_fraction']:.6f}`",
        f"- Proof/theorem token retention: `{report['retention_checks']['proof_or_theorem_token_retention_fraction']:.6f}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{item}`" for item in report["blockers"]] or ["- None"])
    lines.extend(["", "## Next Required Step", "", str(report["next_required_step"]), ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Math selector v3.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=VALIDATION_REPORT)
    parser.add_argument("--md-output", type=Path, default=VALIDATION_MD_REPORT)
    args = parser.parse_args()
    report = build(args.config, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if report["status"] == "math_selector_v3_materialized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
