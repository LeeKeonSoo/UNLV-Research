#!/usr/bin/env python3
"""Run isolated Quality-policy development arms without changing A-B-C runtime."""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from composition_audit import annotate_records, build_composition_audit
from curation_artifacts import save_json, sha256_file
from general_web_span_compaction import build_plan, materialize_candidate_plan
from reason_code_audit import build_reason_code_impact_audit
from stage_c_selection import select_chunks


JsonMap = dict[str, Any]
TokenCounter = Callable[[str], int]
QUALITY_RULE_ARMS = {
    "explicit_generated_artifact": {"explicit_generated_artifact": True},
    "license_comment_only": {"license_comment_only_chunk": True},
    "empty_html_shell": {"empty_html_shell": True},
    "web_chrome_only": {"web_chrome_only_chunk": True},
}
QUALITY_CANDIDATE_ARMS = {
    "explicit_error_navigation_only_candidate": {
        "explicit_error_navigation_only_chunk_candidate": True
    },
    "url_directory_only_candidate": {"url_directory_only_chunk_candidate": True},
}


def _copy_rows(rows: Iterable[JsonMap]) -> list[JsonMap]:
    return [copy.deepcopy(row) for row in rows]


def _selection_config(enabled_rules: JsonMap) -> JsonMap:
    return {
        "near_duplicate_compaction": {"candidate_enabled": False, "shingle_size": 5, "minimum_lexical_tokens": 40, "symmetric_overlap_threshold": 0.95},
        "structural_artifact_rules": enabled_rules,
    }


def _composition(source_rows: list[JsonMap], curated_rows: list[JsonMap]) -> JsonMap:
    return build_composition_audit(
        {"raw_input": annotate_records(_copy_rows(source_rows)), "stage_c_curated": annotate_records(_copy_rows(curated_rows))}
    )


def _coverage(not_selected: list[JsonMap], transformations: list[JsonMap], minimum_chunk_chars: int, curated: list[JsonMap]) -> JsonMap:
    explicit_reasons = {
        "explicit_generated_artifact",
        "license_comment_only_chunk",
        "empty_html_shell",
        "explicit_web_chrome_only_chunk",
        "explicit_error_navigation_only_chunk",
        "url_directory_only_chunk",
    }
    quality_removals = [row for row in not_selected if str(row.get("stage_c_selection", {}).get("removed_reason")) in explicit_reasons]
    residual_valid = all(len(str(row.get("text") or "").strip()) >= minimum_chunk_chars for row in curated if row.get("chunk_uid") in {item["chunk_uid"] for item in transformations})
    return {
        "authority": "audit_only",
        "quality_whole_chunk_removals": len(quality_removals),
        "span_transformations": len(transformations),
        "residual_payload_passed": residual_valid,
        "passed": residual_valid,
    }


def _arm(
    arm_id: str,
    source_rows: list[JsonMap],
    transformed_rows: list[JsonMap],
    transformations: list[JsonMap],
    enabled_rules: JsonMap,
    minimum_chunk_chars: int,
    token_counter: TokenCounter,
    runtime_active: bool,
) -> JsonMap:
    curated, not_selected, selection = select_chunks(_copy_rows(transformed_rows), _selection_config(enabled_rules))
    return {
        "runtime_active": runtime_active,
        "summary": {
            "input_chunks": len(source_rows),
            "curated_chunks": len(curated),
            "input_tokens": sum(token_counter(str(row["text"])) for row in source_rows),
            "curated_tokens": sum(token_counter(str(row["text"])) for row in curated),
            "transformed_span_count": len(transformations),
        },
        "stage_c_selection": selection,
        "reason_code_impact_audit": build_reason_code_impact_audit([], [], not_selected, transformations),
        "coverage": _coverage(not_selected, transformations, minimum_chunk_chars, curated),
        "composition": _composition(source_rows, curated),
        "selector_boundary": {"source_identity_read": False, "composition_read": False, "utility_read": False, "benchmark_outcomes_read": False, "target_retention_fraction_read": False},
    }


def run_quality_matrix(rows: Iterable[JsonMap], *, minimum_chunk_chars: int, token_counter: TokenCounter) -> JsonMap:
    """Materialize rule-isolated Quality arms for development evidence only."""
    if minimum_chunk_chars < 1:
        raise ValueError("minimum_chunk_chars must be positive")
    source_rows = _copy_rows(rows)
    if not source_rows:
        raise ValueError("Quality development matrix requires at least one Stage-B pass chunk")
    arms = {
        "baseline": _arm("baseline", source_rows, source_rows, [], {}, minimum_chunk_chars, token_counter, False),
        **{
            arm_id: _arm(arm_id, source_rows, source_rows, [], rules, minimum_chunk_chars, token_counter, False)
            for arm_id, rules in QUALITY_RULE_ARMS.items()
        },
        **{
            arm_id: _arm(arm_id, source_rows, source_rows, [], rules, minimum_chunk_chars, token_counter, False)
            for arm_id, rules in QUALITY_CANDIDATE_ARMS.items()
        },
        "all_active_quality": _arm("all_active_quality", source_rows, source_rows, [], {rule: True for rules in QUALITY_RULE_ARMS.values() for rule in rules}, minimum_chunk_chars, token_counter, False),
    }
    plan = build_plan(
        source_rows,
        minimum_residual_chars=minimum_chunk_chars,
        token_counter=token_counter,
    )
    candidate = materialize_candidate_plan(source_rows, plan, token_counter=token_counter)
    arms["web_control_span_candidate"] = _arm("web_control_span_candidate", source_rows, candidate["records"], candidate["transformations"], {}, minimum_chunk_chars, token_counter, False)
    cumulative_rules = {
        rule: True
        for group in (*QUALITY_RULE_ARMS.values(), *QUALITY_CANDIDATE_ARMS.values())
        for rule in group
    }
    arms["cumulative_quality_candidate"] = _arm(
        "cumulative_quality_candidate",
        source_rows,
        candidate["records"],
        candidate["transformations"],
        cumulative_rules,
        minimum_chunk_chars,
        token_counter,
        False,
    )
    return {
        "schema_version": "quality-rule-development-matrix-v1",
        "status": "development_only_not_runtime_active",
        "runtime_active": False,
        "rules": {
            "active_quality": sorted(QUALITY_RULE_ARMS),
            "candidate_quality": [
                "stage_c_explicit_error_navigation_only_candidate",
                "stage_c_explicit_web_control_span_candidate",
                "stage_c_url_directory_only_candidate",
            ],
        },
        "arms": arms,
        "claim_boundary": "Rule-isolated structural evidence only. No arm can modify an active profile, read benchmark outcomes, or establish downstream effectiveness.",
    }


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_report(input_path: Path, output_path: Path, minimum_chunk_chars: int) -> JsonMap:
    report = run_quality_matrix(_read_jsonl(input_path), minimum_chunk_chars=minimum_chunk_chars, token_counter=lambda text: len(text.split()))
    report["input"] = {"path": str(input_path), "sha256": sha256_file(input_path), "token_count_kind": "whitespace_proxy"}
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run development-only isolated Quality-policy arms.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-chunk-chars", type=int, default=40)
    args = parser.parse_args()
    report = _write_report(args.input, args.output, args.minimum_chunk_chars)
    print(json.dumps({"status": report["status"], "arms": list(report["arms"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
