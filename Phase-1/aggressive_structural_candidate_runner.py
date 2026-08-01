#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from composition_audit import annotate_records, build_composition_audit
from curation_artifacts import save_json, sha256_file
from inline_license_comment_block_compaction import build_plan as build_license_block_plan
from inline_license_comment_block_compaction import materialize_candidate_plan as materialize_license_blocks
from inline_license_header_compaction import build_plan as build_license_header_plan
from inline_license_header_compaction import materialize_candidate_plan as materialize_license_headers
from reason_code_audit import build_reason_code_impact_audit
from span_level_template_compaction import build_plan as build_repeated_span_plan
from span_level_template_compaction import materialize_candidate_plan as materialize_repeated_spans
from stage_c_selection import select_chunks


JsonMap = dict[str, Any]
TokenCounter = Callable[[str], int]
ARM_IDS = (
    "active_profile_baseline",
    "license_span_compaction",
    "repeated_span_compaction",
    "strengthened_duplicate_family",
    "cumulative_aggressive_candidate",
)


def _copy_rows(rows: Iterable[JsonMap]) -> list[JsonMap]:
    return [copy.deepcopy(row) for row in rows]


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _text_only_rows(rows: Iterable[JsonMap]) -> list[JsonMap]:
    """Erase non-text selector inputs before any candidate arm executes."""
    prepared: list[JsonMap] = []
    for raw in rows:
        row = copy.deepcopy(raw)
        row["stage_c_policy_metadata"] = {}
        row["stage_c_selector_visible"] = {
            "declared_language": False,
            "declared_language_version": False,
            "declared_content_type": False,
            "declared_path": False,
            "declared_artifact_context": False,
            "source_name": False,
            "source_pool_role": False,
            "composition": False,
            "utility": False,
            "benchmark_outcomes": False,
        }
        prepared.append(row)
    return prepared


def _run_license_compaction(rows: list[JsonMap], minimum_residual_chars: int) -> tuple[list[JsonMap], list[JsonMap]]:
    header_plan = build_license_header_plan(rows, minimum_residual_chars=minimum_residual_chars)
    header_result = materialize_license_headers(rows, header_plan)
    block_plan = build_license_block_plan(header_result["records"], minimum_residual_chars=minimum_residual_chars)
    block_result = materialize_license_blocks(header_result["records"], block_plan)
    transformations = [*header_result["transformations"], *block_result["transformations"]]
    return block_result["records"], transformations


def _run_repeated_span_compaction(rows: list[JsonMap], minimum_residual_chars: int) -> tuple[list[JsonMap], list[JsonMap]]:
    plan = build_repeated_span_plan(rows, minimum_span_tokens=12, minimum_residual_chars=minimum_residual_chars)
    result = materialize_repeated_spans(rows, plan)
    return result["records"], result["transformations"]


def _selection_config(base: JsonMap, threshold: float | None = None) -> JsonMap:
    config = copy.deepcopy(base)
    duplicate = dict(config.get("near_duplicate_compaction") or {})
    duplicate["candidate_enabled"] = threshold is not None
    if threshold is not None:
        duplicate.update({"shingle_size": 5, "minimum_lexical_tokens": 40, "symmetric_overlap_threshold": threshold})
    config["near_duplicate_compaction"] = duplicate
    return config


def _coverage_audit(source_rows: list[JsonMap], curated_rows: list[JsonMap]) -> JsonMap:
    return build_composition_audit(
        {
            "raw_input": annotate_records(_copy_rows(source_rows)),
            "stage_c_curated": annotate_records(_copy_rows(curated_rows)),
        }
    )


def _build_arm(
    *,
    arm_id: str,
    source_rows: list[JsonMap],
    transformed_rows: list[JsonMap],
    transformations: list[JsonMap],
    stage_c_selection: JsonMap,
    token_counter: TokenCounter,
) -> JsonMap:
    selected, not_selected, selection_audit = select_chunks(transformed_rows, stage_c_selection)
    reason_audit = build_reason_code_impact_audit(
        stage_a_quarantined=[],
        stage_b_rejected=[],
        stage_c_not_selected=not_selected,
        stage_c_transformations=transformations,
    )
    return {
        "arm_id": arm_id,
        "runtime_active": False,
        "stage_c_selection": selection_audit,
        "curated_chunks": selected,
        "not_selected_chunks": not_selected,
        "transformations": transformations,
        "reason_code_impact_audit": reason_audit,
        "coverage_impact_audit": _coverage_audit(source_rows, selected),
        "summary": {
            "input_chunks": len(source_rows),
            "transformed_span_count": len(transformations),
            "not_selected_chunks": len(not_selected),
            "curated_chunks": len(selected),
            "input_token_count": sum(token_counter(str(row["text"])) for row in source_rows),
            "curated_token_count": sum(token_counter(str(row["text"])) for row in selected),
        },
        "selector_boundary": {
            "source_identity_read": False,
            "source_pool_role_read": False,
            "composition_read": False,
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_retention_fraction_read": False,
        },
    }


def run_candidate_arms(
    rows: Iterable[JsonMap], *, stage_c_selection: JsonMap, minimum_residual_chars: int, token_counter: TokenCounter
) -> JsonMap:
    """Build isolated text-only development arms without changing the active runtime."""
    if minimum_residual_chars < 1:
        raise ValueError("minimum_residual_chars must be positive")
    source_rows = _text_only_rows(rows)
    license_rows, license_transformations = _run_license_compaction(_copy_rows(source_rows), minimum_residual_chars)
    repeated_rows, repeated_transformations = _run_repeated_span_compaction(_copy_rows(source_rows), minimum_residual_chars)
    cumulative_license_rows, cumulative_license_transformations = _run_license_compaction(
        _copy_rows(source_rows), minimum_residual_chars
    )
    cumulative_rows, cumulative_repeated_transformations = _run_repeated_span_compaction(
        cumulative_license_rows, minimum_residual_chars
    )
    arms = {
        "active_profile_baseline": _build_arm(
            arm_id="active_profile_baseline", source_rows=source_rows, transformed_rows=_copy_rows(source_rows), transformations=[],
            stage_c_selection=_selection_config(stage_c_selection), token_counter=token_counter,
        ),
        "license_span_compaction": _build_arm(
            arm_id="license_span_compaction", source_rows=source_rows, transformed_rows=license_rows,
            transformations=license_transformations, stage_c_selection=_selection_config(stage_c_selection), token_counter=token_counter,
        ),
        "repeated_span_compaction": _build_arm(
            arm_id="repeated_span_compaction", source_rows=source_rows, transformed_rows=repeated_rows,
            transformations=repeated_transformations, stage_c_selection=_selection_config(stage_c_selection), token_counter=token_counter,
        ),
        "strengthened_duplicate_family": _build_arm(
            arm_id="strengthened_duplicate_family", source_rows=source_rows, transformed_rows=_copy_rows(source_rows), transformations=[],
            stage_c_selection=_selection_config(stage_c_selection, threshold=0.90), token_counter=token_counter,
        ),
        "cumulative_aggressive_candidate": _build_arm(
            arm_id="cumulative_aggressive_candidate", source_rows=source_rows, transformed_rows=cumulative_rows,
            transformations=[*cumulative_license_transformations, *cumulative_repeated_transformations],
            stage_c_selection=_selection_config(stage_c_selection, threshold=0.90), token_counter=token_counter,
        ),
    }
    threshold_sweep = {
        f"{threshold:.2f}": {
            key: value
            for key, value in _build_arm(
                arm_id=f"strengthened_duplicate_threshold_{threshold:.2f}",
                source_rows=source_rows,
                transformed_rows=_copy_rows(source_rows),
                transformations=[],
                stage_c_selection=_selection_config(stage_c_selection, threshold=threshold),
                token_counter=token_counter,
            ).items()
            if key not in {"curated_chunks", "not_selected_chunks", "transformations"}
        }
        for threshold in (0.90, 0.92, 0.95)
    }
    return {
        "schema_version": "aggressive-structural-candidate-run-v1",
        "status": "development_candidate_complete_not_runtime_active",
        "runtime_active": False,
        "candidate_profile": "aggressive_structural_candidate_v1",
        "token_count_kind": "frozen_tokenizer",
        "arms": arms,
        "near_duplicate_threshold_sweep": threshold_sweep,
        "claim_boundary": "Development-only evidence. No arm may change the active profile or consume external benchmark outcomes.",
    }


def _token_counter(tokenizer_path: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    @lru_cache(maxsize=None)
    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def materialize_candidate_run(config_path: Path) -> JsonMap:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("status") != "frozen_candidate_development":
        raise RuntimeError("Candidate config must be frozen_candidate_development")
    input_path = Path(str(config["input"]["stage_b_pass_path"]))
    output_dir = Path(str(config["output_dir"]))
    tokenizer_path = str(config["tokenizer"]["pretrained_model_name_or_path"])
    result = run_candidate_arms(
        _read_jsonl(input_path),
        stage_c_selection=dict(config["stage_c_selection"]),
        minimum_residual_chars=int(config["stage_b_minimum_residual_chars"]),
        token_counter=_token_counter(tokenizer_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm_id, arm in result["arms"].items():
        arm_dir = output_dir / arm_id
        _write_jsonl(arm_dir / "stage_c_curated_chunks.jsonl", arm["curated_chunks"])
        _write_jsonl(arm_dir / "stage_c_not_selected_chunks.jsonl", arm["not_selected_chunks"])
        _write_jsonl(arm_dir / "stage_c_transformations.jsonl", arm["transformations"])
        report = {key: value for key, value in arm.items() if key not in {"curated_chunks", "not_selected_chunks", "transformations"}}
        report["outputs"] = {
            "curated": str(arm_dir / "stage_c_curated_chunks.jsonl"),
            "not_selected": str(arm_dir / "stage_c_not_selected_chunks.jsonl"),
            "transformations": str(arm_dir / "stage_c_transformations.jsonl"),
        }
        save_json(arm_dir / "candidate_arm_report.json", report)
    result["input_sha256"] = sha256_file(input_path)
    result["config_sha256"] = sha256_file(config_path)
    report_arms = {
        arm_id: {key: value for key, value in arm.items() if key not in {"curated_chunks", "not_selected_chunks", "transformations"}}
        for arm_id, arm in result["arms"].items()
    }
    save_json(
        output_dir / "candidate_development_report.json",
        {key: value for key, value in result.items() if key != "arms"} | {"arms": report_arms},
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated aggressive structural candidate arms from a frozen Stage-B snapshot.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_candidate_run(args.config)
    print(json.dumps({"status": result["status"], "arms": list(result["arms"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
