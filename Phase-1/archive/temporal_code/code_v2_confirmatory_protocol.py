#!/usr/bin/env python3
"""Freeze the code-domain v2 Qwen3-4B confirmatory protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import token_proxy_count


DEFAULT_DEVELOPMENT_PLAN = Path("configs") / "code_domain_v2_development_plan_qwen3_4b.json"
DEFAULT_DEVELOPMENT_DECISION = OUTPUT_DIR / "validation" / "code_domain_v2_development_decision_report.json"
DEFAULT_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_balanced"
DEFAULT_STAGE_B_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2"
DEFAULT_RETENTION_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
DEFAULT_EVALPLUS_SPLIT = Path("configs") / "temporal_code_evalplus_guardrail_split_v1.json"
DEFAULT_OUTPUT_CONFIG = Path("configs") / "code_domain_v2_confirmatory_protocol_qwen3_4b.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_v2_confirmatory_qwen3_4b"
DEFAULT_REPORT = OUTPUT_DIR / "validation" / "code_domain_v2_confirmatory_protocol_qwen3_4b_report.json"
PROJECT_DIR = Path(__file__).resolve().parents[2]

CONFIRMATORY_HELDOUT_SEED = 20260623
MARGIN_ROUNDING_UNIT = 0.0005


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_order(records: Iterable[Dict[str, Any]], seed: int, label: str) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}:{label}:{row['chunk_uid']}".encode("utf-8")
        ).hexdigest(),
    )


def _freeze_confirmatory_heldout(
    source_file: Path,
    output_dir: Path,
    *,
    token_proxy_budget: int,
) -> Dict[str, Any]:
    allowed = {"code", "test"}
    rows = [
        {**row, "token_proxy_count": token_proxy_count(str(row.get("text") or ""))}
        for row in _jsonl(source_file)
        if row.get("split") == "confirmatory"
        and row.get("stage_a_pass") is True
        and row.get("content_type") in allowed
    ]
    selected: List[Dict[str, Any]] = []
    tokens = 0
    for row in _stable_order(rows, CONFIRMATORY_HELDOUT_SEED, "confirmatory_code_nll_heldout_v2"):
        count = int(row["token_proxy_count"])
        if tokens + count > token_proxy_budget and selected:
            continue
        selected.append(row)
        tokens += count
        if tokens >= token_proxy_budget:
            break
    if not selected:
        raise RuntimeError("V2 confirmatory heldout selection produced no records.")
    output = output_dir / "heldouts" / "confirmatory_code_nll_heldout.jsonl"
    _write_jsonl(output, selected)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "source_path": str(source_file),
        "source_sha256": sha256_file(source_file),
        "source_split": "confirmatory",
        "selection_rule": (
            "Sort by sha256(seed + ':' + slice_name + ':' + chunk_uid), "
            "then take chunks until the token-proxy budget is reached."
        ),
        "seed": CONFIRMATORY_HELDOUT_SEED,
        "candidate_records": len(rows),
        "selected_records": len(selected),
        "selected_token_proxy": tokens,
        "token_proxy_budget": token_proxy_budget,
        "allowed_content_types": sorted(allowed),
        "content_type_counts": {
            value: sum(1 for row in selected if row.get("content_type") == value)
            for value in sorted({row.get("content_type") for row in selected})
        },
        "repository_count": len({row.get("repository_identity") for row in selected}),
    }


def _ceil_to_unit(value: float, unit: float) -> float:
    return round(math.ceil(value / unit) * unit, 10)


def _calibrate_margin(development_decision: Dict[str, Any]) -> Dict[str, Any]:
    primary = development_decision["summary"]["nll_gate"]["paired_deltas"][
        "stageA_random_minus_curated"
    ]
    deltas = [float(value) for value in primary["per_seed_delta"].values()]
    mean_delta = sum(deltas) / len(deltas)
    sample_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    absolute_floor = 0.0025
    development_fraction_floor = 0.40 * mean_delta
    variance_floor = 2.0 * sample_std
    unrounded = max(absolute_floor, development_fraction_floor, variance_floor)
    frozen_margin = _ceil_to_unit(unrounded, MARGIN_ROUNDING_UNIT)
    return {
        "status": "frozen_before_v2_confirmatory_outcomes",
        "source": "development_only_primary_paired_seed_deltas",
        "development_primary_mean_delta": mean_delta,
        "development_primary_sample_std_delta": sample_std,
        "development_primary_min_delta": min(deltas),
        "development_primary_max_delta": max(deltas),
        "calibration_formula": (
            "ceil_to_0.0005(max(0.0025 absolute NLL floor, "
            "0.40 * development primary mean delta, "
            "2.0 * development paired-seed sample std))"
        ),
        "absolute_floor": absolute_floor,
        "development_fraction_floor": development_fraction_floor,
        "variance_floor": variance_floor,
        "unrounded_margin": unrounded,
        "frozen_absolute_nll_margin": frozen_margin,
        "confirmatory_outcomes_used": False,
    }


def _training_block_manifest(output_dir: Path, arms: Iterable[str]) -> Dict[str, Any]:
    blocks = {}
    for arm in arms:
        if arm == "base_no_update":
            continue
        path = output_dir / "token_blocks" / f"{arm}.pt"
        blocks[arm] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    manifest = output_dir / "token_blocks" / "block_manifest.json"
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "blocks": blocks,
    }


def freeze(
    development_plan_path: Path,
    development_decision_path: Path,
    stage_a_dir: Path,
    stage_b_dir: Path,
    retention_plan_path: Path,
    evalplus_split_path: Path,
    output_config_path: Path,
    output_dir: Path,
    report_path: Path,
) -> Dict[str, Any]:
    development_plan = load_json(development_plan_path)
    development_decision = load_json(development_decision_path)
    retention_plan = load_json(retention_plan_path)
    evalplus_split = load_json(evalplus_split_path)
    blockers: List[str] = []

    if development_decision.get("status") != "development_decision_promote_to_confirmatory":
        blockers.append(f"development_not_promoted:{development_decision.get('status')}")
    if development_decision.get("confirmatory_outcomes_read") is not False:
        blockers.append("development_decision_read_confirmatory_outcomes")
    if development_plan.get("confirmatory_outcomes_read") is not False:
        blockers.append("development_plan_read_confirmatory_outcomes")

    stage_b_report_path = stage_b_dir / "stage_b_v2_arms_report.json"
    stage_b_report = load_json(stage_b_report_path)
    if stage_b_report.get("status") != "stage_b_v2_arms_frozen_before_stage_c":
        blockers.append(f"stage_b_v2_not_frozen:{stage_b_report.get('status')}")
    if stage_b_report.get("disjointness", {}).get("curated_v2_stageA_random_disjoint") is not True:
        blockers.append("curated_v2_stageA_random_not_disjoint")

    retention_contract = retention_plan["contract"]
    confirmatory_seeds = [
        int(seed) for seed in retention_contract["seed_contract"]["confirmatory_training_seeds"]
    ]
    evalplus_confirmatory_seeds = [int(seed) for seed in evalplus_split["confirmatory_training_seeds"]]
    development_seeds = [
        int(seed) for seed in development_plan["training_recipe"]["development_training_seeds"]
    ]
    if confirmatory_seeds != evalplus_confirmatory_seeds:
        blockers.append("confirmatory_seed_contract_mismatch:retention_vs_evalplus")
    if set(confirmatory_seeds).intersection(development_seeds):
        blockers.append("confirmatory_seeds_overlap_development_seeds")
    if retention_contract["decision_rule"].get("all_guardrails_mandatory") is not True:
        blockers.append("retention_guardrails_not_mandatory")
    if retention_contract["decision_rule"].get("confirmatory_may_not_select_recipe") is not True:
        blockers.append("confirmatory_may_select_recipe_not_forbidden")

    source_file = stage_a_dir / "confirmatory" / "stage_a_pass.jsonl"
    if not source_file.exists():
        blockers.append(f"missing_confirmatory_stage_a_pass:{source_file}")
        heldout = None
    else:
        heldout = _freeze_confirmatory_heldout(
            source_file,
            output_dir,
            token_proxy_budget=int(development_plan["heldout_nll"]["token_proxy_budget"]),
        )

    margin = _calibrate_margin(development_decision)
    recipe = development_plan["training_recipe"]
    confirmatory_recipe = {
        key: value
        for key, value in recipe.items()
        if key not in {"development_training_seeds", "same_seed_set_for_every_arm"}
    }
    confirmatory_recipe["confirmatory_training_seeds"] = confirmatory_seeds
    confirmatory_recipe["same_seed_set_for_every_arm"] = True

    training_arms = [str(arm) for arm in development_plan["training_arms"]]
    trained_arms = [arm for arm in training_arms if arm != "base_no_update"]
    training_payloads = {
        arm: {
            "jsonl_path": str(stage_b_dir / f"{arm}.jsonl"),
            "jsonl_sha256": sha256_file(stage_b_dir / f"{arm}.jsonl"),
        }
        for arm in trained_arms
    }
    training_blocks = _training_block_manifest(
        _resolve("outputs/code_domain_v2_development_qwen3_4b"),
        training_arms,
    )

    protocol = {
        "schema_version": "code-domain-v2-confirmatory-protocol-qwen3-4b",
        "status": "frozen_before_v2_confirmatory_training_outcomes"
        if not blockers
        else "v2_confirmatory_protocol_blocked",
        "purpose": (
            "Freeze the untouched v2 confirmatory contract after the v2 development "
            "decision promoted the curated Stage-B recipe."
        ),
        "target_model": development_plan["target_model"],
        "source_development_decision": {
            "path": str(development_decision_path),
            "status": development_decision.get("status"),
            "sha256": sha256_file(development_decision_path),
        },
        "source_stage_b_arms": {
            "report_path": str(stage_b_report_path),
            "report_sha256": sha256_file(stage_b_report_path),
            "status": stage_b_report.get("status"),
            "curated_v2_stageA_random_disjoint": stage_b_report["disjointness"][
                "curated_v2_stageA_random_disjoint"
            ],
        },
        "training_arms": training_arms,
        "primary_comparison": development_plan["primary_comparison"],
        "confirmatory_training_recipe": confirmatory_recipe,
        "training_payloads": training_payloads,
        "training_blocks": training_blocks,
        "heldout_nll": {
            "confirmatory_slice_name": "confirmatory_code_nll_heldout",
            "source_split": "confirmatory",
            "source_file": str(source_file),
            "frozen_heldout": heldout,
            "metric": "mean_nll",
            "direction": "lower_is_better",
        },
        "primary_success_rule": {
            "all_conditions_required": True,
            "primary_metric": "confirmatory_code_nll_heldout mean NLL",
            "direction": "lower_is_better",
            "primary_treatment": "curated_v2_equal_budget",
            "primary_baseline": "stageA_random_equal_budget",
            "required_absolute_nll_reduction": margin["frozen_absolute_nll_margin"],
            "paired_seed_requirement": "all curated_v2 seed-level NLLs must be lower than paired Stage-A-random NLLs",
            "supporting_raw_random_rule": "curated_v2 mean NLL must be lower than raw-random mean NLL",
            "guardrail_rule": "all frozen confirmatory Stage-C guardrails must pass",
            "failure_action": "report negative finding or abstain; do not tune on confirmatory evidence",
        },
        "margin_calibration": margin,
        "stage_c_guardrails": {
            "evalplus_confirmatory": {
                "source_split_plan": str(evalplus_split_path),
                "required_split": "confirmatory",
                "non_inferiority": evalplus_split["non_inferiority"],
                "role": evalplus_split["role"],
            },
            "general_text_nll_retention": retention_contract["general_text_guardrail"],
            "general_task_retention": retention_contract["general_task_guardrail"],
            "decision_rule": retention_contract["decision_rule"],
        },
        "forbidden_uses": sorted(
            set(development_plan["forbidden_uses"])
            | set(retention_contract["forbidden_uses"])
            | {
                "using Utility, benchmark outcomes, retention outcomes, development outcomes, confirmatory outcomes, or human/LLM review labels in Stage B",
                "using confirmatory outcomes to select a new recipe",
                "changing the primary metric or margin after confirmatory outcomes",
                "using different Stage-A baselines for sensitivity arms",
            }
        ),
        "confirmatory_outcomes_read": False,
        "utility_scope": development_plan["utility_scope"],
        "claim_boundary": (
            "V2 confirmatory protocol freeze only. It authorizes untouched "
            "confirmatory training/evaluation but makes no confirmatory, release, "
            "or paper-success claim."
        ),
    }
    save_json(output_config_path, protocol)

    source_sha256 = {
        str(development_plan_path): sha256_file(development_plan_path),
        str(development_decision_path): sha256_file(development_decision_path),
        str(stage_b_report_path): sha256_file(stage_b_report_path),
        str(retention_plan_path): sha256_file(retention_plan_path),
        str(evalplus_split_path): sha256_file(evalplus_split_path),
        str(output_config_path): sha256_file(output_config_path),
    }
    if heldout is not None:
        source_sha256[str(Path(heldout["path"]))] = sha256_file(Path(heldout["path"]))

    report = {
        "schema_version": "code-domain-v2-confirmatory-protocol-freeze-report-v1",
        "status": "v2_confirmatory_protocol_frozen" if not blockers else "v2_confirmatory_protocol_blocked",
        "source_sha256": source_sha256,
        "summary": {
            "protocol_path": str(output_config_path),
            "target_model": development_plan["target_model"],
            "training_arms": training_arms,
            "confirmatory_training_seeds": confirmatory_seeds,
            "development_training_seeds": development_seeds,
            "optimizer_steps": confirmatory_recipe["optimizer_steps"],
            "common_packed_token_budget": confirmatory_recipe["common_packed_token_budget"],
            "training_token_budget_cap": confirmatory_recipe["training_token_budget_cap"],
            "heldout": heldout,
            "primary_success_rule": protocol["primary_success_rule"],
            "margin_calibration": margin,
            "stage_c_guardrails": protocol["stage_c_guardrails"],
            "blockers": blockers,
        },
        "confirmatory_outcomes_read": False,
        "utility_scope": development_plan["utility_scope"],
        "claim_boundary": protocol["claim_boundary"],
    }
    save_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain v2 Qwen3-4B confirmatory protocol.")
    parser.add_argument("--development-plan", type=Path, default=DEFAULT_DEVELOPMENT_PLAN)
    parser.add_argument("--development-decision", type=Path, default=DEFAULT_DEVELOPMENT_DECISION)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--stage-b-dir", type=Path, default=DEFAULT_STAGE_B_DIR)
    parser.add_argument("--retention-plan", type=Path, default=DEFAULT_RETENTION_PLAN)
    parser.add_argument("--evalplus-split", type=Path, default=DEFAULT_EVALPLUS_SPLIT)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = freeze(
        args.development_plan,
        args.development_decision,
        args.stage_a_dir,
        args.stage_b_dir,
        args.retention_plan,
        args.evalplus_split,
        args.output_config,
        args.output_dir,
        args.report,
    )
    print({"status": report["status"], "blockers": report["summary"]["blockers"]})
    return 0 if report["status"] == "v2_confirmatory_protocol_frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
