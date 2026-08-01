#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = Dict[str, Any]

CODE_STAGE0_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined"
CODE_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_balanced"
CODE_STAGE_B_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2"
CODE_OUTPUT_DIR = OUTPUT_DIR / "code_domain_natural_budget_qwen3_4b"
CODE_PLAN_PATH = Path("configs") / "code_domain_natural_budget_protocol_qwen3_4b_v1.json"
CODE_BASE_PLAN = Path("configs") / "code_domain_v2_confirmatory_protocol_qwen3_4b.json"

MATH_MATERIALIZATION_DIR = OUTPUT_DIR / "math_domain_stage_materialization"
MATH_OUTPUT_DIR = OUTPUT_DIR / "math_domain_natural_budget_qwen3_4b"
MATH_PLAN_PATH = Path("configs") / "math_domain_natural_budget_protocol_qwen3_4b_v1.json"
MATH_BASE_PLAN = Path("configs") / "math_domain_stage_c_protocol_qwen3_4b_v1.json"


def _jsonl(path: Path) -> List[JsonMap]:
    rows: List[JsonMap] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _token(row: JsonMap) -> int:
    value = row.get("token_proxy_count", row.get("token_proxy", 0))
    if isinstance(value, bool):
        return 0
    return int(value) if isinstance(value, (int, float, str)) and str(value).strip() else 0


def _summary(path: Path) -> JsonMap:
    rows = _jsonl(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(rows),
        "token_proxy_count": sum(_token(row) for row in rows),
    }


def _load_code_freezer() -> Dict[str, Any]:
    source = Path(__file__).resolve().parent / "157_freeze_code_domain_v2_stage_b_arms.py"
    spec = importlib.util.spec_from_file_location("code_domain_v2_stage_b_arms", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def _retag(rows: Iterable[JsonMap], arm: str, source_pool: str) -> List[JsonMap]:
    out = []
    for row in rows:
        out.append(
            {
                **row,
                "arm": arm,
                "source_pool": source_pool,
                "natural_budget_role": "uncapped_full_arm",
            }
        )
    return out


def materialize_code() -> JsonMap:
    CODE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    freezer = _load_code_freezer()
    raw = freezer["_raw_chunks"](CODE_STAGE0_DIR)
    arm_record = freezer["_arm_record"]
    raw_full = [
        arm_record(row, "raw_full_natural", "raw_stage0_chunkable_before_stage_a")
        for row in raw["chunks"]
    ]
    stage_a_full = [
        arm_record(row, "stageA_full_natural", "stageA_pass_full")
        for row in _jsonl(CODE_STAGE_A_DIR / "train" / "stage_a_pass.jsonl")
    ]
    curated = _retag(_jsonl(CODE_STAGE_B_DIR / "curated_v2_equal_budget.jsonl"), "curated_v2_natural", "stage_b_v2_selected_full")
    _write_jsonl(CODE_OUTPUT_DIR / "raw_full_natural.jsonl", raw_full)
    _write_jsonl(CODE_OUTPUT_DIR / "stageA_full_natural.jsonl", stage_a_full)
    _write_jsonl(CODE_OUTPUT_DIR / "curated_v2_natural.jsonl", curated)
    report = {
        "schema_version": "code-domain-natural-budget-arms-v1",
        "status": "code_natural_budget_arms_materialized",
        "arms": {
            "raw_full_natural": _summary(CODE_OUTPUT_DIR / "raw_full_natural.jsonl"),
            "stageA_full_natural": _summary(CODE_OUTPUT_DIR / "stageA_full_natural.jsonl"),
            "curated_v2_natural": _summary(CODE_OUTPUT_DIR / "curated_v2_natural.jsonl"),
        },
        "raw_train_unchunkable_records": len(raw["unchunkable"]),
        "claim_boundary": "Natural-budget arm materialization only; no Stage-C outcome or release claim.",
    }
    save_json(CODE_OUTPUT_DIR / "natural_budget_arms_report.json", report)
    save_json(OUTPUT_DIR / "validation" / "code_domain_natural_budget_arms_report.json", report)
    return report


def materialize_math() -> JsonMap:
    MATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MATH_MATERIALIZATION_DIR / "stage0_retained.jsonl", MATH_OUTPUT_DIR / "raw_full_natural.jsonl")
    shutil.copyfile(MATH_MATERIALIZATION_DIR / "stage_a_pass.jsonl", MATH_OUTPUT_DIR / "stageA_full_natural.jsonl")
    curated = _retag(_jsonl(MATH_MATERIALIZATION_DIR / "stage_b_selected.jsonl"), "curated_math_natural", "stage_b_selected_full")
    _write_jsonl(MATH_OUTPUT_DIR / "curated_math_natural.jsonl", curated)
    report = {
        "schema_version": "math-domain-natural-budget-arms-v1",
        "status": "math_natural_budget_arms_materialized",
        "arms": {
            "raw_full_natural": _summary(MATH_OUTPUT_DIR / "raw_full_natural.jsonl"),
            "stageA_full_natural": _summary(MATH_OUTPUT_DIR / "stageA_full_natural.jsonl"),
            "curated_math_natural": _summary(MATH_OUTPUT_DIR / "curated_math_natural.jsonl"),
        },
        "claim_boundary": "Natural-budget arm materialization only; no Stage-C outcome or release claim.",
    }
    save_json(MATH_OUTPUT_DIR / "natural_budget_arms_report.json", report)
    save_json(OUTPUT_DIR / "validation" / "math_domain_natural_budget_arms_report.json", report)
    return report


def _arm_payload(path: Path) -> JsonMap:
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_plan(
    *,
    base_plan_path: Path,
    output_plan_path: Path,
    output_dir: Path,
    arms: List[str],
    primary_treatment: str,
    primary_baseline: str,
    heldout_path: Path | None = None,
) -> JsonMap:
    plan = load_json(base_plan_path)
    recipe = plan["confirmatory_training_recipe"]
    recipe["optimizer_steps"] = 1
    recipe["optimizer_steps_by_arm"] = {arm: 1 for arm in arms}
    recipe["natural_budget_step_rule"] = (
        "After token blocks are prepared, optimizer_steps_by_arm is set to "
        "ceil(num_blocks / gradient_accumulation_steps), preserving one natural pass over each arm."
    )
    recipe["same_step_count_for_every_arm"] = False
    recipe["training_token_budget_cap"] = None
    plan["schema_version"] = str(plan.get("schema_version", "protocol")) + "-natural-budget"
    plan["status"] = "natural_budget_protocol_frozen_before_training_outcomes"
    plan["training_arms"] = ["base_no_update", *arms]
    plan["primary_comparison"] = {
        "treatment": primary_treatment,
        "primary_baseline": primary_baseline,
        "supporting_baselines": ["base_no_update"],
    }
    plan["arm_token_counts"] = {
        arm: _arm_payload(output_dir / f"{arm}.jsonl")
        for arm in arms
    }
    plan["training_payloads"] = {
        arm: {
            "jsonl_path": str(output_dir / f"{arm}.jsonl"),
            "jsonl_sha256": sha256_file(output_dir / f"{arm}.jsonl"),
        }
        for arm in arms
    }
    if heldout_path is not None:
        plan["heldout_nll"] = {
            "frozen_heldout": {"path": str(heldout_path), "sha256": sha256_file(heldout_path)},
            "metric": "mean_nll",
            "direction": "lower_is_better",
        }
    plan["natural_budget_claim"] = (
        "Compare raw-full training against framework-curated natural output without equalizing token budgets."
    )
    plan["claim_boundary"] = "Natural-budget protocol freeze only; no training, benchmark, release, or paper-success claim."
    save_json(output_plan_path, plan)
    return plan


def freeze_plans() -> JsonMap:
    code_plan = _write_plan(
        base_plan_path=CODE_BASE_PLAN,
        output_plan_path=CODE_PLAN_PATH,
        output_dir=CODE_OUTPUT_DIR,
        arms=["raw_full_natural", "curated_v2_natural"],
        primary_treatment="curated_v2_natural",
        primary_baseline="raw_full_natural",
    )
    math_plan = _write_plan(
        base_plan_path=MATH_BASE_PLAN,
        output_plan_path=MATH_PLAN_PATH,
        output_dir=MATH_OUTPUT_DIR,
        arms=["raw_full_natural", "curated_math_natural"],
        primary_treatment="curated_math_natural",
        primary_baseline="raw_full_natural",
        heldout_path=OUTPUT_DIR / "math_domain_stage_c_qwen3_4b" / "heldouts" / "math_nll_heldout.jsonl",
    )
    report = {
        "schema_version": "natural-budget-protocol-freeze-v1",
        "status": "natural_budget_protocols_frozen",
        "plans": {
            str(CODE_PLAN_PATH): {"sha256": sha256_file(CODE_PLAN_PATH), "arms": code_plan["training_arms"]},
            str(MATH_PLAN_PATH): {"sha256": sha256_file(MATH_PLAN_PATH), "arms": math_plan["training_arms"]},
        },
    }
    save_json(OUTPUT_DIR / "validation" / "natural_budget_protocols_report.json", report)
    return report


def update_steps(plan_path: Path, blocks_manifest: Path) -> JsonMap:
    plan = load_json(plan_path)
    manifest = load_json(blocks_manifest)
    blocks = manifest["blocks"]
    recipe = plan["confirmatory_training_recipe"]
    grad_accum = int(recipe["gradient_accumulation_steps"])
    arms = [str(arm) for arm in plan["training_arms"] if str(arm) != "base_no_update"]
    by_arm = {}
    packed_tokens = {}
    for arm in arms:
        if arm not in blocks:
            raise KeyError(f"Missing block manifest entry for {arm}")
        block_count = int(blocks[arm]["blocks"])
        by_arm[arm] = max(1, math.ceil(block_count / grad_accum))
        packed_tokens[arm] = int(blocks[arm]["tokens_in_blocks"])
    recipe["optimizer_steps_by_arm"] = by_arm
    recipe["optimizer_steps"] = max(by_arm.values())
    recipe["natural_budget_packed_tokens_by_arm"] = packed_tokens
    recipe["natural_budget_blocks_manifest"] = {
        "path": str(blocks_manifest),
        "sha256": sha256_file(blocks_manifest),
    }
    save_json(plan_path, plan)
    report = {
        "schema_version": "natural-budget-steps-update-v1",
        "status": "natural_budget_optimizer_steps_updated",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "optimizer_steps_by_arm": by_arm,
        "packed_tokens_by_arm": packed_tokens,
    }
    save_json(blocks_manifest.parent / "natural_budget_steps_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize raw-full versus curated-natural Stage-C arms.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    sub.add_parser("freeze-plans")
    update = sub.add_parser("update-steps")
    update.add_argument("--plan", type=Path, required=True)
    update.add_argument("--blocks-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        print(json.dumps({"code": materialize_code(), "math": materialize_math()}, indent=2))
    elif args.command == "freeze-plans":
        print(json.dumps(freeze_plans(), indent=2))
    elif args.command == "update-steps":
        print(json.dumps(update_steps(args.plan, args.blocks_manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
