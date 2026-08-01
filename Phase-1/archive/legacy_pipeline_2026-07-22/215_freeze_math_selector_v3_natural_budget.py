#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
SOURCE_DIR = OUTPUT_DIR / "math_domain_stage_materialization_v3"
OUTPUT_DIR_V3 = OUTPUT_DIR / "math_domain_natural_budget_v3_qwen3_4b"
PLAN_PATH = Path("configs") / "math_domain_natural_budget_v3_protocol_qwen3_4b.json"
BASE_PLAN_PATH = Path("configs") / "math_domain_natural_budget_protocol_qwen3_4b_v1.json"
VALIDATION_REPORT = OUTPUT_DIR / "validation" / "math_domain_natural_budget_v3_freeze_report.json"


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


def _token(row: JsonMap) -> int:
    value = row.get("token_proxy_count", row.get("token_proxy", 0))
    if isinstance(value, bool):
        return 0
    return int(value) if isinstance(value, int | float | str) and str(value).strip() else 0


def _style_token_counts(path: Path) -> JsonMap:
    counts: JsonMap = {}
    for row in _jsonl(path):
        style = str(row.get("style_bucket", row.get("math_source_type", "unknown")))
        counts[style] = int(counts.get(style, 0)) + _token(row)
    return dict(sorted(counts.items()))


def _summary(path: Path) -> JsonMap:
    rows = _jsonl(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(rows),
        "token_proxy_count": sum(_token(row) for row in rows),
        "style_token_counts": _style_token_counts(path),
    }


def _copy_arm(source_name: str, target_name: str) -> JsonMap:
    source = SOURCE_DIR / f"{source_name}.jsonl"
    target = OUTPUT_DIR_V3 / f"{target_name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return _summary(target)


def _token_reduction(treatment: JsonMap, baseline: JsonMap) -> float:
    return 1.0 - (int(treatment["token_proxy_count"]) / int(baseline["token_proxy_count"]))


def _freeze_plan(arms: JsonMap, source_report: JsonMap) -> JsonMap:
    plan = load_json(BASE_PLAN_PATH)
    recipe = plan["confirmatory_training_recipe"]
    recipe["confirmatory_training_seeds"] = [101, 131, 163]
    recipe["optimizer_steps"] = 1
    recipe["optimizer_steps_by_arm"] = {
        "raw_full_natural": 1,
        "curated_math_v2_natural": 1,
        "curated_math_v3_natural": 1,
    }
    recipe["natural_budget_step_rule"] = (
        "After v3 token blocks are prepared, optimizer_steps_by_arm is set to "
        "ceil(num_blocks / gradient_accumulation_steps), preserving one natural pass per arm."
    )
    recipe["same_step_count_for_every_arm"] = False
    recipe["training_token_budget_cap"] = None
    for stale_key in ("natural_budget_packed_tokens_by_arm", "natural_budget_blocks_manifest"):
        recipe.pop(stale_key, None)
    plan["schema_version"] = "math-domain-stage-c-protocol-qwen3-4b-v3-natural-budget"
    plan["status"] = "math_selector_v3_natural_budget_protocol_frozen_before_training_outcomes"
    plan["source_materialization"] = {
        "path": str(SOURCE_DIR / "math_selector_v3_materialization_report.json"),
        "status": source_report["status"],
        "sha256": sha256_file(SOURCE_DIR / "math_selector_v3_materialization_report.json"),
    }
    plan["training_arms"] = [
        "base_no_update",
        "raw_full_natural",
        "curated_math_v2_natural",
        "curated_math_v3_natural",
    ]
    plan["primary_comparison"] = {
        "treatment": "curated_math_v3_natural",
        "primary_baseline": "raw_full_natural",
        "failure_reference": "curated_math_v2_natural",
        "supporting_baselines": ["base_no_update"],
    }
    plan["arm_token_counts"] = {
        arm: {"path": row["path"], "sha256": row["sha256"], "token_proxy_count": row["token_proxy_count"]}
        for arm, row in arms.items()
    }
    plan["training_payloads"] = {
        arm: {"jsonl_path": row["path"], "jsonl_sha256": row["sha256"]} for arm, row in arms.items()
    }
    plan["command_templates"] = [
        (
            "python 216_prepare_math_selector_v3_natural_budget_blocks.py"
        ),
        (
            "python 141_run_code_domain_development_qlora.py train-missing "
            "--plan configs/math_domain_natural_budget_v3_protocol_qwen3_4b.json "
            "--output-dir outputs/math_domain_natural_budget_v3_qwen3_4b "
            "--blocks-dir outputs/math_domain_natural_budget_v3_qwen3_4b/token_blocks "
            "--arms raw_full_natural,curated_math_v2_natural,curated_math_v3_natural "
            "--seeds 101,131,163 --allow-download"
        ),
        (
            "python 141_run_code_domain_development_qlora.py prepare-eval-blocks "
            "--plan configs/math_domain_natural_budget_v3_protocol_qwen3_4b.json "
            "--output-dir outputs/math_domain_natural_budget_v3_qwen3_4b --allow-download"
        ),
        (
            "python 141_run_code_domain_development_qlora.py eval-missing "
            "--plan configs/math_domain_natural_budget_v3_protocol_qwen3_4b.json "
            "--output-dir outputs/math_domain_natural_budget_v3_qwen3_4b "
            "--arms base_no_update,raw_full_natural,curated_math_v2_natural,curated_math_v3_natural "
            "--seeds 101,131,163 --allow-download"
        ),
    ]
    plan["natural_budget_claim"] = (
        "Compare raw-full training against framework-curated v3 without equalizing token budgets; "
        "keep v2 as the preserved failure reference."
    )
    plan["claim_boundary"] = "Math selector v3 natural-budget protocol freeze only; no training or success claim."
    plan["stage_c_outcomes_read"] = False
    save_json(PLAN_PATH, plan)
    return plan


def freeze() -> JsonMap:
    source_report = load_json(SOURCE_DIR / "math_selector_v3_materialization_report.json")
    arms = {
        "raw_full_natural": _copy_arm("raw_full_natural", "raw_full_natural"),
        "curated_math_v2_natural": _copy_arm("curated_math_v2_natural", "curated_math_v2_natural"),
        "curated_math_v3_natural": _copy_arm("curated_math_v3_natural", "curated_math_v3_natural"),
    }
    plan = _freeze_plan(arms, source_report)
    report = {
        "schema_version": "math-domain-natural-budget-v3-freeze-report",
        "status": "math_natural_budget_v3_protocol_frozen",
        "plan": {"path": str(PLAN_PATH), "sha256": sha256_file(PLAN_PATH)},
        "arms": arms,
        "v3_vs_raw_natural_budget": {
            "record_reduction_fraction": 1.0
            - (arms["curated_math_v3_natural"]["records"] / arms["raw_full_natural"]["records"]),
            "token_proxy_reduction_fraction": _token_reduction(
                arms["curated_math_v3_natural"], arms["raw_full_natural"]
            ),
        },
        "v3_vs_v2_repair_checks": {
            "v3_token_proxy_greater_than_v2": arms["curated_math_v3_natural"]["token_proxy_count"]
            > arms["curated_math_v2_natural"]["token_proxy_count"],
            "v3_proof_or_theorem_tokens_greater_than_v2": int(
                arms["curated_math_v3_natural"]["style_token_counts"].get("proof_or_theorem", 0)
            )
            > int(arms["curated_math_v2_natural"]["style_token_counts"].get("proof_or_theorem", 0)),
        },
        "training_arms": plan["training_arms"],
        "primary_comparison": plan["primary_comparison"],
        "stage_c_outcomes_read": False,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(OUTPUT_DIR_V3 / "natural_budget_v3_freeze_report.json", report)
    save_json(VALIDATION_REPORT, report)
    return report


def main() -> int:
    print(json.dumps(freeze(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
