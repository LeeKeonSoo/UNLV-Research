#!/usr/bin/env python3
"""Freeze the code-domain Qwen3-4B confirmatory protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import token_proxy_count


DEFAULT_DEVELOPMENT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_DEVELOPMENT_DECISION = OUTPUT_DIR / "validation" / "code_domain_development_decision_report.json"
DEFAULT_RETENTION_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
DEFAULT_EVALPLUS_SPLIT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
DEFAULT_ARMS_REPORT = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "equal_token_arms" / "equal_token_training_arms_report.json"
DEFAULT_OUTPUT_CONFIG = Path("configs") / "code_domain_confirmatory_protocol_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_confirmatory_qwen3_4b_v1"
DEFAULT_REPORT = OUTPUT_DIR / "validation" / "code_domain_confirmatory_protocol_qwen3_4b_report.json"
PROJECT_DIR = Path(__file__).resolve().parents[2]

CONFIRMATORY_HELDOUT_SEED = 20260621
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "known_high_quality_equal_budget",
)


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


def _stable_order(records: Iterable[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{row['chunk_uid']}".encode("utf-8")).hexdigest(),
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
    for row in _stable_order(rows, CONFIRMATORY_HELDOUT_SEED):
        count = int(row["token_proxy_count"])
        if tokens + count > token_proxy_budget and selected:
            continue
        selected.append(row)
        tokens += count
        if tokens >= token_proxy_budget:
            break
    if not selected:
        raise RuntimeError("Confirmatory heldout selection produced no records.")
    output = output_dir / "heldouts" / "confirmatory_code_nll_heldout.jsonl"
    _write_jsonl(output, selected)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "source_path": str(source_file),
        "source_sha256": sha256_file(source_file),
        "source_split": "confirmatory",
        "selection_rule": "Sort by sha256(seed + ':' + chunk_uid), then take chunks until the token-proxy budget is reached.",
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


def freeze(
    development_plan_path: Path,
    development_decision_path: Path,
    retention_plan_path: Path,
    evalplus_split_path: Path,
    arms_report_path: Path,
    output_config_path: Path,
    output_dir: Path,
    report_path: Path,
) -> Dict[str, Any]:
    development_plan = load_json(development_plan_path)
    development_decision = load_json(development_decision_path)
    retention_plan = load_json(retention_plan_path)
    evalplus_split = load_json(evalplus_split_path)
    arms_report = load_json(arms_report_path)
    blockers: List[str] = []

    if development_decision.get("status") != "development_decision_promote_to_confirmatory":
        blockers.append(f"development_not_promoted:{development_decision.get('status')}")
    if arms_report.get("status") != "training_arms_frozen":
        blockers.append(f"training_arms_not_frozen:{arms_report.get('status')}")
    retention_contract = retention_plan["contract"]
    evalplus_contract = evalplus_split["contract"]
    confirmatory_seeds = [int(seed) for seed in retention_contract["seed_contract"]["confirmatory_training_seeds"]]
    if confirmatory_seeds != [int(seed) for seed in evalplus_contract["confirmatory_training_seeds"]]:
        blockers.append("confirmatory_seed_contract_mismatch:retention_vs_evalplus")
    if set(confirmatory_seeds).intersection(set(development_plan["training_recipe"]["development_training_seeds"])):
        blockers.append("confirmatory_seeds_overlap_development_seeds")
    if retention_contract["decision_rule"].get("all_guardrails_mandatory") is not True:
        blockers.append("retention_guardrails_not_mandatory")

    source_file = _resolve(
        OUTPUT_DIR / "temporal_code_collection" / "stage_a_path_stratified_tranche" / "confirmatory" / "stage_a_pass.jsonl"
    )
    if not source_file.exists():
        blockers.append(f"missing_confirmatory_stage_a_pass:{source_file}")
        heldout = None
    else:
        heldout = _freeze_confirmatory_heldout(
            source_file,
            output_dir,
            token_proxy_budget=int(development_plan["heldout_nll"]["token_proxy_budget"]),
        )

    recipe = development_plan["training_recipe"]
    immutable_training_recipe = {
        key: value
        for key, value in recipe.items()
        if key not in {"development_training_seeds", "same_seed_set_for_every_arm"}
    }
    immutable_training_recipe["confirmatory_training_seeds"] = confirmatory_seeds
    immutable_training_recipe["same_seed_set_for_every_arm"] = True

    confirmatory_protocol = {
        "schema_version": "code-domain-confirmatory-protocol-qwen3-4b-v1",
        "status": "frozen_before_confirmatory_training_outcomes" if not blockers else "confirmatory_protocol_blocked",
        "purpose": (
            "Bind the promoted development-stage Qwen3-4B code-domain curation recipe "
            "to an untouched confirmatory training/evaluation contract."
        ),
        "target_model": development_plan["target_model"],
        "source_development_decision": {
            "path": str(development_decision_path),
            "status": development_decision.get("status"),
            "sha256": sha256_file(development_decision_path),
        },
        "training_arms": ["base_no_update", *TRAINED_ARMS],
        "primary_comparison": development_plan["primary_comparison"],
        "confirmatory_training_recipe": immutable_training_recipe,
        "training_payloads": {
            arm: {
                "path": str(OUTPUT_DIR / "temporal_code_training_freeze_v1" / "equal_token_arms" / f"{arm}.jsonl"),
                "sha256": sha256_file(OUTPUT_DIR / "temporal_code_training_freeze_v1" / "equal_token_arms" / f"{arm}.jsonl"),
            }
            for arm in TRAINED_ARMS
        },
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
            "conditions": [
                "all frozen confirmatory training runs complete for every required trained arm and seed",
                "curated mean NLL is lower than Stage-A-random mean NLL by at least the predeclared absolute margin",
                "curated mean NLL is lower than raw-random mean NLL",
                "all Stage-C confirmatory retention guardrails pass",
                "no confirmatory model outcomes are used to change the selector, margin, seed set, token budget, or guardrail thresholds",
            ],
            "primary_metric": "confirmatory_code_nll_heldout mean NLL",
            "curated_vs_stageA_random_required_absolute_nll_reduction": development_plan[
                "practical_effect_margin"
            ]["curated_vs_stageA_random_required_absolute_nll_reduction"],
            "curated_vs_raw_random_required_direction": "curated mean NLL must be lower than raw-random mean NLL",
            "seed_aggregation": "paired by the same five frozen confirmatory seeds; report every seed and arithmetic mean",
            "failure_action": "report negative finding or abstain; do not tune on confirmatory evidence",
        },
        "stage_c_guardrails": {
            "evalplus_confirmatory": {
                "source_split_plan": str(evalplus_split_path),
                "split_counts": evalplus_split["summary"]["suite_split_counts"],
                "required_split": "confirmatory",
                "non_inferiority": evalplus_contract["non_inferiority"],
                "role": evalplus_contract["role"],
            },
            "general_text_nll_retention": retention_contract["general_text_guardrail"],
            "general_task_retention": retention_contract["general_task_guardrail"],
            "decision_rule": retention_contract["decision_rule"],
        },
        "forbidden_uses": sorted(
            set(development_plan["forbidden_uses"])
            | set(retention_contract["forbidden_uses"])
            | {
                "using confirmatory outcomes to select a new recipe",
                "using Utility or benchmark outcomes in Stage B",
                "changing common-baseline sensitivity design after confirmatory outcomes",
            }
        ),
        "confirmatory_outcomes_read": False,
        "utility_scope": development_plan["utility_scope"],
        "claim_boundary": (
            "Confirmatory protocol freeze only. It authorizes untouched confirmatory "
            "training/evaluation but makes no confirmatory, release, or paper claim."
        ),
    }
    save_json(output_config_path, confirmatory_protocol)

    source_sha256 = {
        str(development_plan_path): sha256_file(development_plan_path),
        str(development_decision_path): sha256_file(development_decision_path),
        str(retention_plan_path): sha256_file(retention_plan_path),
        str(evalplus_split_path): sha256_file(evalplus_split_path),
        str(arms_report_path): sha256_file(arms_report_path),
        str(output_config_path): sha256_file(output_config_path),
    }
    if heldout is not None:
        source_sha256[str(Path(heldout["path"]))] = sha256_file(Path(heldout["path"]))

    report = {
        "schema_version": "code-domain-confirmatory-protocol-freeze-report-v1",
        "status": "confirmatory_protocol_frozen" if not blockers else "confirmatory_protocol_blocked",
        "source_sha256": source_sha256,
        "summary": {
            "protocol_path": str(output_config_path),
            "target_model": development_plan["target_model"],
            "training_arms": confirmatory_protocol["training_arms"],
            "confirmatory_training_seeds": confirmatory_seeds,
            "optimizer_steps": immutable_training_recipe["optimizer_steps"],
            "common_packed_token_budget": immutable_training_recipe["common_packed_token_budget"],
            "training_token_budget_cap": immutable_training_recipe["training_token_budget_cap"],
            "heldout": heldout,
            "primary_success_rule": confirmatory_protocol["primary_success_rule"],
            "stage_c_guardrails": confirmatory_protocol["stage_c_guardrails"],
            "blockers": blockers,
        },
        "confirmatory_outcomes_read": False,
        "utility_scope": development_plan["utility_scope"],
        "claim_boundary": confirmatory_protocol["claim_boundary"],
    }
    save_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain Qwen3-4B confirmatory protocol.")
    parser.add_argument("--development-plan", type=Path, default=DEFAULT_DEVELOPMENT_PLAN)
    parser.add_argument("--development-decision", type=Path, default=DEFAULT_DEVELOPMENT_DECISION)
    parser.add_argument("--retention-plan", type=Path, default=DEFAULT_RETENTION_PLAN)
    parser.add_argument("--evalplus-split", type=Path, default=DEFAULT_EVALPLUS_SPLIT)
    parser.add_argument("--arms-report", type=Path, default=DEFAULT_ARMS_REPORT)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = freeze(
        args.development_plan,
        args.development_decision,
        args.retention_plan,
        args.evalplus_split,
        args.arms_report,
        args.output_config,
        args.output_dir,
        args.report,
    )
    print({"status": report["status"], "blockers": report["summary"]["blockers"]})
    return 0 if report["status"] == "confirmatory_protocol_frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
