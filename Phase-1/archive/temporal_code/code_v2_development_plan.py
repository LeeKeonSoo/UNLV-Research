#!/usr/bin/env python3
"""Freeze the code-domain v2 Qwen3-4B development plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import token_proxy_count


DEFAULT_DESIGN = Path("configs") / "code_domain_next_development_cycle_v2_design.json"
DEFAULT_STAGE_B_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2"
DEFAULT_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_balanced"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_v2_development_qwen3_4b"
DEFAULT_CONFIG_OUTPUT = Path("configs") / "code_domain_v2_development_plan_qwen3_4b.json"
DEFAULT_REPORT = OUTPUT_DIR / "validation" / "code_domain_v2_development_plan_qwen3_4b_report.json"
DEFAULT_EVALPLUS = Path("configs") / "temporal_code_evalplus_guardrail_split_v1.json"
DEFAULT_RETENTION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
PROJECT_DIR = Path(__file__).resolve().parents[2]


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_order(rows: Iterable[Dict[str, Any]], seed: int, label: str) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{label}:{row['chunk_uid']}".encode("utf-8")).hexdigest())


def _token_stream(path: Path, tokenizer: Any, token_cap: int) -> Iterable[int]:
    emitted = 0
    eos = tokenizer.eos_token_id
    for row in _jsonl(path):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if eos is not None:
            ids.append(int(eos))
        for token_id in ids:
            if emitted >= token_cap:
                return
            yield int(token_id)
            emitted += 1


def _load_tokenizer(plan: Dict[str, Any], allow_download: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        plan["target_model"]["tokenizer_id"],
        revision=plan["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
        use_fast=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _freeze_training_blocks(plan: Dict[str, Any], stage_b_dir: Path, output_dir: Path, allow_download: bool) -> Dict[str, Any]:
    tokenizer = _load_tokenizer(plan, allow_download)
    recipe = plan["training_recipe"]
    sequence_length = int(recipe["sequence_length"])
    token_cap = int(recipe["training_token_budget_cap"])
    blocks_dir = output_dir / "token_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in plan["training_arms"]:
        if arm == "base_no_update":
            continue
        source = stage_b_dir / f"{arm}.jsonl"
        buffer: List[int] = []
        blocks: List[torch.Tensor] = []
        consumed = 0
        for token_id in _token_stream(source, tokenizer, token_cap):
            buffer.append(token_id)
            consumed += 1
            if len(buffer) == sequence_length:
                blocks.append(torch.tensor(buffer, dtype=torch.int32))
                buffer = []
        if not blocks:
            raise RuntimeError(f"No complete training blocks for {arm}: consumed_tokens={consumed}")
        tensor = torch.stack(blocks)
        output = blocks_dir / f"{arm}.pt"
        torch.save({"input_ids": tensor}, output)
        results[arm] = {
            "source_jsonl": str(source),
            "source_sha256": sha256_file(source),
            "path": str(output),
            "sha256": sha256_file(output),
            "blocks": int(tensor.shape[0]),
            "sequence_length": sequence_length,
            "packed_tokens": int(tensor.numel()),
            "consumed_tokens_before_packing": int(consumed),
            "training_token_budget_cap": token_cap,
            "dropped_tail_tokens": int(consumed - tensor.numel()),
        }
    packed = {row["packed_tokens"] for row in results.values()}
    if len(packed) != 1:
        raise RuntimeError(f"Packed token budgets differ across v2 arms: {results}")
    manifest = {
        "schema_version": "code-domain-v2-qwen3-4b-training-blocks-v1",
        "status": "v2_development_training_blocks_frozen",
        "training_token_budget_cap": token_cap,
        "common_packed_token_budget": next(iter(packed)),
        "blocks": results,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "V2 development token blocks only; no Stage-C outcome or Utility claim.",
    }
    save_json(blocks_dir / "block_manifest.json", manifest)
    return manifest


def _freeze_heldout(plan: Dict[str, Any], stage_a_dir: Path, output_dir: Path) -> Dict[str, Any]:
    heldout = plan["heldout_nll"]
    source = stage_a_dir / heldout["source_split"] / "stage_a_pass.jsonl"
    allowed = set(heldout["allowed_content_types"])
    budget = int(heldout["token_proxy_budget"])
    rows = [
        {**row, "token_proxy_count": token_proxy_count(str(row.get("text") or ""))}
        for row in _jsonl(source)
        if row.get("split") == heldout["source_split"]
        and row.get("stage_a_pass") is True
        and row.get("content_type") in allowed
    ]
    selected = []
    tokens = 0
    for row in _stable_order(rows, int(heldout["seed"]), heldout["development_slice_name"]):
        count = int(row["token_proxy_count"])
        if tokens + count > budget and selected:
            continue
        selected.append(row)
        tokens += count
        if tokens >= budget:
            break
    if not selected:
        raise RuntimeError("V2 development heldout selection produced no records.")
    output = output_dir / "heldouts" / f"{heldout['development_slice_name']}.jsonl"
    _write_jsonl(output, selected)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_split": heldout["source_split"],
        "selection_rule": heldout["selection_rule"],
        "seed": int(heldout["seed"]),
        "candidate_records": len(rows),
        "selected_records": len(selected),
        "selected_token_proxy": tokens,
        "token_proxy_budget": budget,
        "allowed_content_types": sorted(allowed),
        "content_type_counts": {
            value: sum(1 for row in selected if row.get("content_type") == value)
            for value in sorted({row.get("content_type") for row in selected})
        },
        "repository_count": len({row.get("repository_identity") for row in selected}),
    }


def _build_plan(design: Dict[str, Any], stage_b_report: Dict[str, Any], retention: Dict[str, Any]) -> Dict[str, Any]:
    token_cap = min(
        int(stage_b_report["primary_arms"][arm]["token_proxy_count"])
        for arm in (
            "curated_v2_equal_budget",
            "stageA_random_equal_budget",
            "raw_random_equal_budget",
            "known_high_quality_equal_budget",
        )
    )
    seeds = retention["contract"]["seed_contract"]["development_training_seeds"]
    return {
        "schema_version": "code-domain-v2-development-plan-qwen3-4b",
        "status": "frozen_before_v2_development_training_outcomes",
        "purpose": "Freeze the v2 Qwen3-4B QLoRA raw-vs-curated comparison after corpus expansion, balanced Stage-A readiness, and Stage-B v2 arm freeze.",
        "target_model": {
            "model_id": "Qwen/Qwen3-4B-Base",
            "tokenizer_id": "Qwen/Qwen3-4B-Base",
            "revision": "main",
        },
        "inputs": {
            "design": str(DEFAULT_DESIGN),
            "stage_b_v2_report": str(DEFAULT_STAGE_B_DIR / "stage_b_v2_arms_report.json"),
            "stage_a_balanced_dir": str(DEFAULT_STAGE_A_DIR),
            "evalplus_guardrail_split": str(DEFAULT_EVALPLUS),
            "retention_guardrail_plan": str(DEFAULT_RETENTION),
        },
        "training_arms": [
            "base_no_update",
            "raw_random_equal_budget",
            "stageA_random_equal_budget",
            "curated_v2_equal_budget",
            "known_high_quality_equal_budget",
        ],
        "primary_comparison": {
            "treatment": "curated_v2_equal_budget",
            "primary_baseline": "stageA_random_equal_budget",
            "supporting_baselines": ["raw_random_equal_budget", "base_no_update"],
            "reference_arm": "known_high_quality_equal_budget",
        },
        "training_recipe": {
            "method": "QLoRA continued pretraining",
            "quantization": "4-bit NF4 with double quantization",
            "compute_dtype": "bf16",
            "sequence_length": 2048,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "optimizer_steps": 20,
            "learning_rate": 0.00005,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "adapter": {"rank": 32, "alpha": 64, "dropout": 0.05, "target_modules": "all-linear"},
            "development_training_seeds": seeds,
            "same_seed_set_for_every_arm": True,
            "training_token_budget_cap": token_cap,
            "common_packed_token_budget": None,
        },
        "heldout_nll": {
            "development_slice_name": "development_code_nll_heldout",
            "source_split": "development",
            "source_file": str(DEFAULT_STAGE_A_DIR / "development" / "stage_a_pass.jsonl"),
            "allowed_content_types": ["code", "test"],
            "selection_rule": "Sort by sha256(seed + ':' + slice_name + ':' + chunk_uid), then take chunks until the token-proxy budget is reached.",
            "seed": 20260622,
            "token_proxy_budget": 131072,
            "confirmatory_read_forbidden": True,
        },
        "external_code_guardrails": {
            "development": [
                "EvalPlus HumanEval+ development split, pass_at_1_temperature_0",
                "EvalPlus MBPP+ development split, pass_at_1_temperature_0",
            ],
            "role": "code capability and retention guardrail; not a selector signal",
            "maximum_allowed_absolute_regression_macro_vs_base": 0.02,
        },
        "general_retention_guardrails": retention["contract"],
        "margin_calibration": design["margin_calibration"],
        "development_decision_rule": design["development_promotion_rule"],
        "forbidden_uses": [
            "using Utility, benchmark outcomes, retention outcomes, development outcomes, confirmatory outcomes, or human/LLM review labels in Stage B",
            "changing training seeds, optimizer steps, token budget, heldout slice, benchmark split, guardrail margin, or practical effect rule after development outcomes",
            "reading confirmatory outcomes before the v2 development decision and confirmatory protocol are frozen",
            "using different Stage-A baselines for sensitivity arms",
        ],
        "confirmatory_outcomes_read": False,
        "utility_scope": design["stage_boundaries"]["utility_scope"],
        "claim_boundary": "V2 development-plan freeze only; no model outcome, Utility result, release decision, or paper claim.",
    }


def freeze(
    design_path: Path,
    stage_b_dir: Path,
    stage_a_dir: Path,
    output_dir: Path,
    config_output: Path,
    report_path: Path,
    evalplus_path: Path,
    retention_path: Path,
    allow_download: bool,
) -> Dict[str, Any]:
    design = load_json(design_path)
    stage_b_report_path = stage_b_dir / "stage_b_v2_arms_report.json"
    stage_b_report = load_json(stage_b_report_path)
    evalplus = load_json(evalplus_path)
    retention = load_json(retention_path)
    blockers = []
    if stage_b_report.get("status") != "stage_b_v2_arms_frozen_before_stage_c":
        blockers.append("stage_b_v2_arms_not_frozen")
    if evalplus.get("status") != "frozen_before_model_outcomes":
        blockers.append("evalplus_guardrail_split_not_frozen")
    if retention.get("status") != "frozen_before_development_model_outcomes":
        blockers.append("retention_guardrail_plan_not_frozen")
    if design.get("confirmatory_outcomes_read_for_v2") is not False:
        blockers.append("v2_confirmatory_outcomes_already_read")

    plan = _build_plan(design, stage_b_report, retention)
    training_blocks = _freeze_training_blocks(plan, stage_b_dir, output_dir, allow_download)
    plan["training_recipe"]["common_packed_token_budget"] = training_blocks["common_packed_token_budget"]
    heldout = _freeze_heldout(plan, stage_a_dir, output_dir)
    plan["heldout_nll"]["frozen_heldout"] = heldout
    save_json(config_output, plan)
    source_sha256 = {
        str(design_path): sha256_file(design_path),
        str(stage_b_report_path): sha256_file(stage_b_report_path),
        str(evalplus_path): sha256_file(evalplus_path),
        str(retention_path): sha256_file(retention_path),
        str(config_output): sha256_file(config_output),
        str(stage_a_dir / "development" / "stage_a_pass.jsonl"): sha256_file(stage_a_dir / "development" / "stage_a_pass.jsonl"),
        str(training_blocks["blocks"]["curated_v2_equal_budget"]["path"]): training_blocks["blocks"]["curated_v2_equal_budget"]["sha256"],
    }
    report = {
        "schema_version": "code-domain-v2-development-plan-freeze-report-v1",
        "status": "v2_development_plan_frozen" if not blockers else "v2_development_plan_blocked",
        "source_sha256": source_sha256,
        "summary": {
            "training_arms": plan["training_arms"],
            "primary_comparison": plan["primary_comparison"],
            "development_training_seeds": plan["training_recipe"]["development_training_seeds"],
            "optimizer_steps": plan["training_recipe"]["optimizer_steps"],
            "gradient_accumulation_steps": plan["training_recipe"]["gradient_accumulation_steps"],
            "training_token_budget_cap": plan["training_recipe"]["training_token_budget_cap"],
            "common_packed_token_budget": plan["training_recipe"]["common_packed_token_budget"],
            "training_blocks": training_blocks,
            "heldout": heldout,
            "blockers": blockers,
        },
        "margin_calibration": plan["margin_calibration"],
        "development_decision_rule": plan["development_decision_rule"],
        "forbidden_uses": plan["forbidden_uses"],
        "confirmatory_outcomes_read": False,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain v2 Qwen3-4B development plan.")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--stage-b-dir", type=Path, default=DEFAULT_STAGE_B_DIR)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evalplus", type=Path, default=DEFAULT_EVALPLUS)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    report = freeze(
        args.design,
        args.stage_b_dir,
        args.stage_a_dir,
        args.output_dir,
        args.config_output,
        args.report,
        args.evalplus,
        args.retention,
        args.allow_download,
    )
    print({"status": report["status"], **report.get("summary", {})})
    return 0 if report["status"] == "v2_development_plan_frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
