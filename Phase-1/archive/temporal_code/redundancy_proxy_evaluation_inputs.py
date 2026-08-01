#!/usr/bin/env python3
"""Freeze mechanism and retention inputs for the redundancy proxy experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors.torch import save_file

from data_eval_common import load_json, save_json, sha256_file
from ingestion.code_selection import _structural_saturation_risk


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_BLOCK_MANIFEST = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_packed_blocks_manifest.json"
)
DEFAULT_RETENTION = (
    ROOT / "outputs" / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
)
DEFAULT_EVALPLUS_SPLIT = (
    ROOT / "outputs" / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
)
DEFAULT_EVALPLUS_PREVALIDATION = (
    ROOT / "outputs" / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
)
DEFAULT_GENERAL_TEXT = (
    ROOT
    / "outputs"
    / "slm_update_experiments"
    / "fineweb_edu_canonical_slm_update_v1"
    / "external_guardrails"
    / "wikitext103_validation_test_guardrail.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "redundancy_saturation_proxy_qwen25_0p5b_v1" / "evaluation_inputs"
)
DEFAULT_CONFIG_OUTPUT = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json"
)
DEFAULT_REPORT_OUTPUT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_evaluation_inputs_freeze_report.json"
)

LM_EVAL_ROOT = Path(importlib.util.find_spec("lm_eval").origin).resolve().parent
LM_EVAL_TASK_FILES = {
    "hellaswag": [
        LM_EVAL_ROOT / "tasks" / "hellaswag" / "hellaswag.yaml",
        LM_EVAL_ROOT / "tasks" / "hellaswag" / "utils.py",
    ],
    "arc_challenge": [
        LM_EVAL_ROOT / "tasks" / "arc" / "arc_challenge.yaml",
        LM_EVAL_ROOT / "tasks" / "arc" / "arc_easy.yaml",
    ],
    "piqa": [LM_EVAL_ROOT / "tasks" / "piqa" / "piqa.yaml"],
    "winogrande": [
        LM_EVAL_ROOT / "tasks" / "winogrande" / "default.yaml",
        LM_EVAL_ROOT / "tasks" / "winogrande" / "preprocess_winogrande.py",
    ],
}

HF_DATASET_ROOTS = {
    "hellaswag": Path(
        "D:/UNLV-Research/hf_cache/datasets/Rowan___hellaswag/default/0.0.0/"
        "218ec52e09a7e7462a5400043bb9a69a41d06b76"
    ),
    "arc_challenge": Path(
        "D:/UNLV-Research/hf_cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/"
        "210d026faf9955653af8916fad021475a3f00453"
    ),
    "piqa": Path(
        "D:/UNLV-Research/hf_cache/datasets/baber___piqa/default/0.0.0/"
        "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"
    ),
    "winogrande": Path(
        "D:/UNLV-Research/hf_cache/datasets/allenai___winogrande/winogrande_xl/"
        "0.0.0/01e74176c63542e6b0bcb004dcdea22d94fb67b5"
    ),
}

HF_VALIDATION_FILES = {
    "hellaswag": "hellaswag-validation.arrow",
    "arc_challenge": "ai2_arc-validation.arrow",
    "piqa": "piqa-validation.arrow",
    "winogrande": "winogrande-validation.arrow",
}

EVALPLUS_CACHE_FILES = {
    "HumanEval+": Path(
        "C:/Users/ksl11/AppData/Local/evalplus/evalplus/Cache/"
        "HumanEvalPlus-v0.1.10.jsonl"
    ),
    "MBPP+": Path(
        "C:/Users/ksl11/AppData/Local/evalplus/evalplus/Cache/"
        "MbppPlus-v0.2.0.jsonl"
    ),
}


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _pack_all_complete_blocks(
    source: Path,
    tokenizer: Any,
    sequence_length: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    ids: List[int] = []
    records = 0
    eos = tokenizer.eos_token_id
    for row in _jsonl(source):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        records += 1
        ids.extend(int(value) for value in tokenizer(text, add_special_tokens=False).input_ids)
        if eos is not None:
            ids.append(int(eos))
    block_count = len(ids) // sequence_length
    if block_count <= 0:
        raise RuntimeError(f"No complete blocks from {source}")
    packed = block_count * sequence_length
    tensor = torch.tensor(ids[:packed], dtype=torch.int32).reshape(
        block_count,
        sequence_length,
    )
    return tensor.contiguous(), {
        "record_count": records,
        "raw_tokens_with_eos": len(ids),
        "blocks": block_count,
        "sequence_length": sequence_length,
        "packed_tokens": packed,
        "dropped_tail_tokens": len(ids) - packed,
        "tensor_content_sha256": _tensor_content_sha256(tensor),
    }


def _exact_stream_exposure(
    source: Path,
    tokenizer: Any,
    exact_tokens: int,
) -> Dict[str, Any]:
    used = 0
    rows = []
    eos = tokenizer.eos_token_id
    for row in _jsonl(source):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        record_ids = list(tokenizer(text, add_special_tokens=False).input_ids)
        if eos is not None:
            record_ids.append(int(eos))
        take = min(len(record_ids), exact_tokens - used)
        if take <= 0:
            break
        evidence = row.get("stage_b_evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        rows.append(
            {
                "tokens": take,
                "match_count": int(evidence.get("soft_structural_match_count") or 0),
                "risk": float(evidence.get("soft_structural_redundancy_risk") or 0.0),
                "content_type": str(provenance.get("content_type") or "unknown"),
                "repository": str(provenance.get("repository_identity") or ""),
            }
        )
        used += take
        if used == exact_tokens:
            break
    if used != exact_tokens:
        raise RuntimeError(f"Could not consume exact stream budget from {source}")
    return {
        "consumed_records": len(rows),
        "exact_tokens": used,
        "repository_count": len({row["repository"] for row in rows if row["repository"]}),
        "token_weighted_match_count": sum(
            row["tokens"] * row["match_count"] for row in rows
        )
        / used,
        "token_weighted_log2_one_plus_match_count": sum(
            row["tokens"] * math.log2(1 + row["match_count"]) for row in rows
        )
        / used,
        "token_weighted_structural_risk": sum(
            row["tokens"] * row["risk"] for row in rows
        )
        / used,
        "single_recurrence_token_share": sum(
            row["tokens"] for row in rows if row["match_count"] == 1
        )
        / used,
        "high_saturation_token_share_count_ge_2": sum(
            row["tokens"] for row in rows if row["match_count"] >= 2
        )
        / used,
        "severe_saturation_token_share_count_ge_4": sum(
            row["tokens"] for row in rows if row["match_count"] >= 4
        )
        / used,
        "test_token_share": sum(
            row["tokens"] for row in rows if row["content_type"] == "test"
        )
        / used,
    }


def _mechanism_contract(plan: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    exact_tokens = int(plan["tokenization_and_packing"]["exact_train_tokens_per_arm"])
    curves = {
        mode: {
            str(count): _structural_saturation_risk(count, mode)
            for count in (0, 1, 2, 4, 8)
        }
        for mode in ("binary_current", "log_count")
    }
    exposure = {
        arm: _exact_stream_exposure(Path(contract["path"]), tokenizer, exact_tokens)
        for arm, contract in plan["arms"].items()
    }
    binary = exposure["binary_current_equal_budget"]
    candidate = exposure["log_count_equal_budget"]
    checks = {
        "candidate_risk_increases_before_cap_and_is_nondecreasing": (
            curves["log_count"]["2"] > curves["log_count"]["1"]
            and curves["log_count"]["4"] > curves["log_count"]["2"]
            and curves["log_count"]["8"] >= curves["log_count"]["4"]
        ),
        "binary_risk_is_flat_after_first_match": (
            curves["binary_current"]["1"]
            == curves["binary_current"]["2"]
            == curves["binary_current"]["4"]
            == curves["binary_current"]["8"]
        ),
        "candidate_does_not_increase_high_saturation_token_share": (
            candidate["high_saturation_token_share_count_ge_2"]
            <= binary["high_saturation_token_share_count_ge_2"] + 1e-12
        ),
        "candidate_does_not_increase_severe_saturation_token_share": (
            candidate["severe_saturation_token_share_count_ge_4"]
            <= binary["severe_saturation_token_share_count_ge_4"] + 1e-12
        ),
        "candidate_preserves_repository_count": (
            candidate["repository_count"] >= binary["repository_count"]
        ),
        "candidate_preserves_test_token_share_within_one_point": (
            candidate["test_token_share"] >= binary["test_token_share"] - 0.01
        ),
    }
    return {
        "definition": (
            "A single structural recurrence (match_count=1) is reported but is not "
            "classified as saturation. High saturation begins at match_count>=2."
        ),
        "policy_response_curve": curves,
        "exact_stream_exposure": exposure,
        "checks": checks,
        "status": (
            "template_saturation_mechanism_precheck_passed"
            if all(checks.values())
            else "template_saturation_mechanism_precheck_failed"
        ),
        "claim_boundary": (
            "Outcome-free mechanism diagnostic only. It validates count response "
            "and selected-stream saturation exposure, not downstream Utility."
        ),
    }


def _general_task_contract() -> Dict[str, Any]:
    task_contracts = {}
    for task, files in LM_EVAL_TASK_FILES.items():
        dataset_root = HF_DATASET_ROOTS[task]
        info = load_json(dataset_root / "dataset_info.json")
        validation_path = dataset_root / HF_VALIDATION_FILES[task]
        task_contracts[task] = {
            "task_implementation": [_artifact(path) for path in files],
            "dataset_cache_root": str(dataset_root),
            "dataset_info": _artifact(dataset_root / "dataset_info.json"),
            "validation_split": _artifact(validation_path),
            "validation_examples": int(info["splits"]["validation"]["num_examples"]),
        }
    return {
        "harness": "lm-evaluation-harness",
        "version": importlib.metadata.version("lm_eval"),
        "tasks": task_contracts,
        "num_fewshot": 0,
        "limit": None,
        "batch_size": "auto",
        "primary_metric": "accuracy",
        "maximum_allowed_absolute_regression_per_suite": 0.02,
        "maximum_allowed_absolute_regression_macro": 0.01,
    }


def _evalplus_contract(
    split_plan_path: Path,
    prevalidation_path: Path,
) -> Dict[str, Any]:
    split_plan = load_json(split_plan_path)
    development_rows = [
        row
        for row in split_plan["records"]
        if row.get("assigned_split") == "development"
    ]
    task_ids = [f"{row['dataset']}\t{row['task_id']}" for row in development_rows]
    task_id_sha = hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()
    prevalidation = load_json(prevalidation_path)
    return {
        "package": "evalplus",
        "version": importlib.metadata.version("evalplus"),
        "development_task_count": len(development_rows),
        "suite_counts": {
            suite: sum(1 for row in development_rows if row["dataset"] == suite)
            for suite in ("HumanEval+", "MBPP+")
        },
        "development_task_id_order_sha256": task_id_sha,
        "split_plan": _artifact(split_plan_path),
        "prevalidation": _artifact(prevalidation_path),
        "execution_support_tier": prevalidation["decision"]["execution_support_tier"],
        "isolated_image_tag": prevalidation["environment"]["isolated_image_tag"],
        "isolated_image_id": prevalidation["environment"]["isolated_image_id"],
        "dataset_cache": {
            name: _artifact(path) for name, path in EVALPLUS_CACHE_FILES.items()
        },
        "temperature": 0,
        "samples_per_task": 1,
        "metric": "pass_at_1",
        "maximum_allowed_absolute_regression_per_suite": 0.02,
        "maximum_allowed_absolute_regression_macro": 0.02,
        "task_content_training_authorized": False,
    }


def freeze(
    experiment_path: Path,
    block_manifest_path: Path,
    retention_path: Path,
    evalplus_split_path: Path,
    evalplus_prevalidation_path: Path,
    general_text_path: Path,
    output_dir: Path,
    config_output: Path,
    report_output: Path,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    experiment = load_json(experiment_path)
    block_manifest = load_json(block_manifest_path)
    retention = load_json(retention_path)
    blockers = []
    if block_manifest["frozen_config"]["sha256"] != sha256_file(experiment_path):
        blockers.append("packed_blocks_do_not_match_frozen_experiment")
    if experiment["status"] != "frozen_before_proxy_training_outcomes":
        blockers.append("proxy_experiment_not_frozen")

    tokenizer_path = Path(experiment["target_model"]["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    sequence_length = int(experiment["tokenization_and_packing"]["sequence_length"])
    general_text_tensor, general_text_audit = _pack_all_complete_blocks(
        general_text_path,
        tokenizer,
        sequence_length,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    general_text_blocks = output_dir / "wikitext103_qwen25_0p5b.safetensors"
    save_file({"input_ids": general_text_tensor}, general_text_blocks)
    general_text_contract = {
        "source": _artifact(general_text_path),
        "blocks": {
            "path": str(general_text_blocks),
            "file_sha256": sha256_file(general_text_blocks),
            **general_text_audit,
        },
        "metric": "mean causal-LM NLL",
        "reference_arm": "base_no_update",
        "maximum_allowed_mean_nll_increase": 0.01,
        "confidence_rule": (
            "one-sided 95% training-seed-level upper confidence bound must not "
            "exceed 0.01"
        ),
    }

    mechanism = _mechanism_contract(experiment, tokenizer)
    if mechanism["status"] != "template_saturation_mechanism_precheck_passed":
        blockers.append("template_saturation_mechanism_precheck_failed")

    base_retention = retention["contract"]
    proxy_arms = [
        "base_no_update",
        "binary_current_equal_budget",
        "log_count_equal_budget",
        "stageA_random_common_disjoint_equal_budget",
    ]
    proxy_seeds = [int(seed) for seed in experiment["training_recipe"]["seeds"]]
    config = {
        "schema_version": "temporal-code-redundancy-proxy-evaluation-inputs-v1",
        "status": (
            "frozen_before_proxy_training_outcomes"
            if not blockers
            else "evaluation_input_freeze_blocked"
        ),
        "purpose": (
            "Freeze the outcome-free mechanism diagnostic and exact Stage-C "
            "retention inputs for the Qwen2.5-0.5B redundancy proxy experiment."
        ),
        "source_contracts": {
            "proxy_experiment": _artifact(experiment_path),
            "packed_blocks_manifest": _artifact(block_manifest_path),
            "retention_guardrail_plan": _artifact(retention_path),
        },
        "comparison_arms": proxy_arms,
        "training_seeds": proxy_seeds,
        "base_evaluated_once": True,
        "mechanism_diagnostic": mechanism,
        "general_text_retention": general_text_contract,
        "general_task_retention": _general_task_contract(),
        "code_retention": _evalplus_contract(
            evalplus_split_path,
            evalplus_prevalidation_path,
        ),
        "decision_contract": {
            "all_guardrails_mandatory": True,
            "reference_arm": base_retention["reference_arm"],
            "missing_evidence_action": "abstain",
            "failed_guardrail_action": "reject_candidate",
            "same_seed_set_for_every_trained_arm": True,
            "mechanism_precheck_must_pass_before_training": True,
            "general_text_margin_inherited": (
                base_retention["general_text_guardrail"][
                    "maximum_allowed_mean_nll_increase"
                ]
            ),
            "general_task_margins_inherited": base_retention[
                "general_task_guardrail"
            ],
            "code_margins_inherited": base_retention["code_guardrail"],
        },
        "forbidden_uses": [
            "using mechanism or retention outcomes in Stage-B scoring",
            "changing task IDs, cached datasets, margins, or seeds after outcomes",
            "using confirmatory EvalPlus task IDs in this development proxy cycle",
            "training on general-text or EvalPlus evaluation content",
        ],
        "confirmatory_outcomes_read": False,
        "blockers": blockers,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Evaluation-input and mechanism precheck freeze only. No model was "
            "trained and no Utility, retention, promotion, or release result exists."
        ),
    }
    save_json(config_output, config)
    report = {
        "schema_version": "redundancy-proxy-evaluation-inputs-freeze-report-v1",
        "status": (
            "redundancy_proxy_evaluation_inputs_frozen"
            if not blockers
            else "redundancy_proxy_evaluation_inputs_freeze_blocked"
        ),
        "config_path": str(config_output),
        "config_sha256": sha256_file(config_output),
        "mechanism_status": mechanism["status"],
        "mechanism_checks": mechanism["checks"],
        "general_text_blocks": general_text_audit["blocks"],
        "general_text_tokens": general_text_audit["packed_tokens"],
        "general_task_validation_examples": {
            task: row["validation_examples"]
            for task, row in config["general_task_retention"]["tasks"].items()
        },
        "evalplus_development_task_count": config["code_retention"][
            "development_task_count"
        ],
        "comparison_arms": proxy_arms,
        "training_seeds": proxy_seeds,
        "blockers": blockers,
        "claim_boundary": config["claim_boundary"],
    }
    save_json(report_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze redundancy proxy mechanism and retention inputs."
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--block-manifest", type=Path, default=DEFAULT_BLOCK_MANIFEST)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--evalplus-split", type=Path, default=DEFAULT_EVALPLUS_SPLIT)
    parser.add_argument(
        "--evalplus-prevalidation",
        type=Path,
        default=DEFAULT_EVALPLUS_PREVALIDATION,
    )
    parser.add_argument("--general-text", type=Path, default=DEFAULT_GENERAL_TEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()
    report = freeze(
        args.experiment,
        args.block_manifest,
        args.retention,
        args.evalplus_split,
        args.evalplus_prevalidation,
        args.general_text,
        args.output_dir,
        args.config_output,
        args.report_output,
    )
    print(report)
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
