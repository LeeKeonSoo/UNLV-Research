#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

MATERIALIZATION_DIR = OUTPUT_DIR / "math_domain_stage_materialization"
OUTPUT_DIR_STAGE_C = OUTPUT_DIR / "math_domain_stage_c_qwen3_4b"
PROTOCOL_PATH = Path("configs") / "math_domain_stage_c_protocol_qwen3_4b_v1.json"
MATERIALIZATION_REPORT = OUTPUT_DIR / "validation" / "math_domain_equal_token_arms_report.json"
HELDOUT_PATH = OUTPUT_DIR_STAGE_C / "heldouts" / "math_nll_heldout.jsonl"


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
    return int(value) if isinstance(value, int | float | str) else 0


def _arm_paths() -> dict[str, Path]:
    return {
        "raw_random_equal_budget": MATERIALIZATION_DIR / "raw_random_equal_budget.jsonl",
        "stageA_random_equal_budget": MATERIALIZATION_DIR / "stageA_random_equal_budget.jsonl",
        "curated_math_equal_budget": MATERIALIZATION_DIR / "curated_math_equal_budget.jsonl",
        "known_high_quality_equal_budget": MATERIALIZATION_DIR / "known_high_quality_equal_budget.jsonl",
    }


def _used_chunk_ids() -> set[str]:
    used: set[str] = set()
    for path in _arm_paths().values():
        used.update(str(row["chunk_uid"]) for row in _jsonl(path))
    return used


def _freeze_heldout(seed: int, budget: int) -> JsonMap:
    candidates = [row for row in _jsonl(MATERIALIZATION_DIR / "stage_a_pass.jsonl") if str(row["chunk_uid"]) not in _used_chunk_ids()]
    ordered = sorted(candidates, key=lambda row: hashlib.sha256(f"{seed}:math-heldout:{row['chunk_uid']}".encode()).hexdigest())
    selected: list[JsonMap] = []
    total = 0
    for row in ordered:
        selected.append(row)
        total += _token(row)
        if total >= budget:
            break
    _write_jsonl(HELDOUT_PATH, selected)
    return {
        "path": str(HELDOUT_PATH),
        "sha256": sha256_file(HELDOUT_PATH),
        "selection_rule": "Exclude every equal-token arm chunk, sort by sha256(seed + ':math-heldout:' + chunk_uid), then take until the token budget is reached.",
        "seed": seed,
        "candidate_records": len(candidates),
        "selected_records": len(selected),
        "selected_token_proxy": total,
        "token_proxy_budget": budget,
    }


def build() -> JsonMap:
    arms_report = load_json(MATERIALIZATION_REPORT)
    training_cap = int(arms_report["training_token_budget_cap"])
    heldout = _freeze_heldout(seed=20260703, budget=65536)
    arm_paths = _arm_paths()
    protocol = {
        "schema_version": "math-domain-stage-c-protocol-qwen3-4b-v1",
        "status": "frozen_before_math_stage_c_training_or_benchmark_outcomes",
        "target_model": {"model_id": "Qwen/Qwen3-4B-Base", "tokenizer_id": "Qwen/Qwen3-4B-Base", "revision": "main"},
        "source_materialization": {
            "path": str(MATERIALIZATION_REPORT),
            "status": arms_report["status"],
            "sha256": sha256_file(MATERIALIZATION_REPORT),
        },
        "training_arms": ["base_no_update", *arm_paths.keys()],
        "primary_comparison": {
            "treatment": "curated_math_equal_budget",
            "primary_baseline": "stageA_random_equal_budget",
            "supporting_baselines": ["raw_random_equal_budget", "base_no_update"],
            "reference_arm": "known_high_quality_equal_budget",
        },
        "confirmatory_training_recipe": {
            "method": "QLoRA continued pretraining",
            "quantization": "4-bit NF4 with double quantization",
            "compute_dtype": "bf16",
            "sequence_length": 2048,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "optimizer_steps": 20,
            "learning_rate": 5e-05,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "adapter": {"rank": 32, "alpha": 64, "dropout": 0.05, "target_modules": "all-linear"},
            "training_token_budget_cap": training_cap,
            "confirmatory_training_seeds": [101, 131, 163],
            "same_seed_set_for_every_arm": True,
        },
        "arm_token_counts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in arm_paths.items()
        },
        "heldout_nll": {"frozen_heldout": heldout, "metric": "mean_nll", "direction": "lower_is_better"},
        "stage_c_benchmarks": [
            {"name": "GSM8K", "dataset_id": "openai/gsm8k", "config": "main", "split": "test", "metric": "exact_match_accuracy"},
            {"name": "MATH", "dataset_id": "hendrycks/competition_math", "split": "test", "metric": "exact_match_accuracy"},
        ],
        "primary_success_rule": {
            "primary_metric": "math_nll_heldout mean NLL",
            "required_absolute_nll_reduction": 0.003,
            "benchmark_support_rule": "Curated must not underperform Stage-A random on both GSM8K and MATH; benchmark accuracy is supporting evidence until commands are implemented.",
            "failure_action": "report negative finding or abstain; do not tune Stage B using Stage-C outcomes",
        },
        "command_templates": [
            "python 37_run_slm_update_training.py prepare-blocks --plan configs/math_domain_stage_c_protocol_qwen3_4b_v1.json --blocks-dir outputs/math_domain_stage_c_qwen3_4b/token_blocks --arms raw_random_equal_budget stageA_random_equal_budget curated_math_equal_budget known_high_quality_equal_budget --sequence-length 2048 --token-budget 119163 --allow-download",
            "python 141_run_code_domain_development_qlora.py train-missing --plan configs/math_domain_stage_c_protocol_qwen3_4b_v1.json --output-dir outputs/math_domain_stage_c_qwen3_4b --blocks-dir outputs/math_domain_stage_c_qwen3_4b/token_blocks --arms raw_random_equal_budget,stageA_random_equal_budget,curated_math_equal_budget,known_high_quality_equal_budget --seeds 101,131,163 --allow-download",
            "python 141_run_code_domain_development_qlora.py prepare-eval-blocks --plan configs/math_domain_stage_c_protocol_qwen3_4b_v1.json --output-dir outputs/math_domain_stage_c_qwen3_4b --allow-download",
            "python 141_run_code_domain_development_qlora.py eval-missing --plan configs/math_domain_stage_c_protocol_qwen3_4b_v1.json --output-dir outputs/math_domain_stage_c_qwen3_4b --arms raw_random_equal_budget,stageA_random_equal_budget,curated_math_equal_budget,known_high_quality_equal_budget --seeds 101,131,163 --allow-download",
            "python 204_run_math_domain_benchmark_eval.py --plan configs/math_domain_stage_c_protocol_qwen3_4b_v1.json --benchmarks GSM8K,MATH --output-dir outputs/math_domain_stage_c_qwen3_4b/math_benchmarks",
        ],
        "forbidden_uses": [
            "using held-out math NLL, GSM8K, MATH, or Utility outcomes in Stage B",
            "changing training seeds, token budget, heldout slice, benchmark split, or practical margin after Stage-C outcomes",
            "using GSM8K or MATH benchmark content as training candidates",
            "claiming math-domain downstream improvement before Stage-C results exist",
        ],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Math Stage-C protocol freeze only; no training, benchmark, release, or paper-success claim.",
        "stage_c_outcomes_read": False,
    }
    save_json(PROTOCOL_PATH, protocol)
    return protocol


def main() -> int:
    protocol = build()
    print(f"[math-domain-stage-c-protocol] {protocol['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
