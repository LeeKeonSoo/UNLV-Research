#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

VALIDATION_DIR = OUTPUT_DIR / "validation"
CONFIG_PATH = Path("configs") / "paper_multidomain_benchmark_protocol_v1.json"
STAGE_B_REPORT = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2" / "stage_b_v2_arms_report.json"
TOKEN_BLOCK_MANIFEST = OUTPUT_DIR / "code_domain_v2_development_qwen3_4b" / "token_blocks" / "block_manifest.json"
STAGE_C_REPORT = VALIDATION_DIR / "stage_c_training_validation_report.json"
COMPARISON_TABLES = VALIDATION_DIR / "paper_comparison_tables.json"
MATH_BLOCK4_REPORT = VALIDATION_DIR / "math_domain_block4_acquisition_report.json"
MATH_EQUAL_TOKEN_REPORT = VALIDATION_DIR / "math_domain_equal_token_arms_report.json"
MATH_STAGE_C_PROTOCOL = Path("configs") / "math_domain_stage_c_protocol_qwen3_4b_v1.json"
DEFAULT_OUTPUT = VALIDATION_DIR / "raw_vs_curated_benchmark_readiness_report.json"
DEFAULT_MD_OUTPUT = VALIDATION_DIR / "raw_vs_curated_benchmark_readiness_report.md"


def _read(path: Path) -> JsonMap:
    payload = load_json(path) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _as_map(value: JsonValue) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _exists(path_text: JsonValue) -> bool:
    return isinstance(path_text, str) and Path(path_text).exists()


def _source(path: Path) -> JsonMap:
    return {"path": str(path), "exists": path.exists()}


def _arm_readiness(stage_b: JsonMap, block_manifest: JsonMap, arm: str) -> JsonMap:
    primary = _as_map(_as_map(stage_b.get("primary_arms")).get(arm))
    block = _as_map(_as_map(block_manifest.get("blocks")).get(arm))
    return {
        "records": primary.get("records"),
        "token_proxy_count": primary.get("token_proxy_count"),
        "repository_count": primary.get("repository_count"),
        "content_type_counts": primary.get("content_type_counts"),
        "source_jsonl": block.get("source_jsonl"),
        "source_jsonl_exists": _exists(block.get("source_jsonl")),
        "source_sha256": block.get("source_sha256"),
        "token_block": block.get("path"),
        "token_block_exists": _exists(block.get("path")),
        "token_block_sha256": block.get("sha256"),
        "packed_tokens": block.get("packed_tokens"),
        "training_token_budget_cap": block.get("training_token_budget_cap"),
    }


def _math_domain_status(math_report: JsonMap, math_arms_report: JsonMap, math_protocol: JsonMap) -> JsonMap:
    if math_protocol.get("status") == "frozen_before_math_stage_c_training_or_benchmark_outcomes":
        return {
            "domain": "math",
            "status": "stage_c_protocol_frozen_training_pending",
            "completed_stage_c": [],
            "completed_inputs": [
                "raw mixed math pool",
                "known high-quality math reference pool",
                "Stage 0/A/B materialization",
                "equal-token math training arms",
                "held-out math NLL slice",
                "GSM8K/MATH benchmark command contract",
            ],
            "remaining": ["run equal-token math fine-tuning", "held-out math NLL", "GSM8K", "MATH"],
        }
    if math_arms_report.get("status") == "math_equal_token_arms_materialized":
        return {
            "domain": "math",
            "status": "equal_token_arms_ready_stage_c_pending",
            "completed_stage_c": [],
            "completed_inputs": [
                "raw mixed math pool",
                "known high-quality math reference pool",
                "Stage 0/A/B materialization",
                "equal-token math training arms",
            ],
            "remaining": ["held-out math NLL", "GSM8K", "MATH"],
        }
    if math_report.get("status") == "math_domain_block4_acquisition_pools_ready":
        return {
            "domain": "math",
            "status": "acquisition_pools_ready_stage_materialization_pending",
            "completed_stage_c": [],
            "completed_inputs": ["raw mixed math pool", "known high-quality math reference pool"],
            "remaining": ["Stage 0/A/B materialization", "equal-token math arms", "GSM8K", "MATH"],
        }
    return {
        "domain": "math",
        "status": "acquisition_required",
        "completed_stage_c": [],
        "remaining": ["raw mixed math pool", "known high-quality math reference pool", "GSM8K", "MATH"],
    }


def _domain_status(
    protocol: JsonMap,
    math_report: JsonMap,
    math_arms_report: JsonMap,
    math_protocol: JsonMap,
) -> list[JsonMap]:
    domains = []
    for raw_domain in _as_list(protocol.get("domains")):
        domain = _as_map(raw_domain)
        name = str(domain.get("domain"))
        if name == "code":
            domains.append(
                {
                    "domain": "code",
                    "status": "payloads_ready_swebench_pending",
                    "completed_stage_c": ["held-out code NLL", "EvalPlus guardrail", "general retention guardrails"],
                    "remaining": ["SWE-bench Lite or Verified execution if compute budget allows"],
                }
            )
        elif name == "math":
            domains.append(_math_domain_status(math_report, math_arms_report, math_protocol))
        elif name == "general_text_instruction":
            domains.append(
                {
                    "domain": "general_text_instruction",
                    "status": "acquisition_required",
                    "completed_stage_c": [],
                    "remaining": ["raw mixed general/instruction pool", "reference pool", "instruction/general benchmarks"],
                }
            )
    return domains


def build(output_path: Path, md_output_path: Path) -> JsonMap:
    protocol = _read(CONFIG_PATH)
    stage_b = _read(STAGE_B_REPORT)
    block_manifest = _read(TOKEN_BLOCK_MANIFEST)
    stage_c = _read(STAGE_C_REPORT)
    comparison = _read(COMPARISON_TABLES)
    math_report = _read(MATH_BLOCK4_REPORT)
    math_arms_report = _read(MATH_EQUAL_TOKEN_REPORT)
    math_protocol = _read(MATH_STAGE_C_PROTOCOL)
    arm_names = [
        "raw_random_equal_budget",
        "stageA_random_equal_budget",
        "curated_v2_equal_budget",
        "known_high_quality_equal_budget",
    ]
    arms = {arm: _arm_readiness(stage_b, block_manifest, arm) for arm in arm_names}
    all_code_payloads_ready = all(
        bool(_as_map(arm).get("source_jsonl_exists")) and bool(_as_map(arm).get("token_block_exists"))
        for arm in arms.values()
    )
    packed_budgets = sorted(
        {
            int(_as_map(arm).get("packed_tokens"))
            for arm in arms.values()
            if isinstance(_as_map(arm).get("packed_tokens"), int)
        }
    )
    equal_packed_tokens = len(packed_budgets) == 1
    nll_supported = _as_map(stage_c.get("claim_decision")).get("target_nll_training_effect_supported") is True
    pairwise = _as_map(comparison.get("stage_c_pairwise_table"))
    report = {
        "schema_version": "raw-vs-curated-benchmark-readiness-v1",
        "status": (
            "code_domain_ready_next_domains_pending"
            if all_code_payloads_ready and equal_packed_tokens and nll_supported
            else "benchmark_readiness_blocked"
        ),
        "claim_contract": {
            "primary_question": "Does a curated equal-token fine-tuning arm outperform raw and Stage-A random equal-token arms on Stage-C benchmarks?",
            "selector_rule": "Benchmarks and Utility are Stage-C only and never Stage-B selector inputs.",
            "current_claim_scope": _as_map(protocol.get("claim_boundary")).get("current_paper_claim"),
            "expanded_claim_requires": _as_map(protocol.get("claim_boundary")).get("expanded_claim_requires"),
        },
        "code_domain": {
            "payloads_ready": all_code_payloads_ready,
            "equal_packed_tokens": equal_packed_tokens,
            "packed_token_values": packed_budgets,
            "training_token_budget_cap": block_manifest.get("training_token_budget_cap"),
            "stage_b_freeze_status": stage_b.get("status"),
            "stage_c_training_status": stage_c.get("status"),
            "curated_vs_stageA_random_mean_nll_reduction": _as_map(
                pairwise.get("curated_vs_stageA_random")
            ).get("mean_nll_reduction"),
            "arms": arms,
        },
        "domain_status": _domain_status(protocol, math_report, math_arms_report, math_protocol),
        "benchmark_order": [
            "held-out domain NLL smoke",
            "domain-specific lightweight benchmark",
            "external high-cost benchmark only after payloads and margins are frozen",
        ],
        "next_block_actions": [
            "run equal-token math fine-tuning arms with the frozen training budget",
            "run held-out math NLL after training outputs exist",
            "implement or run frozen GSM8K/MATH benchmark evaluator without feeding outcomes back into Stage B",
            "decide whether SWE-bench Lite is feasible as a later code-domain capstone",
        ],
        "sources": {
            "protocol": _source(CONFIG_PATH),
            "stage_b_report": _source(STAGE_B_REPORT),
            "token_block_manifest": _source(TOKEN_BLOCK_MANIFEST),
            "stage_c_report": _source(STAGE_C_REPORT),
            "comparison_tables": _source(COMPARISON_TABLES),
            "math_block4_report": _source(MATH_BLOCK4_REPORT),
            "math_equal_token_report": _source(MATH_EQUAL_TOKEN_REPORT),
            "math_stage_c_protocol": _source(MATH_STAGE_C_PROTOCOL),
        },
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    code = _as_map(report.get("code_domain"))
    lines = [
        "# Raw-vs-Curated Benchmark Readiness",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Code Domain",
        "",
        f"Payloads ready: `{code.get('payloads_ready')}`",
        f"Equal packed tokens: `{code.get('equal_packed_tokens')}`",
        f"Packed token values: `{code.get('packed_token_values')}`",
        f"Stage-C status: `{code.get('stage_c_training_status')}`",
        f"Curated vs Stage-A random mean NLL reduction: `{code.get('curated_vs_stageA_random_mean_nll_reduction')}`",
        "",
        "## Next Block Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in _as_list(report.get("next_block_actions")))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.output, args.md_output)
    print(f"[raw-vs-curated-benchmark-readiness] {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
