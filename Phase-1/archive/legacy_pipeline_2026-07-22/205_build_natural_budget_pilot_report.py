#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, save_json


JsonMap = Dict[str, Any]


def _load(path: Path) -> JsonMap:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize(domain: str, out_dir: Path, raw_arm: str, curated_arm: str) -> JsonMap:
    arms = _load(out_dir / "natural_budget_arms_report.json")["arms"]
    steps = _load(out_dir / "token_blocks" / "natural_budget_steps_report.json")
    nll_dir = out_dir / "heldout_nll"
    base = _load(nll_dir / "base_no_update.json")
    raw = _load(nll_dir / f"{raw_arm}_seed101.json")
    curated = _load(nll_dir / f"{curated_arm}_seed101.json")
    raw_tokens = int(arms[raw_arm]["token_proxy_count"])
    curated_tokens = int(arms[curated_arm]["token_proxy_count"])
    raw_packed = int(steps["packed_tokens_by_arm"][raw_arm])
    curated_packed = int(steps["packed_tokens_by_arm"][curated_arm])
    return {
        "domain": domain,
        "seed_scope": "pilot_seed_101_only",
        "arms": {
            "base_no_update": {
                "mean_nll": float(base["mean_nll"]),
                "optimizer_steps": 0,
                "eval_tokens": int(base["tokens"]),
            },
            raw_arm: {
                "mean_nll": float(raw["mean_nll"]),
                "optimizer_steps": int(raw["optimizer_steps"]),
                "record_count": int(arms[raw_arm]["records"]),
                "token_proxy_count": raw_tokens,
                "packed_training_tokens": raw_packed,
            },
            curated_arm: {
                "mean_nll": float(curated["mean_nll"]),
                "optimizer_steps": int(curated["optimizer_steps"]),
                "record_count": int(arms[curated_arm]["records"]),
                "token_proxy_count": curated_tokens,
                "packed_training_tokens": curated_packed,
            },
        },
        "natural_budget_reduction": {
            "record_reduction_fraction": 1.0 - (int(arms[curated_arm]["records"]) / int(arms[raw_arm]["records"])),
            "token_proxy_reduction_fraction": 1.0 - (curated_tokens / raw_tokens),
            "packed_training_token_reduction_fraction": 1.0 - (curated_packed / raw_packed),
            "optimizer_step_reduction_fraction": 1.0 - (int(curated["optimizer_steps"]) / int(raw["optimizer_steps"])),
        },
        "nll_deltas_lower_is_better": {
            "curated_minus_raw_full": float(curated["mean_nll"]) - float(raw["mean_nll"]),
            "raw_full_minus_base": float(raw["mean_nll"]) - float(base["mean_nll"]),
            "curated_minus_base": float(curated["mean_nll"]) - float(base["mean_nll"]),
        },
        "pilot_decision": (
            "curated_better_than_raw_full_on_nll"
            if float(curated["mean_nll"]) < float(raw["mean_nll"])
            else "raw_full_better_than_curated_on_nll"
        ),
    }


def build() -> JsonMap:
    report = {
        "schema_version": "natural-budget-stage-c-pilot-report-v1",
        "status": "pilot_seed_101_natural_budget_nll_completed",
        "scope": "1 seed pilot; not confirmatory. Uses natural arm sizes and arm-specific one-pass optimizer steps.",
        "domains": {
            "code": _summarize(
                "code",
                OUTPUT_DIR / "code_domain_natural_budget_qwen3_4b",
                "raw_full_natural",
                "curated_v2_natural",
            ),
            "math": _summarize(
                "math",
                OUTPUT_DIR / "math_domain_natural_budget_qwen3_4b",
                "raw_full_natural",
                "curated_math_natural",
            ),
        },
        "claim_boundary": (
            "Evidence is sufficient for pilot direction only; multi-seed and downstream benchmark confirmation "
            "remain required before paper claim."
        ),
    }
    save_json(OUTPUT_DIR / "validation" / "natural_budget_stage_c_pilot_report.json", report)
    return report


def main() -> int:
    print(json.dumps(build(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
