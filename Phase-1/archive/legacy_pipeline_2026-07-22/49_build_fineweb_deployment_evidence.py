#!/usr/bin/env python3
"""Build release-policy evidence from the frozen FineWeb confirmatory outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "fineweb_deployment_evidence.json"


def _nll(path: Path) -> float:
    payload = load_json(path)
    value = payload.get("mean_nll")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing mean_nll: {path}")
    return float(value)


def build_evidence(experiment_dir: Path) -> Dict[str, Any]:
    eval_dir = experiment_dir / "eval_results"
    broad = "confirmatory_broad_stageA_eval"
    target = "confirmatory_coverage_stratified_stageA_eval"
    return {
        "schema_version": "deployment-evidence-v1",
        "evidence_identity": "fineweb_edu_qwen25_0p5b_confirmatory_seed20260609",
        "usable_data_sufficient": True,
        "arms": {
            "base_no_update": {
                "evaluations": {
                    broad: _nll(eval_dir / f"context_seed20260608_base_{broad}.json"),
                    target: _nll(eval_dir / f"context_seed20260608_base_{target}.json"),
                }
            },
            "selected_only": {
                "evaluations": {
                    broad: _nll(eval_dir / f"context_seed20260608_curated_{broad}.json"),
                    target: _nll(eval_dir / f"context_seed20260608_curated_{target}.json"),
                },
                "evidence_role": "mechanism_context_only; exploratory seed",
            },
            "coverage_backfilled": {
                "evaluations": {
                    broad: _nll(eval_dir / f"confirm_seed20260609_backfilled_interleaved50_{broad}.json"),
                    target: _nll(eval_dir / f"confirm_seed20260609_backfilled_interleaved50_{target}.json"),
                }
            },
            "stageA_broad": {
                "evaluations": {
                    broad: _nll(eval_dir / f"confirm_seed20260609_stageA_random_{broad}.json"),
                    target: _nll(eval_dir / f"confirm_seed20260609_stageA_random_{target}.json"),
                }
            },
        },
        "evidence_boundary": {
            "confirmatory_primary": broad,
            "confirmatory_secondary": target,
            "fresh_confirmatory_seed": 20260609,
            "selected_only_seed": 20260608,
            "external_guardrails_complete": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FineWeb deployment evidence.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evidence = build_evidence(args.experiment_dir)
    save_json(args.output, evidence)
    print({"evidence_identity": evidence["evidence_identity"], "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
