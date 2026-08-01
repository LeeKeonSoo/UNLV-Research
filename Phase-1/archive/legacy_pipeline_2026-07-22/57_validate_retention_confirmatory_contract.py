#!/usr/bin/env python3
"""Validate the frozen retention-aware confirmatory contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import load_json, sha256_file


DEFAULT_PLAN = Path("configs") / "retention_recipe_confirmatory_plan_qwen25_0p5b_fineweb.json"


def _validate_file(payload: Dict[str, Any], label: str) -> Path:
    path = Path(str(payload.get("path") or ""))
    if not path.exists():
        raise AssertionError(f"{label}: missing {path}")
    if sha256_file(path) != str(payload.get("sha256") or ""):
        raise AssertionError(f"{label}: sha256 mismatch")
    return path


def validate(plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    if plan.get("schema_version") != "retention-recipe-confirmatory-plan-v1":
        raise AssertionError("Unexpected schema")
    if plan.get("fresh_training_seeds") != [20260612, 20260613]:
        raise AssertionError("Fresh seed contract changed")
    if ((plan.get("framework_scope") or {}).get("utility_scope")) != "Stage C validation only; never selector objective":
        raise AssertionError("Utility scope changed")
    candidate = plan["candidate"]
    if candidate["general_replay_fraction"] != 0.01 or candidate["training_recipe"]["learning_rate"] != 0.000005:
        raise AssertionError("Candidate recipe changed")
    _validate_file(candidate, "candidate")
    _validate_file(plan["matched_comparator"], "comparator")
    holdouts = plan.get("confirmatory_holdouts") or {}
    if holdouts.get("status") != "bound_before_confirmatory_training":
        raise AssertionError("Confirmatory holdouts are not bound")
    manifest_path = _validate_file(holdouts["manifest"], "holdout manifest")
    _validate_file(holdouts["target"], "target holdout")
    _validate_file(holdouts["external"], "external holdout")
    manifest = load_json(manifest_path)
    disjointness = manifest.get("disjointness") or {}
    required_zero = [
        "exact_uid_target_vs_train_and_previous_target",
        "exact_normalized_text_external_vs_train_and_previous_external",
        "coarse_minhash_signature_overlap_vs_train",
    ]
    if any(int(disjointness.get(key, -1)) != 0 for key in required_zero):
        raise AssertionError(f"Holdout disjointness failed: {disjointness}")
    return {
        "status": "pass",
        "candidate": candidate["arm"],
        "fresh_seeds": plan["fresh_training_seeds"],
        "target_holdout": holdouts["target"]["path"],
        "external_holdout": holdouts["external"]["path"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate retention confirmatory contract.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    print(validate(args.plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
