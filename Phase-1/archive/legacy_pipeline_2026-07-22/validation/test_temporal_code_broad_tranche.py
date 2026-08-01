#!/usr/bin/env python3
"""Contract checks for the frozen broad temporal-code tranche."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def _module():
    path = PROJECT_DIR / "86_freeze_temporal_code_broad_tranche.py"
    spec = importlib.util.spec_from_file_location("temporal_code_broad_tranche", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _module()
    assert module._quantile_indices(10, 4) == [0, 3, 6, 9]
    path = PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_broad_tranche_plan.json"
    if not path.exists():
        print("[temporal-code-broad-tranche] generated plan absent; quantile contract pass")
        return 0
    plan = load_json(path)
    assert plan["schema_version"] == "temporal-code-broad-tranche-plan-v1"
    assert plan["status"] == "frozen_before_tranche_content_fetch"
    assert plan["summary"]["repository_count"] == 20
    assert plan["summary"]["split_counts"] == {"train": 12, "development": 4, "confirmatory": 4}
    assert plan["content_fetch_limits"]["maximum_pull_requests_per_repository"] == 2
    forbidden = set(plan["contract"]["selection_forbids"])
    assert {"Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(forbidden)
    assert plan["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-broad-tranche] deterministic quantile selection and no-leak contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
