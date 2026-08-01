#!/usr/bin/env python3
"""Regression checks for the bounded temporal-code smoke fetch plan contract."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    path = PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_smoke_fetch_plan.json"
    if not path.exists():
        print("[temporal-code-smoke-plan] generated plan absent; contract test skipped")
        return 0
    plan = load_json(path)
    assert plan["schema_version"] == "temporal-code-smoke-fetch-plan-v1"
    assert plan["status"] == "frozen_before_content_fetch"
    assert set(plan["selected_repositories"]) == {"train", "development", "confirmatory"}
    assert plan["content_fetch_limits"]["maximum_pull_requests_per_repository"] <= 2
    assert plan["content_fetch_limits"]["issue_and_pull_request_prose"] == "do_not_fetch_for_training_payload"
    assert plan["frozen_repository_manifest_status"] == "not_frozen_smoke_only"
    assert plan["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-smoke-plan] bounded fetch and non-freeze contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
