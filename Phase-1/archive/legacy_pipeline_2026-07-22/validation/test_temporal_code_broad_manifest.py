#!/usr/bin/env python3
"""Contract checks for a generated broad temporal-code repository manifest."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    path = PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
    if not path.exists():
        print("[temporal-code-broad-manifest] generated manifest absent; contract test skipped")
        return 0
    manifest = load_json(path)
    assert manifest["schema_version"] == "temporal-code-broad-repository-manifest-v1"
    assert manifest["status"] == "frozen_before_broad_content_fetch"
    assert manifest["summary"]["frozen_repository_count"] == len(manifest["repositories"])
    assert all(manifest["summary"]["split_counts"].values())
    assert all(row["membership_is_training_approval"] is False for row in manifest["repositories"].values())
    assert manifest["freeze_contract"]["content_fetch_limits"]["issue_and_pull_request_prose"] == (
        "do_not_fetch_for_training_payload"
    )
    assert manifest["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-broad-manifest] frozen eligibility, split, and no-training-approval contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
