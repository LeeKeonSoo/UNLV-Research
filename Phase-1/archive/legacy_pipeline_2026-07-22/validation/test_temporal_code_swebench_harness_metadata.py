#!/usr/bin/env python3
"""Contract checks for outcome-free SWE-bench harness metadata profiling."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _module():
    path = PROJECT_DIR / "104_acquire_swebench_harness_metadata.py"
    spec = importlib.util.spec_from_file_location("swebench_harness_metadata", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _module()
    rule = {
        "development_repository_buckets": [0, 49],
        "confirmatory_repository_buckets": [50, 99],
        "development_task_end_inclusive": "2022-12-31T23:59:59Z",
        "confirmatory_task_start_inclusive": "2023-01-01T00:00:00Z",
    }
    before = datetime(2022, 1, 1, tzinfo=timezone.utc)
    after = datetime(2024, 1, 1, tzinfo=timezone.utc)
    identity = "fixture/repository"
    bucket = module._bucket(identity)
    expected_before = "development" if bucket <= 49 else "excluded_split_or_time_rule"
    expected_after = "confirmatory" if bucket >= 50 else "excluded_split_or_time_rule"
    assert module._assign_split(identity, before, rule) == expected_before
    assert module._assign_split(identity, after, rule) == expected_after
    assert module._assign_split(identity, None, rule) == "excluded_missing_timestamp"
    required = module._required_task_count(
        {
            "sample_size_rule": {
                "desired_task_distribution_half_width": 0.05,
                "conservative_variance_bound_for_paired_difference": 1.0,
                "practical_effect_margin_absolute": 0.05,
                "training_seed_count": 5,
            }
        }
    )
    assert required["required_task_count"] == 1083, required
    print("[temporal-code-swebench-metadata] outcome-free split and precision contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
