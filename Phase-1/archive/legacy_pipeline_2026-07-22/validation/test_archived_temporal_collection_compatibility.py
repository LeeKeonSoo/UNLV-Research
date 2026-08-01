#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


MODULES = (
    ("121_freeze_temporal_code_forward_collection_schedule", "archive.temporal_code.forward_collection_schedule", "freeze"),
    ("122_collect_temporal_code_forward_snapshot_shard", "archive.temporal_code.forward_snapshot_shard", "collect"),
    ("123_build_temporal_code_forward_candidate_ledger", "archive.temporal_code.forward_candidate_ledger", "build"),
    ("124_build_temporal_code_forward_operations_status", "archive.temporal_code.forward_operations_status", "build"),
    ("125_run_temporal_code_forward_operations", "archive.temporal_code.forward_operations", "collect"),
    ("126_freeze_temporal_code_forward_recipe_batch", "archive.temporal_code.forward_recipe_batch", "freeze"),
    ("127_freeze_temporal_code_retrospective_development_schedule", "archive.temporal_code.retrospective_development_schedule", "freeze"),
    ("128_collect_temporal_code_retrospective_shard", "archive.temporal_code.retrospective_shard", "collect"),
    ("129_run_temporal_code_retrospective_collection", "archive.temporal_code.retrospective_collection", "main"),
    ("130_build_temporal_code_retrospective_development_report", "archive.temporal_code.retrospective_development_report", "build"),
    ("131_freeze_temporal_code_retrospective_expansion_schedule", "archive.temporal_code.retrospective_expansion_schedule", "freeze"),
    ("132_build_temporal_code_retrospective_combined_ledger", "archive.temporal_code.retrospective_combined_ledger", "build"),
    ("133_build_temporal_code_retrospective_operations_status", "archive.temporal_code.retrospective_operations_status", "build"),
    ("134_freeze_temporal_code_retrospective_execution_order", "archive.temporal_code.retrospective_execution_order", "build"),
    ("135_build_temporal_code_retrospective_e2_capacity_audit", "archive.temporal_code.retrospective_e2_capacity_audit", "build"),
)


def main() -> int:
    for wrapper_name, archive_name, callable_name in MODULES:
        wrapper = importlib.import_module(wrapper_name)
        archived = importlib.import_module(archive_name)
        assert getattr(wrapper, callable_name) is getattr(archived, callable_name)
        assert wrapper.main is archived.main
    print("[archived-temporal-collection] root compatibility wrappers: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
