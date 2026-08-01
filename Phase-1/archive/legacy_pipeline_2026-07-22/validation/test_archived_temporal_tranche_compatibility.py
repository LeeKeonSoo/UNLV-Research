#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


MODULES = (
    ("85_freeze_temporal_code_broad_manifest", "archive.temporal_code.broad_manifest", "freeze"),
    ("86_freeze_temporal_code_broad_tranche", "archive.temporal_code.broad_tranche", "_quantile_indices"),
    ("87_freeze_temporal_code_broad_test_commands", "archive.temporal_code.broad_test_commands", "_bundle_paths"),
    ("88_build_temporal_code_broad_tranche_report", "archive.temporal_code.broad_tranche_report", "build"),
    ("89_run_temporal_code_broad_stage_b_ablations", "archive.temporal_code.broad_stage_b_ablations", "run"),
    ("90_collect_temporal_code_pr_path_metadata", "archive.temporal_code.pr_path_metadata", "classify_changed_paths"),
    ("91_freeze_temporal_code_path_stratified_tranche", "archive.temporal_code.path_stratified_tranche", "_quantile_indices"),
    ("92_freeze_temporal_code_confirmatory_execution_expansion", "archive.temporal_code.confirmatory_execution_expansion", "freeze"),
    ("93_prepare_temporal_code_stage_c_smoke", "archive.temporal_code.stage_c_smoke_prepare", "prepare"),
    ("95_build_temporal_code_stage_c_smoke_report", "archive.temporal_code.stage_c_smoke_report", "build"),
    ("96_freeze_temporal_code_development_execution_expansion", "archive.temporal_code.development_execution_expansion", "freeze"),
    ("97_build_temporal_code_development_expansion_report", "archive.temporal_code.development_expansion_report", "build"),
    ("98_freeze_temporal_code_native_execution_recipes", "archive.temporal_code.native_execution_recipes", "freeze"),
    ("99_build_temporal_code_native_execution_report", "archive.temporal_code.native_execution_report", "build"),
)


def main() -> int:
    for wrapper_name, archive_name, callable_name in MODULES:
        wrapper = importlib.import_module(wrapper_name)
        archived = importlib.import_module(archive_name)
        assert getattr(wrapper, callable_name) is getattr(archived, callable_name)
        assert wrapper.main is archived.main
    print("[archived-temporal-tranche] root compatibility wrappers: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
