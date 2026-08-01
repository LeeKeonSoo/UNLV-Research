#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


MODULES = (
    ("78_build_temporal_code_stage_b_blind_review", "archive.temporal_code.stage_b_blind_review", "build"),
    ("79_analyze_temporal_code_stage_b_blind_review", "archive.temporal_code.stage_b_blind_review_analysis", "analyze"),
    ("80_freeze_temporal_code_proxy_review_expansion", "archive.temporal_code.proxy_review_expansion_freeze", "freeze"),
    ("81_prepare_temporal_code_proxy_review_expansion", "archive.temporal_code.proxy_review_expansion_prepare", "prepare"),
    ("82_build_temporal_code_stage_b_multi_reviewer_packets", "archive.temporal_code.stage_b_multi_reviewer_packets", "build"),
    ("83_analyze_temporal_code_stage_b_multi_review", "archive.temporal_code.stage_b_multi_review_analysis", "analyze"),
    ("84_manage_temporal_code_stage_b_review", "archive.temporal_code.stage_b_review_management", "status"),
)


def main() -> int:
    for wrapper_name, archive_name, callable_name in MODULES:
        wrapper = importlib.import_module(wrapper_name)
        archived = importlib.import_module(archive_name)
        assert getattr(wrapper, callable_name) is getattr(archived, callable_name)
        assert wrapper.main is archived.main
    print("[archived-temporal-review] root compatibility wrappers: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
