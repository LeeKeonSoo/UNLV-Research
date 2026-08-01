#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


MODULES = (
    ("100_freeze_temporal_code_development_fresh_expansion", "archive.temporal_code.development_fresh_expansion", "freeze"),
    ("101_build_temporal_code_development_fresh_expansion_report", "archive.temporal_code.development_fresh_expansion_report", "build"),
    ("102_build_temporal_code_execution_support_report", "archive.temporal_code.execution_support_report", "build"),
    ("103_freeze_temporal_code_executable_task_harness", "archive.temporal_code.executable_task_harness", "freeze"),
    ("104_acquire_swebench_harness_metadata", "archive.temporal_code.swebench_harness_metadata", "acquire"),
    ("105_prevalidate_evalplus_guardrail", "archive.temporal_code.evalplus_guardrail_prevalidation", "main"),
    ("106_freeze_evalplus_guardrail_split", "archive.temporal_code.evalplus_guardrail_split", "freeze"),
    ("107_freeze_temporal_code_retention_guardrails", "archive.temporal_code.retention_guardrails", "freeze"),
    ("108_build_temporal_primary_source_assessment", "archive.temporal_code.primary_source_assessment", "build"),
    ("109_freeze_temporal_code_forward_e2_acquisition", "archive.temporal_code.forward_e2_acquisition", "freeze"),
    ("110_discover_temporal_code_forward_e2_pilot", "archive.temporal_code.forward_e2_pilot", "discover"),
    ("111_freeze_temporal_code_forward_e2_pilot_recipes", "archive.temporal_code.forward_e2_pilot_recipes", "freeze"),
    ("112_verify_temporal_code_forward_e2_pilot", "archive.temporal_code.forward_e2_pilot_verification", "verify"),
    ("113_build_temporal_code_forward_e2_productivity_report", "archive.temporal_code.forward_e2_productivity_report", "build"),
    ("114_freeze_temporal_code_forward_development_snapshot", "archive.temporal_code.forward_development_snapshot", "freeze"),
    ("115_discover_temporal_code_forward_development_candidates", "archive.temporal_code.forward_development_candidates", "discover"),
    ("116_build_temporal_code_forward_development_snapshot_report", "archive.temporal_code.forward_development_snapshot_report", "build"),
    ("117_freeze_temporal_code_forward_development_accumulation", "archive.temporal_code.forward_development_accumulation", "freeze"),
    ("118_build_temporal_code_forward_capacity_audit", "archive.temporal_code.forward_capacity_audit", "build"),
    ("119_merge_temporal_code_forward_repository_discovery", "archive.temporal_code.forward_repository_discovery", "merge"),
    ("120_build_temporal_code_forward_discovery_capacity_report", "archive.temporal_code.forward_discovery_capacity_report", "build"),
)


def main() -> int:
    for wrapper_name, archive_name, callable_name in MODULES:
        wrapper = importlib.import_module(wrapper_name)
        archived = importlib.import_module(archive_name)
        assert getattr(wrapper, callable_name) is getattr(archived, callable_name)
        assert wrapper.main is archived.main
    wrapper = importlib.import_module("110_discover_temporal_code_forward_e2_pilot")
    archived = importlib.import_module("archive.temporal_code.forward_e2_pilot")
    assert wrapper.Client is archived.Client
    print("[archived-temporal-operations] root compatibility wrappers: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
