#!/usr/bin/env python3
"""Build the heldout Stage-0 detector benchmark report."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR


DEFAULT_FIXTURES = Path("validation") / "fixtures" / "stage0_detector_heldout_cases.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage0_detector_heldout_benchmark_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage0_detector_heldout_benchmark_report.md"


def _load_validation_builder() -> Any:
    path = Path(__file__).resolve().with_name("170_build_stage0_detector_validation.py")
    spec = importlib.util.spec_from_file_location("stage0_detector_validation_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Build heldout Stage-0 detector benchmark report.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    module = _load_validation_builder()
    report = module.build(args.fixtures, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"], "summary": report["summary"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
