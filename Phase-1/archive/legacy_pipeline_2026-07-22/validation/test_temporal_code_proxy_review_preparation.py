#!/usr/bin/env python3
"""Regression checks for review-only temporal-code content preparation."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    module = importlib.import_module("81_prepare_temporal_code_proxy_review_expansion")
    fixture = load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_change_bundles.json")["bundles"][0]
    fixture["execution_validation"]["test_command_verified"] = False
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle_dir = root / "bundles"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.json").write_text(json.dumps(fixture), encoding="utf-8")
        primary = root / "primary.jsonl"
        primary.write_text("", encoding="utf-8")
        report = module.prepare(bundle_dir, primary, root / "out")
    assert report["summary"]["review_only_stage_a_pass_chunks"] > 0, report
    assert report["review_only_boundary"]["training_approval"] is False, report
    assert report["review_only_boundary"]["stage0_release_candidate"] is False, report
    assert report["review_only_boundary"]["test_command_verified"] is False, report
    print("[temporal-code-proxy-review-preparation] review chunks prepared: pass")
    print("[temporal-code-proxy-review-preparation] training and Stage-0 release remain forbidden: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
