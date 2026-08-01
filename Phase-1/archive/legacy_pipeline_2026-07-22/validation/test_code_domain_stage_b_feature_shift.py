#!/usr/bin/env python3
"""Validate selected-vs-budget-not-selected Stage-B diagnostics."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "162_build_code_domain_stage_b_feature_shift_report.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("code_domain_stage_b_feature_shift", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_builder()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "code_domain_stage_b_feature_shift_report.json"
        md_report_path = tmp_path / "code_domain_stage_b_feature_shift_report.md"
        report = module.build(
            ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "train_scored_full_selector.jsonl",
            ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "curated_v2_equal_budget.jsonl",
            ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "stage_b_v2_arms_report.json",
            ROOT / "configs" / "lm_curation_operational_framework_v1.json",
            report_path,
            md_report_path,
        )
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved == report
        assert saved["status"] == "code_domain_stage_b_feature_shift_report_ready"
        assert saved["utility_scope"] == "Stage C validation only; never selector objective"
        assert saved["interpretation"]["not_utility_evidence"] is True
        assert saved["interpretation"]["not_selector_tuning_permission"] is True
        assert saved["interpretation"]["budget_not_selected_is_rejection"] is False
        assert saved["interpretation"]["all_compared_records_remain_in_full_curated_pool"] is True

        pool = saved["summary"]["pool_counts"]
        assert pool["scored_stage_a_pass_records"] > 0
        assert pool["selected_records"] > 0
        assert pool["budget_not_selected_records"] > 0
        assert pool["budget_not_selected_is_rejection"] is False

        signals = saved["summary"]["operational_signal_shifts"]
        for signal in (
            "concise_useful_candidate",
            "concise_test_or_regression_candidate",
            "api_usage_candidate",
            "bugfix_or_regression_test_signal",
            "concise_example_support",
            "template_or_boilerplate_risk",
        ):
            assert signal in signals
            assert "selected_share" in signals[signal]
            assert "budget_not_selected_share" in signals[signal]

        numeric = saved["summary"]["numeric_feature_shifts"]
        assert "code_quality_proxy" in numeric
        assert "soft_redundancy_risk" in numeric
        assert "token_proxy_count" in numeric
        assert md_report_path.exists()
    print("[code-domain-stage-b-feature-shift] selected-vs-budget-not-selected diagnostics: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
