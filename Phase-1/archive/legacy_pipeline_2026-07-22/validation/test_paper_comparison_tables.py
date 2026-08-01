#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    path = ROOT / "197_build_paper_comparison_tables.py"
    spec = importlib.util.spec_from_file_location("paper_comparison_tables", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            tmp_path / "paper_comparison_tables.json",
            tmp_path / "paper_comparison_tables.md",
            tmp_path / "paper_comparison_tables.csv",
        )
    assert report["status"] == "paper_comparison_tables_frozen"
    assert report["claim_boundary"]["utility_scope"] == "Stage C only; never selector objective"
    assert report["summary"]["stage_c_arm_count"] == 4
    assert report["summary"]["ablation_arm_count"] >= 4
    assert report["summary"]["remaining_required_tables"] == []
    assert report["stage_c_arm_table"]["curated_v2_equal_budget"]["mean_nll"] == 1.2016515642017513
    assert report["stage_c_pairwise_table"]["curated_vs_stageA_random"]["mean_nll_reduction"] > 0.005
    assert report["stage_c_pairwise_table"]["curated_vs_raw_random"]["direction_pass"] is True
    assert report["stage_b_ablation_table"]["full_selector"]["selected_chunks"] == 423
    assert report["redundancy_ablation_table"]["binary_current"]["selected_count"] == 1424
    assert report["sources"]["v2_confirmatory_decision"]["exists"] is True
    print("[paper-comparison-tables] frozen comparison tables ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
