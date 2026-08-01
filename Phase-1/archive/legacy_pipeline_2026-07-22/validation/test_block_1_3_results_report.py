from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "validation" / "block_1_3_results_report.json"


def test_block_1_math_v2_reports_curated_failure() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    block_1 = report["block_1_math_v2"]

    assert block_1["decision"] == "math_v2_failed_curated_worse_than_raw"
    assert block_1["raw_minus_curated_nll"] < 0
    assert len(block_1["per_seed"]["raw_full_natural"]) == 3
    assert len(block_1["per_seed"]["curated_math_v2_natural"]) == 3


def test_integrated_table_keeps_historical_code_and_math_failures() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    results = {row["domain_run"]: row["result"] for row in report["block_2_integrated_table"]}

    assert results["Code v2 natural-budget 3-seed NLL"] == "historical_positive_rerun_required"
    assert results["Math v1 natural-budget seed101 NLL"] == "fail"
    assert results["Math v2 natural-budget 3-seed NLL"] == "fail"


def main() -> int:
    test_block_1_math_v2_reports_curated_failure()
    test_integrated_table_keeps_historical_code_and_math_failures()
    print("[block-1-3-results] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
