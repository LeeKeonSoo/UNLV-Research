#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_block4_acquisition_report.json"


def _first_record(path: Path) -> dict[str, str | int | dict[str, str | int]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def main() -> int:
    report = load_json(REPORT_PATH)
    assert report["status"] == "math_domain_block4_acquisition_pools_ready"
    assert "Stage-C validation only" in report["utility_scope"]
    assert report["stage_materialization_status"] == "pending_after_acquisition"

    pools = report["pools"]
    raw = pools["raw_mixed_pool"]
    reference = pools["known_high_quality_reference_pool"]
    assert raw["records"] == 512
    assert reference["records"] == 512
    assert raw["token_proxy"] > 10000
    assert reference["token_proxy"] > 10000

    raw_path = Path(raw["path"])
    reference_path = Path(reference["path"])
    assert raw_path.exists()
    assert reference_path.exists()

    raw_record = _first_record(raw_path)
    reference_record = _first_record(reference_path)
    assert raw_record["domain"] == "math"
    assert reference_record["domain"] == "math"
    assert raw_record["source_dataset_id"] != "openai/gsm8k"
    assert reference_record["source_dataset_id"] != "hendrycks/competition_math"
    assert "text" in raw_record
    assert "text" in reference_record

    quarantine = report["stage_c_benchmark_quarantine"]
    assert {item["name"] for item in quarantine} == {"GSM8K", "MATH"}

    print("[math-domain-block4-acquisition] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
