#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "benchmark_sensitivity_protocol_report.json"


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "230_build_benchmark_sensitivity_protocol.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    assert completed.returncode == 0
    import json

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "benchmark_sensitivity_protocol_frozen_external_transfer_inconclusive"
    assert not report["blockers"]
    assert report["frozen_outcome_isolation"]["stage_b_policy_change_permitted"] is False
    assert report["equal_token_primary"]["minimum_paired_training_seeds"] == 5
    assert report["equal_token_primary"]["paired_ci"]["method"] == "paired_bootstrap_percentile"
    assert report["external_transfer"]["current_status"] == "completed_multiseed_external_transfer_inconclusive"
    assert report["external_transfer"]["claim"] == "external_transfer_not_demonstrated_on_frozen_livecodebench_confirmation"
    assert report["external_transfer"]["sample_size_rule"]["minimum_non_easy_tasks"] == 200
    print("[benchmark-sensitivity-protocol] frozen statistical and external-transfer boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
