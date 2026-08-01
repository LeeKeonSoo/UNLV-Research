#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


BLOCK_REPORT = PROJECT_DIR / "outputs" / "validation" / "math_domain_natural_budget_v3_blocks_report.json"
PLAN_PATH = PROJECT_DIR / "configs" / "math_domain_natural_budget_v3_protocol_qwen3_4b.json"


def main() -> int:
    subprocess.run(
        [sys.executable, "216_prepare_math_selector_v3_natural_budget_blocks.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    report = load_json(BLOCK_REPORT)
    plan = load_json(PLAN_PATH)
    blocks = report["blocks"]
    assert report["status"] == "math_natural_budget_v3_blocks_frozen"
    assert blocks["raw_full_natural"]["tokens_in_blocks"] > blocks["curated_math_v3_natural"]["tokens_in_blocks"]
    assert blocks["curated_math_v3_natural"]["tokens_in_blocks"] > blocks["curated_math_v2_natural"]["tokens_in_blocks"]
    assert plan["confirmatory_training_recipe"]["natural_budget_blocks_manifest"]["sha256"] == report["sha256"]
    for arm, row in blocks.items():
        expected_steps = math.ceil(int(row["blocks"]) / int(report["gradient_accumulation_steps"]))
        assert plan["confirmatory_training_recipe"]["optimizer_steps_by_arm"][arm] == expected_steps
    assert report["stage_c_outcomes_read"] is False
    print("[math-domain-natural-budget-v3-blocks] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
