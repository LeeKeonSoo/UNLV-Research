#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "validation" / "math_domain_selector_v3_materialization_report.json"


def main() -> int:
    subprocess.run([sys.executable, "214_materialize_math_selector_v3.py"], cwd=ROOT, check=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "math_selector_v3_materialized"
    assert report["utility_outcomes_read"] is False
    assert report["v2_failure_preserved"] is True
    assert report["preoutcome_mapping_complete"] is True
    assert report["v3_ready_for_stage_c_freeze"] is True
    assert report["blockers"] == []

    arms = report["arms"]
    assert arms["curated_math_v3_natural"]["token_proxy_count"] > arms["curated_math_v2_natural"]["token_proxy_count"]
    assert arms["curated_math_v3_natural"]["token_proxy_count"] < arms["raw_full_natural"]["token_proxy_count"]

    proof_v2 = arms["curated_math_v2_natural"]["style_token_counts"]["proof_or_theorem"]
    proof_v3 = arms["curated_math_v3_natural"]["style_token_counts"]["proof_or_theorem"]
    assert proof_v3 > proof_v2
    assert report["retention_checks"]["proof_or_theorem_token_retention_fraction"] >= 0.95
    assert report["retention_checks"]["stage_a_token_retention_fraction"] >= 0.9

    print("[math-selector-v3] materialization contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
