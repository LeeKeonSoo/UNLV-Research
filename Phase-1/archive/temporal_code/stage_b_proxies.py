#!/usr/bin/env python3
"""Validate temporal-code Stage-B Core proxies against frozen automated fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import score_stage_b


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = PROJECT_DIR / "validation" / "fixtures" / "temporal_code_stage_b_proxy_cases.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_stage_b_proxy_validation.json"


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def validate(fixtures: Dict[str, Any]) -> Dict[str, Any]:
    records = []
    for row in fixtures["records"]:
        records.append(
            {
                **row,
                "split": "fixture",
                "stage_a_pass": True,
                "bundle_id": "stage-b-proxy-fixtures",
                "repository_identity": "fixture/temporal-code-stage-b",
                "change_type": "modified",
                "chunk_kind": "documentation_paragraph_group" if row["content_type"] == "documentation" else "function",
            }
        )
    scored = score_stage_b(records, quality_weight=0.8, redundancy_weight=0.2)
    by_id = {row["chunk_uid"]: row for row in scored}
    results = []
    for assertion in fixtures["assertions"]:
        kind = assertion["type"]
        details: Dict[str, Any]
        if kind == "quality_pair":
            higher = float(by_id[assertion["higher"]]["stage_b_evidence"]["code_quality_proxy"])
            lower = float(by_id[assertion["lower"]]["stage_b_evidence"]["code_quality_proxy"])
            margin = higher - lower
            details = {"higher": higher, "lower": lower, "observed_margin": round(margin, 6)}
            passed = margin >= float(assertion["minimum_margin"])
        elif kind == "quality_invariance":
            left = float(by_id[assertion["left"]]["stage_b_evidence"]["code_quality_proxy"])
            right = float(by_id[assertion["right"]]["stage_b_evidence"]["code_quality_proxy"])
            delta = abs(left - right)
            details = {"left": left, "right": right, "observed_absolute_delta": round(delta, 6)}
            passed = delta <= float(assertion["maximum_absolute_delta"])
        elif kind == "redundancy_group":
            higher_values = [
                float(by_id[uid]["stage_b_evidence"]["soft_redundancy_risk"])
                for uid in assertion["higher_risk_group"]
            ]
            lower_values = [
                float(by_id[uid]["stage_b_evidence"]["soft_redundancy_risk"])
                for uid in assertion["lower_risk_group"]
            ]
            higher_mean = _mean(higher_values)
            lower_mean = _mean(lower_values)
            margin = higher_mean - lower_mean
            details = {
                "higher_risk_values": higher_values,
                "lower_risk_values": lower_values,
                "higher_risk_mean": round(higher_mean, 6),
                "lower_risk_mean": round(lower_mean, 6),
                "observed_mean_margin": round(margin, 6),
            }
            passed = margin >= float(assertion["minimum_mean_margin"])
        elif kind == "redundancy_pair_floor":
            left = float(by_id[assertion["left"]]["stage_b_evidence"]["soft_redundancy_risk"])
            right = float(by_id[assertion["right"]]["stage_b_evidence"]["soft_redundancy_risk"])
            pair_risk = min(left, right)
            details = {"left_risk": left, "right_risk": right, "observed_pair_risk": round(pair_risk, 6)}
            passed = pair_risk >= float(assertion["minimum_pair_risk"])
        else:
            raise ValueError(f"Unsupported assertion type: {kind}")
        results.append({**assertion, "passed": bool(passed), "details": details})
    passed_count = sum(row["passed"] for row in results)
    return {
        "schema_version": "temporal-code-stage-b-proxy-validation-v1",
        "fixture_schema_version": fixtures["schema_version"],
        "summary": {
            "assertion_count": len(results),
            "passed_count": passed_count,
            "failed_count": len(results) - passed_count,
        },
        "assertions": results,
        "scored_fixture_records": [
            {"chunk_uid": row["chunk_uid"], "stage_b_evidence": row["stage_b_evidence"]}
            for row in scored
        ],
        "forbidden_evidence": fixtures["forbidden_evidence"],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Automated Core-proxy direction checks only. Passing does not establish Utility, "
            "training benefit, cross-repository generality, or release readiness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate temporal-code Stage-B Core proxies.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate(load_json(args.fixtures))
    save_json(args.output, report)
    print(report["summary"])
    return 0 if report["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
