#!/usr/bin/env python3
"""Execute labeled Core-policy fixtures against the active A/B/C runtime."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.core_behavior_contracts import CORE_DIMENSIONS, behavior_invariants
from validation.core_behavior_executors import execute_case


JsonMap = dict[str, Any]


def _matrix_reconciliation(registry: JsonMap) -> JsonMap:
    matrix = json.loads(
        (ROOT / "validation" / "fixtures" / "core_case_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    registry_cores = {policy["id"]: policy["core"] for policy in registry["policies"]}
    mismatches = [
        {
            "case_id": case["id"],
            "policy_id": policy_id,
            "matrix_core": case["core"],
            "registry_core": registry_cores.get(policy_id),
        }
        for case in matrix["cases"]
        for policy_id in case["policy_ids"]
        if registry_cores.get(policy_id) != case["core"]
    ]
    return {"passed": not mismatches, "mismatches": mismatches}


def _outcome(expected: bool, observed: bool) -> str:
    if expected:
        return "true_positive" if observed else "false_negative"
    return "false_positive" if observed else "true_negative"


def _policy_report(
    counts_by_policy: dict[str, Counter[str]], registry_policies: dict[str, JsonMap]
) -> JsonMap:
    return {
        policy_id: {
            "core": registry_policies[policy_id]["core"],
            "status": registry_policies[policy_id]["status"],
            "true_positives": counts["true_positive"],
            "false_positives": counts["false_positive"],
            "false_negatives": counts["false_negative"],
            "true_negatives": counts["true_negative"],
            "positive_cases": counts["true_positive"] + counts["false_negative"],
            "negative_cases": counts["true_negative"] + counts["false_positive"],
            "behavior_invariant_failures": counts["invariant_failure"],
        }
        for policy_id, counts in sorted(counts_by_policy.items())
    }


def _core_report(counts_by_core: dict[str, Counter[str]]) -> JsonMap:
    return {
        core: {
            "required_dimensions": CORE_DIMENSIONS[core],
            "true_positives": counts["true_positive"],
            "false_positives": counts["false_positive"],
            "false_negatives": counts["false_negative"],
            "true_negatives": counts["true_negative"],
            "behavior_invariant_failures": counts["invariant_failure"],
            "behavior_gate_passed": counts["false_positive"] == 0
            and counts["false_negative"] == 0
            and counts["invariant_failure"] == 0
            and counts["true_positive"] >= 1
            and counts["true_negative"] >= 1,
        }
        for core, counts in sorted(counts_by_core.items())
    }


def build_audit(cases_path: Path) -> JsonMap:
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "core-behavior-audit-v3-fixtures":
        raise ValueError("Unexpected Core behavior fixture schema.")
    registry = json.loads(
        (ROOT / "configs" / "core_policy_registry.json").read_text(encoding="utf-8")
    )
    registry_policies = {policy["id"]: policy for policy in registry["policies"]}
    policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    core_counts: dict[str, Counter[str]] = defaultdict(Counter)
    case_results: list[JsonMap] = []
    for case in fixture["cases"]:
        policy = registry_policies[case["policy_id"]]
        core = policy["core"]
        expected = bool(case["expected_trigger"])
        event = execute_case(case)
        outcome = _outcome(expected, bool(event["triggered"]))
        invariant = behavior_invariants(core, expected, event)
        policy_counts[case["policy_id"]][outcome] += 1
        policy_counts[case["policy_id"]]["invariant_failure"] += not invariant["passed"]
        core_counts[core][outcome] += 1
        core_counts[core]["invariant_failure"] += not invariant["passed"]
        case_results.append(
            {
                "id": case["id"],
                "policy_id": case["policy_id"],
                "policy_status": policy["status"],
                "core": core,
                "fixture_kind": case.get(
                    "fixture_kind",
                    "labeled_positive" if expected else "false_positive_guard",
                ),
                "expected_code": case["expected_code"],
                "expected_trigger": expected,
                "observed_trigger": bool(event["triggered"]),
                "observed_action": event["action"],
                "outcome": outcome,
                "behavior_checks": invariant["checks"],
                "behavior_invariants_passed": invariant["passed"],
            }
        )
    policies = _policy_report(policy_counts, registry_policies)
    cores = _core_report(core_counts)
    matrix_reconciliation = _matrix_reconciliation(registry)
    return {
        "schema_version": "core-behavior-audit-v3",
        "claim_boundary": (
            "Executable labeled, false-positive, metamorphic, and adversarial fixture "
            "behavior only; this is not an estimate of corpus-wide precision or recall."
        ),
        "fixture_path": str(cases_path),
        "summary": {
            "cases": len(case_results),
            "true_positives": sum(item["true_positives"] for item in policies.values()),
            "false_positives": sum(item["false_positives"] for item in policies.values()),
            "false_negatives": sum(item["false_negatives"] for item in policies.values()),
            "true_negatives": sum(item["true_negatives"] for item in policies.values()),
            "behavior_invariant_failures": sum(
                item["behavior_invariant_failures"] for item in policies.values()
            ),
            "core_behavior_gates_passed": all(
                item["behavior_gate_passed"] for item in cores.values()
            ),
            "case_matrix_reconciled": matrix_reconciliation["passed"],
        },
        "cores": cores,
        "policies": policies,
        "case_matrix_reconciliation": matrix_reconciliation,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a labeled Core behavior audit.")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_audit(args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
