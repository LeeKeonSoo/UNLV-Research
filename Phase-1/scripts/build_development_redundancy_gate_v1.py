#!/usr/bin/env python3
# Run: python scripts/build_development_redundancy_gate_v1.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_redundancy_gate import build_development_redundancy_gate
from development_redundancy_gate_contract import RedundancyGateStatus, load_redundancy_gate_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build E2 empirical Redundancy evidence from the admitted development matrix.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "protocols" / "development_redundancy_gate_registry_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "development_redundancy_gate_report_v1.json",
    )
    args = parser.parse_args()
    report = build_development_redundancy_gate(load_redundancy_gate_registry(args.registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    print(
        f"[development-redundancy-gate-v1] status={report.status.value} "
        f"exact_families={report.recovered_exact_family_count}/{report.expected_exact_family_count} "
        f"clean_false_merges={report.clean_false_merged_record_count} "
        f"perturbation_safe_merges={report.perturbation_safe_merge_count}"
    )
    return 0 if report.status is RedundancyGateStatus.PASSED else 2


if __name__ == "__main__":
    raise SystemExit(main())
