#!/usr/bin/env python3
# Run: python scripts/build_development_quality_gate_v1.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_quality_gate import build_development_quality_gate
from development_quality_gate_contract import QualityGateStatus, load_quality_gate_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the E3 empirical Quality evidence gate.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "protocols" / "development_quality_gate_registry_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "development_quality_gate_report_v1.json",
    )
    args = parser.parse_args()
    report = build_development_quality_gate(load_quality_gate_registry(args.registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    print(
        f"[development-quality-gate-v1] status={report.status.value} "
        f"routes={sum(item.route_evidence_ready for item in report.routes)}/{len(report.routes)} "
        f"calibrated={sum(item.calibration_passed for item in report.routes)}/{len(report.routes)} "
        f"blockers={len(report.blocker_codes)}"
    )
    return 0 if report.status is QualityGateStatus.PASSED else 2


if __name__ == "__main__":
    raise SystemExit(main())
