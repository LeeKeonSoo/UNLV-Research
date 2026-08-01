#!/usr/bin/env python3
"""Materialize the record/text-disjoint external-validation integrity gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation_artifacts import save_json
from external_evaluation.validation_integrity import build_validation_integrity_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen-policy, record/text-disjoint external evaluation inputs."
    )
    parser.add_argument("--development-curation-report", required=True, type=Path)
    parser.add_argument("--confirmatory-curation-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_validation_integrity_report(
        development_curation_report=args.development_curation_report,
        confirmatory_curation_report=args.confirmatory_curation_report,
    )
    save_json(args.output, report)
    print(json.dumps({"status": report["status"], "blocking_reasons": report["blocking_reasons"]}, ensure_ascii=False))
    return 0 if report["status"] == "confirmatory_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
