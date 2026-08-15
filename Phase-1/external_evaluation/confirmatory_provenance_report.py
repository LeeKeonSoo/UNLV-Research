"""Build the complete 60-cell confirmatory benchmark provenance report."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path

from external_evaluation.benchmark_provenance_audit import (
    CellAudit,
    audit_bigcodebench_cell,
    audit_cruxeval_cell,
    audit_ds1000_cell,
    audit_evalplus_cell,
)
from external_evaluation.collect_confirmatory_benchmark_results import (
    result_paths,
    selected_model_arms,
)


def _cell_checks(
    root: Path, arm: str, seed: int | None
) -> tuple[tuple[str, Callable[[], CellAudit]], ...]:
    suffix = "base" if seed is None else f"seed{seed}"
    paths = result_paths(root, arm, seed)
    samples = root / "samples"
    suites = samples / "official_suite_samples"
    ds_details = (
        paths["DS-1000"].with_suffix(".details.json")
        if seed is None
        else paths["DS-1000"].with_suffix(".json")
    )
    return (
        ("HumanEval+", lambda: audit_evalplus_cell(samples / "evalplus" / f"humaneval_{arm}_{suffix}.jsonl", paths["HumanEval+"], expected_count=164)),
        ("MBPP+", lambda: audit_evalplus_cell(samples / "evalplus" / f"mbpp_{arm}_{suffix}.jsonl", paths["MBPP+"], expected_count=378)),
        ("BigCodeBench Complete", lambda: audit_bigcodebench_cell(suites / f"bigcodebench_{arm}_{suffix}-sanitized-calibrated.jsonl", suites / f"bigcodebench_{arm}_{suffix}-sanitized-calibrated_eval_results.json", paths["BigCodeBench Complete"], expected_count=1_140)),
        ("CRUXEval-I", lambda: audit_cruxeval_cell(suites / f"cruxeval_input_{arm}_{suffix}.json", paths["CRUXEval-I"], expected_count=800)),
        ("CRUXEval-O", lambda: audit_cruxeval_cell(suites / f"cruxeval_output_{arm}_{suffix}.json", paths["CRUXEval-O"], expected_count=800)),
        ("DS-1000", lambda: audit_ds1000_cell(suites / f"ds1000_{arm}_{suffix}.jsonl", ds_details, paths["DS-1000"], expected_count=1_000)),
    )


def audit_confirmatory_root(root: Path) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    verified = 0
    missing = 0
    for arm, seed in selected_model_arms((101, 202, 303)):
        for benchmark, check in _cell_checks(root, arm, seed):
            try:
                result = check()
            except FileNotFoundError as error:
                cells.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "benchmark": benchmark,
                        "status": "missing",
                        "path": str(error.filename),
                    }
                )
                missing += 1
                continue
            cells.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "benchmark": benchmark,
                    "status": "verified",
                    **asdict(result),
                }
            )
            verified += 1
    return {
        "schema_version": "confirmatory-benchmark-provenance-audit-v1",
        "expected_cells": 60,
        "verified_cells": verified,
        "missing_cells": missing,
        "all_verified": verified == 60,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_confirmatory_root(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: report[key]
        for key in ("expected_cells", "verified_cells", "missing_cells", "all_verified")
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["all_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
