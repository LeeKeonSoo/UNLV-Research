#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.collect_confirmatory_benchmark_results import (
    PRIMARY_REASONING_BENCHMARKS,
    parse_bigcodebench_percent,
    parse_cruxeval_percent,
    parse_ds1000_percent,
    parse_evalplus_percent,
    selected_model_arms,
    summarize_rows,
)


def test_official_result_parsers_normalize_every_score_to_percent() -> None:
    # Given: minimal outputs matching the four official evaluator schemas.
    with TemporaryDirectory() as directory:
        root = Path(directory)
        evalplus = root / "evalplus.json"
        evalplus.write_text(
            json.dumps(
                {
                    "eval": {
                        "task-1": [{"base_status": "pass", "plus_status": "pass"}],
                        "task-2": [{"base_status": "pass", "plus_status": "fail"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        bigcodebench = root / "bigcodebench.json"
        bigcodebench.write_text(json.dumps({"pass@1": 0.375}), encoding="utf-8")
        cruxeval = root / "cruxeval.json"
        cruxeval.write_text(json.dumps({"pass_at_1": 42.5}), encoding="utf-8")
        ds1000 = root / "ds1000.txt"
        ds1000.write_text("      score\ncount  1000.000\nmean      0.125\n", encoding="utf-8")

        # When/Then: every parser reports the benchmark score on one percent scale.
        assert parse_evalplus_percent(evalplus) == 50.0
        assert parse_bigcodebench_percent(bigcodebench) == 37.5
        assert parse_cruxeval_percent(cruxeval) == 42.5
        assert parse_ds1000_percent(ds1000) == 12.5


def test_evalplus_parser_requires_base_and_plus_tests_to_pass() -> None:
    # Given: one fully passing task and one task that passes only the Plus tests.
    with TemporaryDirectory() as directory:
        result = Path(directory) / "evalplus.json"
        result.write_text(
            json.dumps(
                {
                    "eval": {
                        "task-1": [{"base_status": "pass", "plus_status": "pass"}],
                        "task-2": [{"base_status": "fail", "plus_status": "pass"}],
                    }
                }
            ),
            encoding="utf-8",
        )

        # When/Then: the Plus score follows EvalPlus's combined base-and-extra contract.
        assert parse_evalplus_percent(result) == 50.0


def test_two_seed_summary_preserves_rows_and_reports_sample_variation() -> None:
    assert PRIMARY_REASONING_BENCHMARKS == (
        "BigCodeBench Complete",
        "CRUXEval-I",
        "CRUXEval-O",
        "DS-1000",
    )
    assert selected_model_arms((101, 202)) == (
        ("base_no_update", None),
        ("raw_audited_natural", 101),
        ("raw_audited_natural", 202),
        ("normal_natural", 101),
        ("normal_natural", 202),
        ("hard_natural", 101),
        ("hard_natural", 202),
    )
    rows = []
    for arm, values in (
        ("raw_audited_natural", (40.0, 44.0)),
        ("normal_natural", (42.0, 46.0)),
        ("hard_natural", (38.0, 50.0)),
    ):
        for seed, value in zip((101, 202), values, strict=True):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "scores_percent": {benchmark: value for benchmark in (
                        "HumanEval+",
                        "MBPP+",
                        "BigCodeBench Complete",
                        "CRUXEval-I",
                        "CRUXEval-O",
                        "DS-1000",
                    )},
                }
            )

    summaries = summarize_rows(rows)
    assert summaries[0]["scores"]["HumanEval+"] == {
        "mean_percent": 42.0,
        "sample_std_percent": 2.828427,
        "seed_count": 2,
    }
    assert summaries[0]["primary_reasoning_macro_percent"] == 42.0


if __name__ == "__main__":
    test_official_result_parsers_normalize_every_score_to_percent()
    test_evalplus_parser_requires_base_and_plus_tests_to_pass()
    test_two_seed_summary_preserves_rows_and_reports_sample_variation()
    print("[confirmatory-benchmark-collector] official score parsing: pass")
