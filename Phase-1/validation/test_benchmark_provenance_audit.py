from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.benchmark_provenance_audit import (
    audit_bigcodebench_cell,
    audit_cruxeval_cell,
    audit_ds1000_cell,
    audit_evalplus_cell,
)


def test_auditors_recompute_scores_from_task_level_evidence() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        evalplus_samples = root / "humaneval.jsonl"
        evalplus_samples.write_text(
            '\n'.join(
                json.dumps({"task_id": task_id, "solution": "pass"})
                for task_id in ("HumanEval/0", "HumanEval/1")
            )
            + '\n',
            encoding="utf-8",
        )
        evalplus_results = root / "humaneval_eval_results.json"
        evalplus_results.write_text(
            json.dumps(
                {
                    "eval": {
                        "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
                        "HumanEval/1": [{"base_status": "fail", "plus_status": "fail"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        evalplus = audit_evalplus_cell(
            evalplus_samples, evalplus_results, expected_count=2
        )

        bcb_samples = root / "bcb.jsonl"
        bcb_samples.write_text(
            '\n'.join(
                json.dumps({"task_id": task_id, "solution": "pass"})
                for task_id in ("BigCodeBench/0", "BigCodeBench/1")
            )
            + '\n',
            encoding="utf-8",
        )
        bcb_results = root / "bcb_eval_results.json"
        bcb_results.write_text(
            json.dumps(
                {
                    "eval": {
                        "BigCodeBench/0": [{"status": "pass"}],
                        "BigCodeBench/1": [{"status": "fail"}],
                    }
                }
            ),
            encoding="utf-8",
        )
        bcb_pass = root / "bcb_pass.json"
        bcb_pass.write_text(json.dumps({"pass@1": 0.5}), encoding="utf-8")
        bcb = audit_bigcodebench_cell(
            bcb_samples, bcb_results, bcb_pass, expected_count=2
        )

        crux_samples = root / "crux.json"
        crux_samples.write_text(json.dumps({"sample_0": ["x"]}), encoding="utf-8")
        crux_results = root / "crux_results.json"
        crux_results.write_text(
            json.dumps(
                {
                    "raw_generations": {"sample_0": ["x"]},
                    "raw_scored_generations": {"sample_0": [True]},
                    "pass_at_1": 100.0,
                }
            ),
            encoding="utf-8",
        )
        crux = audit_cruxeval_cell(crux_samples, crux_results, expected_count=1)

        ds_samples = root / "ds.jsonl"
        ds_samples.write_text('{"code":"x"}\n{"code":"y"}\n', encoding="utf-8")
        ds_details = root / "ds.json"
        ds_details.write_text(
            json.dumps(
                [
                    {"problem_id": 0, "passed": True, "status": "passed"},
                    {"problem_id": 1, "passed": False, "status": "failed_exit_1"},
                ]
            ),
            encoding="utf-8",
        )
        ds_summary = root / "ds.txt"
        ds_summary.write_text("count  2.000\nmean      0.500000\n", encoding="utf-8")
        ds = audit_ds1000_cell(
            ds_samples, ds_details, ds_summary, expected_count=2
        )

    assert [evalplus.score_percent, bcb.score_percent, crux.score_percent, ds.score_percent] == [
        50.0,
        50.0,
        100.0,
        50.0,
    ]
    assert all(result.verified for result in (evalplus, bcb, crux, ds))


if __name__ == "__main__":
    test_auditors_recompute_scores_from_task_level_evidence()
    print("Benchmark provenance audit contract passed")
