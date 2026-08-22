#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.evalplus_generator import (
    ARMS,
    DATASETS,
    adapter_directory,
    benchmark_root,
    is_complete_output,
    resolve_datasets,
    resolve_model_run,
    trim_completion,
)


def main() -> int:
    assert "framework_curated_natural" in ARMS
    assert "nemo_natural" in ARMS
    assert "random72_matched" in ARMS
    assert "data_juicer_natural" in ARMS
    with TemporaryDirectory() as directory:
        output = Path(directory) / "samples.jsonl"
        output.touch()
        assert is_complete_output(output) is False
        output.write_text(
            '{"task_id":"HumanEval/0","completion":"pass"}\n',
            encoding="utf-8",
        )
        assert is_complete_output(output) is True
    assert adapter_directory(Path("D:/runs"), "normal_natural", 202, 311) == Path(
        "D:/runs/qlora_runs/normal_natural_seed202_steps311"
    )
    protocol = {
        "training": {
            "arms": ["raw_audited_natural", "normal_natural", "hard_natural"],
            "seeds": [101, 202, 303],
            "output_root": "D:/runs",
        }
    }
    report = {"arms": {"normal_natural": {"optimizer_steps": 311}}}
    resolved = resolve_model_run(protocol, report, "normal_natural", 202)
    assert resolved["adapter_path"] == Path("D:/runs/qlora_runs/normal_natural_seed202_steps311")
    assert benchmark_root(protocol) == Path("D:/runs/benchmarks_v1")
    assert resolve_datasets(None) == DATASETS
    assert resolve_datasets("HumanEval+") == ("HumanEval+",)
    assert resolve_datasets("MBPP+") == ("MBPP+",)
    assert trim_completion("```python\nreturn x\n```\n# Task next") == "return x\n"
    print("[active-evalplus-generator] pure contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
