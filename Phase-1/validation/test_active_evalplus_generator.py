#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.evalplus_generator import (
    DATASETS,
    adapter_directory,
    benchmark_root,
    resolve_datasets,
    resolve_model_run,
    trim_completion,
)


def main() -> int:
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
