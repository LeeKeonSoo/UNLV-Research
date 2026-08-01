#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_code_benchmark_snapshots import (
    build_snapshot,
    cruxeval_tasks,
    ds1000_tasks,
    livecodebench_tasks,
)


def main() -> int:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        lcb = root / "test.jsonl"
        lcb.write_text(json.dumps({"question_id": "lcb/1", "question_content": "Solve this problem.", "starter_code": "def solve():", "public_test_cases": [{"input": "1"}], "private_test_cases": [{"output": "2"}]}) + "\n", encoding="utf-8")
        crux = root / "crux.jsonl"
        crux.write_text(json.dumps({"id": "crux/1", "code": "def f(x): return x + 1", "input": [1], "output": 2}) + "\n", encoding="utf-8")
        ds = root / "ds.jsonl.gz"
        with gzip.open(ds, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"prompt": "import numpy as np\n# task", "reference_code": "np.array([1])", "code_context": "numpy", "metadata": {"problem_id": "ds/1"}}) + "\n")

        lcb_snapshot = build_snapshot("livecodebench_code_generation_lite", "fixture-lcb", livecodebench_tasks([lcb]))
        crux_snapshot = build_snapshot("cruxeval_input_prediction", "fixture-crux", cruxeval_tasks(crux))
        ds_snapshot = build_snapshot("ds1000", "fixture-ds", ds1000_tasks(ds))

        assert lcb_snapshot["tasks"][0]["task_id"] == "lcb/1"
        assert "Solve this problem." in lcb_snapshot["tasks"][0]["prompt"]
        assert "output" not in lcb_snapshot["tasks"][0]["test"]
        assert crux_snapshot["tasks"][0]["task_id"] == "crux/1"
        assert "def f" in crux_snapshot["tasks"][0]["code"]
        assert ds_snapshot["tasks"][0]["task_id"] == "ds/1"
        assert ds_snapshot["tasks"][0]["canonical_solution"] == "np.array([1])"
    print("[code-benchmark-snapshots] official-source normalization fixture: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
