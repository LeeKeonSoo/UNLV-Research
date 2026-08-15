#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.runtime_paths import BenchmarkWorkerPaths
from external_evaluation.evalplus_generator import benchmark_root, resolve_model_run
from external_evaluation.official_suite_generator import cruxeval_data_path, ds1000_data_path


def main() -> int:
    environment = {
        "UNLV_TRAINING_OUTPUT_ROOT": "/content/drive/MyDrive/unlv/training",
        "UNLV_BENCHMARK_OUTPUT_ROOT": "/content/drive/MyDrive/unlv/results",
        "UNLV_MODEL_SNAPSHOT_PATH": "/content/models/qwen3-4b",
        "UNLV_INPUT_REPORT_PATH": "/content/drive/MyDrive/unlv/training_inputs_report.json",
        "UNLV_THIRD_PARTY_ROOT": "/content/third_party",
        "HF_HUB_CACHE": "/content/hf_cache/hub",
        "HF_DATASETS_CACHE": "/content/hf_cache/datasets",
        "BIGCODEBENCH_DATA_ROOT": "/content/benchmarks/bigcodebench",
    }

    paths = BenchmarkWorkerPaths.from_environment(environment)

    assert paths.training_output_root(Path("D:/fallback")) == Path(
        "/content/drive/MyDrive/unlv/training"
    )
    assert paths.benchmark_root(Path("D:/fallback")) == Path(
        "/content/drive/MyDrive/unlv/results"
    )
    assert paths.model_snapshot(Path("D:/fallback/model")) == Path(
        "/content/models/qwen3-4b"
    )
    assert paths.input_report == Path(
        "/content/drive/MyDrive/unlv/training_inputs_report.json"
    )
    assert paths.third_party_root == Path("/content/third_party")
    assert paths.hf_hub_root == Path("/content/hf_cache/hub")
    assert paths.hf_datasets_root == Path("/content/hf_cache/datasets")
    assert paths.bigcodebench_data_root == Path("/content/benchmarks/bigcodebench")
    protocol = {
        "training": {
            "arms": ["normal_natural"],
            "seeds": [101],
            "output_root": "D:/fallback",
        }
    }
    report_data = {"arms": {"normal_natural": {"optimizer_steps": 373}}}
    with patch.dict(os.environ, environment, clear=False):
        resolved = resolve_model_run(protocol, report_data, "normal_natural", 101)
        assert resolved["adapter_path"] == Path(
            "/content/drive/MyDrive/unlv/training/qlora_runs/"
            "normal_natural_seed101_steps373"
        )
        assert benchmark_root(protocol) == Path(
            "/content/drive/MyDrive/unlv/results"
        )
        assert cruxeval_data_path() == Path(
            "/content/third_party/cruxeval/data/cruxeval.jsonl"
        )
        assert ds1000_data_path() == Path(
            "/content/third_party/DS-1000/data/ds1000.jsonl.gz"
        )
    print("[benchmark-worker-paths] Colab overrides: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
