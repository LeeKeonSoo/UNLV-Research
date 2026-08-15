#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Final, Self


DEFAULT_INPUT_REPORT: Final = Path(
    "D:/UNLV-Research/final_all_policy_v1/luna_final_v1/"
    "training_inputs_v1/training_inputs_report.json"
)
DEFAULT_THIRD_PARTY_ROOT: Final = Path("D:/UNLV-Research/third_party")
DEFAULT_HF_HUB_ROOT: Final = Path("D:/UNLV-Research/hf_cache/hub")
DEFAULT_HF_DATASETS_ROOT: Final = Path("D:/UNLV-Research/hf_datasets_cache")


@dataclass(frozen=True, slots=True)
class BenchmarkWorkerPaths:
    """Filesystem contract shared by local and remote benchmark workers."""

    training_output_override: Path | None
    benchmark_output_override: Path | None
    model_snapshot_override: Path | None
    input_report: Path
    third_party_root: Path
    hf_hub_root: Path
    hf_datasets_root: Path
    bigcodebench_data_root: Path | None

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> Self:
        source = os.environ if environment is None else environment

        def optional_path(*names: str) -> Path | None:
            for name in names:
                value = source.get(name)
                if value:
                    return Path(value)
            return None

        return cls(
            training_output_override=optional_path("UNLV_TRAINING_OUTPUT_ROOT"),
            benchmark_output_override=optional_path("UNLV_BENCHMARK_OUTPUT_ROOT"),
            model_snapshot_override=optional_path("UNLV_MODEL_SNAPSHOT_PATH"),
            input_report=optional_path("UNLV_INPUT_REPORT_PATH")
            or DEFAULT_INPUT_REPORT,
            third_party_root=optional_path("UNLV_THIRD_PARTY_ROOT")
            or DEFAULT_THIRD_PARTY_ROOT,
            hf_hub_root=optional_path("UNLV_HF_HUB_ROOT", "HF_HUB_CACHE")
            or DEFAULT_HF_HUB_ROOT,
            hf_datasets_root=optional_path(
                "UNLV_HF_DATASETS_ROOT",
                "HF_DATASETS_CACHE",
            )
            or DEFAULT_HF_DATASETS_ROOT,
            bigcodebench_data_root=optional_path("BIGCODEBENCH_DATA_ROOT"),
        )

    def training_output_root(self, protocol_default: Path) -> Path:
        return self.training_output_override or protocol_default

    def benchmark_root(self, protocol_training_root: Path) -> Path:
        if self.benchmark_output_override is not None:
            return self.benchmark_output_override
        return self.training_output_root(protocol_training_root) / "benchmarks_v1"

    def model_snapshot(self, protocol_default: Path) -> Path:
        return self.model_snapshot_override or protocol_default
