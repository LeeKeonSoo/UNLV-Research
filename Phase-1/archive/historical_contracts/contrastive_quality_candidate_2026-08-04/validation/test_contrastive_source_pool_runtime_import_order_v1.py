#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_contrastive_source_pool_v1 import _load_runtime_dependencies


def test_datasets_loads_before_transformers_to_avoid_windows_pyarrow_access_violation() -> None:
    imported: list[str] = []

    def fake_import(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(load_dataset="dataset-loader", AutoTokenizer="tokenizer-factory")

    load_dataset, auto_tokenizer = _load_runtime_dependencies(fake_import)

    assert imported == ["datasets", "transformers"]
    assert load_dataset == "dataset-loader"
    assert auto_tokenizer == "tokenizer-factory"


if __name__ == "__main__":
    test_datasets_loads_before_transformers_to_avoid_windows_pyarrow_access_violation()
    print("[contrastive-source-pool-runtime-import-order-v1] datasets precedes transformers: pass")
