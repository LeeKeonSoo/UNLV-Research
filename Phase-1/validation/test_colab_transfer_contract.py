#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PART_DIRECTORY = "adapter_parts_4m"
PART_COUNT = 379
PART_SIZE_BYTES = 4 * 1024 * 1024


def main() -> int:
    protocol = json.loads(
        (ROOT / "protocols" / "colab_benchmark_worker_v1.json").read_text(
            encoding="utf-8"
        )
    )
    notebook = json.loads(
        (ROOT / "notebooks" / "colab_benchmark_worker_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    helper = (
        ROOT
        / "output"
        / "colab_worker_v1"
        / "helper_extension"
        / "extension"
        / "extension.js"
    ).read_text(encoding="utf-8")

    transfer = protocol["adapter_transfer"]
    assert transfer["part_count"] == PART_COUNT
    assert transfer["part_size_bytes"] == PART_SIZE_BYTES

    notebook_source = "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    assert PART_DIRECTORY in notebook_source
    assert f"range({PART_COUNT})" in notebook_source
    assert PART_DIRECTORY in helper
    print("[colab-transfer-contract] 4 MiB upload contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
