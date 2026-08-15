#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.confirmatory_qlora_training import resolve_run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        blocks = root / "blocks.pt"
        blocks.write_bytes(b"frozen blocks")
        report = root / "inputs.json"
        report.write_text(
            json.dumps({"status": "tokenizer_materialization_complete", "arms": {"raw_audited_natural": {"blocks_path": str(blocks), "blocks_sha256": _sha256(blocks), "optimizer_steps": 1}}}),
            encoding="utf-8",
        )
        protocol = root / "protocol.json"
        protocol.write_text(
            json.dumps({"status": "frozen_before_curation_and_tokenizer_materialization", "training": {"arms": ["raw_audited_natural"], "seeds": [101], "output_root": str(root / "runs"), "snapshot_path": str(root / "protocol-model")}}),
            encoding="utf-8",
        )
        worker_runs = root / "worker-runs"
        worker_model = root / "worker-model"
        with patch.dict(
            os.environ,
            {
                "UNLV_TRAINING_OUTPUT_ROOT": str(worker_runs),
                "UNLV_MODEL_SNAPSHOT_PATH": str(worker_model),
            },
        ):
            run = resolve_run(protocol, report, "raw_audited_natural", 101)
        assert run["blocks_path"] == blocks
        assert run["run_dir"] == worker_runs / "qlora_runs" / "raw_audited_natural_seed101_steps1"
        assert run["snapshot_path"] == worker_model
    print("[confirmatory-qlora-training] frozen arm preflight: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
