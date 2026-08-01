#!/usr/bin/env python3
"""Validate immutable-block redundancy proxy runner semantics."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "183_run_redundancy_proxy_qlora.py"
    spec = importlib.util.spec_from_file_location("redundancy_proxy_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "runner_audit.json"
        audit = module.audit_runner(
            ROOT
            / "configs"
            / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json",
            ROOT
            / "validation"
            / "frozen_contracts"
            / "redundancy_proxy_packed_blocks_manifest.json",
            ROOT
            / "configs"
            / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json",
            output,
        )
    assert audit["status"] == "redundancy_proxy_qlora_runner_ready"
    assert not audit["blockers"]
    contract = audit["training_contract"]
    assert contract["optimizer_steps"] == 40
    assert contract["gradient_accumulation_steps"] == 8
    assert contract["required_micro_steps"] == 320
    assert contract["tokens_per_arm"] == 327680
    assert contract["single_epoch_exact_consumption"] is True

    for seed, arms in audit["seed_shuffle_contract"].items():
        hashes = {row["order_sha256"] for row in arms.values()}
        assert len(hashes) == 1, seed
        assert all(row["block_count"] == 320 for row in arms.values())
        assert all(row["all_indices_consumed_once"] for row in arms.values())
    assert len(
        {
            next(iter(arms.values()))["order_sha256"]
            for arms in audit["seed_shuffle_contract"].values()
        }
    ) == 3
    assert audit["completion_contract"]["partial_run_reusable"] is False
    assert "adapter_model.safetensors" in audit["completion_contract"]["required_files"]
    assert audit["utility_scope"].startswith("Stage C validation only")
    print("[redundancy-proxy-runner] exact one-epoch compute and matched seed shuffles: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
