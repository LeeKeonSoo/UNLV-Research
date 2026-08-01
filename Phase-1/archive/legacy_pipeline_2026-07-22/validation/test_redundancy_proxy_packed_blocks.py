#!/usr/bin/env python3
"""Validate exact packed tensors for the redundancy proxy experiment."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json, sha256_file


MANIFEST = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_packed_blocks_manifest.json"
)


def _content_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def main() -> int:
    manifest = load_json(MANIFEST)
    assert manifest["status"] == "redundancy_proxy_exact_blocks_materialized"
    assert not manifest["blockers"]
    assert manifest["serialization"]["format"] == "safetensors"
    assert manifest["serialization"]["tensor_key"] == "input_ids"
    assert sha256_file(Path(manifest["frozen_config"]["path"])) == manifest[
        "frozen_config"
    ]["sha256"]

    training = manifest["training_contract"]
    assert training["arm_count"] == 3
    assert training["exact_tokens_per_arm"] == 327680
    assert training["exact_blocks_per_arm"] == 320
    assert training["sequence_length"] == 1024
    assert training["all_training_shapes_equal"] is True
    assert training["all_training_content_hashes_unique"] is True
    assert training["seed_set"] == [11, 23, 37]

    train_tensors = []
    for name, artifact in manifest["artifacts"].items():
        path = Path(artifact["path"])
        assert path.exists(), f"Missing packed artifact: {path}"
        assert sha256_file(path) == artifact["file_sha256"]
        payload = load_file(path)
        assert set(payload) == {"input_ids"}
        tensor = payload["input_ids"]
        assert tensor.dtype == torch.int32
        assert tuple(tensor.shape) == (
            artifact["blocks"],
            artifact["sequence_length"],
        )
        assert tensor.numel() == artifact["exact_tokens"]
        assert _content_sha256(tensor) == artifact["tensor_content_sha256"]
        assert _content_sha256(tensor[0]) == artifact["first_block_content_sha256"]
        assert _content_sha256(tensor[-1]) == artifact["last_block_content_sha256"]
        assert artifact["minimum_token_id"] >= 0
        assert artifact["maximum_token_id"] < manifest["tokenizer"][
            "tokenizer_size_with_added_tokens"
        ]
        if artifact["role"] == "training_arm":
            assert tuple(tensor.shape) == (320, 1024)
            train_tensors.append(tensor)
        else:
            assert name == "development_code_nll_heldout"
            assert tuple(tensor.shape) == (143, 1024)

    assert len(train_tensors) == 3
    assert not torch.equal(train_tensors[0], train_tensors[1])
    assert not torch.equal(train_tensors[0], train_tensors[2])
    assert not torch.equal(train_tensors[1], train_tensors[2])
    assert manifest["heldout_contract"]["repository_overlap_with_train"] == 0
    assert manifest["utility_scope"].startswith("Stage C validation only")
    print("[redundancy-proxy-blocks] exact shapes, hashes, and unique token streams: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
