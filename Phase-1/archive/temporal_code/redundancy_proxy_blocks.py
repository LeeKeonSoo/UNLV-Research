#!/usr/bin/env python3
"""Materialize exact-token proxy training and heldout blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors.torch import save_file

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_FREEZE_REPORT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_experiment_freeze_report.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "redundancy_saturation_proxy_qwen25_0p5b_v1" / "token_blocks"
)
DEFAULT_MANIFEST = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_packed_blocks_manifest.json"
)


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _pack_exact(
    source_path: Path,
    tokenizer: Any,
    *,
    exact_tokens: int,
    sequence_length: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if exact_tokens % sequence_length:
        raise ValueError(
            f"Token budget {exact_tokens} is not divisible by sequence length "
            f"{sequence_length}."
        )
    token_ids: List[int] = []
    eos = tokenizer.eos_token_id
    source_records_read = 0
    nonempty_records_read = 0
    complete_records_consumed = 0
    final_record_uid = None
    final_record_tokens_available = 0
    final_record_tokens_consumed = 0
    cut_inside_final_record = False

    for row in _jsonl(source_path):
        source_records_read += 1
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        nonempty_records_read += 1
        record_ids = list(tokenizer(text, add_special_tokens=False).input_ids)
        if eos is not None:
            record_ids.append(int(eos))
        remaining = exact_tokens - len(token_ids)
        if remaining <= 0:
            break
        take = min(remaining, len(record_ids))
        token_ids.extend(int(value) for value in record_ids[:take])
        final_record_uid = str(
            row.get("chunk_uid") or row.get("record_id") or source_records_read
        )
        final_record_tokens_available = len(record_ids)
        final_record_tokens_consumed = take
        cut_inside_final_record = take < len(record_ids)
        if take == len(record_ids):
            complete_records_consumed += 1
        if len(token_ids) == exact_tokens:
            break

    if len(token_ids) != exact_tokens:
        raise RuntimeError(
            f"Insufficient tokens in {source_path}: "
            f"required={exact_tokens}, materialized={len(token_ids)}"
        )
    tensor = torch.tensor(token_ids, dtype=torch.int32).reshape(
        exact_tokens // sequence_length,
        sequence_length,
    )
    audit = {
        "source_records_read": source_records_read,
        "nonempty_records_read": nonempty_records_read,
        "complete_records_consumed": complete_records_consumed,
        "final_record_uid": final_record_uid,
        "final_record_tokens_available": final_record_tokens_available,
        "final_record_tokens_consumed": final_record_tokens_consumed,
        "cut_inside_final_record": cut_inside_final_record,
        "exact_tokens": exact_tokens,
        "sequence_length": sequence_length,
        "blocks": exact_tokens // sequence_length,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_content_sha256": _tensor_content_sha256(tensor),
        "first_block_content_sha256": _tensor_content_sha256(tensor[0]),
        "last_block_content_sha256": _tensor_content_sha256(tensor[-1]),
        "minimum_token_id": int(tensor.min().item()),
        "maximum_token_id": int(tensor.max().item()),
    }
    return tensor.contiguous(), audit


def materialize(
    config_path: Path,
    freeze_report_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    plan = load_json(config_path)
    freeze_report = load_json(freeze_report_path)
    config_sha256 = sha256_file(config_path)
    if config_sha256 != freeze_report["config_sha256"]:
        raise RuntimeError(
            "Frozen proxy config hash mismatch: "
            f"expected={freeze_report['config_sha256']} actual={config_sha256}"
        )
    if plan.get("status") != "frozen_before_proxy_training_outcomes":
        raise RuntimeError(f"Proxy config is not frozen: {plan.get('status')}")

    tokenizer_path = Path(plan["target_model"]["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_hash = sha256_file(tokenizer_path / "tokenizer.json")
    expected_tokenizer_hash = plan["target_model"]["snapshot_artifacts"][
        "tokenizer.json"
    ]["sha256"]
    if tokenizer_hash != expected_tokenizer_hash:
        raise RuntimeError(
            "Frozen tokenizer hash mismatch: "
            f"expected={expected_tokenizer_hash} actual={tokenizer_hash}"
        )

    packing = plan["tokenization_and_packing"]
    sequence_length = int(packing["sequence_length"])
    exact_train_tokens = int(packing["exact_train_tokens_per_arm"])
    exact_heldout_tokens = int(plan["heldout_nll"]["exact_evaluation_tokens"])
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Dict[str, Any]] = {}
    for arm, contract in plan["arms"].items():
        source_path = Path(contract["path"])
        if sha256_file(source_path) != contract["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch for {arm}: {source_path}")
        tensor, audit = _pack_exact(
            source_path,
            tokenizer,
            exact_tokens=exact_train_tokens,
            sequence_length=sequence_length,
        )
        output_path = output_dir / f"{arm}.safetensors"
        save_file({"input_ids": tensor}, output_path)
        artifacts[arm] = {
            "role": "training_arm",
            "source_path": str(source_path),
            "source_sha256": contract["sha256"],
            "path": str(output_path),
            "file_sha256": sha256_file(output_path),
            **audit,
        }

    heldout_contract = plan["heldout_nll"]
    heldout_source = Path(heldout_contract["path"])
    if sha256_file(heldout_source) != heldout_contract["sha256"]:
        raise RuntimeError(f"Frozen heldout hash mismatch: {heldout_source}")
    heldout_tensor, heldout_audit = _pack_exact(
        heldout_source,
        tokenizer,
        exact_tokens=exact_heldout_tokens,
        sequence_length=sequence_length,
    )
    heldout_output = output_dir / "development_code_nll_heldout.safetensors"
    save_file({"input_ids": heldout_tensor}, heldout_output)
    artifacts["development_code_nll_heldout"] = {
        "role": "heldout_nll",
        "source_path": str(heldout_source),
        "source_sha256": heldout_contract["sha256"],
        "path": str(heldout_output),
        "file_sha256": sha256_file(heldout_output),
        **heldout_audit,
    }

    train_content_hashes = {
        artifacts[arm]["tensor_content_sha256"] for arm in plan["arms"]
    }
    blockers = []
    if len(train_content_hashes) != len(plan["arms"]):
        blockers.append("training_arm_token_tensors_are_not_unique")
    if artifacts["development_code_nll_heldout"]["tensor_content_sha256"] in train_content_hashes:
        blockers.append("heldout_tensor_matches_training_tensor")

    manifest = {
        "schema_version": "redundancy-proxy-packed-blocks-manifest-v1",
        "status": (
            "redundancy_proxy_exact_blocks_materialized"
            if not blockers
            else "redundancy_proxy_exact_blocks_materialized_with_blockers"
        ),
        "frozen_config": {
            "path": str(config_path),
            "sha256": config_sha256,
        },
        "freeze_report": {
            "path": str(freeze_report_path),
            "sha256": sha256_file(freeze_report_path),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "revision": plan["target_model"]["revision"],
            "tokenizer_json_sha256": tokenizer_hash,
            "base_vocab_size": int(tokenizer.vocab_size),
            "tokenizer_size_with_added_tokens": int(len(tokenizer)),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        "serialization": {
            "format": "safetensors",
            "tensor_key": "input_ids",
            "dtype": "int32",
            "byte_order_for_content_hash": "native_little_endian_on_frozen_windows_host",
        },
        "artifacts": artifacts,
        "training_contract": {
            "arm_count": len(plan["arms"]),
            "exact_tokens_per_arm": exact_train_tokens,
            "exact_blocks_per_arm": exact_train_tokens // sequence_length,
            "sequence_length": sequence_length,
            "all_training_shapes_equal": len(
                {
                    (row["blocks"], row["sequence_length"])
                    for name, row in artifacts.items()
                    if row["role"] == "training_arm"
                }
            )
            == 1,
            "all_training_content_hashes_unique": len(train_content_hashes)
            == len(plan["arms"]),
            "shuffle_policy": plan["training_recipe"][
                "shuffle_training_blocks_per_seed"
            ],
            "seed_set": plan["training_recipe"]["seeds"],
        },
        "heldout_contract": {
            "exact_tokens": exact_heldout_tokens,
            "exact_blocks": exact_heldout_tokens // sequence_length,
            "sequence_length": sequence_length,
            "repository_overlap_with_train": heldout_contract[
                "train_repository_overlap_count"
            ],
        },
        "blockers": blockers,
        "utility_scope": plan["primary_comparison"]["utility_scope"],
        "claim_boundary": (
            "Exact-token block materialization only. No model was trained and no "
            "Utility, retention, promotion, release, or framework-validity result "
            "was produced."
        ),
    }
    save_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize frozen redundancy proxy token blocks."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze-report", type=Path, default=DEFAULT_FREEZE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = materialize(
        args.config,
        args.freeze_report,
        args.output_dir,
        args.manifest,
    )
    print(
        {
            "status": manifest["status"],
            "output_dir": str(args.output_dir),
            "artifacts": {
                name: {
                    "blocks": row["blocks"],
                    "tokens": row["exact_tokens"],
                    "file_sha256": row["file_sha256"],
                    "tensor_content_sha256": row["tensor_content_sha256"],
                }
                for name, row in manifest["artifacts"].items()
            },
            "blockers": manifest["blockers"],
        }
    )
    return 0 if not manifest["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
