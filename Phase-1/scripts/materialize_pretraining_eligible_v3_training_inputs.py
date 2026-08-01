#!/usr/bin/env python3
"""Materialize the natural-budget v3 external-training inputs with the frozen tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol, TypedDict

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "protocols" / "code_7benchmark_pretraining_eligible_v3_materialization.json"


class Encoded(Protocol):
    input_ids: list[int]


class Tokenizer(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool) -> Encoded: ...


class ArmReport(TypedDict):
    source_stage: str
    records: int
    source_path: str
    source_sha256: str
    arm_path: str
    arm_sha256: str
    blocks_path: str
    blocks_sha256: str
    sequence_length: int
    blocks: int
    stream_tokens: int
    materialized_tokens: int
    dropped_tail_tokens: int
    optimizer_steps: int


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 fingerprint for one immutable artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read nonempty JSONL records while preserving their frozen source order."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            text = str(raw.get("text") or "")
            if not text.strip():
                continue
            record_id = str(raw.get("record_id") or raw.get("stage_a_record_id") or raw.get("chunk_uid") or line_number)
            rows.append({"record_id": record_id, "text": text})
    if not rows:
        raise RuntimeError(f"No nonempty text records found at {path}")
    return rows


def write_arm_rows(path: Path, rows: list[dict[str, str]]) -> str:
    """Write the exact text sequence used for tokenizer materialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return sha256_file(path)


def token_stream(rows: list[dict[str, str]], tokenizer: Tokenizer) -> list[int]:
    """Tokenize every frozen record and append one EOS boundary per record."""
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Frozen tokenizer does not declare an EOS token")
    tokens: list[int] = []
    for row in rows:
        encoded = tokenizer(row["text"], add_special_tokens=False)
        tokens.extend(int(token_id) for token_id in encoded.input_ids)
        tokens.append(int(tokenizer.eos_token_id))
    return tokens


def materialize_arm(
    *,
    arm: str,
    source: Path,
    source_stage: str,
    output_root: Path,
    tokenizer: Tokenizer,
    sequence_length: int,
    gradient_accumulation_steps: int,
) -> ArmReport:
    """Create only complete optimizer-update groups for one natural-budget arm."""
    rows = load_rows(source)
    arm_path = output_root / "arms" / f"{arm}.jsonl"
    arm_sha256 = write_arm_rows(arm_path, rows)
    tokens = token_stream(rows, tokenizer)
    complete_blocks = len(tokens) // sequence_length
    optimizer_steps = complete_blocks // gradient_accumulation_steps
    retained_blocks = optimizer_steps * gradient_accumulation_steps
    if retained_blocks == 0:
        raise RuntimeError(f"No complete optimizer update is available for {arm}")
    materialized_tokens = retained_blocks * sequence_length
    blocks_path = output_root / "token_blocks" / f"{arm}.pt"
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(tokens[:materialized_tokens], dtype=torch.int32).reshape(retained_blocks, sequence_length)
    torch.save({"input_ids": tensor}, blocks_path)
    return {
        "source_stage": source_stage,
        "records": len(rows),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "arm_path": str(arm_path),
        "arm_sha256": arm_sha256,
        "blocks_path": str(blocks_path),
        "blocks_sha256": sha256_file(blocks_path),
        "sequence_length": sequence_length,
        "blocks": retained_blocks,
        "stream_tokens": len(tokens),
        "materialized_tokens": materialized_tokens,
        "dropped_tail_tokens": len(tokens) - materialized_tokens,
        "optimizer_steps": optimizer_steps,
    }


def materialize(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, object]:
    """Materialize both frozen v3 arms and write their audited input report."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["status"] != "frozen_before_tokenizer_materialization":
        raise RuntimeError("Materialization contract is not in its frozen pre-run state")
    packing = contract["packing"]
    sequence_length = int(packing["sequence_length"])
    gradient_accumulation_steps = int(packing["gradient_accumulation_steps"])
    output_root = Path(contract["output_root"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract["tokenizer"]["snapshot_path"],
        local_files_only=bool(contract["tokenizer"]["local_files_only"]),
        use_fast=True,
    )
    arms: dict[str, ArmReport] = {}
    for arm, arm_config in contract["arms"].items():
        arms[arm] = materialize_arm(
            arm=arm,
            source=Path(arm_config["source"]),
            source_stage=str(arm_config["source_stage"]),
            output_root=output_root,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    report = {
        "schema_version": "code-7benchmark-pretraining-eligible-v3-training-inputs-v1",
        "status": "tokenizer_materialization_complete",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "tokenizer": contract["tokenizer"],
        "packing": packing,
        "arms": arms,
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_token_fraction_read": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    report_path = output_root / "training_inputs_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the frozen v3 natural-budget training inputs.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    report = materialize(args.contract)
    print(json.dumps({"status": report["status"], "arms": report["arms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
