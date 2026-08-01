from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Protocol

import torch

from data_eval_common import save_json, sha256_file


class Tokenizer(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool) -> "Encoded": ...


class Encoded(Protocol):
    input_ids: list[int]


def _rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count, sha256_file(path)


def _token_ids(rows: Iterable[dict[str, object]], tokenizer: Tokenizer) -> Iterable[int]:
    for row in rows:
        encoded = tokenizer(str(row["text"]), add_special_tokens=False)
        for token_id in encoded.input_ids:
            yield int(token_id)
        if tokenizer.eos_token_id is not None:
            yield tokenizer.eos_token_id


def _pack(token_ids: Iterable[int], path: Path, sequence_length: int) -> dict[str, int | str]:
    values = list(token_ids)
    packed_tokens = len(values) // sequence_length * sequence_length
    if packed_tokens == 0:
        raise RuntimeError("No complete training block available")
    tensor = torch.tensor(values[:packed_tokens], dtype=torch.int32).reshape(-1, sequence_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": tensor}, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "blocks": int(tensor.shape[0]),
        "packed_tokens": int(tensor.numel()),
        "dropped_tail_tokens": len(values) - packed_tokens,
    }


def _arm_rows(rows: list[dict[str, object]], arm: str, source_stage: str) -> list[dict[str, object]]:
    return [
        {
            "arm": arm,
            "record_id": str(row.get("record_id") or row.get("chunk_uid")),
            "text": str(row["text"]),
            "source_stage": source_stage,
        }
        for row in rows
        if str(row.get("text") or "").strip()
    ]


def _arm_report(
    rows: list[dict[str, object]],
    output_root: Path,
    arm: str,
    source_stage: str,
    tokenizer: Tokenizer,
    sequence_length: int,
    gradient_accumulation_steps: int,
) -> dict[str, object]:
    arm_path = output_root / "arms" / f"{arm}.jsonl"
    records, arm_sha256 = _write_jsonl(arm_path, rows)
    blocks = _pack(_token_ids(rows, tokenizer), output_root / "token_blocks" / f"{arm}.pt", sequence_length)
    optimizer_steps = int(blocks["blocks"]) // gradient_accumulation_steps
    if optimizer_steps == 0:
        raise RuntimeError(f"No complete gradient-accumulation group available for {arm}")
    return {
        "source_stage": source_stage,
        "records": records,
        "arm_path": str(arm_path),
        "arm_sha256": arm_sha256,
        **blocks,
        "optimizer_steps": optimizer_steps,
        "effective_training_tokens": optimizer_steps * gradient_accumulation_steps * sequence_length,
    }


def materialize(
    raw_safe_path: Path,
    curated_path: Path,
    output_root: Path,
    tokenizer: Tokenizer,
    sequence_length: int,
    gradient_accumulation_steps: int,
) -> dict[str, object]:
    raw_rows = _arm_rows(_rows(raw_safe_path), "raw_safe_natural", "Stage 0 release")
    curated_rows = _arm_rows(_rows(curated_path), "curated_natural", "Stage B selected")
    arms = {
        "raw_safe_natural": _arm_report(raw_rows, output_root, "raw_safe_natural", "Stage 0 release", tokenizer, sequence_length, gradient_accumulation_steps),
        "curated_natural": _arm_report(curated_rows, output_root, "curated_natural", "Stage B selected", tokenizer, sequence_length, gradient_accumulation_steps),
    }
    report = {
        "schema_version": "code-5m-external-validation-inputs-v1",
        "status": "code_5m_external_validation_inputs_frozen",
        "sources": {"raw_safe": str(raw_safe_path), "curated": str(curated_path)},
        "source_sha256": {"raw_safe": sha256_file(raw_safe_path), "curated": sha256_file(curated_path)},
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "arms": arms,
        "utility_scope": "External validation only; never selector objective",
    }
    save_json(output_root / "code_5m_external_validation_inputs_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-safe", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    report = materialize(args.raw_safe, args.curated, args.output_root, tokenizer, args.sequence_length, args.gradient_accumulation_steps)
    print(json.dumps({"status": report["status"], "arms": report["arms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
