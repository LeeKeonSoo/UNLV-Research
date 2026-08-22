#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers"]
# ///
# --- How to run ---
# uv run scripts/materialize_matched_random_control.py --contract protocols/code_7m_random72_matched_materialization_v1.json
"""Materialize one fixed full-record random control matched to a target arm."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, TypedDict

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "protocols" / "code_7m_random72_matched_materialization_v1.json"


class Encoded(Protocol):
    input_ids: list[int]


class Tokenizer(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool) -> Encoded: ...


@dataclass(frozen=True, slots=True)
class SourceRow:
    source_index: int
    record_id: str
    text: str
    stream_tokens: int


class ArmOutput(TypedDict):
    source_path: str
    source_sha256: str
    sampling_seed: int
    selection_algorithm: str
    records: int
    source_records: int
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
    retention_fraction_of_raw_materialized: float


class SelectorBoundary(TypedDict):
    utility_read: bool
    benchmark_outcomes_read: bool
    source_reputation_read: bool


class ControlBoundary(TypedDict):
    runtime_policy: bool
    target_size_is_diagnostic_only: bool
    full_records_only: bool
    source_order_preserved: bool


class MatchedRandomReport(TypedDict):
    schema_version: str
    status: str
    contract: str
    contract_sha256: str
    target_report: str
    target_report_sha256: str
    target_arm: str
    output_arm: str
    arms: dict[str, ArmOutput]
    selector_boundary: SelectorBoundary
    control_boundary: ControlBoundary


class MaterializationError(RuntimeError):
    """Raised when a frozen matched-control contract cannot be satisfied."""


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, tokenizer: Tokenizer) -> list[SourceRow]:
    """Parse and count full source records under the frozen tokenizer boundary."""
    rows: list[SourceRow] = []
    with path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise MaterializationError(f"Expected JSON object at {path}:{source_index + 1}")
            text = str(raw.get("text") or "")
            if not text.strip():
                continue
            record_id = str(
                raw.get("record_id")
                or raw.get("stage_a_record_id")
                or raw.get("chunk_uid")
                or source_index
            )
            encoded = tokenizer(text, add_special_tokens=False)
            rows.append(
                SourceRow(
                    source_index=source_index,
                    record_id=record_id,
                    text=text,
                    stream_tokens=len(encoded.input_ids) + 1,
                )
            )
    if not rows:
        raise MaterializationError(f"No nonempty source records found at {path}")
    return rows


def random_priority(row: SourceRow, seed: int) -> bytes:
    """Return a stable pseudo-random priority independent of file iteration state."""
    payload = f"{seed}:{row.source_index}:{row.record_id}".encode()
    return hashlib.sha256(payload).digest()


def select_rows(
    rows: list[SourceRow], *, seed: int, target_tokens: int, group_tokens: int
) -> list[SourceRow]:
    """Select full records randomly while preserving one exact optimizer-step count."""
    upper_bound = target_tokens + group_tokens - 1
    selected: list[SourceRow] = []
    stream_tokens = 0
    for row in sorted(rows, key=lambda candidate: random_priority(candidate, seed)):
        if stream_tokens >= target_tokens:
            break
        if stream_tokens + row.stream_tokens <= upper_bound:
            selected.append(row)
            stream_tokens += row.stream_tokens
    if stream_tokens < target_tokens:
        raise MaterializationError(
            f"Random full-record selection reached {stream_tokens:,}, below target {target_tokens:,}"
        )
    return sorted(selected, key=lambda row: row.source_index)


def write_rows(path: Path, rows: list[SourceRow]) -> str:
    """Write selected full records in their original source order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "record_id": row.record_id,
                        "source_index": row.source_index,
                        "text": row.text,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
    return sha256_file(path)


def encode_selected(rows: list[SourceRow], tokenizer: Tokenizer) -> list[int]:
    """Encode the selected records and append exactly one EOS per record."""
    if tokenizer.eos_token_id is None:
        raise MaterializationError("Frozen tokenizer does not declare an EOS token")
    tokens: list[int] = []
    for row in rows:
        tokens.extend(int(token) for token in tokenizer(row.text, add_special_tokens=False).input_ids)
        tokens.append(int(tokenizer.eos_token_id))
    return tokens


def materialize(
    contract_path: Path = DEFAULT_CONTRACT, *, tokenizer: Tokenizer | None = None
) -> MatchedRandomReport:
    """Materialize the fixed random control declared by one frozen contract."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_materialization":
        raise MaterializationError("Matched-random contract is not frozen")
    source_path = Path(str(contract["source"]["path"]))
    declared_source_sha = contract["source"].get("sha256")
    source_sha = sha256_file(source_path)
    if declared_source_sha is not None and source_sha != declared_source_sha:
        raise MaterializationError("Matched-random source SHA-256 mismatch")
    target_report_path = Path(str(contract["target"]["report"]))
    declared_target_sha = contract["target"].get("report_sha256")
    target_report_sha = sha256_file(target_report_path)
    if declared_target_sha is not None and target_report_sha != declared_target_sha:
        raise MaterializationError("Matched-random target report SHA-256 mismatch")
    target_report = json.loads(target_report_path.read_text(encoding="utf-8"))
    target_arm = str(contract["target"]["arm"])
    output_arm = str(contract.get("output_arm", "random72_matched"))
    if not output_arm or not output_arm.replace("_", "").isalnum():
        raise MaterializationError("Output arm must be a nonempty alphanumeric snake-case name")
    target_tokens = int(target_report["arms"][target_arm]["materialized_tokens"])
    packing = contract["packing"]
    sequence_length = int(packing["sequence_length"])
    accumulation = int(packing["gradient_accumulation_steps"])
    group_tokens = sequence_length * accumulation
    if target_tokens <= 0 or target_tokens % group_tokens:
        raise MaterializationError("Target materialized tokens are not complete optimizer groups")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(contract["tokenizer"]["snapshot_path"]),
            local_files_only=bool(contract["tokenizer"]["local_files_only"]),
            use_fast=True,
        )
    rows = load_rows(source_path, tokenizer)
    seed = int(contract["sampling"]["seed"])
    selected = select_rows(rows, seed=seed, target_tokens=target_tokens, group_tokens=group_tokens)
    output_root = Path(str(contract["output_root"]))
    arm_path = output_root / "arms" / f"{output_arm}.jsonl"
    arm_sha = write_rows(arm_path, selected)
    tokens = encode_selected(selected, tokenizer)
    blocks = target_tokens // sequence_length
    blocks_path = output_root / "token_blocks" / f"{output_arm}.pt"
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(tokens[:target_tokens], dtype=torch.int32).reshape(blocks, sequence_length)
    torch.save({"input_ids": tensor}, blocks_path)
    raw_materialized = int(contract["source"]["raw_materialized_tokens"])
    arm: ArmOutput = {
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "sampling_seed": seed,
        "selection_algorithm": "sha256_priority_full_records_greedy_fit_v1",
        "records": len(selected),
        "source_records": len(rows),
        "arm_path": str(arm_path),
        "arm_sha256": arm_sha,
        "blocks_path": str(blocks_path),
        "blocks_sha256": sha256_file(blocks_path),
        "sequence_length": sequence_length,
        "blocks": blocks,
        "stream_tokens": len(tokens),
        "materialized_tokens": target_tokens,
        "dropped_tail_tokens": len(tokens) - target_tokens,
        "optimizer_steps": blocks // accumulation,
        "retention_fraction_of_raw_materialized": target_tokens / raw_materialized,
    }
    report: MatchedRandomReport = {
        "schema_version": "matched-random-control-report-v1",
        "status": "tokenizer_materialization_complete",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "target_report": str(target_report_path),
        "target_report_sha256": target_report_sha,
        "target_arm": target_arm,
        "output_arm": output_arm,
        "arms": {output_arm: arm},
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "source_reputation_read": False,
        },
        "control_boundary": {
            "runtime_policy": False,
            "target_size_is_diagnostic_only": True,
            "full_records_only": True,
            "source_order_preserved": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matched_random_control_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    report = materialize(args.contract)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
