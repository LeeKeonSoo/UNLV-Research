#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_chunks import apply_stage_a_hard_gates, syntax_aware_chunks


ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT = ROOT / "configs" / "raw_corpus_matrix_stage_c_study_v1.json"
DEFAULT_OUTPUT_ROOT = OUTPUT_DIR / "raw_corpus_matrix_stage_c_v1"
SEQUENCE_LENGTH = 2048


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tokens = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            tokens += int(row["target_token_count_with_eos"])
    return {"path": str(path), "records": count, "target_tokens_with_eos": tokens, "sha256": sha256_file(path)}


def _chunks(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    unchunkable = 0
    for row in rows:
        partition = row.get("partition") if isinstance(row.get("partition"), dict) else {}
        result = syntax_aware_chunks(row)
        if not result["parseable"]:
            unchunkable += 1
            continue
        for index, chunk in enumerate(result["chunks"]):
            chunks.append(
                {
                    "chunk_uid": f"{row['record_id']}::chunk-{index:04d}",
                    "record_id": row["record_id"],
                    "split": split,
                    "bundle_id": partition.get("bundle_id"),
                    "repository_identity": partition.get("repository_identity"),
                    "path": partition.get("path"),
                    "change_type": partition.get("change_type"),
                    "content_type": partition.get("content_type"),
                    "chunking_mode": result["chunking_mode"],
                    "chunk_kind": chunk["kind"],
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                    "text": chunk["text"],
                }
            )
    return chunks, unchunkable


def _target_tokens(row: dict[str, Any], tokenizer: Any) -> int:
    return len(tokenizer(str(row["text"]), add_special_tokens=False).input_ids) + 1


def _with_tokens(rows: Iterable[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    return [{**row, "target_token_count_with_eos": _target_tokens(row, tokenizer)} for row in rows]


def _stable(rows: Iterable[dict[str, Any]], seed: int, label: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{label}:{row['chunk_uid']}".encode("utf-8")).hexdigest(),
    )


def _audit_row(row: dict[str, Any], arm: str, source_pool: str, source_by_record: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = source_by_record.get(str(row["record_id"]), {})
    return {
        "arm": arm,
        "chunk_uid": row["chunk_uid"],
        "record_id": row["record_id"],
        "text": row["text"],
        "target_token_count_with_eos": row["target_token_count_with_eos"],
        "source_pool": source_pool,
        "training_audit_provenance": {
            "source_uri": (source.get("provenance") or {}).get("source_uri"),
            "collected_at": (source.get("provenance") or {}).get("collected_at"),
            "license_family": ((source.get("rights") or {}).get("license")),
            "source_tier": ((source.get("audit_provenance") or {}).get("source_tier")),
        },
        "stage_a_pass": row.get("stage_a_pass"),
        "stage_b_selection": row.get("stage_b_selection"),
        "stage_b_baseline": row.get("stage_b_baseline"),
    }


def _token_stream(rows: Iterable[dict[str, Any]], tokenizer: Any) -> Iterable[int]:
    eos = tokenizer.eos_token_id
    for row in rows:
        for token_id in tokenizer(str(row["text"]), add_special_tokens=False).input_ids:
            yield int(token_id)
        if eos is not None:
            yield int(eos)


def _pack_blocks(
    rows: Iterable[dict[str, Any]], tokenizer: Any, output_path: Path, token_budget: int
) -> dict[str, Any]:
    packed_budget = (token_budget // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
    if packed_budget <= 0:
        raise RuntimeError("Packed token budget is empty")
    token_ids: list[int] = []
    for token_id in _token_stream(rows, tokenizer):
        token_ids.append(token_id)
        if len(token_ids) == packed_budget:
            break
    if len(token_ids) != packed_budget:
        raise RuntimeError(f"Insufficient token stream for packed budget: {len(token_ids)} < {packed_budget}")
    tensor = torch.tensor(token_ids, dtype=torch.int32).reshape(-1, SEQUENCE_LENGTH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": tensor}, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "blocks": int(tensor.shape[0]),
        "sequence_length": SEQUENCE_LENGTH,
        "packed_tokens": int(tensor.numel()),
        "source_token_budget": token_budget,
        "dropped_tail_tokens": token_budget - int(tensor.numel()),
    }


def _holdout(rows: list[dict[str, Any]], split: str, output_path: Path) -> dict[str, Any]:
    chunks, unchunkable = _chunks(rows, split)
    decisions = apply_stage_a_hard_gates(chunks)
    passed = [row for row in decisions if row["stage_a_pass"]]
    rejected = [row for row in decisions if not row["stage_a_pass"]]
    _write_jsonl(output_path / "stage_a_pass.jsonl", [{**row, "target_token_count_with_eos": 0} for row in passed])
    _write_jsonl(output_path / "stage_a_rejected.jsonl", [{**row, "target_token_count_with_eos": 0} for row in rejected])
    return {
        "stage_b_read": False,
        "input_release_candidate_count": len(rows),
        "stage_a_chunk_count": len(decisions),
        "stage_a_pass_count": len(passed),
        "stage_a_rejected_count": len(rejected),
        "stage_a_unchunkable_count": unchunkable,
        "stage_a_pass_path": str(output_path / "stage_a_pass.jsonl"),
        "stage_a_pass_sha256": sha256_file(output_path / "stage_a_pass.jsonl"),
    }


def prepare(contract_path: Path, output_root: Path, *, allow_download: bool) -> dict[str, Any]:
    from transformers import AutoTokenizer

    contract = load_json(contract_path)
    sources = {name: Path(value) for name, value in contract["training_sources"].items() if name != "stage_b_report"}
    release_rows = _read_jsonl(sources["raw_mixed_release_candidates"])
    source_by_record = {str(row["record_id"]): row for row in release_rows}
    curated = _read_jsonl(sources["raw_mixed_curated"])
    baseline = _read_jsonl(sources["raw_mixed_common_stage_a_random"])
    raw_chunks, raw_unchunkable = _chunks(release_rows, "train")
    baseline_ids = {str(row["chunk_uid"]) for row in baseline}
    curated_ids = {str(row["chunk_uid"]) for row in curated}
    overlap = curated_ids.intersection(baseline_ids)
    if overlap:
        raise RuntimeError(f"Curated/common Stage-A baseline overlap: {sorted(overlap)[:5]}")
    raw_without_baseline = [row for row in raw_chunks if str(row["chunk_uid"]) not in baseline_ids]
    tokenizer = AutoTokenizer.from_pretrained(
        contract["target_model"]["tokenizer_id"], local_files_only=not allow_download, use_fast=True
    )
    tokenized = {
        "curated_equal_token": _with_tokens(curated, tokenizer),
        "stage_a_random_equal_token": _with_tokens(baseline, tokenizer),
        "raw_mixed_random_equal_token": _with_tokens(raw_without_baseline, tokenizer),
    }
    available = {name: sum(int(row["target_token_count_with_eos"]) for row in rows) for name, rows in tokenized.items()}
    budget = min(available.values())
    equal_rows = {
        name: _stable(rows, 20260717, name) if name == "raw_mixed_random_equal_token" else rows
        for name, rows in tokenized.items()
    }
    arm_dir = output_root / "arms"
    equal_report = {
        name: _write_jsonl(arm_dir / f"{name}.jsonl", [_audit_row(row, name, name, source_by_record) for row in rows])
        for name, rows in equal_rows.items()
    }
    natural_rows = {
        "raw_mixed_all_natural": _with_tokens(raw_without_baseline, tokenizer),
        "curated_natural": _with_tokens(curated, tokenizer),
    }
    natural_report = {
        name: _write_jsonl(arm_dir / f"{name}.jsonl", [_audit_row(row, name, name, source_by_record) for row in rows])
        for name, rows in natural_rows.items()
    }
    blocks_dir = output_root / "token_blocks"
    equal_blocks = {
        name: _pack_blocks(rows, tokenizer, blocks_dir / f"{name}.pt", budget)
        for name, rows in equal_rows.items()
    }
    natural_blocks = {
        name: _pack_blocks(
            rows,
            tokenizer,
            blocks_dir / f"{name}.pt",
            sum(int(row["target_token_count_with_eos"]) for row in rows),
        )
        for name, rows in natural_rows.items()
    }
    holdout_contract = contract["holdout_contract"]
    holdouts = {
        name: _holdout(_read_jsonl(Path(spec["source"])), name, output_root / "holdouts" / name)
        for name, spec in holdout_contract.items()
    }
    report = {
        "schema_version": "raw-corpus-matrix-stage-c-inputs-v1",
        "status": "raw_corpus_matrix_stage_c_inputs_frozen",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "target_model": contract["target_model"],
        "tokenizer": {"name_or_path": str(tokenizer.name_or_path), "vocab_size": int(tokenizer.vocab_size)},
        "equal_token_budget_target_tokens_with_eos": budget,
        "equal_token_arms": {**equal_report, "curated_common_baseline_overlap_count": 0},
        "natural_budget_arms": natural_report,
        "token_blocks": {
            "equal_token_blocks": equal_blocks,
            "natural_budget_blocks": natural_blocks,
            "common_packed_token_budget": equal_blocks["curated_equal_token"]["packed_tokens"],
        },
        "common_stage_a_baseline_sha256": equal_report["stage_a_random_equal_token"]["sha256"],
        "raw_random_excluded_common_baseline_count": len(raw_chunks) - len(raw_without_baseline),
        "raw_unchunkable_records": raw_unchunkable,
        "holdouts": holdouts,
        "stage_b_isolation": contract["stage_b_isolation"],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Frozen input construction only; no model execution or downstream outcome claim.",
    }
    save_json(output_root / "raw_corpus_matrix_stage_c_inputs_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    report = prepare(args.contract, args.output_root, allow_download=bool(args.allow_download))
    print({"status": report["status"], "equal_token_budget": report["equal_token_budget_target_tokens_with_eos"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
