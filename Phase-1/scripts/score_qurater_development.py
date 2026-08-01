#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO, TypedDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_taxonomy import classify_coverage


@dataclass(frozen=True, slots=True)
class ProviderScoreError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class QuRaterScores:
    writing_style: float
    required_expertise: float
    facts_trivia: float
    educational_value: float


class ScoreMetadata(TypedDict):
    normalized_text_sha256: str
    general_informational_prose: bool


def score_metadata(text: str) -> ScoreMetadata:
    normalized = " ".join(text.split())
    annotation = classify_coverage(text)
    return {
        "normalized_text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "general_informational_prose": (
            "general_knowledge" in annotation["semantic_domain"]["labels"]
            and "prose" in annotation["format_genre"]["labels"]
        ),
    }


def aggregate_window_scores(
    windows: tuple[QuRaterScores, ...],
    token_counts: tuple[int, ...],
) -> QuRaterScores:
    if not windows or len(windows) != len(token_counts):
        raise ProviderScoreError("Window scores require matching non-empty token counts")
    if any(count <= 0 for count in token_counts):
        raise ProviderScoreError("Window token counts must be positive")
    rows = tuple(tuple(asdict(window).values()) for window in windows)
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ProviderScoreError("QuRater logits must be finite")
    total = sum(token_counts)
    weighted = tuple(
        sum(row[index] * count for row, count in zip(rows, token_counts, strict=True)) / total
        for index in range(4)
    )
    return QuRaterScores(*weighted)


def _score_text(text: str, tokenizer, model, torch_module, device: str) -> tuple[QuRaterScores, int, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ProviderScoreError("QuRater cannot score empty token input")
    windows = tuple(token_ids[offset : offset + 512] for offset in range(0, len(token_ids), 512))
    scores: list[QuRaterScores] = []
    with torch_module.inference_mode():
        for window in windows:
            input_ids = torch_module.tensor([window], dtype=torch_module.long, device=device)
            attention_mask = torch_module.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0].float().cpu().tolist()
            if len(logits) != 4:
                raise ProviderScoreError("QuRater must emit exactly four logits")
            scores.append(QuRaterScores(*(float(value) for value in logits)))
    return aggregate_window_scores(tuple(scores), tuple(map(len, windows))), len(windows), len(token_ids)


def _score_record(
    row: dict[str, str],
    tokenizer,
    model,
    torch_module,
    args: argparse.Namespace,
) -> dict[str, str | int | dict[str, float]]:
    uid = row.get(args.id_field)
    text = row.get(args.text_field)
    if not uid or not text:
        raise ProviderScoreError(f"Input requires non-empty {args.id_field} and {args.text_field}")
    scores, window_count, token_count = _score_text(text, tokenizer, model, torch_module, args.device)
    metadata = score_metadata(text)
    return {
        "schema_version": "qurater-development-score-v1",
        "chunk_uid": uid,
        "provider_id": "princeton-nlp/QuRater-1.3B",
        "provider_revision": args.provider_revision,
        "score_scale": "provider_native_raw_logit",
        "window_count": window_count,
        "scored_tokens": token_count,
        "normalized_text_sha256": metadata["normalized_text_sha256"],
        "general_informational_prose": int(metadata["general_informational_prose"]),
        "raw_scores": asdict(scores),
    }


def _write_scores(source: TextIO, target: TextIO, tokenizer, model, torch_module, args: argparse.Namespace) -> int:
    written = 0
    for line_number, line in enumerate(source, start=1):
        if args.limit is not None and written >= args.limit:
            break
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ProviderScoreError(f"JSONL line {line_number} must be an object")
        row = {str(key): str(value) for key, value in raw.items() if isinstance(value, str)}
        target.write(json.dumps(_score_record(row, tokenizer, model, torch_module, args), ensure_ascii=False) + "\n")
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Score development JSONL with frozen QuRater raw logits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="chunk_uid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        written = _write_scores(source, target, tokenizer, model, torch, args)
    print(json.dumps({"status": "complete", "records_scored": written, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
