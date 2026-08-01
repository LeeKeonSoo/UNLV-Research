#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TextIO


DCLM_ID: Final = "mlfoundations/fasttext-oh-eli5"
FINEWEB_ID: Final = "HuggingFaceFW/fineweb-edu-classifier"


@dataclass(frozen=True, slots=True)
class ProviderScoreError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class NativeFastText(Protocol):
    def predict(
        self,
        text: str,
        k: int,
        threshold: float,
        on_unicode_error: str,
    ) -> list[tuple[float, str]]: ...


class FastTextModel(Protocol):
    f: NativeFastText


def normalize_fasttext_document(text: str) -> str:
    return " ".join(text.split())


def dclm_high_quality_probability(
    labels: tuple[str, ...],
    probabilities: tuple[float, ...],
) -> float:
    if len(labels) != len(probabilities):
        raise ValueError("DCLM labels and probabilities must have equal length")
    probability_by_label = dict(zip(labels, probabilities, strict=True))
    if "__label__hq" not in probability_by_label:
        raise ValueError("DCLM output must include __label__hq")
    score = float(probability_by_label["__label__hq"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("DCLM high-quality probability must be finite and in [0, 1]")
    return score


def predict_dclm_labels(model: FastTextModel, text: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    predictions = model.f.predict(f"{text}\n", -1, 0.0, "strict")
    return (
        tuple(label for _, label in predictions),
        tuple(float(probability) for probability, _ in predictions),
    )


def fineweb_regression_score(logits: tuple[float, ...]) -> float:
    if len(logits) != 1:
        raise ValueError("FineWeb-Edu must emit one regression logit")
    score = float(logits[0])
    if not math.isfinite(score):
        raise ValueError("FineWeb-Edu regression score must be finite")
    return score


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_fasttext_document(text).encode("utf-8")).hexdigest()


def _load_rows(source: TextIO, id_field: str, text_field: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(source, start=1):
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ProviderScoreError(f"JSONL line {line_number} must be an object")
        uid = raw.get(id_field)
        text = raw.get(text_field)
        if not isinstance(uid, str) or not uid or not isinstance(text, str) or not text:
            raise ProviderScoreError(f"JSONL line {line_number} requires non-empty {id_field} and {text_field}")
        rows.append((uid, text))
    return rows


def _base_record(uid: str, text: str, provider_id: str, revision: str) -> dict[str, str | int | float | bool]:
    return {
        "schema_version": "general-provider-candidate-score-v2",
        "chunk_uid": uid,
        "provider_id": provider_id,
        "provider_revision": revision,
        "evidence_head": "route_specific_evidence",
        "normalized_text_sha256": _normalized_hash(text),
        "character_count": len(text),
        "complete_quality_bundle": False,
        "runtime_authority": False,
    }


def _score_dclm(rows: list[tuple[str, str]], target: TextIO, model_path: Path, revision: str) -> None:
    import fasttext

    model = fasttext.load_model(str(model_path))
    for uid, text in rows:
        normalized = normalize_fasttext_document(text)
        labels, probabilities = predict_dclm_labels(model, normalized)
        record = _base_record(uid, text, DCLM_ID, revision)
        record.update(
            {
                "score": dclm_high_quality_probability(tuple(labels), tuple(map(float, probabilities))),
                "score_scale": "probability_of_oh2.5_or_reddit_eli5_reference_distribution",
                "scored_tokens": len(normalized.split()),
                "truncated": False,
            }
        )
        target.write(json.dumps(record, ensure_ascii=False) + "\n")


def _score_fineweb(
    rows: list[tuple[str, str]],
    target: TextIO,
    model_path: Path,
    revision: str,
    device: str,
    batch_size: int,
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    maximum_tokens = min(512, int(getattr(model.config, "max_position_embeddings", 512)))
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            encoded = tokenizer(
                [text for _, text in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=maximum_tokens,
            )
            token_counts = encoded["attention_mask"].sum(dim=1).tolist()
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            logits = model(**encoded).logits.float().cpu().tolist()
            for (uid, text), values, token_count in zip(batch, logits, token_counts, strict=True):
                full_token_count = len(
                    tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
                )
                record = _base_record(uid, text, FINEWEB_ID, revision)
                record.update(
                    {
                        "score": fineweb_regression_score(tuple(map(float, values))),
                        "score_scale": "llama3_annotated_educational_value_regression_0_to_5",
                        "scored_tokens": int(token_count),
                        "truncated": full_token_count > maximum_tokens,
                    }
                )
                target.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score frozen General controls with one candidate provider.")
    parser.add_argument("--provider", choices=("dclm_fasttext", "fineweb_edu"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--id-field", default="chunk_uid")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as source:
        rows = _load_rows(source, args.id_field, args.text_field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        match args.provider:
            case "dclm_fasttext":
                _score_dclm(rows, target, args.model, args.provider_revision)
            case "fineweb_edu":
                _score_fineweb(rows, target, args.model, args.provider_revision, args.device, args.batch_size)
            case unreachable:
                raise ProviderScoreError(f"Unsupported provider: {unreachable}")
    print(json.dumps({"status": "complete", "provider": args.provider, "records_scored": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
