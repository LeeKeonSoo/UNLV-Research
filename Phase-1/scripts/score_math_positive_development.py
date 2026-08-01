#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_positive_evidence import has_explicit_math_notation


def build_score_row(
    row: dict[str, Any],
    math_score: float,
    explicit_math_notation: bool,
    finemath_score: float,
    math_score_revision: str,
    finemath_revision: str,
) -> dict[str, Any]:
    record_id = row.get("record_id")
    token_count = row.get("token_count", row.get("token_proxy"))
    if not isinstance(record_id, str) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError("Every development row requires record_id and positive token count")
    return {
        "schema_version": "math-positive-development-score-v1",
        "record_id": record_id,
        "token_count": token_count,
        "status": "incomplete_candidate_evidence",
        "decision": "abstain",
        "reason_code": "math_positive_bundle_missing_structural_heads_abstain",
        "route_confidence": 1.0 if explicit_math_notation else float(math_score),
        "route_confidence_evidence": {
            "explicit_math_notation": explicit_math_notation,
            "mathscore_probability": float(math_score),
        },
        "substantive_payload": None,
        "coherence_completeness": None,
        "route_specific_evidence": float(finemath_score),
        "missing_heads": ["substantive_payload", "coherence_completeness"],
        "providers": {
            "route_confidence": {
                "provider_id": "open-web-math/MathScore",
                "provider_revision": math_score_revision,
            },
            "route_specific_evidence": {
                "provider_id": "HuggingFaceTB/finemath-classifier",
                "provider_revision": finemath_revision,
            },
        },
        "runtime_activation": False,
    }


def predict_positive_probability(model: Any, normalized_text: str) -> float:
    predictions = model.f.predict(normalized_text + "\n", 2, 0.0, "strict")
    return float(dict((label, probability) for probability, label in predictions).get("__label__positive", 0.0))


def load_official_math_score(model_path: Path, normalizer_path: Path) -> Callable[[str], float]:
    import fasttext

    spec = importlib.util.spec_from_file_location("open_web_math_text_normalizer", normalizer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen OpenWebMath normalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalize = module.normalize
    model = fasttext.load_model(str(model_path))

    def score(text: str) -> float:
        normalized = normalize(text).replace("\n", " ").replace("[EQUATION]", "")
        return predict_positive_probability(model, normalized)

    return score


class FineMathScorer:
    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self._model.to(self._device).eval()

    def score_many(self, texts: list[str]) -> list[float]:
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits.squeeze(-1).float().cpu()
        return [float(value) for value in logits]


def _batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Score incomplete Math positive-evidence provider heads.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--math-score-model", type=Path, required=True)
    parser.add_argument("--math-score-normalizer", type=Path, required=True)
    parser.add_argument("--math-score-revision", required=True)
    parser.add_argument("--finemath-model", type=Path, required=True)
    parser.add_argument("--finemath-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    with args.input.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    math_score = load_official_math_score(args.math_score_model, args.math_score_normalizer)
    finemath = FineMathScorer(args.finemath_model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.output.open("w", encoding="utf-8") as destination:
        for batch in _batches(rows, args.batch_size):
            texts = [str(row.get("text") or "") for row in batch]
            route_scores = [math_score(text) for text in texts]
            explicit_notation = [has_explicit_math_notation(text) for text in texts]
            usefulness_scores = finemath.score_many(texts)
            for row, route_score, notation, usefulness_score in zip(
                batch, route_scores, explicit_notation, usefulness_scores, strict=True
            ):
                output = build_score_row(
                    row,
                    route_score,
                    notation,
                    usefulness_score,
                    args.math_score_revision,
                    args.finemath_revision,
                )
                destination.write(json.dumps(output) + "\n")
            completed += len(batch)
            if completed % 320 == 0 or completed == len(rows):
                print(json.dumps({"scored": completed, "total": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
