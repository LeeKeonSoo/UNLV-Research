#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers"]
# ///
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_positive_evidence import inspect_python_complete_source


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ProviderScorer: TypeAlias = Callable[[str], float]


@dataclass(frozen=True, slots=True)
class StackEduScoreError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def build_score_row(
    row: dict[str, JsonValue],
    provider_scorer: ProviderScorer,
    provider_revision: str,
) -> dict[str, JsonValue]:
    """Build one candidate-only Code evidence row without reading source identity or path."""
    record_id = row.get("record_id")
    text = row.get("text")
    if not isinstance(record_id, str) or not record_id or not isinstance(text, str) or not text:
        raise StackEduScoreError("Every input requires non-empty record_id and text")
    language = _mapping(row.get("language"))
    language_code = language.get("code")
    declaration = language.get("declaration")
    record_shape = row.get("record_shape")
    structural = inspect_python_complete_source(
        text,
        language_code if isinstance(language_code, str) else "und",
        declaration if isinstance(declaration, str) else None,
        record_shape if isinstance(record_shape, str) else "unknown",
    )
    provider_score: float | None = None
    if (
        structural.status == "in_scope"
        and structural.substantive_payload == 1.0
        and structural.coherence_completeness == 1.0
    ):
        provider_score = provider_scorer(text)
        if not math.isfinite(provider_score):
            raise StackEduScoreError("Stack-Edu provider score must be finite")
    return {
        "schema_version": "stack-edu-python-development-score-v1",
        "record_id": record_id,
        "provider_id": "HuggingFaceTB/stack-edu-classifier-python",
        "provider_revision": provider_revision,
        "status": structural.status,
        "reason_code": structural.reason_code,
        "structural_heads": {
            "route_confidence": structural.route_confidence,
            "substantive_payload": structural.substantive_payload,
            "coherence_completeness": structural.coherence_completeness,
        },
        "route_specific_evidence": provider_score,
        "score_scale": "provider_native_raw_regression_score_0_to_5_annotation_target",
        "provider_context_tokens": 1024,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score declared complete Python sources with frozen Stack-Edu.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-revision", required=True)
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

    def provider_scorer(text: str) -> float:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(args.device)
        with torch.inference_mode():
            return float(model(**encoded).logits.squeeze().float().item())

    written = 0
    scored = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if args.limit is not None and written >= args.limit:
                break
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise StackEduScoreError("Every JSONL row must be an object")
            result = build_score_row(raw, provider_scorer, args.provider_revision)
            scored += result["route_specific_evidence"] is not None
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"status": "complete", "records_written": written, "provider_scored": scored}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
