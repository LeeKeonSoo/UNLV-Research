from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from quality_teacher_adapters import CompletionRequest
from quality_teacher_openai import ReasoningEffort, openai_text_format


LUNA_BATCH_UNCACHED_INPUT_USD_PER_MILLION = 0.5
LUNA_BATCH_CACHED_INPUT_USD_PER_MILLION = 0.05
LUNA_BATCH_OUTPUT_USD_PER_MILLION = 3.0


@dataclass(frozen=True, slots=True)
class BatchResultError(RuntimeError):
    custom_id: str
    detail: str

    def __str__(self) -> str:
        return f"OpenAI Batch result failed for {self.custom_id}: {self.detail}"


def build_batch_request(
    *,
    custom_id: str,
    request: CompletionRequest,
    reasoning_effort: ReasoningEffort,
) -> dict[str, object]:
    if not custom_id:
        raise ValueError("custom_id cannot be empty")
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": request.model_id,
            "instructions": request.messages[0].content,
            "input": f"Return a json object only.\n{request.messages[1].content}",
            "max_output_tokens": request.maximum_new_tokens,
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": openai_text_format(request.response_format),
                "verbosity": "low",
            },
            "store": False,
        },
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def extract_batch_output_text(row: Mapping[str, Any]) -> str:
    custom_id = str(row.get("custom_id") or "unknown")
    if row.get("error") is not None:
        raise BatchResultError(custom_id, "request_error")
    response = _mapping(row.get("response"))
    if int(response.get("status_code") or 0) != 200:
        raise BatchResultError(custom_id, f"http_status_{response.get('status_code')}")
    body = _mapping(response.get("body"))
    if body.get("status") not in (None, "completed"):
        raise BatchResultError(custom_id, f"response_status_{body.get('status')}")

    direct = str(body.get("output_text") or "").strip()
    if direct:
        return direct

    parts: list[str] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            content = _mapping(item).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_mapping = _mapping(block)
                if block_mapping.get("type") == "output_text":
                    text = str(block_mapping.get("text") or "").strip()
                    if text:
                        parts.append(text)
    combined = "".join(parts).strip()
    if not combined:
        raise BatchResultError(custom_id, "no_output_text")
    return combined


def summarize_luna_batch_usage(rows: list[Mapping[str, Any]]) -> dict[str, int | float]:
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    request_count = 0
    for row in rows:
        body = _mapping(_mapping(row.get("response")).get("body"))
        usage = _mapping(body.get("usage"))
        if not usage:
            continue
        request_count += 1
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        details = _mapping(usage.get("input_tokens_details"))
        cached_input_tokens += int(details.get("cached_tokens") or 0)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    cost = (
        uncached_input_tokens * LUNA_BATCH_UNCACHED_INPUT_USD_PER_MILLION
        + cached_input_tokens * LUNA_BATCH_CACHED_INPUT_USD_PER_MILLION
        + output_tokens * LUNA_BATCH_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return {
        "request_count": request_count,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "estimated_batch_cost_usd": round(cost, 9),
        "price_usd_per_million_uncached_input": (
            LUNA_BATCH_UNCACHED_INPUT_USD_PER_MILLION
        ),
        "price_usd_per_million_cached_input": LUNA_BATCH_CACHED_INPUT_USD_PER_MILLION,
        "price_usd_per_million_output": LUNA_BATCH_OUTPUT_USD_PER_MILLION,
    }


__all__ = [
    "BatchResultError",
    "build_batch_request",
    "extract_batch_output_text",
    "summarize_luna_batch_usage",
]
