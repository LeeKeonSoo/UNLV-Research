#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_adapters import ChatMessage, CompletionRequest, StructuredResponseFormat
from quality_teacher_openai_batch import (
    BatchResultError,
    build_batch_request,
    extract_batch_output_text,
    summarize_luna_batch_usage,
)


def _request() -> CompletionRequest:
    return CompletionRequest(
        model_id="gpt-5.6-luna",
        messages=(
            ChatMessage(role="system", content="Return JSON only."),
            ChatMessage(role="user", content='{"units":[]}'),
        ),
        maximum_new_tokens=4096,
        response_format=StructuredResponseFormat("json_object", ("payload_present",)),
    )


def test_batch_request_targets_responses_api_without_synchronous_fallback() -> None:
    row = build_batch_request(
        custom_id="calibration-p1-000001",
        request=_request(),
        reasoning_effort="low",
    )

    assert row == {
        "custom_id": "calibration-p1-000001",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5.6-luna",
            "instructions": "Return JSON only.",
            "input": 'Return a json object only.\n{"units":[]}',
            "max_output_tokens": 4096,
            "reasoning": {"effort": "low"},
            "text": {
                "format": {"type": "json_object"},
                "verbosity": "low",
            },
            "store": False,
        },
    }


def test_batch_request_uses_strict_json_schema_when_supplied() -> None:
    schema = {
        "type": "object",
        "properties": {"units": {"type": "array", "items": {"type": "string"}}},
        "required": ["units"],
        "additionalProperties": False,
    }
    request = _request()
    strict_request = CompletionRequest(
        model_id=request.model_id,
        messages=request.messages,
        maximum_new_tokens=request.maximum_new_tokens,
        response_format=StructuredResponseFormat(
            "json_object",
            request.response_format.allowed_reason_codes,
            json_schema_name="quality_policy_votes",
            json_schema=schema,
        ),
    )

    row = build_batch_request(
        custom_id="calibration-p1-000002",
        request=strict_request,
        reasoning_effort="low",
    )

    assert row["body"]["text"]["format"] == {
        "type": "json_schema",
        "name": "quality_policy_votes",
        "strict": True,
        "schema": schema,
    }


def test_batch_result_extracts_output_text_from_responses_body() -> None:
    row = {
        "custom_id": "calibration-p1-000001",
        "response": {
            "status_code": 200,
            "body": {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"units":[]}'},
                        ],
                    },
                ],
            },
        },
        "error": None,
    }

    assert extract_batch_output_text(row) == '{"units":[]}'


def test_batch_result_rejects_failed_or_empty_rows() -> None:
    failed = {
        "custom_id": "failed",
        "response": {"status_code": 400, "body": {}},
        "error": {"code": "invalid_request"},
    }
    empty = {
        "custom_id": "empty",
        "response": {"status_code": 200, "body": {"output": []}},
        "error": None,
    }

    for row in (failed, empty):
        try:
            extract_batch_output_text(row)
        except BatchResultError as error:
            assert error.custom_id == row["custom_id"]
        else:
            raise AssertionError("Invalid Batch output must not become teacher evidence")


def test_luna_batch_usage_summary_separates_cached_tokens_and_estimates_cost() -> None:
    rows = [
        {
            "response": {
                "body": {
                    "usage": {
                        "input_tokens": 1_000,
                        "output_tokens": 200,
                        "input_tokens_details": {"cached_tokens": 100},
                    }
                }
            }
        }
    ]

    assert summarize_luna_batch_usage(rows) == {
        "request_count": 1,
        "input_tokens": 1_000,
        "cached_input_tokens": 100,
        "uncached_input_tokens": 900,
        "output_tokens": 200,
        "estimated_batch_cost_usd": 0.001055,
        "price_usd_per_million_uncached_input": 0.5,
        "price_usd_per_million_cached_input": 0.05,
        "price_usd_per_million_output": 3.0,
    }


if __name__ == "__main__":
    test_batch_request_targets_responses_api_without_synchronous_fallback()
    test_batch_request_uses_strict_json_schema_when_supplied()
    test_batch_result_extracts_output_text_from_responses_body()
    test_batch_result_rejects_failed_or_empty_rows()
    test_luna_batch_usage_summary_separates_cached_tokens_and_estimates_cost()
    print("[quality-teacher-openai-batch-v1] Batch request/result contract: pass")
