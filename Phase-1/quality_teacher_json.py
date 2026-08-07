from __future__ import annotations

from json import JSONDecodeError, JSONDecoder
from typing import TypeVar

from pydantic import BaseModel, ValidationError


JsonModel = TypeVar("JsonModel", bound=BaseModel)


def parse_unique_json_model(raw: str, model_type: type[JsonModel]) -> JsonModel | None:
    """Extract exactly one contract-valid JSON object from a model response."""
    decoder = JSONDecoder()
    parsed: list[JsonModel] = []
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            _, end = decoder.raw_decode(raw, index)
        except JSONDecodeError:
            continue
        candidate = raw[index:end]
        try:
            parsed.append(model_type.model_validate_json(candidate))
        except ValidationError:
            continue
    return parsed[0] if len(parsed) == 1 else None


__all__ = ["parse_unique_json_model"]
