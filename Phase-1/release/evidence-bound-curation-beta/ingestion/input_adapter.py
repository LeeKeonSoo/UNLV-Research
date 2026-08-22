from __future__ import annotations

from collections.abc import Iterable
from typing import Any


JsonMap = dict[str, Any]
DEFAULT_TEXT_FIELDS = ("text", "content", "document", "body")


def _mapping(value: Any) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _text(raw: JsonMap, text_fields: Iterable[str]) -> tuple[Any, str | None]:
    populated = [
        (field, raw[field])
        for field in text_fields
        if field in raw and raw[field] is not None and raw[field] != ""
    ]
    if len(populated) > 1:
        names = ", ".join(field for field, _ in populated)
        raise RuntimeError(f"Input record must populate exactly one declared text field; found: {names}")
    if not populated:
        return "", None
    return populated[0][1], populated[0][0]


def _value(raw: JsonMap, nested: JsonMap, defaults: JsonMap, field: str) -> Any:
    return defaults.get(field) or nested.get(field) or raw.get(field)


def _rights(raw_rights: JsonMap, default_rights: JsonMap) -> JsonMap:
    rank = {"allowed": 0, "unknown": 1, "restricted": 2}
    default_status = str(default_rights.get("status") or "unknown")
    raw_status = str(raw_rights.get("status") or default_status)
    if default_status == "unknown":
        return {"status": raw_status, "license": raw_rights.get("license")}
    selected_status = raw_status if rank.get(raw_status, 1) > rank.get(default_status, 1) else default_status
    selected_license = raw_rights.get("license") if selected_status == raw_status else default_rights.get("license")
    return {"status": selected_status, "license": selected_license}


def _artifact_context(raw: JsonMap, defaults: JsonMap) -> JsonMap:
    """Preserve only source-declared artifact labels under their canonical names.

    A source may map its direct fields to the two canonical fields, but mappings
    are not heuristics: values pass through unchanged and the candidate contract
    validates them later.  A canonical record-level artifact_context takes
    precedence over a source mapping.
    """
    mapping = _mapping(defaults.get("artifact_context_fields"))
    mapped = {
        canonical: raw.get(str(source_field))
        for canonical, source_field in mapping.items()
        if canonical in {"generation", "dependency_copy"} and isinstance(source_field, str) and source_field in raw
    }
    return {
        **_mapping(defaults.get("artifact_context")),
        **mapped,
        **_mapping(raw.get("artifact_context")),
    }


def adapt_raw_record(raw: JsonMap, defaults: JsonMap, text_fields: Iterable[str], index: int) -> JsonMap:
    provenance = _mapping(raw.get("provenance"))
    default_provenance = _mapping(defaults.get("provenance"))
    provenance_defaults = {**defaults, **default_provenance}
    default_rights = _mapping(defaults.get("rights"))
    raw_rights = _mapping(raw.get("rights"))
    raw_partition = _mapping(raw.get("partition"))
    default_partition = _mapping(defaults.get("partition"))
    language = {**_mapping(defaults.get("language")), **_mapping(raw.get("language"))}
    artifact_context = _artifact_context(raw, defaults)
    text, selected_text_field = _text(raw, text_fields)
    pii_context = raw.get("pii_context") or defaults.get("pii_context") or "general"
    normalization_context = (
        raw.get("normalization_context")
        or defaults.get("normalization_context")
        or "preserve"
    )
    return {
        "record_id": str(raw.get("record_id") or raw.get("id") or raw.get("uid") or f"candidate-{index:06d}"),
        "text": text,
        "provenance": {
            "source_name": _value(raw, provenance, provenance_defaults, "source_name"),
            "source_uri": _value(raw, provenance, provenance_defaults, "source_uri"),
            "collected_at": _value(raw, provenance, provenance_defaults, "collected_at"),
        },
        "language": language or {"code": "und", "confidence": None},
        "artifact_context": artifact_context or None,
        "rights": _rights(raw_rights, default_rights),
        "pii_context": pii_context,
        "normalization_context": normalization_context,
        "input_adapter": {"selected_text_field": selected_text_field},
        "min_text_chars": raw.get("min_text_chars") or defaults.get("min_text_chars"),
        "partition": {**default_partition, **raw_partition} or None,
    }


def adapt_raw_records(raw_records: Iterable[JsonMap], input_config: JsonMap) -> list[JsonMap]:
    defaults = _mapping(input_config.get("defaults"))
    configured_fields = input_config.get("text_fields")
    text_fields = tuple(str(field) for field in configured_fields) if isinstance(configured_fields, list) else DEFAULT_TEXT_FIELDS
    return [adapt_raw_record(record, defaults, text_fields, index) for index, record in enumerate(raw_records)]
