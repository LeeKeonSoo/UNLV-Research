from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum

from validity_recovery import ValidityDecision as RecoveryDecision
from validity_recovery import ValidityUnit as RecoveryUnit
from validity_recovery import evaluate_validity


BINARY_MAGIC_PREFIXES = (
    b"\x89PNG\r\n\x1a\n",
    b"\x7fELF",
    b"MZ",
    b"PK\x03\x04",
    b"GIF87a",
    b"GIF89a",
    b"\xff\xd8\xff",
    b"%PDF-",
)


class ValidityContractError(RuntimeError):
    """Raised when a Validity input cannot be represented unambiguously."""


class ValidityStatus(str, Enum):
    VALID = "valid"
    VALID_AFTER_REVERSIBLE_REPAIR = "valid_after_reversible_repair"
    QUARANTINE = "quarantine"
    INVALID = "invalid"


class ValidityAction(str, Enum):
    PASS = "pass"
    REPAIR = "repair"
    RECHUNK = "rechunk"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class TextField:
    name: str
    text: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidityContractError("Text field names must be non-empty")


@dataclass(frozen=True, slots=True)
class ValidityInput:
    text_fields: tuple[TextField, ...] = ()
    raw_bytes: bytes | None = None
    declared_encoding: str = "utf-8"
    source_record_text: str | None = None

    def __post_init__(self) -> None:
        if self.raw_bytes is not None and self.text_fields:
            raise ValidityContractError("Use text fields or raw bytes, never both")
        if not self.declared_encoding:
            raise ValidityContractError("declared_encoding must be non-empty")

    @classmethod
    def from_text(cls, text: str, *, source_record_text: str | None = None) -> "ValidityInput":
        return cls(text_fields=(TextField("text", text),), source_record_text=source_record_text)


@dataclass(frozen=True, slots=True)
class TransformationTrace:
    code: str
    input_sha256: str
    output_sha256: str
    reversible: bool


@dataclass(frozen=True, slots=True)
class ValidityV2Decision:
    status: ValidityStatus
    action: ValidityAction
    reason_codes: tuple[str, ...]
    transformation_trace: tuple[TransformationTrace, ...]
    original_text_fields: tuple[TextField, ...]
    original_field_hashes: tuple[tuple[str, str], ...]
    original_bytes: bytes | None
    original_bytes_sha256: str | None
    recovered_text: str
    recovered_sha256: str
    source_record_sha256: str | None
    training_eligible: bool
    requires_rechunk: bool
    authority: str = "validity_v2_candidate_only"
    provider_outputs_read: bool = False
    benchmark_outcomes_read: bool = False
    utility_read: bool = False


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _field_hashes(fields: tuple[TextField, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, _sha256_text(field.text)) for field in fields)


def _closed_decision(
    *,
    status: ValidityStatus,
    action: ValidityAction,
    reason: str,
    fields: tuple[TextField, ...] = (),
    raw_bytes: bytes | None = None,
) -> ValidityV2Decision:
    empty_hash = _sha256_text("")
    return ValidityV2Decision(
        status=status,
        action=action,
        reason_codes=(reason,),
        transformation_trace=(),
        original_text_fields=fields,
        original_field_hashes=_field_hashes(fields),
        original_bytes=raw_bytes,
        original_bytes_sha256=_sha256_bytes(raw_bytes) if raw_bytes is not None else None,
        recovered_text="",
        recovered_sha256=empty_hash,
        source_record_sha256=None,
        training_eligible=False,
        requires_rechunk=False,
    )


def _closed_binary_payload(payload: bytes) -> bool:
    if not payload:
        return False
    if payload.startswith(BINARY_MAGIC_PREFIXES):
        return True
    return not any(byte >= 128 or 32 <= byte <= 126 for byte in payload)


def _status_for(action: str) -> ValidityStatus:
    if action == "pass":
        return ValidityStatus.VALID
    if action in {"repair", "rechunk"}:
        return ValidityStatus.VALID_AFTER_REVERSIBLE_REPAIR
    if action == "quarantine":
        return ValidityStatus.QUARANTINE
    if action == "reject":
        return ValidityStatus.INVALID
    raise ValidityContractError(f"Unsupported recovery action: {action}")


def _action_for(action: str) -> ValidityAction:
    try:
        return ValidityAction(action)
    except ValueError as error:
        raise ValidityContractError(f"Unsupported recovery action: {action}") from error


def _adapt_recovery(
    recovery: RecoveryDecision,
    fields: tuple[TextField, ...],
    raw_bytes: bytes | None,
    prefix_reasons: tuple[str, ...] = (),
    prefix_trace: tuple[TransformationTrace, ...] = (),
) -> ValidityV2Decision:
    trace = list(prefix_trace)
    if recovery.transformation_codes:
        trace.extend(
            TransformationTrace(code, recovery.original_sha256, recovery.recovered_sha256, True)
            for code in recovery.transformation_codes
        )
    action = _action_for(recovery.final_action)
    return ValidityV2Decision(
        status=_status_for(recovery.final_action),
        action=action,
        reason_codes=tuple(dict.fromkeys((*prefix_reasons, *recovery.reason_codes))),
        transformation_trace=tuple(trace),
        original_text_fields=fields,
        original_field_hashes=_field_hashes(fields),
        original_bytes=raw_bytes,
        original_bytes_sha256=_sha256_bytes(raw_bytes) if raw_bytes is not None else None,
        recovered_text=recovery.recovered_text,
        recovered_sha256=recovery.recovered_sha256,
        source_record_sha256=recovery.source_record_sha256,
        training_eligible=action in {ValidityAction.PASS, ValidityAction.REPAIR},
        requires_rechunk=action is ValidityAction.RECHUNK,
    )


def _evaluate_text(text: str, unit: ValidityInput, prefix_reasons: tuple[str, ...] = (), prefix_trace: tuple[TransformationTrace, ...] = ()) -> ValidityV2Decision:
    recovery = evaluate_validity(RecoveryUnit(text=text, source_record_text=unit.source_record_text))
    fields = unit.text_fields or (TextField("decoded_bytes", text),)
    decision = _adapt_recovery(recovery, fields, unit.raw_bytes, prefix_reasons, prefix_trace)
    if prefix_trace and decision.action is ValidityAction.PASS:
        return replace(
            decision,
            status=ValidityStatus.VALID_AFTER_REVERSIBLE_REPAIR,
            action=ValidityAction.REPAIR,
            training_eligible=True,
        )
    return decision


def evaluate_validity_v2(unit: ValidityInput) -> ValidityV2Decision:
    if unit.raw_bytes is None:
        if len(unit.text_fields) != 1:
            reason = "validity_ambiguous_text_fields" if unit.text_fields else "validity_missing_text_field"
            return _closed_decision(status=ValidityStatus.INVALID, action=ValidityAction.REJECT, reason=reason, fields=unit.text_fields)
        return _evaluate_text(unit.text_fields[0].text, unit)
    payload = unit.raw_bytes
    if _closed_binary_payload(payload):
        return _closed_decision(status=ValidityStatus.INVALID, action=ValidityAction.REJECT, reason="validity_binary_payload", raw_bytes=payload)
    try:
        has_utf8_bom = payload.startswith(b"\xef\xbb\xbf") and unit.declared_encoding.casefold() in {"utf-8", "utf8"}
        text = payload.decode("utf-8-sig" if has_utf8_bom else unit.declared_encoding, errors="strict")
    except (UnicodeDecodeError, LookupError):
        return _closed_decision(status=ValidityStatus.QUARANTINE, action=ValidityAction.QUARANTINE, reason="validity_declared_decoding_failed", raw_bytes=payload)
    if not has_utf8_bom:
        return _evaluate_text(text, unit)
    raw_hash = _sha256_bytes(payload)
    recovered_hash = _sha256_text(text)
    trace = (TransformationTrace("utf8_bom_removal", raw_hash, recovered_hash, True),)
    return _evaluate_text(text, unit, ("validity_utf8_bom_repaired",), trace)
