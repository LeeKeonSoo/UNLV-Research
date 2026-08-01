from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

from explicit_structural_coherence import explicit_coherence_evidence


ValidityAction = Literal["pass", "repair", "rechunk", "quarantine", "reject"]
InterventionAction = Literal["repair", "rechunk", "quarantine", "reject"]


@dataclass(frozen=True, slots=True)
class ValidityUnit:
    text: str
    source_record_text: str | None = None


@dataclass(frozen=True, slots=True)
class ControlRepair:
    text: str
    transformation_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidityDecision:
    final_action: ValidityAction
    action_trace: tuple[InterventionAction, ...]
    reason_codes: tuple[str, ...]
    transformation_codes: tuple[str, ...]
    original_text: str
    recovered_text: str
    original_sha256: str
    recovered_sha256: str
    source_record_sha256: str | None
    authority: str = "candidate_validity_only"
    may_select_for_training: bool = False


@dataclass(frozen=True, slots=True)
class DecisionDraft:
    final_action: ValidityAction
    action_trace: tuple[InterventionAction, ...]
    reason_codes: tuple[str, ...]


COHERENCE_REASON_MAP: Final = {
    "coherence_unicode_replacement_burst": "validity_unicode_replacement_burst",
    "coherence_forbidden_control_character": "validity_forbidden_control_character",
    "coherence_unmatched_latex_environment": "validity_unclosed_latex_environment",
    "coherence_unmatched_explicit_xml_tag": "validity_unclosed_explicit_xml_tag",
    "coherence_dangling_markdown_fence": "validity_dangling_markdown_fence",
    "coherence_repeated_delimiter_damage": "validity_trailing_delimiter_damage",
}
STRUCTURAL_REASONS: Final = frozenset(
    {
        "validity_unclosed_latex_environment",
        "validity_unclosed_explicit_xml_tag",
        "validity_dangling_markdown_fence",
        "validity_trailing_delimiter_damage",
    }
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _contextual_cp1252_replacement(text: str, index: int) -> str | None:
    character = text[index]
    left = text[index - 1] if index else ""
    right = text[index + 1] if index + 1 < len(text) else ""
    if character == "\u0092" and left.isalnum() and right.isalnum():
        return "\u2019"
    if character == "\u0091" and right.isalnum() and (not left or left.isspace() or left in "([{\""):
        return "\u2018"
    if character == "\u0093" and right.isalnum() and (not left or left.isspace() or left in "([{\""):
        return "\u201c"
    if character == "\u0094" and left.isalnum() and (not right or right.isspace() or right in ".,;:!?)]}"):
        return "\u201d"
    if character in {"\u0096", "\u0097"} and left.isspace() and right.isspace():
        return "\u2013" if character == "\u0096" else "\u2014"
    return None


def _repair_contextual_controls(text: str) -> ControlRepair:
    characters: list[str] = []
    transformations: list[str] = []
    for index, character in enumerate(text):
        replacement = _contextual_cp1252_replacement(text, index)
        if replacement is None:
            characters.append(character)
            continue
        characters.append(replacement)
        transformations.append(f"cp1252_c1_{ord(character):04x}_to_{ord(replacement):04x}")
    return ControlRepair("".join(characters), tuple(sorted(set(transformations))))


def _has_interpretable_payload(text: str) -> bool:
    return any(
        not character.isspace()
        and character != "\ufffd"
        and unicodedata.category(character) != "Cc"
        for character in text
    )


def _mapped_defects(text: str) -> tuple[str, ...]:
    evidence = explicit_coherence_evidence(text)
    return tuple(COHERENCE_REASON_MAP[reason] for reason in evidence.reason_codes)


def _source_context_is_structurally_complete(source_record_text: str | None) -> bool:
    if source_record_text is None:
        return False
    repaired = _repair_contextual_controls(source_record_text).text
    return _has_interpretable_payload(repaired) and not _mapped_defects(repaired)


def _decision(
    unit: ValidityUnit,
    recovered: ControlRepair,
    draft: DecisionDraft,
) -> ValidityDecision:
    source_hash = _sha256(unit.source_record_text) if unit.source_record_text is not None else None
    return ValidityDecision(
        final_action=draft.final_action,
        action_trace=draft.action_trace,
        reason_codes=tuple(dict.fromkeys(draft.reason_codes)),
        transformation_codes=recovered.transformation_codes,
        original_text=unit.text,
        recovered_text=recovered.text,
        original_sha256=_sha256(unit.text),
        recovered_sha256=_sha256(recovered.text),
        source_record_sha256=source_hash,
    )


def evaluate_validity(unit: ValidityUnit) -> ValidityDecision:
    recovered = _repair_contextual_controls(unit.text)
    repaired = bool(recovered.transformation_codes)
    repair_reasons = ("validity_cp1252_punctuation_repaired",) if repaired else ()
    if not _has_interpretable_payload(recovered.text):
        return _decision(
            unit,
            recovered,
            DecisionDraft(
                "reject",
                (("repair",) if repaired else ()) + ("reject",),
                repair_reasons + ("validity_payload_absence_after_recovery",),
            ),
        )
    defects = _mapped_defects(recovered.text)
    if not defects:
        return _decision(
            unit,
            recovered,
            DecisionDraft(
                "repair" if repaired else "pass",
                ("repair",) if repaired else (),
                repair_reasons,
            ),
        )
    only_structural = set(defects) <= STRUCTURAL_REASONS
    if only_structural and _source_context_is_structurally_complete(unit.source_record_text):
        return _decision(
            unit,
            recovered,
            DecisionDraft(
                "rechunk",
                (("repair",) if repaired else ()) + ("rechunk",),
                repair_reasons + defects + ("validity_chunk_boundary_structural_split",),
            ),
        )
    return _decision(
        unit,
        recovered,
        DecisionDraft(
            "quarantine",
            (("repair",) if repaired else ()) + ("quarantine",),
            repair_reasons + defects,
        ),
    )
