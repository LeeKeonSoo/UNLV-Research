from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal, assert_never


LEXICAL_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.UNICODE)
TERMINAL_CHARACTERS = frozenset(".!?;:)]}>'\"")
DELIMITER_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))
LATEX_BEGIN_RE = re.compile(r"\\begin\{([^{}]+)\}")
LATEX_END_RE = re.compile(r"\\end\{([^{}]+)\}")
XML_TAG_RE = re.compile(r"<(/?)([A-Za-z][\w:.-]*)(?:\s[^<>]*)?(/?)>")
FENCE_RE = re.compile(r"(?m)^\s*(```|~~~)")
FeatureSchema = Literal["v1", "v2"]


@dataclass(frozen=True, slots=True)
class StructuralFeatures:
    log_characters: float
    lexical_tokens: int
    log_lexical_tokens: float
    unique_lexical_ratio: float
    line_count: int
    repeated_line_fraction: float
    replacement_character_ratio: float
    alphanumeric_character_ratio: float
    terminal_boundary: bool
    delimiter_balance: float
    log_sentence_count: float
    mean_sentence_tokens: float
    repeated_ngram_fraction: float
    boundary_completeness: float
    markup_pair_balance: float
    line_boundary_fraction: float

    def vector(self, schema: FeatureSchema = "v1") -> tuple[float, ...]:
        base = (
            self.log_characters,
            self.log_lexical_tokens,
            self.unique_lexical_ratio,
            math.log1p(self.line_count),
            self.repeated_line_fraction,
            self.replacement_character_ratio,
            self.alphanumeric_character_ratio,
            float(self.terminal_boundary),
            self.delimiter_balance,
            self.log_sentence_count,
            self.mean_sentence_tokens,
        )
        match schema:
            case "v1":
                return base
            case "v2":
                return base + (
                    self.repeated_ngram_fraction,
                    self.boundary_completeness,
                    self.markup_pair_balance,
                    self.line_boundary_fraction,
                )
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class CorruptedText:
    corruption_id: str
    text: str


def _lines(text: str) -> tuple[str, ...]:
    return tuple(line for raw in text.splitlines() if (line := " ".join(raw.split())))


def _delimiter_balance(text: str) -> float:
    observed = []
    for opening, closing in DELIMITER_PAIRS:
        left, right = text.count(opening), text.count(closing)
        if left + right:
            observed.append(1.0 - abs(left - right) / (left + right))
    return sum(observed) / len(observed) if observed else 1.0


def _counter_balance(left: Counter[str], right: Counter[str]) -> float | None:
    names = left.keys() | right.keys()
    if not names:
        return None
    total = sum(left[name] + right[name] for name in names)
    mismatch = sum(abs(left[name] - right[name]) for name in names)
    return 1.0 - mismatch / total


def _markup_pair_balance(text: str) -> float:
    observed: list[float] = []
    latex = _counter_balance(Counter(LATEX_BEGIN_RE.findall(text)), Counter(LATEX_END_RE.findall(text)))
    if latex is not None:
        observed.append(latex)
    xml_open: Counter[str] = Counter()
    xml_close: Counter[str] = Counter()
    for slash, name, self_closing in XML_TAG_RE.findall(text):
        if self_closing:
            continue
        (xml_close if slash else xml_open)[name.casefold()] += 1
    xml = _counter_balance(xml_open, xml_close)
    if xml is not None:
        observed.append(xml)
    fences = len(FENCE_RE.findall(text))
    if fences:
        observed.append(1.0 if fences % 2 == 0 else max(0.0, (fences - 1) / fences))
    return sum(observed) / len(observed) if observed else 1.0


def _repeated_ngram_fraction(words: tuple[str, ...], width: int = 3) -> float:
    if len(words) < width:
        return 0.0
    counts = Counter(tuple(words[index : index + width]) for index in range(len(words) - width + 1))
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / sum(counts.values())


def extract_structural_features(text: str) -> StructuralFeatures:
    normalized = "\n".join(_lines(text))
    words = tuple(token.casefold() for token in LEXICAL_RE.findall(normalized))
    lines = _lines(normalized)
    line_counts = Counter(line.casefold() for line in lines)
    repeated_fraction = max(line_counts.values()) / len(lines) if len(lines) >= 2 else 0.0
    characters = max(1, len(normalized))
    sentences = tuple(segment for segment in SENTENCE_RE.findall(normalized) if LEXICAL_RE.search(segment))
    first_character = normalized[0] if normalized else ""
    boundary_completeness = (
        float(bool(first_character) and not first_character.islower())
        + float(bool(normalized) and normalized[-1] in TERMINAL_CHARACTERS)
    ) / 2.0
    line_boundary_fraction = (
        sum(line[-1] in TERMINAL_CHARACTERS for line in lines) / len(lines) if lines else 0.0
    )
    return StructuralFeatures(
        math.log1p(len(normalized)),
        len(words),
        math.log1p(len(words)),
        len(set(words)) / len(words) if words else 0.0,
        len(lines),
        repeated_fraction,
        normalized.count("\ufffd") / characters,
        sum(character.isalnum() for character in normalized) / characters,
        bool(normalized) and normalized[-1] in TERMINAL_CHARACTERS,
        _delimiter_balance(normalized),
        math.log1p(len(sentences)),
        len(words) / len(sentences) if sentences else 0.0,
        _repeated_ngram_fraction(words),
        boundary_completeness,
        _markup_pair_balance(normalized),
        line_boundary_fraction,
    )


def _repeat_to_lexical_count(unit: str, target: int) -> str:
    unit_tokens = max(1, len(LEXICAL_RE.findall(unit)))
    repetitions = max(2, math.ceil(target / unit_tokens))
    return "\n".join(unit for _ in range(repetitions))


def payload_corruptions(text: str, stable_id: str) -> tuple[CorruptedText, ...]:
    """Create score-blind non-distinct payload controls at approximately matched scale."""
    lines = _lines(text)
    title = lines[0] if lines else "Untitled section"
    target = max(1, len(LEXICAL_RE.findall(text)))
    variants = (
        "Section overview. Continue reading. Navigation and related links.",
        "Chapter metadata. Table of contents. Previous section. Next section.",
        "Document heading. Reference index. Copyright and navigation information.",
    )
    variant_index = hashlib.sha256(stable_id.encode()).digest()[0] % len(variants)
    return (
        CorruptedText("repeated_title_shell", _repeat_to_lexical_count(title, target)),
        CorruptedText("matched_length_navigation_shell", _repeat_to_lexical_count(variants[variant_index], target)),
    )


def _cut_position(text: str, stable_id: str) -> int:
    fraction = 0.45 + hashlib.sha256(stable_id.encode()).digest()[1] / 2550.0
    return min(max(1, int(len(text) * fraction)), max(1, len(text) - 1))


def coherence_corruptions(text: str, stable_id: str) -> tuple[CorruptedText, ...]:
    """Create deterministic truncation, rotation, and delimiter-damage controls."""
    normalized = "\n".join(_lines(text))
    cut = _cut_position(normalized, stable_id)
    truncated = normalized[:cut].rstrip(".!?;:)]}>'\" ")
    rotated = (normalized[cut:] + " " + normalized[:cut]).strip(".!?;:)]}>'\" ")
    damaged = normalized.translate(str.maketrans("", "", ")] }".replace(" ", "")))
    if damaged == normalized:
        damaged = normalized + " ((( [[ {{"
    return (
        CorruptedText("mid_document_truncation", truncated),
        CorruptedText("mid_document_rotation", rotated),
        CorruptedText("closing_delimiter_damage", damaged),
    )
