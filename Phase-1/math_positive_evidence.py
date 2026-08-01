from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MathCandidateEvidence:
    route_confidence: float
    mathscore_probability: float
    explicit_math_notation: bool
    substantive_payload: None
    coherence_completeness: None
    route_specific_evidence: float
    can_emit_keep: bool = False


DISPLAY_MATH_RE = re.compile(
    r"\$\$.+?\$\$|\\\[.+?\\\]|\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}",
    re.DOTALL,
)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)([^$\n]{1,240})(?<!\\)\$(?!\$)")
MATHML_RE = re.compile(r"<math(?:\s|>)", re.IGNORECASE)
INLINE_MATH_OPERATOR_RE = re.compile(r"\\[A-Za-z]+|[=+*/^_<>]|[∑∫√≤≥≠∈∀∃]")
LATEX_COMMAND_RE = re.compile(r"\\(?:frac|sum|prod|int|sqrt|lim|log|sin|cos|tan|alpha|beta|gamma|theta)\b")
UNICODE_MATH_RE = re.compile(r"[∑∫√≤≥≠∈∀∃]")


def has_explicit_math_notation(text: str) -> bool:
    if DISPLAY_MATH_RE.search(text) or MATHML_RE.search(text) or LATEX_COMMAND_RE.search(text) or UNICODE_MATH_RE.search(text):
        return True
    return any(INLINE_MATH_OPERATOR_RE.search(match.group(1)) for match in INLINE_MATH_RE.finditer(text))


def build_math_candidate_evidence(
    text: str,
    relevance_scorer: Callable[[str], float],
    usefulness_scorer: Callable[[str], float],
) -> MathCandidateEvidence:
    """Build two independent provider heads without inventing missing evidence."""
    relevance = float(relevance_scorer(text))
    usefulness = float(usefulness_scorer(text))
    if not 0.0 <= relevance <= 1.0 or not math.isfinite(usefulness):
        raise ValueError("Math provider outputs must be finite and relevance must be within [0, 1]")
    explicit_notation = has_explicit_math_notation(text)
    return MathCandidateEvidence(
        1.0 if explicit_notation else relevance,
        relevance,
        explicit_notation,
        None,
        None,
        usefulness,
    )
