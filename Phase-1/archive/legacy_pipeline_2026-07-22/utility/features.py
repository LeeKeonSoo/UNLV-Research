#!/usr/bin/env python3
"""Feature extraction for utility calibration and prediction."""

from __future__ import annotations

import re
from typing import Any, Dict

from data_eval_common import alpha_ratio, clamp01, repeated_token_ratio, sentence_count


DEFINITION_PATTERNS = (
    "is defined as",
    "refers to",
    "means that",
    "is the process of",
    "can be described as",
)
EXPLANATION_MARKERS = (
    "because",
    "therefore",
    "for example",
    "for instance",
    "as a result",
    "this means",
    "in order to",
)
QUESTION_MARKERS = ("question", "answer", "why", "how", "what is", "what are")
PROCEDURAL_MARKERS = (
    "click",
    "tap",
    "scroll",
    "sign up",
    "open the app",
    "go to the",
    "select the",
    "press the",
)


def _count_contains(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _bullet_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith(("-", "*")) or re.match(r"^\d+\.", line))
    return bullet_lines / len(lines)


def utility_feature_vector(text: str, quality_score: float, validity_score: float) -> Dict[str, float]:
    lowered = text.lower()
    words = text.split()
    n_words = max(len(words), 1)
    n_sent = max(sentence_count(text), 1)
    concept_ratio = sum(1 for w in words if len(w) > 6 and w.isalpha()) / n_words
    explanatory_signal = clamp01(_count_contains(lowered, EXPLANATION_MARKERS) / 4.0)
    definition_signal = clamp01(_count_contains(lowered, DEFINITION_PATTERNS) / 3.0)
    qa_signal = clamp01(_count_contains(lowered, QUESTION_MARKERS) / 4.0)
    procedural_penalty = clamp01(_count_contains(lowered, PROCEDURAL_MARKERS) / 6.0)
    glossary_penalty = 1.0 if lowered.strip().startswith("glossary") or "glossary:" in lowered else 0.0
    conclusion_penalty = 1.0 if lowered.strip().startswith("conclusion") or lowered.strip().startswith("in conclusion") else 0.0
    bullet_penalty = clamp01((_bullet_ratio(text) - 0.35) * 2.0) if _bullet_ratio(text) > 0.35 else 0.0
    list_density_penalty = clamp01((text.count(" - ") + text.count("\n- ")) / 8.0)
    return {
        "word_count": float(n_words),
        "sentence_count": float(n_sent),
        "alpha_ratio": float(alpha_ratio(text)),
        "repeated_token_ratio": float(repeated_token_ratio(text)),
        "quality_score": float(quality_score),
        "validity_score": float(validity_score),
        "concept_ratio": float(concept_ratio),
        "explanatory_signal": float(explanatory_signal),
        "definition_signal": float(definition_signal),
        "qa_signal": float(qa_signal),
        "procedural_penalty": float(procedural_penalty),
        "glossary_penalty": float(glossary_penalty),
        "conclusion_penalty": float(conclusion_penalty),
        "bullet_penalty": float(bullet_penalty),
        "list_density_penalty": float(list_density_penalty),
    }
