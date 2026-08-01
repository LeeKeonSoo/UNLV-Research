from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, TypedDict


TAXONOMY_VERSION: Final = "coverage-taxonomy-v1"
AXES: Final = ("semantic_domain", "language_script", "format_genre", "content_morphology")
UNKNOWN: Final = "unknown"


class AxisClassification(TypedDict):
    labels: list[str]
    status: Literal["classified", "unknown"]
    evidence: dict[str, list[str]]


class CoverageAnnotation(TypedDict):
    taxonomy_version: str
    authority: str
    semantic_domain: AxisClassification
    language_script: AxisClassification
    format_genre: AxisClassification
    content_morphology: AxisClassification
    unknown_axes: list[str]


@dataclass(frozen=True, slots=True)
class EvidenceRule:
    label: str
    patterns: tuple[re.Pattern[str], ...]
    minimum_hits: int


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


DOMAIN_RULES: Final = (
    EvidenceRule("code", (_compile(r"\b(?:def|class|import|return|function|const|let|public\s+class)\b"),), 2),
    EvidenceRule("math", (_compile(r"\b(?:theorem|lemma|proof|integral|derivative|equation|matrix|polynomial)\b"), _compile(r"\\(?:frac|sum|int|sqrt)\b")), 2),
    EvidenceRule("science", (_compile(r"\b(?:experiment|hypothesis|molecule|chemical|physics|density|crystal|species|laboratory)\b"),), 2),
    EvidenceRule("medicine", (_compile(r"\b(?:patient|diagnosis|treatment|clinical|disease|symptom|therapy)\b"), _compile(r"(?:환자|진단|치료|질환|증상|약물)")), 2),
    EvidenceRule("law", (_compile(r"\b(?:court|plaintiff|defendant|statute|regulation|jurisdiction|legislation)\b"),), 2),
    EvidenceRule("finance_economics", (_compile(r"\b(?:revenue|investment|inflation|monetary|fiscal|equity|dividend|economy)\b"),), 2),
    EvidenceRule("history_humanities", (_compile(r"\b(?:century|historical|philosophy|literary|civilization|archaeology|historiography)\b"),), 2),
)

HTML_RE: Final = _compile(r"<[/!]?[a-z][^>]*>")
SPEAKER_RE: Final = _compile(r"^[\w][\w ._-]{0,30}:\s+\S.*$")
QUESTION_RE: Final = _compile(r"^(?:q|question):\s+\S.*$")
ANSWER_RE: Final = _compile(r"^(?:a|answer):\s+\S.*$")
URL_RE: Final = _compile(r"^(?:https?://|www\.)\S+$")
LIST_RE: Final = _compile(r"^\s*(?:[-*+] |\d+[.)]\s+)\S+")
TABLE_RE: Final = _compile(r"^\s*\|.+\|\s*$")
PROCEDURE_RE: Final = _compile(r"\b(?:how to|step\s+\d+|first,?\s|next,?\s|finally,?\s|instructions?)\b")
REFERENCE_RE: Final = _compile(r"\b(?:references|bibliography|sources|appendix|table of contents|citation)\b")
EXPLANATION_RE: Final = _compile(r"\b(?:because|therefore|means|defined as|in other words|explains?)\b|(?:설명|때문|의미한다)")
ARGUMENT_RE: Final = _compile(r"\b(?:however|therefore|consequently|evidence|claim|counterargument)\b")
TRANSACTIONAL_RE: Final = _compile(r"\b(?:buy now|shop now|add to cart|get a quote|order now|book now)\b")
WORD_RE: Final = re.compile(r"[^\W_]+", re.UNICODE)


def _axis(labels: list[str], evidence: dict[str, list[str]]) -> AxisClassification:
    if not labels:
        return {"labels": [UNKNOWN], "status": "unknown", "evidence": {UNKNOWN: ["no_closed_evidence"]}}
    return {"labels": sorted(set(labels)), "status": "classified", "evidence": evidence}


def _rule_axis(text: str, rules: tuple[EvidenceRule, ...]) -> AxisClassification:
    labels: list[str] = []
    evidence: dict[str, list[str]] = {}
    for rule in rules:
        hits = sum(len(pattern.findall(text)) for pattern in rule.patterns)
        if hits >= rule.minimum_hits:
            labels.append(rule.label)
            evidence[rule.label] = [f"closed_marker_hits={hits}"]
    return _axis(labels, evidence)


def semantic_domain(text: str) -> AxisClassification:
    result = _rule_axis(text, DOMAIN_RULES)
    if result["status"] == "classified":
        return result
    words = WORD_RE.findall(text)
    sentences = len(re.findall(r"[.!?。！？](?:\s|$)", text))
    if len(words) >= 12 and sentences >= 2:
        return _axis(["general_knowledge"], {"general_knowledge": ["prose_words>=12", "sentences>=2"]})
    return result


def _script(character: str) -> str | None:
    codepoint = ord(character)
    if 0x0041 <= codepoint <= 0x024F:
        return "latin"
    if 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
        return "hangul"
    if 0x4E00 <= codepoint <= 0x9FFF:
        return "han"
    if 0x3040 <= codepoint <= 0x30FF:
        return "kana"
    if 0x0400 <= codepoint <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= codepoint <= 0x06FF:
        return "arabic"
    if 0x0900 <= codepoint <= 0x097F:
        return "devanagari"
    return None


def language_script(text: str) -> AxisClassification:
    counts: dict[str, int] = {}
    for character in text:
        script = _script(character)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    labels = [label for label, count in counts.items() if count >= 2]
    evidence = {label: [f"unicode_letters={counts[label]}"] for label in labels}
    return _axis(labels, evidence)


def format_genre(text: str) -> AxisClassification:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    labels: list[str] = []
    evidence: dict[str, list[str]] = {}
    code_hits = len(DOMAIN_RULES[0].patterns[0].findall(text))
    math_hits = sum(len(pattern.findall(text)) for pattern in DOMAIN_RULES[1].patterns)
    checks = (
        ("source_code", code_hits >= 2, f"code_marker_hits={code_hits}"),
        ("dialogue", sum(bool(SPEAKER_RE.fullmatch(line)) for line in lines) >= 3, "speaker_lines>=3"),
        ("question_answer", any(QUESTION_RE.fullmatch(line) for line in lines) and any(ANSWER_RE.fullmatch(line) for line in lines), "question_and_answer_markers"),
        ("table", sum(bool(TABLE_RE.fullmatch(line)) for line in lines) >= 2, "pipe_table_lines>=2"),
        ("list", sum(bool(LIST_RE.fullmatch(line)) for line in lines) >= 3, "list_item_lines>=3"),
        ("formula", math_hits >= 2, f"math_marker_hits={math_hits}"),
        ("markup", bool(HTML_RE.search(text)), "explicit_markup_tag"),
        ("link_directory", sum(bool(URL_RE.fullmatch(line)) for line in lines) >= 3, "standalone_url_lines>=3"),
        ("prose", len(WORD_RE.findall(text)) >= 12 and bool(re.search(r"[.!?。！？]", text)), "prose_words>=12_and_sentence_end"),
    )
    for label, matched, reason in checks:
        if matched:
            labels.append(label)
            evidence[label] = [reason]
    return _axis(labels, evidence)


def content_morphology(text: str, formats: AxisClassification) -> AxisClassification:
    labels: list[str] = []
    evidence: dict[str, list[str]] = {}
    checks = (
        ("explanation", bool(EXPLANATION_RE.search(text)), "explicit_explanation_marker"),
        ("procedure", bool(PROCEDURE_RE.search(text)), "explicit_procedure_marker"),
        ("reference", bool(REFERENCE_RE.search(text)), "explicit_reference_marker"),
        ("implementation", "source_code" in formats["labels"], "source_code_structure"),
        ("argument", len(ARGUMENT_RE.findall(text)) >= 2, "argument_markers>=2"),
        ("transactional_ui", bool(TRANSACTIONAL_RE.search(text)), "explicit_transaction_marker"),
    )
    for label, matched, reason in checks:
        if matched:
            labels.append(label)
            evidence[label] = [reason]
    return _axis(labels, evidence)


def classify_coverage(text: str) -> CoverageAnnotation:
    formats = format_genre(text)
    axes = {
        "semantic_domain": semantic_domain(text),
        "language_script": language_script(text),
        "format_genre": formats,
        "content_morphology": content_morphology(text, formats),
    }
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "authority": "audit_only",
        "semantic_domain": axes["semantic_domain"],
        "language_script": axes["language_script"],
        "format_genre": axes["format_genre"],
        "content_morphology": axes["content_morphology"],
        "unknown_axes": [axis for axis in AXES if axes[axis]["status"] == "unknown"],
    }
