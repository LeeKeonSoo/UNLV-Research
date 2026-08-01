from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Final, Literal, assert_never


CoherenceGuardOutcome = Literal["guard_passed", "explicit_defect"]
CoherenceGuardVersion = Literal["v1", "v2"]
REPLACEMENT_BURST_MINIMUM: Final = 3
UNMATCHED_DELIMITER_MINIMUM: Final = 3
DELIMITER_PAIRS: Final = (("(", ")"), ("[", "]"), ("{", "}"))
LATEX_BEGIN_RE: Final = re.compile(r"\\begin\{([^{}]+)\}")
LATEX_END_RE: Final = re.compile(r"\\end\{([^{}]+)\}")
XML_TAG_RE: Final = re.compile(r"<(/?)([A-Za-z][\w:.-]*)(?:\s[^<>]*)?(/?)>")
FENCE_RE: Final = re.compile(r"(?m)^\s*(```|~~~)")
HTML_VOID_ELEMENTS: Final = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
EXPLICIT_XML_ELEMENTS: Final = frozenset(
    {
        "article", "blockquote", "body", "code", "div", "h1", "h2", "h3", "h4", "h5", "h6", "html",
        "li", "math", "ol", "p", "pre", "proof", "section", "span", "table", "td", "theorem", "tr", "ul",
    }
)


@dataclass(frozen=True, slots=True)
class ExplicitCoherenceEvidence:
    outcome: CoherenceGuardOutcome
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExplicitCoherenceCorruption:
    corruption_id: str
    text: str


def explicit_coherence_evidence(
    text: str, version: CoherenceGuardVersion = "v2"
) -> ExplicitCoherenceEvidence:
    """Return the frozen explicit structural-coherence guard outcome."""
    visible_text = _strip_latex_comments(text)
    reasons: list[str] = []
    if text.count("\ufffd") >= REPLACEMENT_BURST_MINIMUM:
        reasons.append("coherence_unicode_replacement_burst")
    if any(unicodedata.category(character) == "Cc" and character not in "\t\n\r" for character in text):
        reasons.append("coherence_forbidden_control_character")
    if _latex_environment_mismatch(visible_text, version):
        reasons.append("coherence_unmatched_latex_environment")
    if _xml_tag_mismatch(visible_text, version):
        reasons.append("coherence_unmatched_explicit_xml_tag")
    if len(FENCE_RE.findall(text)) % 2:
        reasons.append("coherence_dangling_markdown_fence")
    if _delimiter_damage(visible_text, version):
        reasons.append("coherence_repeated_delimiter_damage")
    return ExplicitCoherenceEvidence("explicit_defect" if reasons else "guard_passed", tuple(reasons))


def _strip_latex_comments(text: str) -> str:
    visible_lines = []
    for line in text.splitlines():
        comment_at = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment_at = index
                break
        visible_lines.append(line[:comment_at])
    return "\n".join(visible_lines)


def _latex_environment_mismatch(text: str, version: CoherenceGuardVersion) -> bool:
    begins, ends = Counter(LATEX_BEGIN_RE.findall(text)), Counter(LATEX_END_RE.findall(text))
    match version:
        case "v1":
            return begins != ends
        case "v2":
            return any(begins[name] > ends[name] for name in begins)
        case unreachable:
            assert_never(unreachable)


def _xml_tag_mismatch(text: str, version: CoherenceGuardVersion) -> bool:
    opens: Counter[str] = Counter()
    closes: Counter[str] = Counter()
    match version:
        case "v1":
            allowed_names: frozenset[str] | None = None
        case "v2":
            allowed_names = EXPLICIT_XML_ELEMENTS
        case unreachable:
            assert_never(unreachable)
    for slash, name, self_closing in XML_TAG_RE.findall(text):
        normalized = name.casefold()
        if self_closing or normalized in HTML_VOID_ELEMENTS:
            continue
        if allowed_names is not None and normalized not in allowed_names:
            continue
        (closes if slash else opens)[normalized] += 1
    match version:
        case "v1":
            return opens != closes
        case "v2":
            return any(opens[name] > closes[name] for name in opens)
        case unreachable:
            assert_never(unreachable)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _unmatched_delimiter_count(text: str) -> int:
    counts: Counter[str] = Counter(
        character
        for index, character in enumerate(text)
        if character in "()[]{}" and not _is_escaped(text, index)
    )
    return sum(abs(counts[opening] - counts[closing]) for opening, closing in DELIMITER_PAIRS)


def _trailing_opening_delimiters(text: str) -> int:
    count = 0
    for index in range(len(text) - 1, -1, -1):
        character = text[index]
        if character.isspace():
            continue
        if character in "([{" and not _is_escaped(text, index):
            count += 1
            continue
        break
    return count


def _delimiter_damage(text: str, version: CoherenceGuardVersion) -> bool:
    match version:
        case "v1":
            return _unmatched_delimiter_count(text) >= UNMATCHED_DELIMITER_MINIMUM
        case "v2":
            return _trailing_opening_delimiters(text) >= UNMATCHED_DELIMITER_MINIMUM
        case unreachable:
            assert_never(unreachable)


def explicit_coherence_corruptions(text: str) -> tuple[ExplicitCoherenceCorruption, ...]:
    """Create only corruption families with an observable text-local witness."""
    midpoint = len(text) // 2
    return (
        ExplicitCoherenceCorruption("unicode_replacement_burst", text[:midpoint] + "\ufffd\ufffd\ufffd" + text[midpoint:]),
        ExplicitCoherenceCorruption("forbidden_control_character", text[:midpoint] + "\x00" + text[midpoint:]),
        ExplicitCoherenceCorruption("unmatched_latex_environment", "\\begin{proof}\n" + text),
        ExplicitCoherenceCorruption("unmatched_explicit_xml_tag", "<theorem>" + text),
        ExplicitCoherenceCorruption("dangling_markdown_fence", text + "\n```text\nincomplete"),
        ExplicitCoherenceCorruption("repeated_delimiter_damage", text + "\n((( [[ {{{"),
    )
