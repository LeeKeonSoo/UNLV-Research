from __future__ import annotations

import re
from typing import Final, Literal, TypedDict

from coverage_taxonomy import AxisClassification, format_genre, language_script, semantic_domain


ROUTER_VERSION: Final = "content-router-v2"
AxisStatus = Literal["classified", "mixed", "unknown", "out_of_distribution"]
RouteStatus = Literal["routed", "mixed", "unknown", "out_of_distribution"]
RouteConfidence = Literal["closed_evidence", "ambiguous_evidence", "none", "unsupported"]


class RouterAxis(TypedDict):
    labels: list[str]
    status: AxisStatus
    evidence: dict[str, list[str]]


class ContentRouting(TypedDict):
    router_version: str
    authority: str
    may_select_or_remove: bool
    may_assign_importance: bool
    route_labels: list[str]
    route_status: RouteStatus
    route_confidence: RouteConfidence
    evidence_codes: list[str]
    content_format: RouterAxis
    structural_state: RouterAxis
    language_script: RouterAxis
    semantic_domain: RouterAxis


FORMAT_ORDER: Final = (
    "prose",
    "source_code",
    "mathematical_notation",
    "table",
    "dialogue",
    "question_answer",
    "instruction",
    "log",
    "markup",
    "mixed",
    "unknown",
)
STRUCTURE_ORDER: Final = (
    "complete_document",
    "complete_artifact",
    "snippet",
    "template",
    "generated_artifact",
    "partial",
    "mixed",
    "unknown",
)
ROUTE_ORDER: Final = (
    "general_prose",
    "code_artifact",
    "mathematical_content",
    "technical_documentation",
    "conversation",
    "instruction",
    "table_structured_data",
    "mixed",
    "unknown",
)
FORMAT_MAP: Final = {
    "source_code": "source_code",
    "formula": "mathematical_notation",
    "table": "table",
    "dialogue": "dialogue",
    "question_answer": "question_answer",
    "markup": "markup",
    "prose": "prose",
}

STEP_LINE_RE: Final = re.compile(r"^\s*step\s+\d+\s*[:.)-]\s*\S+", re.IGNORECASE)
LOG_LINE_RE: Final = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\[(?:trace|debug|info|warn|error|fatal)\])",
    re.IGNORECASE,
)
TECH_DOC_RE: Final = re.compile(
    r"^\s*(?:api reference|parameters?|returns?|raises?|examples?)\s*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
SHEBANG_RE: Final = re.compile(r"^#!\s*/\S+", re.MULTILINE)
FULL_HTML_RE: Final = re.compile(r"<html\b[^>]*>.*</html\s*>", re.IGNORECASE | re.DOTALL)
SNIPPET_RE: Final = re.compile(r"(?:\[snippet\]|\bcode snippet\b|\bexcerpt\b|\.\.\.\s*omitted\s*\.\.\.)", re.IGNORECASE)
TEMPLATE_RE: Final = re.compile(r"(?:\{\{[^{}]+\}\}|\{%[^%]+%\}|<%[^%]+%>)")
GENERATED_RE: Final = re.compile(r"\b(?:generated|auto-generated)\b", re.IGNORECASE)
DO_NOT_EDIT_RE: Final = re.compile(r"\bdo not edit\b", re.IGNORECASE)
PARTIAL_RE: Final = re.compile(r"(?:\[(?:truncated|incomplete)\]|\bcontinued in part\b)", re.IGNORECASE)


def _ordered(labels: set[str], order: tuple[str, ...]) -> list[str]:
    return [label for label in order if label in labels]


def _axis(labels: set[str], evidence: dict[str, list[str]], order: tuple[str, ...]) -> RouterAxis:
    if not labels:
        return {"labels": ["unknown"], "status": "unknown", "evidence": {"unknown": ["no_closed_evidence"]}}
    status: AxisStatus = "mixed" if "mixed" in labels else "classified"
    return {"labels": _ordered(labels, order), "status": status, "evidence": evidence}


def _adapt_axis(source: AxisClassification, order: tuple[str, ...]) -> RouterAxis:
    labels = set(source["labels"])
    return _axis(set() if labels == {"unknown"} else labels, dict(source["evidence"]), order)


def _content_format(text: str) -> RouterAxis:
    legacy = format_genre(text)
    labels = {FORMAT_MAP[label] for label in legacy["labels"] if label in FORMAT_MAP}
    evidence = {
        FORMAT_MAP[label]: reasons
        for label, reasons in legacy["evidence"].items()
        if label in FORMAT_MAP
    }
    lines = [line for line in text.splitlines() if line.strip()]
    step_hits = sum(bool(STEP_LINE_RE.match(line)) for line in lines)
    log_hits = sum(bool(LOG_LINE_RE.match(line)) for line in lines)
    if step_hits >= 2:
        labels.add("instruction")
        evidence["instruction"] = [f"explicit_step_lines={step_hits}"]
        labels.discard("dialogue")
        evidence.pop("dialogue", None)
    if log_hits >= 3:
        labels.add("log")
        evidence["log"] = [f"explicit_log_lines={log_hits}"]
    specialized = labels & {"source_code", "mathematical_notation", "table", "dialogue", "question_answer", "instruction", "log"}
    incompatible = specialized - {"question_answer"} if "dialogue" in specialized else specialized
    if len(incompatible) >= 2:
        labels.add("mixed")
        evidence["mixed"] = ["multiple_incompatible_specialized_formats"]
    return _axis(labels, evidence, FORMAT_ORDER)


def _structural_state(text: str, formats: RouterAxis) -> RouterAxis:
    labels: set[str] = set()
    evidence: dict[str, list[str]] = {}
    checks = (
        ("complete_document", bool(FULL_HTML_RE.search(text)), "closed_html_root"),
        ("complete_artifact", "source_code" in formats["labels"] and bool(SHEBANG_RE.search(text)), "source_code_with_shebang"),
        ("snippet", bool(SNIPPET_RE.search(text)), "explicit_snippet_or_omission_marker"),
        ("template", len(TEMPLATE_RE.findall(text)) >= 2, "template_markers>=2"),
        ("generated_artifact", bool(GENERATED_RE.search(text)) and bool(DO_NOT_EDIT_RE.search(text)), "generated_and_do_not_edit"),
        ("partial", bool(PARTIAL_RE.search(text)), "explicit_partial_marker"),
    )
    for label, matched, reason in checks:
        if matched:
            labels.add(label)
            evidence[label] = [reason]
    if len(labels) >= 2:
        labels.add("mixed")
        evidence["mixed"] = ["multiple_explicit_structural_states"]
    return _axis(labels, evidence, STRUCTURE_ORDER)


def _language_axis(text: str) -> RouterAxis:
    result = _adapt_axis(language_script(text), ("latin", "hangul", "han", "kana", "cyrillic", "arabic", "devanagari", "mixed", "unknown"))
    known_labels = [label for label in result["labels"] if label != "unknown"]
    if len(known_labels) >= 2:
        result["labels"].insert(-1 if "unknown" in result["labels"] else len(result["labels"]), "mixed")
        result["status"] = "mixed"
        result["evidence"]["mixed"] = ["multiple_registered_scripts"]
    if result["status"] == "unknown" and sum(character.isalpha() for character in text) >= 8:
        result["status"] = "out_of_distribution"
        result["evidence"] = {"unknown": ["alphabetic_script_outside_registered_taxonomy"]}
    return result


def _route_labels(text: str, formats: RouterAxis) -> list[str]:
    routes: set[str] = set()
    labels = set(formats["labels"])
    if "source_code" in labels:
        routes.add("code_artifact")
    if "mathematical_notation" in labels:
        routes.add("mathematical_content")
    if labels & {"dialogue", "question_answer"}:
        routes.add("conversation")
    if "instruction" in labels:
        routes.add("instruction")
    if "table" in labels:
        routes.add("table_structured_data")
    if len(TECH_DOC_RE.findall(text)) >= 2:
        routes.add("technical_documentation")
    if not routes and "prose" in labels:
        routes.add("general_prose")
    if len(routes) >= 2:
        routes.add("mixed")
    if not routes:
        routes.add("unknown")
    return _ordered(routes, ROUTE_ORDER)


def _evidence_codes(axes: tuple[tuple[str, RouterAxis], ...]) -> list[str]:
    return sorted(
        f"{axis}:{label}:{reason}"
        for axis, result in axes
        for label, reasons in result["evidence"].items()
        for reason in reasons
    )


def route_content(text: str) -> ContentRouting:
    formats = _content_format(text)
    structure = _structural_state(text, formats)
    scripts = _language_axis(text)
    domains = _adapt_axis(semantic_domain(text), ("code", "math", "science", "medicine", "law", "finance_economics", "history_humanities", "general_knowledge", "unknown"))
    routes = _route_labels(text, formats)
    if scripts["status"] == "out_of_distribution":
        status: RouteStatus = "out_of_distribution"
        confidence: RouteConfidence = "unsupported"
    elif "mixed" in routes:
        status = "mixed"
        confidence = "ambiguous_evidence"
    elif routes == ["unknown"]:
        status = "unknown"
        confidence = "none"
    else:
        status = "routed"
        confidence = "closed_evidence"
    axes = (("content_format", formats), ("structural_state", structure), ("language_script", scripts), ("semantic_domain", domains))
    return {
        "router_version": ROUTER_VERSION,
        "authority": "shared_observable_metadata_only",
        "may_select_or_remove": False,
        "may_assign_importance": False,
        "route_labels": routes,
        "route_status": status,
        "route_confidence": confidence,
        "evidence_codes": _evidence_codes(axes),
        "content_format": formats,
        "structural_state": structure,
        "language_script": scripts,
        "semantic_domain": domains,
    }
