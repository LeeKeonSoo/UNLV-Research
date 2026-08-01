from __future__ import annotations

import re
from math import log2
from collections import Counter
from collections.abc import Iterable
from typing import Any

from coverage_taxonomy import AXES, TAXONOMY_VERSION, classify_coverage


JsonMap = dict[str, Any]
CODE_PATTERNS = (re.compile(r"\bdef\s+\w+\s*\("), re.compile(r"\bclass\s+\w+"), re.compile(r"\bimport\s+\w+"))
MATH_PATTERNS = (re.compile(r"\b(?:theorem|lemma|proof|derivative|integral|equation)\b", re.IGNORECASE), re.compile(r"[=+*/^]{2,}"))
SCIENCE_PATTERNS = (re.compile(r"\b(?:experiment|hypothesis|method|results|molecule|clinical)\b", re.IGNORECASE),)
DOMAIN_PRIORITY = ("code", "mathematics", "science")
HTML_TAG_RE = re.compile(r"<[/!]?[a-z][^>]*>", re.IGNORECASE)
DIALOGUE_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9 _-]{0,30}:\s+\S+")
URL_LINE_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
CONTROL_LINE_SET = frozenset({"cookie preferences", "accept all", "reject all", "manage preferences", "manage cookies", "privacy settings"})
TRANSACTIONAL_RE = re.compile(r"\b(?:buy now|shop now|add to cart|get (?:a )?quote|request (?:a )?quote|order now|book now)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"\b(?:references|bibliography|sources|table of contents|appendix)\b", re.IGNORECASE)
INSTRUCTIONAL_RE = re.compile(r"\b(?:step \d+|how to|instructions|tutorial)\b", re.IGNORECASE)


def _hits(text: str, patterns: Iterable[re.Pattern[str]]) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def content_domain(text: str) -> str:
    scores = {
        "code": _hits(text, CODE_PATTERNS),
        "mathematics": _hits(text, MATH_PATTERNS),
        "science": _hits(text, SCIENCE_PATTERNS),
    }
    label = max(DOMAIN_PRIORITY, key=lambda candidate: (scores[candidate], -DOMAIN_PRIORITY.index(candidate)))
    return label if scores[label] else "general"


def language_script(text: str) -> str:
    alphabetic = [character for character in text if character.isalpha()]
    latin = sum(character.isascii() for character in alphabetic)
    non_latin = len(alphabetic) - latin
    if not alphabetic:
        return "unknown"
    if non_latin and latin:
        return "mixed"
    return "non_latin" if non_latin else "latin"


def _nonblank_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def document_format(text: str) -> str:
    lines = _nonblank_lines(text)
    if _hits(text, CODE_PATTERNS) >= 2:
        return "code"
    if len(lines) >= 3 and sum(bool(DIALOGUE_LINE_RE.match(line)) for line in lines) >= 3:
        return "dialogue"
    if HTML_TAG_RE.search(text):
        return "markup"
    if _hits(text, MATH_PATTERNS) >= 2:
        return "formula_structured"
    if len(lines) >= 3 and sum(bool(URL_LINE_RE.fullmatch(line)) for line in lines) >= 3:
        return "link_list"
    return "prose"


def document_function(text: str) -> str:
    lines = _nonblank_lines(text)
    normalized_lines = {_normalized_line(line) for line in lines}
    if len(normalized_lines & CONTROL_LINE_SET) >= 2:
        return "navigation_ui"
    if TRANSACTIONAL_RE.search(text):
        return "transactional"
    if REFERENCE_RE.search(text):
        return "reference"
    if INSTRUCTIONAL_RE.search(text):
        return "instructional"
    if document_format(text) == "dialogue":
        return "discussion"
    return "explanatory"


def _normalized_line(line: str) -> str:
    return " ".join(line.casefold().split())


def annotate_record(record: JsonMap) -> JsonMap:
    text = str(record["text"])
    return {
        "content_domain": content_domain(text),
        "document_format": document_format(text),
        "document_function": document_function(text),
        "language_script": language_script(text),
        "coverage_v1": classify_coverage(text),
        "method": "deterministic_four_axis_audit_v2",
        "authority": "audit_only",
    }


def annotate_records(records: Iterable[JsonMap]) -> list[JsonMap]:
    annotated: list[JsonMap] = []
    for record in records:
        record["composition"] = annotate_record(record)
        annotated.append(record)
    return annotated


def _distribution(rows: Iterable[JsonMap], field: str) -> JsonMap:
    record_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    for row in rows:
        composition = row["composition"]
        label = str(composition[field])
        record_counts[label] += 1
        token_counts[label] += int(row.get("token_proxy") or len(str(row["text"]).split()))
    total_tokens = sum(token_counts.values())
    return {
        "records": dict(sorted(record_counts.items())),
        "token_proxy": dict(sorted(token_counts.items())),
        "token_share": {label: count / total_tokens for label, count in sorted(token_counts.items())} if total_tokens else {},
    }


def _stage_distribution(rows: Iterable[JsonMap]) -> JsonMap:
    materialized = list(rows)
    language = _distribution(materialized, "language_script")
    token_share = language["token_share"]
    return {
        "content_domain": _distribution(materialized, "content_domain"),
        "document_format": _distribution(materialized, "document_format"),
        "document_function": _distribution(materialized, "document_function"),
        "language_script": language,
        "non_latin_or_mixed_token_share": float(token_share.get("mixed", 0.0)) + float(token_share.get("non_latin", 0.0)),
    }


def _token_share_delta(reference: JsonMap, current: JsonMap) -> JsonMap:
    labels = set(reference["token_share"]) | set(current["token_share"])
    return {
        label: float(current["token_share"].get(label, 0.0)) - float(reference["token_share"].get(label, 0.0))
        for label in sorted(labels)
    }


def _stage_delta(raw: JsonMap, current: JsonMap) -> JsonMap:
    return {
        "content_domain": {"token_share": _token_share_delta(raw["content_domain"], current["content_domain"])},
        "document_format": {"token_share": _token_share_delta(raw["document_format"], current["document_format"])},
        "document_function": {"token_share": _token_share_delta(raw["document_function"], current["document_function"])},
        "language_script": {"token_share": _token_share_delta(raw["language_script"], current["language_script"])},
        "non_latin_or_mixed_token_share": float(current["non_latin_or_mixed_token_share"]) - float(raw["non_latin_or_mixed_token_share"]),
    }


def _coverage_axis_distribution(
    rows: list[JsonMap], annotations: list[JsonMap], axis: str
) -> JsonMap:
    record_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    total_tokens = sum(int(row.get("token_proxy") or len(str(row["text"]).split())) for row in rows)
    for row, annotation in zip(rows, annotations, strict=True):
        labels = annotation[axis]["labels"]
        token_count = int(row.get("token_proxy") or len(str(row["text"]).split()))
        for label in labels:
            record_counts[str(label)] += 1
            token_counts[str(label)] += token_count
    unknown_records = record_counts.get("unknown", 0)
    return {
        "records": dict(sorted(record_counts.items())),
        "token_proxy": dict(sorted(token_counts.items())),
        "token_incidence_share_of_stage": {
            label: count / total_tokens for label, count in sorted(token_counts.items())
        } if total_tokens else {},
        "unknown_record_rate": unknown_records / len(rows) if rows else 0.0,
        "multi_label_note": "incidence shares may sum above one",
    }


def _coverage_stage_distribution(rows: list[JsonMap]) -> JsonMap:
    annotations = [classify_coverage(str(row["text"])) for row in rows]
    return {
        axis: _coverage_axis_distribution(rows, annotations, axis)
        for axis in AXES
    }


def _incidence_distribution(report: JsonMap) -> dict[str, float]:
    counts = {label: float(value) for label, value in report["token_proxy"].items()}
    total = sum(counts.values())
    return {label: value / total for label, value in counts.items()} if total else {}


def _jensen_shannon_divergence(left: JsonMap, right: JsonMap) -> float:
    left_distribution = _incidence_distribution(left)
    right_distribution = _incidence_distribution(right)
    labels = set(left_distribution) | set(right_distribution)
    divergence = 0.0
    for label in labels:
        left_value = left_distribution.get(label, 0.0)
        right_value = right_distribution.get(label, 0.0)
        midpoint = (left_value + right_value) / 2.0
        if left_value:
            divergence += 0.5 * left_value * log2(left_value / midpoint)
        if right_value:
            divergence += 0.5 * right_value * log2(right_value / midpoint)
    return divergence


def _coverage_delta(raw: JsonMap, current: JsonMap) -> JsonMap:
    return {
        axis: {
            "token_incidence_share_of_stage": _token_share_delta(
                {"token_share": raw[axis]["token_incidence_share_of_stage"]},
                {"token_share": current[axis]["token_incidence_share_of_stage"]},
            )
        }
        for axis in AXES
    }


def build_composition_audit(stage_rows: dict[str, Iterable[JsonMap]]) -> JsonMap:
    materialized = {name: list(rows) for name, rows in stage_rows.items()}
    stages = {name: _stage_distribution(rows) for name, rows in materialized.items()}
    coverage_stages = {name: _coverage_stage_distribution(rows) for name, rows in materialized.items()}
    raw = stages["raw_input"]
    coverage_raw = coverage_stages["raw_input"]
    return {
        "authority": "audit_only",
        "consumed_by_stage_a": False,
        "consumed_by_stage_b": False,
        "consumed_by_stage_c": False,
        "method": "deterministic_four_axis_audit_v2",
        "stages": stages,
        "delta_from_raw": {name: _stage_delta(raw, stage) for name, stage in stages.items() if name != "raw_input"},
        "coverage_v1": {
            "taxonomy_version": TAXONOMY_VERSION,
            "authority": "audit_only",
            "consumed_by_selection": False,
            "classification": "multi_label_with_unknown",
            "stages": coverage_stages,
            "delta_from_raw": {
                name: _coverage_delta(coverage_raw, stage)
                for name, stage in coverage_stages.items()
                if name != "raw_input"
            },
            "jensen_shannon_divergence_from_raw": {
                name: {
                    axis: _jensen_shannon_divergence(coverage_raw[axis], stage[axis])
                    for axis in AXES
                }
                for name, stage in coverage_stages.items()
                if name != "raw_input"
            },
        },
    }
