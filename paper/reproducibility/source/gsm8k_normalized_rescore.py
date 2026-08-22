#!/usr/bin/env python3
"""Re-score saved GSM8K generations with an auditable final-answer parser."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Final, Literal

TASK: Final = "gsm8k_cot_zeroshot"
NEXT_PROBLEM: Final = re.compile(r"\n\s*(?:Q:|Question:|\[Question\])", re.IGNORECASE)
NUMBER: Final = r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
EXPLICIT_ANSWER: Final = (
    re.compile(
        rf"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*"
        rf"(?:\\boxed\s*\{{\s*)?(?P<number>{NUMBER})",
        re.IGNORECASE,
    ),
    re.compile(rf"####\s*(?P<number>{NUMBER})", re.IGNORECASE),
    re.compile(rf"\\boxed\s*\{{\s*(?P<number>{NUMBER})\s*\}}", re.IGNORECASE),
)
ANY_NUMBER: Final = re.compile(NUMBER)
ExtractionMethod = Literal["explicit_final_answer", "numeric_fallback", "unparsed"]


@dataclass(frozen=True, slots=True)
class RescoreSource:
    label: str
    path: Path


@dataclass(frozen=True, slots=True)
class ResponseScore:
    correct: bool
    extracted_answer: str | None
    target_answer: str | None
    extraction_method: ExtractionMethod
    truncated_next_problem: bool


@dataclass(frozen=True, slots=True)
class RescoreResult:
    label: str
    source_path: str
    source_sha256: str
    records: int
    correct: int
    normalized_accuracy: float
    explicit_final_answer: int
    numeric_fallback: int
    unparsed: int
    truncated_next_problem: int
    official_strict_accuracy: float
    official_flexible_accuracy: float
    both_official_correct: int
    strict_only_correct: int
    flexible_only_correct: int
    neither_official_correct: int


@dataclass(frozen=True, slots=True)
class RescoreArtifact:
    schema_version: str
    status: str
    parser_contract: tuple[str, ...]
    results: tuple[RescoreResult, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncate_response(response: str) -> tuple[str, bool]:
    match = NEXT_PROBLEM.search(response)
    if match is None:
        return response, False
    return response[: match.start()], True


def numeric_value(token: str) -> Fraction | None:
    cleaned = token.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text: str) -> tuple[Fraction | None, ExtractionMethod]:
    explicit = [
        (match.start(), match.group("number"))
        for pattern in EXPLICIT_ANSWER
        for match in pattern.finditer(text)
    ]
    if explicit:
        value = numeric_value(max(explicit, key=lambda item: item[0])[1])
        if value is not None:
            return value, "explicit_final_answer"
    matches = list(ANY_NUMBER.finditer(text))
    if matches:
        value = numeric_value(matches[-1].group(0))
        if value is not None:
            return value, "numeric_fallback"
    return None, "unparsed"


def target_answer(target: str) -> Fraction | None:
    marker = list(re.finditer(rf"####\s*(?P<number>{NUMBER})", target))
    if marker:
        return numeric_value(marker[-1].group("number"))
    value, _ = extract_answer(target)
    return value


def score_response(response: str, target: str) -> ResponseScore:
    truncated, was_truncated = truncate_response(response)
    extracted, method = extract_answer(truncated)
    expected = target_answer(target)
    return ResponseScore(
        correct=extracted is not None and expected is not None and extracted == expected,
        extracted_answer=None if extracted is None else str(extracted),
        target_answer=None if expected is None else str(expected),
        extraction_method=method,
        truncated_next_problem=was_truncated,
    )


def _response_text(sample: dict[str, object]) -> str:
    responses = sample.get("resps")
    match responses:
        case [[str(text), *_], *_]:
            return text
        case _:
            return ""


def _float_metric(results: dict[str, object], name: str) -> float:
    value = results.get(name)
    match value:
        case int() | float():
            return float(value)
        case _:
            return 0.0


def rescore_source(source: RescoreSource) -> RescoreResult:
    payload = json.loads(source.path.read_text(encoding="utf-8-sig"))
    task_results = payload["results"][TASK]
    samples = payload["samples"][TASK]
    responses: dict[str, tuple[str, str]] = {}
    strict: dict[str, int] = {}
    flexible: dict[str, int] = {}
    for raw_sample in samples:
        sample: dict[str, object] = raw_sample
        doc_id = str(sample["doc_id"])
        filter_name = str(sample["filter"])
        exact = int(float(sample.get("exact_match", 0.0)))
        if filter_name == "strict-match":
            strict[doc_id] = exact
            responses[doc_id] = (_response_text(sample), str(sample["target"]))
        elif filter_name == "flexible-extract":
            flexible[doc_id] = exact
            responses.setdefault(doc_id, (_response_text(sample), str(sample["target"])))
    scores = [score_response(response, target) for response, target in responses.values()]
    both = sum(strict.get(doc, 0) == 1 and flexible.get(doc, 0) == 1 for doc in responses)
    strict_only = sum(strict.get(doc, 0) == 1 and flexible.get(doc, 0) == 0 for doc in responses)
    flexible_only = sum(strict.get(doc, 0) == 0 and flexible.get(doc, 0) == 1 for doc in responses)
    records = len(scores)
    correct = sum(score.correct for score in scores)
    return RescoreResult(
        label=source.label,
        source_path=str(source.path),
        source_sha256=sha256_file(source.path),
        records=records,
        correct=correct,
        normalized_accuracy=correct / records if records else 0.0,
        explicit_final_answer=sum(score.extraction_method == "explicit_final_answer" for score in scores),
        numeric_fallback=sum(score.extraction_method == "numeric_fallback" for score in scores),
        unparsed=sum(score.extraction_method == "unparsed" for score in scores),
        truncated_next_problem=sum(score.truncated_next_problem for score in scores),
        official_strict_accuracy=_float_metric(task_results, "exact_match,strict-match"),
        official_flexible_accuracy=_float_metric(task_results, "exact_match,flexible-extract"),
        both_official_correct=both,
        strict_only_correct=strict_only,
        flexible_only_correct=flexible_only,
        neither_official_correct=records - both - strict_only - flexible_only,
    )


def build_rescore_artifact(sources: tuple[RescoreSource, ...]) -> RescoreArtifact:
    return RescoreArtifact(
        schema_version="gsm8k-normalized-rescore-v1",
        status="complete",
        parser_contract=(
            "truncate_before a generated Q:, Question:, or [Question] block",
            "prefer the last explicit answer marker inside the retained response",
            "otherwise use the last numeric token inside the retained response",
            "compare exact normalized rational values",
        ),
        results=tuple(rescore_source(source) for source in sources),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = tuple(
        RescoreSource(label, Path(path))
        for label, path in (item.split("=", 1) for item in args.source)
    )
    artifact = build_rescore_artifact(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(artifact), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
