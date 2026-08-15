"""Verify benchmark scores against complete task-level artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True, slots=True)
class CellAudit:
    verified: bool
    task_count: int
    passed_tasks: int
    score_percent: float
    evaluator_contract: str
    artifact_sha256: tuple[str, ...]


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_object(path: Path) -> dict[str, object]:
    value = _json(path)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_count(actual: int, expected: int, path: Path) -> None:
    if actual != expected:
        raise ValueError(f"Expected {expected} tasks, found {actual}: {path}")


def audit_evalplus_cell(
    samples_path: Path, result_path: Path, *, expected_count: int
) -> CellAudit:
    sample_ids = [record.get("task_id") for record in _jsonl_objects(samples_path)]
    evaluation = _json_object(result_path).get("eval")
    if not isinstance(evaluation, dict):
        raise TypeError(f"Missing eval mapping: {result_path}")
    _require_count(len(sample_ids), expected_count, samples_path)
    if set(sample_ids) != set(evaluation):
        raise ValueError(f"EvalPlus task-set mismatch: {result_path}")
    passed = 0
    for task_id, task_results in evaluation.items():
        if not isinstance(task_results, list) or len(task_results) != 1:
            raise ValueError(f"Expected one EvalPlus judgment for {task_id}")
        result = task_results[0]
        if not isinstance(result, dict):
            raise TypeError(f"Invalid EvalPlus judgment for {task_id}")
        passed += result.get("base_status") == result.get("plus_status") == "pass"
    return CellAudit(
        True,
        expected_count,
        passed,
        round(100.0 * passed / expected_count, 6),
        "official EvalPlus evaluate() base+plus tests",
        (_sha256(samples_path), _sha256(result_path)),
    )


def audit_bigcodebench_cell(
    samples_path: Path,
    result_path: Path,
    pass_rate_path: Path,
    *,
    expected_count: int,
) -> CellAudit:
    sample_ids = [record.get("task_id") for record in _jsonl_objects(samples_path)]
    evaluation = _json_object(result_path).get("eval")
    if not isinstance(evaluation, dict):
        raise TypeError(f"Missing eval mapping: {result_path}")
    _require_count(len(sample_ids), expected_count, samples_path)
    if set(sample_ids) != set(evaluation):
        raise ValueError(f"BigCodeBench task-set mismatch: {result_path}")
    passed = 0
    for task_id, task_results in evaluation.items():
        if not isinstance(task_results, list) or len(task_results) != 1:
            raise ValueError(f"Expected one BigCodeBench judgment for {task_id}")
        result = task_results[0]
        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
            raise TypeError(f"Invalid BigCodeBench judgment for {task_id}")
        passed += result["status"] == "pass"
    stored = _json_object(pass_rate_path).get("pass@1")
    score = passed / expected_count
    if not isinstance(stored, (int, float)) or abs(float(stored) - score) > 1e-12:
        raise ValueError(f"BigCodeBench pass@1 mismatch: {pass_rate_path}")
    return CellAudit(
        True,
        expected_count,
        passed,
        round(100.0 * score, 6),
        "public bigcode/bigcodebench-evaluator task execution",
        (_sha256(samples_path), _sha256(result_path), _sha256(pass_rate_path)),
    )


def audit_cruxeval_cell(
    samples_path: Path, result_path: Path, *, expected_count: int
) -> CellAudit:
    samples = _json_object(samples_path)
    results = _json_object(result_path)
    raw = results.get("raw_generations")
    scored = results.get("raw_scored_generations")
    if raw != samples or not isinstance(scored, dict):
        raise ValueError(f"CRUXEval generation provenance mismatch: {result_path}")
    _require_count(len(samples), expected_count, samples_path)
    if set(scored) != set(samples):
        raise ValueError(f"CRUXEval task-set mismatch: {result_path}")
    passed = 0
    for task_id, judgments in scored.items():
        if not isinstance(judgments, list) or len(judgments) != 1:
            raise ValueError(f"Expected one CRUXEval judgment for {task_id}")
        if not isinstance(judgments[0], bool):
            raise TypeError(f"Invalid CRUXEval judgment for {task_id}")
        passed += judgments[0]
    stored = results.get("pass_at_1")
    score = 100.0 * passed / expected_count
    if not isinstance(stored, (int, float)) or abs(float(stored) - score) > 1e-9:
        raise ValueError(f"CRUXEval pass_at_1 mismatch: {result_path}")
    return CellAudit(
        True,
        expected_count,
        passed,
        round(score, 6),
        "upstream CRUXEval evaluate_generations()",
        (_sha256(samples_path), _sha256(result_path)),
    )


def audit_ds1000_cell(
    samples_path: Path,
    details_path: Path,
    summary_path: Path,
    *,
    expected_count: int,
) -> CellAudit:
    samples = _jsonl_objects(samples_path)
    details = _json(details_path)
    if not isinstance(details, list):
        raise TypeError(f"Expected DS-1000 detail list: {details_path}")
    _require_count(len(samples), expected_count, samples_path)
    _require_count(len(details), expected_count, details_path)
    ids: list[int] = []
    passed = 0
    for detail in details:
        if not isinstance(detail, dict):
            raise TypeError(f"Invalid DS-1000 detail: {details_path}")
        problem_id = detail.get("problem_id")
        is_passed = detail.get("passed")
        if not isinstance(problem_id, int) or not isinstance(is_passed, bool):
            raise TypeError(f"Invalid DS-1000 judgment: {details_path}")
        if (detail.get("status") == "passed") != is_passed:
            raise ValueError(f"DS-1000 status mismatch for problem {problem_id}")
        ids.append(problem_id)
        passed += is_passed
    if sorted(ids) != list(range(expected_count)):
        raise ValueError(f"DS-1000 problem IDs are incomplete: {details_path}")
    match = re.search(
        r"^mean\s+([0-9]*\.?[0-9]+)\s*$",
        summary_path.read_text(encoding="utf-8-sig"),
        flags=re.MULTILINE,
    )
    score = passed / expected_count
    if match is None or abs(float(match.group(1)) - score) > 5e-7:
        raise ValueError(f"DS-1000 summary mismatch: {summary_path}")
    return CellAudit(
        True,
        expected_count,
        passed,
        round(100.0 * score, 6),
        "official DS-1000 test programs in Windows subprocess isolation",
        (_sha256(samples_path), _sha256(details_path), _sha256(summary_path)),
    )
