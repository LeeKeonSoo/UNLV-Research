from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from data_eval_common import OUTPUT_DIR, save_json, sha256_file


TOKEN_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s\w]")
BENCHMARK_MARKER_PATTERN: Final = re.compile(
    r"\b(?:HumanEval|MBPP|EvalPlus|LiveCodeBench|BigCodeBench|SWE[- ]bench)\b",
    re.IGNORECASE,
)
DEFAULT_RAW: Final = OUTPUT_DIR / "code_domain_natural_budget_qwen3_4b" / "raw_full_natural.jsonl"
DEFAULT_STAGE_A: Final = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_b_code_domain_v2"
    / "train_scored_full_selector.jsonl"
)
DEFAULT_CURATED: Final = OUTPUT_DIR / "code_domain_natural_budget_qwen3_4b" / "curated_v2_natural.jsonl"
DEFAULT_OUTPUT: Final = OUTPUT_DIR / "validation" / "code_format_affinity_audit_report.json"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AuditInputs:
    raw_path: Path
    stage_a_path: Path
    curated_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    text: str
    token_count: int
    repository: str
    content_type: str


def _load_records(path: Path) -> tuple[CorpusRecord, ...]:
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else row
            stage_b_evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
            records.append(
                CorpusRecord(
                    text=str(row.get("text") or ""),
                    token_count=int(row.get("token_proxy_count") or stage_b_evidence.get("token_proxy_count") or 0),
                    repository=str(provenance.get("repository_identity") or "unknown"),
                    content_type=str(provenance.get("content_type") or "unknown"),
                )
            )
    return tuple(records)


def _load_evalplus_tasks() -> tuple[BenchmarkTask, ...]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    tasks: list[BenchmarkTask] = []
    for suite, rows in (("HumanEval+", get_human_eval_plus()), ("MBPP+", get_mbpp_plus())):
        for task_id, row in rows.items():
            text = f"{row.get('prompt') or ''}\n{row.get('canonical_solution') or ''}"
            tasks.append(BenchmarkTask(task_id=f"{suite}:{task_id}", text=text))
    return tuple(tasks)


def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _token_ngrams(text: str, width: int = 8) -> frozenset[tuple[str, ...]]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return frozenset(tuple(tokens[index : index + width]) for index in range(max(0, len(tokens) - width + 1)))


def _ast_features(text: str) -> tuple[bool, bool, bool, bool, bool]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False, False, False, False, False
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    has_docstring = any(ast.get_docstring(node, clean=False) for node in [*functions, *classes])
    has_assert = any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    standalone_function = len(functions) == 1 and not classes
    return True, bool(functions), standalone_function, bool(has_docstring), has_assert


def _format_summary(records: tuple[CorpusRecord, ...]) -> JsonMap:
    count = len(records)
    token_counts = [record.token_count for record in records]
    ast_rows = [_ast_features(record.text) for record in records]
    repositories = Counter(record.repository for record in records)
    content_types = Counter(record.content_type for record in records)
    share = lambda hits: round(hits / count, 6) if count else 0.0
    return {
        "records": count,
        "token_proxy_count": sum(token_counts),
        "mean_token_proxy_count": round(statistics.fmean(token_counts), 6) if token_counts else 0.0,
        "median_token_proxy_count": statistics.median(token_counts) if token_counts else 0.0,
        "concise_share": share(sum(value <= 256 for value in token_counts)),
        "python_parseable_share": share(sum(row[0] for row in ast_rows)),
        "top_level_function_share": share(sum(row[1] for row in ast_rows)),
        "standalone_function_share": share(sum(row[2] for row in ast_rows)),
        "function_with_docstring_share": share(sum(row[3] for row in ast_rows)),
        "assert_share": share(sum(row[4] for row in ast_rows)),
        "test_content_share": share(content_types.get("test", 0)),
        "code_content_share": share(content_types.get("code", 0)),
        "documentation_content_share": share(content_types.get("documentation", 0)),
        "repository_count": len(repositories),
        "largest_repository_record_share": share(max(repositories.values(), default=0)),
        "content_type_counts": dict(sorted(content_types.items())),
    }


def _contamination_summary(
    records: tuple[CorpusRecord, ...],
    tasks: tuple[BenchmarkTask, ...],
) -> JsonMap:
    task_rows = [(_canonical_text(task.text), _token_ngrams(task.text), task.task_id) for task in tasks]
    exact_candidates = 0
    near_candidates = 0
    marker_records = 0
    top_candidates: list[JsonMap] = []
    for record_index, record in enumerate(records):
        canonical = _canonical_text(record.text)
        grams = _token_ngrams(record.text)
        marker_records += bool(BENCHMARK_MARKER_PATTERN.search(record.text))
        best_score = 0.0
        best_task = ""
        exact = False
        for task_text, task_grams, task_id in task_rows:
            exact = exact or canonical == task_text or (
                min(len(canonical), len(task_text)) >= 80
                and (canonical in task_text or task_text in canonical)
            )
            denominator = min(len(grams), len(task_grams))
            shared = len(grams & task_grams)
            score = shared / denominator if denominator and shared >= 4 else 0.0
            if score > best_score:
                best_score = score
                best_task = task_id
        exact_candidates += exact
        near_candidates += best_score >= 0.5
        if exact or best_score >= 0.5:
            top_candidates.append(
                {"record_index": record_index, "task_id": best_task, "exact": exact, "ngram_containment": round(best_score, 6)}
            )
    top_candidates.sort(key=lambda row: (bool(row["exact"]), float(row["ngram_containment"])), reverse=True)
    return {
        "records": len(records),
        "benchmark_marker_records": marker_records,
        "exact_copy_candidate_records": exact_candidates,
        "near_overlap_candidate_records": near_candidates,
        "top_candidates": top_candidates[:20],
    }


def build(
    inputs: AuditInputs,
    benchmark_tasks: tuple[BenchmarkTask, ...] | None = None,
) -> JsonMap:
    tasks = benchmark_tasks or _load_evalplus_tasks()
    corpora = {
        "raw": _load_records(inputs.raw_path),
        "stage_a": _load_records(inputs.stage_a_path),
        "curated": _load_records(inputs.curated_path),
    }
    format_rows = {name: _format_summary(records) for name, records in corpora.items()}
    contamination_rows = {name: _contamination_summary(records, tasks) for name, records in corpora.items()}
    raw_format = format_rows["raw"]
    curated_format = format_rows["curated"]
    shifts = {
        key: round(float(curated_format[key]) - float(raw_format[key]), 6)
        for key in (
            "concise_share",
            "python_parseable_share",
            "top_level_function_share",
            "standalone_function_share",
            "function_with_docstring_share",
            "assert_share",
            "test_content_share",
        )
    }
    benchmark_fingerprint = hashlib.sha256(
        "\n".join(f"{task.task_id}\t{task.text}" for task in tasks).encode("utf-8")
    ).hexdigest()
    has_copy_candidate = any(
        int(row["exact_copy_candidate_records"]) > 0 for row in contamination_rows.values()
    )
    has_near_candidate = any(
        int(row["near_overlap_candidate_records"]) > 0 for row in contamination_rows.values()
    )
    report: JsonMap = {
        "schema_version": "code-format-affinity-audit-v1",
        "status": "code_format_affinity_audit_ready",
        "source_sha256": {
            str(inputs.raw_path): sha256_file(inputs.raw_path),
            str(inputs.stage_a_path): sha256_file(inputs.stage_a_path),
            str(inputs.curated_path): sha256_file(inputs.curated_path),
            "evalplus_task_text_fingerprint": benchmark_fingerprint,
        },
        "benchmark_scope": {"task_count": len(tasks), "suites": ["HumanEval+", "MBPP+"]},
        "contamination_screen": {
            "method": "canonical text containment plus exact lexical token 8-gram containment",
            "corpora": contamination_rows,
            "decision": (
                "copy_candidates_require_review"
                if has_copy_candidate
                else "near_overlap_candidates_require_review"
                if has_near_candidate
                else "no_copy_candidate_detected_by_lexical_screen"
            ),
            "benchmark_contamination_absence_proven": False,
        },
        "format_affinity": {
            "corpora": format_rows,
            "curated_minus_raw_share_shifts": shifts,
            "format_affinity_alternative_explanation_present": max(abs(value) for value in shifts.values()) >= 0.1,
        },
        "interpretation": {
            "scope": "Post-hoc Stage-C diagnostic of benchmark overlap and code-format shift.",
            "not_selector_tuning_permission": True,
            "not_intrinsic_quality_proof": True,
            "required_counterfactual": "length-and-format-matched Stage-A baseline plus an independent code benchmark",
        },
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(inputs.output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the code format-affinity and contamination audit.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(AuditInputs(args.raw, args.stage_a, args.curated, args.output))
    print(json.dumps({"status": report["status"], "contamination": report["contamination_screen"]["decision"]}, indent=2))
    return 0
