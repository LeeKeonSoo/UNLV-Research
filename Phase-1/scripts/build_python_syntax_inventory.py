#!/usr/bin/env python3
"""Measure source-level Python parseability before chunking; diagnostic only."""
from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonMap = dict[str, Any]
PYTHON2_COMPATIBILITY_MESSAGES = {
    "Missing parentheses in call to 'print'. Did you mean print(...)?",
    "multiple exception types must be parenthesized",
    "leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers",
    "Lambda expression parameters cannot be parenthesized",
}


def _error_category(message: str) -> str:
    if message in PYTHON2_COMPATIBILITY_MESSAGES:
        return "python2_compatibility"
    if "non-printable character" in message:
        return "encoding_nonprintable_character"
    if "indent" in message or "tabs and spaces" in message:
        return "indentation_error"
    if "unterminated string" in message:
        return "unterminated_string"
    return "ambiguous_syntax_error"


def _read_jsonl(path: Path) -> Iterable[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_inventory(rows: Iterable[JsonMap], sample_limit: int) -> JsonMap:
    counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    error_category_counts: Counter[str] = Counter()
    error_samples: dict[str, list[JsonMap]] = defaultdict(list)
    for row in rows:
        language = row.get("language") if isinstance(row.get("language"), dict) else {}
        if language.get("code") != "python":
            counts["non_python"] += 1
            continue
        counts["python_records"] += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(str(row["text"]))
        except SyntaxError as error:
            counts["syntax_error"] += 1
            message = str(error.msg or "unknown_syntax_error")
            category = _error_category(message)
            error_counts[message] += 1
            error_category_counts[category] += 1
            if len(error_samples[category]) < sample_limit:
                source_lines = str(row["text"]).splitlines()
                source_line = source_lines[error.lineno - 1] if error.lineno and error.lineno <= len(source_lines) else ""
                error_samples[category].append(
                    {
                        "record_id": row["record_id"],
                        "path": ((row.get("partition") or {}).get("path")),
                        "error": message,
                        "line": error.lineno,
                        "offset": error.offset,
                        "source_line": source_line[:240],
                    }
                )
        else:
            counts["parseable"] += 1
    return {
        "schema_version": "python-source-syntax-inventory-v1",
        "status": "diagnostic_only_not_a_selection_policy",
        "scope": "Full source records before chunking. A Python syntax error may reflect language-version mismatch or source corruption and is not an automatic removal decision.",
        "counts": dict(counts),
        "syntax_error_messages": dict(error_counts.most_common()),
        "syntax_error_categories": dict(error_category_counts),
        "syntax_error_samples": dict(error_samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-level Python syntax diagnostic inventory.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=30)
    args = parser.parse_args()
    if args.sample_limit < 1:
        raise RuntimeError("sample-limit must be positive")
    report = build_inventory(_read_jsonl(args.input), args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
