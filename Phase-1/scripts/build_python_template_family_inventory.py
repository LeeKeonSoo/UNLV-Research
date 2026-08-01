#!/usr/bin/env python3
"""Diagnose alpha-normalized Python template families without selecting data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import keyword
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonMap = dict[str, Any]
IGNORED_TOKEN_TYPES = {tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL}


def _read_jsonl(path: Path) -> Iterable[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized_python_template(text: str) -> str | None:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        normalized = []
        for token in tokens:
            if token.type in IGNORED_TOKEN_TYPES:
                continue
            if token.type == tokenize.NAME:
                normalized.append(token.string if keyword.iskeyword(token.string) else "NAME")
            elif token.type == tokenize.NUMBER:
                normalized.append("NUMBER")
            elif token.type == tokenize.STRING:
                normalized.append("STRING")
            else:
                normalized.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return " ".join(normalized)


def build_inventory(rows: Iterable[JsonMap], minimum_tokens: int, sample_limit: int) -> JsonMap:
    families: dict[str, list[JsonMap]] = defaultdict(list)
    diagnostics: Counter[str] = Counter()
    for row in rows:
        language = row.get("language") if isinstance(row.get("language"), dict) else {}
        if language.get("code") != "python":
            diagnostics["non_python"] += 1
            continue
        normalized = normalized_python_template(str(row["text"]))
        if normalized is None:
            diagnostics["tokenize_failed"] += 1
            continue
        token_count = len(normalized.split())
        if token_count < minimum_tokens:
            diagnostics["below_minimum_tokens"] += 1
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        families[digest].append(row)
        diagnostics["eligible_python_records"] += 1

    duplicate_families = {digest: members for digest, members in families.items() if len(members) >= 2}
    samples = []
    for digest, members in sorted(duplicate_families.items(), key=lambda item: (-len(item[1]), item[0]))[:sample_limit]:
        samples.append(
            {
                "template_sha256": digest,
                "family_size": len(members),
                "records": [
                    {
                        "record_id": member["record_id"],
                        "path": ((member.get("partition") or {}).get("path")),
                    }
                    for member in sorted(members, key=lambda row: str(row["record_id"]))[:10]
                ],
            }
        )
    family_size_counts = Counter(len(members) for members in duplicate_families.values())
    return {
        "schema_version": "python-template-family-inventory-v1",
        "status": "diagnostic_only_not_a_selection_policy",
        "scope": "Alpha-normalized full-source Python templates. A family is a candidate for manual false-positive audit, not proof that every member is redundant.",
        "minimum_normalized_tokens": minimum_tokens,
        "diagnostics": dict(diagnostics),
        "duplicate_family_count": len(duplicate_families),
        "records_in_duplicate_families": sum(len(members) for members in duplicate_families.values()),
        "duplicate_family_size_distribution": dict(sorted(family_size_counts.items())),
        "family_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a diagnostic inventory of Python template families.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tokens", type=int, default=40)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_tokens < 1 or args.sample_limit < 1:
        raise RuntimeError("minimum-tokens and sample-limit must be positive")
    report = build_inventory(_read_jsonl(args.input), args.minimum_tokens, args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "duplicate_family_count", "records_in_duplicate_families")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
