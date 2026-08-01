#!/usr/bin/env python3
"""Collect repository files while preserving explicit Git attribute declarations."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]
ATTRIBUTE_NAME = "linguist-generated"


def _git(repository: Path, arguments: list[str], *, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _tracked_paths(repository: Path) -> list[str]:
    return [value.decode("utf-8") for value in _git(repository, ["ls-files", "-z"]).split(b"\0") if value]


def _attribute_values(repository: Path, paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    output = _git(repository, ["check-attr", "-z", "--stdin", ATTRIBUTE_NAME], input_bytes=b"\0".join(path.encode("utf-8") for path in paths) + b"\0")
    fields = output.split(b"\0")
    result: dict[str, str] = {}
    for index in range(0, len(fields) - 2, 3):
        path, attribute, value = fields[index : index + 3]
        if path and attribute.decode("utf-8") == ATTRIBUTE_NAME:
            result[path.decode("utf-8")] = value.decode("utf-8")
    return result


def _generation(attribute_value: str) -> str:
    if attribute_value == "true":
        return "generated"
    if attribute_value == "false":
        return "authored"
    return "unknown"


def _record(path: str, text: str, *, repository_identity: str, source_uri: str, collected_at: str, generation: str, rights_status: str, license_name: str | None) -> JsonMap:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "record_id": f"git-attribute::{repository_identity}::{path}::{digest}",
        "text": text,
        "provenance": {"source_name": repository_identity, "source_uri": source_uri, "collected_at": collected_at},
        "artifact_context": {"generation": generation},
        "rights": {"status": rights_status, "license": license_name},
        "pii_context": "repository_code",
        "partition": {
            "source_dataset": "git-attribute-repository",
            "source_tier": "raw_like",
            "repository_identity": repository_identity,
            "path": path,
            "source_content_sha256": digest,
        },
    }


def collect_rows(repository: Path, *, repository_identity: str, source_uri: str, collected_at: str, rights_status: str = "unknown", license_name: str | None = None) -> list[JsonMap]:
    """Return UTF-8 tracked files with only explicit Git attribute labels preserved."""
    paths = _tracked_paths(repository)
    attributes = _attribute_values(repository, paths)
    rows: list[JsonMap] = []
    for path in paths:
        file_path = repository / path
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue
        rows.append(
            _record(
                path,
                text,
                repository_identity=repository_identity,
                source_uri=source_uri,
                collected_at=collected_at,
                generation=_generation(attributes.get(path, "unspecified")),
                rights_status=rights_status,
                license_name=license_name,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a Git repository with source-declared linguist-generated metadata.")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--repository-identity", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--collected-at", required=True)
    parser.add_argument("--rights-status", choices=["allowed", "unknown", "restricted"], default="unknown")
    parser.add_argument("--license")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_rows(
        args.repository,
        repository_identity=args.repository_identity,
        source_uri=args.source_uri,
        collected_at=args.collected_at,
        rights_status=args.rights_status,
        license_name=args.license,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    generation_counts = {generation: sum(row["artifact_context"]["generation"] == generation for row in rows) for generation in ("generated", "authored", "unknown")}
    print(json.dumps({"records": len(rows), "generation_counts": generation_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
