#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_control_units import extract_latex_heading_units


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
TokenCounter: TypeAlias = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class CleanControlRecord:
    record_id: str
    source_group: str
    text: str
    source_path: str
    normalized_text_sha256: str

    @classmethod
    def from_text(cls, record_id: str, source_group: str, text: str, source_path: str) -> "CleanControlRecord":
        normalized_text = _normalized_lines(text)
        digest = hashlib.sha256(" ".join(normalized_text.split()).encode("utf-8")).hexdigest()
        return cls(record_id, source_group, normalized_text, source_path, digest)


@dataclass(frozen=True, slots=True)
class MaterializationError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _normalized_lines(text: str) -> str:
    return "\n".join(line for raw in text.splitlines() if (line := " ".join(raw.split())))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_title(element: ET.Element) -> str:
    for child in element:
        if _local_name(child.tag) == "title":
            return " ".join(" ".join(child.itertext()).split())
    return ""


def _body_without_direct_title(element: ET.Element) -> str:
    parts = [element.text or ""]
    for child in element:
        if _local_name(child.tag) != "title":
            parts.extend(child.itertext())
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def _has_unit_descendant(element: ET.Element, unit_tags: frozenset[str]) -> bool:
    return any(descendant is not element and _local_name(descendant.tag) in unit_tags for descendant in element.iter())


def extract_xml_units(
    xml: str,
    source_group: str,
    source_path: str,
    unit_tags: frozenset[str],
    minimum_characters: int,
) -> tuple[CleanControlRecord, ...]:
    """Extract complete leaf structural units while retaining ancestor title context."""
    root = ET.fromstring(xml)
    rows: list[CleanControlRecord] = []

    def visit(element: ET.Element, titles: tuple[str, ...], ordinal: int) -> int:
        title = _direct_title(element)
        local = _local_name(element.tag)
        next_titles = (*titles, title) if title else titles
        if local in unit_tags and not _has_unit_descendant(element, unit_tags):
            body = _body_without_direct_title(element)
            text = "\n".join((*titles, title, body))
            normalized = _normalized_lines(text)
            if len(normalized) >= minimum_characters:
                xml_id = element.attrib.get("{http://www.w3.org/XML/1998/namespace}id") or element.attrib.get("id")
                unit_id = xml_id or f"unit-{ordinal:06d}"
                rows.append(
                    CleanControlRecord.from_text(
                        f"{source_group}::{source_path}::{unit_id}", source_group, normalized, source_path
                    )
                )
                ordinal += 1
            return ordinal
        for child in element:
            ordinal = visit(child, next_titles, ordinal)
        return ordinal

    visit(root, (), 0)
    return tuple(rows)


def extract_xml_tree(
    root: Path,
    source_group: str,
    pattern: str,
    unit_tags: frozenset[str],
    minimum_characters: int,
) -> tuple[CleanControlRecord, ...]:
    rows: list[CleanControlRecord] = []
    for path in sorted(root.glob(pattern)):
        try:
            xml = path.read_text(encoding="utf-8")
            rows.extend(extract_xml_units(xml, source_group, path.relative_to(root).as_posix(), unit_tags, minimum_characters))
        except (UnicodeDecodeError, ET.ParseError):
            continue
    return tuple(rows)


def extract_latex_files(
    root: Path, source_group: str, pattern: str, minimum_characters: int
) -> tuple[CleanControlRecord, ...]:
    rows = []
    for path in sorted(root.glob(pattern)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        normalized = _normalized_lines(text)
        if len(normalized) < minimum_characters:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(CleanControlRecord.from_text(f"{source_group}::{relative}", source_group, normalized, relative))
    return tuple(rows)


def extract_latex_patterns(
    root: Path, source_group: str, patterns: tuple[str, ...], minimum_characters: int
) -> tuple[CleanControlRecord, ...]:
    rows = {
        row.record_id: row
        for pattern in patterns
        for row in extract_latex_files(root, source_group, pattern, minimum_characters)
    }
    return tuple(rows[record_id] for record_id in sorted(rows))


def extract_latex_heading_files(
    root: Path, source_group: str, pattern: str, minimum_characters: int, encoding: str = "utf-8"
) -> tuple[CleanControlRecord, ...]:
    rows = []
    for path in sorted(root.glob(pattern)):
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        rows.extend(
            CleanControlRecord.from_text(
                f"{source_group}::{relative}::{unit.unit_id}", source_group, unit.text, relative
            )
            for unit in extract_latex_heading_units(text, minimum_characters)
        )
    return tuple(rows)


def stable_source_sample(rows: tuple[CleanControlRecord, ...], maximum_records: int) -> tuple[CleanControlRecord, ...]:
    if maximum_records <= 0:
        raise MaterializationError("maximum_records must be positive")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{row.record_id}\0{row.normalized_text_sha256}".encode()).hexdigest(),
    )
    return tuple(ordered[:maximum_records])


def ensure_candidate_disjoint(
    controls: tuple[CleanControlRecord, ...], candidate_hashes: frozenset[str]
) -> None:
    if {row.normalized_text_sha256 for row in controls} & candidate_hashes:
        raise MaterializationError("Clean controls and candidates have normalized-text hash overlap")


def build_materialization_report(
    protocol: dict[str, JsonValue],
    rows: tuple[CleanControlRecord, ...],
    source_counts: dict[str, int],
    token_counter: TokenCounter,
) -> dict[str, JsonValue]:
    protocol_version = str(protocol["schema_version"])
    version_suffix = protocol_version.rsplit("-", 1)[-1]
    return {
        "schema_version": f"math-open-educational-clean-control-report-{version_suffix}",
        "protocol_schema_version": protocol_version,
        "status": "materialized_before_provider_scoring",
        "records": len(rows),
        "tokens": sum(token_counter(row.text) for row in rows),
        "source_counts_before_cross_source_deduplication": source_counts,
        "target_retention_fraction_used": False,
        "external_results_visible": False,
    }


def _git_head(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _candidate_hashes(path: Path) -> frozenset[str]:
    hashes = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            text = row.get("text")
            if isinstance(text, str):
                hashes.add(hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest())
    return frozenset(hashes)


def source_file_pattern(format_name: str, declared_pattern: str | None) -> str:
    if declared_pattern:
        return declared_pattern
    return "**/*.ptx" if format_name == "pretext_xml" else "**/*.cnxml"


def _source_records(raw: dict[str, JsonValue], repo_root: Path, minimum_characters: int) -> tuple[CleanControlRecord, ...]:
    source_group = str(raw["source_group"])
    root = repo_root / str(raw["root_directory"])
    if _git_head(root if (root / ".git").exists() else root.parents[0]) != str(raw["commit"]):
        raise MaterializationError(f"Commit mismatch for {source_group}")
    format_name = str(raw["format"])
    if format_name in {"pretext_xml", "cnxml"}:
        declared_pattern = raw.get("file_glob")
        pattern = source_file_pattern(format_name, str(declared_pattern) if declared_pattern else None)
        return extract_xml_tree(root, source_group, pattern, frozenset(str(tag) for tag in raw["unit_tags"]), minimum_characters)
    if format_name == "latex_complete_file":
        declared_patterns = raw.get("file_globs")
        if isinstance(declared_patterns, list):
            return extract_latex_patterns(
                root, source_group, tuple(str(pattern) for pattern in declared_patterns), minimum_characters
            )
        return extract_latex_files(root, source_group, str(raw["file_glob"]), minimum_characters)
    if format_name == "latex_heading_units":
        return extract_latex_heading_files(
            root, source_group, str(raw["file_glob"]), minimum_characters, str(raw.get("encoding", "utf-8"))
        )
    raise MaterializationError(f"Unsupported clean-control format: {format_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize score-blind Math clean controls.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rule = protocol["selection_rule"]
    minimum_characters = int(rule["minimum_normalized_characters"])
    maximum_records = int(rule["maximum_records_per_source_group"])
    all_rows: list[CleanControlRecord] = []
    source_counts: dict[str, int] = {}
    for source in protocol["active_sources"]:
        extracted = _source_records(source, args.repo_root, minimum_characters)
        sampled = stable_source_sample(extracted, maximum_records)
        source_counts[str(source["source_group"])] = len(sampled)
        all_rows.extend(sampled)
    unique = {row.normalized_text_sha256: row for row in sorted(all_rows, key=lambda row: row.record_id)}
    rows = tuple(sorted(unique.values(), key=lambda row: row.record_id))
    ensure_candidate_disjoint(rows, _candidate_hashes(Path(protocol["candidate_disjointness"]["candidate_artifact"])))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_counter: TokenCounter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(asdict(row) | {"token_count": token_counter(row.text)}, ensure_ascii=False) + "\n")
    report = build_materialization_report(protocol, rows, source_counts, token_counter)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
