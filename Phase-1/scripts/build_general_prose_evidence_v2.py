#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_router import route_content
from validity_recovery import ValidityUnit, evaluate_validity


WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def paragraph_control_text(text: str, minimum_chars: int, maximum_chars: int) -> str:
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    selected: list[str] = []
    length = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + (2 if selected else 0)
        if selected and length + extra > maximum_chars:
            break
        if not selected and len(paragraph) > maximum_chars:
            cut = paragraph[:maximum_chars]
            boundary = cut.rfind(" ")
            return cut[:boundary] if boundary >= minimum_chars else cut
        selected.append(paragraph)
        length += extra
        if length >= minimum_chars:
            break
    result = "\n\n".join(selected)
    return result if len(result) >= minimum_chars else ""


def training_text_hashes(rows: Iterable[Mapping[str, object]]) -> set[str]:
    hashes: set[str] = set()
    for row in rows:
        texts = row.get("texts")
        if not isinstance(texts, list):
            continue
        hashes.update(normalized_text_hash(text) for text in texts if isinstance(text, str) and text.strip())
    return hashes


def build_control(
    source_group: str,
    record_id: str,
    text: str,
    provider_training_hashes: set[str],
    *,
    minimum_chars: int = 100,
    maximum_chars: int = 4000,
) -> dict[str, object] | None:
    control_text = paragraph_control_text(text, minimum_chars, maximum_chars)
    if not control_text:
        return None
    validity = evaluate_validity(ValidityUnit(control_text))
    if validity.final_action not in {"pass", "repair"}:
        return None
    control_text = validity.recovered_text
    digest = normalized_text_hash(control_text)
    if digest in provider_training_hashes:
        return None
    routing = route_content(control_text)
    if routing["route_status"] != "routed" or routing["route_labels"] != ["general_prose"]:
        return None
    return {
        "schema_version": "general-prose-clean-control-v2",
        "chunk_uid": f"{source_group}::{record_id}",
        "source_group": source_group,
        "source_record_id": record_id,
        "normalized_text_sha256": digest,
        "route_status": routing["route_status"],
        "route_labels": routing["route_labels"],
        "semantic_domain_labels": routing["semantic_domain"]["labels"],
        "language_script_labels": routing["language_script"]["labels"],
        "validity_action": validity.final_action,
        "validity_transformations": list(validity.transformation_codes),
        "text": control_text,
    }


def _permuted_tokens(text: str) -> str:
    tokens = WORD_RE.findall(text)
    ranked = sorted(enumerate(tokens), key=lambda row: hashlib.sha256(f"{row[0]}:{row[1]}".encode()).digest())
    return " ".join(token for _, token in ranked)


def build_stress_variants(base: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    uid = str(base["chunk_uid"])
    source = str(base["source_group"])
    text = str(base["text"])
    paragraphs = [part for part in text.split("\n\n") if part]
    variants = (
        ("format_html", "retention_decision_invariant", "<article>" + "".join(f"<p>{html.escape(part)}</p>" for part in paragraphs) + "</article>"),
        ("format_markdown_quote", "retention_decision_invariant", "\n\n".join(f"> {part}" for part in paragraphs)),
        ("semantic_destruction_token_permutation", "must_not_outscore_clean_pair", _permuted_tokens(text)),
    )
    return tuple(
        {
            "schema_version": "general-prose-stress-fixture-v2",
            "chunk_uid": f"{uid}::{variant}",
            "base_chunk_uid": uid,
            "source_group": source,
            "variant": variant,
            "expected_relation": relation,
            "text": variant_text,
        }
        for variant, relation, variant_text in variants
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-disjoint General-prose v2 evidence controls.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    from datasets import load_dataset

    training = config["provider_training_dataset"]
    training_rows = load_dataset(training["dataset_id"], revision=training["revision"], split="train", streaming=True)
    provider_hashes = training_text_hashes(training_rows)
    controls: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    for source in config["clean_control_sources"]:
        dataset = load_dataset(source["dataset_id"], revision=source["revision"], split=source["split"], streaming=True)
        accepted = 0
        for row in dataset:
            control = build_control(
                source["source_group"], str(row[source["id_field"]]), str(row[source["text_field"]]), provider_hashes,
                minimum_chars=int(config["minimum_chars"]), maximum_chars=int(config["maximum_chars"]),
            )
            if control is None or control["normalized_text_sha256"] in seen_hashes:
                continue
            control["dataset_id"] = source["dataset_id"]
            control["dataset_revision"] = source["revision"]
            controls.append(control)
            seen_hashes.add(str(control["normalized_text_sha256"]))
            accepted += 1
            if accepted >= int(config["controls_per_source"]):
                break
        if accepted < int(config["controls_per_source"]):
            raise RuntimeError(f"Insufficient controls for {source['source_group']}: {accepted}")
    stress: list[dict[str, object]] = []
    per_source: dict[str, int] = {}
    for control in controls:
        source = str(control["source_group"])
        if per_source.get(source, 0) >= int(config["stress_bases_per_source"]):
            continue
        stress.extend(build_stress_variants(control))
        per_source[source] = per_source.get(source, 0) + 1
    control_path = Path(config["outputs"]["clean_controls"])
    stress_path = Path(config["outputs"]["stress_fixtures"])
    _write_jsonl(control_path, controls)
    _write_jsonl(stress_path, stress)
    collection_report = {
        "schema_version": "general-prose-control-collection-report-v2",
        "provider_training_dataset": training,
        "provider_training_hashes": len(provider_hashes),
        "final_control_overlap_with_provider_training": sum(
            str(row["normalized_text_sha256"]) in provider_hashes for row in controls
        ),
        "controls": len(controls),
        "controls_by_source": dict(sorted(Counter(str(row["source_group"]) for row in controls).items())),
        "stress_variants": len(stress),
        "artifacts": {
            "clean_controls": {"path": str(control_path), "sha256": _file_hash(control_path)},
            "stress_fixtures": {"path": str(stress_path), "sha256": _file_hash(stress_path)},
        },
        "source_identity_available_to_runtime_selector": False,
        "runtime_activation": False,
    }
    report_path = Path(config["outputs"]["collection_report"])
    report_path.write_text(json.dumps(collection_report, indent=2), encoding="utf-8")
    print(json.dumps({"controls": len(controls), "stress_variants": len(stress), "provider_training_hashes": len(provider_hashes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
