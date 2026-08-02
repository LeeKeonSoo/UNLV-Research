from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from development_corpus_inventory import build_development_corpus_inventory
from development_corpus_inventory_contract import (
    DevelopmentCorpusInventoryManifest,
    DevelopmentCorpusInventoryRegistry,
    InventoryDomain,
    InventorySliceEvidence,
    InventorySourceSpec,
    ScenarioOrigin,
    SliceStatus,
    SourceRole,
)


type Row = dict[str, object]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_rows(spec: InventorySourceSpec) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    with Path(spec.path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: Row = json.loads(line)
            uid = "::".join(str(row[field]) for field in spec.id_fields)
            text = row[spec.text_field]
            if not isinstance(text, str):
                raise ValueError(f"development_materialization_text_invalid:{spec.source_id}")
            rows.append((uid, text))
    return tuple(rows)


def _order(rows: tuple[tuple[str, str], ...], minimum_tokens: int) -> tuple[tuple[str, str], ...]:
    eligible = tuple(row for row in rows if len(re.findall(r"\w+", row[1], flags=re.UNICODE)) >= minimum_tokens)
    return tuple(sorted(eligible, key=lambda row: hashlib.sha256(f"{row[0]}\n{_normalize(row[1])}".encode()).hexdigest()))


def _fixture(slice_id: str, parent_id: str, relation: str, text: str) -> dict[str, str]:
    fixture_id = hashlib.sha256(f"{slice_id}\n{parent_id}\n{relation}".encode()).hexdigest()
    return {
        "fixture_id": fixture_id,
        "slice_id": slice_id,
        "parent_record_id": parent_id,
        "metamorphic_relation": relation,
        "text": text,
        "normalized_text_sha256": hashlib.sha256(_normalize(text).encode()).hexdigest(),
    }


def _delete_middle_token(text: str) -> str:
    spans = tuple(re.finditer(r"\w+", text, flags=re.UNICODE))
    target = spans[len(spans) // 2]
    return text[:target.start()] + text[target.end():]


def _records(slice_id: str, scenario: str, parents: tuple[tuple[str, str], ...]) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    chrome = "Home | About | Contact | Privacy | Cookie settings | Accept all | Reject all\n"
    for uid, text in parents:
        records.append(_fixture(slice_id, uid, "parent-retained-v1", text))
        if scenario == "duplicate_heavy":
            records.extend(
                (
                    _fixture(slice_id, uid, "exact-copy-1-v1", text),
                    _fixture(slice_id, uid, "exact-copy-2-v1", text),
                    _fixture(slice_id, uid, "length-relative-single-token-deletion-v1", _delete_middle_token(text)),
                )
            )
        elif scenario == "malformed":
            records.extend(
                (
                    _fixture(slice_id, uid, "empty-payload-v1", ""),
                    _fixture(slice_id, uid, "invalid-utf8-tail-replacement-v1", text + "\ufffd"),
                )
            )
        elif scenario == "boilerplate_heavy":
            records.append(_fixture(slice_id, uid, "explicit-chrome-wrapper-v1", chrome * 3 + text + "\n" + chrome * 3))
    return tuple(records)


def _write_slice(
    path: Path,
    slice_id: str,
    scenario: str,
    parents: tuple[tuple[str, str], ...],
    origin: ScenarioOrigin,
    transformation_ids: tuple[str, ...],
) -> InventorySliceEvidence:
    records = _records(slice_id, scenario, parents)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    fixture_ids = {item["fixture_id"] for item in records}
    return InventorySliceEvidence(
        slice_id=slice_id,
        domain=InventoryDomain(slice_id.split("-", 1)[0]),
        scenario=scenario,
        origin=origin,
        base_source_id="",
        status=SliceStatus.MATERIALIZED,
        transformation_ids=transformation_ids,
        artifact_path=path.as_posix(),
        artifact_sha256=_sha256_file(path),
        parent_record_ids_sha256=hashlib.sha256("\n".join(sorted(uid for uid, _ in parents)).encode()).hexdigest(),
        parent_record_count=len(parents),
        materialized_record_count=len(records),
        unique_fixture_id_count=len(fixture_ids),
    )


def materialize_development_corpus_matrix(registry: DevelopmentCorpusInventoryRegistry) -> DevelopmentCorpusInventoryManifest:
    root = Path(registry.output_root)
    source_map = {(item.domain, item.role): item for item in registry.sources}
    evidence: list[InventorySliceEvidence] = []
    seen_parents: set[str] = set()
    parent_overlap = 0
    count = registry.parent_records_per_slice
    for domain in InventoryDomain:
        clean_spec = source_map[(domain, SourceRole.CLEAN_CONTROL)]
        raw_spec = source_map[(domain, SourceRole.RAW_LIKE)]
        clean = _order(_source_rows(clean_spec), 5)[:count]
        raw = _order(_source_rows(raw_spec), 20)
        if len(clean) < count or len(raw) < count * 4:
            raise ValueError(f"development_materialization_source_too_small:{domain.value}")
        groups = {
            "duplicate_heavy": raw[:count],
            "malformed": raw[count:count * 2],
            "boilerplate_heavy": raw[count * 2:count * 3],
            "mixed_raw_like": raw[count * 3:count * 4],
        }
        entries = (("clean", clean, clean_spec.source_id, ScenarioOrigin.OBSERVED),) + tuple(
            (scenario, parents, raw_spec.source_id, ScenarioOrigin.OBSERVED if scenario == "mixed_raw_like" else ScenarioOrigin.METAMORPHIC)
            for scenario, parents in groups.items()
        )
        for scenario, parents, source_id, origin in entries:
            parent_keys = {f"{source_id}::{uid}" for uid, _ in parents}
            parent_overlap += len(seen_parents & parent_keys)
            seen_parents.update(parent_keys)
            slice_id = f"{domain.value}-{scenario}"
            transformations = registry.metamorphic_transformations.get(scenario, ())
            item = _write_slice(root / domain.value / f"{scenario}.jsonl", slice_id, scenario, parents, origin, transformations)
            evidence.append(item.model_copy(update={"base_source_id": source_id}))
    return build_development_corpus_inventory(registry, tuple(evidence), parent_overlap)


__all__ = ["materialize_development_corpus_matrix"]
