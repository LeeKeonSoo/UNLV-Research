from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

from development_corpus_inventory_contract import (
    ConfirmatoryReference,
    DevelopmentCorpusInventoryManifest,
    DevelopmentCorpusInventoryRegistry,
    DomainPairEvidence,
    InventoryDomain,
    InventorySliceEvidence,
    InventorySourceEvidence,
    InventorySourceSpec,
    InventoryStatus,
    ScenarioOrigin,
    SliceStatus,
    SourceRole,
    hash_json,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(spec: InventorySourceSpec) -> tuple[InventorySourceEvidence, frozenset[str], frozenset[str]]:
    path = Path(spec.path)
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != spec.expected_file_sha256:
        raise ValueError(f"development_source_hash_mismatch:{spec.source_id}")
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    mismatches = 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            record_id = "::".join(str(row[field]) for field in spec.id_fields)
            text = row[spec.text_field]
            if not record_id or not isinstance(text, str) or not text:
                raise ValueError(f"development_source_record_invalid:{spec.source_id}:{count}")
            normalized_hash = hashlib.sha256(_normalize(text).encode()).hexdigest()
            stored_hash = row.get("normalized_text_sha256")
            mismatches += int(stored_hash is not None and stored_hash != normalized_hash)
            record_ids.add(record_id)
            text_hashes.add(normalized_hash)
    if count != len(record_ids) or count != len(text_hashes):
        raise ValueError(f"development_source_identity_not_unique:{spec.source_id}")
    evidence = InventorySourceEvidence(
        source_id=spec.source_id,
        domain=spec.domain,
        role=spec.role,
        file_sha256=actual_sha256,
        record_count=count,
        unique_record_id_count=len(record_ids),
        unique_normalized_text_count=len(text_hashes),
        stored_normalized_hash_mismatch_count=mismatches,
    )
    return evidence, frozenset(record_ids), frozenset(text_hashes)


def _slices(registry: DevelopmentCorpusInventoryRegistry) -> tuple[InventorySliceEvidence, ...]:
    source_by_role = {(item.domain, item.role): item.source_id for item in registry.sources}
    slices: list[InventorySliceEvidence] = []
    for domain in InventoryDomain:
        for scenario in ("clean", "duplicate_heavy", "malformed", "boilerplate_heavy", "mixed_raw_like"):
            observed = scenario in ("clean", "mixed_raw_like")
            role = SourceRole.CLEAN_CONTROL if scenario == "clean" else SourceRole.RAW_LIKE
            transformations = () if observed else registry.metamorphic_transformations[scenario]
            slices.append(
                InventorySliceEvidence(
                    slice_id=f"{domain.value}-{scenario}",
                    domain=domain,
                    scenario=scenario,
                    origin=ScenarioOrigin.OBSERVED if observed else ScenarioOrigin.METAMORPHIC,
                    base_source_id=source_by_role[(domain, role)],
                    status=SliceStatus.INVENTORIED if observed else SliceStatus.MATERIALIZATION_PENDING,
                    transformation_ids=transformations,
                )
            )
    return tuple(slices)


def build_development_corpus_inventory(registry: DevelopmentCorpusInventoryRegistry) -> DevelopmentCorpusInventoryManifest:
    built = tuple((spec, *_source(spec)) for spec in registry.sources)
    sources = tuple(item[1] for item in built)
    pairs: list[DomainPairEvidence] = []
    for domain in InventoryDomain:
        clean = next(item for item in built if item[0].domain is domain and item[0].role is SourceRole.CLEAN_CONTROL)
        raw = next(item for item in built if item[0].domain is domain and item[0].role is SourceRole.RAW_LIKE)
        pairs.append(DomainPairEvidence(domain=domain, clean_raw_record_id_overlap_count=len(clean[2] & raw[2]), clean_raw_normalized_text_overlap_count=len(clean[3] & raw[3])))
    record_overlap = sum(len(left[2] & right[2]) for left, right in combinations(built, 2))
    text_overlap = sum(len(left[3] & right[3]) for left, right in combinations(built, 2))
    slices = _slices(registry)
    blockers = ["metamorphic_slices_not_materialized", "benchmark_exclusion_not_run"]
    if record_overlap or text_overlap:
        blockers.append("development_source_overlap_detected")
    blockers.extend(
        f"{domain.value}_confirmatory_reference_not_frozen"
        for domain, state in registry.confirmatory_references.items()
        if state is not ConfirmatoryReference.FROZEN
    )
    payload = {
        "registry_sha256": registry.identity_sha256(),
        "sources": [item.model_dump(mode="json") for item in sources],
        "domain_pairs": [item.model_dump(mode="json") for item in pairs],
        "cross_source_record_id_overlap_count": record_overlap,
        "cross_source_normalized_text_overlap_count": text_overlap,
        "slices": [item.model_dump(mode="json") for item in slices],
        "blocker_codes": sorted(blockers),
    }
    return DevelopmentCorpusInventoryManifest(
        schema_version="development-corpus-inventory-manifest-v1",
        status=InventoryStatus.BLOCKED if blockers else InventoryStatus.ADMITTED,
        registry_sha256=registry.identity_sha256(),
        sources=sources,
        domain_pairs=tuple(pairs),
        cross_source_record_id_overlap_count=record_overlap,
        cross_source_normalized_text_overlap_count=text_overlap,
        slices=slices,
        blocker_codes=tuple(sorted(blockers)),
        manifest_sha256=hash_json(payload),
    )


__all__ = ["build_development_corpus_inventory"]
