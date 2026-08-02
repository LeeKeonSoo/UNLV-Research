from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_corpus_inventory import build_development_corpus_inventory
from development_corpus_inventory_contract import (
    ConfirmatoryReference,
    DevelopmentCorpusInventoryRegistry,
    InventoryDomain,
    InventorySourceSpec,
    InventoryStatus,
    SourceRole,
)


def _write(path: Path, uid: str, text: str, stored_hash: str | None = None) -> str:
    row = {"record_id": uid, "text": text}
    if stored_hash is not None:
        row["normalized_text_sha256"] = stored_hash
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(root: Path) -> DevelopmentCorpusInventoryRegistry:
    sources: list[InventorySourceSpec] = []
    for domain in InventoryDomain:
        for role in SourceRole:
            path = root / f"{domain.value}-{role.value}.jsonl"
            digest = _write(path, f"{domain.value}-{role.value}", f"{domain.value}  payload", "0" * 64 if role is SourceRole.CLEAN_CONTROL else None)
            sources.append(
                InventorySourceSpec(
                    source_id=f"{domain.value}-{role.value}",
                    domain=domain,
                    role=role,
                    path=str(path),
                    id_fields=("record_id",),
                    text_field="text",
                    expected_file_sha256=digest,
                    selector_visible_source_metadata=False,
                )
            )
    return DevelopmentCorpusInventoryRegistry(
        schema_version="development-corpus-inventory-registry-v1",
        status="block-8-inventory-only",
        normalization="unicode-nfkc-whitespace-collapse-v1",
        sources=tuple(sources),
        confirmatory_references={domain: ConfirmatoryReference.PENDING for domain in InventoryDomain},
        metamorphic_transformations={
            "duplicate_heavy": ("duplicate-v1",),
            "malformed": ("malformed-v1",),
            "boilerplate_heavy": ("boilerplate-v1",),
        },
        benchmark_outcomes_available=False,
        selector_membership_mutation_allowed=False,
    )


def test_inventory_reports_real_sources_without_admitting_pending_slices() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manifest = build_development_corpus_inventory(_registry(Path(directory)))
        assert manifest.status is InventoryStatus.BLOCKED
        assert len(manifest.sources) == 6
        assert len(manifest.slices) == 15
        assert all(item.clean_raw_record_id_overlap_count == 0 for item in manifest.domain_pairs)
        assert manifest.cross_source_record_id_overlap_count == 0
        assert manifest.cross_source_normalized_text_overlap_count == 3
        assert "development_source_overlap_detected" in manifest.blocker_codes
        assert sum(item.stored_normalized_hash_mismatch_count for item in manifest.sources) == 3
        assert "metamorphic_slices_not_materialized" in manifest.blocker_codes
        assert manifest.benchmark_outcomes_read is False


def test_source_hash_drift_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _registry(root)
        Path(registry.sources[0].path).write_text('{}\n', encoding="utf-8")
        try:
            build_development_corpus_inventory(registry)
        except ValueError as error:
            assert str(error).startswith("development_source_hash_mismatch:")
        else:
            raise AssertionError("A drifted source entered the development inventory")


def test_repository_registry_is_closed_and_source_metadata_hidden() -> None:
    payload = json.loads((ROOT / "protocols" / "development_corpus_inventory_registry_v1.json").read_text(encoding="utf-8"))
    registry = DevelopmentCorpusInventoryRegistry.model_validate(payload)
    assert len(registry.sources) == 6
    assert all(item.selector_visible_source_metadata is False for item in registry.sources)
    assert registry.benchmark_outcomes_available is False


if __name__ == "__main__":
    test_inventory_reports_real_sources_without_admitting_pending_slices()
    test_source_hash_drift_fails_closed()
    test_repository_registry_is_closed_and_source_metadata_hidden()
    print("[development-corpus-inventory-v1] closed matrix inventory and fail-closed admission: pass")
