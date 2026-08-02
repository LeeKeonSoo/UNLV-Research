from __future__ import annotations

from development_corpus_admission_contract import DevelopmentCorpusAdmissionError, DevelopmentCorpusAdmissionRegistry
from development_corpus_inventory_contract import DevelopmentCorpusInventoryRegistry


def validate_admission_inventory_alignment(
    admission: DevelopmentCorpusAdmissionRegistry,
    inventory: DevelopmentCorpusInventoryRegistry,
) -> None:
    admission_by_id = {item.reference_id: item for item in admission.development_sources}
    if set(admission_by_id) != {item.source_id for item in inventory.sources}:
        raise DevelopmentCorpusAdmissionError("admission_inventory_source_ids_mismatch")
    for source in inventory.sources:
        reference = admission_by_id[source.source_id]
        aligned = (
            reference.domain is source.domain
            and reference.path == source.path
            and reference.id_fields == source.id_fields
            and reference.text_field == source.text_field
            and reference.expected_file_sha256 == source.expected_file_sha256
            and reference.selector_visible_source_metadata is source.selector_visible_source_metadata
        )
        if not aligned:
            raise DevelopmentCorpusAdmissionError(f"admission_inventory_source_contract_mismatch:{source.source_id}")


__all__ = ["validate_admission_inventory_alignment"]
