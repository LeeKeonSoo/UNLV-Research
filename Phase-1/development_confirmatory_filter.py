from __future__ import annotations

import json
from pathlib import Path

from development_corpus_admission_contract import (
    CorpusRole,
    DevelopmentCorpusAdmissionError,
    DevelopmentCorpusAdmissionRegistry,
    DevelopmentCorpusAdmissionReport,
    FilteredConfirmatoryReference,
)
from development_corpus_benchmark_exclusion import sha256_file
from development_corpus_inventory_contract import InventoryDomain


type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


def filter_confirmatory_reference(
    registry: DevelopmentCorpusAdmissionRegistry,
    report: DevelopmentCorpusAdmissionReport,
    domain: InventoryDomain,
    output_path: Path,
) -> FilteredConfirmatoryReference:
    reference = next(item for item in registry.confirmatory_references if item.domain is domain)
    removed_ids = tuple(
        sorted(
            {
                item.record_id
                for item in report.contamination_matches
                if item.domain is domain and item.role is CorpusRole.CONFIRMATORY
            }
        )
    )
    if not removed_ids:
        raise DevelopmentCorpusAdmissionError(f"confirmatory_filter_has_no_exclusions:{domain.value}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_removed: set[str] = set()
    input_count = 0
    output_count = 0
    with Path(reference.path).open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if not line.strip():
                continue
            row: JsonValue = json.loads(line)
            if not isinstance(row, dict):
                raise DevelopmentCorpusAdmissionError(f"confirmatory_filter_row_invalid:{reference.reference_id}")
            record_id = "::".join(str(row[field]) for field in reference.id_fields)
            input_count += 1
            if record_id in removed_ids:
                seen_removed.add(record_id)
                continue
            target.write(line if line.endswith("\n") else line + "\n")
            output_count += 1
    if set(removed_ids) != seen_removed or input_count != reference.expected_record_count:
        raise DevelopmentCorpusAdmissionError(f"confirmatory_filter_identity_mismatch:{reference.reference_id}")
    return FilteredConfirmatoryReference(
        schema_version="filtered-confirmatory-reference-v1",
        reference_id=reference.reference_id,
        domain=domain,
        source_path=reference.path,
        source_sha256=reference.expected_file_sha256,
        output_path=output_path.as_posix(),
        output_sha256=sha256_file(output_path),
        input_record_count=input_count,
        output_record_count=output_count,
        removed_record_ids=removed_ids,
        admission_report_sha256=report.report_sha256,
    )


__all__ = ["FilteredConfirmatoryReference", "filter_confirmatory_reference"]
