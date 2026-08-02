#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_corpus_admission import build_development_corpus_admission
from development_corpus_admission_alignment import validate_admission_inventory_alignment
from development_corpus_admission_contract import AdmissionStatus, load_admission_registry
from development_corpus_inventory_contract import (
    DevelopmentCorpusInventoryManifest,
    InventoryAdmissionEvidence,
    load_inventory_registry,
)
from development_corpus_materialization import reuse_materialized_development_corpus_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E1 benchmark exclusion and confirmatory-reference admission.")
    parser.add_argument(
        "--admission-registry",
        type=Path,
        default=ROOT / "protocols" / "development_corpus_admission_registry_v1.json",
    )
    parser.add_argument(
        "--inventory-registry",
        type=Path,
        default=ROOT / "protocols" / "development_corpus_inventory_registry_v1.json",
    )
    parser.add_argument(
        "--admission-output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "development_corpus_admission_report_v1.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "configs" / "development_corpus_manifest_v1.json",
    )
    args = parser.parse_args()
    admission_registry = load_admission_registry(args.admission_registry)
    inventory_registry = load_inventory_registry(args.inventory_registry)
    validate_admission_inventory_alignment(admission_registry, inventory_registry)
    report = build_development_corpus_admission(admission_registry)
    evidence = InventoryAdmissionEvidence(
        report_sha256=report.report_sha256,
        benchmark_exclusion_complete=report.benchmark_exclusion_complete,
        frozen_confirmatory_domains=report.frozen_confirmatory_domains,
        blocker_codes=report.blocker_codes,
    )
    previous = DevelopmentCorpusInventoryManifest.model_validate_json(args.manifest_output.read_text(encoding="utf-8"))
    manifest = reuse_materialized_development_corpus_matrix(inventory_registry, previous, evidence)
    args.admission_output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    args.manifest_output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"[development-corpus-admission-v1] status={report.status.value} "
        f"benchmark_contaminated_records={report.total_benchmark_contaminated_record_count} "
        f"confirmatory_text_overlap={report.total_confirmatory_development_text_overlap_count} "
        f"manifest={manifest.status.value}"
    )
    return 0 if report.status is AdmissionStatus.ADMITTED else 2


if __name__ == "__main__":
    raise SystemExit(main())
