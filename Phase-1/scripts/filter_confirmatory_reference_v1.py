#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_confirmatory_filter import filter_confirmatory_reference
from development_corpus_admission_contract import DevelopmentCorpusAdmissionReport, load_admission_registry
from development_corpus_inventory_contract import InventoryDomain


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a benchmark-clean confirmatory corpus reference.")
    parser.add_argument("--domain", type=InventoryDomain, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "protocols" / "development_corpus_admission_registry_v1.json",
    )
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "development_corpus_admission_report_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    registry = load_admission_registry(args.registry)
    report = DevelopmentCorpusAdmissionReport.model_validate_json(args.admission_report.read_text(encoding="utf-8"))
    evidence = filter_confirmatory_reference(registry, report, args.domain, args.output)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"[filtered-confirmatory-reference-v1] domain={args.domain.value} "
        f"input={evidence.input_record_count} output={evidence.output_record_count} "
        f"removed={len(evidence.removed_record_ids)} sha256={evidence.output_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
