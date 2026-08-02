#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_corpus_inventory_contract import load_inventory_registry
from development_corpus_materialization import materialize_development_corpus_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the disjoint Block 8 development scenario matrix.")
    parser.add_argument("--registry", type=Path, default=ROOT / "protocols" / "development_corpus_inventory_registry_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "development_corpus_manifest_v1.json")
    args = parser.parse_args()
    manifest = materialize_development_corpus_matrix(load_inventory_registry(args.registry))
    args.output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    records = sum(item.materialized_record_count or 0 for item in manifest.slices)
    print(f"[development-corpus-matrix-v1] status={manifest.status.value} slices={len(manifest.slices)} records={records} blockers={len(manifest.blocker_codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
