#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_corpus_inventory import build_development_corpus_inventory
from development_corpus_inventory_contract import load_inventory_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory Block 8 development corpus sources without admitting pending slices.")
    parser.add_argument("--registry", type=Path, default=ROOT / "protocols" / "development_corpus_inventory_registry_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "frozen_contracts" / "development_corpus_source_inventory_v1.json")
    args = parser.parse_args()
    manifest = build_development_corpus_inventory(load_inventory_registry(args.registry))
    args.output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"[development-corpus-inventory-v1] status={manifest.status.value} sources={len(manifest.sources)} slices={len(manifest.slices)} blockers={len(manifest.blocker_codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
