#!/usr/bin/env python3
"""Verify labeled retain decisions for diagnostic Python template families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _label_by_digest(labels: JsonMap) -> dict[str, JsonMap]:
    families = labels.get("families")
    if not isinstance(families, list):
        raise RuntimeError("Template-family labels must contain a families list.")
    return {str(family["template_sha256"]): family for family in families if isinstance(family, dict)}


def build_audit(inventory: JsonMap, labels: JsonMap) -> JsonMap:
    if inventory.get("schema_version") != "python-template-family-inventory-v1":
        raise RuntimeError("Unexpected template-family inventory schema.")
    if labels.get("schema_version") != "python-template-family-false-positive-audit-labels-v1":
        raise RuntimeError("Unexpected template-family label schema.")
    inventory_families = inventory.get("family_samples")
    if not isinstance(inventory_families, list):
        raise RuntimeError("Template-family inventory must contain family_samples.")

    label_by_digest = _label_by_digest(labels)
    results: list[JsonMap] = []
    unlabeled = 0
    path_mismatches = 0
    retained = 0
    for family in inventory_families:
        if not isinstance(family, dict):
            continue
        digest = str(family["template_sha256"])
        label = label_by_digest.get(digest)
        if label is None:
            unlabeled += 1
            results.append({"template_sha256": digest, "outcome": "unlabeled"})
            continue
        disposition = str(label.get("disposition"))
        if disposition != "retain":
            raise RuntimeError("This audit supports retain labels only; removal requires a separate policy-card audit.")
        retained += 1
        prefixes = label.get("path_prefixes")
        if not isinstance(prefixes, list) or not prefixes:
            raise RuntimeError("Each template-family retain label requires path_prefixes.")
        records = family.get("records")
        if not isinstance(records, list):
            raise RuntimeError("Each inventory family requires records.")
        matching_records = 0
        for record in records:
            path = str(record.get("path") or "") if isinstance(record, dict) else ""
            if any(path.startswith(str(prefix)) for prefix in prefixes):
                matching_records += 1
        mismatch = matching_records == 0
        path_mismatches += int(mismatch)
        results.append(
            {
                "template_sha256": digest,
                "family_size": family.get("family_size"),
                "disposition": disposition,
                "basis": label.get("basis"),
                "matching_sample_records": matching_records,
                "path_mismatch": mismatch,
            }
        )
    status = "all_labeled_families_retained_not_a_selection_policy" if unlabeled == 0 and path_mismatches == 0 else "audit_incomplete_or_path_mismatch"
    return {
        "schema_version": "python-template-family-false-positive-audit-v1",
        "status": status,
        "scope": "Family-level false-positive audit. Retain labels confirm that observed structural similarity alone is not removal evidence.",
        "summary": {
            "families": len(results),
            "retained_families": retained,
            "unlabeled_families": unlabeled,
            "path_mismatches": path_mismatches,
        },
        "families": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit retain decisions for Python template-family candidates.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    report = build_audit(inventory, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
