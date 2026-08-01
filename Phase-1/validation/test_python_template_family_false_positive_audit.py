#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_python_template_families import build_audit


def test_template_family_false_positive_audit() -> None:
    inventory = {
        "schema_version": "python-template-family-inventory-v1",
        "family_samples": [
            {"template_sha256": "public", "family_size": 2, "records": [{"path": "src/package/__init__.py"}]},
            {"template_sha256": "migration", "family_size": 3, "records": [{"path": "app/migrations/0001_initial.py"}]},
        ],
    }
    labels = {
        "schema_version": "python-template-family-false-positive-audit-labels-v1",
        "families": [
            {"template_sha256": "public", "disposition": "retain", "basis": "public_api_export_structure", "path_prefixes": ["src/package/"]},
            {"template_sha256": "migration", "disposition": "retain", "basis": "versioned_schema_migration", "path_prefixes": ["app/migrations/"]},
        ],
    }

    report = build_audit(inventory, labels)

    assert report["status"] == "all_labeled_families_retained_not_a_selection_policy"
    assert report["summary"] == {"families": 2, "retained_families": 2, "unlabeled_families": 0, "path_mismatches": 0}


if __name__ == "__main__":
    test_template_family_false_positive_audit()
    print("[python-template-family-audit] retain-boundary fixture: pass")
