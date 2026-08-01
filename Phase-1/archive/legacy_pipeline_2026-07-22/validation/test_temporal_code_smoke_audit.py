#!/usr/bin/env python3
"""Regression checks for bounded temporal-code smoke bundle auditing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    audit_module = importlib.import_module("71_audit_temporal_code_smoke_bundles")
    fixture = load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_change_bundles.json")["bundles"][0]
    audited = audit_module.bundle_with_content_signatures(fixture)
    assert audited["content_signatures"], audited
    assert all(len(row["normalized_sha256"]) == 64 for row in audited["content_signatures"])
    assert audited["prose"]["body"] not in {
        row.get("normalized_sha256") for row in audited["content_signatures"]
    }
    print("[temporal-code-smoke-audit] normalized code/test signatures exclude PR prose: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
