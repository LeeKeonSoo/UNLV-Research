from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load("164_build_selector_utility_leakage_audit.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "policy" / "subsets.py",
            ROOT / "ingestion" / "code_selection.py",
            ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "train_scored_full_selector.jsonl",
            tmp_path / "selector_utility_leakage_audit.json",
            tmp_path / "selector_utility_leakage_audit.md",
            None,
        )
    assert report["schema_version"] == "selector-utility-leakage-audit-v2"
    assert report["status"] == "selector_utility_leakage_audit_passed"
    assert not report["blockers"]
    assert report["stage_b_evidence_scan"]["truncated"] is False
    assert report["stage_b_evidence_scan"]["records_checked"] == report["stage_b_evidence_scan"]["total_records_known"]
    assert not report["stage_b_evidence_scan"]["unexpected_stage_b_evidence_keys"]
    assert not report["stage_b_evidence_scan"]["forbidden_terms_seen"]
    for audit in report["selector_files"].values():
        assert not audit["missing_functions"]
        for row in audit["functions"].values():
            assert row["function_found"] is True
            assert not row["forbidden_terms_found"]
    print("[selector-utility-leakage-audit-v2] temporal selector and full Stage-B scan pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
