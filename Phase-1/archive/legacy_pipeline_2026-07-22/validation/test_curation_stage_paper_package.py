#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    path = ROOT / "196_build_curation_stage_paper_package.py"
    spec = importlib.util.spec_from_file_location("curation_stage_paper_package", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        package = module.build(
            tmp_path / "curation_stage_paper_package.json",
            tmp_path / "curation_stage_paper_package.md",
        )
    assert package["status"] == "curation_stage_paper_package_ready"
    assert package["paper_claim"]["tier"] == "curation_stage_research_framework"
    assert package["paper_claim"]["supported"] is True
    assert package["production_boundary"]["supported"] is False
    assert "production_core_validity_not_supported" in package["production_boundary"]["blockers"]
    assert package["method_section"]["ready"] is True
    assert Path(package["method_section"]["path"]).as_posix() == "docs/paper_method_core_metric_policy.md"
    assert package["comparison_tables"]["ready"] is True
    assert Path(package["comparison_tables"]["path"]).as_posix().endswith(
        "outputs/validation/paper_comparison_tables.json"
    )
    assert package["limitations_section"]["ready"] is True
    assert Path(package["limitations_section"]["path"]).as_posix() == "docs/paper_limitations_and_threats.md"
    assert package["reproducibility_manifest"]["ready"] is True
    assert Path(package["reproducibility_manifest"]["path"]).as_posix().endswith(
        "outputs/validation/paper_reproducibility_manifest.json"
    )
    assert "production_ready_universal_filter" in package["forbidden_claims"]
    assert "utility_as_stage_b_selector_objective" in package["forbidden_claims"]
    assert len(package["evidence_table"]) == 4
    assert "write_method_section_for_core_metric_policy_and_stage_a_b_c_boundaries" in package["completed_before_submission"]
    assert "freeze_tables_for_raw_stageA_random_curated_and_ablation_comparisons" in package["completed_before_submission"]
    assert "write_limitations_and_threats_to_validity_with_production_boundary" in package["completed_before_submission"]
    assert (
        "freeze_reproducibility_manifest_with_commands_configs_artifacts_and_hardware_notes"
        in package["completed_before_submission"]
    )
    assert "write_method_section_for_core_metric_policy_and_stage_a_b_c_boundaries" not in package["remaining_before_submission"]
    assert "freeze_tables_for_raw_stageA_random_curated_and_ablation_comparisons" not in package["remaining_before_submission"]
    assert "write_limitations_and_threats_to_validity_with_production_boundary" not in package["remaining_before_submission"]
    assert "freeze_reproducibility_manifest_with_commands_configs_artifacts_and_hardware_notes" not in package[
        "remaining_before_submission"
    ]
    assert len(package["remaining_before_submission"]) == 0
    assert package["missing_inputs"] == []
    print("[curation-stage-paper-package] paper package ready with production boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
