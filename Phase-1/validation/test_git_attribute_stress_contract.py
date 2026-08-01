#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import resolve_curation_mode, validate_run_policy_overrides


CONTRACT = ROOT / "protocols" / "git_attribute_wgpu_stress_curation_contract.json"


def test_git_attribute_stress_contract_preserves_source_declarations_without_auto_removal() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = contract["input"]["sources"][0]

    assert contract["schema_version"] == "curation-run-contract-v1"
    assert contract["status"] == "frozen_before_stage_a_b_c_materialization"
    assert source["metadata_capabilities"]["artifact_context.generation"] == {
        "availability": "provided_by_git_check_attr",
        "attribute": "linguist-generated",
        "inference": "forbidden",
    }
    assert contract["stage_c_selection"]["structural_artifact_rules"]["explicit_generated_artifact"] is True
    assert "artifact_context.generation alone is not a removal reason" in contract["claim_boundary"]
    try:
        validate_run_policy_overrides(
            contract,
            resolve_curation_mode("normal")["effective_policy"],
        )
    except RuntimeError as error:
        assert "stage_c_selection" in str(error)
    else:
        raise AssertionError("Historical run-local policy switches must fail under the immutable runtime.")


if __name__ == "__main__":
    test_git_attribute_stress_contract_preserves_source_declarations_without_auto_removal()
    print("[git-attribute-stress-contract] explicit metadata and no-auto-removal boundary: pass")
