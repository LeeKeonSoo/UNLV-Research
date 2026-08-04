#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "configs" / "curation_framework_v1.json"
SCHEMA_PATH = ROOT / "configs" / "schemas" / "curation_framework_v1.schema.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_roots_the_frozen_research_contract() -> None:
    # Given: the central redesign manifest.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    contract = manifest["research_contract"]
    contract_path = ROOT / contract["path"]

    # When: its declared identity is compared with the tracked document.
    observed = _sha256(contract_path)

    # Then: the manifest is rooted in the exact frozen contract bytes.
    assert contract["sha256"] == observed
    assert manifest["schema_version"] == "curation-framework-v1"
    assert manifest["activation"] == "runtime_integrated_block_7"


def test_manifest_declares_the_complete_core_and_stage_boundary() -> None:
    # Given: the machine-readable framework boundary.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # When: canonical ownership is projected from it.
    core_ids = {core["id"] for core in manifest["cores"]}
    stage_ids = tuple(stage["id"] for stage in manifest["stages"])

    # Then: no legacy alias or external evaluation appears as a Core or Stage.
    assert core_ids == {"validity", "redundancy", "quality", "coverage"}
    assert stage_ids == ("stage_a", "stage_b", "stage_c")
    assert manifest["external_evaluation"]["runtime_stage"] is False
    assert manifest["external_evaluation"]["selector_visible"] is False
    profile = manifest["profile_contract"]
    assert profile["shared_policy_families_required"] is True
    assert profile["independent_operating_point_calibration_required"] is True
    assert profile["arbitrary_threshold_override_allowed"] is False
    assert manifest["registry_references"]["quality_teacher_panel"] == (
        "configs/quality_teacher_panel_v2.json"
    )


def test_threshold_provenance_is_non_optional() -> None:
    # Given: the central threshold provenance contract.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    provenance = set(manifest["threshold_provenance"]["required_fields"])

    # When / Then: every scientific and operational identity field is required.
    assert provenance == {
        "value",
        "unit",
        "comparison_direction",
        "derivation_procedure",
        "development_corpus_sha256",
        "sample_count",
        "supported_routes",
        "provider_identity_sha256",
        "tokenizer_identity_sha256",
        "uncertainty_procedure",
        "fixture_artifact_sha256",
        "ablation_artifact_sha256",
        "external_evidence_sha256",
        "lifecycle",
        "invalidation_conditions",
    }
    assert manifest["threshold_provenance"]["missing_field_action"] == "abstain_retain"


def test_json_schema_rejects_undeclared_fields_by_contract() -> None:
    # Given: the published JSON Schema for the root manifest.
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    # When / Then: it is closed and pins the root identity fields.
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["schema_version"]["const"] == "curation-framework-v1"


if __name__ == "__main__":
    test_manifest_roots_the_frozen_research_contract()
    test_manifest_declares_the_complete_core_and_stage_boundary()
    test_threshold_provenance_is_non_optional()
    test_json_schema_rejects_undeclared_fields_by_contract()
    print("[framework-manifest-v1] root identity and provenance boundary: pass")
