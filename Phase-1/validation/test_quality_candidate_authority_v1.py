#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import POLICY_FINGERPRINT_CONFIGS
from quality_teacher_panel import load_teacher_panel


def _read_json(relative_path: str) -> dict[str, object]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_distilled_ranker_is_runtime_quality_authority_and_panel_is_oracle() -> None:
    manifest = _read_json("configs/curation_framework_v1.json")
    objects = _read_json("configs/framework_objects_v1.json")
    profiles = _read_json("configs/framework_profiles_v1.json")
    providers = _read_json("configs/model_provider_registry_v1.json")

    registry_references = manifest["registry_references"]
    assert registry_references["quality_teacher_oracle"] == (
        "configs/quality_teacher_luna_single_v1.json"
    )
    assert registry_references["quality_ranker"] == "configs/quality_ranker_v1.json"
    assert all("contrastive" not in key for key in registry_references)
    panel = load_teacher_panel(ROOT / registry_references["quality_teacher_oracle"])
    assert panel.runtime_activation is True
    assert panel.transport_mode == "all_policies_per_unit_request"
    assert panel.unit_batch_size == 16

    policy_ids = {policy["id"] for policy in objects["policies"]}
    provider_ids = {provider["id"] for provider in objects["providers"]}
    assert "quality.distilled_ranker_v1" in policy_ids
    assert "quality.teacher_luna_single_v1" not in policy_ids
    assert "quality.distilled_ranker_v1" in provider_ids
    assert "quality.teacher_luna_single_v1" in provider_ids
    assert all("contrastive" not in item for item in policy_ids | provider_ids)

    profiled_policy_ids = {
        policy_id
        for profile in profiles["profiles"]
        for policy_id in profile["policy_ids"]
    }
    registered_provider_ids = {provider["provider_id"] for provider in providers["providers"]}
    assert "quality.distilled_ranker_v1" in profiled_policy_ids
    assert "quality-distilled-ranker-v1" in registered_provider_ids
    assert "quality-teacher-luna-single-v1" in registered_provider_ids
    assert all("contrastive" not in item for item in profiled_policy_ids | registered_provider_ids)
    assert "configs/quality_teacher_luna_single_v1.json" in POLICY_FINGERPRINT_CONFIGS
    assert "configs/quality_ranker_v1.json" in POLICY_FINGERPRINT_CONFIGS
    assert all("contrastive" not in path for path in POLICY_FINGERPRINT_CONFIGS)


def test_contrastive_candidate_files_are_outside_the_active_surface() -> None:
    active_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and "archive" not in path.parts
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    }
    assert not {path for path in active_files if "contrastive" in Path(path).name.lower()}


if __name__ == "__main__":
    test_distilled_ranker_is_runtime_quality_authority_and_panel_is_oracle()
    test_contrastive_candidate_files_are_outside_the_active_surface()
    print("[quality-candidate-authority-v1] ranker authority and Teacher oracle boundary: pass")
