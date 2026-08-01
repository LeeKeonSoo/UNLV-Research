#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "general_provider_candidate_protocol_v2.json"


def test_provider_candidates_share_controls_but_not_claim_semantics() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    providers = protocol["providers"]

    assert len({provider["clean_controls"] for provider in providers}) == 1
    assert len({provider["stress_fixtures"] for provider in providers}) == 1
    assert len({provider["claim_semantics"] for provider in providers}) == 2
    assert all(provider["supported_heads"] == ["route_specific_evidence"] for provider in providers)


def test_protocol_forbids_threshold_copying_and_ensemble_laundering() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["published_threshold_is_runtime_authority"] is False
    assert protocol["ensemble_may_fill_missing_quality_heads"] is False
    assert protocol["runtime_activation"] is False
    assert all(len(provider["revision"]) == 40 for provider in protocol["providers"])
    assert all(len(provider["model_weight_sha256"]) == 64 for provider in protocol["providers"])


if __name__ == "__main__":
    test_provider_candidates_share_controls_but_not_claim_semantics()
    test_protocol_forbids_threshold_copying_and_ensemble_laundering()
    print("general provider candidate protocol v2: ok")
