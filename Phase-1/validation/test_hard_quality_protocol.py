#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "hard_quality_protocol_v1.json"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["schema_version"] == "hard-quality-protocol-v1"
    assert protocol["status"] == "opt_in_candidate_only_not_runtime_active"
    assert protocol["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert protocol["required_user_declaration"] == [
        "model_id",
        "tokenizer_sha256",
        "training_recipe_fingerprint",
        "intended_use",
        "max_training_tokens",
    ]
    assert protocol["budget_contract"]["kind"] == "explicit_user_declared_max_training_tokens"
    assert protocol["budget_contract"]["hidden_retention_fraction"] is False
    assert protocol["ranking"] == "descending_expected_marginal_contribution_per_token_then_group_id"
    assert protocol["requires_mid_estimator_report"] == "mid-quality-development-report-v1"
    assert "target_retention_fraction" in protocol["forbidden_inputs"]
    assert "budget" not in protocol["forbidden_inputs"]

    print("[hard-quality-protocol] explicit opt-in budget boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
