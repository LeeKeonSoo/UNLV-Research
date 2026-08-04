from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from contrastive_quality_audit import (
    ContrastiveAuditInputs,
    build_contrastive_quality_audit,
)
from contrastive_quality_provider import (
    ContrastiveEvidenceBundle,
    ContrastiveQualityObservation,
    load_contrastive_provider,
)


CONFIG = ROOT / "configs" / "contrastive_quality_provider_qwen3_4b_8b_v1.json"


def _evidence() -> ContrastiveEvidenceBundle:
    records = (
        ContrastiveQualityObservation(
            record_uid="parent",
            route="code_artifact",
            token_ids_sha256="1" * 64,
            scored_token_count=15,
            target_nll=2.0,
            reference_nll=1.5,
            excess_nll=0.5,
            target_entropy=3.0,
            reference_entropy=2.5,
            truncated=False,
        ),
        ContrastiveQualityObservation(
            record_uid="copy",
            route="code_artifact",
            token_ids_sha256="1" * 64,
            scored_token_count=15,
            target_nll=2.0,
            reference_nll=1.5,
            excess_nll=0.5,
            target_entropy=3.0,
            reference_entropy=2.5,
            truncated=False,
        ),
    )
    payload = {
        "schema_version": "contrastive-quality-evidence-bundle-v1",
        "provider_identity_sha256": "a" * 64,
        "scoring_contract_identity_sha256": "b" * 64,
        "target_bundle_sha256": "c" * 64,
        "reference_bundle_sha256": "d" * 64,
        "tokenizer_identity_sha256": "e" * 64,
        "input_artifact_sha256": "f" * 64,
        "records": [item.model_dump(mode="json") for item in records],
        "scalar_quality_score_emitted": False,
        "threshold_decision_emitted": False,
        "runtime_authority": False,
        "direct_deletion_authority": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    from contrastive_quality_provider import hash_json

    return ContrastiveEvidenceBundle(
        evidence_bundle_sha256=hash_json(payload),
        **payload,
    )


def test_audit_reports_observations_without_promoting_a_quality_threshold() -> None:
    provider = load_contrastive_provider(CONFIG)
    sample_rows = (
        {
            "fixture_id": "parent",
            "parent_record_id": "family-1",
            "contrastive_domain": "code",
            "contrastive_route": "code_artifact",
            "contrastive_scenario": "duplicate_heavy",
            "contrastive_source_id": "source-a",
            "metamorphic_relation": "parent-retained-v1",
        },
        {
            "fixture_id": "copy",
            "parent_record_id": "family-1",
            "contrastive_domain": "code",
            "contrastive_route": "code_artifact",
            "contrastive_scenario": "duplicate_heavy",
            "contrastive_source_id": "source-a",
            "metamorphic_relation": "exact-copy-1-v1",
        },
        {
            "fixture_id": "empty",
            "parent_record_id": "family-2",
            "contrastive_domain": "code",
            "contrastive_route": "code_artifact",
            "contrastive_scenario": "malformed",
            "contrastive_source_id": "source-b",
            "metamorphic_relation": "empty-payload-v1",
        },
    )

    report = build_contrastive_quality_audit(
        ContrastiveAuditInputs(
            provider=provider,
            evidence=_evidence(),
            sample_rows=sample_rows,
            sample_artifact_sha256="f" * 64,
            required_routes=("code_artifact",),
            minimum_source_groups_per_route=3,
            empirical_effect_bins_by_route={"code_artifact": 0},
            common_baseline_artifact_sha256=None,
            provider_training_disjointness_artifact_sha256=None,
        )
    )

    assert report.status == "blocked"
    assert report.scored_record_count == 2
    assert report.omitted_record_count == 1
    assert report.omitted_relation_counts == {"empty-payload-v1": 1}
    assert report.exact_copy_consistency.checked_copy_count == 1
    assert report.exact_copy_consistency.mismatch_count == 0
    assert report.route_reports[0].source_group_count == 2
    assert len(report.evidence_group_reports) == 2
    exact_delta = next(
        item for item in report.relation_delta_reports if item.relation == "exact-copy-1-v1"
    )
    assert exact_delta.paired_record_count == 1
    assert exact_delta.excess_nll_delta.mean == 0.0
    assert "reference_quantization_unvalidated" in report.blocker_codes
    assert "shared_tokenizer_compatibility_unverified" in report.blocker_codes
    assert "provider_training_disjointness_unverified" in report.blocker_codes
    assert "common_baseline_missing" in report.blocker_codes
    assert "insufficient_source_groups:code_artifact" in report.blocker_codes
    assert "empirical_effect_bins_missing:code_artifact" in report.blocker_codes
    assert report.scalar_quality_score_emitted is False
    assert report.threshold_decision_emitted is False
    assert report.runtime_activation is False


if __name__ == "__main__":
    test_audit_reports_observations_without_promoting_a_quality_threshold()
    print("[contrastive-quality-audit-v1] descriptive evidence and blockers: pass")
