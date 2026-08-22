#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_contract import CoverageChunk, CoverageView, RepresentativeFamily, StratumState
from semantic_coverage_bundle import build_semantic_coverage_request, coverage_tag_from_text
from semantic_coverage_graph import (
    CoverageTag,
    SemanticCoverageGraphRequest,
    SemanticEmbedding,
    build_multiview_strata,
    build_semantic_coverage_graph,
)


CONTRACT = ROOT / "configs" / "semantic_coverage_v3.json"


def _embedding(uid: str, *values: float) -> SemanticEmbedding:
    return SemanticEmbedding(uid, values)


def test_consensus_components_are_stable_and_provider_disagreement_is_uncertain() -> None:
    request = SemanticCoverageGraphRequest(
        primary_provider_id="primary-multilingual",
        primary_provider_identity_sha256="1" * 64,
        audit_provider_id="audit-multilingual",
        audit_provider_identity_sha256="2" * 64,
        primary_embeddings=(
            _embedding("en-a", 1.0, 0.0, 0.0),
            _embedding("ko-a", 0.99, 0.01, 0.0),
            _embedding("ar-a", 0.0, 1.0, 0.0),
            _embedding("math-a", 0.0, 0.99, 0.01),
            _embedding("mixed-a", 0.0, 0.0, 1.0),
        ),
        audit_embeddings=(
            _embedding("en-a", 1.0, 0.0, 0.0),
            _embedding("ko-a", 0.98, 0.02, 0.0),
            _embedding("ar-a", 0.0, 1.0, 0.0),
            _embedding("math-a", 0.0, 0.98, 0.02),
            _embedding("mixed-a", 0.7, 0.0, 0.7),
        ),
        neighbor_count=1,
    )

    result = build_semantic_coverage_graph(request)
    stable = {
        stratum.member_uids
        for stratum in result.strata
        if stratum.state is StratumState.STABLE
    }
    uncertain = {
        stratum.member_uids
        for stratum in result.strata
        if stratum.state is StratumState.UNCERTAIN
    }

    assert frozenset({"en-a", "ko-a"}) in stable
    assert frozenset({"ar-a", "math-a"}) in stable
    assert any("mixed-a" in members for members in uncertain)
    assert all(stratum.view in {CoverageView.SEMANTIC_SKILL, CoverageView.UNCERTAIN_INTERSECTION} for stratum in result.strata)
    assert result.primary_provider_identity_sha256 != result.audit_provider_identity_sha256
    assert result.benchmark_outcomes_read is False
    assert result.utility_read is False


def test_multilingual_tags_create_views_without_assigning_target_proportions() -> None:
    tags = (
        CoverageTag("code-ko", ("code_artifact",), ("hangul", "latin"), ("source_code",), True),
        CoverageTag("math-ar", ("mathematical_content",), ("arabic",), ("formula",), True),
        CoverageTag("unknown", ("unknown",), ("unknown",), ("unknown",), False),
    )

    strata = build_multiview_strata(tags, "3" * 64)
    by_view = {view: [item for item in strata if item.view is view] for view in CoverageView}

    assert any("code-ko" in item.member_uids for item in by_view[CoverageView.CONTENT_ROUTE])
    assert any("math-ar" in item.member_uids for item in by_view[CoverageView.LANGUAGE_SCRIPT])
    assert any("unknown" in item.member_uids for item in by_view[CoverageView.UNCERTAIN_INTERSECTION])
    assert all(not hasattr(item, "target_share") for item in strata)

    routed = coverage_tag_from_text(
        "routed-ko",
        "# 설명을 보존합니다\nimport math\ndef solve(value):\n    return value + 1",
    )
    assert routed.stable is True
    assert "code_artifact" in routed.route_labels
    assert {"hangul", "latin"} <= set(routed.script_labels)


def test_graph_and_tags_form_one_typed_coverage_request() -> None:
    graph = build_semantic_coverage_graph(
        SemanticCoverageGraphRequest(
            "primary",
            "1" * 64,
            "audit",
            "2" * 64,
            (_embedding("a", 1.0, 0.0), _embedding("b", 0.99, 0.01)),
            (_embedding("a", 1.0, 0.0), _embedding("b", 0.98, 0.02)),
            1,
        )
    )
    tags = (
        CoverageTag("a", ("general_prose",), ("latin",), ("prose",), True),
        CoverageTag("b", ("general_prose",), ("hangul",), ("prose",), True),
    )

    request = build_semantic_coverage_request(
        chunks=(CoverageChunk("a", 10), CoverageChunk("b", 20)),
        proposed_survivors=frozenset({"a"}),
        redundancy_families=(
            RepresentativeFamily("family", frozenset({"a", "b"}), "4" * 64, "a"),
        ),
        exclusions=(),
        graph=graph,
        tags=tags,
        tag_evidence_sha256="3" * 64,
        primary_provider_id="primary",
    )

    assert request.provider_identity_sha256 == graph.primary_provider_identity_sha256
    assert {stratum.view for stratum in request.strata} >= {
        CoverageView.SEMANTIC_SKILL,
        CoverageView.CONTENT_ROUTE,
        CoverageView.LANGUAGE_SCRIPT,
    }


def test_provider_identity_or_uid_drift_is_rejected() -> None:
    rejected_same_provider = False
    try:
        SemanticCoverageGraphRequest(
            "same",
            "1" * 64,
            "same",
            "2" * 64,
            (_embedding("a", 1.0, 0.0), _embedding("b", 0.0, 1.0)),
            (_embedding("a", 1.0, 0.0), _embedding("b", 0.0, 1.0)),
            1,
        )
    except RuntimeError:
        rejected_same_provider = True

    rejected_uid_drift = False
    try:
        SemanticCoverageGraphRequest(
            "primary",
            "1" * 64,
            "audit",
            "2" * 64,
            (_embedding("a", 1.0, 0.0), _embedding("b", 0.0, 1.0)),
            (_embedding("a", 1.0, 0.0), _embedding("c", 0.0, 1.0)),
            1,
        )
    except RuntimeError:
        rejected_uid_drift = True

    assert rejected_same_provider is True
    assert rejected_uid_drift is True


def test_contract_keeps_multilingual_semantics_strong_but_unpromoted() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["runtime_activation"] == "development_and_confirmatory_only"
    assert contract["authority"] == (
        "final_corpus_veto_with_explicit_required_retain_and_full_recheck"
    )
    assert contract["mode_contract"]["normal"] == contract["mode_contract"]["hard"]
    assert contract["mode_contract"]["hard_may_weaken_coverage"] is False
    assert contract["graph"]["embedding_similarity_alone_may_delete"] is False
    assert "domain_quota" in contract["forbidden_inputs"]
    assert "benchmark_outcomes" in contract["forbidden_inputs"]


if __name__ == "__main__":
    test_consensus_components_are_stable_and_provider_disagreement_is_uncertain()
    test_multilingual_tags_create_views_without_assigning_target_proportions()
    test_provider_identity_or_uid_drift_is_rejected()
    test_contract_keeps_multilingual_semantics_strong_but_unpromoted()
    test_graph_and_tags_form_one_typed_coverage_request()
    print("[semantic-coverage-graph-v3] multilingual consensus graph: pass")
