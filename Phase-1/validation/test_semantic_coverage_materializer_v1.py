#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_contract import CoverageExecutionScope
from coverage_contract import RepresentativeFamily
from model_provider_contract import ProviderLifecycle, ProviderManifest, ProviderRole
from semantic_coverage_materializer import materialize_semantic_coverage


def _provider() -> ProviderManifest:
    return ProviderManifest(
        provider_id="fixture-semantic",
        role=ProviderRole.SEMANTIC,
        provider_type="deterministic",
        lifecycle=ProviderLifecycle.RUNTIME_EXPERIMENT,
        artifacts=(),
        tokenizer_id=None,
        tokenizer_revision=None,
        normalization="fixture",
        output_semantics="fixture",
        supported_routes=("fixture",),
        supported_languages=("fixture",),
        policy_contribution_authority=True,
        direct_deletion_authority=False,
        calibration=None,
        validation=None,
    )


def test_stage_c_restores_only_an_extinct_support_representative() -> None:
    rows = (
        {"chunk_uid": "a", "text": "A proof explains the theorem.", "token_proxy": 5},
        {"chunk_uid": "b", "text": "A second proof explains the theorem.", "token_proxy": 6},
        {"chunk_uid": "c", "text": "def solve(x): return x + 1", "token_proxy": 6},
    )
    corpus_payload = "".join(
        json.dumps({"uid": row["chunk_uid"], "text": row["text"]}) + "\n" for row in rows
    )
    provider = _provider()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        corpus.write_text(corpus_payload, encoding="utf-8")
        corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
        graph = root / "graph.json"
        graph.write_text(
            json.dumps(
                {
                    "schema_version": "semantic-coverage-graph-v1",
                    "corpus_sha256": corpus_sha,
                    "graph_sha256": "3" * 64,
                    "primary_provider_id": provider.provider_id,
                    "primary_provider_identity_sha256": provider.identity_sha256(),
                    "stable_strata": [["a", "b"]],
                    "uncertain_strata": [["c"]],
                    "similarities": [
                        {"left_uid": "a", "right_uid": "b", "similarity": 0.9}
                    ],
                }
            ),
            encoding="utf-8",
        )
        final, audit = materialize_semantic_coverage(
            universe=rows,
            proposed_survivors=(rows[2],),
            removal_proposals=(),
            corpus_path=corpus,
            graph_path=graph,
            provider=provider,
            execution_scope=CoverageExecutionScope.CONFIRMATORY,
            representative_families=(
                RepresentativeFamily(
                    family_id="redundancy-family-a-b",
                    member_uids=frozenset({"a", "b"}),
                    evidence_artifact_sha256="4" * 64,
                    preferred_representative_uid="a",
                ),
            ),
        )

    assert {row["chunk_uid"] for row in final} == {"a", "c"}
    assert audit["required_retain_uids"] == ["a"]
    assert audit["complete_recheck_passed"] is True
    assert audit["may_create_new_removal"] is False
    assert audit["execution_scope"] == "confirmatory"
    assert audit["provider_id"] == provider.provider_id
    assert audit["scientific_promotion_claimed"] is False
    assert audit["explicit_representative_families_consumed"] == 1


if __name__ == "__main__":
    test_stage_c_restores_only_an_extinct_support_representative()
    print("[semantic-coverage-materializer-v1] explicit veto and recheck: pass")
