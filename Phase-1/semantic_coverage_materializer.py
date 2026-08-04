from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from coverage_contract import (
    CoverageChunk,
    CoverageExecutionScope,
    CoverageRequest,
    CoverageStatus,
    CoverageStratum,
    CoverageView,
    FrozenSimilarity,
    RepresentativeFamily,
    StratumState,
)
from coverage_rematerialization import rematerialize_with_coverage
from model_provider_contract import ProviderManifest
from semantic_coverage_bundle import coverage_tag_from_text
from semantic_coverage_graph import build_multiview_strata


JsonMap = dict[str, Any]


class SemanticCoverageMaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _families(proposals: tuple[JsonMap, ...]) -> tuple[RepresentativeFamily, ...]:
    members: dict[str, set[str]] = {}
    for row in proposals:
        trace = row.get("stage_b_policy")
        if not isinstance(trace, dict):
            continue
        representative = trace.get("representative_chunk_uid")
        if isinstance(representative, str) and representative:
            members.setdefault(representative, {representative}).add(str(row["chunk_uid"]))
    result = []
    for representative, family_members in sorted(members.items()):
        payload = json.dumps(sorted(family_members), separators=(",", ":")).encode()
        evidence = hashlib.sha256(payload).hexdigest()
        result.append(
            RepresentativeFamily(
                f"stage-b-family:{evidence[:16]}",
                frozenset(family_members),
                evidence,
                representative,
            )
        )
    return tuple(result)


def materialize_semantic_coverage(
    *,
    universe: tuple[JsonMap, ...],
    proposed_survivors: tuple[JsonMap, ...],
    removal_proposals: tuple[JsonMap, ...],
    corpus_path: Path,
    graph_path: Path,
    provider: ProviderManifest,
    execution_scope: CoverageExecutionScope,
) -> tuple[list[JsonMap], JsonMap]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    corpus_sha = _sha256(corpus_path)
    if graph.get("corpus_sha256") != corpus_sha:
        raise SemanticCoverageMaterializationError("Coverage graph and corpus hashes differ")
    if graph.get("primary_provider_id") != provider.provider_id:
        raise SemanticCoverageMaterializationError("Coverage graph provider ID differs")
    if graph.get("primary_provider_identity_sha256") != provider.identity_sha256():
        raise SemanticCoverageMaterializationError("Coverage graph provider identity differs")
    by_uid = {str(row["chunk_uid"]): row for row in universe}
    proposed_ids = frozenset(str(row["chunk_uid"]) for row in proposed_survivors)
    if len(by_uid) != len(universe) or not proposed_ids <= set(by_uid):
        raise SemanticCoverageMaterializationError("Coverage materialization universe is invalid")
    corpus_rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    corpus_text = {str(row["uid"]): str(row["text"]) for row in corpus_rows}
    if corpus_text != {uid: str(row["text"]) for uid, row in by_uid.items()}:
        raise SemanticCoverageMaterializationError("Coverage graph text universe differs")
    graph_hash = str(graph["graph_sha256"])
    semantic_strata = tuple(
        CoverageStratum(
            f"semantic-stable-{index}",
            CoverageView.SEMANTIC_SKILL,
            frozenset(group),
            StratumState.STABLE,
            graph_hash,
        )
        for index, group in enumerate(graph["stable_strata"])
    ) + tuple(
        CoverageStratum(
            f"semantic-uncertain-{index}",
            CoverageView.UNCERTAIN_INTERSECTION,
            frozenset(group),
            StratumState.UNCERTAIN,
            graph_hash,
        )
        for index, group in enumerate(graph["uncertain_strata"])
    )
    tags = tuple(
        coverage_tag_from_text(uid, str(row["text"])) for uid, row in sorted(by_uid.items())
    )
    request = CoverageRequest(
        chunks=tuple(
            CoverageChunk(uid, max(1, int(row.get("token_proxy") or len(str(row["text"]).split()))))
            for uid, row in sorted(by_uid.items())
        ),
        proposed_survivors=proposed_ids,
        strata=semantic_strata + build_multiview_strata(tags, corpus_sha),
        redundancy_families=_families(removal_proposals),
        similarities=tuple(
            FrozenSimilarity(
                edge["left_uid"], edge["right_uid"], float(edge["similarity"]), graph_hash
            )
            for edge in graph["similarities"]
        ),
        exclusions=(),
        provider_id=provider.provider_id,
        provider_identity_sha256=provider.identity_sha256(),
        execution_scope=execution_scope,
    )
    result = rematerialize_with_coverage(request, provider)
    if result.initial_decision.status is CoverageStatus.ABSTAIN:
        raise SemanticCoverageMaterializationError(result.initial_decision.reason_code)
    final_ids = frozenset(result.final_survivor_uids)
    restored = set(result.required_retain_uids)
    final = []
    for uid, row in by_uid.items():
        if uid not in final_ids:
            continue
        materialized = dict(row)
        materialized["stage_c_coverage"] = {
            "action": "veto_retain" if uid in restored else "accept_stage_b_proposal",
            "graph_sha256": graph_hash,
        }
        final.append(materialized)
    audit = {
        "schema_version": "semantic-coverage-materialization-v1",
        "execution_scope": execution_scope.value,
        "corpus_sha256": corpus_sha,
        "graph_sha256": graph_hash,
        "provider_id": provider.provider_id,
        "provider_identity_sha256": provider.identity_sha256(),
        "initial_status": result.initial_decision.status.value,
        "final_status": result.final_decision.status.value,
        "required_retain_uids": list(result.required_retain_uids),
        "complete_recheck_passed": result.final_decision.status is CoverageStatus.PASS,
        "rematerialization_applied": result.rematerialization_applied,
        "proposed_survivors": len(proposed_ids),
        "final_survivors": len(final),
        "may_create_new_removal": False,
        "scientific_promotion_claimed": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    return final, audit
