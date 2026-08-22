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


def _validated_artifacts(
    universe: tuple[JsonMap, ...],
    corpus_path: Path,
    graph_path: Path,
    provider: ProviderManifest,
) -> tuple[JsonMap, str, dict[str, JsonMap]]:
    if not corpus_path.is_file() or not graph_path.is_file():
        raise SemanticCoverageMaterializationError(
            "Semantic Coverage corpus or graph is missing"
        )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    corpus_sha = _sha256(corpus_path)
    if graph.get("corpus_sha256") != corpus_sha:
        raise SemanticCoverageMaterializationError("Coverage graph and corpus hashes differ")
    if graph.get("primary_provider_id") != provider.provider_id:
        raise SemanticCoverageMaterializationError("Coverage graph provider ID differs")
    if graph.get("primary_provider_identity_sha256") != provider.identity_sha256():
        raise SemanticCoverageMaterializationError("Coverage graph provider identity differs")
    by_uid = {str(row["chunk_uid"]): row for row in universe}
    if len(by_uid) != len(universe):
        raise SemanticCoverageMaterializationError("Coverage materialization universe is invalid")
    with corpus_path.open(encoding="utf-8") as handle:
        corpus_rows = [json.loads(line) for line in handle if line.strip()]
    corpus_text = {str(row["uid"]): str(row["text"]) for row in corpus_rows}
    if corpus_text != {uid: str(row["text"]) for uid, row in by_uid.items()}:
        raise SemanticCoverageMaterializationError("Coverage graph text universe differs")
    return graph, corpus_sha, by_uid


def validate_semantic_coverage_artifacts(
    *,
    universe: tuple[JsonMap, ...],
    corpus_path: Path,
    graph_path: Path,
    provider: ProviderManifest,
) -> JsonMap:
    graph, corpus_sha, _ = _validated_artifacts(
        universe, corpus_path, graph_path, provider
    )
    return {
        "status": "semantic_coverage_artifacts_ready",
        "corpus_sha256": corpus_sha,
        "graph_sha256": str(graph["graph_sha256"]),
        "provider_id": provider.provider_id,
        "provider_identity_sha256": provider.identity_sha256(),
        "universe_chunks": len(universe),
    }


def _families(proposals: tuple[JsonMap, ...]) -> tuple[RepresentativeFamily, ...]:
    members: dict[str, set[str]] = {}
    identities: dict[str, tuple[str | None, str | None, str]] = {}
    for row in proposals:
        trace = row.get("stage_b_policy")
        if not isinstance(trace, dict):
            continue
        representative = trace.get("representative_chunk_uid")
        if isinstance(representative, str) and representative:
            family_id = trace.get("family_id")
            evidence_sha256 = trace.get("evidence_sha256")
            has_upstream_identity = (
                isinstance(family_id, str)
                and bool(family_id)
                and isinstance(evidence_sha256, str)
                and bool(evidence_sha256)
            )
            key = f"family:{family_id}" if has_upstream_identity else f"representative:{representative}"
            members.setdefault(key, {representative}).add(str(row["chunk_uid"]))
            identity = (
                family_id if has_upstream_identity else None,
                evidence_sha256 if has_upstream_identity else None,
                representative,
            )
            if key in identities and identities[key] != identity:
                raise SemanticCoverageMaterializationError(
                    "Representative family trace identity differs"
                )
            identities[key] = identity
    result = []
    for key, family_members in sorted(members.items()):
        family_id, upstream_evidence, representative = identities[key]
        payload = json.dumps(sorted(family_members), separators=(",", ":")).encode()
        inferred_evidence = hashlib.sha256(payload).hexdigest()
        result.append(
            RepresentativeFamily(
                family_id or f"stage-b-family:{inferred_evidence[:16]}",
                frozenset(family_members),
                upstream_evidence or inferred_evidence,
                representative,
            )
        )
    return tuple(result)


def materialize_semantic_coverage(
    *,
    universe: tuple[JsonMap, ...],
    proposed_survivors: tuple[JsonMap, ...],
    non_selection_proposals: tuple[JsonMap, ...],
    corpus_path: Path,
    graph_path: Path,
    provider: ProviderManifest,
    execution_scope: CoverageExecutionScope,
    restoration_candidate_uids: frozenset[str] | None = None,
    representative_families: tuple[RepresentativeFamily, ...] = (),
) -> tuple[list[JsonMap], JsonMap]:
    graph, corpus_sha, by_uid = _validated_artifacts(
        universe, corpus_path, graph_path, provider
    )
    proposed_ids = frozenset(str(row["chunk_uid"]) for row in proposed_survivors)
    if not proposed_ids <= set(by_uid):
        raise SemanticCoverageMaterializationError("Coverage materialization universe is invalid")
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
    combined_families = tuple(
        {
            family.family_id: family
            for family in (
                *_families(non_selection_proposals),
                *representative_families,
            )
        }.values()
    )
    families_outside_restoration_ceiling = (
        ()
        if restoration_candidate_uids is None
        else tuple(
            family.family_id
            for family in combined_families
            if not family.member_uids & restoration_candidate_uids
        )
    )
    active_families = tuple(
        family
        for family in combined_families
        if family.family_id not in families_outside_restoration_ceiling
    )
    request = CoverageRequest(
        chunks=tuple(
            CoverageChunk(uid, max(1, int(row.get("token_proxy") or len(str(row["text"]).split()))))
            for uid, row in sorted(by_uid.items())
        ),
        proposed_survivors=proposed_ids,
        strata=semantic_strata + build_multiview_strata(tags, corpus_sha),
        redundancy_families=active_families,
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
        restoration_candidate_uids=restoration_candidate_uids,
    )
    result = rematerialize_with_coverage(request, provider)
    if result.initial_decision.status is CoverageStatus.ABSTAIN:
        raise SemanticCoverageMaterializationError(result.initial_decision.reason_code)
    final_ids = frozenset(result.final_survivor_uids)
    if restoration_candidate_uids is not None and not final_ids <= restoration_candidate_uids:
        raise SemanticCoverageMaterializationError(
            "Coverage materialization escaped the declared restoration ceiling"
        )
    restored = set(result.required_retain_uids)
    restoration_trace_by_uid = {
        trace.chunk_uid: trace for trace in result.initial_decision.restoration_traces
    }
    if set(restoration_trace_by_uid) != restored:
        raise SemanticCoverageMaterializationError(
            "Coverage restoration trace does not match required retain UIDs"
        )
    final = []
    for uid, row in by_uid.items():
        if uid not in final_ids:
            continue
        materialized = dict(row)
        coverage_metadata = {
            "action": "veto_retain" if uid in restored else "accept_stage_b_proposal",
            "graph_sha256": graph_hash,
        }
        trace = restoration_trace_by_uid.get(uid)
        if trace is not None:
            coverage_metadata["restoration_trace"] = {
                "group_id": trace.group_id,
                "view": trace.view.value,
                "evidence_artifact_sha256": trace.evidence_artifact_sha256,
                "selection_method": trace.selection_method,
            }
        materialized["stage_c_coverage"] = coverage_metadata
        final.append(materialized)
    restoration_traces = [
        {
            "chunk_uid": trace.chunk_uid,
            "group_id": trace.group_id,
            "view": trace.view.value,
            "evidence_artifact_sha256": trace.evidence_artifact_sha256,
            "selection_method": trace.selection_method,
        }
        for trace in result.initial_decision.restoration_traces
    ]
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
        "restoration_traces": restoration_traces,
        "restoration_trace_complete": len(restoration_traces) == len(restored),
        "complete_recheck_passed": result.final_decision.status is CoverageStatus.PASS,
        "rematerialization_applied": result.rematerialization_applied,
        "proposed_survivors": len(proposed_ids),
        "final_survivors": len(final),
        "may_create_new_removal": False,
        "scientific_promotion_claimed": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "explicit_representative_families_consumed": len(representative_families),
        "representative_families_evaluated": len(active_families),
        "families_outside_restoration_ceiling": list(
            families_outside_restoration_ceiling
        ),
        "restoration_ceiling_applied": restoration_candidate_uids is not None,
        "restoration_candidate_count": (
            len(restoration_candidate_uids)
            if restoration_candidate_uids is not None
            else len(by_uid)
        ),
    }
    return final, audit
