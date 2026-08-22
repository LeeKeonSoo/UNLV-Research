from __future__ import annotations

from coverage_contract import (
    CoverageChunk,
    CoverageRequest,
    ExclusionEvidence,
    RepresentativeFamily,
)
from content_router import route_content
from semantic_coverage_graph import (
    CoverageTag,
    SemanticCoverageContractError,
    SemanticCoverageGraphResult,
    build_multiview_strata,
)


def coverage_tag_from_text(uid: str, text: str) -> CoverageTag:
    routing = route_content(text)
    return CoverageTag(
        uid=uid,
        route_labels=tuple(routing["route_labels"]),
        script_labels=tuple(routing["language_script"]["labels"]),
        format_labels=tuple(routing["content_format"]["labels"]),
        stable=routing["route_confidence"] == "closed_evidence",
    )


def build_semantic_coverage_request(
    *,
    chunks: tuple[CoverageChunk, ...],
    proposed_survivors: frozenset[str],
    redundancy_families: tuple[RepresentativeFamily, ...],
    exclusions: tuple[ExclusionEvidence, ...],
    graph: SemanticCoverageGraphResult,
    tags: tuple[CoverageTag, ...],
    tag_evidence_sha256: str,
    primary_provider_id: str,
) -> CoverageRequest:
    chunk_uids = {chunk.uid for chunk in chunks}
    if {tag.uid for tag in tags} != chunk_uids:
        raise SemanticCoverageContractError(
            "Every Coverage chunk requires exactly one multilingual tag"
        )
    tagged = build_multiview_strata(tags, tag_evidence_sha256)
    strata = graph.strata + tagged
    stratum_ids = tuple(stratum.stratum_id for stratum in strata)
    if len(stratum_ids) != len(set(stratum_ids)):
        raise SemanticCoverageContractError(
            "Semantic and deterministic Coverage stratum IDs must be disjoint"
        )
    return CoverageRequest(
        chunks=chunks,
        proposed_survivors=proposed_survivors,
        strata=strata,
        redundancy_families=redundancy_families,
        similarities=graph.similarities,
        exclusions=exclusions,
        provider_id=primary_provider_id,
        provider_identity_sha256=graph.primary_provider_identity_sha256,
    )
