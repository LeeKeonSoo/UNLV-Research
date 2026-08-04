#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_source_pool_contract import (
    ContrastiveSourcePoolProtocol,
    PoolRole,
    Route,
    load_source_pool_protocol,
)


def _protocol() -> ContrastiveSourcePoolProtocol:
    return ContrastiveSourcePoolProtocol.model_validate_json(
        (ROOT / "protocols" / "contrastive_operating_point_source_pool_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_source_pool_has_one_baseline_and_two_eligible_sources_per_route() -> None:
    protocol = _protocol()

    assert len(protocol.sources) == 9
    for route in Route:
        route_sources = [source for source in protocol.sources if source.route is route]
        assert sum(source.pool_role is PoolRole.COMMON_BASELINE for source in route_sources) == 1
        assert sum(source.pool_role is PoolRole.ELIGIBLE_ARM for source in route_sources) == 2
    assert protocol.boundary.normal_and_hard_share_eligible_record_ids is True
    assert protocol.boundary.effect_bins_are_separate_arms is False


def test_source_pool_rejects_baseline_source_reused_by_an_arm() -> None:
    protocol = _protocol()
    payload = protocol.model_dump(mode="json")
    baseline_group = next(
        source.source_group_id
        for source in protocol.sources
        if source.pool_role is PoolRole.COMMON_BASELINE
    )
    eligible_index = next(
        index
        for index, source in enumerate(protocol.sources)
        if source.pool_role is PoolRole.ELIGIBLE_ARM
    )
    payload["sources"][eligible_index]["source_group_id"] = baseline_group

    try:
        ContrastiveSourcePoolProtocol.model_validate(payload)
    except ValidationError as error:
        assert "source_group_ids_not_unique" in str(error)
    else:
        raise AssertionError("baseline/eligible source overlap must fail closed")


def test_source_pool_freezes_exact_remote_revisions_and_disjoint_confirmatory_groups() -> None:
    protocol = _protocol()
    remote = [source for source in protocol.sources if source.location_kind.value == "huggingface_file"]

    assert len(remote) == 3
    assert all(source.revision is not None and source.data_file for source in remote)
    development_groups = {source.source_group_id for source in protocol.sources}
    assert development_groups.isdisjoint(protocol.confirmatory_source_group_ids)
    assert protocol.boundary.source_metadata_selector_visible is False
    assert protocol.boundary.benchmark_outcomes_read is False
    assert protocol.boundary.utility_read is False
    assert len({source.source_group_id for source in protocol.sources}) == 9
    math_remote = next(source for source in remote if source.route is Route.MATH)
    assert math_remote.dataset_id == "common-pile/arxiv_papers"
    assert math_remote.required_text_route is Route.MATH
    assert protocol.sampling.stage_a_policy == "text_only_v2"


def test_v2_revision_changes_only_math_collection_size_after_frozen_preflight() -> None:
    original = _protocol()
    revised = load_source_pool_protocol(
        ROOT / "protocols" / "contrastive_operating_point_source_pool_v2.json"
    )
    original_by_id = {source.source_id: source for source in original.sources}
    revised_by_id = {source.source_id: source for source in revised.sources}

    assert revised_by_id["math-arxiv-papers-raw-v1"].exact_token_collection_target == 5_000_000
    assert revised_by_id["math-arxiv-papers-raw-v1"].collection_output.endswith("_v2.jsonl")
    for source_id in original_by_id.keys() - {"math-arxiv-papers-raw-v1"}:
        assert revised_by_id[source_id] == original_by_id[source_id]


if __name__ == "__main__":
    test_frozen_source_pool_has_one_baseline_and_two_eligible_sources_per_route()
    test_source_pool_rejects_baseline_source_reused_by_an_arm()
    test_source_pool_freezes_exact_remote_revisions_and_disjoint_confirmatory_groups()
    test_v2_revision_changes_only_math_collection_size_after_frozen_preflight()
    print("[contrastive-source-pool-v1] frozen source roles and disjointness: pass")
