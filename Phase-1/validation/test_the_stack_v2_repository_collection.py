#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_the_stack_v2_repository_samples import collect_repository_disjoint_samples


def test_collection_is_repository_disjoint_and_preserves_source_declared_artifacts() -> None:
    upstream = iter(
        [
            {"blob_id": "a", "content_id": "content-a", "repo_name": "owner/a", "path": "src/a.py", "src_encoding": "UTF-8", "license_type": "permissive", "is_generated": True, "is_vendor": False},
            {"blob_id": "b", "content_id": "content-b", "repo_name": "owner/a", "path": "src/b.py", "src_encoding": "UTF-8", "license_type": "permissive", "is_generated": False, "is_vendor": True},
            {"blob_id": "c", "content_id": "content-c", "repo_name": "owner/c", "path": "src/c.py", "src_encoding": "UTF-8", "license_type": "no_license", "is_generated": False, "is_vendor": False},
        ]
    )
    contents = {"a": "alpha beta gamma", "b": "delta epsilon zeta", "c": "must not be fetched"}

    samples, report = collect_repository_disjoint_samples(
        upstream,
        fetch_content=lambda row: contents[str(row["blob_id"])],
        count_tokens=lambda text: len(text.split()),
        sample_count=3,
        target_tokens=6,
        assignment_seed="fixture-seed",
    )

    records = [record for sample in samples for record in sample]
    assert len(records) == 2
    assert report["skipped_by_license"] == 1
    assert {record["partition"]["repository_identity"] for record in records} == {"owner/a"}
    assert len({record["partition"]["sample_id"] for record in records}) == 1
    assert records[0]["artifact_context"] == {"generation": "generated", "dependency_copy": False}
    assert records[1]["artifact_context"] == {"generation": "authored", "dependency_copy": True}
    assert all(record["rights"] == {"status": "allowed", "license": "permissive"} for record in records)


if __name__ == "__main__":
    test_collection_is_repository_disjoint_and_preserves_source_declared_artifacts()
    print("[the-stack-v2-collection] repository-disjoint source-preserving collection: pass")
