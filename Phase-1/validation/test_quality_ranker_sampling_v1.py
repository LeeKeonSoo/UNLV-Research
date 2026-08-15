from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_ranker_sampling import (
    CalibrationSampleConfig,
    normalized_text_sha256,
    select_calibration_rows,
    select_protected_rows,
)


def _row(uid: str, route: str, token_proxy: int, format_label: str) -> dict:
    return {
        "uid": uid,
        "text": f"payload for {uid}",
        "route_labels": [route],
        "script_labels": ["latin"],
        "format_labels": [format_label],
        "token_proxy": token_proxy,
    }


def test_sampling_is_order_independent_and_spans_observed_strata() -> None:
    rows = [
        *(_row(f"code-short-{index}", "code_artifact", 16, "source_code") for index in range(4)),
        *(_row(f"code-long-{index}", "code_artifact", 700, "source_code") for index in range(4)),
        *(_row(f"prose-short-{index}", "general_prose", 20, "prose") for index in range(4)),
        *(_row(f"prose-long-{index}", "general_prose", 900, "prose") for index in range(4)),
    ]
    config = CalibrationSampleConfig(target_size=8, seed="quality-ranker-v1")

    selected = select_calibration_rows(rows, config)
    reversed_selected = select_calibration_rows(list(reversed(rows)), config)

    assert [row["uid"] for row in selected] == [row["uid"] for row in reversed_selected]
    assert len(selected) == 8
    assert {row["route_labels"][0] for row in selected} == {
        "code_artifact",
        "general_prose",
    }
    assert {"short", "long"} <= {row["quality_calibration_stratum"]["length_bin"] for row in selected}
    assert all(row["quality_calibration_sample"] is True for row in selected)
    assert all(row["chunk_uid"] == row["uid"] for row in selected)


def test_sampling_never_uses_source_reputation_or_dataset_identity() -> None:
    rows = [
        {
            **_row(f"unit-{index}", "code_artifact", 100 + index, "source_code"),
            "source_reputation": "known_high_quality" if index % 2 else "raw_like",
            "source_dataset": "dataset-a" if index < 4 else "dataset-b",
        }
        for index in range(8)
    ]

    selected = select_calibration_rows(
        rows,
        CalibrationSampleConfig(target_size=4, seed="source-blind"),
    )

    assert len(selected) == 4
    assert all(
        set(row["quality_calibration_stratum"]) == {"route", "script", "format", "length_bin"}
        for row in selected
    )


def test_protected_sampling_is_uid_and_normalized_text_disjoint() -> None:
    rows = [
        _row(f"unit-{index}", "general_prose", 100 + index, "prose")
        for index in range(20)
    ]
    rows.append({**_row("duplicate-text", "general_prose", 120, "prose"), "text": rows[0]["text"]})
    calibration = select_calibration_rows(
        rows,
        CalibrationSampleConfig(target_size=8, seed="calibration"),
    )

    protected = select_protected_rows(
        rows,
        calibration_rows=calibration,
        config=CalibrationSampleConfig(target_size=8, seed="protected"),
    )

    calibration_uids = {row["chunk_uid"] for row in calibration}
    protected_uids = {row["chunk_uid"] for row in protected}
    calibration_hashes = {normalized_text_sha256(str(row["text"])) for row in calibration}
    protected_hashes = {normalized_text_sha256(str(row["text"])) for row in protected}
    assert calibration_uids.isdisjoint(protected_uids)
    assert calibration_hashes.isdisjoint(protected_hashes)
    assert len(protected_hashes) == len(protected)
    assert all(row["quality_protected_sample"] is True for row in protected)


if __name__ == "__main__":
    test_sampling_is_order_independent_and_spans_observed_strata()
    test_sampling_never_uses_source_reputation_or_dataset_identity()
    test_protected_sampling_is_uid_and_normalized_text_disjoint()
    print("[quality-ranker-sampling-v1] deterministic source-blind sampling: pass")
