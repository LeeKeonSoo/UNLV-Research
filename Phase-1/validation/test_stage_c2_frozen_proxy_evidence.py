from __future__ import annotations

from stage_c2_frozen_proxy_evidence import build_frozen_proxy_evidence


def test_build_frozen_proxy_evidence_calibrates_familiarity_without_quality_input() -> None:
    # Given: frozen LM outputs with lower proxy NLL for a familiar chunk.
    rows = [
        {"chunk_uid": "known", "semantic_bucket": "bucket", "embedding": [1.0, 0.0], "proxy_nll": 0.1, "gradient_alignment": 0.0},
        {"chunk_uid": "new", "semantic_bucket": "bucket", "embedding": [0.9, 0.1], "proxy_nll": 1.0, "gradient_alignment": 0.6},
    ]

    # When: the frozen evidence artifact is calibrated.
    evidence, manifest = build_frozen_proxy_evidence(rows, {"model_id": "frozen-fixture", "model_sha256": "abc", "calibration_snapshot_sha256": "def"})

    # Then: the output exposes only selector evidence and an immutable model manifest.
    by_uid = {row["chunk_uid"]: row for row in evidence}
    assert by_uid["known"]["familiarity"] > by_uid["new"]["familiarity"]
    assert by_uid["known"]["novelty"] < by_uid["new"]["novelty"]
    assert "quality_score" not in by_uid["known"]
    assert manifest["status"] == "frozen_proxy_evidence_ready"


def test_build_frozen_proxy_evidence_rejects_quality_field() -> None:
    # Given: an upstream record that attempts to carry an intrinsic quality field.
    rows = [{"chunk_uid": "bad", "semantic_bucket": "bucket", "embedding": [1.0], "proxy_nll": 0.1, "gradient_alignment": 0.0, "quality_score": 0.9}]

    # When / Then: calibration rejects the forbidden field.
    try:
        build_frozen_proxy_evidence(rows, {"model_id": "frozen", "model_sha256": "abc", "calibration_snapshot_sha256": "def"})
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("Quality field must not enter frozen proxy evidence")


if __name__ == "__main__":
    test_build_frozen_proxy_evidence_calibrates_familiarity_without_quality_input()
    test_build_frozen_proxy_evidence_rejects_quality_field()
    print("[stage-c2-frozen-proxy-evidence] frozen evidence calibration: pass")
