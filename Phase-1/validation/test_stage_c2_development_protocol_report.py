from __future__ import annotations

from stage_c2_development_protocol_report import build_development_protocol_report


def _manifest() -> dict[str, object]:
    return {
        "status": "frozen_proxy_evidence_ready",
        "model_id": "Qwen/Qwen3-4B-Base",
        "model_sha256": "frozen-model",
        "input_records": 512,
        "scoring": {"max_length": 256, "semantic_index": "last_hidden_state_lsh"},
    }


def _audit(removed: int) -> dict[str, object]:
    return {
        "candidate_removed_chunks": removed,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "evidence_thresholds": {"minimum_familiarity": 0.8, "maximum_novelty": 0.2, "maximum_gradient_alignment": 0.05},
    }


def test_build_development_protocol_report_accepts_one_frozen_protocol_for_three_corpora() -> None:
    # Given: code, math, and general artifacts with the same frozen protocol.
    artifacts = {
        "code_raw_like": {"manifest": _manifest(), "audit": _audit(3)},
        "math_raw_like": {"manifest": _manifest(), "audit": _audit(0)},
        "general_text_raw_like": {"manifest": _manifest(), "audit": _audit(3)},
    }

    # When: the protocol integrity report is built.
    report = build_development_protocol_report(artifacts)

    # Then: it records unequal outcomes without treating them as a policy change.
    assert report["protocol_integrity_passed"] is True
    assert report["thresholds_identical_across_corpora"] is True
    assert report["corpora"]["math_raw_like"]["candidate_removed_chunks"] == 0


def test_build_development_protocol_report_rejects_domain_specific_thresholds() -> None:
    # Given: one corpus with a different novelty threshold.
    artifacts = {
        "code_raw_like": {"manifest": _manifest(), "audit": _audit(3)},
        "math_raw_like": {"manifest": _manifest(), "audit": {**_audit(0), "evidence_thresholds": {"minimum_familiarity": 0.8, "maximum_novelty": 0.3, "maximum_gradient_alignment": 0.05}}},
        "general_text_raw_like": {"manifest": _manifest(), "audit": _audit(3)},
    }

    # When / Then: the protocol rejects the domain-specific configuration.
    try:
        build_development_protocol_report(artifacts)
    except RuntimeError as error:
        assert "threshold" in str(error)
    else:
        raise AssertionError("Domain-specific thresholds must be rejected")


if __name__ == "__main__":
    test_build_development_protocol_report_accepts_one_frozen_protocol_for_three_corpora()
    test_build_development_protocol_report_rejects_domain_specific_thresholds()
    print("[stage-c2-development-protocol-report] frozen cross-domain protocol: pass")
