from __future__ import annotations

from stage_c2_model_relative_selector import select_model_relative_candidates


CONFIG = {
    "semantic_index": {"cosine_threshold": 0.98},
    "evidence_thresholds": {
        "minimum_familiarity": 0.80,
        "maximum_novelty": 0.20,
        "maximum_gradient_alignment": 0.05,
    },
}


def _row(chunk_uid: str, evidence: dict[str, object]) -> dict[str, object]:
    return {"chunk_uid": chunk_uid, "stage_c2_proxy_evidence": {"semantic_bucket": "fixture-family", **evidence}, "text": chunk_uid}


def test_select_model_relative_candidates_removes_only_supported_nonrepresentative() -> None:
    # Given: a semantic family with one evidence-supported representative and one redundant member.
    rows = [
        _row("representative", {"embedding": [1.0, 0.0], "familiarity": 0.15, "novelty": 0.85, "gradient_alignment": 0.70}),
        _row("redundant", {"embedding": [0.999, 0.001], "familiarity": 0.92, "novelty": 0.08, "gradient_alignment": 0.01}),
        _row("novel", {"embedding": [0.998, 0.002], "familiarity": 0.90, "novelty": 0.75, "gradient_alignment": 0.01}),
    ]

    # When: candidate-only model-relative selection runs.
    selected, rejected, audit = select_model_relative_candidates(rows, CONFIG)

    # Then: only the redundant candidate is removed with an auditable reason code.
    assert {row["chunk_uid"] for row in selected} == {"representative", "novel"}
    assert [row["chunk_uid"] for row in rejected] == ["redundant"]
    assert rejected[0]["stage_c2_selection"]["removed_reason"] == "model_relative_redundant_family_member"
    assert audit["candidate_removed_chunks"] == 1
    assert audit["runtime_authorization"] == "none_candidate_cannot_select_or_remove"


def test_select_model_relative_candidates_preserves_missing_evidence() -> None:
    # Given: a row without frozen proxy evidence.
    rows = [{"chunk_uid": "unknown", "text": "kept because evidence is unavailable"}]

    # When: candidate-only selection runs.
    selected, rejected, audit = select_model_relative_candidates(rows, CONFIG)

    # Then: the row is retained and explicitly marked not evaluated.
    assert [row["chunk_uid"] for row in selected] == ["unknown"]
    assert rejected == []
    assert audit["not_evaluated_chunks"] == 1


def test_select_model_relative_candidates_rejects_forbidden_policy_input() -> None:
    # Given: evidence carrying a forbidden quality-like field.
    rows = [_row("bad", {"embedding": [1.0, 0.0], "familiarity": 0.9, "novelty": 0.1, "gradient_alignment": 0.0, "quality_score": 1.0})]

    # When / Then: the boundary rejects it before selection.
    try:
        select_model_relative_candidates(rows, CONFIG)
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("Forbidden policy input must be rejected")


def test_select_model_relative_candidates_separates_semantic_and_proxy_ablation_arms() -> None:
    # Given: a semantic family whose nonrepresentative has strong proxy evidence.
    rows = [
        _row("representative", {"embedding": [1.0, 0.0], "familiarity": 0.1, "novelty": 0.9, "gradient_alignment": 0.7}),
        _row("candidate", {"embedding": [0.999, 0.001], "familiarity": 0.9, "novelty": 0.8, "gradient_alignment": 0.7}),
    ]

    # When: semantic-only and proxy-only diagnostic arms run independently.
    _, semantic_rejected, _ = select_model_relative_candidates(rows, {**CONFIG, "ablation_mode": "semantic_only"})
    _, proxy_rejected, _ = select_model_relative_candidates(rows, {**CONFIG, "ablation_mode": "proxy_only"})

    # Then: semantic evidence alone removes the family member while proxy-only does not.
    assert [row["chunk_uid"] for row in semantic_rejected] == ["candidate"]
    assert proxy_rejected == []


if __name__ == "__main__":
    test_select_model_relative_candidates_removes_only_supported_nonrepresentative()
    test_select_model_relative_candidates_preserves_missing_evidence()
    test_select_model_relative_candidates_rejects_forbidden_policy_input()
    test_select_model_relative_candidates_separates_semantic_and_proxy_ablation_arms()
    print("[stage-c2-model-relative-selector] candidate-only selection: pass")
