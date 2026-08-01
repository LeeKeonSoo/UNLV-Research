#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _assert_contains(text: str, phrases: tuple[str, ...]) -> None:
    for phrase in phrases:
        assert phrase in text, phrase


def test_project_docs_state_bounded_claim() -> None:
    docs = "\n".join(
        [
            _read("README.md"),
            _read("HANDOFF.md"),
            _read("docs/framework_requirements_and_test_matrix.md"),
            _read("docs/paper_claim_redefinition.md"),
        ]
    )

    _assert_contains(
        docs,
        (
            "Code natural-budget",
            "Math natural-budget",
            "universal data-quality",
            "deployment-conditioned",
            "Stage C",
        ),
    )


def test_math_postmortem_reports_failed_stage_c_evidence() -> None:
    postmortem = _read("docs/math_domain_failure_postmortem.md")

    _assert_contains(
        postmortem,
        (
            "Math Domain Failure Postmortem",
            "failed Stage-C natural-budget validation",
            "1.495650",
            "1.527065",
            "+0.031415",
            "Utility remains Stage C only",
        ),
    )


def test_paper_tex_rejects_universal_quality_claim() -> None:
    paper = _read("paper/curation_stage_framework_ieee.tex")

    _assert_contains(
        paper,
        (
            "positive code-domain validation",
            "negative math-domain validation",
            "EvalPlus",
            "universal data-quality detector",
            "all-domain improvement",
        ),
    )


def main() -> int:
    test_project_docs_state_bounded_claim()
    test_math_postmortem_reports_failed_stage_c_evidence()
    test_paper_tex_rejects_universal_quality_claim()
    print("[paper-claim-redefinition] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
