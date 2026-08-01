#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_math_clean_controls import (
    CleanControlRecord,
    build_materialization_report,
    ensure_candidate_disjoint,
    extract_latex_heading_files,
    extract_latex_patterns,
    extract_xml_units,
    source_file_pattern,
    stable_source_sample,
)
from latex_control_units import extract_latex_heading_units


def test_xml_extraction_preserves_context_and_math() -> None:
    xml = """
    <section><title>Limits</title><example xml:id="e1">
      <title>Derivative</title><p>Show that <m>f'(x)=2x</m> for <m>f(x)=x^2</m>.</p>
    </example></section>
    """

    records = extract_xml_units(xml, "book", "unit.ptx", frozenset({"example"}), 20)

    assert len(records) == 1
    assert records[0].text == "Limits\nDerivative\nShow that f'(x)=2x for f(x)=x^2."
    assert records[0].source_group == "book"


def test_stable_sampling_depends_on_identity_not_input_order() -> None:
    rows = tuple(
        CleanControlRecord.from_text(f"id-{index}", "source", f"payload {index} " * 20, f"p/{index}")
        for index in range(8)
    )

    forward = stable_source_sample(rows, 3)
    reverse = stable_source_sample(tuple(reversed(rows)), 3)

    assert tuple(row.record_id for row in forward) == tuple(row.record_id for row in reverse)


def test_candidate_hash_overlap_is_rejected() -> None:
    control = CleanControlRecord.from_text("control", "source", "same normalized text", "control.txt")
    candidate = CleanControlRecord.from_text("candidate", "candidate", "same   normalized\ntext", "candidate.txt")

    try:
        ensure_candidate_disjoint((control,), frozenset({candidate.normalized_text_sha256}))
    except ValueError as error:
        assert "normalized-text" in str(error)
    else:
        raise AssertionError("Expected normalized-text overlap to fail")


def test_protocol_freezes_score_blind_control_selection() -> None:
    protocol = json.loads((ROOT / "configs" / "math_open_educational_clean_control_protocol_v2.json").read_text())

    assert protocol["selection_blind_to"] == [
        "MathScore",
        "FineMath",
        "candidate_provider_scores",
        "benchmark_results",
        "target_retention_fraction",
    ]
    assert protocol["candidate_disjointness"]["normalized_text_sha256"] == "required"
    assert len(protocol["active_sources"]) >= 3
    assert "openstax/osbooks-physics" not in {source["source_group"] for source in protocol["active_sources"]}


def test_report_identifies_the_protocol_version() -> None:
    row = CleanControlRecord.from_text("r", "s", "substantive math payload " * 20, "p")

    report = build_materialization_report(
        {"schema_version": "math-control-protocol-v2"},
        (row,),
        {"s": 1},
        lambda _text: 42,
    )

    assert report["protocol_schema_version"] == "math-control-protocol-v2"
    assert report["schema_version"] == "math-open-educational-clean-control-report-v2"
    assert report["tokens"] == 42


def test_declared_xml_glob_overrides_the_format_default() -> None:
    assert source_file_pattern("pretext_xml", "**/*.xml") == "**/*.xml"
    assert source_file_pattern("pretext_xml", None) == "**/*.ptx"


def test_multiple_latex_globs_are_combined_without_duplicate_records() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "chapter-a").mkdir()
        (root / "chapter-b").mkdir()
        (root / "chapter-a" / "one.tex").write_text("Complete theorem and proof. " * 20)
        (root / "chapter-b" / "two.tex").write_text("Complete example and explanation. " * 20)

        records = extract_latex_patterns(
            root,
            "book",
            ("chapter-a/**/*.tex", "chapter-*/**/*.tex"),
            256,
        )

    assert len(records) == 2


def test_latex_heading_units_exclude_preamble_and_preserve_complete_sections() -> None:
    text = """Preamble ignored.
\\chapter{Vectors}
This chapter introduces vectors with enough explanatory material to retain.
\\section{Bases}
This section explains bases with enough explanatory material to retain.
\\subsection{Short}
Tiny.
"""

    units = extract_latex_heading_units(text, minimum_characters=60)

    assert tuple(unit.unit_id for unit in units) == ("heading-000000", "heading-000001")
    assert units[0].text.startswith("\\chapter{Vectors}")


def test_latex_heading_files_honor_declared_source_encoding() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        text = "\\chapter{Körper}\n" + "Complete mathematical explanation. " * 20
        (root / "book.tex").write_bytes(text.encode("iso-8859-15"))

        records = extract_latex_heading_files(root, "book", "book.tex", 256, "iso-8859-15")

    assert len(records) == 1
    assert "Körper" in records[0].text


if __name__ == "__main__":
    test_xml_extraction_preserves_context_and_math()
    test_stable_sampling_depends_on_identity_not_input_order()
    test_candidate_hash_overlap_is_rejected()
    test_protocol_freezes_score_blind_control_selection()
    test_report_identifies_the_protocol_version()
    test_declared_xml_glob_overrides_the_format_default()
    test_multiple_latex_globs_are_combined_without_duplicate_records()
    test_latex_heading_units_exclude_preamble_and_preserve_complete_sections()
    test_latex_heading_files_honor_declared_source_encoding()
    print("[math-clean-controls] materialization contract: pass")
