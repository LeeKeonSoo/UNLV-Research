#!/usr/bin/env python3
"""Run one frozen Math-compatible NeMo Curator recipe."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Final

from external_baselines.nemo_curator_code import (
    FrozenRecipe,
    canonicalize,
    count_records,
    load_frozen_recipe as load_code_recipe,
    run_dedup,
    sha256_file,
    token_counts,
)


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE: Final = ROOT / "protocols" / "math_7m_nemo_curator_baseline_v1.json"


@dataclass(frozen=True, slots=True)
class MathFilterConfig:
    min_words: int
    max_words: int
    max_word_length: int
    remove_boilerplate_at_edges: bool
    max_boilerplate_ratio: float
    max_repeated_line_fraction: float
    max_repeated_lines_char_ratio: float
    max_repeated_paragraphs_ratio: float
    max_repeated_paragraphs_char_ratio: float
    top_ngram_ratios: tuple[tuple[int, float], ...]


def load_frozen_recipe(path: Path = DEFAULT_RECIPE) -> FrozenRecipe:
    """Load the shared runtime fields from the frozen Math protocol."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    recipe = raw["recipe"]
    fuzzy = recipe["fuzzy_dedup"]
    return FrozenRecipe(
        protocol_path=path,
        image_digest=f"nemo-curator=={raw['environment']['nemo_curator_version']}",
        input_path=Path(raw["input"]["path"]),
        input_sha256=str(raw["input"]["sha256"]),
        input_records=int(raw["input"]["records"]),
        output_root=Path(raw["output_root"]),
        text_field=str(raw["input"]["text_field"]),
        id_field=str(raw["input"]["id_field"]),
        code_filters=tuple(str(name) for name in recipe["order"][:-2]),
        fuzzy_seed=int(fuzzy["seed"]),
        fuzzy_char_ngrams=int(fuzzy["char_ngrams"]),
        fuzzy_num_bands=int(fuzzy["num_bands"]),
        fuzzy_minhashes_per_band=int(fuzzy["minhashes_per_band"]),
        target_token_budget=recipe["target_token_budget"],
        tokenizer_path=Path(raw["tokenizer"]["snapshot_path"]),
        sequence_length=int(raw["tokenizer"]["sequence_length"]),
        gradient_accumulation_steps=int(raw["tokenizer"]["gradient_accumulation_steps"]),
        claim_boundary=str(raw["claim_boundary"]),
    )


def load_filter_config(path: Path = DEFAULT_RECIPE) -> MathFilterConfig:
    """Parse the explicit Math-compatible heuristic thresholds."""
    filters = json.loads(path.read_text(encoding="utf-8"))["recipe"]["filters"]
    return MathFilterConfig(
        min_words=int(filters["word_count"]["min_words"]),
        max_words=int(filters["word_count"]["max_words"]),
        max_word_length=int(filters["long_word"]["max_word_length"]),
        remove_boilerplate_at_edges=bool(filters["boilerplate"]["remove_if_at_top_or_bottom"]),
        max_boilerplate_ratio=float(filters["boilerplate"]["max_boilerplate_string_ratio"]),
        max_repeated_line_fraction=float(filters["repeated_lines"]["max_repeated_line_fraction"]),
        max_repeated_lines_char_ratio=float(filters["repeated_lines_by_char"]["max_repeated_lines_char_ratio"]),
        max_repeated_paragraphs_ratio=float(filters["repeated_paragraphs"]["max_repeated_paragraphs_ratio"]),
        max_repeated_paragraphs_char_ratio=float(filters["repeated_paragraphs_by_char"]["max_repeated_paragraphs_char_ratio"]),
        top_ngram_ratios=tuple(
            (int(filters[key]["n"]), float(filters[key]["max_repeating_ngram_ratio"]))
            for key in ("top_ngram_2", "top_ngram_3", "top_ngram_4")
        ),
    )


def run_math_filters(
    recipe: FrozenRecipe,
    config: MathFilterConfig,
) -> tuple[Path, list[dict[str, int | str]]]:
    """Apply only notation-safe NeMo heuristics and retain per-filter counts."""
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.filters import ScoreFilter
    from nemo_curator.stages.text.filters.heuristic import (
        BoilerPlateStringFilter,
        LongWordFilter,
        WordCountFilter,
    )
    from nemo_curator.stages.text.filters.heuristic.repetition import (
        RepeatedLinesByCharFilter,
        RepeatedLinesFilter,
        RepeatedParagraphsByCharFilter,
        RepeatedParagraphsFilter,
        RepeatingTopNGramsFilter,
    )
    from nemo_curator.stages.text.io.reader import JsonlReader
    from nemo_curator.stages.text.io.writer import JsonlWriter

    filters = [
        ("WordCountFilter", WordCountFilter(min_words=config.min_words, max_words=config.max_words)),
        ("LongWordFilter", LongWordFilter(max_word_length=config.max_word_length)),
        (
            "BoilerPlateStringFilter",
            BoilerPlateStringFilter(
                remove_if_at_top_or_bottom=config.remove_boilerplate_at_edges,
                max_boilerplate_string_ratio=config.max_boilerplate_ratio,
            ),
        ),
        ("RepeatedLinesFilter", RepeatedLinesFilter(config.max_repeated_line_fraction)),
        ("RepeatedLinesByCharFilter", RepeatedLinesByCharFilter(config.max_repeated_lines_char_ratio)),
        ("RepeatedParagraphsFilter", RepeatedParagraphsFilter(config.max_repeated_paragraphs_ratio)),
        (
            "RepeatedParagraphsByCharFilter",
            RepeatedParagraphsByCharFilter(config.max_repeated_paragraphs_char_ratio),
        ),
    ]
    filters.extend(
        (f"RepeatingTopNGramsFilter-{n}", RepeatingTopNGramsFilter(n=n, max_repeating_ngram_ratio=ratio))
        for n, ratio in config.top_ngram_ratios
    )
    current = recipe.input_path
    before = recipe.input_records
    audit: list[dict[str, int | str]] = []
    for index, (name, filter_obj) in enumerate(filters, start=1):
        output = recipe.output_root / "filters" / f"{index:02d}_{name}"
        Pipeline(
            name=f"nemo_math_{name}",
            stages=[
                JsonlReader(file_paths=str(current), fields=[recipe.id_field, recipe.text_field]),
                ScoreFilter(filter_obj=filter_obj, text_field=recipe.text_field),
                JsonlWriter(path=str(output), fields=[recipe.id_field, recipe.text_field], mode="overwrite"),
            ],
        ).run()
        after = count_records(output)
        audit.append({"filter": name, "input_records": before, "retained_records": after, "removed_records": before - after})
        current = output
        before = after
    return current, audit


def run(recipe_path: Path = DEFAULT_RECIPE) -> Path:
    """Materialize the frozen Math baseline and its exact audit report."""
    recipe = load_frozen_recipe(recipe_path)
    if sha256_file(recipe.input_path) != recipe.input_sha256:
        raise RuntimeError("Frozen NeMo Math baseline input SHA-256 mismatch")
    recipe.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    filtered, filter_audit = run_math_filters(recipe, load_filter_config(recipe_path))
    deduplicated, dedup_audit = run_dedup(recipe, filtered)
    curated = recipe.output_root / "curated" / "nemo_curated.jsonl"
    canonicalize(deduplicated, curated, recipe.id_field, recipe.text_field)
    report = {
        "schema_version": "nemo-curator-math-baseline-report-v1",
        "status": "complete",
        "protocol": str(recipe.protocol_path),
        "protocol_sha256": sha256_file(recipe.protocol_path),
        "input": token_counts(recipe.input_path, recipe),
        "filters": filter_audit,
        "deduplication": dedup_audit,
        "curated": token_counts(curated, recipe) | {"path": str(curated), "sha256": sha256_file(curated)},
        "total_seconds": time.perf_counter() - started,
        "target_token_budget": recipe.target_token_budget,
        "claim_boundary": recipe.claim_boundary,
    }
    report_path = recipe.output_root / "nemo_curator_math_7m_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    args = parser.parse_args()
    print(run(args.recipe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
