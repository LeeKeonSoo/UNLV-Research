#!/usr/bin/env python3
"""Run one frozen NeMo Curator recipe on the audited Code corpus."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE: Final = ROOT / "protocols" / "code_7m_nemo_curator_baseline_v1.json"


@dataclass(frozen=True, slots=True)
class FrozenRecipe:
    protocol_path: Path
    image_digest: str
    input_path: Path
    input_sha256: str
    input_records: int
    output_root: Path
    text_field: str
    id_field: str
    code_filters: tuple[str, ...]
    fuzzy_seed: int
    fuzzy_char_ngrams: int
    fuzzy_num_bands: int
    fuzzy_minhashes_per_band: int
    target_token_budget: int | None
    tokenizer_path: Path
    sequence_length: int
    gradient_accumulation_steps: int
    claim_boundary: str


def load_frozen_recipe(path: Path = DEFAULT_RECIPE) -> FrozenRecipe:
    """Parse the immutable external-baseline protocol."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    recipe = raw["recipe"]
    fuzzy = recipe["fuzzy_dedup"]
    return FrozenRecipe(
        protocol_path=path,
        image_digest=str(raw["image"]["digest"]),
        input_path=Path(raw["input"]["path"]),
        input_sha256=str(raw["input"]["sha256"]),
        input_records=int(raw["input"]["records"]),
        output_root=Path(raw["output_root"]),
        text_field=str(raw["input"]["text_field"]),
        id_field=str(raw["input"]["id_field"]),
        code_filters=tuple(str(item["name"]) for item in recipe["code_filters"]),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_files(path: Path) -> tuple[Path, ...]:
    return tuple(sorted((*path.rglob("*.jsonl"), *path.rglob("*.json"))))


def count_records(path: Path) -> int:
    return sum(1 for file_path in jsonl_files(path) for line in file_path.open(encoding="utf-8") if line.strip())


def run_code_filters(recipe: FrozenRecipe) -> tuple[Path, list[dict[str, int | str]]]:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.filters import ScoreFilter
    from nemo_curator.stages.text.filters.heuristic.code import (
        AlphaFilter,
        NumberOfLinesOfCodeFilter,
        PythonCommentToCodeFilter,
        XMLHeaderFilter,
    )
    from nemo_curator.stages.text.io.reader import JsonlReader
    from nemo_curator.stages.text.io.writer import JsonlWriter

    filters = (
        ("PythonCommentToCodeFilter", PythonCommentToCodeFilter()),
        ("NumberOfLinesOfCodeFilter", NumberOfLinesOfCodeFilter()),
        ("AlphaFilter", AlphaFilter()),
        ("XMLHeaderFilter", XMLHeaderFilter()),
    )
    current = recipe.input_path
    before = recipe.input_records
    audit: list[dict[str, int | str]] = []
    for index, (name, filter_obj) in enumerate(filters, start=1):
        output = recipe.output_root / "filters" / f"{index:02d}_{name}"
        pipeline = Pipeline(
            name=f"nemo_{name}",
            stages=[
                JsonlReader(file_paths=str(current), fields=[recipe.id_field, recipe.text_field]),
                ScoreFilter(filter_obj=filter_obj, text_field=recipe.text_field),
                JsonlWriter(path=str(output), fields=[recipe.id_field, recipe.text_field], mode="overwrite"),
            ],
        )
        pipeline.run()
        after = count_records(output)
        audit.append({"filter": name, "input_records": before, "retained_records": after, "removed_records": before - after})
        current = output
        before = after
    return current, audit


def remove_duplicates(source: Path, ids: Path, output: Path, id_generator: str) -> None:
    from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

    TextDuplicatesRemovalWorkflow(
        input_path=str(source),
        ids_to_remove_path=str(ids),
        output_path=str(output),
        input_filetype="jsonl",
        id_field="_curator_dedup_id",
        duplicate_id_field="_curator_dedup_id",
        id_generator_path=id_generator,
        output_filetype="jsonl",
        output_fields=["record_id", "text"],
        output_mode="overwrite",
    ).run()


def run_dedup(recipe: FrozenRecipe, source: Path) -> tuple[Path, dict[str, int | float]]:
    from nemo_curator.core.client import RayClient
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

    client = RayClient(num_cpus=8, num_gpus=1, include_dashboard=False)
    client.start()
    try:
        exact_root = recipe.output_root / "exact"
        exact = ExactDeduplicationWorkflow(
            input_path=str(source), output_path=str(exact_root), input_filetype="jsonl", text_field=recipe.text_field
        ).run()
        exact_count = int(exact.metadata["num_duplicates"])
        exact_clean = source
        if exact_count:
            exact_clean = recipe.output_root / "exact_clean"
            remove_duplicates(source, exact_root / "ExactDuplicateIds", exact_clean, str(exact.metadata["id_generator_path"]))

        fuzzy_root = recipe.output_root / "fuzzy"
        fuzzy_cache = recipe.output_root / "fuzzy_cache"
        fuzzy = FuzzyDeduplicationWorkflow(
            input_path=str(exact_clean), cache_path=str(fuzzy_cache), output_path=str(fuzzy_root), input_filetype="jsonl",
            text_field=recipe.text_field, seed=recipe.fuzzy_seed, char_ngrams=recipe.fuzzy_char_ngrams,
            num_bands=recipe.fuzzy_num_bands, minhashes_per_band=recipe.fuzzy_minhashes_per_band,
        ).run()
        fuzzy_count = int(fuzzy.metadata.get("num_duplicates", 0))
        final = exact_clean
        if fuzzy_count:
            final = recipe.output_root / "fuzzy_clean"
            remove_duplicates(exact_clean, fuzzy_root / "FuzzyDuplicateIds", final, str(fuzzy.metadata["id_generator_path"]))
        return final, {
            "exact_duplicates": exact_count,
            "exact_seconds": float(exact.metadata["total_time"]),
            "fuzzy_duplicates": fuzzy_count,
            "fuzzy_seconds": float(fuzzy.metadata["total_time"]),
        }
    finally:
        client.stop()


def canonicalize(source: Path, output: Path, id_field: str, text_field: str) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for file_path in jsonl_files(source):
            with file_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sink.write(json.dumps({"record_id": str(row[id_field]), "text": str(row[text_field])}, ensure_ascii=True, sort_keys=True) + "\n")
                    records += 1
    return records


def token_counts(path: Path, recipe: FrozenRecipe) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(recipe.tokenizer_path, local_files_only=True, use_fast=True)
    stream_tokens = 0
    records = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stream_tokens += len(tokenizer(str(row["text"]), add_special_tokens=False).input_ids) + 1
            records += 1
    group = recipe.sequence_length * recipe.gradient_accumulation_steps
    materialized = stream_tokens // group * group
    return {"records": records, "stream_tokens": stream_tokens, "materialized_tokens": materialized, "dropped_tail_tokens": stream_tokens - materialized}


def run(recipe_path: Path = DEFAULT_RECIPE) -> Path:
    recipe = load_frozen_recipe(recipe_path)
    if sha256_file(recipe.input_path) != recipe.input_sha256:
        raise RuntimeError("Frozen NeMo baseline input SHA-256 mismatch")
    recipe.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    filtered, filter_audit = run_code_filters(recipe)
    deduplicated, dedup_audit = run_dedup(recipe, filtered)
    curated = recipe.output_root / "curated" / "nemo_curated.jsonl"
    canonicalize(deduplicated, curated, recipe.id_field, recipe.text_field)
    report = {
        "schema_version": "nemo-curator-code-baseline-report-v1",
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
    report_path = recipe.output_root / "nemo_curator_code_7m_report.json"
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
