from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

from data_eval_common import save_json, sha256_file
from paper_evidence.livecodebench_pilot import (
    BenchmarkTask,
    screen_lexical_overlap,
    select_stratified_pilot,
)

PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CACHE: Final = Path(
    r"D:\UNLV-Research\hf_cache\datasets\livecodebench___code_generation_lite"
    r"\release_latest-version_tag=release_latest\0.0.0"
    r"\4c038560f391c4c05fdf7fd7ae61ae0e6dbd8672f8fe5b95597b78a8dc40a417"
)
DEFAULT_LCB_REPO: Final = Path(r"D:\UNLV-Research\third_party\LiveCodeBench")
DEFAULT_RAW: Final = PROJECT_DIR / "outputs" / "code_domain_natural_budget_qwen3_4b" / "raw_full_natural.jsonl"
DEFAULT_CURATED: Final = PROJECT_DIR / "outputs" / "code_domain_natural_budget_qwen3_4b" / "curated_v2_natural.jsonl"
DEFAULT_OUTPUT: Final = PROJECT_DIR / "configs" / "code_livecodebench_pilot_v1.json"
DEFAULT_TASK_BUNDLE: Final = (
    PROJECT_DIR / "outputs" / "code_livecodebench_pilot_qwen3_4b" / "frozen_tasks.json"
)
LCB_COMMIT: Final = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class FreezeReport(TypedDict):
    schema_version: str
    status: str
    scope: str
    selector_tuning_permission: bool
    utility_stage: str
    dataset: JsonValue
    official_runner: JsonValue
    selection: JsonValue
    leakage_screen: JsonValue
    execution: JsonValue
    inputs: JsonValue
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class FreezeInputs:
    dataset_cache: Path
    lcb_repo: Path
    raw_path: Path
    curated_path: Path
    output_path: Path
    task_bundle_path: Path


@dataclass(frozen=True, slots=True)
class FreezeContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _load_dataset(cache_dir: Path):
    from datasets import Dataset, concatenate_datasets

    shards = sorted(cache_dir.glob("*.arrow"))
    if not shards:
        raise FileNotFoundError(cache_dir)
    return concatenate_datasets([Dataset.from_file(str(path)) for path in shards])


def _task_from_row(row) -> BenchmarkTask:
    prompt_text = "\n".join(
        (str(row["question_title"]), str(row["question_content"]), str(row["starter_code"]))
    )
    return BenchmarkTask(
        question_id=str(row["question_id"]),
        contest_date=date.fromisoformat(str(row["contest_date"])[:10]),
        platform=str(row["platform"]),
        difficulty=str(row["difficulty"]),
        prompt_text=prompt_text,
    )


def _canonical_hash(row) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _training_records(path: Path) -> tuple[tuple[str, str], ...]:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(
            (str(row["chunk_uid"]), str(row["text"]))
            for line in handle
            if line.strip()
            for row in (json.loads(line),)
        )


def _git_commit(repo: Path) -> str:
    commit = (repo / ".git" / "HEAD").read_text(encoding="ascii").strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise FreezeContractError(f"LiveCodeBench checkout is not detached: {commit}")
    return commit


def build_freeze(inputs: FreezeInputs) -> FreezeReport:
    commit = _git_commit(inputs.lcb_repo)
    if commit != LCB_COMMIT:
        raise FreezeContractError(f"LiveCodeBench commit mismatch: {commit}")
    dataset = _load_dataset(inputs.dataset_cache)
    window_rows = tuple(
        row
        for row in dataset
        if "2025-01-01" <= str(row["contest_date"])[:10] <= "2025-04-06"
    )
    tasks = tuple(_task_from_row(row) for row in window_rows)
    selected = select_stratified_pilot(tasks, per_cell=8, selection_seed="lcb-pilot-v1")
    selected_ids = {task.question_id for task in selected}
    selected_rows = {
        str(row["question_id"]): row
        for row in window_rows
        if str(row["question_id"]) in selected_ids
    }
    raw_overlap = screen_lexical_overlap(
        _training_records(inputs.raw_path), selected, ngram_size=8
    )
    curated_overlap = screen_lexical_overlap(
        _training_records(inputs.curated_path), selected, ngram_size=8
    )
    execution_allowed = (
        raw_overlap.candidate_count == 0 and curated_overlap.candidate_count == 0
    )
    frozen_rows = [selected_rows[task.question_id] for task in selected]
    save_json(inputs.task_bundle_path, frozen_rows)
    report: FreezeReport = {
        "schema_version": "code-livecodebench-stratified-pilot-v1",
        "status": "frozen_before_outcomes" if execution_allowed else "abstain_overlap_detected",
        "scope": "stage_c_independent_benchmark_diagnostic",
        "selector_tuning_permission": False,
        "utility_stage": "Stage C only",
        "dataset": {
            "id": "livecodebench/code_generation_lite",
            "version_tag": "release_latest",
            "cache_builder_hash": inputs.dataset_cache.name,
            "window_start": "2025-01-01",
            "window_end": "2025-04-06",
            "window_task_count": len(window_rows),
            "frozen_task_bundle": {
                "path": str(inputs.task_bundle_path),
                "sha256": sha256_file(inputs.task_bundle_path),
            },
        },
        "official_runner": {
            "repository": "https://github.com/LiveCodeBench/LiveCodeBench",
            "commit": commit,
            "local_path": str(inputs.lcb_repo),
        },
        "selection": {
            "strategy": "sha256_rank_within_platform_x_difficulty",
            "seed": "lcb-pilot-v1",
            "per_cell": 8,
            "task_count": len(selected),
            "tasks": [
                {
                    "question_id": task.question_id,
                    "contest_date": task.contest_date.isoformat(),
                    "platform": task.platform,
                    "difficulty": task.difficulty,
                    "content_sha256": _canonical_hash(selected_rows[task.question_id]),
                }
                for task in selected
            ],
        },
        "leakage_screen": {
            "method": "exact normalized lexical 8-gram overlap against prompt and starter code",
            "raw_candidate_count": raw_overlap.candidate_count,
            "curated_candidate_count": curated_overlap.candidate_count,
            "raw_candidates": [
                {
                    "training_id": item.training_id,
                    "question_id": item.question_id,
                    "shared_ngram_count": item.shared_ngram_count,
                    "containment": item.containment,
                }
                for item in raw_overlap.candidates
            ],
            "curated_candidates": [
                {
                    "training_id": item.training_id,
                    "question_id": item.question_id,
                    "shared_ngram_count": item.shared_ngram_count,
                    "containment": item.containment,
                }
                for item in curated_overlap.candidates
            ],
            "limitation": "This screen does not prove absence of semantic or solution-level contamination.",
        },
        "execution": {
            "allowed": execution_allowed,
            "arms": ["base_no_update", "raw_full_natural", "curated_v2_natural"],
            "seed": 101,
            "metric": "pass@1",
            "samples_per_task": 1,
            "temperature": 0.0,
            "max_new_tokens": 1024,
            "stop_strings": ["###"],
            "model_style": "GenericBase",
            "required_gpu": "NVIDIA GeForce RTX 3070 Ti",
        },
        "inputs": {
            "raw": {
                "path": str(inputs.raw_path),
                "sha256": sha256_file(inputs.raw_path),
            },
            "curated": {
                "path": str(inputs.curated_path),
                "sha256": sha256_file(inputs.curated_path),
            },
        },
        "claim_boundary": (
            "A 48-task stratified pilot tests whether the EvalPlus gain transfers to a different "
            "competitive-programming format. It is not a full LiveCodeBench score or a universal claim."
        ),
    }
    save_json(inputs.output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    parser.add_argument("--lcb-repo", type=Path, default=DEFAULT_LCB_REPO)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-bundle", type=Path, default=DEFAULT_TASK_BUNDLE)
    args = parser.parse_args()
    inputs = FreezeInputs(
        dataset_cache=args.dataset_cache,
        lcb_repo=args.lcb_repo,
        raw_path=args.raw,
        curated_path=args.curated,
        output_path=args.output,
        task_bundle_path=args.task_bundle,
    )
    print(json.dumps(build_freeze(inputs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
