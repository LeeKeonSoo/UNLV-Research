from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmark_snapshot import build_benchmark_registry
from benchmark_snapshot_contract import (
    BenchmarkAdapter,
    BenchmarkPanel,
    BenchmarkSnapshotContractError,
    BenchmarkSnapshotRegistry,
    BenchmarkSnapshotSpec,
    FrozenBenchmarkRegistry,
    load_benchmark_snapshot_registry,
)


REVISION = "1" * 40


def _write(path: Path, rows: list[dict[str, str | int | list[str] | dict[str, list[str]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _materialize_sources(cache: Path) -> tuple[BenchmarkSnapshotSpec, ...]:
    rows_by_adapter = {
        BenchmarkAdapter.GSM8K: [{"question": "What is 1 + 1?", "answer": "2"}],
        BenchmarkAdapter.HENDRYCKS_MATH: [{"problem": "Solve x=1.", "solution": "x=1", "level": "1", "type": "algebra"}],
        BenchmarkAdapter.MMLU_PRO: [{"question_id": 7, "question": "Pick A", "options": ["A", "B"], "answer": "A", "cot_content": "A follows.", "category": "other"}],
        BenchmarkAdapter.BBH: [{"input": "True and True", "target": "True"}],
        BenchmarkAdapter.ARC_CHALLENGE: [{"id": "arc-1", "question": "Sky color?", "choices": {"text": ["blue", "green"], "label": ["A", "B"]}, "answerKey": "A"}],
        BenchmarkAdapter.HELLASWAG: [
            {"ind": 4, "source_id": "activitynet~clip-1", "ctx": "A person starts", "endings": ["walking", "sleeping"], "label": "0"},
            {"ind": 4, "source_id": "wikihow~4", "ctx": "A cook starts", "endings": ["mixing", "driving"], "label": "0"},
        ],
    }
    panel_by_adapter = {
        BenchmarkAdapter.GSM8K: BenchmarkPanel.MATH,
        BenchmarkAdapter.HENDRYCKS_MATH: BenchmarkPanel.MATH,
        BenchmarkAdapter.MMLU_PRO: BenchmarkPanel.GENERAL,
        BenchmarkAdapter.BBH: BenchmarkPanel.GENERAL,
        BenchmarkAdapter.ARC_CHALLENGE: BenchmarkPanel.GENERAL,
        BenchmarkAdapter.HELLASWAG: BenchmarkPanel.GENERAL,
    }
    specs: list[BenchmarkSnapshotSpec] = []
    for adapter, rows in rows_by_adapter.items():
        source = Path(adapter.value) / "test.parquet"
        _write(cache / source, rows)
        specs.append(
            BenchmarkSnapshotSpec(
                benchmark_id=adapter.value,
                panel=panel_by_adapter[adapter],
                repository_id=f"fixture/{adapter.value}",
                revision=REVISION,
                adapter=adapter,
                declared_split="test",
                source_files=(source.as_posix(),),
            )
        )
    return tuple(specs)


def _registry(cache: Path) -> BenchmarkSnapshotRegistry:
    return BenchmarkSnapshotRegistry(
        schema_version="benchmark-snapshot-registry-v1",
        status="development_exclusion_snapshot_only",
        benchmark_outcomes_available=False,
        selector_membership_mutation_allowed=False,
        source_reputation_used=False,
        snapshots=_materialize_sources(cache),
    )


def test_all_adapters_build_deterministic_canonical_snapshots() -> None:
    # Given one typed task for every frozen Math and General benchmark adapter.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _registry(root / "cache")

        # When snapshots are built twice, then counts and registry hash replay.
        first = build_benchmark_registry(registry, root / "cache", root / "out")
        replay = build_benchmark_registry(registry, root / "cache", root / "out")
        assert first.manifest_sha256 == replay.manifest_sha256
        expected_counts = {adapter.value: 1 for adapter in BenchmarkAdapter}
        expected_counts[BenchmarkAdapter.HELLASWAG.value] = 2
        assert {item.benchmark_id: item.task_count for item in first.snapshots} == expected_counts
        assert all(item.unique_text_hash_count == item.task_count for item in first.snapshots)

        # Then canonical MMLU-Pro text contains question, choices, rationale, and answer.
        mmlu = root / "out" / "mmlu_pro.jsonl"
        record = json.loads(mmlu.read_text(encoding="utf-8"))
        assert record["segments"] == ["Pick A", "A", "B", "A follows.", "A"]

        # Then HellaSwag preserves both source-local tasks whose numeric IDs collide.
        hellaswag = root / "out" / "hellaswag.jsonl"
        task_ids = [json.loads(line)["task_id"] for line in hellaswag.read_text(encoding="utf-8").splitlines()]
        assert task_ids == ["activitynet~clip-1/4", "wikihow~4/4"]


def test_missing_source_file_fails_closed() -> None:
    # Given an otherwise valid registry whose first parquet is absent.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _registry(root / "cache")
        missing = registry.snapshots[0].model_copy(update={"source_files": ("missing/test.parquet",)})
        drifted = registry.model_copy(update={"snapshots": (missing, *registry.snapshots[1:])})

        # When the builder resolves inputs, then no partial registry is emitted.
        try:
            build_benchmark_registry(drifted, root / "cache", root / "out")
        except BenchmarkSnapshotContractError as error:
            assert error.reason_code == "benchmark_source_file_missing"
        else:
            raise AssertionError("A missing benchmark source was accepted")


def test_duplicate_source_file_and_feedback_field_are_rejected() -> None:
    # Given a source file listed twice, then the registry boundary rejects it.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _registry(root / "cache")
        first = registry.snapshots[0]
        try:
            first.model_copy(update={"source_files": (first.source_files[0], first.source_files[0])}).validate_contract()
        except BenchmarkSnapshotContractError as error:
            assert error.reason_code == "benchmark_source_files_invalid"
        else:
            raise AssertionError("Duplicate benchmark source files were accepted")

        # Given benchmark outcomes, then they cannot enter the parsed contract.
        payload = registry.model_dump(mode="json")
        payload["benchmark_results"] = {"gsm8k": 1.0}
        try:
            BenchmarkSnapshotRegistry.model_validate(payload)
        except ValidationError:
            pass
        else:
            raise AssertionError("Benchmark outcomes entered the snapshot contract")


def test_frozen_repository_registry_has_exact_six_benchmarks() -> None:
    # Given the repository registry, when loaded, then its six revisions are pinned.
    registry = load_benchmark_snapshot_registry(ROOT / "protocols" / "math_general_benchmark_snapshot_registry_v1.json")
    assert {item.benchmark_id for item in registry.snapshots} == {adapter.value for adapter in BenchmarkAdapter}
    assert all(len(item.revision) == 40 for item in registry.snapshots)


def test_frozen_manifest_rejects_metadata_hash_drift() -> None:
    # Given a generated manifest whose task metadata changes after freezing.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen = build_benchmark_registry(_registry(root / "cache"), root / "cache", root / "out")
        payload = frozen.model_dump(mode="json")
        payload["snapshots"][0]["task_count"] = 2

        # When the manifest is parsed, then its pinned hash fails closed.
        try:
            FrozenBenchmarkRegistry.model_validate(payload)
        except BenchmarkSnapshotContractError as error:
            assert error.reason_code == "frozen_benchmark_manifest_hash_mismatch"
        else:
            raise AssertionError("Drifted frozen benchmark metadata was accepted")


if __name__ == "__main__":
    test_all_adapters_build_deterministic_canonical_snapshots()
    test_missing_source_file_fails_closed()
    test_duplicate_source_file_and_feedback_field_are_rejected()
    test_frozen_repository_registry_has_exact_six_benchmarks()
    test_frozen_manifest_rejects_metadata_hash_drift()
    print("[benchmark-snapshot-registry-v1] six canonical Math/General snapshots: pass")
