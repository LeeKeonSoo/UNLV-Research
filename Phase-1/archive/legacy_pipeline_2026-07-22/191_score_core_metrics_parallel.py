#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from data_eval_common import (
    CORE_SELECTION_METRICS,
    DIAGNOSTIC_METRICS,
    INDEX_DIR,
    METRIC_SPEC_PATH,
    SCHEMA_VERSION,
    SCORED_DIR,
    clamp01,
    fingerprint_files,
    save_json,
    scoring_metric_spec_fingerprint,
    sha256_file,
)
from signals.core import CoreMetricScorer


PROJECT_DIR = Path(__file__).resolve().parent
SCORER_SCRIPT_PATH = PROJECT_DIR / "03_score_core_metrics.py"
INDEX_DB_PATH = INDEX_DIR / "index.sqlite"
SCORING_MANIFEST_PATH = SCORED_DIR / "scoring_manifest.json"
BATCH_SIZE = int(os.environ.get("DATA_EVAL_SCORE_BATCH_SIZE", "4096"))
PROGRESS_INTERVAL = int(os.environ.get("DATA_EVAL_SCORE_PROGRESS_INTERVAL", "100000"))


def _load_scorer_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_core_metrics", SCORER_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scorer script: {SCORER_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_datasets(index_db_path: Path) -> list[str]:
    with sqlite3.connect(str(index_db_path)) as conn:
        rows = conn.execute("SELECT DISTINCT dataset FROM chunks ORDER BY dataset").fetchall()
    return [str(row[0]) for row in rows]


def build_scorer_cache(index_db_path: Path) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    simhash_prefix_counts: dict[str, int] = {}
    with sqlite3.connect(str(index_db_path)) as conn:
        total_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        for prefix, count in conn.execute("SELECT simhash_prefix, COUNT(*) FROM chunks GROUP BY simhash_prefix"):
            simhash_prefix_counts[str(prefix)] = int(count)
        for dataset, cluster_size, count in conn.execute(
            "SELECT dataset, cluster_size, COUNT(*) FROM chunks "
            "GROUP BY dataset, cluster_size ORDER BY dataset, cluster_size"
        ):
            grouped[str(dataset)].append((int(cluster_size), int(count)))

    dataset_cluster_size_rarity: dict[str, dict[int, float]] = {}
    for dataset, rows in grouped.items():
        total = sum(count for _, count in rows)
        if total <= 0:
            continue
        csum = 0
        rarity_map: dict[int, float] = {}
        for cluster_size, count in rows:
            csum += count
            rarity_map[int(cluster_size)] = clamp01(1.0 - (csum / total))
        dataset_cluster_size_rarity[dataset] = rarity_map

    return {
        "simhash_prefix_counts": simhash_prefix_counts,
        "dataset_cluster_size_rarity": dataset_cluster_size_rarity,
        "total_chunks": int(total_row[0]) if total_row else 0,
    }


def _query_total(conn: sqlite3.Connection, dataset: str) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks WHERE dataset = ?", [dataset]).fetchone()
    return int(row[0]) if row else 0


def _iter_dataset_rows(conn: sqlite3.Connection, dataset: str) -> sqlite3.Cursor:
    return conn.execute(
        "SELECT chunk_uid, dataset, source, doc_id, chunk_id, word_count, text, text_hash, "
        "simhash, simhash_prefix, cluster_id, cluster_size, metadata_json, input_source "
        "FROM chunks WHERE dataset = ? "
        "ORDER BY simhash_prefix, doc_id, chunk_id",
        [dataset],
    )


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata = json.loads(str(row["metadata_json"]) or "{}")
    return {
        "chunk_uid": str(row["chunk_uid"]),
        "dataset": str(row["dataset"]),
        "source": str(row["source"]),
        "doc_id": str(row["doc_id"]),
        "chunk_id": int(row["chunk_id"]),
        "word_count": int(row["word_count"]),
        "text": str(row["text"]),
        "text_hash": str(row["text_hash"]),
        "simhash": str(row["simhash"]),
        "simhash_prefix": str(row["simhash_prefix"]),
        "cluster_id": int(row["cluster_id"]),
        "cluster_size": int(row["cluster_size"]),
        "input_source": str(row["input_source"]),
        "metadata": metadata,
    }


def _write_scored_record(
    handle: Any,
    chunk_meta: dict[str, Any],
    metric_groups: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    core_metrics = metric_groups["core_metrics"]
    diagnostic_metrics = metric_groups["diagnostic_metrics"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": chunk_meta["dataset"],
        "source": chunk_meta["source"],
        "doc_id": chunk_meta["doc_id"],
        "chunk_id": chunk_meta["chunk_id"],
        "chunk_uid": chunk_meta["chunk_uid"],
        "word_count": chunk_meta["word_count"],
        "core_metrics": core_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "diagnostics": {
            "cluster_id": chunk_meta["cluster_id"],
            "cluster_size": chunk_meta["cluster_size"],
            "text_hash": chunk_meta["text_hash"],
        },
        "provenance": {
            "input_source": chunk_meta["input_source"],
            "metadata": chunk_meta["metadata"],
            "text_preview": chunk_meta["text"][:220] + ("..." if len(chunk_meta["text"]) > 220 else ""),
        },
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return core_metrics, diagnostic_metrics


def score_dataset(index_db_path: Path, dataset: str, scorer_cache: dict[str, Any]) -> dict[str, Any]:
    scorer_module = _load_scorer_script()
    running_stat = scorer_module.RunningStat
    SCORED_DIR.mkdir(parents=True, exist_ok=True)

    scorer = CoreMetricScorer(
        index_db_path=index_db_path,
        simhash_prefix_counts=scorer_cache["simhash_prefix_counts"],
        dataset_cluster_size_rarity=scorer_cache["dataset_cluster_size_rarity"],
        total_chunks=int(scorer_cache["total_chunks"]),
    )
    conn = sqlite3.connect(str(index_db_path))
    conn.row_factory = sqlite3.Row
    total = _query_total(conn, dataset)
    stats: dict[str, dict[str, Any]] = {"core_metrics": {}, "diagnostic_metrics": {}}
    records = 0
    output_path = SCORED_DIR / f"{dataset}.jsonl"
    tmp_path = SCORED_DIR / f"{dataset}.jsonl.tmp"
    output_path.unlink(missing_ok=True)
    tmp_path.unlink(missing_ok=True)
    batch: list[dict[str, Any]] = []

    def flush_batch(handle: Any) -> None:
        nonlocal records
        if not batch:
            return
        before = records
        metrics_batch = scorer.score_chunks_grouped(batch)
        for chunk_meta, metric_groups in zip(batch, metrics_batch):
            core_metrics, diagnostic_metrics = _write_scored_record(handle, chunk_meta, metric_groups)
            for name, payload in core_metrics.items():
                stats["core_metrics"].setdefault(name, running_stat()).add(float(payload["score"]))
            for name, payload in diagnostic_metrics.items():
                stats["diagnostic_metrics"].setdefault(name, running_stat()).add(float(payload["score"]))
            records += 1
        batch.clear()
        if records // PROGRESS_INTERVAL > before // PROGRESS_INTERVAL:
            handle.flush()
            print(f"[191] {dataset}: {records}/{total}", flush=True)

    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in _iter_dataset_rows(conn, dataset):
            batch.append(_record_from_row(row))
            if len(batch) >= BATCH_SIZE:
                flush_batch(handle)
        flush_batch(handle)

    conn.close()
    scorer.close()
    if records != total:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{dataset} scored {records} of {total} records")
    tmp_path.replace(output_path)
    return {
        "path": str(output_path),
        "records": records,
        "expected_records": total,
        "core_metrics": {
            metric: stats["core_metrics"][metric].as_dict()
            for metric in CORE_SELECTION_METRICS
            if metric in stats["core_metrics"]
        },
        "diagnostic_metrics": {
            metric: stats["diagnostic_metrics"][metric].as_dict()
            for metric in DIAGNOSTIC_METRICS
            if metric in stats["diagnostic_metrics"]
        },
    }


def build_manifest(index_db_path: Path, dataset_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scorer_module = _load_scorer_script()
    return {
        "schema_version": SCHEMA_VERSION,
        "index_db_path": str(index_db_path),
        "index_db_sha256": sha256_file(index_db_path) if index_db_path.exists() else None,
        "metric_spec_path": str(METRIC_SPEC_PATH),
        "metric_spec_fingerprint": fingerprint_files([METRIC_SPEC_PATH]),
        "scoring_metric_spec_fingerprint": scoring_metric_spec_fingerprint(METRIC_SPEC_PATH),
        "scoring_reproducibility": scorer_module.build_scoring_reproducibility_manifest(index_db_path),
        "scoring_execution": {
            "runner": str(Path(__file__).name),
            "parallelism": "dataset",
            "batch_size": BATCH_SIZE,
        },
        "datasets": dataset_meta,
    }


def score_datasets_parallel(index_db_path: Path, datasets: Sequence[str], workers: int) -> dict[str, Any]:
    SCORING_MANIFEST_PATH.unlink(missing_ok=True)
    scorer_cache = build_scorer_cache(index_db_path)
    print("[191] shared scorer cache ready", flush=True)
    dataset_meta: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(score_dataset, index_db_path, dataset, scorer_cache): dataset for dataset in datasets}
        for future in as_completed(futures):
            dataset = futures[future]
            meta = future.result()
            if int(meta["records"]) != int(meta["expected_records"]):
                raise RuntimeError(f"{dataset} scored {meta['records']} of {meta['expected_records']} records")
            dataset_meta[dataset] = meta
            print(f"[191] {dataset}: {meta['records']} records -> {meta['path']}", flush=True)
    manifest = build_manifest(index_db_path, dict(sorted(dataset_meta.items())))
    save_json(SCORING_MANIFEST_PATH, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Core metrics in parallel by dataset.")
    parser.add_argument("--index-db", type=Path, default=INDEX_DB_PATH)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()

    datasets = [str(x) for x in (args.datasets or discover_datasets(args.index_db)) if str(x).strip()]
    manifest = score_datasets_parallel(args.index_db, datasets, workers=int(args.workers))
    print(f"[191] manifest: {SCORING_MANIFEST_PATH}")
    print(f"[191] datasets: {', '.join(sorted(manifest['datasets']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
