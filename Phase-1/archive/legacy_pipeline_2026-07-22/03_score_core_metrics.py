#!/usr/bin/env python3
"""Canonical entrypoint: score canonical selection metrics and diagnostics for every chunk."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from data_eval_common import (
    CORE_SELECTION_METRICS,
    DIAGNOSTIC_METRICS,
    INDEX_DIR,
    METRIC_SPEC_PATH,
    PROJECT_DIR,
    QUALITY_REFERENCE_META_PATH,
    QUALITY_REFERENCE_MODEL_PATH,
    SCHEMA_VERSION,
    SCORED_DIR,
    fingerprint_files,
    scoring_metric_spec_fingerprint,
    save_json,
    sha256_file,
)
from signals.core import CoreMetricScorer


INDEX_DB_PATH = INDEX_DIR / "index.sqlite"
SCORING_MANIFEST_PATH = SCORED_DIR / "scoring_manifest.json"
BATCH_SIZE = 2048
SCORING_SOURCE_FILES = (
    Path("03_score_core_metrics.py"),
    Path("signals") / "core.py",
    Path("quality") / "reference_quality.py",
    Path("data_eval_common.py"),
)


def _artifact_hash(path: Path) -> Dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_DIR / path
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "sha256": sha256_file(resolved) if resolved.exists() else None,
    }


def build_scoring_reproducibility_manifest(index_db_path: Path) -> Dict[str, Any]:
    source_files = {str(path): _artifact_hash(path) for path in SCORING_SOURCE_FILES}
    model_artifacts = {
        "reference_quality_model": _artifact_hash(QUALITY_REFERENCE_MODEL_PATH),
        "reference_quality_metadata": _artifact_hash(QUALITY_REFERENCE_META_PATH),
    }
    return {
        "purpose": (
            "Reproduce score generation beyond the metric spec: scorer source, "
            "reference-quality model artifacts, and index input are part of the "
            "frozen scoring surface."
        ),
        "source_files": source_files,
        "model_artifacts": model_artifacts,
        "index_input": _artifact_hash(index_db_path),
        "complete": all(row["exists"] and row["sha256"] for row in [*source_files.values(), *model_artifacts.values()]),
    }


def split_metric_groups(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split raw scorer output into canonical Core metrics and diagnostics."""
    missing_core = [name for name in CORE_SELECTION_METRICS if name not in metrics]
    missing_diagnostic = [name for name in DIAGNOSTIC_METRICS if name not in metrics]
    if missing_core or missing_diagnostic:
        raise KeyError(
            "scorer output missing metric groups: "
            f"core={missing_core}, diagnostic={missing_diagnostic}"
        )
    core_metrics = {name: metrics[name] for name in CORE_SELECTION_METRICS}
    diagnostic_metrics = {name: metrics[name] for name in DIAGNOSTIC_METRICS}
    overlap = set(core_metrics).intersection(diagnostic_metrics)
    if overlap:
        raise ValueError(f"metric group overlap is forbidden: {sorted(overlap)}")
    return core_metrics, diagnostic_metrics


class RunningStat:
    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.min = None
        self.max = None

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def as_dict(self) -> Dict[str, Any]:
        mean = (self.total / self.n) if self.n else 0.0
        return {
            "n": self.n,
            "mean": round(mean, 6),
            "min": round(self.min, 6) if self.min is not None else None,
            "max": round(self.max, 6) if self.max is not None else None,
        }


def score_all(index_db_path: Path = INDEX_DB_PATH, datasets: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    scorer = CoreMetricScorer(index_db_path=index_db_path)
    conn = sqlite3.connect(str(index_db_path))
    conn.row_factory = sqlite3.Row

    out_files: Dict[str, Any] = {}
    metric_stats: Dict[str, Dict[str, RunningStat]] = defaultdict(dict)
    counts: Dict[str, int] = defaultdict(int)

    dataset_filter = [str(x) for x in (datasets or []) if str(x).strip()]
    where_sql = ""
    params: list[str] = []
    if dataset_filter:
        placeholders = ",".join("?" for _ in dataset_filter)
        where_sql = f"WHERE dataset IN ({placeholders})"
        params.extend(dataset_filter)

    total_row = conn.execute(
        f"SELECT COUNT(*) FROM chunks {where_sql}",
        params,
    ).fetchone()
    total_chunks = int(total_row[0]) if total_row else 0
    cursor = conn.execute(
        """
        SELECT chunk_uid, dataset, source, doc_id, chunk_id, word_count, text, text_hash,
               simhash, simhash_prefix, cluster_id, cluster_size, metadata_json, input_source
        FROM chunks
        {where_sql}
        ORDER BY dataset, simhash_prefix, doc_id, chunk_id
        """.format(where_sql=where_sql),
        params,
    )

    def flush_batch(batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        metrics_batch = scorer.score_chunks_grouped(batch)
        for chunk_meta, metric_groups in zip(batch, metrics_batch):
            dataset = str(chunk_meta["dataset"])
            path = SCORED_DIR / f"{dataset}.jsonl"
            if dataset not in out_files:
                out_files[dataset] = path.open("w", encoding="utf-8")

            core_metrics = metric_groups["core_metrics"]
            diagnostic_metrics = metric_groups["diagnostic_metrics"]
            record = {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset,
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
            out_files[dataset].write(json.dumps(record, ensure_ascii=False) + "\n")
            for name, payload in core_metrics.items():
                metric_stats[dataset].setdefault(name, RunningStat()).add(float(payload["score"]))
            for name, payload in diagnostic_metrics.items():
                metric_stats[dataset].setdefault(name, RunningStat()).add(float(payload["score"]))
            counts[dataset] += 1

    batch: List[Dict[str, Any]] = []
    for row in tqdm(cursor, total=total_chunks, desc="[03] scoring", unit="chunk"):
        metadata = json.loads(str(row["metadata_json"]) or "{}")
        batch.append(
            {
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
        )
        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch = []

    flush_batch(batch)

    for handle in out_files.values():
        handle.close()
    conn.close()
    scorer.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_db_path": str(index_db_path),
        "index_db_sha256": sha256_file(index_db_path) if index_db_path.exists() else None,
        "metric_spec_path": str(METRIC_SPEC_PATH),
        "metric_spec_fingerprint": fingerprint_files([METRIC_SPEC_PATH]),
        "scoring_metric_spec_fingerprint": scoring_metric_spec_fingerprint(METRIC_SPEC_PATH),
        "scoring_reproducibility": build_scoring_reproducibility_manifest(index_db_path),
        "datasets": {},
    }
    if dataset_filter and SCORING_MANIFEST_PATH.exists():
        existing = json.loads(SCORING_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["datasets"] = dict((existing.get("datasets") or {}))
        if existing.get("scoring_reproducibility"):
            manifest["previous_scoring_reproducibility"] = existing["scoring_reproducibility"]
    for dataset in sorted(counts):
        manifest["datasets"][dataset] = {
            "path": str(SCORED_DIR / f"{dataset}.jsonl"),
            "records": counts[dataset],
            "core_metrics": {
                metric: metric_stats[dataset][metric].as_dict()
                for metric in CORE_SELECTION_METRICS
                if metric in metric_stats[dataset]
            },
            "diagnostic_metrics": {
                metric: metric_stats[dataset][metric].as_dict()
                for metric in DIAGNOSTIC_METRICS
                if metric in metric_stats[dataset]
            },
        }
    save_json(SCORING_MANIFEST_PATH, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Score core metrics for all chunks.")
    parser.add_argument("--index-db", type=Path, default=INDEX_DB_PATH)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    manifest = score_all(args.index_db, datasets=args.datasets)
    target_sets = set(str(x) for x in (args.datasets or []))
    for dataset, meta in manifest["datasets"].items():
        if target_sets and dataset not in target_sets:
            continue
        print(f"[03] {dataset}: {meta['records']} records -> {meta['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
