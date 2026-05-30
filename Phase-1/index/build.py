#!/usr/bin/env python3
"""Build reusable index artifacts for data evaluation."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm

from data_eval_common import (
    INDEX_DIR,
    PROJECT_DIR,
    SCHEMA_VERSION,
    build_chunk_uid,
    fingerprint_files,
    iter_chunks,
    normalize_dataset_config,
    safe_float,
    save_json,
    simhash64,
    simhash_prefix,
    text_hash,
)


INDEX_DB_PATH = INDEX_DIR / "index.sqlite"
INDEX_META_PATH = INDEX_DIR / "index_manifest.json"
VECTORIZER_FEATURES = 2**16
CLUSTER_BATCH_SIZE = 512


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS hash_counts")
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_uid TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            source TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            simhash TEXT NOT NULL,
            simhash_prefix TEXT NOT NULL,
            cluster_id INTEGER,
            cluster_size INTEGER,
            metadata_json TEXT NOT NULL,
            input_source TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hash_counts (
            text_hash TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_chunks_prefix ON chunks(simhash_prefix)")
    conn.execute("CREATE INDEX idx_chunks_hash ON chunks(text_hash)")
    conn.execute("CREATE INDEX idx_chunks_cluster ON chunks(cluster_id)")
    conn.execute("CREATE INDEX idx_chunks_dataset_uid ON chunks(dataset, chunk_uid)")
    conn.execute("CREATE INDEX idx_chunks_dataset_cluster_uid ON chunks(dataset, cluster_id, chunk_uid)")
    conn.commit()
    return conn


def _count_chunks(specs: List[Dict[str, Any]]) -> int:
    return sum(1 for _ in iter_chunks(specs))


def _cluster_count(total_chunks: int) -> int:
    if total_chunks <= 0:
        return 1
    if total_chunks < 32:
        return max(2, total_chunks // 2)
    return min(64, max(8, int(math.sqrt(total_chunks / 2.0))))


def build_index(dataset_config: Path) -> Dict[str, Any]:
    specs = normalize_dataset_config(dataset_config)
    total_chunks = _count_chunks(specs)
    clusters = _cluster_count(total_chunks)

    vectorizer = HashingVectorizer(
        n_features=VECTORIZER_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=max(1, clusters),
        random_state=42,
        batch_size=CLUSTER_BATCH_SIZE,
        n_init=3,
    )

    conn = _open_db(INDEX_DB_PATH)
    hash_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()

    batch_texts: List[str] = []

    def _flush_fit() -> None:
        nonlocal batch_texts
        if not batch_texts:
            return
        matrix = vectorizer.transform(batch_texts)
        kmeans.partial_fit(matrix)
        batch_texts = []

    insert_rows: List[tuple] = []
    for chunk in tqdm(iter_chunks(specs), total=total_chunks, desc="[02] index pass 1", unit="chunk"):
        simhash_value = simhash64(chunk["text"])
        h = text_hash(chunk["text"])
        hash_counts[h] += 1
        dataset_counts[chunk["dataset"]] += 1
        insert_rows.append(
            (
                chunk["chunk_uid"],
                chunk["dataset"],
                chunk["source"],
                chunk["doc_id"],
                int(chunk["chunk_id"]),
                int(chunk["word_count"]),
                chunk["text"],
                h,
                f"{simhash_value:016x}",
                simhash_prefix(simhash_value),
                json.dumps(chunk["metadata"], ensure_ascii=False),
                chunk["input_source"],
            )
        )
        batch_texts.append(chunk["text"])
        if len(insert_rows) >= 1000:
            conn.executemany(
                """
                INSERT INTO chunks(
                    chunk_uid, dataset, source, doc_id, chunk_id, word_count,
                    text, text_hash, simhash, simhash_prefix, metadata_json, input_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
            conn.commit()
            insert_rows = []
        if len(batch_texts) >= CLUSTER_BATCH_SIZE:
            _flush_fit()

    if insert_rows:
        conn.executemany(
            """
            INSERT INTO chunks(
                chunk_uid, dataset, source, doc_id, chunk_id, word_count,
                text, text_hash, simhash, simhash_prefix, metadata_json, input_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
        )
        conn.commit()
    _flush_fit()

    conn.executemany(
        "INSERT INTO hash_counts(text_hash, count) VALUES (?, ?)",
        sorted(hash_counts.items()),
    )
    conn.commit()

    cluster_sizes: Counter[int] = Counter()
    update_rows: List[tuple] = []
    for chunk in tqdm(iter_chunks(specs), total=total_chunks, desc="[02] index pass 2", unit="chunk"):
        matrix = vectorizer.transform([chunk["text"]])
        cluster_id = int(kmeans.predict(matrix)[0])
        cluster_sizes[cluster_id] += 1
        update_rows.append((cluster_id, chunk["chunk_uid"]))
        if len(update_rows) >= 1000:
            conn.executemany(
                "UPDATE chunks SET cluster_id = ? WHERE chunk_uid = ?",
                update_rows,
            )
            conn.commit()
            update_rows = []
    if update_rows:
        conn.executemany(
            "UPDATE chunks SET cluster_id = ? WHERE chunk_uid = ?",
            update_rows,
        )
        conn.commit()

    for cluster_id, size in cluster_sizes.items():
        conn.execute(
            "UPDATE chunks SET cluster_size = ? WHERE cluster_id = ?",
            (int(size), int(cluster_id)),
        )
    conn.commit()
    conn.close()

    input_fingerprint = fingerprint_files(
        [Path(spec["source"]) for spec in specs if spec["format"] == "json_list"]
        + [p for spec in specs if spec["format"] == "json_batch_dir" for p in sorted(Path(spec["source"]).glob(spec["batch_glob"]))]
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "index_version": "v1",
        "dataset_config_path": str(dataset_config),
        "dataset_names": [spec["name"] for spec in specs],
        "total_chunks": total_chunks,
        "cluster_count": len(cluster_sizes),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "dataset_chunk_counts": dict(dataset_counts),
        "index_db_path": str(INDEX_DB_PATH),
        "input_fingerprint": input_fingerprint,
        "vectorizer": {
            "type": "HashingVectorizer",
            "n_features": VECTORIZER_FEATURES,
            "alternate_sign": False,
            "norm": "l2",
        },
        "clustering": {
            "type": "MiniBatchKMeans",
            "random_state": 42,
            "batch_size": CLUSTER_BATCH_SIZE,
            "n_clusters": max(1, clusters),
        },
    }
    save_json(INDEX_META_PATH, manifest)
    return manifest


def main() -> int:
    manifest = build_index(PROJECT_DIR / "datasets_config.json")
    print(f"[02] total_chunks: {manifest['total_chunks']}")
    print(f"[02] cluster_count: {manifest['cluster_count']}")
    print(f"[02] index_db: {manifest['index_db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
