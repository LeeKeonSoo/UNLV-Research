#!/usr/bin/env python3
"""Generate label-template CSVs for Phase-1 metric identity validation.

Outputs:
  validation/labels/domain_labels_v1.csv
  validation/labels/quality_labels_v1.csv
  validation/labels/difficulty_sanity_v1.csv
  validation/labels/ood_labels_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from phase1_autosave import autosave_stage

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
LABEL_DIR = PROJECT_DIR / "validation" / "labels"

ANCHOR_DATASETS = {"khan_academy", "tiny_textbooks"}


def _load_manifest() -> Dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_outputs(manifest: Dict[str, Any]) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    datasets = manifest.get("dataset_versions_and_counts", {}) or {}
    for dataset_name, meta in datasets.items():
        if not isinstance(meta, dict):
            continue
        output_file = Path(str(meta.get("output_file") or f"{dataset_name}_analysis.jsonl")).name
        outputs[dataset_name] = OUTPUT_DIR / output_file
    return outputs


def _safe_float(v: Any):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _safe_bool(v: Any):
    if isinstance(v, bool):
        return v
    return None


def _topk_domain_labels(record: Dict[str, Any], k: int = 3) -> str:
    labels = record.get("domain_labels") or {}
    if not isinstance(labels, dict):
        return ""
    ranked: List[Tuple[str, float]] = []
    for name, score in labels.items():
        s = _safe_float(score)
        if s is None:
            continue
        ranked.append((str(name), s))
    if not ranked:
        return ""
    ranked.sort(key=lambda x: x[1], reverse=True)
    return "|".join(name for name, _ in ranked[:k])


def _text_preview(record: Dict[str, Any], max_chars: int = 320) -> str:
    text = str(record.get("text") or record.get("text_preview") or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _reservoir_sample(path: Path, dataset: str, k: int, rng: random.Random) -> List[Dict[str, Any]]:
    sample: List[Dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            seen += 1
            item = {
                "dataset": dataset,
                "source": str(record.get("source") or dataset),
                "doc_id": str(record.get("doc_id") or ""),
                "chunk_id": int(record.get("chunk_id") or 0),
                "ood_label": str(((record.get("ood") or {}).get("label") or "ood_far")),
                "quality_score": _safe_float(record.get("quality_score")),
                "fk_grade": _safe_float((record.get("difficulty") or {}).get("flesch_kincaid_grade")),
                # Human-labeling aids (read-only hints)
                "text_preview": _text_preview(record),
                "pred_top_domain": str(record.get("top_domain_full") or record.get("top_domain") or ""),
                "pred_top3_domains_pipe": _topk_domain_labels(record, k=3),
                "pred_has_examples": _safe_bool((record.get("educational_markers") or {}).get("has_examples")),
                "pred_has_explanation": _safe_bool((record.get("educational_markers") or {}).get("has_explanation")),
                "pred_has_structure": _safe_bool((record.get("educational_markers") or {}).get("has_structure")),
                "pred_difficulty_valid": _safe_bool((record.get("validity_flags") or {}).get("difficulty_valid")),
                "pred_ood_label": str(((record.get("ood") or {}).get("label") or "")),
            }
            if len(sample) < k:
                sample.append(item)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    sample[j - 1] = item
    return sample


def _quantile_thresholds(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.2, 0.4, 0.6
    values = sorted(values)

    def q(p: float) -> float:
        idx = int(round((len(values) - 1) * p))
        idx = max(0, min(len(values) - 1, idx))
        return float(values[idx])

    return q(0.25), q(0.50), q(0.75)


def _quality_bin(v: float | None, qs: Tuple[float, float, float]) -> str:
    if v is None:
        return "q_unknown"
    q1, q2, q3 = qs
    if v <= q1:
        return "q1"
    if v <= q2:
        return "q2"
    if v <= q3:
        return "q3"
    return "q4"


def _difficulty_bin(v: float | None) -> str:
    if v is None:
        return "d_unknown"
    if v < 4:
        return "d_0_4"
    if v < 8:
        return "d_4_8"
    if v < 12:
        return "d_8_12"
    if v < 16:
        return "d_12_16"
    return "d_16_plus"


def _key(item: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return item["dataset"], item["source"], item["doc_id"], int(item["chunk_id"])


def _group_key(item: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return item["dataset"], item["ood_label"], item["quality_bin"], item["difficulty_bin"]


def _balanced_sample(
    candidates: List[Dict[str, Any]],
    target_n: int,
    datasets: List[str],
    used_keys: set,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if target_n <= 0 or not datasets:
        return []

    pool = [x for x in candidates if x["dataset"] in datasets and _key(x) not in used_keys]
    if not pool:
        return []

    by_group: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_dataset_groups: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)

    for item in pool:
        g = _group_key(item)
        by_group[g].append(item)

    for g in by_group:
        by_dataset_groups[g[0]].append(g)
        rng.shuffle(by_group[g])

    for ds in by_dataset_groups:
        rng.shuffle(by_dataset_groups[ds])

    picked: List[Dict[str, Any]] = []

    def pop_from_dataset(ds: str, need: int):
        if need <= 0:
            return
        groups = by_dataset_groups.get(ds, [])
        if not groups:
            return
        idx = 0
        while need > 0 and groups:
            g = groups[idx % len(groups)]
            bucket = by_group[g]
            if bucket:
                item = bucket.pop()
                k = _key(item)
                if k not in used_keys:
                    used_keys.add(k)
                    picked.append(item)
                    need -= 1
            if not bucket:
                groups.remove(g)
                if not groups:
                    break
            else:
                idx += 1

    # Dataset-balanced first pass
    base = target_n // len(datasets)
    rem = target_n % len(datasets)
    quotas = {ds: base for ds in datasets}
    for ds in datasets[:rem]:
        quotas[ds] += 1

    for ds in datasets:
        pop_from_dataset(ds, quotas.get(ds, 0))

    # Fill leftovers from global groups
    if len(picked) < target_n:
        remaining = [g for g, rows in by_group.items() if rows]
        rng.shuffle(remaining)
        i = 0
        while len(picked) < target_n and remaining:
            g = remaining[i % len(remaining)]
            bucket = by_group[g]
            if bucket:
                item = bucket.pop()
                k = _key(item)
                if k not in used_keys:
                    used_keys.add(k)
                    picked.append(item)
            if not bucket:
                remaining.remove(g)
                if not remaining:
                    break
            else:
                i += 1

    return picked


def _write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate label templates for Phase-1 metric identity.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reservoir-per-dataset", type=int, default=30000)
    parser.add_argument("--main-domain", type=int, default=200)
    parser.add_argument("--main-quality", type=int, default=200)
    parser.add_argument("--main-difficulty", type=int, default=100)
    parser.add_argument("--main-ood", type=int, default=200)
    parser.add_argument("--transfer-domain", type=int, default=120)
    parser.add_argument("--transfer-quality", type=int, default=120)
    parser.add_argument("--transfer-difficulty", type=int, default=60)
    parser.add_argument("--transfer-ood", type=int, default=120)
    parser.add_argument("--calibration-ood", type=int, default=200)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest = _load_manifest()
    outputs = _dataset_outputs(manifest)
    if not outputs:
        raise SystemExit("No dataset analysis outputs found in run_manifest.json")

    candidates: List[Dict[str, Any]] = []
    for dataset, path in outputs.items():
        if not path.exists():
            continue
        sampled = _reservoir_sample(path, dataset, args.reservoir_per_dataset, rng)
        candidates.extend(sampled)

    if not candidates:
        raise SystemExit("No candidates sampled from analysis outputs.")

    q_values = [x["quality_score"] for x in candidates if x["quality_score"] is not None]
    q_thresholds = _quantile_thresholds(q_values)
    for item in candidates:
        item["quality_bin"] = _quality_bin(item.get("quality_score"), q_thresholds)
        item["difficulty_bin"] = _difficulty_bin(item.get("fk_grade"))

    datasets = sorted(set(x["dataset"] for x in candidates))
    main_datasets = [d for d in datasets if d in ANCHOR_DATASETS]
    if not main_datasets:
        main_datasets = datasets[:]
    transfer_datasets = [d for d in datasets if d not in main_datasets]

    print("[08] Sampled candidates:", len(candidates))
    print("[08] Main datasets:", main_datasets)
    print("[08] Transfer datasets:", transfer_datasets if transfer_datasets else "(none)")

    used_domain, used_quality, used_diff, used_ood = set(), set(), set(), set()

    domain_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []
    ood_rows: List[Dict[str, Any]] = []

    def add_domain(rows_src: List[Dict[str, Any]], split: str):
        for x in rows_src:
            domain_rows.append({
                "dataset": x["dataset"],
                "source": x["source"],
                "doc_id": x["doc_id"],
                "chunk_id": x["chunk_id"],
                "gold_primary": "",
                "gold_alternates_pipe": "",
                "split": split,
                "pred_top_domain": x.get("pred_top_domain", ""),
                "pred_top3_domains_pipe": x.get("pred_top3_domains_pipe", ""),
                "text_preview": x.get("text_preview", ""),
            })

    def add_quality(rows_src: List[Dict[str, Any]], split: str):
        for x in rows_src:
            quality_rows.append({
                "dataset": x["dataset"],
                "source": x["source"],
                "doc_id": x["doc_id"],
                "chunk_id": x["chunk_id"],
                "gold_has_examples": "",
                "gold_has_explanation": "",
                "gold_has_structure": "",
                "split": split,
                "pred_has_examples": x.get("pred_has_examples", ""),
                "pred_has_explanation": x.get("pred_has_explanation", ""),
                "pred_has_structure": x.get("pred_has_structure", ""),
                "text_preview": x.get("text_preview", ""),
            })

    def add_diff(rows_src: List[Dict[str, Any]], split: str):
        for x in rows_src:
            diff_rows.append({
                "dataset": x["dataset"],
                "source": x["source"],
                "doc_id": x["doc_id"],
                "chunk_id": x["chunk_id"],
                "gold_valid": "",
                "reason": "",
                "split": split,
                "pred_difficulty_valid": x.get("pred_difficulty_valid", ""),
                "fk_grade_hint": x.get("fk_grade", ""),
                "text_preview": x.get("text_preview", ""),
            })

    def add_ood(rows_src: List[Dict[str, Any]], split: str):
        for x in rows_src:
            ood_rows.append({
                "dataset": x["dataset"],
                "source": x["source"],
                "doc_id": x["doc_id"],
                "chunk_id": x["chunk_id"],
                "gold_in_domain": "",
                "split": split,
                "pred_ood_label": x.get("pred_ood_label", ""),
                "pred_top_domain": x.get("pred_top_domain", ""),
                "pred_top3_domains_pipe": x.get("pred_top3_domains_pipe", ""),
                "text_preview": x.get("text_preview", ""),
            })

    add_domain(_balanced_sample(candidates, args.main_domain, main_datasets, used_domain, rng), "main_eval")
    add_quality(_balanced_sample(candidates, args.main_quality, main_datasets, used_quality, rng), "main_eval")
    add_diff(_balanced_sample(candidates, args.main_difficulty, main_datasets, used_diff, rng), "main_eval")
    add_ood(_balanced_sample(candidates, args.main_ood, main_datasets, used_ood, rng), "main_eval")

    if transfer_datasets:
        add_domain(_balanced_sample(candidates, args.transfer_domain, transfer_datasets, used_domain, rng), "transfer_eval")
        add_quality(_balanced_sample(candidates, args.transfer_quality, transfer_datasets, used_quality, rng), "transfer_eval")
        add_diff(_balanced_sample(candidates, args.transfer_difficulty, transfer_datasets, used_diff, rng), "transfer_eval")
        add_ood(_balanced_sample(candidates, args.transfer_ood, transfer_datasets, used_ood, rng), "transfer_eval")

    add_ood(_balanced_sample(candidates, args.calibration_ood, main_datasets, used_ood, rng), "calibration")

    _write_csv(
        LABEL_DIR / "domain_labels_v1.csv",
        [
            "dataset",
            "source",
            "doc_id",
            "chunk_id",
            "gold_primary",
            "gold_alternates_pipe",
            "split",
            "pred_top_domain",
            "pred_top3_domains_pipe",
            "text_preview",
        ],
        domain_rows,
    )
    _write_csv(
        LABEL_DIR / "quality_labels_v1.csv",
        [
            "dataset",
            "source",
            "doc_id",
            "chunk_id",
            "gold_has_examples",
            "gold_has_explanation",
            "gold_has_structure",
            "split",
            "pred_has_examples",
            "pred_has_explanation",
            "pred_has_structure",
            "text_preview",
        ],
        quality_rows,
    )
    _write_csv(
        LABEL_DIR / "difficulty_sanity_v1.csv",
        [
            "dataset",
            "source",
            "doc_id",
            "chunk_id",
            "gold_valid",
            "reason",
            "split",
            "pred_difficulty_valid",
            "fk_grade_hint",
            "text_preview",
        ],
        diff_rows,
    )
    _write_csv(
        LABEL_DIR / "ood_labels_v1.csv",
        [
            "dataset",
            "source",
            "doc_id",
            "chunk_id",
            "gold_in_domain",
            "split",
            "pred_ood_label",
            "pred_top_domain",
            "pred_top3_domains_pipe",
            "text_preview",
        ],
        ood_rows,
    )

    print("[08] Wrote templates:")
    print(f"  domain:    {len(domain_rows)}")
    print(f"  quality:   {len(quality_rows)}")
    print(f"  difficulty:{len(diff_rows)}")
    print(f"  ood:       {len(ood_rows)}")

    try:
        autosave_stage(PROJECT_DIR, "generate_label_templates")
    except Exception as e:
        print(f"[Autosave] warning (08): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
